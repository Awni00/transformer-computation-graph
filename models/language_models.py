import torch
from torch import nn
from .transformer_blocks import EncoderBlock, create_norm
from .attention import precompute_freqs_cis
import math

class TransformerLM(nn.Module):
    """Transformer Language Model"""

    def __init__(self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dff: int,
        dropout_rate: float,
        activation: str,
        norm_first: bool,
        max_block_size: int,
        norm_type: str = 'layernorm',
        bias: bool = True,
        pos_enc_type: str = 'pos_emb',
        use_flash_attention=True,
        block_kwargs: dict = None
        ):
        """
        Transformer autoregressive language model.

        given (x_1, ..., x_T) causally predicts (y_1, ..., y_T)

        Parameters
        ----------
        vocab_size : int
            vocabulary size.
        d_model : int
            model dimension.
        n_layers : int
            number of layers.
        n_heads : int
            number of attention heads.
        dff : int
            size of intermediate layer in feedforward blocks.
        dropout_rate : float
            dropout rate.
        activation : str
            name of activation function (e.g., 'relu', 'gelu', or 'swiglu').
        norm_first : bool
            whether to apply layer normalization before or after attention.
        max_block_size : int
            maximum context size.
        bias : bool, optional
            whether to use bias in attention, by default True
        pos_enc_type : 'pos_emb' or 'RoPE', optional
            type of positional encoding to use, by default 'pos_emb'
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.norm_first = norm_first
        self.block_size = max_block_size
        self.norm_type = norm_type
        self.bias = bias
        self.pos_enc_type = pos_enc_type
        self.block_kwargs = block_kwargs if block_kwargs is not None else {}
        self.use_flash_attention = use_flash_attention
        self._need_weights = not use_flash_attention # used to specify whether flash attention is used

        layers = dict(
            token_embedder = nn.Embedding(vocab_size, d_model),
            dropout = nn.Dropout(dropout_rate),
            blocks = nn.ModuleList([EncoderBlock(
                d_model=d_model, n_heads=n_heads, dff=dff, dropout_rate=dropout_rate, activation=activation,
                norm_first=norm_first, norm_type=norm_type, bias=bias, causal=True, **self.block_kwargs) for _ in range(n_layers)]),
            norm = create_norm(d_model, self.norm_type),
            final_out = nn.Linear(d_model, vocab_size, bias=False)
            )

        if pos_enc_type == 'pos_emb':
            # if using positional embedding, create positional embedding layer
            positional_embedder = nn.Embedding(max_block_size, d_model)
            layers['positional_embedder'] = positional_embedder
        elif pos_enc_type == 'RoPE':
            # if using RoPE, precompute RoPE sine-cosine rotation matrices
            freqs_cos, freqs_sin = precompute_freqs_cis(self.d_model // self.n_heads, self.block_size)
            self.register_buffer("freqs_cos", freqs_cos, persistent=False)
            self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        else:
            raise ValueError('`pos_enc_type` invalid')

        self.layers = nn.ModuleDict(layers)

        # weight-tying embedder and final layer
        self.layers.token_embedder.weight = self.layers.final_out.weight

        # initialize weights
        self.apply(self._init_weights)
        # NOTE: previously, I did not apply special initialization, but it turns out that it is important


        # per-GPT2 paper, scale intialization of output projection and last layer of mlp
        # apply special n_layer-scaled initialization to layers that add to the residual stream
        # (output projection of attention and last layer of mlp)
        # this ensures that, at initialization, adding to the residual stream does not cause things to blow up
        # note: while the _init_weights seemed to have a big effect, it is unclear what effect this is having
        mlp_special_init_layer = 'linear3' if activation == 'swiglu' else 'linear2'
        for pn, p in self.named_parameters():
            if pn.endswith(f'{mlp_special_init_layer}.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, x, targets=None):
        device = x.device
        b, t = x.size()
        assert t <= self.block_size, f'Input sequence length {t} exceeds maximum block size {self.block_size}'

        x = self.layers.token_embedder(x)

        if self.pos_enc_type == 'pos_emb':
            # if using positional embeddings, add positional embeeddings
            positions = torch.arange(0, t, dtype=torch.long, device=device)
            positional_embedding = self.layers.positional_embedder(positions)
            x = x + positional_embedding
            freqs_cos, freqs_sin = None, None # not using RoPE
        elif self.pos_enc_type == 'RoPE':
            # otherwise, get the RoPE matrices
            freqs_cos = self.freqs_cos[:t]
            freqs_sin = self.freqs_sin[:t]

        for enc_block in self.layers.blocks:
            x = enc_block(x, freqs_cos=freqs_cos, freqs_sin=freqs_sin, need_weights=self._need_weights)

        x = self.layers.norm(x)

        logits = self.layers.final_out(x)

        loss = None
        if targets is not None:
            # compute loss if given targets
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=-1)

        return logits, loss


    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311

        N = self.get_num_params()
        L, H, Q, T = self.n_layers, self.n_heads, self.d_model//self.n_heads, self.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.pos_enc_type=='pos_emb':
            n_params -= self.layers.positional_embedder.weight.numel()
        return n_params

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None):
        """
        Generate max_new_tokens new tokens, conditioning on the input idx.

        Parameters
        ----------
        idx : Tensor[int]
            tensor of shape (batch_size, seq_len) with input tokens.
        max_new_tokens : int
            number of new tokens to generate
        temperature : float, optional
            temperature parameter of softmax, by default 1.0
        top_k : int, optional
            top-k sampling parameter, by default None

        Returns
        -------
        Tensor[int]
            tensor of shape (batch_size, seq_len + max_new_tokens) with generated tokens.
        """
        for _ in range(max_new_tokens):
            # crop the sequence if it is longer thanblock_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond) # forward pass
            logits = logits[:, -1, :] / temperature # scale by temperature

            # optionally, crop logits to top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = torch.nn.functional.softmax(logits, dim=-1) # convert to probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # sample from distribution
            idx = torch.cat((idx, idx_next), dim=1) # append to sequence

        return idx

class RecurrentTransformerLM(nn.Module):
    """Transformer Language Model"""

    def __init__(self,
        vocab_size: int,
        d_model: int,
        init_block_depth: int,
        recurrent_block_depth: int,
        head_block_depth: int,
        default_n_iters: int,
        n_heads: int,
        dff: int,
        dropout_rate: float,
        activation: str,
        norm_first: bool,
        max_block_size: int,
        norm_type: str = 'layernorm',
        bias: bool = True,
        pos_enc_type: str = 'pos_emb',
        use_flash_attention=True,
        block_kwargs: dict = None
        ):
        """
        Transformer autoregressive language model.

        given (x_1, ..., x_T) causally predicts (y_1, ..., y_T)

        Parameters
        ----------
        vocab_size : int
            vocabulary size.
        d_model : int
            model dimension.
        init_block_depth : int
            depth of the initial block, applied before beginning of recurrence.
        recurrent_block_depth : int
            depth of the recurrent block. At each iteration, this block is applied.
        head_block_depth : int
            depth of the head block. After the recurrence, this block is applied to generate the final output.
        default_n_iters : int
            default number of iterations for the recurrent block. This can be specified arbitrarily at inference time.
        n_heads : int
            number of attention heads.
        dff : int
            size of intermediate layer in feedforward blocks.
        dropout_rate : float
            dropout rate.
        activation : str
            name of activation function (e.g., 'relu', 'gelu', or 'swiglu').
        norm_first : bool
            whether to apply layer normalization before or after attention.
        max_block_size : int
            maximum context size.
        norm_type : str, optional
            type of normalization to use, by default 'layernorm'.
        bias : bool, optional
            whether to use bias in attention, by default True.
        pos_enc_type : str, optional
            type of positional encoding to use, by default 'pos_emb'.
        use_flash_attention : bool, optional
            whether to use flash attention, by default True.
        block_kwargs : dict, optional
            additional arguments for the blocks, by default None.
        """

        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.default_n_iters = default_n_iters
        self.init_block_depth = init_block_depth
        self.recurrent_block_depth = recurrent_block_depth
        self.head_block_depth = head_block_depth
        self.n_heads = n_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.norm_first = norm_first
        self.block_size = max_block_size
        self.norm_type = norm_type
        self.bias = bias
        self.pos_enc_type = pos_enc_type
        self.block_kwargs = block_kwargs if block_kwargs is not None else {}
        self.use_flash_attention = use_flash_attention
        self._need_weights = not use_flash_attention # used to specify whether flash attention is used

        layers = dict(
            token_embedder = nn.Embedding(vocab_size, d_model),
            dropout = nn.Dropout(dropout_rate),
            init_blocks = nn.ModuleList([EncoderBlock(
                d_model=d_model, n_heads=n_heads, dff=dff, dropout_rate=dropout_rate, activation=activation,
                norm_first=norm_first, norm_type=norm_type, bias=bias, causal=True, **self.block_kwargs) for _ in range(init_block_depth)]),
            recurrent_blocks = nn.ModuleList([EncoderBlock(
                d_model=d_model, n_heads=n_heads, dff=dff, dropout_rate=dropout_rate, activation=activation,
                norm_first=norm_first, norm_type=norm_type, bias=bias, causal=True, **self.block_kwargs) for _ in range(recurrent_block_depth)]),
            head_blocks = nn.ModuleList([EncoderBlock(
                d_model=d_model, n_heads=n_heads, dff=dff, dropout_rate=dropout_rate, activation=activation,
                norm_first=norm_first, norm_type=norm_type, bias=bias, causal=True, **self.block_kwargs) for _ in range(head_block_depth)]),
            norm = create_norm(d_model, self.norm_type),
            final_out = nn.Linear(d_model, vocab_size, bias=False)
            )

        if pos_enc_type == 'pos_emb':
            # if using positional embedding, create positional embedding layer
            positional_embedder = nn.Embedding(max_block_size, d_model)
            layers['positional_embedder'] = positional_embedder
        elif pos_enc_type == 'RoPE':
            # if using RoPE, precompute RoPE sine-cosine rotation matrices
            freqs_cos, freqs_sin = precompute_freqs_cis(self.d_model // self.n_heads, self.block_size)
            self.register_buffer("freqs_cos", freqs_cos, persistent=False)
            self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        else:
            raise ValueError('`pos_enc_type` invalid')

        self.layers = nn.ModuleDict(layers)

        # weight-tying embedder and final layer
        self.layers.token_embedder.weight = self.layers.final_out.weight

        # initialize weights
        self.apply(self._init_weights)
        # NOTE: previously, I did not apply special initialization, but it turns out that it is important


        # per-GPT2 paper, scale intialization of output projection and last layer of mlp
        # apply special n_layer-scaled initialization to layers that add to the residual stream
        # (output projection of attention and last layer of mlp)
        # this ensures that, at initialization, adding to the residual stream does not cause things to blow up
        # note: while the _init_weights seemed to have a big effect, it is unclear what effect this is having
        mlp_special_init_layer = 'linear3' if activation == 'swiglu' else 'linear2'
        for pn, p in self.named_parameters():
            if pn.endswith(f'{mlp_special_init_layer}.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * default_n_iters))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, x, targets=None, n_iters=None):
        device = x.device
        b, t = x.size()
        assert t <= self.block_size, f'Input sequence length {t} exceeds maximum block size {self.block_size}'

        x = self.layers.token_embedder(x)

        if self.pos_enc_type == 'pos_emb':
            # if using positional embeddings, add positional embeeddings
            positions = torch.arange(0, t, dtype=torch.long, device=device)
            positional_embedding = self.layers.positional_embedder(positions)
            x = x + positional_embedding
            freqs_cos, freqs_sin = None, None # not using RoPE
        elif self.pos_enc_type == 'RoPE':
            # otherwise, get the RoPE matrices
            freqs_cos = self.freqs_cos[:t]
            freqs_sin = self.freqs_sin[:t]

        if n_iters is None:
            n_iters = self.default_n_iters

        # first, apply the initial blocks
        for enc_block in self.layers.init_blocks:
            x = enc_block(x, freqs_cos=freqs_cos, freqs_sin=freqs_sin, need_weights=self._need_weights)

        # then, reccurently apply the recurrent blocks n_iters times
        for _ in range(n_iters):
            for enc_block in self.layers.recurrent_blocks:
                x = enc_block(x, freqs_cos=freqs_cos, freqs_sin=freqs_sin, need_weights=self._need_weights)

        # finally, apply the head blocks to generate the final output
        for enc_block in self.layers.head_blocks:
            x = enc_block(x, freqs_cos=freqs_cos, freqs_sin=freqs_sin, need_weights=self._need_weights)

        x = self.layers.norm(x)

        logits = self.layers.final_out(x)

        loss = None
        if targets is not None:
            # compute loss if given targets
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.contiguous().view(-1), ignore_index=-1)

        return logits, loss


    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311

        N = self.get_num_params()
        L, H, Q, T = self.default_n_iters, self.n_heads, self.d_model//self.n_heads, self.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.pos_enc_type=='pos_emb':
            n_params -= self.layers.positional_embedder.weight.numel()
        return n_params

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None):
        """
        Generate max_new_tokens new tokens, conditioning on the input idx.

        Parameters
        ----------
        idx : Tensor[int]
            tensor of shape (batch_size, seq_len) with input tokens.
        max_new_tokens : int
            number of new tokens to generate
        temperature : float, optional
            temperature parameter of softmax, by default 1.0
        top_k : int, optional
            top-k sampling parameter, by default None

        Returns
        -------
        Tensor[int]
            tensor of shape (batch_size, seq_len + max_new_tokens) with generated tokens.
        """
        for _ in range(max_new_tokens):
            # crop the sequence if it is longer thanblock_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, _ = self(idx_cond) # forward pass
            logits = logits[:, -1, :] / temperature # scale by temperature

            # optionally, crop logits to top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = torch.nn.functional.softmax(logits, dim=-1) # convert to probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # sample from distribution
            idx = torch.cat((idx, idx_next), dim=1) # append to sequence

        return idx

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    use_fused = (device_type == 'cuda')
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
    print(f"using fused AdamW: {use_fused}")

    return optimizer