import unittest
import torch

from models.language_models import RecurrentTransformerLM

class TestRecurrentTransformerLM(unittest.TestCase):
    def test_model_initialization(self):
        vocab_size = 100
        d_model = 64
        default_n_iters = 5
        n_heads = 4
        dff = 128
        dropout_rate = 0.1
        activation = 'relu'
        norm_first = True
        max_block_size = 128
        norm_type = 'layernorm'
        bias = True
        pos_enc_type = 'pos_emb'
        use_flash_attention = True
        block_kwargs = {}


        # note: depth of each type of block includes 0 (i.e., no block)
        for init_block_depth in range(5):
            for recurrent_block_depth in range(5):
                for head_block_depth in range(5):
                    model = RecurrentTransformerLM(
                        vocab_size=vocab_size,
                        d_model=d_model,
                        init_block_depth=init_block_depth,
                        recurrent_block_depth=recurrent_block_depth,
                        head_block_depth=head_block_depth,
                        default_n_iters=default_n_iters,
                        n_heads=n_heads,
                        dff=dff,
                        dropout_rate=dropout_rate,
                        activation=activation,
                        norm_first=norm_first,
                        max_block_size=max_block_size,
                        norm_type=norm_type,
                        bias=bias,
                        pos_enc_type=pos_enc_type,
                        use_flash_attention=use_flash_attention,
                        block_kwargs=block_kwargs
                    )
                    self.assertIsNotNone(model)

                    # Create dummy input data
                    x = torch.randint(0, vocab_size, (1, 10))

                    # check that model can run with default number of iterations
                    logits, _ = model(x)
                    self.assertIsNotNone(logits)

                    # check that model can run with default number of iterations and compute loss if targets are provided
                    logits, loss = model(x, targets=x)
                    self.assertIsNotNone(loss)

                    # check that model can run with different number of iterations
                    for n_iters in range(default_n_iters*2):
                        logits, _ = model(x, n_iters=n_iters)
                        self.assertIsNotNone(logits)

if __name__ == '__main__':
    unittest.main()