import argparse

import torch
import torchinfo
import wandb

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar

from utils import get_experiment_name, get_model, get_dataset
from computation_graph_dgm import Tokenizer
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument("--wandb_project", default=None, type=str)
parser.add_argument("--wandb_entity", default='transformer-computation-graph', type=str)

parser.add_argument("--data_path", default='data/modval_16', type=str)

parser.add_argument('--n_heads', default=12, type=int, help='number of self-attention heads')
parser.add_argument('--n_layers', default=8, type=int, help='number of layers')
parser.add_argument('--d_model', default=384, type=int, help='model dimension')
parser.add_argument('--dff', default=None, type=int, help='feedforward hidden dimension')
parser.add_argument('--activation', default='gelu', type=str, help='MLP activation')
parser.add_argument('--dropout_rate', default=0., type=float, help='dropout rate')
parser.add_argument('--norm_first', default=1, type=int, help='whether to use pre-LN or post-LN')
parser.add_argument('--norm_type', default='layernorm', type=str, choices=["layernorm", "rmsnorm"], help='normalization layer type')
# parser.add_argument('--n_kv_heads', type=int, default=None, help='Number of key/value heads (e.g., MQA if 1)')
parser.add_argument('--bias', default=1, type=int, help='whether to use bias')
parser.add_argument("--pos_enc_type", default="RoPE", type=str, choices=["RoPE", "pos_emb"])
parser.add_argument("--max_block_size", default=1024, type=int)

parser.add_argument("--batch_size", default=1024, type=int)
parser.add_argument("--eval_batch_size", default=None, type=int)
parser.add_argument("--max_epochs", default=100, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--min_lr", default=1e-5, type=float)
parser.add_argument("--warmup_epoch", default=5, type=int)
parser.add_argument("--beta1", default=0.9, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
parser.add_argument("--off_benchmark", action="store_true")
parser.add_argument("--dry_run", action="store_true")
parser.add_argument("--weight_decay", default=5e-5, type=float)
parser.add_argument("--train_cumulative", action="store_true")

parser.add_argument("--precision", default='bf16-mixed', type=str)
parser.add_argument("--compile", action="store_true")

parser.add_argument("--seed", default=None, type=int)
args = parser.parse_args()

if args.seed is None:
    print("Seed not specified by script arguments. Will generate randomly.")
    args.seed = random.randrange(2**32 - 1)

print(f"Setting seed to: {args.seed}")
torch.manual_seed(args.seed)
np.random.seed(args.seed)


# process args
args.norm_first = True if args.norm_first==1 else False
args.bias = args.bias

args.benchmark = True if not args.off_benchmark else False
args.gpus = torch.cuda.device_count()
args.num_workers = 4*args.gpus if args.gpus else 8
if not args.gpus:
    args.precision=32

args.eval_batch_size = args.eval_batch_size or args.batch_size

train_data, val_data, dgm_spec = get_dataset(args)

args.dgm_spec = dgm_spec

min_nvars = min(list(train_data.keys()))
max_nvars = max(list(train_data.keys()))
data_split_name_map = {k: v for k, v in enumerate(range(min_nvars, max_nvars+1))}

tokenizer = dgm_spec['tokenizer']
tokenizer = Tokenizer(vocab=tokenizer['vocab'])
args.vocab_size = len(tokenizer.vocab)

def create_datamodule(curriculum_nvars, train_cummulative=True):

    if train_cummulative:
        pad_token_idx = dgm_spec['tokenizer']['tok2idx'][dgm_spec['tokenizer']['pad_token']]
        pad_length = max([train_data[i].shape[1] for i in range(min_nvars, curriculum_nvars + 1)])

        pad_transform = lambda x: torch.nn.functional.pad(x, pad=(0, pad_length - x.shape[1]), value=pad_token_idx)
        train_ds_ = torch.concat([pad_transform(train_data[i]) for i in range(min_nvars, curriculum_nvars + 1)])
        print(f"train_ds_ shape: {train_ds_.shape}")
        train_loaders = torch.utils.data.DataLoader(train_ds_,
            batch_size=args.eval_batch_size, num_workers=args.num_workers, shuffle=True)
    else:
        train_loaders = torch.utils.data.DataLoader(train_data[curriculum_nvars], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    val_loaders = [torch.utils.data.DataLoader(val_data[i], batch_size=args.eval_batch_size, num_workers=args.num_workers) for i in range(min_nvars, curriculum_nvars + 1)]

    data_module = dict(train_dataloaders=train_loaders, val_dataloaders=val_loaders)

    return data_module


class LitLM(pl.LightningModule):
    def __init__(self, hparams):
        super(LitLM, self).__init__()
        # self.hparams = hparams
        self.hparams.update(vars(hparams))
        self.model = get_model(hparams)

        self.log_image_flag = hparams.wandb_project is None

    def forward(self, x, y=None):
        return self.model(x, y)

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2), weight_decay=self.hparams.weight_decay)
        self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.min_lr)
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=0.1, end_factor=1.0, last_epoch=self.hparams.warmup_epoch)

        self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer,
            schedulers=[self.warmup_scheduler, self.base_scheduler], milestones=[self.hparams.warmup_epoch])

        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch[:, :-1], batch[:, 1:]

        logits, loss = self(x, y)

        self.log("loss/train", loss)

        return loss

    def on_train_epoch_end(self):
        self.log("lr", self.optimizer.param_groups[0]["lr"], on_epoch=True)#self.current_epoch)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch[:, :-1], batch[:, 1:]
        logits, loss = self(x, y)

        data_split_name = f"nvars={data_split_name_map[dataloader_idx]}"
        self.log(f"val_loss/{data_split_name}", loss, add_dataloader_idx=False)
        self.log(f"val_perplexity/{data_split_name}", torch.exp(loss), add_dataloader_idx=False)

        # where model's predicted output would be
        answer_index = torch.where(y==tokenizer.tok2idx['<answer>'])[1].unsqueeze(0).T # get the index of the answer token
        answer_index += 1 # shift by 1 to get the next token

        predicted_tokens = logits.argmax(-1) # token assigned the highest probability
        predicted_answer = torch.gather(predicted_tokens, 1, answer_index).squeeze() # get the predicted answer token

        true_answer = torch.gather(y, 1, answer_index).squeeze()
        acc = (predicted_answer == true_answer).float().mean()
        self.log(f"val_acc/{data_split_name}", acc, add_dataloader_idx=False)

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch[:, :-1], batch[:, 1:]
        logits, loss = self(x, y)

        data_split_name = f"test/nvars={data_split_name_map[dataloader_idx]}"
        self.log(f"loss/{data_split_name}", loss, add_dataloader_idx=False)
        self.log(f"perplexity/{data_split_name}", torch.exp(loss), add_dataloader_idx=False)

        # where model's predicted output would be
        answer_index = torch.where(y==tokenizer.tok2idx['<answer>'])[1].unsqueeze(0).T # get the index of the answer token
        answer_index += 1 # shift by 1 to get the next token

        predicted_tokens = logits.argmax(-1) # token assigned the highest probability
        predicted_answer = torch.gather(predicted_tokens, 1, answer_index).squeeze() # get the predicted answer token

        true_answer = torch.gather(y, 1, answer_index).squeeze()
        acc = (predicted_answer == true_answer).float().mean()
        self.log(f"test_acc/{data_split_name}", acc, add_dataloader_idx=False)

        return loss


def create_logger(curriculum_nvars):
    experiment_name, run_name = get_experiment_name(args, curriculum_step=f'nvars={curriculum_nvars}')
    print(f"experiment name: {experiment_name}")


    if args.wandb_project is not None:
        print("[INFO] Logging with W&B...")
        logger = pl.loggers.WandbLogger(
            project=args.wandb_project,
            name=run_name,
            group=experiment_name,
            entity=args.wandb_entity,
            config=args
        )

    else:
        print("[INFO] Logging with CSV...")
        logger = pl.loggers.CSVLogger(
            save_dir="logs",
            name=experiment_name
        )

    return logger


if __name__ == "__main__":

    net = LitLM(args)

    model_summary = torchinfo.summary(net.model, input_data=torch.zeros(1, args.max_block_size, dtype=int),
        col_names=("input_size", "output_size", "num_params", "params_percent"))
    print(model_summary)

    model_summary_dict = {
        'Input size (MB)': model_summary.to_megabytes(model_summary.total_input),
        'Params size (MB)': model_summary.to_megabytes(model_summary.total_param_bytes),
        'Forward/backward pass size  (MB)': model_summary.to_megabytes(model_summary.total_output_bytes),
        'Estimated total size (MB)': model_summary.to_megabytes(model_summary.total_output_bytes + model_summary.total_param_bytes + model_summary.total_input),
        'Total Mult-Adds': model_summary.total_mult_adds,

        'trainable_params': model_summary.trainable_params, # note: numbers from torchinfo are not always accurate
        'total_params': model_summary.total_params, # note: numbers from torchinfo are not always accurate

        'num_params': sum(p.numel() for p in net.model.parameters()),
        'num_trainable_params': sum(p.numel() for p in net.model.parameters() if p.requires_grad)
    }

    print(f'num params: {model_summary_dict["num_params"]:,}')
    print(f'num trainable params: {model_summary_dict["num_trainable_params"]:,}')
    args.model_summary = model_summary_dict

    if args.compile:
        # net.model = torch.compile(net.model)
        net = torch.compile(net)

    for curriculum_nvars in range(min_nvars, max_nvars+1):
        print('='*100)
        print(f"Training for curriculum_nvars={curriculum_nvars}")
        print('='*100)
        datamodule = create_datamodule(curriculum_nvars, args.train_cumulative)
        train_dls, val_dls = datamodule['train_dataloaders'], datamodule['val_dataloaders']

        logger = create_logger(curriculum_nvars)

        refresh_rate = 100
        trainer = pl.Trainer(max_epochs=args.max_epochs, precision=args.precision, fast_dev_run=args.dry_run, benchmark=args.benchmark, # devices=args.gpus,
            logger=logger, callbacks=[TQDMProgressBar(refresh_rate=refresh_rate)])

        trainer.fit(model=net, train_dataloaders=train_dls, val_dataloaders=val_dls)

        trainer.test(model=net, dataloaders=val_dls)
        print('='*100)
        print()
        wandb.finish()


    # if not args.dry_run:
    #     model_path = f"model_checkpoints/{experiment_name}.pth"
    #     torch.save(net.state_dict(), model_path)
        # if args.wandb_project is not None:
        #     logger.experiment.log_asset(file_name=experiment_name, file_data=model_path)