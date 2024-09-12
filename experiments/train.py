import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import ignite
import torchinfo

import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar

from utils import get_experiment_name, get_model, get_dataset
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

parser.add_argument("--batch-size", default=128, type=int)
parser.add_argument("--eval-batch-size", default=1024, type=int)
parser.add_argument("--max-epochs", default=100, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--min-lr", default=1e-5, type=float)
parser.add_argument("--warmup-epoch", default=5, type=int)
parser.add_argument("--beta1", default=0.9, type=float)
parser.add_argument("--beta2", default=0.999, type=float)
parser.add_argument("--off-benchmark", action="store_true")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--weight-decay", default=5e-5, type=float)

parser.add_argument("--precision", default='bf16', type=str)
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

train_data, val_data, dgm_spec = get_dataset(args)

args.dgm_spec = dgm_spec

data_split_name_map = dict(enumerate(list(train_data.keys())))
min_nvars = min(list(train_data.keys()))
max_nvars = max(list(train_data.keys()))

def create_datamodule(curriculum_nvars):

    train_loader = torch.utils.data.DataLoader(train_data[curriculum_nvars], batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loaders = [torch.utils.data.DataLoader(val_data[i], batch_size=args.eval_batch_size, num_workers=args.num_workers) for i in range(min_nvars, curriculum_nvars)]

    data_module = dict(train_dataloaders=train_loader, val_dataloaders=val_loaders)

    return data_module


class LitLM(pl.LightningModule):
    def __init__(self, hparams):
        super(LitLM, self).__init__()
        # self.hparams = hparams
        self.hparams.update(vars(hparams))
        self.model = get_model(hparams)
        if args.compile:
            self.model = torch.compile(self.model)

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

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch[:, :-1], batch[:, 1:]
        logits, loss = self(x)

        data_split_name = f"val_{data_split_name_map[dataloader_idx]}"
        self.log(f"loss/{data_split_name}", loss)
        self.log(f"perplexity/{data_split_name}", torch.exp(loss))

        query_pred = logits[:, -1].argmax(-1)
        query_true = y[:, -1]
        acc = (query_pred == query_true).float().mean()
        self.log(f"acc/{data_split_name}", acc)

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

    for curriculum_nvars in range(min_nvars, max_nvars+1):
        print(f"Training for curriculum_nvars={curriculum_nvars}")
        datamodule = create_datamodule(curriculum_nvars)
        train_dl, val_dls = datamodule['train_dataloaders'], datamodule['val_dataloaders']

        logger = create_logger(curriculum_nvars)

        refresh_rate = 1
        trainer = pl.Trainer(max_epochs=args.max_epochs, precision=args.precision, fast_dev_run=args.dry_run, devices=args.gpus, benchmark=args.benchmark,
            logger=logger, callbacks=[TQDMProgressBar(refresh_rate=refresh_rate)])

        trainer.fit(model=net, train_dataloaders=train_dl, val_dataloaders=val_dls)

    # if not args.dry_run:
    #     model_path = f"model_checkpoints/{experiment_name}.pth"
    #     torch.save(net.state_dict(), model_path)
        # if args.wandb_project is not None:
        #     logger.experiment.log_asset(file_name=experiment_name, file_data=model_path)