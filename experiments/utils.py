import datetime
from models import language_models
import json
import os
from collections import OrderedDict
import torch

def get_dataset(args):

    with open(f'{args.data_path}/dgm_spec.json', 'r') as f:
        dgm_spec = json.load(f)

    # get files ending with .pt in data_path
    files = [f for f in os.listdir(args.data_path) if f.endswith('.pt')]
    n_varss = [int(file.split('_')[1].split('.')[0]) for file in files]
    n_varss.sort()

    # load data into dictionaries
    train_data = OrderedDict()
    val_data = OrderedDict()
    for n_vars in n_varss:
        train_data[n_vars] = torch.load(f'{args.data_path}/train_{n_vars}.pt')
        val_data[n_vars] = torch.load(f'{args.data_path}/val_{n_vars}.pt')

    return train_data, val_data, dgm_spec

def get_model(args):
    if getattr(args, 'recurrent', False):
        model = language_models.RecurrentTransformerLM(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            default_n_iters=args.n_layers,
            n_heads=args.n_heads,
            dff=args.dff,
            dropout_rate=args.dropout_rate,
            activation=args.activation,
            norm_first=args.norm_first,
            max_block_size=args.max_block_size,
            norm_type=args.norm_type,
            bias=args.bias,
            pos_enc_type=args.pos_enc_type,
            use_flash_attention=True,
            block_kwargs=None)
    else:
        model = language_models.TransformerLM(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            dff=args.dff,
            dropout_rate=args.dropout_rate,
            activation=args.activation,
            norm_first=args.norm_first,
            max_block_size=args.max_block_size,
            norm_type=args.norm_type,
            bias=args.bias,
            pos_enc_type=args.pos_enc_type,
            use_flash_attention=True,
            block_kwargs=None)

    return model


def get_experiment_name(args, curriculum_step=None):

    experiment_name = f'L={args.n_layers}-D={args.d_model}-H={args.n_heads}'

    if curriculum_step is not None:
        experiment_name += f'-curriculum_step={curriculum_step}'

    run_name = f'{experiment_name} ({args.seed})'

    return experiment_name, run_name