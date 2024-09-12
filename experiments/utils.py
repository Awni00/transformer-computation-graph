import datetime
from models import language_models
import json
import os
import torch

def get_dataset(args):

    with open(f'{args.data_path}/dgm_spec.json', 'r') as f:
        dgm_spec = json.load(f)

    # get files ending with .pt in data_path
    files = [f for f in os.listdir(args.data_path) if f.endswith('.pt')]

    # load data into dictionaries
    train_data = {}
    val_data = {}
    for file in files:
        if file.startswith('train'):
            n_vars = int(file.split('_')[1].split('.')[0])
            train_data[n_vars] = torch.load(f'{args.data_path}/{file}')
        elif file.startswith('val'):
            n_vars = int(file.split('_')[1].split('.')[0])
            val_data[n_vars] = torch.load(f'{args.data_path}/{file}')

    return train_data, val_data, dgm_spec

def get_model(args):

    model = language_models.TransformerLM(
        vocab_size=len(args.dgm_spec['vocab']), # TODO: change this to explicitly depend on tokenizer in DGM?
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

    datetimestr = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")

    run_name = f'{experiment_name} ({datetimestr})'

    return experiment_name, run_name