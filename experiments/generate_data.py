from tqdm import tqdm, trange
import networkx as nx
from computation_graph_dgm import ComputationGraphDGM, draw_topological_order
import string
import torch

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--mod_val', type=int, default=16)
parser.add_argument('--n_vars_start', type=int, default=2)
parser.add_argument('--n_vars_end', type=int, default=20)
parser.add_argument('--samples_per_n_var', type=int, default=10_000) # for now, the same for all n_vars
parser.add_argument('--val_samples_per_n_var', type=int, default=1_000)

args = parser.parse_args()

var_vocab = list(string.ascii_lowercase) # letters a-z
function_map = {'sum': lambda x: sum(x) % mod_val}
mod_val = args.mod_val

if not os.path.exists('data/'):
    os.makedirs('data/')

data_path = f'data/modval_{args.mod_val}'
if not os.path.exists(data_path):
    os.makedirs(data_path)

dgm = ComputationGraphDGM(var_vocab=var_vocab, function_map=function_map, mod_val=mod_val)

n_var_range = list(range(args.n_vars_start, args.n_vars_end+1))

# save dgm to json file
dgm_spec = dict(vars(dgm))
import json
with open(f'{data_path}/dgm_spec.json', 'w') as f:
    json.dump(dgm_spec, f, default=repr, indent=2)

# generate curriculum of training examples
for n_vars in tqdm(n_var_range):
    train_data = []
    for _ in trange(args.samples_per_n_var, leave=False):
        example = dgm.sample_example(n_vars=n_vars, func_degree=2)
        train_data.append(example['tokenized_query_prompt'])

    train_data = torch.tensor(train_data)
    torch.save(train_data, f'{data_path}/train_{n_vars}.pt')

    val_data = []
    for _ in trange(args.val_samples_per_n_var, leave=False):
        example = dgm.sample_example(n_vars=n_vars, func_degree=2)
        val_data.append(example['tokenized_query_prompt'])

    val_data = torch.tensor(val_data)
    torch.save(val_data, f'{data_path}/val_{n_vars}.pt')