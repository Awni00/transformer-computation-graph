from loopTF_FAP.experiments.starter import train_start
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument('--use_loss_n', type=bool, default=True, help='Use NTP loss')
    parser.add_argument('--use_parent_loss', type=bool, default=True, help='Use parent loss')
    parser.add_argument('--max_dep', type=int, default=6, help='Maximum depth')
    parser.add_argument('--med_loss_ratio', type=float, nargs='+', default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], required=False, help='List of float values for med_loss_ratio')
    parser.add_argument('--last_run_name', type=str, required=False, help='Name of the last run')
    parser.add_argument('--ckpt_file_name', type=str, required=False, help='Checkpoint file name')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for data loading')
    return parser.parse_args()

args = parse_args()

train_start(
    data_file_name='dag_ADDonly.json',
    vocab_file_name='vocab.yaml',
    **{
        'use_wandb': True,
        'use_loss_n': args.use_loss_n,
        'seed': 21,
        'max_dep': args.max_dep,
        'med_loss_ratio': args.med_loss_ratio,
        'num_workers': args.num_workers
    }
)

# debug purpose
# train_start(
#     **{
#         'use_wandb': False,
#         'use_ntp_loss': False,
#         'seed': 42,
#         'max_dep': 6,
#         'med_loss_ratio': [0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 
#         'data_file_name': 'dag_Addonly.json'
#     }
# )