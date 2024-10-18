from loopTF_FAP.experiments.starter import train_start
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Training parameters")
    parser.add_argument('--use_ntp_loss', type=bool, default=True, help='Use NTP loss')
    parser.add_argument('--max_dep', type=int, required=True, help='Maximum depth')
    parser.add_argument('--med_loss_ratio', type=float, nargs='+', required=True, help='List of float values for med_loss_ratio')
    return parser.parse_args()

args = parse_args()

train_start(
    **{
        'use_wandb': True,
        'use_ntp_loss': args.use_ntp_loss,
        'seed': None,
        'max_dep': args.max_dep,
        'med_loss_ratio': args.med_loss_ratio, 
        'data_file_name': 'dag_ADDonly.json'
    }
)