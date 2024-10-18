from loopTF_FAP.experiments.starter import train_start

train_start(
    **{
        'use_wandb': True,
        'use_ntp_loss': True,
        'seed': None
    }
)
