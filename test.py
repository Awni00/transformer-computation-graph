from loopTF_FAP.experiments.starter import train_start

train_start(
    **{
        'use_wandb': False,
        'use_ntp_loss': False,
    }
)
