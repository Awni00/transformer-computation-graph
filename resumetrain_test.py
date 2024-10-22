from loopTF_FAP.experiments.starter import train_continue

# type = 'resumetrain'
type = 'test'

if __name__ == '__main__':
    if type == 'resumetrain'
        train_continue('L2H16-N64-D6-medprob-[1.0e-1,1.0e+0,1.0e-1,1.0e+0,1.0e-1,1.0e+0]1018-1747', 'epoch=173-val_loss=0.0505.ckpt')
    elif type == 'test':
        