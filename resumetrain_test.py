from loopTF_FAP.experiments.starter import train_continue, test_start

# type = 'resumetrain'
type = 'test'
last_run_name = 'L2H16-N64-D6-medprob-[1.0e-1,1.0e+0,1.0e-1,1.0e+0,1.0e-1,1.0e+0]1018-1747'
ckpt_file_name = 'epoch=173-val_loss=0.0505.ckpt'

if __name__ == '__main__':
    if type == 'resumetrain':
        train_continue(last_run_name, ckpt_file_name)
    elif type == 'test':
        test_start(last_run_name, ckpt_file_name)