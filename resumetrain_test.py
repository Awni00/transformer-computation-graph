from loopTF_FAP.experiments.starter import train_continue, test_start

# type = 'resumetrain'
type = 'resumetrain'
last_run_name = 'L2H16D256-N64DP6[ADD]-MR[1.0e+0,1.0e+0,1.0e+0,1.0e+0,1.0e+0,1.0e+0]NTP1.0-1020-1547'
ckpt_file_name = 'epoch=470-val_loss=0.0016.ckpt'

if __name__ == '__main__':
    if type == 'resumetrain':
        train_continue(last_run_name, ckpt_file_name)
    elif type == 'test':
        test_start(last_run_name, ckpt_file_name)