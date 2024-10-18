#!/bin/bash
#SBATCH --job-name=loopTF_FAP # Job name
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:a100:1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00 
#SBATCH --output=loopTF_FAP/slurm_output/%j.out 
#SBATCH --error=loopTF_FAP/slurm_output/%j.err 
#SBATCH --requeue 

echo '-------------------------------'
cd ${SLURM_SUBMIT_DIR}
echo ${SLURM_SUBMIT_DIR}
echo Running on host $(hostname)
echo Time is $(date)
echo SLURM_NODES are $(echo ${SLURM_NODELIST})
echo '-------------------------------'
echo -e '\n\n'

export PROCS=${SLURM_CPUS_ON_NODE}

# Set the working directory
cd /home/sc3226/project/transformer-computation-graph

module load miniconda
conda activate scgpt1


# Array of different configurations
declare -a configs=(
    "--use_ntp_loss False --max_dep 2 --add_med_loss_prob 0.1 1.0"
    "--use_ntp_loss False --max_dep 4 --add_med_loss_prob 0.001 0.01 0.1 1.0"
    "--use_ntp_loss False --max_dep 6 --add_med_loss_prob 0.00001 0.0001 0.001 0.01 0.1 1.0"
    "--use_ntp_loss False --max_dep 2 --add_med_loss_prob 1.0 1.0"
    "--use_ntp_loss False --max_dep 4 --add_med_loss_prob 1.0 1.0 1.0 1.0"
    "--use_ntp_loss False --max_dep 6 --add_med_loss_prob 1.0 1.0 1.0 1.0 1.0 1.0"
)

# Loop through configurations and submit each as a separate job
for config in "${configs[@]}"
do
    sbatch --wrap="python test.py $config"
done