#!/bin/bash
#SBATCH --job-name=loopTF_FAP # Job name
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:a100:1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00 
#SBATCH --output=slurm_output/%j.out 
#SBATCH --error=slurm_output/%j.err 
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

python test.py --max_dep 6 --med_loss_ratio 1.0 1.0 1.0 1.0 1.0 1.0 --num_workers 8

