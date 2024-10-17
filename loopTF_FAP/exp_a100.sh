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

# python Tk_comp_atomic.py --only_dst
python test.py