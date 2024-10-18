#!/bin/bash
#SBATCH --job-name=loopTF_FAP # Job name
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:a100:1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00 
#SBATCH --output=loopTF_FAP/slurm_output/%j_%A_%a-%N.out 
#SBATCH --error=loopTF_FAP/slurm_output/%j.err 
#SBATCH --requeue 

# DO NOT EDIT LINE BELOW
/gpfs/radev/apps/avx512/software/dSQ/1.05/dSQBatch.py --job-file /gpfs/radev/project/zhuoran_yang/sc3226/transformer-computation-graph/loopTF_FAP/joblist.txt --status-dir /gpfs/radev/project/zhuoran_yang/sc3226/transformer-computation-graph/loopTF_FAP

