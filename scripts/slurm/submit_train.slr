#!/bin/bash -l
#SBATCH --time=04:00:00
#SBATCH -C gpu
#SBATCH --account=cusp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=10
#SBATCH -o scripts/sout/train_latest_2.out


conda activate tfp
srun python scripts/train_ae.py
srun python scripts/train_flow.py 
srun python scripts/run_posterior_analysis.py

