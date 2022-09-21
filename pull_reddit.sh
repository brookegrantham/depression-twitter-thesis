#!/bin/bash
#
#SBATCH --partition=cpu
#SBATCH --job-name=pull-reddit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:0
#SBATCH --mem=10000M
#SBATCH --output=pull_reddit.out

cd "${SLURM_SUBMIT_DIR}"


python3 pull_reddit.py