#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --job-name=run-roberta
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:0
#SBATCH --mem=5000M
#SBATCH --output=roberta-large-log.out

cd "${SLURM_SUBMIT_DIR}"


python3 roberta-large-multi-gpu.py