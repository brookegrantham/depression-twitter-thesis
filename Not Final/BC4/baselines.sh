#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --job-name=run-baselines
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --mem=5000M
#SBATCH --output=baselines.out


cd "${SLURM_SUBMIT_DIR}"


python3 baselines_v2.py