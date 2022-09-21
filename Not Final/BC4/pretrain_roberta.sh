#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --job-name=pre_roberta
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:0
#SBATCH --mem=50000M
#SBATCH --output=pre_robertaB.out

cd "${SLURM_SUBMIT_DIR}"


python3 pretrain_roberta.py