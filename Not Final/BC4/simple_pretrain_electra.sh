#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --job-name=pre-elec
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:0
#SBATCH --mem=5000M
#SBATCH --output=pretrain_electra.out


cd "${SLURM_SUBMIT_DIR}"


python3 simple_pretrain_electra.py