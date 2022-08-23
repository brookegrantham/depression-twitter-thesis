#!/bin/bash
#
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:02:0
#SBATCH --mem=100M

cd "${SLURM_SUBMIT_DIR}"


python3 testbc4.py