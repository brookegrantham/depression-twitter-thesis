#!/bin/bash
#SBATCH --job-name=jupyter
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=50GB
#SBATCH --output=/user/home/bg17893/jupyter.log

module load anaconda
source activate /share/sw/open/anaconda/3

cat /etc/hosts
jupyter lab --ip=0.0.0.0 --port=8888