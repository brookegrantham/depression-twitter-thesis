#!/bin/bash
#
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:02:0
#SBATCH --mem=100M

cd /user/home/bg17893

python3 testbc4.py