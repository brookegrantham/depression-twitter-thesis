#!/bin/bash
#
#
#SBATCH --job-name=test
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --mem=500M
#SBATCH --output=testing.out




python test.py