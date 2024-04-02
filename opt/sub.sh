#!/bin/bash
#SBATCH --qos main
#SBATCH -J kmc-md
#SBATCH --mail-type=END
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=1G

python3 main.py
