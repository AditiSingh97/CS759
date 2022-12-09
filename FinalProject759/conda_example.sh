#!/usr/bin/env bash
#SBATCH -t 0-00:30:00
#SBATCH -p wacc
#SBATCH -J test
#SBATCH -o test.out -e test.err
#SBATCH -c 1 --gres=gpu:1

conda init bash
source ~/.bashrc
conda activate test
python test.py
conda deactivate
