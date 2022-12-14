#!/usr/bin/env bash
#SBATCH -t 0-00:30:00
#SBATCH -p wacc
#SBATCH -J env
#SBATCH -o env.out -e env.err
#SBATCH -c 1 --gres=gpu:1

conda init bash
source ~/.bashrc
conda activate project_old_python
conda install -c numba pyculib
conda deactivate
