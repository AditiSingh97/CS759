#!/usr/bin/env bash
#SBATCH -t 0-00:30:00
#SBATCH -p wacc
#SBATCH -J conda
#SBATCH -o conda.out -e conda.err
#SBATCH -c 1

conda env remove -n project_old_python
conda create -n project_old_python python=3.4 numpy scipy pandas sortedcontainers

