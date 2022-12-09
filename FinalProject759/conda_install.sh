#!/usr/bin/env bash
#SBATCH -t 0-00:30:00
#SBATCH -p wacc
#SBATCH -J conda
#SBATCH -o conda.out -e conda.err
#SBATCH -c 1

bash Anaconda3-2022.10-Linux-x86_64.sh  < <(yes $'yes\n')
