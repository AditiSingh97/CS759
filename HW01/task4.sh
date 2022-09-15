#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -J FirstSlurm
#SBATCH -o FirstSlurm.out -e FirstSlurm.err
#SBATCH -c 2

hostname
