#!/bin/bash

#SBATCH --job-name=debug
#SBATCH --account=amath
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1G
#SBATCH --gpus=0
#SBATCH --time=00-00:10:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --export=all
#SBATCH --output=slurm.out # where STDOUT goes
#SBATCH --error=slurm.err # where STDERR goes
# Modules to use (optional).
# <e.g., module load singularity>

# Your programs to run.
g++ -std=c++14 -o main.o main.cpp;

./main.o;

rm *.o
