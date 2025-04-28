#!/bin/bash
#SBATCH --job-name=nbody_cuda
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=nbody_cuda.out
#SBATCH --error=nbody_cuda.err

# Load CUDA module
module load cuda/12.4

# Run your program
./nbody_cuda 100000 0.01 50 101 128

