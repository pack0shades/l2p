#!/bin/bash
# Job name:
#SBATCH --job-name=train_cifar100
# Partition:
#SBATCH --partition=btech # Use the appropriate partition name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1 # Define the number of GPUs

# Load required modules
module load python/3.10.pytorch

# Activate your conda environment (if applicable)
# source /csehome/b23es1024/.conda/envs/bin/activate l2p  # or any other environment name

# Run your PyTorch script with distributed launch
python image_embedding.py

        