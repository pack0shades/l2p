#!/bin/bash
# Job name:
#SBATCH --job-name=train_cifar100
# Partition:
#SBATCH --partition=small # Use the appropriate partition name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1 # Define the number of GPUs

# Load required modules
module load python/3.10.pytorch

# Activate your conda environment (if applicable)
source /csehome/b23es1024/.conda/envs/l2p/bin/activate # or any other environment name

# Run your PyTorch script with distributed launch
python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --use_env main.py \
          cifar100_l2p \
        --model vit_base_patch16_224 \
        --batch-size 16 \
        --data-path /local_datasets/ \
        --output_dir ./output  
