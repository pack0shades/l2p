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
# source /csehome/b23es1024/.conda/bin/activate l2p  # or any other environment name

# Run your PyTorch script with distributed launch
python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --use_env main.py \
          cifar100_l2p \
        --model vit_base_patch16_224 \
        --batch-size 16 \
        --data-path /scratch/b23es1024/l2p-pytorch/local_datasets/ \
        --output_dir ./output \
        --epochs 5 \
        --size 20 \
        --embedding_key cls \
        --top_k 4 \
        --prompt_key_init uniform \
        --prompt_init uniform \
        --shared_prompt_pool True \
        --shared_prompt_key False \
        --head_type prompt

        