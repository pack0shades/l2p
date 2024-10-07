#!/bin/bash


# Load required modules
# module load python/3.10.pytorch

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
        --epochs 8 \
        --prompt_key_init 'normal' \
        --embedding_key 'mean_max' \
        --model 'vit_large_patch16_224' \
        --top_k 8 


        python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path /scratch/b23es1024/l2p-pytorch/local_datasets/ --output_dir ./output --epochs 8 --prompt_key_init 'normal' --embedding_key 'mean_max' --model 'vit_large_patch16_224'--top_k 8