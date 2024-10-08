#!/bin/bash


# Load required modules
# module load python/3.10.pytorch

# Activate your conda environment (if applicable)
# source /csehome/b23es1024/.conda/bin/activate l2p  # or any other environment name

# Run your PyTorch script with distributed launch
python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --use_env main.py \
          five_datasets_l2p \
        --model vit_base_patch16_224 \
        --batch-size 16 \
        --output_dir ./output \
        --epochs 8 \
        --size 10 \
        --embedding_key 'mean_max' \
        --top_k 8 \
        --prompt_key_init 'normal' \
        --prompt_init 'normal' \
        --shared_prompt_pool False \
        --shared_prompt_key False \
        --head_type 'prompt'