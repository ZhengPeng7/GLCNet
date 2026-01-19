#!/bin/bash

# Usage: ./scripts/train.sh [gpu_ids] [method]
# Example: ./scripts/train.sh 0 exp1
# Multi-GPU: ./scripts/train.sh 0,1,2,3 exp1
# Task is read from configs/task.py
# Output: ckpts/${method}/${task}/

devices=${1:-"0"}
method=${2:-"tmp"}
task=$(python3 -m configs --print_task)
nproc_per_node=$(($(echo ${devices%%,} | grep -o "," | wc -l)+1))

echo "Training started at $(date)"
echo "Task: ${task}, GPUs: ${devices}."

CUDA_VISIBLE_DEVICES=${devices} \
    torchrun --standalone --nproc_per_node ${nproc_per_node} \
    train.py --ckpt_dir ckpts/${method}_${task}
