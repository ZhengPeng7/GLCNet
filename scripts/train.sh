#!/bin/bash
# Usage: ./scripts/train.sh [gpu_ids] [method] [--cbgm]
#   Single-GPU: ./scripts/train.sh 0 exp1
#   Multi-GPU:  ./scripts/train.sh 0,1,2,3 exp1
#   --cbgm: use CBGM in the in-training epoch evals.
# Task is read from configs/task.py. Output: ckpts/${method}_${task}/
start_time=$(date +%s)

devices=${1:-"0"} && devices=${devices%%,} && devices=${devices##,}
method=${2:-$(basename "$(pwd)")}
cbgm_flag=""; [ "${3}" = "--cbgm" ] && cbgm_flag="--cbgm"
task=$(python3 -m configs --print_task)
nproc_per_node=$(($(echo ${devices} | grep -o "," | wc -l) + 1))
output_dir="ckpts/${method}_${task}"

echo "Training started at $(date)"
echo "Task: ${task} | GPUs: ${devices} (${nproc_per_node}-proc) | Output: ${output_dir} ${cbgm_flag}"

CUDA_VISIBLE_DEVICES=${devices} \
    torchrun --standalone --nproc_per_node ${nproc_per_node} \
    train.py --ckpt_dir ${output_dir} ${cbgm_flag}
train_status=$?

elapsed=$(( $(date +%s) - start_time ))
echo "Training finished at $(date) — elapsed $(date -ud @${elapsed} +'%H:%M:%S')"
exit ${train_status}
