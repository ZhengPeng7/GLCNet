#!/bin/bash

# Usage: ./scripts/test.sh [gpu_id] [method] [--cbgm]
# Example: ./scripts/test.sh 0 exp1
# With CBGM: ./scripts/test.sh 0 exp1 --cbgm
# Task is read from configs/task.py
# Checkpoint: ckpts/${method}/${task}/epoch_best.pth

start_time=$(date +%s)

devices=${1:-"0"}
method=${2:-"GLCNet"}
cbgm_flag=${3:-""}
task=$(python3 -m configs --print_task)

ckpt="ckpts/${method}_${task}/epoch_last.pth"

echo "Testing: Task=${task}, GPU=${devices}, Checkpoint=${ckpt}"

CUDA_VISIBLE_DEVICES=${devices} \
    python train.py --eval --ckpt ${ckpt} ${cbgm_flag}

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Testing finished in ${elapsed}s ($(date -ud @${elapsed} +'%H:%M:%S'))"
