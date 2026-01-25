#!/bin/bash
# Example: `./scripts/test.sh 0 $(basename "$(pwd)")`
start_time=$(date +%s)

devices=${1:-"0"}
method=${2:-$(basename "$(pwd)")}
task=$(python3 -m configs --print_task)

ckpt="ckpts/${method}_${task}/epoch_best.pth"

echo "Testing: Task=${task}, GPU=${devices}, Checkpoint=${ckpt}"

CUDA_VISIBLE_DEVICES=${devices} \
    python train.py --eval --ckpt ${ckpt} --cbgm

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Testing finished in ${elapsed}s ($(date -ud @${elapsed} +'%H:%M:%S'))"
