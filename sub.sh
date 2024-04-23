#!/bin/sh
export PYTHONUNBUFFERED=1

script=${1:-"./run_all.sh"}

devices=${2:-0}

sbatch --nodes=1 -p vip_gpu_ailab -A ai4bio \
--ntasks-per-node=1 \
--gres=gpu:$(($(echo ${devices%%,} | grep -o "," | wc -l)+1)) \
--cpus-per-task=32 \
${script} ${devices}

hostname
