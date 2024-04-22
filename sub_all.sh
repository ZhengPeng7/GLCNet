#!/bin/sh

devices=${1:-0}

sleep ${2:-0}h

srun --nodes=1 -p vip_gpu_ailab -A ai4bio \
--ntasks-per-node=1 \
--gres=gpu:$(($(echo ${devices%%,} | grep -o "," | wc -l)+1)) \
--cpus-per-task=32 \
bash run_all.sh ${devices}

hostname
