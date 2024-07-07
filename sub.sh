#!/bin/sh
module load compilers/cuda/11.1  compilers/gcc/9.3.0
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/miniconda3/lib:/home/bingxing2/apps/cudnn/8.2.1.32_cuda11.x/lib64

script=${1:-"./run_all.sh"}

devices=${2:-0}

sbatch --nodes=1 -p vip_gpu_ailab -A ai4bio \
--ntasks-per-node=1 \
--gres=gpu:$(($(echo ${devices%%,} | grep -o "," | wc -l)+1)) \
--cpus-per-task=32 \
${script} ${devices}

hostname
