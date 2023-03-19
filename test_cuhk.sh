#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:v100:1
#SBATCH --time=48:00:00


dataset="cuhk"
epoch=18

# Test
# CBGM
python train.py --cfg ./exp_${dataset}/config.yaml --eval --ckpt ./exp_${dataset}/epoch_${epoch}.pth
python train.py --cfg ./exp_${dataset}/config.yaml --eval --ckpt ./exp_${dataset}/epoch_${epoch}.pth EVAL_USE_CBGM True


nvidia-smi
hostname
