#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:v100:1
#SBATCH --time=48:00:00


# cuhk / prw / mvn
dataset="mvn"

# Keep consistant with N in sub_prw.sh
NGPU=1

if [ ${dataset} == "cuhk" ]
    then let epoch=18*${NGPU}-1
elif [ ${dataset} == "prw" ]
    then let epoch=18*${NGPU}-1
else let epoch=20*${NGPU}-1
fi

# Train
CUDA_VISIBLE_DEVICES=$1 python train.py --cfg configs/${dataset}.yaml

# Test

# CBGM
CUDA_VISIBLE_DEVICES=$1 python train.py --cfg ./exp_${dataset}/config.yaml --eval --ckpt ./exp_${dataset}/epoch_${epoch}.pth EVAL_USE_CBGM True

# using GT bbox
CUDA_VISIBLE_DEVICES=$1 python train.py --cfg ./exp_${dataset}/config.yaml --eval --ckpt ./exp_${dataset}/epoch_${epoch}.pth EVAL_USE_GT True

# CBGM, using GT bbox
CUDA_VISIBLE_DEVICES=$1 python train.py --cfg ./exp_${dataset}/config.yaml --eval --ckpt ./exp_${dataset}/epoch_${epoch}.pth EVAL_USE_CBGM True EVAL_USE_GT True


nvidia-smi
hostname
