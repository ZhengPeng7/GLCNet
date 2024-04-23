#!/bin/sh

# cuhk / prw / mvn
dataset="mvn"

# Train
CUDA_VISIBLE_DEVICES=${1:-0} python train.py --cfg configs/${dataset}.yaml

# # Test w/ CBGM
# CUDA_VISIBLE_DEVICES=${1:-0} python train.py --cfg ./exp_${dataset}/config.yaml --eval --ckpt ./exp_${dataset}/epoch_best.pth EVAL_USE_CBGM False
