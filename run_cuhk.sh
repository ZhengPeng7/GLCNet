# cuhk / prw / mvn
dataset="cuhk"

# Train
CUDA_VISIBLE_DEVICES=$1 python train.py --cfg configs/${dataset}.yaml

# Test

epoch=24

# CBGM
CUDA_VISIBLE_DEVICES=$1 python train.py --cfg ./exp_${dataset}/config.yaml --eval --ckpt ./exp_${dataset}/epoch_${epoch}.pth EVAL_USE_CBGM True

# using GT bbox
CUDA_VISIBLE_DEVICES=$1 python train.py --cfg ./exp_${dataset}/config.yaml --eval --ckpt ./exp_${dataset}/epoch_${epoch}.pth EVAL_USE_GT True

# CBGM, using GT bbox
CUDA_VISIBLE_DEVICES=$1 python train.py --cfg ./exp_${dataset}/config.yaml --eval --ckpt ./exp_${dataset}/epoch_${epoch}.pth EVAL_USE_CBGM True EVAL_USE_GT True


nvidia-smi
hostname
