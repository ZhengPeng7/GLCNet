#!/bin/bash

# cuhk / prw / mvn
dataset=${1:-"prw"}
devices=${2:-"0"}
nproc_per_node=$(($(echo ${devices%%,} | grep -o "," | wc -l)+1))
to_be_distributed=`echo ${nproc_per_node} | awk '{if($e > 1) print "True"; else print "False";}'`


echo Training started at $(date)

if [ ${to_be_distributed} == "True" ]
then
    echo "Multi-GPU mode received..."
    CUDA_VISIBLE_DEVICES=${devices} \
    torchrun --standalone --nproc_per_node=${nproc_per_node} --nnodes=1 \
        train.py \
        --cfg configs/${dataset}.yaml
else
    echo "Single-GPU mode received..."
    CUDA_VISIBLE_DEVICES=${devices} \
    python \
        train.py \
        --cfg configs/${dataset}.yaml
fi

# Test w/ CBGM
python train.py --cfg ./exp_${dataset}/config.yaml --eval --ckpt ./exp_${dataset}/epoch_best.pth EVAL_USE_CBGM True
