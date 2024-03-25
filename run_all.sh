sleep ${2:-0}h

# cuhk / prw / mvn
dataset="prw"

# Train
CUDA_VISIBLE_DEVICES=${1:-0} python train.py --cfg configs/${dataset}.yaml


# cuhk / prw / mvn
dataset="cuhk"

# Train
CUDA_VISIBLE_DEVICES=${1:-0} python train.py --cfg configs/${dataset}.yaml
