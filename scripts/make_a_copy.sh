#!/bin/bash
# Set dst repo here.
repo=$1
mkdir ../${repo}
cp ./*.py ../${repo}

echo "$(pwd) -> ../${repo}"

mkdir ../${repo}/scripts
cp ./scripts/*.sh ../${repo}/scripts

mkdir ../${repo}/configs
cp ./configs/*.py ../${repo}/configs

mkdir ../${repo}/utils
cp ./utils/*.py ../${repo}/utils

mkdir ../${repo}/losses
cp ./losses/*.py ../${repo}/losses

mkdir ../${repo}/models
cp ./models/*.py ../${repo}/models

mkdir ../${repo}/datasets
cp ./datasets/*.py ../${repo}/datasets

# mkdir ../${repo}/models/modules
# cp ./models/modules/*.py ../${repo}/models/modules

cp -r ./.git* ../${repo}
