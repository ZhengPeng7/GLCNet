#!/bin/bash
# Runs INSIDE the submitted cluster job (see sub.sh): set up the H-cluster env, then train+test.
devices=${1}
N_RUNS=${2:-1}
task_name=${3:-$(basename "$(pwd)")}

p2p=/mnt/shared-storage-gpfs2/single-cell/zhengpeng
source ${p2p}/scripts/append_me_to_bashrc.sh

conda activate cv
./scripts/train_test.sh ${devices} ${N_RUNS} ${task_name}

hostname
