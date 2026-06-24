#!/bin/bash
# Submit a GLCNet train+test job to the H cluster via rjob.
# Example: bash scripts/sub.sh 2 small 1   (2 GPUs, ma4agismall group, 1 run)
GPU_num=${1:-1}
group_suffix=${2:-'small'}
group_GPU=ma4agi${group_suffix}_gpu
N_RUNS=${3:-1}
work_dir=${4:-$(pwd)}
task_name=$(basename "${work_dir}")
devices=$(seq -s, 0 $(( GPU_num - 1 )))
rjob_name=rjob-p-glc


rjob submit --name=${rjob_name} --gpu=${GPU_num} --memory=$(( GPU_num * 200000)) --cpu=$(( GPU_num * 16 )) \
    --charged-group=${group_GPU} --task-type=normal \
    --priority=9 --backoff_limit 5 \
    --mount=gpfs://gpfs1/ma4agi-gpu:/mnt/shared-storage-user/ma4agi-gpu \
    --mount=gpfs://gpfs2/single-cell:/mnt/shared-storage-gpfs2/single-cell \
    --image=registry.h.pjlab.org.cn/ailab-ma4agismall-ma4agismall_cpu/peng-p:peng-dev-p-20260524213801 \
    --enable-sshd --private-machine=group --custom-resources brainpp.cn/fuse=1 \
    --auto-restart=always --gang-start=true -e DISTRIBUTED_JOB=true \
    -- bash -exc "
        cd ${work_dir}
        bash ./scripts/sub_inner.sh ${devices} ${N_RUNS} ${task_name}
    "
