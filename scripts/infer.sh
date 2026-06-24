#!/bin/bash
# Person-search inference / visualization on arbitrary images via demo.py (single-GPU).
# Usage: ./scripts/infer.sh [gpu] [method|ckpt] [query] [query_box] [gallery] [out_dir]
#   method|ckpt : an explicit .pth path, or a method name (resolves best/last/latest under ckpts/${method}_${task})
#   query_box  : 'x1,y1,x2,y2'    gallery : a directory or glob pattern
# Examples:
#   ./scripts/infer.sh 0                                              # bundled demo_imgs with the best ckpt
#   ./scripts/infer.sh 0 exp1 my/q.jpg 10,20,120,400 my/gallery e_preds
start_time=$(date +%s)

devices=${1:-"0"}
method=${2:-$(basename "$(pwd)")}
query=${3:-demo_imgs/query.jpg}
query_box=${4:-"0,0,466,943"}
gallery=${5:-demo_imgs/gallery-*.jpg}
out_dir=${6:-}
task=$(python3 -m configs --print_task)

# Resolve ckpt: explicit .pth path, else best -> last -> latest under ckpts/${method}_${task}.
if [ -f "${method}" ]; then
    ckpt="${method}"
else
    ckpt_dir="ckpts/${method}_${task}"
    ckpt="${ckpt_dir}/epoch_best.pth"
    [ -f "${ckpt}" ] || ckpt="${ckpt_dir}/epoch_last.pth"
    [ -f "${ckpt}" ] || ckpt=$(ls -t ${ckpt_dir}/*.pth 2>/dev/null | head -1)
fi
[ -f "${ckpt}" ] || { echo "No checkpoint found ('${method}')"; exit 1; }

out_arg=""; [ -n "${out_dir}" ] && out_arg="--out_dir ${out_dir}"
echo "Inferring: GPU=${devices} | Ckpt=${ckpt} | Query=${query} | Gallery=${gallery} ${out_arg}"
CUDA_VISIBLE_DEVICES=${devices} python demo.py \
    --ckpt "${ckpt}" --query "${query}" --query_box "${query_box}" --gallery "${gallery}" ${out_arg}

elapsed=$(( $(date +%s) - start_time ))
echo "Inference finished in $(date -ud @${elapsed} +'%H:%M:%S')"
