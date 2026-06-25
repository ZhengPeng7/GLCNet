#!/bin/bash
# Evaluate a GLCNet checkpoint. Unlike BiRefNet2, GLCNet eval.py performs GPU
# inference and metric computation in one process, so callers must avoid
# same-GPU overlap with training.
# Usage: ./scripts/eval.sh [gpu] [method] [which] [--no-cbgm] [--use_cache] [--save_cache]
start_time=$(date +%s)

devices=${1:-"0"}
method=${2:-$(basename "$(pwd)")}
which=${3:-best}
cbgm_flag="--cbgm"
cache_flag=""
save_cache_flag=""
for arg in "${@:4}"; do
    [ "${arg}" = "--no-cbgm" ] && cbgm_flag=""
    [ "${arg}" = "--use_cache" ] && cache_flag="--use_cache"
    [ "${arg}" = "--save_cache" ] && save_cache_flag="--save_cache"
done
task=$(python3 -m configs --print_task)
ckpt_dir="ckpts/${method}_${task}"

case "${which}" in
    best)   ckpt="${ckpt_dir}/epoch_best.pth" ;;
    last)   ckpt="${ckpt_dir}/epoch_last.pth" ;;
    latest) ckpt=$(ls -t ${ckpt_dir}/*.pth 2>/dev/null | head -1) ;;
    *)      ckpt="${which}" ;;
esac

if [ ! -f "${ckpt}" ]; then
    ckpt="${ckpt_dir}/epoch_best.pth"
    [ -f "${ckpt}" ] || ckpt="${ckpt_dir}/epoch_last.pth"
    [ -f "${ckpt}" ] || ckpt=$(ls -t ${ckpt_dir}/*.pth 2>/dev/null | head -1)
fi
[ -f "${ckpt}" ] || { echo "No checkpoint found (requested '${which}' under ${ckpt_dir})" >&2; exit 1; }

echo "Evaluating: Task=${task} | GPU=${devices} | Checkpoint=${ckpt} ${cbgm_flag} ${cache_flag} ${save_cache_flag}"
CUDA_VISIBLE_DEVICES=${devices} python eval.py --ckpt "${ckpt}" ${cbgm_flag} ${cache_flag} ${save_cache_flag}
eval_status=$?

elapsed=$(( $(date +%s) - start_time ))
echo "Evaluation finished at $(date) — elapsed $(date -ud @${elapsed} +'%H:%M:%S')"
exit ${eval_status}
