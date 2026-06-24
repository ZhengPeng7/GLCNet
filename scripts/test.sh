#!/bin/bash
# Test = evaluate (GLCNet's eval.py runs inference over gallery+query AND computes mAP/top-1 in one pass).
# Usage: ./scripts/test.sh [gpu] [method] [which] [--no-cbgm] [--use_cache]
#   which = best (default) | last | latest | <explicit .pth path>
# Examples:
#   ./scripts/test.sh 0                                   # best ckpt of current method, CBGM on
#   ./scripts/test.sh 0 exp1 last                         # epoch_last of exp1
#   ./scripts/test.sh 0 exp1 path/to.pth --no-cbgm        # explicit ckpt, no CBGM
#   ./scripts/test.sh 0 exp1 best --use_cache             # reuse cached gallery features
start_time=$(date +%s)

devices=${1:-"0"}
method=${2:-$(basename "$(pwd)")}
which=${3:-best}
cbgm_flag="--cbgm"; cache_flag=""
for arg in "${@:4}"; do
    [ "${arg}" = "--no-cbgm" ] && cbgm_flag=""
    [ "${arg}" = "--use_cache" ] && cache_flag="--use_cache"
done
task=$(python3 -m configs --print_task)
ckpt_dir="ckpts/${method}_${task}"

case "${which}" in
    best)   ckpt="${ckpt_dir}/epoch_best.pth" ;;
    last)   ckpt="${ckpt_dir}/epoch_last.pth" ;;
    latest) ckpt=$(ls -t ${ckpt_dir}/*.pth 2>/dev/null | head -1) ;;
    *)      ckpt="${which}" ;;   # explicit path
esac
# Fall back best -> last -> latest if the requested checkpoint is missing.
if [ ! -f "${ckpt}" ]; then
    ckpt="${ckpt_dir}/epoch_best.pth"
    [ -f "${ckpt}" ] || ckpt="${ckpt_dir}/epoch_last.pth"
    [ -f "${ckpt}" ] || ckpt=$(ls -t ${ckpt_dir}/*.pth 2>/dev/null | head -1)
fi
[ -f "${ckpt}" ] || { echo "No checkpoint found (requested '${which}' under ${ckpt_dir})"; exit 1; }

echo "Testing: Task=${task} | GPU=${devices} | Checkpoint=${ckpt} ${cbgm_flag} ${cache_flag}"
CUDA_VISIBLE_DEVICES=${devices} python eval.py --ckpt ${ckpt} ${cbgm_flag} ${cache_flag}

elapsed=$(( $(date +%s) - start_time ))
echo "Testing finished in $(date -ud @${elapsed} +'%H:%M:%S')"
