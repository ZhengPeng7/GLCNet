#!/bin/bash
# Example: ./scripts/train_test.sh 0 2 $(basename "$(pwd)")
# Optional dedicated eval GPU: ./scripts/train_test.sh 0 2 exp_name 3
# If the 4th arg is omitted, all training runs finish first, then eval is queued.

devices=${1:-"0"} && devices=${devices%%,} && devices=${devices##,}
N_RUNS=${2:-1}
task_name=${3:-$(basename "$(pwd)")}
eval_devices=${4:-}
EVAL_BLOCKING="${EVAL_BLOCKING:-0}"   # 1=sub/rjob-safe foreground wait; 0=local background wait.
EVAL_WHICH="${EVAL_WHICH:-best}"

if [ -n "${eval_devices}" ]; then
    eval_devices=${eval_devices%%,} && eval_devices=${eval_devices##,}
    train_list=",${devices},"
    IFS=',' read -ra eval_list <<< "${eval_devices}"
    for gpu in "${eval_list[@]}"; do
        if [ -n "${gpu}" ] && [[ "${train_list}" == *",${gpu},"* ]]; then
            echo "Dedicated eval GPU(s) must be disjoint from train GPU(s): train=${devices}, eval=${eval_devices}" >&2
            exit 2
        fi
    done
fi

rm -rf .eval_done && mkdir -p .eval_done
methods=()
prev_eval_method=""

launch_eval() {
    local method=$1
    local prev_method=$2
    (
        if [ -n "${prev_method}" ]; then
            until [ -f ".eval_done/${prev_method}" ]; do sleep 5; done
        fi
        echo "[train_test] eval ${method} starting on GPU(s) ${eval_devices} at $(date)"
        bash scripts/eval.sh "${eval_devices}" "${method}" "${EVAL_WHICH}"
        st=$?
        echo "${st}" > ".eval_done/${method}"
        [ ${st} -eq 0 ] || echo "[train_test] eval ${method} failed (${st})." >&2
        exit ${st}
    ) &
}

wait_evals_then_report() {
    local expected=$1
    local status=0
    while [ "$(find .eval_done -maxdepth 1 -type f | wc -l)" -lt "${expected}" ]; do sleep 10; done
    for marker in .eval_done/*; do
        [ -f "${marker}" ] || continue
        marker_status=$(cat "${marker}")
        [ "${marker_status}" = "0" ] || status=${marker_status}
    done
    rm -rf .eval_done
    return ${status}
}

echo "[train_test] train GPUs=${devices}; dedicated_eval GPUs=${eval_devices:-none}; EVAL_BLOCKING=${EVAL_BLOCKING}"
for run_id in $(seq 0 $((N_RUNS - 1))); do
    method="${task_name}$( [ ${run_id} -gt 0 ] && echo "_${run_id}" )"
    bash scripts/train.sh "${devices}" "${method}" || { st=$?; echo "[train_test] train ${method} failed (${st})." >&2; exit ${st}; }
    methods+=("${method}")
    if [ -n "${eval_devices}" ]; then
        launch_eval "${method}" "${prev_eval_method}"
        prev_eval_method="${method}"
    fi
done

if [ -z "${eval_devices}" ]; then
    eval_devices=${devices%%,*}
    for method in "${methods[@]}"; do
        launch_eval "${method}" "${prev_eval_method}"
        prev_eval_method="${method}"
    done
fi

if [ "${EVAL_BLOCKING}" = "1" ]; then
    wait_evals_then_report "${#methods[@]}"
    eval_status=$?
    [ ${eval_status} -eq 0 ] || exit ${eval_status}
else
    wait_evals_then_report "${#methods[@]}" &
fi

hostname
