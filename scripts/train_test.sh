#!/bin/bash
# Example: `./scripts/train_test.sh 0,1 3 $(basename "$(pwd)") 0`

devices=${1:-"0"} && devices=${devices%%,} && devices=${devices##,}
N_RUNS=${2:-1}
task_name=${3:-$(basename "$(pwd)")}
devices_test=${4:-${devices%%,*}}

for run_id in $(seq 0 $((N_RUNS - 1))); do
    method="${task_name}$( [ ${run_id} -gt 0 ] && echo "_${run_id}" )"
    bash scripts/train.sh ${devices} ${method}
    bash scripts/test.sh ${devices_test} ${method}
done

hostname
