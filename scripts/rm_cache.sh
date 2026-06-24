#!/bin/bash
# Remove training/eval/cache artifacts (never tracked by git). Output dirs (ckpts/, exp_*)
# also contain tf_log/ and training.log, removed with them.
rm -rf __pycache__ */__pycache__ */*/__pycache__

# Train / eval outputs
rm -rf ckpts/ exp_*/
rm -rf vis/ data/

# Eval feature cache / prediction dumps
rm -rf e_preds/ e_logs/

# Logs / core dumps
rm -f *.out nohup.out* nohup.log* slurm* core-*-python-*

clear
