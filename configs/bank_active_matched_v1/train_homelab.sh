#!/usr/bin/env bash
set -euo pipefail
data_root=${1:?Specify the categorized dataset directory (e.g. results/dataset_trial)}
shift
python -u -m asi train-bank --config-dir configs/bank_active_matched_v1 \
  --data-root "$data_root" --output results/bank_active_matched_seed1337 \
  --batch-size 2 --device cuda --resume --save-interval 25 \
  --keep-checkpoints 3 --max-restarts 5 --restart-delay 15 "$@"
