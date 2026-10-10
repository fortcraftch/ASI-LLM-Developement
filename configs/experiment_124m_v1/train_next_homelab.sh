#!/usr/bin/env bash
# Run from repository root: bash configs/experiment_124m_v1/train_next_homelab.sh /path/to/data [--dry-run]
set -euo pipefail
data_root=${1:?Specify the full categorized dataset directory}
shift
# Same seed, batch and token schedule as the completed domain checkpoint.
# Fresh learned base: post-hoc classification will reuse this checkpoint.
export ASI_MOE_BACKEND=bmm
python -m asi train \
  --recipe configs/experiment_124m_v1/base_16_learned.json \
  --pool-manifest configs/experiment_124m_v1/expert_pools.json \
  --data-root "$data_root" --batch-size 8 --device cuda \
  --log-dir results/base_124m_seed1337 \
  --auto-restart --save-interval 25 --keep-checkpoints 3 \
  --max-restarts 5 --restart-delay 15 "$@"
