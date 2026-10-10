#!/usr/bin/env bash
# Run from the repository root. Repeating this command resumes the collection.
set -euo pipefail
data_root=${1:?Specify the full categorized dataset directory}
shift
python -u -m asi train-bank --data-root "$data_root" \
  --output results/router_124m_seed1337 --batch-size 8 --device cuda \
  --resume --save-interval 25 --keep-checkpoints 3 \
  --max-restarts 5 --restart-delay 15 "$@"
