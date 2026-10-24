#!/usr/bin/env bash
# ASI-matched architecture/data/schedule, free learned top-2 of 16 routing.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
data_root=${1:?Specify the same categorized dataset directory used by ASI}
shift
python - "$data_root" <<'PY'
import sys,json
from pathlib import Path
from asi.data.identity import manifest_matches
recipe=json.loads(Path('configs/experiment_124m_v1/base_16_learned.json').read_text())
path=Path(sys.argv[1])/'manifest.json'
if not path.is_file() or not manifest_matches(path,recipe['source_manifest_sha256']):
    raise SystemExit('Dataset manifest does not match ASI: use the original categorized dataset, not flat DeepSeek FineWeb shards.')
PY
export ASI_MOE_BACKEND=bmm
exec python -m asi train \
  --recipe configs/experiment_124m_v1/base_16_learned.json \
  --pool-manifest configs/experiment_124m_v1/expert_pools.json \
  --data-root "$data_root" --device cuda --seed 1337 \
  --batch-size 8 --pairing-audit \
  --log-dir results/deepseek16_asi_matched_seed1337 \
  --auto-restart --save-interval 25 --keep-checkpoints 3 \
  --max-restarts 5 --restart-delay 15 "$@"
