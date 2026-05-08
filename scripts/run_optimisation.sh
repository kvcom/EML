#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs outputs data/db

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="logs/optimisation_${RUN_STAMP}.log"
STUDY_NAME="${STUDY_NAME:-eml_hetzner}"
STORAGE="${STORAGE:-sqlite:///outputs/optuna_study.sqlite}"
TRIALS="${TRIALS:-5000}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-21600}"
N_JOBS="${N_JOBS:-1}"
MODE="${MODE:-fast}"
TOP="${TOP:-3}"

exec > >(tee -a "$RUN_LOG") 2>&1

echo "started_at=$(date --iso-8601=seconds)"
echo "root=$ROOT_DIR"
echo "run_log=$RUN_LOG"
echo "study_name=$STUDY_NAME"
echo "storage=$STORAGE"
echo "trials=$TRIALS"
echo "timeout_seconds=$TIMEOUT_SECONDS"
echo "n_jobs=$N_JOBS"
echo "mode=$MODE"
echo "top=$TOP"

if [[ ! -d ".venv" ]]; then
  echo "ERROR: .venv not found. Run scripts/bootstrap_hetzner.sh first."
  exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate

python -m euromillions.cli smoke-test

python -m euromillions.cli optimise \
  --study-name "$STUDY_NAME" \
  --storage "$STORAGE" \
  --trials "$TRIALS" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
  --n-jobs "$N_JOBS" \
  --mode "$MODE" \
  --top "$TOP"

python -m euromillions.cli predict --top "$TOP"

echo "final_predictions_json=outputs/predictions_latest.json"
python - <<'PY'
import json
from pathlib import Path

predictions = json.loads(Path("outputs/predictions_latest.json").read_text(encoding="utf-8"))
for row in predictions[:3]:
    mains = " ".join(f"{n:02d}" for n in row["mains"])
    stars = " ".join(f"{n:02d}" for n in row["stars"])
    print(f"Rank {row['rank']}: mains={mains} stars={stars} score={row['score']:.5f}")
PY

echo "finished_at=$(date --iso-8601=seconds)"
