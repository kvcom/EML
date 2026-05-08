#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p logs outputs data/db

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="logs/hetzner_optimisation_${RUN_STAMP}.log"
STUDY_NAME="${STUDY_NAME:-eml_hetzner}"
STORAGE="${STORAGE:-sqlite:///outputs/optuna_study.sqlite}"
TRIALS="${TRIALS:-500}"
MODE="${MODE:-fast}"
N_JOBS="${N_JOBS:-1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-21600}"
TOP="${TOP:-3}"
DB_PATH="${DB_PATH:-data/db/euromillions.sqlite}"

exec > >(tee -a "$RUN_LOG") 2>&1

echo "started_at=$(date --iso-8601=seconds)"
echo "root=$ROOT_DIR"
echo "run_log=$RUN_LOG"
echo "study_name=$STUDY_NAME"
echo "storage=$STORAGE"
echo "trials=$TRIALS"
echo "mode=$MODE"
echo "n_jobs=$N_JOBS"
echo "timeout_seconds=$TIMEOUT_SECONDS"

if [[ ! -d ".venv" ]]; then
  echo "ERROR: .venv not found. Create it and install dependencies before running this script."
  exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate

git fetch origin main
git checkout main
git pull --ff-only origin main

python -m pip install -e ".[dev]"

if [[ ! -s "$DB_PATH" ]]; then
  if [[ -f "data/All Numbers.xlsx" ]]; then
    EXCEL_PATH="data/All Numbers.xlsx"
  elif [[ -f "All Numbers.xlsx" ]]; then
    EXCEL_PATH="All Numbers.xlsx"
  else
    echo "ERROR: database missing and no historical Excel file found."
    exit 1
  fi
  echo "initialising database from $EXCEL_PATH"
  python -m euromillions.cli init-db --excel "$EXCEL_PATH"
fi

python -m euromillions.cli smoke-test

python -m euromillions.cli optimise \
  --study-name "$STUDY_NAME" \
  --storage "$STORAGE" \
  --trials "$TRIALS" \
  --mode "$MODE" \
  --n-jobs "$N_JOBS" \
  --timeout-seconds "$TIMEOUT_SECONDS" \
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
