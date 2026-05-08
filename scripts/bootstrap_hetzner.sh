#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p data/db logs outputs

if command -v sudo >/dev/null 2>&1; then
  SUDO="sudo"
else
  SUDO=""
fi

echo "Installing apt dependencies"
$SUDO apt-get update
$SUDO apt-get install -y \
  build-essential \
  ca-certificates \
  curl \
  git \
  libsqlite3-dev \
  python3 \
  python3-dev \
  python3-pip \
  python3-venv \
  tmux

PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" - <<'PY'
import sys

if sys.version_info < (3, 12):
    raise SystemExit(f"Python 3.12+ required, found {sys.version.split()[0]}")
PY

if [[ ! -d ".venv" ]]; then
  echo "Creating .venv"
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck source=/dev/null
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"

DB_PATH="${DB_PATH:-data/db/euromillions.sqlite}"
EXCEL_PATH=""
if [[ -f "data/All Numbers.xlsx" ]]; then
  EXCEL_PATH="data/All Numbers.xlsx"
elif [[ -f "All Numbers.xlsx" ]]; then
  EXCEL_PATH="All Numbers.xlsx"
fi

if [[ ! -s "$DB_PATH" && -z "$EXCEL_PATH" ]]; then
  echo "ERROR: required data missing. Provide $DB_PATH or All Numbers.xlsx/data/All Numbers.xlsx."
  exit 1
fi

python -m pytest
python -m ruff check .
python -m mypy src

if [[ ! -s "$DB_PATH" ]]; then
  echo "Initialising database from $EXCEL_PATH"
  python -m euromillions.cli init-db --excel "$EXCEL_PATH"
else
  echo "Database already exists at $DB_PATH"
fi

python -m euromillions.cli smoke-test

echo "Bootstrap complete"
