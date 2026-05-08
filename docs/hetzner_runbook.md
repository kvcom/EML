# Hetzner Optimisation Runbook

This runbook is for long-running EuroMillions optimisation on a Hetzner server.

## First-Time Setup

1. Clone the repository and enter it.

   ```bash
   git clone git@github.com:kvcom/EML.git
   cd EML
   ```

2. Create the virtual environment.

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e ".[dev]"
   ```

3. Put the historical Excel file at either `All Numbers.xlsx` or `data/All Numbers.xlsx`.

## Recommended Run

Run inside `tmux` or `screen` so the terminal can disconnect safely.

```bash
tmux new -s eml
TRIALS=5000 MODE=fast N_JOBS=1 TIMEOUT_SECONDS=21600 scripts/run_hetzner_optimisation.sh
```

The script will:

- activate `.venv`
- pull latest `main`
- install the package
- initialise the SQLite draw database if needed
- run `python -m euromillions.cli smoke-test`
- run persistent Optuna optimisation
- run `predict --top 3`
- print the final top 3 predictions

## Persistent State

Optuna state is stored by default in:

```text
outputs/optuna_study.sqlite
```

The default study name is:

```text
eml_hetzner
```

To resume after a reboot or SSH disconnection, run the same command again with the same `STUDY_NAME` and `STORAGE`. The optimiser uses `load_if_exists=True` and continues the existing study.

## Logs And Outputs

Run logs:

```text
logs/hetzner_optimisation_YYYYMMDD_HHMMSS.log
logs/optimise_YYYYMMDD_HHMMSS.log
```

Core outputs:

```text
outputs/optimisation_report.json
outputs/best_params.json
outputs/predictions_latest.json
outputs/predictions_latest.csv
```

The optimisation report includes:

- commit hash
- draw count
- latest draw date
- config hash
- started_at
- finished_at
- study name and storage
- existing, new, and completed trial counts
- holdout metrics against random baseline

## Recovery

If the server reboots:

```bash
cd EML
source .venv/bin/activate
python -m euromillions.cli smoke-test
STUDY_NAME=eml_hetzner STORAGE=sqlite:///outputs/optuna_study.sqlite scripts/run_hetzner_optimisation.sh
```

If the run stops during optimisation, do not delete `outputs/optuna_study.sqlite`. That file is the checkpoint.

If the repository changed during a run, inspect the current report before resuming:

```bash
cat outputs/optimisation_report.json
git rev-parse HEAD
```

Resume only when you are comfortable continuing the same study under the current code/config. For a clean independent run, choose a new `STUDY_NAME` or a new SQLite storage file.

## Useful Commands

Fast local validation:

```bash
python -m euromillions.cli smoke-test
python -m pytest
python -m ruff check .
python -m mypy src
```

Manual optimisation:

```bash
python -m euromillions.cli optimise \
  --study-name eml_hetzner \
  --storage sqlite:///outputs/optuna_study.sqlite \
  --trials 5000 \
  --mode fast \
  --n-jobs 1 \
  --timeout-seconds 21600
```

Manual prediction:

```bash
python -m euromillions.cli predict --top 3
```
