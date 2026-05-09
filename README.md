# EuroMillions Predictor

Production-oriented Python application for analysing historical EuroMillions draws, running leakage-safe walk-forward backtests, and producing ranked candidate combinations.

## Caution

EuroMillions is a random lottery. This project does not claim to beat randomness reliably.  
It ranks statistically preferred combinations from historical patterns and benchmarks performance against random baselines.

## Commands

- `python -m euromillions.cli init-db --excel "data/All Numbers.xlsx"`
- `python -m euromillions.cli update-results`
- `python -m euromillions.cli build-combinations`
- `python -m euromillions.cli backtest --top 3`
- `python -m euromillions.cli smoke-test`
- `python -m euromillions.cli optimise --study-name eml_main --storage sqlite:///outputs/optuna_study.sqlite --trials 500 --mode fast --timeout-seconds 21600 --objective exact-rank`
- `python -m euromillions.cli predict --top 3`
- `python -m euromillions.cli rank-history --mode fast --thresholds 1,3,10,100,500,1000,3000`
- `python -m euromillions.cli run`

## What each command does

- `init-db`: creates SQLite schema, ingests historical draws from Excel, and builds all main/star combinations.
- `update-results`: fetches latest observations from configured public sources and inserts safe reconciled updates.
- `backtest`: runs walk-forward evaluation with no data leakage and random baseline comparison.
- `smoke-test`: validates the local environment, database, one fast walk-forward round, and one prediction before expensive runs.
- `optimise`: resumes or creates a persistent Optuna study, logs progress to `logs/`, records run metadata, and reports holdout performance. The default objective is exact full-ticket historical rank; use `--objective top-k --top 3` for the older Top-K hit objective.
- `predict`: updates data, refreshes features incrementally, and outputs top-N ranked predictions to terminal/JSON/CSV.
- `rank-history`: ranks historical winning tickets using only prior draws and writes exact rank bucket reports to JSON/CSV.
- `run`: convenience pipeline for update + feature refresh + prediction.

## Hetzner

Use `scripts/bootstrap_hetzner.sh` to prepare a new server, then `scripts/run_optimisation.sh` for long-running optimisation. The run script resumes persistent SQLite-backed Optuna optimisation, writes logs, runs prediction, and prints the final top 3 predictions.

Recovery instructions are in `docs/hetzner_runbook.md`.

Optimisation objective guidance is in `docs/optimisation_objective.md`.

## Data sources

Source adapters are under `src/euromillions/sources/`.  
To add a new source, implement the protocol in `base.py`, add parser tests, and include the adapter in source registry.
