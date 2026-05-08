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
- `python -m euromillions.cli optimise --trials 500 --top 3`
- `python -m euromillions.cli predict --top 3`
- `python -m euromillions.cli run`

## What each command does

- `init-db`: creates SQLite schema, ingests historical draws from Excel, and builds all main/star combinations.
- `update-results`: fetches latest observations from configured public sources and inserts safe reconciled updates.
- `backtest`: runs walk-forward evaluation with no data leakage and random baseline comparison.
- `optimise`: tunes weighted model parameters with Optuna and reports holdout performance.
- `predict`: updates data, refreshes features incrementally, and outputs top-N ranked predictions to terminal/JSON/CSV.
- `run`: convenience pipeline for update + feature refresh + prediction.

## Data sources

Source adapters are under `src/euromillions/sources/`.  
To add a new source, implement the protocol in `base.py`, add parser tests, and include the adapter in source registry.
