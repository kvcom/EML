# Hetzner Runbook

This is the operational runbook for running the EML EuroMillions optimiser on a Hetzner server.

## 1. Connect To The Server

From your local machine:

```bash
ssh root@YOUR_SERVER_IP
```

If you created a non-root user, use that instead:

```bash
ssh YOUR_USER@YOUR_SERVER_IP
```

## 2. Clone The Repository

On the Hetzner server:

```bash
git clone https://github.com/kvcom/EML.git
cd EML
```

If you prefer SSH deploy keys:

```bash
git clone git@github.com:kvcom/EML.git
cd EML
```

## 3. Add Required Data

The bootstrap script needs either an existing SQLite database or the historical Excel workbook.

Expected locations:

```text
data/db/euromillions.sqlite
data/All Numbers.xlsx
All Numbers.xlsx
```

If copying the Excel file from your local machine:

```bash
scp "All Numbers.xlsx" root@YOUR_SERVER_IP:/root/EML/
```

Adjust `/root/EML/` if you cloned somewhere else or use a non-root user.

## 4. Run Bootstrap

Bootstrap installs apt packages, creates `.venv`, installs the project, runs quality checks, verifies data, initialises the DB if needed, and runs the smoke test.

```bash
chmod +x scripts/bootstrap_hetzner.sh scripts/run_optimisation.sh
scripts/bootstrap_hetzner.sh
```

Successful bootstrap should end with:

```text
smoke-test: ok
Bootstrap complete
```

## 5. Start Optimisation In tmux

Use `tmux` so the optimiser continues if your SSH session disconnects.

```bash
tmux new -s eml
```

Inside tmux:

```bash
cd EML
TRIALS=5000 TIMEOUT_SECONDS=21600 N_JOBS=1 MODE=fast STUDY_NAME=eml_hetzner STORAGE=sqlite:///outputs/optuna_study.sqlite scripts/run_optimisation.sh
```

For a full evaluation run:

```bash
TRIALS=5000 TIMEOUT_SECONDS=21600 N_JOBS=1 MODE=full STUDY_NAME=eml_hetzner_full STORAGE=sqlite:///outputs/optuna_study.sqlite scripts/run_optimisation.sh
```

The script resumes the same Optuna study when `STUDY_NAME` and `STORAGE` are unchanged.

## 6. Detach And Reattach tmux

Detach without stopping the optimiser:

```text
Ctrl-b then d
```

List sessions:

```bash
tmux ls
```

Reattach:

```bash
tmux attach -t eml
```

## 7. Monitor Progress

Logs are written to:

```text
logs/optimisation_YYYYMMDD_HHMMSS.log
logs/optimise_YYYYMMDD_HHMMSS.log
```

Watch the latest optimisation log:

```bash
tail -f logs/optimise_*.log
```

Core outputs:

```text
outputs/optuna_study.sqlite
outputs/optimisation_report.json
outputs/best_params.json
outputs/predictions_latest.json
outputs/predictions_latest.csv
```

The optimisation report includes commit hash, draw count, latest draw date, config hash, start time, finish time, trial counts, and holdout metrics.

## 8. Copy Results Back

From your local machine:

```bash
mkdir -p eml_results
scp root@YOUR_SERVER_IP:/root/EML/outputs/optimisation_report.json eml_results/
scp root@YOUR_SERVER_IP:/root/EML/outputs/best_params.json eml_results/
scp root@YOUR_SERVER_IP:/root/EML/outputs/predictions_latest.json eml_results/
scp root@YOUR_SERVER_IP:/root/EML/outputs/predictions_latest.csv eml_results/
scp root@YOUR_SERVER_IP:/root/EML/outputs/optuna_study.sqlite eml_results/
```

To copy logs:

```bash
scp 'root@YOUR_SERVER_IP:/root/EML/logs/*.log' eml_results/
```

Adjust paths and usernames if needed.

## 9. Resume After Reboot Or Disconnect

SSH back in:

```bash
ssh root@YOUR_SERVER_IP
cd EML
tmux new -s eml
STUDY_NAME=eml_hetzner STORAGE=sqlite:///outputs/optuna_study.sqlite scripts/run_optimisation.sh
```

Do not delete `outputs/optuna_study.sqlite`; it is the Optuna checkpoint.

## 10. Safely Delete The Hetzner Server

Before deletion, copy back anything you need from `outputs/` and `logs/`.

Then either delete it in the Hetzner Cloud Console, or use the Hetzner CLI from your local machine:

```bash
hcloud server list
hcloud server shutdown SERVER_NAME_OR_ID
hcloud server delete SERVER_NAME_OR_ID
```

Confirm the server is gone:

```bash
hcloud server list
```

Only delete the server after confirming `outputs/optuna_study.sqlite`, `outputs/optimisation_report.json`, and `outputs/predictions_latest.*` have been copied back.
