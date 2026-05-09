# Optimisation objective notes

Measured on 2026-05-09 against commit `9ae36ec454a23bc24188fe0c7f7c1f60d4a45618`
with 1,943 draws through 2026-05-05.

## Timing baseline

- `smoke-test`: about 1.0s.
- Fast Top-3 backtest, 175 rounds: about 4.2s.
- One exact historical rank round: about 4.3s.
- Optuna Top-3, 1 trial plus holdout: about 12.9s.
- Optuna Top-3, 10 trials plus holdout: about 66.7s.
- Optuna Top-1000, 1 trial plus holdout: about 122.6s.
- Optuna Top-1000, 10 requested trials with a 180s timeout: 7 trials completed
  in about 213.4s including holdout.

## Why Top-1000 optimisation is slow

`optimise` evaluates each trial through `run_walk_forward`, which calls
`generate_predictions(history, top=top)` for every evaluated draw.

For `top=1000`, `generate_predictions` derives a model candidate pool from
`top * candidate_pool_multiplier`. Optuna currently samples the multiplier from
50 to 200, so a Top-1000 trial can request 50,000 to 200,000 candidates for
each walk-forward round. The ensemble then repeatedly ranks large weighted and
Bayesian candidate sets, applies diversity constraints, and repeats that work
for roughly 125 to 145 fast-mode rounds per trial plus holdout rounds.

This is exact deterministic prediction-output work, not exact historical
rank-by-counting work.

## Current architecture

Use exact full-ticket rank as the default Optuna objective:

```powershell
python -m euromillions.cli optimise --objective exact-rank
```

This objective preserves exact rank semantics and maximises negative average
rank, so lower historical winner ranks are better.

Only scoring parameters are sampled for exact-rank optimisation. Prediction
output parameters such as candidate-pool size and diversity constraints are
left at defaults because they do not affect full-ticket score ranking.

Top-3 remains available when the goal is specifically to tune the small final
prediction set:

```powershell
python -m euromillions.cli optimise --objective top-k --top 3
```

Exact-rank runs can use validation early stopping:

```powershell
python -m euromillions.cli optimise --objective exact-rank --early-stop-patience 1 --early-stop-validation-rounds 10
```

This does not stop on the final holdout. It reserves a validation slice before
the final holdout, checks average exact rank after each completed trial, and
stops when validation rank does not improve for the configured patience. The
final holdout remains a last unbiased check.

## Long-run monitoring

Every optimisation writes lightweight monitor files:

- `outputs/optimisation_progress.json`: current status, active trial number,
  completed trial count, best value, and rough remaining-time estimate based on
  the last completed trial.
- `outputs/optimisation_trials.csv`: trial history for the current invocation
  with value, best value, duration, and parameters for each completed trial.

On Linux servers, watch progress with:

```bash
watch -n 5 cat outputs/optimisation_progress.json
tail -f outputs/optimisation_trials.csv
```

On Windows PowerShell:

```powershell
Get-Content outputs/optimisation_progress.json -Wait
Get-Content outputs/optimisation_trials.csv -Wait
```

Do not use Top-1000 prediction output as the main optimiser objective. It mostly
optimises the expensive "best hit among 1000 generated tickets" surface and
candidate/diversity settings, which is a different question from ranking the
actual historical winner among all possible tickets.

A future combined objective should be explicit and agreed before implementation.
It should preserve exact rank semantics:
`rank = 1 + count(candidate_score > winner_score)`.

## GPU note

The exact-rank implementation now uses a vectorized NumPy backend: it precomputes
all 2,118,760 main combinations once, scores them with array operations, sorts
the score vector, and uses vectorized binary searches to count the exact rank.

Measured after the vectorized backend:

- `rank-history --mode fast --max-rounds 20`: about 2.1s.
- `optimise --objective exact-rank --trials 1 --mode fast`: about 9.3s.

The rank backend can be selected with `--rank-backend auto`, `--rank-backend cpu`,
or `--rank-backend gpu`. `auto` uses the GPU when a usable CuPy/CUDA runtime is
available and falls back to CPU otherwise, so CPU-only servers remain compatible.

Install optional GPU dependencies on CUDA-capable machines:

```powershell
python -m pip install ".[gpu]"
```

Measured on the local RTX 5080 after installing the GPU extra:

- CPU backend, `optimise --objective exact-rank --trials 1`: about 8.8s.
- GPU backend, `optimise --objective exact-rank --trials 1`: about 7.9s.

The GPU backend keeps exact CPU/GPU rank reproducibility by building the score
vectors with the same CPU arithmetic and using the GPU for sort/search/count.
This is a modest speedup because the vectorized CPU backend is already very
fast, and host-to-device transfer is now a meaningful part of the cost.
