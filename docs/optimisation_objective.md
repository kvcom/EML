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

## Recommended architecture

Use Top-3 as the primary Optuna objective because it matches the production
prediction surface and is fast enough for many trials.

Use `rank-history` as a post-optimisation audit for exact historical winner
rank buckets: `top_1`, `top_3`, `top_10`, `top_100`, `top_500`, `top_1000`,
`top_3000`, and `outside`.

Do not use Top-1000 prediction output as the main optimiser objective. It mostly
optimises the expensive "best hit among 1000 generated tickets" surface and
candidate/diversity settings, which is a different question from ranking the
actual historical winner among all possible tickets.

A future combined objective should be explicit and agreed before implementation,
for example Top-3 uplift plus a small exact rank-bucket audit sample. It should
preserve exact rank semantics: `rank = 1 + count(candidate_score > winner_score)`.
