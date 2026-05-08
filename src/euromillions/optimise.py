from __future__ import annotations

import optuna

from euromillions.backtest import run_walk_forward
from euromillions.features import DrawRecord


def recommended_trials(draw_count: int) -> int:
    if draw_count <= 0:
        return 200
    return min(5000, max(200, 2 * draw_count))


def optimise_weights(draws: list[DrawRecord], trials: int = 50, top: int = 3) -> dict[str, float]:
    def objective(trial: optuna.Trial) -> float:
        min_training = trial.suggest_int("min_training_draws", 100, 300)
        seed = trial.suggest_int("random_seed", 1, 1000)
        result = run_walk_forward(draws, top=top, min_training_draws=min_training, seed=seed)
        return (result.average_best_main_hits * 10.0 + result.average_best_star_hits * 4.0) - (
            result.random_baseline_main_hits * 10.0 + result.random_baseline_star_hits * 4.0
        )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
    return {k: float(v) for k, v in study.best_params.items()}
