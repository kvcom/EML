from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import optuna

from euromillions.backtest import run_walk_forward
from euromillions.features import DrawRecord


def recommended_trials(draw_count: int) -> int:
    if draw_count <= 0:
        return 200
    return min(5000, max(200, 2 * draw_count))


def optimise_weights(
    draws: list[DrawRecord],
    trials: int = 50,
    top: int = 3,
    study_name: str = "eml_optimisation",
    storage: str = "sqlite:///outputs/optuna_study.sqlite",
    n_jobs: int = 1,
    timeout_seconds: int | None = None,
    evaluation_mode: Literal["fast", "full"] = "fast",
    holdout_fraction: float = 0.2,
) -> dict[str, Any]:
    if not 0.0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be between 0 and 0.5")
    Path("outputs").mkdir(parents=True, exist_ok=True)
    split_idx = max(1, int(len(draws) * (1.0 - holdout_fraction)))
    train_draws = draws[:split_idx]
    holdout_draws = draws

    def objective(trial: optuna.Trial) -> float:
        min_training = trial.suggest_int("min_training_draws", 100, 300)
        seed = trial.suggest_int("random_seed", 1, 1000)
        result = run_walk_forward(
            train_draws,
            top=top,
            min_training_draws=min_training,
            seed=seed,
            evaluation_mode=evaluation_mode,
        )
        return result.uplift_points

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=trials, n_jobs=n_jobs, timeout=timeout_seconds)

    best_min_training = int(study.best_params.get("min_training_draws", 200))
    best_seed = int(study.best_params.get("random_seed", 42))
    holdout_result = run_walk_forward(
        holdout_draws,
        top=top,
        min_training_draws=best_min_training,
        seed=best_seed,
        evaluation_mode=evaluation_mode,
        start_index=split_idx,
    )
    report: dict[str, Any] = {
        "study_name": study.study_name,
        "storage": storage,
        "completed_trials": len(study.trials),
        "best_value": float(study.best_value),
        "best_params": {k: float(v) for k, v in study.best_params.items()},
        "evaluation_mode": evaluation_mode,
        "holdout": {
            "rounds": holdout_result.rounds,
            "evaluation_stride": holdout_result.evaluation_stride,
            "sampled": holdout_result.evaluation_stride > 1,
            "model_points": holdout_result.model_points,
            "baseline_points": holdout_result.baseline_points,
            "uplift_points": holdout_result.uplift_points,
            "average_best_main_hits": holdout_result.average_best_main_hits,
            "average_best_star_hits": holdout_result.average_best_star_hits,
            "average_random_main_hits": holdout_result.random_baseline_main_hits,
            "average_random_star_hits": holdout_result.random_baseline_star_hits,
        },
    }
    Path("outputs/optimisation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
