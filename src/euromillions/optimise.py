from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Literal

import optuna
from optuna.trial import FrozenTrial

from euromillions.backtest import run_walk_forward
from euromillions.features import DrawRecord
from euromillions.model_params import DEFAULT_MODEL_PARAMS
from euromillions.rank_history import DEFAULT_THRESHOLDS, rank_historical_winners

OptimisationObjective = Literal["top-k", "exact-rank"]


def recommended_trials(draw_count: int) -> int:
    if draw_count <= 0:
        return 200
    return min(5000, max(200, 2 * draw_count))


def _ensure_sqlite_storage_dir(storage: str) -> None:
    prefix = "sqlite:///"
    if not storage.startswith(prefix):
        return
    db_path = Path(storage.removeprefix(prefix))
    if db_path.parent != Path("."):
        db_path.parent.mkdir(parents=True, exist_ok=True)


def _optimise_logger(log_path: Path | None) -> logging.Logger:
    logger = logging.getLogger("euromillions.optimise")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def suggest_model_params(
    trial: optuna.Trial,
    include_prediction_params: bool = True,
) -> dict[str, float]:
    params = {
        "weighted_freq_weight": trial.suggest_float("weighted_freq_weight", 0.0, 1.0),
        "weighted_delay_weight": trial.suggest_float("weighted_delay_weight", 0.0, 1.0),
        "weighted_main_weight": trial.suggest_float("weighted_main_weight", 0.0, 1.0),
        "weighted_star_weight": trial.suggest_float("weighted_star_weight", 0.0, 1.0),
        "bayesian_alpha": trial.suggest_float("bayesian_alpha", 0.1, 5.0, log=True),
        "bayesian_main_weight": trial.suggest_float("bayesian_main_weight", 0.0, 1.0),
        "bayesian_star_weight": trial.suggest_float("bayesian_star_weight", 0.0, 1.0),
        "ensemble_weighted_weight": trial.suggest_float("ensemble_weighted_weight", 0.0, 1.0),
        "ensemble_bayesian_weight": trial.suggest_float("ensemble_bayesian_weight", 0.0, 1.0),
    }
    if include_prediction_params:
        params.update(
            {
                "weighted_main_pool_size": float(
                    trial.suggest_int("weighted_main_pool_size", 100, 1000, step=100)
                ),
                "weighted_star_pool_size": float(trial.suggest_int("weighted_star_pool_size", 12, 66)),
                "weighted_top_number_count": float(
                    trial.suggest_int("weighted_top_number_count", 10, 25)
                ),
                "bayesian_main_pool_size": float(
                    trial.suggest_int("bayesian_main_pool_size", 100, 1000, step=100)
                ),
                "bayesian_star_pair_count": float(trial.suggest_int("bayesian_star_pair_count", 5, 66)),
                "bayesian_top_number_count": float(
                    trial.suggest_int("bayesian_top_number_count", 10, 25)
                ),
                "candidate_pool_multiplier": float(
                    trial.suggest_int("candidate_pool_multiplier", 50, 200, step=25)
                ),
                "candidate_pool_min": float(trial.suggest_int("candidate_pool_min", 250, 1000, step=250)),
                "max_main_overlap": float(trial.suggest_int("max_main_overlap", 2, 4)),
                "require_distinct_star_pairs": float(
                    trial.suggest_categorical("require_distinct_star_pairs", [0, 1])
                ),
            }
        )
    return params


def optimise_weights(
    draws: list[DrawRecord],
    trials: int = 50,
    top: int = 3,
    objective_name: OptimisationObjective = "top-k",
    study_name: str = "eml_optimisation",
    storage: str = "sqlite:///outputs/optuna_study.sqlite",
    n_jobs: int = 1,
    timeout_seconds: int | None = None,
    evaluation_mode: Literal["fast", "full"] = "fast",
    holdout_fraction: float = 0.2,
    log_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not 0.0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be between 0 and 0.5")
    Path("outputs").mkdir(parents=True, exist_ok=True)
    _ensure_sqlite_storage_dir(storage)
    logger = _optimise_logger(log_path)
    split_idx = max(1, int(len(draws) * (1.0 - holdout_fraction)))
    train_draws = draws[:split_idx]
    holdout_draws = draws

    logger.info(
        "starting optimisation study=%s storage=%s trials=%s n_jobs=%s timeout_seconds=%s mode=%s objective=%s draws=%s",
        study_name,
        storage,
        trials,
        n_jobs,
        timeout_seconds,
        evaluation_mode,
        objective_name,
        len(draws),
    )

    def objective(trial: optuna.Trial) -> float:
        min_training = trial.suggest_int("min_training_draws", 100, 300)
        seed = trial.suggest_int("random_seed", 1, 1000) if objective_name == "top-k" else 42
        model_params = suggest_model_params(
            trial,
            include_prediction_params=objective_name == "top-k",
        )
        logger.info(
            "trial %s started min_training_draws=%s random_seed=%s model_params=%s",
            trial.number,
            min_training,
            seed,
            json.dumps(model_params, sort_keys=True),
        )
        if objective_name == "exact-rank":
            rows, summary = rank_historical_winners(
                train_draws,
                min_training_draws=min_training,
                mode=evaluation_mode,
                thresholds=DEFAULT_THRESHOLDS,
                model_params=model_params,
            )
            average_rank = float(summary.get("average_rank", 0.0))
            median_rank = float(summary.get("median_rank", 0.0))
            value = -average_rank
            logger.info(
                "trial %s finished exact_rank_value=%.6f average_rank=%.6f median_rank=%.6f rounds=%s",
                trial.number,
                value,
                average_rank,
                median_rank,
                len(rows),
            )
            return value
        result = run_walk_forward(
            train_draws,
            top=top,
            min_training_draws=min_training,
            seed=seed,
            evaluation_mode=evaluation_mode,
            model_params=model_params,
        )
        logger.info(
            "trial %s finished uplift_points=%.6f model_points=%.6f baseline_points=%.6f rounds=%s",
            trial.number,
            result.uplift_points,
            result.model_points,
            result.baseline_points,
            result.rounds,
        )
        return result.uplift_points

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    existing_trials = len(study.trials)
    logger.info("loaded study=%s existing_trials=%s", study.study_name, existing_trials)

    def on_trial_complete(study: optuna.Study, trial: FrozenTrial) -> None:
        message = (
            f"completed trial={trial.number} state={trial.state.name} "
            f"value={trial.value} total_trials={len(study.trials)} best_value={study.best_value}"
        )
        logger.info(message)
        if progress_callback is not None:
            progress_callback(message)

    study.optimize(
        objective,
        n_trials=trials,
        n_jobs=n_jobs,
        timeout=timeout_seconds,
        callbacks=[on_trial_complete],
    )

    best_min_training = int(study.best_params.get("min_training_draws", 200))
    best_seed = int(study.best_params.get("random_seed", 42))
    best_model_params = {
        key: float(study.best_params.get(key, default))
        for key, default in DEFAULT_MODEL_PARAMS.items()
    }
    if objective_name == "exact-rank":
        _holdout_rows, holdout_summary = rank_historical_winners(
            holdout_draws,
            min_training_draws=best_min_training,
            mode=evaluation_mode,
            thresholds=DEFAULT_THRESHOLDS,
            model_params=best_model_params,
            start_index=split_idx,
        )
        holdout_payload: dict[str, float | int | str | bool] = {
            **holdout_summary,
            "sampled": int(holdout_summary["evaluation_stride"]) > 1,
        }
        logger.info(
            "holdout finished exact_rank evaluated_draws=%s average_rank=%s median_rank=%s",
            holdout_summary["evaluated_draws"],
            holdout_summary.get("average_rank"),
            holdout_summary.get("median_rank"),
        )
    else:
        holdout_result = run_walk_forward(
            holdout_draws,
            top=top,
            min_training_draws=best_min_training,
            seed=best_seed,
            evaluation_mode=evaluation_mode,
            start_index=split_idx,
            model_params=best_model_params,
        )
        logger.info(
            "holdout finished rounds=%s model_points=%.6f baseline_points=%.6f uplift_points=%.6f",
            holdout_result.rounds,
            holdout_result.model_points,
            holdout_result.baseline_points,
            holdout_result.uplift_points,
        )
        holdout_payload = {
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
        }
    completed_trials = len(study.trials)
    report: dict[str, Any] = {
        "study_name": study.study_name,
        "storage": storage,
        "objective": objective_name,
        "requested_trials": trials,
        "existing_trials": existing_trials,
        "completed_trials": completed_trials,
        "new_trials": completed_trials - existing_trials,
        "best_value": float(study.best_value),
        "best_params": {k: float(v) for k, v in study.best_params.items()},
        "best_model_params": best_model_params,
        "evaluation_mode": evaluation_mode,
        "mode": evaluation_mode,
        "metadata": metadata or {},
        "log_path": str(log_path) if log_path is not None else None,
        "holdout": holdout_payload,
    }
    Path("outputs/optimisation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("wrote optimisation report outputs/optimisation_report.json")
    return report
