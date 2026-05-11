from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4
from typing import Any, Callable, Literal

import optuna
from optuna.trial import FrozenTrial

from euromillions.backtest import run_walk_forward
from euromillions.features import DrawRecord
from euromillions.model_params import DEFAULT_MODEL_PARAMS
from euromillions.portfolio_backtest import run_portfolio_backtest
from euromillions.rank_history import DEFAULT_THRESHOLDS, RankBackend, rank_historical_winners

OptimisationObjective = Literal["top-k", "exact-rank", "exact-rank-sum", "portfolio-uplift"]


def _complete_trials_by_value(study: optuna.Study, limit: int) -> list[FrozenTrial]:
    complete_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    return sorted(complete_trials, key=lambda trial: float(trial.value or float("-inf")), reverse=True)[
        :limit
    ]


def _rolling_window_ranges(
    *,
    total_draws: int,
    holdout_start_idx: int,
    min_training_draws: int,
    window_count: int,
    window_rounds: int | None,
    mode: Literal["fast", "full"],
) -> list[tuple[int, int]]:
    if window_count < 1:
        raise ValueError("rolling_window_count must be at least 1")
    if window_count == 1:
        return [(min_training_draws, holdout_start_idx)]

    stride = 10 if mode == "fast" else 1
    rounds_per_window = window_rounds or 10
    window_span = max(stride, rounds_per_window * stride)
    latest_end = min(holdout_start_idx, total_draws)
    earliest_start = max(min_training_draws, latest_end - (window_span * window_count))
    ranges: list[tuple[int, int]] = []
    for idx in range(window_count):
        start = earliest_start + idx * window_span
        end = min(start + window_span, latest_end)
        if start < end:
            ranges.append((start, end))
    return ranges


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


def _model_params_from_study_params(params: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(params.get(key, default))
        for key, default in DEFAULT_MODEL_PARAMS.items()
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".{uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        tmp_path.replace(path)
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise


class OptimisationMonitor:
    def __init__(
        self,
        progress_path: Path,
        trials_path: Path,
        requested_trials: int,
        existing_trials: int,
        study_name: str,
        objective_name: str,
        rank_backend: str,
        started_at: str | None,
    ) -> None:
        self.progress_path = progress_path
        self.trials_path = trials_path
        self.requested_trials = requested_trials
        self.existing_trials = existing_trials
        self.study_name = study_name
        self.objective_name = objective_name
        self.rank_backend = rank_backend
        self.started_at = started_at
        self.current_trial_number: int | None = None
        self.current_trial_started_at: str | None = None
        self.current_trial_timer: float | None = None
        self.last_trial_seconds: float | None = None
        self.best_value: float | None = None
        self.best_trial: int | None = None
        self.completed_new_trials = 0
        self.total_completed_trials = existing_trials
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        self.trials_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trials_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "trial",
                    "state",
                    "value",
                    "best_value",
                    "duration_seconds",
                    "completed_new_trials",
                    "total_completed_trials",
                    "params_json",
                ],
            )
            writer.writeheader()

    def _payload(self, status: str) -> dict[str, Any]:
        remaining = max(0, self.requested_trials - self.completed_new_trials)
        estimated_remaining_seconds = (
            remaining * self.last_trial_seconds if self.last_trial_seconds is not None else None
        )
        return {
            "status": status,
            "study_name": self.study_name,
            "objective": self.objective_name,
            "rank_backend": self.rank_backend,
            "started_at": self.started_at,
            "requested_trials": self.requested_trials,
            "existing_trials": self.existing_trials,
            "completed_new_trials": self.completed_new_trials,
            "total_completed_trials": self.total_completed_trials,
            "remaining_requested_trials": remaining,
            "current_trial": self.current_trial_number,
            "current_trial_started_at": self.current_trial_started_at,
            "last_trial_seconds": self.last_trial_seconds,
            "estimated_remaining_seconds": estimated_remaining_seconds,
            "best_value": self.best_value,
            "best_trial": self.best_trial,
        }

    def write(self, status: str) -> None:
        try:
            _atomic_write_json(self.progress_path, self._payload(status))
        except OSError:
            return

    def trial_started(self, trial_number: int, started_at: str) -> None:
        self.current_trial_number = trial_number
        self.current_trial_started_at = started_at
        self.current_trial_timer = perf_counter()
        self.write("running")

    def trial_completed(self, study: optuna.Study, trial: FrozenTrial) -> None:
        self.last_trial_seconds = (
            perf_counter() - self.current_trial_timer
            if self.current_trial_timer is not None
            else None
        )
        self.completed_new_trials = max(0, len(study.trials) - self.existing_trials)
        self.total_completed_trials = len(study.trials)
        self.best_value = float(study.best_value)
        self.best_trial = study.best_trial.number
        try:
            with self.trials_path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "trial",
                        "state",
                        "value",
                        "best_value",
                        "duration_seconds",
                        "completed_new_trials",
                        "total_completed_trials",
                        "params_json",
                    ],
                )
                writer.writerow(
                    {
                        "trial": trial.number,
                        "state": trial.state.name,
                        "value": trial.value,
                        "best_value": self.best_value,
                        "duration_seconds": self.last_trial_seconds,
                        "completed_new_trials": self.completed_new_trials,
                        "total_completed_trials": self.total_completed_trials,
                        "params_json": json.dumps(dict(trial.params), sort_keys=True),
                    }
                )
        except OSError:
            pass
        self.current_trial_number = None
        self.current_trial_started_at = None
        self.current_trial_timer = None
        self.write("running")


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
    validation_fraction: float = 0.2,
    early_stop_patience: int | None = None,
    early_stop_min_delta: float = 0.0,
    early_stop_validation_rounds: int | None = 10,
    rolling_windows: int = 1,
    rolling_window_rounds: int | None = 10,
    top_trial_holdout_count: int = 10,
    top_trial_holdout_rounds: int | None = None,
    portfolio_objective_rounds: int | None = 100,
    portfolio_random_baseline_runs: int = 10,
    portfolio_holdout_random_baseline_runs: int = 25,
    rank_backend: RankBackend = "auto",
    log_path: Path | None = None,
    progress_path: Path | None = None,
    trials_path: Path | None = None,
    metadata: dict[str, Any] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if not 0.0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must be between 0 and 0.5")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    if early_stop_patience is not None and early_stop_patience < 1:
        raise ValueError("early_stop_patience must be at least 1")
    if rolling_windows < 1:
        raise ValueError("rolling_windows must be at least 1")
    if top_trial_holdout_count < 0:
        raise ValueError("top_trial_holdout_count must be at least 0")
    if portfolio_random_baseline_runs < 1:
        raise ValueError("portfolio_random_baseline_runs must be at least 1")
    if portfolio_holdout_random_baseline_runs < 1:
        raise ValueError("portfolio_holdout_random_baseline_runs must be at least 1")
    Path("outputs").mkdir(parents=True, exist_ok=True)
    _ensure_sqlite_storage_dir(storage)
    logger = _optimise_logger(log_path)
    split_idx = max(1, int(len(draws) * (1.0 - holdout_fraction)))
    is_exact_rank_objective = objective_name in {"exact-rank", "exact-rank-sum"}
    validation_enabled = is_exact_rank_objective and early_stop_patience is not None
    validation_start_idx = (
        max(1, int(split_idx * (1.0 - validation_fraction))) if validation_enabled else split_idx
    )
    train_draws = draws[:validation_start_idx]
    holdout_draws = draws

    logger.info(
        "starting optimisation study=%s storage=%s trials=%s n_jobs=%s timeout_seconds=%s mode=%s objective=%s draws=%s validation_enabled=%s",
        study_name,
        storage,
        trials,
        n_jobs,
        timeout_seconds,
        evaluation_mode,
        objective_name,
        len(draws),
        validation_enabled,
    )

    def objective(trial: optuna.Trial) -> float:
        min_training = trial.suggest_int("min_training_draws", 100, 300)
        seed = trial.suggest_int("random_seed", 1, 1000) if objective_name in {"top-k", "portfolio-uplift"} else 42
        model_params = suggest_model_params(
            trial,
            include_prediction_params=objective_name in {"top-k", "portfolio-uplift"},
        )
        monitor.trial_started(trial.number, datetime.now(timezone.utc).isoformat(timespec="seconds"))
        logger.info(
            "trial %s started min_training_draws=%s random_seed=%s model_params=%s",
            trial.number,
            min_training,
            seed,
            json.dumps(model_params, sort_keys=True),
        )
        if is_exact_rank_objective:
            objective_ranges = (
                _rolling_window_ranges(
                    total_draws=len(draws),
                    holdout_start_idx=validation_start_idx,
                    min_training_draws=min_training,
                    window_count=rolling_windows,
                    window_rounds=rolling_window_rounds,
                    mode=evaluation_mode,
                )
                if rolling_windows > 1
                else []
            )
            window_summaries: list[dict[str, float | int | str]] = []
            average_ranks: list[float] = []
            median_ranks: list[float] = []
            rank_sum = 0.0
            evaluated_draws = 0
            if rolling_windows == 1:
                rows, summary = rank_historical_winners(
                    train_draws,
                    min_training_draws=min_training,
                    mode=evaluation_mode,
                    thresholds=DEFAULT_THRESHOLDS,
                    model_params=model_params,
                    rank_backend=rank_backend,
                )
                if "average_rank" in summary:
                    average_ranks.append(float(summary["average_rank"]))
                if "median_rank" in summary:
                    median_ranks.append(float(summary["median_rank"]))
                rank_sum += float(summary.get("rank_sum", sum(row.exact_rank for row in rows)))
                evaluated_draws += len(rows)
                window_summaries.append(summary)
            else:
                for start_idx, end_idx in objective_ranges:
                    rows, summary = rank_historical_winners(
                        draws,
                        min_training_draws=min_training,
                        mode=evaluation_mode,
                        thresholds=DEFAULT_THRESHOLDS,
                        model_params=model_params,
                        max_rounds=rolling_window_rounds,
                        start_index=start_idx,
                        end_index=end_idx,
                        rank_backend=rank_backend,
                    )
                    if "average_rank" in summary:
                        average_ranks.append(float(summary["average_rank"]))
                    if "median_rank" in summary:
                        median_ranks.append(float(summary["median_rank"]))
                    rank_sum += float(summary.get("rank_sum", sum(row.exact_rank for row in rows)))
                    evaluated_draws += len(rows)
                    window_summaries.append(summary)
            average_rank = sum(average_ranks) / len(average_ranks) if average_ranks else float("inf")
            median_rank = sum(median_ranks) / len(median_ranks) if median_ranks else float("inf")
            value = -rank_sum if objective_name == "exact-rank-sum" else -average_rank
            trial.set_user_attr("objective_average_rank", average_rank)
            trial.set_user_attr("objective_median_rank", median_rank)
            trial.set_user_attr("objective_rank_sum", rank_sum)
            trial.set_user_attr("objective_evaluated_draws", evaluated_draws)
            trial.set_user_attr("objective_window_count", len(window_summaries))
            logger.info(
                "trial %s finished exact_rank_value=%.6f rank_sum=%.6f average_rank=%.6f median_rank=%.6f rounds=%s windows=%s",
                trial.number,
                value,
                rank_sum,
                average_rank,
                median_rank,
                evaluated_draws,
                len(window_summaries),
            )
            return value
        if objective_name == "portfolio-uplift":
            report = run_portfolio_backtest(
                train_draws,
                top=top,
                min_training_draws=min_training,
                seed=seed,
                mode=evaluation_mode,
                max_rounds=portfolio_objective_rounds,
                model_params=model_params,
                random_baseline_runs=portfolio_random_baseline_runs,
            )
            model_rate = float(report["model"]["winning_round_rate"])
            random_rate = float(report["random_baseline"]["winning_round_rate"])
            value = model_rate - random_rate
            trial.set_user_attr("portfolio_model_winning_round_rate", model_rate)
            trial.set_user_attr("portfolio_random_winning_round_rate", random_rate)
            trial.set_user_attr("portfolio_winning_round_uplift", value)
            logger.info(
                "trial %s finished portfolio_uplift=%.6f model_winning_round_rate=%.6f random_winning_round_rate=%.6f rounds=%s",
                trial.number,
                value,
                model_rate,
                random_rate,
                report["rounds"],
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
    monitor = OptimisationMonitor(
        progress_path=progress_path or Path("outputs/optimisation_progress.json"),
        trials_path=trials_path or Path("outputs/optimisation_trials.csv"),
        requested_trials=trials,
        existing_trials=existing_trials,
        study_name=study.study_name,
        objective_name=objective_name,
        rank_backend=rank_backend,
        started_at=str((metadata or {}).get("started_at") or ""),
    )
    monitor.write("starting")
    early_stop_state: dict[str, float | int | bool | None] = {
        "best_validation_rank": None,
        "best_trial": None,
        "stale_trials": 0,
        "stopped": False,
    }

    def on_trial_complete(study: optuna.Study, trial: FrozenTrial) -> None:
        if validation_enabled and trial.state == optuna.trial.TrialState.COMPLETE:
            min_training = int(trial.params.get("min_training_draws", 200))
            model_params = _model_params_from_study_params(trial.params)
            _, validation_summary = rank_historical_winners(
                draws,
                min_training_draws=min_training,
                mode=evaluation_mode,
                thresholds=DEFAULT_THRESHOLDS,
                model_params=model_params,
                max_rounds=early_stop_validation_rounds,
                start_index=validation_start_idx,
                end_index=split_idx,
                rank_backend=rank_backend,
            )
            validation_rank = float(validation_summary.get("average_rank", float("inf")))
            best_rank = early_stop_state["best_validation_rank"]
            improved = best_rank is None or validation_rank < float(best_rank) - early_stop_min_delta
            if improved:
                early_stop_state["best_validation_rank"] = validation_rank
                early_stop_state["best_trial"] = trial.number
                early_stop_state["stale_trials"] = 0
            else:
                early_stop_state["stale_trials"] = int(early_stop_state["stale_trials"] or 0) + 1
            study.set_user_attr("early_stop_best_validation_rank", early_stop_state["best_validation_rank"])
            study.set_user_attr("early_stop_best_trial", early_stop_state["best_trial"])
            study.set_user_attr("early_stop_stale_trials", early_stop_state["stale_trials"])
            logger.info(
                "trial %s validation average_rank=%.6f best_validation_rank=%s stale_trials=%s",
                trial.number,
                validation_rank,
                early_stop_state["best_validation_rank"],
                early_stop_state["stale_trials"],
            )
            if int(early_stop_state["stale_trials"] or 0) >= int(early_stop_patience or 1):
                early_stop_state["stopped"] = True
                study.set_user_attr("early_stop_stopped", True)
                logger.info(
                    "early stopping triggered stale_trials=%s patience=%s",
                    early_stop_state["stale_trials"],
                    early_stop_patience,
                )
                study.stop()
        message = (
            f"completed trial={trial.number} state={trial.state.name} "
            f"value={trial.value} total_trials={len(study.trials)} best_value={study.best_value}"
        )
        logger.info(message)
        monitor.trial_completed(study, trial)
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
    best_model_params = _model_params_from_study_params(study.best_params)
    top_trial_report: list[dict[str, Any]] = []
    if is_exact_rank_objective:
        _holdout_rows, holdout_summary = rank_historical_winners(
            holdout_draws,
            min_training_draws=best_min_training,
            mode=evaluation_mode,
            thresholds=DEFAULT_THRESHOLDS,
            model_params=best_model_params,
            start_index=split_idx,
            rank_backend=rank_backend,
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
        for candidate in _complete_trials_by_value(study, top_trial_holdout_count):
            candidate_min_training = int(candidate.params.get("min_training_draws", 200))
            candidate_model_params = _model_params_from_study_params(candidate.params)
            validation_summary: dict[str, float | int | str] | None = None
            if validation_enabled:
                _, validation_summary = rank_historical_winners(
                    draws,
                    min_training_draws=candidate_min_training,
                    mode=evaluation_mode,
                    thresholds=DEFAULT_THRESHOLDS,
                    model_params=candidate_model_params,
                    max_rounds=early_stop_validation_rounds,
                    start_index=validation_start_idx,
                    end_index=split_idx,
                    rank_backend=rank_backend,
                )
            _, candidate_holdout_summary = rank_historical_winners(
                holdout_draws,
                min_training_draws=candidate_min_training,
                mode=evaluation_mode,
                thresholds=DEFAULT_THRESHOLDS,
                model_params=candidate_model_params,
                max_rounds=top_trial_holdout_rounds,
                start_index=split_idx,
                rank_backend=rank_backend,
            )
            top_trial_report.append(
                {
                    "trial": candidate.number,
                    "objective_value": float(candidate.value or 0.0),
                    "objective_average_rank": float(candidate.user_attrs.get("objective_average_rank", -float(candidate.value or 0.0))),
                    "objective_rank_sum": candidate.user_attrs.get("objective_rank_sum"),
                    "validation_average_rank": (
                        float(validation_summary["average_rank"])
                        if validation_summary is not None and "average_rank" in validation_summary
                        else None
                    ),
                    "holdout_average_rank": candidate_holdout_summary.get("average_rank"),
                    "holdout_median_rank": candidate_holdout_summary.get("median_rank"),
                    "holdout_evaluated_draws": candidate_holdout_summary["evaluated_draws"],
                    "params": {k: float(v) for k, v in candidate.params.items()},
                }
            )
        top_trial_report.sort(
            key=lambda row: float(row["holdout_average_rank"])
            if row["holdout_average_rank"] is not None
            else float("inf")
        )
        _atomic_write_json(
            Path("outputs/top_trial_holdout_report.json"),
            {"trials": top_trial_report},
        )
    elif objective_name == "portfolio-uplift":
        holdout_report = run_portfolio_backtest(
            holdout_draws,
            top=top,
            min_training_draws=best_min_training,
            seed=best_seed,
            mode=evaluation_mode,
            start_index=split_idx,
            model_params=best_model_params,
            random_baseline_runs=portfolio_holdout_random_baseline_runs,
        )
        logger.info(
            "holdout finished portfolio rounds=%s model_winning_round_rate=%.6f random_winning_round_rate=%.6f uplift=%.6f",
            holdout_report["rounds"],
            holdout_report["model"]["winning_round_rate"],
            holdout_report["random_baseline"]["winning_round_rate"],
            holdout_report["model"]["winning_round_rate"] - holdout_report["random_baseline"]["winning_round_rate"],
        )
        holdout_payload = {
            **holdout_report,
            "sampled": int(holdout_report["evaluation_stride"]) > 1,
            "winning_round_uplift": (
                float(holdout_report["model"]["winning_round_rate"])
                - float(holdout_report["random_baseline"]["winning_round_rate"])
            ),
        }
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
        "progress_path": str(monitor.progress_path),
        "trials_path": str(monitor.trials_path),
        "early_stop": {
            "enabled": validation_enabled,
            "patience": early_stop_patience,
            "min_delta": early_stop_min_delta,
            "validation_fraction": validation_fraction,
            "validation_rounds": early_stop_validation_rounds,
            "rank_backend": rank_backend,
            "best_validation_rank": early_stop_state["best_validation_rank"],
            "best_trial": early_stop_state["best_trial"],
            "stale_trials": early_stop_state["stale_trials"],
            "stopped": early_stop_state["stopped"],
        },
        "rolling_objective": {
            "enabled": is_exact_rank_objective and rolling_windows > 1,
            "windows": rolling_windows,
            "rounds_per_window": rolling_window_rounds,
        },
        "top_trial_holdout": {
            "count": top_trial_holdout_count,
            "rounds": top_trial_holdout_rounds,
            "report_path": "outputs/top_trial_holdout_report.json" if is_exact_rank_objective else None,
            "trials": top_trial_report,
        },
        "portfolio_objective": {
            "enabled": objective_name == "portfolio-uplift",
            "objective_rounds": portfolio_objective_rounds,
            "random_baseline_runs": portfolio_random_baseline_runs,
            "holdout_random_baseline_runs": portfolio_holdout_random_baseline_runs,
        },
        "holdout": holdout_payload,
    }
    Path("outputs/optimisation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    monitor.write("finished")
    logger.info("wrote optimisation report outputs/optimisation_report.json")
    return report
