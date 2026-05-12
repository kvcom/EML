from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Literal

import optuna

from euromillions.features import DrawRecord
from euromillions.model_params import DEFAULT_MODEL_PARAMS
from euromillions.optimise import suggest_model_params
from euromillions.rank_history import DEFAULT_THRESHOLDS, RankBackend, rank_historical_winners


def _score_one_draw(
    draws: list[DrawRecord],
    *,
    target_index: int,
    min_training_draws: int,
    model_params: dict[str, float],
    mode: Literal["fast", "full"],
    rank_backend: RankBackend,
) -> dict[str, float | int | str]:
    _, summary = rank_historical_winners(
        draws,
        min_training_draws=min_training_draws,
        mode=mode,
        thresholds=DEFAULT_THRESHOLDS,
        model_params=model_params,
        max_rounds=1,
        start_index=target_index,
        end_index=target_index + 1,
        rank_backend=rank_backend,
    )
    return summary


def _optimise_oracle_params_for_draw(
    draws: list[DrawRecord],
    *,
    target_index: int,
    trials: int,
    mode: Literal["fast", "full"],
    rank_backend: RankBackend,
) -> dict[str, Any]:
    def objective(trial: optuna.Trial) -> float:
        min_training = trial.suggest_int("min_training_draws", 100, 300)
        model_params = suggest_model_params(trial, include_prediction_params=False)
        summary = _score_one_draw(
            draws,
            target_index=target_index,
            min_training_draws=min_training,
            model_params=model_params,
            mode=mode,
            rank_backend=rank_backend,
        )
        return -float(summary.get("rank_sum", summary.get("average_rank", float("inf"))))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials)
    params = {
        "min_training_draws": float(study.best_params.get("min_training_draws", 200)),
        **{
            key: float(study.best_params.get(key, default))
            for key, default in DEFAULT_MODEL_PARAMS.items()
            if key in study.best_params or key not in {
                "weighted_main_pool_size",
                "weighted_star_pool_size",
                "weighted_top_number_count",
                "bayesian_main_pool_size",
                "bayesian_star_pair_count",
                "bayesian_top_number_count",
                "candidate_pool_multiplier",
                "candidate_pool_min",
                "max_main_overlap",
                "require_distinct_star_pairs",
            }
        },
    }
    best_rank = -float(study.best_value)
    return {
        "target_index": target_index,
        "best_rank": best_rank,
        "params": params,
        "trials": trials,
    }


def _forecast_params(previous_params: list[dict[str, float]], lookback: int) -> dict[str, float]:
    window = previous_params[-lookback:]
    keys = sorted({key for params in window for key in params})
    return {
        key: float(mean(params.get(key, DEFAULT_MODEL_PARAMS.get(key, 200.0)) for params in window))
        for key in keys
    }


def run_dynamic_params_experiment(
    draws: list[DrawRecord],
    *,
    baseline_params: dict[str, float],
    start_index: int | None = None,
    end_index: int | None = None,
    max_targets: int = 20,
    stride: int = 10,
    oracle_trials: int = 20,
    forecast_lookback: int = 5,
    mode: Literal["fast", "full"] = "fast",
    rank_backend: RankBackend = "auto",
) -> dict[str, Any]:
    if max_targets < 1:
        raise ValueError("max_targets must be at least 1")
    if stride < 1:
        raise ValueError("stride must be at least 1")
    if oracle_trials < 1:
        raise ValueError("oracle_trials must be at least 1")
    if forecast_lookback < 1:
        raise ValueError("forecast_lookback must be at least 1")

    first_target = max(300, start_index or 300)
    stop = min(end_index or len(draws), len(draws))
    target_indices = list(range(first_target, stop, stride))[:max_targets]
    oracle_sequence: list[dict[str, Any]] = []
    oracle_params: list[dict[str, float]] = []
    evaluation_rows: list[dict[str, Any]] = []
    baseline_min_training = int(baseline_params.get("min_training_draws", 200))

    for target_index in target_indices:
        baseline_summary = _score_one_draw(
            draws,
            target_index=target_index,
            min_training_draws=baseline_min_training,
            model_params=baseline_params,
            mode=mode,
            rank_backend=rank_backend,
        )
        row: dict[str, Any] = {
            "target_index": target_index,
            "draw_id": draws[target_index].draw_id,
            "draw_date": draws[target_index].draw_date.isoformat()
            if draws[target_index].draw_date is not None
            else "",
            "baseline_rank": baseline_summary.get("rank_sum", baseline_summary.get("average_rank")),
            "dynamic_rank": None,
            "oracle_rank": None,
        }
        if oracle_params:
            forecast = _forecast_params(oracle_params, forecast_lookback)
            dynamic_summary = _score_one_draw(
                draws,
                target_index=target_index,
                min_training_draws=int(forecast.get("min_training_draws", baseline_min_training)),
                model_params=forecast,
                mode=mode,
                rank_backend=rank_backend,
            )
            row["dynamic_rank"] = dynamic_summary.get("rank_sum", dynamic_summary.get("average_rank"))
            row["forecast_params"] = forecast

        oracle = _optimise_oracle_params_for_draw(
            draws,
            target_index=target_index,
            trials=oracle_trials,
            mode=mode,
            rank_backend=rank_backend,
        )
        row["oracle_rank"] = oracle["best_rank"]
        oracle_sequence.append(oracle)
        oracle_params.append(oracle["params"])
        evaluation_rows.append(row)

    dynamic_ranks = [float(row["dynamic_rank"]) for row in evaluation_rows if row["dynamic_rank"] is not None]
    baseline_ranks = [
        float(row["baseline_rank"]) for row in evaluation_rows if row["dynamic_rank"] is not None
    ]
    oracle_ranks = [float(row["oracle_rank"]) for row in evaluation_rows]
    summary = {
        "targets": len(target_indices),
        "evaluated_dynamic_targets": len(dynamic_ranks),
        "baseline_average_rank": sum(baseline_ranks) / len(baseline_ranks) if baseline_ranks else None,
        "dynamic_average_rank": sum(dynamic_ranks) / len(dynamic_ranks) if dynamic_ranks else None,
        "oracle_average_rank": sum(oracle_ranks) / len(oracle_ranks) if oracle_ranks else None,
        "dynamic_vs_baseline_rank_delta": (
            (sum(dynamic_ranks) / len(dynamic_ranks)) - (sum(baseline_ranks) / len(baseline_ranks))
            if dynamic_ranks and baseline_ranks
            else None
        ),
    }
    return {
        "mode": mode,
        "stride": stride,
        "oracle_trials": oracle_trials,
        "forecast_lookback": forecast_lookback,
        "summary": summary,
        "rows": evaluation_rows,
        "oracle_sequence": oracle_sequence,
    }


def save_dynamic_params_report(
    report: dict[str, Any],
    out_path: str = "outputs/dynamic_params_report.json",
) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
