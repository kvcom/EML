from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from euromillions.features import DrawRecord
from euromillions.rank_history import DEFAULT_THRESHOLDS, RankBackend, rank_historical_winners


@dataclass(frozen=True)
class CandidateSpec:
    label: str
    params: dict[str, float]
    objective_value: float | None = None
    objective_average_rank: float | None = None


def load_params_file(path: str) -> CandidateSpec:
    params_path = Path(path)
    raw = json.loads(params_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return CandidateSpec(
        label=params_path.stem,
        params={str(k): float(v) for k, v in raw.items() if isinstance(v, int | float)},
    )


def build_validation_windows(
    *,
    total_draws: int,
    holdout_start_index: int,
    window_count: int = 5,
    window_size: int = 20,
    gap: int = 20,
    min_start_index: int = 100,
) -> list[dict[str, int | str]]:
    windows: list[dict[str, int | str]] = []
    for idx in range(window_count):
        stop = holdout_start_index - ((window_count - idx - 1) * (window_size + gap))
        start = max(min_start_index, stop - window_size)
        if start >= stop or stop > total_draws:
            continue
        windows.append(
            {
                "name": f"validation_{idx + 1}",
                "start_index": start,
                "end_index": stop,
                "draws": stop - start,
            }
        )
    return windows


def validate_candidates(
    draws: list[DrawRecord],
    candidates: list[CandidateSpec],
    *,
    holdout_fraction: float,
    mode: Literal["fast", "full"] = "full",
    rank_backend: RankBackend = "auto",
    window_count: int = 5,
    window_size: int = 20,
    gap: int = 20,
) -> dict[str, Any]:
    holdout_start_index = max(1, int(len(draws) * (1.0 - holdout_fraction)))
    validation_windows = build_validation_windows(
        total_draws=len(draws),
        holdout_start_index=holdout_start_index,
        window_count=window_count,
        window_size=window_size,
        gap=gap,
    )
    report: dict[str, Any] = {
        "draw_count": len(draws),
        "holdout_start_index": holdout_start_index,
        "mode": mode,
        "rank_backend": rank_backend,
        "validation_windows": validation_windows,
        "candidates": [],
    }
    for candidate_spec in candidates:
        min_training_draws = int(candidate_spec.params.get("min_training_draws", 200))
        candidate: dict[str, Any] = {
            "label": candidate_spec.label,
            "objective_value": candidate_spec.objective_value,
            "objective_average_rank": candidate_spec.objective_average_rank,
            "min_training_draws": min_training_draws,
            "params": candidate_spec.params,
            "validation": [],
        }
        validation_average_ranks: list[float] = []
        validation_median_ranks: list[float] = []
        for window in validation_windows:
            _, summary = rank_historical_winners(
                draws,
                min_training_draws=min_training_draws,
                mode=mode,
                thresholds=DEFAULT_THRESHOLDS,
                model_params=candidate_spec.params,
                start_index=int(window["start_index"]),
                end_index=int(window["end_index"]),
                rank_backend=rank_backend,
            )
            average_rank = float(summary["average_rank"])
            median_rank = float(summary["median_rank"])
            validation_average_ranks.append(average_rank)
            validation_median_ranks.append(median_rank)
            candidate["validation"].append(
                {
                    "name": window["name"],
                    "start_index": window["start_index"],
                    "end_index": window["end_index"],
                    "evaluated_draws": summary["evaluated_draws"],
                    "average_rank": average_rank,
                    "median_rank": median_rank,
                }
            )
        _, holdout_summary = rank_historical_winners(
            draws,
            min_training_draws=min_training_draws,
            mode=mode,
            thresholds=DEFAULT_THRESHOLDS,
            model_params=candidate_spec.params,
            start_index=holdout_start_index,
            rank_backend=rank_backend,
        )
        candidate["validation_mean_average_rank"] = (
            sum(validation_average_ranks) / len(validation_average_ranks)
            if validation_average_ranks
            else None
        )
        candidate["validation_mean_median_rank"] = (
            sum(validation_median_ranks) / len(validation_median_ranks)
            if validation_median_ranks
            else None
        )
        candidate["holdout"] = {
            "evaluated_draws": holdout_summary["evaluated_draws"],
            "average_rank": float(holdout_summary["average_rank"]),
            "median_rank": float(holdout_summary["median_rank"]),
        }
        report["candidates"].append(candidate)

    report["ranked_by_validation_mean"] = sorted(
        [
            {
                "label": candidate["label"],
                "validation_mean_average_rank": candidate["validation_mean_average_rank"],
                "holdout_average_rank": candidate["holdout"]["average_rank"],
                "holdout_median_rank": candidate["holdout"]["median_rank"],
            }
            for candidate in report["candidates"]
        ],
        key=lambda row: float(row["validation_mean_average_rank"] or float("inf")),
    )
    report["ranked_by_holdout"] = sorted(
        [
            {
                "label": candidate["label"],
                "validation_mean_average_rank": candidate["validation_mean_average_rank"],
                "holdout_average_rank": candidate["holdout"]["average_rank"],
                "holdout_median_rank": candidate["holdout"]["median_rank"],
            }
            for candidate in report["candidates"]
        ],
        key=lambda row: float(row["holdout_average_rank"]),
    )
    return report


def save_candidate_validation_report(
    report: dict[str, Any],
    out_path: str = "outputs/candidate_validation_report.json",
) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
