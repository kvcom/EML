from __future__ import annotations

from collections.abc import Mapping
from typing import Any


DEFAULT_MODEL_PARAMS: dict[str, float] = {
    "weighted_main_pool_size": 500.0,
    "weighted_star_pool_size": 66.0,
    "weighted_top_number_count": 15.0,
    "weighted_freq_weight": 0.6,
    "weighted_delay_weight": 0.4,
    "weighted_main_weight": 0.8,
    "weighted_star_weight": 0.2,
    "bayesian_alpha": 1.0,
    "bayesian_main_pool_size": 500.0,
    "bayesian_star_pair_count": 10.0,
    "bayesian_top_number_count": 15.0,
    "bayesian_main_weight": 0.75,
    "bayesian_star_weight": 0.25,
    "ensemble_weighted_weight": 0.6,
    "ensemble_bayesian_weight": 0.4,
    "candidate_pool_multiplier": 100.0,
    "candidate_pool_min": 500.0,
    "max_main_overlap": 3.0,
    "require_distinct_star_pairs": 1.0,
}


def merge_model_params(params: Mapping[str, Any] | None = None) -> dict[str, float]:
    merged = dict(DEFAULT_MODEL_PARAMS)
    if params is None:
        return merged
    for key, value in params.items():
        if key not in merged:
            continue
        merged[key] = float(value)
    return merged


def int_param(params: Mapping[str, float], key: str) -> int:
    return max(1, int(round(params[key])))


def bool_param(params: Mapping[str, float], key: str) -> bool:
    return bool(round(params[key]))
