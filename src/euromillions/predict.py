from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from itertools import combinations
from pathlib import Path
from typing import TypedDict

from euromillions.features import DrawRecord
from euromillions.model_params import bool_param, int_param, merge_model_params
from euromillions.models.bayesian_frequency import BayesianFrequencyModel
from euromillions.models.ensemble import EnsembleModel
from euromillions.models.weighted_statistical import WeightedStatisticalModel


class PredictionRow(TypedDict):
    rank: int
    mains: tuple[int, int, int, int, int]
    stars: tuple[int, int]
    score: float
    why: str


def _is_diverse_enough(
    selected: list[PredictionRow],
    candidate_mains: tuple[int, int, int, int, int],
    candidate_stars: tuple[int, int],
    max_main_overlap: int,
    require_distinct_star_pairs: bool,
) -> bool:
    for row in selected:
        overlap = len(set(row["mains"]) & set(candidate_mains))
        if overlap > max_main_overlap:
            return False
    if require_distinct_star_pairs and any(row["stars"] == candidate_stars for row in selected):
        return False
    return True


def generate_predictions(
    history: list[DrawRecord],
    top: int,
    seed: int = 42,
    max_main_overlap: int | None = None,
    require_distinct_star_pairs: bool | None = None,
    model_params: dict[str, float] | None = None,
) -> list[PredictionRow]:
    params = merge_model_params(model_params)
    effective_max_main_overlap = (
        int_param(params, "max_main_overlap") if max_main_overlap is None else max_main_overlap
    )
    effective_require_distinct_star_pairs = (
        bool_param(params, "require_distinct_star_pairs")
        if require_distinct_star_pairs is None
        else require_distinct_star_pairs
    )
    model = EnsembleModel(
        weighted=WeightedStatisticalModel(
            main_pool_size=int_param(params, "weighted_main_pool_size"),
            star_pool_size=int_param(params, "weighted_star_pool_size"),
            top_number_count=int_param(params, "weighted_top_number_count"),
            frequency_weight=params["weighted_freq_weight"],
            delay_weight=params["weighted_delay_weight"],
            main_weight=params["weighted_main_weight"],
            star_weight=params["weighted_star_weight"],
        ),
        bayesian=BayesianFrequencyModel(
            alpha=params["bayesian_alpha"],
            main_pool_size=int_param(params, "bayesian_main_pool_size"),
            star_pair_count=int_param(params, "bayesian_star_pair_count"),
            top_number_count=int_param(params, "bayesian_top_number_count"),
            main_weight=params["bayesian_main_weight"],
            star_weight=params["bayesian_star_weight"],
        ),
        w_weighted=params["ensemble_weighted_weight"],
        w_bayesian=params["ensemble_bayesian_weight"],
    )
    _ = seed
    candidate_pool = max(
        top * int_param(params, "candidate_pool_multiplier"),
        int_param(params, "candidate_pool_min"),
    )
    ranked = model.predict(history, top=candidate_pool)
    out: list[PredictionRow] = []
    for mains, stars, score in ranked:
        if not _is_diverse_enough(
            out,
            mains,
            stars,
            effective_max_main_overlap,
            effective_require_distinct_star_pairs,
        ):
            continue
        idx = len(out) + 1
        out.append(
            {
                "rank": idx,
                "mains": mains,
                "stars": stars,
                "score": float(score),
                "why": "balanced score from frequency, delay, and smoothed probability",
            }
        )
        if len(out) == top:
            break
    if len(out) < top:
        used_stars = {row["stars"] for row in out}
        for mains, stars, score in ranked:
            if len(out) == top:
                break
            if stars in used_stars:
                continue
            if any(row["mains"] == mains and row["stars"] == stars for row in out):
                continue
            out.append(
                {
                    "rank": len(out) + 1,
                    "mains": mains,
                    "stars": stars,
                    "score": float(score),
                    "why": "balanced score from frequency, delay, and smoothed probability",
                }
            )
            used_stars.add(stars)
    if len(out) < top:
        for mains, stars, score in ranked:
            if any(row["mains"] == mains and row["stars"] == stars for row in out):
                continue
            out.append(
                {
                    "rank": len(out) + 1,
                    "mains": mains,
                    "stars": stars,
                    "score": float(score),
                    "why": "balanced score from frequency, delay, and smoothed probability",
                }
            )
            if len(out) == top:
                break
    if len(out) < top:
        out = repair_main_diversity(
            ranked,
            out,
            effective_max_main_overlap,
            effective_require_distinct_star_pairs,
            target_count=top,
        )
    if len(out) < top:
        return out
    if effective_require_distinct_star_pairs and len({row["stars"] for row in out}) < min(top, 66):
        repair_star_pairs(out)
    if not _selection_is_diverse(
        out,
        effective_max_main_overlap,
        effective_require_distinct_star_pairs,
    ):
        out = repair_main_diversity(
            ranked,
            out,
            effective_max_main_overlap,
            effective_require_distinct_star_pairs,
        )
    return out


def _selection_is_diverse(
    rows: list[PredictionRow],
    max_main_overlap: int,
    require_distinct_star_pairs: bool,
) -> bool:
    seen_stars: set[tuple[int, int]] = set()
    for idx, row in enumerate(rows):
        if require_distinct_star_pairs:
            if row["stars"] in seen_stars:
                return False
            seen_stars.add(row["stars"])
        for other in rows[idx + 1 :]:
            if len(set(row["mains"]) & set(other["mains"])) > max_main_overlap:
                return False
    return True


def repair_star_pairs(rows: list[PredictionRow]) -> None:
    replacement_pairs = list(combinations(range(1, 13), 2))
    used: set[tuple[int, int]] = set()
    for row in rows:
        if row["stars"] not in used:
            used.add(row["stars"])
            continue
        for pair in replacement_pairs:
            if pair in used:
                continue
            row["stars"] = pair
            row["why"] = "balanced score with enforced star-pair diversity"
            used.add(pair)
            break


def repair_main_diversity(
    ranked: list[tuple[tuple[int, int, int, int, int], tuple[int, int], float]],
    current: list[PredictionRow],
    max_main_overlap: int,
    require_distinct_star_pairs: bool,
    target_count: int | None = None,
) -> list[PredictionRow]:
    repaired: list[PredictionRow] = []
    used_tickets: set[tuple[tuple[int, int, int, int, int], tuple[int, int]]] = set()
    target = len(current) if target_count is None else target_count
    for original in current:
        if _is_diverse_enough(repaired, original["mains"], original["stars"], max_main_overlap, require_distinct_star_pairs):
            repaired.append(original)
            used_tickets.add((original["mains"], original["stars"]))
            continue
        for mains, stars, score in ranked:
            if (mains, stars) in used_tickets:
                continue
            if not _is_diverse_enough(repaired, mains, stars, max_main_overlap, require_distinct_star_pairs):
                continue
            repaired.append(
                {
                    "rank": len(repaired) + 1,
                    "mains": mains,
                    "stars": stars,
                    "score": float(score),
                    "why": "balanced score with enforced combination diversity",
                }
            )
            used_tickets.add((mains, stars))
            break
    if len(repaired) < target:
        for mains, stars, score in fallback_diverse_candidates():
            if len(repaired) == target:
                break
            if (mains, stars) in used_tickets:
                continue
            if not _is_diverse_enough(repaired, mains, stars, max_main_overlap, require_distinct_star_pairs):
                continue
            repaired.append(
                {
                    "rank": len(repaired) + 1,
                    "mains": mains,
                    "stars": stars,
                    "score": score,
                    "why": "deterministic fallback with enforced diversity",
                }
            )
            used_tickets.add((mains, stars))
    for idx, row in enumerate(repaired, start=1):
        row["rank"] = idx
    return repaired


def fallback_diverse_candidates() -> Iterator[tuple[tuple[int, int, int, int, int], tuple[int, int], float]]:
    star_pairs = list(combinations(range(1, 13), 2))
    for main_idx, mains in enumerate(combinations(range(1, 51), 5)):
        stars = star_pairs[main_idx % len(star_pairs)]
        yield mains, stars, 0.0


def save_predictions(predictions: list[PredictionRow], out_dir: str = "outputs") -> None:
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    json_path = p / "predictions_latest.json"
    csv_path = p / "predictions_latest.csv"
    json_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["rank", "mains", "stars", "score", "why"])
        writer.writeheader()
        for row in predictions:
            writer.writerow(row)
