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


def _main_mask(mains: tuple[int, int, int, int, int]) -> int:
    mask = 0
    for number in mains:
        mask |= 1 << number
    return mask


def _is_diverse_enough(
    selected: list[PredictionRow],
    candidate_mains: tuple[int, int, int, int, int],
    candidate_stars: tuple[int, int],
    max_main_overlap: int,
    require_distinct_star_pairs: bool,
) -> bool:
    candidate_mask = _main_mask(candidate_mains)
    for row in selected:
        overlap = (_main_mask(row["mains"]) & candidate_mask).bit_count()
        if overlap > max_main_overlap:
            return False
    if require_distinct_star_pairs:
        used_stars = {row["stars"] for row in selected}
        if len(used_stars) < 66 and candidate_stars in used_stars:
            return False
    return True


def _star_pair_target(top: int, require_distinct_star_pairs: bool) -> int:
    if not require_distinct_star_pairs:
        return 0
    return min(top, 66)


def _has_enough_star_pair_diversity(rows: list[PredictionRow], target: int) -> bool:
    if target == 0:
        return True
    return len({row["stars"] for row in rows}) >= min(len(rows), target)


def _should_skip_repeated_star_pair(
    used_stars: set[tuple[int, int]],
    stars: tuple[int, int],
    target: int,
) -> bool:
    if target == 0 or len(used_stars) >= target:
        return False
    return stars in used_stars


def _renumber(rows: list[PredictionRow]) -> list[PredictionRow]:
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return rows


def _sort_for_prediction(
    rows: list[PredictionRow],
    star_pair_target: int,
) -> list[PredictionRow]:
    if star_pair_target == 0:
        return _renumber(rows)
    seen: set[tuple[int, int]] = set()
    front: list[PredictionRow] = []
    rest: list[PredictionRow] = []
    for row in rows:
        if len(seen) < star_pair_target and row["stars"] not in seen:
            seen.add(row["stars"])
            front.append(row)
        else:
            rest.append(row)
    return _renumber(front + rest)


def _can_select_candidate(
    banned_main_subsets: set[tuple[int, ...]],
    used_stars: set[tuple[int, int]],
    candidate_mains: tuple[int, int, int, int, int],
    candidate_stars: tuple[int, int],
    max_main_overlap: int,
    star_pair_target: int,
) -> bool:
    if star_pair_target and len(used_stars) < star_pair_target and candidate_stars in used_stars:
        return False
    subset_size = max_main_overlap + 1
    return not any(subset in banned_main_subsets for subset in combinations(candidate_mains, subset_size))


def _add_banned_main_subsets(
    banned_main_subsets: set[tuple[int, ...]],
    mains: tuple[int, int, int, int, int],
    max_main_overlap: int,
) -> None:
    subset_size = max_main_overlap + 1
    banned_main_subsets.update(combinations(mains, subset_size))


def _append_prediction(
    out: list[PredictionRow],
    selected_masks: list[int],
    banned_main_subsets: set[tuple[int, ...]],
    used_tickets: set[tuple[tuple[int, int, int, int, int], tuple[int, int]]],
    used_stars: set[tuple[int, int]],
    mains: tuple[int, int, int, int, int],
    stars: tuple[int, int],
    score: float,
    why: str,
    max_main_overlap: int,
) -> None:
    out.append(
        {
            "rank": len(out) + 1,
            "mains": mains,
            "stars": stars,
            "score": float(score),
            "why": why,
        }
    )
    selected_masks.append(_main_mask(mains))
    _add_banned_main_subsets(banned_main_subsets, mains, max_main_overlap)
    used_tickets.add((mains, stars))
    used_stars.add(stars)


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
    star_pair_target = _star_pair_target(top, effective_require_distinct_star_pairs)
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
    selected_masks: list[int] = []
    banned_main_subsets: set[tuple[int, ...]] = set()
    used_tickets: set[tuple[tuple[int, int, int, int, int], tuple[int, int]]] = set()
    used_stars: set[tuple[int, int]] = set()
    for mains, stars, score in ranked:
        if not _can_select_candidate(
            banned_main_subsets,
            used_stars,
            mains,
            stars,
            effective_max_main_overlap,
            star_pair_target,
        ):
            continue
        _append_prediction(
            out,
            selected_masks,
            banned_main_subsets,
            used_tickets,
            used_stars,
            mains,
            stars,
            score,
            "balanced score from frequency, delay, and smoothed probability",
            effective_max_main_overlap,
        )
        if len(out) == top:
            break
    if len(out) < top:
        for mains, stars, score in ranked:
            if len(out) == top:
                break
            if _should_skip_repeated_star_pair(used_stars, stars, star_pair_target):
                continue
            if (mains, stars) in used_tickets:
                continue
            if not _can_select_candidate(
                banned_main_subsets,
                used_stars,
                mains,
                stars,
                effective_max_main_overlap,
                0,
            ):
                continue
            _append_prediction(
                out,
                selected_masks,
                banned_main_subsets,
                used_tickets,
                used_stars,
                mains,
                stars,
                score,
                "balanced score from frequency, delay, and smoothed probability",
                effective_max_main_overlap,
            )
    if len(out) < top:
        for mains, stars, score in ranked:
            if (mains, stars) in used_tickets:
                continue
            if not _can_select_candidate(
                banned_main_subsets,
                used_stars,
                mains,
                stars,
                effective_max_main_overlap,
                0,
            ):
                continue
            _append_prediction(
                out,
                selected_masks,
                banned_main_subsets,
                used_tickets,
                used_stars,
                mains,
                stars,
                score,
                "balanced score from frequency, delay, and smoothed probability",
                effective_max_main_overlap,
            )
            if len(out) == top:
                break
    if len(out) < top:
        for mains, stars, score in fallback_diverse_candidates():
            if len(out) == top:
                break
            if (mains, stars) in used_tickets:
                continue
            if not _can_select_candidate(
                banned_main_subsets,
                used_stars,
                mains,
                stars,
                effective_max_main_overlap,
                star_pair_target,
            ):
                continue
            _append_prediction(
                out,
                selected_masks,
                banned_main_subsets,
                used_tickets,
                used_stars,
                mains,
                stars,
                score,
                "deterministic fallback with enforced diversity",
                effective_max_main_overlap,
            )
    if len(out) < top:
        return _sort_for_prediction(out, star_pair_target)
    if not _has_enough_star_pair_diversity(out, star_pair_target):
        repair_star_pairs(out)
    if not _selection_is_diverse(
        out,
        effective_max_main_overlap,
        effective_require_distinct_star_pairs,
        star_pair_target,
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
    star_pair_target: int | None = None,
) -> bool:
    target = _star_pair_target(len(rows), require_distinct_star_pairs) if star_pair_target is None else star_pair_target
    seen_stars: set[tuple[int, int]] = set()
    for idx, row in enumerate(rows):
        if require_distinct_star_pairs:
            if len(seen_stars) < target and row["stars"] in seen_stars:
                return False
            seen_stars.add(row["stars"])
        for other in rows[idx + 1 :]:
            if len(set(row["mains"]) & set(other["mains"])) > max_main_overlap:
                return False
    return _has_enough_star_pair_diversity(rows, target)


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
    return _renumber(repaired)


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
