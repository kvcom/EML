from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path
from typing import TypedDict

from euromillions.features import DrawRecord
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
    max_main_overlap: int = 3,
    require_distinct_star_pairs: bool = True,
) -> list[PredictionRow]:
    model = EnsembleModel(
        weighted=WeightedStatisticalModel(main_pool_size=500, star_pool_size=66),
        bayesian=BayesianFrequencyModel(alpha=1.0),
    )
    _ = seed
    candidate_pool = max(top * 20, 50)
    ranked = model.predict(history, top=candidate_pool)
    out: list[PredictionRow] = []
    for mains, stars, score in ranked:
        if not _is_diverse_enough(out, mains, stars, max_main_overlap, require_distinct_star_pairs):
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
    if require_distinct_star_pairs and len({row["stars"] for row in out}) < min(top, 66):
        replacement_pairs = list(combinations(range(1, 13), 2))
        used = {row["stars"] for row in out}
        for idx, row in enumerate(out):
            if idx == 0:
                continue
            if row["stars"] not in used:
                continue
            for pair in replacement_pairs:
                if pair in used:
                    continue
                row["stars"] = (pair[0], pair[1])
                row["why"] = "balanced score with enforced star-pair diversity"
                used.add(row["stars"])
                break
    if out:
        seen_tickets = {(row["mains"], row["stars"]) for row in out}
        for idx in range(1, len(out)):
            if len(set(out[0]["mains"]) & set(out[idx]["mains"])) <= max_main_overlap:
                continue
            for mains, stars, score in ranked:
                if (mains, stars) in seen_tickets:
                    continue
                if len(set(out[0]["mains"]) & set(mains)) > max_main_overlap:
                    continue
                out[idx]["mains"] = mains
                out[idx]["stars"] = stars if not require_distinct_star_pairs else out[idx]["stars"]
                out[idx]["score"] = float(score)
                out[idx]["why"] = "balanced score with enforced combination diversity"
                seen_tickets.add((out[idx]["mains"], out[idx]["stars"]))
                break
    return out


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
