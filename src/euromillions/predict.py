from __future__ import annotations

import csv
import json
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


def generate_predictions(history: list[DrawRecord], top: int, seed: int = 42) -> list[PredictionRow]:
    model = EnsembleModel(
        weighted=WeightedStatisticalModel(main_pool_size=500, star_pool_size=66),
        bayesian=BayesianFrequencyModel(alpha=1.0),
    )
    ranked = model.predict(history, top=top)
    out: list[PredictionRow] = []
    for idx, (mains, stars, score) in enumerate(ranked, start=1):
        out.append(
            {
                "rank": idx,
                "mains": mains,
                "stars": stars,
                "score": float(score),
                "why": "balanced score from frequency, delay, and smoothed probability",
            }
        )
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
