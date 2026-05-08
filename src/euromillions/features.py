from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DrawRecord:
    draw_id: int
    mains: tuple[int, int, int, int, int]
    stars: tuple[int, int]
    draw_date: date | None = None


def compute_frequency_features(history: list[DrawRecord], window: int | None = None) -> dict[int, float]:
    sample = history[-window:] if window else history
    cnt: Counter[int] = Counter()
    for draw in sample:
        cnt.update(draw.mains)
    total = max(1, len(sample) * 5)
    return {n: cnt[n] / total for n in range(1, 51)}


def compute_delay_features(history: list[DrawRecord]) -> dict[int, int]:
    last_seen = {n: -1 for n in range(1, 51)}
    for idx, draw in enumerate(history):
        for n in draw.mains:
            last_seen[n] = idx
    latest_idx = len(history) - 1
    return {n: latest_idx - last_seen[n] if last_seen[n] >= 0 else len(history) for n in range(1, 51)}


def compute_pair_counts(history: list[DrawRecord]) -> dict[tuple[int, int], int]:
    counts: Counter[tuple[int, int]] = Counter()
    for draw in history:
        for p in combinations(draw.mains, 2):
            counts[p] += 1
    return dict(counts)


def draws_to_dataframe(history: list[DrawRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "draw_id": d.draw_id,
                "m1": d.mains[0],
                "m2": d.mains[1],
                "m3": d.mains[2],
                "m4": d.mains[3],
                "m5": d.mains[4],
                "s1": d.stars[0],
                "s2": d.stars[1],
            }
            for d in history
        ]
    )


def decayed_frequency(history: list[DrawRecord], decay: float = 0.98) -> dict[int, float]:
    weights = np.array([decay ** (len(history) - i - 1) for i in range(len(history))], dtype=float)
    score = {n: 0.0 for n in range(1, 51)}
    for idx, draw in enumerate(history):
        for n in draw.mains:
            score[n] += float(weights[idx])
    total = max(1e-12, float(weights.sum()) * 5.0)
    return {k: v / total for k, v in score.items()}
