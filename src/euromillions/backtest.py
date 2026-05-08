from __future__ import annotations

import random
from dataclasses import dataclass

from euromillions.features import DrawRecord
from euromillions.predict import generate_predictions


@dataclass(frozen=True)
class BacktestResult:
    rounds: int
    average_best_main_hits: float
    average_best_star_hits: float
    random_baseline_main_hits: float
    random_baseline_star_hits: float


def _hits(pred_mains: tuple[int, ...], pred_stars: tuple[int, ...], actual: DrawRecord) -> tuple[int, int]:
    return len(set(pred_mains) & set(actual.mains)), len(set(pred_stars) & set(actual.stars))


def run_walk_forward(
    draws: list[DrawRecord],
    top: int = 3,
    min_training_draws: int = 200,
    seed: int = 42,
    evaluation_stride: int = 10,
    max_rounds: int = 250,
) -> BacktestResult:
    rng = random.Random(seed)
    rounds = 0
    total_main = 0.0
    total_star = 0.0
    baseline_main = 0.0
    baseline_star = 0.0
    for idx in range(min_training_draws, len(draws), max(1, evaluation_stride)):
        if rounds >= max_rounds:
            break
        history = draws[:idx]
        actual = draws[idx]
        preds = generate_predictions(history, top=top, seed=seed + idx)
        best_m, best_s = 0, 0
        for p in preds:
            m_hits, s_hits = _hits(p["mains"], p["stars"], actual)
            best_m, best_s = max(best_m, m_hits), max(best_s, s_hits)
        total_main += best_m
        total_star += best_s
        rounds += 1
        rm = tuple(sorted(rng.sample(range(1, 51), 5)))
        rs = tuple(sorted(rng.sample(range(1, 13), 2)))
        bm, bs = _hits(rm, rs, actual)
        baseline_main += bm
        baseline_star += bs
    if rounds == 0:
        return BacktestResult(0, 0.0, 0.0, 0.0, 0.0)
    return BacktestResult(
        rounds=rounds,
        average_best_main_hits=total_main / rounds,
        average_best_star_hits=total_star / rounds,
        random_baseline_main_hits=baseline_main / rounds,
        random_baseline_star_hits=baseline_star / rounds,
    )
