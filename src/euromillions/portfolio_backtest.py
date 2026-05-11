from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from euromillions.backtest import _hits
from euromillions.features import DrawRecord
from euromillions.predict import PredictionRow, generate_predictions

PRIZE_TIERS: tuple[tuple[int, int], ...] = (
    (5, 2),
    (5, 1),
    (5, 0),
    (4, 2),
    (4, 1),
    (3, 2),
    (4, 0),
    (2, 2),
    (3, 1),
    (3, 0),
    (1, 2),
    (2, 1),
    (2, 0),
)


def prize_tier(main_hits: int, star_hits: int) -> str | None:
    return f"{main_hits}+{star_hits}" if (main_hits, star_hits) in PRIZE_TIERS else None


def _empty_tier_counts() -> dict[str, int]:
    return {f"{main}+{stars}": 0 for main, stars in PRIZE_TIERS}


def _random_portfolio(rng: random.Random, top: int) -> list[PredictionRow]:
    rows: list[PredictionRow] = []
    used: set[tuple[tuple[int, int, int, int, int], tuple[int, int]]] = set()
    while len(rows) < top:
        mains = tuple(sorted(rng.sample(range(1, 51), 5)))
        stars = tuple(sorted(rng.sample(range(1, 13), 2)))
        if (mains, stars) in used:
            continue
        used.add((mains, stars))
        rows.append(
            {
                "rank": len(rows) + 1,
                "mains": mains,
                "stars": stars,
                "score": 0.0,
                "why": "random baseline",
            }
        )
    return rows


def _portfolio_summary(predictions: list[PredictionRow], actual: DrawRecord) -> dict[str, Any]:
    best_main_hits = 0
    best_star_hits = 0
    best_total_hits = 0
    tier_counts = Counter()
    winning_tickets = 0
    for row in predictions:
        main_hits, star_hits = _hits(row["mains"], row["stars"], actual)
        best_main_hits = max(best_main_hits, main_hits)
        best_star_hits = max(best_star_hits, star_hits)
        best_total_hits = max(best_total_hits, main_hits + star_hits)
        tier = prize_tier(main_hits, star_hits)
        if tier is not None:
            tier_counts[tier] += 1
            winning_tickets += 1
    counts = _empty_tier_counts()
    counts.update(tier_counts)
    return {
        "best_main_hits": best_main_hits,
        "best_star_hits": best_star_hits,
        "best_total_hits": best_total_hits,
        "winning_tickets": winning_tickets,
        "tier_counts": counts,
    }


def run_portfolio_backtest(
    draws: list[DrawRecord],
    *,
    top: int = 3,
    min_training_draws: int = 200,
    seed: int = 42,
    mode: Literal["fast", "full"] = "fast",
    evaluation_stride: int | None = None,
    max_rounds: int | None = None,
    start_index: int | None = None,
    end_index: int | None = None,
    model_params: dict[str, float] | None = None,
    max_main_overlap: int | None = None,
    require_distinct_star_pairs: bool | None = None,
    random_baseline_runs: int = 1,
) -> dict[str, Any]:
    if random_baseline_runs < 1:
        raise ValueError("random_baseline_runs must be at least 1")
    stride = evaluation_stride if evaluation_stride is not None else (10 if mode == "fast" else 1)
    limit = max_rounds if max_rounds is not None else (250 if mode == "fast" else len(draws))
    start = max(min_training_draws, start_index or min_training_draws)
    stop = end_index if end_index is not None else len(draws)
    rng = random.Random(seed)

    model_tiers = _empty_tier_counts()
    random_tiers = _empty_tier_counts()
    model_winning_rounds = 0
    random_winning_rounds = 0
    model_winning_tickets = 0
    random_winning_tickets = 0
    model_best_main_hits = 0.0
    model_best_star_hits = 0.0
    model_best_total_hits = 0.0
    random_best_main_hits = 0.0
    random_best_star_hits = 0.0
    random_best_total_hits = 0.0
    rounds = 0

    for idx in range(start, stop, max(1, stride)):
        if rounds >= limit:
            break
        history = draws[:idx]
        actual = draws[idx]
        predictions = generate_predictions(
            history,
            top=top,
            seed=seed + idx,
            max_main_overlap=max_main_overlap,
            require_distinct_star_pairs=require_distinct_star_pairs,
            model_params=model_params,
        )
        model_summary = _portfolio_summary(predictions, actual)

        model_winning_tickets += int(model_summary["winning_tickets"])
        model_winning_rounds += int(model_summary["winning_tickets"] > 0)
        model_best_main_hits += float(model_summary["best_main_hits"])
        model_best_star_hits += float(model_summary["best_star_hits"])
        model_best_total_hits += float(model_summary["best_total_hits"])
        for tier, count in model_summary["tier_counts"].items():
            model_tiers[tier] += int(count)
        for _ in range(random_baseline_runs):
            random_summary = _portfolio_summary(_random_portfolio(rng, top), actual)
            random_winning_tickets += int(random_summary["winning_tickets"])
            random_winning_rounds += int(random_summary["winning_tickets"] > 0)
            random_best_main_hits += float(random_summary["best_main_hits"])
            random_best_star_hits += float(random_summary["best_star_hits"])
            random_best_total_hits += float(random_summary["best_total_hits"])
            for tier, count in random_summary["tier_counts"].items():
                random_tiers[tier] += int(count)
        rounds += 1

    if rounds == 0:
        average_denominator = 1
    else:
        average_denominator = rounds
    random_denominator = average_denominator * random_baseline_runs
    return {
        "top": top,
        "mode": mode,
        "rounds": rounds,
        "evaluated_draws": rounds * max(1, stride),
        "evaluation_stride": max(1, stride),
        "random_baseline_runs": random_baseline_runs,
        "model": {
            "winning_rounds": model_winning_rounds,
            "winning_tickets": model_winning_tickets,
            "winning_round_rate": model_winning_rounds / average_denominator,
            "average_best_main_hits": model_best_main_hits / average_denominator,
            "average_best_star_hits": model_best_star_hits / average_denominator,
            "average_best_total_hits": model_best_total_hits / average_denominator,
            "tier_counts": model_tiers,
        },
        "random_baseline": {
            "winning_rounds": random_winning_rounds / random_baseline_runs,
            "winning_tickets": random_winning_tickets / random_baseline_runs,
            "winning_round_rate": random_winning_rounds / random_denominator,
            "average_best_main_hits": random_best_main_hits / random_denominator,
            "average_best_star_hits": random_best_star_hits / random_denominator,
            "average_best_total_hits": random_best_total_hits / random_denominator,
            "tier_counts": {tier: count / random_baseline_runs for tier, count in random_tiers.items()},
        },
    }


def save_portfolio_backtest_report(
    report: dict[str, Any],
    out_path: str = "outputs/portfolio_backtest_report.json",
) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
