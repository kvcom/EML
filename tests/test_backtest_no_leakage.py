from __future__ import annotations

from euromillions.backtest import run_walk_forward
from euromillions.features import DrawRecord


def test_backtest_deterministic_with_seed() -> None:
    draws = [
        DrawRecord(
            i,
            tuple(sorted((((i + j) % 50) + 1 for j in range(5)))),
            tuple(sorted(((i % 12) + 1, ((i + 1) % 12) + 1))),
        )
        for i in range(1, 260)
    ]
    r1 = run_walk_forward(draws, top=3, min_training_draws=200, seed=42)
    r2 = run_walk_forward(draws, top=3, min_training_draws=200, seed=42)
    assert r1 == r2
