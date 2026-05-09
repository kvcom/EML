from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations

from euromillions.features import DrawRecord
from euromillions.rank_history import (
    bucket_for_rank,
    exact_ticket_rank,
    exact_ticket_rank_vectorized,
    parse_thresholds,
    rank_historical_winners,
)


def test_rank_history_uses_only_previous_draws(monkeypatch) -> None:
    draws = [
        DrawRecord(i, tuple(range(1, 6)), (1, 2))
        for i in range(1, 10)
    ]
    seen_history_lengths: list[int] = []

    def fake_exact_ticket_rank(
        history,
        actual_mains,
        actual_stars,
        model_params=None,
        rank_backend="auto",
    ):
        _ = actual_mains, actual_stars, model_params, rank_backend
        seen_history_lengths.append(len(history))
        return 0.5, 42

    monkeypatch.setattr("euromillions.rank_history.exact_ticket_rank", fake_exact_ticket_rank)
    rows, summary = rank_historical_winners(
        draws,
        min_training_draws=3,
        mode="full",
        thresholds=(10, 100),
        max_rounds=2,
    )

    assert seen_history_lengths == [3, 4]
    assert [row.draw_id for row in rows] == [4, 5]
    assert all(row.exact_rank == 42 for row in rows)
    assert summary["evaluated_draws"] == 2
    assert summary["pct_top_100"] == 1.0


def test_rank_history_threshold_helpers() -> None:
    assert parse_thresholds("1000,1,10,10") == (1, 10, 1000)
    assert bucket_for_rank(3, (1, 3, 10)) == "top_3"
    assert bucket_for_rank(11, (1, 3, 10)) == "outside"


def test_exact_ticket_rank_does_not_count_equal_score_ties(monkeypatch) -> None:
    draws = [DrawRecord(i, tuple(range(1, 6)), (1, 2)) for i in range(1, 5)]

    def tiny_combinations(iterable: Iterable[int], r: int) -> Iterable[tuple[int, ...]]:
        values = tuple(iterable)
        if len(values) == 50 and r == 5:
            return ((1, 2, 3, 4, 5), (1, 2, 3, 4, 6))
        if len(values) == 12 and r == 2:
            return ((1, 2), (1, 3))
        return combinations(values, r)

    monkeypatch.setattr("euromillions.rank_history.combinations", tiny_combinations)
    score, rank = exact_ticket_rank(
        draws,
        (1, 2, 3, 4, 6),
        (1, 3),
        model_params={
            "ensemble_weighted_weight": 0,
            "ensemble_bayesian_weight": 0,
        },
    )

    assert score == 0
    assert rank == 1


def test_vectorized_exact_ticket_rank_does_not_count_equal_score_ties() -> None:
    draws = [DrawRecord(i, tuple(range(1, 6)), (1, 2)) for i in range(1, 5)]
    score, rank = exact_ticket_rank_vectorized(
        draws,
        (1, 2, 3, 4, 6),
        (1, 3),
        model_params={
            "ensemble_weighted_weight": 0,
            "ensemble_bayesian_weight": 0,
        },
    )

    assert score == 0
    assert rank == 1
