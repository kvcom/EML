from __future__ import annotations

from euromillions.candidate_validation import CandidateSpec, build_validation_windows, validate_candidates
from euromillions.features import DrawRecord


def _draws(n: int = 180) -> list[DrawRecord]:
    return [
        DrawRecord(
            i,
            tuple(sorted((((i + offset) % 50) + 1 for offset in range(5)))),
            tuple(sorted(((i % 12) + 1, ((i + 1) % 12) + 1))),
        )
        for i in range(1, n + 1)
    ]


def test_build_validation_windows_before_holdout() -> None:
    windows = build_validation_windows(
        total_draws=200,
        holdout_start_index=160,
        window_count=3,
        window_size=10,
        gap=5,
    )

    assert len(windows) == 3
    assert all(int(window["end_index"]) <= 160 for window in windows)
    assert all(int(window["draws"]) == 10 for window in windows)


def test_validate_candidates_ranks_by_validation_mean(monkeypatch) -> None:
    def fake_rank_historical_winners(
        draws,
        min_training_draws,
        mode="fast",
        thresholds=(1, 3, 10, 100, 500, 1000, 3000),
        model_params=None,
        max_rounds=None,
        start_index=None,
        end_index=None,
        rank_backend="auto",
    ):
        _ = draws, mode, thresholds, model_params, max_rounds, start_index, end_index, rank_backend
        average_rank = float(min_training_draws)
        return [], {
            "evaluated_draws": 2,
            "average_rank": average_rank,
            "median_rank": average_rank,
        }

    monkeypatch.setattr(
        "euromillions.candidate_validation.rank_historical_winners",
        fake_rank_historical_winners,
    )

    report = validate_candidates(
        _draws(),
        [
            CandidateSpec("later", {"min_training_draws": 150.0}),
            CandidateSpec("earlier", {"min_training_draws": 120.0}),
        ],
        holdout_fraction=0.2,
        window_count=2,
        window_size=5,
        gap=5,
    )

    assert report["ranked_by_validation_mean"][0]["label"] == "earlier"
    assert report["ranked_by_holdout"][0]["label"] == "earlier"
