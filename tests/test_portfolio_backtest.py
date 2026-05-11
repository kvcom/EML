from __future__ import annotations

from euromillions.features import DrawRecord
from euromillions.portfolio_backtest import prize_tier, run_portfolio_backtest


def _draws(n: int = 260) -> list[DrawRecord]:
    return [DrawRecord(i, (1, 2, 3, 4, 5), (1, 2)) for i in range(1, n + 1)]


def test_prize_tier_matches_euromillions_prize_categories() -> None:
    assert prize_tier(5, 2) == "5+2"
    assert prize_tier(2, 0) == "2+0"
    assert prize_tier(1, 1) is None
    assert prize_tier(0, 2) is None


def test_portfolio_backtest_reports_model_and_random(monkeypatch) -> None:
    def fake_generate_predictions(
        history,
        top,
        seed=42,
        max_main_overlap=None,
        require_distinct_star_pairs=None,
        model_params=None,
    ):
        _ = history, seed, max_main_overlap, require_distinct_star_pairs, model_params
        return [
            {
                "rank": rank,
                "mains": (1, 2, 3, 4, 5),
                "stars": (1, 2),
                "score": 1.0,
                "why": "test",
            }
            for rank in range(1, top + 1)
        ]

    monkeypatch.setattr("euromillions.portfolio_backtest.generate_predictions", fake_generate_predictions)

    report = run_portfolio_backtest(
        _draws(),
        top=3,
        min_training_draws=100,
        mode="fast",
        max_rounds=2,
    )

    assert report["rounds"] == 2
    assert report["model"]["winning_rounds"] == 2
    assert report["model"]["tier_counts"]["5+2"] == 6
    assert "random_baseline" in report
