from __future__ import annotations

from euromillions.dynamic_params import run_dynamic_params_experiment
from euromillions.features import DrawRecord


def _draws(n: int = 360) -> list[DrawRecord]:
    return [
        DrawRecord(
            i,
            tuple(sorted((((i + offset) % 50) + 1 for offset in range(5)))),
            tuple(sorted(((i % 12) + 1, ((i + 1) % 12) + 1))),
        )
        for i in range(1, n + 1)
    ]


def test_dynamic_params_forecasts_from_previous_oracles(monkeypatch) -> None:
    optimised_targets: list[int] = []

    def fake_score_one_draw(
        draws,
        *,
        target_index,
        min_training_draws,
        model_params,
        mode,
        rank_backend,
    ):
        _ = draws, min_training_draws, mode, rank_backend
        rank = 1000.0 - float(model_params.get("weighted_freq_weight", 0.0) * 100.0)
        return {"rank_sum": rank, "average_rank": rank, "median_rank": rank, "evaluated_draws": 1}

    def fake_optimise_oracle_params_for_draw(
        draws,
        *,
        target_index,
        trials,
        mode,
        rank_backend,
    ):
        _ = draws, trials, mode, rank_backend
        optimised_targets.append(target_index)
        value = len(optimised_targets) / 10.0
        return {
            "target_index": target_index,
            "best_rank": 100.0,
            "params": {
                "min_training_draws": 100.0,
                "weighted_freq_weight": value,
            },
            "trials": trials,
        }

    monkeypatch.setattr("euromillions.dynamic_params._score_one_draw", fake_score_one_draw)
    monkeypatch.setattr(
        "euromillions.dynamic_params._optimise_oracle_params_for_draw",
        fake_optimise_oracle_params_for_draw,
    )

    report = run_dynamic_params_experiment(
        _draws(),
        baseline_params={"min_training_draws": 100.0, "weighted_freq_weight": 0.0},
        start_index=300,
        max_targets=3,
        stride=1,
        oracle_trials=2,
        forecast_lookback=2,
    )

    assert report["summary"]["targets"] == 3
    assert report["summary"]["evaluated_dynamic_targets"] == 2
    assert report["rows"][0]["dynamic_rank"] is None
    assert report["rows"][1]["forecast_params"]["weighted_freq_weight"] == 0.1
    assert report["rows"][2]["forecast_params"]["weighted_freq_weight"] == 0.15000000000000002
    assert optimised_targets == [300, 301, 302]
