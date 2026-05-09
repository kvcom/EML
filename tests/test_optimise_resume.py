from __future__ import annotations

from pathlib import Path

from euromillions.features import DrawRecord
from euromillions.optimise import optimise_weights


def _synthetic_draws(n: int = 320) -> list[DrawRecord]:
    return [
        DrawRecord(
            i,
            tuple(sorted((((i + j) % 50) + 1 for j in range(5)))),
            tuple(sorted(((i % 12) + 1, ((i + 1) % 12) + 1))),
        )
        for i in range(1, n)
    ]


def test_optuna_resume_with_sqlite_storage(tmp_path: Path) -> None:
    db = tmp_path / "optuna.sqlite"
    log_path = tmp_path / "optimise.log"
    storage = f"sqlite:///{db.as_posix()}"
    study_name = "resume_test"
    draws = _synthetic_draws()

    report1 = optimise_weights(
        draws,
        trials=1,
        top=3,
        study_name=study_name,
        storage=storage,
        evaluation_mode="fast",
        log_path=log_path,
        metadata={"commit_hash": "test", "draw_count": len(draws)},
    )
    report2 = optimise_weights(
        draws,
        trials=1,
        top=3,
        study_name=study_name,
        storage=storage,
        evaluation_mode="fast",
        log_path=log_path,
        metadata={"commit_hash": "test", "draw_count": len(draws)},
    )
    assert report1["completed_trials"] >= 1
    assert report2["completed_trials"] >= report1["completed_trials"] + 1
    assert report2["existing_trials"] >= report1["completed_trials"]
    assert report2["metadata"]["commit_hash"] == "test"
    assert "weighted_freq_weight" in report2["best_params"]
    assert "bayesian_alpha" in report2["best_params"]
    assert "ensemble_weighted_weight" in report2["best_params"]
    assert "weighted_freq_weight" in report2["best_model_params"]
    assert report2["best_params"]["max_main_overlap"] >= 2
    assert log_path.exists()
    assert "starting optimisation" in log_path.read_text(encoding="utf-8")


def test_exact_rank_objective_uses_rank_history(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / "exact_rank_optuna.sqlite"
    log_path = tmp_path / "exact_rank.log"
    storage = f"sqlite:///{db.as_posix()}"
    draws = _synthetic_draws()
    seen_start_indexes: list[int | None] = []

    def fake_rank_historical_winners(
        draws,
        min_training_draws,
        mode="fast",
        thresholds=(1, 3, 10, 100, 500, 1000, 3000),
        model_params=None,
        max_rounds=None,
        start_index=None,
        end_index=None,
    ):
        _ = draws, min_training_draws, mode, thresholds, model_params, max_rounds, end_index
        seen_start_indexes.append(start_index)
        summary = {
            "mode": "fast",
            "evaluated_draws": 2,
            "evaluation_stride": 10,
            "total_ticket_count": 139_838_160,
            "random_expected_top_1000_rate": 1000 / 139_838_160,
            "average_rank": 123.0,
            "median_rank": 100.0,
        }
        return [], summary

    monkeypatch.setattr("euromillions.optimise.rank_historical_winners", fake_rank_historical_winners)

    report = optimise_weights(
        draws,
        trials=1,
        objective_name="exact-rank",
        study_name="exact_rank_test",
        storage=storage,
        evaluation_mode="fast",
        log_path=log_path,
    )

    assert report["objective"] == "exact-rank"
    assert report["best_value"] == -123.0
    assert report["holdout"]["average_rank"] == 123.0
    assert "candidate_pool_multiplier" not in report["best_params"]
    assert "max_main_overlap" not in report["best_params"]
    assert "weighted_freq_weight" in report["best_params"]
    assert seen_start_indexes[-1] is not None
