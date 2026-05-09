from __future__ import annotations

import csv
import json
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
        rank_backend="auto",
    ):
        _ = draws, min_training_draws, mode, thresholds, model_params, max_rounds, end_index, rank_backend
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
        progress_path=tmp_path / "progress.json",
        trials_path=tmp_path / "trials.csv",
        top_trial_holdout_count=0,
    )

    assert report["objective"] == "exact-rank"
    assert report["best_value"] == -123.0
    assert report["holdout"]["average_rank"] == 123.0
    assert "candidate_pool_multiplier" not in report["best_params"]
    assert "max_main_overlap" not in report["best_params"]
    assert "weighted_freq_weight" in report["best_params"]
    assert seen_start_indexes[-1] is not None
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "finished"
    assert progress["completed_new_trials"] == 1
    with (tmp_path / "trials.csv").open(newline="", encoding="utf-8") as fh:
        trial_rows = list(csv.DictReader(fh))
    assert len(trial_rows) == 1
    assert trial_rows[0]["state"] == "COMPLETE"


def test_exact_rank_early_stopping_uses_validation_not_holdout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    db = tmp_path / "early_stop.sqlite"
    storage = f"sqlite:///{db.as_posix()}"
    draws = _synthetic_draws()
    validation_ranks = iter([100.0, 110.0])
    validation_calls = 0
    holdout_calls = 0

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
        nonlocal validation_calls, holdout_calls
        _ = draws, min_training_draws, mode, thresholds, model_params, rank_backend
        average_rank = 50.0
        if start_index is not None and end_index is not None:
            validation_calls += 1
            average_rank = next(validation_ranks)
        elif start_index is not None:
            holdout_calls += 1
            average_rank = 120.0
        summary = {
            "mode": "fast",
            "evaluated_draws": max_rounds or 2,
            "evaluation_stride": 10,
            "total_ticket_count": 139_838_160,
            "random_expected_top_1000_rate": 1000 / 139_838_160,
            "average_rank": average_rank,
            "median_rank": average_rank,
        }
        return [], summary

    monkeypatch.setattr("euromillions.optimise.rank_historical_winners", fake_rank_historical_winners)

    report = optimise_weights(
        draws,
        trials=3,
        objective_name="exact-rank",
        study_name="early_stop_test",
        storage=storage,
        evaluation_mode="fast",
        early_stop_patience=1,
        early_stop_validation_rounds=1,
        top_trial_holdout_count=0,
    )

    assert report["completed_trials"] == 2
    assert report["early_stop"]["stopped"] is True
    assert report["early_stop"]["best_validation_rank"] == 100.0
    assert report["early_stop"]["stale_trials"] == 1
    assert validation_calls == 2
    assert holdout_calls == 1


def test_exact_rank_reports_top_trials_by_holdout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    db = tmp_path / "top_trials.sqlite"
    storage = f"sqlite:///{db.as_posix()}"
    draws = _synthetic_draws()

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
        _ = draws, mode, thresholds, model_params, max_rounds, end_index, rank_backend
        if start_index is None or start_index < 200:
            average_rank = float(400 - min_training_draws)
        else:
            average_rank = float(min_training_draws)
        summary = {
            "mode": "fast",
            "evaluated_draws": 2,
            "evaluation_stride": 10,
            "total_ticket_count": 139_838_160,
            "random_expected_top_1000_rate": 1000 / 139_838_160,
            "average_rank": average_rank,
            "median_rank": average_rank,
        }
        return [], summary

    monkeypatch.setattr("euromillions.optimise.rank_historical_winners", fake_rank_historical_winners)

    report = optimise_weights(
        draws,
        trials=3,
        objective_name="exact-rank",
        study_name="top_trial_test",
        storage=storage,
        evaluation_mode="fast",
        top_trial_holdout_count=3,
    )

    top_trials = report["top_trial_holdout"]["trials"]
    assert 1 <= len(top_trials) <= 3
    holdout_ranks = [row["holdout_average_rank"] for row in top_trials]
    assert holdout_ranks == sorted(holdout_ranks)
    report_path = Path("outputs/top_trial_holdout_report.json")
    assert report_path.exists()


def test_exact_rank_rolling_objective_uses_multiple_windows(
    monkeypatch,
    tmp_path: Path,
) -> None:
    db = tmp_path / "rolling.sqlite"
    storage = f"sqlite:///{db.as_posix()}"
    draws = _synthetic_draws()
    objective_ranges: list[tuple[int | None, int | None, int | None]] = []

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
        _ = draws, min_training_draws, mode, thresholds, model_params, rank_backend
        if end_index is not None:
            objective_ranges.append((start_index, end_index, max_rounds))
        average_rank = float(start_index or 100)
        summary = {
            "mode": "fast",
            "evaluated_draws": max_rounds or 2,
            "evaluation_stride": 10,
            "total_ticket_count": 139_838_160,
            "random_expected_top_1000_rate": 1000 / 139_838_160,
            "average_rank": average_rank,
            "median_rank": average_rank,
        }
        return [], summary

    monkeypatch.setattr("euromillions.optimise.rank_historical_winners", fake_rank_historical_winners)

    report = optimise_weights(
        draws,
        trials=1,
        objective_name="exact-rank",
        study_name="rolling_test",
        storage=storage,
        evaluation_mode="fast",
        rolling_windows=3,
        rolling_window_rounds=1,
        top_trial_holdout_count=0,
    )

    assert report["rolling_objective"]["enabled"] is True
    assert report["rolling_objective"]["windows"] == 3
    assert len(objective_ranges) == 3
    assert {max_rounds for _, _, max_rounds in objective_ranges} == {1}
