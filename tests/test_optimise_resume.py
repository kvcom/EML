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
    assert log_path.exists()
    assert "starting optimisation" in log_path.read_text(encoding="utf-8")
