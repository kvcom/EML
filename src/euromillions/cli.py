from __future__ import annotations

import json
from typing import Literal
from typing import Any
from pathlib import Path

import typer
from sqlalchemy import func, select

from euromillions.backtest import run_walk_forward
from euromillions.combinations import build_main_combinations, build_star_combinations
from euromillions.config import load_config
from euromillions.db import begin, create_all_tables, create_db_engine
from euromillions.ingest_excel import ingest_draw_rows, read_excel_draws
from euromillions.ingest_web import reconcile_and_insert
from euromillions.io import load_draw_records
from euromillions.optimise import optimise_weights, recommended_trials
from euromillions.predict import generate_predictions, save_predictions
from euromillions.schema import draws
from euromillions.sources.beatlottery import BeatLotterySource
from euromillions.sources.euro_millions_com import EuroMillionsComSource
from euromillions.sources.national_lottery import NationalLotterySource
from euromillions.sources.pedro_api import PedroApiSource

app = typer.Typer()


def _engine() -> Any:
    cfg = load_config()
    return create_db_engine(cfg.database_path)


@app.command("init-db")
def init_db(excel: str = typer.Option(..., "--excel")) -> None:
    engine = _engine()
    create_all_tables(engine)
    with begin(engine) as conn:
        rows = read_excel_draws(excel)
        inserted = ingest_draw_rows(conn, rows)
        mains = build_main_combinations(conn)
        stars = build_star_combinations(conn)
        draw_count = conn.execute(select(func.count()).select_from(draws)).scalar_one()
        first = conn.execute(select(func.min(draws.c.draw_date))).scalar_one()
        latest = conn.execute(select(func.max(draws.c.draw_date))).scalar_one()
        typer.echo(
            f"draws={draw_count} first={first} latest={latest} "
            f"inserted={inserted} main_combinations_added={mains} star_combinations_added={stars}"
        )


@app.command("build-combinations")
def build_combinations() -> None:
    engine = _engine()
    with begin(engine) as conn:
        mains = build_main_combinations(conn)
        stars = build_star_combinations(conn)
        typer.echo(f"main added={mains}, stars added={stars}")


@app.command("update-results")
def update_results() -> None:
    engine = _engine()
    with begin(engine) as conn:
        inserted, warnings = reconcile_and_insert(
            conn, [EuroMillionsComSource(), BeatLotterySource(), PedroApiSource(), NationalLotterySource()]
        )
        typer.echo(f"inserted_new_draws={inserted}")
        for w in warnings:
            typer.echo(f"warning: {w}")


@app.command("backtest")
def backtest(
    top: int = typer.Option(3, "--top"),
    from_date: str | None = typer.Option(None, "--from-date"),
    evaluation_mode: Literal["fast", "full"] = typer.Option("fast", "--evaluation-mode"),
) -> None:
    engine = _engine()
    with begin(engine) as conn:
        records = load_draw_records(conn)
    if from_date:
        _ = from_date
    cfg = load_config()
    result = run_walk_forward(
        records,
        top=top,
        min_training_draws=cfg.min_training_draws,
        seed=cfg.random_seed,
        evaluation_mode=evaluation_mode,
    )
    typer.echo(json.dumps(result.__dict__, indent=2))


@app.command("optimise")
def optimise(
    trials: int | None = typer.Option(None, "--trials"),
    top: int = typer.Option(3, "--top"),
    study_name: str = typer.Option("eml_optimisation", "--study-name"),
    storage: str = typer.Option("sqlite:///outputs/optuna_study.sqlite", "--storage"),
    n_jobs: int = typer.Option(1, "--n-jobs"),
    timeout_seconds: int | None = typer.Option(None, "--timeout-seconds"),
    evaluation_mode: Literal["fast", "full"] = typer.Option("fast", "--evaluation-mode"),
) -> None:
    engine = _engine()
    with begin(engine) as conn:
        records = load_draw_records(conn)
    picked_trials = recommended_trials(len(records)) if trials is None else trials
    report = optimise_weights(
        records,
        trials=picked_trials,
        top=top,
        study_name=study_name,
        storage=storage,
        n_jobs=n_jobs,
        timeout_seconds=timeout_seconds,
        evaluation_mode=evaluation_mode,
        holdout_fraction=load_config().optimisation.holdout_fraction,
    )
    Path("outputs").mkdir(parents=True, exist_ok=True)
    Path("outputs/best_params.json").write_text(json.dumps(report["best_params"], indent=2), encoding="utf-8")
    typer.echo(f"recommended_trials={recommended_trials(len(records))}")
    typer.echo(f"trials_used={picked_trials}")
    typer.echo(f"completed_trials={report['completed_trials']}")
    typer.echo(f"evaluation_mode={report['evaluation_mode']}")
    typer.echo(
        "holdout_metrics="
        + json.dumps(
            {
                "model_points": report["holdout"]["model_points"],
                "baseline_points": report["holdout"]["baseline_points"],
                "uplift_points": report["holdout"]["uplift_points"],
                "rounds": report["holdout"]["rounds"],
                "sampled": report["holdout"]["sampled"],
                "evaluation_stride": report["holdout"]["evaluation_stride"],
            }
        )
    )
    typer.echo(json.dumps(report, indent=2))


@app.command("predict")
def predict(
    top: int = typer.Option(3, "--top"),
    max_main_overlap: int = typer.Option(3, "--max-main-overlap"),
    require_distinct_star_pairs: bool = typer.Option(True, "--require-distinct-star-pairs"),
) -> None:
    update_results()
    engine = _engine()
    with begin(engine) as conn:
        records = load_draw_records(conn)
    ranked = generate_predictions(
        records,
        top=top,
        max_main_overlap=max_main_overlap,
        require_distinct_star_pairs=require_distinct_star_pairs,
    )
    save_predictions(ranked)
    for row in ranked:
        mains_tuple = row["mains"]
        stars_tuple = row["stars"]
        mains = " ".join(f"{n:02d}" for n in mains_tuple)
        stars = " ".join(f"{n:02d}" for n in stars_tuple)
        typer.echo(
            f"Rank {row['rank']}\n"
            f"Main numbers: {mains}\n"
            f"Lucky Stars: {stars}\n"
            f"Score: {row['score']:.5f}\n"
            f"Why: {row['why']}\n"
        )


@app.command("run")
def run_pipeline() -> None:
    update_results()
    predict(top=load_config().top_predictions)


if __name__ == "__main__":
    app()
