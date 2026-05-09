from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import typer
from sqlalchemy import func, select

from euromillions.backtest import run_walk_forward
from euromillions.combinations import build_main_combinations, build_star_combinations
from euromillions.config import load_config
from euromillions.db import begin, create_all_tables, create_db_engine
from euromillions.ingest_excel import ingest_draw_rows, read_excel_draws
from euromillions.ingest_web import reconcile_and_insert
from euromillions.io import load_draw_records
from euromillions.optimise import OptimisationObjective, optimise_weights, recommended_trials
from euromillions.predict import generate_predictions, save_predictions
from euromillions.rank_history import parse_thresholds, rank_historical_winners, save_rank_history
from euromillions.schema import draws
from euromillions.sources.beatlottery import BeatLotterySource
from euromillions.sources.euro_millions_com import EuroMillionsComSource
from euromillions.sources.national_lottery import NationalLotterySource
from euromillions.sources.pedro_api import PedroApiSource

app = typer.Typer()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _config_hash(path: str = "config/default.yaml") -> str:
    config_path = Path(path)
    if not config_path.exists():
        return "missing"
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def _log_path(prefix: str = "optimise") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("logs") / f"{prefix}_{stamp}.log"


def _load_model_params(path: str) -> dict[str, float] | None:
    params_path = Path(path)
    if not params_path.exists():
        return None
    raw = json.loads(params_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return {str(k): float(v) for k, v in raw.items() if isinstance(v, int | float)}


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
    mode: Literal["fast", "full"] = typer.Option("fast", "--mode", "--evaluation-mode"),
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
        evaluation_mode=mode,
    )
    typer.echo(json.dumps(result.__dict__, indent=2))


@app.command("optimise")
def optimise(
    trials: int | None = typer.Option(None, "--trials"),
    top: int = typer.Option(3, "--top"),
    objective: OptimisationObjective = typer.Option("exact-rank", "--objective"),
    study_name: str = typer.Option("eml_optimisation", "--study-name"),
    storage: str = typer.Option("sqlite:///outputs/optuna_study.sqlite", "--storage"),
    n_jobs: int = typer.Option(1, "--n-jobs"),
    timeout_seconds: int | None = typer.Option(None, "--timeout-seconds"),
    mode: Literal["fast", "full"] = typer.Option("fast", "--mode", "--evaluation-mode"),
    early_stop_patience: int | None = typer.Option(None, "--early-stop-patience"),
    early_stop_min_delta: float = typer.Option(0.0, "--early-stop-min-delta"),
    early_stop_validation_rounds: int | None = typer.Option(10, "--early-stop-validation-rounds"),
) -> None:
    started_at = _utc_now()
    engine = _engine()
    with begin(engine) as conn:
        records = load_draw_records(conn)
        latest = conn.execute(select(func.max(draws.c.draw_date))).scalar_one()
    picked_trials = recommended_trials(len(records)) if trials is None else trials
    log_path = _log_path()
    metadata = {
        "commit_hash": _commit_hash(),
        "draw_count": len(records),
        "latest_draw_date": latest.isoformat() if latest is not None else None,
        "config_hash": _config_hash(),
        "started_at": started_at,
        "finished_at": None,
    }
    report = optimise_weights(
        records,
        trials=picked_trials,
        top=top,
        objective_name=objective,
        study_name=study_name,
        storage=storage,
        n_jobs=n_jobs,
        timeout_seconds=timeout_seconds,
        evaluation_mode=mode,
        holdout_fraction=load_config().optimisation.holdout_fraction,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        early_stop_validation_rounds=early_stop_validation_rounds,
        log_path=log_path,
        metadata=metadata,
        progress_callback=typer.echo,
    )
    report["metadata"]["finished_at"] = _utc_now()
    Path("outputs/optimisation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path("outputs").mkdir(parents=True, exist_ok=True)
    Path("outputs/best_params.json").write_text(json.dumps(report["best_params"], indent=2), encoding="utf-8")
    typer.echo(f"log_path={log_path}")
    typer.echo(f"recommended_trials={recommended_trials(len(records))}")
    typer.echo(f"trials_used={picked_trials}")
    typer.echo(f"existing_trials={report['existing_trials']}")
    typer.echo(f"new_trials={report['new_trials']}")
    typer.echo(f"completed_trials={report['completed_trials']}")
    typer.echo(f"mode={report['mode']}")
    typer.echo(f"objective={report['objective']}")
    typer.echo("early_stop=" + json.dumps(report["early_stop"]))
    typer.echo("metadata=" + json.dumps(report["metadata"]))
    if objective == "exact-rank":
        typer.echo(
            "holdout_metrics="
            + json.dumps(
                {
                    "average_rank": report["holdout"].get("average_rank"),
                    "median_rank": report["holdout"].get("median_rank"),
                    "evaluated_draws": report["holdout"]["evaluated_draws"],
                    "sampled": report["holdout"]["sampled"],
                    "evaluation_stride": report["holdout"]["evaluation_stride"],
                }
            )
        )
    else:
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


@app.command("smoke-test")
def smoke_test() -> None:
    cfg = load_config()
    typer.echo("smoke-test: starting")
    typer.echo(f"commit_hash={_commit_hash()}")
    typer.echo(f"config_hash={_config_hash()}")
    engine = _engine()
    create_all_tables(engine)
    with begin(engine) as conn:
        records = load_draw_records(conn)
        latest = conn.execute(select(func.max(draws.c.draw_date))).scalar_one()
    typer.echo(f"database_path={cfg.database_path}")
    typer.echo(f"draw_count={len(records)}")
    typer.echo(f"latest_draw_date={latest.isoformat() if latest is not None else None}")
    if len(records) < cfg.min_training_draws + 1:
        raise typer.BadParameter(
            f"need at least {cfg.min_training_draws + 1} draws for smoke test, found {len(records)}"
        )
    backtest_result = run_walk_forward(
        records,
        top=1,
        min_training_draws=cfg.min_training_draws,
        seed=cfg.random_seed,
        evaluation_mode="fast",
        max_rounds=1,
    )
    preds = generate_predictions(records, top=1)
    typer.echo(
        "smoke_backtest="
        + json.dumps(
            {
                "rounds": backtest_result.rounds,
                "model_points": backtest_result.model_points,
                "baseline_points": backtest_result.baseline_points,
                "uplift_points": backtest_result.uplift_points,
            }
        )
    )
    typer.echo("smoke_prediction=" + json.dumps(preds[0]))
    typer.echo("smoke-test: ok")


@app.command("predict")
def predict(
    top: int = typer.Option(3, "--top"),
    max_main_overlap: int | None = typer.Option(None, "--max-main-overlap"),
    require_distinct_star_pairs: bool | None = typer.Option(None, "--require-distinct-star-pairs"),
    params_path: str = typer.Option("outputs/best_params.json", "--params-path"),
) -> None:
    update_results()
    engine = _engine()
    with begin(engine) as conn:
        records = load_draw_records(conn)
    model_params = _load_model_params(params_path)
    if model_params is None:
        typer.echo(f"model_params=defaults; params file not found: {params_path}")
    else:
        typer.echo(f"model_params={params_path}")
    ranked = generate_predictions(
        records,
        top=top,
        max_main_overlap=max_main_overlap,
        require_distinct_star_pairs=require_distinct_star_pairs,
        model_params=model_params,
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


@app.command("rank-history")
def rank_history(
    mode: Literal["fast", "full"] = typer.Option("fast", "--mode"),
    thresholds: str = typer.Option("1,3,10,100,500,1000,3000", "--thresholds"),
    params_path: str = typer.Option("outputs/best_params.json", "--params-path"),
    max_rounds: int | None = typer.Option(None, "--max-rounds"),
) -> None:
    engine = _engine()
    with begin(engine) as conn:
        records = load_draw_records(conn)
    cfg = load_config()
    model_params = _load_model_params(params_path)
    parsed_thresholds = parse_thresholds(thresholds)
    rows, summary = rank_historical_winners(
        records,
        min_training_draws=cfg.min_training_draws,
        mode=mode,
        thresholds=parsed_thresholds,
        model_params=model_params,
        max_rounds=max_rounds,
    )
    save_rank_history(rows, summary)
    typer.echo(json.dumps(summary, indent=2))
    typer.echo("wrote outputs/historical_winner_ranks.json")
    typer.echo("wrote outputs/historical_winner_ranks.csv")


@app.command("run")
def run_pipeline() -> None:
    update_results()
    predict(top=load_config().top_predictions)


if __name__ == "__main__":
    app()
