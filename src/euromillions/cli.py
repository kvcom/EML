from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import typer
import optuna
from sqlalchemy import func, select

from euromillions.backtest import run_walk_forward
from euromillions.candidate_validation import (
    CandidateSpec,
    load_params_file,
    save_candidate_validation_report,
    validate_candidates,
)
from euromillions.combinations import build_main_combinations, build_star_combinations
from euromillions.config import load_config
from euromillions.db import begin, create_all_tables, create_db_engine
from euromillions.dynamic_params import run_dynamic_params_experiment, save_dynamic_params_report
from euromillions.ingest_excel import ingest_draw_rows, read_excel_draws
from euromillions.ingest_web import reconcile_and_insert
from euromillions.io import load_draw_records
from euromillions.optimise import (
    OptimisationObjective,
    _model_params_from_study_params,
    optimise_weights,
    recommended_trials,
)
from euromillions.portfolio_backtest import run_portfolio_backtest, save_portfolio_backtest_report
from euromillions.predict import generate_predictions, save_predictions
from euromillions.rank_history import RankBackend, parse_thresholds, rank_historical_winners, save_rank_history
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


def _load_candidate_specs(
    params_paths: list[str],
    trial_numbers: list[int],
    study_name: str,
    storage: str,
) -> list[CandidateSpec]:
    specs = [load_params_file(path) for path in params_paths]
    if trial_numbers:
        study = optuna.load_study(study_name=study_name, storage=storage)
        wanted = set(trial_numbers)
        trials = {trial.number: trial for trial in study.trials if trial.number in wanted}
        missing = sorted(wanted - set(trials))
        if missing:
            raise typer.BadParameter(f"trial(s) not found in study: {missing}")
        for number in trial_numbers:
            trial = trials[number]
            specs.append(
                CandidateSpec(
                    label=f"trial_{number}",
                    params={
                        "min_training_draws": float(trial.params.get("min_training_draws", 200)),
                        **_model_params_from_study_params(trial.params),
                    },
                    objective_value=float(trial.value) if trial.value is not None else None,
                    objective_average_rank=(
                        float(trial.user_attrs["objective_average_rank"])
                        if "objective_average_rank" in trial.user_attrs
                        else (-float(trial.value) if trial.value is not None else None)
                    ),
                )
            )
    if not specs:
        raise typer.BadParameter("provide at least one --params-path or --trial")
    return specs


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
    rolling_windows: int = typer.Option(1, "--rolling-windows"),
    rolling_window_rounds: int | None = typer.Option(10, "--rolling-window-rounds"),
    top_trial_holdout_count: int = typer.Option(10, "--top-trial-holdout-count"),
    top_trial_holdout_rounds: int | None = typer.Option(None, "--top-trial-holdout-rounds"),
    portfolio_objective_rounds: int | None = typer.Option(100, "--portfolio-objective-rounds"),
    portfolio_random_baseline_runs: int = typer.Option(10, "--portfolio-random-baseline-runs"),
    portfolio_holdout_random_baseline_runs: int = typer.Option(25, "--portfolio-holdout-random-baseline-runs"),
    rank_backend: RankBackend = typer.Option("auto", "--rank-backend"),
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
        rolling_windows=rolling_windows,
        rolling_window_rounds=rolling_window_rounds,
        top_trial_holdout_count=top_trial_holdout_count,
        top_trial_holdout_rounds=top_trial_holdout_rounds,
        portfolio_objective_rounds=portfolio_objective_rounds,
        portfolio_random_baseline_runs=portfolio_random_baseline_runs,
        portfolio_holdout_random_baseline_runs=portfolio_holdout_random_baseline_runs,
        rank_backend=rank_backend,
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
    typer.echo(f"progress_path={report['progress_path']}")
    typer.echo(f"trials_path={report['trials_path']}")
    typer.echo("early_stop=" + json.dumps(report["early_stop"]))
    typer.echo("rolling_objective=" + json.dumps(report["rolling_objective"]))
    typer.echo("top_trial_holdout=" + json.dumps(report["top_trial_holdout"]))
    typer.echo("portfolio_objective=" + json.dumps(report["portfolio_objective"]))
    typer.echo("metadata=" + json.dumps(report["metadata"]))
    if objective in {"exact-rank", "exact-rank-sum"}:
        typer.echo(
            "holdout_metrics="
            + json.dumps(
                {
                    "rank_sum": report["holdout"].get("rank_sum"),
                    "average_rank": report["holdout"].get("average_rank"),
                    "median_rank": report["holdout"].get("median_rank"),
                    "evaluated_draws": report["holdout"]["evaluated_draws"],
                    "sampled": report["holdout"]["sampled"],
                    "evaluation_stride": report["holdout"]["evaluation_stride"],
                }
            )
        )
    elif objective == "portfolio-uplift":
        typer.echo(
            "holdout_metrics="
            + json.dumps(
                {
                    "model_winning_round_rate": report["holdout"]["model"]["winning_round_rate"],
                    "random_winning_round_rate": report["holdout"]["random_baseline"]["winning_round_rate"],
                    "winning_round_uplift": report["holdout"]["winning_round_uplift"],
                    "rounds": report["holdout"]["rounds"],
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


@app.command("validate-candidates")
def validate_candidates_cmd(
    params_path: list[str] | None = typer.Option(None, "--params-path"),
    trial: list[int] | None = typer.Option(None, "--trial"),
    study_name: str = typer.Option("monitor_rolling_windows_20260510", "--study-name"),
    storage: str = typer.Option("sqlite:///outputs/monitor_rolling_windows_20260510.sqlite", "--storage"),
    mode: Literal["fast", "full"] = typer.Option("full", "--mode"),
    window_count: int = typer.Option(5, "--window-count"),
    window_size: int = typer.Option(20, "--window-size"),
    gap: int = typer.Option(20, "--gap"),
    rank_backend: RankBackend = typer.Option("auto", "--rank-backend"),
    out_path: str = typer.Option("outputs/candidate_validation_report.json", "--out-path"),
) -> None:
    engine = _engine()
    with begin(engine) as conn:
        records = load_draw_records(conn)
    specs = _load_candidate_specs(
        params_paths=params_path or [],
        trial_numbers=trial or [],
        study_name=study_name,
        storage=storage,
    )
    report = validate_candidates(
        records,
        specs,
        holdout_fraction=load_config().optimisation.holdout_fraction,
        mode=mode,
        rank_backend=rank_backend,
        window_count=window_count,
        window_size=window_size,
        gap=gap,
    )
    save_candidate_validation_report(report, out_path)
    typer.echo("wrote " + out_path)
    typer.echo("ranked_by_validation_mean=" + json.dumps(report["ranked_by_validation_mean"]))
    typer.echo("ranked_by_holdout=" + json.dumps(report["ranked_by_holdout"]))


@app.command("portfolio-backtest")
def portfolio_backtest(
    top: int = typer.Option(3, "--top"),
    params_path: str = typer.Option("outputs/best_params.json", "--params-path"),
    min_training_draws: int | None = typer.Option(None, "--min-training-draws"),
    mode: Literal["fast", "full"] = typer.Option("fast", "--mode"),
    max_rounds: int | None = typer.Option(None, "--max-rounds"),
    start_index: int | None = typer.Option(None, "--start-index"),
    end_index: int | None = typer.Option(None, "--end-index"),
    max_main_overlap: int | None = typer.Option(None, "--max-main-overlap"),
    require_distinct_star_pairs: bool | None = typer.Option(None, "--require-distinct-star-pairs"),
    random_baseline_runs: int = typer.Option(25, "--random-baseline-runs"),
    seed: int = typer.Option(42, "--seed"),
    out_path: str = typer.Option("outputs/portfolio_backtest_report.json", "--out-path"),
) -> None:
    engine = _engine()
    with begin(engine) as conn:
        records = load_draw_records(conn)
    model_params = _load_model_params(params_path)
    effective_min_training = (
        min_training_draws
        if min_training_draws is not None
        else int((model_params or {}).get("min_training_draws", load_config().min_training_draws))
    )
    report = run_portfolio_backtest(
        records,
        top=top,
        min_training_draws=effective_min_training,
        seed=seed,
        mode=mode,
        max_rounds=max_rounds,
        start_index=start_index,
        end_index=end_index,
        model_params=model_params,
        max_main_overlap=max_main_overlap,
        require_distinct_star_pairs=require_distinct_star_pairs,
        random_baseline_runs=random_baseline_runs,
    )
    report["params_path"] = params_path
    report["min_training_draws"] = effective_min_training
    save_portfolio_backtest_report(report, out_path)
    typer.echo("wrote " + out_path)
    typer.echo(
        "portfolio_summary="
        + json.dumps(
            {
                "rounds": report["rounds"],
                "model_winning_round_rate": report["model"]["winning_round_rate"],
                "random_winning_round_rate": report["random_baseline"]["winning_round_rate"],
                "model_tier_counts": report["model"]["tier_counts"],
                "random_tier_counts": report["random_baseline"]["tier_counts"],
            }
        )
    )


@app.command("dynamic-params")
def dynamic_params(
    params_path: str = typer.Option("outputs/best_params.json", "--params-path"),
    start_index: int | None = typer.Option(None, "--start-index"),
    end_index: int | None = typer.Option(None, "--end-index"),
    max_targets: int = typer.Option(20, "--max-targets"),
    stride: int = typer.Option(10, "--stride"),
    oracle_trials: int = typer.Option(20, "--oracle-trials"),
    forecast_lookback: int = typer.Option(5, "--forecast-lookback"),
    mode: Literal["fast", "full"] = typer.Option("fast", "--mode"),
    rank_backend: RankBackend = typer.Option("auto", "--rank-backend"),
    out_path: str = typer.Option("outputs/dynamic_params_report.json", "--out-path"),
) -> None:
    engine = _engine()
    with begin(engine) as conn:
        records = load_draw_records(conn)
    baseline_params = _load_model_params(params_path)
    if baseline_params is None:
        raise typer.BadParameter(f"params file not found: {params_path}")
    report = run_dynamic_params_experiment(
        records,
        baseline_params=baseline_params,
        start_index=start_index,
        end_index=end_index,
        max_targets=max_targets,
        stride=stride,
        oracle_trials=oracle_trials,
        forecast_lookback=forecast_lookback,
        mode=mode,
        rank_backend=rank_backend,
    )
    save_dynamic_params_report(report, out_path)
    typer.echo("wrote " + out_path)
    typer.echo("dynamic_summary=" + json.dumps(report["summary"]))


@app.command("rank-history")
def rank_history(
    mode: Literal["fast", "full"] = typer.Option("fast", "--mode"),
    thresholds: str = typer.Option("1,3,10,100,500,1000,3000", "--thresholds"),
    params_path: str = typer.Option("outputs/best_params.json", "--params-path"),
    max_rounds: int | None = typer.Option(None, "--max-rounds"),
    rank_backend: RankBackend = typer.Option("auto", "--rank-backend"),
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
        rank_backend=rank_backend,
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
