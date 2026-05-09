from __future__ import annotations

import bisect
import csv
import json
import os
import site
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Any, Literal

import numpy as np

from euromillions.features import DrawRecord, compute_delay_features, compute_frequency_features
from euromillions.model_params import merge_model_params
from euromillions.scoring import score_main_combination_from_features, score_star_combination_from_counts

TOTAL_TICKETS = 139_838_160
DEFAULT_THRESHOLDS = (1, 3, 10, 100, 500, 1000, 3000)
_ORIGINAL_COMBINATIONS = combinations
RankBackend = Literal["auto", "cpu", "gpu"]


@dataclass(frozen=True)
class HistoricalRankRow:
    draw_id: int
    draw_date: str
    mains: tuple[int, int, int, int, int]
    stars: tuple[int, int]
    model_score: float
    exact_rank: int
    percentile: float
    bucket: str


def bucket_for_rank(rank: int, thresholds: tuple[int, ...]) -> str:
    for threshold in thresholds:
        if rank <= threshold:
            return f"top_{threshold}"
    return "outside"


def parse_thresholds(raw: str | None) -> tuple[int, ...]:
    if not raw:
        return DEFAULT_THRESHOLDS
    thresholds = tuple(sorted({int(part.strip()) for part in raw.split(",") if part.strip()}))
    if not thresholds or any(value <= 0 for value in thresholds):
        raise ValueError("thresholds must be positive comma-separated integers")
    return thresholds


@lru_cache(maxsize=1)
def _main_combinations_array() -> np.ndarray:
    return np.array(list(combinations(range(1, 51), 5)), dtype=np.uint8)


@lru_cache(maxsize=1)
def _star_combinations_array() -> np.ndarray:
    return np.array(list(combinations(range(1, 13), 2)), dtype=np.uint8)


def _maybe_set_cuda_path() -> None:
    if os.environ.get("CUDA_PATH"):
        return
    candidates = [Path(path) for path in site.getsitepackages()]
    candidates.extend(Path(path) for path in site.getusersitepackages().split(os.pathsep))
    for parent in candidates:
        candidate = parent / "nvidia" / "cuda_nvrtc"
        if (candidate / "bin").exists():
            os.environ["CUDA_PATH"] = str(candidate)
            return


@lru_cache(maxsize=1)
def _cupy() -> Any | None:
    try:
        _maybe_set_cuda_path()
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() <= 0:
            return None
        probe = cp.arange(1, dtype=cp.float32)
        cp.asnumpy(probe + 1)
        return cp
    except Exception:
        return None


def gpu_available() -> bool:
    return _cupy() is not None


def resolve_rank_backend(rank_backend: RankBackend) -> Literal["cpu", "gpu"]:
    if rank_backend == "cpu":
        return "cpu"
    if rank_backend == "gpu":
        if not gpu_available():
            raise RuntimeError("rank_backend='gpu' requested, but no usable CuPy/CUDA runtime is available")
        return "gpu"
    return "gpu" if gpu_available() else "cpu"


@lru_cache(maxsize=1)
def _main_combinations_gpu() -> Any:
    cp = _cupy()
    if cp is None:
        raise RuntimeError("no usable CuPy/CUDA runtime is available")
    return cp.asarray(_main_combinations_array())


@lru_cache(maxsize=1)
def _star_combinations_gpu() -> Any:
    cp = _cupy()
    if cp is None:
        raise RuntimeError("no usable CuPy/CUDA runtime is available")
    return cp.asarray(_star_combinations_array())


def _score_vectors_cpu(
    main_components: np.ndarray,
    star_components: np.ndarray,
    actual_mains: tuple[int, int, int, int, int],
    actual_stars: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, float]:
    main_combos = _main_combinations_array()
    star_combos = _star_combinations_array()
    main_scores = (
        main_components[main_combos[:, 0]]
        + main_components[main_combos[:, 1]]
        + main_components[main_combos[:, 2]]
        + main_components[main_combos[:, 3]]
        + main_components[main_combos[:, 4]]
    )
    star_scores = star_components[star_combos[:, 0]] + star_components[star_combos[:, 1]]
    actual_score = float(
        main_components[np.fromiter(actual_mains, dtype=np.uint8)].sum()
        + star_components[np.fromiter(actual_stars, dtype=np.uint8)].sum()
    )
    return main_scores, star_scores, actual_score


def _posterior_main_prob(history: list[DrawRecord], alpha: float) -> dict[int, float]:
    counts = {n: 0 for n in range(1, 51)}
    for draw in history:
        for n in draw.mains:
            counts[n] += 1
    denom = len(history) * 5 + 50 * alpha
    return {n: (counts[n] + alpha) / denom for n in counts}


def _posterior_star_prob(history: list[DrawRecord], alpha: float) -> dict[int, float]:
    counts = {s: 0 for s in range(1, 13)}
    for draw in history:
        counts[draw.stars[0]] += 1
        counts[draw.stars[1]] += 1
    denom = len(history) * 2 + 12 * alpha
    return {s: (counts[s] + alpha) / denom for s in counts}


def _star_counts(history: list[DrawRecord]) -> dict[int, int]:
    counts = {s: 0 for s in range(1, 13)}
    for draw in history:
        counts[draw.stars[0]] += 1
        counts[draw.stars[1]] += 1
    return counts


def _main_component(
    mains: tuple[int, int, int, int, int],
    freq: dict[int, float],
    delay: dict[int, int],
    posterior: dict[int, float],
    history_len: int,
    params: dict[str, float],
) -> float:
    weighted = score_main_combination_from_features(
        mains,
        freq,
        delay,
        history_len,
        w_freq=params["weighted_freq_weight"],
        w_delay=params["weighted_delay_weight"],
    )
    bayesian = sum(posterior[n] for n in mains) / 5.0
    return (
        params["ensemble_weighted_weight"] * params["weighted_main_weight"] * weighted
        + params["ensemble_bayesian_weight"] * params["bayesian_main_weight"] * bayesian
    )


def _star_component(
    stars: tuple[int, int],
    counts: dict[int, int],
    posterior: dict[int, float],
    history_len: int,
    params: dict[str, float],
) -> float:
    weighted = score_star_combination_from_counts(stars, counts, max(1, history_len * 2))
    bayesian = (posterior[stars[0]] + posterior[stars[1]]) / 2.0
    return (
        params["ensemble_weighted_weight"] * params["weighted_star_weight"] * weighted
        + params["ensemble_bayesian_weight"] * params["bayesian_star_weight"] * bayesian
    )


def _history_arrays(
    history: list[DrawRecord],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    main_counts = np.zeros(51, dtype=np.float64)
    star_counts = np.zeros(13, dtype=np.float64)
    last_seen = np.full(51, -1, dtype=np.int64)
    for idx, draw in enumerate(history):
        for number in draw.mains:
            main_counts[number] += 1.0
            last_seen[number] = idx
        star_counts[draw.stars[0]] += 1.0
        star_counts[draw.stars[1]] += 1.0
    history_len = len(history)
    latest_idx = history_len - 1
    delays = np.zeros(51, dtype=np.float64)
    for number in range(1, 51):
        delays[number] = latest_idx - last_seen[number] if last_seen[number] >= 0 else history_len
    main_posterior = (main_counts + alpha) / (history_len * 5.0 + 50.0 * alpha)
    star_posterior = (star_counts + alpha) / (history_len * 2.0 + 12.0 * alpha)
    return main_counts, star_counts, delays, main_posterior, star_posterior


def _number_score_components(
    history_len: int,
    main_counts: np.ndarray,
    star_counts: np.ndarray,
    delays: np.ndarray,
    main_posterior: np.ndarray,
    star_posterior: np.ndarray,
    params: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    main_freq = main_counts / max(1.0, history_len * 5.0)
    main_weighted = (
        params["weighted_freq_weight"] * main_freq / 5.0
        + params["weighted_delay_weight"] * delays / max(1.0, history_len * 5.0)
    )
    main_components = (
        params["ensemble_weighted_weight"] * params["weighted_main_weight"] * main_weighted
        + params["ensemble_bayesian_weight"] * params["bayesian_main_weight"] * main_posterior / 5.0
    )
    star_weighted = star_counts / max(1.0, history_len * 2.0)
    star_components = (
        params["ensemble_weighted_weight"] * params["weighted_star_weight"] * star_weighted
        + params["ensemble_bayesian_weight"] * params["bayesian_star_weight"] * star_posterior / 2.0
    )
    return main_components, star_components


def exact_ticket_rank_vectorized(
    history: list[DrawRecord],
    actual_mains: tuple[int, int, int, int, int],
    actual_stars: tuple[int, int],
    model_params: dict[str, float] | None = None,
    rank_backend: RankBackend = "auto",
) -> tuple[float, int]:
    resolved_backend = resolve_rank_backend(rank_backend)
    params = merge_model_params(model_params)
    history_len = len(history)
    main_counts, star_counts, delays, main_posterior, star_posterior = _history_arrays(
        history,
        params["bayesian_alpha"],
    )
    main_components, star_components = _number_score_components(
        history_len,
        main_counts,
        star_counts,
        delays,
        main_posterior,
        star_posterior,
        params,
    )
    if resolved_backend == "gpu":
        return _exact_ticket_rank_gpu(
            main_components,
            star_components,
            actual_mains,
            actual_stars,
        )
    return _exact_ticket_rank_cpu(
        main_components,
        star_components,
        actual_mains,
        actual_stars,
    )


def _exact_ticket_rank_cpu(
    main_components: np.ndarray,
    star_components: np.ndarray,
    actual_mains: tuple[int, int, int, int, int],
    actual_stars: tuple[int, int],
) -> tuple[float, int]:
    main_scores, star_scores, actual_score = _score_vectors_cpu(
        main_components,
        star_components,
        actual_mains,
        actual_stars,
    )
    sorted_main_scores = np.sort(main_scores)
    thresholds = actual_score - star_scores
    insertion_points = np.searchsorted(sorted_main_scores, thresholds, side="right")
    better = int(np.sum(len(sorted_main_scores) - insertion_points, dtype=np.int64))
    return actual_score, better + 1


def _exact_ticket_rank_gpu(
    main_components: np.ndarray,
    star_components: np.ndarray,
    actual_mains: tuple[int, int, int, int, int],
    actual_stars: tuple[int, int],
) -> tuple[float, int]:
    cp = _cupy()
    if cp is None:
        raise RuntimeError("no usable CuPy/CUDA runtime is available")
    main_scores, star_scores, actual_score = _score_vectors_cpu(
        main_components,
        star_components,
        actual_mains,
        actual_stars,
    )
    sorted_main_scores = cp.sort(cp.asarray(main_scores))
    thresholds = cp.asarray(actual_score - star_scores)
    insertion_points = cp.searchsorted(sorted_main_scores, thresholds, side="right")
    better = cp.sum(len(sorted_main_scores) - insertion_points, dtype=cp.int64)
    exact_rank = int(cp.asnumpy(better)) + 1
    return actual_score, exact_rank


def exact_ticket_rank(
    history: list[DrawRecord],
    actual_mains: tuple[int, int, int, int, int],
    actual_stars: tuple[int, int],
    model_params: dict[str, float] | None = None,
    rank_backend: RankBackend = "auto",
) -> tuple[float, int]:
    if combinations is _ORIGINAL_COMBINATIONS:
        return exact_ticket_rank_vectorized(
            history,
            actual_mains,
            actual_stars,
            model_params,
            rank_backend=rank_backend,
        )
    params = merge_model_params(model_params)
    freq = compute_frequency_features(history)
    delay = compute_delay_features(history)
    main_posterior = _posterior_main_prob(history, params["bayesian_alpha"])
    star_posterior = _posterior_star_prob(history, params["bayesian_alpha"])
    star_counts = _star_counts(history)
    history_len = len(history)
    main_scores = [
        (
            _main_component(mains, freq, delay, main_posterior, history_len, params),
            mains,
        )
        for mains in combinations(range(1, 51), 5)
    ]
    main_scores.sort(key=lambda item: item[0])
    sorted_scores = [item[0] for item in main_scores]
    star_scores = [
        (
            _star_component(stars, star_counts, star_posterior, history_len, params),
            stars,
        )
        for stars in combinations(range(1, 13), 2)
    ]
    actual_main_score = _main_component(actual_mains, freq, delay, main_posterior, history_len, params)
    actual_star_score = _star_component(actual_stars, star_counts, star_posterior, history_len, params)
    actual_score = actual_main_score + actual_star_score
    better = 0
    for star_score, _stars in star_scores:
        threshold = actual_score - star_score
        better += len(sorted_scores) - bisect.bisect_right(sorted_scores, threshold)
    return actual_score, better + 1


def rank_historical_winners(
    draws: list[DrawRecord],
    min_training_draws: int,
    mode: Literal["fast", "full"] = "fast",
    thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS,
    model_params: dict[str, float] | None = None,
    max_rounds: int | None = None,
    start_index: int | None = None,
    end_index: int | None = None,
    rank_backend: RankBackend = "auto",
) -> tuple[list[HistoricalRankRow], dict[str, float | int | str]]:
    stride = 10 if mode == "fast" else 1
    resolved_backend = resolve_rank_backend(rank_backend)
    limit = 250 if max_rounds is None and mode == "fast" else max_rounds
    rows: list[HistoricalRankRow] = []
    start = max(min_training_draws, start_index or min_training_draws)
    stop = end_index if end_index is not None else len(draws)
    for idx in range(start, stop, stride):
        if limit is not None and len(rows) >= limit:
            break
        actual = draws[idx]
        history = draws[:idx]
        score, rank = exact_ticket_rank(
            history,
            actual.mains,
            actual.stars,
            model_params,
            rank_backend=resolved_backend,
        )
        rows.append(
            HistoricalRankRow(
                draw_id=actual.draw_id,
                draw_date=actual.draw_date.isoformat() if actual.draw_date is not None else "",
                mains=actual.mains,
                stars=actual.stars,
                model_score=score,
                exact_rank=rank,
                percentile=rank / TOTAL_TICKETS,
                bucket=bucket_for_rank(rank, thresholds),
            )
        )
    ranks = [row.exact_rank for row in rows]
    summary: dict[str, float | int | str] = {
        "mode": mode,
        "evaluated_draws": len(rows),
        "evaluation_stride": stride,
        "rank_backend": resolved_backend,
        "total_ticket_count": TOTAL_TICKETS,
        "random_expected_top_1000_rate": 1000 / TOTAL_TICKETS,
    }
    if ranks:
        summary["median_rank"] = float(median(ranks))
        summary["average_rank"] = float(mean(ranks))
        for threshold in thresholds:
            summary[f"pct_top_{threshold}"] = sum(rank <= threshold for rank in ranks) / len(ranks)
    return rows, summary


def save_rank_history(
    rows: list[HistoricalRankRow],
    summary: dict[str, float | int | str],
    out_dir: str = "outputs",
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_payload = {"summary": summary, "rows": [asdict(row) for row in rows]}
    (out / "historical_winner_ranks.json").write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    with (out / "historical_winner_ranks.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "draw_id",
                "draw_date",
                "mains",
                "stars",
                "model_score",
                "exact_rank",
                "percentile",
                "bucket",
            ],
        )
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            data["mains"] = " ".join(f"{n:02d}" for n in row.mains)
            data["stars"] = " ".join(f"{n:02d}" for n in row.stars)
            writer.writerow(data)
