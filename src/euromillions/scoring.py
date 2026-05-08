from __future__ import annotations

from dataclasses import dataclass

from euromillions.features import DrawRecord, compute_delay_features, compute_frequency_features


@dataclass(frozen=True)
class ScoredMain:
    mains: tuple[int, int, int, int, int]
    score: float


@dataclass(frozen=True)
class ScoredStars:
    stars: tuple[int, int]
    score: float


def score_main_combination(
    comb: tuple[int, int, int, int, int],
    history: list[DrawRecord],
    w_freq: float = 0.6,
    w_delay: float = 0.4,
) -> float:
    freq = compute_frequency_features(history)
    delay = compute_delay_features(history)
    return score_main_combination_from_features(
        comb,
        freq,
        delay,
        len(history),
        w_freq=w_freq,
        w_delay=w_delay,
    )


def score_main_combination_from_features(
    comb: tuple[int, int, int, int, int],
    freq: dict[int, float],
    delay: dict[int, int],
    history_len: int,
    w_freq: float = 0.6,
    w_delay: float = 0.4,
) -> float:
    freq_component = sum(freq[n] for n in comb) / 5.0
    delay_component = sum(delay[n] for n in comb) / max(1, history_len * 5)
    return w_freq * freq_component + w_delay * delay_component


def score_star_combination(stars: tuple[int, int], history: list[DrawRecord]) -> float:
    counts = {s: 0 for s in range(1, 13)}
    for draw in history:
        counts[draw.stars[0]] += 1
        counts[draw.stars[1]] += 1
    total = max(1, len(history) * 2)
    return score_star_combination_from_counts(stars, counts, total)


def score_star_combination_from_counts(
    stars: tuple[int, int],
    counts: dict[int, int],
    total: int,
) -> float:
    return (counts[stars[0]] + counts[stars[1]]) / total
