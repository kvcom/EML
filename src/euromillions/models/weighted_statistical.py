from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from euromillions.features import DrawRecord, compute_delay_features, compute_frequency_features
from euromillions.scoring import score_main_combination_from_features, score_star_combination_from_counts


@dataclass
class WeightedStatisticalModel:
    main_pool_size: int = 2000
    star_pool_size: int = 66
    top_number_count: int = 15
    frequency_weight: float = 0.6
    delay_weight: float = 0.4
    main_weight: float = 0.8
    star_weight: float = 0.2

    def predict(
        self, history: list[DrawRecord], top: int = 3
    ) -> list[tuple[tuple[int, int, int, int, int], tuple[int, int], float]]:
        freq = compute_frequency_features(history)
        delay = compute_delay_features(history)
        history_len = len(history)
        number_scores = {
            n: score_main_combination_from_features(
                (n, n + 1, n + 2, n + 3, n + 4),
                freq,
                delay,
                history_len,
                w_freq=self.frequency_weight,
                w_delay=self.delay_weight,
            )
            for n in range(1, 47)
        }
        top_number_count = max(5, min(50, self.top_number_count))
        top_numbers = sorted(number_scores, key=lambda n: number_scores[n], reverse=True)[:top_number_count]
        ordered_numbers = sorted(top_numbers)
        main_scores = [
            (
                comb,
                score_main_combination_from_features(
                    comb,
                    freq,
                    delay,
                    history_len,
                    w_freq=self.frequency_weight,
                    w_delay=self.delay_weight,
                ),
            )
            for comb in combinations(ordered_numbers, 5)
        ]
        main_scores.sort(key=lambda x: x[1], reverse=True)
        mains_top_count = min(self.main_pool_size, max(top, 100))
        mains_top = main_scores[:mains_top_count]
        star_counts = {s: 0 for s in range(1, 13)}
        for draw in history:
            star_counts[draw.stars[0]] += 1
            star_counts[draw.stars[1]] += 1
        star_total = max(1, history_len * 2)
        star_candidates = [
            (stars, score_star_combination_from_counts(stars, star_counts, star_total))
            for stars in combinations(range(1, 13), 2)
        ]
        star_candidates.sort(key=lambda x: x[1], reverse=True)
        stars_top = star_candidates[: self.star_pool_size]
        merged: list[tuple[tuple[int, int, int, int, int], tuple[int, int], float]] = []
        for mains, ms in mains_top:
            for stars, ss in stars_top:
                merged.append((mains, stars, ms * self.main_weight + ss * self.star_weight))
        merged.sort(key=lambda x: x[2], reverse=True)
        return merged[:top]
