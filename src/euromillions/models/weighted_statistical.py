from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from euromillions.features import DrawRecord
from euromillions.scoring import score_main_combination, score_star_combination


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
        number_scores = {
            n: score_main_combination(
                (n, n + 1, n + 2, n + 3, n + 4),
                history,
                w_freq=self.frequency_weight,
                w_delay=self.delay_weight,
            )
            for n in range(1, 47)
        }
        top_numbers = sorted(number_scores, key=lambda n: number_scores[n], reverse=True)[
            : max(5, min(50, self.top_number_count))
        ]
        ordered_numbers = sorted(top_numbers)
        main_scores = [
            (
                comb,
                score_main_combination(
                    comb,
                    history,
                    w_freq=self.frequency_weight,
                    w_delay=self.delay_weight,
                ),
            )
            for comb in combinations(ordered_numbers, 5)
        ]
        main_scores.sort(key=lambda x: x[1], reverse=True)
        stars_scores = [
            (stars, score_star_combination(stars, history))
            for stars in combinations(range(1, 13), 2)
        ]
        stars_scores.sort(key=lambda x: x[1], reverse=True)
        mains_top = main_scores[: self.main_pool_size]
        stars_top = stars_scores[: self.star_pool_size]
        merged: list[tuple[tuple[int, int, int, int, int], tuple[int, int], float]] = []
        for mains, ms in mains_top[: max(top, 100)]:
            for stars, ss in stars_top:
                merged.append((mains, stars, ms * self.main_weight + ss * self.star_weight))
        merged.sort(key=lambda x: x[2], reverse=True)
        return merged[:top]
