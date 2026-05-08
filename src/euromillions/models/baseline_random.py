from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class BaselineRandomModel:
    seed: int = 42

    def predict(self, top: int) -> list[tuple[tuple[int, int, int, int, int], tuple[int, int], float]]:
        rng = random.Random(self.seed)
        out = []
        for _ in range(top):
            mains_raw = sorted(rng.sample(range(1, 51), 5))
            stars_raw = sorted(rng.sample(range(1, 13), 2))
            mains = (mains_raw[0], mains_raw[1], mains_raw[2], mains_raw[3], mains_raw[4])
            stars = (stars_raw[0], stars_raw[1])
            out.append((mains, stars, float(rng.random())))
        return out
