from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from euromillions.features import DrawRecord


@dataclass
class BayesianFrequencyModel:
    alpha: float = 1.0

    def _posterior_main_prob(self, history: list[DrawRecord]) -> dict[int, float]:
        counts = {n: 0 for n in range(1, 51)}
        for draw in history:
            for n in draw.mains:
                counts[n] += 1
        denom = len(history) * 5 + 50 * self.alpha
        return {n: (counts[n] + self.alpha) / denom for n in counts}

    def _posterior_star_prob(self, history: list[DrawRecord]) -> dict[int, float]:
        counts = {s: 0 for s in range(1, 13)}
        for draw in history:
            counts[draw.stars[0]] += 1
            counts[draw.stars[1]] += 1
        denom = len(history) * 2 + 12 * self.alpha
        return {s: (counts[s] + self.alpha) / denom for s in counts}

    def predict(
        self, history: list[DrawRecord], top: int = 3
    ) -> list[tuple[tuple[int, int, int, int, int], tuple[int, int], float]]:
        mp = self._posterior_main_prob(history)
        sp = self._posterior_star_prob(history)
        top_numbers = sorted(mp, key=lambda n: mp[n], reverse=True)[:15]
        mains = sorted(
            ((c, sum(mp[n] for n in c) / 5.0) for c in combinations(sorted(top_numbers), 5)),
            key=lambda x: x[1],
            reverse=True,
        )[:500]
        stars = sorted(
            ((c, (sp[c[0]] + sp[c[1]]) / 2.0) for c in combinations(range(1, 13), 2)),
            key=lambda x: x[1],
            reverse=True,
        )
        out = []
        for m, ms in mains[: max(top, 25)]:
            for s, ss in stars[: min(66, max(top, 10))]:
                out.append((m, s, ms * 0.75 + ss * 0.25))
        out.sort(key=lambda x: x[2], reverse=True)
        return out[:top]
