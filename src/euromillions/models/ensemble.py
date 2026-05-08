from __future__ import annotations

from dataclasses import dataclass

from euromillions.features import DrawRecord
from euromillions.models.bayesian_frequency import BayesianFrequencyModel
from euromillions.models.weighted_statistical import WeightedStatisticalModel


@dataclass
class EnsembleModel:
    weighted: WeightedStatisticalModel
    bayesian: BayesianFrequencyModel
    w_weighted: float = 0.6
    w_bayesian: float = 0.4

    def predict(
        self, history: list[DrawRecord], top: int = 3
    ) -> list[tuple[tuple[int, int, int, int, int], tuple[int, int], float]]:
        w_preds = self.weighted.predict(history, top=max(top, 20))
        b_preds = self.bayesian.predict(history, top=max(top, 20))
        scores: dict[tuple[tuple[int, int, int, int, int], tuple[int, int]], float] = {}
        for m, s, score in w_preds:
            scores[(m, s)] = scores.get((m, s), 0.0) + score * self.w_weighted
        for m, s, score in b_preds:
            scores[(m, s)] = scores.get((m, s), 0.0) + score * self.w_bayesian
        ranked = sorted(((m, s, sc) for (m, s), sc in scores.items()), key=lambda x: x[2], reverse=True)
        return ranked[:top]
