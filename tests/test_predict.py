from __future__ import annotations

from euromillions.features import DrawRecord
from euromillions.predict import generate_predictions


def test_predict_returns_top_n_valid() -> None:
    history = [
        DrawRecord(i, (1, 2, 3, 4, 5), (1, 2)) for i in range(1, 260)
    ]
    preds = generate_predictions(history, top=3, seed=42)
    assert len(preds) == 3
    for p in preds:
        mains = p["mains"]
        stars = p["stars"]
        assert len(mains) == 5 and len(set(mains)) == 5 and tuple(sorted(mains)) == mains
        assert len(stars) == 2 and len(set(stars)) == 2 and tuple(sorted(stars)) == stars


def test_predict_diversity_constraints() -> None:
    history = [DrawRecord(i, (1, 2, 3, 4, 5), (1, 2)) for i in range(1, 320)]
    preds = generate_predictions(
        history,
        top=3,
        seed=42,
        max_main_overlap=3,
        require_distinct_star_pairs=True,
    )
    assert len(preds) == 3
    assert len({p["stars"] for p in preds}) == 3
    for idx, pred in enumerate(preds):
        for other in preds[idx + 1 :]:
            assert len(set(pred["mains"]) & set(other["mains"])) <= 3


def test_predict_accepts_tuned_model_params() -> None:
    history = [DrawRecord(i, tuple(range(1, 6)), (1, 2)) for i in range(1, 260)]
    preds = generate_predictions(
        history,
        top=3,
        model_params={
            "weighted_main_pool_size": 100,
            "weighted_star_pool_size": 12,
            "weighted_top_number_count": 10,
            "weighted_freq_weight": 0.9,
            "weighted_delay_weight": 0.1,
            "weighted_main_weight": 0.9,
            "weighted_star_weight": 0.1,
            "bayesian_alpha": 0.5,
            "bayesian_main_pool_size": 100,
            "bayesian_star_pair_count": 5,
            "bayesian_top_number_count": 10,
            "bayesian_main_weight": 0.9,
            "bayesian_star_weight": 0.1,
            "ensemble_weighted_weight": 0.8,
            "ensemble_bayesian_weight": 0.2,
            "candidate_pool_multiplier": 50,
            "candidate_pool_min": 250,
            "max_main_overlap": 3,
            "require_distinct_star_pairs": 1,
        },
    )
    assert len(preds) == 3
