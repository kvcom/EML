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
