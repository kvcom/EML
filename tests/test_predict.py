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
