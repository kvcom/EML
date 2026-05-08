from __future__ import annotations

from euromillions.features import DrawRecord, compute_delay_features, compute_frequency_features, compute_pair_counts


def _sample() -> list[DrawRecord]:
    return [
        DrawRecord(1, (1, 2, 3, 4, 5), (1, 2)),
        DrawRecord(2, (1, 6, 7, 8, 9), (1, 3)),
        DrawRecord(3, (10, 11, 12, 13, 14), (4, 5)),
    ]


def test_frequency_and_delays() -> None:
    freq = compute_frequency_features(_sample())
    delay = compute_delay_features(_sample())
    assert freq[1] > 0.0
    assert delay[1] == 1
    assert delay[50] == 3


def test_pair_counts() -> None:
    pairs = compute_pair_counts(_sample())
    assert pairs[(1, 2)] == 1
    assert pairs[(1, 6)] == 1
