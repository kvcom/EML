from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from euromillions.ingest_excel import read_excel_draws


def test_excel_ingestion_parses_and_sorts(tmp_path: Path) -> None:
    f = tmp_path / "seed.xlsx"
    pd.DataFrame(
        [
            {"draw_date": date(2024, 1, 1), "m1": 5, "m2": 2, "m3": 1, "m4": 3, "m5": 4, "s1": 12, "s2": 1},
            {"draw_date": date(2024, 1, 5), "m1": 10, "m2": 11, "m3": 12, "m4": 13, "m5": 14, "s1": 2, "s2": 1},
        ]
    ).to_excel(f, index=False)
    rows = read_excel_draws(str(f))
    assert len(rows) == 2
    assert rows[0].mains == (1, 2, 3, 4, 5)
    assert rows[0].stars == (1, 12)
