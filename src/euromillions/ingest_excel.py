from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from hashlib import sha256
from typing import Any

import pandas as pd
from sqlalchemy import Connection, select

from euromillions.schema import draws


@dataclass(frozen=True)
class DrawRow:
    draw_date: date
    mains: tuple[int, int, int, int, int]
    stars: tuple[int, int]
    source: str = "excel_seed"
    source_url: str | None = None


def _coerce_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()
    parsed = pd.to_datetime(value)
    if hasattr(parsed, "date"):
        return parsed.date()
    raise ValueError(f"cannot parse draw date: {value}")


def _norm_draw(draw_date: date, mains: list[int], stars: list[int]) -> DrawRow:
    smv = sorted(int(x) for x in mains)
    ssv = sorted(int(x) for x in stars)
    sm = (smv[0], smv[1], smv[2], smv[3], smv[4])
    ss = (ssv[0], ssv[1])
    if len(set(sm)) != 5 or any(n < 1 or n > 50 for n in sm):
        raise ValueError(f"invalid mains for draw {draw_date}: {sm}")
    if len(set(ss)) != 2 or any(n < 1 or n > 12 for n in ss):
        raise ValueError(f"invalid stars for draw {draw_date}: {ss}")
    return DrawRow(draw_date=draw_date, mains=sm, stars=ss)


def checksum_for_draw(row: DrawRow) -> str:
    payload = f"{row.draw_date.isoformat()}|{','.join(map(str, row.mains))}|{','.join(map(str, row.stars))}"
    return sha256(payload.encode("utf-8")).hexdigest()


def read_excel_draws(excel_path: str) -> list[DrawRow]:
    df = pd.read_excel(excel_path)
    cols = {c.lower().strip(): c for c in df.columns}
    expected = ["draw_date", "m1", "m2", "m3", "m4", "m5", "s1", "s2"]
    if all(col in cols for col in expected):
        use = [cols[x] for x in expected]
        subset = df[use].copy()
        subset.columns = expected
    else:
        subset = df.iloc[:, :8].copy()
        subset.columns = expected

    out: list[DrawRow] = []
    for _, row in subset.dropna(subset=["draw_date"]).iterrows():
        out.append(
            _norm_draw(
                _coerce_date(row["draw_date"]),
                [int(row[f"m{i}"]) for i in range(1, 6)],
                [int(row["s1"]), int(row["s2"])],
            )
        )
    return out


def ingest_draw_rows(conn: Connection, rows: list[DrawRow]) -> int:
    existing_checksums = {
        r[0] for r in conn.execute(select(draws.c.checksum)).fetchall()
    }
    now = datetime.utcnow().isoformat()
    payload: list[dict[str, Any]] = []
    for row in rows:
        checksum = checksum_for_draw(row)
        if checksum in existing_checksums:
            continue
        payload.append(
            {
                "draw_date": row.draw_date,
                "m1": row.mains[0],
                "m2": row.mains[1],
                "m3": row.mains[2],
                "m4": row.mains[3],
                "m5": row.mains[4],
                "s1": row.stars[0],
                "s2": row.stars[1],
                "source": row.source,
                "source_url": row.source_url,
                "created_at": now,
                "updated_at": now,
                "checksum": checksum,
            }
        )
    if not payload:
        return 0
    result = conn.execute(draws.insert().prefix_with("OR IGNORE"), payload)
    return int(result.rowcount or 0)
