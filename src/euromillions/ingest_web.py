from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from hashlib import sha256

from sqlalchemy import Connection

from euromillions.ingest_excel import DrawRow, ingest_draw_rows
from euromillions.schema import source_observations
from euromillions.sources.base import DrawResult, ResultSource


def _checksum(draw: DrawResult) -> str:
    payload = f"{draw.draw_date.isoformat()}|{draw.mains}|{draw.stars}"
    return sha256(payload.encode("utf-8")).hexdigest()


def reconcile_and_insert(conn: Connection, sources: list[ResultSource]) -> tuple[int, list[str]]:
    now = datetime.utcnow().isoformat()
    grouped: dict[str, list[tuple[str, DrawResult]]] = defaultdict(list)
    warnings: list[str] = []
    for source in sources:
        draws = source.fetch_latest()
        for draw in draws:
            key = draw.draw_date.isoformat()
            grouped[key].append((source.name, draw))
            conn.execute(
                source_observations.insert(),
                {
                    "source": source.name,
                    "source_url": draw.source_url,
                    "observed_at": now,
                    "draw_date": draw.draw_date,
                    "m1": draw.mains[0],
                    "m2": draw.mains[1],
                    "m3": draw.mains[2],
                    "m4": draw.mains[3],
                    "m5": draw.mains[4],
                    "s1": draw.stars[0],
                    "s2": draw.stars[1],
                    "raw_payload": draw.raw_payload[:10000],
                    "status": draw.status,
                    "checksum": _checksum(draw),
                },
            )
    rows: list[DrawRow] = []
    for day, obs in grouped.items():
        unique = {(d.mains, d.stars) for _, d in obs if d.status == "ok"}
        if len(unique) > 1:
            warnings.append(f"source disagreement on {day}; skipped")
            continue
        if len(unique) == 0:
            continue
        mains, stars = unique.pop()
        rows.append(
            DrawRow(draw_date=obs[0][1].draw_date, mains=mains, stars=stars, source="|".join(s for s, _ in obs))
        )
    inserted = ingest_draw_rows(conn, rows)
    return inserted, warnings
