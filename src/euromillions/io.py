from __future__ import annotations

from euromillions.features import DrawRecord
from euromillions.schema import draws
from sqlalchemy import Connection, select


def load_draw_records(conn: Connection) -> list[DrawRecord]:
    rows = conn.execute(select(draws).order_by(draws.c.draw_date.asc())).mappings().all()
    return [
        DrawRecord(
            draw_id=int(r["id"]),
            mains=(int(r["m1"]), int(r["m2"]), int(r["m3"]), int(r["m4"]), int(r["m5"])),
            stars=(int(r["s1"]), int(r["s2"])),
            draw_date=r["draw_date"],
        )
        for r in rows
    ]
