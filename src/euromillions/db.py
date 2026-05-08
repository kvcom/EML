from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import Connection, Engine, create_engine, select

from euromillions.schema import draws, metadata


def create_db_engine(database_path: str) -> Engine:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{path.as_posix()}", future=True)


def create_all_tables(engine: Engine) -> None:
    metadata.create_all(engine)


@contextmanager
def begin(engine: Engine) -> Iterator[Connection]:
    with engine.begin() as conn:
        yield conn


def insert_ignore(conn: Connection, table: Any, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    stmt = table.insert().prefix_with("OR IGNORE")
    result = conn.execute(stmt, rows)
    return int(result.rowcount or 0)


def draw_count(conn: Connection) -> int:
    return int(conn.execute(select(draws.c.id)).fetchall().__len__())
