from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol


@dataclass(frozen=True)
class DrawResult:
    draw_date: date
    mains: tuple[int, int, int, int, int]
    stars: tuple[int, int]
    source_url: str
    raw_payload: str
    status: str = "ok"


class ResultSource(Protocol):
    name: str

    def fetch_latest(self) -> list[DrawResult]: ...

    def fetch_since(self, since_date: date) -> list[DrawResult]: ...
