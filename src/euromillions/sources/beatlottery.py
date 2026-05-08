from __future__ import annotations

from datetime import date

from euromillions.sources.base import DrawResult


class BeatLotterySource:
    name = "beatlottery"

    def fetch_latest(self) -> list[DrawResult]:
        return []

    def fetch_since(self, since_date: date) -> list[DrawResult]:
        return []
