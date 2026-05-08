from __future__ import annotations

from datetime import date

from euromillions.sources.base import DrawResult


class NationalLotterySource:
    name = "national_lottery"
    url = "https://www.national-lottery.co.uk/results/euromillions"

    def fetch_latest(self) -> list[DrawResult]:
        return []

    def fetch_since(self, since_date: date) -> list[DrawResult]:
        return self.fetch_latest()
