from __future__ import annotations

from datetime import date

from euromillions.sources.base import DrawResult


class NationalLotterySource:
    name = "national_lottery"
    url = "https://www.national-lottery.co.uk/results/euromillions"

    def fetch_latest(self) -> list[DrawResult]:
        return [
            DrawResult(
                draw_date=date.today(),
                mains=(1, 2, 3, 4, 5),
                stars=(1, 2),
                source_url=self.url,
                raw_payload="unavailable_or_js_blocked",
                status="unavailable",
            )
        ]

    def fetch_since(self, since_date: date) -> list[DrawResult]:
        return self.fetch_latest()
