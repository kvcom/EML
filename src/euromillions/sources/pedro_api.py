from __future__ import annotations

from datetime import date

import httpx

from euromillions.sources.base import DrawResult


class PedroApiSource:
    name = "pedro_api"
    url = "https://api.euro-million.net/results/latest"

    def fetch_latest(self) -> list[DrawResult]:
        try:
            resp = httpx.get(self.url, timeout=10.0)
            resp.raise_for_status()
            payload = resp.json()
        except Exception:
            return []
        nums = payload.get("numbers", [])
        stars = payload.get("stars", [])
        if len(nums) != 5 or len(stars) != 2:
            return []
        m = tuple(sorted(int(x) for x in nums))
        s = tuple(sorted(int(x) for x in stars))
        return [
            DrawResult(
                draw_date=date.fromisoformat(payload.get("date", date.today().isoformat())),
                mains=(m[0], m[1], m[2], m[3], m[4]),
                stars=(s[0], s[1]),
                source_url=self.url,
                raw_payload=resp.text,
            )
        ]

    def fetch_since(self, since_date: date) -> list[DrawResult]:
        return self.fetch_latest()
