from __future__ import annotations

from datetime import date, datetime
import re

import httpx
from bs4 import BeautifulSoup

from euromillions.sources.base import DrawResult


class EuroMillionsComSource:
    name = "euro_millions_com"
    url = "https://www.euro-millions.com/results"

    @staticmethod
    def _parse_draw_date(soup: BeautifulSoup) -> date | None:
        for tag in soup.select("time[datetime]"):
            raw = tag.get("datetime")
            if not isinstance(raw, str) or not raw:
                continue
            try:
                return date.fromisoformat(raw[:10])
            except ValueError:
                continue
        text = soup.get_text(" ", strip=True)
        match = re.search(r"\b(\d{1,2})\s*(?:st|nd|rd|th)?\s+([A-Za-z]+)\s+(\d{4})\b", text)
        if match is None:
            return None
        try:
            day, month, year = match.groups()
            return datetime.strptime(f"{day} {month} {year}", "%d %B %Y").date()
        except ValueError:
            return None

    def fetch_latest(self) -> list[DrawResult]:
        try:
            resp = httpx.get(self.url, timeout=10.0)
            resp.raise_for_status()
        except Exception:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        parsed_date = self._parse_draw_date(soup)
        if parsed_date is None:
            return []
        nums: list[int] = []
        for ball_list in soup.select("ul.balls"):
            candidate = [
                int(n.text)
                for n in ball_list.select(".ball, .lucky-star")
                if n.text.strip().isdigit()
            ]
            if len(candidate) >= 7:
                nums = candidate[:7]
                break
        if len(nums) < 7:
            return []
        mains = tuple(sorted(nums[:5]))
        stars = tuple(sorted(nums[5:7]))
        return [
            DrawResult(
                draw_date=parsed_date,
                mains=(mains[0], mains[1], mains[2], mains[3], mains[4]),
                stars=(stars[0], stars[1]),
                source_url=self.url,
                raw_payload=resp.text[:5000],
            )
        ]

    def fetch_since(self, since_date: date) -> list[DrawResult]:
        return self.fetch_latest()
