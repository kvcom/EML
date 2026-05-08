from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable

from sqlalchemy import Connection, select

from euromillions.schema import main_combinations, star_combinations


def _decade_signature(nums: tuple[int, ...]) -> str:
    buckets = [0, 0, 0, 0, 0]
    for n in nums:
        buckets[min((n - 1) // 10, 4)] += 1
    return "-".join(str(x) for x in buckets)


def _main_row(nums: tuple[int, int, int, int, int]) -> dict[str, Any]:
    gaps = [nums[i + 1] - nums[i] for i in range(4)]
    odd = sum(1 for n in nums if n % 2)
    return {
        "m1": nums[0],
        "m2": nums[1],
        "m3": nums[2],
        "m4": nums[3],
        "m5": nums[4],
        "sum_main": sum(nums),
        "odd_count": odd,
        "even_count": 5 - odd,
        "span": nums[-1] - nums[0],
        "gap1": gaps[0],
        "gap2": gaps[1],
        "gap3": gaps[2],
        "gap4": gaps[3],
        "min_gap": min(gaps),
        "max_gap": max(gaps),
        "avg_gap": sum(gaps) / 4.0,
        "decade_bucket_signature": _decade_signature(nums),
        "combination_key": ",".join(map(str, nums)),
    }


def _star_row(stars: tuple[int, int]) -> dict[str, Any]:
    odd = sum(1 for n in stars if n % 2)
    return {
        "s1": stars[0],
        "s2": stars[1],
        "sum_stars": stars[0] + stars[1],
        "gap": stars[1] - stars[0],
        "odd_count": odd,
        "even_count": 2 - odd,
        "combination_key": ",".join(map(str, stars)),
    }


def _batched(rows: Iterable[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_main_combinations(conn: Connection, batch_size: int = 25_000) -> int:
    existing = conn.execute(select(main_combinations.c.id).limit(1)).first()
    if existing is not None:
        return 0
    inserted = 0
    rows = (_main_row(cast) for cast in combinations(range(1, 51), 5))
    for batch in _batched(rows, batch_size):
        result = conn.execute(main_combinations.insert(), batch)
        inserted += int(result.rowcount or 0)
    return inserted


def build_star_combinations(conn: Connection) -> int:
    existing = conn.execute(select(star_combinations.c.id).limit(1)).first()
    if existing is not None:
        return 0
    payload = [_star_row(s) for s in combinations(range(1, 13), 2)]
    result = conn.execute(star_combinations.insert(), payload)
    return int(result.rowcount or 0)
