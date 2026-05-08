from __future__ import annotations

from heapq import nsmallest
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


def deterministic_top_k(
    items: Iterable[T],
    k: int,
    score: Callable[[T], float],
) -> list[T]:
    if k <= 0:
        return []
    scored = [(-score(item), index, item) for index, item in enumerate(items)]
    if len(scored) <= k:
        scored.sort()
        return [item for _, _, item in scored]
    if k * 8 >= len(scored):
        scored.sort()
        return [item for _, _, item in scored[:k]]
    best = nsmallest(k, scored)
    best.sort()
    return [item for _, _, item in best]
