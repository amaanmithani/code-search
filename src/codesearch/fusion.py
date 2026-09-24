"""Reciprocal-rank fusion (Cormack, Clarke & Buettcher, SIGIR 2009)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence


def reciprocal_rank_fusion[T: Hashable](
    rankings: Sequence[Sequence[T]], k: int = 60
) -> list[tuple[T, float]]:
    """Fuse ranked lists: ``score(d) = sum_r 1 / (k + rank_r(d))`` with 1-based ranks.

    Only ranks matter, so BM25 scores and cosine similarities (different scales)
    can be combined without calibration. Ties keep first-seen order.
    """
    scores: dict[T, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])
