import math

import pytest

from codesearch.metrics import bootstrap_ci, hit_at, reciprocal_rank, summarize


def test_reciprocal_rank_and_hits() -> None:
    assert reciprocal_rank(1) == 1.0
    assert reciprocal_rank(4) == 0.25
    assert reciprocal_rank(11) == 0.0
    assert reciprocal_rank(None) == 0.0
    assert hit_at(5, 5) == 1.0 and hit_at(6, 5) == 0.0 and hit_at(None, 10) == 0.0


def test_bootstrap_ci_brackets_mean_and_is_deterministic() -> None:
    vals = [1.0] * 30 + [0.0] * 70
    lo, hi = bootstrap_ci(vals, n_resamples=500, seed=1)
    assert lo < 0.3 < hi
    assert lo > 0.15 and hi < 0.45
    assert bootstrap_ci(vals, n_resamples=500, seed=1) == (lo, hi)
    assert bootstrap_ci([0.5] * 10) == (0.5, 0.5)
    assert all(math.isnan(x) for x in bootstrap_ci([]))


def test_summarize() -> None:
    s = summarize([1, 2, None, 10], [1.0, 2.0, 3.0, 4.0], n_resamples=100)
    assert s["n_queries"] == 4
    assert s["mrr@10"]["mean"] == pytest.approx((1 + 0.5 + 0 + 0.1) / 4)  # type: ignore[index]
    assert s["recall@1"]["mean"] == 0.25  # type: ignore[index]
    assert s["recall@5"]["mean"] == 0.5  # type: ignore[index]
    assert s["recall@10"]["mean"] == 0.75  # type: ignore[index]
    assert s["latency_ms"]["p50"] == 2.5  # type: ignore[index]
    empty = summarize([], [])
    assert math.isnan(empty["latency_ms"]["p50"])  # type: ignore[index]
