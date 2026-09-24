"""Ranking metrics for single-relevant-item queries, with bootstrap CIs."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def reciprocal_rank(rank: int | None, cutoff: int = 10) -> float:
    """1/rank if the relevant item is within ``cutoff`` (1-based rank), else 0."""
    return 1.0 / rank if rank is not None and rank <= cutoff else 0.0


def hit_at(rank: int | None, k: int) -> float:
    """Recall@k with exactly one relevant item: 1 if it's in the top k."""
    return 1.0 if rank is not None and rank <= k else 0.0


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling queries with replacement."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


def summarize(
    ranks: Sequence[int | None],
    latencies_ms: Sequence[float],
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, object]:
    """Point estimates + 95% CIs for MRR@10 and recall@{1,5,10}, and latency p50/p95."""
    per_query = {
        "mrr@10": [reciprocal_rank(r, 10) for r in ranks],
        "recall@1": [hit_at(r, 1) for r in ranks],
        "recall@5": [hit_at(r, 5) for r in ranks],
        "recall@10": [hit_at(r, 10) for r in ranks],
    }
    out: dict[str, object] = {"n_queries": len(ranks)}
    for name, vals in per_query.items():
        lo, hi = bootstrap_ci(vals, n_resamples=n_resamples, seed=seed)
        out[name] = {"mean": float(np.mean(vals)) if vals else float("nan"), "ci95": [lo, hi]}
    lat = np.asarray(latencies_ms, dtype=np.float64)
    out["latency_ms"] = {
        "p50": float(np.percentile(lat, 50)) if lat.size else float("nan"),
        "p95": float(np.percentile(lat, 95)) if lat.size else float("nan"),
    }
    return out
