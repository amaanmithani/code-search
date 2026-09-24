"""End-to-end evaluation: index pinned repos (docstrings stripped), run every mode.

Model outputs (query embeddings, cross-encoder scores) are cached on disk under
``data/cache`` so an interrupted run resumes cheaply. Because of that cache,
the per-mode latency measures retrieval with model outputs already available;
the uncached cost of one query embedding and one rerank call is measured
separately on a small sample and reported as ``model_latency_ms``.
"""

from __future__ import annotations

import json
import platform
import re
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from codesearch.benchmark import (
    MIN_DOC_WORDS,
    MIN_QUERY_WORDS,
    PINNED_REPOS,
    Pair,
    RepoSpec,
    build_benchmark,
    ensure_repo,
    strip_transform,
)
from codesearch.cache import CachedQueryEmbedder, CachedReranker, KVCache
from codesearch.embeddings import Embedder
from codesearch.index import IndexStore, index_repository
from codesearch.metrics import summarize
from codesearch.rerank import Reranker
from codesearch.search import MODES, Mode, Searcher

EVAL_K = 10
MODEL_LATENCY_SAMPLE = 5  # per repo; the eval machine's Ollama server is shared


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()


def _p50(values: list[float]) -> float | None:
    return float(np.percentile(values, 50)) if values else None


def run_eval(
    data_dir: Path,
    out_path: Path,
    embedder: Embedder,
    reranker: Reranker | None,
    repos: Sequence[RepoSpec] = PINNED_REPOS,
    modes: Sequence[Mode] = MODES,
    limit: int | None = None,
    n_resamples: int = 2000,
    rerank_depth: int = 20,
    log: Callable[[str], None] = print,
) -> dict[str, object]:
    modes = [m for m in modes if m != "hybrid+rerank" or reranker is not None]
    cache = KVCache(data_dir / "cache" / "models.sqlite")
    q_embedder = CachedQueryEmbedder(embedder, cache)
    c_reranker = CachedReranker(reranker, cache) if reranker is not None else None

    ranks: dict[str, dict[str, list[int | None]]] = {m: {} for m in modes}
    lats: dict[str, dict[str, list[float]]] = {m: {} for m in modes}
    repo_info: list[dict[str, object]] = []
    examples: list[dict[str, object]] = []
    embed_lats: list[float] = []
    rerank_lats: list[float] = []

    for spec in repos:
        root = ensure_repo(spec, data_dir / "repos")
        index_dir = data_dir / "index" / f"{spec.name}-{_slug(embedder.name)}"
        stats = index_repository(root, index_dir, embedder, transform=strip_transform)
        store = IndexStore(index_dir)
        try:
            _, chunks, _ = store.load()
        finally:
            store.close()
        bench = build_benchmark(spec.name, root, [c.text for c in chunks])
        pairs: list[Pair] = bench.pairs[:limit] if limit else bench.pairs
        log(
            f"{spec.name}: {len(chunks)} chunks, {len(pairs)} queries "
            f"(dropped {bench.dropped_duplicate} duplicate, {bench.dropped_verbatim} verbatim); "
            f"index +{stats.added} ~{stats.updated} ={stats.unchanged}"
        )
        repo_info.append(
            {
                "name": spec.name,
                "url": spec.url,
                "ref": spec.ref,
                "commit": spec.commit,
                "chunks": len(chunks),
                "queries": len(pairs),
                "candidate_pairs": bench.candidates,
                "dropped_duplicate_query": bench.dropped_duplicate,
                "dropped_verbatim_in_index": bench.dropped_verbatim,
            }
        )
        searcher = Searcher.from_index(index_dir, q_embedder, c_reranker)
        searcher.rerank_depth = rerank_depth
        searcher.precompute_queries([p.query for p in pairs])

        # Uncached model cost on a small sample.
        for pair in pairs[:MODEL_LATENCY_SAMPLE]:
            t0 = time.perf_counter()
            embedder.embed_query(pair.query)
            embed_lats.append((time.perf_counter() - t0) * 1000)
            if reranker is not None:
                head = searcher.search(pair.query, "hybrid", rerank_depth)
                docs = [searcher.document_for(h.chunk) for h in head]
                t0 = time.perf_counter()
                reranker.score(pair.query, docs)
                rerank_lats.append((time.perf_counter() - t0) * 1000)

        for mode in modes:
            r_list: list[int | None] = []
            l_list: list[float] = []
            for n, pair in enumerate(pairs, 1):
                t0 = time.perf_counter()
                hits = searcher.search(pair.query, mode, EVAL_K)
                l_list.append((time.perf_counter() - t0) * 1000)
                keys = [(h.chunk.path, h.chunk.start_line, h.chunk.end_line) for h in hits]
                r_list.append(keys.index(pair.target) + 1 if pair.target in keys else None)
                if mode == "hybrid+rerank" and n % 25 == 0:
                    log(f"  {mode}: {n}/{len(pairs)} queries")
            ranks[mode][spec.name] = r_list
            lats[mode][spec.name] = l_list
            for j, pair in enumerate(pairs[:3]):
                ex = next((e for e in examples if e["query"] == pair.query), None)
                if ex is None:
                    ex = {
                        "repo": spec.name,
                        "query": pair.query,
                        "target": f"{pair.path}:{pair.start_line} {pair.qualname}",
                        "rank": {},
                    }
                    examples.append(ex)
                ex["rank"][mode] = r_list[j]  # type: ignore[index]
            s = summarize(r_list, l_list, n_resamples=200)
            log(f"  {mode:<14} mrr@10={s['mrr@10']['mean']:.3f}")  # type: ignore[index]
    cache.close()

    results: dict[str, object] = {}
    for mode in modes:
        all_r = [r for name in ranks[mode] for r in ranks[mode][name]]
        all_l = [x for name in lats[mode] for x in lats[mode][name]]
        results[mode] = {
            "overall": summarize(all_r, all_l, n_resamples=n_resamples),
            "per_repo": {
                name: summarize(ranks[mode][name], lats[mode][name], n_resamples=n_resamples)
                for name in ranks[mode]
            },
        }
    report: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "embedder": embedder.name,
            "reranker": reranker.name if reranker is not None else None,
            "k": EVAL_K,
            "bm25": {"k1": 1.2, "b": 0.75},
            "rrf_k": 60,
            "candidates_per_retriever": 100,
            "rerank_depth": rerank_depth,
            "min_docstring_words": MIN_DOC_WORDS,
            "min_query_words": MIN_QUERY_WORDS,
            "bootstrap": {"resamples": n_resamples, "ci": 0.95, "unit": "query"},
            "latency_note": "latency_ms is wall-clock retrieval time per query with model "
            "outputs (query embedding, cross-encoder scores) already computed and cached: "
            "BM25 + vector search + fusion + cache lookups. Uncached model cost is in "
            "model_latency_ms.",
            "machine": f"{platform.system()} {platform.machine()} / Python "
            f"{platform.python_version()}",
            "limit": limit,
        },
        "repos": repo_info,
        "results": results,
        "model_latency_ms": {
            "query_embedding_p50": _p50(embed_lats),
            "rerank_call_p50": _p50(rerank_lats),
            "rerank_candidates_per_call": rerank_depth,
            "n": len(embed_lats),
        },
        "examples": examples,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    return report
