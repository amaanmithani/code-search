"""Query-time retrieval: BM25, dense, hybrid (RRF) and hybrid + cross-encoder rerank."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import numpy as np

from codesearch.bm25 import BM25, top_k_indices
from codesearch.chunking import Chunk
from codesearch.embeddings import Embedder, Matrix
from codesearch.fusion import reciprocal_rank_fusion
from codesearch.index import IndexStore, document_text
from codesearch.rerank import Reranker
from codesearch.tokenize import tokenize

Mode = Literal["bm25", "dense", "hybrid", "hybrid+rerank"]
MODES: tuple[Mode, ...] = get_args(Mode)


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int


class Searcher:
    def __init__(
        self,
        chunks: list[Chunk],
        vectors: Matrix | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        candidates: int = 100,
        rerank_depth: int = 50,
        rrf_k: int = 60,
    ) -> None:
        self.chunks = chunks
        self.docs = [document_text(c) for c in chunks]
        self.bm25 = BM25([tokenize(d) for d in self.docs])
        self.vectors = vectors
        self.embedder = embedder
        self.reranker = reranker
        self.candidates = candidates
        self.rerank_depth = rerank_depth
        self.rrf_k = rrf_k
        self._query_cache: dict[str, Matrix] = {}

    @classmethod
    def from_index(
        cls,
        index_dir: Path,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> Searcher:
        if not (index_dir / "index.sqlite").exists():
            raise FileNotFoundError(f"no index at {index_dir}; run `codesearch index` first")
        store = IndexStore(index_dir)
        try:
            _, chunks, vectors = store.load()
            stored = store.get_meta("embedder")
        finally:
            store.close()
        if embedder is not None and stored != embedder.name:
            raise ValueError(f"index was embedded with {stored!r}, not {embedder.name!r}")
        return cls(chunks, vectors, embedder, reranker)

    def precompute_queries(self, queries: list[str]) -> None:
        """Batch-embed queries up front (evaluation); ``search`` then reuses them."""
        if self.embedder is None or not queries:
            return
        todo = [q for q in dict.fromkeys(queries) if q not in self._query_cache]
        if todo:
            for q, vec in zip(todo, self.embedder.embed_queries(todo), strict=True):
                self._query_cache[q] = vec

    # -- individual retrievers --------------------------------------------
    def _bm25(self, query: str, n: int) -> list[tuple[int, float]]:
        return self.bm25.top_k(tokenize(query), n)

    def _dense(self, query: str, n: int) -> list[tuple[int, float]]:
        if self.embedder is None or self.vectors is None:
            raise RuntimeError("dense retrieval needs an embedder and an embedded index")
        q = self._query_cache.get(query)
        if q is None:
            q = self.embedder.embed_query(query)
        sims = (self.vectors @ q).astype(np.float64)
        return top_k_indices(sims, n)

    def _hybrid(self, query: str, n: int) -> list[tuple[int, float]]:
        lexical = [i for i, _ in self._bm25(query, n)]
        dense = [i for i, _ in self._dense(query, n)]
        return reciprocal_rank_fusion([lexical, dense], k=self.rrf_k)[:n]

    def _rerank(self, query: str, fused: list[tuple[int, float]]) -> list[tuple[int, float]]:
        if self.reranker is None:
            raise RuntimeError("hybrid+rerank needs a reranker")
        head = fused[: self.rerank_depth]
        scores = self.reranker.score(query, [self.docs[i] for i, _ in head])
        reranked = sorted(zip((i for i, _ in head), scores, strict=True), key=lambda kv: -kv[1])
        # Anything below the rerank depth keeps its fused order after the reranked head.
        floor = min(scores, default=0.0)
        tail = [(i, floor - 1.0 - r) for r, (i, _) in enumerate(fused[self.rerank_depth :])]
        return reranked + tail

    def search(self, query: str, mode: Mode = "hybrid", k: int = 10) -> list[Hit]:
        n = max(k, self.candidates)
        if mode == "bm25":
            ranked = self._bm25(query, n)
        elif mode == "dense":
            ranked = self._dense(query, n)
        elif mode == "hybrid":
            ranked = self._hybrid(query, n)
        elif mode == "hybrid+rerank":
            ranked = self._rerank(query, self._hybrid(query, n))
        else:
            raise ValueError(f"unknown mode {mode!r}; choose from {', '.join(MODES)}")
        return [Hit(self.chunks[i], score, rank) for rank, (i, score) in enumerate(ranked[:k], 1)]
