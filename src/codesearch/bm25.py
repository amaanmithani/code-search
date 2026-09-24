"""A small Okapi BM25 implementation over pre-tokenised documents."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


class BM25:
    """Okapi BM25 with an inverted index (scores only touch matching postings)."""

    def __init__(self, documents: Sequence[Sequence[str]], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs = len(documents)
        self.doc_len = np.array([len(d) for d in documents], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_id, doc in enumerate(documents):
            for term, tf in Counter(doc).items():
                postings[term].append((doc_id, tf))
        self._postings = dict(postings)
        self.idf = {
            term: math.log(1.0 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self._postings.items()
        }

    def scores(self, query: Sequence[str]) -> npt.NDArray[np.float64]:
        """BM25 score of every document for ``query`` (0 where nothing matches)."""
        out = np.zeros(self.n_docs, dtype=np.float64)
        if not self.n_docs:
            return out
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-9))
        for term, qtf in Counter(query).items():
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            ids = np.fromiter((d for d, _ in postings), dtype=np.int64, count=len(postings))
            tfs = np.fromiter((t for _, t in postings), dtype=np.float64, count=len(postings))
            out[ids] += qtf * idf * tfs * (self.k1 + 1.0) / (tfs + norm[ids])
        return out

    def top_k(self, query: Sequence[str], k: int) -> list[tuple[int, float]]:
        """Top-``k`` (doc_id, score) pairs with a positive score, best first."""
        s = self.scores(query)
        return top_k_indices(s, k, positive_only=True)


def top_k_indices(
    scores: npt.NDArray[np.float64], k: int, *, positive_only: bool = False
) -> list[tuple[int, float]]:
    """Indices of the ``k`` largest scores, best first; ties broken by index."""
    if k <= 0 or scores.size == 0:
        return []
    k = min(k, scores.size)
    idx = np.argpartition(-scores, k - 1)[:k]
    # Stable ordering: score desc, then doc id asc.
    ordered = sorted(idx.tolist(), key=lambda i: (-scores[i], i))
    result = [(int(i), float(scores[i])) for i in ordered]
    if positive_only:
        result = [(i, s) for i, s in result if s > 0]
    return result
