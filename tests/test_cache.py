from collections.abc import Sequence
from pathlib import Path

import numpy as np

from codesearch.cache import CachedQueryEmbedder, CachedReranker, KVCache
from codesearch.embeddings import HashingEmbedder
from codesearch.rerank import OverlapReranker


class CountingReranker(OverlapReranker):
    def __init__(self) -> None:
        self.pairs = 0

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        self.pairs += len(documents)
        return super().score(query, documents)


class CountingEmbedder(HashingEmbedder):
    def __init__(self) -> None:
        super().__init__(dim=8)
        self.queries = 0

    def embed_queries(self, texts: Sequence[str]):  # type: ignore[no-untyped-def]
        self.queries += len(texts)
        return super().embed_queries(texts)


def test_cached_reranker_persists_across_instances(tmp_path: Path) -> None:
    inner = CountingReranker()
    cache = KVCache(tmp_path / "c.sqlite")
    r = CachedReranker(inner, cache)
    first = r.score("read config", ["def read_config(): ...", "def other(): ..."])
    assert inner.pairs == 2 and r.name == "overlap"
    cache.close()
    r2 = CachedReranker(inner, KVCache(tmp_path / "c.sqlite"))
    again = r2.score("read config", ["def read_config(): ...", "def other(): ...", "config"])
    assert again[:2] == first and inner.pairs == 3  # only the new document was scored
    assert r2.misses == 1


def test_cached_query_embedder(tmp_path: Path) -> None:
    inner = CountingEmbedder()
    e = CachedQueryEmbedder(inner, KVCache(tmp_path / "c.sqlite"))
    a = e.embed_queries(["alpha", "beta"])
    b = e.embed_query("alpha")
    assert inner.queries == 2 and np.allclose(a[0], b)
    assert e.name == inner.name
    assert e.embed_documents(["x"]).shape == (1, 8)
    assert e.embed_queries([]).shape == (0, 0)


def test_kvcache_batches_large_lookups(tmp_path: Path) -> None:
    cache = KVCache(tmp_path / "c.sqlite")
    items = {cache.key(str(i)): bytes([i % 256]) for i in range(1200)}
    cache.put_many(items)
    assert cache.get_many(list(items)) == items
