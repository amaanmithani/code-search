"""A tiny SQLite key/value cache so expensive model calls survive restarts.

Used by the evaluation to persist cross-encoder scores and query embeddings:
an interrupted ``codesearch eval`` resumes without redoing model work.
"""

from __future__ import annotations

import hashlib
import sqlite3
import struct
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from codesearch.embeddings import Embedder, Matrix, l2_normalize
from codesearch.rerank import Reranker


class KVCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value BLOB)")

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("\0".join(parts).encode()).hexdigest()

    def get_many(self, keys: Sequence[str]) -> dict[str, bytes]:
        out: dict[str, bytes] = {}
        for i in range(0, len(keys), 500):
            batch = list(keys[i : i + 500])
            q = f"SELECT key, value FROM kv WHERE key IN ({','.join('?' * len(batch))})"
            out.update(self.db.execute(q, batch).fetchall())
        return out

    def put_many(self, items: dict[str, bytes]) -> None:
        self.db.executemany("INSERT OR REPLACE INTO kv VALUES (?, ?)", items.items())
        self.db.commit()

    def close(self) -> None:
        self.db.close()


class CachedReranker:
    """Wraps a reranker; scores are cached per (model, query, document)."""

    def __init__(self, inner: Reranker, cache: KVCache) -> None:
        self.inner = inner
        self.cache = cache

    @property
    def name(self) -> str:
        return self.inner.name

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        keys = [self.cache.key(self.inner.name, query, d) for d in documents]
        hit = self.cache.get_many(keys)
        missing = [i for i, k in enumerate(keys) if k not in hit]
        if missing:
            fresh = self.inner.score(query, [documents[i] for i in missing])
            new = {keys[i]: struct.pack("<d", s) for i, s in zip(missing, fresh, strict=True)}
            self.cache.put_many(new)
            hit.update(new)
        return [struct.unpack("<d", hit[k])[0] for k in keys]


class CachedQueryEmbedder:
    """Wraps an embedder; query vectors are cached (documents pass through)."""

    def __init__(self, inner: Embedder, cache: KVCache) -> None:
        self.inner = inner
        self.cache = cache

    @property
    def name(self) -> str:
        return self.inner.name

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> Matrix:
        vec: Matrix = self.embed_queries([text])[0]
        return vec

    def embed_queries(self, texts: Sequence[str]) -> Matrix:
        keys = [self.cache.key(self.inner.name, "query", t) for t in texts]
        hit = self.cache.get_many(keys)
        missing = [i for i, k in enumerate(keys) if k not in hit]
        if missing:
            mat = self.inner.embed_queries([texts[i] for i in missing])
            new = {
                keys[i]: row.astype(np.float32).tobytes()
                for i, row in zip(missing, mat, strict=True)
            }
            self.cache.put_many(new)
            hit.update(new)
        if not keys:
            return np.zeros((0, 0), dtype=np.float32)
        return l2_normalize(np.stack([np.frombuffer(hit[k], dtype=np.float32) for k in keys]))
