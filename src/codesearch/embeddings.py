"""Embedding clients behind a small protocol so tests never touch the network."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol

import httpx
import numpy as np
import numpy.typing as npt

from codesearch.tokenize import tokenize

Matrix = npt.NDArray[np.float32]

# Code is token-dense; ~3000 chars stays inside a 2048-token embedding context.
MAX_EMBED_CHARS = 3000


class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    def embed_documents(self, texts: Sequence[str]) -> Matrix: ...

    def embed_query(self, text: str) -> Matrix: ...

    def embed_queries(self, texts: Sequence[str]) -> Matrix: ...


def l2_normalize(m: Matrix) -> Matrix:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


# nomic-embed-text was trained with task prefixes; mxbai uses a query instruction.
_PREFIXES: dict[str, tuple[str, str]] = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
    "mxbai-embed-large": ("", "Represent this sentence for searching relevant passages: "),
}


class OllamaEmbedder:
    """Calls Ollama's OpenAI-compatible ``/v1/embeddings`` endpoint."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        batch_size: int = 32,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self._client = client or httpx.Client(base_url=base_url, timeout=timeout)
        self._doc_prefix, self._query_prefix = _PREFIXES.get(model.split(":")[0], ("", ""))

    @property
    def name(self) -> str:
        return f"ollama/{self.model}"

    def _embed(self, texts: Sequence[str]) -> Matrix:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t[:MAX_EMBED_CHARS] for t in texts[i : i + self.batch_size]]
            resp = self._client.post("/v1/embeddings", json={"model": self.model, "input": batch})
            resp.raise_for_status()
            data = sorted(resp.json()["data"], key=lambda d: d["index"])
            out.extend(d["embedding"] for d in data)
        if not out:
            return np.zeros((0, 0), dtype=np.float32)
        return l2_normalize(np.asarray(out, dtype=np.float32))

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return self._embed([self._doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> Matrix:
        vec: Matrix = self._embed([self._query_prefix + text])[0]
        return vec

    def embed_queries(self, texts: Sequence[str]) -> Matrix:
        return self._embed([self._query_prefix + t for t in texts])


class HashingEmbedder:
    """Deterministic bag-of-subwords hashing embedder.

    Not semantic at all: used as a test double and for offline smoke runs.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    @property
    def name(self) -> str:
        return f"hashing/{self.dim}"

    def _vec(self, text: str) -> Matrix:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return v

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> Matrix:
        vec: Matrix = self.embed_documents([text])[0]
        return vec

    def embed_queries(self, texts: Sequence[str]) -> Matrix:
        return self.embed_documents(texts)
