"""Cross-encoder reranking behind a protocol.

The real implementation runs ``cross-encoder/ms-marco-MiniLM-L-6-v2`` through
ONNX Runtime + ``tokenizers`` (~90 MB, no PyTorch), installed via the
``rerank`` extra. Tests use :class:`OverlapReranker`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from codesearch.tokenize import tokenize

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class OverlapReranker:
    """Test double: scores by the fraction of query tokens present in the document."""

    name = "overlap"

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        q = set(tokenize(query))
        if not q:
            return [0.0] * len(documents)
        return [len(q & set(tokenize(d))) / len(q) for d in documents]


class OnnxCrossEncoder:  # pragma: no cover - needs the downloaded model
    def __init__(
        self, model_id: str = RERANK_MODEL, max_length: int = 256, batch_size: int = 16
    ) -> None:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self.model_id = model_id
        self.batch_size = batch_size
        model_path = hf_hub_download(model_id, "onnx/model.onnx")
        tok_path = hf_hub_download(model_id, "tokenizer.json")
        self._tok: Any = Tokenizer.from_file(tok_path)
        self._tok.enable_truncation(max_length=max_length)
        self._tok.enable_padding()
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._sess: Any = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._sess.get_inputs()}

    @property
    def name(self) -> str:
        return self.model_id

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        scores: list[float] = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]
            enc = self._tok.encode_batch([(query, d) for d in batch])
            feeds = {
                "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64),
            }
            feeds = {k: v for k, v in feeds.items() if k in self._inputs}
            logits = self._sess.run(None, feeds)[0]
            scores.extend(float(x) for x in np.asarray(logits).reshape(len(batch), -1)[:, 0])
        return scores
