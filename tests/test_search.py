from pathlib import Path

import pytest

from codesearch.embeddings import HashingEmbedder
from codesearch.index import index_repository
from codesearch.rerank import OverlapReranker
from codesearch.search import MODES, Searcher


@pytest.fixture
def searcher(repo: Path, tmp_path: Path) -> Searcher:
    idx = tmp_path / "idx"
    emb = HashingEmbedder(dim=64)
    index_repository(repo, idx, emb)
    return Searcher.from_index(idx, emb, OverlapReranker())


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_finds_obvious_target(searcher: Searcher, mode: str) -> None:
    hits = searcher.search("count vowels in text", mode, k=3)  # type: ignore[arg-type]
    assert hits[0].chunk.qualname == "count_vowels"
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))


def test_rerank_keeps_tail_below_head(searcher: Searcher) -> None:
    searcher.rerank_depth = 2
    hits = searcher.search("mean of numbers", "hybrid+rerank", k=10)
    assert len(hits) > 2
    assert hits[1].score > hits[2].score


def test_precomputed_queries_are_used(searcher: Searcher) -> None:
    searcher.precompute_queries(["largest number values"])
    assert "largest number values" in searcher._query_cache
    assert searcher.search("largest number values", "dense", 1)[0].chunk.qualname == "Stats.maximum"


def test_errors(repo: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Searcher.from_index(tmp_path / "missing")
    idx = tmp_path / "idx"
    index_repository(repo, idx, None)
    plain = Searcher.from_index(idx)
    assert plain.search("reverse words", "bm25", 1)[0].chunk.name == "reverse_words"
    with pytest.raises(RuntimeError):
        plain.search("x", "dense")
    with pytest.raises(RuntimeError):
        plain._rerank("x", [])
    with pytest.raises(ValueError):
        plain.search("x", "nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Searcher.from_index(idx, HashingEmbedder())
