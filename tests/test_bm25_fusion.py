import numpy as np
import pytest

from codesearch.bm25 import BM25, top_k_indices
from codesearch.fusion import reciprocal_rank_fusion


def test_bm25_ranks_matching_document_first() -> None:
    docs = [["parse", "config", "file"], ["send", "http", "request"], ["http", "http", "adapter"]]
    bm = BM25(docs)
    assert bm.top_k(["http", "request"], 3)[0][0] == 1
    assert bm.top_k(["config"], 3) == [(0, pytest.approx(bm.scores(["config"])[0]))]
    assert bm.top_k(["missing"], 3) == []


def test_bm25_idf_and_length_normalisation() -> None:
    bm = BM25([["a", "rare"], ["a"], ["a"], ["a", "x", "x", "x", "x", "x", "x", "rare"]])
    assert bm.idf["rare"] > bm.idf["a"]
    s = bm.scores(["rare"])
    assert s[0] > s[3] > 0  # same tf, shorter doc wins
    assert s[1] == 0


def test_bm25_empty_corpus() -> None:
    bm = BM25([])
    assert bm.scores(["x"]).shape == (0,)
    assert bm.top_k(["x"], 5) == []


def test_top_k_indices_ties_and_bounds() -> None:
    scores = np.array([0.5, 0.9, 0.5, 0.1])
    assert [i for i, _ in top_k_indices(scores, 3)] == [1, 0, 2]
    assert len(top_k_indices(scores, 10)) == 4
    assert top_k_indices(scores, 0) == []


def test_rrf_combines_ranks() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]], k=60)
    assert fused[0][0] == "b"
    scores = dict(fused)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 63)


def test_rrf_items_in_one_list_only_and_tie_order() -> None:
    fused = reciprocal_rank_fusion([["x"], ["y"]], k=1)
    assert fused == [("x", 0.5), ("y", 0.5)]
    assert reciprocal_rank_fusion([]) == []
