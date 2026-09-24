import json
from pathlib import Path

from codesearch.benchmark import (
    RepoSpec,
    build_benchmark,
    clean_docstring,
    extract_pairs,
    first_sentence,
    strip_docstrings,
    strip_transform,
)
from codesearch.chunking import chunk_source
from codesearch.embeddings import HashingEmbedder
from codesearch.evaluate import run_eval
from codesearch.index import iter_source_files
from codesearch.rerank import OverlapReranker

SRC = '''class Stats:
    """Simple statistics helpers."""

    def mean(self, values):
        """Compute the arithmetic mean.

        Longer explanation here.
        """
        return sum(values) / len(values)


def only_doc():
    """Placeholder whose body is only this docstring."""
'''


def test_strip_docstrings_preserves_lines_and_parses() -> None:
    out = strip_docstrings(SRC)
    assert '"""' not in out
    assert len(out.splitlines()) == len(SRC.splitlines())
    assert "return sum(values)" in out
    # empty bodies become `pass`, so the result still chunks cleanly
    names = {c.qualname for c in chunk_source("m.py", out)}
    assert {"Stats", "Stats.mean", "only_doc"} <= names
    assert strip_transform("x.go", '"""keep"""') == '"""keep"""'


def test_clean_docstring_and_first_sentence() -> None:
    assert clean_docstring('"""Use :func:`~pkg.run` and ``x``."""') == "Use pkg.run and x."
    assert clean_docstring("f'{x}'") is None
    assert clean_docstring("b'bytes'") is None
    assert first_sentence("Do a thing. Then more.") == "Do a thing."
    assert first_sentence("Line one\ncontinues here\n\nNew para.") == "Line one continues here"


def test_extract_pairs(repo: Path) -> None:
    path = "pkg/math_util.py"
    pairs = extract_pairs("r", path, (repo / path).read_text())
    by = {p.qualname: p for p in pairs}
    assert set(by) == {"Stats.mean", "Stats.maximum", "only_doc"}
    assert by["Stats.mean"].query.startswith("Compute the arithmetic mean")
    assert by["Stats.mean"].target == (path, 4, 6)
    # "Too short to count." has < 8 words
    short = extract_pairs("r", "s.py", (repo / "pkg/strings_util.py").read_text())
    assert "short" not in {p.qualname for p in short}


def test_queries_never_appear_verbatim_in_indexed_code(repo: Path) -> None:
    texts = []
    for rel, p in iter_source_files(repo):
        texts += [c.text for c in chunk_source(rel, strip_transform(rel, p.read_text()))]
    bench = build_benchmark("r", repo, texts)
    assert bench.dropped_duplicate == 2  # dup.py's a() and b()
    assert len(bench.pairs) == 5
    haystack = " ".join(" ".join(t.lower().split()) for t in texts)
    for pair in bench.pairs:
        assert " ".join(pair.query.lower().split()) not in haystack


def test_verbatim_query_is_dropped(repo: Path) -> None:
    leak = "# count how many vowels appear in the given text, ignoring letter case entirely.\n"
    bench = build_benchmark("r", repo, [leak])
    assert bench.dropped_verbatim == 1


def test_run_eval_end_to_end(repo: Path, tmp_path: Path) -> None:
    data = tmp_path / "data"
    (data / "repos").mkdir(parents=True)
    (data / "repos" / "fx").symlink_to(repo)
    out = tmp_path / "eval.json"
    report = run_eval(
        data,
        out,
        HashingEmbedder(dim=64),
        OverlapReranker(),
        repos=[RepoSpec("fx", None, "local")],
        n_resamples=50,
        log=lambda _m: None,
    )
    saved = json.loads(out.read_text())
    assert saved["results"].keys() == {"bm25", "dense", "hybrid", "hybrid+rerank"}
    bm = saved["results"]["bm25"]["overall"]
    assert bm["n_queries"] == 5
    assert 0.0 <= bm["mrr@10"]["mean"] <= 1.0
    assert saved["repos"][0]["dropped_duplicate_query"] == 2
    assert saved["examples"] and "rank" in saved["examples"][0]
    assert report["config"]["reranker"] == "overlap"  # type: ignore[index]
    # no reranker -> that mode is skipped
    rep2 = run_eval(
        data,
        out,
        HashingEmbedder(dim=64),
        None,
        repos=[RepoSpec("fx", None, "local")],
        n_resamples=20,
        log=lambda _m: None,
    )
    assert "hybrid+rerank" not in rep2["results"]  # type: ignore[operator]
