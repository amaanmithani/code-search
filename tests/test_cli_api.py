import json
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from codesearch.api import create_app, mode_cast
from codesearch.cli import format_hit, main
from codesearch.embeddings import HashingEmbedder, OllamaEmbedder
from codesearch.index import index_repository
from codesearch.search import Searcher


def test_cli_index_and_query(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["index", str(repo), "--embedder", "hashing", "-v"]) == 0
    out = capsys.readouterr().out
    assert "+3 added" in out and "indexed pkg/" in out
    code = main(
        [
            "query",
            "reverse the words",
            "--repo",
            str(repo),
            "--embedder",
            "hashing",
            "--mode",
            "hybrid",
            "-k",
            "2",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "pkg/strings_util.py:1-3" in out and "reverse_words" in out
    main(["query", "zzzqqq", "--repo", str(repo), "--embedder", "hashing", "--mode", "bm25"])
    assert "no results" in capsys.readouterr().out


def test_cli_eval_without_rerank(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codesearch import benchmark, evaluate

    monkeypatch.setattr(evaluate, "PINNED_REPOS", (benchmark.RepoSpec("fx", None, "x"),))
    data = tmp_path / "data"
    (data / "repos").mkdir(parents=True)
    (data / "repos" / "fx").symlink_to(repo)
    out = tmp_path / "r.json"
    # run_eval's default argument was bound at import, so pass through the CLI's call path
    monkeypatch.setattr(
        evaluate.run_eval,
        "__defaults__",
        (evaluate.PINNED_REPOS, *evaluate.run_eval.__defaults__[1:]),  # type: ignore[index]
    )
    assert (
        main(
            [
                "eval",
                "--data-dir",
                str(data),
                "--out",
                str(out),
                "--no-rerank",
                "--embedder",
                "hashing",
                "--limit",
                "2",
            ]
        )
        == 0
    )
    assert json.loads(out.read_text())["results"]["bm25"]["overall"]["n_queries"] == 2


def test_format_hit_truncates(repo: Path, tmp_path: Path) -> None:
    index_repository(repo, tmp_path / "i", None)
    hit = Searcher.from_index(tmp_path / "i").search("mean", "bm25", 1)[0]
    text = format_hit(hit, snippet_lines=1)
    assert text.splitlines()[0].startswith(" 1. pkg/math_util.py")
    assert text.endswith("...")


def test_api(repo: Path, tmp_path: Path) -> None:
    emb = HashingEmbedder(dim=32)
    index_repository(repo, tmp_path / "i", emb)
    client = TestClient(create_app(Searcher.from_index(tmp_path / "i", emb)))
    r = client.get("/search", params={"q": "count vowels", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["hits"][0]["name"] == "count_vowels" and len(body["hits"]) == 2
    assert client.get("/search", params={"q": "x", "mode": "bad"}).status_code == 400
    assert client.get("/search", params={"q": "x", "mode": "hybrid+rerank"}).status_code == 400
    assert client.get("/search", params={"q": ""}).status_code == 422
    with pytest.raises(ValueError):
        mode_cast("nope")


def test_ollama_embedder_with_mock_transport() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append(payload)
        data = [
            {"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(payload["input"])
        ]
        return httpx.Response(200, json={"data": list(reversed(data))})

    client = httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))
    emb = OllamaEmbedder("nomic-embed-text", client=client, batch_size=2)
    assert emb.name == "ollama/nomic-embed-text"
    docs = emb.embed_documents(["a", "bb", "c" * 5000])
    assert docs.shape == (3, 2) and len(seen) == 2
    assert seen[0]["input"] == ["search_document: a", "search_document: bb"]
    assert len(seen[1]["input"][0]) == 3000  # type: ignore[index]
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0)
    q = emb.embed_query("hello")
    assert q.shape == (2,) and seen[-1]["input"] == ["search_query: hello"]
    assert emb.embed_queries(["a", "b"]).shape == (2, 2)
    assert emb.embed_documents([]).shape == (0, 0)


def test_hashing_embedder_properties() -> None:
    emb = HashingEmbedder(dim=16)
    a, b = emb.embed_documents(["parse config", "parse config"])
    assert np.allclose(a, b) and emb.embed_documents([]).shape == (0, 16)
    assert emb.embed_query("x").shape == (16,)
