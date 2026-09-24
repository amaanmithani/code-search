from pathlib import Path

import numpy as np
import pytest

from codesearch.embeddings import HashingEmbedder
from codesearch.index import IndexStore, index_repository, iter_source_files


class CountingEmbedder(HashingEmbedder):
    def __init__(self) -> None:
        super().__init__(dim=32)
        self.calls = 0
        self.texts = 0

    def embed_documents(self, texts):  # type: ignore[no-untyped-def]
        self.calls += 1
        self.texts += len(texts)
        return super().embed_documents(texts)


def _load(index_dir: Path):  # type: ignore[no-untyped-def]
    store = IndexStore(index_dir)
    try:
        return store.load()
    finally:
        store.close()


def test_full_then_incremental(repo: Path, tmp_path: Path) -> None:
    idx = tmp_path / "idx"
    emb = CountingEmbedder()
    s1 = index_repository(repo, idx, emb)
    assert (s1.added, s1.updated, s1.removed, s1.unchanged) == (3, 0, 0, 0)
    _, chunks, vecs = _load(idx)
    assert vecs is not None and vecs.shape == (len(chunks), 32)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)

    # nothing changed -> nothing embedded
    before = emb.texts
    s2 = index_repository(repo, idx, emb)
    assert (s2.added, s2.updated, s2.unchanged) == (0, 0, 3)
    assert emb.texts == before

    # modify one file, add one, delete one
    p = repo / "pkg" / "strings_util.py"
    p.write_text(p.read_text() + "\n\ndef shout(s):\n    return s.upper()\n")
    (repo / "pkg" / "new.go").write_text("package p\n\nfunc Hello() {}\n")
    (repo / "pkg" / "dup.py").unlink()
    s3 = index_repository(repo, idx, emb)
    assert (s3.added, s3.updated, s3.removed, s3.unchanged) == (1, 1, 1, 1)
    # unchanged chunks inside the modified file reuse cached vectors
    assert s3.chunks_embedded == 3  # shout + Hello + new.go module chunk
    _, chunks, vecs = _load(idx)
    names = {c.qualname for c in chunks}
    assert {"shout", "Hello", "reverse_words"} <= names
    assert "a" not in names and "b" not in names
    assert vecs is not None and len(vecs) == len(chunks)


def test_embedder_change_rebuilds(repo: Path, tmp_path: Path) -> None:
    idx = tmp_path / "idx"
    index_repository(repo, idx, HashingEmbedder(dim=16))
    msgs: list[str] = []
    s = index_repository(repo, idx, HashingEmbedder(dim=8), log=msgs.append)
    assert s.added == 3 and any("rebuilding" in m for m in msgs)
    assert _load(idx)[2].shape[1] == 8  # type: ignore[union-attr]


def test_no_embedder_and_transform(repo: Path, tmp_path: Path) -> None:
    idx = tmp_path / "idx"
    index_repository(repo, idx, None, transform=lambda p, t: t.replace("vowels", "VOWELS"))
    _, chunks, vecs = _load(idx)
    assert vecs is None
    assert any("VOWELS" in c.text for c in chunks)


def test_walk_skips_hidden_and_vendor(repo: Path) -> None:
    (repo / ".git").mkdir()
    (repo / ".git" / "x.py").write_text("x = 1\n")
    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "y.ts").write_text("let y = 1;\n")
    (repo / "notes.md").write_text("hi")
    rels = [r for r, _ in iter_source_files(repo)]
    assert rels == sorted(rels)
    assert all(r.startswith("pkg/") for r in rels)


def test_default_index_dir(repo: Path) -> None:
    index_repository(repo)
    assert (repo / ".codesearch" / "index.sqlite").exists()


def test_chunker_version_change_rebuilds(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codesearch import index as index_mod

    idx = tmp_path / "idx"
    emb = CountingEmbedder()
    index_repository(repo, idx, emb)
    monkeypatch.setattr(index_mod, "CHUNKER_VERSION", 999)
    s = index_repository(repo, idx, emb)
    assert s.added == 3 and s.unchanged == 0
    assert s.chunks_embedded == 0  # vectors come from the embedding cache
