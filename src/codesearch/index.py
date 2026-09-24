"""On-disk index: SQLite holds files, chunks and their embedding vectors.

Re-indexing is incremental: a file is re-chunked and re-embedded only when the
SHA-256 of its (possibly transformed) content changed; deleted files are
dropped. Changing the embedding model invalidates everything.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from codesearch.chunking import CHUNKER_VERSION, Chunk, chunk_source, detect_language
from codesearch.embeddings import Embedder, Matrix

INDEX_DIRNAME = ".codesearch"
SKIP_DIRS = frozenset(
    {"node_modules", "__pycache__", "venv", ".venv", "build", "dist", "vendor", "site-packages"}
)
MAX_FILE_BYTES = 1_000_000

# (relative posix path, original text) -> text to index
Transform = Callable[[str, str], str]

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, sha256 TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
    language TEXT NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    parent TEXT,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    text TEXT NOT NULL,
    vector BLOB
);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
CREATE TABLE IF NOT EXISTS embed_cache (key TEXT PRIMARY KEY, vector BLOB NOT NULL);
"""


def document_text(chunk: Chunk) -> str:
    """What BM25 and the embedder see: path/kind/qualname header plus the code."""
    return f"{chunk.context()}\n{chunk.text}"


@dataclass
class IndexStats:
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    chunks_embedded: int = 0


def iter_source_files(root: Path) -> Iterator[tuple[str, Path]]:
    """Yield (relative posix path, absolute path) for supported source files, sorted."""
    found: list[tuple[str, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS)
        for fn in filenames:
            if detect_language(fn) is None:
                continue
            p = Path(dirpath) / fn
            if p.stat().st_size > MAX_FILE_BYTES:
                continue
            found.append((p.relative_to(root).as_posix(), p))
    yield from sorted(found)


def _embed_cached(
    store: IndexStore, embedder: Embedder, chunks: list[Chunk], stats: IndexStats
) -> list[bytes | None]:
    """Embed chunks, reusing vectors for identical (model, text) pairs seen before."""
    texts = [document_text(c) for c in chunks]
    keys = [hashlib.sha256(f"{embedder.name}\0{t}".encode()).hexdigest() for t in texts]
    cached = dict(
        store.db.execute(
            f"SELECT key, vector FROM embed_cache WHERE key IN ({','.join('?' * len(keys))})",
            keys,
        ).fetchall()
    )
    missing = [i for i, k in enumerate(keys) if k not in cached]
    if missing:
        mat = embedder.embed_documents([texts[i] for i in missing])
        for i, row in zip(missing, mat, strict=True):
            cached[keys[i]] = row.astype(np.float32).tobytes()
            store.db.execute(
                "INSERT OR REPLACE INTO embed_cache VALUES (?, ?)", (keys[i], cached[keys[i]])
            )
        stats.chunks_embedded += len(missing)
    return [cached[k] for k in keys]


class IndexStore:
    def __init__(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        self.path = index_dir / "index.sqlite"
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def get_meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))

    def file_hashes(self) -> dict[str, str]:
        return {p: h for p, h in self.db.execute("SELECT path, sha256 FROM files")}

    def load(self) -> tuple[list[int], list[Chunk], Matrix | None]:
        """All chunks (ordered by id) and their vectors (None if not embedded)."""
        ids: list[int] = []
        chunks: list[Chunk] = []
        vecs: list[bytes | None] = []
        for row in self.db.execute(
            "SELECT id, path, language, kind, name, parent, start_line, end_line, text, vector "
            "FROM chunks ORDER BY id"
        ):
            ids.append(int(row[0]))
            chunks.append(Chunk(*row[1:9]))
            vecs.append(row[9])
        if not vecs or any(v is None for v in vecs):
            return ids, chunks, None
        matrix = np.stack([np.frombuffer(v, dtype=np.float32) for v in vecs if v is not None])
        return ids, chunks, matrix

    def reset(self) -> None:
        self.db.execute("DELETE FROM chunks")
        self.db.execute("DELETE FROM files")
        self.db.commit()


def index_repository(
    root: Path,
    index_dir: Path | None = None,
    embedder: Embedder | None = None,
    transform: Transform | None = None,
    log: Callable[[str], None] | None = None,
) -> IndexStats:
    """Build or incrementally update the index for ``root``."""
    root = root.resolve()
    store = IndexStore(index_dir or root / INDEX_DIRNAME)
    stats = IndexStats()
    try:
        embed_name = embedder.name if embedder is not None else "none"
        chunker = str(CHUNKER_VERSION)
        populated = bool(store.file_hashes())
        if populated and (
            store.get_meta("embedder") != embed_name or store.get_meta("chunker") != chunker
        ):
            if log:
                log("embedder or chunker changed; rebuilding index")
            store.reset()
        store.set_meta("embedder", embed_name)
        store.set_meta("chunker", chunker)

        known = store.file_hashes()
        seen: set[str] = set()
        for rel, abspath in iter_source_files(root):
            seen.add(rel)
            text = abspath.read_text(encoding="utf-8", errors="replace")
            if transform is not None:
                text = transform(rel, text)
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if known.get(rel) == digest:
                stats.unchanged += 1
                continue
            chunks = chunk_source(rel, text)
            vectors: list[bytes | None] = [None] * len(chunks)
            if embedder is not None and chunks:
                vectors = _embed_cached(store, embedder, chunks, stats)
            if rel in known:
                stats.updated += 1
            else:
                stats.added += 1
            store.db.execute("DELETE FROM chunks WHERE path = ?", (rel,))
            store.db.execute("INSERT OR REPLACE INTO files VALUES (?, ?)", (rel, digest))
            store.db.executemany(
                "INSERT INTO chunks (path, language, kind, name, parent, start_line, end_line, "
                "text, vector) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        c.path,
                        c.language,
                        c.kind,
                        c.name,
                        c.parent,
                        c.start_line,
                        c.end_line,
                        c.text,
                        v,
                    )
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )
            store.db.commit()
            if log:
                log(f"indexed {rel} ({len(chunks)} chunks)")
        for rel in sorted(set(known) - seen):
            store.db.execute("DELETE FROM chunks WHERE path = ?", (rel,))
            store.db.execute("DELETE FROM files WHERE path = ?", (rel,))
            stats.removed += 1
        store.db.commit()
    finally:
        store.close()
    return stats
