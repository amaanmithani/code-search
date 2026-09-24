"""Minimal FastAPI app: ``GET /search?q=...&mode=hybrid&k=10``."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException, Query

from codesearch.search import MODES, Mode, Searcher


def create_app(searcher: Searcher) -> FastAPI:
    app = FastAPI(title="codesearch")

    @app.get("/search")
    def search(
        q: str = Query(..., min_length=1),
        mode: str = "hybrid",
        k: int = Query(10, ge=1, le=100),
    ) -> dict[str, Any]:
        if mode not in MODES:
            raise HTTPException(400, f"mode must be one of {list(MODES)}")
        try:
            hits = searcher.search(q, mode_cast(mode), k)
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc
        return {
            "query": q,
            "mode": mode,
            "hits": [
                {
                    "rank": h.rank,
                    "score": h.score,
                    "path": h.chunk.path,
                    "start_line": h.chunk.start_line,
                    "end_line": h.chunk.end_line,
                    "kind": h.chunk.kind,
                    "name": h.chunk.qualname,
                    "snippet": "\n".join(h.chunk.text.splitlines()[:10]),
                }
                for h in hits
            ],
        }

    return app


def mode_cast(mode: str) -> Mode:
    for m in MODES:
        if m == mode:
            return m
    raise ValueError(mode)
