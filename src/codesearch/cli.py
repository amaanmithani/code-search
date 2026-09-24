"""``codesearch`` command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from codesearch.embeddings import Embedder, HashingEmbedder, OllamaEmbedder
from codesearch.index import INDEX_DIRNAME, index_repository
from codesearch.rerank import Reranker
from codesearch.search import MODES, Hit, Searcher

SNIPPET_LINES = 6


def make_embedder(args: argparse.Namespace) -> Embedder:
    if args.embedder == "hashing":
        return HashingEmbedder()
    return OllamaEmbedder(model=args.model, base_url=args.ollama_url)


def make_reranker() -> Reranker:  # pragma: no cover - loads the ONNX model
    from codesearch.rerank import OnnxCrossEncoder

    return OnnxCrossEncoder()


def format_hit(hit: Hit, snippet_lines: int = SNIPPET_LINES) -> str:
    c = hit.chunk
    head = f"{hit.rank:>2}. {c.path}:{c.start_line}-{c.end_line}  {c.kind} {c.qualname}"
    head += f"  (score {hit.score:.4f})"
    lines = c.text.splitlines()
    body = [f"      {c.start_line + i:>5} | {line}" for i, line in enumerate(lines[:snippet_lines])]
    if len(lines) > snippet_lines:
        body.append("            ...")
    return "\n".join([head, *body])


def _add_embed_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--embedder", choices=["ollama", "hashing"], default="ollama")
    p.add_argument("--model", default="nomic-embed-text", help="Ollama embedding model")
    p.add_argument("--ollama-url", default="http://localhost:11434")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codesearch", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="index (or incrementally re-index) a repository")
    p_index.add_argument("repo", type=Path)
    p_index.add_argument("--index-dir", type=Path, default=None)
    p_index.add_argument("-v", "--verbose", action="store_true")
    _add_embed_args(p_index)

    p_query = sub.add_parser("query", help="search an indexed repository")
    p_query.add_argument("text")
    p_query.add_argument("--repo", type=Path, default=Path("."))
    p_query.add_argument("--index-dir", type=Path, default=None)
    p_query.add_argument("--mode", choices=MODES, default="hybrid")
    p_query.add_argument("-k", type=int, default=10)
    _add_embed_args(p_query)

    p_eval = sub.add_parser("eval", help="run the benchmark and write results/eval.json")
    p_eval.add_argument("--data-dir", type=Path, default=Path("data"))
    p_eval.add_argument("--out", type=Path, default=Path("results/eval.json"))
    p_eval.add_argument("--no-rerank", action="store_true")
    p_eval.add_argument("--limit", type=int, default=None, help="max queries per repo")
    _add_embed_args(p_eval)

    p_serve = sub.add_parser("serve", help="serve GET /search over HTTP (needs uvicorn)")
    p_serve.add_argument("--repo", type=Path, default=Path("."))
    p_serve.add_argument("--index-dir", type=Path, default=None)
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--rerank", action="store_true")
    _add_embed_args(p_serve)
    return parser


def _index_dir(args: argparse.Namespace) -> Path:
    path: Path = args.index_dir or args.repo / INDEX_DIRNAME
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "index":
        log = print if args.verbose else None
        stats = index_repository(args.repo, _index_dir(args), make_embedder(args), log=log)
        print(
            f"files: +{stats.added} added, ~{stats.updated} updated, "
            f"-{stats.removed} removed, ={stats.unchanged} unchanged; "
            f"{stats.chunks_embedded} chunks embedded"
        )
        return 0
    if args.command == "query":
        reranker = make_reranker() if args.mode == "hybrid+rerank" else None
        searcher = Searcher.from_index(_index_dir(args), make_embedder(args), reranker)
        hits = searcher.search(args.text, args.mode, args.k)
        if not hits:
            print("no results")
        for hit in hits:
            print(format_hit(hit))
        return 0
    if args.command == "eval":
        from codesearch.evaluate import run_eval

        reranker = None if args.no_rerank else make_reranker()
        run_eval(args.data_dir, args.out, make_embedder(args), reranker, limit=args.limit)
        print(f"wrote {args.out}")
        return 0
    if args.command == "serve":  # pragma: no cover - blocking server
        import uvicorn

        from codesearch.api import create_app

        reranker = make_reranker() if args.rerank else None
        searcher = Searcher.from_index(_index_dir(args), make_embedder(args), reranker)
        uvicorn.run(create_app(searcher), port=args.port)
        return 0
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
