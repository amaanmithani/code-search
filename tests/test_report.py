import importlib.util
from pathlib import Path

import pytest

from codesearch.metrics import summarize

spec = importlib.util.spec_from_file_location(
    "report", Path(__file__).parents[1] / "scripts" / "report.py"
)
assert spec and spec.loader
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def _fake_report() -> dict[str, object]:
    s = summarize([1, None, 3], [1.0, 2.0, 3.0], n_resamples=20)
    return {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "config": {
            "embedder": "e",
            "reranker": "r",
            "rerank_depth": 50,
            "bootstrap": {"resamples": 20},
        },
        "repos": [
            {
                "name": "fx",
                "ref": "v1",
                "commit": "abcdef1234567890",
                "chunks": 9,
                "queries": 3,
                "dropped_duplicate_query": 0,
                "dropped_verbatim_in_index": 0,
            }
        ],
        "results": {"bm25": {"overall": s, "per_repo": {"fx": s}}},
        "query_embedding_latency_ms": {"p50": 12.0, "n": 3},
    }


def test_render_and_update() -> None:
    block = report.render(_fake_report())
    assert "| `bm25` | 3 |" in block and "**fx** @ `v1`" in block and "12 ms" in block
    readme = f"intro\n{report.START}\nold\n{report.END}\ntail\n"
    new = report.update_readme(readme, block)
    assert "old" not in new and new.startswith("intro") and new.endswith("tail\n")
    with pytest.raises(SystemExit):
        report.update_readme("no markers", block)
