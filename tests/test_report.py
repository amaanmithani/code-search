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
        "paired_comparisons": {
            "hybrid vs bm25": {
                "mrr@10_diff": 0.05,
                "ci95": [0.01, 0.09],
                "a_better": 7,
                "b_better": 2,
            }
        },
        "model_latency_ms": {
            "query_embedding_p50": 12.0,
            "rerank_call_p50": 99.0,
            "rerank_candidates_per_call": 20,
            "n": 3,
            "uncached_rerank_pairs_in_timed_loop": 0,
        },
    }


def test_render_and_update() -> None:
    block = report.render(_fake_report())
    assert "| `bm25` | 3 |" in block and "**fx** @ `v1`" in block and "p50 12 ms" in block
    assert "on-disk cache" in block
    assert "| hybrid vs bm25 | +0.050 | [+0.010, +0.090] | 7 / 2 |" in block
    readme = f"intro\n{report.START}\nold\n{report.END}\ntail\n"
    new = report.update_readme(readme, block)
    assert "old" not in new and new.startswith("intro") and new.endswith("tail\n")
    with pytest.raises(SystemExit):
        report.update_readme("no markers", block)
