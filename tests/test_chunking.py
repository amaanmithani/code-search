from pathlib import Path

import pytest

from codesearch.chunking import chunk_source, detect_language


def _by_name(chunks):  # type: ignore[no-untyped-def]
    return {c.qualname: c for c in chunks}


def test_python_chunks(fixtures: Path) -> None:
    chunks = chunk_source("sample.py", (fixtures / "sample.py").read_text())
    by = _by_name(chunks)
    assert by["parse_config"].kind == "function"
    assert (by["parse_config"].start_line, by["parse_config"].end_line) == (8, 11)
    # decorators and the attached comment are part of the chunk
    assert by["decorated"].text.startswith("# Attached comment.\n@staticmethod")
    assert by["HttpClient"].kind == "class"
    assert by["HttpClient"].end_line == 23  # header only, trailing blank trimmed
    assert "timeout = 10" in by["HttpClient"].text
    assert by["HttpClient.get_json"].kind == "method"
    assert by["HttpClient.get_json"].parent == "HttpClient"
    # nested function stays inside its parent, not its own chunk
    assert "helper" not in by
    assert "def helper" in by["HttpClient.get_json"].text
    assert by["HttpClient.Inner"].parent == "HttpClient"
    assert by["HttpClient.Inner.ping"].kind == "method"
    module = by["sample"]
    assert module.kind == "module"
    assert "MAX_RETRIES = 3" in module.text and "if __name__" in module.text
    assert "def parse_config" not in module.text
    assert [c.start_line for c in chunks] == sorted(c.start_line for c in chunks)


def test_line_spans_match_source(fixtures: Path) -> None:
    src = (fixtures / "sample.py").read_text().splitlines()
    for c in chunk_source("sample.py", "\n".join(src)):
        if c.kind != "module":
            assert c.text == "\n".join(src[c.start_line - 1 : c.end_line])


def test_go_chunks(fixtures: Path) -> None:
    by = _by_name(chunk_source("sample.go", (fixtures / "sample.go").read_text()))
    assert by["Server"].kind == "type"
    assert by["Server.Start"].kind == "method"
    assert by["Server.Start"].text.startswith("// Start opens")
    assert by["NewServer"].kind == "function"
    assert by["sample"].kind == "module"


def test_typescript_chunks(fixtures: Path) -> None:
    by = _by_name(chunk_source("sample.ts", (fixtures / "sample.ts").read_text()))
    assert by["User"].kind == "interface"
    assert by["UserStore"].kind == "class"
    assert by["UserStore.addUser"].kind == "method"
    assert by["formatName"].kind == "function"
    assert by["toUpper"].kind == "function"
    assert by["sample"].text == 'import { thing } from "./thing";'  # class braces covered


def test_context_header() -> None:
    (c,) = [
        c
        for c in chunk_source("a/b.py", "class A:\n    def f(self):\n        pass\n")
        if c.kind == "method"
    ]
    assert c.context() == "a/b.py method A.f"


def test_language_detection() -> None:
    assert detect_language("x/y.PY") == "python"
    assert detect_language("x.tsx") == "tsx"
    assert detect_language("README.md") is None
    with pytest.raises(ValueError):
        chunk_source("README.md", "hi")
    assert chunk_source("x.tsx", "export const f = () => <div/>;\n")[0].name == "f"
