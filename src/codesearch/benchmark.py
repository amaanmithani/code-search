"""Build a (query, function) benchmark from docstrings of pinned open-source repos.

For every Python function/method whose docstring has >= 8 words, the query is
the docstring's first sentence and the single relevant target is that
function's chunk. All function and class docstrings are removed from the
indexed code (replaced by ``pass`` plus blank lines, so line numbers are
unchanged), and any query that still appears verbatim somewhere in the indexed
text is dropped, so retrieval cannot succeed by string-matching the docstring.
"""

from __future__ import annotations

import ast
import inspect
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Node

from codesearch.chunking import chunk_source, parse
from codesearch.index import iter_source_files

MIN_DOC_WORDS = 8
MIN_QUERY_WORDS = 3


@dataclass(frozen=True)
class RepoSpec:
    name: str
    url: str | None  # None: already on disk (tests)
    commit: str
    ref: str = ""


PINNED_REPOS: tuple[RepoSpec, ...] = (
    RepoSpec(
        "requests",
        "https://github.com/psf/requests",
        "0e322af87745eff34caffe4df68456ebc20d9068",
        "v2.32.3",
    ),
    RepoSpec(
        "click",
        "https://github.com/pallets/click",
        "874ca2bc1c30d93a4ac6e36a15ed685eafe89097",
        "8.1.7",
    ),
)


@dataclass(frozen=True)
class Pair:
    repo: str
    query: str
    path: str
    qualname: str
    start_line: int
    end_line: int

    @property
    def target(self) -> tuple[str, int, int]:
        return (self.path, self.start_line, self.end_line)


def ensure_repo(spec: RepoSpec, repos_dir: Path) -> Path:  # pragma: no cover - network
    dest = repos_dir / spec.name
    if spec.url is None:
        return dest
    if not dest.exists():
        repos_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--quiet", "--depth", "1", "--branch", spec.ref, spec.url, str(dest)],
            check=True,
        )
    head = subprocess.run(
        ["git", "-C", str(dest), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != spec.commit:
        raise RuntimeError(f"{spec.name}: expected commit {spec.commit}, found {head}")
    return dest


# -- docstrings ---------------------------------------------------------------
def _docstring_node(defn: Node) -> Node | None:
    body = defn.child_by_field_name("body")
    if body is None or not body.named_children:
        return None
    first = body.named_children[0]
    is_doc = (
        first.type == "expression_statement"
        and len(first.named_children) == 1
        and first.named_children[0].type == "string"
    )
    return first if is_doc else None


def _walk(node: Node) -> list[Node]:
    out: list[Node] = []
    stack = [node]
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(reversed(n.named_children))
    return out


def strip_docstrings(source: str) -> str:
    """Replace every function/class docstring with ``pass``, preserving line numbers."""
    root = parse(source, "python")
    data = source.encode("utf-8")
    spans = []
    for node in _walk(root):
        if node.type in ("function_definition", "class_definition"):
            doc = _docstring_node(node)
            if doc is not None:
                spans.append((doc.start_byte, doc.end_byte))
    for start, end in sorted(spans, reverse=True):
        newlines = data[start:end].count(b"\n")
        data = data[:start] + b"pass" + b"\n" * newlines + data[end:]
    return data.decode("utf-8")


def strip_transform(_path: str, source: str) -> str:
    return strip_docstrings(source) if _path.endswith(".py") else source


_ROLE_RE = re.compile(r":[a-z]+(?::[a-z]+)?:`~?([^`]+)`")


def clean_docstring(raw: str) -> str | None:
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(value, str):
        return None
    text = inspect.cleandoc(value)
    text = _ROLE_RE.sub(r"\1", text)
    return text.replace("``", "")


def first_sentence(doc: str) -> str:
    paragraph = re.split(r"\n\s*\n", doc.strip(), maxsplit=1)[0]
    paragraph = " ".join(paragraph.split())
    return re.split(r"(?<=[.!?])\s+", paragraph, maxsplit=1)[0]


def extract_pairs(repo: str, path: str, source: str) -> list[Pair]:
    """Docstring/function pairs for the function and method chunks of one file."""
    docs: dict[tuple[int, str], str] = {}
    for node in _walk(parse(source, "python")):
        if node.type != "function_definition":
            continue
        doc = _docstring_node(node)
        name = node.child_by_field_name("name")
        if doc is None or name is None or name.text is None:
            continue
        text = clean_docstring(doc.text.decode("utf-8", "replace") if doc.text else "")
        if text:
            docs.setdefault((node.end_point[0] + 1, name.text.decode()), text)
    pairs = []
    for chunk in chunk_source(path, source, "python"):
        if chunk.kind not in ("function", "method"):
            continue
        doc_text = docs.get((chunk.end_line, chunk.name))
        if doc_text is None or len(doc_text.split()) < MIN_DOC_WORDS:
            continue
        query = first_sentence(doc_text)
        if len(query.split()) < MIN_QUERY_WORDS:
            continue
        pairs.append(Pair(repo, query, path, chunk.qualname, chunk.start_line, chunk.end_line))
    return pairs


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


@dataclass
class Benchmark:
    pairs: list[Pair]
    dropped_duplicate: int
    dropped_verbatim: int
    candidates: int


def build_benchmark(repo: str, root: Path, indexed_texts: list[str]) -> Benchmark:
    """Extract pairs from ``root`` and filter them against the indexed chunk texts."""
    raw: list[Pair] = []
    for rel, abspath in iter_source_files(root):
        if rel.endswith(".py"):
            raw.extend(extract_pairs(repo, rel, abspath.read_text("utf-8", errors="replace")))
    counts = Counter(_norm(p.query) for p in raw)
    unique = [p for p in raw if counts[_norm(p.query)] == 1]
    haystack = "\n".join(_norm(t) for t in indexed_texts)
    kept = [p for p in unique if _norm(p.query) not in haystack]
    return Benchmark(
        pairs=kept,
        dropped_duplicate=len(raw) - len(unique),
        dropped_verbatim=len(unique) - len(kept),
        candidates=len(raw),
    )
