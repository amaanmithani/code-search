"""Syntax-aware chunking with tree-sitter.

One chunk per function / method / class header / type, carrying the file path,
the enclosing class (for methods) and 1-based inclusive line spans. Nested
functions stay inside their parent's chunk (they are rarely useful search hits
on their own). Top-level code not covered by any definition (imports, constants,
``if __name__ == "__main__"`` blocks) becomes one ``module`` chunk per file.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import PurePosixPath

import tree_sitter_go
import tree_sitter_python
import tree_sitter_typescript
from tree_sitter import Language, Node, Parser

# Bump when chunk boundaries change so existing indexes are rebuilt.
CHUNKER_VERSION = 2

EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".go": "go",
    ".ts": "typescript",
    ".tsx": "tsx",
}


@dataclass(frozen=True)
class Chunk:
    path: str
    language: str
    kind: str  # function | method | class | type | interface | module
    name: str
    parent: str | None
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    text: str

    @property
    def qualname(self) -> str:
        return f"{self.parent}.{self.name}" if self.parent else self.name

    def context(self) -> str:
        """Header prepended to the code for BM25 and embedding."""
        return f"{self.path} {self.kind} {self.qualname}"


@dataclass(frozen=True)
class LanguageSpec:
    functions: frozenset[str]
    classes: frozenset[str]  # containers whose function children are methods
    types: frozenset[str] = frozenset()  # whole-node chunks (Go types, TS interfaces)
    wrappers: frozenset[str] = frozenset()  # decorated_definition / export_statement


def _field_text(node: Node, name: str) -> str | None:
    child = node.child_by_field_name(name)
    if child is None or child.text is None:
        return None
    return child.text.decode("utf-8", "replace")


def _go_receiver_type(node: Node) -> str | None:
    receiver = node.child_by_field_name("receiver")
    if receiver is None:
        return None
    for param in receiver.named_children:
        typ = param.child_by_field_name("type")
        while typ is not None and typ.type in ("pointer_type", "generic_type"):
            inner = typ.named_children[0] if typ.named_children else None
            typ = inner
        if typ is not None and typ.text is not None:
            return typ.text.decode("utf-8", "replace")
    return None


def _go_type_name(node: Node) -> str | None:
    for spec in node.named_children:
        if spec.type in ("type_spec", "type_alias"):
            return _field_text(spec, "name")
    return None


SPECS: dict[str, LanguageSpec] = {
    "python": LanguageSpec(
        functions=frozenset({"function_definition"}),
        classes=frozenset({"class_definition"}),
        wrappers=frozenset({"decorated_definition"}),
    ),
    "go": LanguageSpec(
        functions=frozenset({"function_declaration", "method_declaration"}),
        classes=frozenset(),
        types=frozenset({"type_declaration"}),
    ),
    "typescript": LanguageSpec(
        functions=frozenset({"function_declaration", "method_definition"}),
        classes=frozenset({"class_declaration", "abstract_class_declaration"}),
        types=frozenset({"interface_declaration", "type_alias_declaration", "enum_declaration"}),
        wrappers=frozenset({"export_statement", "lexical_declaration"}),
    ),
}
SPECS["tsx"] = SPECS["typescript"]


@cache
def _parser(language: str) -> Parser:
    if language == "python":
        lang = Language(tree_sitter_python.language())
    elif language == "go":
        lang = Language(tree_sitter_go.language())
    elif language == "typescript":
        lang = Language(tree_sitter_typescript.language_typescript())
    elif language == "tsx":
        lang = Language(tree_sitter_typescript.language_tsx())
    else:
        raise ValueError(f"unsupported language: {language}")
    return Parser(lang)


def detect_language(path: str) -> str | None:
    return EXTENSIONS.get(PurePosixPath(path).suffix.lower())


def parse(source: str, language: str) -> Node:
    return _parser(language).parse(source.encode("utf-8")).root_node


class _Chunker:
    def __init__(self, path: str, source: str, language: str) -> None:
        self.path = path
        self.language = language
        self.spec = SPECS[language]
        self.lines = source.splitlines()
        self.chunks: list[Chunk] = []
        self.covered: set[int] = set()  # 0-based line numbers inside some chunk

    # -- helpers -----------------------------------------------------------
    def _span_text(self, start: int, end: int) -> str:
        return "\n".join(self.lines[start : end + 1])

    def _leading_comment_start(self, node: Node) -> int:
        """Extend a chunk upwards over directly-attached comments (Go/TS doc comments)."""
        start = node.start_point[0]
        prev = node.prev_named_sibling
        while prev is not None and prev.type == "comment" and prev.end_point[0] >= start - 1:
            start = prev.start_point[0]
            prev = prev.prev_named_sibling
        return start

    def _emit(
        self,
        kind: str,
        name: str,
        parent: str | None,
        start: int,
        end: int,
        text: str | None = None,
    ) -> None:
        self.chunks.append(
            Chunk(
                path=self.path,
                language=self.language,
                kind=kind,
                name=name,
                parent=parent,
                start_line=start + 1,
                end_line=end + 1,
                text=text if text is not None else self._span_text(start, end),
            )
        )
        self.covered.update(range(start, end + 1))

    def _unwrap(self, node: Node) -> Node | None:
        """Return the definition inside a wrapper node (decorators, export, const)."""
        spec = self.spec
        if node.type == "decorated_definition":
            return node.child_by_field_name("definition")
        if node.type == "export_statement":
            decl = node.child_by_field_name("declaration")
            if decl is None:
                return None
            return self._unwrap(decl) if decl.type in spec.wrappers else decl
        if node.type == "lexical_declaration":
            # const handler = (req) => {...}
            for decl in node.named_children:
                value = decl.child_by_field_name("value")
                if value is not None and value.type in ("arrow_function", "function_expression"):
                    return decl
            return None
        return node

    def _name(self, node: Node) -> str:
        if self.language == "go" and node.type == "type_declaration":
            return _go_type_name(node) or "<type>"
        return _field_text(node, "name") or "<anonymous>"

    # -- walk ----------------------------------------------------------------
    def visit(self, node: Node, parent: str | None, in_class: bool) -> None:
        spec = self.spec
        for child in node.named_children:
            outer = child
            inner = self._unwrap(child) if child.type in spec.wrappers else child
            if inner is None:
                continue
            start = self._leading_comment_start(outer)
            end = outer.end_point[0]
            if inner.type in spec.functions or inner.type == "variable_declarator":
                name = self._name(inner)
                owner = parent
                kind = "method" if in_class else "function"
                if self.language == "go" and inner.type == "method_declaration":
                    owner = _go_receiver_type(inner)
                    kind = "method"
                self._emit(kind, name, owner, start, end)
            elif inner.type in spec.classes:
                self._class(inner, parent, start, end)
            elif inner.type in spec.types:
                kind = "interface" if inner.type == "interface_declaration" else "type"
                self._emit(kind, self._name(inner), parent, start, end)

    def _class(self, node: Node, parent: str | None, start: int, end: int) -> None:
        name = self._name(node)
        qual = f"{parent}.{name}" if parent else name
        body = node.child_by_field_name("body")
        before = len(self.chunks)
        if body is not None:
            self.visit(body, qual, in_class=True)
        members = self.chunks[before:]
        # Class chunk = header + class-level statements before the first member.
        header_end = min((c.start_line - 1 for c in members), default=end + 1) - 1
        header_end = max(start, min(header_end, end))
        while header_end > start and not self.lines[header_end].strip():
            header_end -= 1
        self.chunks.insert(
            before,
            Chunk(
                path=self.path,
                language=self.language,
                kind="class",
                name=name,
                parent=parent,
                start_line=start + 1,
                end_line=header_end + 1,
                text=self._span_text(start, header_end),
            ),
        )
        # The whole class body counts as covered (closing braces, blank lines).
        self.covered.update(range(start, end + 1))

    def module_chunk(self) -> None:
        rest = [i for i, line in enumerate(self.lines) if i not in self.covered and line.strip()]
        if not rest:
            return
        parts: list[str] = []
        prev = None
        for i in rest:
            if prev is not None and i != prev + 1:
                parts.append("...")
            parts.append(self.lines[i])
            prev = i
        name = PurePosixPath(self.path).stem
        self.chunks.append(
            Chunk(
                path=self.path,
                language=self.language,
                kind="module",
                name=name,
                parent=None,
                start_line=rest[0] + 1,
                end_line=rest[-1] + 1,
                text="\n".join(parts),
            )
        )


def chunk_source(path: str, source: str, language: str | None = None) -> list[Chunk]:
    """Chunk one file. ``language`` defaults to detection from the extension."""
    lang = language or detect_language(path)
    if lang is None:
        raise ValueError(f"cannot detect language for {path}")
    chunker = _Chunker(path, source, lang)
    chunker.visit(parse(source, lang), parent=None, in_class=False)
    chunker.module_chunk()
    return sorted(chunker.chunks, key=lambda c: (c.start_line, c.end_line))
