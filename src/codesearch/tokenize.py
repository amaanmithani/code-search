"""Identifier-aware tokenisation shared by the BM25 index and queries.

Code identifiers are compound words (``getUserName``, ``parse_http_header``,
``HTTPAdapter``). A developer querying in English writes "user name", so we emit
both the full identifier and its sub-words. Everything is lower-cased.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")
# Split points inside an identifier: lower->Upper, ACRONYM->Word, letter<->digit.
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")

# Deliberately tiny: BM25's IDF already down-weights common tokens; this only
# drops English glue words that carry no signal in a natural-language query.
STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
        "which",
        "when",
    ]
)


def split_identifier(identifier: str) -> list[str]:
    """Split one identifier into lower-cased sub-words.

    >>> split_identifier("getHTTPResponseCode")
    ['get', 'http', 'response', 'code']
    >>> split_identifier("parse_url_v2")
    ['parse', 'url', 'v', '2']
    """
    parts: list[str] = []
    for piece in identifier.split("_"):
        if not piece:
            continue
        parts.extend(m.group(0).lower() for m in _CAMEL_RE.finditer(piece))
    return parts


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Tokenise code or prose into identifiers plus their sub-words.

    A compound identifier contributes the full lower-cased form *and* each
    sub-word, so ``read_config`` matches both the query "read config" and the
    query "read_config".
    """
    tokens: list[str] = []
    for match in _WORD_RE.finditer(text):
        word = match.group(0)
        subwords = split_identifier(word)
        full = word.lower().strip("_")
        if len(subwords) > 1 and full:
            tokens.append(full)
        tokens.extend(subwords)
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens
