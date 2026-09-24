from codesearch.tokenize import split_identifier, tokenize


def test_split_camel_and_acronyms() -> None:
    assert split_identifier("getHTTPResponseCode") == ["get", "http", "response", "code"]
    assert split_identifier("HTTPAdapter") == ["http", "adapter"]
    assert split_identifier("XMLHttpRequest") == ["xml", "http", "request"]


def test_split_snake_digits_and_dunder() -> None:
    assert split_identifier("parse_url_v2") == ["parse", "url", "v", "2"]
    assert split_identifier("__init__") == ["init"]
    assert split_identifier("_") == []


def test_tokenize_keeps_full_identifier_and_subwords() -> None:
    toks = tokenize("def read_config(path): return loadJSON(path)")
    assert "read_config" in toks and "read" in toks and "config" in toks
    assert "loadjson" in toks and "load" in toks and "json" in toks
    # single-part identifiers are not duplicated
    assert toks.count("path") == 2


def test_tokenize_stopwords() -> None:
    assert tokenize("Return the value of a key") == ["return", "value", "key"]
    assert "the" in tokenize("the", drop_stopwords=False)
