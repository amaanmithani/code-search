from __future__ import annotations

import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A writable copy of the small fixture repository."""
    dest = tmp_path / "repo"
    shutil.copytree(FIXTURES / "repo", dest)
    return dest
