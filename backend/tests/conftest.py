"""Shared fixtures. Pair cache is process-local — wipe it between tests."""
from __future__ import annotations

import pytest

from app.agent import pair_cache


@pytest.fixture(autouse=True)
def _clear_pair_cache():
    pair_cache.clear()
    yield
    pair_cache.clear()
