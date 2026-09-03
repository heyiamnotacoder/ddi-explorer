"""Shared fixtures. Pair cache is process-local — wipe it between tests."""
from __future__ import annotations

import pytest

from app import service
from app.agent import pair_cache

# The seams `service.run_check` calls out through, by role. One place to fix
# when the pipeline's call chain changes.
_SEAMS: dict[str, tuple[str, str]] = {
    "extract": ("extract", "extract_drugs"),
    "normalize": ("norm", "normalize_drugs"),
    "prefilter": ("prefilter", "check_known_pairs"),
    "waterfall": ("waterfall", "evaluate_pairs"),
    "avoid": ("avoid_mod", "lookup"),
}


@pytest.fixture(autouse=True)
def _clear_pair_cache():
    pair_cache.clear()
    yield
    pair_cache.clear()


@pytest.fixture
def patch_seams(monkeypatch):
    """Stub the pipeline seams by role: patch_seams(extract=fn, waterfall=fn)."""
    def patch(**fakes) -> None:
        for role, fake in fakes.items():
            module, attr = _SEAMS[role]
            monkeypatch.setattr(getattr(service, module), attr, fake)
    return patch
