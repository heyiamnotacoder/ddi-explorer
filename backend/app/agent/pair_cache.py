"""Process-local cache of waterfall pair results.

Keyed by unordered component names. Callers only store results graded
WITHOUT patient context, so no field can carry patient text. Error-tier
results are not cached so a later retry can hunt again.
"""
from __future__ import annotations

from collections import OrderedDict
from threading import Lock

from ..models import PairResult, SourceTier

MAX_ENTRIES = 256

_lock = Lock()
_store: OrderedDict[tuple[str, str], PairResult] = OrderedDict()


def key(drug_a: str, drug_b: str) -> tuple[str, str]:
    return tuple(sorted((drug_a.strip().lower(), drug_b.strip().lower())))


def get(drug_a: str, drug_b: str) -> PairResult | None:
    k = key(drug_a, drug_b)
    with _lock:
        hit = _store.get(k)
        if hit is None:
            return None
        _store.move_to_end(k)
        return hit.model_copy(deep=True)


def put(result: PairResult) -> None:
    if result.source_tier == SourceTier.ERROR:
        return
    a, b = result.drugs
    if not a or not b:
        return
    # Belt and braces: the caller already refuses to store a context-graded
    # result, so this field must be empty by construction.
    stored = result.model_copy(
        deep=True,
        update={"patient_specific_note": None},
    )
    k = key(a, b)
    with _lock:
        _store[k] = stored
        _store.move_to_end(k)
        while len(_store) > MAX_ENTRIES:
            _store.popitem(last=False)


def clear() -> None:
    with _lock:
        _store.clear()


def entries() -> list[PairResult]:
    """Snapshot of what is cached. For tests and privacy assertions."""
    with _lock:
        return [v.model_copy(deep=True) for v in _store.values()]
