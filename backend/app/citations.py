"""Mapping LLM-named identifiers back to records the tools actually retrieved.

Shared by the agent waterfall and the deterministic avoid-with lookup, so the
`pipeline/` package never has to import the LLM waterfall to reuse it.
"""
from __future__ import annotations

from .models import Citation

_ID_FIELDS = ("pmid", "nct_id", "setid", "url")


def as_list(cited) -> list:
    if cited is None:
        return []
    if isinstance(cited, str):
        return [cited] if cited.strip() else []
    if isinstance(cited, (list, tuple, set)):
        return list(cited)
    return [cited]


def _ids_of(rec: dict) -> set[str]:
    return {str(rec.get(k, "")).strip().lower() for k in _ID_FIELDS} - {""}


def record_ids(pool: list[dict]) -> set[str]:
    ids: set[str] = set()
    for rec in pool:
        ids |= _ids_of(rec)
    return ids


def all_cited_mapped(cited, pool: list[dict]) -> bool:
    """False if the synthesizer named any identifier the tools did not retrieve."""
    ids = record_ids(pool)
    for c in as_list(cited):
        s = str(c).strip().lower()
        if s and s not in ids:
            return False
    return True


def citations_from(cited, pool: list[dict], source: str) -> list[Citation]:
    """Map cited identifiers back to REAL retrieved records only."""
    out: list[Citation] = []
    cited_norm = {str(c).strip().lower() for c in as_list(cited)} - {""}
    for rec in pool:
        if _ids_of(rec) & cited_norm:
            out.append(Citation(
                source=source,
                title=rec.get("title", f"{source} record"),
                url=rec.get("url"),
                identifier=rec.get("pmid") or rec.get("nct_id") or rec.get("setid"),
            ))
    return out
