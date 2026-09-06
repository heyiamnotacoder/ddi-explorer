"""Non-drug avoid-with lookup from openFDA labels.

Alcohol, grapefruit, tobacco, and named herbals are never pair-checked.
Each is searched in listed drugs' interaction fields. Hits cite retrieved
records only. No record → honest empty copy. No LLM.
"""
from __future__ import annotations

import asyncio
import re

from ..agent import tools
from ..citations import citations_from
from ..config import get_settings
from ..models import AvoidWithItem, NormalizedDrug
from ..textmatch import contains_any_term

EMPTY_NOTE = (
    "No labeled interaction found between {substance} and the listed medications."
)

_CANON_TERMS: dict[str, tuple[str, ...]] = {
    "alcohol": ("alcohol", "ethanol"),
    "grapefruit": ("grapefruit",),
    "tobacco": ("tobacco", "smoking", "cigarette"),
}
_ALIASES: dict[str, str] = {
    "alcohol": "alcohol",
    "ethanol": "alcohol",
    "wine": "alcohol",
    "beer": "alcohol",
    "drinking": "alcohol",
    "grapefruit": "grapefruit",
    "grapefruit juice": "grapefruit",
    "tobacco": "tobacco",
    "smoking": "tobacco",
    "cigarette": "tobacco",
    "cigarettes": "tobacco",
}


def _canon(substance: str) -> str:
    key = " ".join(substance.lower().split())
    return _ALIASES.get(key, key)


def _terms(substance: str) -> tuple[str, ...]:
    canon = _canon(substance)
    if canon in _CANON_TERMS:
        return _CANON_TERMS[canon]
    token = " ".join(substance.split())
    return (token,) if token else ()


def _mentions(text: str, terms: tuple[str, ...]) -> bool:
    return contains_any_term(text, terms)


def _snippet(text: str, terms: tuple[str, ...], limit: int = 400) -> str | None:
    blob = " ".join((text or "").split())
    if not _mentions(blob, terms):
        return None
    parts = re.split(r"(?<=[.!?])\s+", blob)
    hits = [s for s in parts if _mentions(s, terms)]
    if not hits:
        return blob[:limit]
    out = " ".join(hits[:2])
    return out[:limit]


def _clean_substances(raw) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in raw or []:
        if not isinstance(item, str):
            continue
        s = " ".join(item.split())
        if not s:
            continue
        key = _canon(s)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _drug_names(drugs: list[NormalizedDrug], extra_names: list[str]) -> list[str]:
    seen: set[str] = set()
    names: list[str] = []
    for d in drugs:
        for c in d.components or []:
            k = c.lower().strip()
            if k and k not in seen:
                seen.add(k)
                names.append(c)
    for raw in extra_names:
        if not isinstance(raw, str):
            continue
        s = raw.strip()
        k = s.lower()
        if s and k not in seen:
            seen.add(k)
            names.append(s)
    return names


def _empty(substance: str) -> AvoidWithItem:
    return AvoidWithItem(substance=substance, note=EMPTY_NOTE.format(substance=substance))


async def lookup(
    substances,
    *,
    drugs: list[NormalizedDrug],
    extra_names: list[str] | None = None,
) -> list[AvoidWithItem]:
    """Label hits for each non-drug vs listed medications. Never invents DDIs."""
    cleaned = _clean_substances(substances)
    if not cleaned:
        return []
    names = _drug_names(drugs, extra_names or [])
    if not names:
        return [_empty(s) for s in cleaned]

    sem = asyncio.Semaphore(get_settings().pair_concurrency)

    async def _one(drug: str, substance: str) -> tuple[str, str, list[dict]]:
        terms = _terms(substance)
        async with sem:
            try:
                recs = await tools.openfda_substance_check(drug, list(terms))
            except Exception:  # noqa: BLE001 — one miss must not fail the check
                recs = []
        return drug, substance, recs or []

    rows = await asyncio.gather(*(_one(d, s) for s in cleaned for d in names))
    by_sub: dict[str, list[tuple[str, dict]]] = {s: [] for s in cleaned}
    for drug, substance, recs in rows:
        terms = _terms(substance)
        for rec in recs:
            blob = " ".join((
                rec.get("interactions_text") or "",
                rec.get("food_interactions_text") or "",
            ))
            if not _mentions(blob, terms):
                continue
            by_sub[substance].append((drug, rec))

    out: list[AvoidWithItem] = []
    for substance in cleaned:
        kept_pairs = by_sub.get(substance) or []
        if not kept_pairs:
            out.append(_empty(substance))
            continue
        terms = _terms(substance)
        meds: list[str] = []
        meds_seen: set[str] = set()
        snippets: list[str] = []
        pool: list[dict] = []
        cited_ids: list[str] = []
        for drug, rec in kept_pairs:
            key = drug.lower()
            if key not in meds_seen:
                meds_seen.add(key)
                meds.append(drug)
            blob = " ".join((
                rec.get("interactions_text") or "",
                rec.get("food_interactions_text") or "",
            ))
            snip = _snippet(blob, terms)
            if snip and snip not in snippets:
                snippets.append(snip)
            pool.append(rec)
            ident = rec.get("setid") or rec.get("url")
            if ident:
                cited_ids.append(str(ident))
        cites = citations_from(cited_ids, pool, "openfda")
        if not cites:
            out.append(_empty(substance))
            continue
        note = " ".join(snippets).strip() or (
            f"Labeling lists an interaction between {substance} and "
            f"{', '.join(meds)}."
        )
        out.append(AvoidWithItem(
            substance=substance,
            medications=meds,
            note=note,
            citations=cites,
        ))
    return out
