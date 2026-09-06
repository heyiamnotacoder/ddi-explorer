"""Web verification for names that miss the Indian dataset and RxNav.

Queries are scrubbed names only (regex PHI strip; NER off so unknown brands
are not redacted as PERSON). Retrieved page text is the only source of
generics — invented ingredients are dropped. This path fills NormalizedDrug;
DDI grades still come from prefilter/waterfall citation mapping.
"""
from __future__ import annotations

import asyncio
import json
import re

import httpx

from ..config import get_settings
from ..models import NormalizedDrug
from ..pipeline import normalize as norm
from ..pipeline import scrubber
from ..textmatch import contains_term
from . import llm, tools

_MAX_FETCH = 2
_PLACEHOLDER = re.compile(r"\[[A-Z_]+_\d+\]")
SYSTEM = """Extract generic ingredient names for one medicine from retrieved web text.

Return JSON: {"generics": ["paracetamol"]}
Rules:
- Each generic MUST appear in the retrieved text. If the pages do not name
  ingredients, return {"generics": []}.
- Split combination products into separate ingredients.
- Ignore people, phone numbers, and anything that is not a drug ingredient.
"""


def _clean_query(raw: str) -> str:
    """Regex-scrub the name. NER stays off so unknown brands survive."""
    text = scrubber.for_reasoning_llm(raw, use_ner=False) or ""
    text = _PLACEHOLDER.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip(" ,;/-")


def _pool_blob(records: list[dict]) -> str:
    parts: list[str] = []
    for rec in records:
        for k in ("title", "snippet", "content"):
            v = rec.get(k)
            if v:
                parts.append(str(v))
    return " ".join(parts).lower()


def _mentioned(name: str, blob: str) -> bool:
    token = " ".join(name.lower().split())
    if len(token) < 3:
        return False
    return contains_term(blob, token)


def _generics_from(parsed: dict, blob: str) -> list[str]:
    raw = parsed.get("generics") if isinstance(parsed, dict) else None
    if not isinstance(raw, list):
        return []
    flat: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        for part in norm.split_components(item):
            if part in seen or not _mentioned(part, blob):
                continue
            seen.add(part)
            flat.append(part)
    return flat


async def resolve_one(raw: str) -> NormalizedDrug | None:
    query = _clean_query(raw)
    if len(query) < 3:
        return None
    hits = await tools.web_search(f"{query} tablet composition generic ingredients")
    pool = await tools.fetch_web_pages(hits, _MAX_FETCH)
    if not pool:
        # Snippets only when a fetch returned nothing — still retrieved text.
        pool = [
            {
                "title": h.get("title") or "search hit",
                "url": h.get("url") or "",
                "snippet": h.get("snippet") or "",
                "content": "",
            }
            for h in hits
            if (h.get("snippet") or h.get("title"))
        ]
    blob = _pool_blob(pool)
    if not pool or not blob.strip():
        return None
    user = (
        f"Brand or name as written: {query}\n"
        f"Retrieved records (JSON):\n{json.dumps(pool, indent=1)[:12000]}"
    )
    parsed = llm.parse_json_object(await llm.complete(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        max_tokens=400,
    ))
    comps = _generics_from(parsed, blob)
    if not comps:
        return None
    if all(norm.bare_name(c) == norm.bare_name(query) for c in comps):
        return None
    async with httpx.AsyncClient() as client:
        cuis = await norm.rxcuis_for_components(client, comps)
    rxcui = next((cuis[c] for c in comps if c in cuis), None)
    return NormalizedDrug(
        input_name=raw,
        generic_name=", ".join(comps),
        rxcui=rxcui,
        components=comps,
        component_rxcuis=cuis,
        resolved_via="agent_web",
    )


async def apply(
    drugs: list[NormalizedDrug],
) -> tuple[list[NormalizedDrug], list[str]]:
    """Fill dataset+RxNav misses via web. Already-resolved drugs pass through."""
    pending_idx = [i for i, d in enumerate(drugs) if not d.components]
    if not pending_idx:
        return drugs, []
    sem = asyncio.Semaphore(get_settings().pair_concurrency)

    async def _one(d: NormalizedDrug) -> NormalizedDrug | None:
        async with sem:
            try:
                return await resolve_one(d.input_name)
            except Exception:  # noqa: BLE001 — one miss must not fail the check
                return None

    found = await asyncio.gather(*(_one(drugs[i]) for i in pending_idx))
    out = list(drugs)
    still: list[str] = []
    for i, resolved in zip(pending_idx, found):
        src = drugs[i]
        if resolved and resolved.components:
            out[i] = resolved.model_copy(update={
                "input_name": src.input_name,
                "dose": src.dose,
                "schedule": src.schedule,
            })
        else:
            still.append(src.input_name)
    return out, still
