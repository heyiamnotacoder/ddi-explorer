"""Drug normalization: raw name -> generic(s) + RxCUI.

Order of resolution:
  1. Local Indian brand dataset (fuzzy match) -> generic composition(s)
  2. RxNav/RxNorm (spell-fix, brand->generic, RxCUI)
  3. (misses returned as `unresolved` — the agent may web-verify them)

Combination products (FDCs) are split into components so every
component pair gets checked downstream.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import httpx
from rapidfuzz import fuzz, process

from ..config import get_settings
from ..models import NormalizedDrug

RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"
DATA_PATH = Path(__file__).parent.parent / "data" / "indian_drugs.json"

FUZZY_THRESHOLD = 88  # rapidfuzz score (0-100); brand names are noisy


# ---------------------------------------------------------------------------
# Indian brand dataset (downloaded by scripts/fetch_indian_dataset.py)
# Format: {"augmentin 625 duo tablet": ["amoxycillin", "clavulanic acid"], ...}
# ---------------------------------------------------------------------------
@lru_cache
def _indian_index() -> dict[str, list[str]]:
    if not DATA_PATH.exists():
        return {}
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def _lookup_indian(name: str) -> list[str] | None:
    """Brand -> composition(s). Prefers exact, then prefix, then fuzzy;
    among fuzzy candidates prefers the fewest components (a plain brand
    should not resolve to a combination product)."""
    index = _indian_index()
    if not index:
        return None
    key = name.lower().strip()
    if key in index:
        return index[key]
    # prefix match: 'telma 40' -> 'telma 40 tablet'
    prefix_hits = [k for k in index if k.startswith(key + " ") or k.startswith(key)]
    if prefix_hits:
        best = min(prefix_hits, key=lambda k: (len(index[k]), len(k)))
        return index[best]
    # fuzzy: take top candidates, prefer fewest components
    matches = process.extract(key, index.keys(), scorer=fuzz.WRatio, limit=5)
    matches = [m for m in matches if m[1] >= FUZZY_THRESHOLD]
    if matches:
        best = min(matches, key=lambda m: (len(index[m[0]]), -m[1]))
        return index[best[0]]
    return None


# ---------------------------------------------------------------------------
# RxNav helpers
# ---------------------------------------------------------------------------
async def _rxnav_get(client: httpx.AsyncClient, path: str, **params) -> dict:
    r = await client.get(f"{RXNAV_BASE}{path}", params=params,
                         timeout=get_settings().http_timeout)
    r.raise_for_status()
    return r.json()


async def _rxnav_resolve(client: httpx.AsyncClient, name: str) -> tuple[str | None, str | None]:
    """name -> (rxcui, generic_name) using approximate matching (handles typos)."""
    try:
        approx = await _rxnav_get(client, "/approximateTerm.json", term=name, maxEntries=1)
        candidates = approx.get("approximateGroup", {}).get("candidate", [])
        if not candidates:
            return None, None
        rxcui = candidates[0].get("rxcui")
        if not rxcui:
            return None, None
        props = await _rxnav_get(client, f"/rxcui/{rxcui}/property.json", propName="RxNorm Name")
        generic = props.get("propConceptGroup", {}).get("propConcept", [{}])[0].get("propValue")
        return rxcui, generic
    except (httpx.HTTPError, KeyError, IndexError):
        return None, None


def split_components(generic_string: str) -> list[str]:
    """'amoxycillin / clavulanic acid' or 'telmisartan + amlodipine' -> list."""
    parts = re.split(r"\s*(?:/|\+|\band\b|,)\s*", generic_string.lower())
    cleaned = []
    for p in parts:
        # strip parenthesized strengths: '(500mg)' / '(125 mg)'
        p = re.sub(r"\([^)]*\d[^)]*\)", "", p)
        # strip bare strengths: 'amoxycillin 500 mg' -> 'amoxycillin'
        p = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|iu|%)\b.*", "", p).strip()
        p = re.sub(r"\b(tablet|capsule|injection|syrup|cream|ointment|drops?)\b", "", p).strip()
        p = p.strip("() -").strip()
        if p:
            cleaned.append(p)
    return cleaned or [generic_string.lower()]


async def normalize_drugs(names: list[str]) -> tuple[list[NormalizedDrug], list[str]]:
    """Resolve each raw name. Returns (normalized, unresolved_names)."""
    settings = get_settings()
    names = [n.strip() for n in names if n.strip()][: settings.max_drugs_per_request]
    normalized: list[NormalizedDrug] = []
    unresolved: list[str] = []

    async with httpx.AsyncClient() as client:
        for raw in names:
            # 1) Indian brand dataset
            comps = _lookup_indian(raw)
            if comps:
                flat = [c for comp in comps for c in split_components(comp)]
                # try to grab an RxCUI for the first component for pre-filter use
                rxcui, _ = await _rxnav_resolve(client, flat[0])
                normalized.append(NormalizedDrug(
                    input_name=raw, generic_name=", ".join(flat),
                    rxcui=rxcui, components=flat, resolved_via="indian_dataset"))
                continue
            # 2) RxNav
            rxcui, generic = await _rxnav_resolve(client, raw)
            if generic:
                comps = split_components(generic)
                normalized.append(NormalizedDrug(
                    input_name=raw, generic_name=generic.lower(),
                    rxcui=rxcui, components=comps, resolved_via="rxnav"))
                continue
            # 3) unresolved -> agent web verification downstream
            unresolved.append(raw)
            normalized.append(NormalizedDrug(input_name=raw, resolved_via=None))

    return normalized, unresolved
