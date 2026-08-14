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
GENERIC_FUZZY_THRESHOLD = 92  # stricter: short generics must not become FDCs

# Tokens that are units / fillers, never drug components
_JUNK = frozenset({
    "ml", "mg", "mcg", "ug", "µg", "g", "gm", "kg",
    "iu", "i.u", "units", "unit",
    "%", "w/w", "w/v", "v/v",
    "na", "nil", "qs", "q.s", "q.s.",
})
_SOLID_FORMS = ("tablet", "capsule")
_OTHER_FORMS = (
    "drop", "syrup", "suspension", "injection", "infusion",
    "gel", "cream", "ointment", "solution", "liquid",
    "inhaler", "spray", "lotion", "patch",
)


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


def _form_penalty(brand_key: str, query: str) -> int:
    """Unspecified form prefers tablet/capsule over drops/syrup/gel."""
    specified = [f for f in _SOLID_FORMS + _OTHER_FORMS if f in query]
    if specified:
        return 0 if any(f in brand_key for f in specified) else 1
    if any(f in brand_key for f in _SOLID_FORMS):
        return 0
    if any(f in brand_key for f in _OTHER_FORMS):
        return 2
    return 1


def _pick_brand(hits: list[str], index: dict[str, list[str]], query: str) -> str:
    return min(hits, key=lambda k: (len(index[k]), _form_penalty(k, query), len(k)))


@lru_cache
def _known_generics() -> frozenset[str]:
    """Generic names that appear as Indian-dataset components."""
    out: set[str] = set()
    for comps in _indian_index().values():
        for raw in comps:
            for part in split_components(raw):
                if len(part) >= 4:
                    out.add(part)
    return frozenset(out)


def _lookup_indian(name: str) -> list[str] | None:
    """Brand -> composition(s). Exact, word-bounded prefix, known generic,
    then fuzzy. A plain generic must not resolve to an FDC; a bare brand
    prefers the oral solid over drops/syrup."""
    index = _indian_index()
    if not index:
        return None
    key = name.lower().strip()
    if not key:
        return None
    if key in index:
        return index[key]
    # word-bounded prefix only: 'dolo' must not match 'dolonex'
    prefix_hits = [k for k in index if k.startswith(key + " ") or k.startswith(key + "-")]
    if prefix_hits:
        return index[_pick_brand(prefix_hits, index, key)]
    # typed name is already a generic (e.g. 'amlodipine') — do not fuzzy an FDC
    generics = _known_generics()
    if key in generics:
        return [key]
    gmatch = process.extractOne(key, generics, scorer=fuzz.WRatio)
    if gmatch and gmatch[1] >= GENERIC_FUZZY_THRESHOLD:
        return [gmatch[0]]
    matches = process.extract(key, index.keys(), scorer=fuzz.WRatio, limit=15)
    matches = [m for m in matches if m[1] >= FUZZY_THRESHOLD]
    if matches:
        singles = [m for m in matches if len(index[m[0]]) == 1]
        pool = singles or matches
        best = min(pool, key=lambda m: (len(index[m[0]]), -m[1], _form_penalty(m[0], key)))
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
    """'amoxycillin / clavulanic acid' or 'telmisartan + amlodipine' -> list.

    Parenthetical strengths are stripped *before* splitting so that
    'paracetamol (100mg/ml)' does not become ['paracetamol', 'ml'].
    """
    s = generic_string.lower().strip()
    s = re.sub(r"\([^)]*\)", " ", s)
    parts = re.split(r"\s*(?:/|\+|\band\b|,|;)\s*", s)
    cleaned: list[str] = []
    seen: set[str] = set()
    for p in parts:
        p = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|ug|µg|g|gm|ml|iu|i\.u\.?|%)\b", " ", p)
        p = re.sub(r"\b(tablet|tablets|capsule|capsules|injection|syrup|cream|ointment|drops?|solution|suspension)\b", " ", p)
        p = re.sub(r"\s+", " ", p).strip("() -.")
        if not p or p in _JUNK or len(p) < 3:
            continue
        if p not in seen:
            seen.add(p)
            cleaned.append(p)
    return cleaned


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
                flat: list[str] = []
                seen: set[str] = set()
                for comp in comps:
                    for c in split_components(comp):
                        if c not in seen:
                            seen.add(c)
                            flat.append(c)
                if not flat:
                    unresolved.append(raw)
                    normalized.append(NormalizedDrug(input_name=raw, resolved_via=None))
                    continue
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
