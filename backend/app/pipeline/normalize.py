"""Drug normalization: raw name -> generic(s) + RxCUI.

Order of resolution:
  1. Local Indian brand dataset (fuzzy match) -> generic composition(s)
  2. RxNav/RxNorm (spell-fix, brand->generic, RxCUI)
  3. Misses stay unresolved here. `agent.web_resolve` may fill them next.

Combination products (FDCs) are split into components so every
component pair gets checked downstream.
"""
from __future__ import annotations

import asyncio
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

# Strength with a unit, e.g. "20mg" / "5 mg". Brand numbers without a unit
# ("dolo 650", "telma 40") are left intact — the Indian index keys need them.
_STRENGTH_RE = re.compile(
    r"\b\d+(\.\d+)?\s*(mg|mcg|ug|µg|g|gm|ml|iu|i\.u\.?|%)\b",
    re.I,
)
_COMBO_RE = re.compile(r"\s*/\s*|\s+\+\s+")


def _bare_name(name: str) -> str:
    s = name.lower().strip()
    s = _STRENGTH_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip(" ,;/-")


def _is_combo_name(s: str) -> bool:
    return bool(_COMBO_RE.search(s or ""))


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


def _generic_hit(token: str, generics: frozenset[str]) -> str | None:
    if not token:
        return None
    if token in generics:
        return token
    gmatch = process.extractOne(token, generics, scorer=fuzz.WRatio)
    if gmatch and gmatch[1] >= GENERIC_FUZZY_THRESHOLD:
        return gmatch[0]
    return None


def _brand_fuzzy(key: str, index: dict[str, list[str]]) -> list[str] | None:
    matches = process.extract(key, index.keys(), scorer=fuzz.WRatio, limit=15)
    matches = [m for m in matches if m[1] >= FUZZY_THRESHOLD]
    if not matches:
        return None
    singles = [m for m in matches if len(index[m[0]]) == 1]
    pool = singles or matches
    best = min(pool, key=lambda m: (len(index[m[0]]), -m[1], _form_penalty(m[0], key)))
    return index[best[0]]


def _lookup_indian(name: str) -> list[str] | None:
    """Brand -> composition(s). Exact, word-bounded prefix, known generic,
    then fuzzy. A plain generic must not resolve to an FDC; a bare brand
    prefers the oral solid over drops/syrup.

    Strength-with-unit ('amlodipine 20mg') is stripped before generic/fuzzy
    so a dose does not match an FDC partner's mg or kill RxNav later.
    Unitless brand numbers ('dolo 650') stay on the full string.
    """
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
    # typed name is already a generic (e.g. 'amlodipine') — do not fuzzy an FDC.
    # Also try the dose-stripped form so 'amlodipine 20mg' stays a singleton.
    generics = _known_generics()
    bare = _bare_name(key)
    for token in (key, bare):
        hit = _generic_hit(token, generics)
        if hit:
            return [hit]
    brand = _brand_fuzzy(key, index)
    if brand:
        return brand
    if bare != key:
        return _brand_fuzzy(bare, index)
    return None


# ---------------------------------------------------------------------------
# RxNav helpers
# ---------------------------------------------------------------------------
async def _rxnav_get(client: httpx.AsyncClient, path: str, **params) -> dict:
    r = await client.get(f"{RXNAV_BASE}{path}", params=params,
                         timeout=get_settings().http_timeout)
    r.raise_for_status()
    return r.json()


async def _rxnorm_name(client: httpx.AsyncClient, rxcui: str) -> str | None:
    props = await _rxnav_get(client, f"/rxcui/{rxcui}/property.json", propName="RxNorm Name")
    concepts = props.get("propConceptGroup", {}).get("propConcept") or []
    if not concepts:
        return None
    value = concepts[0].get("propValue")
    return value or None


def _pick_rxnav_candidate(
    query: str, rows: list[tuple[str, str, float]],
) -> tuple[str, str] | None:
    """Prefer a named non-combo ingredient when the query is a single drug.

    `rows` are (rxcui, rxnorm_name, score). Nameless hits are skipped — some
    approximateTerm RxCUIs have no RxNorm Name and used to fail the whole
    resolve (e.g. 'warfar 100 mg').
    """
    named = [(cui, n, s) for cui, n, s in rows if cui and n]
    if not named:
        return None
    if not _is_combo_name(query):
        singles = [r for r in named if not _is_combo_name(r[1])]
        pool = singles or named
    else:
        pool = named
    best = max(pool, key=lambda r: (r[2], -len(r[1])))
    return best[0], best[1]


async def _rxcuis_for_components(
    client: httpx.AsyncClient, names: list[str],
) -> dict[str, str]:
    """One RxCUI per ingredient. Never copy the first component's id onto the rest."""
    if not names:
        return {}
    resolved = await asyncio.gather(*[_rxnav_resolve(client, n) for n in names])
    out: dict[str, str] = {}
    for name, (cui, _) in zip(names, resolved):
        if cui:
            out[name] = cui
    return out


async def _rxnav_resolve(client: httpx.AsyncClient, name: str) -> tuple[str | None, str | None]:
    """name -> (rxcui, generic_name) using approximate matching (handles typos).

    Dose-stripped form is tried first so 'amlodipine 20mg' is not matched to
    a 20 mg combo partner. Several candidates are scored; combos lose unless
    the query itself looks like a combination.
    """
    terms: list[str] = []
    for t in (_bare_name(name), name.strip()):
        if t and t.lower() not in {x.lower() for x in terms}:
            terms.append(t)
    try:
        for term in terms:
            approx = await _rxnav_get(
                client, "/approximateTerm.json", term=term, maxEntries=5)
            raw = approx.get("approximateGroup", {}).get("candidate") or []
            ordered: list[tuple[str, float]] = []
            seen: set[str] = set()
            for c in raw:
                rxcui = c.get("rxcui")
                if not rxcui or rxcui in seen:
                    continue
                seen.add(rxcui)
                try:
                    score = float(c.get("score") or 0)
                except (TypeError, ValueError):
                    score = 0.0
                ordered.append((rxcui, score))
            names = await asyncio.gather(
                *[_rxnorm_name(client, cui) for cui, _ in ordered],
                return_exceptions=True,
            )
            rows: list[tuple[str, str, float]] = []
            for (rxcui, score), generic in zip(ordered, names):
                if isinstance(generic, Exception):
                    generic = None
                rows.append((rxcui, generic or "", score))
            picked = _pick_rxnav_candidate(term, rows)
            if picked:
                return picked
        return None, None
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
        p = re.sub(r"\b(tablet|tablets|capsule|capsules|injection|syrup|cream|ointment|drops?|solution|suspension|oral)\b", " ", p)
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
                cuis = await _rxcuis_for_components(client, flat)
                rxcui = next((cuis[c] for c in flat if c in cuis), None)
                normalized.append(NormalizedDrug(
                    input_name=raw, generic_name=", ".join(flat),
                    rxcui=rxcui, components=flat, component_rxcuis=cuis,
                    resolved_via="indian_dataset"))
                continue
            # 2) RxNav
            rxcui, generic = await _rxnav_resolve(client, raw)
            if generic:
                comps = split_components(generic)
                if len(comps) <= 1:
                    cuis = {comps[0]: rxcui} if comps and rxcui else {}
                else:
                    cuis = await _rxcuis_for_components(client, comps)
                    rxcui = next((cuis[c] for c in comps if c in cuis), rxcui)
                normalized.append(NormalizedDrug(
                    input_name=raw, generic_name=generic.lower(),
                    rxcui=rxcui, components=comps, component_rxcuis=cuis,
                    resolved_via="rxnav"))
                continue
            # 3) miss — service may web-verify via agent.web_resolve
            unresolved.append(raw)
            normalized.append(NormalizedDrug(input_name=raw, resolved_via=None))

    return normalized, unresolved
