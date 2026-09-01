"""Local pre-filter: resolve KNOWN interactions without spending LLM tokens.

Uses RxNav's interaction endpoint (NLM, free). Pairs resolved here are
Grade A candidates and never enter the agent waterfall.
"""
from __future__ import annotations

import httpx

from ..config import get_settings
from ..models import Category, Citation, Grade, NormalizedDrug, PairResult

RXNAV_INTERACTION = "https://rxnav.nlm.nih.gov/REST/interaction/list.json"


def _severity_rank(desc: str) -> str:
    d = desc.lower()
    if any(w in d for w in ("contraindicat", "life-threat", "serious", "major", "high risk")):
        return "major"
    if any(w in d for w in ("moderate", "monitor", "caution")):
        return "moderate"
    return "minor"


def _is_contraindicated(desc: str) -> bool:
    return "contraindicat" in desc.lower()


def _concept_cui(concept: dict) -> str | None:
    mini = concept.get("minConceptItem") or {}
    cui = mini.get("rxcui") or concept.get("rxcui")
    return str(cui) if cui else None


async def check_known_pairs(
    drugs: list[NormalizedDrug],
) -> tuple[list[PairResult], list[tuple[str, str]]]:
    """Returns (resolved_pair_results, unresolved_component_pairs).

    Only pairs where BOTH sides have RxCUIs can be checked via RxNav;
    everything else flows to the agent waterfall. Each FDC ingredient uses
    its own RxCUI — never a sibling's.
    """
    settings = get_settings()

    # Build component pairs: (comp_a, comp_b, rxcui_a, rxcui_b)
    pairs: list[tuple[str, str, str | None, str | None]] = []
    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            for ca in drugs[i].components:
                for cb in drugs[j].components:
                    if ca.lower() != cb.lower():
                        pairs.append(
                            (ca, cb, drugs[i].rxcui_for(ca), drugs[j].rxcui_for(cb)))

    cui_pairs = [p for p in pairs if p[2] and p[3] and p[2] != p[3]]
    no_cui_pairs = {(a, b) for a, b, ra, rb in pairs if not (ra and rb) or ra == rb}

    resolved: list[PairResult] = []
    resolved_names: set[tuple[str, str]] = set()
    our_by_cuis: dict[tuple[str, str], tuple[str, str]] = {}
    for a, b, ra, rb in cui_pairs:
        our_by_cuis[tuple(sorted((ra, rb)))] = (a, b)

    cuis = sorted({c for p in cui_pairs for c in (p[2], p[3]) if c})
    if cuis:
        async with httpx.AsyncClient() as client:
            try:
                r = await client.get(RXNAV_INTERACTION, params={"rxcuis": " ".join(cuis)},
                                     timeout=settings.http_timeout)
                r.raise_for_status()
                data = r.json()
            except httpx.HTTPError:
                data = {}

        for group in data.get("fullInteractionTypeGroup", []):
            for fit in group.get("fullInteractionType", []):
                for pair in fit.get("interactionPair", []):
                    desc = pair.get("description", "")
                    sev = pair.get("severity", "")
                    concepts = pair.get("interactionConcept") or []
                    hit_cuis = [_concept_cui(c) for c in concepts]
                    hit_cuis = [c for c in hit_cuis if c]
                    if len(hit_cuis) != 2:
                        continue
                    ours = our_by_cuis.get(tuple(sorted(hit_cuis)))
                    if not ours:
                        continue
                    names = tuple(sorted((ours[0].lower(), ours[1].lower())))
                    full_desc = f"{desc} (severity: {sev})" if sev else desc
                    cat = (Category.CONTRAINDICATED if _is_contraindicated(full_desc)
                           else Category.INTERACTION)
                    resolved.append(PairResult(
                        drugs=names,
                        grade=Grade.A,
                        category=cat,
                        severity=_severity_rank(full_desc + " " + sev),
                        summary=desc,
                        citations=[Citation(source="rxnav", title="RxNav Drug Interaction (NLM)",
                                            url="https://lhncbc.nlm.nih.gov/RxNav/APIs")],
                        source_tier="local",
                    ))
                    resolved_names.add(names)

    # Anything RxNav didn't cover goes to the agent waterfall
    unresolved: set[tuple[str, str]] = set()
    for a, b in no_cui_pairs:
        unresolved.add(tuple(sorted((a, b))))
    for a, b, _, _ in cui_pairs:
        key = tuple(sorted((a, b)))
        if key not in resolved_names:
            unresolved.add(key)

    return resolved, sorted(unresolved)
