"""Orchestrator: /api/check pipeline.

  images -> OCR -> text
  ALL text -> Presidio scrub
  scrubbed text -> LLM extraction (drugs, non-drugs, patient context)
  drugs -> normalize (Indian dataset / RxNav / FDC split)
  known pairs -> local pre-filter (Grade A, zero LLM tokens)
  unknown pairs -> agent waterfall (A -> B -> C -> none, early exit)
  assemble -> contraindication banner, avoid-with-medications, disclaimers
"""
from __future__ import annotations

from .agent import alternatives as alts
from .agent import extract, waterfall
from .models import (
    AlternativeSuggestion,
    AlternativesRequest,
    AlternativesResponse,
    Category,
    CheckRequest,
    CheckResponse,
    Grade,
    PairResult,
    ReplaceableDrug,
)
from .pipeline import normalize as norm
from .pipeline import ocr, prefilter, scrubber

DISCLAIMER = (
    "DDI Explorer is a clinical decision-support aid for healthcare "
    "professionals. Evidence grades reflect source strength (A: approved "
    "labeling; B: human trial literature; C: case reports/weak evidence). "
    "It does not replace clinical judgment or approved prescribing "
    "information. Verify critical decisions against primary sources."
)


def _dedupe_pairs(pairs: list[PairResult]) -> list[PairResult]:
    """Keep one result per unordered component pair; prefer a stronger grade."""
    rank = {Grade.A: 0, Grade.B: 1, Grade.C: 2, None: 3}
    best: dict[tuple[str, str], PairResult] = {}
    order: list[tuple[str, str]] = []
    for p in pairs:
        key = tuple(sorted(p.drugs))
        if key not in best:
            best[key] = p
            order.append(key)
            continue
        if rank.get(p.grade, 3) < rank.get(best[key].grade, 3):
            best[key] = p
    return [best[k] for k in order]


async def run_check(req: CheckRequest) -> CheckResponse:
    raw_texts: list[str] = []

    # 1) OCR images -> raw text (local first, vision fallback)
    for img in req.images:
        result = await ocr.extract_text(img)
        if result["text"].strip():
            raw_texts.append(result["text"])

    if req.text:
        raw_texts.append(req.text)
    if req.patient_context:
        raw_texts.append(f"Patient context: {req.patient_context}")
    if req.timing:
        raw_texts.append(f"Timing: {req.timing}")

    # 2) Presidio scrub — BEFORE any LLM sees anything
    scrubbed = scrubber.scrub_text("\n".join(raw_texts))

    # 3) Structured extraction (first LLM contact; input is clean)
    extracted = await extract.extract_drugs(scrubbed.text)
    drug_names = [d["name"] for d in extracted["drugs"] if d.get("name")]
    patient_ctx = req.patient_context or extracted.get("patient_context")

    # 4) Normalize (local Indian dataset -> RxNav -> unresolved)
    normalized, unresolved = await norm.normalize_drugs(drug_names)

    # 5) Local pre-filter: known pairs resolved without LLM tokens
    prefilter_input = [
        {"generic": n.generic_name, "rxcui": n.rxcui, "components": n.components}
        for n in normalized if n.components
    ]
    known_results, unknown_pairs = await prefilter.check_known_pairs(prefilter_input)

    # 6) Agent waterfall for unknown pairs only
    agent_results = await waterfall.evaluate_pairs(unknown_pairs, patient_ctx)

    # 7) Assemble (dedupe in case RxNav names and waterfall names collide)
    all_pairs = _dedupe_pairs(known_results + agent_results)
    banner = [p for p in all_pairs if p.category == Category.CONTRAINDICATED]
    insufficient = [p.drugs for p in all_pairs if p.source_tier == "insufficient"]

    avoid: list[str] = []
    for substance in extracted.get("non_drugs", []):
        avoid.append(f"{substance}: check against all listed medications "
                     f"(non-drug substance — pair checking skipped)")

    return CheckResponse(
        scrubbed_text=scrubbed.text,
        normalized_drugs=normalized,
        unresolved_drugs=unresolved,
        pairs=all_pairs,
        contraindicated_banner=banner,
        avoid_with_medications=avoid,
        insufficient_evidence=insufficient,
        disclaimer=DISCLAIMER,
    )


ALT_DISCLAIMER = (
    DISCLAIMER + " Substitution suggestions are decision support only. "
    "Confirm indication, dose, comorbidities, and local formulary before "
    "changing therapy. Prefer the lower-stakes (symptomatic) medicine; "
    "do not stop a disease-modifying or high-ADR-risk drug without review."
)


def _pairs_involving(pairs: list[PairResult], names: set[str]) -> list[PairResult]:
    keys = {n.lower() for n in names}
    return [p for p in pairs if any(d.lower() in keys for d in p.drugs)]


async def _recheck_against_rest(
    alt_name: str,
    remaining: list[dict],
    patient_ctx: str | None,
) -> tuple[list[PairResult], list[str], str | None]:
    """Normalize one substitute and grade it only vs leftover components."""
    normalized, unresolved = await norm.normalize_drugs([alt_name])
    if not normalized or not normalized[0].components:
        return [], [], (unresolved[0] if unresolved else alt_name)
    alt = normalized[0]
    new_keys = {c.lower() for c in alt.components}
    known, unknown = await prefilter.check_known_pairs(
        remaining + [{
            "generic": alt.generic_name, "rxcui": alt.rxcui,
            "components": alt.components,
        }]
    )
    known = _pairs_involving(known, new_keys)
    unknown = [
        (a, b) for a, b in unknown
        if a.lower() in new_keys or b.lower() in new_keys
    ]
    agent = await waterfall.evaluate_pairs(unknown, patient_ctx) if unknown else []
    return _dedupe_pairs(known + agent), alt.components, None


async def run_alternatives(req: AlternativesRequest) -> AlternativesResponse:
    """Second loop over an already-graded prescription.

    1. Scrub any patient text again (never send raw notes to the LLM).
    2. Rank which interacting drug is lowest-stakes to change.
    3. LLM proposes same-indication substitutes for those drugs only.
    4. Recheck each substitute against the rest of the list.
    """
    ctx = None
    if req.patient_context:
        ctx = scrubber.scrub_text(req.patient_context).text
    rx_text = None
    if req.scrubbed_text:
        rx_text = scrubber.scrub_text(req.scrubbed_text).text

    candidates = alts.pick_candidates(req.normalized_drugs, req.pairs)
    keep = alts.keep_list(req.normalized_drugs, req.pairs, candidates)
    timing = alts.timing_notes(req.pairs)

    proposed: dict = {}
    if candidates:
        try:
            proposed = await alts.propose(
                req.normalized_drugs, req.pairs, candidates, ctx,
                req.avoid_with_medications, rx_text,
            )
        except Exception:  # noqa: BLE001 — ranking still useful if the LLM fails
            proposed = {}

    allowed = alts.allowed_from_names(candidates)
    change_rows = proposed.get("change") or []
    if allowed:
        filtered = []
        for row in change_rows:
            src = str(row.get("from") or "").strip()
            if src.lower() in allowed:
                filtered.append(row)
        change_rows = filtered

    suggestions: list[AlternativeSuggestion] = []
    remaining_base = [
        {"generic": d.generic_name, "rxcui": d.rxcui, "components": d.components}
        for d in req.normalized_drugs if d.components
    ]

    for row in change_rows:
        src = str(row.get("from") or "").strip()
        if not src:
            continue
        src_l = src.lower()
        product = alts._parent_product(req.normalized_drugs, src)
        old_pairs = _pairs_involving(req.pairs, {src_l})
        remaining = []
        for item in remaining_base:
            comps = [c for c in item["components"] if c.lower() != src_l]
            if comps:
                remaining.append({**item, "components": comps})

        for alt in (row.get("alternatives") or [])[:2]:
            name = str(alt.get("name") or "").strip()
            if not name or name.lower() == src_l:
                continue
            already = {
                c.lower()
                for d in req.normalized_drugs
                for c in d.components
                if c.lower() != src_l
            }
            if name.lower() in already:
                suggestions.append(AlternativeSuggestion(
                    change_from=src, change_from_product=product,
                    change_to=name, indication=row.get("indication") or "",
                    rationale=alt.get("rationale") or "",
                    adr_note=alt.get("adr_note") or "",
                    safer=False,
                    reject_reason="Already on this prescription.",
                ))
                continue
            try:
                new_pairs, comps, unresolved = await _recheck_against_rest(
                    name, remaining, ctx)
            except Exception as e:  # noqa: BLE001
                suggestions.append(AlternativeSuggestion(
                    change_from=src, change_from_product=product,
                    change_to=name, indication=row.get("indication") or "",
                    rationale=alt.get("rationale") or "",
                    adr_note=alt.get("adr_note") or "",
                    safer=False,
                    reject_reason=f"Recheck failed: {type(e).__name__}",
                ))
                continue
            if unresolved:
                suggestions.append(AlternativeSuggestion(
                    change_from=src, change_from_product=product,
                    change_to=name, indication=row.get("indication") or "",
                    rationale=alt.get("rationale") or "",
                    adr_note=alt.get("adr_note") or "",
                    safer=False,
                    reject_reason=f"Could not resolve “{unresolved}” to a generic.",
                ))
                continue
            leftover = alts.leftover_ddis(new_pairs)
            safer = alts.is_safer(old_pairs, leftover)
            reject = None
            if not safer:
                new_a = sum(1 for p in leftover if p.grade == Grade.A)
                old_a = sum(1 for p in old_pairs if p.grade == Grade.A)
                if any(p.category == Category.CONTRAINDICATED for p in leftover):
                    reject = "Introduces a contraindicated pair with the rest of the list."
                elif new_a > 0 and new_a >= old_a:
                    reject = "Still has a Grade A interaction with the rest of the list."
                elif alts.pair_burden(leftover) >= alts.pair_burden(old_pairs):
                    reject = "Does not reduce the interaction burden vs the current pair."
            suggestions.append(AlternativeSuggestion(
                change_from=src,
                change_from_product=product,
                change_to=name,
                change_to_components=comps,
                indication=row.get("indication") or "",
                rationale=alt.get("rationale") or "",
                adr_note=alt.get("adr_note") or "",
                safer=safer,
                reject_reason=reject,
                remaining_ddis=leftover,
            ))

    suggestions.sort(key=lambda s: (not s.safer, s.reject_reason is not None))

    strategy = (proposed.get("strategy") or "").strip() or alts.fallback_strategy(
        candidates, timing, req.pairs)
    if proposed.get("timing_first"):
        extra = [str(t) for t in proposed["timing_first"] if t]
        for t in extra:
            if t not in timing:
                timing.append(t)

    replaceable = [
        ReplaceableDrug(
            name=c.name, input_name=c.input_name, importance=c.importance,
            why_this_one=c.why, involved_pairs=c.involved_pairs,
        )
        for c in candidates
    ]

    return AlternativesResponse(
        strategy=strategy,
        replaceable=replaceable,
        suggestions=suggestions,
        keep=keep,
        timing_first=timing,
        disclaimer=ALT_DISCLAIMER,
    )
