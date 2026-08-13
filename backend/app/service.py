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

from .agent import extract, waterfall
from .models import Category, CheckRequest, CheckResponse
from .pipeline import normalize as norm
from .pipeline import ocr, prefilter, scrubber

DISCLAIMER = (
    "DDI Explorer is a clinical decision-support aid for healthcare "
    "professionals. Evidence grades reflect source strength (A: approved "
    "labeling; B: human trial literature; C: case reports/weak evidence). "
    "It does not replace clinical judgment or approved prescribing "
    "information. Verify critical decisions against primary sources."
)


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

    # 7) Assemble
    all_pairs = known_results + agent_results
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
