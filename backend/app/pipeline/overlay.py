"""Deterministic overlay for Grade A pairs that skipped the synthesizer.

RxNav (`local`) and openFDA partner-mention hits skip the synthesizer.
This module attaches timing-separable category, missing-dose copy, and
patient/severe-if notes from already scrubbed extract fields — never from
raw request text, never via LLM.
"""
from __future__ import annotations

import re

from ..models import (
    Category,
    Grade,
    NormalizedDrug,
    PairResult,
    SourceTier,
    parent_drug,
)
from ..textmatch import has_word

# Absorption / chelation: separate administration instead of swapping.
_SEPARABLE = (
    "levothyroxine", "liothyronine",
    "ciprofloxacin", "levofloxacin", "moxifloxacin", "ofloxacin", "norfloxacin",
    "gemifloxacin",
    "tetracycline", "doxycycline", "minocycline",
    "alendronate", "risedronate", "ibandronate",
    "raltegravir", "dolutegravir", "elvitegravir", "bictegravir",
)
_BINDERS = (
    "calcium", "iron", "ferrous", "ferric", "magnesium", "aluminum", "aluminium",
    "zinc", "sucralfate", "sevelamer", "lanthanum",
    "cholestyramine", "colestyramine", "colesevelam", "colestipol",
    "antacid",
)
_TIMING_KEYS = (
    "separate", "stagger", "hours before", "hours after", "chelat",
    "absorption of",
)

# (side A tokens, side B tokens, victim name, threshold copy)
_DOSE_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...], str, str], ...] = (
    (("simvastatin",), ("amlodipine",), "simvastatin", "20 mg"),
    (("simvastatin",), ("diltiazem", "verapamil"), "simvastatin", "10 mg"),
    (("simvastatin",), ("amiodarone",), "simvastatin", "20 mg"),
    (("lovastatin",), ("amlodipine", "diltiazem", "verapamil", "amiodarone"),
     "lovastatin", "20 mg"),
    (("colchicine",),
     ("clarithromycin", "erythromycin", "cyclosporine", "ciclosporin",
      "ketoconazole"),
     "colchicine", "the labeled maximum"),
)

_DOSE_NUM = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mg|mcg|ug|µg|g|iu)\b", re.I,
)
_THRESHOLD = re.compile(
    r"(?:doses?|dose)\s*(?:greater than|exceeding|above|of more than|>)\s*"
    r"(\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|iu))",
    re.I,
)
_MAX_DOSE = re.compile(
    r"(?:should not exceed|not exceed|maximum(?: dose)?(?: of)?|limit(?:ed)? to)\s*"
    r"(\d+(?:\.\d+)?\s*(?:mg|mcg|µg|g|iu))",
    re.I,
)
_AGE = re.compile(
    r"\b(?:age|aged)\s*[:\s]*(\d{1,3})\b|\b(\d{1,3})\s*(?:y/o|yo|years?\s*old)\b",
    re.I,
)


def apply(
    pairs: list[PairResult],
    *,
    drugs: list[NormalizedDrug],
    patient_ctx: str | None,
) -> list[PairResult]:
    """Fill overlay fields on local/openFDA Grade A pairs. Other pairs pass through."""
    out: list[PairResult] = []
    for p in pairs:
        if p.grade == Grade.A and p.source_tier in (
                SourceTier.LOCAL, SourceTier.OPENFDA):
            out.append(_overlay_one(p, drugs, patient_ctx))
        else:
            out.append(p)
    return out


def _overlay_one(
    p: PairResult,
    drugs: list[NormalizedDrug],
    patient_ctx: str | None,
) -> PairResult:
    updates: dict = {}
    if p.category != Category.CONTRAINDICATED and _is_timing_separable(p):
        updates["category"] = Category.TIMING
    dose_c = _dose_condition(p, drugs)
    if dose_c:
        updates["dose_condition"] = dose_c
    ctx = (patient_ctx or "").strip()
    if ctx:
        updates["patient_specific_note"] = _patient_note(p, ctx)
        updates["severe_if"] = []
    else:
        updates["severe_if"] = _severe_if(p)
    return p.model_copy(update=updates)


def _is_timing_separable(p: PairResult) -> bool:
    a, b = p.drugs[0].lower(), p.drugs[1].lower()
    a_sep, b_sep = has_word(a, _SEPARABLE), has_word(b, _SEPARABLE)
    a_bind, b_bind = has_word(a, _BINDERS), has_word(b, _BINDERS)
    if (a_sep and b_bind) or (b_sep and a_bind):
        return True
    blob = (p.summary or "").lower()
    if not any(k in blob for k in _TIMING_KEYS):
        return False
    return a_sep or b_sep or a_bind or b_bind


def _has_numeric_dose(dose: str | None) -> bool:
    return bool(dose and _DOSE_NUM.search(dose))


def _condition_copy(victim: str, threshold: str) -> str:
    if threshold.startswith("the "):
        return f"DDI possible if {victim} dose exceeds {threshold}."
    return f"DDI possible if {victim} dose > {threshold}."


def _pair_hits(a: str, b: str, left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return (
        (has_word(a, left) and has_word(b, right))
        or (has_word(b, left) and has_word(a, right))
    )


def _without_numeric_dose(
    p: PairResult, drugs: list[NormalizedDrug],
) -> list[str]:
    """Pair members whose input product carried no numeric dose."""
    out = []
    for name in p.drugs:
        parent = parent_drug(drugs, name)
        if not (parent and _has_numeric_dose(parent.dose)):
            out.append(name)
    return out


def _dose_condition(p: PairResult, drugs: list[NormalizedDrug]) -> str | None:
    a, b = p.drugs[0], p.drugs[1]
    for left, right, victim, thresh in _DOSE_RULES:
        if _pair_hits(a.lower(), b.lower(), left, right):
            parent = parent_drug(drugs, victim)
            if parent and _has_numeric_dose(parent.dose):
                return None
            return _condition_copy(victim, thresh)
    blob = p.summary or ""
    m = _THRESHOLD.search(blob) or _MAX_DOSE.search(blob)
    if m:
        missing = _without_numeric_dose(p, drugs)
        if missing:
            return _condition_copy(missing[0], m.group(1))
        return None
    if "high dose" in blob.lower() or "dose-dependent" in blob.lower():
        missing = _without_numeric_dose(p, drugs)
        if missing:
            return (
                f"DDI possible if {missing[0]} dose is high; "
                "confirm the labeled maximum for this pair."
            )
    return None


def _facts(ctx: str) -> list[str]:
    facts: list[str] = []
    age = _AGE.search(ctx)
    if age:
        facts.append(f"age {age.group(1) or age.group(2)}")
    low = ctx.lower()
    if re.search(r"\b(ckd|crcl|egfr|creatinine|dialysis|renal)\b", low):
        facts.append("reduced renal function")
    if re.search(r"\b(hepatic|cirrhos|child-pugh|liver)\b", low):
        facts.append("hepatic impairment")
    if re.search(r"\bpregnan", low):
        facts.append("pregnancy")
    if re.search(r"\b(elderly|geriatric)\b", low):
        facts.append("elderly")
    return facts


def _patient_note(p: PairResult, ctx: str) -> str:
    facts = _facts(ctx)
    fact_s = ", ".join(facts) if facts else "clinical notes on file"
    major = (p.severity or "").lower() == "major"
    note = (
        f"Patient notes mention {fact_s}. This is a known Grade A"
        f"{' major' if major else ''} pair; check labeling for this patient."
    )
    blob = f"{p.summary or ''} {p.severity or ''}".lower()
    if "reduced renal function" in facts and any(
        w in blob for w in ("renal", "kidney", "major")
    ):
        note += " This pair may be more severe in reduced GFR — confirm labeling."
    elif "pregnancy" in facts:
        note += " Check labeling for pregnancy."
    return note


def _severe_if(p: PairResult) -> list[str]:
    blob = f"{p.summary or ''} {p.severity or ''}".lower()
    items: list[str] = []
    if any(w in blob for w in ("renal", "kidney", "egfr", "crcl", "gfr")):
        items.append("severe if eGFR is reduced")
    if any(w in blob for w in ("hepatic", "liver", "cirrhos")):
        items.append("severe if hepatic impairment")
    if any(w in blob for w in ("elderly", "geriatric")):
        items.append("severe if age > 65")
    if any(w in blob for w in ("pregnan", "fetal", "teratogen")):
        items.append("severe if pregnant")
    if any(w in blob for w in ("high dose", "dose-dependent")):
        items.append("severe if high dose")
    if not items:
        if (p.severity or "").lower() == "major":
            items = [
                "severe if elderly",
                "severe if renal or hepatic impairment",
                "severe if high dose",
            ]
        else:
            items = [
                "severe if high dose",
                "severe if renal or hepatic impairment",
            ]
    return items
