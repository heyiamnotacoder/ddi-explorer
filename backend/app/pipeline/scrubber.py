"""PHI scrubber — runs AFTER OCR, BEFORE anything reaches an LLM.

Uses Presidio (spaCy NER + regex recognizers). Intake uses scrub_text();
reasoning completions must go through for_reasoning_llm() (also applied
inside agent.llm.complete). Clinical numbers stay; identifiers go.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Regex recognizers (deterministic, no model needed) — Indian + international
# ---------------------------------------------------------------------------
REGEX_PATTERNS: dict[str, str] = {
    "PHONE": r"(?<!\d)(?:\+?\d{1,3}[-.\s]?)?(?:\d{10}|\d{5}[-.\s]\d{5})(?!\d)",
    "EMAIL": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    # Indian ABHA / ABHA address
    "ABHA_NUMBER": r"(?<!\d)\d{2}-?\d{4}-?\d{4}-?\d{4}(?!\d)",
    "ABHA_ADDRESS": r"[a-zA-Z0-9._-]+@abdm\b",
    # Indian Aadhaar
    "AADHAAR": r"(?<!\d)\d{4}\s\d{4}\s\d{4}(?!\d)",
    # Indian PAN
    "PAN": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    # Generic MRN / hospital numbers
    "MRN": r"\b(?:MRN|UHID|IP\s?No|OP\s?No|Reg(?:istration)?\s?No|Patient\s?ID)\s*[:#]?\s*[A-Z0-9/-]{4,}\b",
    # Dates of birth (labelled only, so lab dates like 'Hb 12/05 report' survive)
    "DOB": r"\b(?:DOB|D\.O\.B|Date of Birth)\s*[:/-]?\s*\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",
}

_COMPILED = {k: re.compile(v, re.IGNORECASE) for k, v in REGEX_PATTERNS.items()}

# Entities Presidio NER should catch (spaCy-backed)
NER_ENTITIES = ["PERSON", "LOCATION", "GPE", "ORG", "DATE_TIME", "NRP", "PHONE_NUMBER", "EMAIL_ADDRESS"]

# DATE_TIME is risky in prescriptions (dosing "1-0-1", lab dates) — we keep
# NER DATE_TIME detections ONLY if they look like absolute calendar dates.
_ABS_DATE = re.compile(r"\b\d{1,2}[-/ ](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{1,2})[-/ ]\d{2,4}\b", re.I)


@dataclass
class ScrubResult:
    text: str
    redactions: list[dict] = field(default_factory=list)  # {type, placeholder, start, end}


class _PresidioBackend:
    """Lazy singleton so tests / OCR-only paths don't pay spaCy load cost."""

    _analyzer = None
    _anonymizer = None

    @classmethod
    def get(cls):
        if cls._analyzer is None:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            cls._analyzer = AnalyzerEngine()
            cls._anonymizer = AnonymizerEngine()
        return cls._analyzer, cls._anonymizer


def _regex_scan(text: str) -> list[tuple[int, int, str]]:
    hits = []
    for label, rx in _COMPILED.items():
        for m in rx.finditer(text):
            hits.append((m.start(), m.end(), label))
    return hits


_FORM_PREFIX = re.compile(
    r"\b(?:tab(?:let)?|cap(?:sule)?|inj(?:ection)?|syp|syrup|drops?|cream|oint(?:ment)?|gel|sachet)\b\.?\s*$",
    re.IGNORECASE)


def _looks_like_drug(text: str, start: int, end: int) -> bool:
    """Guard: NER sometimes flags brand names ('Telma', 'Dolo') as PERSON.
    A span is treated as a drug if (a) it matches the Indian brand dataset,
    or (b) it's immediately preceded by a dosage-form word (Tab/Cap/Inj...)."""
    span = text[start:end].strip().lower()
    if not span:
        return False
    # NER spans often swallow the dosage-form word: 'Tab Telma' -> check both
    span_core = re.sub(
        r"^(?:tab(?:let)?|cap(?:sule)?|inj(?:ection)?|syp|syrup|drops?|cream|oint(?:ment)?|gel|sachet)\.?\s+",
        "", span)
    candidates = {span, span_core}
    # (a) dataset match on any candidate
    try:
        from .normalize import indian_index  # lazy; avoids import cycle
        index = indian_index()
        if index:
            for cand in candidates:
                if not cand:
                    continue
                if any(cand in key or key in cand for key in
                       (k for k in index if abs(len(k) - len(cand)) <= 12)):
                    return True
    except Exception:  # noqa: BLE001 — guard must never break scrubbing
        pass
    # (b) dosage-form prefix: 'Tab Telma' -> Telma is a drug
    prefix = text[max(0, start - 20):start]
    return bool(_FORM_PREFIX.search(prefix.strip()))


def _presidio_scan(text: str) -> list[tuple[int, int, str]]:
    try:
        analyzer, _ = _PresidioBackend.get()
    except Exception:
        return []  # Presidio unavailable -> regex-only mode
    results = analyzer.analyze(text=text, entities=NER_ENTITIES, language="en")
    spans = []
    for r in results:
        if _looks_like_drug(text, r.start, r.end):
            continue  # never redact medications
        if r.entity_type == "DATE_TIME" and not _ABS_DATE.search(text[r.start:r.end]):
            continue  # keep relative dosing info, drop only absolute dates
        spans.append((r.start, r.end, r.entity_type))
    return spans


def for_reasoning_llm(text: str | None, *, use_ner: bool = True) -> str | None:
    """The only text a reasoning LLM may see. None/blank stays None."""
    if text is None:
        return None
    cleaned = scrub_text(str(text), use_ner=use_ner).text
    return cleaned if cleaned.strip() else None


def scrub_text(text: str, use_ner: bool = True) -> ScrubResult:
    """Remove PHI. Returns scrubbed text + redaction audit trail."""
    if not text:
        return ScrubResult(text="")

    spans = _regex_scan(text)
    if use_ner:
        spans += _presidio_scan(text)

    # De-overlap: sort by start, keep longer span on conflict
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    merged: list[tuple[int, int, str]] = []
    for s in spans:
        if merged and s[0] < merged[-1][1]:
            prev = merged[-1]
            if s[1] > prev[1]:
                merged[-1] = (prev[0], s[1], prev[2])
            continue
        merged.append(s)

    out, redactions, cursor = [], [], 0
    counters: dict[str, int] = {}
    for start, end, label in merged:
        out.append(text[cursor:start])
        counters[label] = counters.get(label, 0) + 1
        placeholder = f"[{label}_{counters[label]}]"
        out.append(placeholder)
        redactions.append({"type": label, "placeholder": placeholder, "start": start, "end": end})
        cursor = end
    out.append(text[cursor:])

    return ScrubResult(text="".join(out), redactions=redactions)
