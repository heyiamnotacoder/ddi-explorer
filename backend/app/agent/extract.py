"""Extraction step: scrubbed free text -> structured drug list.

One cheap LLM call. Input is ALREADY de-identified (Presidio ran first),
so this is safe to send to the reasoning model.
"""
from __future__ import annotations

import json

from . import llm

PROMPT = """Extract structured data from this (de-identified) prescription / medication text.

Return STRICT JSON:
{
  "drugs": [{"name": "<as written>", "dose": "<e.g. 500 mg or null>",
             "timing": "<e.g. 1-0-1 / BD / null>"}],
  "non_drugs": ["<alcohol | tobacco | grapefruit | herbal/supplement names>"],
  "patient_context": "<age/sex/weight/renal/hepatic/pregnancy/comorbidity info found, or null>"
}

Rules:
- Include EVERY medication, including combination products.
- "name" = brand/generic name INCLUDING strength if written (e.g. "Telma 40", "Dolo 650") — strength is critical for correct brand resolution. Put the full strength in "dose" too.
- Non-drugs: only clearly non-medication substances (alcohol, tobacco, food items, herbals).
- Do NOT invent drugs not present in the text.
- Ignore redaction placeholders like [PERSON_1], [MRN_1].

TEXT:
"""


async def extract_drugs(scrubbed_text: str) -> dict:
    raw = await llm.complete(
        [{"role": "user", "content": PROMPT + scrubbed_text}],
        response_format={"type": "json_object"}, max_tokens=1500)
    try:
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        data = {}
    data.setdefault("drugs", [])
    data.setdefault("non_drugs", [])
    data.setdefault("patient_context", None)
    return data
