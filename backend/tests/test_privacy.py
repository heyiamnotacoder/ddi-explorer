"""Privacy gate: reasoning LLMs never see raw patient identifiers."""
from __future__ import annotations

import json

import pytest

from app import service
from app.agent import llm
from app.models import (
    AlternativesRequest,
    Category,
    CheckRequest,
    Grade,
    NormalizedDrug,
    PairResult,
)
from app.service import DISCLAIMER


PHI_NOTES = (
    "Ramesh Kumar, phone 9876543210, MRN: AIIMS-20451. Age 56, CrCl 42."
)
PHONE = "9876543210"
MRN = "AIIMS-20451"


def _no_phi(text: str | None) -> None:
    blob = text or ""
    assert PHONE not in blob
    assert MRN not in blob
    assert "Ramesh" not in blob


@pytest.mark.asyncio
async def test_check_does_not_send_raw_patient_notes(monkeypatch):
    extract_seen: list[str] = []
    waterfall_seen: list[str | None] = []

    async def fake_extract(text: str) -> dict:
        extract_seen.append(text)
        return {
            "drugs": [{"name": "warfarin 5 mg", "dose": "5 mg", "timing": None}],
            "non_drugs": [],
            "patient_context": "Age 56, CrCl 42",
        }

    async def fake_norm(names: list[str]):
        drugs = [
            NormalizedDrug(
                input_name=n, generic_name="warfarin", rxcui="11289",
                components=["warfarin"], resolved_via="rxnav",
            )
            for n in names
        ]
        return drugs, []

    async def fake_prefilter(_items):
        return [], [("warfarin", "amiodarone")]

    async def fake_waterfall(_pairs, patient_ctx=None):
        waterfall_seen.append(patient_ctx)
        return []

    monkeypatch.setattr(service.extract, "extract_drugs", fake_extract)
    monkeypatch.setattr(service.norm, "normalize_drugs", fake_norm)
    monkeypatch.setattr(service.prefilter, "check_known_pairs", fake_prefilter)
    monkeypatch.setattr(service.waterfall, "evaluate_pairs", fake_waterfall)

    resp = await service.run_check(CheckRequest(
        text="warfarin 5 mg",
        patient_context=PHI_NOTES,
    ))

    assert resp.disclaimer == DISCLAIMER
    assert extract_seen
    _no_phi(extract_seen[0])
    assert "[PHONE_" in extract_seen[0]
    assert "[MRN_" in extract_seen[0]
    _no_phi(resp.scrubbed_text)
    assert waterfall_seen
    _no_phi(waterfall_seen[0])
    assert waterfall_seen[0] and "Age 56" in waterfall_seen[0]
    assert "CrCl 42" in waterfall_seen[0]


@pytest.mark.asyncio
async def test_check_waterfall_never_gets_raw_request_fallback(monkeypatch):
    """Even if extract omits patient_context, use the scrubbed blob — not req."""

    async def fake_extract(text: str) -> dict:
        return {"drugs": [{"name": "warfarin 5 mg"}], "non_drugs": [],
                "patient_context": None}

    async def fake_norm(names: list[str]):
        return ([NormalizedDrug(
            input_name=n, generic_name="warfarin", rxcui="11289",
            components=["warfarin"], resolved_via="rxnav",
        ) for n in names], [])

    async def fake_prefilter(_items):
        return [], [("warfarin", "amiodarone")]

    seen: list[str | None] = []

    async def fake_waterfall(_pairs, patient_ctx=None):
        seen.append(patient_ctx)
        return []

    monkeypatch.setattr(service.extract, "extract_drugs", fake_extract)
    monkeypatch.setattr(service.norm, "normalize_drugs", fake_norm)
    monkeypatch.setattr(service.prefilter, "check_known_pairs", fake_prefilter)
    monkeypatch.setattr(service.waterfall, "evaluate_pairs", fake_waterfall)

    await service.run_check(CheckRequest(
        text="warfarin 5 mg",
        patient_context=PHI_NOTES,
    ))
    assert seen
    _no_phi(seen[0])
    assert seen[0]
    assert "Age 56" in seen[0]


@pytest.mark.asyncio
async def test_alternatives_cannot_be_pointed_at_raw_notes(monkeypatch):
    seen: dict[str, str | None] = {}

    async def fake_propose(_drugs, _pairs, _candidates, patient_context, _avoid,
                           scrubbed_text=None):
        seen["ctx"] = patient_context
        seen["rx"] = scrubbed_text
        return {}

    monkeypatch.setattr(service.alts, "propose", fake_propose)

    drugs = [
        NormalizedDrug(input_name="Dolo 650", generic_name="paracetamol",
                       components=["paracetamol"]),
        NormalizedDrug(input_name="warfarin", generic_name="warfarin",
                       components=["warfarin"]),
    ]
    pairs = [PairResult(
        drugs=("paracetamol", "warfarin"), grade=Grade.A,
        category=Category.INTERACTION, severity="major",
        summary="paracetamol + warfarin",
    )]
    resp = await service.run_alternatives(AlternativesRequest(
        normalized_drugs=drugs,
        pairs=pairs,
        patient_context=PHI_NOTES,
        scrubbed_text="warfarin 5 mg + Dolo 650",
    ))
    assert DISCLAIMER in resp.disclaimer
    assert "ctx" in seen
    _no_phi(seen["ctx"])
    _no_phi(seen["rx"])
    assert seen["ctx"] and "Age 56" in seen["ctx"]


@pytest.mark.asyncio
async def test_complete_scrubs_reasoning_prompts(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(llm, "get_settings", lambda: SimpleNamespace(
        llm_model="anthropic/claude-sonnet-5",
        vision_model="anthropic/claude-sonnet-5",
        anthropic_api_key="sk-ant-test",
        deepseek_api_key=None,
        openai_api_key=None,
        gemini_api_key=None,
    ))
    outbound: list = []

    class _Msg:
        content = "{}"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    async def fake_acompletion(**kwargs):
        outbound.append(kwargs["messages"])
        return _Resp()

    monkeypatch.setattr(llm.litellm, "acompletion", fake_acompletion)
    await llm.complete([{
        "role": "user",
        "content": f"Patient phone {PHONE} MRN: {MRN}. Age 56.",
    }])
    blob = json.dumps(outbound)
    assert PHONE not in blob
    assert MRN not in blob
    assert "[PHONE_" in blob
    assert "[MRN_" in blob
