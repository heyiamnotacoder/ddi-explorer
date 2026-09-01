"""Deterministic Grade A overlay — no network, no LLM."""
from __future__ import annotations

import inspect

import pytest

from app import service
from app.agent import alternatives as alts
from app.models import Category, CheckRequest, Grade, NormalizedDrug, PairResult
from app.pipeline import overlay
from app.pipeline.overlay import apply
from app.service import DISCLAIMER


def _drug(name: str, *comps: str, dose: str | None = None,
          schedule: str | None = None) -> NormalizedDrug:
    parts = list(comps) or [name]
    return NormalizedDrug(
        input_name=name, generic_name=", ".join(parts),
        components=parts, dose=dose, schedule=schedule,
    )


def _local(a: str, b: str, summary: str, *,
           category: Category = Category.INTERACTION,
           severity: str = "moderate") -> PairResult:
    return PairResult(
        drugs=(a, b), grade=Grade.A, category=category, severity=severity,
        summary=summary, source_tier="local",
    )


def test_overlay_is_sync_and_does_not_import_llm():
    assert inspect.iscoroutinefunction(apply) is False
    assert not hasattr(overlay, "llm")
    src = inspect.getsource(overlay)
    assert "litellm" not in src
    assert "complete(" not in src


def test_levothyroxine_calcium_is_timing_not_a_swap():
    p = _local(
        "levothyroxine", "calcium",
        "The absorption of Levothyroxine can be decreased when combined with Calcium.",
    )
    out = apply(
        [p],
        drugs=[_drug("Thyronorm", "levothyroxine", schedule="1-0-0"),
               _drug("Calcium", "calcium", schedule="0-0-1")],
        patient_ctx=None,
    )
    assert len(out) == 1
    assert out[0].category == Category.TIMING
    assert out[0].grade == Grade.A
    assert not alts.is_actionable(out[0])
    assert out[0].severe_if


def test_contraindicated_is_not_downgraded_to_timing():
    p = _local(
        "levothyroxine", "calcium",
        "The absorption of Levothyroxine can be decreased when combined with Calcium.",
        category=Category.CONTRAINDICATED, severity="major",
    )
    out = apply([p], drugs=[_drug("levothyroxine"), _drug("calcium")],
                patient_ctx=None)
    assert out[0].category == Category.CONTRAINDICATED


def test_missing_dose_on_dose_dependent_pair_is_conditional():
    p = _local(
        "amlodipine", "simvastatin",
        "The metabolism of Simvastatin can be decreased when combined with Amlodipine.",
    )
    out = apply(
        [p],
        drugs=[_drug("Amlong", "amlodipine", dose="5 mg"),
               _drug("simvastatin", "simvastatin")],
        patient_ctx=None,
    )
    assert out[0].dose_condition
    assert out[0].dose_condition.startswith("DDI possible if")
    assert "simvastatin" in out[0].dose_condition.lower()


def test_known_dose_skips_dose_condition():
    p = _local(
        "amlodipine", "simvastatin",
        "The metabolism of Simvastatin can be decreased when combined with Amlodipine.",
    )
    out = apply(
        [p],
        drugs=[_drug("amlodipine", "amlodipine", dose="5 mg"),
               _drug("simvastatin", "simvastatin", dose="10 mg")],
        patient_ctx=None,
    )
    assert out[0].dose_condition is None


def test_patient_context_sets_note_not_severe_if():
    p = _local(
        "warfarin", "amiodarone",
        "The risk or severity of bleeding can be increased.",
        severity="major",
    )
    out = apply(
        [p],
        drugs=[_drug("warfarin"), _drug("amiodarone")],
        patient_ctx="Age 56, CrCl 42.",
    )
    assert out[0].patient_specific_note
    assert "age 56" in out[0].patient_specific_note.lower()
    assert "renal" in out[0].patient_specific_note.lower()
    assert out[0].severe_if == []


def test_absent_patient_context_lists_severe_if():
    p = _local(
        "warfarin", "amiodarone",
        "The risk or severity of bleeding can be increased.",
        severity="major",
    )
    out = apply(
        [p],
        drugs=[_drug("warfarin"), _drug("amiodarone")],
        patient_ctx=None,
    )
    assert out[0].patient_specific_note is None
    assert out[0].severe_if


def test_waterfall_pairs_are_not_overwritten():
    p = PairResult(
        drugs=("warfarin", "omeprazole"), grade=Grade.A,
        category=Category.INTERACTION, summary="from synthesizer",
        source_tier="openfda", dose_condition="keep me",
        patient_specific_note="synth note", severe_if=["x"],
    )
    out = apply([p], drugs=[_drug("warfarin"), _drug("omeprazole")],
                patient_ctx="Age 80")
    assert out[0].dose_condition == "keep me"
    assert out[0].patient_specific_note == "synth note"
    assert out[0].severe_if == ["x"]


@pytest.mark.asyncio
async def test_check_keeps_dose_schedule_and_overlays_without_waterfall(monkeypatch):
    waterfall_calls: list = []

    async def fake_extract(_text: str) -> dict:
        return {
            "drugs": [
                {"name": "Thyronorm 50", "dose": "50 mcg", "timing": "1-0-0"},
                {"name": "calcium", "dose": "500 mg", "timing": "0-0-1"},
            ],
            "non_drugs": [],
            "patient_context": "Age 56, CrCl 42",
        }

    async def fake_norm(names: list[str]):
        table = {
            "Thyronorm 50": _drug("Thyronorm 50", "levothyroxine"),
            "calcium": _drug("calcium", "calcium"),
        }
        return [table[n] for n in names], []

    async def fake_prefilter(_items):
        return [_local(
            "calcium", "levothyroxine",
            "The absorption of Levothyroxine can be decreased when combined with Calcium.",
        )], []

    async def fake_waterfall(pairs, patient_ctx=None):
        waterfall_calls.append((pairs, patient_ctx))
        return []

    monkeypatch.setattr(service.extract, "extract_drugs", fake_extract)
    monkeypatch.setattr(service.norm, "normalize_drugs", fake_norm)
    monkeypatch.setattr(service.prefilter, "check_known_pairs", fake_prefilter)
    monkeypatch.setattr(service.waterfall, "evaluate_pairs", fake_waterfall)

    resp = await service.run_check(CheckRequest(
        text="Thyronorm 50 1-0-0, calcium 0-0-1",
        patient_context="Ramesh Kumar, phone 9876543210, MRN: AIIMS-20451. Age 56, CrCl 42.",
    ))
    assert resp.disclaimer == DISCLAIMER
    by = {d.input_name: d for d in resp.normalized_drugs}
    assert by["Thyronorm 50"].dose == "50 mcg"
    assert by["Thyronorm 50"].schedule == "1-0-0"
    assert by["calcium"].dose == "500 mg"
    assert by["calcium"].schedule == "0-0-1"

    assert waterfall_calls == []
    assert len(resp.pairs) == 1
    p = resp.pairs[0]
    assert p.category == Category.TIMING
    assert p.grade == Grade.A
    assert p.patient_specific_note
    assert "9876543210" not in p.patient_specific_note
    assert "Ramesh" not in p.patient_specific_note
    assert "AIIMS-20451" not in p.patient_specific_note
    assert "age 56" in p.patient_specific_note.lower()
    assert p.severe_if == []
