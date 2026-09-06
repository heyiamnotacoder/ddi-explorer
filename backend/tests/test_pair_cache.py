"""Process-local pair cache and 429 → that pair error, not the whole request."""
from __future__ import annotations

import json

import httpx
import pytest

from app.agent import pair_cache, tools
from app.agent.waterfall import evaluate_pair, evaluate_pairs
from app.models import Category, Citation, Grade, PairResult

PHI = "Ramesh Kumar phone 9876543210 MRN AIIMS-20451"


def _grade_a(**over) -> dict:
    base = {
        "verdict": "interaction",
        "summary": "Labelled CYP interaction.",
        "cited": ["set-1"],
        "category": "interaction",
        "severity": "moderate",
        "mechanism": "CYP",
        "severe_if": [],
        "patient_specific_note": None,
        "evidence_conflict": None,
        "dose_condition": None,
    }
    base.update(over)
    return base


def _patch_fda(monkeypatch, fn):
    monkeypatch.setattr("app.agent.waterfall.tools.openfda_label_check", fn)


def _patch_synth(monkeypatch, payload: dict):
    async def fake_complete(*_a, **_k):
        return json.dumps(payload)

    monkeypatch.setattr("app.agent.waterfall.llm.complete", fake_complete)


@pytest.mark.asyncio
async def test_identical_pairs_hunt_once(monkeypatch):
    calls: list[tuple[str, str]] = []

    async def fake_fda(a, b):
        calls.append((a, b))
        return [{
            "setid": "set-1", "title": "label",
            "url": "https://dailymed/set-1",
            "interactions_text": f"Concomitant {b} with {a}.",
        }]

    _patch_fda(monkeypatch, fake_fda)
    _patch_synth(monkeypatch, _grade_a())

    results = await evaluate_pairs(
        [("warfarin", "amiodarone"), ("Warfarin", "Amiodarone")],
    )
    assert len(calls) == 1
    assert len(results) == 2
    assert results[0].grade == results[1].grade == Grade.A
    assert results[0].source_tier == "openfda"


@pytest.mark.asyncio
async def test_cache_hit_same_grade_no_second_fetch(monkeypatch):
    calls: list[tuple[str, str]] = []

    async def fake_fda(a, b):
        calls.append((a, b))
        return [{
            "setid": "set-1", "title": "label",
            "url": "https://dailymed/set-1",
            "interactions_text": f"Concomitant {b} with {a}.",
        }]

    _patch_fda(monkeypatch, fake_fda)
    _patch_synth(monkeypatch, _grade_a())

    first = await evaluate_pair("warfarin", "amiodarone")
    second = await evaluate_pair("amiodarone", "warfarin")
    assert len(calls) == 1
    assert first.grade == second.grade == Grade.A
    assert first.source_tier == second.source_tier == "openfda"
    assert [c.identifier for c in second.citations] == ["set-1"]


def _cache_blob() -> str:
    return json.dumps(
        [p.model_dump(mode="json") for p in pair_cache.entries()], default=str)


@pytest.mark.asyncio
async def test_cache_does_not_store_patient_text(monkeypatch):
    async def fake_fda(a, b):
        return [{
            "setid": "set-1", "title": "label",
            "url": "https://dailymed/set-1",
            "interactions_text": f"Concomitant {b} with {a}.",
        }]

    _patch_fda(monkeypatch, fake_fda)
    _patch_synth(monkeypatch, _grade_a(patient_specific_note=PHI))

    live = await evaluate_pair(
        "warfarin", "amiodarone", patient_context=PHI,
    )
    # Label Grade A skips the synthesizer; overlay (service) adds notes later.
    assert live.patient_specific_note is None
    # A result graded WITH patient context is never stored: summary,
    # severe_if, and dose_condition can all carry that patient.
    blob = _cache_blob()
    assert PHI not in blob
    assert "9876543210" not in blob
    assert "Ramesh" not in blob
    assert pair_cache.get("warfarin", "amiodarone") is None


def test_put_strips_patient_note_and_skips_errors():
    pair_cache.put(PairResult(
        drugs=("warfarin", "amiodarone"),
        grade=Grade.A,
        category=Category.INTERACTION,
        summary="Known pair.",
        patient_specific_note=PHI,
        source_tier="openfda",
        citations=[Citation(source="openfda", title="label", identifier="set-1")],
    ))
    hit = pair_cache.get("amiodarone", "warfarin")
    assert hit is not None
    assert hit.grade == Grade.A
    assert hit.patient_specific_note is None
    assert PHI not in _cache_blob()

    pair_cache.put(PairResult(
        drugs=("foo", "bar"),
        source_tier="error",
        summary=PHI,
    ))
    assert pair_cache.get("foo", "bar") is None


@pytest.mark.asyncio
async def test_429_one_pair_others_still_return(monkeypatch):
    async def fake_fda(a, b):
        if "amiodarone" in (a, b):
            raise tools.ToolRateLimit("openfda")
        return [{
            "setid": "set-1", "title": "label",
            "url": "https://dailymed/set-1",
            "interactions_text": f"Concomitant {b} with {a}.",
        }]

    _patch_fda(monkeypatch, fake_fda)
    _patch_synth(monkeypatch, _grade_a())

    results = await evaluate_pairs([
        ("warfarin", "amiodarone"),
        ("warfarin", "omeprazole"),
    ])
    by = {tuple(sorted(p.drugs)): p for p in results}
    err = by[("amiodarone", "warfarin")]
    ok = by[("omeprazole", "warfarin")]
    assert err.source_tier == "error"
    assert err.grade is None
    assert "rate-limited" in err.summary.lower()
    assert ok.grade == Grade.A
    assert ok.source_tier == "openfda"


@pytest.mark.asyncio
async def test_one_pair_exception_does_not_fail_request(monkeypatch):
    async def fake_fda(a, b):
        if "explode" in (a, b):
            raise RuntimeError("boom")
        return [{
            "setid": "set-1", "title": "label",
            "url": "https://dailymed/set-1",
            "interactions_text": f"Concomitant {b} with {a}.",
        }]

    _patch_fda(monkeypatch, fake_fda)
    _patch_synth(monkeypatch, _grade_a())

    results = await evaluate_pairs([
        ("warfarin", "explode"),
        ("warfarin", "omeprazole"),
    ])
    by = {tuple(sorted(p.drugs)): p for p in results}
    assert by[("explode", "warfarin")].source_tier == "error"
    assert by[("omeprazole", "warfarin")].grade == Grade.A


@pytest.mark.asyncio
async def test_429_is_not_cached_so_retry_hunts(monkeypatch):
    n = 0

    async def fake_fda(*_a, **_k):
        nonlocal n
        n += 1
        raise tools.ToolRateLimit("openfda")

    _patch_fda(monkeypatch, fake_fda)
    first = await evaluate_pair("warfarin", "amiodarone")
    second = await evaluate_pair("warfarin", "amiodarone")
    assert n == 2
    assert first.source_tier == second.source_tier == "error"


def test_raise_http_429_is_tool_rate_limit():
    r = httpx.Response(429, request=httpx.Request("GET", "https://api.fda.gov/"))
    with pytest.raises(tools.ToolRateLimit) as ei:
        tools.raise_for_tool_status(r, "openfda")
    assert ei.value.tool == "openfda"


@pytest.mark.asyncio
async def test_patient_context_request_does_not_read_a_context_free_cache_hit(
        monkeypatch):
    """Patient-specific grading is never skipped by a cache hit."""
    calls: list[tuple[str, str]] = []

    async def fake_fda(a, b):
        calls.append((a, b))
        return [{
            "setid": "set-1", "title": "label",
            "url": "https://dailymed/set-1",
            "interactions_text": f"Concomitant {b} with {a}.",
        }]

    _patch_fda(monkeypatch, fake_fda)
    _patch_synth(monkeypatch, _grade_a())

    await evaluate_pair("warfarin", "amiodarone")
    assert len(calls) == 1
    await evaluate_pair("warfarin", "amiodarone")
    assert len(calls) == 1  # context-free repeat is served from cache
    await evaluate_pair("warfarin", "amiodarone", patient_context="CrCl 28")
    assert len(calls) == 2  # context re-hunts instead of replaying a stale grade
