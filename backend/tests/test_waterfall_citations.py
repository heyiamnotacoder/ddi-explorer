"""Graded interaction requires mapped citations; invented PMIDs never grade."""
from __future__ import annotations

import json

import pytest

from app.agent.waterfall import _verdict_result, evaluate_pair
from app.citations import citations_from
from app.models import Category, Grade
from app.service import DISCLAIMER


POOL = [
    {
        "pmid": "12345",
        "title": "Warfarin and omeprazole",
        "url": "https://pubmed.ncbi.nlm.nih.gov/12345",
    },
    {
        "nct_id": "NCT1",
        "title": "A trial",
        "url": "https://clinicaltrials.gov/study/NCT1",
    },
]


def _synth(**over):
    base = {
        "verdict": "interaction",
        "summary": "CYP2C19-mediated interaction.",
        "cited": ["12345"],
        "category": "interaction",
        "severity": "moderate",
        "mechanism": "CYP2C19",
        "severe_if": [],
        "patient_specific_note": None,
        "evidence_conflict": None,
        "dose_condition": None,
    }
    base.update(over)
    return base


def test_invented_pmid_never_appears_as_citation():
    cites = citations_from(["99999", "12345"], POOL, "pubmed")
    assert [c.identifier for c in cites] == ["12345"]
    assert all(c.identifier != "99999" for c in cites)


def test_empty_mapped_citations_no_grade():
    s = _synth(cited=[])
    result = _verdict_result(
        "warfarin", "omeprazole", s, Grade.A, [], "openfda", pool=POOL,
    )
    assert result.grade is None
    assert result.source_tier == "insufficient"
    assert result.citations == []


def test_invented_only_citations_no_grade():
    s = _synth(cited=["99999"])
    cites = citations_from(s["cited"], POOL, "pubmed")
    result = _verdict_result(
        "warfarin", "omeprazole", s, Grade.B, cites, "pubmed_ct", pool=POOL,
    )
    assert result.grade is None
    assert result.source_tier == "insufficient"
    assert result.citations == []
    assert all(
        "99999" not in (c.identifier or "") and "99999" not in (c.title or "")
        for c in result.citations
    )


def test_partial_unmapped_cited_ids_no_grade():
    s = _synth(cited=["12345", "99999"])
    cites = citations_from(s["cited"], POOL, "pubmed")
    result = _verdict_result(
        "warfarin", "omeprazole", s, Grade.A, cites, "openfda", pool=POOL,
    )
    assert result.grade is None
    assert result.source_tier == "insufficient"
    assert [c.identifier for c in result.citations] == ["12345"]
    assert all(c.identifier != "99999" for c in result.citations)


def test_mapped_citations_keep_grade():
    s = _synth(cited=["12345"])
    cites = citations_from(s["cited"], POOL, "pubmed")
    result = _verdict_result(
        "warfarin", "omeprazole", s, Grade.A, cites, "openfda", pool=POOL,
    )
    assert result.grade == Grade.A
    assert result.source_tier == "openfda"
    assert result.category == Category.INTERACTION
    assert [c.identifier for c in result.citations] == ["12345"]


def test_none_verdict_never_graded():
    s = _synth(verdict="none", cited=["12345"], summary="No interaction.")
    cites = citations_from(s["cited"], POOL, "pubmed")
    result = _verdict_result(
        "warfarin", "omeprazole", s, Grade.A, cites, "openfda", pool=POOL,
    )
    assert result.grade is None
    assert result.category == Category.NONE


@pytest.mark.asyncio
async def test_evaluate_pair_invented_pmid_is_insufficient(monkeypatch):
    async def fake_fda(*_a, **_k):
        return []

    async def fake_pubmed(*_a, **_k):
        return [{
            "pmid": "12345",
            "title": "A trial",
            "pubtype": ["Randomized Controlled Trial"],
            "abstract": "Interaction observed.",
            "url": "https://pubmed.ncbi.nlm.nih.gov/12345/",
        }]

    async def fake_ct(*_a, **_k):
        return []

    async def fake_web(*_a, **_k):
        return []

    async def fake_complete(*_a, **_k):
        return json.dumps({
            "verdict": "interaction",
            "summary": "Serious interaction.",
            "cited": ["99999999"],
            "category": "interaction",
            "severity": "major",
        })

    monkeypatch.setattr(
        "app.agent.waterfall.tools.openfda_label_check", fake_fda)
    monkeypatch.setattr("app.agent.waterfall.tools.pubmed_search", fake_pubmed)
    monkeypatch.setattr(
        "app.agent.waterfall.tools.clinicaltrials_search", fake_ct)
    monkeypatch.setattr("app.agent.waterfall.tools.web_search", fake_web)
    monkeypatch.setattr("app.agent.waterfall.llm.complete", fake_complete)

    result = await evaluate_pair("warfarin", "amiodarone")
    assert result.grade is None
    assert result.source_tier == "insufficient"
    blob = json.dumps(result.model_dump())
    assert "99999999" not in blob


@pytest.mark.asyncio
async def test_evaluate_pair_mapped_setid_keeps_grade_a(monkeypatch):
    async def fake_fda(a, b):
        return [{
            "setid": "abc",
            "title": "label",
            "url": "https://dailymed/abc",
            "interactions_text": f"Concomitant {b} with {a}.",
        }]

    monkeypatch.setattr(
        "app.agent.waterfall.tools.openfda_label_check", fake_fda)

    result = await evaluate_pair("warfarin", "amiodarone")
    assert result.grade == Grade.A
    assert result.source_tier == "openfda"
    assert [c.identifier for c in result.citations] == ["abc"]


def test_check_disclaimer_constant_unchanged():
    assert "decision-support" in DISCLAIMER.lower()
    assert "does not replace clinical judgment" in DISCLAIMER.lower()
