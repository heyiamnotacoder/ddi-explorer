"""PK is Grade B; case-report vs trial is Grade C with conflict; fetch to cite."""
from __future__ import annotations

import json

import pytest

from app.agent.waterfall import (
    _is_pk_study,
    _is_strong_human,
    evaluate_pair,
    split_pubmed,
)
from app.models import Category, Grade

PK = {
    "pmid": "111",
    "title": "Pharmacokinetics of warfarin coadministered with omeprazole",
    "pubtype": ["Journal Article"],
    "abstract": "A pharmacokinetic study in 12 healthy volunteers showed increased AUC.",
    "url": "https://pubmed.ncbi.nlm.nih.gov/111/",
}
RCT = {
    "pmid": "222",
    "title": "Randomized trial of warfarin with amiodarone",
    "pubtype": ["Randomized Controlled Trial"],
    "abstract": "No clinically meaningful change in INR.",
    "url": "https://pubmed.ncbi.nlm.nih.gov/222/",
}
CASE = {
    "pmid": "333",
    "title": "A case report of a suspected warfarin–amiodarone interaction",
    "pubtype": ["Case Reports"],
    "abstract": "An 80-year-old developed a marked INR rise.",
    "url": "https://pubmed.ncbi.nlm.nih.gov/333/",
}
WEB_HIT = {
    "title": "Case of interaction",
    "url": "https://example.com/case",
    "snippet": "Serious interaction observed.",
}


def test_pk_journal_article_is_strong_human():
    assert _is_pk_study(PK)
    assert _is_strong_human(PK)
    strong, weak = split_pubmed([PK, CASE])
    assert strong == [PK]
    assert weak == [CASE]


def test_case_report_is_not_pk_even_if_title_says_pharmacokinetic():
    rec = {
        "pmid": "9",
        "title": "A case report of a pharmacokinetic interaction",
        "pubtype": ["Case Reports"],
        "abstract": "Single patient.",
    }
    assert not _is_pk_study(rec)
    assert not _is_strong_human(rec)


def _patch_empty_fda_ct(monkeypatch):
    async def fake_fda(*_a, **_k):
        return []

    async def fake_ct(*_a, **_k):
        return []

    monkeypatch.setattr(
        "app.agent.waterfall.tools.openfda_label_check", fake_fda)
    monkeypatch.setattr(
        "app.agent.waterfall.tools.clinicaltrials_search", fake_ct)


@pytest.mark.asyncio
async def test_human_pk_study_grades_b_without_web(monkeypatch):
    _patch_empty_fda_ct(monkeypatch)
    web_calls: list[str] = []

    async def fake_pubmed(*_a, **_k):
        return [PK]

    async def fake_search(query, **_k):
        web_calls.append(query)
        return []

    async def fake_complete(*_a, **_k):
        return json.dumps({
            "verdict": "interaction",
            "summary": "PK study shows increased warfarin AUC.",
            "cited": ["111"],
            "category": "interaction",
            "severity": "moderate",
            "evidence_conflict": None,
        })

    monkeypatch.setattr("app.agent.waterfall.tools.pubmed_search", fake_pubmed)
    monkeypatch.setattr("app.agent.waterfall.tools.web_search", fake_search)
    monkeypatch.setattr("app.agent.waterfall.llm.complete", fake_complete)

    result = await evaluate_pair("warfarin", "omeprazole")
    assert result.grade == Grade.B
    assert result.source_tier == "pubmed_ct"
    assert [c.identifier for c in result.citations] == ["111"]
    assert result.evidence_conflict is None
    assert web_calls == []


@pytest.mark.asyncio
async def test_case_report_plus_negative_trial_is_grade_c_with_conflict(monkeypatch):
    _patch_empty_fda_ct(monkeypatch)
    completes: list[str] = []

    async def fake_pubmed(*_a, **_k):
        return [RCT, CASE]

    async def fake_search(*_a, **_k):
        return []

    async def fake_fetch(*_a, **_k):
        return ""

    async def fake_complete(messages, **_k):
        blob = json.dumps(messages)
        completes.append(blob)
        if "Evidence tier: human trial/PK literature (Grade B)" in blob:
            return json.dumps({
                "verdict": "none",
                "summary": "RCT found no clinically meaningful interaction.",
                "cited": ["222"],
                "category": "none",
                "evidence_conflict": None,
            })
        return json.dumps({
            "verdict": "interaction",
            "summary": "Case report of a marked INR rise; trial was negative.",
            "cited": ["333", "222"],
            "category": "interaction",
            "severity": "moderate",
            "evidence_conflict": (
                "RCT negative; case report positive."
            ),
        })

    monkeypatch.setattr("app.agent.waterfall.tools.pubmed_search", fake_pubmed)
    monkeypatch.setattr("app.agent.waterfall.tools.web_search", fake_search)
    monkeypatch.setattr("app.agent.waterfall.tools.web_fetch", fake_fetch)
    monkeypatch.setattr("app.agent.waterfall.llm.complete", fake_complete)

    result = await evaluate_pair("warfarin", "amiodarone")
    assert result.grade == Grade.C
    assert result.category == Category.INTERACTION
    assert result.evidence_conflict
    assert "333" in [c.identifier for c in result.citations]
    assert len(completes) == 2


@pytest.mark.asyncio
async def test_conflict_filled_when_synthesizer_omits_it(monkeypatch):
    _patch_empty_fda_ct(monkeypatch)

    async def fake_pubmed(*_a, **_k):
        return [RCT, CASE]

    async def fake_search(*_a, **_k):
        return []

    async def fake_complete(messages, **_k):
        blob = json.dumps(messages)
        if "Evidence tier: human trial/PK literature (Grade B)" in blob:
            return json.dumps({
                "verdict": "none",
                "summary": "No interaction in the RCT.",
                "cited": ["222"],
            })
        return json.dumps({
            "verdict": "interaction",
            "summary": "Case report of bleeding.",
            "cited": ["333"],
            "category": "interaction",
            "severity": "moderate",
        })

    monkeypatch.setattr("app.agent.waterfall.tools.pubmed_search", fake_pubmed)
    monkeypatch.setattr("app.agent.waterfall.tools.web_search", fake_search)
    monkeypatch.setattr("app.agent.waterfall.llm.complete", fake_complete)

    result = await evaluate_pair("warfarin", "amiodarone")
    assert result.grade == Grade.C
    assert result.evidence_conflict
    assert "case report" in result.evidence_conflict.lower()


@pytest.mark.asyncio
async def test_unfetched_search_url_is_not_a_citation(monkeypatch):
    _patch_empty_fda_ct(monkeypatch)
    fetched: list[str] = []

    async def fake_pubmed(*_a, **_k):
        return []

    async def fake_search(*_a, **_k):
        return [WEB_HIT]

    async def fake_fetch(url, **_k):
        fetched.append(url)
        return ""

    complete_calls = 0

    async def fake_complete(*_a, **_k):
        nonlocal complete_calls
        complete_calls += 1
        return json.dumps({
            "verdict": "interaction",
            "summary": "Should not grade from a snippet.",
            "cited": ["https://example.com/case"],
        })

    monkeypatch.setattr("app.agent.waterfall.tools.pubmed_search", fake_pubmed)
    monkeypatch.setattr("app.agent.waterfall.tools.web_search", fake_search)
    monkeypatch.setattr("app.agent.waterfall.tools.web_fetch", fake_fetch)
    monkeypatch.setattr("app.agent.waterfall.llm.complete", fake_complete)

    result = await evaluate_pair("warfarin", "amiodarone")
    assert fetched == ["https://example.com/case"]
    assert complete_calls == 0
    assert result.grade is None
    assert result.source_tier == "none"
    blob = json.dumps(result.model_dump())
    assert "example.com/case" not in blob


@pytest.mark.asyncio
async def test_fetched_page_can_be_cited_as_grade_c(monkeypatch):
    _patch_empty_fda_ct(monkeypatch)

    async def fake_pubmed(*_a, **_k):
        return []

    async def fake_search(*_a, **_k):
        return [WEB_HIT]

    async def fake_fetch(url, **_k):
        assert url == "https://example.com/case"
        return "# Case\n\nSerious INR rise after coadministration."

    async def fake_complete(*_a, **_k):
        return json.dumps({
            "verdict": "interaction",
            "summary": "Case-level web evidence of an interaction.",
            "cited": ["https://example.com/case"],
            "category": "interaction",
            "severity": "moderate",
        })

    monkeypatch.setattr("app.agent.waterfall.tools.pubmed_search", fake_pubmed)
    monkeypatch.setattr("app.agent.waterfall.tools.web_search", fake_search)
    monkeypatch.setattr("app.agent.waterfall.tools.web_fetch", fake_fetch)
    monkeypatch.setattr("app.agent.waterfall.llm.complete", fake_complete)

    result = await evaluate_pair("warfarin", "amiodarone")
    assert result.grade == Grade.C
    assert [c.url for c in result.citations] == ["https://example.com/case"]
    assert all(c.identifier is None or c.identifier != "https://example.com/case"
               for c in result.citations)


@pytest.mark.asyncio
async def test_tier3_negative_verdict_is_not_rewritten_into_grade_c(monkeypatch):
    """A weak-evidence pool the synthesizer reads as negative stays ungraded.

    Disclosing a conflict must never manufacture one: only a positive tier-3
    finding earns Grade C.
    """
    _patch_empty_fda_ct(monkeypatch)

    async def fake_pubmed(*_a, **_k):
        return [CASE]

    async def fake_search(*_a, **_k):
        return []

    async def fake_complete(messages, **_k):
        return json.dumps({
            "verdict": "none",
            "summary": "The case report describes a different mechanism.",
            "cited": ["333"],
            "category": "none",
        })

    monkeypatch.setattr("app.agent.waterfall.tools.pubmed_search", fake_pubmed)
    monkeypatch.setattr("app.agent.waterfall.tools.web_search", fake_search)
    monkeypatch.setattr("app.agent.waterfall.llm.complete", fake_complete)

    result = await evaluate_pair("warfarin", "amiodarone")
    assert result.grade is None
    assert result.category == Category.NONE
    assert result.evidence_conflict is None


@pytest.mark.asyncio
async def test_tier3_insufficient_verdict_is_not_rewritten(monkeypatch):
    _patch_empty_fda_ct(monkeypatch)

    async def fake_pubmed(*_a, **_k):
        return [CASE]

    async def fake_search(*_a, **_k):
        return []

    async def fake_complete(messages, **_k):
        return json.dumps({
            "verdict": "insufficient",
            "summary": "Records retrieved do not answer the question.",
            "cited": ["333"],
        })

    monkeypatch.setattr("app.agent.waterfall.tools.pubmed_search", fake_pubmed)
    monkeypatch.setattr("app.agent.waterfall.tools.web_search", fake_search)
    monkeypatch.setattr("app.agent.waterfall.llm.complete", fake_complete)

    result = await evaluate_pair("warfarin", "amiodarone")
    assert result.grade is None
    assert result.source_tier == "insufficient"
    assert result.evidence_conflict is None
