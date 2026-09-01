"""openFDA label hits: set_id, partner window, CI/boxed fields, aliases."""
from __future__ import annotations

import pytest

from app.agent import tools
from app.agent.waterfall import evaluate_pair
from app.models import Grade
from app.service import DISCLAIMER


def _resp(status: int, payload: dict | None = None):
    class Resp:
        status_code = status

        def json(self):
            return payload or {}

        def raise_for_status(self):
            if status >= 400 and status != 404:
                raise tools.httpx.HTTPStatusError(
                    "err", request=tools.httpx.Request("GET", "https://api.fda.gov/"),
                    response=tools.httpx.Response(status),
                )

    return Resp()


def _patch_client(monkeypatch, handler):
    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return False

        async def get(self, url, params=None):
            return handler(url, params or {})

    monkeypatch.setattr(tools, "_client", lambda: FakeClient())


def test_setid_from_set_id_and_spl_set_id():
    assert tools._setid_from_record({"set_id": "from-set-id"}) == "from-set-id"
    assert tools._setid_from_record({
        "openfda": {"spl_set_id": ["from-spl"]},
    }) == "from-spl"
    assert tools._setid_from_record({"setid": None, "set_id": None}) is None
    # raw openFDA never uses `setid`; leftover key still accepted
    assert tools._setid_from_record({"setid": "legacy"}) == "legacy"


def test_window_keeps_partner_past_char_3000():
    prefix = "HIGHLIGHTS OF PRESCRIBING INFORMATION " + ("x " * 1800)
    rec = {
        "set_id": "atorva-1",
        "drug_interactions": [
            prefix + "Avoid concomitant clarithromycin with atorvastatin."
        ],
    }
    assert "clarithromycin" not in prefix[:3000].lower()
    hit = tools._label_hit(rec, "atorvastatin", tools._label_aliases("clarithromycin"))
    assert hit is not None
    assert hit["setid"] == "atorva-1"
    assert "clarithromycin" in hit["interactions_text"].lower()
    assert "atorva-1" in hit["url"]


def test_window_without_partner_is_not_a_hit():
    rec = {
        "set_id": "toc-1",
        "drug_interactions": ["HIGHLIGHTS see full prescribing information. " * 80],
    }
    assert tools._label_hit(rec, "atorvastatin", tools._label_aliases("clarithromycin")) is None


def test_boxed_warning_field_is_a_hit():
    rec = {
        "set_id": "box-1",
        "boxed_warning": ["Concomitant clarithromycin increases myopathy risk."],
    }
    hit = tools._label_hit(rec, "atorvastatin", tools._label_aliases("clarithromycin"))
    assert hit is not None
    assert "clarithromycin" in hit["interactions_text"].lower()


def test_contraindications_field_is_a_hit():
    rec = {
        "set_id": "sild-1",
        "contraindications": [
            "Do not use with organic nitrates in any form."
        ],
        "drug_interactions": ["See contraindications."],
    }
    hit = tools._label_hit(
        rec, "sildenafil", tools._label_aliases("isosorbide mononitrate"))
    assert hit is not None
    assert "nitrate" in hit["interactions_text"].lower()


def test_inn_usan_spelling_aliases():
    assert "acetaminophen" in tools._label_aliases("paracetamol")
    assert "paracetamol" in tools._label_aliases("acetaminophen")
    assert "amoxicillin" in tools._label_aliases("amoxycillin")
    assert "amoxycillin" in tools._label_aliases("amoxicillin")


def test_rifampicin_aliases_include_rifampin():
    aliases = tools._label_aliases("rifampicin")
    assert "rifampin" in aliases
    assert "rifampicin" in aliases
    rec = {
        "openfda": {"spl_set_id": ["warf-rif"]},
        "drug_interactions": ["Rifampin may decrease warfarin exposure."],
    }
    hit = tools._label_hit(rec, "warfarin", aliases)
    assert hit is not None
    assert hit["setid"] == "warf-rif"
    assert "rifampin" in hit["interactions_text"].lower()


def test_isosorbide_aliases_include_nitrates():
    aliases = tools._label_aliases("isosorbide mononitrate")
    assert "nitrates" in aliases
    assert "nitrate" in aliases


def test_disclaimer_still_present():
    assert "decision-support" in DISCLAIMER.lower()


@pytest.mark.asyncio
async def test_label_check_search_uses_ci_boxed_and_aliases(monkeypatch):
    captured: list[str] = []

    def handler(_url, params):
        captured.append(params.get("search", ""))
        return _resp(200, {"results": [{
            "set_id": "sild-ci",
            "openfda": {"generic_name": ["sildenafil"]},
            "contraindications": ["Concomitant organic nitrates are contraindicated."],
        }]})

    _patch_client(monkeypatch, handler)
    rows = await tools.openfda_label_check("sildenafil", "isosorbide mononitrate")
    assert captured
    blob = " ".join(captured)
    assert "contraindications:" in blob
    assert "boxed_warning:" in blob
    assert "drug_interactions:" in blob
    assert 'contraindications:"nitrates"' in blob or 'drug_interactions:"nitrates"' in blob
    assert rows
    assert rows[0]["setid"] == "sild-ci"
    assert "nitrate" in rows[0]["interactions_text"].lower()


@pytest.mark.asyncio
async def test_label_check_drops_toc_only_api_hit(monkeypatch):
    def handler(_url, params):
        return _resp(200, {"results": [{
            "set_id": "toc-only",
            "drug_interactions": ["HIGHLIGHTS OF PRESCRIBING INFORMATION. " * 50],
        }]})

    _patch_client(monkeypatch, handler)
    rows = await tools.openfda_label_check("atorvastatin", "clarithromycin")
    assert rows == []


@pytest.mark.asyncio
async def test_empty_fda_window_falls_through_to_pubmed(monkeypatch):
    pubmed_called = {"n": 0}

    async def fake_fda(*_a, **_k):
        return []

    async def fake_pubmed(*_a, **_k):
        pubmed_called["n"] += 1
        return [{
            "pmid": "111",
            "title": "PK study",
            "pubtype": ["Journal Article"],
            "abstract": "Pharmacokinetic interaction in volunteers.",
            "url": "https://pubmed.ncbi.nlm.nih.gov/111/",
        }]

    async def fake_ct(*_a, **_k):
        return []

    async def fake_web(*_a, **_k):
        return []

    async def fake_complete(*_a, **_k):
        import json
        return json.dumps({
            "verdict": "interaction",
            "summary": "PK interaction.",
            "cited": ["111"],
            "category": "interaction",
            "severity": "moderate",
        })

    monkeypatch.setattr(tools, "openfda_label_check", fake_fda)
    monkeypatch.setattr(
        "app.agent.waterfall.tools.openfda_label_check", fake_fda)
    monkeypatch.setattr("app.agent.waterfall.tools.pubmed_search", fake_pubmed)
    monkeypatch.setattr(
        "app.agent.waterfall.tools.clinicaltrials_search", fake_ct)
    monkeypatch.setattr("app.agent.waterfall.tools.web_search", fake_web)
    monkeypatch.setattr("app.agent.waterfall.llm.complete", fake_complete)

    result = await evaluate_pair("rifampicin", "warfarin")
    assert pubmed_called["n"] == 1
    assert result.grade == Grade.B
    assert result.source_tier == "pubmed_ct"


@pytest.mark.asyncio
async def test_substance_check_uses_set_id(monkeypatch):
    def handler(_url, params):
        return _resp(200, {"results": [{
            "set_id": "warf-alc",
            "openfda": {"generic_name": ["warfarin"]},
            "drug_interactions": ["Patients should avoid alcohol."],
            "food_interactions": [],
        }]})

    _patch_client(monkeypatch, handler)
    rows = await tools.openfda_substance_check("warfarin", ["alcohol"])
    assert len(rows) == 1
    assert rows[0]["setid"] == "warf-alc"
    assert "alcohol" in rows[0]["interactions_text"].lower()
    assert "warf-alc" in rows[0]["url"]


@pytest.mark.asyncio
async def test_citation_gate_still_rejects_invented_setid(monkeypatch):
    import json

    async def fake_fda(a, b):
        return [{
            "setid": "abc",
            "title": "label",
            "url": "https://dailymed/abc",
            "interactions_text": f"Concomitant {b} with {a}.",
        }]

    async def fake_complete(*_a, **_k):
        return json.dumps({
            "verdict": "interaction",
            "summary": "Serious.",
            "cited": ["not-a-real-id"],
            "category": "interaction",
            "severity": "major",
        })

    monkeypatch.setattr(
        "app.agent.waterfall.tools.openfda_label_check", fake_fda)
    monkeypatch.setattr("app.agent.waterfall.llm.complete", fake_complete)
    result = await evaluate_pair("warfarin", "amiodarone")
    assert result.grade == Grade.A
    assert result.source_tier == "openfda"
    assert [c.identifier for c in result.citations] == ["abc"]
    blob = json.dumps(result.model_dump())
    assert "not-a-real-id" not in blob
