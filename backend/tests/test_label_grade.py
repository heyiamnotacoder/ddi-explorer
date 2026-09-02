"""Grade A from a label partner mention — no synthesizer."""
from __future__ import annotations

import json

import pytest

from app.agent.waterfall import evaluate_pair
from app.models import Category, CheckRequest, Grade, NormalizedDrug, PairResult
from app.pipeline.overlay import apply
from app.service import DISCLAIMER, run_check


def _fda_rec(a: str, b: str, *, setid: str = "set-1", extra: str = "") -> dict:
    return {
        "setid": setid,
        "title": f"{a} labeling",
        "url": f"https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid={setid}",
        "interactions_text": (
            f"Concomitant {b} with {a} increases exposure. {extra}"
        ).strip(),
        "label_contraindicated": "contraindicat" in extra.lower(),
    }


@pytest.mark.asyncio
async def test_partner_mention_is_grade_a_without_synthesizer(monkeypatch):
    synth = {"n": 0}

    async def fake_fda(a, b):
        return [_fda_rec(a, b)]

    async def fake_complete(*_a, **_k):
        synth["n"] += 1
        return json.dumps({"verdict": "insufficient", "cited": ["99999"]})

    monkeypatch.setattr(
        "app.agent.waterfall.tools.openfda_label_check", fake_fda)
    monkeypatch.setattr("app.agent.waterfall.llm.complete", fake_complete)

    result = await evaluate_pair("warfarin", "amiodarone")
    assert synth["n"] == 0
    assert result.grade == Grade.A
    assert result.source_tier == "openfda"
    assert result.category == Category.INTERACTION
    assert [c.identifier for c in result.citations] == ["set-1"]
    assert all(c.source == "openfda" for c in result.citations)
    blob = json.dumps(result.model_dump())
    assert "99999" not in blob


@pytest.mark.asyncio
async def test_contraindicat_snippet_sets_category(monkeypatch):
    async def fake_fda(a, b):
        return [_fda_rec(
            a, b,
            extra=f"Contraindicated with {b}. Do not coadminister {b}.",
        )]

    monkeypatch.setattr(
        "app.agent.waterfall.tools.openfda_label_check", fake_fda)

    result = await evaluate_pair("sildenafil", "isosorbide mononitrate")
    assert result.grade == Grade.A
    assert result.category == Category.CONTRAINDICATED
    assert result.severity == "major"


@pytest.mark.asyncio
async def test_unscoped_contraindicat_is_not_a_banner(monkeypatch):
    async def fake_fda(a, b):
        return [_fda_rec(
            a, b,
            extra="Contraindications: severe renal impairment, metabolic acidosis.",
        )]

    monkeypatch.setattr(
        "app.agent.waterfall.tools.openfda_label_check", fake_fda)

    result = await evaluate_pair("metformin", "sitagliptin")
    assert result.grade == Grade.A
    assert result.category == Category.INTERACTION


@pytest.mark.asyncio
async def test_janumet_colist_is_not_grade_a(monkeypatch):
    pubmed_n = {"n": 0}

    async def fake_fda(a, b):
        return [{
            "setid": "janumet",
            "subject_drug": a,
            "title": "JANUMET labeling",
            "url": "https://dailymed/janumet",
            "interactions_text": (
                f"JANUMET contains {b} and {a}. This product contains {b}. "
                "Contraindications: severe renal impairment."
            ),
            "di_text": (
                f"JANUMET contains {b} and {a}. This product contains {b}."
            ),
            "ci_text": "Contraindications: severe renal impairment.",
            "label_contraindicated": True,
        }]

    async def fake_pubmed(*_a, **_k):
        pubmed_n["n"] += 1
        return []

    async def fake_ct(*_a, **_k):
        return []

    async def fake_web(*_a, **_k):
        return []

    monkeypatch.setattr(
        "app.agent.waterfall.tools.openfda_label_check", fake_fda)
    monkeypatch.setattr("app.agent.waterfall.tools.pubmed_search", fake_pubmed)
    monkeypatch.setattr(
        "app.agent.waterfall.tools.clinicaltrials_search", fake_ct)
    monkeypatch.setattr("app.agent.waterfall.tools.web_search", fake_web)

    result = await evaluate_pair("sitagliptin", "metformin")
    assert pubmed_n["n"] == 1
    assert result.grade is None
    assert result.category != Category.CONTRAINDICATED


@pytest.mark.asyncio
async def test_maoi_ssri_pair_scoped_is_contraindicated(monkeypatch):
    async def fake_fda(a, b):
        return [{
            "setid": "zoloft-maoi",
            "subject_drug": "fluoxetine",
            "title": "fluoxetine labeling",
            "url": "https://dailymed/fluox",
            "interactions_text": (
                "ZOLOFT is contraindicated with MAOIs including phenelzine. "
                "Do not use with phenelzine."
            ),
            "di_text": None,
            "ci_text": (
                "ZOLOFT is contraindicated with MAOIs including phenelzine. "
                "Do not use with phenelzine."
            ),
            "label_contraindicated": True,
        }]

    monkeypatch.setattr(
        "app.agent.waterfall.tools.openfda_label_check", fake_fda)
    result = await evaluate_pair("phenelzine", "fluoxetine")
    assert result.grade == Grade.A
    assert result.category == Category.CONTRAINDICATED


@pytest.mark.asyncio
async def test_levothyroxine_calcium_label_is_timing_not_banner(monkeypatch):
    async def fake_fda(a, b):
        return [{
            "setid": "synthroid-ca",
            "subject_drug": "levothyroxine",
            "title": "levothyroxine labeling",
            "url": "https://dailymed/lt4",
            "interactions_text": (
                "Calcium supplements may decrease the absorption of levothyroxine. "
                "Separate administration by 4 hours."
            ),
            "di_text": (
                "Calcium supplements may decrease the absorption of levothyroxine. "
                "Separate administration by 4 hours."
            ),
            "ci_text": "Contraindicated in uncorrected adrenal insufficiency.",
            "label_contraindicated": True,
        }]

    monkeypatch.setattr(
        "app.agent.waterfall.tools.openfda_label_check", fake_fda)
    result = await evaluate_pair("levothyroxine", "calcium")
    assert result.grade == Grade.A
    assert result.category == Category.INTERACTION
    out = apply(
        [result],
        drugs=[
            NormalizedDrug(
                input_name="levothyroxine", generic_name="levothyroxine",
                components=["levothyroxine"], schedule="1-0-0"),
            NormalizedDrug(
                input_name="calcium", generic_name="calcium",
                components=["calcium"], schedule="0-0-1"),
        ],
        patient_ctx=None,
    )
    assert out[0].category == Category.TIMING
    assert out[0].grade == Grade.A


@pytest.mark.asyncio
async def test_no_partner_mention_does_not_grade_and_falls_through(monkeypatch):
    pubmed_n = {"n": 0}

    async def fake_fda(*_a, **_k):
        return [{
            "setid": "toc",
            "title": "label",
            "url": "https://dailymed/toc",
            "interactions_text": "HIGHLIGHTS OF PRESCRIBING INFORMATION.",
        }]

    async def fake_pubmed(*_a, **_k):
        pubmed_n["n"] += 1
        return [{
            "pmid": "111",
            "title": "Pharmacokinetic study",
            "pubtype": ["Journal Article"],
            "abstract": "A pharmacokinetic study in volunteers.",
            "url": "https://pubmed.ncbi.nlm.nih.gov/111/",
        }]

    async def fake_ct(*_a, **_k):
        return []

    async def fake_web(*_a, **_k):
        return []

    async def fake_complete(*_a, **_k):
        return json.dumps({
            "verdict": "interaction",
            "summary": "PK interaction.",
            "cited": ["111"],
            "category": "interaction",
            "severity": "moderate",
        })

    monkeypatch.setattr(
        "app.agent.waterfall.tools.openfda_label_check", fake_fda)
    monkeypatch.setattr("app.agent.waterfall.tools.pubmed_search", fake_pubmed)
    monkeypatch.setattr(
        "app.agent.waterfall.tools.clinicaltrials_search", fake_ct)
    monkeypatch.setattr("app.agent.waterfall.tools.web_search", fake_web)
    monkeypatch.setattr("app.agent.waterfall.llm.complete", fake_complete)

    result = await evaluate_pair("atorvastatin", "clarithromycin")
    assert pubmed_n["n"] == 1
    assert result.grade == Grade.B
    assert result.source_tier == "pubmed_ct"


@pytest.mark.asyncio
async def test_missing_setid_does_not_grade_from_labels(monkeypatch):
    pubmed_n = {"n": 0}

    async def fake_fda(a, b):
        rec = _fda_rec(a, b)
        rec["setid"] = None
        return [rec]

    async def fake_pubmed(*_a, **_k):
        pubmed_n["n"] += 1
        return []

    async def fake_ct(*_a, **_k):
        return []

    async def fake_web(*_a, **_k):
        return []

    monkeypatch.setattr(
        "app.agent.waterfall.tools.openfda_label_check", fake_fda)
    monkeypatch.setattr("app.agent.waterfall.tools.pubmed_search", fake_pubmed)
    monkeypatch.setattr(
        "app.agent.waterfall.tools.clinicaltrials_search", fake_ct)
    monkeypatch.setattr("app.agent.waterfall.tools.web_search", fake_web)

    result = await evaluate_pair("warfarin", "rifampicin")
    assert pubmed_n["n"] == 1
    assert result.grade is None


def test_openfda_grade_a_gets_overlay_without_llm():
    p = PairResult(
        drugs=("simvastatin", "amlodipine"), grade=Grade.A,
        category=Category.INTERACTION,
        summary="Label: amlodipine increases simvastatin exposure.",
        source_tier="openfda",
        citations=[],
    )
    out = apply(
        [p],
        drugs=[
            NormalizedDrug(
                input_name="simvastatin", generic_name="simvastatin",
                components=["simvastatin"]),
            NormalizedDrug(
                input_name="amlodipine", generic_name="amlodipine",
                components=["amlodipine"], dose="5 mg"),
        ],
        patient_ctx="Age 80, CKD",
    )
    assert out[0].grade == Grade.A
    assert out[0].source_tier == "openfda"
    assert out[0].dose_condition
    assert out[0].patient_specific_note
    assert "age 80" in out[0].patient_specific_note.lower()


def test_grade_b_is_not_overlaid():
    p = PairResult(
        drugs=("warfarin", "rifampicin"), grade=Grade.B,
        category=Category.INTERACTION, summary="from pubmed",
        source_tier="pubmed_ct", dose_condition="keep me",
        patient_specific_note="synth note", severe_if=["x"],
    )
    out = apply(
        [p],
        drugs=[
            NormalizedDrug(input_name="warfarin", generic_name="warfarin",
                           components=["warfarin"]),
            NormalizedDrug(input_name="rifampicin", generic_name="rifampin",
                           components=["rifampicin"]),
        ],
        patient_ctx="Age 80",
    )
    assert out[0].dose_condition == "keep me"
    assert out[0].patient_specific_note == "synth note"
    assert out[0].severe_if == ["x"]


@pytest.mark.asyncio
async def test_contraindicated_pair_fills_banner(monkeypatch):
    async def fake_extract(_text: str) -> dict:
        return {
            "drugs": [
                {"name": "sildenafil", "dose": "50 mg", "timing": None},
                {"name": "isosorbide mononitrate", "dose": "30 mg", "timing": None},
            ],
            "non_drugs": [],
            "patient_context": None,
        }

    async def fake_norm(names: list[str]):
        return [
            NormalizedDrug(input_name=n, generic_name=n, components=[n])
            for n in names
        ], []

    async def fake_prefilter(_items):
        return [], [("isosorbide mononitrate", "sildenafil")]

    async def fake_eval(pairs, patient_ctx=None):
        return [PairResult(
            drugs=("isosorbide mononitrate", "sildenafil"),
            grade=Grade.A,
            category=Category.CONTRAINDICATED,
            severity="major",
            summary="Contraindicated with organic nitrates.",
            source_tier="openfda",
        )]

    monkeypatch.setattr("app.service.extract.extract_drugs", fake_extract)
    monkeypatch.setattr("app.service.norm.normalize_drugs", fake_norm)
    monkeypatch.setattr("app.service.prefilter.check_known_pairs", fake_prefilter)
    monkeypatch.setattr("app.service.waterfall.evaluate_pairs", fake_eval)
    monkeypatch.setattr("app.service.avoid_mod.lookup", _empty_avoid)

    resp = await run_check(CheckRequest(text="sildenafil, isosorbide mononitrate"))
    assert resp.disclaimer == DISCLAIMER
    assert len(resp.contraindicated_banner) == 1
    assert resp.contraindicated_banner[0].category == Category.CONTRAINDICATED
    assert resp.pairs[0].grade == Grade.A


async def _empty_avoid(*_a, **_k):
    return []
