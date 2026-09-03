"""Label-backed avoid-with — no network, no LLM, no invented herbals."""
from __future__ import annotations

import inspect

import pytest

from app import service
from app.models import CheckRequest, NormalizedDrug
from app.pipeline import avoid
from app.pipeline.avoid import EMPTY_NOTE, lookup
from app.service import DISCLAIMER


def _drug(name: str, *comps: str) -> NormalizedDrug:
    parts = list(comps) or [name]
    return NormalizedDrug(
        input_name=name, generic_name=", ".join(parts), components=parts,
    )


def test_avoid_does_not_import_llm():
    assert inspect.iscoroutinefunction(lookup)
    assert not hasattr(avoid, "llm")
    src = inspect.getsource(avoid)
    assert "litellm" not in src
    assert "complete(" not in src


def test_empty_copy_is_honest():
    note = EMPTY_NOTE.format(substance="kava")
    assert "kava" in note
    assert "skipped" not in note.lower()
    assert "pair checking" not in note.lower()


@pytest.mark.asyncio
async def test_label_hit_maps_citation_to_retrieved_record(monkeypatch):
    async def fake_check(drug: str, terms: list[str]) -> list[dict]:
        assert "alcohol" in [t.lower() for t in terms]
        return [{
            "subject_drug": drug,
            "title": "warfarin labeling",
            "interactions_text": (
                "Patients should avoid alcohol while taking warfarin "
                "because of increased bleeding risk."
            ),
            "food_interactions_text": "",
            "setid": "set-warfarin-1",
            "url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=set-warfarin-1",
        }]

    monkeypatch.setattr(avoid.tools, "openfda_substance_check", fake_check)
    items = await lookup(["alcohol"], drugs=[_drug("warfarin")])
    assert len(items) == 1
    hit = items[0]
    assert hit.substance == "alcohol"
    assert hit.medications == ["warfarin"]
    assert "alcohol" in hit.note.lower()
    assert "skipped" not in hit.note.lower()
    assert [c.identifier for c in hit.citations] == ["set-warfarin-1"]
    assert all(c.source == "openfda" for c in hit.citations)
    assert all(c.url and "set-warfarin-1" in c.url for c in hit.citations)


@pytest.mark.asyncio
async def test_unmapped_or_empty_records_are_honest_empty(monkeypatch):
    async def fake_check(_drug: str, _terms: list[str]) -> list[dict]:
        return [{
            "subject_drug": "warfarin",
            "title": "warfarin labeling",
            "interactions_text": "Avoid alcohol.",
            "food_interactions_text": "",
            "setid": None,
            "url": None,
        }]

    monkeypatch.setattr(avoid.tools, "openfda_substance_check", fake_check)
    items = await lookup(["alcohol"], drugs=[_drug("warfarin")])
    assert len(items) == 1
    assert items[0].citations == []
    assert items[0].medications == []
    assert items[0].note == EMPTY_NOTE.format(substance="alcohol")
    assert "skipped" not in items[0].note.lower()


@pytest.mark.asyncio
async def test_no_label_hit_does_not_invent_herbal_ddi(monkeypatch):
    called: list[tuple[str, tuple[str, ...]]] = []

    async def fake_check(drug: str, terms: list[str]) -> list[dict]:
        called.append((drug, tuple(terms)))
        return []

    monkeypatch.setattr(avoid.tools, "openfda_substance_check", fake_check)
    items = await lookup(["kava"], drugs=[_drug("warfarin"), _drug("amiodarone")])
    assert called
    assert all(t[1] == ("kava",) for t in called)
    assert len(items) == 1
    assert items[0].substance == "kava"
    assert items[0].citations == []
    assert items[0].medications == []
    assert "interaction" in items[0].note.lower()
    assert "kava" in items[0].note.lower()
    blob = items[0].note.lower()
    assert "hepatotox" not in blob
    assert "cyp" not in blob
    assert "skipped" not in blob


@pytest.mark.asyncio
async def test_record_without_substance_mention_is_not_a_hit(monkeypatch):
    async def fake_check(_drug: str, _terms: list[str]) -> list[dict]:
        return [{
            "subject_drug": "warfarin",
            "title": "warfarin labeling",
            "interactions_text": "May interact with amiodarone via CYP2C9.",
            "food_interactions_text": "",
            "setid": "other-1",
            "url": "https://example.test/other-1",
        }]

    monkeypatch.setattr(avoid.tools, "openfda_substance_check", fake_check)
    items = await lookup(["alcohol"], drugs=[_drug("warfarin")])
    assert items[0].citations == []
    assert items[0].note == EMPTY_NOTE.format(substance="alcohol")


@pytest.mark.asyncio
async def test_wine_searches_alcohol_label_terms(monkeypatch):
    seen: list[list[str]] = []

    async def fake_check(_drug: str, terms: list[str]) -> list[dict]:
        seen.append(terms)
        return []

    monkeypatch.setattr(avoid.tools, "openfda_substance_check", fake_check)
    await lookup(["wine"], drugs=[_drug("metronidazole")])
    assert seen
    lowered = {t.lower() for t in seen[0]}
    assert "alcohol" in lowered
    assert "ethanol" in lowered


@pytest.mark.asyncio
async def test_no_non_drugs_skips_lookup(monkeypatch):
    async def boom(*_a, **_k):
        raise AssertionError("openFDA should not run when extract has no non-drugs")

    monkeypatch.setattr(avoid.tools, "openfda_substance_check", boom)
    assert await lookup([], drugs=[_drug("warfarin")]) == []
    assert await lookup(None, drugs=[_drug("warfarin")]) == []


@pytest.mark.asyncio
async def test_non_drugs_never_enter_pair_checking(monkeypatch, patch_seams):
    prefilter_names: list[str] = []
    waterfall_pairs: list = []

    async def fake_extract(_text: str) -> dict:
        return {
            "drugs": [
                {"name": "warfarin", "dose": "5 mg", "timing": None},
                {"name": "amiodarone", "dose": None, "timing": None},
            ],
            "non_drugs": ["alcohol", "grapefruit"],
            "patient_context": None,
        }

    async def fake_norm(names: list[str]):
        return [_drug(n) for n in names], []

    async def fake_prefilter(items):
        prefilter_names.extend(
            c.lower() for d in items for c in d.components
        )
        return [], [("amiodarone", "warfarin")]

    async def fake_waterfall(pairs, patient_ctx=None):
        waterfall_pairs.extend(pairs)
        return []

    async def fake_check(_drug: str, _terms: list[str]) -> list[dict]:
        return []

    patch_seams(extract=fake_extract, normalize=fake_norm,
                prefilter=fake_prefilter, waterfall=fake_waterfall)
    monkeypatch.setattr(avoid.tools, "openfda_substance_check", fake_check)

    resp = await service.run_check(CheckRequest(
        text="warfarin, amiodarone, alcohol, grapefruit juice",
    ))
    assert resp.disclaimer == DISCLAIMER
    blob = " ".join(prefilter_names)
    assert "alcohol" not in blob
    assert "grapefruit" not in blob
    flat = " ".join(a + " " + b for a, b in waterfall_pairs).lower()
    assert "alcohol" not in flat
    assert "grapefruit" not in flat
    substances = [a.substance for a in resp.avoid_with_medications]
    assert substances == ["alcohol", "grapefruit"]
    for item in resp.avoid_with_medications:
        assert item.citations == []
        assert "skipped" not in item.note.lower()
        assert "pair checking" not in item.note.lower()
