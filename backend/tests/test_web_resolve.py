"""Web verification for dataset+RxNav misses — offline, no live keys."""
from __future__ import annotations

import pytest

from app import service
from app.agent import web_resolve
from app.models import CheckRequest, NormalizedDrug
from app.service import DISCLAIMER

PHONE = "9876543210"
MRN = "AIIMS-20451"


def _miss(name: str, *, dose: str | None = None,
          schedule: str | None = None) -> NormalizedDrug:
    return NormalizedDrug(
        input_name=name, dose=dose, schedule=schedule, resolved_via=None,
    )


async def _no_rxcui(_client, names):
    return {}


@pytest.mark.asyncio
async def test_dataset_rxnav_miss_resolves_via_web(monkeypatch):
    searches: list[str] = []

    async def fake_search(query: str, *, limit: int = 5):
        searches.append(query)
        return [{
            "title": "Telmafoo composition",
            "url": "https://example.test/telmafoo",
            "snippet": "Telmafoo contains telmisartan 40 mg",
        }]

    async def fake_fetch(url: str) -> str:
        assert url == "https://example.test/telmafoo"
        return "Telmafoo tablet composition: telmisartan 40 mg. Used for hypertension."

    async def fake_complete(messages, **_k) -> str:
        return '{"generics": ["telmisartan"]}'

    monkeypatch.setattr(web_resolve.tools, "web_search", fake_search)
    monkeypatch.setattr(web_resolve.tools, "web_fetch", fake_fetch)
    monkeypatch.setattr(web_resolve.llm, "complete", fake_complete)
    monkeypatch.setattr(web_resolve.norm, "_rxcuis_for_components", _no_rxcui)

    out, still = await web_resolve.apply([_miss("Telmafoo 40")])
    assert still == []
    d = out[0]
    assert d.resolved_via == "agent_web"
    assert d.components == ["telmisartan"]
    assert searches
    assert "Telmafoo" in searches[0] or "telmafoo" in searches[0].lower()
    assert "composition" in searches[0]


@pytest.mark.asyncio
async def test_invented_generic_not_in_page_is_dropped(monkeypatch):
    async def fake_search(query: str, *, limit: int = 5):
        return [{
            "title": "Brand X",
            "url": "https://example.test/x",
            "snippet": "contains telmisartan",
        }]

    async def fake_fetch(_url: str) -> str:
        return "Brand X contains telmisartan 40 mg only."

    async def fake_complete(messages, **_k) -> str:
        return '{"generics": ["telmisartan", "ibuprofen"]}'

    monkeypatch.setattr(web_resolve.tools, "web_search", fake_search)
    monkeypatch.setattr(web_resolve.tools, "web_fetch", fake_fetch)
    monkeypatch.setattr(web_resolve.llm, "complete", fake_complete)
    monkeypatch.setattr(web_resolve.norm, "_rxcuis_for_components", _no_rxcui)

    out, still = await web_resolve.apply([_miss("BrandX")])
    assert still == []
    assert out[0].components == ["telmisartan"]
    assert "ibuprofen" not in out[0].components


@pytest.mark.asyncio
async def test_queries_use_scrubbed_names_only(monkeypatch):
    searches: list[str] = []
    prompts: list[str] = []

    async def fake_search(query: str, *, limit: int = 5):
        searches.append(query)
        return []

    async def fake_fetch(_url: str) -> str:
        raise AssertionError("no fetch without search hits")

    async def fake_complete(messages, **_k) -> str:
        prompts.append(str(messages))
        return '{"generics": []}'

    monkeypatch.setattr(web_resolve.tools, "web_search", fake_search)
    monkeypatch.setattr(web_resolve.tools, "web_fetch", fake_fetch)
    monkeypatch.setattr(web_resolve.llm, "complete", fake_complete)

    dirty = f"mysterybrand {PHONE} MRN: {MRN}"
    out, still = await web_resolve.apply([_miss(dirty)])
    assert still == [dirty]
    assert out[0].components == []
    assert out[0].resolved_via is None
    blob = " ".join(searches)
    assert PHONE not in blob
    assert MRN not in blob
    assert "mysterybrand" in blob.lower()
    assert prompts == []  # no LLM when nothing was retrieved


@pytest.mark.asyncio
async def test_echoing_the_brand_is_not_a_resolution(monkeypatch):
    async def fake_search(query: str, *, limit: int = 5):
        return [{
            "title": "Telmafoo", "url": "https://example.test/t",
            "snippet": "Telmafoo tablets",
        }]

    async def fake_fetch(_url: str) -> str:
        return "Telmafoo is a tablet sold in India."

    async def fake_complete(messages, **_k) -> str:
        return '{"generics": ["telmafoo"]}'

    monkeypatch.setattr(web_resolve.tools, "web_search", fake_search)
    monkeypatch.setattr(web_resolve.tools, "web_fetch", fake_fetch)
    monkeypatch.setattr(web_resolve.llm, "complete", fake_complete)

    out, still = await web_resolve.apply([_miss("Telmafoo")])
    assert still == ["Telmafoo"]
    assert out[0].resolved_via is None
    assert out[0].components == []


@pytest.mark.asyncio
async def test_no_web_when_already_resolved(monkeypatch):
    async def boom(*_a, **_k):
        raise AssertionError("web search must not run for resolved names")

    monkeypatch.setattr(web_resolve.tools, "web_search", boom)
    resolved = NormalizedDrug(
        input_name="warfarin", generic_name="warfarin",
        components=["warfarin"], resolved_via="rxnav",
    )
    out, still = await web_resolve.apply([resolved])
    assert still == []
    assert out[0].resolved_via == "rxnav"


@pytest.mark.asyncio
async def test_keeps_dose_schedule_on_web_hit(monkeypatch):
    async def fake_search(query: str, *, limit: int = 5):
        return [{
            "title": "page", "url": "https://example.test/p",
            "snippet": "contains amlodipine",
        }]

    async def fake_fetch(_url: str) -> str:
        return "Contains amlodipine 5 mg."

    async def fake_complete(messages, **_k) -> str:
        return '{"generics": ["amlodipine"]}'

    monkeypatch.setattr(web_resolve.tools, "web_search", fake_search)
    monkeypatch.setattr(web_resolve.tools, "web_fetch", fake_fetch)
    monkeypatch.setattr(web_resolve.llm, "complete", fake_complete)
    monkeypatch.setattr(web_resolve.norm, "_rxcuis_for_components", _no_rxcui)

    out, still = await web_resolve.apply([
        _miss("Amlongfoo", dose="5 mg", schedule="1-0-0"),
    ])
    assert still == []
    assert out[0].dose == "5 mg"
    assert out[0].schedule == "1-0-0"
    assert out[0].resolved_via == "agent_web"


@pytest.mark.asyncio
async def test_check_web_resolved_name_enters_prefilter_not_unresolved(monkeypatch):
    prefilter_names: list[str] = []

    async def fake_extract(_text: str) -> dict:
        return {
            "drugs": [
                {"name": "Telmafoo", "dose": "40 mg", "timing": "1-0-0"},
                {"name": "warfarin", "dose": "5 mg", "timing": None},
            ],
            "non_drugs": [],
            "patient_context": None,
        }

    async def fake_norm(names: list[str]):
        table = {
            "Telmafoo": _miss("Telmafoo", dose="40 mg", schedule="1-0-0"),
            "warfarin": NormalizedDrug(
                input_name="warfarin", generic_name="warfarin",
                components=["warfarin"], resolved_via="rxnav",
            ),
        }
        drugs = [table[n] for n in names]
        still = [n for n in names if not table[n].components]
        return drugs, still

    async def fake_search(query: str, *, limit: int = 5):
        return [{
            "title": "Telmafoo", "url": "https://example.test/t",
            "snippet": "telmisartan",
        }]

    async def fake_fetch(_url: str) -> str:
        return "Telmafoo: telmisartan 40 mg tablets."

    async def fake_complete(messages, **_k) -> str:
        return '{"generics": ["telmisartan"]}'

    async def fake_prefilter(items):
        prefilter_names.extend(c for d in items for c in d.components)
        return [], []

    async def fake_waterfall(pairs, patient_ctx=None):
        return []

    async def fake_avoid(*_a, **_k):
        return []

    monkeypatch.setattr(service.extract, "extract_drugs", fake_extract)
    monkeypatch.setattr(service.norm, "normalize_drugs", fake_norm)
    monkeypatch.setattr(web_resolve.tools, "web_search", fake_search)
    monkeypatch.setattr(web_resolve.tools, "web_fetch", fake_fetch)
    monkeypatch.setattr(web_resolve.llm, "complete", fake_complete)
    monkeypatch.setattr(web_resolve.norm, "_rxcuis_for_components", _no_rxcui)
    monkeypatch.setattr(service.prefilter, "check_known_pairs", fake_prefilter)
    monkeypatch.setattr(service.waterfall, "evaluate_pairs", fake_waterfall)
    monkeypatch.setattr(service.avoid_mod, "lookup", fake_avoid)

    resp = await service.run_check(CheckRequest(text="Telmafoo, warfarin"))
    assert resp.disclaimer == DISCLAIMER
    assert resp.unresolved_drugs == []
    by = {d.input_name: d for d in resp.normalized_drugs}
    assert by["Telmafoo"].resolved_via == "agent_web"
    assert by["Telmafoo"].components == ["telmisartan"]
    assert by["Telmafoo"].dose == "40 mg"
    assert "telmisartan" in prefilter_names
    assert "warfarin" in prefilter_names
