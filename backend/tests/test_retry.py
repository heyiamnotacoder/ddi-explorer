"""HTTP 429 retries on evidence GETs; errors stay uncached."""
from __future__ import annotations

import json

import httpx
import pytest

from app.agent import pair_cache, tools
from app.agent.waterfall import evaluate_pair, evaluate_pairs
from app.models import Category, Grade, NormalizedDrug
from app.pipeline.prefilter import check_known_pairs

FDA_URL = "https://api.fda.gov/drug/label.json"
FDA_HIT = {
    "results": [{
        "drug_interactions": [
            "Concomitant amiodarone potentiates warfarin. "
            "Omeprazole may increase warfarin INR."
        ],
        "set_id": "set-1",
    }],
}
RXNAV_HIT = {
    "fullInteractionTypeGroup": [{
        "fullInteractionType": [{
            "interactionPair": [{
                "description": "Warfarin may interact with amiodarone.",
                "severity": "moderate",
                "interactionConcept": [
                    {"minConceptItem": {"rxcui": "11289", "name": "Warfarin"}},
                    {"minConceptItem": {"rxcui": "703", "name": "Amiodarone"}},
                ],
            }],
        }],
    }],
}


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def _noop(*_a, **_k):
        return None

    monkeypatch.setattr("app.agent.tools.asyncio.sleep", _noop)


def _resp(status: int, url: str = FDA_URL, body=None) -> httpx.Response:
    kw: dict = {"request": httpx.Request("GET", url)}
    if body is not None:
        kw["json"] = body
    elif status == 200:
        kw["json"] = {}
    return httpx.Response(status, **kw)


class QueueClient:
    def __init__(self, responses: list):
        self._queue = list(responses)
        self.calls: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False

    def _next(self, method: str, url: str, kwargs: dict) -> httpx.Response:
        self.calls.append({"method": method, "url": url, **kwargs})
        if not self._queue:
            raise AssertionError("no stubbed HTTP responses left")
        item = self._queue.pop(0)
        if isinstance(item, httpx.Response):
            return item
        if isinstance(item, int):
            return _resp(item, url)
        status, body = item
        return _resp(status, url, body)

    async def get(self, url, **kwargs):
        return self._next("GET", url, kwargs)

    async def post(self, url, **kwargs):
        return self._next("POST", url, kwargs)


class RepeatClient(QueueClient):
    def __init__(self, item):
        super().__init__([])
        self._item = item

    def _next(self, method: str, url: str, kwargs: dict) -> httpx.Response:
        self.calls.append({"method": method, "url": url, **kwargs})
        item = self._item
        if isinstance(item, httpx.Response):
            return item
        if isinstance(item, int):
            return _resp(item, url)
        status, body = item
        return _resp(status, url, body)


class FnClient(QueueClient):
    def __init__(self, fn):
        super().__init__([])
        self._fn = fn

    def _next(self, method: str, url: str, kwargs: dict) -> httpx.Response:
        self.calls.append({"method": method, "url": url, **kwargs})
        return self._fn(url, kwargs)


def _patch_tools_client(monkeypatch, client):
    monkeypatch.setattr(tools, "_client", lambda: client)


def _patch_synth(monkeypatch, payload: dict):
    async def fake_complete(*_a, **_k):
        return json.dumps(payload)

    monkeypatch.setattr("app.agent.waterfall.llm.complete", fake_complete)


def _grade_a() -> dict:
    return {
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


def _drug(name: str, rxcui: str) -> NormalizedDrug:
    return NormalizedDrug(
        input_name=name, generic_name=name, rxcui=rxcui,
        components=[name], component_rxcuis={name: rxcui},
    )


def _patch_rxnav_client(monkeypatch, client_obj):
    class FakeAsyncClient:
        def __init__(self, *_a, **_k):
            pass

        async def __aenter__(self):
            return client_obj

        async def __aexit__(self, *_a):
            return False

    monkeypatch.setattr("app.pipeline.prefilter.httpx.AsyncClient", FakeAsyncClient)


@pytest.mark.asyncio
async def test_request_429_then_200_succeeds():
    client = QueueClient([429, (200, {"ok": True})])
    r = await tools.request_with_retry(
        client, "GET", FDA_URL, tool="openfda")
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_request_persistent_429_raises_tool_rate_limit():
    client = RepeatClient(429)
    with pytest.raises(tools.ToolRateLimit) as ei:
        await tools.request_with_retry(client, "GET", FDA_URL, tool="openfda")
    assert ei.value.tool == "openfda"
    assert len(client.calls) == 1 + tools._429_RETRIES


@pytest.mark.asyncio
async def test_404_is_not_retried():
    client = QueueClient([404, 200])
    r = await tools.request_with_retry(client, "GET", FDA_URL, tool="openfda")
    assert r.status_code == 404
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_5xx_is_retried_once():
    client = QueueClient([500, (200, {"ok": True}), 200])
    r = await tools.request_with_retry(client, "GET", FDA_URL, tool="openfda")
    assert r.status_code == 200
    assert len(client.calls) == 2

    client = QueueClient([503, 503, 200])
    r = await tools.request_with_retry(client, "GET", FDA_URL, tool="openfda")
    assert r.status_code == 503
    assert len(client.calls) == 1 + tools._5XX_RETRIES


@pytest.mark.asyncio
async def test_openfda_429_then_200_returns_records(monkeypatch):
    client = QueueClient([429, (200, FDA_HIT), (200, {"results": []})])
    _patch_tools_client(monkeypatch, client)

    hits = await tools.openfda_label_check("warfarin", "amiodarone")
    assert hits
    assert hits[0]["setid"] == "set-1"
    assert len(client.calls) == 3


@pytest.mark.asyncio
async def test_evaluate_pair_429_then_200_is_not_error(monkeypatch):
    client = QueueClient([429, (200, FDA_HIT), (200, {"results": []})])
    _patch_tools_client(monkeypatch, client)
    _patch_synth(monkeypatch, _grade_a())

    result = await evaluate_pair("warfarin", "amiodarone")
    assert result.source_tier != "error"
    assert result.grade == Grade.A
    assert result.source_tier == "openfda"
    assert [c.identifier for c in result.citations] == ["set-1"]


@pytest.mark.asyncio
async def test_persistent_429_one_pair_others_still_return(monkeypatch):
    def _route(url, kwargs):
        search = (kwargs.get("params") or {}).get("search", "")
        if "amiodarone" in search:
            return _resp(429, url)
        return _resp(200, url, FDA_HIT)

    client = FnClient(_route)
    _patch_tools_client(monkeypatch, client)
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
async def test_exhausted_429_is_not_cached(monkeypatch):
    client = RepeatClient(429)
    _patch_tools_client(monkeypatch, client)

    first = await evaluate_pair("warfarin", "amiodarone")
    n_after_first = len(client.calls)
    second = await evaluate_pair("warfarin", "amiodarone")
    assert first.source_tier == second.source_tier == "error"
    assert pair_cache.get("warfarin", "amiodarone") is None
    assert len(client.calls) == 2 * n_after_first
    assert n_after_first == 1 + tools._429_RETRIES


@pytest.mark.asyncio
async def test_prefilter_429_then_200_succeeds(monkeypatch):
    client = QueueClient([429, (200, RXNAV_HIT)])
    _patch_rxnav_client(monkeypatch, client)

    resolved, unknown = await check_known_pairs([
        _drug("warfarin", "11289"),
        _drug("amiodarone", "703"),
    ])
    assert len(client.calls) == 2
    assert unknown == []
    assert len(resolved) == 1
    assert tuple(sorted(resolved[0].drugs)) == ("amiodarone", "warfarin")
    assert resolved[0].grade == Grade.A
    assert resolved[0].source_tier == "local"
    assert resolved[0].category == Category.INTERACTION


@pytest.mark.asyncio
async def test_prefilter_429_exhausted_is_unresolved_not_known_empty(monkeypatch):
    client = RepeatClient(429)
    _patch_rxnav_client(monkeypatch, client)

    resolved, unknown = await check_known_pairs([
        _drug("warfarin", "11289"),
        _drug("amiodarone", "703"),
    ])
    assert len(client.calls) == 1 + tools._429_RETRIES
    assert resolved == []
    assert unknown == [("amiodarone", "warfarin")]
    assert all(p.source_tier != "local" for p in resolved)


@pytest.mark.asyncio
async def test_prefilter_404_is_not_retried_and_is_unresolved(monkeypatch):
    client = QueueClient([404, (200, RXNAV_HIT)])
    _patch_rxnav_client(monkeypatch, client)

    resolved, unknown = await check_known_pairs([
        _drug("warfarin", "11289"),
        _drug("amiodarone", "703"),
    ])
    assert len(client.calls) == 1
    assert resolved == []
    assert unknown == [("amiodarone", "warfarin")]
