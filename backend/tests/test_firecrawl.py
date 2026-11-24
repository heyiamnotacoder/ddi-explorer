"""Firecrawl v2 REST wrapper tests for web_search and web_fetch."""
from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from app.agent import tools


class _StubClient:
    def __init__(self, response: httpx.Response | Exception):
        self._response = response
        self.calls: list[dict] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_a):
        return False

    async def post(self, url: str, **kwargs) -> httpx.Response:
        self.calls.append({"url": url, **kwargs})
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _json_response(url: str, payload: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload, request=httpx.Request("POST", url))


@pytest.fixture
def _firecrawl_key(monkeypatch):
    monkeypatch.setattr(
        tools, "get_settings", lambda: SimpleNamespace(firecrawl_api_key="fc-test-key"),
    )


@pytest.mark.asyncio
async def test_web_search_uses_v2_endpoint_and_reads_data_web(monkeypatch, _firecrawl_key):
    url = "https://api.firecrawl.dev/v2/search"
    stub = _StubClient(_json_response(url, {
        "success": True,
        "data": {
            "web": [
                {
                    "title": "Warfarin and amiodarone case report",
                    "url": "https://example.test/case-1",
                    "description": "Elevated INR after co-administration.",
                },
            ],
        },
    }))
    monkeypatch.setattr(tools, "_client", lambda: stub)

    hits = await tools.web_search("warfarin amiodarone interaction")

    assert len(stub.calls) == 1
    assert stub.calls[0]["url"] == "https://api.firecrawl.dev/v2/search"
    assert stub.calls[0]["headers"] == {"Authorization": "Bearer fc-test-key"}
    assert stub.calls[0]["json"] == {
        "query": "warfarin amiodarone interaction",
        "limit": 5,
    }
    assert hits == [{
        "title": "Warfarin and amiodarone case report",
        "url": "https://example.test/case-1",
        "snippet": "Elevated INR after co-administration.",
    }]


@pytest.mark.asyncio
async def test_web_search_handles_empty_or_missing_web_results(monkeypatch, _firecrawl_key):
    url = "https://api.firecrawl.dev/v2/search"
    stub = _StubClient(_json_response(url, {"success": True, "data": {}}))
    monkeypatch.setattr(tools, "_client", lambda: stub)

    assert await tools.web_search("nothing here", limit=3) == []
    assert stub.calls[0]["json"] == {"query": "nothing here", "limit": 3}


@pytest.mark.asyncio
async def test_web_fetch_uses_v2_scrape_and_truncates_markdown(monkeypatch, _firecrawl_key):
    url = "https://api.firecrawl.dev/v2/scrape"
    long_markdown = "A" * 7000
    stub = _StubClient(_json_response(url, {
        "success": True,
        "data": {"markdown": long_markdown},
    }))
    monkeypatch.setattr(tools, "_client", lambda: stub)

    body = await tools.web_fetch("https://example.test/case-1")

    assert len(stub.calls) == 1
    assert stub.calls[0]["url"] == "https://api.firecrawl.dev/v2/scrape"
    assert stub.calls[0]["headers"] == {"Authorization": "Bearer fc-test-key"}
    assert stub.calls[0]["json"] == {
        "url": "https://example.test/case-1",
        "formats": ["markdown"],
        "onlyMainContent": True,
    }
    assert body == "A" * 6000


@pytest.mark.asyncio
async def test_firecrawl_returns_empty_without_api_key(monkeypatch):
    monkeypatch.setattr(
        tools, "get_settings", lambda: SimpleNamespace(firecrawl_api_key=None),
    )

    assert await tools.web_search("warfarin") == []
    assert await tools.web_fetch("https://example.test/case-1") == ""


@pytest.mark.asyncio
async def test_firecrawl_raises_rate_limit_on_429(monkeypatch, _firecrawl_key):
    async def _noop(*_a, **_k):
        return None

    monkeypatch.setattr(tools.asyncio, "sleep", _noop)
    stub = _StubClient(_json_response("https://api.firecrawl.dev/v2/search", {}, status=429))
    monkeypatch.setattr(tools, "_client", lambda: stub)

    with pytest.raises(tools.ToolRateLimit, match="firecrawl"):
        await tools.web_search("warfarin")
