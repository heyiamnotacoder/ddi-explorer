"""Shared LLM JSON-object parser — used by extract, waterfall, alternatives."""
import pytest

from app.agent.extract import extract_drugs
from app.agent.llm import parse_json_object


def test_parse_json_object_plain():
    assert parse_json_object('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_parse_json_object_fenced_and_prose():
    raw = 'Sure.\n```json\n{"verdict": "none", "cited": []}\n```\n'
    assert parse_json_object(raw) == {"verdict": "none", "cited": []}


def test_parse_json_object_garbage_is_empty():
    assert parse_json_object("") == {}
    assert parse_json_object(None) == {}
    assert parse_json_object("no braces here") == {}
    assert parse_json_object("{not json") == {}
    assert parse_json_object("[1, 2]") == {}


@pytest.mark.asyncio
async def test_extract_uses_shared_parser(monkeypatch):
    async def fake_complete(*_a, **_k):
        return 'Sure. {"drugs": [{"name": "warfarin"}], "non_drugs": [], "patient_context": null}'

    monkeypatch.setattr("app.agent.extract.llm.complete", fake_complete)
    data = await extract_drugs("warfarin 5 mg")
    assert data["drugs"][0]["name"] == "warfarin"
    assert data["non_drugs"] == []
