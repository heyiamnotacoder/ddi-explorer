"""LiteLLM must use pydantic settings keys, not process env alone."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.agent import llm
from app.config import Settings, _BACKEND_ROOT


def _settings(**over) -> SimpleNamespace:
    base = dict(
        llm_model="anthropic/claude-sonnet-5",
        vision_model="anthropic/claude-sonnet-5",
        anthropic_api_key="sk-ant-test",
        deepseek_api_key=None,
        openai_api_key=None,
        gemini_api_key=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


class _Msg:
    content = "{}"


class _Choice:
    message = _Msg()


class _Resp:
    choices = [_Choice()]


def test_settings_read_backend_dotenv_path():
    env_file = Settings.model_config["env_file"]
    assert env_file == _BACKEND_ROOT / ".env"
    assert env_file.name == ".env"
    assert env_file.parent.name == "backend"


def test_get_settings_rereads_when_env_mtime_changes(monkeypatch):
    from app import config

    first = config.Settings()
    config._cached = first
    config._cached_mtime = -1.0
    second = config.get_settings()
    assert second is not first
    assert config._cached is second


def test_auth_kwargs_maps_provider_keys():
    s = _settings(
        deepseek_api_key="sk-ds",
        openai_api_key="sk-oa",
        gemini_api_key="gem",
    )
    assert llm._auth_kwargs("anthropic/claude-sonnet-5", s)["api_key"] == "sk-ant-test"
    assert llm._auth_kwargs("deepseek/deepseek-v4-flash", s)["api_key"] == "sk-ds"
    assert llm._auth_kwargs("openai/gpt-4o", s)["api_key"] == "sk-oa"
    assert llm._auth_kwargs("gemini/gemini-2.0-flash", s)["api_key"] == "gem"
    assert llm._auth_kwargs("ollama/llama3", s) == {}


def test_missing_key_fails_before_network():
    s = _settings(anthropic_api_key=None)
    with pytest.raises(llm.LLMConfigError, match="ANTHROPIC_API_KEY"):
        llm._auth_kwargs("anthropic/claude-sonnet-5", s)


@pytest.mark.asyncio
async def test_complete_passes_settings_key_not_process_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(llm, "get_settings", lambda: _settings())
    seen: dict = {}

    async def fake_acompletion(**kwargs):
        seen.update(kwargs)
        return _Resp()

    monkeypatch.setattr(llm.litellm, "acompletion", fake_acompletion)
    await llm.complete([{"role": "user", "content": "hi"}])
    assert seen["api_key"] == "sk-ant-test"
    assert seen["model"] == "anthropic/claude-sonnet-5"


@pytest.mark.asyncio
async def test_complete_missing_key_does_not_call_vendor(monkeypatch):
    monkeypatch.setattr(
        llm, "get_settings",
        lambda: _settings(anthropic_api_key=None),
    )
    called: list = []

    async def fake_acompletion(**kwargs):
        called.append(kwargs)
        return _Resp()

    monkeypatch.setattr(llm.litellm, "acompletion", fake_acompletion)
    with pytest.raises(llm.LLMConfigError, match="backend/.env"):
        await llm.complete([{"role": "user", "content": "hi"}])
    assert called == []


@pytest.mark.asyncio
async def test_complete_maps_auth_error_to_config_error(monkeypatch):
    monkeypatch.setattr(llm, "get_settings", lambda: _settings())

    class AuthenticationError(Exception):
        pass

    async def fake_acompletion(**kwargs):
        raise AuthenticationError('{"type":"error","error":{"type":"authentication_error"}}')

    monkeypatch.setattr(llm.litellm, "acompletion", fake_acompletion)
    with pytest.raises(llm.LLMConfigError, match="ANTHROPIC_API_KEY"):
        await llm.complete([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_complete_vision_uses_vision_model_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(llm, "get_settings", lambda: _settings(
        vision_model="gemini/gemini-2.0-flash",
        gemini_api_key="gem-test",
    ))
    seen: dict = {}

    async def fake_acompletion(**kwargs):
        seen.update(kwargs)
        return _Resp()

    monkeypatch.setattr(llm.litellm, "acompletion", fake_acompletion)
    await llm.complete_vision("transcribe", ["data:image/png;base64,xx"])
    assert seen["model"] == "gemini/gemini-2.0-flash"
    assert seen["api_key"] == "gem-test"
