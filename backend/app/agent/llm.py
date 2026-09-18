"""Provider-agnostic LLM access via litellm.

Default reasoning model: anthropic/claude-sonnet-5 (set LLM_MODEL to swap).
Vision is a SEPARATE setting — DeepSeek is text-only.
"""
from __future__ import annotations

import json
import os

import litellm

from ..config import Settings, get_settings
from ..pipeline import scrubber

litellm.drop_params = True  # tolerate provider-specific param mismatches


class LLMConfigError(RuntimeError):
    """Missing provider key — fail immediately, do not hang on the vendor."""


def _provider_key(model: str, settings: Settings) -> tuple[str, str | None]:
    """Return (env var name, key from settings) for a litellm model string."""
    m = (model or "").lower()
    if m.startswith("anthropic/") or m.startswith("claude"):
        return "ANTHROPIC_API_KEY", settings.anthropic_api_key
    if m.startswith("deepseek"):
        return "DEEPSEEK_API_KEY", settings.deepseek_api_key
    if m.startswith("openai/") or m.startswith("gpt-"):
        return "OPENAI_API_KEY", settings.openai_api_key
    if m.startswith("gemini") or m.startswith("google"):
        return "GEMINI_API_KEY", settings.gemini_api_key
    return "", None


def _auth_kwargs(model: str, settings: Settings) -> dict:
    """Pass the settings key into LiteLLM; do not rely on process env."""
    env_name, key = _provider_key(model, settings)
    if env_name and not key:
        raise LLMConfigError(
            f"No API key configured for {model}. "
            f"Set {env_name} in backend/.env — uvicorn loads it; no shell export needed."
        )
    extra: dict = {}
    if key:
        extra["api_key"] = key
        if env_name and not os.environ.get(env_name):
            os.environ[env_name] = key
    return extra


def parse_json_object(text: str | None) -> dict:
    """First JSON object in model output, or {}."""
    if not text:
        return {}
    try:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end < start:
            return {}
        data = json.loads(text[start:end + 1])
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _gate_reasoning_messages(messages: list[dict]) -> list[dict]:
    """Last-chance PHI strip. Regex-only so pair synthesizers skip spaCy."""
    gated = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            gated.append({**m, "content": scrubber.for_reasoning_llm(
                content, use_ner=False) or ""})
        else:
            gated.append(m)
    return gated


async def complete(messages: list[dict], *, model: str | None = None,
                   temperature: float = 0.0, max_tokens: int = 2000,
                   response_format: dict | None = None) -> str:
    """Plain chat completion. Returns assistant text.

    Every string in `messages` is scrubbed before it leaves this function.
    Vision stays on complete_vision (image-level redaction is not v1).
    """
    settings = get_settings()
    chosen = model or settings.llm_model
    kwargs: dict = {
        "model": chosen,
        "messages": _gate_reasoning_messages(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
        **_auth_kwargs(chosen, settings),
    }
    if response_format:
        kwargs["response_format"] = response_format
    try:
        resp = await litellm.acompletion(**kwargs)
    except Exception as e:
        if type(e).__name__ == "AuthenticationError" or "authentication" in str(e).lower():
            env_name, _ = _provider_key(chosen, settings)
            raise LLMConfigError(
                f"Provider rejected the API key for {chosen}. "
                f"Update {env_name or 'the provider key'} in backend/.env and retry."
            ) from e
        raise
    return resp.choices[0].message.content or ""


async def complete_vision(prompt: str, image_data_urls: list[str]) -> str:
    """Vision completion — OCR fallback path only."""
    settings = get_settings()
    content: list[dict] = [{"type": "text", "text": prompt}]
    content += [{"type": "image_url", "image_url": {"url": u}} for u in image_data_urls]
    chosen = settings.vision_model
    try:
        resp = await litellm.acompletion(
            model=chosen,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            max_tokens=3000,
            **_auth_kwargs(chosen, settings),
        )
    except Exception as e:
        if type(e).__name__ == "AuthenticationError" or "authentication" in str(e).lower():
            env_name, _ = _provider_key(chosen, settings)
            raise LLMConfigError(
                f"Provider rejected the API key for {chosen}. "
                f"Update {env_name or 'the provider key'} in backend/.env and retry."
            ) from e
        raise
    return resp.choices[0].message.content or ""
