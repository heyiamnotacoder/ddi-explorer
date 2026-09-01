"""Provider-agnostic LLM access via litellm.

Default reasoning model: deepseek-v4-flash (set LLM_MODEL to swap).
Vision model is a SEPARATE setting — DeepSeek is text-only.
"""
from __future__ import annotations

import litellm

from ..config import get_settings
from ..pipeline import scrubber

litellm.drop_params = True  # tolerate provider-specific param mismatches


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
    kwargs: dict = {
        "model": model or settings.llm_model,
        "messages": _gate_reasoning_messages(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format:
        kwargs["response_format"] = response_format
    resp = await litellm.acompletion(**kwargs)
    return resp.choices[0].message.content or ""


async def complete_vision(prompt: str, image_data_urls: list[str]) -> str:
    """Vision completion — OCR fallback path only."""
    settings = get_settings()
    content: list[dict] = [{"type": "text", "text": prompt}]
    content += [{"type": "image_url", "image_url": {"url": u}} for u in image_data_urls]
    resp = await litellm.acompletion(
        model=settings.vision_model,
        messages=[{"role": "user", "content": content}],
        temperature=0.0,
        max_tokens=3000,
    )
    return resp.choices[0].message.content or ""
