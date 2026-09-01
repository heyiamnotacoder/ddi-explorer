"""OCR stage: image -> raw text.

Hybrid strategy:
  1. Tesseract locally (free, PHI never leaves the box).
  2. If mean confidence < threshold -> vision-LLM fallback.

PRIVACY: whichever path produces the text, the caller MUST pass it through
scrubber.scrub_text() before any reasoning-LLM call. Honest v1: the vision
fallback may see the raw image. Image-level redaction is not v1. The
transcript is still scrubbed before extract.
"""
from __future__ import annotations

import base64
import io
import shutil

from ..config import get_settings

TESSERACT_AVAILABLE = shutil.which("tesseract") is not None

OCR_PROMPT = """Transcribe this medical document verbatim.
Output ONLY the extracted text — no commentary. Preserve structure:
drug names, doses, frequencies (e.g. 1-0-1), and any patient details.
If a region is illegible, write [illegible]."""


def _decode(data_url: str) -> bytes:
    _, _, b64 = data_url.partition(",")
    return base64.b64decode(b64)


def _tesseract(data_url: str) -> tuple[str, float]:
    """Returns (text, mean_confidence 0..1)."""
    import pytesseract
    from PIL import Image

    img = Image.open(io.BytesIO(_decode(data_url)))
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    words, confs = [], []
    for word, conf in zip(data["text"], data["conf"]):
        if word.strip():
            words.append(word)
            try:
                c = float(conf)
            except (TypeError, ValueError):
                c = -1
            if c >= 0:
                confs.append(c)
    mean = (sum(confs) / len(confs) / 100.0) if confs else 0.0
    return " ".join(words), mean


async def extract_text(data_url: str) -> dict:
    """Returns {'text': str, 'engine': 'tesseract'|'vision', 'confidence': float}."""
    from ..agent import llm  # local import: keeps vision path optional

    settings = get_settings()

    if TESSERACT_AVAILABLE:
        try:
            text, conf = _tesseract(data_url)
            if conf >= settings.ocr_confidence_threshold:
                return {"text": text, "engine": "tesseract", "confidence": conf}
        except Exception:  # noqa: BLE001 — fall through to vision
            text, conf = "", 0.0
    else:
        text, conf = "", 0.0

    vision_text = await llm.complete_vision(OCR_PROMPT, [data_url])
    return {"text": vision_text, "engine": "vision", "confidence": conf}
