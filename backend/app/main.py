"""FastAPI entrypoint."""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .agent.llm import LLMConfigError
from .config import get_settings
from .models import (
    AlternativesRequest,
    AlternativesResponse,
    CheckRequest,
    CheckResponse,
    OcrRequest,
)
from .pipeline import ocr, scrubber
from .service import run_alternatives, run_check

app = FastAPI(title="DDI Explorer", version="0.1.0")

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _settings.cors_origins.split(",") if o.strip()],
    allow_origin_regex=_settings.cors_origin_regex or None,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(LLMConfigError)
async def llm_config_error(_request: Request, exc: LLMConfigError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "tesseract": ocr.TESSERACT_AVAILABLE}


@app.post("/api/check", response_model=CheckResponse)
async def check(req: CheckRequest) -> CheckResponse:
    return await run_check(req)


@app.post("/api/alternatives", response_model=AlternativesResponse)
async def alternatives(req: AlternativesRequest) -> AlternativesResponse:
    """Second loop: which medicine to change, given the whole graded list."""
    return await run_alternatives(req)


@app.post("/api/ocr")
async def ocr_only(req: OcrRequest) -> dict:
    """Standalone image -> scrubbed text (lets the UI preview before checking)."""
    result = await ocr.extract_text(req.image)
    scrubbed = scrubber.scrub_text(result["text"])
    return {"text": scrubbed.text, "engine": result["engine"],
            "confidence": result["confidence"],
            "redactions": len(scrubbed.redactions)}
