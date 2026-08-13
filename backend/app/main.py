"""FastAPI entrypoint."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .models import CheckRequest, CheckResponse
from .pipeline import ocr, scrubber
from .service import run_check

app = FastAPI(title="DDI Explorer", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "tesseract": ocr.TESSERACT_AVAILABLE}


@app.post("/api/check", response_model=CheckResponse)
async def check(req: CheckRequest) -> CheckResponse:
    return await run_check(req)


@app.post("/api/ocr")
async def ocr_only(payload: dict) -> dict:
    """Standalone image -> scrubbed text (lets the UI preview before checking)."""
    result = await ocr.extract_text(payload["image"])
    scrubbed = scrubber.scrub_text(result["text"])
    return {"text": scrubbed.text, "engine": result["engine"],
            "confidence": result["confidence"],
            "redactions": len(scrubbed.redactions)}
