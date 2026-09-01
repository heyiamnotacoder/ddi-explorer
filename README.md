# DDI Explorer

Evidence-graded drug–drug interaction explorer. Single linear AI agent with
early-exit waterfall (openFDA → PubMed/ClinicalTrials → web), PHI de-identification
before any reasoning LLM, OCR support for prescription photos, and Indian brand-name
resolution.

Default reasoning/vision model is Claude (`anthropic/claude-sonnet-5`). DeepSeek
and OpenAI are swaps via `LLM_MODEL`.

## Quick start

```bash
# 1. Backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
python -m spacy download en_core_web_lg
cd backend && cp .env.example .env   # fill ANTHROPIC_API_KEY (and FIRECRAWL_API_KEY)
../.venv/bin/uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
# keys come from backend/.env — no shell export needed

# 2. Frontend (new terminal)
cd frontend && npm install && npm run dev   # http://127.0.0.1:5173

# 3. Optional: local OCR (vision-LLM fallback works without it)
brew install tesseract
```

## Architecture

```
images ─▶ OCR (tesseract → vision fallback) ─┐
text / patient context / timing ─────────────┤
                                             ▼
                                     Presidio PHI scrub   ← nothing reaches a reasoning LLM before this
                                             ▼
                              LLM extraction (drugs, doses, timing)
                                             ▼
              normalize: Indian dataset (242K brands) → RxNav → FDC split
                                             ▼
        ┌─ known pairs ─▶ RxNav pre-filter (Grade A, zero LLM tokens)
        └─ unknown pairs ─▶ AGENT WATERFALL (early exit):
             openFDA → Grade A, stop
             PubMed + ClinicalTrials.gov → Grade B, stop
             web (case reports) → Grade C
             nothing → "No DDI found"
```

## Evidence grades

| Grade | Source |
|---|---|
| **A** | Approved labeling (openFDA / RxNav interaction data) |
| **B** | Human RCT / PK study / meta-analysis (PubMed, ClinicalTrials.gov) |
| **C** | Case reports / weak evidence only |

Contraindicated pairs surface as a banner; timing-manageable pairs get their own
category; every claim requires a retrieved citation. The results screen includes
a pair matrix.

v1 vision OCR may see the raw image; the transcript is still scrubbed. Image-level
redaction is not v1.

## Tests

Offline unit tests (no live API keys):

```bash
cd backend && ../.venv/bin/python -m pytest tests/ -q
```

A ~30-pair full live `/api/check` eval lives on a **branch off main**, not in
default pytest. See `PLAN.md` and `AGENTS.md` for locked decisions.
