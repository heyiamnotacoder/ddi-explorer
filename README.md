# DDI Explorer

Evidence-graded drug–drug interaction explorer. Single linear AI agent with
early-exit waterfall (openFDA → PubMed/ClinicalTrials → web), PHI de-identification
before any LLM contact, OCR support for prescription photos, and Indian brand-name
resolution.

## Quick start

```bash
# 1. Backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
python -m spacy download en_core_web_lg
cd backend && cp .env.example .env   # fill in DEEPSEEK_API_KEY (+ GEMINI_API_KEY, FIRECRAWL_API_KEY)
uvicorn app.main:app --reload        # http://localhost:8000/docs

# 2. Frontend (new terminal)
cd frontend && npm install && npm run dev   # http://localhost:5173

# 3. Optional: local OCR (vision-LLM fallback works without it)
brew install tesseract
```

## Architecture

```
images ─▶ OCR (tesseract → vision fallback) ─┐
text / patient context / timing ─────────────┤
                                             ▼
                                     Presidio PHI scrub   ← nothing reaches an LLM before this
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
category; every claim requires a retrieved citation.

## Tests

```bash
cd backend && ../.venv/bin/python -m pytest tests/ -q
```

See `PLAN.md` for full design decisions and edge-case handling.
