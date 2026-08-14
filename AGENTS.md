# AGENTS.md — DDI Explorer

Clinician-facing **drug–drug interaction (DDI)** checker. A clinician pastes a
prescription (or uploads a photo) plus optional patient/timing notes. The app
returns evidence-graded pair results with citations. It is decision support,
not a replacement for a clinical pharmacist or approved labeling.

This file is the project map for humans and coding agents. Design history and
locked product rules live in `PLAN.md`. Keep both in sync when behavior changes.

---

## What it does

1. Read typed text and/or prescription images.
2. Strip PHI **before any LLM sees the text**.
3. Extract drugs, doses, schedules, non-drugs, and patient context.
4. Normalize names: Indian brands (~242K local) → RxNorm generics, including
   FDC/combination splits.
5. Check every component pair:
   - known pairs via RxNav (Grade A, no LLM tokens)
   - unknown pairs via an early-exit evidence waterfall
6. Assemble results: contraindication banner, timing-manageable pairs,
   “avoid with medications”, patient-specific notes, citations, disclaimer.

---

## Repo layout

```
.
├── AGENTS.md                 ← this file
├── PLAN.md                   locked design decisions
├── README.md                 human quick start
├── .gitignore                ignores .env, .venv, node_modules
├── backend/                  FastAPI app (Python 3.14 venv at repo-root `.venv`)
│   ├── .env                  LIVE KEYS — never commit
│   ├── .env.example          placeholders only
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py           routes
│   │   ├── service.py        /api/check orchestrator
│   │   ├── models.py         request/response contract
│   │   ├── config.py         pydantic-settings
│   │   ├── data/indian_drugs.json   brand → composition(s)
│   │   ├── pipeline/         deterministic, no LLM
│   │   │   ├── ocr.py
│   │   │   ├── scrubber.py
│   │   │   ├── normalize.py
│   │   │   └── prefilter.py
│   │   └── agent/            LLM + evidence tools
│   │       ├── llm.py        litellm adapter
│   │       ├── extract.py    first LLM call (drugs from scrubbed text)
│   │       ├── tools.py      openFDA, PubMed, CT.gov, Firecrawl
│   │       └── waterfall.py  per-pair A→B→C early exit
│   ├── scripts/fetch_indian_dataset.py
│   └── tests/                pytest (offline: normalize + scrubber)
└── frontend/                 React 19 + Vite 8 + TypeScript
    └── src/
        ├── App.tsx           single-screen UI
        └── api.ts            POST /api/check
```

No auth, no database. Backend is stateless. Any history would be
browser-only (`localStorage` is planned; not required today).

---

## Stack

| Layer | Choice |
|---|---|
| Backend | FastAPI, uvicorn, pydantic v2, httpx |
| Frontend | React + Vite + TypeScript; Vite proxies `/api` → `localhost:8000` |
| LLM | litellm (provider-agnostic). Current `.env.example` default: `anthropic/claude-sonnet-5` |
| Vision / OCR fallback | Separate `VISION_MODEL` (Claude or Gemini). DeepSeek is text-only |
| PHI | Microsoft Presidio + spaCy `en_core_web_lg` + Indian-specific regex |
| Brands | Local `indian_drugs.json` (rapidfuzz) then RxNav/RxNorm |
| Evidence | openFDA labels, PubMed eutils, ClinicalTrials.gov v2, Firecrawl |
| OCR | Tesseract if installed; else vision-LLM |

`PLAN.md` still mentions DeepSeek as the original default and PubMed/CT.gov
as “MCP servers”. The running code uses REST wrappers in `agent/tools.py`.

---

## Pipeline (`POST /api/check`)

Implemented in `backend/app/service.py`. Order is load-bearing.

```
images ─▶ OCR (tesseract if confidence ≥ 0.75, else vision)
text / patient_context / timing
        ▼
Presidio scrub          ← first (and last) gate before any reasoning LLM
        ▼
LLM extract             drugs[], non_drugs[], patient_context
        ▼
normalize               Indian dataset → RxNav → unresolved
        ▼
RxNav pre-filter        known pairs → Grade A (source_tier="local")
        ▼
waterfall               unknown pairs only, early exit
        ▼
assemble                banner, avoid-with, insufficient, disclaimer
```

### 1. OCR — `pipeline/ocr.py`

- Input: base64 data-URLs.
- Tesseract first when `tesseract` is on `PATH`.
- Vision fallback if missing, exception, or mean confidence below
  `OCR_CONFIDENCE_THRESHOLD` (default 0.75).
- Vision currently receives the raw image (image-level redaction is v2).
  Output text is still scrubbed before extraction.

### 2. Scrub — `pipeline/scrubber.py`

Must run on **all** text (typed + OCR + patient + timing) before extract.

Removes: names, phones, emails, Aadhaar, ABHA, PAN, labelled MRN/UHID, labelled
DOB, NER PERSON/LOCATION/ORG (with guards).

**Preserves:** drug names (Indian brands are often false-positive PERSONs —
never redact a dataset hit or a `Tab/Cap/Inj` span), clinical numbers
(age, CrCl, eGFR), relative dosing (`1-0-1`, `3 months`).

Placeholders look like `[PERSON_1]`, `[MRN_1]`. Extraction is told to ignore them.

### 3. Extract — `agent/extract.py`

One JSON completion. Input is already de-identified. Returns
`{drugs: [{name, dose, timing}], non_drugs, patient_context}`.
Keep strength on the name (`Telma 40`, `Dolo 650`) — brand resolution needs it.

### 4. Normalize — `pipeline/normalize.py`

Per name, cap `MAX_DRUGS_PER_REQUEST` (default 15):

1. Indian dataset exact → word-bounded prefix (not `dolo`→`dolonex`) →
   known generic (so `amlodipine` stays a single ingredient) → fuzzy
   (rapidfuzz ≥ 88). Prefer fewer components so a plain brand does not
   become an FDC; unspecified form prefers tablet/capsule over drops.
2. Else RxNav `/approximateTerm` → RxCUI + RxNorm name.
3. Else mark unresolved.

`split_components` turns `amoxycillin (500mg) / clavulanic acid (125mg)` into
`["amoxycillin", "clavulanic acid"]`. Strengths like `(100mg/ml)` are stripped
before the `/` split so `ml` is never a component. Every *component* pair
across different input drugs is checked — an FDC therefore adds pairs.

### 5. Pre-filter — `pipeline/prefilter.py`

RxNav interaction list for pairs that have two distinct RxCUIs. Hits become
Grade A, `source_tier="local"`, and never enter the waterfall.

### 6. Waterfall — `agent/waterfall.py`

Per unresolved pair, stop at the first tier that returns records:

| Step | Tool | Grade if interaction |
|---|---|---|
| 1 | `openfda_label_check` | A |
| 2 | PubMed eutils + ClinicalTrials.gov (RCT / PK / meta / systematic review) | B |
| 3 | leftover PubMed + Firecrawl case-report search | C |
| 4 | nothing | no DDI (`grade=null`, `category=none`) |

Hard rules (also in the synthesizer prompt):

- Cite **only** records the tools just retrieved. No record → no citation → no graded claim.
- `verdict` is `interaction` | `none` | `insufficient`. Only `interaction` gets a grade.
- Conflict (case report vs absent/negative trial) → Grade C + `evidence_conflict`.
- Unknown dose → conditional `dose_condition` (`DDI possible if … > X mg`).
- Separable admin (e.g. cations + levothyroxine) → `category=timing`.
- Contraindicated → `category=contraindicated` (also copied into the banner).
- Patient context present → `patient_specific_note`. Absent → `severe_if[]`.

Pairs run with `PAIR_CONCURRENCY` (default 5). One pair failing must not fail
the whole request (`source_tier="error"`).

Non-drugs (alcohol, tobacco, grapefruit, herbals) skip pair-checking and land
in `avoid_with_medications`.

---

## Evidence grades

| Grade | Meaning | Typical source |
|---|---|---|
| **A** | Approved / known DDI | openFDA label “Drug Interactions”, RxNav |
| **B** | Human trial literature, not in labeling | PubMed RCT/PK/meta, ClinicalTrials.gov |
| **C** | Weak evidence | Case reports, in-vitro, mechanistic only |
| — | No DDI / insufficient | Explicit; never silent extrapolation |

---

## HTTP API

Defined in `backend/app/main.py` and `models.py`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | `{status, tesseract}` |
| `POST` | `/api/check` | Full pipeline |
| `POST` | `/api/ocr` | Image → scrubbed text (UI preview) |

`CheckRequest`: `{ text?, images[]?, patient_context?, timing? }`
(`images` are data-URLs).

`CheckResponse`: `scrubbed_text`, `normalized_drugs`, `unresolved_drugs`,
`pairs`, `contraindicated_banner`, `avoid_with_medications`,
`insufficient_evidence`, `disclaimer`.

Frontend: one screen in `frontend/src/App.tsx`. Submit disabled until there is
text or an image. Results sort contraindicated → A → B → C. CORS allows
`http://localhost:5173` only.

---

## Configuration and secrets

`backend/app/config.py` reads `backend/.env` via pydantic-settings.

| Variable | Role |
|---|---|
| `LLM_MODEL` | litellm model string |
| `ANTHROPIC_API_KEY` / `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` | reasoning key for the chosen provider |
| `VISION_MODEL` | OCR fallback only |
| `GEMINI_API_KEY` | only if vision is Gemini |
| `FIRECRAWL_API_KEY` | Grade C web search/fetch |
| `OPENFDA_API_KEY` | optional; works without at lower rate limits |
| `OCR_CONFIDENCE_THRESHOLD` | default 0.75 |
| `MAX_DRUGS_PER_REQUEST` | default 15 |
| `PAIR_CONCURRENCY` | default 5 |
| `HTTP_TIMEOUT` | default 30s |

**Never commit `backend/.env`.** Only `.env.example` (placeholders) is in git.

**Startup gotcha:** pydantic loads keys into `Settings`, but `agent/llm.py`
calls litellm without passing `api_key`. LiteLLM reads **process env**.
`uvicorn` from `backend/` without exporting `.env` → 500
`Missing Anthropic API Key`. Start with:

```bash
cd backend && set -a && source .env && set +a
../.venv/bin/uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

If you fix this, pass `settings.anthropic_api_key` (or the matching provider
key) into `litellm.acompletion` so a plain `uvicorn` works.

---

## Commands

Repo-root venv: `.venv` (Python 3.14).

```bash
# backend
source .venv/bin/activate
pip install -r backend/requirements.txt
python -m spacy download en_core_web_lg
cd backend && cp .env.example .env   # fill real keys
set -a && source .env && set +a
../.venv/bin/uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

# frontend (second terminal)
cd frontend && npm install && npm run dev   # http://127.0.0.1:5173

# tests (offline; no network / no LLM)
cd backend && ../.venv/bin/python -m pytest tests/ -q

# optional local OCR
brew install tesseract

# rebuild Indian brand index
python backend/scripts/fetch_indian_dataset.py
```

A full `/api/check` is ~30–40s (extract + normalize + at least one evidence
tier). Do not treat that latency as a hang.

---

## Tests and invariants

`backend/tests/test_normalize.py` — Indian exact/fuzzy/FDC split, unknown names.

`backend/tests/test_scrubber.py` — PHI gone, clinical context kept, brands not
redacted, `1-0-1` / relative time survive.

When changing scrub or normalize, extend these tests. Do not add tests that
need live API keys.

Privacy invariant: **no reasoning LLM call on unscrubbed text.** Vision may
see a raw image today; its transcript is still scrubbed.

Citation invariant: a graded `interaction` must map `cited` identifiers back
to tool records (`waterfall._citations_from`). Invented PMIDs are a bug.

---

## Editing conventions

- Keep the pipeline order in `service.py`. Do not let extract or waterfall
  run before `scrubber.scrub_text`.
- New evidence sources belong in `agent/tools.py` and a waterfall tier — not
  ad-hoc fetches inside the synthesizer prompt.
- `models.py` is the contract with the frontend. Change `api.ts` in the same PR.
- Comments: short, factual, only for non-obvious constraints (PHI guards,
  brand-as-PERSON, early-exit). No changelog comments.
- Do not add Markdown files the user did not ask for.
- Do not commit `.env`, `.venv`, `node_modules`, or `__pycache__`.
- Indian dataset is ~14MB JSON; do not regenerate unless the fetch script
  or source data changed.

---

## Known gaps (do not “fix” unless asked)

- LiteLLM key wiring (export `.env` or pass key explicitly) — see above.
- Tesseract often absent; health reports `"tesseract": false`; vision fallback used.
- Image-level PHI redaction before vision is not implemented.
- Duplicate-therapy detection (two NSAIDs) omitted on purpose (PLAN §6 #8).
- No pair-result cache, rate limits, or golden e2e DDI eval set (PLAN P8).
- Frontend has no matrix view yet (PLAN mentioned it).
- `PLAN.md` defaults (DeepSeek, Gemini vision, MCP tools) have drifted from
  the running `.env.example` / REST tools.

---

## Product tone

Output is for clinicians. Keep summaries mechanism-first, cite sources, show
uncertainty (`insufficient`, conflicts) instead of guessing. The disclaimer in
`service.py` must stay on every `CheckResponse`.
