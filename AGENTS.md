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
   label-backed “avoid with medications”, patient-specific notes, citations, disclaimer.
7. Optional second loop (button): rank which interacting drug is safest to
   change (prefer symptomatic / lower-ADR over disease-modifying), propose
   same-indication substitutes, recheck each vs the rest of the list.

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
│   │   │   ├── prefilter.py
│   │   │   ├── overlay.py    Grade A timing/dose/patient copy, no LLM
│   │   │   └── avoid.py      non-drug vs labels, mapped citations, no LLM
│   │   └── agent/            LLM + evidence tools
│   │       ├── llm.py        litellm adapter
│   │       ├── extract.py    first LLM call (drugs from scrubbed text)
│   │       ├── alternatives.py  second loop: which drug to change
│   │       ├── tools.py      openFDA, PubMed, CT.gov, Firecrawl (429 → pair error)
│   │       ├── pair_cache.py process-local pair results; no patient text
│   │       ├── waterfall.py  per-pair A→B→C early exit
│   │       └── web_resolve.py  dataset+RxNav misses via scrubbed web search
│   ├── scripts/fetch_indian_dataset.py
│   └── tests/                pytest (offline; no live API keys)
└── frontend/                 React 19 + Vite 8 + TypeScript
    └── src/
        ├── App.tsx           single-screen UI
        ├── api.ts            POST /api/check, /api/alternatives
        └── history.ts        localStorage: scrubbed CheckResponse snapshots
```

No auth, no database. Backend is stateless. The browser keeps up to five
**scrubbed** `/api/check` snapshots (drugs, grades, citations). Raw patient
notes, timing, images, and unscrubbed text are never written.

---

## Stack

| Layer | Choice |
|---|---|
| Backend | FastAPI, uvicorn, pydantic v2, httpx |
| Frontend | React + Vite + TypeScript; Vite proxies `/api` → `localhost:8000` |
| LLM | litellm. **Default `anthropic/claude-sonnet-5`.** DeepSeek / OpenAI are swaps via `LLM_MODEL` |
| Vision / OCR fallback | Separate `VISION_MODEL` (default Claude). DeepSeek is text-only. Gemini is a swap |
| PHI | Microsoft Presidio + spaCy `en_core_web_lg` + Indian-specific regex |
| Brands | Local `indian_drugs.json` (rapidfuzz) then RxNav/RxNorm |
| Evidence | REST wrappers in `agent/tools.py`: openFDA, PubMed eutils, ClinicalTrials.gov v2, Firecrawl. MCP is optional, not required |
| OCR | Tesseract if installed; else vision-LLM |

Where `PLAN.md` and the code disagree, **code wins**. Keep both files in sync.

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
normalize               Indian dataset → RxNav → web verification on misses
        ▼
RxNav pre-filter        known pairs → Grade A (source_tier="local")
        ▼
waterfall               unknown pairs only, early exit
        ▼
assemble                banner, label-backed avoid-with, insufficient, disclaimer
        ▼
[button] alternatives   rank replaceable drug → propose substitutes
                        → recheck vs leftover list (prefilter + waterfall)
```

### 1. OCR — `pipeline/ocr.py`

- Input: base64 data-URLs.
- Tesseract first when `tesseract` is on `PATH`.
- Vision fallback if missing, exception, or mean confidence below
  `OCR_CONFIDENCE_THRESHOLD` (default 0.75).
- **Honest v1:** vision may receive the raw image. Image-level redaction
  is not v1. Output text is still scrubbed before extraction.

### 2. Scrub — `pipeline/scrubber.py`

Must run on **all** text (typed + OCR + patient + timing) before extract.

Removes: names, phones, emails, Aadhaar, ABHA, PAN, labelled MRN/UHID, labelled
DOB, NER PERSON/LOCATION/ORG (with guards).

**Preserves:** drug names (Indian brands are often false-positive PERSONs —
never redact a dataset hit or a `Tab/Cap/Inj` span), clinical numbers
(age, CrCl, eGFR), relative dosing (`1-0-1`, `3 months`).

Placeholders look like `[PERSON_1]`, `[MRN_1]`. Extraction is told to ignore them.

`scrubber.for_reasoning_llm()` is the text that may enter extract, waterfall,
and alternatives. `llm.complete` re-scrubs prompt strings (regex-only).
Never pass raw `req.patient_context` into a reasoning call.

### 3. Extract — `agent/extract.py`

One JSON completion. Input is already de-identified. Returns
`{drugs: [{name, dose, timing}], non_drugs, patient_context}`.
Keep strength on the name (`Telma 40`, `Dolo 650`) — brand resolution needs it.
Dose and 1-0-1-style `timing` are copied onto `NormalizedDrug.dose` /
`NormalizedDrug.schedule` so they survive to the check response.

### 4. Normalize — `pipeline/normalize.py`

Per name, cap `MAX_DRUGS_PER_REQUEST` (default 15):

1. Indian dataset exact → word-bounded prefix (not `dolo`→`dolonex`) →
   known generic (so `amlodipine` stays a single ingredient) → fuzzy
   (rapidfuzz ≥ 88). Prefer fewer components so a plain brand does not
   become an FDC; unspecified form prefers tablet/capsule over drops.
2. Else RxNav `/approximateTerm` → RxCUI + RxNorm name.
3. Else `agent/web_resolve.py`: Firecrawl search/fetch on the **scrubbed**
   name (regex PHI only — NER off so unknown brands are not PERSON). The
   LLM may only return generics that appear in retrieved text.
   `resolved_via="agent_web"`. Still unresolved if nothing maps.
   DDI grades still require mapped citations downstream.

`split_components` turns `amoxycillin (500mg) / clavulanic acid (125mg)` into
`["amoxycillin", "clavulanic acid"]`. Strengths like `(100mg/ml)` are stripped
before the `/` split so `ml` is never a component. Every *component* pair
across different input drugs is checked — an FDC therefore adds pairs.
Each ingredient gets its own RxCUI (`component_rxcuis`); the first is also
copied to `rxcui` for display. A sibling's identifier is never reused.

### 5. Pre-filter — `pipeline/prefilter.py`

RxNav interaction list for pairs that have two distinct **component** RxCUIs.
Hits become Grade A, `source_tier="local"`, and never enter the waterfall.
Match on RxCUI so Indian spellings (`amoxycillin`) still hit.

A deterministic overlay (`pipeline/overlay.py`) then fills `category=timing`
(separable admin), `dose_condition` (missing dose on a dose-dependent pair),
and `patient_specific_note` / `severe_if` from extracted dose/schedule plus
**scrubbed** patient notes on Grade A `local` and `openfda` pairs. No
synthesizer call.

### 6. Waterfall — `agent/waterfall.py`

Per unresolved pair, stop at the first tier that returns records:

| Step | Tool | Grade if interaction |
|---|---|---|
| 1 | `openfda_label_check` (DI / CI / boxed warning; window around partner) | A |
| 2 | PubMed eutils + ClinicalTrials.gov (RCT / **human PK** / meta / systematic review) | B |
| 3 | leftover case reports + Firecrawl **fetched** pages | C |
| 4 | nothing | no DDI (`grade=null`, `category=none`) |

`openfda_label_check` searches `drug_interactions`, `contraindications`, and
`boxed_warning`. The snippet is a window around the partner mention, not the
first 3000 characters (highlights TOC). INN/USAN aliases: rifampicin/rifampin,
paracetamol/acetaminophen, amoxycillin/amoxicillin; isosorbide* also matches
`nitrate`/`nitrates`. Citation id is `set_id` or `openfda.spl_set_id`, stored
as `setid`. No partner/alias in the window → not a hit; the waterfall continues
to PubMed. A partner mention with a mapped `setid` is Grade A with **no
synthesizer**; `contraindicat` in that snippet/CI/boxed warning sets
`category=contraindicated` (banner). Overlay then fills dose/timing/patient
notes the same way as RxNav local hits.

Hard rules (also in the synthesizer prompt):

- Cite **only** records the tools just retrieved. No record → no citation → no graded claim.
- `verdict` is `interaction` | `none` | `insufficient`. Only `interaction` gets a grade.
- Human PK / coadministration studies are Grade B when they are the strongest retrieved human evidence (not only RCT-tagged papers).
- Conflict (case report vs absent/negative trial) → Grade C + `evidence_conflict` (never upgraded to B).
- Grade C may `web_fetch` a search URL; unfetched pages are not citations.
- Unknown dose → conditional `dose_condition` (`DDI possible if … > X mg`).
- Separable admin (e.g. cations + levothyroxine) → `category=timing`.
- Contraindicated → `category=contraindicated` (also copied into the banner).
- Patient context present → `patient_specific_note`. Absent → `severe_if[]`.

Pairs run with `PAIR_CONCURRENCY` (default 5). One pair failing must not fail
the whole request (`source_tier="error"`). A tool HTTP 429 is retried twice
with short backoff; if it still fails it is that pair’s error, not “no DDI”,
and does not abort other pairs. 404 is not retried. Identical component
pairs reuse a process-local cache (`agent/pair_cache.py`); hits keep the
same grade. The cache stores graded pair fields only — never patient notes,
raw Rx text, or error-tier rows.

Non-drugs (alcohol, tobacco, grapefruit, herbals) skip pair-checking. Each is
looked up on listed drugs’ openFDA `drug_interactions` / `food_interactions`.
Hits become `AvoidWithItem` rows with citations mapped to retrieved records
(`pipeline/avoid.py`). No record → honest empty copy (never “pair checking
skipped”). No LLM, so herbals are never invented.

### 7. Alternatives — `agent/alternatives.py` + `POST /api/alternatives`

Optional. Does not run until the clinician clicks **Suggest safer alternatives**
under the pair results. Input is the already-graded list (plus patient notes,
scrubbed again).

1. Deterministic importance: `adjuvant` (symptomatic/PRN) > `controller`
   (chronic disease-modifying) > `anchor` (withdrawal can worsen disease or
   cause a serious ADR). Unknown names default to `controller`.
2. Only drugs in a Grade A, Grade B, or contraindicated pair are change
   candidates. Grade C (case reports / weak evidence) never justifies a swap.
   Timing pairs recommend separation, not a swap.
3. Adjuvants are always eligible. Controllers only if the pair is major or
   contraindicated. Anchors only if every partner is also an anchor.
4. LLM proposes ≤2 same-indication substitutes per candidate. It may not
   invent citations or change a drug the ranker rejected.
5. Each substitute is normalized and rechecked only against leftover
   components. A suggestion is `safer` only if it does not introduce a
   contraindication and the leftover interaction burden drops.

The button sits under the pair cards. Empty suggestions still show the
ranking so the clinician knows *which* medicine is lower-stakes.

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
| `POST` | `/api/alternatives` | Second loop: which drug to change |
| `POST` | `/api/ocr` | Image → scrubbed text (UI preview) |

`CheckRequest`: `{ text?, images[]?, patient_context?, timing? }`
(`images` are data-URLs).

`NormalizedDrug` also carries `dose` and `schedule` when extract found them.

`CheckResponse`: `scrubbed_text`, `normalized_drugs`, `unresolved_drugs`,
`pairs`, `contraindicated_banner`,
`avoid_with_medications` (`{substance, medications[], note, citations[]}`),
`insufficient_evidence`, `disclaimer`.

`AlternativesRequest`: `{ normalized_drugs, pairs, patient_context?, avoid_with_medications? }`
(the graded `/api/check` snapshot).

`AlternativesResponse`: `strategy`, `replaceable`, `suggestions`
(`change_from` → `change_to`, `safer`, leftover `remaining_ddis`),
`keep`, `timing_first`, `disclaimer`.

Frontend: one screen in `frontend/src/App.tsx`. Submit disabled until there is
text or an image. Results sort contraindicated → A → B → C. A **pair matrix**
(component × component) sits with the pair cards. Up to five scrubbed checks
live in `localStorage` (`history.ts`); the clinician can clear them. CORS
allows `http://localhost:5173` only.

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

Settings always load `backend/.env` (path is pinned in `config.py`, cwd
does not matter). `llm.complete` / `complete_vision` pass the matching
settings key into LiteLLM. Missing key raises `LLMConfigError` immediately.

---

## Commands

Repo-root venv: `.venv` (Python 3.14).

```bash
# backend
source .venv/bin/activate
pip install -r backend/requirements.txt
python -m spacy download en_core_web_lg
cd backend && cp .env.example .env   # fill real keys
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

`backend/tests/test_scrubber.py` / `test_privacy.py` — PHI gone, clinical
context kept, brands not redacted, raw patient notes never reach extract,
waterfall, or web-resolve queries.

`backend/tests/test_alternatives.py` — importance ranking, adjuvant preferred
over anchor, timing is not a swap, safer() rejects a new contraindication.

`backend/tests/test_llm_json.py` / `test_prefilter.py` / `test_llm_keys.py` —
shared JSON parse, NormalizedDrug pre-filter shape, settings keys into LiteLLM.

`backend/tests/test_waterfall_citations.py` — empty or unmapped synthesizer
citations never earn a grade; invented PMIDs never appear.

`backend/tests/test_waterfall_grades.py` — human PK grades B; case report vs
negative trial is C with `evidence_conflict`; unfetched URLs are not citations.

`backend/tests/test_overlay.py` — local/openFDA Grade A pairs get timing / missing-dose
copy / patient notes without a synthesizer call; extract dose and schedule
survive on `NormalizedDrug`.

`backend/tests/test_avoid.py` — non-drugs stay out of pair-checking; label hits
cite retrieved setids; no hit / unmapped record → honest empty copy; herbals
are not invented.

`backend/tests/test_web_resolve.py` — dataset+RxNav miss can resolve via
scrubbed web verification; invented generics not in the page are dropped;
already-resolved names skip the web.

`backend/tests/test_pair_cache.py` — identical pairs hunt once; cache hits
keep the grade; patient notes never enter the cache; a tool 429 errors that
pair only.

When changing scrub, normalize, web resolve, ranking, overlay, or LLM wiring,
extend these tests. Do not add tests that need live API keys. The ~30-pair **full live
`/api/check` eval lives on a branch off main** (not default pytest).

Privacy invariant: **no reasoning LLM call on unscrubbed text.** Vision may
see a raw image in v1; its transcript is still scrubbed. Image-level
redaction is not v1.

Citation invariant: a graded `interaction` must map **every** cited identifier
back to tool records (`waterfall._citations_from`). Empty or unmapped citations
become `source_tier="insufficient"` (no A/B/C). Invented PMIDs never appear.
The results screen lists those pairs separately; the alternatives panel shows
its own disclaimer when open. Avoid-with citations use the same mapper against
retrieved openFDA records; no mapped citation → empty copy, not a claimed hit.

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

- Tesseract often absent; health reports `"tesseract": false`; vision fallback used.
- Image-level PHI redaction before vision is **not v1** (locked; do not build it here).
- Duplicate-therapy detection (two NSAIDs) omitted on purpose (PLAN §6 #8).
- Live 30-pair full `/api/check` eval is a **branch off main**, not default pytest.

---

## Product tone

Output is for clinicians. Keep summaries mechanism-first, cite sources, show
uncertainty (`insufficient`, conflicts) instead of guessing. The disclaimer in
`service.py` must stay on every `CheckResponse`.
