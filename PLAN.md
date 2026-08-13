# DDI Explorer — Project Plan

> Clinician-facing drug–drug interaction explorer. Single linear AI agent with
> tool access, evidence-graded output (A/B/C), privacy-first input handling.

**Status:** Plan v1 — approved decisions locked; defaults marked ⚙️ can be changed before build.

---

## 1. Locked Decisions

| Decision | Choice |
|---|---|
| Stack | FastAPI (Python) backend + React (Vite, TypeScript) frontend |
| Agent architecture | **Single linear agent** with tools (no in-app subagents) |
| LLM layer | Provider-agnostic adapter (litellm); default reasoning model `deepseek-v4-flash` (OpenAI-compatible, 1M ctx, ~$0.14/$0.28 per 1M tok) |
| Vision (OCR fallback) | ⚙️ Separate provider via env var — default Gemini vision. DeepSeek is text-only, never used for vision |
| De-identification | Server-side Presidio (regex + NER), applied **after OCR, before the agent** |
| OCR | Hybrid: local Tesseract → low confidence → vision-LLM fallback (PHI-safe: text scrubbed after extraction regardless of path) |
| Users/scope | Clinician-facing decision support (disclaimers, auditability, citation-required grading) |
| Evidence retrieval | **Waterfall with early exit** (see §4) — saves tokens, solid source wins |
| Persistence | ⚙️ Stateless backend; history in browser localStorage only |

---

## 2. System Architecture

```
React frontend
  │  text | image(s) | optional patient context | optional timing info
  ▼
FastAPI — PRE-AGENT PIPELINE (deterministic, no LLM)
  1. OCR stage (images only):
       Tesseract local ──confidence < 0.75──▶ vision-LLM fallback (redacted image)
  2. Presidio scrubber: strip PHI (names, MRN, phone, dates, addresses)
       from ALL text — typed input AND OCR output — before anything else
  3. Drug normalization (no LLM first pass):
       a. Local Indian brand dataset (junioralive/Indian-Medicine-Dataset,
          ~250K brands w/ compositions) — fuzzy match → generic(s)
       b. RxNav/RxNorm API for generic/international names + spell-fix
       c. FDC split: combination products → component list
       d. Misses → agent resolves via websearch verification (Indian brands)
  4. Local known-DDI pre-filter (RxNav interaction API / seed dataset):
       resolves known pairs instantly → Grade A candidates, zero LLM tokens
  ▼
SINGLE AGENT (deepseek-v4-flash) — only for pairs NOT resolved locally
  Tools:
    • openfda_label_check(drug_a, drug_b)     → label interaction sections
    • pubmed_search(query)                    → PubMed MCP
    • clinicaltrials_search(query)            → ClinicalTrials.gov MCP
    • web_search(query) / web_fetch(url)      → Firecrawl
  ▼
RESPONSE ASSEMBLER (deterministic)
  merges local pre-filter + agent results, applies rubric, builds output
```

**Why this shape:** known pairs never burn LLM tokens; the agent only does
genuine evidence hunting; PHI never reaches any LLM as text (and redacted
images only reach the vision fallback).

---

## 3. Evidence Grading Rubric

| Grade | Meaning | Sources |
|---|---|---|
| **A** | Approved/known DDI | FDA/openFDA label "Drug Interactions" section, RxNav interaction data |
| **B** | Literature evidence, not in labeling | Human RCT, PK study, meta-analysis (PubMed / ClinicalTrials.gov) |
| **C** | Weak evidence | Case reports, in-vitro/animal, mechanistic extrapolation |
| — | No DDI found | Explicit "no interaction found for this prescription" output |

**Rules:**
- **Citation-required:** a claim with no retrievable citation is *refused*, never graded (anti-hallucination guardrail).
- **Conflict rule (locked):** positive case report + negative/absent RCT evidence → **Grade C**, with the conflict disclosed.
- **Contraindicated pairs** (any grade) → red banner above all results, never buried.
- **Insufficient evidence** (new/novel drug, nothing retrievable) → distinct tag/screen, honest "insufficient evidence" — never silent extrapolation.

---

## 4. Agent Waterfall (early exit — locked)

For each unresolved pair (Drug A, Drug B):

```
1. openfda_label_check ──found──▶ Grade A → DONE, stop
        │ not found
2. pubmed_search + clinicaltrials_search (RCT/PK/meta)
        │ ──found──▶ Grade B → DONE, stop
        │ not found
3. web_search for weak evidence (case reports etc.)
        │ ──found──▶ Grade C → DONE
        │ nothing
4. "No DDI found" for this pair
```

---

## 5. Patient Context Modulation (locked)

- **Optional patient-context text box** (age, sex, weight, renal/hepatic
  function, pregnancy, comorbidities) — also extracted from prescription
  photos via OCR. Scrubbed by Presidio like everything else.
- If context present → grader states severity **for this patient**
  (e.g., same pair: manageable at 30 y/o vs contraindicated in CKD-4).
- If context absent → output includes a **"DDI will be severe if:"** section
  listing risk scenarios (renal impairment, elderly, high dose, etc.).
- **Missing dose:** phrase output conditionally — "DDI possible if Drug A
  dose > 500 mg" — never assume silently.
- **Timing input box:** accepts 1-0-1 style schedules (also parsed from
  prescription photos). Pairs manageable by separation (e.g., calcium +
  levothyroxine) get a distinct **"manageable by timing/separation"**
  category instead of an alarm grade.

---

## 6. Edge-Case Handling (locked)

| # | Case | Handling |
|---|---|---|
| 1 | Combination products / FDCs | Split into components; every component pair checked |
| 2 | Indian brand names | Local dataset fuzzy match → generic; misses → agent websearch verification |
| 3 | Non-drugs (alcohol, tobacco, herbals, grapefruit) | Rejected from pair-checking; separate **"Avoid with medications"** feature listing substance interactions |
| 4 | N-drug explosion | Two-tier: local pre-filter resolves known pairs free; only unknown pairs hit agent. Cap 15 drugs, bounded concurrency, severity-sorted matrix view |
| 5 | Missing dose | Conditional phrasing ("DDI possible if dose > X") |
| 6 | Patient modulation | §5 |
| 7 | Timing-separation pairs | Distinct output category; 1-0-1 parsing |
| 8 | Duplicate therapy (2 NSAIDs) | **Omitted** from v1 |
| 9 | PHI leakage | OCR → text → Presidio → agent; vision fallback receives redacted image only; stateless backend |
| 10 | Vision fallback provider | Non-DeepSeek, env-swappable (default Gemini) |
| 11 | Novel drugs, zero literature | "Insufficient evidence" tag/screen |
| 12 | Contraindicated pairs | Banner above results |
| 13 | Conflicting evidence | Grade C + disclosure |

---

## 7. API Surface (draft)

```
POST /api/check
  body: { text?, images[]?, patient_context?, timing? }
  → { pairs: [{drugs, grade, severity, summary, citations[],
               category: "interaction"|"timing"|"contraindicated",
               severe_if[]?}],
      contraindicated_banner: [...],
      avoid_with_medications: [...],
      unresolved_drugs: [...],
      insufficient_evidence: [...] }

POST /api/ocr          (standalone image → scrubbed text, for UI preview)
GET  /api/health
```

Frontend pages: single main screen (input → results), matrix view for
multi-drug, distinct styling for contraindicated / timing / insufficient.

---

## 8. Build Phases

| Phase | Deliverable |
|---|---|
| **P0** Scaffold | Repo layout, FastAPI skeleton, Vite React, env config, litellm adapter w/ DeepSeek |
| **P1** Privacy pipeline | Presidio scrubber + tests (typed text first, OCR after) |
| **P2** Normalization | Indian dataset ingest + fuzzy match, RxNav client, FDC split |
| **P3** Local pre-filter | RxNav interaction endpoint integration |
| **P4** Agent core | Tool definitions, waterfall loop, early exit, citation enforcement |
| **P5** Evidence tools | PubMed MCP, ClinicalTrials MCP, Firecrawl search/fetch wrappers |
| **P6** OCR | Tesseract + confidence gate + vision fallback with image redaction |
| **P7** Output & UI | Rubric assembler, banner/categories/"severe if", matrix view, disclaimers |
| **P8** Hardening | Rate limits, caching (pair→result cache), eval set of known DDIs, clinician disclaimer copy |

**Testing spine:** golden set of ~30 known pairs (warfarin+aspirin → A,
MAOI+SSRI → contraindicated banner, etc.) run end-to-end in CI.

---

## 9. Open Assumptions (change anytime)

1. ⚙️ Vision fallback = Gemini (needs one extra API key).
2. ⚙️ Stateless backend; browser-local history only.
3. Indian dataset (GitHub, scraped from 1mg) is acceptable for v1 — licensing/data-freshness caveat noted; websearch fallback covers misses.
4. PubMed/ClinicalTrials MCP servers available locally at build time; otherwise thin REST wrappers (eutils / CT.gov v2 API) implement the same tool interface.
