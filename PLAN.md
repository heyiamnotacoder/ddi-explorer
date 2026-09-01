# DDI Explorer — Project Plan

> Clinician-facing drug–drug interaction explorer. Single linear AI agent with
> tool access, evidence-graded output (A/B/C), privacy-first input handling.

**Status:** Running on `main`. Where this file and the code disagree, **code wins**.
Locked review decisions (below) are current product rules, not backlog guesses.

---

## 1. Locked Decisions

| Decision | Choice |
|---|---|
| Stack | FastAPI (Python) backend + React (Vite, TypeScript) frontend |
| Agent architecture | **Single linear agent** with tools (no in-app subagents) |
| LLM layer | Provider-agnostic adapter (litellm). **Default reasoning model `anthropic/claude-sonnet-5`.** DeepSeek and OpenAI are supported swaps via `LLM_MODEL` |
| Vision (OCR fallback) | Separate `VISION_MODEL` (default: same Claude). DeepSeek is text-only, never used for vision. Gemini is an optional swap |
| Extract | **Keep.** Runs after Presidio scrub, before normalize. Do not skip or reorder |
| De-identification | Server-side Presidio (regex + NER) on **all text** after OCR, before any reasoning LLM. `llm.complete` re-scrubs prompt strings |
| OCR / vision honesty (v1) | Hybrid: local Tesseract → low confidence → vision-LLM. **Raw image may reach vision.** Transcript is scrubbed before extract. **Image-level redaction is not v1** |
| Users/scope | Clinician-facing decision support (disclaimers, auditability, citation-required grading) |
| Evidence retrieval | **Waterfall with early exit** (see §4). Tools are REST wrappers in `agent/tools.py` (openFDA, PubMed eutils, ClinicalTrials.gov v2, Firecrawl). MCP servers are optional, not required |
| Persistence | Stateless backend. Browser history is **not shipped**; when added, **scrubbed snapshots only** (never raw patient/timing text) |
| Grade A patient/dose/timing | Deterministic overlay on the known pair — no extra LLM. Extracted dose/schedule stay on `NormalizedDrug` |
| Avoid-with | openFDA label-backed lookup of non-drugs vs listed medications. Hits cite retrieved records. No hit → honest empty copy. No LLM |
| Eval | ~30 **full live `/api/check`** cases on a **branch off main**, not default pytest. Offline unit tests stay on `main` and must not need live API keys |

---

## 2. System Architecture

```
React frontend (single screen: input → pair cards + pair matrix)
  │  text | image(s) | optional patient context | optional timing info
  ▼
FastAPI — PRE-AGENT PIPELINE (deterministic, no LLM)
  1. OCR stage (images only):
       Tesseract local ──confidence < 0.75──▶ vision-LLM fallback
       (v1: vision may see the raw image; output text is still scrubbed)
  2. Presidio scrubber: strip PHI (names, MRN, phone, dates, addresses)
       from ALL text — typed input AND OCR output — before extract
  3. LLM extract (first reasoning call; input is already scrubbed):
       drugs[] (name/dose/timing), non_drugs[], patient_context
  4. Drug normalization (no LLM):
       a. Local Indian brand dataset (~242K brands w/ compositions)
          — fuzzy match → generic(s)
       b. RxNav/RxNorm API for generic/international names + spell-fix
       c. FDC split: combination products → component list;
          each ingredient gets its own RxCUI
       d. Misses → web verification on the scrubbed name (`resolved_via=agent_web`);
          still unresolved if retrieved pages do not name a generic
  5. Local known-DDI pre-filter (RxNav interaction API):
       same NormalizedDrug shape as a fresh check; match on component RxCUI
       resolves known pairs instantly → Grade A, zero LLM tokens
       then a deterministic overlay (timing / dose_condition / patient note)
       from scrubbed extract fields — still no synthesizer
  ▼
SINGLE AGENT (default Claude; swap via LLM_MODEL) — unknown pairs only
  Tools (REST, not MCP):
    • openfda_label_check(drug_a, drug_b)     → label interaction sections
    • pubmed_search(query)                    → PubMed eutils
    • clinicaltrials_search(query)            → ClinicalTrials.gov v2
    • web_search(query) / web_fetch(url)      → Firecrawl
  ▼
RESPONSE ASSEMBLER (deterministic)
  merges local pre-filter + agent results, applies rubric, builds output
```

**Why this shape:** known pairs never burn LLM tokens; the agent only does
genuine evidence hunting; no reasoning LLM sees unscrubbed text. Vision is
transcription-only and is allowed to see the raw image in v1.

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
        │ ──supports DDI──▶ Grade B → DONE, stop
        │ ──negative/absent + no case reports──▶ none / insufficient
        │ ──negative/absent + case reports──▶ continue
3. leftover case reports + web_search URLs that web_fetch actually retrieved
        │ ──found──▶ Grade C → DONE (conflict disclosed if trials absent/negative)
        │ nothing
4. "No DDI found" for this pair
```
Unfetched search hits are not citations.

---

## 5. Patient Context Modulation (locked)

- **Optional patient-context text box** (age, sex, weight, renal/hepatic
  function, pregnancy, comorbidities) — also extracted from prescription
  photos via OCR. Scrubbed by Presidio like everything else. Downstream
  (waterfall, alternatives) never falls back to the raw request field.
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
| 2 | Indian brand names | Local dataset fuzzy match → generic; misses → RxNav; still missing → web verification on scrubbed name (`agent_web`) |
| 3 | Non-drugs (alcohol, tobacco, herbals, grapefruit) | Rejected from pair-checking; **"Avoid with medications"** from openFDA labels, citations mapped to retrieved records; no hit → honest empty copy |
| 4 | N-drug explosion | Two-tier: local pre-filter resolves known pairs free; only unknown pairs hit agent. Cap 15 drugs, bounded concurrency, **severity-sorted pair matrix on the results screen** |
| 5 | Missing dose | Conditional phrasing ("DDI possible if dose > X") |
| 6 | Patient modulation | §5 |
| 7 | Timing-separation pairs | Distinct output category; 1-0-1 parsing |
| 8 | Duplicate therapy (2 NSAIDs) | **Omitted** from v1 |
| 9 | PHI leakage | OCR → text → Presidio → extract; reasoning LLM never sees raw notes. **v1 vision may see the raw image**; transcript is still scrubbed. Image-level redaction is not v1 |
| 10 | Vision fallback provider | Non-DeepSeek, env-swappable. **Default Claude** (same key as reasoning). Gemini is a swap |
| 11 | Novel drugs, zero literature | "Insufficient evidence" tag/screen |
| 12 | Contraindicated pairs | Banner above results |
| 13 | Conflicting evidence | Grade C + disclosure |

---

## 7. API Surface

```
POST /api/check
  body: { text?, images[]?, patient_context?, timing? }
  → { scrubbed_text,
      normalized_drugs: [{input_name, components, dose?, schedule?,
                          resolved_via: indian_dataset|rxnav|agent_web, ...}],
      unresolved_drugs,
      pairs: [{drugs, grade, severity, summary, citations[],
               category: "interaction"|"timing"|"contraindicated"|"none",
               severe_if[]?, patient_specific_note?, dose_condition?}],
      contraindicated_banner: [...],
      avoid_with_medications: [{substance, medications[], note, citations[]}],
      insufficient_evidence: [...],
      disclaimer }

POST /api/ocr          (standalone image → scrubbed text, for UI preview)
POST /api/alternatives (second loop: which drug to change)
  body: { normalized_drugs, pairs, patient_context?, scrubbed_text?,
          avoid_with_medications? }
  → { strategy, replaceable[], suggestions[{change_from, change_to, safer,
      remaining_ddis[]}], keep[], timing_first[], disclaimer }
GET  /api/health
```

Frontend: one screen (`frontend/src/App.tsx`). Submit disabled until there is
text or an image. Results sort contraindicated → A → B → C. A **pair matrix**
sits with the pair cards (components × components, links into the cards).
A **Suggest safer alternatives** button under the pair results runs the
second loop: change the lowest-importance interacting drug (adjuvant
before controller before anchor); timing pairs stay on a schedule.
Grade C pairs never justify a substitution.
CORS allows `http://localhost:5173` only.

---

## 8. Build Phases

| Phase | Deliverable |
|---|---|
| **P0** Scaffold | Repo layout, FastAPI skeleton, Vite React, env config, litellm adapter (Claude default) |
| **P1** Privacy pipeline | Presidio scrubber + tests (typed text first, OCR after) |
| **P2** Normalization | Indian dataset ingest + fuzzy match, RxNav client, FDC split |
| **P3** Local pre-filter | RxNav interaction endpoint integration |
| **P4** Agent core | Tool definitions, waterfall loop, early exit, citation enforcement |
| **P5** Evidence tools | REST wrappers: PubMed eutils, ClinicalTrials.gov v2, Firecrawl (MCP optional) |
| **P6** OCR | Tesseract + confidence gate + vision fallback. Image-level redaction is **not v1** |
| **P7** Output & UI | Rubric assembler, banner/categories/"severe if", **pair matrix**, disclaimers |
| **P8** Hardening | Rate limits, pair→result cache, **eval branch** of ~30 live full `/api/check` cases |

**Testing spine:** offline pytest on `main` (no live keys). Golden set of ~30
known pairs as **full live `/api/check`** on a **branch off main** (not default
CI pytest) — e.g. warfarin+aspirin → A, MAOI+SSRI → contraindicated banner.

---

## 9. Open Assumptions (env-swappable)

1. ⚙️ Vision fallback default = Claude (`VISION_MODEL`). Gemini needs `GEMINI_API_KEY`.
2. ⚙️ Stateless backend; browser-local history, when added, is scrubbed snapshots only.
3. Indian dataset (GitHub, scraped from 1mg) is acceptable for v1 — licensing/data-freshness caveat noted; websearch fallback covers dataset+RxNav misses.
4. Evidence tools are REST wrappers in-tree. MCP servers are optional later; they are not required to run the app.
