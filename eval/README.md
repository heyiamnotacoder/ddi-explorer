# Live `/api/check` eval (this branch only)

Golden list of 30 prescriptions. Each case is a **full live** `POST /api/check`
(extract → normalize → pre-filter → waterfall → assembler). This is the
accuracy stress test. It is **not** default pytest and does not live on `main`.

## Keys

Uses `backend/.env` the same way the app does. You need:

- an LLM key for the configured `LLM_MODEL` (`ANTHROPIC_API_KEY` / `DEEPSEEK_API_KEY` / `OPENAI_API_KEY`)
- `FIRECRAWL_API_KEY` for Grade C / web-resolve cases
- optional `OPENFDA_API_KEY` (higher rate limits)

**Do not commit `.env` or any key.**

A full run is slow (~30–40s per case, often 15–20 minutes) and spends API
tokens. Rate-limit (429) on one pair is an error for that pair, not a
process crash.

## Run

From the repo root, with the repo venv:

```bash
# schema only (no network, no keys)
.venv/bin/python eval/run_eval.py --dry-run

# all 30 live cases
.venv/bin/python eval/run_eval.py

# one known anchor
.venv/bin/python eval/run_eval.py --only warfarin-aspirin

# write a local report (gitignored if you use eval/results/)
.venv/bin/python eval/run_eval.py --json-out eval/results/last.json
```

`PYTHONPATH` is set inside the runner (`backend/` is on `sys.path`). Offline
unit tests stay:

```bash
cd backend && ../.venv/bin/python -m pytest tests/ -q
```

That command must not collect these eval cases.

## Anchors

| id | Expect |
|---|---|
| `warfarin-aspirin` | Grade A |
| `maoi-ssri` | contraindication banner |
| `screenshot-quartet` | four singleton drugs; warfarin–omeprazole is checked |

Other cases cover Indian brands, FDC split, timing, avoid-with, patient
overlay, and additional labeled pairs.

## Invariants checked on every live response

- `disclaimer` is present
- a graded `A`/`B`/`C` pair has mapped citations (non-empty)
