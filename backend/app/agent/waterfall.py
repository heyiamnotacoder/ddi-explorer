"""The single linear agent: per-pair evidence waterfall with early exit.

  1. openFDA labels      -> found -> Grade A, STOP
  2. PubMed + CT.gov     -> human RCT/PK/meta -> Grade B, STOP
                            (only case reports? carry them down to tier 3)
  3. Web search          -> case reports / weak evidence -> Grade C
  4. nothing             -> "No DDI found"

Anti-hallucination rule: the LLM may ONLY cite records handed to it by a
tool in this run. No retrieved record -> no citation -> no claim.
"""
from __future__ import annotations

import asyncio
import json

from ..config import get_settings
from ..models import Category, Citation, Grade, PairResult
from . import llm, tools

SYSTEM = """You are a clinical pharmacology evidence grader for drug-drug interactions.

HARD RULES:
- Cite ONLY records provided to you in this conversation (by PMID, NCT ID, or URL). Never invent citations.
- If provided evidence does not support an interaction, say so — do not extrapolate silently.
- If evidence conflicts (e.g., RCT negative but case reports positive), grade by the STRONGEST human
  evidence tier available and disclose the conflict in "evidence_conflict".
- If a dose threshold matters and the dose is unknown, phrase conditionally:
  "DDI possible if <drug> dose > X mg".
- If the pair is manageable by separating administration timing (e.g., divalent cations +
  levothyroxine/fluoroquinolones), set category="timing".
- If the combination is contraindicated, set category="contraindicated".
- If patient context is provided, give "patient_specific_note" (severity FOR THIS PATIENT).
  If not, list risk scenarios in "severe_if" (e.g., "severe if eGFR < 30", "severe if age > 65").

Return STRICT JSON with keys:
verdict ("interaction" | "none" | "insufficient"), summary, mechanism,
severity (major|moderate|minor), category
(interaction|timing|contraindicated), patient_specific_note (string|null),
severe_if (list[string]), evidence_conflict (string|null), dose_condition (string|null),
cited (list of identifiers/URLs you actually used).

verdict="none": the retrieved evidence affirmatively indicates NO clinically
meaningful interaction (say so plainly in summary).
verdict="insufficient": records retrieved but they do not answer the question.
Only verdict="interaction" earns a grade."""


def _citations_from(cited: list, pool: list[dict], source: str) -> list[Citation]:
    """Map LLM-cited identifiers back to REAL retrieved records only."""
    out = []
    cited_norm = {str(c).lower() for c in cited}
    for rec in pool:
        ids = {str(rec.get(k, "")).lower()
               for k in ("pmid", "nct_id", "setid", "url")} - {""}
        if ids & cited_norm:
            out.append(Citation(
                source=source,
                title=rec.get("title", f"{source} record"),
                url=rec.get("url"),
                identifier=rec.get("pmid") or rec.get("nct_id") or rec.get("setid"),
            ))
    return out


async def _synthesize(drug_a: str, drug_b: str, tier_label: str,
                      evidence: list[dict], patient_context: str | None) -> dict:
    user = (
        f"Drug pair: {drug_a} + {drug_b}\n"
        f"Evidence tier: {tier_label}\n"
        f"Retrieved records (JSON):\n{json.dumps(evidence, indent=1)[:12000]}\n"
        f"Patient context: {patient_context or 'not provided'}"
    )
    raw = await llm.complete(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        response_format={"type": "json_object"})
    return llm.parse_json_object(raw)


def _verdict_result(a: str, b: str, s: dict, grade: Grade | None,
                    citations: list[Citation], tier: str) -> PairResult:
    """Apply the verdict: 'none'/'insufficient' never carry a grade."""
    verdict = s.get("verdict", "interaction")
    # Anti-garbage guard: parse failure or empty synthesis can never earn a grade
    if not s or (verdict == "interaction" and not s.get("summary")):
        verdict = "insufficient"
    if verdict in ("none", "insufficient"):
        return PairResult(
            drugs=(a, b), grade=None,
            category=Category.NONE if verdict == "none" else Category.INTERACTION,
            summary=(s.get("summary", "") or
                     ("No clinically meaningful interaction per retrieved evidence."
                      if verdict == "none" else "Insufficient evidence to determine.")),
            citations=citations, source_tier=tier if verdict == "none" else "insufficient",
        )
    return PairResult(
        drugs=(a, b), grade=grade,
        category=Category(s.get("category", "interaction")),
        severity=s.get("severity"), summary=s.get("summary", ""),
        mechanism=s.get("mechanism"), citations=citations,
        severe_if=s.get("severe_if", []),
        patient_specific_note=s.get("patient_specific_note"),
        evidence_conflict=s.get("evidence_conflict"),
        dose_condition=s.get("dose_condition"),
        source_tier=tier,
    )


async def evaluate_pair(drug_a: str, drug_b: str,
                        patient_context: str | None = None) -> PairResult:
    a, b = drug_a.lower(), drug_b.lower()

    # ---- Tier 1: openFDA labels -------------------------------------------
    fda = await tools.openfda_label_check(a, b)
    if fda:
        s = await _synthesize(a, b, "approved FDA labeling (Grade A)", fda, patient_context)
        return _verdict_result(a, b, s, Grade.A,
                               _citations_from(s.get("cited", []), fda, "openfda"),
                               "openfda")

    # ---- Tier 2: PubMed + ClinicalTrials.gov (parallel) -------------------
    pubs, trials = await asyncio.gather(
        tools.pubmed_search(a, b), tools.clinicaltrials_search(a, b))
    strong = [p for p in pubs if any(
        t in ("Randomized Controlled Trial", "Clinical Trial", "Meta-Analysis",
              "Systematic Review")
        for t in p.get("pubtype", []))] + trials

    if strong:
        s = await _synthesize(a, b, "human trial/PK literature (Grade B)", strong,
                              patient_context)
        return _verdict_result(
            a, b, s, Grade.B,
            (_citations_from(s.get("cited", []), pubs, "pubmed")
             + _citations_from(s.get("cited", []), trials, "clinicaltrials")),
            "pubmed_ct")

    # ---- Tier 3: weak evidence (case reports etc.) ------------------------
    weak_pool = [p for p in pubs if p]  # case reports etc. from the same PubMed search
    web = await tools.web_search(
        f"{a} {b} drug interaction case report")
    weak_pool += web

    if weak_pool:
        s = await _synthesize(a, b, "weak evidence only: case reports / unverified "
                                    "(Grade C)", weak_pool, patient_context)
        return _verdict_result(
            a, b, s, Grade.C,
            (_citations_from(s.get("cited", []), pubs, "pubmed")
             + _citations_from(s.get("cited", []), web, "web")),
            "web")

    # ---- Nothing found -----------------------------------------------------
    return PairResult(
        drugs=(a, b), grade=None, category=Category.NONE,
        summary=f"No documented interaction found between {a} and {b} "
                f"(checked labeling, trials, and literature).",
        source_tier="none",
    )


async def evaluate_pairs(pairs: list[tuple[str, str]],
                         patient_context: str | None = None) -> list[PairResult]:
    """Bounded-concurrency fan-out over unknown pairs."""
    sem = asyncio.Semaphore(get_settings().pair_concurrency)

    async def one(a: str, b: str) -> PairResult:
        async with sem:
            try:
                return await evaluate_pair(a, b, patient_context)
            except Exception as e:  # noqa: BLE001 — never lose a whole run to one pair
                return PairResult(drugs=(a, b), grade=None, category=Category.NONE,
                                  summary=f"Evaluation failed: {type(e).__name__}",
                                  source_tier="error")

    return await asyncio.gather(*(one(a, b) for a, b in pairs))
