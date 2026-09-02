"""The single linear agent: per-pair evidence waterfall with early exit.

  1. openFDA labels      -> pair-scoped DI / CI hit -> Grade A, no LLM, STOP
  2. PubMed + CT.gov     -> human RCT/PK/meta that support a DDI -> Grade B, STOP
                            (case reports withheld; negative/absent strong
                            evidence falls through to tier 3)
  3. Case reports + fetched web pages -> Grade C
     Conflict (case report vs absent/negative trial) stays C and is disclosed.
     Search URLs are not citations until web_fetch returns content.
  4. nothing             -> "No DDI found"

Anti-hallucination rule: the LLM may ONLY cite records handed to it by a
tool in this run. No retrieved record -> no citation -> no claim.

A 429 from any evidence tool marks that pair `source_tier="error"` (not
"no DDI"). Identical component pairs reuse a process-local cache; the
cache never stores patient notes or error-tier rows.
"""
from __future__ import annotations

import asyncio
import json
import re

from ..config import get_settings
from ..models import Category, Citation, Grade, PairResult
from . import llm, pair_cache, tools

_STRONG_PUBTYPES = (
    "Randomized Controlled Trial",
    "Clinical Trial",
    "Controlled Clinical Trial",
    "Pragmatic Clinical Trial",
    "Equivalence Trial",
    "Meta-Analysis",
    "Systematic Review",
    "Clinical Study",
)
_WEAK_PUBTYPES = ("Case Reports", "Letter", "Comment", "Editorial")
_PK_RE = re.compile(
    r"\bpharmacokinet|\bcoadministration study\b|\bdrug[- ]drug interaction study\b",
    re.I,
)
_CASE_RE = re.compile(r"\bcase reports?\b", re.I)
_MAX_WEB_FETCHES = 3
_CONFLICT_NEGATIVE = (
    "Case reports suggest an interaction; retrieved trial or PK evidence "
    "does not confirm it (absent or negative)."
)
_CONFLICT_ABSENT = (
    "Case reports suggest an interaction; no confirmatory human trial or "
    "PK study was retrieved."
)

SYSTEM = """You are a clinical pharmacology evidence grader for drug-drug interactions.

HARD RULES:
- Cite ONLY records provided to you in this conversation (by PMID, NCT ID, or URL). Never invent citations.
- If provided evidence does not support an interaction, say so — do not extrapolate silently.
- Human PK / coadministration studies are Grade B when they are the strongest retrieved human evidence.
- If evidence conflicts (positive case report plus absent or negative trial/PK evidence), verdict is
  interaction at Grade C and you MUST fill evidence_conflict. Do not upgrade that conflict to B.
- Cite ONLY identifiers in the retrieved records. Web pages count only when page content was fetched;
  a search snippet or unfetched URL is not evidence.
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


def _cited_list(cited) -> list:
    if cited is None:
        return []
    if isinstance(cited, str):
        return [cited] if cited.strip() else []
    if isinstance(cited, (list, tuple, set)):
        return list(cited)
    return [cited]


def _record_ids(pool: list[dict]) -> set[str]:
    ids: set[str] = set()
    for rec in pool:
        for k in ("pmid", "nct_id", "setid", "url"):
            v = rec.get(k)
            if v:
                ids.add(str(v).strip().lower())
    return ids


def _all_cited_mapped(cited, pool: list[dict]) -> bool:
    """False if the synthesizer named any identifier the tools did not retrieve."""
    ids = _record_ids(pool)
    for c in _cited_list(cited):
        s = str(c).strip().lower()
        if s and s not in ids:
            return False
    return True


def _citations_from(cited, pool: list[dict], source: str) -> list[Citation]:
    """Map LLM-cited identifiers back to REAL retrieved records only."""
    out = []
    cited_norm = {str(c).strip().lower() for c in _cited_list(cited)} - {""}
    for rec in pool:
        ids = {str(rec.get(k, "")).strip().lower()
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
                    citations: list[Citation], tier: str,
                    *, pool: list[dict] | None = None) -> PairResult:
    """Apply the verdict: 'none'/'insufficient' never carry a grade.

    An 'interaction' is graded only when every cited identifier maps to a
    retrieved record and at least one mapped citation remains. Invented
    PMIDs never appear on the result.
    """
    verdict = s.get("verdict", "interaction")
    # Anti-garbage guard: parse failure or empty synthesis can never earn a grade
    if not s or (verdict == "interaction" and not s.get("summary")):
        verdict = "insufficient"
    if verdict == "interaction":
        cited = s.get("cited", [])
        if not citations or (pool is not None and not _all_cited_mapped(cited, pool)):
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


def _is_case_report(rec: dict) -> bool:
    types = rec.get("pubtype") or []
    if any(t in _WEAK_PUBTYPES for t in types):
        return True
    blob = f"{rec.get('title', '')} {rec.get('abstract', '')}"
    return bool(_CASE_RE.search(blob))


def _is_pk_study(rec: dict) -> bool:
    """Human PK / coadministration study. Case reports never count as PK-B."""
    if _is_case_report(rec):
        return False
    blob = f"{rec.get('title', '')} {rec.get('abstract', '')}"
    return bool(_PK_RE.search(blob))


def _is_strong_human(rec: dict) -> bool:
    """RCT / clinical trial / meta / systematic review / human PK."""
    if _is_case_report(rec):
        return False
    types = rec.get("pubtype") or []
    if any(t in _STRONG_PUBTYPES for t in types):
        return True
    return _is_pk_study(rec)


def split_pubmed(pubs: list[dict]) -> tuple[list[dict], list[dict]]:
    """(strong human evidence, leftover case reports / other)."""
    strong: list[dict] = []
    weak: list[dict] = []
    for rec in pubs:
        if _is_strong_human(rec):
            strong.append(rec)
        else:
            weak.append(rec)
    return strong, weak


async def _fetched_web(hits: list[dict]) -> list[dict]:
    """Fetch search hits. Unfetched URLs never enter the citation pool."""
    chosen = [h for h in hits if str(h.get("url") or "").strip()][:_MAX_WEB_FETCHES]
    if not chosen:
        return []
    bodies = await asyncio.gather(
        *(tools.web_fetch(str(h["url"])) for h in chosen))
    out: list[dict] = []
    for hit, body in zip(chosen, bodies):
        text = (body or "").strip()
        if not text:
            continue
        out.append({
            "title": hit.get("title") or "web page",
            "url": str(hit["url"]),
            "snippet": hit.get("snippet", ""),
            "content": text[:4000],
        })
    return out


def _with_conflict(result: PairResult, text: str) -> PairResult:
    if result.grade != Grade.C or result.evidence_conflict:
        return result
    return result.model_copy(update={"evidence_conflict": text})


def _none_pair(a: str, b: str) -> PairResult:
    return PairResult(
        drugs=(a, b), grade=None, category=Category.NONE,
        summary=f"No documented interaction found between {a} and {b} "
                f"(checked labeling, trials, and literature).",
        source_tier="none",
    )


def _error_pair(a: str, b: str, reason: str) -> PairResult:
    return PairResult(
        drugs=(a, b), grade=None, category=Category.NONE,
        summary=reason, source_tier="error",
    )


def _label_text_mentions_pair(text: str, a: str, b: str) -> bool:
    aliases = tools._label_aliases(a) + tools._label_aliases(b)
    return tools._first_mention(text or "", aliases) is not None


def _partner_aliases_for_rec(rec: dict, a: str, b: str) -> list[str]:
    subject = (rec.get("subject_drug") or "").strip().lower()
    if subject == a.lower():
        return tools._label_aliases(b)
    if subject == b.lower():
        return tools._label_aliases(a)
    return tools._label_aliases(a) + tools._label_aliases(b)


def _label_role(rec: dict, a: str, b: str) -> str | None:
    """Map one label hit to contraindicated, interaction, or drop (fall through)."""
    if not rec.get("setid"):
        return None
    names = tools._ingredient_names_from(rec)
    if tools._spl_contains_both_pair_members(names, a, b):
        return None
    di = rec.get("di_text")
    ci = rec.get("ci_text")
    blob = rec.get("interactions_text") or ""
    if di is None and ci is None:
        di = blob or None
        ci = blob or None
    mention = " ".join(x for x in (di, ci, blob) if x)
    if not _label_text_mentions_pair(mention, a, b):
        return None
    aliases = _partner_aliases_for_rec(rec, a, b)
    if tools._pair_scoped_contraindication(ci or "", aliases) or (
            tools._pair_scoped_contraindication(di or "", aliases)):
        return "contraindicated"
    if not di or not _label_text_mentions_pair(di, a, b):
        return None
    if tools._negative_pair_mention(di, aliases):
        return None
    if tools._is_ingredient_colist(di, aliases) and not tools._has_interaction_language(di):
        return None
    return "interaction"


def _pair_from_labels(a: str, b: str, fda: list[dict]) -> PairResult | None:
    """Grade A from a pair-scoped label hit. No synthesizer.

    Combo SPLs whose ingredients already include both pair members are
    dropped. A 'no clinically significant interaction with {partner}'
    sentence is not Grade A. Drug-interactions windows grade interaction
    unless they are
    ingredient co-lists without interaction language. CI/boxed (or DI)
    grades contraindicated only with with/concomitant/coadminister +
    partner. Unusable hits fall through.
    """
    usable: list[dict] = []
    contra = False
    for rec in fda:
        role = _label_role(rec, a, b)
        if role is None:
            continue
        usable.append(rec)
        if role == "contraindicated":
            contra = True
    if not usable:
        return None
    cites = _citations_from(
        [str(rec["setid"]) for rec in usable], usable, "openfda")
    if not cites:
        return None
    blob = " ".join(rec.get("interactions_text") or "" for rec in usable)
    cat = Category.CONTRAINDICATED if contra else Category.INTERACTION
    return PairResult(
        drugs=(a, b),
        grade=Grade.A,
        category=cat,
        severity="major" if contra else "moderate",
        summary=blob[:800],
        citations=cites,
        source_tier="openfda",
    )


async def evaluate_pair(drug_a: str, drug_b: str,
                        patient_context: str | None = None) -> PairResult:
    a, b = drug_a.lower(), drug_b.lower()
    hit = pair_cache.get(a, b)
    if hit is not None:
        return hit.model_copy(update={"drugs": (a, b)})
    try:
        result = await _hunt_pair(a, b, patient_context)
    except tools.ToolRateLimit as e:
        return _error_pair(a, b, f"Evidence tool rate-limited ({e.tool}).")
    pair_cache.put(result)
    return result


async def _hunt_pair(a: str, b: str, patient_context: str | None) -> PairResult:
    # ---- Tier 1: openFDA labels -------------------------------------------
    fda = await tools.openfda_label_check(a, b)
    labeled = _pair_from_labels(a, b, fda)
    if labeled is not None:
        return labeled

    # ---- Tier 2: PubMed + ClinicalTrials.gov (parallel) -------------------
    pubs, trials = await asyncio.gather(
        tools.pubmed_search(a, b), tools.clinicaltrials_search(a, b))
    strong_pubs, weak_pubs = split_pubmed(pubs)
    strong = strong_pubs + list(trials)

    b_result: PairResult | None = None
    if strong:
        s = await _synthesize(
            a, b,
            "human trial/PK literature (Grade B). These records are RCT, PK, "
            "meta, systematic review, or registered trials. Case reports are "
            "withheld. If they do not support a clinically meaningful "
            "interaction, verdict=none or insufficient.",
            strong, patient_context)
        b_result = _verdict_result(
            a, b, s, Grade.B,
            (_citations_from(s.get("cited", []), strong_pubs, "pubmed")
             + _citations_from(s.get("cited", []), trials, "clinicaltrials")),
            "pubmed_ct", pool=strong)
        if b_result.grade == Grade.B:
            return b_result

    # ---- Tier 3: weak evidence (case reports + fetched pages only) --------
    web_hits = await tools.web_search(
        f"{a} {b} drug interaction case report")
    fetched = await _fetched_web(web_hits)
    weak_only = list(weak_pubs) + list(fetched)
    if not weak_only:
        return b_result if b_result is not None else _none_pair(a, b)

    weak_pool = list(weak_only)
    if b_result is not None and b_result.grade != Grade.B:
        weak_pool = list(strong) + weak_pool

    s = await _synthesize(
        a, b,
        "weak evidence: case reports and fetched pages (Grade C). "
        "If trial/PK evidence is absent or negative alongside a positive "
        "case report, verdict=interaction at Grade C and fill "
        "evidence_conflict. Do not cite unfetched URLs.",
        weak_pool, patient_context)
    cited = s.get("cited", [])
    cites = (
        _citations_from(cited, pubs, "pubmed")
        + _citations_from(cited, trials, "clinicaltrials")
        + _citations_from(cited, fetched, "web")
    )
    conflict_text = _CONFLICT_NEGATIVE if strong else _CONFLICT_ABSENT
    disclose = not strong or (b_result is not None and b_result.grade != Grade.B)
    if disclose and cites and s.get("verdict") != "interaction":
        s = {
            **s,
            "verdict": "interaction",
            "summary": s.get("summary") or conflict_text,
            "evidence_conflict": s.get("evidence_conflict") or conflict_text,
            "category": s.get("category") or "interaction",
        }
    result = _verdict_result(
        a, b, s, Grade.C, cites, "web", pool=weak_pool)
    if disclose:
        result = _with_conflict(result, conflict_text)
    return result


async def evaluate_pairs(pairs: list[tuple[str, str]],
                         patient_context: str | None = None) -> list[PairResult]:
    """Bounded-concurrency fan-out. Duplicate component pairs hunt once."""
    sem = asyncio.Semaphore(get_settings().pair_concurrency)
    computed: dict[tuple[str, str], PairResult] = {}

    async def hunt(a: str, b: str) -> None:
        k = pair_cache.key(a, b)
        async with sem:
            try:
                computed[k] = await evaluate_pair(a, b, patient_context)
            except Exception as e:  # noqa: BLE001 — never lose a whole run to one pair
                computed[k] = _error_pair(
                    a.lower(), b.lower(),
                    f"Evaluation failed: {type(e).__name__}",
                )

    unique: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for a, b in pairs:
        k = pair_cache.key(a, b)
        if k not in seen:
            seen.add(k)
            unique.append((a, b))
    await asyncio.gather(*(hunt(a, b) for a, b in unique))
    out: list[PairResult] = []
    for a, b in pairs:
        src = computed[pair_cache.key(a, b)]
        out.append(src.model_copy(update={"drugs": (a.lower(), b.lower())}))
    return out
