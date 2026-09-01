"""Second loop: which medicine to change, given the whole prescription.

Deterministic importance ranking picks the lowest-stakes drug involved in
a real DDI. An LLM then proposes same-indication substitutes. Those names
are normalized and rechecked against the rest of the list (prefilter +
waterfall) so a suggestion cannot silently introduce a worse pair.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from ..models import (
    Category,
    Grade,
    Importance,
    NormalizedDrug,
    PairResult,
    ReplaceableDrug,
)
from . import llm

# Word-boundary tokens. Short stems are omitted on purpose (ace, pam, na).
_ANCHOR = (
    "insulin", "warfarin", "heparin", "enoxaparin", "dalteparin", "fondaparinux",
    "rivaroxaban", "apixaban", "dabigatran", "edoxaban",
    "clopidogrel", "prasugrel", "ticagrelor",
    "alteplase", "tenecteplase", "streptokinase",
    "digoxin", "amiodarone", "sotalol", "dofetilide", "flecainide",
    "levothyroxine", "liothyronine", "carbimazole", "methimazole",
    "propylthiouracil",
    "phenytoin", "carbamazepine", "oxcarbazepine", "valproate", "valproic",
    "levetiracetam", "lamotrigine", "topiramate", "lacosamide",
    "phenobarbitone", "phenobarbital", "clobazam",
    "lithium", "clozapine",
    "tacrolimus", "cyclosporine", "ciclosporin", "mycophenolate",
    "sirolimus", "everolimus", "azathioprine",
    "methotrexate", "cyclophosphamide",
    "isoniazid", "rifampicin", "rifampin", "pyrazinamide", "ethambutol",
    "bedaquiline",
    "efavirenz", "tenofovir", "emtricitabine", "dolutegravir", "lamivudine",
    "zidovudine", "ritonavir", "atazanavir", "lopinavir", "nevirapine",
    "sofosbuvir", "daclatasvir", "ledipasvir", "velpatasvir",
    "cisplatin", "carboplatin", "doxorubicin", "paclitaxel", "vincristine",
    "imatinib",
    "pyridostigmine", "fludrocortisone",
)

_CONTROLLER = (
    "telmisartan", "losartan", "olmesartan", "valsartan", "candesartan",
    "irbesartan", "azilsartan",
    "ramipril", "enalapril", "lisinopril", "perindopril", "captopril",
    "amlodipine", "cilnidipine", "nifedipine", "felodipine",
    "metoprolol", "atenolol", "bisoprolol", "carvedilol", "nebivolol",
    "propranolol",
    "spironolactone", "eplerenone", "chlorthalidone", "hydrochlorothiazide",
    "indapamide", "furosemide", "torsemide", "bumetanide",
    "hydralazine", "sacubitril",
    "metformin", "glimepiride", "gliclazide", "sitagliptin", "vildagliptin",
    "linagliptin", "empagliflozin", "dapagliflozin", "canagliflozin",
    "semaglutide", "liraglutide", "pioglitazone",
    "atorvastatin", "rosuvastatin", "simvastatin", "pitavastatin",
    "budesonide", "fluticasone", "beclomethasone", "formoterol", "salmeterol",
    "tiotropium", "montelukast",
    "sertraline", "escitalopram", "fluoxetine", "paroxetine", "venlafaxine",
    "duloxetine", "mirtazapine", "amitriptyline",
    "levodopa", "carbidopa", "pramipexole", "ropinirole",
    "donepezil", "rivastigmine", "memantine",
    "allopurinol", "febuxostat",
    "tamsulosin", "finasteride",
    "aspirin", "acetylsalicylic", "ecosprin",
)

_ADJUVANT = (
    "paracetamol", "acetaminophen",
    "ibuprofen", "diclofenac", "aceclofenac", "naproxen", "mefenamic",
    "nimesulide", "etoricoxib", "celecoxib", "ketorolac", "piroxicam",
    "indomethacin", "mefenamic acid",
    "pantoprazole", "omeprazole", "rabeprazole", "esomeprazole",
    "lansoprazole", "famotidine", "ranitidine",
    "cetirizine", "levocetirizine", "fexofenadine", "loratadine",
    "chlorpheniramine", "hydroxyzine",
    "ondansetron", "domperidone", "metoclopramide",
    "cholecalciferol", "ergocalciferol", "methylcobalamin", "cyanocobalamin",
    "folic", "ferrous", "multivitamin",
    "ambroxol", "bromhexine", "guaifenesin", "dextromethorphan",
    "phenylephrine", "pseudoephedrine",
    "alprazolam", "diazepam", "zolpidem",
    "thiocolchicoside", "chlorzoxazone",
    "tramadol", "tapentadol", "codeine",
    "lactulose", "ispaghula",
    "calcium", "calcitriol",
)

_PROMPT = """You are a clinical pharmacologist advising which medicine to change
when a prescription has drug–drug interactions.

HARD RULES:
- Change the LOWEST-importance drug that participates in a real DDI.
  adjuvant (symptomatic/PRN) before controller (chronic disease) before
  anchor (withdrawal can worsen serious disease or cause severe ADR).
- Do NOT recommend changing an anchor (insulin, therapeutic anticoagulant,
  antiretroviral, anti-TB, chemo, transplant immunosuppressant, anti-epileptic
  for epilepsy, levothyroxine, post-stent P2Y12, clozapine, lithium) unless
  EVERY other member of the contraindicated pair is also an anchor. Then
  disclose disease-worsening risk.
- Timing-manageable pairs: recommend separating administration. Do not substitute.
- Do NOT recommend changing a medicine because of a Grade C pair. Grade C is
  weak evidence (case reports, in-vitro, mechanism only). Substitution is
  justified only by Grade A, Grade B, or a contraindicated pair. If every
  flagged pair is Grade C, return change=[] and say so in strategy.
- Every alternative must treat the SAME indication as the drug being replaced.
- Prefer the alternative with the lower chance of worsening THIS patient's
  disease or causing ADR (renal, hepatic, elderly, pregnancy, GI bleed, etc.).
- Do not invent citations. Refer only to grades/summaries already provided.
- Do not propose a generic already on the list.
- Propose at most 2 drugs to change and 2 alternatives each.
- Prefer names from "preferred_change" — those are the deterministic ranking.

Return STRICT JSON:
{
  "strategy": "<one paragraph: which drug to change and why it is the lower-stakes one>",
  "change": [
    {
      "from": "<component name exactly as listed>",
      "why_this_one": "<why this drug, not its pair partner>",
      "indication": "<what it is treating>",
      "alternatives": [
        {"name": "<generic>", "rationale": "<why this substitute>",
         "adr_note": "<ADR / disease-worsening comparison vs the original>"}
      ]
    }
  ],
  "keep": [{"drug": "<component>", "reason": "<why not to change it>"}],
  "timing_first": ["<pair + how to separate>"]
}
"""


@dataclass
class Candidate:
    name: str
    input_name: str
    importance: Importance
    ddi_count: int
    has_major_or_contra: bool
    partners_all_anchor: bool
    involved_pairs: list[tuple[str, str]] = field(default_factory=list)
    why: str = ""


def _has_token(name: str, tokens: tuple[str, ...]) -> bool:
    n = name.lower()
    return any(re.search(rf"\b{re.escape(t)}\b", n) for t in tokens)


def importance_of(name: str) -> Importance:
    """Conservative default: unknown names are controllers, not adjuvants."""
    if _has_token(name, _ANCHOR):
        return Importance.ANCHOR
    if _has_token(name, _ADJUVANT):
        return Importance.ADJUVANT
    if _has_token(name, _CONTROLLER):
        return Importance.CONTROLLER
    n = name.lower()
    if re.search(r"\b(vitamin|calcium|iron|zinc|folate)\b", n):
        return Importance.ADJUVANT
    return Importance.CONTROLLER


def replaceability(imp: Importance) -> int:
    return {Importance.ADJUVANT: 2, Importance.CONTROLLER: 1, Importance.ANCHOR: 0}[imp]


def is_actionable(p: PairResult) -> bool:
    """Substitution is only for Grade A/B or contraindicated — never Grade C."""
    if p.category == Category.TIMING:
        return False
    if p.grade == Grade.C:
        return False
    if p.category == Category.CONTRAINDICATED:
        return True
    return p.category == Category.INTERACTION and p.grade in (Grade.A, Grade.B)


def is_major_or_contra(p: PairResult) -> bool:
    if p.category == Category.CONTRAINDICATED:
        return True
    if p.grade == Grade.A:
        return True
    return (p.severity or "").lower() == "major"


def _pair_mentions(p: PairResult, name: str) -> bool:
    n = name.lower()
    return any(d.lower() == n for d in p.drugs)


def _parent_product(drugs: list[NormalizedDrug], component: str) -> str:
    c = component.lower()
    for d in drugs:
        if any(x.lower() == c for x in d.components):
            return d.input_name
        if d.generic_name and d.generic_name.lower() == c:
            return d.input_name
    return component


def _components_of(drugs: list[NormalizedDrug]) -> list[tuple[str, str]]:
    """(component, input_name) in first-seen order."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for d in drugs:
        for c in d.components:
            k = c.lower()
            if k and k not in seen:
                seen.add(k)
                out.append((c, d.input_name))
    return out


def pick_candidates(drugs: list[NormalizedDrug],
                    pairs: list[PairResult]) -> list[Candidate]:
    """Lowest-stakes drugs that are actually in an actionable DDI."""
    actionable = [p for p in pairs if is_actionable(p)]
    comps = _components_of(drugs)
    imp_map = {c.lower(): importance_of(c) for c, _ in comps}
    out: list[Candidate] = []

    for name, input_name in comps:
        involved = [p for p in actionable if _pair_mentions(p, name)]
        if not involved:
            continue
        partners: list[str] = []
        for p in involved:
            partners.extend(d for d in p.drugs if d.lower() != name.lower())
        cand = Candidate(
            name=name,
            input_name=input_name,
            importance=imp_map[name.lower()],
            ddi_count=len(involved),
            has_major_or_contra=any(is_major_or_contra(p) for p in involved),
            partners_all_anchor=all(
                imp_map.get(p.lower(), Importance.CONTROLLER) == Importance.ANCHOR
                for p in partners
            ) if partners else False,
            involved_pairs=[tuple(p.drugs) for p in involved],
        )
        if cand.importance == Importance.ADJUVANT:
            cand.why = (
                f"{name} is symptomatic / lower-stakes; prefer changing it "
                f"over a disease-modifying partner."
            )
            out.append(cand)
        elif cand.importance == Importance.CONTROLLER and cand.has_major_or_contra:
            cand.why = (
                f"{name} is a chronic controller involved in a major or "
                f"contraindicated pair; change only if a same-class option is safer."
            )
            out.append(cand)
        elif (cand.importance == Importance.ANCHOR and cand.has_major_or_contra
              and cand.partners_all_anchor):
            cand.why = (
                f"{name} is an anchor and so is its pair partner; any change "
                f"needs specialist review because withdrawal can worsen disease."
            )
            out.append(cand)

    out.sort(key=lambda c: (-replaceability(c.importance), -c.ddi_count, c.name))
    return out


def keep_list(drugs: list[NormalizedDrug], pairs: list[PairResult],
              candidates: list[Candidate]) -> list[ReplaceableDrug]:
    changing = {c.name.lower() for c in candidates}
    keep: list[ReplaceableDrug] = []
    seen: set[str] = set()
    for name, input_name in _components_of(drugs):
        k = name.lower()
        if k in changing or k in seen:
            continue
        seen.add(k)
        imp = importance_of(name)
        if imp == Importance.ADJUVANT and not any(
                is_actionable(p) and _pair_mentions(p, name) for p in pairs):
            continue
        reason = {
            Importance.ANCHOR: (
                f"Keep {name}: withdrawal or casual substitution can worsen "
                f"the underlying disease or precipitate a serious ADR."
            ),
            Importance.CONTROLLER: (
                f"Keep {name} unless a same-class alternative clears a major DDI."
            ),
            Importance.ADJUVANT: f"Keep {name}: not driving a graded interaction.",
        }[imp]
        keep.append(ReplaceableDrug(
            name=name, input_name=input_name, importance=imp, why_this_one=reason,
        ))
    return keep


def timing_notes(pairs: list[PairResult]) -> list[str]:
    notes = []
    for p in pairs:
        if p.category != Category.TIMING:
            continue
        a, b = p.drugs
        extra = f" {p.summary}" if p.summary else ""
        notes.append(f"Keep {a} and {b}; separate administration.{extra}".strip())
    return notes


def pair_burden(pairs: list[PairResult]) -> int:
    """Higher = worse leftover interaction load (contra >> A >> B >> C >> timing)."""
    score = 0
    for p in pairs:
        if p.category == Category.CONTRAINDICATED:
            score += 100
        elif p.grade == Grade.A:
            score += 10
        elif p.grade == Grade.B:
            score += 5
        elif p.grade == Grade.C:
            score += 2
        elif p.category == Category.TIMING:
            score += 1
    return score


def is_safer(old_pairs: list[PairResult], new_pairs: list[PairResult]) -> bool:
    """True only if leftover burden drops and we did not keep the same Grade A."""
    if any(p.category == Category.CONTRAINDICATED for p in new_pairs):
        return False
    if pair_burden(new_pairs) >= pair_burden(old_pairs):
        return False
    new_a = sum(1 for p in new_pairs if p.grade == Grade.A)
    old_a = sum(1 for p in old_pairs if p.grade == Grade.A)
    old_contra = sum(1 for p in old_pairs if p.category == Category.CONTRAINDICATED)
    # Same-molecule / same-class swaps often re-hit the original Grade A.
    if new_a > 0 and new_a >= old_a and old_contra == 0:
        return False
    return True


def leftover_ddis(pairs: list[PairResult]) -> list[PairResult]:
    """Show leftover A/B/contra, timing, and Grade C (C is informational only)."""
    return [
        p for p in pairs
        if is_actionable(p) or p.category == Category.TIMING or p.grade == Grade.C
    ]


def _payload(drugs: list[NormalizedDrug], pairs: list[PairResult],
             candidates: list[Candidate], patient_context: str | None,
             avoid: list[str], scrubbed_text: str | None = None) -> str:
    listed = []
    for name, input_name in _components_of(drugs):
        listed.append({
            "component": name,
            "product": input_name,
            "importance": importance_of(name).value,
        })
    pair_rows = []
    for p in pairs:
        pair_rows.append({
            "drugs": list(p.drugs),
            "grade": p.grade.value if p.grade else None,
            "category": p.category.value,
            "severity": p.severity,
            "summary": p.summary,
            "mechanism": p.mechanism,
            "dose_condition": p.dose_condition,
            "patient_specific_note": p.patient_specific_note,
        })
    preferred = [
        {"component": c.name, "product": c.input_name,
         "importance": c.importance.value, "why": c.why}
        for c in candidates
    ]
    return (
        f"Medicines (component, product, importance):\n{json.dumps(listed, indent=1)}\n"
        f"Preferred drugs to change (deterministic ranking):\n{json.dumps(preferred, indent=1)}\n"
        f"Pair results already graded:\n{json.dumps(pair_rows, indent=1)[:14000]}\n"
        f"Patient context: {patient_context or 'not provided'}\n"
        f"De-identified prescription text:\n{(scrubbed_text or '')[:4000]}\n"
        f"Avoid-with / non-drugs: {avoid or []}"
    )


async def propose(drugs: list[NormalizedDrug], pairs: list[PairResult],
                  candidates: list[Candidate], patient_context: str | None,
                  avoid: list[str], scrubbed_text: str | None = None) -> dict:
    raw = await llm.complete(
        [{"role": "system", "content": _PROMPT},
         {"role": "user", "content": _payload(
             drugs, pairs, candidates, patient_context, avoid, scrubbed_text)}],
        response_format={"type": "json_object"},
        max_tokens=2500,
    )
    return llm.parse_json_object(raw)


def allowed_from_names(candidates: list[Candidate]) -> set[str]:
    names: set[str] = set()
    for c in candidates:
        names.add(c.name.lower())
        names.add(c.input_name.lower())
    return names


def fallback_strategy(candidates: list[Candidate], timing: list[str],
                      pairs: list[PairResult] | None = None) -> str:
    pairs = pairs or []
    only_weak = (
        not candidates
        and any(p.grade == Grade.C for p in pairs)
        and not any(is_actionable(p) for p in pairs)
    )
    if only_weak and timing:
        return (
            "No substitution for the Grade C pairs (weak evidence). "
            "Timing-manageable pairs should be separated, not swapped."
        )
    if only_weak:
        return (
            "No substitution. The flagged pairs are Grade C (case reports / "
            "weak evidence). Do not change a working medicine on that basis."
        )
    if not candidates and timing:
        return (
            "No substitution needed. The flagged pairs are manageable by "
            "separating administration rather than changing a medicine."
        )
    if not candidates:
        return (
            "No lower-stakes medicine is a safe substitution target. "
            "The interacting drugs are disease-modifying or high-ADR-risk; "
            "review with a clinical pharmacist rather than swapping casually."
        )
    top = candidates[0]
    return (
        f"Prefer changing {top.name} ({top.importance.value}) — {top.why} "
        "Same-indication alternatives were not proposed this run; "
        "the ranking still stands."
    )
