"""Evidence tools available to the agent's waterfall.

  tier 1: openfda_label_check   -> approved labeling (Grade A)
  tier 2: pubmed_search         -> PubMed eutils (Grade B candidates)
          clinicaltrials_search -> ClinicalTrials.gov v2 (Grade B candidates)
  tier 3: web_search/web_fetch  -> Firecrawl (Grade C candidates)

Each tool returns compact dicts the waterfall feeds to the LLM for
synthesis. The LLM never fabricates: no retrieved record -> no citation.
"""
from __future__ import annotations

import asyncio
import os
import random
import re
import urllib.parse
import xml.etree.ElementTree as ET

import httpx

from ..config import get_settings

# Two retries after the first 429. Total sleep stays well under HTTP_TIMEOUT (30s).
_429_RETRIES = 2
_5XX_RETRIES = 1
_BACKOFF_BASE_S = 0.4
_MAX_SLEEP_S = 8.0


class ToolRateLimit(Exception):
    """HTTP 429 from an evidence tool. That pair degrades to error, not 'no DDI'."""

    def __init__(self, tool: str):
        self.tool = tool
        super().__init__(f"{tool} rate-limited (429)")


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=get_settings().http_timeout)


def _raise_http(r: httpx.Response, tool: str) -> None:
    if r.status_code == 429:
        raise ToolRateLimit(tool)
    r.raise_for_status()


async def _sleep_backoff(attempt: int, slept: float) -> float:
    remaining = _MAX_SLEEP_S - slept
    if remaining <= 0:
        return slept
    delay = min(_BACKOFF_BASE_S * (2 ** attempt), remaining)
    delay = min(delay + random.uniform(0.0, min(0.2, delay)), remaining)
    if delay <= 0:
        return slept
    await asyncio.sleep(delay)
    return slept + delay


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    tool: str,
    **kwargs,
) -> httpx.Response:
    """Evidence GET/POST with short 429 backoff. 404 is not retried."""
    send = getattr(client, method.lower())
    n_429 = 0
    n_5xx = 0
    slept = 0.0
    while True:
        r = await send(url, **kwargs)
        code = r.status_code
        if code == 429:
            if n_429 >= _429_RETRIES or slept >= _MAX_SLEEP_S:
                _raise_http(r, tool)
            n_429 += 1
            slept = await _sleep_backoff(n_429 - 1, slept)
            continue
        if code >= 500:
            if n_5xx >= _5XX_RETRIES or slept >= _MAX_SLEEP_S:
                return r
            n_5xx += 1
            slept = await _sleep_backoff(n_5xx - 1, slept)
            continue
        return r


# --------------------------------------------------------------------------
# Tier 1 — openFDA drug labels
# --------------------------------------------------------------------------
LABEL_SEARCH_FIELDS = ("drug_interactions", "contraindications", "boxed_warning")
SUBSTANCE_SEARCH_FIELDS = ("drug_interactions", "food_interactions")
LABEL_WINDOW_CHARS = 3000

# Indian/British -> US spelling variants; literature is indexed under US names.
SPELLING_VARIANTS: dict[str, list[str]] = {
    "amoxycillin": ["amoxicillin"],
    "sulphate": ["sulfate"],
    "sulpha": ["sulfa"],
    "oestrogen": ["estrogen"],
    "oestradiol": ["estradiol"],
    "adrenaline": ["epinephrine"],
    "noradrenaline": ["norepinephrine"],
    "frusemide": ["furosemide"],
    "lignocaine": ["lidocaine"],
    "pethidine": ["meperidine"],
    "methylphenobarbitone": ["methylphenobarbital"],
    "paracetamol": ["acetaminophen"],
    "rifampicin": ["rifampin"],
}

_NITRATE_NAMES = frozenset({
    "isosorbide", "isosorbide mononitrate", "isosorbide dinitrate",
    "nitroglycerin", "glyceryl trinitrate", "nitrate", "nitrates",
})
_NITRATE_ALIASES = (
    "isosorbide", "isosorbide mononitrate", "isosorbide dinitrate",
    "nitrate", "nitrates",
)
_PDE5_NAMES = frozenset({
    "sildenafil", "tadalafil", "vardenafil", "avanafil",
})
_PDE5_ALIASES = (
    "sildenafil", "tadalafil", "vardenafil", "avanafil",
    "pde-5", "pde5", "phosphodiesterase", "pde inhibitor", "pde inhibitors",
)


def _clean_term(name: str) -> str:
    return (name or "").replace('"', "").strip()


def _label_aliases(name: str) -> list[str]:
    """INN/USAN plus nitrate / PDE-5 class terms for label search and windows."""
    raw = _clean_term(name)
    if not raw:
        return []
    n = raw.lower()
    variants = {n}
    for brit, us in SPELLING_VARIANTS.items():
        if brit in n:
            variants |= {n.replace(brit, u) for u in us}
        for u in us:
            if u in n:
                variants.add(n.replace(u, brit))
    if n in _NITRATE_NAMES or n.startswith("isosorbide"):
        variants.update(_NITRATE_ALIASES)
    folded = n.replace("-", "").replace(" ", "")
    if n in _PDE5_NAMES or "phosphodiesterase" in n or folded in {"pde5"}:
        variants.update(_PDE5_ALIASES)
    out: list[str] = []
    seen: set[str] = set()
    for item in (n, *sorted(variants)):
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _setid_from_record(rec: dict) -> str | None:
    """openFDA uses set_id / openfda.spl_set_id, not setid."""
    for key in ("set_id", "setid"):
        val = rec.get(key)
        if val:
            return str(val)
    openfda = rec.get("openfda") or {}
    spl = openfda.get("spl_set_id")
    if isinstance(spl, list):
        return str(spl[0]) if spl else None
    if spl:
        return str(spl)
    return None


def _dailymed_url(setid: str | None, query: str) -> str:
    if setid:
        return (
            "https://dailymed.nlm.nih.gov/dailymed/"
            f"drugInfo.cfm?setid={urllib.parse.quote(str(setid))}"
        )
    return (
        "https://dailymed.nlm.nih.gov/dailymed/search.cfm"
        f"?query={urllib.parse.quote(query)}"
    )


def _join_fields(rec: dict, fields: tuple[str, ...]) -> str:
    parts: list[str] = []
    for field in fields:
        val = rec.get(field) or []
        if isinstance(val, str):
            if val.strip():
                parts.append(val)
        else:
            parts.extend(str(x) for x in val if x)
    return " ".join(parts)


def _alias_pattern(alias: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![a-z0-9]){re.escape(alias.lower())}(?![a-z0-9])")


def _first_mention(text: str, aliases: list[str]) -> int | None:
    low = text.lower()
    best: int | None = None
    for alias in aliases:
        if not alias:
            continue
        m = _alias_pattern(alias).search(low)
        if m and (best is None or m.start() < best):
            best = m.start()
    return best


def _window_around(text: str, aliases: list[str],
                   limit: int = LABEL_WINDOW_CHARS) -> str | None:
    """Keep a window around the first partner/alias mention, not the TOC head."""
    blob = " ".join((text or "").split())
    if not blob or not aliases:
        return None
    idx = _first_mention(blob, aliases)
    if idx is None:
        return None
    half = max(limit // 2, 1)
    start = max(0, idx - half)
    end = min(len(blob), start + limit)
    if end - start < limit:
        start = max(0, end - limit)
    return blob[start:end]


_INTERACTION_RE = re.compile(
    r"\b(?:drug[- ]drug|interact(?:ion|s|ing)?|concomitant|co-?administ\w*|"
    r"exposure|auc|cmax|cyp\d*|p-?gp|clearance|metabolis\w*|"
    r"increas\w+|decreas\w+|reduc\w+|avoid|monitor|caution|"
    r"potentiat\w+|prolong\w+|myopath\w+|rhabdo\w*|bleed\w*|"
    r"absorption|chelat\w*|separate|stagger)\b",
    re.I,
)


def _has_interaction_language(text: str) -> bool:
    return bool(_INTERACTION_RE.search(text or ""))


def _any_alias_in_blob(aliases: list[str], blob: str) -> bool:
    low = blob or ""
    for alias in aliases:
        if alias and _alias_pattern(alias).search(low):
            return True
    return False


def _ingredient_names_from(rec: dict) -> list[str]:
    """Product ingredients: attached list, or openFDA generic/substance names."""
    attached = rec.get("ingredient_names")
    if attached:
        return [str(x) for x in attached if x]
    openfda = rec.get("openfda") or {}
    out: list[str] = []
    for key in ("generic_name", "substance_name"):
        val = openfda.get(key) or []
        if isinstance(val, str):
            val = [val] if val.strip() else []
        out.extend(str(x) for x in val if x)
    return out


def _spl_contains_both_pair_members(
    ingredient_names: list[str] | None, drug_a: str, drug_b: str,
) -> bool:
    """True when this SPL already lists both pair members as ingredients."""
    names = [str(n) for n in (ingredient_names or []) if n]
    if not names:
        return False
    blob = " ".join(names).lower()
    return (
        _any_alias_in_blob(_label_aliases(drug_a), blob)
        and _any_alias_in_blob(_label_aliases(drug_b), blob)
    )


# Trailing salt/ester words; not drug names (do not include nitrate).
_SALT_SUFFIXES = frozenset({
    "hydrochloride", "hcl", "besylate", "besilate", "fumarate", "maleate",
    "succinate", "mesylate", "tartrate", "citrate", "sulfate", "sulphate",
    "phosphate", "acetate", "sodium", "potassium", "calcium", "bromide",
    "hydrobromide", "chloride", "pamoate", "embonate", "dipropionate",
    "valerate", "propionate", "dihydrate", "monohydrate", "anhydrous",
    "hydrate",
})


def _active_moieties(ingredient_names: list[str] | None) -> list[str]:
    """Distinct active ingredients, salts stripped, split on and/;/,/."""
    parts: list[str] = []
    for raw in ingredient_names or []:
        text = str(raw).lower()
        for sep in (" and ", ";", "/", ","):
            text = text.replace(sep, "|")
        for piece in text.split("|"):
            words = piece.split()
            while words and words[-1] in _SALT_SUFFIXES:
                words.pop()
            tok = " ".join(words).strip()
            if tok and tok not in parts:
                parts.append(tok)
    return parts


def _is_combination_spl(ingredient_names: list[str] | None) -> bool:
    """True when this product has two or more active ingredients."""
    return len(_active_moieties(ingredient_names)) >= 2


_NEGATIVE_INTERACTION_RE = re.compile(
    r"(?:did not|does not|do not|no(?:t)?)\s+"
    r"(?:result in\s+|cause\s+|show\s+|produce\s+|demonstrate\s+)?"
    r"(?:a\s+)?(?:clinically\s+)?(?:significant\s+)?interaction",
    re.I,
)


def _negative_pair_mention(text: str, aliases: list[str]) -> bool:
    """True when the partner is named in a 'no significant interaction' sentence."""
    low = " ".join((text or "").lower().split())
    if not low:
        return False
    idx = _first_mention(low, aliases)
    if idx is None:
        return False
    start = low.rfind(".", 0, idx) + 1
    end = low.find(".", idx)
    if end < 0:
        end = len(low)
    sent = low[start:end]
    if not _any_alias_in_blob(aliases, sent):
        return False
    return bool(_NEGATIVE_INTERACTION_RE.search(sent))


def _is_ingredient_colist(text: str, aliases: list[str]) -> bool:
    """True when the partner is named as an ingredient of this product, not a DDI."""
    low = " ".join((text or "").lower().split())
    if not low:
        return False
    for alias in aliases:
        if not alias:
            continue
        a = re.escape(alias.lower())
        pats = (
            rf"this product contains\s+{a}",
            rf"(?:each|the)\s+(?:tablet|capsule|film|product)\s+contains\s+[^.]{{0,80}}{a}",
            rf"contains\s+(?:the\s+active\s+ingredients?\s+)?{a}",
            rf"combination (?:product\s+)?(?:of|containing)\s+[^.]{{0,60}}{a}",
            rf"fixed[- ]dose combination[^.]{{0,80}}{a}",
            rf"{a}\s*/\s*[a-z][a-z0-9\- ]{{2,40}}\s+combination",
            rf"[a-z][a-z0-9\- ]{{2,40}}\s*/\s*{a}\s+combination",
        )
        if any(re.search(p, low) for p in pats):
            return True
    return False


# Class/article words allowed between "with" and the partner ("organic nitrates").
# Not "hypersensitivity" / "allergy" — those are product-ingredient CI, not a DDI.
_CI_WITH_MOD = (
    r"(?:organic|any|other|the|a|an|concomitant|strong|potent|"
    r"oral|parenteral|systemic|inhaled|topical|all)\s+"
)


def _pair_scoped_contraindication(text: str, aliases: list[str]) -> bool:
    """CI only when the partner is named with with / concomitant / coadminister.

    'Do not use … telmisartan' in a hypersensitivity sentence is not CI.
    """
    low = " ".join((text or "").lower().split())
    if not low:
        return False
    mod = _CI_WITH_MOD
    for alias in aliases:
        if not alias:
            continue
        a = re.escape(alias.lower())
        with_a = rf"with\s+(?:{mod}){{0,3}}{a}"
        pats = (
            rf"contraindicat\w*(?:\s+\w+){{0,8}}\s+(?:in\s+combination\s+)?{with_a}",
            rf"contraindicat\w*(?:\s+\w+){{0,8}}\s+"
            rf"(?:taking|receiving|using|on|treated with)\s+(?:{mod}){{0,3}}{a}",
            rf"concomitant(?:\s+\w+){{0,6}}\s+(?:{mod}){{0,3}}{a}\b[^.]{{0,80}}"
            rf"contraindicat",
            rf"concomitant(?:\s+\w+){{0,6}}\s+contraindicat\w*[^.]{{0,80}}\b{a}\b",
            rf"do\s+not\s+(?:use|give|take|combine)(?:\s+\w+){{0,6}}\s+{with_a}",
            rf"do\s+not\s+co-?administer(?:\s+\w+){{0,6}}\s+(?:{with_a}|{a})",
            rf"(?:must|should)\s+not\s+be\s+"
            rf"(?:used|given|taken|combined|co-?administered)\s+{with_a}",
            rf"co-?administ\w*(?:\s+\w+){{0,6}}\s+(?:{with_a}|{a})\b[^.]{{0,80}}"
            rf"contraindicat",
        )
        if any(re.search(p, low) for p in pats):
            return True
    return False


def _field_query(aliases: list[str], fields: tuple[str, ...]) -> str:
    clauses = [
        f'{field}:"{alias}"'
        for alias in aliases
        for field in fields
        if alias
    ]
    return "(" + " OR ".join(clauses) + ")" if clauses else ""


def _label_hit(rec: dict, subject: str, aliases: list[str],
               partner: str | None = None,
               _fields: tuple[str, ...] = LABEL_SEARCH_FIELDS) -> dict | None:
    """Window DI and CI/boxed separately. Combo SPLs of both pair drugs are not hits."""
    other = partner or (aliases[0] if aliases else "")
    ingredients = _ingredient_names_from(rec)
    if other and _spl_contains_both_pair_members(ingredients, subject, other):
        return None
    di = _window_around(_join_fields(rec, ("drug_interactions",)), aliases)
    ci = _window_around(
        _join_fields(rec, ("contraindications", "boxed_warning")), aliases)
    if di and _negative_pair_mention(di, aliases):
        di = None
    if not di and not ci:
        return None
    contra = _pair_scoped_contraindication(
        ci or "", aliases) or _pair_scoped_contraindication(di or "", aliases)
    snippet = ci if contra and ci else (di or ci)
    setid = _setid_from_record(rec)
    openfda = rec.get("openfda") or {}
    names = openfda.get("generic_name") or [subject]
    title_name = names[0] if names else subject
    return {
        "subject_drug": subject,
        "title": f"{title_name} labeling",
        "interactions_text": snippet,
        "di_text": di,
        "ci_text": ci,
        "setid": setid,
        "url": _dailymed_url(setid, subject),
        "label_contraindicated": contra,
        "ingredient_names": ingredients,
    }


async def openfda_label_check(drug_a: str, drug_b: str) -> list[dict]:
    """Search labels both ways for a partner mention in DI / CI / boxed warning.

    Hits must contain the partner (or an alias) in the kept window. Empty /
    TOC-only snippets are not records — the waterfall then continues to PubMed.
    """
    settings = get_settings()
    results: list[dict] = []
    seen: set[str] = set()
    async with _client() as client:
        for subject, other in ((drug_a, drug_b), (drug_b, drug_a)):
            subject = _clean_term(subject)
            other = _clean_term(other)
            aliases = _label_aliases(other)
            if not subject or not aliases:
                continue
            field_q = _field_query(aliases, LABEL_SEARCH_FIELDS)
            params = {
                "search": (
                    f"{field_q} AND "
                    f'(openfda.generic_name:"{subject}" OR '
                    f'openfda.brand_name:"{subject}")'
                ),
                "limit": 3,
            }
            if settings.openfda_api_key:
                params["api_key"] = settings.openfda_api_key
            try:
                r = await request_with_retry(
                    client, "GET", "https://api.fda.gov/drug/label.json",
                    tool="openfda", params=params)
                if r.status_code == 404:
                    continue
                _raise_http(r, "openfda")
                for rec in r.json().get("results", []):
                    row = _label_hit(rec, subject, aliases, partner=other)
                    if not row:
                        continue
                    key = str(row.get("setid") or row.get("url") or "")
                    if key and key in seen:
                        continue
                    if key:
                        seen.add(key)
                    results.append(row)
            except httpx.HTTPError:
                continue
    return results


async def openfda_substance_check(drug: str, terms: list[str]) -> list[dict]:
    """Search one drug's label for non-drug terms in interaction fields.

    Only this direction (drug label mentions the substance). Alcohol / herbals
    are not queried as if they were the labeled product.
    """
    drug = _clean_term(drug)
    cleaned = []
    seen: set[str] = set()
    for raw in terms:
        t = _clean_term(str(raw))
        if not t:
            continue
        key = t.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(t)
    if not drug or not cleaned:
        return []
    aliases = [t.lower() for t in cleaned]
    field_q = _field_query(cleaned, SUBSTANCE_SEARCH_FIELDS)
    settings = get_settings()
    params = {
        "search": (
            f"{field_q} AND "
            f'(openfda.generic_name:"{drug}" OR openfda.brand_name:"{drug}")'
        ),
        "limit": 3,
    }
    if settings.openfda_api_key:
        params["api_key"] = settings.openfda_api_key
    try:
        async with _client() as client:
            r = await request_with_retry(
                client, "GET", "https://api.fda.gov/drug/label.json",
                tool="openfda", params=params)
            if r.status_code == 404:
                return []
            _raise_http(r, "openfda")
            out: list[dict] = []
            for rec in r.json().get("results", []):
                di = _window_around(
                    _join_fields(rec, ("drug_interactions",)), aliases)
                food = _window_around(
                    _join_fields(rec, ("food_interactions",)), aliases)
                if not di and not food:
                    continue
                setid = _setid_from_record(rec)
                openfda = rec.get("openfda") or {}
                names = openfda.get("generic_name") or [drug]
                title_name = names[0] if names else drug
                out.append({
                    "subject_drug": drug,
                    "title": f"{title_name} labeling",
                    "interactions_text": di or "",
                    "food_interactions_text": food or "",
                    "setid": setid,
                    "url": _dailymed_url(setid, drug),
                })
            return out
    except httpx.HTTPError:
        return []


# --------------------------------------------------------------------------
# Tier 2 — PubMed (NCBI eutils)
# --------------------------------------------------------------------------
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _term_variants(name: str) -> str:
    """'amoxycillin' -> (\"amoxycillin\"[tiab] OR \"amoxicillin\"[tiab])"""
    variants = {name}
    for brit, us in SPELLING_VARIANTS.items():
        if brit in name:
            variants |= {name.replace(brit, u) for u in us}
        for u in us:
            if u in name:
                variants.add(name.replace(u, brit))
    return "(" + " OR ".join(f'"{v}"[Title/Abstract]' for v in sorted(variants)) + ")"


async def pubmed_search(drug_a: str, drug_b: str, *, retmax: int = 5) -> list[dict]:
    query = (f'{_term_variants(drug_a)} AND {_term_variants(drug_b)} AND '
             f'(drug interaction[Title/Abstract] OR pharmacokinetic*[Title/Abstract] '
             f'OR "drug-drug interaction"[Title/Abstract])')
    async with _client() as client:
        try:
            r = await request_with_retry(
                client, "GET", f"{EUTILS}/esearch.fcgi",
                tool="pubmed",
                params={"db": "pubmed", "term": query,
                        "retmax": retmax, "retmode": "json",
                        "sort": "relevance"})
            _raise_http(r, "pubmed")
            pmids = r.json().get("esearchresult", {}).get("idlist", [])
            if not pmids:
                return []
            r2 = await request_with_retry(
                client, "GET", f"{EUTILS}/esummary.fcgi",
                tool="pubmed",
                params={"db": "pubmed", "id": ",".join(pmids),
                        "retmode": "json"})
            _raise_http(r2, "pubmed")
            data = r2.json().get("result", {})
            # Fetch abstracts too — titles alone are not gradeable evidence
            abstracts = await _fetch_abstracts(client, pmids)
            out = []
            for pmid in pmids:
                rec = data.get(pmid, {})
                if not rec:
                    continue
                out.append({
                    "pmid": pmid,
                    "title": rec.get("title", ""),
                    "pubdate": rec.get("pubdate", ""),
                    "pubtype": rec.get("pubtype", []),
                    "abstract": abstracts.get(pmid, ""),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                })
            return out
        except httpx.HTTPError:
            return []


async def _fetch_abstracts(client: httpx.AsyncClient, pmids: list[str]) -> dict[str, str]:
    """efetch abstracts (truncated) keyed by PMID."""
    try:
        r = await request_with_retry(
            client, "GET", f"{EUTILS}/efetch.fcgi",
            tool="pubmed",
            params={"db": "pubmed", "id": ",".join(pmids),
                    "rettype": "abstract", "retmode": "xml"})
        _raise_http(r, "pubmed")
        out: dict[str, str] = {}
        for article in ET.fromstring(r.text).iter("PubmedArticle"):
            pmid_el = article.find(".//PMID")
            abs_els = article.findall(".//Abstract/AbstractText")
            if pmid_el is not None and abs_els:
                text = " ".join("".join(a.itertext()) for a in abs_els)
                out[pmid_el.text or ""] = text[:1800]
        return out
    except (httpx.HTTPError, ET.ParseError):
        return {}


# --------------------------------------------------------------------------
# Tier 2 — ClinicalTrials.gov v2
# --------------------------------------------------------------------------
def _plain_variants(name: str) -> list[str]:
    variants = {name}
    for brit, us in SPELLING_VARIANTS.items():
        if brit in name:
            variants |= {name.replace(brit, u) for u in us}
    return sorted(variants)


async def clinicaltrials_search(drug_a: str, drug_b: str, *, page_size: int = 5) -> list[dict]:
    term = (f'({" OR ".join(_plain_variants(drug_a))}) AND '
            f'({" OR ".join(_plain_variants(drug_b))}) AND '
            f'(interaction OR pharmacokinetics OR coadministration)')
    async with _client() as client:
        try:
            r = await request_with_retry(
                client, "GET", "https://clinicaltrials.gov/api/v2/studies",
                tool="clinicaltrials",
                params={"query.term": term, "pageSize": page_size,
                        "fields": "NCTId,BriefTitle,OverallStatus,Phase,StudyType,LeadSponsorName"})
            _raise_http(r, "clinicaltrials")
            out = []
            for st in r.json().get("studies", []):
                proto = st.get("protocolSection", {})
                ident = proto.get("identificationModule", {})
                status = proto.get("statusModule", {})
                design = proto.get("designModule", {})
                nct = ident.get("nctId", "")
                out.append({
                    "nct_id": nct,
                    "title": ident.get("briefTitle", ""),
                    "status": status.get("overallStatus", ""),
                    "phase": design.get("phases", []),
                    "url": f"https://clinicaltrials.gov/study/{nct}",
                })
            return out
        except httpx.HTTPError:
            return []


# --------------------------------------------------------------------------
# Tier 3 — Firecrawl web search / fetch (weak evidence: case reports etc.)
# --------------------------------------------------------------------------
async def web_search(query: str, *, limit: int = 5) -> list[dict]:
    settings = get_settings()
    if not settings.firecrawl_api_key:
        return []
    async with _client() as client:
        try:
            r = await request_with_retry(
                client, "POST", "https://api.firecrawl.dev/v1/search",
                tool="firecrawl",
                headers={"Authorization": f"Bearer {settings.firecrawl_api_key}"},
                json={"query": query, "limit": limit})
            _raise_http(r, "firecrawl")
            return [{"title": d.get("title", ""), "url": d.get("url", ""),
                     "snippet": d.get("description", "")}
                    for d in r.json().get("data", [])]
        except httpx.HTTPError:
            return []


async def web_fetch(url: str) -> str:
    settings = get_settings()
    if not settings.firecrawl_api_key:
        return ""
    async with _client() as client:
        try:
            r = await request_with_retry(
                client, "POST", "https://api.firecrawl.dev/v1/scrape",
                tool="firecrawl",
                headers={"Authorization": f"Bearer {settings.firecrawl_api_key}"},
                json={"url": url, "formats": ["markdown"], "onlyMainContent": True})
            _raise_http(r, "firecrawl")
            return (r.json().get("data", {}).get("markdown") or "")[:6000]
        except httpx.HTTPError:
            return ""


# Keep os import used (API keys may arrive via env passthrough for litellm)
_ = os.environ
