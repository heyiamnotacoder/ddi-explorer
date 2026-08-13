"""Evidence tools available to the agent's waterfall.

  tier 1: openfda_label_check   -> approved labeling (Grade A)
  tier 2: pubmed_search         -> PubMed eutils (Grade B candidates)
          clinicaltrials_search -> ClinicalTrials.gov v2 (Grade B candidates)
  tier 3: web_search/web_fetch  -> Firecrawl (Grade C candidates)

Each tool returns compact dicts the waterfall feeds to the LLM for
synthesis. The LLM never fabricates: no retrieved record -> no citation.
"""
from __future__ import annotations

import os
import urllib.parse
import xml.etree.ElementTree as ET

import httpx

from ..config import get_settings


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=get_settings().http_timeout)


# --------------------------------------------------------------------------
# Tier 1 — openFDA drug labels
# --------------------------------------------------------------------------
async def openfda_label_check(drug_a: str, drug_b: str) -> list[dict]:
    """Search openFDA labels for `drug_b` mentioned in `drug_a`'s
    drug_interactions section (and vice versa)."""
    settings = get_settings()
    results = []
    async with _client() as client:
        for subject, other in ((drug_a, drug_b), (drug_b, drug_a)):
            params = {
                "search": (f'drug_interactions:"{other}" AND '
                           f'(openfda.generic_name:"{subject}" OR openfda.brand_name:"{subject}")'),
                "limit": 3,
            }
            if settings.openfda_api_key:
                params["api_key"] = settings.openfda_api_key
            try:
                r = await client.get("https://api.fda.gov/drug/label.json", params=params)
                if r.status_code == 404:
                    continue
                r.raise_for_status()
                for rec in r.json().get("results", []):
                    results.append({
                        "subject_drug": subject,
                        "interactions_text": " ".join(rec.get("drug_interactions", []))[:3000],
                        "setid": rec.get("setid"),
                        "url": f"https://dailymed.nlm.nih.gov/dailymed/search.cfm?query={urllib.parse.quote(subject)}",
                    })
            except httpx.HTTPError:
                continue
    return results


# --------------------------------------------------------------------------
# Tier 2 — PubMed (NCBI eutils)
# --------------------------------------------------------------------------
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

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
}


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
            r = await client.get(f"{EUTILS}/esearch.fcgi",
                                 params={"db": "pubmed", "term": query,
                                         "retmax": retmax, "retmode": "json",
                                         "sort": "relevance"})
            r.raise_for_status()
            pmids = r.json().get("esearchresult", {}).get("idlist", [])
            if not pmids:
                return []
            r2 = await client.get(f"{EUTILS}/esummary.fcgi",
                                  params={"db": "pubmed", "id": ",".join(pmids),
                                          "retmode": "json"})
            r2.raise_for_status()
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
        r = await client.get(f"{EUTILS}/efetch.fcgi",
                             params={"db": "pubmed", "id": ",".join(pmids),
                                     "rettype": "abstract", "retmode": "xml"})
        r.raise_for_status()
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
            r = await client.get(
                "https://clinicaltrials.gov/api/v2/studies",
                params={"query.term": term, "pageSize": page_size,
                        "fields": "NCTId,BriefTitle,OverallStatus,Phase,StudyType,LeadSponsorName"})
            r.raise_for_status()
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
            r = await client.post(
                "https://api.firecrawl.dev/v1/search",
                headers={"Authorization": f"Bearer {settings.firecrawl_api_key}"},
                json={"query": query, "limit": limit})
            r.raise_for_status()
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
            r = await client.post(
                "https://api.firecrawl.dev/v1/scrape",
                headers={"Authorization": f"Bearer {settings.firecrawl_api_key}"},
                json={"url": url, "formats": ["markdown"], "onlyMainContent": True})
            r.raise_for_status()
            return (r.json().get("data", {}).get("markdown") or "")[:6000]
        except httpx.HTTPError:
            return ""


# Keep os import used (API keys may arrive via env passthrough for litellm)
_ = os.environ
