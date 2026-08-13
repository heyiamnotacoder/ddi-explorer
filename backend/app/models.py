"""API schemas — the contract between pipeline, agent, and frontend."""
from enum import Enum

from pydantic import BaseModel, Field


class Grade(str, Enum):
    A = "A"  # approved labeling / known compendia
    B = "B"  # human RCT / PK / meta-analysis, not in labeling
    C = "C"  # case reports, in-vitro, mechanistic extrapolation


class Category(str, Enum):
    INTERACTION = "interaction"
    TIMING = "timing"              # manageable by separation/scheduling
    CONTRAINDICATED = "contraindicated"
    NONE = "none"                  # no DDI found


class Citation(BaseModel):
    source: str          # "openfda" | "pubmed" | "clinicaltrials" | "web"
    title: str
    url: str | None = None
    identifier: str | None = None  # PMID / NCT ID / label setid


class PairResult(BaseModel):
    drugs: tuple[str, str]
    grade: Grade | None = None
    category: Category = Category.NONE
    severity: str | None = None            # e.g. "major" | "moderate" | "minor"
    summary: str = ""
    mechanism: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    severe_if: list[str] = Field(default_factory=list)   # risk scenarios when no patient context
    patient_specific_note: str | None = None             # severity given THIS patient
    evidence_conflict: str | None = None                 # disclosed when sources disagree
    dose_condition: str | None = None                    # "DDI possible if Drug A > 500 mg"
    source_tier: str | None = None           # "local" | "openfda" | "pubmed_ct" | "web"


class NormalizedDrug(BaseModel):
    input_name: str
    generic_name: str | None = None
    rxcui: str | None = None
    components: list[str] = Field(default_factory=list)  # >1 for FDCs/combos
    resolved_via: str | None = None  # "indian_dataset" | "rxnav" | "agent_web" | None


class CheckRequest(BaseModel):
    text: str | None = None
    images: list[str] = Field(default_factory=list)  # base64 data-URLs
    patient_context: str | None = None
    timing: str | None = None


class CheckResponse(BaseModel):
    scrubbed_text: str                       # what the agent actually saw (transparency)
    normalized_drugs: list[NormalizedDrug]
    unresolved_drugs: list[str]
    pairs: list[PairResult]
    contraindicated_banner: list[PairResult]
    avoid_with_medications: list[str]        # alcohol/tobacco/herbal notes
    insufficient_evidence: list[tuple[str, str]]
    disclaimer: str
