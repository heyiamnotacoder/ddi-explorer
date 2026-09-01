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
    rxcui: str | None = None  # first resolved component; prefer rxcui_for()
    components: list[str] = Field(default_factory=list)  # >1 for FDCs/combos
    component_rxcuis: dict[str, str] = Field(default_factory=dict)
    resolved_via: str | None = None  # "indian_dataset" | "rxnav" | "agent_web" | None
    dose: str | None = None          # from extract, e.g. "5 mg"
    schedule: str | None = None      # from extract, e.g. "1-0-1" / "BD"

    def rxcui_for(self, component: str) -> str | None:
        """RxCUI for one ingredient. Never reuse a sibling FDC component's id."""
        key = component.lower()
        for name, cui in self.component_rxcuis.items():
            if name.lower() == key and cui:
                return cui
        if (
            len(self.components) == 1
            and self.components[0].lower() == key
            and self.rxcui
        ):
            return self.rxcui
        return None


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


class Importance(str, Enum):
    ANCHOR = "anchor"          # withdrawal can worsen serious disease / high-ADR
    CONTROLLER = "controller"  # chronic disease-modifying; change only if needed
    ADJUVANT = "adjuvant"      # symptomatic / PRN — prefer changing these


class AlternativesRequest(BaseModel):
    """Second loop: whole-prescription substitution, not a per-pair lookup."""
    normalized_drugs: list[NormalizedDrug]
    pairs: list[PairResult]
    patient_context: str | None = None
    scrubbed_text: str | None = None  # already de-identified /api/check input
    avoid_with_medications: list[str] = Field(default_factory=list)


class ReplaceableDrug(BaseModel):
    name: str
    input_name: str
    importance: Importance
    why_this_one: str
    involved_pairs: list[tuple[str, str]] = Field(default_factory=list)


class AlternativeSuggestion(BaseModel):
    change_from: str
    change_from_product: str
    change_to: str
    change_to_components: list[str] = Field(default_factory=list)
    indication: str = ""
    rationale: str = ""
    adr_note: str = ""
    safer: bool = False
    reject_reason: str | None = None
    remaining_ddis: list[PairResult] = Field(default_factory=list)


class AlternativesResponse(BaseModel):
    strategy: str
    replaceable: list[ReplaceableDrug] = Field(default_factory=list)
    suggestions: list[AlternativeSuggestion] = Field(default_factory=list)
    keep: list[ReplaceableDrug] = Field(default_factory=list)
    timing_first: list[str] = Field(default_factory=list)
    disclaimer: str
