export interface Citation {
  source: string;
  title: string;
  url?: string | null;
  identifier?: string | null;
}

export interface PairResult {
  drugs: [string, string];
  grade: "A" | "B" | "C" | null;
  category: "interaction" | "timing" | "contraindicated" | "none";
  severity?: string | null;
  summary: string;
  mechanism?: string | null;
  citations: Citation[];
  severe_if: string[];
  patient_specific_note?: string | null;
  evidence_conflict?: string | null;
  dose_condition?: string | null;
  source_tier?: string | null;
}

export interface NormalizedDrug {
  input_name: string;
  generic_name?: string | null;
  rxcui?: string | null;
  components: string[];
  component_rxcuis?: Record<string, string>;
  resolved_via?: string | null;
  dose?: string | null;
  schedule?: string | null;
}

export interface AvoidWithItem {
  substance: string;
  medications: string[];
  note: string;
  citations: Citation[];
}

export interface CheckResponse {
  scrubbed_text: string;
  normalized_drugs: NormalizedDrug[];
  unresolved_drugs: string[];
  pairs: PairResult[];
  contraindicated_banner: PairResult[];
  avoid_with_medications: AvoidWithItem[];
  insufficient_evidence: [string, string][];
  disclaimer: string;
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`API error ${r.status}`);
  return r.json() as Promise<T>;
}

export async function checkInteractions(body: {
  text?: string;
  images?: string[];
  patient_context?: string;
  timing?: string;
}): Promise<CheckResponse> {
  return postJson<CheckResponse>("/api/check", body);
}

export type Importance = "anchor" | "controller" | "adjuvant";

export interface ReplaceableDrug {
  name: string;
  input_name: string;
  importance: Importance;
  why_this_one: string;
  involved_pairs: [string, string][];
}

export interface AlternativeSuggestion {
  change_from: string;
  change_from_product: string;
  change_to: string;
  change_to_components: string[];
  indication: string;
  rationale: string;
  adr_note: string;
  safer: boolean;
  reject_reason?: string | null;
  remaining_ddis: PairResult[];
}

export interface AlternativesResponse {
  strategy: string;
  replaceable: ReplaceableDrug[];
  suggestions: AlternativeSuggestion[];
  keep: ReplaceableDrug[];
  timing_first: string[];
  disclaimer: string;
}

export async function suggestAlternatives(body: {
  normalized_drugs: NormalizedDrug[];
  pairs: PairResult[];
  patient_context?: string;
  scrubbed_text?: string;
  avoid_with_medications?: AvoidWithItem[];
}): Promise<AlternativesResponse> {
  return postJson<AlternativesResponse>("/api/alternatives", body);
}
