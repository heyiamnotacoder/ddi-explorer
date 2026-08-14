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
  resolved_via?: string | null;
}

export interface CheckResponse {
  scrubbed_text: string;
  normalized_drugs: NormalizedDrug[];
  unresolved_drugs: string[];
  pairs: PairResult[];
  contraindicated_banner: PairResult[];
  avoid_with_medications: string[];
  insufficient_evidence: [string, string][];
  disclaimer: string;
}

export async function checkInteractions(body: {
  text?: string;
  images?: string[];
  patient_context?: string;
  timing?: string;
}): Promise<CheckResponse> {
  const r = await fetch("/api/check", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`API error ${r.status}`);
  return r.json();
}
