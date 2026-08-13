import { useState } from "react";
import { checkInteractions, type CheckResponse, type PairResult } from "./api";

const GRADE_STYLE: Record<string, { bg: string; label: string }> = {
  A: { bg: "#16a34a", label: "A — Approved labeling" },
  B: { bg: "#d97706", label: "B — Trial literature" },
  C: { bg: "#dc2626", label: "C — Weak evidence" },
  none: { bg: "#6b7280", label: "No DDI found" },
};

function GradeBadge({ grade }: { grade: string | null }) {
  const s = GRADE_STYLE[grade ?? "none"];
  return (
    <span style={{ background: s.bg, color: "#fff", borderRadius: 6, padding: "2px 10px", fontWeight: 700, fontSize: 13 }}>
      {s.label}
    </span>
  );
}

function PairCard({ p }: { p: PairResult }) {
  return (
    <div style={{ border: "1px solid #e5e7eb", borderRadius: 10, padding: 14, marginBottom: 12, background: p.category === "contraindicated" ? "#fef2f2" : p.category === "timing" ? "#eff6ff" : "#fff" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
        <strong style={{ fontSize: 16 }}>{p.drugs[0]} + {p.drugs[1]}</strong>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          {p.severity && <span style={{ fontSize: 12, color: "#6b7280" }}>severity: {p.severity}</span>}
          {p.category === "timing" && <span style={{ fontSize: 12, background: "#2563eb", color: "#fff", borderRadius: 6, padding: "2px 8px" }}>manageable by timing</span>}
          <GradeBadge grade={p.grade} />
        </div>
      </div>
      <p style={{ margin: "8px 0 4px" }}>{p.summary}</p>
      {p.mechanism && <p style={{ margin: "4px 0", fontSize: 13, color: "#4b5563" }}><em>Mechanism:</em> {p.mechanism}</p>}
      {p.dose_condition && <p style={{ margin: "4px 0", fontSize: 13, color: "#92400e" }}>⚖ {p.dose_condition}</p>}
      {p.patient_specific_note && <p style={{ margin: "4px 0", fontSize: 13, color: "#7c3aed" }}>👤 For this patient: {p.patient_specific_note}</p>}
      {p.evidence_conflict && <p style={{ margin: "4px 0", fontSize: 13, color: "#b45309" }}>⚠ Conflicting evidence: {p.evidence_conflict}</p>}
      {p.severe_if.length > 0 && (
        <div style={{ margin: "6px 0", fontSize: 13 }}>
          <strong>DDI will be severe if:</strong>
          <ul style={{ margin: "4px 0" }}>{p.severe_if.map((s, i) => <li key={i}>{s}</li>)}</ul>
        </div>
      )}
      {p.citations.length > 0 && (
        <div style={{ fontSize: 12, color: "#6b7280", marginTop: 6 }}>
          Sources: {p.citations.map((c, i) => (
            <span key={i} style={{ marginRight: 10 }}>
              {c.url ? <a href={c.url} target="_blank" rel="noreferrer">[{c.source}{c.identifier ? ` ${c.identifier}` : ""}]</a> : `[${c.source}${c.identifier ? ` ${c.identifier}` : ""}]`}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [text, setText] = useState("");
  const [patient, setPatient] = useState("");
  const [timing, setTiming] = useState("");
  const [images, setImages] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CheckResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showScrubbed, setShowScrubbed] = useState(false);

  const onFiles = (files: FileList | null) => {
    if (!files) return;
    Array.from(files).forEach((f) => {
      const reader = new FileReader();
      reader.onload = () => setImages((prev) => [...prev, reader.result as string]);
      reader.readAsDataURL(f);
    });
  };

  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const res = await checkInteractions({
        text: text || undefined,
        images: images.length ? images : undefined,
        patient_context: patient || undefined,
        timing: timing || undefined,
      });
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const sorted = result ? [...result.pairs].sort((a, b) => {
    const rank = (p: PairResult) => p.category === "contraindicated" ? 0 : p.grade === "A" ? 1 : p.grade === "B" ? 2 : p.grade === "C" ? 3 : 4;
    return rank(a) - rank(b);
  }) : [];

  return (
    <div style={{ maxWidth: 860, margin: "0 auto", padding: 24, fontFamily: "system-ui, sans-serif" }}>
      <h1 style={{ marginBottom: 4 }}>DDI Explorer</h1>
      <p style={{ color: "#6b7280", marginTop: 0 }}>Evidence-graded drug–drug interaction checking. Patient identifiers are removed before any AI processing.</p>

      <textarea value={text} onChange={(e) => setText(e.target.value)} rows={5}
        placeholder={"Type or paste a prescription, e.g.\nTab Telma 40 1-0-0\nTab Dolo 650 1-0-1\n..."}
        style={{ width: "100%", borderRadius: 8, border: "1px solid #d1d5db", padding: 10, fontSize: 14 }} />

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, margin: "10px 0" }}>
        <textarea value={patient} onChange={(e) => setPatient(e.target.value)} rows={2}
          placeholder="Patient context (optional): age, sex, weight, renal/hepatic function, pregnancy..."
          style={{ borderRadius: 8, border: "1px solid #d1d5db", padding: 10, fontSize: 13 }} />
        <textarea value={timing} onChange={(e) => setTiming(e.target.value)} rows={2}
          placeholder="Timing (optional): e.g. drug A 1-0-1 before food, drug B 0-0-1..."
          style={{ borderRadius: 8, border: "1px solid #d1d5db", padding: 10, fontSize: 13 }} />
      </div>

      <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 14 }}>
        <label style={{ fontSize: 14 }}>
          📷 Prescription photo: <input type="file" accept="image/*" multiple onChange={(e) => onFiles(e.target.files)} />
        </label>
        {images.length > 0 && <span style={{ fontSize: 13, color: "#6b7280" }}>{images.length} image(s) attached</span>}
        <button onClick={run} disabled={loading || (!text && images.length === 0)}
          style={{ marginLeft: "auto", background: "#111827", color: "#fff", border: 0, borderRadius: 8, padding: "10px 22px", fontSize: 15, cursor: "pointer", opacity: loading ? 0.6 : 1 }}>
          {loading ? "Checking…" : "Check interactions"}
        </button>
      </div>

      {error && <div style={{ background: "#fef2f2", color: "#b91c1c", padding: 12, borderRadius: 8 }}>{error}</div>}

      {result && (
        <div>
          {result.contraindicated_banner.length > 0 && (
            <div style={{ background: "#991b1b", color: "#fff", borderRadius: 10, padding: 14, marginBottom: 16 }}>
              <strong>⛔ CONTRAINDICATED COMBINATIONS</strong>
              {result.contraindicated_banner.map((p, i) => (
                <div key={i} style={{ marginTop: 6 }}>{p.drugs[0]} + {p.drugs[1]} — {p.summary}</div>
              ))}
            </div>
          )}

          {result.unresolved_drugs.length > 0 && (
            <div style={{ background: "#fffbeb", border: "1px solid #f59e0b", borderRadius: 8, padding: 10, marginBottom: 12, fontSize: 13 }}>
              Could not resolve: {result.unresolved_drugs.join(", ")} — check spelling or use generic names.
            </div>
          )}

          {result.avoid_with_medications.length > 0 && (
            <div style={{ background: "#fdf4ff", border: "1px solid #c084fc", borderRadius: 8, padding: 10, marginBottom: 12, fontSize: 13 }}>
              <strong>Avoid with medications:</strong>
              <ul style={{ margin: "4px 0" }}>{result.avoid_with_medications.map((a, i) => <li key={i}>{a}</li>)}</ul>
            </div>
          )}

          <h2 style={{ fontSize: 18 }}>Results ({result.pairs.length} pairs)</h2>
          {sorted.map((p, i) => <PairCard key={i} p={p} />)}
          {result.pairs.length === 0 && <p>No drug pairs to check.</p>}

          <details style={{ marginTop: 12 }} open={showScrubbed} onToggle={(e) => setShowScrubbed((e.target as HTMLDetailsElement).open)}>
            <summary style={{ cursor: "pointer", fontSize: 13, color: "#6b7280" }}>🔒 What the AI saw (de-identified input)</summary>
            <pre style={{ background: "#f3f4f6", padding: 10, borderRadius: 8, fontSize: 12, whiteSpace: "pre-wrap" }}>{result.scrubbed_text}</pre>
          </details>

          <p style={{ fontSize: 12, color: "#9ca3af", marginTop: 16, borderTop: "1px solid #e5e7eb", paddingTop: 10 }}>{result.disclaimer}</p>
        </div>
      )}
    </div>
  );
}
