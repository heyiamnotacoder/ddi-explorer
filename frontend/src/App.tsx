import { useMemo, useState } from "react";
import { checkInteractions, type CheckResponse, type PairResult } from "./api";

type Filter = "all" | "flagged" | "timing";

function isDdi(p: PairResult) {
  return p.category === "contraindicated" || p.category === "timing" || p.grade != null;
}

const VIA: Record<string, string> = {
  indian_dataset: "Indian brand index",
  rxnav: "RxNorm",
};

function uniqueComponents(result: CheckResponse): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const d of result.normalized_drugs) {
    for (const c of d.components) {
      const k = c.toLowerCase();
      if (!seen.has(k)) {
        seen.add(k);
        out.push(c);
      }
    }
  }
  return out;
}

function pairKey(a: string, b: string) {
  return [a.toLowerCase(), b.toLowerCase()].sort().join("|");
}

function rank(p: PairResult) {
  if (p.category === "contraindicated") return 0;
  if (p.grade === "A") return 1;
  if (p.grade === "B") return 2;
  if (p.grade === "C") return 3;
  if (p.category === "timing") return 4;
  return 5;
}

function GradePill({ grade, category }: { grade: string | null; category?: string }) {
  if (category === "contraindicated") return <span className="pill contra">CONTRA</span>;
  if (category === "timing") return <span className="pill timing">TIMING</span>;
  if (grade === "A") return <span className="pill A">A</span>;
  if (grade === "B") return <span className="pill B">B</span>;
  if (grade === "C") return <span className="pill C">C</span>;
  return <span className="pill none">NONE</span>;
}

function PairCard({ p }: { p: PairResult }) {
  return (
    <article id={`pair-${pairKey(p.drugs[0], p.drugs[1])}`} className={`pair ${p.category}`}>
      <div className="pair-top">
        <h4>{p.drugs[0]} + {p.drugs[1]}</h4>
        <div className="pair-meta">
          {p.severity && <span className="sev">{p.severity}</span>}
          <GradePill grade={p.grade} category={p.category} />
        </div>
      </div>
      <p>{p.summary}</p>
      {p.mechanism && <p className="mech"><em>Mechanism.</em> {p.mechanism}</p>}
      {p.dose_condition && <p className="dose">{p.dose_condition}</p>}
      {p.patient_specific_note && <p className="pt">For this patient: {p.patient_specific_note}</p>}
      {p.evidence_conflict && <p className="conflict">Conflicting evidence: {p.evidence_conflict}</p>}
      {p.severe_if.length > 0 && (
        <div className="severe">
          <strong>More severe if</strong>
          <ul>{p.severe_if.map((s, i) => <li key={i}>{s}</li>)}</ul>
        </div>
      )}
      {p.citations.length > 0 && (
        <div className="cites">
          Sources:{" "}
          {p.citations.map((c, i) =>
            c.url ? (
              <a key={i} href={c.url} target="_blank" rel="noreferrer">
                [{c.source}{c.identifier ? ` ${c.identifier}` : ""}]
              </a>
            ) : (
              <span key={i}>[{c.source}{c.identifier ? ` ${c.identifier}` : ""}]</span>
            ),
          )}
        </div>
      )}
    </article>
  );
}

function PairMatrix({ components, pairs }: { components: string[]; pairs: PairResult[] }) {
  if (components.length < 2) return null;
  const map = new Map(pairs.filter(isDdi).map((p) => [pairKey(p.drugs[0], p.drugs[1]), p]));
  return (
    <div className="matrix-wrap">
      <table className="matrix">
        <thead>
          <tr>
            <th />
            {components.map((c) => <th key={c}>{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {components.map((row, i) => (
            <tr key={row}>
              <th>{row}</th>
              {components.map((col, j) => {
                if (i === j) return <td key={col} className="diag">·</td>;
                if (j < i) return <td key={col} />;
                const p = map.get(pairKey(row, col));
                if (!p) return <td key={col}>—</td>;
                const label = p.category === "contraindicated" ? "X" : p.grade ?? "–";
                return (
                  <td key={col}>
                    <a href={`#pair-${pairKey(row, col)}`}>
                      <GradePill grade={p.grade} category={p.category} />
                      <span className="sr-only">{label}</span>
                    </a>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
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
  const [filter, setFilter] = useState<Filter>("all");

  const onFiles = (files: FileList | null) => {
    if (!files) return;
    Array.from(files).forEach((f) => {
      const reader = new FileReader();
      reader.onload = () => setImages((prev) => [...prev, reader.result as string]);
      reader.readAsDataURL(f);
    });
  };

  const run = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setFilter("all");
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

  const components = result ? uniqueComponents(result) : [];
  const expectedPairs = components.length >= 2
    ? (components.length * (components.length - 1)) / 2
    : 0;

  const ddiPairs = useMemo(() => {
    if (!result) return [];
    return result.pairs.filter(isDdi).sort((a, b) => rank(a) - rank(b));
  }, [result]);

  const counts = useMemo(() => {
    const c = { flagged: 0, timing: 0 };
    for (const p of ddiPairs) {
      if (p.category === "timing") c.timing += 1;
      else c.flagged += 1;
    }
    return c;
  }, [ddiPairs]);

  const visible = ddiPairs.filter((p) => {
    if (filter === "timing") return p.category === "timing";
    if (filter === "flagged") return p.category !== "timing";
    return true;
  });

  const fdcs = result?.normalized_drugs.filter((d) => d.components.length > 1) ?? [];

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="mark" aria-hidden>D×D</div>
          <div>
            <h1>DDI Explorer</h1>
            <p className="lede">Evidence-graded drug–drug interaction check for clinicians</p>
          </div>
        </div>
        <p className="top-note">
          Decision support only. Not a substitute for a clinical pharmacist or approved labeling.
        </p>
      </header>

      <section className="explainer">
        <div className="panel panel-pad">
          <p className="kicker">What this does</p>
          <h2>Paste a prescription. Every component pair is graded against retrieved evidence.</h2>
          <p className="intro">
            Typed text and prescription photos are de-identified on the server before any
            reasoning model sees them. Indian brand names resolve to generics; combination
            products are split. Known pairs come from labeling and RxNav; the rest go through
            an evidence waterfall and stop at the first solid source.
          </p>
          <ol className="steps">
            <li><span className="n">1</span><div><strong>Read</strong> <span>text and/or a photo of the Rx</span></div></li>
            <li><span className="n">2</span><div><strong>Strip identifiers</strong> <span>names, phones, Aadhaar, MRN — before any LLM</span></div></li>
            <li><span className="n">3</span><div><strong>Resolve</strong> <span>Indian brands → RxNorm generics; FDCs split into ingredients</span></div></li>
            <li><span className="n">4</span><div><strong>Pair</strong> <span>every ingredient with every other ingredient from a different product</span></div></li>
            <li><span className="n">5</span><div><strong>Grade</strong> <span>A labeling · B trials · C case reports · or no DDI — with citations</span></div></li>
          </ol>
        </div>
        <div className="panel panel-pad">
          <p className="kicker">How to read a grade</p>
          <div className="grades">
            <div className="grade-row">
              <span className="pill A">A</span>
              <p><b>Approved / known.</b> FDA label “Drug Interactions” or RxNav. Strongest source we accept.</p>
            </div>
            <div className="grade-row">
              <span className="pill B">B</span>
              <p><b>Human trial literature.</b> RCT, PK study, meta-analysis, or a registered trial — not yet in labeling.</p>
            </div>
            <div className="grade-row">
              <span className="pill C">C</span>
              <p><b>Weak evidence.</b> Case reports, in-vitro, or mechanism only. Conflicts are disclosed, not hidden.</p>
            </div>
            <div className="grade-row">
              <span className="pill none">NONE</span>
              <p><b>No DDI found</b> in the sources we retrieved. Absence of evidence is stated, never inferred as safe.</p>
            </div>
          </div>
        </div>
      </section>

      <section className="workspace panel panel-pad">
        <p className="kicker">Check a list</p>
        <div className="rx-box">
          <label className="field" htmlFor="rx">
            Prescription or medication list
          </label>
          <textarea
            id="rx"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={"Tab Telma 40  1-0-0\nTab Dolo 650  1-0-1\nTab Ecosprin 75  0-1-0\n…or just names, one per line."}
          />
        </div>
        <div className="side-fields">
          <div className="field">
            <label htmlFor="pt">Patient context (optional)</label>
            <textarea
              id="pt"
              value={patient}
              onChange={(e) => setPatient(e.target.value)}
              placeholder="Age, sex, weight, CrCl / eGFR, hepatic function, pregnancy, comorbidities"
            />
          </div>
          <div className="field">
            <label htmlFor="tm">Timing (optional)</label>
            <textarea
              id="tm"
              value={timing}
              onChange={(e) => setTiming(e.target.value)}
              placeholder="e.g. levothyroxine 1-0-0 empty stomach; calcium 0-0-1"
            />
          </div>
        </div>
        <div className="toolbar">
          <div>
            <label className="file-btn">
              Attach prescription photo
              <input type="file" accept="image/*" multiple onChange={(e) => onFiles(e.target.files)} />
            </label>
            {images.length > 0 && (
              <div className="thumbs" style={{ marginTop: 8 }}>
                {images.map((src, i) => (
                  <div className="thumb" key={i}>
                    <img src={src} alt={`Prescription ${i + 1}`} />
                    <button type="button" onClick={() => setImages((prev) => prev.filter((_, j) => j !== i))} aria-label="Remove image">×</button>
                  </div>
                ))}
              </div>
            )}
          </div>
          <button className="run" onClick={run} disabled={loading || (!text && images.length === 0)}>
            {loading ? "Checking pairs…" : "Check interactions"}
          </button>
        </div>
        <p className="hint">
          A full check takes about 30–40 seconds. Identifiers never reach the reasoning model.
          Five single-ingredient drugs produce 10 pairs — C(n, 2). Combination products add more.
        </p>
      </section>

      {error && <div className="error" role="alert">{error}</div>}

      {loading && (
        <div className="loading panel">
          <h3>Working through the list</h3>
          <p>
            De-identifying the text, extracting medicines, resolving Indian brands to generics,
            then grading each pair. Known interactions return from RxNav without a model call;
            unknown pairs walk openFDA → PubMed / trials → case reports.
          </p>
        </div>
      )}

      {result && (
        <div className="results">
          <section className="panel resolved">
            <div className="resolved-head">
              <h3>Resolved medicines</h3>
              <p className="pair-math">
                {result.normalized_drugs.length} input{result.normalized_drugs.length === 1 ? "" : "s"}
                {" → "}
                <b>{components.length} component{components.length === 1 ? "" : "s"}</b>
                {" → "}
                <b>{ddiPairs.length} DDI{ddiPairs.length === 1 ? "" : "s"}</b>
                {` of ${result.pairs.length} pair${result.pairs.length === 1 ? "" : "s"} checked`}
                {expectedPairs > 0 && result.pairs.length === expectedPairs
                  ? ` · C(${components.length}, 2)`
                  : null}
              </p>
            </div>
            <div className="chips">
              {result.normalized_drugs.map((d) => (
                <div className="chip" key={d.input_name}>
                  <span className="from">{d.input_name}</span>
                  <span className="to">
                    {d.components.length ? d.components.join(" + ") : (d.generic_name || "unresolved")}
                  </span>
                  <span className="via">{d.resolved_via ? (VIA[d.resolved_via] || d.resolved_via) : "not in index"}</span>
                </div>
              ))}
            </div>
            {fdcs.length > 0 && (
              <p className="fdc-note">
                Combination product{fdcs.length > 1 ? "s" : ""} split into ingredients
                ({fdcs.map((d) => `${d.input_name} → ${d.components.join(" + ")}`).join("; ")}).
                Each ingredient is paired with every ingredient from the other medicines.
              </p>
            )}
            <PairMatrix components={components} pairs={result.pairs} />
          </section>

          {result.contraindicated_banner.length > 0 && (
            <div className="banner" role="alert">
              <strong>CONTRAINDICATED</strong>
              {result.contraindicated_banner.map((p, i) => (
                <div key={i}>{p.drugs[0]} + {p.drugs[1]} — {p.summary}</div>
              ))}
            </div>
          )}

          {result.unresolved_drugs.length > 0 && (
            <div className="warn-box">
              Could not resolve: {result.unresolved_drugs.join(", ")}. Check spelling or use the generic name.
            </div>
          )}

          {result.avoid_with_medications.length > 0 && (
            <div className="avoid-box">
              <strong>Avoid with these medications</strong>
              <ul>{result.avoid_with_medications.map((a, i) => <li key={i}>{a}</li>)}</ul>
            </div>
          )}

          {ddiPairs.length > 0 && (
            <h3 className="ddi-heading">DDI pairs ({ddiPairs.length})</h3>
          )}
          {ddiPairs.length > 0 && counts.timing > 0 && (
            <div className="tabs" role="tablist">
              <button className={filter === "all" ? "on" : ""} onClick={() => setFilter("all")} type="button">
                All DDIs ({ddiPairs.length})
              </button>
              <button className={filter === "flagged" ? "on" : ""} onClick={() => setFilter("flagged")} type="button">
                Interactions ({counts.flagged})
              </button>
              <button className={filter === "timing" ? "on" : ""} onClick={() => setFilter("timing")} type="button">
                Timing-manageable ({counts.timing})
              </button>
            </div>
          )}

          {visible.map((p, i) => <PairCard key={i} p={p} />)}
          {visible.length === 0 && (
            <p className="empty-pairs">No drug–drug interaction found among the checked pairs.</p>
          )}

          <details className="audit">
            <summary>What the model saw (de-identified input)</summary>
            <pre>{result.scrubbed_text}</pre>
          </details>

          <p className="disclaimer">{result.disclaimer}</p>
        </div>
      )}
    </div>
  );
}
