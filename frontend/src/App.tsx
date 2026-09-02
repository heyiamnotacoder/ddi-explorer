import { useMemo, useState } from "react";
import {
  checkInteractions,
  suggestAlternatives,
  type AlternativesResponse,
  type AlternativeSuggestion,
  type AvoidWithItem,
  type CheckResponse,
  type Citation,
  type Importance,
  type PairResult,
} from "./api";
import {
  clearHistory,
  loadHistory,
  saveCheck,
  snapshotLabel,
  type CheckSnapshot,
} from "./history";

type Filter = "all" | "flagged" | "timing";

function isDdi(p: PairResult) {
  return p.category === "contraindicated" || p.category === "timing" || p.grade != null;
}

function isInsufficient(p: PairResult) {
  return p.source_tier === "insufficient";
}

const VIA: Record<string, string> = {
  indian_dataset: "Indian brand index",
  rxnav: "RxNorm",
  agent_web: "web verification",
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

function GradePill({
  grade,
  category,
  sourceTier,
}: {
  grade: string | null;
  category?: string;
  sourceTier?: string | null;
}) {
  if (sourceTier === "insufficient") return <span className="pill insuff">INSUFF</span>;
  if (category === "contraindicated") {
    return (
      <span className="pill contra" title="Contraindicated">
        <span aria-hidden="true">⚠</span> CONTRA
      </span>
    );
  }
  if (category === "timing") return <span className="pill timing">TIMING</span>;
  if (grade === "A") return <span className="pill A">A</span>;
  if (grade === "B") return <span className="pill B">B</span>;
  if (grade === "C") return <span className="pill C">C</span>;
  return <span className="pill none">NONE</span>;
}

const IMP_LABEL: Record<Importance, string> = {
  adjuvant: "symptomatic",
  controller: "disease-modifying",
  anchor: "do not stop casually",
};

function SuggestionCard({ s }: { s: AlternativeSuggestion }) {
  return (
    <article className={`alt-card ${s.safer ? "safer" : "not-safer"}`}>
      <div className="alt-swap">
        <div>
          <p className="alt-k">Change</p>
          <h4>{s.change_from} → {s.change_to}</h4>
          <p className="alt-prod">On the list as {s.change_from_product}</p>
        </div>
        {s.safer
          ? <span className="pill A">SAFER</span>
          : <span className="pill none">NOT SAFER</span>}
      </div>
      {s.indication && <p><em>Indication.</em> {s.indication}</p>}
      {s.rationale && <p>{s.rationale}</p>}
      {s.adr_note && (
        <p className="alt-adr"><em>ADR / disease risk.</em> {s.adr_note}</p>
      )}
      {s.reject_reason && <p className="alt-reject">{s.reject_reason}</p>}
      {s.remaining_ddis.length > 0 && (
        <div className="alt-recheck">
          <strong>Recheck vs the rest of the list</strong>
          <ul>
            {s.remaining_ddis.map((p, i) => (
              <li key={i}>
                {p.drugs[0]} + {p.drugs[1]}{" "}
                <GradePill grade={p.grade} category={p.category} sourceTier={p.source_tier} />
                {p.summary ? ` — ${p.summary}` : ""}
              </li>
            ))}
          </ul>
        </div>
      )}
      {s.safer && s.remaining_ddis.length === 0 && (
        <p className="alt-clear">No leftover DDI with the remaining medicines.</p>
      )}
    </article>
  );
}

function PairCard({ p }: { p: PairResult }) {
  const kind = isInsufficient(p) ? "insufficient" : p.category;
  const conflict = Boolean(p.evidence_conflict);
  return (
    <article
      id={`pair-${pairKey(p.drugs[0], p.drugs[1])}`}
      className={`pair ${kind}${conflict ? " has-conflict" : ""}`}
    >
      <div className="pair-top">
        <h4>{p.drugs[0]} + {p.drugs[1]}</h4>
        <div className="pair-meta">
          {p.severity && <span className="sev">{p.severity}</span>}
          <GradePill grade={p.grade} category={p.category} sourceTier={p.source_tier} />
          {conflict && <span className="pill conflict">CONFLICT</span>}
        </div>
      </div>
      <p>{p.summary}</p>
      {p.mechanism && <p className="mech"><em>Mechanism.</em> {p.mechanism}</p>}
      {p.dose_condition && <p className="dose">{p.dose_condition}</p>}
      {p.patient_specific_note && <p className="pt">For this patient: {p.patient_specific_note}</p>}
      {p.evidence_conflict && (
        <p className="conflict" role="status">
          <strong>Conflicting evidence.</strong> {p.evidence_conflict}
        </p>
      )}
      {p.severe_if.length > 0 && (
        <div className="severe">
          <strong>More severe if</strong>
          <ul>{p.severe_if.map((s, i) => <li key={i}>{s}</li>)}</ul>
        </div>
      )}
      <CiteList citations={p.citations} />
    </article>
  );
}

function CiteList({ citations }: { citations: Citation[] }) {
  if (citations.length === 0) return null;
  return (
    <div className="cites">
      Sources:{" "}
      {citations.map((c, i) =>
        c.url ? (
          <a key={i} href={c.url} target="_blank" rel="noreferrer">
            [{c.source}{c.identifier ? ` ${c.identifier}` : ""}]
          </a>
        ) : (
          <span key={i}>[{c.source}{c.identifier ? ` ${c.identifier}` : ""}]</span>
        ),
      )}
    </div>
  );
}

function AvoidBox({ items }: { items: AvoidWithItem[] }) {
  if (items.length === 0) return null;
  return (
    <div className="avoid-box">
      <strong>Avoid with these medications</strong>
      <ul>
        {items.map((a, i) => {
          const hit = a.citations.length > 0;
          return (
            <li key={i} className={hit ? "avoid-hit" : "avoid-miss"}>
              <strong>{a.substance}</strong>
              {a.medications.length > 0 && ` — ${a.medications.join(", ")}`}
              <p>{a.note}</p>
              <CiteList citations={a.citations} />
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function PairMatrix({ components, pairs }: { components: string[]; pairs: PairResult[] }) {
  if (components.length < 2) return null;
  const map = new Map(
    pairs
      .filter((p) => isDdi(p) || isInsufficient(p))
      .map((p) => [pairKey(p.drugs[0], p.drugs[1]), p]),
  );
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
                      <GradePill grade={p.grade} category={p.category} sourceTier={p.source_tier} />
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
  const [snapshots, setSnapshots] = useState<CheckSnapshot[]>(() => loadHistory());
  const [result, setResult] = useState<CheckResponse | null>(
    () => loadHistory()[0]?.result ?? null,
  );
  const [activeSavedAt, setActiveSavedAt] = useState<string | null>(
    () => loadHistory()[0]?.saved_at ?? null,
  );
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>("all");
  const [altLoading, setAltLoading] = useState(false);
  const [alt, setAlt] = useState<AlternativesResponse | null>(null);
  const [altError, setAltError] = useState<string | null>(null);

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
    setAlt(null);
    setAltError(null);
    try {
      const res = await checkInteractions({
        text: text || undefined,
        images: images.length ? images : undefined,
        patient_context: patient || undefined,
        timing: timing || undefined,
      });
      setResult(res);
      const next = saveCheck(res);
      setSnapshots(next);
      setActiveSavedAt(next[0]?.saved_at ?? null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const runAlternatives = async () => {
    if (!result) return;
    setAltLoading(true);
    setAltError(null);
    setAlt(null);
    try {
      const res = await suggestAlternatives({
        normalized_drugs: result.normalized_drugs,
        pairs: result.pairs,
        patient_context: patient || undefined,
        scrubbed_text: result.scrubbed_text,
        avoid_with_medications: result.avoid_with_medications,
      });
      setAlt(res);
    } catch (e) {
      setAltError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setAltLoading(false);
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

  const insuffPairs = useMemo(() => {
    if (!result) return [];
    const fromPairs = result.pairs.filter(isInsufficient);
    if (fromPairs.length) return fromPairs;
    return result.insufficient_evidence.map(([a, b]) => ({
      drugs: [a, b] as [string, string],
      grade: null,
      category: "interaction" as const,
      summary: "Insufficient evidence to determine.",
      citations: [],
      severe_if: [],
      source_tier: "insufficient",
    }));
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

  const openSnapshot = (s: CheckSnapshot) => {
    setResult(s.result);
    setActiveSavedAt(s.saved_at);
    setFilter("all");
    setAlt(null);
    setAltError(null);
    setError(null);
  };

  const onClearHistory = () => {
    clearHistory();
    setSnapshots([]);
    setActiveSavedAt(null);
    setResult(null);
    setAlt(null);
    setAltError(null);
  };

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
            <li><span className="n">5</span><div><strong>Grade</strong> <span>A labeling · B trials · C case reports · insufficient · or no DDI — with citations</span></div></li>
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
              <span className="pill insuff">INSUFF</span>
              <p><b>Insufficient evidence.</b> Retrieved records do not answer the question, or a claim had no mapped citation. Never a silent A/B/C.</p>
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

      {snapshots.length > 0 && (
        <section className="history-bar panel panel-pad" aria-label="Saved checks">
          <div className="history-head">
            <div>
              <p className="kicker">This browser</p>
              <h3>Recent checks</h3>
              <p className="history-note">
                Scrubbed results only — drugs, grades, citations.
                Patient notes and timing are never stored.
              </p>
            </div>
            <button type="button" className="history-clear" onClick={onClearHistory}>
              Clear history
            </button>
          </div>
          <ul className="history-list">
            {snapshots.map((s) => (
              <li key={s.saved_at}>
                <button
                  type="button"
                  className={s.saved_at === activeSavedAt ? "on" : ""}
                  onClick={() => openSnapshot(s)}
                  title={s.saved_at}
                >
                  {snapshotLabel(s)}
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

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
          {result.contraindicated_banner.length > 0 && (
            <div className="ci-banner" role="alert">
              <div className="ci-banner-stripe" aria-hidden="true" />
              <div className="ci-banner-body">
                <svg className="ci-banner-icon" viewBox="0 0 24 24" aria-hidden="true">
                  <path d="M12 2.2 22.8 21.5H1.2L12 2.2z" />
                  <rect x="11" y="9" width="2" height="6.2" rx="0.4" />
                  <rect x="11" y="16.6" width="2" height="2" rx="0.4" />
                </svg>
                <div>
                  <p className="ci-banner-kicker">Caution — do not co-administer</p>
                  <h2>Contraindicated combination</h2>
                  <p className="ci-banner-lead">
                    {result.contraindicated_banner.length === 1
                      ? "This pair is labeled contraindicated. Do not start or continue both medicines together unless a specialist has already accepted the risk."
                      : "These pairs are labeled contraindicated. Do not start or continue both medicines together unless a specialist has already accepted the risk."}
                  </p>
                  <ul>
                    {result.contraindicated_banner.map((p, i) => (
                      <li key={i}>
                        <strong>{p.drugs[0]} + {p.drugs[1]}</strong>
                        {p.summary ? ` — ${p.summary}` : ""}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}
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
                  {(d.dose || d.schedule) && (
                    <span className="dose-sched">
                      {[d.dose, d.schedule].filter(Boolean).join(" · ")}
                    </span>
                  )}
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

          {result.unresolved_drugs.length > 0 && (
            <div className="warn-box">
              Could not resolve: {result.unresolved_drugs.join(", ")}. Check spelling or use the generic name.
            </div>
          )}

          <AvoidBox items={result.avoid_with_medications} />

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
          {visible.length === 0 && insuffPairs.length === 0 && (
            <p className="empty-pairs">No drug–drug interaction found among the checked pairs.</p>
          )}

          {insuffPairs.length > 0 && (
            <section className="insuff-section">
              <h3 className="ddi-heading">Insufficient evidence ({insuffPairs.length})</h3>
              <p className="insuff-intro">
                These pairs are not graded. Either the retrieved records did not
                answer the question, or a synthesizer citation did not map to a
                record the tools just retrieved.
              </p>
              {insuffPairs.map((p, i) => <PairCard key={`insuff-${i}`} p={p} />)}
            </section>
          )}

          {ddiPairs.length > 0 && (
            <section className="alt-panel panel panel-pad" aria-live="polite">
              <p className="kicker">Whole-prescription substitution</p>
              <h3>Which medicine can change</h3>
              <p className="alt-intro">
                A second pass over every flagged pair. It prefers changing a
                symptomatic or lower-stakes drug so the disease-modifying or
                high-ADR-risk medicine stays. Grade C pairs (weak evidence) are
                not a reason to change a medicine. Proposed substitutes are
                rechecked against the rest of this list.
              </p>
              <button
                type="button"
                className="alt-run"
                onClick={runAlternatives}
                disabled={altLoading}
                aria-busy={altLoading}
              >
                {altLoading ? "Ranking and rechecking…" : "Suggest safer alternatives"}
              </button>
              {altLoading && (
                <p className="hint">
                  Ranking replaceable medicines, proposing same-indication
                  alternatives, then grading each candidate against the leftover list.
                  This is another evidence loop — often 20–40 seconds.
                </p>
              )}
              {altError && <div className="error" role="alert">{altError}</div>}
              {alt && (
                <div className="alt-out">
                  <p className="alt-strategy">{alt.strategy}</p>
                  {alt.timing_first.length > 0 && (
                    <div className="alt-timing">
                      <strong>Separate administration first</strong>
                      <ul>{alt.timing_first.map((t, i) => <li key={i}>{t}</li>)}</ul>
                    </div>
                  )}
                  {alt.replaceable.length > 0 && (
                    <div className="alt-tags">
                      {alt.replaceable.map((d) => (
                        <span className={`imp imp-${d.importance}`} key={d.name}>
                          Change {d.name}
                          <small>{IMP_LABEL[d.importance]}</small>
                        </span>
                      ))}
                    </div>
                  )}
                  {alt.keep.length > 0 && (
                    <div className="alt-keep">
                      <strong>Keep</strong>
                      <ul>
                        {alt.keep.map((d) => (
                          <li key={d.name}>
                            <b>{d.name}</b> ({IMP_LABEL[d.importance]}) — {d.why_this_one}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {alt.suggestions.map((s, i) => <SuggestionCard key={i} s={s} />)}
                  {alt.suggestions.length === 0 && !altLoading && (
                    <p className="empty-pairs">
                      No substitute cleared the rest of this list. Use the ranking
                      above and a clinical pharmacist if a change is still needed.
                    </p>
                  )}
                  <p className="disclaimer alt-disclaimer">{alt.disclaimer}</p>
                </div>
              )}
            </section>
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
