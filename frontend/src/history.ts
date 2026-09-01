/** Browser check history: scrubbed CheckResponse snapshots only. */
import type { CheckResponse } from "./api";

export const HISTORY_KEY = "ddi-explorer.history.v1";
export const MAX_SNAPSHOTS = 5;

export interface CheckSnapshot {
  saved_at: string;
  result: CheckResponse;
}

function storage(): Storage | null {
  try {
    if (typeof window === "undefined") return null;
    return window.localStorage;
  } catch {
    return null;
  }
}

/** Allowlisted check payload. Never copies form fields (text/patient/timing/images). */
export function snapshotResult(result: CheckResponse): CheckResponse {
  return {
    scrubbed_text: result.scrubbed_text,
    normalized_drugs: result.normalized_drugs,
    unresolved_drugs: result.unresolved_drugs,
    pairs: result.pairs,
    contraindicated_banner: result.contraindicated_banner,
    avoid_with_medications: result.avoid_with_medications,
    insufficient_evidence: result.insufficient_evidence,
    disclaimer: result.disclaimer,
  };
}

function isSnapshot(raw: unknown): raw is CheckSnapshot {
  if (!raw || typeof raw !== "object") return false;
  const o = raw as CheckSnapshot;
  const r = o.result;
  return (
    typeof o.saved_at === "string" &&
    !!r &&
    Array.isArray(r.normalized_drugs) &&
    Array.isArray(r.pairs) &&
    Array.isArray(r.unresolved_drugs)
  );
}

export function loadHistory(): CheckSnapshot[] {
  const store = storage();
  if (!store) return [];
  try {
    const parsed = JSON.parse(store.getItem(HISTORY_KEY) || "[]") as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isSnapshot).slice(0, MAX_SNAPSHOTS);
  } catch {
    return [];
  }
}

export function saveCheck(result: CheckResponse, now = new Date()): CheckSnapshot[] {
  const entry: CheckSnapshot = {
    saved_at: now.toISOString(),
    result: snapshotResult(result),
  };
  const next = [entry, ...loadHistory()].slice(0, MAX_SNAPSHOTS);
  const store = storage();
  if (store) {
    try {
      store.setItem(HISTORY_KEY, JSON.stringify(next));
    } catch {
      // Quota or private mode — keep the in-memory list for this session.
    }
  }
  return next;
}

export function clearHistory(): void {
  storage()?.removeItem(HISTORY_KEY);
}

export function snapshotLabel(s: CheckSnapshot): string {
  const names = s.result.normalized_drugs.map((d) => d.input_name).filter(Boolean);
  if (names.length === 0) return "Saved check";
  const head = names.slice(0, 3).join(", ");
  return names.length > 3 ? `${head} +${names.length - 3}` : head;
}
