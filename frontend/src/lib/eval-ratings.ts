// ─── Eval ratings persistence ────────────────────────────────
//
// Blind-test ratings for the layout-evaluation gallery, stored in localStorage
// keyed by dataset fingerprint + artifact revision so a rating is tied to the
// exact map generation it describes (re-runs, re-reveals and other datasets stay
// isolated). Pure and dependency-free so it is unit-testable without a DOM.

export const RATING_CATEGORIES = [
  "neighbours",
  "organisation",
  "crowding",
  "overall",
] as const;
export type RatingCategory = (typeof RATING_CATEGORIES)[number];

/** The four 1–5 scores plus a free-text note. */
export interface EvalRatings {
  neighbours: number | null;
  organisation: number | null;
  crowding: number | null;
  overall: number | null;
  note: string;
}

/** A stored record: the ratings plus the map identity they describe. */
export interface EvalRatingRecord extends EvalRatings {
  recipeId: string;
  revision: string;
  /** Algorithm/recipe name; blank until revealed, filled in on reveal. */
  label: string;
  updatedAt: number;
}

const LS_KEY = "trackspace:layout-eval-ratings";

/** ``null`` for every score; note starts empty. */
export function emptyRatings(): EvalRatings {
  return { neighbours: null, organisation: null, crowding: null, overall: null, note: "" };
}

function isCategory(v: unknown): v is RatingCategory {
  return RATING_CATEGORIES.includes(v as RatingCategory);
}

/** How many of the four categories have a 1–5 score. */
export function ratedCount(r: EvalRatings | null | undefined): number {
  if (!r) return 0;
  let n = 0;
  for (const c of RATING_CATEGORIES) if (r[c] != null) n++;
  return n;
}

/** Load every stored record, grouped fingerprint → revision. */
export function loadAllRatings(): Record<string, Record<string, EvalRatingRecord>> {
  try {
    const raw = JSON.parse(localStorage.getItem(LS_KEY) ?? "null");
    if (!raw || typeof raw !== "object") return {};
    const out: Record<string, Record<string, EvalRatingRecord>> = {};
    for (const [fp, revisions] of Object.entries(raw as Record<string, unknown>)) {
      if (!revisions || typeof revisions !== "object") continue;
      for (const [rev, rec] of Object.entries(revisions as Record<string, unknown>)) {
        const r = normalizeRecord(rec, rev);
        if (!r) continue;
        (out[fp] ??= {})[rev] = r;
      }
    }
    return out;
  } catch {
    return {};
  }
}

/** The record for (fingerprint, revision), or ``null`` when none is stored. */
export function loadRating(fp: string, revision: string): EvalRatingRecord | null {
  return loadAllRatings()[fp]?.[revision] ?? null;
}

/** Validate + coerce a raw stored record, or ``null`` if it is unusable. */
function normalizeRecord(raw: unknown, fallbackRevision: string): EvalRatingRecord | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  const rec: EvalRatingRecord = {
    ...emptyRatings(),
    recipeId: typeof o.recipeId === "string" ? o.recipeId : "",
    revision: typeof o.revision === "string" ? o.revision : fallbackRevision,
    label: typeof o.label === "string" ? o.label : "",
    updatedAt: typeof o.updatedAt === "number" ? o.updatedAt : Date.now(),
  };
  for (const c of RATING_CATEGORIES) {
    const v = o[c];
    if (typeof v === "number" && v >= 1 && v <= 5) rec[c] = v;
  }
  if (typeof o.note === "string") rec.note = o.note;
  return rec;
}

/**
 * Persist one record for (fingerprint, revision), merging into existing storage.
 * Returns the stored record.
 */
export function saveRating(
  fp: string,
  revision: string,
  patch: Partial<EvalRatingRecord>,
): EvalRatingRecord {
  const all = loadAllRatings();
  const prev = all[fp]?.[revision] ?? {
    ...emptyRatings(),
    recipeId: "",
    revision,
    label: "",
    updatedAt: 0,
  };
  const next: EvalRatingRecord = { ...prev, ...patch, revision, updatedAt: Date.now() };
  all[fp] = { ...(all[fp] ?? {}), [revision]: next };
  try {
    localStorage.setItem(LS_KEY, JSON.stringify(all));
  } catch {
    /* storage full/unavailable — ratings just won't persist this session */
  }
  return next;
}

/** A data-dataset summary for a JSON export: one row per stored revision. */
export interface ExportRow {
  recipeId: string;
  revision: string;
  label: string;
  neighbours: number | null;
  organisation: number | null;
  crowding: number | null;
  overall: number | null;
  note: string;
  updatedAt: number;
}

/** Build the exportable JSON string for a dataset's stored records. */
export function buildEvalExport(
  datasetFingerprint: string,
  records: ReadonlyArray<EvalRatingRecord>,
): string {
  const rows: ExportRow[] = records
    .map((r) => ({
      recipeId: r.recipeId,
      revision: r.revision,
      label: r.label,
      neighbours: r.neighbours,
      organisation: r.organisation,
      crowding: r.crowding,
      overall: r.overall,
      note: r.note,
      updatedAt: r.updatedAt,
    }))
    .sort((a, b) => a.updatedAt - b.updatedAt);
  return JSON.stringify(
    {
      schema_version: 1,
      dataset_fingerprint: datasetFingerprint,
      exported_at: new Date().toISOString(),
      maps: rows,
    },
    null,
    2,
  );
}
