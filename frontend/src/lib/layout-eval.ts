// ─── Layout-evaluation parsing + helpers ─────────────────────
//
// Pure, DOM-free parsing and selection logic for the offline layout-evaluation
// artifacts served by `GET /api/layout-evaluation` and `GET /api/layout-evaluation/<id>`.
// Kept out of the coordinator so it is unit-testable in isolation and so the
// blind-name assignment, universe computation and artifact validation follow one
// rule regardless of UI state.

/** One recipe as listed in the manifest (a completed entry). */
export interface EvalManifestRecipe {
  id: string;
  label: string;
  tracks?: number;
  timings?: Record<string, number>;
  alignedTo?: string | null;
}

/** A parsed, validated manifest of the completed eval recipes. */
export interface EvalManifest {
  schemaVersion: number;
  status: string;
  datasetFingerprint: string;
  recipes: EvalManifestRecipe[];
}

/** The layout-evaluation manifest schema this client reads. */
export const EVAL_SCHEMA_VERSION = 1;

/**
 * Validate + normalise the manifest payload. Enforces the schema version and
 * that at least one recipe is complete. Recipes the manifest marks ``complete``
 * are listed even while the generation is still in progress — the server serves
 * the completed recipes of an in-progress run — so nothing else is required.
 * Returns ``null`` for a manifest that is not a usable, complete generation.
 */
export function parseEvalManifest(raw: unknown): EvalManifest | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  // A different schema is a hard rejection: a gallery from another build must
  // not render as if it were this one.
  if (o.schema_version !== EVAL_SCHEMA_VERSION) return null;
  const recipesRaw = o.recipes;
  if (!recipesRaw || typeof recipesRaw !== "object") return null;
  if (typeof o.dataset_fingerprint !== "string") return null;
  const recipes: EvalManifestRecipe[] = [];
  for (const [id, entry] of Object.entries(recipesRaw as Record<string, unknown>)) {
    if (!entry || typeof entry !== "object") continue;
    const r = entry as Record<string, unknown>;
    if (r.status !== "complete") continue;
    const label = r.label;
    if (typeof label !== "string") continue;
    const rec: EvalManifestRecipe = { id, label };
    if (typeof r.tracks === "number") rec.tracks = r.tracks;
    if (r.timings && typeof r.timings === "object") {
      const timings: Record<string, number> = {};
      for (const [k, v] of Object.entries(r.timings as Record<string, unknown>)) {
        if (typeof v === "number") timings[k] = v;
      }
      rec.timings = timings;
    }
    if (typeof r.aligned_to === "string" || r.aligned_to === null) {
      rec.alignedTo = r.aligned_to as string | null;
    }
    recipes.push(rec);
  }
  if (recipes.length === 0) return null;
  return {
    schemaVersion: EVAL_SCHEMA_VERSION,
    status: typeof o.status === "string" ? o.status : "unknown",
    datasetFingerprint: o.dataset_fingerprint,
    recipes,
  };
}

/**
 * True when the payload is a manifest this client cannot read: it exists,
 * declares a ``schema_version``, and that version differs from the one this
 * build understands. Payloads with no schema at all are not manifests, so they
 * fall through to the generic "no usable gallery" message. Lets callers tell
 * "no usable gallery" apart from "gallery too new for this build".
 */
export function isUnsupportedEvalSchema(raw: unknown): boolean {
  if (!raw || typeof raw !== "object") return false;
  const schema = (raw as Record<string, unknown>).schema_version;
  return schema !== undefined && schema !== EVAL_SCHEMA_VERSION;
}

/** A parsed, validated recipe artifact with its positions map. */
export interface EvalArtifact {
  id: string;
  revision: string;
  datasetFingerprint: string;
  positions: Map<string, { x: number; y: number }>;
  tracks: number;
  label: string;
  alignedTo: string | null;
  timings: Record<string, number>;
}

/**
 * Validate + normalise one recipe artifact. Mirrors the backend's own checks:
 * every position must be finite and carry a string path. Returns ``null`` when
 * the artifact is missing/corrupt or its positions are unusable.
 */
export function parseEvalArtifact(raw: unknown): EvalArtifact | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as Record<string, unknown>;
  const positions = o.positions;
  if (!Array.isArray(positions)) return null;
  if (typeof o.id !== "string" || typeof o.revision !== "string") return null;
  const map = new Map<string, { x: number; y: number }>();
  for (const pos of positions) {
    if (!pos || typeof pos !== "object") return null;
    const p = pos as Record<string, unknown>;
    if (typeof p.path !== "string") return null;
    const x = p.x;
    const y = p.y;
    if (typeof x !== "number" || typeof y !== "number") return null;
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    map.set(p.path, { x, y });
  }
  if (map.size === 0) return null;
  const timings: Record<string, number> = {};
  if (o.timings && typeof o.timings === "object") {
    for (const [k, v] of Object.entries(o.timings as Record<string, unknown>)) {
      if (typeof v === "number") timings[k] = v;
    }
  }
  return {
    id: o.id,
    revision: o.revision,
    datasetFingerprint: typeof o.dataset_fingerprint === "string" ? o.dataset_fingerprint : "",
    positions: map,
    tracks: typeof o.tracks === "number" ? o.tracks : map.size,
    label: typeof o.label === "string" ? o.label : o.id,
    alignedTo: typeof o.aligned_to === "string" ? (o.aligned_to as string) : null,
    timings,
  };
}

/** The blind display letter for a 0-based map index (A, B, … Z). */
export function blindLetter(index: number): string {
  return String.fromCharCode(65 + (index % 26));
}

/**
 * The URL to navigate to when leaving eval mode: strips the eval-only params
 * (``layout_eval``, ``layout``, and the named-set ``set``) while preserving every
 * unrelated query param.
 */
export function evalExitUrl(href: string): string {
  const url = new URL(href);
  url.searchParams.delete("layout_eval");
  url.searchParams.delete("layout");
  url.searchParams.delete("set");
  return url.toString();
}

/**
 * The eval-view universe: the artifact's own paths that exist in the loaded
 * library. Returns the eligible set and how many artifact paths are missing from
 * the loaded library ("unavailable files"), so the toolbar can report them.
 */
export function computeEvalUniverse(
  artifactPaths: ReadonlySet<string>,
  loadedPaths: ReadonlySet<string>,
): { eligible: Set<string>; unavailable: number } {
  const eligible = new Set<string>();
  let unavailable = 0;
  for (const p of artifactPaths) {
    if (loadedPaths.has(p)) eligible.add(p);
    else unavailable++;
  }
  return { eligible, unavailable };
}
