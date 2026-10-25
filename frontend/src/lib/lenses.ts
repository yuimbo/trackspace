// ─── Lenses ──────────────────────────────────────────────────
//
// A lens is *what the canvas is showing*. The old model had a single binary
// toggle (`viewMode`: tag axes vs the combined embedding map); a lens generalises
// that to:
//
// - **tags** — the user-tag / audio-feature axes (the original tag view).
// - **projection** — a t-SNE layout over a set of source blocks. The combined
//   lens projects every frozen source; a per-source lens projects one block (or the
//   MAEST pair) so each embedding space can be inspected on its own.
// - **deterministic** — a pure function of the cached audio features, computed
//   client-side with no server round-trip (Camelot wheel, tempo spiral).
//
// Presets (a base lens plus tweaked display options) live on the model, not here;
// this module is the built-in registry and the deterministic geometry.

import type { Track } from "../model";
import { clamp } from "./util";
import featureGeometry from "./feature-geometry.json";

// Feature geometry constants live in ``feature-geometry.json`` (shared with the
// server) so the deterministic lenses cannot silently drift from the backend.
export const BPM_LO = featureGeometry.bpm_lo;
export const BPM_HI = featureGeometry.bpm_hi;

export const CAMELOT_MAJOR_RADIUS = featureGeometry.camelot_major_radius;
export const CAMELOT_MINOR_RADIUS = featureGeometry.camelot_minor_radius;
const MINOR_FIFTHS_OFFSET = featureGeometry.minor_fifths_offset;
export const SPIRAL_R_MIN = featureGeometry.spiral_r_min;
export const SPIRAL_R_MAX = featureGeometry.spiral_r_max;
const BPM_OCTAVES = Math.log2(BPM_HI / BPM_LO);

/** A lens position in polar form: angle (radians) and radius from the centre. */
export interface Polar {
  angle: number;
  radius: number;
}

export type LensKind = "tags" | "projection" | "deterministic";

/** How dots are coloured; mirrors the folder/cluster palettes and the two feature
 *  lenses (`camelot`, `tempo`), whose polar geometry drives the hue/brightness.
 *  `head` colours by the folder head's nearest learned folder, not the track's own. */
export type ColorMode = "folder" | "head" | "cluster" | "tempo" | "key";

/** Colour modes in **C** cycle order. */
export const COLOR_MODES: readonly ColorMode[] = [
  "folder",
  "head",
  "cluster",
  "tempo",
  "key",
];

/** Tab labels for the colour-mode tabber, keyed by mode. */
export const COLOR_MODE_LABELS: Record<ColorMode, string> = {
  folder: "Folder colour",
  head: "Predicted folder",
  cluster: "Cluster colour",
  tempo: "Tempo colour",
  key: "Key colour",
};

/** Colour-mode tabs (id + label) for the top tabber and the View-tab select. */
export function colorModeTabs(): { id: string; label: string }[] {
  return COLOR_MODES.map((id) => ({ id, label: COLOR_MODE_LABELS[id] }));
}

/** Step a colour mode through the **C** / **D** order; wraps at either end. */
export function stepColorMode(mode: ColorMode, dir: 1 | -1 = 1): ColorMode {
  const idx = COLOR_MODES.indexOf(mode);
  const n = COLOR_MODES.length;
  return COLOR_MODES[(idx + dir + n) % n];
}

/** A saved view: a base lens plus the display options the user tweaked. */
export interface LensPreset {
  id: string;
  name: string;
  baseLens: string;
  weights: Record<string, number>;
  colorMode: ColorMode;
  clusterResolution: number;
}

export interface Lens {
  id: string;
  label: string;
  kind: LensKind;
  /** Projection lenses: source blocks. Undefined = every frozen source. */
  sources?: readonly string[];
  /** Deterministic lenses: [0,1]² position, or null when the track lacks features. */
  compute?: (t: Track) => { x: number; y: number } | null;
  /** Canvas guide drawn under the dots. */
  guide?: "camelot" | "spiral";
}

/** Deterministic hash → [0, 1), so same-key tracks spread stably across runs. */
function _hash01(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) % 100000) / 100000;
}

/**
 * Small stable offset so tracks sharing a key/tempo do not stack on one pixel.
 * Deterministic (path hash), so the layout does not shuffle between frames.
 */
function _jitter(t: Track, radius: number): { x: number; y: number } {
  const a = 2 * Math.PI * _hash01(t.path + "|a");
  const r = radius * Math.sqrt(_hash01(t.path + "|r"));
  return { x: r * Math.cos(a), y: r * Math.sin(a) };
}

/** Camelot wheel: major outer ring, minor inner ring rotated 3 fifths (90°). */
export function camelotPolar(keyTurns: number, mode: number): Polar {
  const angle = 2 * Math.PI * (keyTurns - 0.5);
  const minor = mode < 0.5;
  const a = minor ? angle - MINOR_FIFTHS_OFFSET * ((2 * Math.PI) / 12) : angle;
  const r = minor ? CAMELOT_MINOR_RADIUS : CAMELOT_MAJOR_RADIUS;
  return { angle: a, radius: r };
}

export function camelotXY(keyTurns: number, mode: number): { x: number; y: number } {
  const { angle, radius } = camelotPolar(keyTurns, mode);
  return { x: 0.5 + 0.5 * radius * Math.cos(angle), y: 0.5 + 0.5 * radius * Math.sin(angle) };
}

/** Tempo spiral: one turn per octave, radius grows with log-BPM. */
export function bpmSpiralPolar(tempoNorm: number): Polar {
  const bpm = Math.max(tempoNorm * (BPM_HI - BPM_LO) + BPM_LO, 1e-6);
  const octave = Math.log2(bpm / BPM_LO);
  const angle = 2 * Math.PI * (octave % 1);
  const tt = clamp(octave / BPM_OCTAVES);
  const radius = SPIRAL_R_MIN + tt * (SPIRAL_R_MAX - SPIRAL_R_MIN);
  return { angle, radius };
}

export function bpmSpiralXY(tempoNorm: number): { x: number; y: number } {
  const { angle, radius } = bpmSpiralPolar(tempoNorm);
  return { x: 0.5 + 0.5 * radius * Math.cos(angle), y: 0.5 + 0.5 * radius * Math.sin(angle) };
}

/** [0,1] tempo for a track: cached features first, else its raw BPM chip. */
export function trackTempoNorm(t: Track): number | null {
  const af = t.audio_features;
  const tnorm = af
    ? af.tempo
    : t.bpm != null
      ? (t.bpm - BPM_LO) / (BPM_HI - BPM_LO)
      : null;
  return tnorm != null && Number.isFinite(tnorm) ? tnorm : null;
}

/** Camelot position for a track, or null when it has no cached features. */
export function camelotPosition(t: Track): { x: number; y: number } | null {
  const af = t.audio_features;
  if (!af) return null;
  const p = camelotXY(af.key, af.mode);
  const j = _jitter(t, 0.05);
  return { x: p.x + j.x, y: p.y + j.y };
}

/** Tempo-spiral position for a track, or null when it has no tempo. */
export function tempoSpiralPosition(t: Track): { x: number; y: number } | null {
  const tnorm = trackTempoNorm(t);
  if (tnorm == null) return null;
  const p = bpmSpiralXY(tnorm);
  const j = _jitter(t, 0.04);
  return { x: p.x + j.x, y: p.y + j.y };
}

/** Camelot major key names by circle-of-fifths position (mirrors the backend). */
export const CAMELOT_KEY_NAMES = [
  "C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F",
] as const;

/** Outer-ring key labels for the Camelot guide, in world coordinates. */
export function camelotGuide(): { label: string; x: number; y: number }[] {
  return CAMELOT_KEY_NAMES.map((label, p) => {
    const a = (2 * Math.PI * p) / 12;
    return {
      label,
      x: 0.5 + 0.5 * CAMELOT_MAJOR_RADIUS * Math.cos(a),
      y: 0.5 + 0.5 * CAMELOT_MAJOR_RADIUS * Math.sin(a),
    };
  });
}

/** Sampled spiral arm points for the tempo guide, in world coordinates. */
export function spiralGuide(steps = 180): { x: number; y: number }[] {
  const out: { x: number; y: number }[] = [];
  for (let i = 0; i <= steps; i++) out.push(bpmSpiralXY(i / steps));
  return out;
}

/** Built-ins, in **E**-cycle order: tags → map lenses → feature lenses → tags. */
export const BUILTIN_LENSES: readonly Lens[] = [
  { id: "tags", label: "Tags", kind: "tags" },
  { id: "combined", label: "Combined map", kind: "projection" },
  { id: "maest", label: "MAEST genre", kind: "projection", sources: ["maest_logits", "maest"] },
  { id: "supervised", label: "Folder head", kind: "projection", sources: ["supervised"] },
  { id: "clap", label: "CLAP timbre", kind: "projection", sources: ["clap"] },
  { id: "rhythm", label: "Rhythm", kind: "projection", sources: ["rhythm"] },
  { id: "camelot", label: "Camelot wheel", kind: "deterministic", compute: camelotPosition, guide: "camelot" },
  { id: "tempo", label: "Tempo spiral", kind: "deterministic", compute: tempoSpiralPosition, guide: "spiral" },
];

/**
 * The opt-in ``genre_full`` lens (logits .40 / maest .20 / layer-7 .25 / MERT .15).
 * Its ``sources`` are the four *genre analysis* blocks the map waits on / requires
 * for placement — the tunable non-genre contributors (rhythm/clap/supervised) are a
 * separate list (``GENRE_FULL_SOURCES``) used for the weight preview and the recipe
 * ``w=`` override, so the analysis gate never waits on a zero-weight contributor.
 */
export const GENRE_FULL_LENS: Lens = {
  id: "genre_full",
  label: "Genre",
  kind: "projection",
  sources: ["maest_logits", "maest", "maest_layer7", "mert_l19"],
};

/**
 * The genre_full recipe's *tunable* sources, in canonical weight order (mirrors
 * ``backend/embeddings/projection_config.py`` ``GENRE_FULL_SOURCES``). These are
 * the sliders and the ``w=`` override keys; ``features`` stays excluded — it is
 * the audio-feature block, never a genre recipe override key.
 */
export const GENRE_FULL_SOURCES: readonly string[] = [
  "maest_logits",
  "maest",
  "maest_layer7",
  "mert_l19",
  "rhythm",
  "clap",
  "supervised",
];

/**
 * The canonical genre_full default weights (mirrors
 * ``backend/embeddings/projection_config.py`` ``GENRE_FULL_RECIPE_WEIGHTS``): the
 * four genre blocks, the tunable rhythm/clap/supervised at 0, ``features`` absent.
 * The query omits ``w=`` when the applied vector equals these defaults, so the
 * default recipe key stays stable.
 */
export const GENRE_FULL_WEIGHTS: Record<string, number> = {
  maest_logits: 0.4,
  maest: 0.2,
  maest_layer7: 0.25,
  mert_l19: 0.15,
  rhythm: 0,
  clap: 0,
  supervised: 0,
};

/** Lenses made redundant by the genre map: hidden from the E-cycle / tabber. */
const GENRE_HIDDEN_IDS = new Set(["maest", "supervised"]);

/** Map a legacy lens id to its genre replacement, or pass it through.
 *
 *  ``combined`` (the old combined map) and the single-source MAEST / Folder-head
 *  probes (``maest``/``supervised``) all resolve to the genre map when it is
 *  enabled. Used by preset migration and the E-cycle so a saved legacy id never
 *  wraps to an invalid (or surprising) target.
 */
export function legacyToGenreLensId(id: string): string {
  if (id === "combined" || id === "maest" || id === "supervised") {
    return GENRE_FULL_LENS.id;
  }
  return id;
}

/** The effective lens set for the current server.
 *
 *  The single-source MAEST / Folder-head probes are gone from the cycle. With
 *  genre enabled the ``combined`` map slot is taken by ``genre_full`` (the genre
 *  map is the default); with genre disabled ``combined`` remains as the fallback
 *  default map. ``tags`` and the feature lenses are always present, so the map
 *  slot always sits right after ``tags`` in the **E** cycle.
 */
export function availableLenses(genreFullEnabled: boolean): readonly Lens[] {
  // BUILTIN minus the redundant single-source probes, keeping ``combined`` in
  // the map slot so the E-cycle order (tags → map → features) stays stable.
  const base = BUILTIN_LENSES.filter((l) => !GENRE_HIDDEN_IDS.has(l.id));
  if (!genreFullEnabled) return base;
  // Replace the combined-map slot with the genre map.
  return base.map((l) => (l.id === "combined" ? GENRE_FULL_LENS : l));
}

/** The default map lens: the genre recipe when enabled, else legacy ``combined``. */
export function defaultMapLensId(genreFullEnabled: boolean): string {
  return genreFullEnabled ? GENRE_FULL_LENS.id : "combined";
}

export function getLens(id: string): Lens {
  return (
    BUILTIN_LENSES.find((l) => l.id === id) ??
    (id === GENRE_FULL_LENS.id ? GENRE_FULL_LENS : undefined) ??
    BUILTIN_LENSES[0]
  );
}

/** Lenses a scope can use: the tag lens has no positions to lay out, and the
 *  genre recipe (or the legacy ``combined`` fallback) is a valid scope projection
 *  kind. ``genreFullEnabled`` selects which map lens is on the list. */
export function scopeLenses(genreFullEnabled = false): Lens[] {
  return availableLenses(genreFullEnabled).filter((l) => l.kind !== "tags");
}

/** Next built-in lens id for the **E** cycle; a preset cycles from its base lens. */
export function nextLensId(
  id: string,
  baseLens?: string,
  available: readonly Lens[] = BUILTIN_LENSES,
): string {
  return stepLensId(id, 1, available, baseLens);
}

/** Previous built-in lens id for the **Shift+E** cycle. */
export function prevLensId(
  id: string,
  baseLens?: string,
  available: readonly Lens[] = BUILTIN_LENSES,
): string {
  return stepLensId(id, -1, available, baseLens);
}

function stepLensId(
  id: string,
  dir: 1 | -1,
  available: readonly Lens[],
  baseLens?: string,
): string {
  let from = baseLens ?? id;
  // A legacy map id that the genre map replaced resolves to the genre map when it
  // is present, so E from a saved legacy lens (a preset base) lands on a valid map
  // target rather than wrapping to tags.
  if (available.some((l) => l.id === GENRE_FULL_LENS.id)) {
    from = legacyToGenreLensId(from);
  }
  const idx = available.findIndex((l) => l.id === from);
  const n = available.length;
  if (idx === -1) return available[0].id;
  return available[(idx + dir + n) % n].id;
}

/** Lens source names for the "wait for analysis" gate; tags waits on nothing. */
export function lensWaitSources(lens: Lens): readonly string[] {
  if (lens.kind === "tags") return [];
  if (lens.kind === "deterministic") return ["features"];
  return lens.sources ?? ["maest_logits", "maest", "rhythm"];
}

/** Per-source weights for the projection query: only the lens's blocks are on. */
export function lensWeights(
  lens: Lens,
  clusterWeights: Record<string, number>,
): Record<string, number> {
  if (!lens.sources) return clusterWeights;
  const out: Record<string, number> = {};
  for (const k of Object.keys(clusterWeights)) out[k] = 0;
  for (const s of lens.sources) out[s] = 1;
  return out;
}
