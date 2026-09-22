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

// Mirrors ``backend/tuning.py`` BPM_LO / BPM_HI; keep in sync.
export const BPM_LO = 60;
export const BPM_HI = 250;

// Mirrors ``backend/embeddings/feature_encoding.py``; keep in sync.
export const CAMELOT_MAJOR_RADIUS = 0.82;
export const CAMELOT_MINOR_RADIUS = 0.45;
const MINOR_FIFTHS_OFFSET = 3;
export const SPIRAL_R_MIN = 0.18;
export const SPIRAL_R_MAX = 0.86;
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

export function getLens(id: string): Lens {
  return BUILTIN_LENSES.find((l) => l.id === id) ?? BUILTIN_LENSES[0];
}

/** Lenses a scope can use: the tag lens has no positions to lay out. */
export function scopeLenses(): Lens[] {
  return BUILTIN_LENSES.filter((l) => l.kind !== "tags");
}

/** Next built-in lens id for the **E** cycle; a preset cycles from its base lens. */
export function nextLensId(id: string, baseLens?: string): string {
  return stepLensId(id, 1, baseLens);
}

/** Previous built-in lens id for the **Shift+E** cycle. */
export function prevLensId(id: string, baseLens?: string): string {
  return stepLensId(id, -1, baseLens);
}

function stepLensId(id: string, dir: 1 | -1, baseLens?: string): string {
  const from = baseLens ?? id;
  const idx = BUILTIN_LENSES.findIndex((l) => l.id === from);
  const n = BUILTIN_LENSES.length;
  return BUILTIN_LENSES[(idx + dir + n) % n].id;
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
