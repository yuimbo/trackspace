// ─── Selection bloom ──────────────────────────────────────────
//
// The selection cursor (moved with **D**) picks the variable the hover bloom
// matches on; the bloom **range** (**D + scroll**) is that match's tolerance. The
// range means something different per variable, so the HUD always names it:
//
// - **folder / predicted folder** — how many parent directories are climbed
//   before comparing, so a wider range includes sibling subtrees under a shared
//   ancestor (and the predicted folder is affected the same way).
// - **tempo** — a ± percentage of the hovered BPM.
// - **key** — ± Camelot-wheel steps.
// - **cluster** — no range: an exact cluster-id match.
//
// One module owns the mapping, so the matcher (`Model.trackBloomsWith`) and the
// HUD (`CanvasView.showBloomHud`) cannot disagree about what the range means.

import type { Track } from "../model";
import type { ColorMode } from "./lenses";

/**
 * Default bloom range: exact. The bloom starts as the hovered dot's own class and
 * **D + scroll** widens it; a range of 0 must be reachable, so the adjustment is
 * additive rather than the volume-style multiplicative scale.
 */
export const DEFAULT_BLOOM_RANGE = 0;

/**
 * Per-variable bloom range slots. Each variable keeps its own range so widening
 * the tempo bloom does not also widen the folder bloom; folder and predicted
 * folder share one slot, since they are the same directory climb.
 */
export type BloomRangeKey = "folder" | "cluster" | "tempo" | "key";

export const BLOOM_RANGE_KEYS: readonly BloomRangeKey[] = [
  "folder",
  "cluster",
  "tempo",
  "key",
];

/** The range slot a colour mode reads and writes. */
export function bloomRangeKey(mode: ColorMode): BloomRangeKey {
  return mode === "head" ? "folder" : mode;
}

/** Range [0,1] → ± this percentage of the hovered tempo, at full range. */
export const BLOOM_TEMPO_MAX_PCT = 25;
/** Range [0,1] → ± this many Camelot-wheel steps, at full range. */
export const BLOOM_KEY_MAX_STEPS = 6;
/** Range [0,1] → this many parent directory levels, at full range. */
export const BLOOM_FOLDER_MAX_LEVELS = 4;

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

/** Parent directory levels the folder/head bloom climbs at *range*. */
export function bloomFolderLevels(range: number): number {
  return Math.round(clamp01(range) * BLOOM_FOLDER_MAX_LEVELS);
}

/** ± BPM tolerance the tempo bloom uses at *range*, as a percentage. */
export function bloomTempoPct(range: number): number {
  return Math.round(clamp01(range) * BLOOM_TEMPO_MAX_PCT);
}

/** ± Camelot-wheel steps the key bloom uses at *range*. */
export function bloomKeySteps(range: number): number {
  return Math.round(clamp01(range) * BLOOM_KEY_MAX_STEPS);
}

/**
 * Semantic readout of the bloom range under *mode*, e.g. `+2 parent directories`
 * or `±13% BPM`; ``null`` for a variable with no range (cluster).
 */
export function bloomRangeLabel(mode: ColorMode, range: number): string | null {
  switch (mode) {
    case "folder":
    case "head": {
      const n = bloomFolderLevels(range);
      if (n === 0) return "exact folder";
      return `+${n} parent director${n === 1 ? "y" : "ies"}`;
    }
    case "tempo":
      return `±${bloomTempoPct(range)}% BPM`;
    case "key": {
      const n = bloomKeySteps(range);
      return n === 0 ? "exact key" : `±${n} step${n === 1 ? "" : "s"}`;
    }
    default:
      return null;
  }
}

/** Ancestor prefix *levels* above *folder* (0 = the folder itself). */
export function folderAncestor(folder: string, levels: number): string {
  const segs = (folder ?? "").split("/").filter(Boolean);
  return segs.slice(0, Math.max(0, segs.length - levels)).join("/");
}

/** Camelot-wheel fifth position (0..11) of a track's cached key, or null. */
export function trackCamelotFifths(t: Track): number | null {
  const af = t.audio_features;
  if (!af) return null;
  // Minor keys sit three fifths counter-clockwise of their major tonic
  // (mirrors `camelotPolar`).
  const p = Math.round(af.key * 12) - (af.mode < 0.5 ? 3 : 0);
  return ((p % 12) + 12) % 12;
}

/** Camelot-wheel distance (in fifths) between two tracks' cached keys, or null. */
export function camelotSteps(a: Track, b: Track): number | null {
  const fa = trackCamelotFifths(a);
  const fb = trackCamelotFifths(b);
  if (fa === null || fb === null) return null;
  // Distance is the shorter way round the wheel.
  const d = (((fa - fb) % 12) + 12) % 12;
  return Math.min(d, 12 - d);
}

/**
 * Cluster tracks by the tempo bloom: repeatedly take the lowest remaining integer
 * BPM as the cluster centre and gather every track within `bloomTempoPct(range)`% of
 * it. The selected-tracks overlay uses this so its BPM rows merge exactly the way
 * the hover bloom would; at range 0 a cluster is one integer BPM.
 */
export function clusterTracksByTempo(
  tracks: readonly Track[],
  range: number,
): Track[][] {
  const pct = bloomTempoPct(range);
  const withBpm = tracks
    .filter((t) => t.bpm != null && Number.isFinite(t.bpm))
    .sort((a, b) => (a.bpm as number) - (b.bpm as number));
  const out: Track[][] = [];
  for (let i = 0; i < withBpm.length; ) {
    const centre = Math.round(withBpm[i].bpm as number);
    const tol = (centre * pct) / 100;
    const cluster: Track[] = [];
    let j = i;
    while (
      j < withBpm.length &&
      Math.round(withBpm[j].bpm as number) - centre <= tol + 1e-9
    ) {
      cluster.push(withBpm[j]);
      j++;
    }
    out.push(cluster);
    i = j;
  }
  return out;
}

/**
 * Cluster tracks by the key bloom: repeatedly take the most common remaining
 * Camelot fifth as the cluster centre and gather every track within
 * `bloomKeySteps(range)` steps of it. At range 0 a cluster is one fifth position
 * (enharmonic equivalents included); wider ranges gather neighbouring keys.
 */
export function clusterTracksByKey(
  tracks: readonly Track[],
  range: number,
): Track[][] {
  const steps = bloomKeySteps(range);
  let remaining = tracks.filter((t) => trackCamelotFifths(t) !== null);
  const out: Track[][] = [];
  while (remaining.length) {
    const counts = new Map<number, number>();
    for (const t of remaining) {
      const f = trackCamelotFifths(t) as number;
      counts.set(f, (counts.get(f) ?? 0) + 1);
    }
    let centre: Track | null = null;
    let centreFifths = -1;
    let best = -1;
    for (const t of remaining) {
      const f = trackCamelotFifths(t) as number;
      const c = counts.get(f) as number;
      if (c > best || (c === best && f < centreFifths)) {
        best = c;
        centreFifths = f;
        centre = t;
      }
    }
    if (centre === null) break;
    const cluster: Track[] = [];
    const rest: Track[] = [];
    for (const t of remaining) {
      const d = camelotSteps(centre, t) as number;
      (d <= steps ? cluster : rest).push(t);
    }
    out.push(cluster);
    remaining = rest;
  }
  return out;
}
