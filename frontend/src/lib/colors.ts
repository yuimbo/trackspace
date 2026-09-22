import {
  CAMELOT_MAJOR_RADIUS,
  CAMELOT_MINOR_RADIUS,
  SPIRAL_R_MAX,
  SPIRAL_R_MIN,
  bpmSpiralPolar,
  camelotPolar,
} from "./lenses";
import { clamp } from "./util";

// ─── CSS colour parsing ─────────────────────────────────────
//
// The canvas palettes emit `hsl()`, but the WebGL glow layer needs linear RGB
// channels. One parser here keeps every colour consumer on the same conversion.

/** `hsl(h, s, l)` with `s`/`l` in 0..1 to linear RGB channels, each 0..1. */
export function hslToRgb(h: number, s: number, l: number): [number, number, number] {
  const a = s * Math.min(l, 1 - l);
  const f = (n: number) => {
    const k = (n + h / 30) % 12;
    return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return [f(0), f(8), f(4)];
}

/** Parse the CSS colours the palette emits (`#rgb`, `#rrggbb`, `rgb()`, `hsl()`). */
export function parseColor(css: string): [number, number, number] {
  const s = css.trim().toLowerCase();
  if (s.startsWith("#")) {
    let hex = s.slice(1);
    if (hex.length === 3)
      hex = hex
        .split("")
        .map((c) => c + c)
        .join("");
    const n = parseInt(hex, 16);
    return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
  }
  const nums = (s.match(/-?\d*\.?\d+/g) ?? []).map(Number);
  if (s.startsWith("hsl")) {
    return hslToRgb(nums[0] ?? 0, (nums[1] ?? 0) / 100, (nums[2] ?? 0) / 100);
  }
  return [(nums[0] ?? 0) / 255, (nums[1] ?? 0) / 255, (nums[2] ?? 0) / 255];
}

// ─── Folder coloring ─────────────────────────────────────────
//
// `dirColor` is a pure function of a virtual folder path (`rootId/sub/…`). Roots
// and their top-level folders keep the original path-hash hue, so every playlist
// stays vivid and distinguishable. Deeper folders are tints/shades of their
// top-level ancestor's hue, and the deviation halves with each nesting level, so a
// subtree reads as one colour family that tightens as it deepens.
//
// The algorithm is mirrored in `backend/services/dir_colors.py` (the folder-tree
// Jinja partial paints the same swatches server-side); both test suites assert the
// same golden vectors in `dir-color-vectors.json`.

const HASH_MASK = 0x3ffff;

/** Original 31-rolling path hash; the mult/seed args salt the offset channels. */
function hashPath(s: string, mult: number, seed: number): number {
  let h = seed & HASH_MASK;
  for (let i = 0; i < s.length; i++)
    h = (h * mult + s.charCodeAt(i)) & HASH_MASK;
  return h;
}

/** Deterministic value in [-1, 1] from a hash. */
function signedUnit(h: number): number {
  return (h % 2001) / 1000 - 1;
}

/** Round half up (matches Python's `math.floor(x + 0.5)`). */
function roundHalfUp(x: number): number {
  return Math.floor(x + 0.5);
}

export function dirColor(folderPath: string): string {
  if (!folderPath) return "hsl(350,60%,55%)";
  const segments = folderPath.split("/");
  const depth = segments.length - 1;
  const base = hashPath(folderPath, 31, 0);
  if (depth <= 1) return `hsl(${base % 360},65%,60%)`;

  const hue = hashPath(`${segments[0]}/${segments[1]}`, 31, 0) % 360;
  // Depth 2 is the widest tint; each level past it halves the deviation. The
  // amplitude is deliberately wide (lightness ±30, saturation ±18 at depth 2) so
  // sibling subgenres stay separable on the map.
  const f = Math.pow(0.5, depth - 2);
  const dl = signedUnit(hashPath(folderPath, 7, 11)) * 30 * f;
  const ds = signedUnit(hashPath(folderPath, 13, 23)) * 18 * f;
  const light = roundHalfUp(60 + dl);
  const sat = roundHalfUp(65 + ds);
  return `hsl(${hue},${sat}%,${light}%)`;
}

/**
 * Colour stops for a folder *tag*: a single colour for a root or top-level
 * folder, or a parent→own gradient for a nested folder, so the tag reads as a
 * deeper tint of its family (mirrors how `dirColor` shades deeper paths).
 */
export function dirTagColors(folderPath: string): string[] {
  const segments = folderPath.split("/");
  if (segments.length - 1 < 2) return [dirColor(folderPath)];
  return [dirColor(segments.slice(0, -1).join("/")), dirColor(folderPath)];
}

// ─── Cluster coloring ────────────────────────────────────────
/** Golden-angle hue for cluster *id*.
 *
 * Successive ids land far apart on the colour wheel, so adjacent clusters
 * stay visually distinct however many there are — unlike hashing, which
 * happily gives two neighbouring clusters near-identical hues. `-1` means
 * "unclustered" and renders grey.
 */
export function clusterColor(id: number): string {
  if (id < 0) return "hsl(0,0%,45%)";
  const hue = (id * 137.508) % 360;
  // Alternate lightness so even a wrap-around hue collision stays separable.
  const light = id % 2 === 0 ? 62 : 52;
  return `hsl(${hue.toFixed(1)},68%,${light}%)`;
}

// ─── Feature-lens coloring ───────────────────────────────────
//
// The tempo and key lenses lay tracks out in polar coordinates. Their colouring
// reuses that same geometry — angle → hue, radius from the centre → brightness —
// so two tracks that sit close on the map also read as close in colour. Keeping one
// polar source (the lens functions) is what makes "colour proximity = spatial
// proximity" hold instead of drifting into a second, unrelated mapping.

/** Polar colour: angle → hue, radius (normalised over `rLo`..`rHi`) → lightness. */
function polarColor(angle: number, radius: number, rLo: number, rHi: number): string {
  const hue = ((((angle / (2 * Math.PI)) % 1) + 1) % 1) * 360;
  const t = rHi > rLo ? clamp((radius - rLo) / (rHi - rLo)) : 0;
  // 36%..72% keeps both rings legible on the dark canvas while still reading
  // as "near = dark, far = bright".
  return `hsl(${hue.toFixed(1)},72%,${(36 + t * 36).toFixed(1)}%)`;
}

/** Colour for a track's tempo: hue is its angle on the spiral, brightness its
 *  log-BPM radius. Mirrors `bpmSpiralXY`. */
export function tempoColor(tempoNorm: number): string {
  const { angle, radius } = bpmSpiralPolar(tempoNorm);
  return polarColor(angle, radius, SPIRAL_R_MIN, SPIRAL_R_MAX);
}

/** Colour for a track's key: hue is its Camelot angle, brightness the major
 *  (outer, bright) vs minor (inner, dark) ring. Mirrors `camelotXY`. */
export function keyColor(keyTurns: number, mode: number): string {
  const { angle, radius } = camelotPolar(keyTurns, mode);
  return polarColor(angle, radius, CAMELOT_MINOR_RADIUS, CAMELOT_MAJOR_RADIUS);
}
