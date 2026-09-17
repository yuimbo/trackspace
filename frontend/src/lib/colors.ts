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
  // Depth 2 is the widest tint; each level past it halves the deviation.
  const f = Math.pow(0.5, depth - 2);
  const dl = signedUnit(hashPath(folderPath, 7, 11)) * 10 * f;
  const ds = signedUnit(hashPath(folderPath, 13, 23)) * 6 * f;
  const light = roundHalfUp(60 + dl);
  const sat = roundHalfUp(65 + ds);
  return `hsl(${hue},${sat}%,${light}%)`;
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
