// ─── Source-weight Jacobian ───────────────────────────────────
//
// The 2-D layout is a non-linear function of the per-source weights, so dragging
// a "Source weights" slider cannot re-project without a full t-SNE refit. The server
// exposes a first-order Jacobian d(position)/d(weight) around the current weights;
// this module applies it linearly for the small deltas of a drag, and the full
// recompute happens on release. It is deliberately a cheap approximation — the sign
// and direction are right, the magnitude only to first order.

export interface WeightJacobian {
  /** The weights the Jacobian was evaluated at (the drag's origin). */
  weights: Record<string, number>;
  /** Path order the base positions and per-source deltas are aligned with. */
  paths: string[];
  /** Base 2-D positions at ``weights``, aligned with ``paths``. */
  base: { x: number; y: number }[];
  /** ``source → per-track (dx, dy)`` in ``paths`` order. */
  jacobian: Record<string, { dx: number; dy: number }[]>;
}

/** Parse ``{positions, jacobian, weights}`` from ``?jacobian=1``; ``null`` on any miss. */
export function parseWeightJacobian(raw: unknown): WeightJacobian | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as {
    positions?: { path: string; x: number; y: number }[];
    jacobian?: unknown;
    weights?: unknown;
  };
  if (!Array.isArray(o.positions) || !o.jacobian || typeof o.jacobian !== "object") {
    return null;
  }
  const jacobian: Record<string, { dx: number; dy: number }[]> = {};
  for (const [source, arr] of Object.entries(
    o.jacobian as Record<string, unknown>,
  )) {
    if (!Array.isArray(arr)) continue;
    jacobian[source] = arr.map((v) => {
      const d = v as { 0?: unknown; 1?: unknown };
      return { dx: Number(d?.[0] ?? 0), dy: Number(d?.[1] ?? 0) };
    });
  }
  return {
    weights: { ...((o.weights as Record<string, number>) ?? {}) },
    paths: o.positions.map((p) => p.path),
    base: o.positions.map((p) => ({ x: Number(p.x), y: Number(p.y) })),
    jacobian,
  };
}

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

/** Approximate positions at *weights* by a first-order step from the base layout.
 *
 *  Always steps from the Jacobian's own base positions, never from the current
 *  map: re-basing on an already-approximated map while ``dw`` stays relative to
 *  the fetch-time weights would compound each drag event's delta.
 *
 *  The result is renormalised to the unit square, matching the server's min-max
 *  normalisation of every layout, so a drag cannot drift or zoom the map out of
 *  frame. The Jacobian already has its translation/uniform-zoom component removed,
 *  so the visible change is the shape change.
 */
export function approximatePositions(
  jac: WeightJacobian,
  weights: Record<string, number>,
): Map<string, { x: number; y: number }> {
  const steps = Object.entries(jac.jacobian).map(([source, deltas]) => ({
    deltas,
    dw: (weights[source] ?? 0) - (jac.weights[source] ?? 0),
  }));
  const moved: { path: string; x: number; y: number }[] = [];
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < jac.paths.length; i++) {
    const pos = jac.base[i];
    if (!pos) continue;
    let dx = 0;
    let dy = 0;
    for (const step of steps) {
      if (!step.dw) continue;
      const d = step.deltas[i];
      if (!d) continue;
      dx += d.dx * step.dw;
      dy += d.dy * step.dw;
    }
    const x = pos.x + dx;
    const y = pos.y + dy;
    moved.push({ path: jac.paths[i], x, y });
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const out = new Map<string, { x: number; y: number }>();
  for (const p of moved) {
    out.set(p.path, { x: clamp01((p.x - minX) / spanX), y: clamp01((p.y - minY) / spanY) });
  }
  return out;
}
