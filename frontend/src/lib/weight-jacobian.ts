// ─── Source-weight Jacobian ───────────────────────────────────
//
// The layout (2-D, or 3-D with a z axis) is a non-linear function of the
// per-source weights, so dragging a "Source weights" slider cannot re-project
// without a full t-SNE refit. The server exposes a first-order Jacobian
// d(position)/d(weight) around the current weights; this module applies it
// linearly for the small deltas of a drag, and the full recompute happens on
// release. It is deliberately a cheap approximation — the sign and direction
// are right, the magnitude only to first order.

import type { MapPos } from "../model";

export interface WeightJacobian {
  /** The weights the Jacobian was evaluated at (the drag's origin). */
  weights: Record<string, number>;
  /** Path order the base positions and per-source deltas are aligned with. */
  paths: string[];
  /** Base positions at ``weights``, aligned with ``paths``; ``z`` only for a 3-D payload. */
  base: MapPos[];
  /** ``source → per-track (dx, dy[, dz])`` in ``paths`` order; ``dz`` only when the row has one. */
  jacobian: Record<string, { dx: number; dy: number; dz?: number }[]>;
  /** Layout revision the Jacobian was evaluated at; empty when the server sent none.
   *  The base positions step from this exact layout, so a drag must never apply a
   *  Jacobian whose revision is no longer the one on screen. */
  revision: string;
}

/** Parse ``{positions, jacobian, weights, revision}`` from ``?jacobian=1``; ``null`` on any miss. */
export function parseWeightJacobian(raw: unknown): WeightJacobian | null {
  if (!raw || typeof raw !== "object") return null;
  const o = raw as {
    positions?: { path: string; x: number; y: number; z?: number }[];
    jacobian?: unknown;
    weights?: unknown;
    revision?: unknown;
  };
  if (!Array.isArray(o.positions) || !o.jacobian || typeof o.jacobian !== "object") {
    return null;
  }
  const jacobian: Record<string, { dx: number; dy: number; dz?: number }[]> = {};
  for (const [source, arr] of Object.entries(
    o.jacobian as Record<string, unknown>,
  )) {
    if (!Array.isArray(arr)) continue;
    jacobian[source] = arr.map((v) => {
      const d = v as { 0?: unknown; 1?: unknown; 2?: unknown };
      const dx = Number(d?.[0] ?? 0);
      const dy = Number(d?.[1] ?? 0);
      // A row with only two components is a 2-D delta: leave dz absent (the
      // accumulation reads it as 0) so 2-D output objects stay 2-D.
      return d?.[2] === undefined
        ? { dx, dy }
        : { dx, dy, dz: Number(d[2]) };
    });
  }
  return {
    weights: { ...((o.weights as Record<string, number>) ?? {}) },
    paths: o.positions.map((p) => p.path),
    base: o.positions.map((p) =>
      p.z === undefined
        ? { x: Number(p.x), y: Number(p.y) }
        : { x: Number(p.x), y: Number(p.y), z: Number(p.z) },
    ),
    jacobian,
    revision: typeof o.revision === "string" ? o.revision : "",
  };
}

/** Min/max of one cloud; ``minZ``/``maxZ`` only meaningful when it carries z. */
interface Bounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
  minZ: number;
  maxZ: number;
}

function bounds(pts: readonly { x: number; y: number; z?: number }[]): Bounds {
  const b: Bounds = {
    minX: Infinity,
    maxX: -Infinity,
    minY: Infinity,
    maxY: -Infinity,
    minZ: Infinity,
    maxZ: -Infinity,
  };
  for (const p of pts) {
    if (p.x < b.minX) b.minX = p.x;
    if (p.x > b.maxX) b.maxX = p.x;
    if (p.y < b.minY) b.minY = p.y;
    if (p.y > b.maxY) b.maxY = p.y;
    if (p.z !== undefined) {
      if (p.z < b.minZ) b.minZ = p.z;
      if (p.z > b.maxZ) b.maxZ = p.z;
    }
  }
  return b;
}

/** Approximate positions at *weights* by a first-order step from the base layout.
 *
 *  Always steps from the Jacobian's own base positions, never from the current
 *  map: re-basing on an already-approximated map while ``dw`` stays relative to
 *  the fetch-time weights would compound each drag event's delta.
 *
 *  Each axis is rescaled onto the base's own extent, never re-filled into the unit
 *  box. The served base is the *fitted* chart after its per-axis display
 *  normalisation, and that chart is then extended by Nyström placement for
 *  external roots; placed rows can lie outside the fitted `[0, 1]` bounds because
 *  they are evaluated from the frozen fit, not re-normalised with it. The base
 *  cloud's own per-axis centre and span is therefore the only frame that
 *  reproduces what is on screen, and a zero delta is exactly the identity.
 *  Normalising the stepped cloud standalone would resize and recentre the map on
 *  the first drag event. Rescaling each axis to the base span still strips the
 *  translation and uniform zoom a step accumulates, which is what a drag must not
 *  show. A non-zero step may change the displayed aspect ratio: that is the shape
 *  change the fitted chart's per-axis normalisation applies, not a shared-span
 *  gauge.
 */
export function approximatePositions(
  jac: WeightJacobian,
  weights: Record<string, number>,
): Map<string, MapPos> {
  const steps = Object.entries(jac.jacobian).map(([source, deltas]) => ({
    deltas,
    dw: (weights[source] ?? 0) - (jac.weights[source] ?? 0),
  }));
  // At the Jacobian's own weights the base *is* the answer; return it untouched
  // rather than round-tripping it through the frame fit, so a drag that starts
  // and returns to the anchor lands on the server's exact positions.
  if (steps.every((step) => !step.dw)) {
    const out = new Map<string, MapPos>();
    jac.paths.forEach((path, i) => {
      const pos = jac.base[i];
      if (pos) out.set(path, pos);
    });
    return out;
  }
  const paths: string[] = [];
  const base: MapPos[] = [];
  const moved: MapPos[] = [];
  for (let i = 0; i < jac.paths.length; i++) {
    const pos = jac.base[i];
    if (!pos) continue;
    let dx = 0;
    let dy = 0;
    let dz = 0;
    for (const step of steps) {
      if (!step.dw) continue;
      const d = step.deltas[i];
      if (!d) continue;
      dx += d.dx * step.dw;
      dy += d.dy * step.dw;
      dz += (d.dz ?? 0) * step.dw;
    }
    const z = pos.z === undefined ? undefined : pos.z + dz;
    paths.push(jac.paths[i]);
    base.push(pos);
    moved.push({ x: pos.x + dx, y: pos.y + dy, ...(z === undefined ? {} : { z }) });
  }
  const b = bounds(base);
  const m = bounds(moved);
  // Anchor on the base span; a degenerate base span (a flat cloud) falls back to
  // filling the unit span from the stepped cloud, like the fitted chart's
  // zero-span fallback.
  const scale = (baseSpan: number, movedSpan: number): number => {
    if (!(movedSpan > 1e-9)) return 1;
    return baseSpan > 1e-9 ? baseSpan / movedSpan : 1 / movedSpan;
  };
  const scaleX = scale(b.maxX - b.minX, m.maxX - m.minX);
  const scaleY = scale(b.maxY - b.minY, m.maxY - m.minY);
  const cxBase = (b.minX + b.maxX) / 2;
  const cyBase = (b.minY + b.maxY) / 2;
  const cxMoved = (m.minX + m.maxX) / 2;
  const cyMoved = (m.minY + m.maxY) / 2;
  const hasZ = moved.some((p) => p.z !== undefined);
  const scaleZ = hasZ ? scale(b.maxZ - b.minZ, m.maxZ - m.minZ) : 1;
  const czBase = (b.minZ + b.maxZ) / 2;
  const czMoved = (m.minZ + m.maxZ) / 2;
  const out = new Map<string, MapPos>();
  for (let i = 0; i < moved.length; i++) {
    const p = moved[i];
    const pos: MapPos = {
      x: cxBase + (p.x - cxMoved) * scaleX,
      y: cyBase + (p.y - cyMoved) * scaleY,
    };
    if (hasZ && p.z !== undefined) {
      pos.z = czBase + (p.z - czMoved) * scaleZ;
    }
    out.set(paths[i], pos);
  }
  return out;
}
