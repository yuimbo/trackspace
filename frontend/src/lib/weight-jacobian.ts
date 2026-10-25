// ─── Source-weight Jacobian ───────────────────────────────────
//
// The layout (2-D, or 3-D with a z axis) is a non-linear function of the
// per-source weights, so dragging a "Source weights" slider cannot re-project
// without a full t-SNE refit. The server exposes a first-order Jacobian
// d(position)/d(weight) around the current weights; this module applies it
// linearly for the small deltas of a drag, and the full recompute happens on
// release. It is deliberately a cheap approximation — the sign and direction
// are right, the magnitude only to first order.

import { clamp01 } from "./util";
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

/** Approximate positions at *weights* by a first-order step from the base layout.
 *
 *  Always steps from the Jacobian's own base positions, never from the current
 *  map: re-basing on an already-approximated map while ``dw`` stays relative to
 *  the fetch-time weights would compound each drag event's delta.
 *
 *  The result is renormalised per axis (min-max, clamped to the unit square —
 *  or the unit cube when the payload carries z), matching the server's min-max
 *  normalisation of every layout, so a drag cannot drift or zoom the map out of
 *  frame. The Jacobian already has its translation/uniform-zoom component removed,
 *  so the visible change is the shape change.
 */
export function approximatePositions(
  jac: WeightJacobian,
  weights: Record<string, number>,
): Map<string, MapPos> {
  const steps = Object.entries(jac.jacobian).map(([source, deltas]) => ({
    deltas,
    dw: (weights[source] ?? 0) - (jac.weights[source] ?? 0),
  }));
  const moved: { path: string; x: number; y: number; z?: number }[] = [];
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  let minZ = Infinity;
  let maxZ = -Infinity;
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
    const x = pos.x + dx;
    const y = pos.y + dy;
    const z = pos.z === undefined ? undefined : pos.z + dz;
    moved.push({ path: jac.paths[i], x, y, ...(z === undefined ? {} : { z }) });
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (z !== undefined) {
      if (z < minZ) minZ = z;
      if (z > maxZ) maxZ = z;
    }
  }
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;
  const hasZ = moved.some((p) => p.z !== undefined);
  const spanZ = hasZ ? maxZ - minZ || 1 : 1;
  const out = new Map<string, MapPos>();
  for (const p of moved) {
    const pos: MapPos = {
      x: clamp01((p.x - minX) / spanX),
      y: clamp01((p.y - minY) / spanY),
    };
    if (hasZ && p.z !== undefined) pos.z = clamp01((p.z - minZ) / spanZ);
    out.set(p.path, pos);
  }
  return out;
}
