import { describe, expect, it } from "vitest";
import {
  approximatePositions,
  parseWeightJacobian,
  type WeightJacobian,
} from "./weight-jacobian";

// A square base spanning [0,1] on both axes, i.e. the frame a first ungauged
// layout gets. Five points, so the interior can move while the extremes define
// the frame.
const jac = (): WeightJacobian => ({
  weights: { maest: 0.5, clap: 0.5 },
  paths: ["a.mp3", "b.mp3", "c.mp3", "d.mp3", "e.mp3"],
  base: [
    { x: 0.0, y: 0.0 },
    { x: 0.25, y: 0.25 },
    { x: 0.5, y: 0.5 },
    { x: 0.75, y: 0.75 },
    { x: 1.0, y: 1.0 },
  ],
  jacobian: {
    maest: [
      { dx: 0.0, dy: 0.0 },
      { dx: 0.1, dy: -0.1 },
      { dx: 0.2, dy: -0.2 },
      { dx: 0.1, dy: -0.1 },
      { dx: 0.0, dy: 0.0 },
    ],
  },
  revision: "rev-1",
});

// A 3-D fixture: base positions carry z and the deltas a dz component.
const jac3d = (): WeightJacobian => ({
  weights: { maest: 0.5, clap: 0.5 },
  paths: ["a.mp3", "b.mp3", "c.mp3", "d.mp3", "e.mp3"],
  base: [
    { x: 0.0, y: 0.0, z: 0.0 },
    { x: 0.25, y: 0.25, z: 0.25 },
    { x: 0.5, y: 0.5, z: 0.5 },
    { x: 0.75, y: 0.75, z: 0.75 },
    { x: 1.0, y: 1.0, z: 1.0 },
  ],
  jacobian: {
    maest: [
      { dx: 0.0, dy: 0.0, dz: 0.0 },
      { dx: 0.1, dy: -0.1, dz: 0.1 },
      { dx: 0.2, dy: -0.2, dz: 0.2 },
      { dx: 0.1, dy: -0.1, dz: 0.1 },
      { dx: 0.0, dy: 0.0, dz: 0.0 },
    ],
  },
  revision: "rev-1",
});

// A rectangular cloud that sits outside the unit box: the server normalises each
// axis *before* the gauge, so a displayed layout can be 4:1, offset and larger
// than [0,1]². Four points; the extremes define its per-axis frame.
const rect = (): WeightJacobian => ({
  weights: { maest: 0.5, clap: 0.5 },
  paths: ["a.mp3", "b.mp3", "c.mp3", "d.mp3"],
  base: [
    { x: 1.2, y: -0.4 },
    { x: 1.8, y: -0.2 },
    { x: 2.6, y: 0.0 },
    { x: 3.2, y: 0.1 },
  ],
  jacobian: {
    maest: [
      { dx: 0.1, dy: 0.2 },
      { dx: -0.1, dy: 0.1 },
      { dx: 0.2, dy: -0.1 },
      { dx: -0.2, dy: -0.2 },
    ],
  },
  revision: "rev-1",
});

const spans = (pts: Iterable<{ x: number; y: number }>) => {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const p of pts) {
    minX = Math.min(minX, p.x);
    maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y);
    maxY = Math.max(maxY, p.y);
  }
  return { x: maxX - minX, y: maxY - minY };
};

describe("parseWeightJacobian", () => {
  it("parses positions, weights and the per-source deltas", () => {
    const parsed = parseWeightJacobian({
      positions: [{ path: "a.mp3", x: 0.2, y: 0.3 }],
      weights: { maest: 0.5 },
      jacobian: { maest: [[0.1, -0.1]] },
      revision: "rev-9",
    });
    expect(parsed).not.toBeNull();
    expect(parsed!.paths).toEqual(["a.mp3"]);
    expect(parsed!.base).toEqual([{ x: 0.2, y: 0.3 }]);
    expect(parsed!.jacobian.maest[0]).toEqual({ dx: 0.1, dy: -0.1 });
    expect(parsed!.revision).toBe("rev-9");
  });

  it("defaults an absent revision to empty", () => {
    const parsed = parseWeightJacobian({
      positions: [{ path: "a.mp3", x: 0.2, y: 0.3 }],
      weights: { maest: 0.5 },
      jacobian: { maest: [[0.1, -0.1]] },
    });
    expect(parsed!.revision).toBe("");
  });

  it("keeps z and dz for a 3-D payload", () => {
    const parsed = parseWeightJacobian({
      positions: [{ path: "a.mp3", x: 0.2, y: 0.3, z: 0.7 }],
      weights: { maest: 0.5 },
      jacobian: { maest: [[0.1, -0.1, 0.2]] },
    });
    expect(parsed).not.toBeNull();
    expect(parsed!.base).toEqual([{ x: 0.2, y: 0.3, z: 0.7 }]);
    expect(parsed!.jacobian.maest[0]).toEqual({ dx: 0.1, dy: -0.1, dz: 0.2 });
  });

  it("leaves z and dz absent for a 2-D payload", () => {
    const parsed = parseWeightJacobian({
      positions: [{ path: "a.mp3", x: 0.2, y: 0.3 }],
      weights: { maest: 0.5 },
      jacobian: { maest: [[0.1, -0.1]] },
    });
    expect(parsed!.base[0]).toEqual({ x: 0.2, y: 0.3 });
    expect(parsed!.jacobian.maest[0]).toEqual({ dx: 0.1, dy: -0.1 });
  });

  it("rejects a response without positions or jacobian", () => {
    expect(parseWeightJacobian(null)).toBeNull();
    expect(parseWeightJacobian({ positions: [] })).toBeNull();
    expect(parseWeightJacobian({ jacobian: {} })).toBeNull();
  });

  it("accepts an empty Jacobian (preview unavailable) and holds the exact base", () => {
    // The 3-D no-landmark path returns the exact base layout with an empty
    // per-source Jacobian: parsing succeeds and a drag holds the base positions.
    const parsed = parseWeightJacobian({
      positions: [
        { path: "a.mp3", x: 0.0, y: 0.0, z: 0.0 },
        { path: "b.mp3", x: 1.0, y: 1.0, z: 1.0 },
      ],
      weights: { maest: 0.5 },
      jacobian: {},
      revision: "rev-1",
    });
    expect(parsed).not.toBeNull();
    expect(parsed!.jacobian).toEqual({});
    // The step is anchored to the base cloud, so a drag leaves every track
    // exactly where the server placed it.
    const out = approximatePositions(parsed!, { maest: 0.7 });
    expect(out.get("a.mp3")).toEqual({ x: 0, y: 0, z: 0 });
    expect(out.get("b.mp3")).toEqual({ x: 1, y: 1, z: 1 });
  });
});

describe("approximatePositions", () => {
  it("is the identity at the Jacobian's own weights", () => {
    const out = approximatePositions(jac(), { maest: 0.5, clap: 0.5 });
    expect(out.get("a.mp3")).toEqual({ x: 0, y: 0 });
    expect(out.get("c.mp3")).toEqual({ x: 0.5, y: 0.5 });
    expect(out.get("e.mp3")).toEqual({ x: 1, y: 1 });
  });

  it("moves the interior and keeps every point in the unit square", () => {
    const out = approximatePositions(jac(), { maest: 0.6, clap: 0.5 });
    // The extremes define the frame, so the interior carries the visible change.
    const c = out.get("c.mp3")!;
    expect(c.y).not.toBeCloseTo(0.5);
    for (const p of out.values()) {
      expect(p.x).toBeGreaterThanOrEqual(0);
      expect(p.x).toBeLessThanOrEqual(1);
      expect(p.y).toBeGreaterThanOrEqual(0);
      expect(p.y).toBeLessThanOrEqual(1);
    }
  });

  it("ignores a pure translation, which carries no shape information", () => {
    const j = jac();
    // Every point offset by the same amount -> renormalised back to the base.
    j.jacobian = {
      maest: [
        { dx: 0.3, dy: 0.3 },
        { dx: 0.3, dy: 0.3 },
        { dx: 0.3, dy: 0.3 },
        { dx: 0.3, dy: 0.3 },
        { dx: 0.3, dy: 0.3 },
      ],
    };
    const out = approximatePositions(j, { maest: 0.9, clap: 0.5 });
    expect(out.get("a.mp3")!.x).toBeCloseTo(0);
    expect(out.get("c.mp3")!.x).toBeCloseTo(0.5);
    expect(out.get("e.mp3")!.x).toBeCloseTo(1);
  });

  it("does not compound across successive steps from the same base", () => {
    const once = approximatePositions(jac(), { maest: 0.6, clap: 0.5 });
    const twice = approximatePositions(jac(), { maest: 0.6, clap: 0.5 });
    expect(twice.get("c.mp3")).toEqual(once.get("c.mp3"));
  });

  it("moves z alongside x and y for a 3-D payload", () => {
    const out = approximatePositions(jac3d(), { maest: 0.6, clap: 0.5 });
    const c = out.get("c.mp3")!;
    // The interior carries the visible change on every axis; z tilts with them.
    expect(c.z).toBeCloseTo(0.52, 5);
    for (const p of out.values()) {
      expect(p.z).toBeDefined();
      expect(p.z!).toBeGreaterThanOrEqual(0);
      expect(p.z!).toBeLessThanOrEqual(1);
    }
  });

  it("renormalises z to [0,1] like x and y", () => {
    const j = jac3d();
    // All z start at 0.5; a strong dw pushes the extremes out of the cube.
    j.base = j.base.map((p) => ({ ...p, z: 0.5 }));
    j.jacobian.maest = [
      { dx: 0, dy: 0, dz: 2 },
      { dx: 0, dy: 0, dz: 0 },
      { dx: 0, dy: 0, dz: 0 },
      { dx: 0, dy: 0, dz: 0 },
      { dx: 0, dy: 0, dz: -2 },
    ];
    const out = approximatePositions(j, { maest: 1.0, clap: 0.5 });
    // Raw z: 1.5 and -0.5 (outside the cube); renormalised back to the frame.
    expect(out.get("a.mp3")!.z).toBe(1);
    expect(out.get("e.mp3")!.z).toBe(0);
    for (const p of out.values()) {
      expect(p.z).toBeGreaterThanOrEqual(0);
      expect(p.z).toBeLessThanOrEqual(1);
    }
  });

  it("stays 2-D for a payload without z", () => {
    const out = approximatePositions(jac(), { maest: 0.6, clap: 0.5 });
    const c = out.get("c.mp3")!;
    expect(c.z).toBeUndefined();
    expect("z" in c).toBe(false);
  });

  it("returns a rectangular, out-of-box base exactly at zero delta", () => {
    // A displayed layout is normalised before the gauge, so it may be 4:1 and
    // sit outside [0,1]². Re-normalising it standalone would recentre and
    // rescale the map on the first drag event; a zero delta must be the identity.
    const j = rect();
    const out = approximatePositions(j, { maest: 0.5, clap: 0.5 });
    j.base.forEach((p, i) => expect(out.get(j.paths[i])).toEqual(p));
  });

  it("maps each axis to its own base extent", () => {
    const j = rect();
    const before = spans(j.base);
    expect(before.x / before.y).toBeCloseTo(4, 5);
    const out = approximatePositions(j, { maest: 0.6, clap: 0.5 });
    const after = spans(out.values());
    // Per-axis min/max anchoring restores both base extents independently:
    // x 2, y 0.5 (4:1), not a shared-span 1.7 / 0.1 raw-step frame.
    expect(after.x).toBeCloseTo(before.x, 6);
    expect(after.y).toBeCloseTo(before.y, 6);
    // Controlled interior: the stepped cloud's x span is 1.97 centred at 2.195,
    // so scaleX = 2/1.97 around the base centre 2.2.
    expect(out.get("b.mp3")!.x).toBeCloseTo(1.788832, 5);
    expect(out.get("c.mp3")!.x).toBeCloseTo(2.631472, 5);
    // Its y span is 0.46 centred at -0.15, so scaleY = 0.5/0.46 restores the
    // base extent; c lands near the base minimum, not the unit-box maximum.
    expect(out.get("c.mp3")!.y).toBeCloseTo(0.002174, 5);
  });

  it("renormalises a short-axis move to the base extent, not the unit box", () => {
    const j = rect();
    j.jacobian = {
      maest: [
        { dx: 0, dy: -0.2 },
        { dx: 0, dy: 0 },
        { dx: 0, dy: 0 },
        { dx: 0, dy: 0.2 },
      ],
    };
    const before = spans(j.base);
    const out = approximatePositions(j, { maest: 0.6, clap: 0.5 });
    const after = spans(out.values());
    // The stepped y span is 0.54; per-axis scaling maps it back to the base's
    // 0.5 (not [0,1]), so a single-axis drag cannot refill the whole box.
    expect(after.y).toBeCloseTo(before.y, 6);
    expect(after.y).toBeLessThan(1);
    expect(out.get("b.mp3")!.y).toBeCloseTo(-0.196296, 5);
    expect(out.get("c.mp3")!.y).toBeCloseTo(-0.011111, 5);
  });

  it("maps x, y and z to independent base extents in a rectangular 3-D base", () => {
    const j = jac3d();
    j.base = [
      { x: 0, y: 0, z: 0 },
      { x: 0.25, y: 0.5, z: 0.25 },
      { x: 0.5, y: 1, z: 0.5 },
      { x: 0.75, y: 1.5, z: 0.75 },
      { x: 1, y: 2, z: 1 },
    ];
    const before = spans(j.base);
    const out = approximatePositions(j, { maest: 0.6, clap: 0.5 });
    const after = spans(out.values());
    expect(after.x).toBeCloseTo(before.x, 6);
    expect(after.y).toBeCloseTo(before.y, 6);
    let minZ = Infinity;
    let maxZ = -Infinity;
    for (const p of out.values()) {
      expect(Number.isFinite(p.x)).toBe(true);
      expect(Number.isFinite(p.y)).toBe(true);
      expect(Number.isFinite(p.z)).toBe(true);
      minZ = Math.min(minZ, p.z!);
      maxZ = Math.max(maxZ, p.z!);
    }
    expect(maxZ - minZ).toBeCloseTo(1, 6);
  });
});
