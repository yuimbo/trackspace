import { describe, expect, it } from "vitest";
import {
  approximatePositions,
  parseWeightJacobian,
  type WeightJacobian,
} from "./weight-jacobian";

// Base spans [0,1] per axis, as the server's min-max normalisation guarantees.
// Five points, so the interior can move while the extremes define the frame.
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
});

describe("parseWeightJacobian", () => {
  it("parses positions, weights and the per-source deltas", () => {
    const parsed = parseWeightJacobian({
      positions: [{ path: "a.mp3", x: 0.2, y: 0.3 }],
      weights: { maest: 0.5 },
      jacobian: { maest: [[0.1, -0.1]] },
    });
    expect(parsed).not.toBeNull();
    expect(parsed!.paths).toEqual(["a.mp3"]);
    expect(parsed!.base).toEqual([{ x: 0.2, y: 0.3 }]);
    expect(parsed!.jacobian.maest[0]).toEqual({ dx: 0.1, dy: -0.1 });
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
});
