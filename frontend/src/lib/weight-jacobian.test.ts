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
});
