import { describe, expect, it } from "vitest";
import {
  approximatePositions,
  parseWeightJacobian,
  type WeightJacobian,
} from "./weight-jacobian";

const jac: WeightJacobian = {
  weights: { maest: 0.5, clap: 0.5 },
  paths: ["a.mp3", "b.mp3"],
  base: [
    { x: 0.2, y: 0.2 },
    { x: 0.8, y: 0.8 },
  ],
  jacobian: {
    maest: [
      { dx: 0.1, dy: -0.1 },
      { dx: 0.2, dy: 0.0 },
    ],
    clap: [
      { dx: 0.0, dy: 0.4 },
      { dx: -0.2, dy: 0.2 },
    ],
  },
};

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
    const out = approximatePositions(jac, { maest: 0.5, clap: 0.5 });
    expect(out.get("a.mp3")).toEqual({ x: 0.2, y: 0.2 });
    expect(out.get("b.mp3")).toEqual({ x: 0.8, y: 0.8 });
  });

  it("steps linearly per source and clamps to the unit square", () => {
    const out = approximatePositions(jac, { maest: 0.6, clap: 0.5 });
    // a.mp3: 0.2 + 0.1 * 0.1 in x, 0.2 - 0.1 * 0.1 in y.
    expect(out.get("a.mp3")!.x).toBeCloseTo(0.21);
    expect(out.get("a.mp3")!.y).toBeCloseTo(0.19);

    const far = approximatePositions(jac, { maest: 2.0, clap: 0.5 });
    expect(far.get("b.mp3")!.x).toBe(1);
    expect(far.get("b.mp3")!.y).toBe(0.8);
  });

  it("does not compound across successive steps from the same base", () => {
    const once = approximatePositions(jac, { maest: 0.6, clap: 0.5 });
    const twice = approximatePositions(jac, { maest: 0.6, clap: 0.5 });
    expect(twice.get("a.mp3")).toEqual(once.get("a.mp3"));
  });
});
