import { describe, expect, it } from "vitest";
import { packGlowVertices, type GlowPoint } from "./glow-layer";

describe("packGlowVertices", () => {
  const point: GlowPoint = {
    x: 10,
    y: 20,
    radius: 30,
    core: 0.4,
    color: [0.1, 0.2, 0.3],
    alpha: 0.5,
  };

  it("emits six vertices per point", () => {
    expect(packGlowVertices([point]).length).toBe(66);
  });

  it("carries the centre, radius, core, colour, alpha and depth on every vertex", () => {
    const data = packGlowVertices([{ ...point, depth: 0.8 }]);
    for (let v = 0; v < 6; v++) {
      const o = v * 11;
      expect([data[o], data[o + 1]]).toEqual([10, 20]);
      expect(data[o + 4]).toBe(30);
      expect(data[o + 5]).toBeCloseTo(0.4);
      expect(data[o + 6]).toBeCloseTo(0.1);
      expect(data[o + 7]).toBeCloseTo(0.2);
      expect(data[o + 8]).toBeCloseTo(0.3);
      expect(data[o + 9]).toBeCloseTo(0.5);
      expect(data[o + 10]).toBeCloseTo(0.8);
    }
  });

  it("defaults the depth to the front (1), so a 2-D halo is never occluded", () => {
    const data = packGlowVertices([point]);
    expect(data[10]).toBe(1);
  });

  it("keeps per-point depth aligned when packing a mixed list", () => {
    const points: GlowPoint[] = [
      { ...point, depth: 0.1 },
      point, // depth defaults to 1
      { ...point, depth: 0.9 },
    ];
    const data = packGlowVertices(points);
    const expected = [0.1, 1, 0.9];
    for (let p = 0; p < points.length; p++) {
      for (let v = 0; v < 6; v++) {
        expect(data[(p * 6 + v) * 11 + 10]).toBeCloseTo(expected[p]);
      }
    }
  });

  it("expands the quad corners across both triangles", () => {
    const data = packGlowVertices([point]);
    const corners = Array.from({ length: 6 }, (_, v) => [
      data[v * 11 + 2],
      data[v * 11 + 3],
    ]);
    expect(corners).toEqual([
      [-1, -1],
      [1, -1],
      [1, 1],
      [-1, -1],
      [1, 1],
      [-1, 1],
    ]);
  });
});
