import { describe, expect, it } from "vitest";
import { packPointVertices, type PointSpec } from "./point-layer";

describe("packPointVertices", () => {
  const point: PointSpec = {
    x: 10,
    y: 20,
    radius: 6.5,
    color: [0.1, 0.2, 0.3],
    alpha: 0.5,
  };

  it("emits eleven floats per vertex across six vertices", () => {
    expect(packPointVertices([point]).length).toBe(66);
  });

  it("carries the centre, radius, colour, alpha, weight and depth on every vertex", () => {
    const data = packPointVertices([{ ...point, weight: 0.15, depth: 0.8 }]);
    for (let v = 0; v < 6; v++) {
      const o = v * 11;
      expect([data[o], data[o + 1]]).toEqual([10, 20]);
      expect(data[o + 4]).toBe(6.5);
      expect(data[o + 5]).toBeCloseTo(0.1);
      expect(data[o + 6]).toBeCloseTo(0.2);
      expect(data[o + 7]).toBeCloseTo(0.3);
      expect(data[o + 8]).toBe(0.5);
      expect(data[o + 9]).toBeCloseTo(0.15);
      expect(data[o + 10]).toBeCloseTo(0.8);
    }
  });

  it("defaults the colour priority to full and the depth to mid", () => {
    const data = packPointVertices([point]);
    expect(data[9]).toBe(1);
    expect(data[10]).toBe(0.5);
  });

  it("keeps per-point depth aligned when packing a mixed list", () => {
    const points: PointSpec[] = [
      { ...point, depth: 0.1 },
      point, // depth defaults to 0.5
      { ...point, depth: 0.9 },
    ];
    const data = packPointVertices(points);
    const expected = [0.1, 0.5, 0.9];
    for (let p = 0; p < points.length; p++) {
      for (let v = 0; v < 6; v++) {
        expect(data[(p * 6 + v) * 11 + 10]).toBeCloseTo(expected[p]);
      }
    }
  });

  it("expands the quad corners across both triangles", () => {
    const data = packPointVertices([point]);
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

  it("packs no vertices for no points", () => {
    expect(packPointVertices([]).length).toBe(0);
  });
});
