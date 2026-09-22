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

  it("emits ten floats per vertex across six vertices", () => {
    expect(packPointVertices([point]).length).toBe(60);
  });

  it("carries the centre, radius, colour, alpha and weight on every vertex", () => {
    const data = packPointVertices([{ ...point, weight: 0.15 }]);
    for (let v = 0; v < 6; v++) {
      const o = v * 10;
      expect([data[o], data[o + 1]]).toEqual([10, 20]);
      expect(data[o + 4]).toBe(6.5);
      expect(data[o + 5]).toBeCloseTo(0.1);
      expect(data[o + 6]).toBeCloseTo(0.2);
      expect(data[o + 7]).toBeCloseTo(0.3);
      expect(data[o + 8]).toBe(0.5);
      expect(data[o + 9]).toBeCloseTo(0.15);
    }
  });

  it("defaults the colour priority to full", () => {
    const data = packPointVertices([point]);
    expect(data[9]).toBe(1);
  });

  it("expands the quad corners across both triangles", () => {
    const data = packPointVertices([point]);
    const corners = Array.from({ length: 6 }, (_, v) => [
      data[v * 10 + 2],
      data[v * 10 + 3],
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
