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
    expect(packGlowVertices([point]).length).toBe(60);
  });

  it("carries the centre, radius, core, colour and alpha on every vertex", () => {
    const data = packGlowVertices([point]);
    for (let v = 0; v < 6; v++) {
      const o = v * 10;
      expect([data[o], data[o + 1]]).toEqual([10, 20]);
      expect(data[o + 4]).toBe(30);
      expect(data[o + 5]).toBeCloseTo(0.4);
      expect(data[o + 6]).toBeCloseTo(0.1);
      expect(data[o + 7]).toBeCloseTo(0.2);
      expect(data[o + 8]).toBeCloseTo(0.3);
      expect(data[o + 9]).toBeCloseTo(0.5);
    }
  });

  it("expands the quad corners across both triangles", () => {
    const data = packGlowVertices([point]);
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
});
