import { describe, expect, it } from "vitest";
import {
  DEFAULT_PERSPECTIVE,
  MAX_PERSPECTIVE,
  MAX_PITCH,
  ORBIT_CENTER,
  ORBIT_RAD_PER_PX,
  blendCamera,
  orbitDrag,
  orbitProject,
  orbitUnproject,
  readPerspective,
  type Orbit,
} from "./camera3d";

/** 3-D distance of a projected point from the cube centre (ignores `scale`). */
function distFromCentre(out: [number, number, number, number]): number {
  return Math.hypot(
    out[0] - ORBIT_CENTER,
    out[1] - ORBIT_CENTER,
    out[2],
  );
}

describe("camera3d orbitProject", () => {
  it("is the identity at yaw=pitch=0, depth centred on 0", () => {
    const [rx, ry, depth] = orbitProject({ yaw: 0, pitch: 0 }, 0.3, 0.7);
    expect(rx).toBeCloseTo(0.3, 10);
    expect(ry).toBeCloseTo(0.7, 10);
    expect(depth).toBeCloseTo(0, 10);
    // An explicit z is centred on 0: depth = z - 0.5.
    const [rx2, ry2, depth2] = orbitProject(
      { yaw: 0, pitch: 0 },
      0.3,
      0.7,
      0.9,
    );
    expect(rx2).toBeCloseTo(0.3, 10);
    expect(ry2).toBeCloseTo(0.7, 10);
    expect(depth2).toBeCloseTo(0.4, 10);
  });

  it("maps the cube centre to the cube centre at any orbit", () => {
    for (const o of [
      { yaw: 0, pitch: 0 },
      { yaw: 1.1, pitch: -0.7 },
      { yaw: -2.9, pitch: 1.4 },
    ]) {
      const [rx, ry, depth] = orbitProject(o, ORBIT_CENTER, ORBIT_CENTER);
      expect(rx).toBeCloseTo(ORBIT_CENTER, 10);
      expect(ry).toBeCloseTo(ORBIT_CENTER, 10);
      expect(depth).toBeCloseTo(0, 10);
    }
  });

  it("90° yaw swings the +z front to +x and a +x point to -z depth", () => {
    const o: Orbit = { yaw: Math.PI / 2, pitch: 0 };
    // The front face of the cube faces +x after a quarter turn.
    const [rx, ry] = orbitProject(o, ORBIT_CENTER, ORBIT_CENTER, 1);
    expect(rx).toBeCloseTo(1, 5);
    expect(ry).toBeCloseTo(ORBIT_CENTER, 5);
    // A +x point is pushed to the back: depth ≈ -0.5.
    const [, , depth] = orbitProject(o, 1, ORBIT_CENTER, ORBIT_CENTER);
    expect(depth).toBeCloseTo(-0.5, 5);
  });

  it("preserves distance from the cube centre", () => {
    const p: [number, number, number] = [0.2, 0.7, 0.9];
    const before = Math.hypot(
      p[0] - ORBIT_CENTER,
      p[1] - ORBIT_CENTER,
      p[2] - ORBIT_CENTER,
    );
    for (const o of [
      { yaw: 0.9, pitch: 0.4 },
      { yaw: -2.4, pitch: -1.2 },
      { yaw: 3.0, pitch: 1.5 },
    ]) {
      expect(distFromCentre(orbitProject(o, p[0], p[1], p[2]))).toBeCloseTo(
        before,
        8,
      );
    }
  });

  it("persp=0 keeps the first three outputs exactly as orthographic, scale 1", () => {
    const o: Orbit = { yaw: 0.9, pitch: -0.4 };
    for (const p of [
      [0.2, 0.7, 0.9],
      [0.8, 0.3, 0.1],
      [0.5, 0.5, 0.5],
    ]) {
      const [rx, ry, depth] = orbitProject(o, p[0], p[1], p[2]);
      const [prx, pry, pdepth, scale] = orbitProject(
        o,
        p[0],
        p[1],
        p[2],
        0,
      );
      expect(prx).toBeCloseTo(rx, 12);
      expect(pry).toBeCloseTo(ry, 12);
      expect(pdepth).toBeCloseTo(depth, 12);
      expect(scale).toBe(1);
    }
  });

  it("scales a nearer point up and a farther point down under perspective", () => {
    const o: Orbit = { yaw: 0, pitch: 0 };
    const persp = 0.28;
    // z=1 is the near face (rotated depth +0.5), z=0 the far face (-0.5).
    const [, , , nearScale] = orbitProject(o, 0.5, 0.5, 1, persp);
    const [, , , farScale] = orbitProject(o, 0.5, 0.5, 0, persp);
    const [, , , centreScale] = orbitProject(o, 0.5, 0.5, 0.5, persp);
    expect(nearScale).toBeCloseTo(1 / (1 - persp * 0.5), 10);
    expect(nearScale).toBeGreaterThan(1);
    expect(farScale).toBeCloseTo(1 / (1 + persp * 0.5), 10);
    expect(farScale).toBeLessThan(1);
    expect(centreScale).toBe(1);
  });

  it("pushes a near off-centre point outward and pulls a far one inward", () => {
    const o: Orbit = { yaw: 0, pitch: 0 };
    const persp = 0.28;
    const [nearRx] = orbitProject(o, 0.8, 0.5, 1, persp);
    const [farRx] = orbitProject(o, 0.8, 0.5, 0, persp);
    const [flatRx] = orbitProject(o, 0.8, 0.5, 0.5, persp);
    // Perspective scales the centred offset, so the near point sits further out.
    expect(nearRx).toBeGreaterThan(flatRx);
    expect(farRx).toBeLessThan(flatRx);
  });
});

describe("camera3d orbitUnproject", () => {
  it("round-trips orbitProject for a range of orbits, points and persp", () => {
    const points: [number, number, number][] = [
      [0.5, 0.5, 0.5],
      [0.2, 0.7, 0.9],
      [0.9, 0.1, 0.05],
      [0.05, 0.95, 1],
    ];
    for (const o of [
      { yaw: 0, pitch: 0 },
      { yaw: 1.1, pitch: -0.7 },
      { yaw: -2.9, pitch: 1.4 },
    ]) {
      for (const persp of [0, 0.15, 0.28, 0.6]) {
        for (const [x, y, z] of points) {
          const [rx, ry, depth] = orbitProject(o, x, y, z, persp);
          const [ux, uy, uz] = orbitUnproject(o, rx, ry, depth, persp);
          expect(ux).toBeCloseTo(x, 8);
          expect(uy).toBeCloseTo(y, 8);
          expect(uz).toBeCloseTo(z, 8);
        }
      }
    }
  });

  it("inverts with persp=0 as the plain rotation inverse", () => {
    const o: Orbit = { yaw: 1.3, pitch: 0.6 };
    const [rx, ry, depth] = orbitProject(o, 0.7, 0.2, 0.1);
    const [x, y, z] = orbitUnproject(o, rx, ry, depth);
    expect(x).toBeCloseTo(0.7, 10);
    expect(y).toBeCloseTo(0.2, 10);
    expect(z).toBeCloseTo(0.1, 10);
  });
});

describe("camera3d blendCamera", () => {
  it("is head-on orthographic at blend 0", () => {
    expect(
      blendCamera({ yaw: 1.2, pitch: 0.6 }, 0, 0.28),
    ).toEqual({ yaw: 0, pitch: 0, persp: 0 });
  });

  it("is the full orbit and perspective at blend 1", () => {
    expect(
      blendCamera({ yaw: 1.2, pitch: 0.6 }, 1, 0.28),
    ).toEqual({ yaw: 1.2, pitch: 0.6, persp: 0.28 });
  });

  it("scales every parameter by the blend at intermediate values", () => {
    expect(
      blendCamera({ yaw: 1.2, pitch: 0.6 }, 0.5, 0.28),
    ).toEqual({ yaw: 0.6, pitch: 0.3, persp: 0.14 });
  });
});

describe("camera3d readPerspective", () => {
  it("defaults when the parameter is absent", () => {
    expect(readPerspective("")).toBe(DEFAULT_PERSPECTIVE);
    expect(readPerspective("?foo=1")).toBe(DEFAULT_PERSPECTIVE);
    expect(readPerspective("?perspex=0.1")).toBe(DEFAULT_PERSPECTIVE);
  });

  it("reads a valid value with or without the leading ?", () => {
    expect(readPerspective("?persp=0.15")).toBeCloseTo(0.15, 10);
    expect(readPerspective("persp=0.15")).toBeCloseTo(0.15, 10);
    expect(readPerspective("?foo=1&persp=0.4")).toBeCloseTo(0.4, 10);
  });

  it("clamps out-of-range values to [0, MAX_PERSPECTIVE]", () => {
    expect(readPerspective("?persp=0.9")).toBe(MAX_PERSPECTIVE);
    expect(readPerspective("?persp=2")).toBe(MAX_PERSPECTIVE);
    expect(readPerspective("?persp=-0.1")).toBe(0);
  });

  it("falls back for non-numeric input", () => {
    expect(readPerspective("?persp=abc")).toBe(DEFAULT_PERSPECTIVE);
    expect(readPerspective("?persp=")).toBe(DEFAULT_PERSPECTIVE);
  });

  it("prefers the supplied fallback (the persisted setting) when absent or junk", () => {
    expect(readPerspective("", 0.12)).toBe(0.12);
    expect(readPerspective("?persp=abc", 0.12)).toBe(0.12);
    expect(readPerspective("?persp=0.3", 0.12)).toBeCloseTo(0.3, 10);
  });
});

describe("camera3d orbitDrag", () => {
  it("drag right increases yaw, drag down increases pitch", () => {
    const o: Orbit = { yaw: 0, pitch: 0 };
    orbitDrag(o, 10, 0);
    expect(o.yaw).toBeCloseTo(10 * ORBIT_RAD_PER_PX, 10);
    expect(o.pitch).toBe(0);

    const down: Orbit = { yaw: 0, pitch: 0 };
    orbitDrag(down, 0, 10);
    expect(down.pitch).toBeCloseTo(10 * ORBIT_RAD_PER_PX, 10);
    expect(down.yaw).toBe(0);
  });

  it("clamps pitch to ±MAX_PITCH", () => {
    const o: Orbit = { yaw: 0, pitch: 0 };
    orbitDrag(o, 0, 1e6);
    expect(o.pitch).toBe(MAX_PITCH);
    orbitDrag(o, 0, -1e6);
    expect(o.pitch).toBe(-MAX_PITCH);
  });

  it("wraps yaw to (-π, π]", () => {
    const over: Orbit = { yaw: Math.PI - 0.001, pitch: 0 };
    orbitDrag(over, 1, 0);
    expect(over.yaw).toBeLessThan(0);
    expect(over.yaw).toBeGreaterThan(-Math.PI);
    expect(over.yaw).toBeCloseTo(-Math.PI + 0.007, 10);

    const small: Orbit = { yaw: 0.5, pitch: 0 };
    orbitDrag(small, 1, 0);
    expect(small.yaw).toBeCloseTo(0.5 + ORBIT_RAD_PER_PX, 10);
  });
});
