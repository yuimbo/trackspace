// ─── 3-D orbit camera (pure) ───────────────────────────────────
//
// The embedding scatter gains a perspective 3-D mode: a point's world position
// becomes (x, y, z) in the unit cube, the cube is rotated about its centre
// (0.5, 0.5, 0.5) by yaw/pitch, and only the rotated (x, y) feeds the existing
// 2-D viewport transform — pan/zoom stay untouched. The rotated z is a depth
// value only (positive towards the viewer). A perspective factor *persp* puts
// the camera on the +z axis at distance 1/persp: a rotated point nearer than
// the cube centre scales up and pushes outward, a farther one scales down.
// *persp* 0 is exactly the old orthographic projection. No DOM, no canvas: the
// drawing slice owns turning the projection into pixels.

export interface Orbit {
  yaw: number;
  pitch: number;
}

/** An orbit plus a perspective strength; what the canvas actually projects with. */
export interface Camera extends Orbit {
  persp: number;
}

export const ORBIT_CENTER = 0.5;

/** Radians of rotation per dragged CSS pixel. */
export const ORBIT_RAD_PER_PX = 0.008;

/** Pitch is clamped just short of straight up/down. */
export const MAX_PITCH = Math.PI / 2 - 0.01;

/**
 * Default perspective strength for the 3-D view; the user can tune it live with
 * ``?persp=0.15`` (see ``readPerspective``).
 */
export const DEFAULT_PERSPECTIVE = 0.28;
/** Perspective is clamped to this ceiling so the near face cannot blow up. */
export const MAX_PERSPECTIVE = 0.6;

/**
 * Read a perspective strength from a query string's ``persp`` parameter: finite
 * and clamped to [0, MAX_PERSPECTIVE], else DEFAULT_PERSPECTIVE. ``search`` may
 * be ``"?persp=0.15"`` or ``"persp=0.15"``.
 */
export function readPerspective(search: string): number {
  const m = /(?:^|[?&])persp=([^&]+)/.exec(search);
  if (!m) return DEFAULT_PERSPECTIVE;
  const v = Number(m[1]);
  if (!Number.isFinite(v)) return DEFAULT_PERSPECTIVE;
  return Math.max(0, Math.min(MAX_PERSPECTIVE, v));
}

/**
 * Rotate (x,y,z) about the cube centre and apply perspective. Returns
 * [rx, ry, depth, scale]: rx,ry are absolute world coords (centre 0.5) in the
 * viewport frame; depth is the rotated z centred on 0, positive towards the
 * viewer; scale multiplies the centred offsets (1 when *persp* is 0, or at the
 * cube centre). z defaults to ORBIT_CENTER.
 */
export function orbitProject(
  o: Orbit,
  x: number,
  y: number,
  z?: number,
  persp = 0,
): [number, number, number, number] {
  const px = x - ORBIT_CENTER;
  const py = y - ORBIT_CENTER;
  const pz = (z ?? ORBIT_CENTER) - ORBIT_CENTER;
  const { yaw, pitch } = o;
  // Yaw about Y: +x right, +z toward the viewer.
  const x1 = px * Math.cos(yaw) + pz * Math.sin(yaw);
  const z1 = -px * Math.sin(yaw) + pz * Math.cos(yaw);
  // Pitch about X: +y up.
  const y2 = py * Math.cos(pitch) - z1 * Math.sin(pitch);
  const z2 = py * Math.sin(pitch) + z1 * Math.cos(pitch);
  // A point at rotated depth z2 is 1/persp from the camera; its centred offsets
  // scale by perspective distance. The floor keeps a depth-straddling point from
  // ever inverting.
  const scale = 1 / Math.max(0.2, 1 - persp * z2);
  return [ORBIT_CENTER + x1 * scale, ORBIT_CENTER + y2 * scale, z2, scale];
}

/**
 * Exact inverse of ``orbitProject`` for a *known rotated depth*: undo the
 * perspective scale at that depth, then pitch, then yaw. Round-trips
 * ``orbitProject(o, x, y, z, persp)`` → ``[x, y, z]`` for any orbit/point.
 */
export function orbitUnproject(
  o: Orbit,
  rx: number,
  ry: number,
  depth: number,
  persp = 0,
): [number, number, number] {
  const scale = 1 / Math.max(0.2, 1 - persp * depth);
  const x1 = (rx - ORBIT_CENTER) / scale;
  const y2 = (ry - ORBIT_CENTER) / scale;
  const { yaw, pitch } = o;
  // Undo pitch about X (+y up).
  const py = y2 * Math.cos(pitch) + depth * Math.sin(pitch);
  const z1 = -y2 * Math.sin(pitch) + depth * Math.cos(pitch);
  // Undo yaw about Y (+x right, +z toward the viewer).
  const px = x1 * Math.cos(yaw) - z1 * Math.sin(yaw);
  const pz = x1 * Math.sin(yaw) + z1 * Math.cos(yaw);
  return [px + ORBIT_CENTER, py + ORBIT_CENTER, pz + ORBIT_CENTER];
}

/**
 * The camera for a 2-D↔3-D transition: every parameter is the orbit scaled by
 * *blend* (0 = head-on orthographic, 1 = the full orbit + perspective). This is
 * what makes toggling a smooth camera move — the view tweens *blend* and the
 * projection fades between orthographic head-on and the user's orbit angle.
 */
export function blendCamera(o: Orbit, blend: number, persp: number): Camera {
  return { yaw: o.yaw * blend, pitch: o.pitch * blend, persp: persp * blend };
}

/** Wrap *a* to (-π, π]. */
function wrapPi(a: number): number {
  const TWO_PI = 2 * Math.PI;
  let r = a % TWO_PI;
  if (r <= -Math.PI) r += TWO_PI;
  else if (r > Math.PI) r -= TWO_PI;
  return r;
}

/**
 * Apply a mouse drag in place: yaw += dx*k; pitch += dy*k clamped to ±MAX_PITCH;
 * yaw wrapped to (-π, π].
 */
export function orbitDrag(o: Orbit, dxPx: number, dyPx: number): void {
  o.yaw = wrapPi(o.yaw + dxPx * ORBIT_RAD_PER_PX);
  o.pitch = Math.max(
    -MAX_PITCH,
    Math.min(MAX_PITCH, o.pitch + dyPx * ORBIT_RAD_PER_PX),
  );
}
