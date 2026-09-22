// ─── Glow overlay ──────────────────────────────────────────
//
// Every glow on the map — the search-field splats and the navigation-target
// halos — is one WebGL pass over a transparent canvas layered *under* the 2-D
// scatter, so the glow reads as light behind the dots and never washes them out.
// Each point is a screen-space quad whose fragment shader evaluates a
// radial falloff (a signed distance to the quad centre), so N glowing points cost
// one draw call instead of N `createRadialGradient` fills or a full-screen
// `ctx.filter` blur.
//
// Overlaps take the brightest coverage (`MAX`), not the sum, and the falloff is a
// gradual Gaussian, so a dense region reads as one soft zone rather than a pile of
// bright circles.
//
// The 2-D canvas stays the single source of truth for *where* things are; this
// module only turns points into light.

import {
  CORNERS,
  WEBGL_OPTS,
  bindAttrib,
  deviceSize,
  link,
} from "./webgl-layer";

/** One glow splat, in CSS pixels. */
export interface GlowPoint {
  x: number;
  y: number;
  /** Glow extent from the centre. */
  radius: number;
  /** Flat core as a fraction of `radius` (0 = pure falloff, 0.45 = disc + halo). */
  core: number;
  /** Linear RGB, each 0..1. */
  color: [number, number, number];
  /** Peak opacity 0..1. */
  alpha: number;
}

const FLOATS_PER_VERTEX = 10;
const VERTICES_PER_POINT = 6;

const VERT_SRC = `
attribute vec2 a_center;
attribute vec2 a_corner;
attribute float a_radius;
attribute float a_core;
attribute vec4 a_color;
uniform vec2 u_resolution;
varying vec2 v_local;
varying float v_radius;
varying float v_core;
varying vec4 v_color;
void main() {
  vec2 px = a_center + a_corner * a_radius;
  vec2 clip = px / u_resolution * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
  v_local = a_corner * a_radius;
  v_radius = a_radius;
  v_core = a_core;
  v_color = a_color;
}`;

const FRAG_SRC = `
precision mediump float;
varying vec2 v_local;
varying float v_radius;
varying float v_core;
varying vec4 v_color;
void main() {
  float d = length(v_local) / v_radius;
  // Flat core (if any), then a gradual Gaussian tail that reaches zero at the rim,
  // so nearby points merge into one soft zone instead of stacking as circles.
  float t = clamp((d - v_core) / (1.0 - v_core), 0.0, 1.0);
  float falloff = exp(-2.5 * t * t) * (1.0 - smoothstep(0.75, 1.0, t));
  float a = v_color.a * falloff;
  gl_FragColor = vec4(v_color.rgb * a, a);
}`;

/**
 * Interleave the points into the vertex buffer the shader reads.
 *
 * Pass *out* to fill a reusable buffer (it must be at least ``points.length``
 * vertices long); the returned view is the exact slice to upload.
 */
export function packGlowVertices(
  points: readonly GlowPoint[],
  out?: Float32Array,
): Float32Array {
  const need = points.length * VERTICES_PER_POINT * FLOATS_PER_VERTEX;
  const data = out ?? new Float32Array(need);
  let o = 0;
  for (const p of points) {
    for (const [cx, cy] of CORNERS) {
      data[o++] = p.x;
      data[o++] = p.y;
      data[o++] = cx;
      data[o++] = cy;
      data[o++] = p.radius;
      data[o++] = p.core;
      data[o++] = p.color[0];
      data[o++] = p.color[1];
      data[o++] = p.color[2];
      data[o++] = p.alpha;
    }
  }
  return data.length === need ? data : data.subarray(0, need);
}

/**
 * A transparent WebGL canvas layered under the scatter. `supported` is false when
 * WebGL or the shaders are unavailable; `render` is then a no-op and the map
 * simply loses its glow rather than falling back to the per-point 2-D path.
 */
export class GlowLayer {
  readonly supported: boolean;
  private readonly canvas: HTMLCanvasElement;
  private readonly reference: HTMLCanvasElement;
  private gl: WebGL2RenderingContext | null = null;
  private program: WebGLProgram | null = null;
  private buffer: WebGLBuffer | null = null;
  private resolutionLoc: WebGLUniformLocation | null = null;
  private width = 0;
  private height = 0;
  /** Reused vertex buffer, so a frame does not allocate one per draw. */
  private _vertexData = new Float32Array(0);

  constructor(host: HTMLElement, reference: HTMLCanvasElement) {
    this.reference = reference;
    this.canvas = document.createElement("canvas");
    this.canvas.setAttribute("aria-hidden", "true");
    this.canvas.style.position = "absolute";
    this.canvas.style.pointerEvents = "none";
    // Behind the scatter canvas (which sits at z-index 1): the glow lights the
    // map without dimming the dots drawn on top of it.
    this.canvas.style.zIndex = "0";
    host.appendChild(this.canvas);
    this.supported = this._init();
  }

  private _init(): boolean {
    // Premultiplied output so the browser composites the glow over the page
    // correctly. WebGL2 is required for the non-additive MAX blend below.
    const gl = this.canvas.getContext("webgl2", WEBGL_OPTS);
    if (!gl) return false;
    const program = link(gl, VERT_SRC, FRAG_SRC);
    if (!program) return false;
    const buffer = gl.createBuffer();
    if (!buffer) return false;
    this.resolutionLoc = gl.getUniformLocation(program, "u_resolution");
    this.gl = gl;
    this.program = program;
    this.buffer = buffer;
    gl.enable(gl.BLEND);
    // Non-additive: an overlap takes the brightest coverage, not the sum, so a
    // dense region reads as one soft zone instead of a pile of bright circles.
    gl.blendEquation(gl.MAX);
    gl.disable(gl.DEPTH_TEST);
    return true;
  }

  /** Match the overlay to the scatter canvas: position, CSS size, and DPR. */
  sync(): void {
    if (!this.supported) return;
    const { cssW, cssH, pw, ph } = deviceSize(this.reference);
    const style = this.canvas.style;
    style.left = this.reference.offsetLeft + "px";
    style.top = this.reference.offsetTop + "px";
    style.width = cssW + "px";
    style.height = cssH + "px";
    if (this.canvas.width !== pw || this.canvas.height !== ph) {
      this.canvas.width = pw;
      this.canvas.height = ph;
    }
    this.width = cssW;
    this.height = cssH;
    this.gl!.viewport(0, 0, pw, ph);
  }

  /** Clear the overlay and additively splat every point in one draw call. */
  render(points: readonly GlowPoint[]): void {
    const gl = this.gl;
    if (!gl || gl.isContextLost()) return;
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (!points.length) return;

    gl.useProgram(this.program);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    const need = points.length * VERTICES_PER_POINT * FLOATS_PER_VERTEX;
    if (this._vertexData.length < need) this._vertexData = new Float32Array(need);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      packGlowVertices(points, this._vertexData),
      gl.DYNAMIC_DRAW,
    );
    const stride = FLOATS_PER_VERTEX * Float32Array.BYTES_PER_ELEMENT;
    const program = this.program!;
    bindAttrib(gl, program, "a_center", 2, stride, 0);
    bindAttrib(gl, program, "a_corner", 2, stride, 2 * Float32Array.BYTES_PER_ELEMENT);
    bindAttrib(gl, program, "a_radius", 1, stride, 4 * Float32Array.BYTES_PER_ELEMENT);
    bindAttrib(gl, program, "a_core", 1, stride, 5 * Float32Array.BYTES_PER_ELEMENT);
    bindAttrib(gl, program, "a_color", 4, stride, 6 * Float32Array.BYTES_PER_ELEMENT);
    gl.uniform2f(this.resolutionLoc, this.width, this.height);
    gl.drawArrays(gl.TRIANGLES, 0, points.length * VERTICES_PER_POINT);
  }
}
