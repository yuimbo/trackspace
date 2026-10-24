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
// In 3-D each halo carries a depth and `render` takes occluders — the blobs in
// front of it. A pre-pass splats the occluders' solid kernels (MAX-blended) into
// a half-float depth target, and the glow pass hides any halo pixel a nearer or
// equal-depth occluder covers, so a halo sits behind its own blob yet shows over
// deeper ones. The caller blits the glow between the blobs and the crisp dots.
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
import { CORE, DEPTH_MASK } from "./point-layer";

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
  /**
   * Normalised depth, 1 = nearest the viewer (always 1 in 2-D). Only read when
   * `render` is given occluders; without them a halo is never occluded.
   */
  depth?: number;
}

/**
 * A solid disc that can hide a halo behind it: the same kernel the metaball
 * layer uses, so the occlusion footprint matches the blob on screen.
 */
export interface GlowOccluder {
  x: number;
  y: number;
  /** Kernel half-extent, in CSS pixels (same space as the glow points). */
  radius: number;
  /** Normalised depth, 1 = nearest the viewer. */
  depth: number;
}

/** How far *behind* an occluder a halo must sit before it can show. */
const DEPTH_EPS = 0.01;

const FLOATS_PER_VERTEX = 11;
const VERTICES_PER_POINT = 6;
const OCC_FLOATS_PER_VERTEX = 6;

const VERT_SRC = `#version 300 es
in vec2 a_center;
in vec2 a_corner;
in float a_radius;
in float a_core;
in vec4 a_color;
in float a_depth;
uniform vec2 u_resolution;
out vec2 v_local;
out float v_radius;
out float v_core;
out vec4 v_color;
out float v_depth;
void main() {
  vec2 px = a_center + a_corner * a_radius;
  vec2 clip = px / u_resolution * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
  v_local = a_corner * a_radius;
  v_radius = a_radius;
  v_core = a_core;
  v_color = a_color;
  v_depth = a_depth;
}`;

const FRAG_SRC = `#version 300 es
precision highp float;
precision highp sampler2D;
in vec2 v_local;
in float v_radius;
in float v_core;
in vec4 v_color;
in float v_depth;
uniform sampler2D u_front;
uniform float u_occlude;
out vec4 outColor;
void main() {
  float d = length(v_local) / v_radius;
  // Flat core (if any), then a gradual Gaussian tail that reaches zero at the rim,
  // so nearby points merge into one soft zone instead of stacking as circles.
  float t = clamp((d - v_core) / (1.0 - v_core), 0.0, 1.0);
  float falloff = exp(-2.5 * t * t) * (1.0 - smoothstep(0.75, 1.0, t));
  float a = v_color.a * falloff;
  // In 3-D a halo shows only where the front surface (from the occluder
  // pre-pass) is strictly behind it — or empty, which reads as depth 0.
  if (u_occlude > 0.5) {
    float front = texelFetch(u_front, ivec2(gl_FragCoord.xy), 0).r;
    a *= step(front, v_depth - ${DEPTH_EPS});
  }
  outColor = vec4(v_color.rgb * a, a);
}`;

// Occluder pre-pass: the same kernel as the metaball depth pre-pass, MAX-blended
// into a second target so each pixel holds the depth of the nearest solid
// occluder (0 where none is). The glow pass reads this to hide halos behind it.
const OCC_VERT_SRC = `#version 300 es
in vec2 a_center;
in vec2 a_corner;
in float a_radius;
in float a_depth;
uniform vec2 u_resolution;
out vec2 v_local;
out float v_radius;
out float v_depth;
void main() {
  vec2 px = a_center + a_corner * a_radius;
  vec2 clip = px / u_resolution * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
  v_local = a_corner * a_radius;
  v_radius = a_radius;
  v_depth = a_depth;
}`;

const OCC_FRAG_SRC = `#version 300 es
precision mediump float;
in vec2 v_local;
in float v_radius;
in float v_depth;
uniform float u_core;
out vec4 outColor;
void main() {
  float d = length(v_local) / v_radius;
  float w = 1.0 - smoothstep(u_core, 1.0, d);
  outColor = vec4(w > ${DEPTH_MASK} ? v_depth : 0.0, 0.0, 0.0, 0.0);
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
      data[o++] = p.depth ?? 1;
    }
  }
  return data.length === need ? data : data.subarray(0, need);
}

/** Interleave the occluders into the vertex buffer the pre-pass shader reads. */
function packOccluderVertices(
  occluders: readonly GlowOccluder[],
  out?: Float32Array,
): Float32Array {
  const need = occluders.length * VERTICES_PER_POINT * OCC_FLOATS_PER_VERTEX;
  const data = out ?? new Float32Array(need);
  let o = 0;
  for (const p of occluders) {
    for (const [cx, cy] of CORNERS) {
      data[o++] = p.x;
      data[o++] = p.y;
      data[o++] = cx;
      data[o++] = cy;
      data[o++] = p.radius;
      data[o++] = p.depth;
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
  /** The glow canvas; in 3-D the caller blits it into the scatter canvas. */
  readonly canvas: HTMLCanvasElement;
  private readonly reference: HTMLCanvasElement;
  private gl: WebGL2RenderingContext | null = null;
  private program: WebGLProgram | null = null;
  private buffer: WebGLBuffer | null = null;
  private resolutionLoc: WebGLUniformLocation | null = null;
  private frontLoc: WebGLUniformLocation | null = null;
  private occludeLoc: WebGLUniformLocation | null = null;
  private occProgram: WebGLProgram | null = null;
  private occBuffer: WebGLBuffer | null = null;
  private occResolutionLoc: WebGLUniformLocation | null = null;
  private occCoreLoc: WebGLUniformLocation | null = null;
  private depthTexture: WebGLTexture | null = null;
  private depthFramebuffer: WebGLFramebuffer | null = null;
  /** True once the depth target has storage and is a complete framebuffer. */
  private occlusionReady = false;
  /** True when the context can render into a half-float target. */
  private depthExt = false;
  private depthW = 0;
  private depthH = 0;
  private width = 0;
  private height = 0;
  /** Reused vertex buffer, so a frame does not allocate one per draw. */
  private _vertexData = new Float32Array(0);
  private _occVertexData = new Float32Array(0);

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
    this.frontLoc = gl.getUniformLocation(program, "u_front");
    this.occludeLoc = gl.getUniformLocation(program, "u_occlude");
    // Occlusion is optional: without a renderable half-float target the halos
    // simply draw un-occluded — the layer itself stays supported.
    this.depthExt =
      !!gl.getExtension("EXT_color_buffer_float") ||
      !!gl.getExtension("EXT_color_buffer_half_float");
    const occProgram = link(gl, OCC_VERT_SRC, OCC_FRAG_SRC);
    const occBuffer = gl.createBuffer();
    const depthTexture = gl.createTexture();
    const depthFramebuffer = gl.createFramebuffer();
    if (occProgram && occBuffer && depthTexture && depthFramebuffer) {
      this.occProgram = occProgram;
      this.occBuffer = occBuffer;
      this.depthTexture = depthTexture;
      this.depthFramebuffer = depthFramebuffer;
      this.occResolutionLoc = gl.getUniformLocation(occProgram, "u_resolution");
      this.occCoreLoc = gl.getUniformLocation(occProgram, "u_core");
      // NEAREST is enough — the glow pass only indexes the target per pixel.
      gl.bindTexture(gl.TEXTURE_2D, depthTexture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.bindTexture(gl.TEXTURE_2D, null);
    }
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
    this._syncDepthTarget(pw, ph);
  }

  /**
   * Give the occlusion target the canvas's device-pixel size. Skipped silently
   * when the context cannot render into a half-float target; occlusion then
   * stays off rather than marking the layer unsupported.
   */
  private _syncDepthTarget(pw: number, ph: number): void {
    const gl = this.gl;
    const tex = this.depthTexture;
    const fbo = this.depthFramebuffer;
    if (!gl || !this.depthExt || !tex || !fbo) return;
    if (this.occlusionReady && this.depthW === pw && this.depthH === ph) return;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA16F,
      pw,
      ph,
      0,
      gl.RGBA,
      gl.HALF_FLOAT,
      null,
    );
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      tex,
      0,
    );
    this.occlusionReady =
      gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    this.depthW = pw;
    this.depthH = ph;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /**
   * Clear the overlay and additively splat every point in one draw call. In 3-D
   * `occluders` (the blobs in front of the halos) first splat their depth into
   * the occlusion target, so each halo shows only where nothing at its own or a
   * nearer depth covers the pixel.
   */
  render(points: readonly GlowPoint[], occluders?: readonly GlowOccluder[]): void {
    const gl = this.gl;
    if (!gl || gl.isContextLost()) return;
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (!points.length) return;

    const occlude =
      occluders !== undefined && occluders.length > 0 && this.occlusionReady;
    if (occlude) this._splatOccluders(occluders!);

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
    bindAttrib(gl, program, "a_depth", 1, stride, 10 * Float32Array.BYTES_PER_ELEMENT);
    gl.uniform2f(this.resolutionLoc, this.width, this.height);
    // Always point `u_front` at the depth target on unit 1, even when occlusion
    // is off: WebGL validates sampler bindings regardless of the dynamic branch,
    // and the default unit 0 holds the canvas this pass renders into (a feedback
    // loop that drops the draw).
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.depthTexture);
    gl.uniform1i(this.frontLoc, 1);
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1f(this.occludeLoc, occlude ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, points.length * VERTICES_PER_POINT);
  }

  /** Splat the occluders' solid kernels' depth into the occlusion target with
   * MAX blending, so each pixel holds the depth of the nearest solid occluder
   * (0 where none is). Only called when the depth target is usable. */
  private _splatOccluders(occluders: readonly GlowOccluder[]): void {
    const gl = this.gl!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.depthFramebuffer);
    gl.viewport(0, 0, this.depthW, this.depthH);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendEquation(gl.MAX);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.useProgram(this.occProgram!);
    const need = occluders.length * VERTICES_PER_POINT * OCC_FLOATS_PER_VERTEX;
    if (this._occVertexData.length < need) {
      this._occVertexData = new Float32Array(need);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.occBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      packOccluderVertices(occluders, this._occVertexData),
      gl.DYNAMIC_DRAW,
    );
    const stride = OCC_FLOATS_PER_VERTEX * Float32Array.BYTES_PER_ELEMENT;
    const program = this.occProgram!;
    bindAttrib(gl, program, "a_center", 2, stride, 0);
    bindAttrib(gl, program, "a_corner", 2, stride, 2 * Float32Array.BYTES_PER_ELEMENT);
    bindAttrib(gl, program, "a_radius", 1, stride, 4 * Float32Array.BYTES_PER_ELEMENT);
    bindAttrib(gl, program, "a_depth", 1, stride, 5 * Float32Array.BYTES_PER_ELEMENT);
    gl.uniform2f(this.occResolutionLoc!, this.width, this.height);
    gl.uniform1f(this.occCoreLoc!, CORE);
    gl.drawArrays(gl.TRIANGLES, 0, occluders.length * VERTICES_PER_POINT);
    // Back to the glow pass's own target and equation.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.blendEquation(gl.MAX);
  }
}
