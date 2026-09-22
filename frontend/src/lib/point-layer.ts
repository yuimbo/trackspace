// ─── Metaball point layer ───────────────────────────────────
//
// Every scatter point is drawn as one gooey field instead of N hard 2-D discs plus a
// per-cell count badge. Two passes over a transparent WebGL canvas:
//
//   1. *Splat*: each point is a screen-space quad whose fragment shader deposits a
//      radial kernel (flat core, smooth falloff) additively into a half-float render
//      target. Overlapping kernels sum, so neighbours share a single iso-contour.
//   2. *Composite*: one full-screen quad thresholds that summed field with a
//      `smoothstep`, so a lone point reads as a crisp disc and a dense knot bulges
//      into one merged blob — the metaball read.
//
// Colour is the kernel-weighted average, so a blob of one hue stays pure and a
// mixed blob blends proportionally; dividing by the summed weight keeps the interior
// at full saturation however many points stack. Point alpha dims the colour (search
// fade, external tracks) without shrinking the geometry, so faded points still merge.
//
// The 2-D canvas stays the single source of truth for *where* things are: this
// module is handed the same screen positions the scatter would have drawn, and the
// caller blits the result between the grid and the overlays. Hit testing stays
// CPU-side on `CanvasView.positions`; merging never affects a click.

import { CORNERS, WEBGL_OPTS, deviceSize, link } from "./webgl-layer";

/** One splatted point, in CSS pixels. */
export interface PointSpec {
  x: number;
  y: number;
  /** Splat half-extent from the centre; a lone point's disc is ~0.62× this. */
  radius: number;
  /** Linear RGB, each 0..1. */
  color: [number, number, number];
  /** Point opacity 0..1: dims the blob's colour without changing its shape. */
  alpha: number;
}

/** Default splat half-extent, tuned so an isolated point is the old 4 px dot. */
export const POINT_SPLAT_R = 6.5;
/** Flat fraction of the splat before the falloff begins. */
const CORE = 0.5;
/** Field level that becomes the blob edge; below it, no point is drawn. */
const THRESHOLD = 0.85;
/**
 * Minimum half-width of the composite's edge ramp, in field units. The real ramp is
 * sized from the field's screen-space gradient (`fwidth`) so every edge is ~1 px
 * wide; this floor only keeps the smoothstep well-defined where the field is flat.
 */
const EDGE_FLOOR = 0.002;
/** Weak dark ring just outside each blob, in screen pixels and peak opacity. */
const OUTLINE_PX = 1.5;
const OUTLINE_ALPHA = 0.35;

const FLOATS_PER_VERTEX = 9;
const VERTICES_PER_POINT = 6;

const SPLAT_VERT_SRC = `#version 300 es
in vec2 a_center;
in vec2 a_corner;
in float a_radius;
in vec4 a_color;
uniform vec2 u_resolution;
out vec2 v_local;
out float v_radius;
out vec4 v_color;
void main() {
  vec2 px = a_center + a_corner * a_radius;
  vec2 clip = px / u_resolution * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
  v_local = a_corner * a_radius;
  v_radius = a_radius;
  v_color = a_color;
}`;

// Deposits `color * weight * alpha` into RGB and the raw geometry `weight` into
// A. Summing over every overlapping point builds the field the composite reads.
const SPLAT_FRAG_SRC = `#version 300 es
precision mediump float;
in vec2 v_local;
in float v_radius;
in vec4 v_color;
uniform float u_core;
out vec4 outColor;
void main() {
  float d = length(v_local) / v_radius;
  float w = 1.0 - smoothstep(u_core, 1.0, d);
  outColor = vec4(v_color.rgb * w * v_color.a, w);
}`;

const COMPOSITE_VERT_SRC = `#version 300 es
in vec2 a_corner;
out vec2 v_uv;
void main() {
  v_uv = a_corner * 0.5 + 0.5;
  gl_Position = vec4(a_corner, 0.0, 1.0);
}`;

const COMPOSITE_FRAG_SRC = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;
uniform sampler2D u_field;
uniform float u_threshold;
uniform float u_minSoft;
uniform float u_outlineWidth;
uniform float u_outlineAlpha;
void main() {
  vec4 f = texture(u_field, v_uv);
  float g = f.a;
  // Size the ramp from the field's own screen-space gradient, so the edge is ~1 px
  // wide however steep the field is — a fixed ramp hardens in dense knots.
  float px = fwidth(g);
  float w = max(px * 0.5, u_minSoft);
  float edge = smoothstep(u_threshold - w, u_threshold + w, g);
  // A weak dark ring sits just outside the blob edge, so blobs stay legible against
  // the grid and the glow. Its width tracks the same gradient, keeping it ~1.5 px.
  float ow = max(px * u_outlineWidth, u_minSoft);
  float ring = smoothstep(u_threshold - ow - w, u_threshold - ow + w, g) * (1.0 - edge);
  vec3 col = f.rgb / max(g, 1e-4);
  // Ring composites as black under the blob: premultiplied rgb is the blob colour only
  // where the blob covers, while alpha also carries the ring.
  float a = edge + ring * u_outlineAlpha * (1.0 - edge);
  outColor = vec4(col * edge, a);
}`;

/**
 * Interleave the points into the vertex buffer the splat shader reads.
 *
 * Pass *out* to fill a reusable buffer (it must be at least ``points.length``
 * vertices long); the returned view is the exact slice to upload.
 */
export function packPointVertices(
  points: readonly PointSpec[],
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
      data[o++] = p.color[0];
      data[o++] = p.color[1];
      data[o++] = p.color[2];
      data[o++] = p.alpha;
    }
  }
  return data.length === need ? data : data.subarray(0, need);
}

/**
 * A transparent WebGL canvas holding the merged scatter. `supported` is false when
 * WebGL2 or a renderable half-float target is unavailable; `render` is then a
 * no-op and the caller falls back to the plain 2-D dot path.
 */
export class MetaballLayer {
  readonly supported: boolean;
  /** Detached canvas; the caller blits it into the scatter canvas. */
  readonly canvas: HTMLCanvasElement;
  private gl: WebGL2RenderingContext | null = null;
  private splatProgram: WebGLProgram | null = null;
  private compositeProgram: WebGLProgram | null = null;
  private buffer: WebGLBuffer | null = null;
  private quad: WebGLBuffer | null = null;
  private texture: WebGLTexture | null = null;
  private framebuffer: WebGLFramebuffer | null = null;
  private splatResolutionLoc: WebGLUniformLocation | null = null;
  private splatCoreLoc: WebGLUniformLocation | null = null;
  private fieldLoc: WebGLUniformLocation | null = null;
  private thresholdLoc: WebGLUniformLocation | null = null;
  private minSoftLoc: WebGLUniformLocation | null = null;
  private outlineWidthLoc: WebGLUniformLocation | null = null;
  private outlineAlphaLoc: WebGLUniformLocation | null = null;
  private width = 0;
  private height = 0;
  /** True once the field target has storage and is a complete framebuffer. */
  private ready = false;
  /** Reused vertex buffer, so a frame does not allocate one per draw. */
  private _vertexData = new Float32Array(0);
  /** Resolved attribute slots, so a pass never re-reads the current program. */
  private splatAttribs: Record<string, number> = {};
  private compositeCornerLoc = -1;

  constructor() {
    this.canvas = document.createElement("canvas");
    this.supported = this._init();
  }

  private _init(): boolean {
    const gl = this.canvas.getContext("webgl2", WEBGL_OPTS);
    if (!gl) return false;
    // A renderable half-float target is what lets overlapping kernels sum past 1
    // without clipping — the growth the metaball read depends on.
    if (
      !gl.getExtension("EXT_color_buffer_float") &&
      !gl.getExtension("EXT_color_buffer_half_float")
    ) {
      return false;
    }
    const splat = link(gl, SPLAT_VERT_SRC, SPLAT_FRAG_SRC);
    const composite = link(gl, COMPOSITE_VERT_SRC, COMPOSITE_FRAG_SRC);
    if (!splat || !composite) return false;
    const buffer = gl.createBuffer();
    const quad = gl.createBuffer();
    const texture = gl.createTexture();
    const framebuffer = gl.createFramebuffer();
    if (!buffer || !quad || !texture || !framebuffer) return false;

    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array(CORNERS.flat()),
      gl.STATIC_DRAW,
    );
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      texture,
      0,
    );

    this.splatResolutionLoc = gl.getUniformLocation(splat, "u_resolution");
    this.splatCoreLoc = gl.getUniformLocation(splat, "u_core");
    this.fieldLoc = gl.getUniformLocation(composite, "u_field");
    this.thresholdLoc = gl.getUniformLocation(composite, "u_threshold");
    this.minSoftLoc = gl.getUniformLocation(composite, "u_minSoft");
    this.outlineWidthLoc = gl.getUniformLocation(composite, "u_outlineWidth");
    this.outlineAlphaLoc = gl.getUniformLocation(composite, "u_outlineAlpha");
    this.splatAttribs = {
      a_center: gl.getAttribLocation(splat, "a_center"),
      a_corner: gl.getAttribLocation(splat, "a_corner"),
      a_radius: gl.getAttribLocation(splat, "a_radius"),
      a_color: gl.getAttribLocation(splat, "a_color"),
    };
    this.compositeCornerLoc = gl.getAttribLocation(composite, "a_corner");
    this.gl = gl;
    this.splatProgram = splat;
    this.compositeProgram = composite;
    this.buffer = buffer;
    this.quad = quad;
    this.texture = texture;
    this.framebuffer = framebuffer;
    gl.disable(gl.DEPTH_TEST);
    return true;
  }

  /** Match the scatter canvas's CSS size and DPR, and resize the field target. */
  sync(reference: HTMLCanvasElement): void {
    if (!this.supported) return;
    const { cssW, cssH, pw, ph } = deviceSize(reference);
    this.width = cssW;
    this.height = cssH;
    if (this.ready && this.canvas.width === pw && this.canvas.height === ph) {
      return;
    }
    this.canvas.width = pw;
    this.canvas.height = ph;
    const gl = this.gl!;
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
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
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    this.ready =
      gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
  }

  /** Threshold every splatted point into one gooey layer on the canvas. */
  render(points: readonly PointSpec[]): void {
    const gl = this.gl;
    // A frame may draw before the first `sync`; the target has no storage yet, so
    // skip it rather than clear an incomplete framebuffer.
    if (!gl || !this.ready || gl.isContextLost()) return;
    const dpr = devicePixelRatio || 1;
    const pw = Math.max(1, Math.round(this.width * dpr));
    const ph = Math.max(1, Math.round(this.height * dpr));

    // Pass 1: accumulate every kernel into the field target.
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.viewport(0, 0, pw, ph);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.useProgram(this.splatProgram);
    this._disableAttribs();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    const need = points.length * VERTICES_PER_POINT * FLOATS_PER_VERTEX;
    if (this._vertexData.length < need) this._vertexData = new Float32Array(need);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      packPointVertices(points, this._vertexData),
      gl.DYNAMIC_DRAW,
    );
    const stride = FLOATS_PER_VERTEX * Float32Array.BYTES_PER_ELEMENT;
    this._pointAttrib("a_center", 2, stride, 0);
    this._pointAttrib("a_corner", 2, stride, 2 * Float32Array.BYTES_PER_ELEMENT);
    this._pointAttrib("a_radius", 1, stride, 4 * Float32Array.BYTES_PER_ELEMENT);
    this._pointAttrib("a_color", 4, stride, 5 * Float32Array.BYTES_PER_ELEMENT);
    gl.uniform2f(this.splatResolutionLoc, this.width, this.height);
    gl.uniform1f(this.splatCoreLoc, CORE);
    if (points.length) {
      gl.drawArrays(gl.TRIANGLES, 0, points.length * VERTICES_PER_POINT);
    }

    // Pass 2: threshold the field onto the canvas, premultiplied over.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, pw, ph);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(this.compositeProgram);
    this._disableAttribs();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.enableVertexAttribArray(this.compositeCornerLoc);
    gl.vertexAttribPointer(
      this.compositeCornerLoc,
      2,
      gl.FLOAT,
      false,
      2 * Float32Array.BYTES_PER_ELEMENT,
      0,
    );
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.uniform1i(this.fieldLoc, 0);
    gl.uniform1f(this.thresholdLoc, THRESHOLD);
    gl.uniform1f(this.minSoftLoc, EDGE_FLOOR);
    gl.uniform1f(this.outlineWidthLoc, OUTLINE_PX);
    gl.uniform1f(this.outlineAlphaLoc, OUTLINE_ALPHA);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  private _pointAttrib(
    name: string,
    size: number,
    stride: number,
    offset: number,
  ): void {
    const gl = this.gl!;
    const loc = this.splatAttribs[name];
    if (loc < 0) return;
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, offset);
  }

  /** Disable every array either program may have left enabled, so a draw with an
   * empty point buffer can never read past it. */
  private _disableAttribs(): void {
    const gl = this.gl!;
    for (const loc of Object.values(this.splatAttribs)) {
      if (loc >= 0) gl.disableVertexAttribArray(loc);
    }
    if (this.compositeCornerLoc >= 0) {
      gl.disableVertexAttribArray(this.compositeCornerLoc);
    }
  }
}
