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
// A down-prioritised (grey) point contributes almost none of its colour, and grey
// points are splatted and composited *before* the coloured ones, so a grey dot never
// occludes a coloured one in the merged field.
//
// In 3-D mode (`render(points, true)`) a depth pre-pass runs first: the same kernels
// are splatted with MAX blending into a second target, keeping per pixel the depth of
// the nearest solid point. The field pass then occludes any point deeper than a thin
// slab behind that front surface, so depth layers stay separate gooey blobs instead
// of blending into one another.
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
  /**
   * Colour priority 0..1, default 1: scales how much this point's colour counts
   * in a merged blob, **without** changing its geometry. A grey point (a track the
   * head cannot place, or an unanalysed one) is down-prioritised, so its greyness
   * cannot wash out the hue of the information-bearing points it merges with.
   */
  weight?: number;
  /**
   * Depth 0..1, default 0.5: 1 = nearest the viewer (orthographic). Only read when
   * `render` is called with `depthAware`; the 2-D path ignores it entirely.
   */
  depth?: number;
}

/** Default splat half-extent, tuned so an isolated point is the old 4 px dot. */
export const POINT_SPLAT_R = 6.5;
/** Flat fraction of the splat before the falloff begins. */
export const CORE = 0.5;
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
/** Kernel weight above which a point counts as solid for the front surface. */
export const DEPTH_MASK = 0.4;
/** Occlusion slab behind the front surface, in normalised depth units. */
const DEPTH_SLAB = 0.06;

const FLOATS_PER_VERTEX = 11;
const VERTICES_PER_POINT = 6;

// The six attributes carry fixed layout locations so the depth pre-pass program —
// linked from this same vertex shader — reuses the attribute slots resolved for the
// splat program without any extra lookup.
const SPLAT_VERT_SRC = `#version 300 es
layout(location = 0) in vec2 a_center;
layout(location = 1) in vec2 a_corner;
layout(location = 2) in float a_radius;
layout(location = 3) in vec4 a_color;
layout(location = 4) in float a_weight;
layout(location = 5) in float a_depth;
uniform vec2 u_resolution;
out vec2 v_local;
out float v_radius;
out vec4 v_color;
out float v_weight;
out float v_depth;
void main() {
  vec2 px = a_center + a_corner * a_radius;
  vec2 clip = px / u_resolution * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
  v_local = a_corner * a_radius;
  v_radius = a_radius;
  v_color = a_color;
  v_weight = a_weight;
  v_depth = a_depth;
}`;

// Deposits `color * weight * alpha * priority` into RGB and the raw geometry
// `weight` into A. Summing over every overlapping point builds the field the
// composite reads; the priority scales only the colour, so a down-prioritised grey
// point still adds shape without desaturating its neighbours. In 3-D mode the
// kernel is damped behind the front surface from the depth pre-pass (`u_front`):
// a point only contributes within a thin slab behind the nearest solid kernel at
// that pixel, so depth layers never blend into each other.
const SPLAT_FRAG_SRC = `#version 300 es
precision highp float;
precision highp sampler2D;
in vec2 v_local;
in float v_radius;
in vec4 v_color;
in float v_weight;
in float v_depth;
uniform float u_core;
uniform sampler2D u_front;
uniform float u_slab;
uniform float u_depthAware;
out vec4 outColor;
void main() {
  float d = length(v_local) / v_radius;
  float w = 1.0 - smoothstep(u_core, 1.0, d);
  if (u_depthAware > 0.5) {
    float front = texelFetch(u_front, ivec2(gl_FragCoord.xy), 0).r;
    w *= 1.0 - smoothstep(0.0, u_slab, front - v_depth);
  }
  outColor = vec4(v_color.rgb * w * v_color.a * v_weight, w);
}`;

// Front-depth pre-pass: the same kernel, MAX-blended into a second target so each
// pixel ends up holding the depth of the nearest point whose kernel is solid there
// (0 where none is). The field pass reads this to occlude what lies behind it.
const DEPTH_FRAG_SRC = `#version 300 es
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
      data[o++] = p.weight ?? 1;
      data[o++] = p.depth ?? 0.5;
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
  private depthProgram: WebGLProgram | null = null;
  private compositeProgram: WebGLProgram | null = null;
  private buffer: WebGLBuffer | null = null;
  private quad: WebGLBuffer | null = null;
  private texture: WebGLTexture | null = null;
  private framebuffer: WebGLFramebuffer | null = null;
  private depthTexture: WebGLTexture | null = null;
  private depthFramebuffer: WebGLFramebuffer | null = null;
  private splatResolutionLoc: WebGLUniformLocation | null = null;
  private splatCoreLoc: WebGLUniformLocation | null = null;
  private splatFrontLoc: WebGLUniformLocation | null = null;
  private splatSlabLoc: WebGLUniformLocation | null = null;
  private splatDepthAwareLoc: WebGLUniformLocation | null = null;
  private depthResolutionLoc: WebGLUniformLocation | null = null;
  private depthCoreLoc: WebGLUniformLocation | null = null;
  private fieldLoc: WebGLUniformLocation | null = null;
  private thresholdLoc: WebGLUniformLocation | null = null;
  private minSoftLoc: WebGLUniformLocation | null = null;
  private outlineWidthLoc: WebGLUniformLocation | null = null;
  private outlineAlphaLoc: WebGLUniformLocation | null = null;
  private width = 0;
  private height = 0;
  /** True once both targets have storage and are complete framebuffers. */
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
    const depth = link(gl, SPLAT_VERT_SRC, DEPTH_FRAG_SRC);
    const composite = link(gl, COMPOSITE_VERT_SRC, COMPOSITE_FRAG_SRC);
    if (!splat || !depth || !composite) return false;
    const buffer = gl.createBuffer();
    const quad = gl.createBuffer();
    const texture = gl.createTexture();
    const framebuffer = gl.createFramebuffer();
    const depthTexture = gl.createTexture();
    const depthFramebuffer = gl.createFramebuffer();
    if (
      !buffer ||
      !quad ||
      !texture ||
      !framebuffer ||
      !depthTexture ||
      !depthFramebuffer
    ) {
      return false;
    }

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
    // Depth pre-pass target: NEAREST is enough — the field pass only indexes it
    // per pixel with texelFetch.
    gl.bindTexture(gl.TEXTURE_2D, depthTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindFramebuffer(gl.FRAMEBUFFER, depthFramebuffer);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      depthTexture,
      0,
    );

    this.splatResolutionLoc = gl.getUniformLocation(splat, "u_resolution");
    this.splatCoreLoc = gl.getUniformLocation(splat, "u_core");
    this.splatFrontLoc = gl.getUniformLocation(splat, "u_front");
    this.splatSlabLoc = gl.getUniformLocation(splat, "u_slab");
    this.splatDepthAwareLoc = gl.getUniformLocation(splat, "u_depthAware");
    this.depthResolutionLoc = gl.getUniformLocation(depth, "u_resolution");
    this.depthCoreLoc = gl.getUniformLocation(depth, "u_core");
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
      a_weight: gl.getAttribLocation(splat, "a_weight"),
      a_depth: gl.getAttribLocation(splat, "a_depth"),
    };
    this.compositeCornerLoc = gl.getAttribLocation(composite, "a_corner");
    this.gl = gl;
    this.splatProgram = splat;
    this.depthProgram = depth;
    this.compositeProgram = composite;
    this.buffer = buffer;
    this.quad = quad;
    this.texture = texture;
    this.framebuffer = framebuffer;
    this.depthTexture = depthTexture;
    this.depthFramebuffer = depthFramebuffer;
    gl.disable(gl.DEPTH_TEST);
    return true;
  }

  /** Match the scatter canvas's CSS size and DPR, and resize the render targets. */
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
    gl.bindTexture(gl.TEXTURE_2D, this.depthTexture);
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
    const fieldOk =
      gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.depthFramebuffer);
    this.ready =
      fieldOk &&
      gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
  }

  /**
   * Threshold the splatted points into one gooey layer on the canvas.
   *
   * Down-prioritised (grey) points are splatted and composited **first**, then
   * the coloured points over them, so a grey dot never occludes a coloured one in
   * the merged field. The first pass clears the canvas; the rest blend over it.
   *
   * With `depthAware`, the grey/colour split is ignored — every point is splatted
   * in one pass and depth alone decides ordering, via a front-depth pre-pass that
   * occludes points behind the nearest solid kernel at each pixel.
   */
  render(points: readonly PointSpec[], depthAware = false): void {
    const gl = this.gl;
    // A frame may draw before the first `sync`; the target has no storage yet, so
    // skip it rather than clear an incomplete framebuffer.
    if (!gl || !this.ready || gl.isContextLost()) return;
    if (depthAware) {
      if (!points.length) {
        this._clearFrame();
        return;
      }
      this._pass(points, true, true);
      return;
    }
    const grey: PointSpec[] = [];
    const colour: PointSpec[] = [];
    for (const p of points) ((p.weight ?? 1) < 1 ? grey : colour).push(p);
    if (!grey.length && !colour.length) {
      this._clearFrame();
      return;
    }
    const first = grey.length ? grey : colour;
    const second = grey.length ? colour : null;
    this._pass(first, true, false);
    if (second) this._pass(second, false, false);
  }

  /** Clear the visible canvas so a frame with no splats is not stale. */
  private _clearFrame(): void {
    const gl = this.gl!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }

  /**
   * Splat *points* into the field target, then composite it onto the canvas. With
   * `depthAware` a front-depth pre-pass runs first and the field pass occludes
   * points behind that front surface.
   */
  private _pass(
    points: readonly PointSpec[],
    clearCanvas: boolean,
    depthAware = false,
  ): void {
    const gl = this.gl!;
    const dpr = devicePixelRatio || 1;
    const pw = Math.max(1, Math.round(this.width * dpr));
    const ph = Math.max(1, Math.round(this.height * dpr));
    const count = points.length * VERTICES_PER_POINT;
    const stride = FLOATS_PER_VERTEX * Float32Array.BYTES_PER_ELEMENT;
    const need = count * FLOATS_PER_VERTEX;
    if (this._vertexData.length < need) this._vertexData = new Float32Array(need);

    // Upload *points* and bind the shared point attributes. Both the splat and the
    // depth program use `SPLAT_VERT_SRC`, whose layout qualifiers give every
    // attribute the same location in each program.
    const bindPoints = (): void => {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
      gl.bufferData(
        gl.ARRAY_BUFFER,
        packPointVertices(points, this._vertexData),
        gl.DYNAMIC_DRAW,
      );
      this._pointAttrib("a_center", 2, stride, 0);
      this._pointAttrib("a_corner", 2, stride, 2 * Float32Array.BYTES_PER_ELEMENT);
      this._pointAttrib("a_radius", 1, stride, 4 * Float32Array.BYTES_PER_ELEMENT);
      this._pointAttrib("a_color", 4, stride, 5 * Float32Array.BYTES_PER_ELEMENT);
      this._pointAttrib("a_weight", 1, stride, 9 * Float32Array.BYTES_PER_ELEMENT);
      this._pointAttrib("a_depth", 1, stride, 10 * Float32Array.BYTES_PER_ELEMENT);
    };

    if (depthAware) {
      // Pass 0: MAX-blend the solid kernels' depth into the depth target, so each
      // pixel holds the depth of the nearest solid point (0 where none is).
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.depthFramebuffer);
      gl.viewport(0, 0, pw, ph);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.enable(gl.BLEND);
      gl.blendEquation(gl.MAX);
      gl.blendFunc(gl.ONE, gl.ONE);
      gl.useProgram(this.depthProgram);
      this._disableAttribs();
      bindPoints();
      gl.uniform2f(this.depthResolutionLoc, this.width, this.height);
      gl.uniform1f(this.depthCoreLoc, CORE);
      gl.drawArrays(gl.TRIANGLES, 0, count);
      // The 2-D path never touches the blend equation, so leave it additive.
      gl.blendEquation(gl.FUNC_ADD);
    }

    // Pass A: accumulate every kernel into the field target.
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.viewport(0, 0, pw, ph);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.useProgram(this.splatProgram);
    this._disableAttribs();
    bindPoints();
    gl.uniform2f(this.splatResolutionLoc, this.width, this.height);
    gl.uniform1f(this.splatCoreLoc, CORE);
    gl.uniform1f(this.splatSlabLoc, DEPTH_SLAB);
    gl.uniform1f(this.splatDepthAwareLoc, depthAware ? 1 : 0);
    // Always point `u_front` at the depth target on unit 1: WebGL validates sampler
    // bindings even when the shader branch is skipped, and the default unit 0 holds the
    // field texture this pass is rendering into (a feedback loop that drops the draw).
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.depthTexture);
    gl.uniform1i(this.splatFrontLoc, 1);
    gl.activeTexture(gl.TEXTURE0);
    gl.drawArrays(gl.TRIANGLES, 0, count);

    // Pass B: threshold the field onto the canvas, premultiplied over.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, pw, ph);
    if (clearCanvas) gl.clear(gl.COLOR_BUFFER_BIT);
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
    // A missing attribute resolves to `undefined`, which would otherwise be
    // coerced to attribute 0 and corrupt every point's centre.
    if (!Number.isInteger(loc) || loc < 0) return;
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
