// ─── Shared WebGL quad-layer scaffolding ─────────────────────
//
// `MetaballLayer` and `GlowLayer` both draw one screen-space quad per point in a
// single draw call, so they share the same WebGL2 context setup, quad geometry,
// shader compile/link, DPR sync, and attribute binding. This module is that shared
// half: each layer keeps only its shaders, blending, and pass structure, so the two
// stay in step on the parts that are not actually different.

/** Two triangles covering the [-1, 1] quad, expanded per vertex. */
export const CORNERS: readonly [number, number][] = [
  [-1, -1],
  [1, -1],
  [1, 1],
  [-1, -1],
  [1, 1],
  [-1, 1],
];

/** Context attributes both layers want: premultiplied output, no depth/stencil. */
export const WEBGL_OPTS: WebGLContextAttributes = {
  alpha: true,
  premultipliedAlpha: true,
  antialias: false,
  depth: false,
  stencil: false,
};

/** CSS size and DPR-pixel size of *reference*, each floored at 1 px. */
export function deviceSize(reference: HTMLCanvasElement): {
  cssW: number;
  cssH: number;
  pw: number;
  ph: number;
} {
  const cssW = reference.clientWidth;
  const cssH = reference.clientHeight;
  const dpr = devicePixelRatio || 1;
  return {
    cssW,
    cssH,
    pw: Math.max(1, Math.round(cssW * dpr)),
    ph: Math.max(1, Math.round(cssH * dpr)),
  };
}

/** Compile one shader, or ``null`` when it does not compile. */
export function compile(
  gl: WebGL2RenderingContext,
  type: number,
  src: string,
): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

/** Link a vertex/fragment pair into a program, or ``null`` when linking fails. */
export function link(
  gl: WebGL2RenderingContext,
  vertSrc: string,
  fragSrc: string,
): WebGLProgram | null {
  const vert = compile(gl, gl.VERTEX_SHADER, vertSrc);
  const frag = compile(gl, gl.FRAGMENT_SHADER, fragSrc);
  if (!vert || !frag) return null;
  const program = gl.createProgram();
  if (!program) return null;
  gl.attachShader(program, vert);
  gl.attachShader(program, frag);
  gl.linkProgram(program);
  gl.deleteShader(vert);
  gl.deleteShader(frag);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    gl.deleteProgram(program);
    return null;
  }
  return program;
}

/** Bind *name* on *program* to one float attribute of the current buffer. */
export function bindAttrib(
  gl: WebGL2RenderingContext,
  program: WebGLProgram,
  name: string,
  size: number,
  stride: number,
  offset: number,
): void {
  const loc = gl.getAttribLocation(program, name);
  if (loc < 0) return;
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, offset);
}
