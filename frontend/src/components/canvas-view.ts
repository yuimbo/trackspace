import type { HoverTarget, Model, Track } from "../model";
import { isAudioFeatureAxisId } from "../model";
import { clusterColor, dirColor } from "../lib/colors";
import { pipelineProgress, phaseLabel, type PipelineProgress } from "../lib/pipeline";
import { buildTooltip, renderTooltip, type InspectState } from "../lib/tooltip";


// ─── Types ───────────────────────────────────────────────────
export interface DotPosition {
  idx: number;
  sx: number;
  sy: number;
}

export interface TxHandle {
  type: string;
  sx: number;
  sy: number;
  cursor: string;
}

export interface BoxSel {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

interface TxBox {
  sl: number;
  sr: number;
  st: number;
  sb: number;
}

interface TxWorld {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

// ─── Constants ───────────────────────────────────────────────
const PAD = 40;
const DOT_R = 4;
const DOT_R_SEL = 6;
const HIT_RADIUS = 10;
const CLUSTER_CELL = 16;
const TX_PAD = 14;
const TX_KNOB_HIT = 9;

/** Halo colour for a key/BPM hover target (folders/clusters use their own colours). */
const TARGET_GLOW_COLOR = "#8e9bff";

export class CanvasView {
  model: Model;
  $canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  $tip: HTMLElement;

  positions: DotPosition[] = [];
  hoveredIdx = -1;
  /**
   * Hovered navigation element — a folder row, cluster row, or card chip (key/BPM).
   * Owned by whichever list renders it (sidebar tree, cluster panel, inspect card);
   * the canvas only reads it to glow matching dots. Purely visual, so it lives here
   * rather than on the model (see `ui_conventions.md` → *Adding new interaction surfaces*).
   */
  hoverTarget: HoverTarget | null = null;
  /** Exact folder path (``""`` = library root): glow siblings while hovering a dot. */
  hoveredSiblingFolder: string | null = null;
  tipReady = false;
  tipTimer = 0;
  /** Identity of the rendered tooltip card, so pan/zoom frames do not rebuild it. */
  private _tipKey = "";
  /**
   * Pinned inspect card state (Alt over a track). ``null`` = the plain hover
   * tooltip; while set, the card is interactive and ignores hover changes.
   */
  inspect: InspectState | null = null;
  /** Anchor (page coords) captured when the card was pinned. */
  private _pinPos: { left: number; top: number } | null = null;
  /** Result row currently auditioning on hover (dedupes repeated mouseover). */
  private _previewRow: HTMLElement | null = null;
  /** Inspect-card action clicked (``anti``/``near``/``proximity``/``focus``/``close``). */
  onInspectAction: ((action: string, path: string, value?: number) => void) | null =
    null;
  lassoPoints: [number, number][] = [];
  boxSel: BoxSel | null = null;
  /** True while a lasso/box drag is in intersect (AND) mode — shown in cyan. */
  intersectMode = false;
  dragGhosts: Map<string, { wx: number; wy: number; folder: string }> | null = null;
  _rafId = 0;

  txHandles: TxHandle[] = [];
  _txBox: TxBox | null = null;
  _txWorld: TxWorld | null = null;
  /** False when both axes are analysis features — box interior is not a drag target. */
  _txAllowBoxDrag = false;

  /** Transient per-track positions used during animated mode transitions. */
  animPositions: Map<string, { x: number; y: number }> | null = null;

  /** 0→1 eases when explode (hold X) is toggled so offsets interpolate smoothly. */
  private _explodeBlend = 0;

  // Offscreen canvas used as render texture for the glow screen-pass
  private _glowCanvas: HTMLCanvasElement | null = null;
  private _glowCtx: CanvasRenderingContext2D | null = null;

  constructor(model: Model) {
    this.model = model;
    this.$canvas = document.getElementById("viewport") as HTMLCanvasElement;
    this.ctx = this.$canvas.getContext("2d")!;
    this.$tip = document.getElementById("tooltip")!;
    this.$tip.addEventListener("click", (e) => this._onTipClick(e));
    this.$tip.addEventListener("input", (e) => this._onTipInput(e));
    this.$tip.addEventListener("mouseover", (e) => this._onTipOver(e));
    this.$tip.addEventListener("mouseout", (e) => this._onTipOut(e));
    // Leaving the open card ends the audition it owns (the card is
    // `pointer-events: none` until pinned, so this only fires while open).
    this.$tip.addEventListener("mouseleave", () =>
      this.onInspectAction?.("leave", this.inspect?.path ?? ""),
    );
  }

  /* ── Coordinate transforms ──────────────────────────────── */

  w2s(wx: number, wy: number): [number, number] {
    const w = this.$canvas.clientWidth,
      h = this.$canvas.clientHeight;
    const iw = Math.max(1e-6, w - 2 * PAD),
      ih = Math.max(1e-6, h - 2 * PAD),
      vp = this.model.vp;
    return [
      PAD + (wx - vp.ox) * vp.zoom * iw,
      h - PAD - (wy - vp.oy) * vp.zoom * ih,
    ];
  }

  s2w(sx: number, sy: number): [number, number] {
    const w = this.$canvas.clientWidth,
      h = this.$canvas.clientHeight;
    const iw = Math.max(1e-6, w - 2 * PAD),
      ih = Math.max(1e-6, h - 2 * PAD),
      vp = this.model.vp;
    return [
      (sx - PAD) / (vp.zoom * iw) + vp.ox,
      (h - PAD - sy) / (vp.zoom * ih) + vp.oy,
    ];
  }

  /** Minimum zoom: let the user zoom out until the longest axis drives the limit. */
  minZoom(): number {
    const iw = Math.max(1, this.$canvas.clientWidth - 2 * PAD);
    const ih = Math.max(1, this.$canvas.clientHeight - 2 * PAD);
    return Math.min(iw, ih) / Math.max(iw, ih);
  }

  /* ── Sizing ─────────────────────────────────────────────── */

  resize(): void {
    const dpr = devicePixelRatio || 1;
    this.$canvas.width = this.$canvas.clientWidth * dpr;
    this.$canvas.height = this.$canvas.clientHeight * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  scheduleDraw(): void {
    cancelAnimationFrame(this._rafId);
    this._rafId = requestAnimationFrame(() => this.draw());
  }

  /* ── Viewport pan clamping ──────────────────────────────── */

  private _clampVP(): void {
    const vp = this.model.vp;
    const w = this.$canvas.clientWidth,
      h = this.$canvas.clientHeight;
    const iw = Math.max(1e-6, w - 2 * PAD),
      ih = Math.max(1e-6, h - 2 * PAD);
    if (w <= 2 * PAD || h <= 2 * PAD) return;
    const cxOff = (w / 2 - PAD) / (vp.zoom * iw);
    const cyOff = (h / 2 - PAD) / (vp.zoom * ih);
    vp.ox = Math.max(-cxOff, Math.min(1 - cxOff, vp.ox));
    vp.oy = Math.max(-cyOff, Math.min(1 - cyOff, vp.oy));
  }

  /* ── Hit testing ────────────────────────────────────────── */

  hitTest(sx: number, sy: number): number {
    let best = -1,
      bestD = HIT_RADIUS;
    for (const p of this.positions) {
      const d = Math.hypot(p.sx - sx, p.sy - sy);
      if (d < bestD) {
        bestD = d;
        best = p.idx;
      }
    }
    return best;
  }

  hitTestTransform(sx: number, sy: number): TxHandle | { type: "move"; cursor: "move" } | null {
    for (const h of this.txHandles) {
      if (
        Math.abs(sx - h.sx) < TX_KNOB_HIT &&
        Math.abs(sy - h.sy) < TX_KNOB_HIT
      )
        return h;
    }
    const b = this._txBox;
    if (
      this._txAllowBoxDrag &&
      b &&
      sx > b.sl &&
      sx < b.sr &&
      sy > b.st &&
      sy < b.sb
    )
      return { type: "move", cursor: "move" };
    return null;
  }

  /* ── Main draw ──────────────────────────────────────────── */

  draw(): void {
    this._clampVP();
    this._updateExplodeBlend();
    const m = this.model;
    const tracks = m.tracks;
    const w = this.$canvas.clientWidth,
      h = this.$canvas.clientHeight;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, w, h);
    this.positions = [];
    this.txHandles = [];
    this._txBox = null;
    this._txWorld = null;
    this._txAllowBoxDrag = false;

    if (tracks.length === 0) {
      ctx.fillStyle = "#444";
      ctx.font = "14px monospace";
      ctx.textAlign = "center";
      ctx.fillText("No tracks loaded", w / 2, h / 2);
      ctx.textAlign = "start";
      this._drawLasso();
      if (this._explodeBlendAnimating()) this.scheduleDraw();
      return;
    }

    this._drawGrid(w, h);
    this._drawDragGhosts();
    this._drawScatter(tracks);
    this._drawFolderGlow(tracks);
    this._drawTransformBox();
    this._drawClusters();
    this._drawBoxSel();
    this._drawLasso();
    this._drawModeBadge(w, h);
    this._updateTooltip(tracks);

    if (this._explodeBlendAnimating()) this.scheduleDraw();
  }

  private _explodeBlendAnimating(): boolean {
    const m = this.model;
    const target =
      m.explodeHold && !m.suppressExplodeVisual && m.explodeStrength > 1e-6
        ? 1
        : 0;
    return Math.abs(this._explodeBlend - target) > 0.003;
  }

  private _updateExplodeBlend(): void {
    const m = this.model;
    const target =
      m.explodeHold && !m.suppressExplodeVisual && m.explodeStrength > 1e-6
        ? 1
        : 0;
    const k = 0.2;
    this._explodeBlend += (target - this._explodeBlend) * k;
    if (this._explodeBlend < 0.002) this._explodeBlend = 0;
    else if (this._explodeBlend > 0.998) this._explodeBlend = 1;
  }

  /* ── Grid + tick labels (adaptive to zoom) ──────────────── */

  private _drawGrid(w: number, h: number): void {
    const ctx = this.ctx;
    const m = this.model;

    const [wxMin] = this.s2w(0, 0);
    const [wxMax] = this.s2w(w, 0);
    const [, wyMin] = this.s2w(0, h);
    const [, wyMax] = this.s2w(0, 0);

    const [sLeft] = this.w2s(0, 0);
    const [sRight] = this.w2s(1, 0);
    const [, sBottom] = this.w2s(0, 0);
    const [, sTop] = this.w2s(0, 1);

    ctx.save();

    // Dark overlay outside [0,1] bounds
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    if (sLeft > 0) ctx.fillRect(0, 0, sLeft, h);
    if (sRight < w) ctx.fillRect(sRight, 0, w - sRight, h);
    if (sBottom < h)
      ctx.fillRect(sLeft, sBottom, sRight - sLeft, h - sBottom);
    if (sTop > 0) ctx.fillRect(sLeft, 0, sRight - sLeft, sTop);

    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.lineWidth = 1;
    ctx.strokeRect(sLeft, sTop, sRight - sLeft, sBottom - sTop);

    this._drawAxisBackgroundLabels(w, h, sLeft, sRight, sTop, sBottom);

    const pickStep = (worldRange: number, pxRange: number): number => {
      const ideal = (worldRange / pxRange) * 60;
      const steps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1];
      for (const s of steps) if (s >= ideal) return s;
      return 1;
    };

    const stepX = pickStep(wxMax - wxMin, w);
    const stepY = pickStep(wyMax - wyMin, h);
    const startX = Math.floor(wxMin / stepX) * stepX;
    const startY = Math.floor(wyMin / stepY) * stepY;

    const EPS = 1e-9;
    const inRange = (v: number) => v > -EPS && v < 1 + EPS;

    // Adaptive grid lines
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.lineWidth = 1;
    for (let v = startX; v <= wxMax + stepX * 0.5; v += stepX) {
      if (!inRange(v)) continue;
      const [sx] = this.w2s(v, 0);
      ctx.beginPath();
      ctx.moveTo(sx, sTop);
      ctx.lineTo(sx, sBottom);
      ctx.stroke();
    }
    for (let v = startY; v <= wyMax + stepY * 0.5; v += stepY) {
      if (!inRange(v)) continue;
      const [, sy] = this.w2s(0, v);
      ctx.beginPath();
      ctx.moveTo(sLeft, sy);
      ctx.lineTo(sRight, sy);
      ctx.stroke();
    }

    // 0.1-step reference lines
    ctx.strokeStyle = "rgba(255,255,255,0.10)";
    ctx.lineWidth = 1.5;
    const REF = 0.1;
    const startXRef = Math.floor(Math.max(wxMin, 0) / REF) * REF;
    const startYRef = Math.floor(Math.max(wyMin, 0) / REF) * REF;
    for (let v = startXRef; v <= Math.min(wxMax, 1) + REF * 0.5; v += REF) {
      if (!inRange(v)) continue;
      const [sx] = this.w2s(v, 0);
      ctx.beginPath();
      ctx.moveTo(sx, sTop);
      ctx.lineTo(sx, sBottom);
      ctx.stroke();
    }
    for (let v = startYRef; v <= Math.min(wyMax, 1) + REF * 0.5; v += REF) {
      if (!inRange(v)) continue;
      const [, sy] = this.w2s(0, v);
      ctx.beginPath();
      ctx.moveTo(sLeft, sy);
      ctx.lineTo(sRight, sy);
      ctx.stroke();
    }

    // 0.5 origin lines
    ctx.strokeStyle = "rgba(255,255,255,0.22)";
    ctx.lineWidth = 1.5;
    const [sx05] = this.w2s(0.5, 0);
    ctx.beginPath();
    ctx.moveTo(sx05, sTop);
    ctx.lineTo(sx05, sBottom);
    ctx.stroke();
    const [, sy05] = this.w2s(0, 0.5);
    ctx.beginPath();
    ctx.moveTo(sLeft, sy05);
    ctx.lineTo(sRight, sy05);
    ctx.stroke();

    // Tick labels along bottom edge
    ctx.fillStyle = "rgba(255,255,255,0.3)";
    ctx.font = "9px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let v = startX; v <= wxMax + stepX * 0.5; v += stepX) {
      if (!inRange(v)) continue;
      const [sx] = this.w2s(v, 0);
      if (sx < 4 || sx > w - 4) continue;
      ctx.fillText(v.toFixed(stepX < 0.1 ? 2 : 1), sx, h - 14);
    }

    // Tick labels along left edge
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (let v = startY; v <= wyMax + stepY * 0.5; v += stepY) {
      if (!inRange(v)) continue;
      const [, sy] = this.w2s(0, v);
      if (sy < 4 || sy > h - 4) continue;
      ctx.fillText(v.toFixed(stepY < 0.1 ? 2 : 1), 32, sy);
    }

    // Embedding mode caption (tag axes use large background labels only)
    ctx.fillStyle = "rgba(255,255,255,0.25)";
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    if (m.viewMode === "embeddings") {
      ctx.fillText(m.lens().label, w / 2, h - 2);
    }

    ctx.restore();
  }

  /**
   * Large watermark-style axis titles at the plot edges (tag mode only).
   * Drawn behind grid lines and points so they read as canvas background.
   */
  private _drawAxisBackgroundLabels(
    w: number,
    h: number,
    sLeft: number,
    sRight: number,
    sTop: number,
    sBottom: number,
  ): void {
    const m = this.model;
    if (m.viewMode === "embeddings") return;

    const ctx = this.ctx;
    ctx.save();
    ctx.fillStyle = "rgba(255,255,255,0.13)";
    ctx.font = "600 28px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textBaseline = "middle";

    if (m.axisX) {
      const cx = (sLeft + sRight) / 2;
      const tickRowTop = h - 14;
      const band = tickRowTop - sBottom;
      const cy =
        band > 6
          ? sBottom + band * 0.48
          : Math.min(sBottom + 18, h - 22);
      ctx.textAlign = "center";
      ctx.fillText(m.axisDisplayName(m.axisX), cx, cy);
    }

    if (m.axisY) {
      const cy = (sTop + sBottom) / 2;
      const idealX = sLeft - 22;
      const cx =
        sLeft > w - 20
          ? 24
          : Math.max(24, Math.min(idealX, w - 24));
      ctx.textAlign = "center";
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(m.axisDisplayName(m.axisY), 0, 0);
      ctx.restore();
    }

    ctx.restore();
  }

  /* ── Scatter plot (filter-aware) ────────────────────────── */

  private _tagPos(t: Track): { wx: number; wy: number } {
    const m = this.model;
    const tx = m.axisX,
      ty = m.axisY;
    return {
      wx: tx ? m.axisScalar(t, tx) : 0.5,
      wy: ty ? m.axisScalar(t, ty) : 0.5,
    };
  }

  /** World (normalised) position for fit / tooling; not clamped. */
  worldPosForTrack(t: Track): { wx: number; wy: number } | null {
    return this._resolveTrackWPos(t);
  }

  private _resolveTrackWPos(t: Track): { wx: number; wy: number } | null {
    const m = this.model;

    // During animation, interpolated positions always take priority
    if (this.animPositions) {
      const ap = this.animPositions.get(t.path);
      if (ap) return this._maybeExplode(t, ap.x, ap.y);
    }

    if (m.viewMode === "embeddings") {
      const ep = m.embeddingPositions.get(t.path);
      if (ep) return this._maybeExplode(t, ep.x, ep.y);
      // No embedding → hide the track (don't fall back to tag positions)
      return null;
    }

    const tag = this._tagPos(t);
    return this._maybeExplode(t, tag.wx, tag.wy);
  }

  /**
   * Hold-X explode preview: direction from tag axes (embedding mode) or toward
   * embedding (tag mode). Magnitude is in **screen pixels** (fraction of plot size
   * from strength 5–20%), then mapped back to world so it scales correctly with zoom.
   */
  private _maybeExplode(t: Track, wx: number, wy: number): { wx: number; wy: number } {
    const m = this.model;
    if (m.suppressExplodeVisual) return { wx, wy };

    const effective = m.explodeStrength * this._explodeBlend;
    if (effective <= 1e-7) return { wx, wy };

    let dwx = 0,
      dwy = 0;
    if (m.viewMode === "embeddings") {
      const tag = this._tagPos(t);
      dwx = (tag.wx - 0.5) * 2;
      dwy = (tag.wy - 0.5) * 2;
    } else {
      const ep = m.embeddingPositions.get(t.path);
      if (!ep) return { wx, wy };
      dwx = ep.x - wx;
      dwy = ep.y - wy;
    }

    const wlen = Math.hypot(dwx, dwy);
    if (wlen < 1e-9) return { wx, wy };
    dwx /= wlen;
    dwy /= wlen;

    const cw = this.$canvas.clientWidth,
      ch = this.$canvas.clientHeight;
    const iw = Math.max(1e-6, cw - 2 * PAD),
      ih = Math.max(1e-6, ch - 2 * PAD);
    const magPx = effective * Math.min(iw, ih);

    const [sx0, sy0] = this.w2s(wx, wy);
    const worldEps = 1e-4;
    const [sx1, sy1] = this.w2s(wx + dwx * worldEps, wy + dwy * worldEps);
    let px = sx1 - sx0,
      py = sy1 - sy0;
    const plen = Math.hypot(px, py);
    if (plen < 1e-9) return { wx, wy };
    px /= plen;
    py /= plen;

    const sx2 = sx0 + px * magPx;
    const sy2 = sy0 + py * magPx;
    const [wx2, wy2] = this.s2w(sx2, sy2);
    return { wx: wx2, wy: wy2 };
  }

  private _drawScatter(tracks: Track[]): void {
    const m = this.model;
    let anyFiltered = false;
    tracks.forEach((t, i) => {
      if (m.viewMode === "tags" && !m.passesFilter(t)) {
        anyFiltered = true;
        return;
      }
      const pos = this._resolveTrackWPos(t);
      if (!pos) return;
      const [sx, sy] = this.w2s(pos.wx, pos.wy);
      this.positions.push({ idx: i, sx, sy });
      this._dot(sx, sy, t, i);
    });
    if (anyFiltered) {
      const ctx = this.ctx;
      const w = this.$canvas.clientWidth,
        h = this.$canvas.clientHeight;
      ctx.save();
      ctx.fillStyle = "rgba(255,170,50,0.55)";
      ctx.font = "10px monospace";
      ctx.textAlign = "right";
      ctx.fillText("filtered", w - 6, h - 6);
      ctx.restore();
    }
  }

  /* ── Hover glow pass (navigation targets + sibling folder) ── */

  /** Tracks that get the navigation-target halo: sidebar/card hover, or dot sibling folder. */
  private _inHoverGlow(track: Track): boolean {
    if (this.hoverTarget !== null)
      return this.model.trackMatchesHoverTarget(track, this.hoverTarget);
    const sib = this.hoveredSiblingFolder;
    if (sib === null) return false;
    return (track.folder ?? "") === sib;
  }

  /** Halo colour: cluster swatch, folder colour, or the neutral accent for key/BPM. */
  private _hoverGlowColor(track: Track): string {
    const t = this.hoverTarget;
    if (t?.kind === "cluster") return clusterColor(t.id);
    if (t?.kind === "key" || t?.kind === "bpm" || t?.kind === "track")
      return TARGET_GLOW_COLOR;
    return this.trackColor(track);
  }

  /**
   * Lazily creates / resizes the offscreen glow render-texture so it always
   * matches the main canvas in physical pixels and shares the same DPR
   * coordinate transform.
   */
  private _ensureGlowCanvas(w: number, h: number): void {
    const dpr = devicePixelRatio || 1;
    const pw = Math.round(w * dpr);
    const ph = Math.round(h * dpr);
    if (!this._glowCanvas) {
      this._glowCanvas = document.createElement("canvas");
      this._glowCtx = this._glowCanvas.getContext("2d")!;
    }
    if (this._glowCanvas.width !== pw || this._glowCanvas.height !== ph) {
      this._glowCanvas.width = pw;
      this._glowCanvas.height = ph;
      // Setting canvas dimensions resets the context transform
      this._glowCtx!.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
  }

  /**
   * Renders the hover glow (navigation-target matches, or a hovered dot's sibling
   * folder) as a single screen-pass blur instead of N individual
   * ctx.shadowBlur draws.
   *
   * All glowing dots are painted (without shadow) onto an offscreen render
   * texture, then the texture is blitted to the main canvas with a single
   * ctx.filter = 'blur()' — one GPU compositing pass for all dots.
   */
  private _drawFolderGlow(tracks: Track[]): void {
    if (this.hoverTarget === null && this.hoveredSiblingFolder === null) return;

    const w = this.$canvas.clientWidth;
    const h = this.$canvas.clientHeight;
    this._ensureGlowCanvas(w, h);
    const gc = this._glowCtx!;

    // Render all halo circles to the offscreen texture (no shadow)
    gc.clearRect(0, 0, w, h);
    gc.globalAlpha = 0.85;
    for (const pos of this.positions) {
      const track = tracks[pos.idx];
      if (!this._inHoverGlow(track)) continue;
      gc.beginPath();
      gc.arc(pos.sx, pos.sy, DOT_R + 6, 0, Math.PI * 2);
      gc.fillStyle = this._hoverGlowColor(track);
      gc.fill();
    }
    gc.globalAlpha = 1;

    // Single screen-pass blur: blit render texture with one filter operation
    const ctx = this.ctx;
    ctx.save();
    ctx.filter = "blur(14px)";
    ctx.globalAlpha = 0.75;
    ctx.drawImage(this._glowCanvas!, 0, 0, w, h);
    ctx.restore();
  }

  /* ── Drag ghosts (original positions while dragging) ────── */

  private _drawDragGhosts(): void {
    const ghosts = this.dragGhosts;
    if (!ghosts || !ghosts.size) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.setLineDash([2, 3]);
    ctx.lineWidth = 1.5;
    for (const { wx, wy, folder } of ghosts.values()) {
      const [sx, sy] = this.w2s(wx, wy);
      const color = dirColor(folder);
      ctx.beginPath();
      ctx.arc(sx, sy, DOT_R, 0, Math.PI * 2);
      ctx.globalAlpha = 0.2;
      ctx.fillStyle = color;
      ctx.fill();
      ctx.globalAlpha = 0.6;
      ctx.strokeStyle = color;
      ctx.stroke();
    }
    ctx.restore();
  }

  /** Base colour for a track: cluster hue when enabled, else folder hue.
   *
   * Falls back to the folder colour whenever the track has no cluster yet
   * (still being analysed), so partially analysed libraries never render as a
   * field of grey dots.
   */
  trackColor(track: Track): string {
    const m = this.model;
    if (m.colorByCluster && m.clusterAssignments.size > 0) {
      const cid = m.clusterAssignments.get(track.path);
      if (cid !== undefined) return clusterColor(cid);
    }
    return dirColor(track.folder ?? "");
  }

  /* ── Single dot ─────────────────────────────────────────── */

  private _dot(sx: number, sy: number, track: Track, idx: number): void {
    const sel = this.model.selected.has(track.path);
    const hov = this.hoveredIdx === idx;
    const r = sel ? DOT_R_SEL : hov ? DOT_R + 1.5 : DOT_R;
    const ctx = this.ctx;

    let color: string;
    if (sel) color = "#ffdd57";
    else if (hov) color = "#ff7eb3";
    else color = this.trackColor(track);

    // External-root tracks (shown via the sidebar toggle) read as secondary.
    const external = !sel && !hov && this.model.isExternalTrack(track);
    if (external) ctx.globalAlpha = 0.5;

    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    if (external) ctx.globalAlpha = 1;

    if (sel) {
      ctx.strokeStyle = "#ffffffaa";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  /* ── Transform bounding box ─────────────────────────────── */

  private _drawTransformBox(): void {
    const m = this.model;
    if (!m.selected.size) return;
    if (m.viewMode === "embeddings") return;
    if (!m.axisX && !m.axisY) return;

    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity;
    for (const path of m.selected) {
      const t = m.trackByPath(path);
      if (!t || !m.passesFilter(t)) continue;
      const wx = m.axisX ? m.axisScalar(t, m.axisX) : 0.5;
      const wy = m.axisY ? m.axisScalar(t, m.axisY) : 0.5;
      if (wx < minX) minX = wx;
      if (wx > maxX) maxX = wx;
      if (wy < minY) minY = wy;
      if (wy > maxY) maxY = wy;
    }
    if (!isFinite(minX)) return;

    const canTxX = !!(m.axisX && !isAudioFeatureAxisId(m.axisX));
    const canTxY = !!(m.axisY && !isAudioFeatureAxisId(m.axisY));
    this._txAllowBoxDrag = canTxX || canTxY;

    this._txWorld = { minX, maxX, minY, maxY };

    const [sx0] = this.w2s(minX, 0);
    const [sx1] = this.w2s(maxX, 0);
    const [, sy0] = this.w2s(0, minY);
    const [, sy1] = this.w2s(0, maxY);

    const sl = sx0 - TX_PAD;
    const sr = sx1 + TX_PAD;
    const st = sy1 - TX_PAD;
    const sb = sy0 + TX_PAD;

    const cx = (sl + sr) / 2,
      cy = (st + sb) / 2;
    const bsl = cx - Math.max((sr - sl) / 2, 16);
    const bsr = cx + Math.max((sr - sl) / 2, 16);
    const bst = cy - Math.max((sb - st) / 2, 16);
    const bsb = cy + Math.max((sb - st) / 2, 16);

    this._txBox = { sl: bsl, sr: bsr, st: bst, sb: bsb };

    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = "rgba(255,221,87,0.35)";
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 4]);
    ctx.strokeRect(bsl, bst, bsr - bsl, bsb - bst);
    ctx.setLineDash([]);

    const hasRange = maxX - minX >= 0.005 || maxY - minY >= 0.005;
    const mx = (bsl + bsr) / 2,
      my = (bst + bsb) / 2;

    this.txHandles = [];
    if (hasRange) {
      if (m.axisX && m.axisY && canTxX && canTxY) {
        this.txHandles.push(
          { type: "nw", sx: bsl, sy: bst, cursor: "nw-resize" },
          { type: "ne", sx: bsr, sy: bst, cursor: "ne-resize" },
          { type: "sw", sx: bsl, sy: bsb, cursor: "sw-resize" },
          { type: "se", sx: bsr, sy: bsb, cursor: "se-resize" },
        );
      }
      if (m.axisY && canTxY)
        this.txHandles.push(
          { type: "n", sx: mx, sy: bst, cursor: "n-resize" },
          { type: "s", sx: mx, sy: bsb, cursor: "s-resize" },
        );
      if (m.axisX && canTxX)
        this.txHandles.push(
          { type: "e", sx: bsr, sy: my, cursor: "e-resize" },
          { type: "w", sx: bsl, sy: my, cursor: "w-resize" },
        );
    }

    for (const h of this.txHandles) {
      ctx.fillStyle = "rgba(255,221,87,0.9)";
      ctx.fillRect(h.sx - 4, h.sy - 4, 8, 8);
      ctx.strokeStyle = "rgba(0,0,0,0.55)";
      ctx.lineWidth = 1;
      ctx.strokeRect(h.sx - 4, h.sy - 4, 8, 8);
    }
    ctx.restore();
  }

  /* ── Cluster badges ─────────────────────────────────────── */

  private _drawClusters(): void {
    if (this.positions.length < 2) return;
    const grid = new Map<string, DotPosition[]>();
    for (const p of this.positions) {
      const key = `${Math.round(p.sx / CLUSTER_CELL)},${Math.round(p.sy / CLUSTER_CELL)}`;
      if (!grid.has(key)) grid.set(key, []);
      grid.get(key)!.push(p);
    }

    const ctx = this.ctx;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    for (const cluster of grid.values()) {
      if (cluster.length < 2) continue;
      const cx = cluster.reduce((s, p) => s + p.sx, 0) / cluster.length;
      const cy = cluster.reduce((s, p) => s + p.sy, 0) / cluster.length;
      const n = cluster.length;
      const br = n > 9 ? 9 : 7;
      const bx = cx + DOT_R + 1;
      const by = cy - DOT_R - 1;

      ctx.beginPath();
      ctx.arc(bx, by, br, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(15,15,30,0.88)";
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.28)";
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = "#fff";
      ctx.font = `bold ${n > 9 ? 7 : 8}px monospace`;
      ctx.fillText(String(n), bx, by);
    }

    ctx.textAlign = "start";
    ctx.textBaseline = "alphabetic";
  }

  /* ── Mode badge ─────────────────────────────────────────── */

  /** Centred pill while embedding space has nothing to plot yet or layout is in flight. */
  private _embeddingWait(m: Model): { label: string; fraction: number | null } | null {
    if (m.viewMode !== "embeddings") return null;
    // The analysis pipeline owns its own progress pill (see _drawPipelinePill), so
    // these fallbacks only cover states the queue cannot describe.
    if (m.projectionPending) {
      const p = m.layoutProgress;
      if (!p) return { label: "Computing layout…", fraction: null };
      const phase = phaseLabel(p.phase);
      return {
        label:
          p.total > 1
            ? `Computing layout… ${phase} ${p.done} / ${p.total}`
            : `Computing layout… ${phase}`,
        // The layout phases report no steps (no iteration callback), so the bar is
        // indeterminate while it runs.
        fraction: p.total > 1 ? p.done / p.total : null,
      };
    }
    if (m.embeddingPositions.size > 0) return null;
    if (m.embeddingsReady) return null;
    if (m.embeddingModelsLoading) return { label: "loading models…", fraction: null };
    if (m.embeddingsGenerating) return { label: "generating embeddings…", fraction: null };
    return { label: "loading…", fraction: null };
  }

  /** Progress pill for the whole analysis pipeline, with a determinate bar.
   *
   * Big and centred while the map has nothing to plot; once positions exist it
   * docks to the top-centre so progress stays visible without covering the map.
   */
  private _drawPipelinePill(
    w: number,
    h: number,
    prog: PipelineProgress,
    docked: boolean,
  ): void {
    const ctx = this.ctx;
    const label = `${prog.label} · ${prog.detail}`;
    const fontSize = docked ? 12 : 16;
    const barW = docked ? 200 : 240;
    const barH = 4;

    ctx.font = `${docked ? "" : "bold "}${fontSize}px monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const pw = Math.max(barW, ctx.measureText(label).width) + 28;
    const ph = fontSize + 34;
    const cx = w / 2;
    const cy = docked ? 34 + ph / 2 : h / 2;
    const x = cx - pw / 2;
    const y = cy - ph / 2;

    ctx.beginPath();
    ctx.roundRect(x, y, pw, ph, 8);
    ctx.fillStyle = "rgba(10,14,30,0.82)";
    ctx.fill();
    ctx.strokeStyle = "rgba(180,220,255,0.35)";
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.fillStyle = "rgba(180,220,255,0.95)";
    ctx.fillText(label, cx, y + 10 + fontSize / 2);

    // Determinate bar: overall finished/total across every queued stage, except
    // for the tail stages, where the reducer substitutes the stage's own steps.
    const bx = cx - barW / 2;
    const by = y + ph - 12;
    ctx.fillStyle = "rgba(255,255,255,0.14)";
    ctx.beginPath();
    ctx.roundRect(bx, by, barW, barH, barH / 2);
    ctx.fill();
    if (prog.indeterminate) {
      // Opaque phase (gather/mining): no measurable fraction, so fill the whole
      // track faintly instead of leaving it looking stalled at 0 %.
      ctx.fillStyle = "rgba(120,200,255,0.35)";
      ctx.beginPath();
      ctx.roundRect(bx, by, barW, barH, barH / 2);
      ctx.fill();
    } else {
      const fw = Math.max(barH, barW * Math.max(0, Math.min(1, prog.fraction)));
      ctx.fillStyle = "rgba(120,200,255,0.9)";
      ctx.beginPath();
      ctx.roundRect(bx, by, fw, barH, barH / 2);
      ctx.fill();
    }

    ctx.textAlign = "start";
    ctx.textBaseline = "alphabetic";
  }

  private _drawCenterStatusPill(
    w: number,
    h: number,
    label: string,
    fraction: number | null = null,
  ): void {
    const ctx = this.ctx;
    const cx = w / 2;
    const cy = h / 2;
    ctx.font = "bold 22px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const metrics = ctx.measureText(label);
    const pw = metrics.width + 28;
    const ph = 44;
    const rx = 8;
    ctx.beginPath();
    ctx.roundRect(cx - pw / 2, cy - ph / 2, pw, ph, rx);
    ctx.fillStyle = "rgba(10,14,30,0.72)";
    ctx.fill();
    ctx.strokeStyle = "rgba(180,220,255,0.35)";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = "rgba(180,220,255,0.9)";
    ctx.fillText(label, cx, cy - 4);

    // A bar under the label: determinate when the phase reports steps, an
    // animated indeterminate track otherwise (the layout base phase is opaque).
    const barW = Math.max(120, pw - 40);
    const barH = 4;
    const bx = cx - barW / 2;
    const by = cy + ph / 2 - 12;
    ctx.fillStyle = "rgba(255,255,255,0.14)";
    ctx.beginPath();
    ctx.roundRect(bx, by, barW, barH, barH / 2);
    ctx.fill();
    if (fraction != null && Number.isFinite(fraction)) {
      const fw = Math.max(barH, barW * Math.max(0, Math.min(1, fraction)));
      ctx.fillStyle = "rgba(120,200,255,0.9)";
      ctx.beginPath();
      ctx.roundRect(bx, by, fw, barH, barH / 2);
      ctx.fill();
    } else {
      // Indeterminate: slide a short fill across the track so a phase with no
      // step count still reads as "working", not stalled.
      const slide = (performance.now() / 1100) % 1;
      const fw = barW * 0.4;
      const fx = bx - fw + slide * (barW + fw);
      ctx.save();
      ctx.beginPath();
      ctx.roundRect(bx, by, barW, barH, barH / 2);
      ctx.clip();
      ctx.fillStyle = "rgba(120,200,255,0.55)";
      ctx.beginPath();
      ctx.roundRect(fx, by, fw, barH, barH / 2);
      ctx.fill();
      ctx.restore();
      // Keep animating while the indeterminate phase is on screen.
      this.scheduleDraw();
    }

    ctx.textAlign = "start";
    ctx.textBaseline = "alphabetic";
  }

  private _drawModeBadge(w: number, h: number): void {
    const m = this.model;
    if (m.viewMode !== "embeddings") return;
    const ctx = this.ctx;
    ctx.save();

    // Small top-right badge
    ctx.font = "bold 10px monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "top";
    ctx.fillStyle = "rgba(120,200,255,0.6)";
    ctx.fillText("EMBEDDING SPACE", w - 8, 8);

    const wait = this._embeddingWait(m);
    const prog = pipelineProgress(m.queueStatus);
    if (prog) {
      // Docked once the map is drawn, so the indicator never covers the points.
      this._drawPipelinePill(w, h, prog, m.embeddingPositions.size > 0);
    } else if (wait) {
      this._drawCenterStatusPill(w, h, wait.label, wait.fraction);
    }

    ctx.restore();
  }

  /* ── Box selection overlay ──────────────────────────────── */

  private _drawBoxSel(): void {
    const bs = this.boxSel;
    if (!bs) return;
    const x = Math.min(bs.x0, bs.x1);
    const y = Math.min(bs.y0, bs.y1);
    const w = Math.abs(bs.x1 - bs.x0);
    const h = Math.abs(bs.y1 - bs.y0);
    if (w < 2 && h < 2) return;
    const ctx = this.ctx;
    const stroke = this.intersectMode ? "rgba(80,220,255,0.75)" : "rgba(255,221,87,0.7)";
    const fill   = this.intersectMode ? "rgba(80,220,255,0.07)" : "rgba(255,221,87,0.06)";
    ctx.save();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
    ctx.fillStyle = fill;
    ctx.fillRect(x, y, w, h);
    if (this.intersectMode && (w >= 2 || h >= 2)) {
      ctx.font = "bold 11px monospace";
      ctx.fillStyle = "rgba(80,220,255,0.7)";
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText("∩ AND", x + 4, y - 3);
    }
    ctx.restore();
  }

  /* ── Lasso overlay ──────────────────────────────────────── */

  private _drawLasso(): void {
    const pts = this.lassoPoints;
    if (pts.length < 2) return;
    const ctx = this.ctx;
    const stroke = this.intersectMode ? "rgba(80,220,255,0.75)" : "rgba(255,221,87,0.7)";
    const fill   = this.intersectMode ? "rgba(80,220,255,0.07)" : "rgba(255,221,87,0.06)";
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.closePath();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = fill;
    ctx.fill();
    if (this.intersectMode) {
      ctx.font = "bold 11px monospace";
      ctx.fillStyle = "rgba(80,220,255,0.7)";
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText("∩ AND", pts[0][0] + 6, pts[0][1] - 4);
    }
  }

  /* ── Tooltip ────────────────────────────────────────────── */

  private _updateTooltip(tracks: Track[]): void {
    const pinned = this.inspect !== null;
    let track: Track | undefined;
    let anchor: { left: number; top: number } | null = null;

    if (pinned) {
      track = this.model.trackByPath(this.inspect!.path);
      if (!track) {
        this.unpinInspect();
        this.$tip.classList.add("hidden");
        return;
      }
      // Follow the track: a tweened viewport move must carry the card with it,
      // not leave it pinned to a stale screen position.
      const idx = tracks.indexOf(track);
      const pos = idx >= 0 ? this.positions.find((p) => p.idx === idx) : undefined;
      if (pos) {
        const r = this.$canvas.getBoundingClientRect();
        anchor = { left: r.left + pos.sx, top: r.top + pos.sy };
      } else {
        anchor = this._pinPos;
      }
    } else if (this.hoveredIdx >= 0 && this.tipReady) {
      track = tracks[this.hoveredIdx];
      const pos = this.positions.find((p) => p.idx === this.hoveredIdx);
      if (track && pos) {
        const r = this.$canvas.getBoundingClientRect();
        anchor = { left: r.left + pos.sx, top: r.top + pos.sy };
      }
    }

    if (!track || !anchor) {
      this.$tip.classList.add("hidden");
      return;
    }

    const card = buildTooltip(track, this.model, this.inspect);
    if (card.key !== this._tipKey) {
      this._tipKey = card.key;
      renderTooltip(this.$tip, card);
      // The old target elements are gone; a stale hover would keep glowing.
      this.hoverTarget = null;
      this._previewRow = null;
    }
    this.$tip.classList.toggle("pinned", pinned);
    this.$tip.classList.remove("hidden");

    // Keep the card inside the viewport, preferring the right of the anchor.
    let tx = anchor.left + 14;
    if (tx + this.$tip.offsetWidth > window.innerWidth)
      tx = anchor.left - this.$tip.offsetWidth - 14;
    let ty = anchor.top - 10;
    tx = Math.max(4, Math.min(tx, window.innerWidth - this.$tip.offsetWidth - 4));
    ty = Math.max(4, Math.min(ty, window.innerHeight - this.$tip.offsetHeight - 4));
    this.$tip.style.left = tx + "px";
    this.$tip.style.top = ty + "px";
  }

  /* ── Inspect card (pinned, interactive) ──────────────────── */

  /** Pin the hovered track's card open and make it interactive (Alt). */
  pinInspect(): void {
    if (this.inspect || this.hoveredIdx < 0 || !this.tipReady) return;
    const track = this.model.tracks[this.hoveredIdx];
    const pos = this.positions.find((p) => p.idx === this.hoveredIdx);
    if (!track || !pos) return;
    const r = this.$canvas.getBoundingClientRect();
    this._pinPos = { left: r.left + pos.sx, top: r.top + pos.sy };
    this.inspect = {
      path: track.path,
      busy: false,
      message: null,
      mode: null,
      results: [],
      proximity: 100,
    };
    this._tipKey = "";
    this.scheduleDraw();
  }

  /** Close the inspect card; returns whether one was open. */
  unpinInspect(): boolean {
    if (!this.inspect) return false;
    this.inspect = null;
    this._pinPos = null;
    this._previewRow = null;
    this._tipKey = "";
    this.hoverTarget = null;
    this.$tip.classList.remove("pinned");
    this.scheduleDraw();
    return true;
  }

  /** Alt toggles the hovered track's inspect card; true when it closed. */
  toggleInspect(): boolean {
    if (this.inspect) return this.unpinInspect();
    this.pinInspect();
    return false;
  }

  /** Merge a patch into the pinned inspect state (busy/results/…). */
  setInspect(patch: Partial<InspectState>): void {
    if (!this.inspect) return;
    this.inspect = { ...this.inspect, ...patch };
    this._tipKey = "";
    this.scheduleDraw();
  }

  /** Delegated click handling for the inspect card's buttons and result rows. */
  private _onTipClick(e: MouseEvent): void {
    const t = e.target as HTMLElement;
    if (t.closest("[data-inspect-close]")) {
      this.onInspectAction?.("close", this.inspect?.path ?? "");
      return;
    }
    const focus = t.closest<HTMLElement>("[data-inspect-focus]");
    if (focus) {
      this.onInspectAction?.("focus", focus.dataset.inspectFocus!);
      return;
    }
    const action = t.closest<HTMLElement>("[data-inspect-action]");
    if (action) {
      this.onInspectAction?.(
        action.dataset.inspectAction!,
        this.inspect?.path ?? "",
      );
    }
  }

  /**
   * Neighbourhood-slider drag. The card is *not* re-rendered here (the slider is
   * absent from the inspect re-render key), so the input keeps the pointer and the
   * readout updates in place; the coordinator applies the new neighbourhood.
   */
  private _onTipInput(e: Event): void {
    const t = e.target as HTMLElement;
    const range = t.closest<HTMLInputElement>("[data-inspect-proximity]");
    if (!range) return;
    const value = Number(range.value);
    if (!Number.isFinite(value)) return;
    const readout = this.$tip.querySelector<HTMLElement>(
      "[data-inspect-proximity-value]",
    );
    if (readout) readout.textContent = `${Math.round(value)}%`;
    if (this.inspect) this.inspect.proximity = value;
    this.onInspectAction?.("proximity", this.inspect?.path ?? "", value);
  }

  /**
   * Card navigation-target hover: the card's folder/cluster/key/BPM elements set
   * the same `hoverTarget` a sidebar row would, so the glow and S/H agree.
   */
  private _onTipOver(e: MouseEvent): void {
    const t = e.target as HTMLElement;
    // A result row auditions its track on hover, straight away.
    const preview = t.closest<HTMLElement>("[data-preview]");
    if (preview && preview !== this._previewRow) {
      this._previewRow = preview;
      this.onInspectAction?.("hover", preview.dataset.preview!);
    }
    const el = t.closest<HTMLElement>("[data-hover-target]");
    const raw = el?.dataset.hoverTarget;
    if (!raw) return;
    try {
      this.hoverTarget = JSON.parse(raw) as HoverTarget;
      this.scheduleDraw();
    } catch {
      /* malformed attribute; ignore */
    }
  }

  private _onTipOut(e: MouseEvent): void {
    const t = e.target as HTMLElement;
    // Leaving a result row does *not* stop its audition: the preview belongs to
    // the open card, so it keeps playing until the card closes.
    const preview = t.closest<HTMLElement>("[data-preview]");
    const previewTo = (e.relatedTarget as HTMLElement | null)?.closest?.(
      "[data-preview]",
    );
    if (preview && previewTo !== preview) this._previewRow = null;
    const el = t.closest<HTMLElement>("[data-hover-target]");
    if (!el) return;
    const to = (e.relatedTarget as HTMLElement | null)?.closest?.(
      "[data-hover-target]",
    );
    if (to === el) return;
    this.hoverTarget = null;
    this.scheduleDraw();
  }

  clearHover(): void {
    clearTimeout(this.tipTimer);
    this.tipReady = false;
    this.hoveredIdx = -1;
    this.hoveredSiblingFolder = null;
    // The pinned inspect card outlives the pointer leaving the canvas.
    if (!this.inspect) this.$tip.classList.add("hidden");
  }
}
