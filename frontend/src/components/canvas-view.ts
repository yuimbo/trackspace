import type { Model, Track } from "../model";
import { isAudioFeatureAxisId } from "../model";
import { dirColor } from "../lib/toast";
import { displayFolderPath } from "../lib/path-presenter";


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

export class CanvasView {
  model: Model;
  $canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  $tip: HTMLElement;

  positions: DotPosition[] = [];
  hoveredIdx = -1;
  hoveredFolderPrefix: string | null = null;
  /** Exact folder path (``""`` = library root): glow siblings while hovering a dot. */
  hoveredSiblingFolder: string | null = null;
  tipReady = false;
  tipTimer = 0;
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

  // Offscreen canvas used as render texture for the glow screen-pass
  private _glowCanvas: HTMLCanvasElement | null = null;
  private _glowCtx: CanvasRenderingContext2D | null = null;

  constructor(model: Model) {
    this.model = model;
    this.$canvas = document.getElementById("viewport") as HTMLCanvasElement;
    this.ctx = this.$canvas.getContext("2d")!;
    this.$tip = document.getElementById("tooltip")!;
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

    // Axis names
    ctx.fillStyle = "rgba(255,255,255,0.25)";
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    if (m.viewMode === "embeddings") {
      ctx.fillText("Embedding Space", w / 2, h - 2);
    } else {
      if (m.axisX) ctx.fillText(m.axisDisplayName(m.axisX), w / 2, h - 2);
      if (m.axisY) {
        ctx.save();
        ctx.translate(8, h / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(m.axisDisplayName(m.axisY), 0, 0);
        ctx.restore();
      }
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
      if (ap) return { wx: ap.x, wy: ap.y };
    }

    if (m.viewMode === "embeddings") {
      const ep = m.embeddingPositions.get(t.path);
      if (ep) return { wx: ep.x, wy: ep.y };
      // No embedding → hide the track (don't fall back to tag positions)
      return null;
    }

    return this._tagPos(t);
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

  /* ── Folder-hover glow pass ────────────────────────────── */

  private _inHoveredFolder(track: Track): boolean {
    const prefix = this.hoveredFolderPrefix;
    if (prefix === null) return false;
    const f = track.folder ?? "";
    if (prefix === "") return true;
    return f === prefix || f.startsWith(prefix + "/");
  }

  /** Tracks that get the folder-style halo: sidebar subtree or canvas sibling folder. */
  private _inFolderGlow(track: Track): boolean {
    if (this.hoveredFolderPrefix !== null) return this._inHoveredFolder(track);
    const sib = this.hoveredSiblingFolder;
    if (sib === null) return false;
    return (track.folder ?? "") === sib;
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
   * Renders the folder-hover glow as a single screen-pass blur instead of
   * N individual ctx.shadowBlur draws.
   *
   * All glowing dots are painted (without shadow) onto an offscreen render
   * texture, then the texture is blitted to the main canvas with a single
   * ctx.filter = 'blur()' — one GPU compositing pass for all dots.
   */
  private _drawFolderGlow(tracks: Track[]): void {
    if (this.hoveredFolderPrefix === null && this.hoveredSiblingFolder === null)
      return;

    const w = this.$canvas.clientWidth;
    const h = this.$canvas.clientHeight;
    this._ensureGlowCanvas(w, h);
    const gc = this._glowCtx!;

    // Render all halo circles to the offscreen texture (no shadow)
    gc.clearRect(0, 0, w, h);
    gc.globalAlpha = 0.85;
    for (const pos of this.positions) {
      const track = tracks[pos.idx];
      if (!this._inFolderGlow(track)) continue;
      gc.beginPath();
      gc.arc(pos.sx, pos.sy, DOT_R + 6, 0, Math.PI * 2);
      gc.fillStyle = dirColor(track.folder ?? "");
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

  /* ── Single dot ─────────────────────────────────────────── */

  private _dot(sx: number, sy: number, track: Track, idx: number): void {
    const sel = this.model.selected.has(track.path);
    const hov = this.hoveredIdx === idx;
    const r = sel ? DOT_R_SEL : hov ? DOT_R + 1.5 : DOT_R;
    const ctx = this.ctx;

    let color: string;
    if (sel) color = "#ffdd57";
    else if (hov) color = "#ff7eb3";
    else color = dirColor(track.folder ?? "");

    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

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
    let y = 22;
    if (m.embeddingsGenerating && m.embeddingProgress) {
      ctx.fillStyle = "rgba(255,200,100,0.6)";
      ctx.fillText(
        `generating ${m.embeddingProgress.done}/${m.embeddingProgress.total}…`,
        w - 8,
        y,
      );
      y += 14;
    }

    // Large centred "clustering…" overlay
    if (m.projectionPending) {
      const cx = w / 2;
      const cy = h / 2;
      const label = "clustering…";
      ctx.font = "bold 22px monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      const metrics = ctx.measureText(label);
      const pw = metrics.width + 28;
      const ph = 44;
      const rx = 8;
      // pill background
      ctx.beginPath();
      ctx.roundRect(cx - pw / 2, cy - ph / 2, pw, ph, rx);
      ctx.fillStyle = "rgba(10,14,30,0.72)";
      ctx.fill();
      ctx.strokeStyle = "rgba(180,220,255,0.35)";
      ctx.lineWidth = 1;
      ctx.stroke();
      // label
      ctx.fillStyle = "rgba(180,220,255,0.9)";
      ctx.fillText(label, cx, cy);
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
    if (this.hoveredIdx < 0 || !this.tipReady) {
      this.$tip.classList.add("hidden");
      return;
    }
    const track = tracks[this.hoveredIdx];
    const pos = this.positions.find((p) => p.idx === this.hoveredIdx);
    if (!track || !pos) {
      this.$tip.classList.add("hidden");
      return;
    }

    const folderLine = displayFolderPath(track.folder || "");
    const ar = (track.artist || "").trim();
    const ti = (track.title || "").trim();
    const titleLine =
      ar && ti ? `${ar} — ${ti}` : ti || ar || track.filename;
    const audioParts: string[] = [];
    if (track.bpm != null && Number.isFinite(track.bpm))
      audioParts.push(`${Math.round(track.bpm)} BPM`);
    if (track.musical_key) audioParts.push(track.musical_key);
    const audioLine = audioParts.join(" · ");

    const tagStr = Object.entries(track.tags)
      .map(([k, v]) => `${k}: ${v.toFixed(2)}`)
      .join("  ");
    const lines = [folderLine, titleLine];
    if (audioLine) lines.push(audioLine);
    if (tagStr) lines.push(tagStr);
    this.$tip.textContent = lines.join("\n");
    this.$tip.classList.remove("hidden");

    const r = this.$canvas.getBoundingClientRect();
    let tx = r.left + pos.sx + 14;
    const ty = r.top + pos.sy - 10;
    if (tx + 320 > window.innerWidth) tx = r.left + pos.sx - 320;
    this.$tip.style.left = tx + "px";
    this.$tip.style.top = ty + "px";
  }

  clearHover(): void {
    clearTimeout(this.tipTimer);
    this.tipReady = false;
    this.hoveredIdx = -1;
    this.hoveredSiblingFolder = null;
    this.$tip.classList.add("hidden");
  }
}
