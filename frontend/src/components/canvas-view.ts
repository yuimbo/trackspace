import type { HoverTarget, Model, Track } from "../model";
import { isAudioFeatureAxisId, scopeContains, scopeHalfExtents } from "../model";
import {
  clusterColor,
  dirColor,
  keyColor,
  parseColor,
  tempoColor,
} from "../lib/colors";
import {
  BUILTIN_LENSES,
  CAMELOT_MAJOR_RADIUS,
  CAMELOT_MINOR_RADIUS,
  camelotGuide,
  colorModeTabs,
  scopeLenses,
  spiralGuide,
  trackTempoNorm,
} from "../lib/lenses";
import { pipelineProgress, phaseLabel, type PipelineProgress } from "../lib/pipeline";
import { GlowLayer, type GlowPoint } from "../lib/glow-layer";
import {
  MetaballLayer,
  POINT_SPLAT_R,
  type PointSpec,
} from "../lib/point-layer";
import {
  buildTooltip,
  describeKey,
  emptyDescribe,
  renderDescribePanel,
  renderGenrePop,
  renderTooltip,
  type InspectState,
  type TooltipCard,
} from "../lib/tooltip";


// ─── Types ───────────────────────────────────────────────────
export interface DotPosition {
  idx: number;
  sx: number;
  sy: number;
}

/** One landmark sample point for the search field (path + query cosine). */
export interface SearchFieldPoint {
  path: string;
  score: number;
}

/** A clickable tab drawn on the canvas (a lens or colour-mode id). */
export interface TabHit {
  id: string;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
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
/** Bottom of the persistent top tabbers (lens row + colour row); status pills dock below. */
const TOP_TABS_BOTTOM = 50;
const DOT_R = 4;
const DOT_R_SEL = 6;
const HIT_RADIUS = 10;
/** Screen-grid cell for the blob-aware hit fallback; ~3×3 cells ≈ 72 px reach. */
const HIT_CELL = 24;
/** Extra screen pixels kept around the viewport before a dot is culled. */
const CULL_MARGIN = 64;
const TX_PAD = 14;
const TX_KNOB_HIT = 9;

/** Halo colour for a key/BPM hover target (folders/clusters use their own colours). */
const TARGET_GLOW_COLOR = "#8e9bff";
/** Search-field glow: accent RGB (matches `TARGET_GLOW_COLOR`) and splat radius in px. */
const FIELD_GLOW_RGB: [number, number, number] = [142 / 255, 155 / 255, 1];
const FIELD_GLOW_R = 64;
/** Hover halo: a dot-sized disc plus a soft edge, sized to match the old blur pass. */
const HOVER_GLOW_R = DOT_R + 20;
const HOVER_GLOW_CORE = 0.42;
const HOVER_GLOW_ALPHA = 0.64;

/**
 * Nearest track when the cursor sits inside a merged blob rather than on a single
 * dot. Returns ``-1`` unless the 3×3 screen-grid neighbourhood around the
 * cursor holds at least two points — a lone dot's cell is not a blob, so a far
 * isolated point is never grabbed. Pure, so the rule is unit-tested directly.
 */
export function nearestInBlob(
  grid: Map<string, DotPosition[]>,
  sx: number,
  sy: number,
  cell = HIT_CELL,
): number {
  const gx = Math.round(sx / cell);
  const gy = Math.round(sy / cell);
  let best = -1;
  let bestD = Infinity;
  let count = 0;
  for (let dx = -1; dx <= 1; dx++) {
    for (let dy = -1; dy <= 1; dy++) {
      const arr = grid.get(`${gx + dx},${gy + dy}`);
      if (!arr) continue;
      count += arr.length;
      for (const p of arr) {
        const d = Math.hypot(p.sx - sx, p.sy - sy);
        if (d < bestD) {
          bestD = d;
          best = p.idx;
        }
      }
    }
  }
  return count >= 2 ? best : -1;
}

/** Id of the tab under *sx,sy*, else null. Pure, so it is unit-tested directly. */
export function tabAt(
  tabs: readonly TabHit[],
  sx: number,
  sy: number,
): string | null {
  for (const t of tabs) {
    if (sx >= t.x0 && sx <= t.x1 && sy >= t.y0 && sy <= t.y1) return t.id;
  }
  return null;
}

export class CanvasView {
  model: Model;
  $canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  $tip: HTMLElement;
  /** Popped-out coordinate panel, shown beside the pinned card on Describe. */
  $coords: HTMLElement;

  positions: DotPosition[] = [];
  hoveredIdx = -1;
  /**
   * A track hovered via a non-canvas row (search hit / Properties row). It gets
   * the same glow, audition and info card as a canvas hover; the row's leave
   * clears it. Purely visual, so it lives here rather than on the model.
   */
  hoveredPath: string | null = null;
  /**
   * True while the search box is focused. The search-field glow shows only then,
   * so a dormant query never tints the map. Purely visual, so it lives here
   * rather than on the model (see `ui_conventions.md` → *Adding new interaction
   * surfaces*).
   */
  searchActive = false;
  /** Landmark query scores the backend returns for the CLAP-lens field. */
  searchField: SearchFieldPoint[] | null = null;
  /**
   * Paths the active search matched. Non-members are drawn faded while it is
   * set, so the results read against the rest of the map. ``null`` = no search.
   */
  searchResults: Set<string> | null = null;
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
  /** The currently rendered card, for the genre chip's hover pop-over. */
  private _tipCard: TooltipCard | null = null;
  /** Identity of the rendered Describe panel (``""`` while hidden). */
  private _describeKey = "";
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
  /** A Describe-tab term clicked: run it as a CLAP search for similar tracks. */
  onSearchTerm: ((term: string) => void) | null = null;
  lassoPoints: [number, number][] = [];
  boxSel: BoxSel | null = null;
  /** True while a lasso/box drag is in intersect (AND) mode — shown in cyan. */
  intersectMode = false;
  dragGhosts: Map<string, { wx: number; wy: number; folder: string }> | null = null;
  /** In-progress scope box in screen coordinates (drag corners). */
  scopeDrag: { x0: number; y0: number; x1: number; y1: number } | null = null;
  /** Clickable lens tabs drawn around the open scope (screen rects). */
  scopeTabs: TabHit[] = [];
  /** Clickable lens tabs in the persistent top tabber. */
  lensTabs: TabHit[] = [];
  /** Clickable colour-mode tabs in the persistent top tabber. */
  colorTabs: TabHit[] = [];
  private _waitTimer = 0;
  _rafId = 0;

  txHandles: TxHandle[] = [];
  _txBox: TxBox | null = null;
  _txWorld: TxWorld | null = null;
  /** False when both axes are analysis features — box interior is not a drag target. */
  _txAllowBoxDrag = false;

  /** Transient per-track positions used during animated mode transitions. */
  animPositions: Map<string, { x: number; y: number }> | null = null;

  /** Transparent WebGL overlay under the scatter: every glow is one additive SDF pass. */
  glow: GlowLayer;

  /**
   * Transparent WebGL layer that merges the scatter points into gooey blobs. Its
   * canvas is blitted between the grid and the overlays; when unsupported the
   * scatter falls back to plain 2-D dots.
   */
  points: MetaballLayer;
  /** This frame's splats (all points except the crisp hover/selected ones). */
  private _pointSpecs: PointSpec[] = [];
  /** Hover/selected dots, excluded from the merge so they read as single tracks. */
  private _crispDots: { sx: number; sy: number; idx: number }[] = [];
  /** Screen grid over the visible dots, for the blob-aware hit fallback. */
  private _hitGrid = new Map<string, DotPosition[]>();

  constructor(model: Model) {
    this.model = model;
    this.$canvas = document.getElementById("viewport") as HTMLCanvasElement;
    this.ctx = this.$canvas.getContext("2d")!;
    this.glow = new GlowLayer(
      this.$canvas.parentElement as HTMLElement,
      this.$canvas,
    );
    this.points = new MetaballLayer();
    this.$tip = document.getElementById("tooltip")!;
    this.$coords = document.getElementById("coords-panel")!;
    this.$tip.addEventListener("click", (e) => this._onTipClick(e));
    this.$tip.addEventListener("input", (e) => this._onTipInput(e));
    this.$tip.addEventListener("mouseover", (e) => this._onTipOver(e));
    this.$tip.addEventListener("mouseout", (e) => this._onTipOut(e));
    // The Describe panel is a sibling of the card, so its term rows (a CLAP
    // search on click) need the same click routing as the card's own controls.
    this.$coords.addEventListener("click", (e) => this._onTipClick(e));
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
    this.glow.sync();
    this.points.sync(this.$canvas);
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
    // A merged blob is wider than a single dot's hit radius, so a click inside
    // one falls through to the nearest member of its neighbourhood.
    if (best < 0 && this.points.supported) {
      best = nearestInBlob(this._hitGrid, sx, sy);
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
    this.scopeTabs = [];
    this.lensTabs = [];
    this.colorTabs = [];
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
      this._drawTopTabs(w);
      this.points.render([]);
      this.glow.render([]);
      return;
    }

    this._drawGrid(w, h);
    this._drawLensGuide();
    this._drawDragGhosts();
    this._drawScatter(tracks);
    this._renderPoints(tracks);
    this._renderGlow(tracks);
    this._drawTransformBox();
    this._drawBoxSel();
    this._drawLasso();
    this._drawScope();
    this._drawTopTabs(w);
    // Docked status pills start below the tabbers, so a running analysis never
    // covers the persistent E/C feedback.
    this._drawModeBadge(w, h, TOP_TABS_BOTTOM + 8);
    this._updateTooltip(tracks);
  }

  /** Lens background guide (Camelot rings + key labels, or the tempo spiral). */
  private _drawLensGuide(): void {
    const m = this.model;
    const lens = m.lens();
    if (m.viewMode !== "embeddings" || !lens.guide) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = "rgba(255,255,255,0.16)";
    ctx.fillStyle = "rgba(255,255,255,0.45)";
    ctx.lineWidth = 1;
    ctx.font = "9px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";

    if (lens.guide === "camelot") {
      const [cx, cy] = this.w2s(0.5, 0.5);
      const [ex] = this.w2s(0.5 + 0.5 * CAMELOT_MAJOR_RADIUS, 0.5);
      const majorR = Math.abs(ex - cx);
      const minorR = majorR * (CAMELOT_MINOR_RADIUS / CAMELOT_MAJOR_RADIUS);
      for (const r of [majorR, minorR]) {
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.stroke();
      }
      for (const p of camelotGuide()) {
        const [sx, sy] = this.w2s(p.x, p.y);
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(sx, sy);
        ctx.stroke();
        ctx.fillText(p.label, sx, sy);
      }
    } else {
      const pts = spiralGuide();
      ctx.beginPath();
      pts.forEach((p, i) => {
        const [sx, sy] = this.w2s(p.x, p.y);
        if (i === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      });
      ctx.stroke();
    }
    ctx.restore();
  }

  /* ── Glow points (search field + hover halo) ────────────── */

  /**
   * The "where does my query match" glow points, shown while the search box is
   * focused and the CLAP lens is on screen.
   *
   * Built from the backend's landmark sample: each landmark is splatted at its
   * *current* map position (scope/animation included), sized and weighted by its
   * query cosine, so the samples read as one smooth additive field. t-SNE only
   * preserves local neighbourhoods, so this shows where matching tracks live —
   * not a calibrated similarity contour.
   */
  collectSearchGlowPoints(): GlowPoint[] {
    const m = this.model;
    // The lens actually on screen: the scope's when one is open, else the map's.
    const lensId = m.scope ? m.scope.lensId : m.lensId;
    if (!this.searchActive || m.viewMode !== "embeddings" || lensId !== "clap") {
      return [];
    }
    const field = this.searchField;
    if (!field || !field.length) return [];

    let lo = Infinity,
      hi = -Infinity;
    const placed: { x: number; y: number; score: number }[] = [];
    for (const p of field) {
      const t = m.trackByPath(p.path);
      if (!t) continue;
      const pos = this._resolveTrackWPos(t);
      if (!pos) continue;
      const [x, y] = this.w2s(pos.wx, pos.wy);
      placed.push({ x, y, score: p.score });
      if (p.score < lo) lo = p.score;
      if (p.score > hi) hi = p.score;
    }
    if (!placed.length) return [];
    const span = Math.max(hi - lo, 1e-6);

    return placed.map((p) => {
      const w = (p.score - lo) / span;
      return {
        x: p.x,
        y: p.y,
        radius: FIELD_GLOW_R * (0.55 + 0.45 * w),
        core: 0,
        color: FIELD_GLOW_RGB,
        alpha: 0.05 + 0.35 * w * w,
      };
    });
  }

  /** Glow points for the hovered navigation target / sibling folder. */
  collectHoverGlowPoints(tracks: Track[]): GlowPoint[] {
    if (this.hoverTarget === null && this.hoveredSiblingFolder === null) {
      return [];
    }
    const points: GlowPoint[] = [];
    for (const pos of this.positions) {
      const track = tracks[pos.idx];
      if (!this._inHoverGlow(track)) continue;
      points.push({
        x: pos.sx,
        y: pos.sy,
        radius: HOVER_GLOW_R,
        core: HOVER_GLOW_CORE,
        color: parseColor(this._hoverGlowColor(track)),
        alpha: HOVER_GLOW_ALPHA,
      });
    }
    return points;
  }

  /** Splat every glow for this frame in one additive overlay pass. */
  private _renderGlow(tracks: Track[]): void {
    this.glow.render([
      ...this.collectSearchGlowPoints(),
      ...this.collectHoverGlowPoints(tracks),
    ]);
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

  /** Outer (pre-scope) position, for scope membership and anchors. */
  outerPosForTrack(t: Track): { wx: number; wy: number } | null {
    return this._outerTrackWPos(t);
  }

  /** Lens id under *sx,sy* when a scope tab is there, else null. */
  scopeTabAt(sx: number, sy: number): string | null {
    return tabAt(this.scopeTabs, sx, sy);
  }

  /** Lens id under *sx,sy* in the top tabber, else null. */
  topLensTabAt(sx: number, sy: number): string | null {
    return tabAt(this.lensTabs, sx, sy);
  }

  /** Colour-mode id under *sx,sy* in the top tabber, else null. */
  topColorTabAt(sx: number, sy: number): string | null {
    return tabAt(this.colorTabs, sx, sy);
  }

  private _resolveTrackWPos(t: Track): { wx: number; wy: number } | null {
    const m = this.model;

    // During animation, interpolated positions always take priority
    if (this.animPositions) {
      const ap = this.animPositions.get(t.path);
      if (ap) return { wx: ap.x, wy: ap.y };
    }

    const outer = this._outerTrackWPos(t);
    if (m.scope) {
      // A scope re-lays out its tracks; scopePositions holds only in-scope paths.
      const sp = m.scopePositions.get(t.path);
      if (sp) return { wx: sp.x, wy: sp.y };
      // A track the scope lens could not place but whose outer position falls
      // inside the circle would clutter the re-layout; hide it.
      if (outer && scopeContains(m.scope, outer.wx, outer.wy)) {
        return null;
      }
    }

    return outer;
  }

  /** Position ignoring any scope: the embedding lens or the tag axes. */
  private _outerTrackWPos(t: Track): { wx: number; wy: number } | null {
    const m = this.model;
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
    const supported = this.points.supported;
    // Reuse the per-frame buffers: allocating three fresh containers for ~14k
    // tracks on every animation frame is pure GC pressure.
    this._pointSpecs.length = 0;
    this._crispDots.length = 0;
    this._hitGrid.clear();
    const w = this.$canvas.clientWidth;
    const h = this.$canvas.clientHeight;
    let anyFiltered = false;
    tracks.forEach((t, i) => {
      if (m.viewMode === "tags" && !m.passesFilter(t)) {
        anyFiltered = true;
        return;
      }
      const pos = this._resolveTrackWPos(t);
      if (!pos) return;
      const [sx, sy] = this.w2s(pos.wx, pos.wy);
      // An off-screen dot cannot be seen or hit, so skip it entirely; while
      // zoomed in this is most of the library.
      if (
        sx < -CULL_MARGIN ||
        sx > w + CULL_MARGIN ||
        sy < -CULL_MARGIN ||
        sy > h + CULL_MARGIN
      )
        return;
      this.positions.push({ idx: i, sx, sy });
      if (!supported) {
        this._dot(sx, sy, t, i);
        return;
      }
      const key = `${Math.round(sx / HIT_CELL)},${Math.round(sy / HIT_CELL)}`;
      const cell = this._hitGrid.get(key);
      if (cell) cell.push({ idx: i, sx, sy });
      else this._hitGrid.set(key, [{ idx: i, sx, sy }]);
      if (m.selected.has(t.path) || this.hoveredIdx === i) {
        this._crispDots.push({ sx, sy, idx: i });
      } else {
        this._pointSpecs.push(this._pointSpec(sx, sy, t));
      }
    });
    if (anyFiltered) {
      const ctx = this.ctx;
      ctx.save();
      ctx.fillStyle = "rgba(255,170,50,0.55)";
      ctx.font = "10px monospace";
      ctx.textAlign = "right";
      ctx.fillText("filtered", w - 6, h - 6);
      ctx.restore();
    }
  }

  /** One splat for a track, with its colour and search/external fade. */
  private _pointSpec(sx: number, sy: number, track: Track): PointSpec {
    const faded =
      this.searchResults !== null && !this.searchResults.has(track.path);
    return {
      x: sx,
      y: sy,
      radius: POINT_SPLAT_R,
      color: parseColor(this.trackColor(track)),
      alpha: faded ? 0.15 : this.model.isExternalTrack(track) ? 0.5 : 1,
    };
  }

  /**
   * Blit the merged field over the grid, then draw the crisp hover/selected dots
   * on top — they are excluded from the field so a single track never dissolves
   * into a blob. Falls through to nothing when the layer is unsupported, since
   * `_drawScatter` already drew every dot.
   */
  private _renderPoints(tracks: Track[]): void {
    if (!this.points.supported) return;
    this.points.render(this._pointSpecs);
    const w = this.$canvas.clientWidth,
      h = this.$canvas.clientHeight;
    this.ctx.drawImage(this.points.canvas, 0, 0, w, h);
    for (const c of this._crispDots) this._dot(c.sx, c.sy, tracks[c.idx], c.idx);
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

  /** Base colour for a track, from the active colour mode.
   *
   * Feature modes (tempo, key) reuse the matching lens's polar geometry, so a
   * track's colour tracks its position on that map. Falls back to the folder
   * colour whenever the chosen mode has nothing to say about the track — no cluster
   * yet, or no cached features — so partially analysed libraries never render as a
   * field of grey dots.
   */
  trackColor(track: Track): string {
    const m = this.model;
    switch (m.colorMode) {
      case "head": {
        const predicted = m.folderPredictions.get(track.path);
        // "Confident only" leaves a weak match on its real folder, so a novel
        // track never reads as a folder it merely resembles.
        const ok =
          predicted !== undefined &&
          (!m.headColorConfident ||
            m.folderPredictionMatches.has(track.path));
        if (ok) return dirColor(predicted);
        break;
      }
      case "cluster": {
        const cid = m.clusterAssignments.get(track.path);
        if (cid !== undefined) return clusterColor(cid);
        break;
      }
      case "tempo": {
        const tnorm = trackTempoNorm(track);
        if (tnorm != null) return tempoColor(tnorm);
        break;
      }
      case "key": {
        const af = track.audio_features;
        if (af) return keyColor(af.key, af.mode);
        break;
      }
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
    else color = this.trackColor(track);

    // External-root tracks (shown via the sidebar toggle) read as secondary.
    const external = !sel && !hov && this.model.isExternalTrack(track);
    // While a search is active, tracks it did not match fade into the map so the
    // results stand out. The selected set *is* the result set, so `!sel` covers it.
    const faded =
      this.searchResults !== null &&
      !sel &&
      !hov &&
      !this.searchResults.has(track.path);
    ctx.globalAlpha = faded ? 0.15 : external ? 0.5 : 1;

    ctx.beginPath();
    ctx.arc(sx, sy, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();

    ctx.globalAlpha = 1;

    // Selected and hovered dots get a white outline, so a track stays legible
    // against a merged blob without changing its own colour.
    if (sel || hov) {
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

  /* ── Mode badge ─────────────────────────────────────────── */

  /** Centred pill while embedding space has nothing to plot yet or layout is in flight. */
  private _embeddingWait(m: Model): { label: string; fraction: number | null } | null {
    if (m.viewMode !== "embeddings") return null;
    // The analysis pipeline owns its own progress pill (see _drawPipelinePill), so
    // these fallbacks only cover states the queue cannot describe.
    if (m.projectionPending) {
      // Name the lens being loaded: the map still shows the previous lens while the
      // new one fits, so "Computing layout…" alone reads as if the shown view is.
      const lens = m.lens().label;
      const p = m.layoutProgress;
      const phase = p ? ` · ${phaseLabel(p.phase)}` : "";
      const counts = p && p.total > 1 ? ` ${p.done} / ${p.total}` : "";
      const elapsed =
        m.projectionStartedAt != null
          ? ` · ${Math.max(0, Math.round((Date.now() - m.projectionStartedAt) / 1000))}s`
          : "";
      return {
        label: `Loading ${lens}${phase}${counts}${elapsed}`,
        // The base layout phases report no steps (no iteration callback), so the bar
        // is indeterminate; the elapsed readout is what shows it is still moving.
        fraction: p && p.total > 1 ? p.done / p.total : null,
      };
    }
    if (m.embeddingPositions.size > 0) return null;
    if (m.embeddingsReady) return null;
    if (m.embeddingModelsLoading) return { label: "loading models…", fraction: null };
    if (m.embeddingsGenerating) return { label: "generating embeddings…", fraction: null };
    return { label: "loading…", fraction: null };
  }

  /** Redraw once while a projection is in flight so its elapsed readout ticks. */
  private _scheduleWaitRedraw(): void {
    clearTimeout(this._waitTimer);
    this._waitTimer = setTimeout(
      () => this.scheduleDraw(),
      500,
    ) as unknown as number;
  }

  /** Progress pill for the whole analysis pipeline, with a determinate bar.
   *
   * Big and centred while the map has nothing to plot and it is the only
   * indicator; otherwise docked at *top* so progress stays visible without
   * covering the map and leaves room for the layout pill below it. Returns the
   * pill's bottom edge, so a second indicator can stack under it.
   */
  private _drawPipelinePill(
    w: number,
    h: number,
    prog: PipelineProgress,
    top: number,
    docked: boolean,
  ): number {
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
    const cy = docked ? top + ph / 2 : h / 2;
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
    return y + ph;
  }

  /** Docked status pill for a layout that is still computing.
   *
   * The map keeps rendering the previous lens while the new one fits, so a
   * **throbber** is what makes the wait legible instead of the map looking
   * idle. Drawn at *top* so it stacks under the analysis pill, and returns its
   * bottom edge for the next indicator.
   */
  private _drawCenterStatusPill(
    w: number,
    label: string,
    fraction: number | null,
    top: number,
  ): number {
    const ctx = this.ctx;
    const pad = 14;
    const spinnerR = 7;
    const gap = 10;
    ctx.font = "bold 13px monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const pw = pad + spinnerR * 2 + gap + ctx.measureText(label).width + pad;
    const ph = 40;
    const cx = w / 2;
    const x = cx - pw / 2;
    const rowY = top + 16;
    ctx.beginPath();
    ctx.roundRect(x, top, pw, ph, 8);
    ctx.fillStyle = "rgba(10,14,30,0.82)";
    ctx.fill();
    ctx.strokeStyle = "rgba(180,220,255,0.35)";
    ctx.lineWidth = 1;
    ctx.stroke();

    this._drawSpinner(x + pad + spinnerR, rowY, spinnerR);
    ctx.fillStyle = "rgba(180,220,255,0.95)";
    ctx.fillText(label, x + pad + spinnerR * 2 + gap, rowY);

    // A bar under the label: determinate when the phase reports steps, an
    // animated indeterminate track otherwise (the layout base phase is opaque).
    const barW = Math.max(120, pw - 2 * pad);
    const barH = 4;
    const bx = cx - barW / 2;
    const by = top + ph - 10;
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
    return top + ph;
  }

  /** A rotating arc, so an indeterminate wait reads as "working". */
  private _drawSpinner(cx: number, cy: number, r: number): void {
    const ctx = this.ctx;
    const a0 = ((performance.now() / 900) % 1) * Math.PI * 2;
    ctx.save();
    ctx.lineWidth = 2;
    ctx.lineCap = "round";
    ctx.strokeStyle = "rgba(180,220,255,0.25)";
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();
    ctx.strokeStyle = "rgba(120,200,255,0.95)";
    ctx.beginPath();
    ctx.arc(cx, cy, r, a0, a0 + Math.PI * 0.55);
    ctx.stroke();
    ctx.restore();
  }

  /** Top-right badge and the docked status pills, starting at *top*. */
  private _drawModeBadge(w: number, h: number, top: number): void {
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
    // The speculative per-lens prefetch is background work: the status bar reports
    // it, but the canvas must not make it look like the shown view is computing.
    const showProg = !!prog && !prog.background;
    // A lens still fitting keeps the previous lens on screen, so the layout wait
    // *stacks below* the analysis pill rather than being hidden by it — otherwise
    // a lens switch looks like the shown map is idle and interactive.
    const layoutWait = wait !== null && m.projectionPending;
    // Docked once the map is drawn, or whenever a second indicator shares the
    // top strip, so neither indicator covers the points.
    const docked = m.embeddingPositions.size > 0 || layoutWait;

    if (showProg && prog) {
      top = this._drawPipelinePill(w, h, prog, top, docked) + 8;
    }
    if (wait && (layoutWait || !showProg)) {
      this._drawCenterStatusPill(w, wait.label, wait.fraction, top);
    }
    if (m.projectionPending) this._scheduleWaitRedraw();

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

  /* ── Scope overlay ─────────────────────────────────────── */

  /** The open scope region, its lens tab switcher, and the in-progress drag. */
  private _drawScope(): void {
    const ctx = this.ctx;
    const scope = this.model.scope;
    const drag = this.scopeDrag;
    if (!scope && !drag) return;
    ctx.save();
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = "rgba(120,220,255,0.85)";
    ctx.fillStyle = "rgba(120,220,255,0.05)";
    if (drag) {
      // The drag is a screen-space box; draw it as-is.
      const x = Math.min(drag.x0, drag.x1);
      const y = Math.min(drag.y0, drag.y1);
      ctx.beginPath();
      ctx.rect(x, y, Math.max(1, Math.abs(drag.x1 - drag.x0)), Math.max(1, Math.abs(drag.y1 - drag.y0)));
      ctx.fill();
      ctx.stroke();
    }
    if (scope) {
      // The region is world-space and w2s is non-uniform: it stretches the unit
      // square to the viewport, so a circle's screen image is an ellipse and a
      // box's is a rectangle. Draw the exact membership region so an
      // out-of-scope track can never sit inside the outline.
      const { hw, hh } = scopeHalfExtents(scope);
      const [cx, cy] = this.w2s(scope.cx, scope.cy);
      const [ex] = this.w2s(scope.cx + hw, scope.cy);
      const [, ey] = this.w2s(scope.cx, scope.cy + hh);
      const rx = Math.abs(ex - cx);
      const ry = Math.abs(ey - cy);
      ctx.beginPath();
      if (scope.shape === "box") ctx.rect(cx - rx, cy - ry, rx * 2, ry * 2);
      else ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.setLineDash([]);
      // The lens tabs sit just outside the region, above unless it is near the top.
      const above = cy - ry - 16 > 26;
      const tabCy = above ? cy - ry - 16 : cy + ry + 16;
      this.scopeTabs = this._drawTabs(
        scopeLenses(),
        scope.lensId,
        cx,
        tabCy,
      );
    }
    ctx.setLineDash([]);
    ctx.restore();
  }

  /**
   * The persistent top tabbers: a lens row and a colour-mode row, both centred
   * at the top of the canvas. Their active tab tracks the model, so they are the
   * always-visible feedback for **E** / **C** (and a click switches directly).
   */
  private _drawTopTabs(w: number): void {
    const m = this.model;
    this.lensTabs = this._drawTabs(
      BUILTIN_LENSES,
      m.activeBaseLensId(),
      w / 2,
      16,
    );
    this.colorTabs = this._drawTabs(colorModeTabs(), m.colorMode, w / 2, 40);
  }

  /**
   * Draw a horizontal row of tabs centred at *(cx, cy)* and return their screen rects
   * so a click can be routed back to the coordinator.
   */
  private _drawTabs(
    tabs: readonly { id: string; label: string }[],
    activeId: string,
    cx: number,
    cy: number,
  ): TabHit[] {
    const ctx = this.ctx;
    const padX = 9;
    const gap = 6;
    const h = 20;
    ctx.save();
    ctx.font = "11px monospace";
    const widths = tabs.map(
      (t) => ctx.measureText(t.label).width + padX * 2,
    );
    const total =
      widths.reduce((a, b) => a + b, 0) + gap * (tabs.length - 1);
    let x = cx - total / 2;
    const y = cy - h / 2;
    const hits: TabHit[] = [];
    tabs.forEach((tab, i) => {
      const w = widths[i];
      const active = tab.id === activeId;
      this._roundRect(x, y, w, h, 5);
      ctx.fillStyle = active ? "rgba(120,220,255,0.92)" : "rgba(18,20,24,0.86)";
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = active
        ? "rgba(120,220,255,1)"
        : "rgba(120,220,255,0.3)";
      ctx.stroke();
      ctx.fillStyle = active ? "#0b1a22" : "rgba(210,220,230,0.9)";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(tab.label, x + w / 2, y + h / 2 + 0.5);
      hits.push({ id: tab.id, x0: x, y0: y, x1: x + w, y1: y + h });
      x += w + gap;
    });
    ctx.restore();
    return hits;
  }

  /** Trace a rounded rectangle on the current path. */
  private _roundRect(x: number, y: number, w: number, h: number, r: number): void {
    const ctx = this.ctx;
    const rr = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.arcTo(x + w, y, x + w, y + h, rr);
    ctx.arcTo(x + w, y + h, x, y + h, rr);
    ctx.arcTo(x, y + h, x, y, rr);
    ctx.arcTo(x, y, x + w, y, rr);
    ctx.closePath();
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
    } else if (this.hoveredPath) {
      // A search hit / Properties row hovered: show the same card a canvas hover
      // would, anchored at the track's dot. No delay — the row is the hover.
      track = this.model.trackByPath(this.hoveredPath);
      const idx = track ? tracks.indexOf(track) : -1;
      const pos = idx >= 0 ? this.positions.find((p) => p.idx === idx) : undefined;
      if (track && pos) {
        const r = this.$canvas.getBoundingClientRect();
        anchor = { left: r.left + pos.sx, top: r.top + pos.sy };
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
      this.$coords.classList.add("hidden");
      this._describeKey = "";
      this._tipCard = null;
      this._hideGenrePop();
      return;
    }

    const card = buildTooltip(track, this.model, this.inspect);
    // On the Describe tab the content folds out into a wide two-column panel
    // beside the card (terms | coordinates), where it gets room the 300 px card
    // cannot give. `card.inspect` is the path-guarded effective state.
    const showPanel = pinned && this.inspect?.tab === "describe";
    const describe = card.inspect?.describe ?? emptyDescribe("");
    if (card.key !== this._tipKey) {
      this._tipKey = card.key;
      renderTooltip(this.$tip, card);
      // The old target elements are gone; a stale hover would keep glowing, and
      // a rebuilt chip cannot keep the old pop-over.
      this.hoverTarget = null;
      this._previewRow = null;
      this._hideGenrePop();
    }
    this._tipCard = card;
    // The panel is a sibling of the card, so it needs its own identity gate: a
    // pan/zoom frame must not replace the term rows under the pointer.
    const panelKey = showPanel ? describeKey(describe) : "";
    if (panelKey !== this._describeKey) {
      this._describeKey = panelKey;
      renderDescribePanel(this.$coords, describe, showPanel);
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

    if (showPanel) {
      let px = tx + this.$tip.offsetWidth + 8;
      if (px + this.$coords.offsetWidth > window.innerWidth)
        px = tx - this.$coords.offsetWidth - 8;
      px = Math.max(4, Math.min(px, window.innerWidth - this.$coords.offsetWidth - 4));
      const py = Math.max(
        4,
        Math.min(ty, window.innerHeight - this.$coords.offsetHeight - 4),
      );
      this.$coords.style.left = px + "px";
      this.$coords.style.top = py + "px";
    }
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
      tab: "navigate",
      describe: emptyDescribe(track.path),
    };
    this._tipKey = "";
    this.scheduleDraw();
    // Load the zero-shot descriptions so the coordinates show next to the stats
    // while the card is pinned, before the Describe tab is opened.
    this.onInspectAction?.("open", track.path);
  }

  /** Close the inspect card; returns whether one was open. */
  unpinInspect(): boolean {
    if (!this.inspect) return false;
    this.inspect = null;
    this._pinPos = null;
    this._previewRow = null;
    this._tipKey = "";
    this._tipCard = null;
    this._hideGenrePop();
    this.hoverTarget = null;
    this.$tip.classList.remove("pinned");
    this.$coords.classList.add("hidden");
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
    // A Describe-tab term is a search: clicking it runs the semantic query.
    const term = t.closest<HTMLElement>("[data-search-term]");
    if (term) {
      this.onSearchTerm?.(term.dataset.searchTerm!);
      return;
    }
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
    // Hovering the genre chip opens a pop-over with the full name, so a truncated
    // one can be read without the fixed-width card reflowing.
    const genre = t.closest<HTMLElement>("[data-genre-pop]");
    if (genre) this._showGenrePop(genre);
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
    const genre = t.closest<HTMLElement>("[data-genre-pop]");
    const genreTo = (e.relatedTarget as HTMLElement | null)?.closest?.(
      "[data-genre-pop]",
    );
    if (genre && genreTo !== genre) this._hideGenrePop();
    const el = t.closest<HTMLElement>("[data-hover-target]");
    if (!el) return;
    const to = (e.relatedTarget as HTMLElement | null)?.closest?.(
      "[data-hover-target]",
    );
    if (to === el) return;
    this.hoverTarget = null;
    this.scheduleDraw();
  }

  /**
   * The genre chip's hover pop-over. It is a body-level fixed element, not a
   * child of the card, so expanding a truncated name never reflows the card.
   */
  private _genrePop: HTMLElement | null = null;

  private _ensureGenrePop(): HTMLElement {
    if (!this._genrePop) {
      const pop = document.createElement("div");
      pop.id = "tt-pop";
      pop.className = "tt-pop hidden";
      document.body.appendChild(pop);
      this._genrePop = pop;
    }
    return this._genrePop;
  }

  /** Show the full genre name (and projected folder) above the chip. */
  private _showGenrePop(anchor: HTMLElement): void {
    const genre = this._tipCard?.genre;
    if (!genre) return;
    const pop = this._ensureGenrePop();
    renderGenrePop(pop, genre);
    const r = anchor.getBoundingClientRect();
    const pw = pop.offsetWidth,
      ph = pop.offsetHeight;
    let top = r.top - ph - 6;
    if (top < 4) top = r.bottom + 6;
    const left = Math.max(4, Math.min(r.left, window.innerWidth - pw - 4));
    pop.style.left = `${left}px`;
    pop.style.top = `${Math.max(4, Math.min(top, window.innerHeight - ph - 4))}px`;
  }

  private _hideGenrePop(): void {
    this._genrePop?.classList.add("hidden");
  }

  /**
   * Cursor-adjacent readout while **M + scroll** adjusts the preview volume. A
   * body-level fixed element, so it follows the pointer without touching layout,
   * and it fades after a short pause so it never lingers.
   */
  private _volumeHud: HTMLElement | null = null;
  private _volumeHudTimer = 0;

  showVolumeHud(clientX: number, clientY: number): void {
    if (!this._volumeHud) {
      const hud = document.createElement("div");
      hud.id = "volume-hud";
      hud.className = "volume-hud hidden";
      document.body.appendChild(hud);
      this._volumeHud = hud;
    }
    const hud = this._volumeHud;
    hud.textContent = `${Math.round(this.model.previewVolume * 100)}%`;
    hud.classList.remove("hidden");
    hud.style.left = `${clientX + 14}px`;
    hud.style.top = `${clientY + 14}px`;
    clearTimeout(this._volumeHudTimer);
    this._volumeHudTimer = setTimeout(
      () => hud.classList.add("hidden"),
      700,
    ) as unknown as number;
  }

  clearHover(): void {
    clearTimeout(this.tipTimer);
    this.tipReady = false;
    this.hoveredIdx = -1;
    this.hoveredPath = null;
    this.hoveredSiblingFolder = null;
    // The pinned inspect card outlives the pointer leaving the canvas.
    if (!this.inspect) {
      this.$tip.classList.add("hidden");
      this.$coords.classList.add("hidden");
      this._hideGenrePop();
    }
  }
}
