import type { Billboard, HoverTarget, MapPos, Model, Track } from "../model";
import { clamp01 } from "../lib/util";
import {
  blendCamera,
  orbitProject,
  orbitUnproject,
  readPerspective,
  type Camera,
} from "../lib/camera3d";
import {
  isAudioFeatureAxisId,
  scopeContains,
  scopeHalfExtents,
} from "../model";
import { bloomFolderLevels, bloomRangeKey, bloomRangeLabel } from "../lib/bloom";
import { displayFolderPath } from "../lib/path-presenter";
import {
  clusterColor,
  dirColor,
  keyColor,
  parseColor,
  tempoColor,
} from "../lib/colors";
import {
  CAMELOT_MAJOR_RADIUS,
  CAMELOT_MINOR_RADIUS,
  availableLenses,
  camelotGuide,
  colorModeTabs,
  scopeLenses,
  spiralGuide,
  trackTempoNorm,
  type ColorMode,
} from "../lib/lenses";
import { pipelineProgress, phaseLabel, type PipelineProgress } from "../lib/pipeline";
import { GlowLayer, type GlowOccluder, type GlowPoint } from "../lib/glow-layer";
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
  renderTooltip,
  updateSeekProgress,
  type InspectState,
} from "../lib/tooltip";


// ─── Types ───────────────────────────────────────────────────
export interface DotPosition {
  idx: number;
  sx: number;
  sy: number;
  /** Normalised depth, 1 = nearest the viewer (always 1 in 2-D). */
  depth: number;
}

/**
 * A track's resolved world position: a cube position, or a sphere-scope member's
 * flat ``billboard`` layout (see ``MapPos``). ``from``/``mix`` describe an
 * in-flight move between frames: the projection of ``from`` is blended into this
 * frame's projection with weight ``mix`` (1 = entirely ``from``, 0 = entirely
 * this position), so cross-frame animations are continuous in view space.
 */
interface ResolvedTrackPos {
  wx: number;
  wy: number;
  wz?: number;
  billboard?: Billboard;
  from?: { wx: number; wy: number; wz?: number; billboard?: Billboard };
  mix?: number;
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
/** A track the folder head cannot place confidently, under "Confident only". */
export const CONFIDENT_MISS_COLOR = "#6b7280";
/**
 * Colour priority of a grey dot in a merged blob: it still adds shape, but almost
 * none of its colour, so it cannot wash out the hue of its coloured neighbours.
 */
const GREY_COLOUR_WEIGHT = 0.15;
const HIT_RADIUS = 10;
/** Screen-grid cell for the blob-aware hit fallback; ~3×3 cells ≈ 72 px reach. */
const HIT_CELL = 24;
/** Extra screen pixels kept around the viewport before a dot is culled. */
const CULL_MARGIN = 64;
/** A 3-D scope gesture snaps to the nearest on-screen dot within this many px. */
const SNAP_PX = 20;
const TX_PAD = 14;
const TX_KNOB_HIT = 9;
/**
 * Depth of the front-most splat is scaled into [DEPTH_FLOOR, 1]; the floor keeps a
 * point's depth strictly positive, because 0 means "no point" in the depth pre-pass.
 */
const DEPTH_FLOOR = 0.05;
/** Depth-cue extremes: how the colour scales at the back (0) vs front (1). */
const DEPTH_SHADE_MIN = 0.45;
const DEPTH_SHADE_SPAN = 0.55;
/** A wormhole-ghosted dot fades to this opacity at full influence (ghost = 1). */
const GHOST_FADE = 0.85;
/**
 * A sphere scope's wormhole pushes in-front non-members out to this multiple of
 * the sphere's screen radius, ghosting them as they clear the disc. Inside the
 * circle the frosted glass takes over (`rimPush`); past this reach nothing moves.
 */
const WORMHOLE_REACH = 1.7;
/**
 * Width of the inner rim band a glassed point is pushed onto, as a fraction of
 * the sphere's screen radius. `rimPush` maps the inside of the circle onto
 * [(1 − RIM_BAND)·r, (1 − RIM_BAND + RIM_BAND·0.8)·r].
 */
const RIM_BAND = 0.18;
/**
 * Frosted-glass tint over the sphere scope at full blur: the disc softens the
 * grid and the outer blobs, and this tint dims what the blur leaves.
 */
export const GLASS_TINT = 0.5;
/**
 * Glass tint without a blur (the canvas `filter` is unavailable, e.g. Safari):
 * the tint alone must hide more, since nothing softens the covered content.
 */
export const GLASS_TINT_FLAT = 0.8;
/** Default sphere-scope glass blur radius in CSS px; `?glass=` tunes it live. */
export const GLASS_BLUR_PX = 7;
/** The glass blur is clamped to this ceiling (CSS px). */
export const MAX_GLASS_BLUR = 24;

/**
 * A cube-unit depth difference → normalised depth units, the scale `_trackScreen`
 * maps raw rotated depth to (a cube-unit Δ lands at (1 - DEPTH_FLOOR)·Δ/√3 there).
 */
function depthUnits(delta: number): number {
  return ((1 - DEPTH_FLOOR) * delta) / Math.sqrt(3);
}

/** GLSL-style smoothstep: 0 below *edge0*, 1 above *edge1*, smooth in between. */
function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp01((x - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
}

/** The 8 corners of the unit cube, (x, y, z) each 0 or 1. */
const CUBE_CORNERS: readonly (readonly [number, number, number])[] = [
  [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
  [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
];
/** The 12 edges of the unit cube, as corner index pairs. */
const CUBE_EDGES: readonly (readonly [number, number])[] = [
  [0, 1], [2, 3], [4, 5], [6, 7],
  [0, 2], [1, 3], [4, 6], [5, 7],
  [0, 4], [1, 5], [2, 6], [3, 7],
];
/** The view-row tabs: a single 3-D orbit toggle. */
const VIEW_TABS: readonly { id: string; label: string }[] = [
  { id: "3d", label: "3D" },
];

/** Halo colour for a key/BPM hover target (folders/clusters use their own colours). */
const TARGET_GLOW_COLOR = "#8e9bff";
/** Search-field glow: accent RGB (matches `TARGET_GLOW_COLOR`) and splat radius in px. */
const FIELD_GLOW_RGB: [number, number, number] = [142 / 255, 155 / 255, 1];
const FIELD_GLOW_R = 64;
/** Hover halo: a dot-sized disc plus a soft edge, sized to match the old blur pass. */
const HOVER_GLOW_R = DOT_R + 20;
const HOVER_GLOW_CORE = 0.42;
const HOVER_GLOW_ALPHA = 0.64;
/** Selection-bloom cursor ring when no dot is hovered (no class colour to borrow). */
const CURSOR_COLOR = "rgba(255,190,90,0.95)";

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

/**
 * 3-D depth shade for a splat: how much brighter a point draws at *depth*
 * (0 back … 1 front) than a flat 2-D dot. The back of the cube is 45 % of the
 * colour; the front is full brightness. Point *size* no longer scales here —
 * perspective (the `scale` from `orbitProject`) owns that, so in 2-D every
 * point keeps its flat size. Pure, so it is unit-tested directly.
 *
 * *blend* scales the whole cue with the 2-D↔3-D camera blend: at 0 every depth
 * draws at full brightness (exactly the 2-D colours, so the toggle has no pop),
 * at 1 the full depth shade, linearly between.
 */
export function depthShade(depth: number, blend = 1): number {
  const b = clamp01(blend);
  const shade = DEPTH_SHADE_MIN + DEPTH_SHADE_SPAN * depth;
  // `shade * b + (1 - b)` (≡ 1 − (1 − shade)·b) keeps the blend-1 endpoint the
  // exact old value: 1 − (1 − shade) cancels catastrophically in floats.
  return shade * b + (1 - b);
}

/**
 * The sphere-scope wormhole: displace a NON-member that is nearer the camera
 * than the sphere's centre and whose screen position falls near the sphere's
 * projected circle radially OUT of the circle, fading it into a ghost. The
 * influence ramps up (with *depthReach*) the nearer the point is, and fades out
 * (with *Rinf = WORMHOLE_REACH·r*) the farther it is from the circle's centre,
 * so the rim band stays continuous with the untouched outside. Pure, so it is
 * unit-tested directly.
 *
 * *depth* and *centreDepth* are the normalised 0.05–1 depths `_trackScreen`
 * returns; the displacement itself is done in screen px on the projected circle.
 */
export function wormhole(
  p: { sx: number; sy: number; depth: number },
  circle: { cx: number; cy: number; r: number },
  centreDepth: number,
  depthReach: number,
): { sx: number; sy: number; ghost: number } {
  // A degenerate reach/circle cannot displace anything; skip before smoothstep
  // divides by zero.
  if (depthReach <= 0 || circle.r <= 0) {
    return { sx: p.sx, sy: p.sy, ghost: 0 };
  }
  // 0 at/behind the centre plane, 1 once the point is depthReach nearer.
  const s = smoothstep(0, depthReach, p.depth - centreDepth);
  if (s === 0) return { sx: p.sx, sy: p.sy, ghost: 0 };
  const Rinf = circle.r * WORMHOLE_REACH;
  const dx = p.sx - circle.cx;
  const dy = p.sy - circle.cy;
  const d = Math.hypot(dx, dy);
  // 1 inside the circle, fading to 0 at Rinf, so everything past it is untouched.
  const f = 1 - smoothstep(circle.r, Rinf, d);
  const w = s * f;
  if (w === 0) return { sx: p.sx, sy: p.sy, ghost: 0 };
  // Monotone: maps [0, Rinf] onto [r, Rinf] — inside points push to the rim band.
  const dOut = circle.r + (Rinf - circle.r) * (d / Rinf);
  // Away from the centre (fixed +x for a point exactly at it, so it is deterministic).
  const [ux, uy] = d < 1e-3 ? [1, 0] : [dx / d, dy / d];
  return {
    sx: p.sx + (circle.cx + ux * dOut - p.sx) * w,
    sy: p.sy + (circle.cy + uy * dOut - p.sy) * w,
    ghost: w,
  };
}

/**
 * Push a non-member that still projects inside a sphere scope's circle onto the
 * inner rim band, so it stays visible under the frosted glass instead of being
 * hidden outright. Monotone: maps the inside (`d < r`) onto
 * [(1 − RIM_BAND)·r, (1 − RIM_BAND + RIM_BAND·0.8)·r] — near the rim, clear of
 * the circle's centre — along the ray away from the centre (fixed +x for a
 * point exactly at it, so the result is deterministic). Points at or beyond the
 * circle are returned unchanged. Pure, so it is unit-tested directly.
 */
export function rimPush(
  p: { sx: number; sy: number },
  circle: { cx: number; cy: number; r: number },
): { sx: number; sy: number } {
  const dx = p.sx - circle.cx;
  const dy = p.sy - circle.cy;
  const d = Math.hypot(dx, dy);
  if (d >= circle.r) return { sx: p.sx, sy: p.sy };
  const dOut =
    circle.r * (1 - RIM_BAND) + circle.r * RIM_BAND * 0.8 * (d / circle.r);
  // Away from the centre (fixed +x for a point exactly at it, so deterministic).
  const [ux, uy] = d < 1e-3 ? [1, 0] : [dx / d, dy / d];
  return { sx: circle.cx + ux * dOut, sy: circle.cy + uy * dOut };
}

/**
 * Read the sphere-scope glass blur radius from a query string's ``glass``
 * parameter: finite and clamped to [0, MAX_GLASS_BLUR], else GLASS_BLUR_PX.
 * ``search`` may be ``"?glass=12"`` or ``"glass=12"``; 0 = no blur, tint only.
 */
export function readGlassBlur(search: string): number {
  const m = /(?:^|[?&])glass=([^&]+)/.exec(search);
  if (!m) return GLASS_BLUR_PX;
  const v = Number(m[1]);
  if (!Number.isFinite(v)) return GLASS_BLUR_PX;
  return Math.max(0, Math.min(MAX_GLASS_BLUR, v));
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
  /** Hovered track: bloom every dot sharing its value on the selection cursor. */
  hoveredBloomTrack: Track | null = null;
  tipReady = false;
  tipTimer = 0;
  /** Identity of the rendered tooltip card, so pan/zoom frames do not rebuild it. */
  private _tipKey = "";
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
  /** Seekbar clicked or dragged: seek the audition to *fraction* of the track. */
  onSeek: ((path: string, fraction: number) => void) | null = null;
  /** True while the pointer is dragging the card's seekbar. */
  private _seekDragging = false;
  lassoPoints: [number, number][] = [];
  boxSel: BoxSel | null = null;
  /** True while a lasso/box drag is in intersect (AND) mode — shown in cyan. */
  intersectMode = false;
  dragGhosts: Map<string, { wx: number; wy: number; wz?: number; folder: string }> | null =
    null;
  /** In-progress scope drag: screen corners, plus the 3-D sphere origin. */
  scopeDrag: {
    x0: number;
    y0: number;
    x1: number;
    y1: number;
    /** Sphere centre in cube coords (3-D drags only; the release re-projects it). */
    sphere?: { cx: number; cy: number; cz: number };
  } | null = null;
  /** Clickable lens tabs drawn around the open scope (screen rects). */
  scopeTabs: TabHit[] = [];
  /** Clickable lens tabs in the persistent top tabber. */
  lensTabs: TabHit[] = [];
  /** Clickable view tabs (the 3-D toggle) in the persistent top tabber. */
  viewTabs: TabHit[] = [];
  /** Clickable colour-mode tabs in the persistent top tabber. */
  colorTabs: TabHit[] = [];
  private _waitTimer = 0;
  _rafId = 0;
  /** Bottom of the top tabbers this frame, so status pills dock below the bloom chip. */
  private _topTabsBottom = TOP_TABS_BOTTOM;
  /** Perspective strength for the 3-D view; tunable live via ``?persp=…``. */
  private persp = readPerspective(
    typeof location !== "undefined" ? location.search : "",
  );
  /** Sphere-scope glass blur radius in CSS px; tunable live via ``?glass=…``. */
  private glassBlur = readGlassBlur(
    typeof location !== "undefined" ? location.search : "",
  );

  txHandles: TxHandle[] = [];
  _txBox: TxBox | null = null;
  _txWorld: TxWorld | null = null;
  /** False when both axes are analysis features — box interior is not a drag target. */
  _txAllowBoxDrag = false;

  /** Transient per-track positions used during animated mode transitions. */
  animPositions: Map<string, MapPos> | null = null;

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
  /**
   * The open sphere scope's member splats (3-D only), rendered in a second pass
   * above the scope's frosted glass; empty everywhere else.
   */
  private _scopeSpecs: PointSpec[] = [];
  /** Hover/selected dots, excluded from the merge so they read as single tracks. */
  private _crispDots: { sx: number; sy: number; idx: number; depth: number }[] = [];
  /** Reused occluder list for the 3-D glow pass (see `_glowOccluders`). */
  private _occluders: GlowOccluder[] = [];
  /** Cached map background colour for the sphere-scope glass tint. */
  private _mapBg: string | null = null;
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
    this.$tip.addEventListener("mousedown", (e) => this._onTipSeekDown(e));
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

  /**
   * The camera for this frame: the user's orbit (and perspective strength)
   * scaled by the 2-D↔3-D blend. At blend 0 it is head-on orthographic; at
   * blend 1 the full orbit + perspective. Everything that projects — tracks,
   * the cube, world positions for fit — goes through this one camera so the
   * transition moves them together.
   */
  private _camera(): Camera {
    return blendCamera(this.model.view3d, this.model.view3dBlend, this.persp);
  }

  /**
   * Project a world position into the view frame: the rotated (rx, ry) that
   * feeds the viewport transform, the rotated depth ``d``, and the perspective
   * ``scale``. In 2-D this is the raw position with depth 0 and scale 1. Shared
   * by ``_trackScreen`` (which turns it into screen px + a depth shade), the
   * sphere scope drawing, which needs the view frame and scale to size its
   * screen circle, and ``worldPosForTrack``.
   *
   * A position carrying a ``billboard`` anchor (a sphere-scope member's flat 2-D
   * layout) is drawn as a disc facing the camera: it takes the anchor's
   * projection and lays the member's world offset out at the anchor's
   * perspective scale, so a member at world offset ``r`` lands on the sphere's
   * projected circle edge and the disc never rotates with the orbit. The anchor
   * rides on the position itself, so it survives the scope closing. In 2-D the
   * flag is ignored — the flat position is drawn as-is.
   *
   * This is the ONE place that blends frames: when ``pos.from`` is set, this
   * frame and the frame of ``from`` (itself from-less) are both projected and
   * their results blended linearly with weight ``mix``, so a cross-frame move
   * (a scope member leaving the cube or landing on the billboard) is continuous
   * in screen space.
   */
  private _viewFrame(pos: ResolvedTrackPos): {
    rx: number;
    ry: number;
    d: number;
    scale: number;
  } {
    let frame: { rx: number; ry: number; d: number; scale: number };
    if (!this.model.shows3d) {
      frame = { rx: pos.wx, ry: pos.wy, d: 0, scale: 1 };
    } else if (pos.billboard) {
      const c = this._viewFrame({
        wx: pos.billboard.cx,
        wy: pos.billboard.cy,
        wz: pos.billboard.cz,
      });
      frame = {
        rx: c.rx + (pos.wx - pos.billboard.cx) * c.scale,
        ry: c.ry + (pos.wy - pos.billboard.cy) * c.scale,
        d: c.d,
        scale: c.scale,
      };
    } else {
      const cam = this._camera();
      const [rx, ry, d, scale] = orbitProject(
        cam,
        pos.wx,
        pos.wy,
        pos.wz,
        cam.persp,
      );
      frame = { rx, ry, d, scale };
    }
    if (pos.from && pos.mix !== undefined && pos.mix > 0) {
      const f = this._viewFrame({
        wx: pos.from.wx,
        wy: pos.from.wy,
        wz: pos.from.wz,
        billboard: pos.from.billboard,
      });
      const mix = pos.mix;
      return {
        rx: frame.rx * (1 - mix) + f.rx * mix,
        ry: frame.ry * (1 - mix) + f.ry * mix,
        d: frame.d * (1 - mix) + f.d * mix,
        scale: frame.scale * (1 - mix) + f.scale * mix,
      };
    }
    return frame;
  }

  /**
   * Project a track's world position to screen for drawing and hit testing. In
   * 3-D the unit cube is rotated about its centre first — the rotated (rx, ry)
   * feeds the existing viewport transform, and the rotated z becomes a depth in
   * [DEPTH_FLOOR, 1], 1 = nearest the viewer. Perspective scales the centred
   * offsets by `scale` (1 in 2-D, and at the cube centre in 3-D). In 2-D this
   * is plain `w2s` and depth/scale are 1. Renders 3-D whenever `shows3d` is
   * true, so the collapse tween keeps projecting until it lands on head-on.
   */
  private _trackScreen(pos: ResolvedTrackPos): {
    sx: number;
    sy: number;
    depth: number;
    scale: number;
  } {
    const { rx, ry, d, scale } = this._viewFrame(pos);
    const [sx, sy] = this.w2s(rx, ry);
    if (!this.model.shows3d) {
      return { sx, sy, depth: 1, scale: 1 };
    }
    const depth =
      DEPTH_FLOOR + (1 - DEPTH_FLOOR) * clamp01(0.5 + d / Math.sqrt(3));
    return { sx, sy, depth, scale };
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

  /**
   * Request one draw on the next frame. Requests coalesce: while a draw is pending
   * further calls are no-ops. (Cancel-and-requeue would let two animation loops that
   * each call this every frame cancel one another's draw — the canvas then freezes
   * until they both stop.)
   */
  scheduleDraw(): void {
    if (this._rafId) return;
    this._rafId = requestAnimationFrame(() => {
      this._rafId = 0;
      this.draw();
    });
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
    const is3d = this.model.shows3d;
    let best = -1,
      bestD = HIT_RADIUS,
      bestDepth = -Infinity;
    for (const p of this.positions) {
      const d = Math.hypot(p.sx - sx, p.sy - sy);
      if (d >= HIT_RADIUS) continue;
      // In 3-D the front-most candidate wins and distance breaks ties; in 2-D
      // the nearest candidate wins, as always.
      const better = is3d
        ? p.depth > bestDepth || (p.depth === bestDepth && d < bestD)
        : d < bestD;
      if (better) {
        bestD = d;
        bestDepth = p.depth;
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
    // In 3-D the glow is blitted into the scatter canvas (see the draw order
    // below); hide its own stacked canvas so it is not composited a second time.
    this.glow.canvas.style.visibility = this.model.shows3d ? "hidden" : "";
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
    this.viewTabs = [];
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

    // The 2-D grid and the 3-D cube frame crossfade during the camera tween:
    // at blend 1 only the cube, at blend 0 (or fully 2-D) only the grid. Each
    // draws at the opposite alpha, so the transition reads as one continuous
    // frame rather than a pop between the two guides.
    if (this.model.shows3d) {
      const blend = this.model.view3dBlend;
      if (blend > 0) this._drawCube(blend);
      if (blend < 1) this._drawGrid(w, h, 1 - blend);
    } else {
      this._drawGrid(w, h, 1);
    }
    this._drawLensGuide();
    this._drawDragGhosts();
    this._drawScatter(tracks);
    if (this.model.shows3d) {
      // The 3-D glow is composed into the scatter canvas between the blobs and
      // the crisp dots: it renders first, occluded by the blobs' depth field,
      // then is blitted over the blobs so a halo shows over deeper blobs yet
      // stays under its own and anything nearer (the crisp dots).
      this._renderBlobs(true);
      this._renderGlow(tracks);
      this._blitGlow();
      this._drawCrispDots(tracks);
    } else {
      this._renderPoints(tracks);
      this._renderGlow(tracks);
    }
    this._drawTransformBox();
    this._drawBoxSel();
    this._drawLasso();
    this._drawScope();
    this._drawTopTabs(w);
    // Docked status pills start below the tabbers, so a running analysis never
    // covers the persistent E/C feedback.
    this._drawModeBadge(w, h, this._topTabsBottom + 8);
    this._updateTooltip(tracks);
  }

  /** Lens background guide (Camelot rings + key labels, or the tempo spiral). */
  private _drawLensGuide(): void {
    const m = this.model;
    const lens = m.lens();
    // Eval maps are arbitrary precomputed layouts, so no lens guide fits.
    if (m.viewMode !== "embeddings" || m.evalMode || !lens.guide) return;
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
    const placed: { x: number; y: number; score: number; depth: number }[] = [];
    for (const p of field) {
      const t = m.trackByPath(p.path);
      if (!t) continue;
      const pos = this._resolveTrackWPos(t);
      if (!pos) continue;
      const { sx, sy, depth } = this._trackScreen(pos);
      // A non-member that projects inside the circle is glassed behind the
      // scope's disc, so its landmark glow must not bleed through the glass.
      if (this._behindScope(p.path, sx, sy)) continue;
      placed.push({ x: sx, y: sy, score: p.score, depth });
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
        depth: p.depth,
        radius: FIELD_GLOW_R * (0.55 + 0.45 * w),
        core: 0,
        color: FIELD_GLOW_RGB,
        alpha: 0.05 + 0.35 * w * w,
      };
    });
  }

  /** Glow points for the hovered navigation target / selection-bloom family. */
  collectHoverGlowPoints(tracks: Track[]): GlowPoint[] {
    if (this.hoverTarget === null && this.hoveredBloomTrack === null) {
      return [];
    }
    const points: GlowPoint[] = [];
    for (const pos of this.positions) {
      const track = tracks[pos.idx];
      if (!this._inHoverGlow(track)) continue;
      points.push({
        x: pos.sx,
        y: pos.sy,
        depth: pos.depth,
        radius: HOVER_GLOW_R,
        core: HOVER_GLOW_CORE,
        color: parseColor(this._hoverGlowColor(track)),
        alpha: HOVER_GLOW_ALPHA,
      });
    }
    return points;
  }

  /** Splat every glow for this frame in one additive overlay pass. In 3-D the
   * blobs (and crisp dots) in front of a halo are passed as occluders, so the
   * glow composits behind its own blob yet above deeper ones. */
  private _renderGlow(tracks: Track[]): void {
    const points = [
      ...this.collectSearchGlowPoints(),
      ...this.collectHoverGlowPoints(tracks),
    ];
    if (this.model.shows3d) {
      this.glow.render(points, this._glowOccluders());
    } else {
      this.glow.render(points);
    }
  }

  /**
   * The 3-D glow's occluders: every splat the metaball layer merges (the outer
   * map field plus the sphere scope's member field) plus the crisp
   * hover/selected dots, each with the depth a halo must sit behind.
   * Point-splat radii are already perspective-scaled, so the occlusion
   * footprint matches the blob on screen; crisp dots get a `DOT_R_SEL` disc.
   */
  private _glowOccluders(): GlowOccluder[] {
    const occ = this._occluders;
    occ.length = 0;
    for (const p of this._pointSpecs) {
      occ.push({ x: p.x, y: p.y, radius: p.radius, depth: p.depth ?? 0.5 });
    }
    for (const p of this._scopeSpecs) {
      occ.push({ x: p.x, y: p.y, radius: p.radius, depth: p.depth ?? 0.5 });
    }
    for (const c of this._crispDots) {
      occ.push({ x: c.sx, y: c.sy, radius: DOT_R_SEL, depth: c.depth });
    }
    return occ;
  }

  /* ── Grid + tick labels (adaptive to zoom) ──────────────── */

  private _drawGrid(w: number, h: number, alpha = 1): void {
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
    // The 2-D↔3-D crossfade fades the whole grid (overlay, lines, labels)
    // together; the frame's own save/restore resets it.
    ctx.globalAlpha = alpha;

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
    if (m.viewMode === "embeddings" && !m.evalMode) {
      ctx.fillText(m.lens().label, w / 2, h - 2);
    }

    ctx.restore();
  }

  /** 3-D frame replacing the 2-D grid: the 12 edges of the unit cube, projected
   * through the same orbit as the tracks. An edge whose average end-depth lies
   * behind the cube centre draws at half alpha, so the near face reads. The
   * *alpha* argument crossfades it with the 2-D grid during the camera tween. */
  private _drawCube(alpha = 1): void {
    const ctx = this.ctx;
    const corners = CUBE_CORNERS.map(([x, y, z]) =>
      this._trackScreen({ wx: x, wy: y, wz: z }),
    );
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.lineWidth = 1;
    for (const [a, b] of CUBE_EDGES) {
      const p0 = corners[a];
      const p1 = corners[b];
      ctx.strokeStyle =
        (p0.depth + p1.depth) / 2 < 0.5
          ? "rgba(255,255,255,0.07)"
          : "rgba(255,255,255,0.14)";
      ctx.beginPath();
      ctx.moveTo(p0.sx, p0.sy);
      ctx.lineTo(p1.sx, p1.sy);
      ctx.stroke();
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

  /**
   * World (normalised) position for fit / tooling; not clamped. In 3-D the
   * returned position is the *rotated* view-frame position (perspective
   * included), so fit-to-paths / fit-to-folder frame exactly what is on screen.
   */
  worldPosForTrack(t: Track): { wx: number; wy: number; wz?: number } | null {
    const pos = this._resolveTrackWPos(t);
    if (!pos) return null;
    if (this.model.shows3d) {
      const { rx, ry } = this._viewFrame(pos);
      return { wx: rx, wy: ry };
    }
    // Drop the billboard anchor and cross-frame blend: they only matter to the
    // 3-D projection (and in 2-D a from/mix entry draws at its own x/y anyway).
    return { wx: pos.wx, wy: pos.wy, wz: pos.wz };
  }

  /** Outer (pre-scope) position, for scope membership and anchors. */
  outerPosForTrack(t: Track): { wx: number; wy: number; wz?: number } | null {
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

  /** View-toggle id under *sx,sy* in the top tabber, else null. */
  topViewTabAt(sx: number, sy: number): string | null {
    return tabAt(this.viewTabs, sx, sy);
  }

  /** Colour-mode id under *sx,sy* in the top tabber, else null. */
  topColorTabAt(sx: number, sy: number): string | null {
    return tabAt(this.colorTabs, sx, sy);
  }

  private _resolveTrackWPos(t: Track): ResolvedTrackPos | null {
    const m = this.model;

    // During animation, interpolated positions always take priority
    if (this.animPositions) {
      const ap = this.animPositions.get(t.path);
      if (ap) {
        return {
          wx: ap.x,
          wy: ap.y,
          wz: ap.z,
          billboard: ap.billboard,
          from: ap.from
            ? {
                wx: ap.from.x,
                wy: ap.from.y,
                wz: ap.from.z,
                billboard: ap.from.billboard,
              }
            : undefined,
          mix: ap.mix,
        };
      }
    }

    const outer = this._outerTrackWPos(t);
    if (m.scope) {
      // A scope re-lays out its tracks; scopePositions holds only in-scope paths.
      // In 3-D with a sphere scope those flat {x, y} layouts are billboarded to
      // face the camera (see `_viewFrame`), the anchor carried by the position
      // itself; everywhere else the flag is ignored.
      const sp = m.scopePositions.get(t.path);
      if (sp) return { wx: sp.x, wy: sp.y, wz: sp.z, billboard: sp.billboard };
      // A track the scope lens could not place but whose outer position falls
      // inside the region would clutter the re-layout; hide it. Spheres are
      // tested in cube coordinates, so the outer wz rides along.
      if (outer && scopeContains(m.scope, outer.wx, outer.wy, outer.wz)) {
        return null;
      }
    }

    return outer;
  }

  /** Position ignoring any scope: the embedding lens or the tag axes. */
  private _outerTrackWPos(t: Track): ResolvedTrackPos | null {
    const m = this.model;
    if (m.viewMode === "embeddings") {
      const ep = m.embeddingPositions.get(t.path);
      if (ep) return { wx: ep.x, wy: ep.y, wz: ep.z };
      // No embedding → hide the track (don't fall back to tag positions)
      return null;
    }
    return this._tagPos(t);
  }

  /**
   * A non-member whose (post-wormhole) screen position lies inside the open
   * sphere scope's circle is glassed: `_drawScatter` pushes it to the inner rim
   * band under the frosted-glass disc, where it is visible but not hit, and its
   * search glow is dropped. Members (`scopePositions`) are the scope's own
   * contents and are never glassed. *circle* is the already-projected sphere
   * circle when the caller has one (the scatter loop's wormhole context); when
   * omitted it is recomputed, and the rule is a no-op unless a 3-D sphere scope
   * is shown.
   */
  private _behindScope(
    path: string,
    sx: number,
    sy: number,
    circle?: { cx: number; cy: number; r: number } | null,
  ): boolean {
    if (this.model.scopePositions.has(path)) return false;
    const c =
      circle ??
      (this.model.shows3d && this.model.scope?.shape === "sphere"
        ? this.scopeScreenCircle()
        : null);
    return c !== null && Math.hypot(sx - c.cx, sy - c.cy) < c.r;
  }

  private _drawScatter(tracks: Track[]): void {
    const m = this.model;
    const supported = this.points.supported;
    // Reuse the per-frame buffers: allocating three fresh containers for ~14k
    // tracks on every animation frame is pure GC pressure.
    this._pointSpecs.length = 0;
    this._scopeSpecs.length = 0;
    this._crispDots.length = 0;
    this._hitGrid.clear();
    const w = this.$canvas.clientWidth;
    const h = this.$canvas.clientHeight;
    let anyFiltered = false;
    // Sphere scope in 3-D: the disc is a frosted-glass backdrop for its members,
    // so a non-member in front of the sphere is wormholed radially out of the
    // projected circle, ghosting as it goes; one that still projects inside the
    // circle lies behind the glass and is pushed to the inner rim band
    // (`rimPush`) — visible under the glass, but not hittable. Members
    // (`scopePositions`) are never displaced or glassed.
    let wormholeCtx: {
      circle: { cx: number; cy: number; r: number };
      centreDepth: number;
      depthReach: number;
    } | null = null;
    if (m.shows3d && m.scope?.shape === "sphere") {
      const circle = this.scopeScreenCircle();
      if (circle) {
        const cz = m.scope.cz ?? 0.5;
        wormholeCtx = {
          circle,
          centreDepth: this._trackScreen({
            wx: m.scope.cx,
            wy: m.scope.cy,
            wz: cz,
          }).depth,
          depthReach: depthUnits(0.25 * m.scope.r),
        };
      }
    }
    tracks.forEach((t, i) => {
      if (!m.isVisible(t)) {
        anyFiltered = true;
        return;
      }
      const pos = this._resolveTrackWPos(t);
      if (!pos) return;
      const { sx: rawSx, sy: rawSy, depth, scale } = this._trackScreen(pos);
      let sx = rawSx;
      let sy = rawSy;
      let ghost = 0;
      if (wormholeCtx && !m.scopePositions.has(t.path)) {
        const w = wormhole(
          { sx, sy, depth },
          wormholeCtx.circle,
          wormholeCtx.centreDepth,
          wormholeCtx.depthReach,
        );
        sx = w.sx;
        sy = w.sy;
        ghost = w.ghost;
      }
      // A non-member that still projects inside the circle lies behind the
      // frosted glass: push it onto the inner rim band so it stays visible
      // (blurred by the disc) instead of vanishing, and draw it at full
      // colour — but, like a heavy ghost, never hittable or selectable.
      let glassed = false;
      if (wormholeCtx && this._behindScope(t.path, sx, sy, wormholeCtx.circle)) {
        const rim = rimPush({ sx, sy }, wormholeCtx.circle);
        sx = rim.sx;
        sy = rim.sy;
        glassed = true;
        ghost = 0;
      }
      // An off-screen dot cannot be seen or hit, so skip it entirely; while
      // zoomed in this is most of the library.
      if (
        sx < -CULL_MARGIN ||
        sx > w + CULL_MARGIN ||
        sy < -CULL_MARGIN ||
        sy > h + CULL_MARGIN
      )
        return;
      // Heavily ghosted dots are rendered but not hittable or selectable, so
      // hover, click, lasso and the hover glow ignore them. Glassed points are
      // not hittable either, even though they draw at full colour.
      if (!glassed && ghost < 0.5) this.positions.push({ idx: i, sx, sy, depth });
      if (!supported) {
        this._dot(sx, sy, t, i);
        return;
      }
      if (!glassed && ghost < 0.5) {
        const key = `${Math.round(sx / HIT_CELL)},${Math.round(sy / HIT_CELL)}`;
        const cell = this._hitGrid.get(key);
        if (cell) cell.push({ idx: i, sx, sy, depth });
        else this._hitGrid.set(key, [{ idx: i, sx, sy, depth }]);
      }
      const crisp = m.selected.has(t.path) || this.hoveredIdx === i;
      if (crisp && !glassed && ghost < 0.5) {
        this._crispDots.push({ sx, sy, idx: i, depth });
      } else {
        // A heavily ghosted crisp dot merges back into the field as a silhouette.
        const spec = this._pointSpec(sx, sy, t, depth, scale, ghost);
        // A sphere scope's members render above its glass, so their splats go
        // to the second field pass (`_renderBlobs`); everyone else — glassed
        // points included — merges with the outer map.
        if (wormholeCtx && m.scopePositions.has(t.path)) {
          this._scopeSpecs.push(spec);
        } else {
          this._pointSpecs.push(spec);
        }
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

  /**
   * One splat for a track, with its colour, search/external fade, and (in 3-D)
   * the depth shade: nearer points draw brighter, scaled by the camera blend so
   * the cue fades in with the 2-D↔3-D transition instead of popping at toggle
   * start. Size comes from the perspective `scale` (1 in 2-D), so a nearer
   * point also draws larger.
   */
  private _pointSpec(
    sx: number,
    sy: number,
    track: Track,
    depth: number,
    scale: number,
    ghost = 0,
  ): PointSpec {
    const faded =
      this.searchResults !== null && !this.searchResults.has(track.path);
    const colour = this.trackColor(track);
    const shows3d = this.model.shows3d;
    const [r, g, b] = parseColor(colour);
    const shade = depthShade(depth, this.model.view3dBlend);
    return {
      x: sx,
      y: sy,
      radius: POINT_SPLAT_R * scale,
      color: shows3d
        ? [r * shade, g * shade, b * shade]
        : [r, g, b],
      alpha:
        (faded ? 0.15 : this.model.isExternalTrack(track) ? 0.5 : 1) *
        (1 - GHOST_FADE * ghost),
      // A grey dot contributes shape but almost no colour, so a blob it merges
      // into keeps the hue of its coloured members instead of desaturating.
      weight: colour === CONFIDENT_MISS_COLOR ? GREY_COLOUR_WEIGHT : 1,
      // A ghost sinks to the back, so the front-depth pre-pass can never
      // occlude the scope contents drawn behind it.
      depth: depth * (1 - ghost) + DEPTH_FLOOR * ghost,
    };
  }

  /**
   * The map's background colour, read once from the page's `--bg` custom
   * property. The scatter canvas is transparent over the page background, so
   * the sphere scope's glass tint must repaint the same colour; computed-style
   * reads are too slow to repeat every frame, hence the cache. Falls back to
   * the CSS default when the property is unset (e.g. headless tests).
   */
  private mapBackground(): string {
    if (this._mapBg === null) {
      this._mapBg =
        getComputedStyle(document.documentElement)
          .getPropertyValue("--bg")
          .trim() || "#1a1a2e";
    }
    return this._mapBg;
  }

  /**
   * Render the metaball field and blit it over the grid; crisp dots come later.
   *
   * With a sphere scope in 3-D the disc is frosted glass: the outer (non-member)
   * field is blitted first, a blurred + tinted circle covers it (`_drawGlass`),
   * then the scope's member field goes over it — the glass softens what sits
   * behind the disc (the grid/cube and the rim-pushed non-members) while the
   * members stay sharp on top. Everywhere else the whole field renders in one
   * pass, exactly as before.
   */
  private _renderBlobs(depthAware: boolean): void {
    if (!this.points.supported) return;
    const w = this.$canvas.clientWidth,
      h = this.$canvas.clientHeight;
    const ctx = this.ctx;
    const circle = this.model.shows3d ? this.scopeScreenCircle() : null;
    if (circle) {
      this.points.render(this._pointSpecs, depthAware);
      ctx.drawImage(this.points.canvas, 0, 0, w, h);
      this._drawGlass(circle, w, h);
      this.points.render(this._scopeSpecs, depthAware);
      ctx.drawImage(this.points.canvas, 0, 0, w, h);
      return;
    }
    this.points.render(this._pointSpecs, depthAware);
    ctx.drawImage(this.points.canvas, 0, 0, w, h);
  }

  /**
   * The sphere scope's frosted-glass disc: clip to the projected circle, blur
   * the canvas bitmap under it (the grid + the outer blobs), then tint the disc
   * with the map background. The blur softens what the disc covers, so the tint
   * alone — at higher alpha — must do the hiding where the canvas `filter` is
   * unavailable (Safari, and headless fakes). The scope's member splats render
   * after and stay sharp.
   */
  private _drawGlass(
    circle: { cx: number; cy: number; r: number },
    w: number,
    h: number,
  ): void {
    const ctx = this.ctx;
    ctx.save();
    ctx.beginPath();
    ctx.arc(circle.cx, circle.cy, circle.r, 0, Math.PI * 2);
    ctx.clip();
    const blurred = this.glassBlur > 0 && typeof ctx.filter === "string";
    if (blurred) {
      // The canvas bitmap is device-pixel sized and drawn under the dpr
      // transform, so a w×h dest copies it 1:1; the self-copy is what blurs.
      ctx.filter = `blur(${this.glassBlur * (devicePixelRatio || 1)}px)`;
      ctx.drawImage(this.$canvas, 0, 0, w, h);
      ctx.filter = "none";
    }
    ctx.globalAlpha = blurred ? GLASS_TINT : GLASS_TINT_FLAT;
    ctx.fillStyle = this.mapBackground();
    ctx.fill();
    ctx.restore();
  }

  /**
   * Blit the merged field over the grid, then draw the crisp hover/selected dots
   * on top — they are excluded from the field so a single track never dissolves
   * into a blob. Falls through to nothing when the layer is unsupported, since
   * `_drawScatter` already drew every dot.
   */
  private _renderPoints(tracks: Track[]): void {
    this._renderBlobs(this.model.shows3d);
    this._drawCrispDots(tracks);
  }

  /** Blit the (hidden) glow canvas over the blobs — the 3-D composition step. */
  private _blitGlow(): void {
    const w = this.$canvas.clientWidth,
      h = this.$canvas.clientHeight;
    this.ctx.drawImage(this.glow.canvas, 0, 0, w, h);
  }

  /** The crisp hover/selected dots on top of the merged field. */
  private _drawCrispDots(tracks: Track[]): void {
    if (!this.points.supported) return;
    for (const c of this._crispDots) this._dot(c.sx, c.sy, tracks[c.idx], c.idx);
  }

  /* ── Hover glow pass (navigation targets + selection bloom) ── */

  /** Tracks that get the halo: sidebar/card hover, or the hovered dot's bloom family. */
  private _inHoverGlow(track: Track): boolean {
    if (this.hoverTarget !== null)
      return this.model.trackMatchesHoverTarget(track, this.hoverTarget);
    const hovered = this.hoveredBloomTrack;
    if (hovered === null) return false;
    return this.model.trackBloomsWith(track, hovered);
  }

  /** Halo colour: cluster swatch, folder colour, or the neutral accent for key/BPM. */
  private _hoverGlowColor(track: Track): string {
    const t = this.hoverTarget;
    if (t?.kind === "cluster") return clusterColor(t.id);
    if (t?.kind === "paths") return t.color ?? TARGET_GLOW_COLOR;
    if (t?.kind === "key" || t?.kind === "bpm" || t?.kind === "track")
      return TARGET_GLOW_COLOR;
    // A navigation-target hover uses the active dot colour; the selection bloom
    // glows in its own variable's colour, so the family reads even when the map
    // is coloured by something else.
    return this.trackColor(track, t ? this.model.colorMode : this.model.selectionBloom);
  }

  /* ── Drag ghosts (original positions while dragging) ────── */

  private _drawDragGhosts(): void {
    const ghosts = this.dragGhosts;
    if (!ghosts || !ghosts.size) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.setLineDash([2, 3]);
    ctx.lineWidth = 1.5;
    for (const { wx, wy, wz, folder } of ghosts.values()) {
      const { sx, sy } = this._trackScreen({ wx, wy, wz });
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
   * yet, or no cached features. The predicted-folder mode is the exception: it
   * greys only a track the head has no prediction for, so a track the head keeps
   * in its own folder still takes that folder's colour.
   */
  trackColor(track: Track, mode: ColorMode = this.model.colorMode): string {
    const m = this.model;
    switch (mode) {
      case "head": {
        const projected = m.folderPredictions.get(track.path);
        // No prediction at all — the head has nothing to say about this track — is
        // grey. A prediction equal to the track's current folder still takes that
        // folder's colour, so the map never hides a track the head places.
        if (projected === undefined) return CONFIDENT_MISS_COLOR;
        // "Confident only" greys a match too weak to count; "All nearest"
        // takes every track's nearest projected folder.
        const confident =
          !m.headColorConfident ||
          m.folderPredictionMatches.has(track.path);
        return confident ? dirColor(projected) : CONFIDENT_MISS_COLOR;
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
      if (!t || !m.isVisible(t)) continue;
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
      const fw = Math.max(barH, barW * clamp01(prog.fraction));
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
      const fw = Math.max(barH, barW * clamp01(fraction));
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

  /* ── Scope picking (3-D) ───────────────────────────────── */

  /**
   * Pick a cube position for a 3-D scope gesture at *sx,sy*: the nearest
   * on-screen dot within `SNAP_PX` (its own cube position, so membership is
   * orbit-invariant), preferring the front-most among dots within 2 px of each
   * other; otherwise the cursor unprojected onto the cube-centre depth plane.
   * Pure projection, so the tests exercise the geometry directly.
   */
  pickCursorWorld3D(
    sx: number,
    sy: number,
  ): { wx: number; wy: number; wz: number; snapped: boolean } {
    let bestD = Infinity;
    for (const p of this.positions) {
      const d = Math.hypot(p.sx - sx, p.sy - sy);
      if (d <= SNAP_PX && d < bestD) bestD = d;
    }
    if (isFinite(bestD)) {
      let bestIdx = -1;
      let bestDepth = -Infinity;
      let bestDist = Infinity;
      for (const p of this.positions) {
        const d = Math.hypot(p.sx - sx, p.sy - sy);
        // A dot within 2 px of the nearest is a tie: the front-most wins, and
        // distance breaks a remaining depth tie.
        if (d > bestD + 2) continue;
        if (p.depth > bestDepth || (p.depth === bestDepth && d < bestDist)) {
          bestIdx = p.idx;
          bestDepth = p.depth;
          bestDist = d;
        }
      }
      const track = this.model.tracks[bestIdx];
      const pos = track ? this._resolveTrackWPos(track) : null;
      if (pos) {
        return { wx: pos.wx, wy: pos.wy, wz: pos.wz ?? 0.5, snapped: true };
      }
    }
    const cam = this._camera();
    const [rx, ry] = this.s2w(sx, sy);
    const [wx, wy, wz] = orbitUnproject(cam, rx, ry, 0, cam.persp);
    return { wx, wy, wz, snapped: false };
  }

  /**
   * World (cube) radius of a sphere centred at (cx,cy,cz) whose projected
   * screen radius is *px*: invert the perspective scale at the centre against
   * the viewport's pixels-per-world-unit. The exact inverse of the sphere
   * drawing in `_drawScope`, so the two stay consistent.
   */
  worldRadiusForPx(cx: number, cy: number, cz: number, px: number): number {
    const { rx, ry, scale } = this._viewFrame({ wx: cx, wy: cy, wz: cz });
    const [s0] = this.w2s(rx, ry);
    const [s1] = this.w2s(rx + 1, ry);
    const zoomPxPerWorld = Math.abs(s1 - s0);
    return px / (zoomPxPerWorld * scale);
  }

  /**
   * The open sphere scope's screen circle (centre + radius px), or null for a
   * circle/box scope (whose screen image is not a circle). `_drawScope` and the
   * click / pointer-near tests share it, so the hit region always matches the
   * drawn outline.
   */
  scopeScreenCircle(): { cx: number; cy: number; r: number } | null {
    const scope = this.model.scope;
    if (!scope || scope.shape !== "sphere") return null;
    const cz = scope.cz ?? 0.5;
    const { sx, sy } = this._trackScreen({ wx: scope.cx, wy: scope.cy, wz: cz });
    const { rx, ry, scale } = this._viewFrame({
      wx: scope.cx,
      wy: scope.cy,
      wz: cz,
    });
    const [edgeX] = this.w2s(rx + scope.r * scale, ry);
    return { cx: sx, cy: sy, r: Math.abs(edgeX - sx) };
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
      if (this.model.shows3d && drag.sphere) {
        // The 3-D drag preview is the sphere the release will open: centre at
        // the drag origin's snapped cursor, radius = the drag distance in px.
        const { sx, sy } = this._trackScreen({
          wx: drag.sphere.cx,
          wy: drag.sphere.cy,
          wz: drag.sphere.cz,
        });
        const r = Math.max(1, Math.hypot(drag.x1 - drag.x0, drag.y1 - drag.y0));
        ctx.beginPath();
        ctx.arc(sx, sy, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      } else {
        // The drag is a screen-space box; draw it as-is.
        const x = Math.min(drag.x0, drag.x1);
        const y = Math.min(drag.y0, drag.y1);
        ctx.beginPath();
        ctx.rect(x, y, Math.max(1, Math.abs(drag.x1 - drag.x0)), Math.max(1, Math.abs(drag.y1 - drag.y0)));
        ctx.fill();
        ctx.stroke();
      }
    }
    if (scope) {
      if (scope.shape === "sphere") {
        const circle = this.scopeScreenCircle();
        if (circle) {
          ctx.beginPath();
          ctx.arc(circle.cx, circle.cy, circle.r, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
          ctx.setLineDash([]);
          // A faint inner ring makes the disc read as a sphere.
          ctx.save();
          ctx.globalAlpha = 0.35;
          ctx.beginPath();
          ctx.arc(circle.cx, circle.cy, circle.r * 0.5, 0, Math.PI * 2);
          ctx.stroke();
          ctx.restore();
          // The lens tabs sit just outside the region, above unless it is near the top.
          const above = circle.cy - circle.r - 16 > 26;
          const tabCy = above ? circle.cy - circle.r - 16 : circle.cy + circle.r + 16;
          this.scopeTabs = this._drawTabs(
            scopeLenses(),
            scope.lensId,
            circle.cx,
            tabCy,
          );
        }
      } else {
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
    // Eval maps have no lenses or colour modes; the toolbar owns the controls.
    if (m.evalMode) return;
    this.lensTabs = this._drawTabs(
      availableLenses(m.genreFullEnabled),
      m.activeBaseLensId(),
      w / 2,
      16,
    );
    // The 3-D toggle tab sits just right of the lens row. It highlights while
    // the client-side orbit is armed even on a lens that cannot show it (only a
    // projection lens is actually 3-D), so the toggle state is always visible.
    const ctx = this.ctx;
    ctx.font = "11px monospace";
    const viewTabW = ctx.measureText("3D").width + 18;
    this.viewTabs = this._drawTabs(
      VIEW_TABS,
      m.view3d.on ? "3d" : "",
      this.lensTabs[this.lensTabs.length - 1].x1 + 14 + viewTabW / 2,
      16,
    );
    this.colorTabs = this._drawTabs(colorModeTabs(), m.colorMode, w / 2, 40);
    // The selection-bloom cursor is a second marker on the colour row, separate
    // from the active-mode highlight: **D** moves it, a colour-tab click sets
    // both, so the map can be coloured by one variable and bloom by another. Its
    // ring borrows the hovered class's colour, so it reads as that class.
    const cursorTab = this.colorTabs.find((t) => t.id === m.selectionBloom);
    const hovered = this.hoveredBloomTrack;
    const cursorColor = hovered
      ? this.trackColor(hovered, m.selectionBloom)
      : CURSOR_COLOR;
    this._drawTabCursor(cursorTab, cursorColor);
    this._topTabsBottom = TOP_TABS_BOTTOM;
    if (cursorTab && hovered) {
      const bottom = this._drawBloomChip(cursorTab, hovered, cursorColor, w);
      if (bottom !== null) this._topTabsBottom = bottom;
    }
  }

  /** Draw the selection-bloom cursor ring around a colour tab, in *color*. */
  private _drawTabCursor(tab: TabHit | undefined, color: string): void {
    if (!tab) return;
    const ctx = this.ctx;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    this._roundRect(
      tab.x0 - 3,
      tab.y0 - 3,
      tab.x1 - tab.x0 + 6,
      tab.y1 - tab.y0 + 6,
      7,
    );
    ctx.stroke();
    ctx.restore();
  }

  /** The hovered dot's class path segments on the bloom variable, or null. */
  private _bloomClassSegments(track: Track): string[] | null {
    const m = this.model;
    switch (m.selectionBloom) {
      case "folder":
        return this._folderSegments(track.folder ?? "");
      case "head": {
        const p = m.folderPredictions.get(track.path);
        return p !== undefined ? this._folderSegments(p) : null;
      }
      case "cluster": {
        const id = m.clusterAssignments.get(track.path);
        return id === undefined
          ? null
          : [m.clusterNames.get(id) ?? `Cluster ${id}`];
      }
      case "tempo":
        return track.bpm != null && Number.isFinite(track.bpm)
          ? [`${Math.round(track.bpm)} BPM`]
          : null;
      case "key":
        return track.musical_key ? [track.musical_key] : null;
    }
  }

  /** A virtual folder path as display segments (root id → its tree label). */
  private _folderSegments(folder: string): string[] {
    return displayFolderPath(folder).split("/").filter(Boolean);
  }

  /**
   * The bloom chip under the cursor: the hovered class and, where the variable
   * has a range, its semantic meaning (`+2 parent directories`, `±13% BPM`). For a
   * folder (or predicted folder) the path is drawn segment by segment and the
   * segment the range climbed to is highlighted — deeper segments fade to grey, so
   * the chip shows *which ancestor* the bloom reached. Its border borrows the class
   * colour, so the chip and the halo agree. Returns the chip's bottom edge, so the
   * status pills dock below it.
   */
  private _drawBloomChip(
    tab: TabHit,
    track: Track,
    color: string,
    canvasW: number,
  ): number | null {
    const m = this.model;
    const segs = this._bloomClassSegments(track);
    if (!segs || !segs.length) return null;
    const range = bloomRangeLabel(m.selectionBloom, m.bloomRange());
    // Only the folder climb has ancestors to point at; other variables are a
    // single segment that is always the highlight.
    const levels =
      bloomRangeKey(m.selectionBloom) === "folder"
        ? bloomFolderLevels(m.bloomRange())
        : 0;
    const anchor = Math.max(0, segs.length - 1 - levels);

    const ctx = this.ctx;
    const padX = 10;
    const h = 20;
    const gap = 6;
    ctx.save();
    ctx.textBaseline = "middle";
    // Each segment is measured in its own font, since the anchor is bold.
    const segW = segs.map((s, i) => {
      ctx.font = i === anchor ? "bold 11px monospace" : "11px monospace";
      return ctx.measureText(s).width;
    });
    const sepW = ctx.measureText("/").width + gap * 2;
    const rangeText = range ? `·  ${range}` : "";
    const rangeW = rangeText ? ctx.measureText(rangeText).width + gap * 2 : 0;
    const chipW =
      segW.reduce((a, b) => a + b, 0) +
      sepW * (segs.length - 1) +
      rangeW +
      padX * 2;
    const cx = (tab.x0 + tab.x1) / 2;
    const x = Math.max(4, Math.min(canvasW - chipW - 4, cx - chipW / 2));
    const y = tab.y1 + 6;
    const mid = y + h / 2 + 0.5;

    this._roundRect(x, y, chipW, h, 6);
    ctx.fillStyle = "rgba(18,20,24,0.92)";
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    ctx.textAlign = "left";
    let tx = x + padX;
    segs.forEach((seg, i) => {
      if (i > 0) {
        ctx.font = "11px monospace";
        ctx.fillStyle = "rgba(150,158,168,0.55)";
        ctx.fillText("/", tx + gap, mid);
        tx += sepW;
      }
      ctx.font = i === anchor ? "bold 11px monospace" : "11px monospace";
      ctx.fillStyle = this._bloomSegColor(i, anchor, color);
      ctx.fillText(seg, tx, mid);
      tx += segW[i];
    });
    if (rangeText) {
      ctx.font = "11px monospace";
      ctx.fillStyle = "rgba(170,180,192,0.7)";
      ctx.fillText(rangeText, tx + gap, mid);
    }
    ctx.restore();
    return y + h;
  }

  /** Chip segment colour: the anchor is the class colour, deeper segments fade. */
  private _bloomSegColor(i: number, anchor: number, color: string): string {
    if (i === anchor) return color;
    if (i > anchor) {
      // One below the anchor is "a bit faded", further below greys out.
      const depth = i - anchor;
      return `rgba(150,158,168,${Math.max(0.3, 0.65 - 0.3 * (depth - 1)).toFixed(2)})`;
    }
    return "rgba(230,236,242,0.92)";
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
      // The old target elements are gone; a stale hover would keep glowing.
      this.hoverTarget = null;
      this._previewRow = null;
    }
    // Playback advances far faster than the card key changes, so the seekbar fill,
    // playhead, and readout are moved directly each frame instead of rebuilding the
    // card. Only the auditioning card follows the live position; every other card keeps
    // the zeroed values it was rendered with.
    if (card.seek.playing)
      updateSeekProgress(this.$tip, this.model.previewTime, card.seek.duration);
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

  /** The path of the card under the pointer: the pinned track, else the hover. */
  private _cardPath(): string | null {
    if (this.inspect) return this.inspect.path;
    if (this.hoveredPath) return this.hoveredPath;
    if (this.hoveredIdx >= 0)
      return this.model.tracks[this.hoveredIdx]?.path ?? null;
    return null;
  }

  /** Fraction of the track at the pointer's x within the seekbar track. */
  private _seekFraction(track: HTMLElement, clientX: number): number {
    const r = track.getBoundingClientRect();
    return clamp01((clientX - r.left) / Math.max(1, r.width));
  }

  /** Start a seek drag on the card's seekbar; the first move seeks too. */
  private _onTipSeekDown(e: MouseEvent): void {
    const track = (e.target as HTMLElement).closest<HTMLElement>("[data-seek]");
    if (!track) return;
    const path = this._cardPath();
    if (!path) return;
    e.preventDefault();
    this._seekDragging = true;
    document.addEventListener("mousemove", this._onDocSeekMove);
    document.addEventListener("mouseup", this._onDocSeekUp);
    this.onSeek?.(path, this._seekFraction(track, e.clientX));
  }

  // The track is re-queried on each move: a seek rebuilds the card (its play
  // state is part of the re-render key), so a held element would go stale.
  private _onDocSeekMove = (e: MouseEvent): void => {
    if (!this._seekDragging) return;
    const track = this.$tip.querySelector<HTMLElement>("[data-seek]");
    const path = this._cardPath();
    if (!track || !path) return;
    this.onSeek?.(path, this._seekFraction(track, e.clientX));
  };

  private _onDocSeekUp = (): void => {
    this._seekDragging = false;
    document.removeEventListener("mousemove", this._onDocSeekMove);
    document.removeEventListener("mouseup", this._onDocSeekUp);
  };

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

  /**
   * Cursor-adjacent readout while a held key + scroll adjusts a meter (**M**
   * preview volume, **D** bloom range). A body-level fixed element, so it
   * follows the pointer without touching layout, and it fades after a short pause
   * so it never lingers.
   */
  private _meterHuds = new Map<string, HTMLElement>();
  private _meterTimers = new Map<string, number>();

  /** **M + scroll** readout: the preview-volume level. */
  showVolumeHud(clientX: number, clientY: number): void {
    const pct = Math.round(this.model.previewVolume * 100);
    this._showMeterHud("volume", "volume-hud", `Volume ${pct}%`, pct, clientX, clientY);
  }

  /** **D + scroll** readout: the selection-bloom range, named per variable. */
  showBloomHud(clientX: number, clientY: number): void {
    const m = this.model;
    const label = bloomRangeLabel(m.selectionBloom, m.bloomRange()) ?? "no range";
    // A variable with no range (cluster) has nothing to slide, so the panel shows
    // the label without a slider rather than a bar that cannot move.
    const pct = bloomRangeLabel(m.selectionBloom, m.bloomRange())
      ? Math.round(m.bloomRange() * 100)
      : null;
    this._showMeterHud("bloom", "bloom-hud", `Bloom · ${label}`, pct, clientX, clientY);
  }

  private _showMeterHud(
    key: string,
    cls: string,
    label: string,
    pct: number | null,
    clientX: number,
    clientY: number,
  ): void {
    let hud = this._meterHuds.get(key);
    if (!hud) {
      hud = document.createElement("div");
      hud.id = cls;
      hud.className = `${cls} hidden`;
      const labelEl = document.createElement("div");
      labelEl.className = `${cls}-label`;
      hud.appendChild(labelEl);
      document.body.appendChild(hud);
      this._meterHuds.set(key, hud);
    }
    (hud.querySelector(`.${cls}-label`) as HTMLElement).textContent = label;
    // No pct → no slider: a variable with no range has nothing to slide.
    let track = hud.querySelector<HTMLElement>(`.${cls}-track`);
    if (pct === null) {
      track?.remove();
    } else {
      if (!track) {
        track = document.createElement("div");
        track.className = `${cls}-track`;
        track.appendChild(document.createElement("i"));
        hud.appendChild(track);
      }
      (track.querySelector("i") as HTMLElement).style.width = `${pct}%`;
    }
    hud.classList.remove("hidden");
    hud.style.left = `${clientX + 14}px`;
    hud.style.top = `${clientY + 14}px`;
    clearTimeout(this._meterTimers.get(key));
    this._meterTimers.set(
      key,
      setTimeout(() => hud!.classList.add("hidden"), 700) as unknown as number,
    );
  }

  clearHover(): void {
    clearTimeout(this.tipTimer);
    this.tipReady = false;
    this.hoveredIdx = -1;
    this.hoveredPath = null;
    this.hoveredBloomTrack = null;
    // The pinned inspect card outlives the pointer leaving the canvas.
    if (!this.inspect) {
      this.$tip.classList.add("hidden");
      this.$coords.classList.add("hidden");
    }
  }
}
