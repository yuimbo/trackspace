import {
  COLOR_MODES,
  GENRE_FULL_LENS,
  availableLenses,
  defaultMapLensId,
  getLens,
  legacyToGenreLensId,
  nextLensId,
  prevLensId,
  stepColorMode,
  trackTempoNorm,
  type ColorMode,
  type Lens,
  type LensPreset,
} from "./lib/lenses";
import {
  BLOOM_RANGE_KEYS,
  DEFAULT_BLOOM_RANGE,
  bloomFolderLevels,
  bloomKeySteps,
  bloomRangeKey,
  bloomRangeLabel,
  bloomTempoPct,
  camelotSteps,
  folderAncestor,
  type BloomRangeKey,
} from "./lib/bloom";
import { clusterColor, dirTagColors, keyColor, tempoColor } from "./lib/colors";
import { MAX_PITCH, type Orbit } from "./lib/camera3d";
import { displayFolderPath } from "./lib/path-presenter";
import type { EvalDims } from "./lib/layout-eval";
import { clamp01 } from "./lib/util";
import clusterWeights from "./lib/cluster-weights.json";
import { GENRE_FULL_WEIGHTS } from "./lib/lenses";

/** Cached librosa descriptors ([0,1]) when the fingerprint has a features row. */
export interface TrackAudioFeatures {
  tempo: number;
  /** Circle-of-fifths angle mapped to [0, 1). */
  key: number;
  mode: number;
  energy: number;
  danceability: number;
}

/** Coarse track structure from the ``structure`` stage; seconds, or null. */
export interface TrackStructure {
  /** Where the audio starts after lead-in silence (``0`` when it starts immediately). */
  intro_start?: number | null;
  intro_end: number | null;
  peak_start: number | null;
  peak_end: number | null;
  break_start: number | null;
  break_end: number | null;
  /** Track length in seconds, probed from the file header; the seekbar timeline. */
  duration?: number | null;
  /** True when the track's excerpts disagree (intro vs drop). */
  genre_spanning?: boolean;
}

/** In-flight phase of a library-wide singleton stage (supervise/cluster). */
export interface QueueStageStatus {
  phase: string;
  done: number;
  total: number;
  at: number;
}

/** Counts for one job kind, as reported by ``/api/jobs/status``. */
export interface QueueKindStatus {
  label: string;
  pending: number;
  running: number;
  done: number;
  failed: number;
  total: number;
  finished: number;
  outstanding: number;
  last?: {
    path: string | null;
    ok: boolean | null;
    detail: Record<string, unknown>;
    at: number;
  };
  /** Present only while a singleton stage is mid-run. */
  stage?: QueueStageStatus;
}

/** Whole-queue snapshot; ``order`` lists kinds in pipeline order. */
export interface QueueStatus {
  kinds: Record<string, QueueKindStatus>;
  order: string[];
  outstanding: number;
  total: number;
  finished: number;
  active: boolean;
  epoch: number;
  models?: { maest_ready: boolean; maest_error: string | null };
  clusters?: { available: boolean; revision: string | null };
}

export interface Track {
  path: string;
  filename: string;
  folder: string;
  tags: Record<string, number>;
  artist: string;
  title: string;
  fingerprint?: string | null;
  /** From cached librosa features when available (after embedding/analysis pass). */
  bpm?: number | null;
  musical_key?: string | null;
  /** Same pipeline as embedding "Audio features"; snake_case from API. */
  audio_features?: TrackAudioFeatures | null;
}

/** Librosa analysis dimensions (reserved; not canvas-writable; listed under Features). */
export const LIBROSA_FEATURE_AXIS_IDS = [
  "tempo",
  "key",
  "mode",
  "energy",
  "danceability",
] as const;

const _LIBROSA_FEATURE_SET = new Set<string>(LIBROSA_FEATURE_AXIS_IDS);

const LEGACY_LIBROSA_AXIS_MAP: Record<string, string> = {
  "audio:tempo": "tempo",
  "audio:key": "key",
  "audio:mode": "mode",
  "audio:energy": "energy",
  "audio:danceability": "danceability",
};

function _migrateLibrosaAxisId(axis: string | null): string | null {
  if (!axis) return null;
  return LEGACY_LIBROSA_AXIS_MAP[axis] ?? axis;
}

function _migrateFilterRanges(
  raw: Record<string, [number, number]> | undefined,
): Record<string, [number, number]> {
  if (!raw) return {};
  const out: Record<string, [number, number]> = {};
  for (const [k, v] of Object.entries(raw)) {
    const nk = LEGACY_LIBROSA_AXIS_MAP[k] ?? k;
    out[nk] = v;
  }
  return out;
}

const _LIBROSA_AXIS_DISPLAY: Record<string, string> = {
  tempo: "Tempo",
  key: "Key",
  mode: "Mode",
  energy: "Energy",
  danceability: "Dance",
};

/**
 * Default per-source weights for clustering **and** the 2-D layout. The values
 * live in ``cluster-weights.json`` (shared with the server) — the map and the
 * clusters must describe the same space, so one user-controlled vector drives
 * both. Persisted; "Reset" restores these values.
 */
export const DEFAULT_CLUSTER_WEIGHTS: Record<string, number> = clusterWeights;

/** Slider labels, keyed by source name (insertion order = display order). */
export const CLUSTER_WEIGHT_LABELS: Record<string, string> = {
  maest_logits: "Genre",
  maest: "MAEST",
  rhythm: "Rhythm",
  clap: "CLAP",
  supervised: "Folders",
  features: "Audio",
  maest_layer7: "MAEST layer-7",
  mert_l19: "MERT L19",
};

/**
 * What each source actually encodes, for the slider hover help. Keep the wording
 * about *information*, not the model name: the sliders are the only place a user
 * sees the sources, and "MAEST" alone says nothing about what it contributes.
 */
export const CLUSTER_WEIGHT_DESCRIPTIONS: Record<string, string> = {
  maest_logits:
    "Discogs style/genre logits (519-D) — the primary genre signal; sparse, near one-hot",
  maest: "MAEST hidden embedding (768-D) — audio semantics",
  rhythm: "librosa/madmom DSP (14-D) — groove, tempo and onset",
  clap: "CLAP audio embedding (512-D) — text-aligned semantics; the only block with a text tower",
  supervised:
    "Folder-supervised head output (128-D) — your own folder organisation; derived, not an encoder",
  features:
    "Audio descriptors (tempo, energy, danceability) — the other three are masked out",
  maest_layer7: "MAEST layer-7 (2304-D) — the opt-in pure-genre block; only active in the genre_full recipe",
  mert_l19:
    "MERT layer-19 (1024-D) — the opt-in MERT genre block; only active in the genre_full recipe",
};

function _migrateClusterWeights(raw: unknown): Record<string, number> {
  const out: Record<string, number> = { ...DEFAULT_CLUSTER_WEIGHTS };
  if (raw && typeof raw === "object") {
    for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
      if (!(k in out)) continue;
      const n = Number(v);
      if (Number.isFinite(n)) out[k] = clamp01(n);
    }
  }
  return out;
}

/** Migrate a saved genre recipe weight vector, ignoring unknown/legacy keys. */
function _migrateGenreWeights(raw: unknown): Record<string, number> {
  const out: Record<string, number> = { ...GENRE_FULL_WEIGHTS };
  if (raw && typeof raw === "object") {
    for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
      if (!(k in out)) continue;
      const n = Number(v);
      if (Number.isFinite(n)) out[k] = clamp01(n);
    }
  }
  return out;
}

/** Saved lens ids the genre map replaced: ``combined``/``maest``/``supervised``
 *  now resolve to the genre recipe. Unknown ids pass through unchanged. */
function _migrateLegacyMapLensId(id: string): string {
  return legacyToGenreLensId(id);
}

/** Compact display / query-string form for a source weight (drops trailing zeros). */
export function formatSourceWeight(v: number): string {
  return Number(v.toFixed(4)).toString();
}

export function isAudioFeatureAxisId(tag: string): boolean {
  return _LIBROSA_FEATURE_SET.has(tag);
}

const COLOR_MODES_LOOKUP = new Set<string>(COLOR_MODES);

/** Read a persisted colour mode, falling back to the legacy cluster boolean. */
function _migrateColorMode(raw: unknown, legacyCluster?: unknown): ColorMode {
  if (typeof raw === "string" && COLOR_MODES_LOOKUP.has(raw)) {
    return raw as ColorMode;
  }
  return Boolean(legacyCluster) ? "cluster" : "folder";
}

/**
 * Read the per-variable bloom ranges. A legacy scalar (the old shared
 * ``bloomRange`` or the older visual ``bloomAmount``) seeds every slot; a stored
 * record overrides slot by slot. A stored 0 is a real range (exact class), so only
 * a missing or non-numeric value falls back to the default.
 */
function _migrateBloomRanges(
  raw: unknown,
  legacy: unknown,
): Record<BloomRangeKey, number> {
  const out = Object.fromEntries(
    BLOOM_RANGE_KEYS.map((k) => [k, DEFAULT_BLOOM_RANGE]),
  ) as Record<BloomRangeKey, number>;
  if (typeof legacy === "number" && Number.isFinite(legacy)) {
    for (const k of BLOOM_RANGE_KEYS) out[k] = clamp01(legacy);
  }
  if (raw && typeof raw === "object") {
    const r = raw as Record<string, unknown>;
    for (const k of BLOOM_RANGE_KEYS) {
      const v = r[k];
      if (typeof v === "number" && Number.isFinite(v)) out[k] = clamp01(v);
    }
  }
  return out;
}

function _migrateLensPresets(raw: unknown): LensPreset[] {
  if (!Array.isArray(raw)) return [];
  const out: LensPreset[] = [];
  for (const item of raw) {
    if (!item || typeof item !== "object") continue;
    const p = item as Record<string, unknown>;
    if (typeof p.id !== "string" || typeof p.name !== "string") continue;
    out.push({
      id: p.id,
      name: p.name,
      // A legacy base lens (combined / maest / supervised) resolves to the genre
      // map, matching the scope-lens migration: the genre map replaced those maps.
      baseLens: legacyToGenreLensId(
        typeof p.baseLens === "string" ? p.baseLens : "combined",
      ),
      weights: _migrateClusterWeights(p.weights),
      colorMode: _migrateColorMode(p.colorMode, p.colorByCluster),
      clusterResolution:
        typeof p.clusterResolution === "number" && p.clusterResolution > 0
          ? p.clusterResolution
          : 1.5,
    });
  }
  return out;
}

export type ViewMode = "tags" | "embeddings";

/**
 * Right-panel tab ids, in tab-strip order. Properties stays pinned above the
 * strip; these three tabs own the lower sections so the panel no longer scrolls to
 * the brim. The view options live on the canvas edge instead. See
 * `ui_conventions.md` → *Right panel*.
 */
export const RIGHT_PANEL_TABS = ["analysis", "tracks"] as const;
export type RightPanelTab = (typeof RIGHT_PANEL_TABS)[number];

/** Hover-preview audition section; cycled with **W** and highlighted in the tooltip. */
export type PreviewSection = "intro" | "drop" | "break";

/**
 * A list-like navigation element under the pointer: a folder row, a cluster row, a
 * card chip (key / BPM), or an overlay row whose class is a set of tracks (a bloom
 * cluster of BPMs / keys). One concept so the glow, **s**, **h**, **f** (select) and
 * **Shift+s** (exclude) affordances behave the same wherever the element is rendered.
 * See `design-patterns.md` → *Navigation targets*.
 */
export type HoverTarget =
  | { kind: "folder"; prefix: string }
  | { kind: "cluster"; id: number }
  | { kind: "key"; value: string }
  | { kind: "bpm"; value: number }
  | { kind: "track"; path: string }
  /** An explicit path set (a bloom cluster); `color` tints the halo. */
  | { kind: "paths"; paths: string[]; color?: string; label?: string };

/** BPM window that counts as "the same tempo" for the BPM chip's hover target. */
export const BPM_MATCH_TOLERANCE = 2;

/**
 * The class a Solo / Excluded chip filtered with, ready to render as a chip pill:
 * a folder path in its colour family, a cluster or key/BPM in its lens colour, or a
 * bare track / path set. ``title`` carries the elaboration the short value drops
 * (a full multi-folder list, a widened bloom range).
 */
export interface RuleChip {
  value: string;
  icon?: "folder" | "tag";
  colors?: string[];
  title?: string;
}

/**
 * Kinds of **active view modifier**: a reversible change to what the canvas
 * shows. Each kind is one chip in the corner stack; the chip's × and **Esc**
 * undo it. See `design-patterns.md` → *Active view modifiers*.
 */
export const VIEW_MODIFIER_KINDS = [
  "search",
  "proximity",
  "scope",
  "exclusions",
  "filters",
  "external",
] as const;
export type ViewModifierKind = (typeof VIEW_MODIFIER_KINDS)[number];

/**
 * One active view modifier. Session-only: the stack resets on reload, like the
 * exclusions and the scope it describes.
 */
export interface ViewModifier {
  id: string;
  kind: ViewModifierKind;
  /** Chip title, e.g. the query, `"42 hidden"`, or `"3 filters"`. */
  label: string;
  /** Optional secondary readout, e.g. `"nearest 40%"`. */
  detail?: string;
  /** Slider position 0–100 for a `proximity` chip. */
  value?: number;
  /** Toggle state for an `external` chip (refit with external). */
  checked?: boolean;
  /**
   * A chip with no × — a permanent view filter (the external toggle) that Esc and
   * a direct close must leave in place.
   */
  permanent?: boolean;
  /**
   * An exclusions chip's action glyph (◉ solo / ⊘ excluded), shown before the rule
   * readout that replaces the label.
   */
  glyph?: string;
  /** The rule the chip filtered with, rendered in place of the plain label. */
  rule?: RuleChip;
}

/**
 * One active view rule: a view modifier that may also filter visibility. A rule
 * without ``matches`` (a search fade, a scope re-layout) only changes what the
 * canvas draws; one with it also decides whether a track is shown. ``matches``
 * returns true to *keep* the track, and ``scope`` says which view it applies to
 * (a tag filter only applies in tags mode). See `design-patterns.md` → *Active
 * view modifiers*.
 */
export interface ViewRule extends ViewModifier {
  matches?(track: Track): boolean;
  scope?: "tags" | "all";
}

export interface Viewport {
  ox: number;
  oy: number;
  zoom: number;
}

/** Sphere centre in cube coordinates: the billboard anchor of a scope member's flat layout. */
export interface Billboard {
  cx: number;
  cy: number;
  cz: number;
}

/** A world position in the unit cube: 2-D or, with ``z``, the 3-D orbit input. */
export interface MapPos {
  x: number;
  y: number;
  z?: number;
  /**
   * Set on a sphere-scope member's flat layout position: it is drawn facing the
   * camera at this sphere centre (x,y are world offsets around (cx,cy)), not as
   * a cube position.
   */
  billboard?: Billboard;
  /**
   * In-flight move between two frames: the position `from` (another frame) is
   * blended into this one in VIEW space, weight `mix` (1 = entirely `from`,
   * 0 = entirely this position). Only ever set on animation overlay entries,
   * never persisted in scopePositions.
   */
  from?: MapPos;
  mix?: number;
}

/**
 * Client-side 3-D orbit view option (persisted). ``on`` arms the mode; the
 * projection and deterministic lenses stay 2-D regardless — see ``is3d``.
 */
export interface View3D extends Orbit {
  on: boolean;
}

/** A region on the canvas whose tracks are re-laid-out by another lens. */
export interface Scope {
  /**
   * Region shape: a circle (X + scroll) or an axis-aligned box (X + drag) in the 2-D
   * map; a sphere in the 3-D view (the one shape that keeps its membership under
   * orbit).
   */
  shape: "circle" | "box" | "sphere";
  /** Centre in world coordinates. */
  cx: number;
  cy: number;
  /** Sphere centre depth in cube coordinates (sphere only; default 0.5). */
  cz?: number;
  /** Circle / sphere radius in world units (a box uses ``rx``/``ry`` instead). */
  r: number;
  /** Box half-width in world units (box only). */
  rx?: number;
  /** Box half-height in world units (box only). */
  ry?: number;
  /** Lens the in-scope tracks are laid out by. */
  lensId: string;
}

/** Half-width / half-height of a scope's region, in world units. */
export function scopeHalfExtents(scope: Scope): { hw: number; hh: number } {
  if (scope.shape === "box")
    return { hw: scope.rx ?? scope.r, hh: scope.ry ?? scope.r };
  return { hw: scope.r, hh: scope.r };
}

/**
 * True when a world position falls inside the scope's region.
 *
 * This is the single membership test: the coordinator uses it to pick the in-scope
 * set and the canvas uses it to hide in-region tracks the scope lens cannot place.
 * A circle/box lives in the flat map, so the canvas must draw its (possibly
 * non-circular) screen image — see ``CanvasView._drawScope``. A sphere is tested in
 * cube coordinates (``wz`` defaults to the centre plane), so orbiting never changes
 * its members.
 */
export function scopeContains(
  scope: Scope,
  wx: number,
  wy: number,
  wz = 0.5,
): boolean {
  if (scope.shape === "box") {
    const { hw, hh } = scopeHalfExtents(scope);
    return Math.abs(wx - scope.cx) <= hw && Math.abs(wy - scope.cy) <= hh;
  }
  if (scope.shape === "sphere") {
    return (
      Math.hypot(wx - scope.cx, wy - scope.cy, wz - (scope.cz ?? 0.5)) <= scope.r
    );
  }
  return Math.hypot(wx - scope.cx, wy - scope.cy) <= scope.r;
}

const LS_KEY = "trackspace";

// ─── Minimal event bus ───────────────────────────────────────
type EventCallback = (...args: unknown[]) => void;

class EventBus {
  private _h: Record<string, EventCallback[]> = {};

  on(e: string, fn: EventCallback): void {
    (this._h[e] ??= []).push(fn);
  }

  off(e: string, fn: EventCallback): void {
    const a = this._h[e];
    if (a) this._h[e] = a.filter((f) => f !== fn);
  }

  emit(e: string, ...a: unknown[]): void {
    for (const fn of this._h[e] ?? []) fn(...a);
  }
}

// ─── Hover preview volume ─────────────────────────────────────

/** Default hover-preview slider position. Moderate, not full scale. */
export const DEFAULT_PREVIEW_VOLUME = 0.8;

/**
 * Slider position → audio gain, with a perceptual taper: half the slider is a
 * quarter of full scale. A linear 0–1 mapping puts almost all of the usable range
 * in the top few pixels, so the preview jumps from inaudible to deafening.
 */
export function previewGain(volume: number): number {
  const v = clamp01(volume);
  return v * v;
}

// ─── Model ───────────────────────────────────────────────────
export class Model extends EventBus {
  allTracks: Track[] = [];
  /**
   * True while the read-only layout-evaluation gallery is active
   * (``?layout_eval=1``). Session-only, never persisted. Forces the view into the
   * embedding map and lets the canvas suppress lens chrome that would not make
   * sense for precomputed maps.
   */
  evalMode = false;
  /**
   * When set (eval gallery), the only paths eligible for the view: the artifact's
   * own paths that exist in the loaded library. Non-artifact library tracks are
   * filtered out so the eval universe is exactly the map's tracks.
   */
  evalUniverse: Set<string> | null = null;
  /**
   * Dimensionality of the layout-evaluation artifact on screen. Session-only,
   * defaults to 2 and resets when eval mode ends; it never touches the persisted
   * production ``view3d`` option.
   */
  evalDims: EvalDims = 2;
  /**
   * Sidebar focus: highlighted tree row and default parent for “+ Folder”.
   * Does **not** filter the canvas — the full loaded library is always in scope
   * (only per-session hides apply). Persisted as ``folder`` in localStorage.
   */
  folder = "";
  recursive = true;
  axisX: string | null = null;
  axisY: string | null = null;
  filterRanges: Record<string, [number, number]> = {};
  selected = new Set<string>();
  hoverPreview = false;
  /** Hover preview slider position [0, 1]; the audio gain is ``previewGain(this)``. */
  previewVolume = DEFAULT_PREVIEW_VOLUME;
  /**
   * Track currently being auditioned by the hover preview (in-memory only).
   * With ``previewSection`` it drives the tooltip chip highlight.
   */
  previewTrackPath: string | null = null;
  /** Section the hover preview is auditioning; ``null`` while not previewing. */
  previewSection: PreviewSection | null = null;
  /** Playback position of the hover preview, in seconds (in-memory only). */
  previewTime = 0;
  /** Duration of the auditioned track, in seconds; ``0`` until metadata loads. */
  previewDuration = 0;
  vp: Viewport = { ox: 0, oy: 0, zoom: 1 };
  /**
   * Client-side 3-D orbit view option; persisted. Only a projection lens with
   * this on is actually 3-D — see ``is3d``.
   */
  view3d: View3D = { on: false, yaw: 0.6, pitch: 0.35 };
  /**
   * Camera blend 0..1 for the 2-D↔3-D transition: 0 = head-on orthographic,
   * 1 = the full orbit + perspective. Transient, never persisted: ``loadLS``
   * snaps it to ``view3d.on`` so a reload does not re-tween. The canvas renders
   * 3-D while ``shows3d`` is true, so the collapse tween keeps drawing the
   * rotated view until it lands back on head-on.
   */
  view3dBlend = 0;
  _knownTags: string[] = [];

  /**
   * Active lens id (built-in or preset). "tags" is the tag-axis lens; every other
   * built-in is an embedding-space lens. See `lib/lenses.ts`.
   */
  private _lensId = "tags";
  /** User presets: a base lens plus saved display options. Persisted. */
  lensPresets: LensPreset[] = [];

  /**
   * True when the server opted into the ``genre_full`` blocks (``--enable-genre-full``),
   * learned from ``/api/embeddings/status``. Gates the genre_full lens: when false
   * the built-in lens set (cycle + tabber) is exactly the pre-recipe one.
   */
  genreFullEnabled = false;

  /**
   * Per-dimensionality openTSNE iteration-budget overrides, keyed by ``d2``/``d3``.
   * Every embedding-map lens shares the budget for its dimensionality: a budget
   * chosen on the 2-D Genre map is shared by all other 2-D map lenses, and 3-D
   * keeps its own separate budget. ``null`` entries mean "use the active recipe's
   * default" (the production multilevel 50/0 budget). Sent as ``tsne_n_iter`` /
   * ``tsne_early_exaggeration_iter`` only when set.
   */
  tsneIteration: Record<string, { nIter: number | null; early: number | null }> = {};

  get lensId(): string {
    return this._lensId;
  }

  set lensId(id: string) {
    this._lensId = id;
  }

  /**
   * Derived compatibility view of the lens: "tags" for the tag-axis lens,
   * "embeddings" for every map/feature lens. Keeping this as the *read* surface
   * means all the existing embedding-mode gating (transform handles, tag drag,
   * hiding unanalysed tracks) needs no per-lens branch.
   *
   * During the read-only eval gallery the view is forced to the embedding map
   * *without* touching ``_lensId`` — so the user's normal mode is never persisted
   * or clobbered by a ``saveLS`` that happens mid-eval.
   */
  get viewMode(): ViewMode {
    if (this.evalMode) return "embeddings";
    return getLens(this._lensId).kind === "tags" ? "tags" : "embeddings";
  }

  set viewMode(mode: ViewMode) {
    this._lensId =
      mode === "embeddings"
        ? getLens(this._lensId).kind === "tags"
          ? defaultMapLensId(this.genreFullEnabled)
          : this._lensId
        : "tags";
  }

  /**
   * True when the canvas should draw the 3-D orbit view: the client-side toggle
   * is on, the eval gallery is not active, and the lens is a projection lens
   * (only those have a z axis — tags/deterministic lenses and the eval gallery
   * stay 2-D).
   */
  get is3d(): boolean {
    if (this.evalMode) return this.evalDims === 3;
    return (
      this.view3d.on &&
      this.lens().kind === "projection"
    );
  }

  /**
   * True while the canvas renders the 3-D view. Same gate as ``is3d`` but the
   * *mode* flag is replaced by "armed or still blending": true while the 2-D↔3-D
   * camera tween is mid-flight (``view3dBlend > 0``), so the collapse tween
   * keeps drawing the rotated view until it reaches head-on. ``is3d`` stays the
   * *mode* gate (query string, interactions, scope/tag rules); this is the
   * *rendering* gate.
   */
  get shows3d(): boolean {
    if (this.evalMode) return this.evalDims === 3 || this.view3dBlend > 0;
    return (
      (this.view3d.on || this.view3dBlend > 0) &&
      this.lens().kind === "projection"
    );
  }

  /** The active lens, resolving a preset id to its base lens. */
  lens(): Lens {
    const preset = this.lensPresets.find((p) => p.id === this._lensId);
    return getLens(preset ? preset.baseLens : this._lensId);
  }

  /** Next built-in lens id for the **E** cycle. */
  nextLens(): string {
    const preset = this.lensPresets.find((p) => p.id === this._lensId);
    return nextLensId(
      this._lensId,
      preset?.baseLens,
      availableLenses(this.genreFullEnabled),
    );
  }

  /** Previous built-in lens id for the **Shift+E** cycle. */
  prevLens(): string {
    const preset = this.lensPresets.find((p) => p.id === this._lensId);
    return prevLensId(
      this._lensId,
      preset?.baseLens,
      availableLenses(this.genreFullEnabled),
    );
  }

  /** Built-in lens the active view is based on: a preset's base, else the lens id. */
  activeBaseLensId(): string {
    const preset = this.lensPresets.find((p) => p.id === this._lensId);
    return preset?.baseLens ?? this._lensId;
  }

  /** Per-dimensionality key for a budget override: ``d2``/``d3``. */
  static tsneIterationKey(dims: 2 | 3): string {
    return `d${dims}`;
  }

  /** The active dimensionality's openTSNE budget override, or
   *  ``{nIter:null, early:null}`` (use the recipe default). Shared by every
   *  embedding-map lens of that dimensionality. The per-dimensionality lookup is
   *  hidden here so the projection query and the iteration control read one seam. */
  tsneIterationFor(): { nIter: number | null; early: number | null } {
    return (
      this.tsneIteration[Model.tsneIterationKey(this.is3d ? 3 : 2)] ??
      { nIter: null, early: null }
    );
  }

  /** Persist a budget override for the *active* dimensionality (``d2``/``d3``),
   *  shared by all embedding-map lenses of that dimensionality. ``null``/``null``
   *  clears it (recipe default). The coordinator owns when to save/refetch; this
   *  only mutates the stored per-dimensionality value. */
  setTsneIterationForActiveView(nIter: number | null, early: number | null): void {
    const key = Model.tsneIterationKey(this.is3d ? 3 : 2);
    if (nIter === null && early === null) delete this.tsneIteration[key];
    else this.tsneIteration[key] = { nIter, early };
  }

  /** The map lens the "tags → embeddings" toggle lands on: genre when enabled. */
  defaultMapLensId(): string {
    return defaultMapLensId(this.genreFullEnabled);
  }

  /** Apply a saved preset: base lens plus its captured display options. */
  applyLensPreset(p: LensPreset): void {
    this._lensId = p.id;
    this.clusterWeights = { ...DEFAULT_CLUSTER_WEIGHTS, ...p.weights };
    this.colorMode = p.colorMode;
    this.clusterResolution = p.clusterResolution;
    this.emit("change");
  }

  /** Capture the current lens + display options as a new preset. */
  addLensPreset(name: string): LensPreset {
    const preset: LensPreset = {
      id: `preset:${Date.now().toString(36)}`,
      name,
      baseLens: this.lens().id,
      weights: { ...this.clusterWeights },
      colorMode: this.colorMode,
      clusterResolution: this.clusterResolution,
    };
    this.lensPresets = [...this.lensPresets, preset];
    this._lensId = preset.id;
    return preset;
  }

  deleteLensPreset(id: string): void {
    this.lensPresets = this.lensPresets.filter((p) => p.id !== id);
    if (this._lensId === id) this._lensId = "tags";
  }

  embeddingPositions: Map<string, MapPos> = new Map();
  embeddingsReady = false;
  embeddingsGenerating = false;
  projectionPending = false;
  /**
   * Query string of the layout request currently in flight. ``layout_progress``
   * events echo their request's query, so a phase from an older overlapping
   * request is ignored instead of overwriting the newer one's progress.
   */
  pendingProjectionQuery: string | null = null;
  /** When the in-flight layout request started (ms), for an elapsed readout. */
  projectionStartedAt: number | null = null;
  /**
   * In-flight phase of a layout request, streamed on `/api/embeddings/stream`.
   * The layout is a plain request rather than a queued job, so it carries its own
   * phase here instead of riding `queueStatus`.
   */
  layoutProgress: { phase: string; done: number; total: number } | null = null;
  /**
   * In-flight phase of the folder colour anchors build, streamed on
   * `/api/jobs/stream` as ``colors_progress``. Like the layout it is a plain
   * background computation rather than a queued job, so it carries its own phase
   * instead of riding `queueStatus`.
   */
  colorsProgress: { phase: string; done: number; total: number } | null = null;
  /** True while the required models are not ready yet (embedding mode waits on `_pollModelReady`). */
  embeddingModelsLoading = false;
  /**
   * Active scope: a circle or box (world coordinates) whose tracks are re-laid-out
   * by ``lensId``. ``scopePositions`` holds only the in-scope tracks' scope-lens
   * positions, so membership is "has a scope position".
   */
  scope: Scope | null = null;
  scopePositions: Map<string, MapPos> = new Map();
  /** Lens a newly drawn scope starts on; persisted so the last choice sticks. */
  scopeLensId = "camelot";
  libraryLoadProgress: {
    done: number;
    total: number;
    lastPath?: string;
    lastTitle?: string;
  } | null = null;

  /* ── Analysis queue ─────────────────────────────────────────
   * Mirrors ``/api/jobs/status``. Counts come from SQL aggregates on the
   * server, so they stay correct across restarts — the UI never has to
   * reconstruct progress from a stream of events it may have missed.
   */
  queueStatus: QueueStatus | null = null;

  /* ── Clustering ─────────────────────────────────────────────
   * ``clusterAssignments`` maps track path → cluster id at the currently
   * selected resolution; ``clusterNames`` labels those ids.
   */
  clusterAssignments: Map<string, number> = new Map();
  clusterNames: Map<number, string> = new Map();
  clusterSizes: Map<number, number> = new Map();
  /** Available Leiden resolutions (coarse → fine), from the server. */
  clusterResolutions: number[] = [];
  /** Currently displayed resolution; persisted. */
  clusterResolution = 1.5;
  /** How dots are coloured (folder, cluster, tempo, key); persisted. */
  colorMode: ColorMode = "folder";
  /**
   * Variable the selection-bloom cursor points at: hovering a dot blooms every
   * track that matches it on this variable. Independent of ``colorMode`` so the map
   * can be coloured by one variable while blooming by another; moved with **D** /
   * **Shift+D** across the colour tabs. Persisted.
   */
  selectionBloom: ColorMode = "folder";
  /**
   * Per-variable selection-bloom range [0,1]; **D + scroll** widens the range of
   * the cursor's current variable. Each variable keeps its own value so widening the
   * tempo bloom does not also widen the folder bloom; folder and predicted folder
   * share one slot. The meaning is per-variable (parent folders / ±BPM / ±key steps)
   * — see `lib/bloom.ts`. Persisted.
   */
  bloomRanges: Record<BloomRangeKey, number> = Object.fromEntries(
    BLOOM_RANGE_KEYS.map((k) => [k, DEFAULT_BLOOM_RANGE]),
  ) as Record<BloomRangeKey, number>;

  /** The bloom range for *mode* (default: the cursor's variable). */
  bloomRange(mode: ColorMode = this.selectionBloom): number {
    return this.bloomRanges[bloomRangeKey(mode)];
  }

  /** Set the bloom range for *mode* (default: the cursor's variable). */
  setBloomRange(value: number, mode: ColorMode = this.selectionBloom): void {
    this.bloomRanges[bloomRangeKey(mode)] = clamp01(value);
  }
  /**
   * Per-source weights for clustering and the layout; persisted. One vector drives
   * both pipelines so the map and cluster assignments describe the same space.
   * (The legacy Combined-map vector — only active when the server did not enable
   * the genre recipe.)
   */
  clusterWeights: Record<string, number> = { ...DEFAULT_CLUSTER_WEIGHTS };

  /* ── Genre map weights ──────────────────────────────────────
   * The genre_full recipe's tunable 7-key vector (logits / maest / layer-7 /
   * MERT / rhythm / clap / supervised). ``genreWeights`` is the *applied*,
   * persisted vector the map is currently fit at; ``genreDraft`` holds a live
   * slider draft that has only been *previewed* via the Jacobian and not yet
   * committed (null = no active draft). */
  genreWeights: Record<string, number> = { ...GENRE_FULL_WEIGHTS };
  /** Live slider draft, or null when the sliders match the applied vector. */
  genreDraft: Record<string, number> | null = null;
  /** True when the server enabled the MERT genre block. Derived from the status
   *  payload: the ``mert_l19`` coverage row is only reported when MERT is wired.
   *  Gates the MERT genre slider (a disabled block can never contribute). */
  mertEnabled = false;
  /** Per-source coverage from the last status poll, for genre-slider availability. */
  coverageSources: { id: string; done: number; pending: number }[] = [];
  clusterRevision: string | null = null;
  clustersAvailable = false;
  /**
   * Virtual folders the next head training run draws labels from; persisted.
   * Empty means every root (the server's default).
   */
  trainFolders: string[] = [];

  /* ── Predicted folders (folder head) ─────────────────────────
   * ``folderPredictions`` maps track path → the folder head's nearest learned
   * folder (a virtual path); ``folderPredictionMatches`` is the subset that also
   * clears its class's match floor. Fetched lazily the first time the "Predicted
   * folder" colour mode is selected, and invalidated when the head changes.
   */
  folderPredictions: Map<string, string> = new Map();
  folderPredictionMatches: Set<string> = new Set();
  folderPredictionsLoaded = false;
  folderPredictionsAvailable = false;
  /** True while the head's predictions are being fetched, for the status bar. */
  folderPredictionsLoading = false;
  /** Predicted-folder colour: confident matches only, or every nearest. Persisted. */
  headColorConfident = true;

  /* ── Track structure ────────────────────────────────────────
   * Coarse intro/drop/break timestamps, fetched lazily per path (the structure
   * stage backfills the cache in the background). Missing entries fall back to a
   * fixed preview position.
   */
  trackStructure: Map<string, TrackStructure> = new Map();

  /* ── Search match sections ─────────────────────────────────
   * Per path, the best-matching CLAP segment centre as a fraction of the track
   * duration, set by the current search. The hover preview seeks there instead of
   * the drop so a hit auditions the part that actually matched the query.
   */
  searchSections: Map<string, number> = new Map();

  /* ── Active view modifiers ─────────────────────────────────
   * The stack of reversible changes to what the canvas shows, oldest → newest.
   * **Esc** dismisses the newest; a chip's × dismisses one. Session-only, like
   * the exclusions and the scope a modifier may describe.
   */
  /**
   * Active view rules, oldest → newest. The chip stack the corner view renders;
   * the rules that carry a ``matches`` predicate are also what ``isVisible``
   * consults, so a filter chip and the visible set share one source.
   */
  viewRules: ViewRule[] = [];

  /** The descriptors the corner chip view renders (rules are modifiers + matcher). */
  get viewModifiers(): ViewModifier[] {
    return this.viewRules;
  }

  private _cache: Track[] | null = null;
  private _pathMap: Map<string, Track> | null = null;
  /**
   * Folder prefixes (and ``""``) that contain at least one visible track, derived
   * from ``tracks``. Cached because the folder tree asks "is this row excluded?"
   * once per row; scanning every track per row is O(rows × tracks) on each hover.
   */
  private _visibleTreePrefixes: Set<string> | null = null;

  /** Explicitly excluded by path (session-only; Alt+click / Shift+S). */
  hiddenPaths = new Set<string>();
  /**
   * Hidden by the distance-filter slider, tracked apart from ``hiddenPaths`` so
   * the Excluded chip counts only explicit exclusions and the two filters can be
   * dismissed independently. Session-only.
   */
  proximityHiddenPaths = new Set<string>();
  /** Hide every track in this folder and subfolders (normalised "", not "."). */
  hiddenFolderPrefixes = new Set<string>();
  /** Root ids whose role is "external" (not My library); from `/api/folders`. */
  externalRootIds = new Set<string>();
  /** Include external-root tracks in the view (persisted, default off). */
  showExternal = false;
  /**
   * Fit the projection with external-root tracks included (persisted, default
   * off). When off, showing externals keeps the library layout and places them
   * onto it by Nyström extension, so toggling visibility never moves the map.
   */
  externalInProjection = false;
  /** Active right-panel tab (persisted); search + properties stay pinned. */
  rightPanelTab: RightPanelTab = "analysis";

  /* ── Computed ──────────────────────────────────────────────── */

  /**
   * True when the track belongs to the current view scope. Under eval the
   * artifact universe is the authority: every artifact track is drawn, including
   * external-root tracks, and every non-artifact track is hidden — without
   * mutating ``showExternal``. Outside eval the normal external filter applies.
   */
  trackInViewUnion(t: Track): boolean {
    if (this.evalUniverse) return this.evalUniverse.has(t.path);
    return this.showExternal || !this.isExternalTrack(t);
  }

  /** True when *t* belongs to a root marked external (analysed, not My library). */
  isExternalTrack(t: Track): boolean {
    const folder = t.folder ?? "";
    const slash = folder.indexOf("/");
    const rootId = slash >= 0 ? folder.slice(0, slash) : folder;
    return this.externalRootIds.has(rootId);
  }

  isExcludedTrack(t: Track): boolean {
    if (this.hiddenPaths.has(t.path)) return true;
    const tf = t.folder ?? "";
    for (const p of this.hiddenFolderPrefixes) {
      if (p === "") return true;
      if (tf === p || tf.startsWith(p + "/")) return true;
    }
    return false;
  }

  isTrackHidden(t: Track): boolean {
    return this.isExcludedTrack(t) || this.proximityHiddenPaths.has(t.path);
  }

  /**
   * True when the current view shows *t*: inside the view union, not excluded or
   * distance-filtered, and passing every active filter rule. The one predicate the
   * canvas and the fit/select paths share, so the chip stack and the visible set
   * cannot disagree.
   */
  isVisible(t: Track): boolean {
    if (!this.trackInViewUnion(t) || this.isTrackHidden(t)) return false;
    for (const r of this.viewRules) {
      if (!r.matches) continue;
      if (r.scope === "tags" && this.viewMode !== "tags") continue;
      if (!r.matches(t)) return false;
    }
    return true;
  }

  /** True while any track or folder subtree is excluded from the current view. */
  get hasExclusions(): boolean {
    return this.hiddenPaths.size > 0 || this.hiddenFolderPrefixes.size > 0;
  }

  /** Visible tracks under ``prefix`` (``""`` or ``"a/b"``), respecting view union + hides. */
  hasVisibleTracksUnderTreePrefix(treePrefix: string): boolean {
    return this.visibleTreePrefixes().has(treePrefix);
  }

  /**
   * Every folder prefix (and ``""``) with at least one visible track under it, in one
   * pass over the tracks. The folder tree tests each rendered row against this set
   * rather than rescanning the library per row.
   */
  visibleTreePrefixes(): Set<string> {
    if (this._visibleTreePrefixes) return this._visibleTreePrefixes;
    const prefixes = new Set<string>();
    for (const t of this.allTracks) {
      if (!this.trackInViewUnion(t) || this.isTrackHidden(t)) continue;
      prefixes.add("");
      const folder = t.folder ?? "";
      const parts = folder.split("/");
      for (let i = 1; i <= parts.length; i++) {
        prefixes.add(parts.slice(0, i).join("/"));
      }
    }
    this._visibleTreePrefixes = prefixes;
    return prefixes;
  }

  get tracks(): Track[] {
    if (this._cache) return this._cache;
    const result = this.allTracks.filter((t) => {
      if (!this.trackInViewUnion(t) || this.isTrackHidden(t)) return false;
      return true;
    });
    this._cache = result;
    return result;
  }

  get tags(): string[] {
    const s = new Set(this._knownTags);
    for (const t of this.allTracks)
      for (const k of Object.keys(t.tags)) s.add(k);
    const withVals = new Set<string>();
    for (const t of this.tracks)
      for (const k of Object.keys(t.tags)) withVals.add(k);
    return [...s]
      .filter((tag) => !isAudioFeatureAxisId(tag))
      .sort((a, b) => {
        const ha = withVals.has(a),
          hb = withVals.has(b);
        if (ha !== hb) return ha ? -1 : 1;
        return a < b ? -1 : a > b ? 1 : 0;
      });
  }

  tagHasValuesInView(tag: string): boolean {
    if (isAudioFeatureAxisId(tag))
      return this.tracks.some(
        (t) =>
          this._audioFeatureScalar(t, tag) !== undefined ||
          t.tags[tag] !== undefined,
      );
    return this.tracks.some((t) => t.tags[tag] !== undefined);
  }

  axisDisplayName(axis: string): string {
    return _LIBROSA_AXIS_DISPLAY[axis] ?? axis;
  }

  /** World-axis value in [0, 1] for tag mode (ID3 tags or cached audio features). */
  axisScalar(track: Track, axis: string | null): number {
    if (!axis) return 0.5;
    const af = this._audioFeatureScalar(track, axis);
    if (af !== undefined) return af;
    return track.tags[axis] ?? 0.5;
  }

  private _audioFeatureScalar(track: Track, axis: string): number | undefined {
    if (!isAudioFeatureAxisId(axis)) return undefined;
    const row = track.audio_features;
    if (!row) return undefined;
    const k = axis as keyof TrackAudioFeatures;
    const v = row[k];
    return typeof v === "number" ? v : undefined;
  }

  trackByPath(path: string): Track | undefined {
    if (!this._pathMap) {
      this._pathMap = new Map();
      for (const t of this.allTracks) this._pathMap.set(t.path, t);
    }
    return this._pathMap.get(path);
  }

  /** True when *track*'s *tag* value is inside its active filter range. */
  passesFilterTag(track: Track, tag: string): boolean {
    const r = this.filterRanges[tag];
    if (!r) return true;
    const [lo, hi] = r;
    let v: number | undefined;
    if (isAudioFeatureAxisId(tag))
      v = this._audioFeatureScalar(track, tag) ?? track.tags[tag];
    else v = track.tags[tag];
    if (v === undefined) return true;
    return v >= lo && v <= hi;
  }

  passesFilter(track: Track): boolean {
    for (const tag of Object.keys(this.filterRanges)) {
      if (!this.passesFilterTag(track, tag)) return false;
    }
    return true;
  }

  /* ── Internal ─────────────────────────────────────────────── */

  private _dirty(): void {
    this._invalidateView();
    this._pathMap = null;
  }

  /** Drop the view-derived caches (visible track list + visible folder prefixes). */
  private _invalidateView(): void {
    this._cache = null;
    this._visibleTreePrefixes = null;
  }

  private _hasAnyVisibleInViewUnion(): boolean {
    for (const t of this.allTracks) {
      if (this.trackInViewUnion(t) && !this.isTrackHidden(t)) return true;
    }
    return false;
  }

  /** If every track in the view union is excluded, clear all exclusions (avoid a blank canvas). */
  private _clearExclusionsIfNoneVisible(): void {
    if (this._hasAnyVisibleInViewUnion()) return;
    if (!this.hasExclusions) return;
    this.hiddenPaths.clear();
    this.hiddenFolderPrefixes.clear();
  }

  /* ── Mutators ─────────────────────────────────────────────── */

  /** Enter/leave the read-only layout-evaluation gallery (session-only). */
  setEvalMode(on: boolean): void {
    if (this.evalMode === on) return;
    this.evalMode = on;
    this.evalDims = 2;
    // Production blend state is persisted through `view3d.on`; an eval session's
    // blend is transient, so leaving must not leave the production camera 3-D.
    this.view3dBlend = this.view3d.on ? 1 : 0;
    // Deliberately does NOT touch `_lensId`: the view-mode getter reports the
    // embedding map while eval is on, so a mid-eval `saveLS` still persists the
    // user's normal lens rather than overwriting it.
    this._invalidateView();
    this.emit("change");
  }

  /** Commit the eval artifact dimensionality (after its positions applied). */
  setEvalDims(dims: EvalDims): void {
    if (this.evalDims === dims) return;
    this.evalDims = dims;
    this._invalidateView();
    this.emit("change");
  }

  /**
   * Restrict the eval-view universe to *paths* (the artifact paths present in the
   * loaded library), or clear the restriction with ``null``. Invalidates the view.
   */
  setEvalUniverse(paths: Set<string> | null): void {
    this.evalUniverse = paths;
    this._invalidateView();
    this.emit("change");
  }

  setAllTracks(tracks: Track[]): void {
    this.allTracks = tracks;
    this._dirty();
    this.selected.clear();
    this.emit("change");
  }

  /** Append tracks without emitting "change" — caller controls when to redraw. */
  appendTracks(tracks: Track[]): void {
    for (const t of tracks) this.allTracks.push(t);
    this._dirty();
  }

  setRecursive(r: boolean): void {
    this.recursive = r;
    this._invalidateView();
    this.selected.clear();
    this.emit("change");
  }

  hidePaths(paths: Iterable<string>): void {
    this.hiddenPaths = new Set([...this.hiddenPaths, ...paths]);
    this._invalidateView();
    this._clearExclusionsIfNoneVisible();
    this.emit("change");
  }

  hideFolderTree(prefixNorm: string): void {
    this.hiddenFolderPrefixes.add(prefixNorm);
    this._invalidateView();
    this._clearExclusionsIfNoneVisible();
    this.emit("change");
  }

  /** Replace the external-root set (from `/api/folders`); no-op when unchanged. */
  setExternalRoots(ids: Iterable<string>): void {
    const next = new Set(ids);
    if (
      next.size === this.externalRootIds.size &&
      [...next].every((id) => this.externalRootIds.has(id))
    )
      return;
    this.externalRootIds = next;
    this._invalidateView();
    this._clearExclusionsIfNoneVisible();
    this.emit("change");
  }

  setShowExternal(show: boolean): void {
    if (this.showExternal === show) return;
    this.showExternal = show;
    this._invalidateView();
    this._clearExclusionsIfNoneVisible();
    this.emit("change");
  }

  setExternalInProjection(included: boolean): void {
    if (this.externalInProjection === included) return;
    this.externalInProjection = included;
    this.emit("change");
  }

  /** Hide every visible-until-now track that is not currently selected. */
  focusSelection(): void {
    this.focusPaths(this.selected);
  }

  /**
   * Hide every visible track whose path is not in *keep* (folder subtree, cluster
   * membership, or selection — all isolate actions reduce to a keep-set).
   */
  focusPaths(keep: ReadonlySet<string>): void {
    const add: string[] = [];
    for (const t of this.allTracks) {
      if (!this.trackInViewUnion(t) || this.isTrackHidden(t)) continue;
      if (keep.has(t.path)) continue;
      add.push(t.path);
    }
    this.hiddenPaths = new Set([...this.hiddenPaths, ...add]);
    this._invalidateView();
    this._clearExclusionsIfNoneVisible();
    this.emit("change");
  }

  /** True when some visible track lies outside *keep* (isolating would hide more). */
  hasVisibleTracksOutside(keep: ReadonlySet<string>): boolean {
    for (const t of this.allTracks) {
      if (!this.trackInViewUnion(t) || this.isTrackHidden(t)) continue;
      if (!keep.has(t.path)) return true;
    }
    return false;
  }

  /**
   * Neighbourhood-slider isolation: reveal what the previous slider step hid, then
   * hide every visible track outside *keep* — one update, so a drag stays smooth
   * (the alternative, `revealPaths` + `focusPaths`, emits "change" twice per step).
   * Returns the paths it hid, for the next step to reveal.
   */
  setProximityKeep(
    keep: ReadonlySet<string>,
    previouslyHidden: ReadonlySet<string>,
  ): Set<string> {
    const next = new Set(this.proximityHiddenPaths);
    for (const p of previouslyHidden) next.delete(p);
    const newlyHidden = new Set<string>();
    for (const t of this.allTracks) {
      if (!this.trackInViewUnion(t)) continue;
      if (keep.has(t.path)) continue;
      if (next.has(t.path)) continue;
      next.add(t.path);
      newlyHidden.add(t.path);
    }
    this.proximityHiddenPaths = next;
    this._invalidateView();
    this.emit("change");
    return newlyHidden;
  }

  /** Reveal every track the distance-filter slider hid (dismiss its chip). */
  clearProximityHidden(): void {
    if (this.proximityHiddenPaths.size === 0) return;
    this.proximityHiddenPaths = new Set();
    this._invalidateView();
    this.emit("change");
  }

  /** Every known track in *paths* is excluded (and at least one exists). */
  pathsFullyHidden(paths: Iterable<string>): boolean {
    let any = false;
    for (const p of paths) {
      const t = this.trackByPath(p);
      if (!t) continue;
      any = true;
      if (!this.isTrackHidden(t)) return false;
    }
    return any;
  }

  /* ── Hover targets ────────────────────────────────────────── */

  /** True when *track* belongs to the hovered navigation element. */
  trackMatchesHoverTarget(track: Track, target: HoverTarget | null): boolean {
    if (!target) return false;
    switch (target.kind) {
      case "folder": {
        const f = track.folder ?? "";
        return (
          target.prefix === "" ||
          f === target.prefix ||
          f.startsWith(target.prefix + "/")
        );
      }
      case "cluster":
        return this.clusterAssignments.get(track.path) === target.id;
      case "key":
        return target.value !== "" && (track.musical_key ?? "") === target.value;
      case "bpm":
        return (
          track.bpm != null &&
          Number.isFinite(track.bpm) &&
          Math.abs(track.bpm - target.value) <= BPM_MATCH_TOLERANCE
        );
      case "track":
        return track.path === target.path;
      case "paths":
        return target.paths.includes(track.path);
    }
  }

  /**
   * True when *track* falls inside the hovered track's bloom on the selection
   * variable (``selectionBloom``) at the current ``bloomRange``, so hovering a dot
   * blooms its "family" the way the old same-folder rule did. Folder and predicted
   * folder climb ``bloomFolderLevels`` parent directories, tempo allows a
   * ``bloomTempoPct`` window, key allows ``bloomKeySteps`` Camelot steps; cluster
   * is an exact id match. A track the variable cannot place never blooms.
   */
  trackBloomsWith(track: Track, hovered: Track): boolean {
    switch (this.selectionBloom) {
      case "folder": {
        const lv = bloomFolderLevels(this.bloomRange());
        return (
          folderAncestor(track.folder ?? "", lv) ===
          folderAncestor(hovered.folder ?? "", lv)
        );
      }
      case "head": {
        const a = this.folderPredictions.get(track.path);
        const b = this.folderPredictions.get(hovered.path);
        if (a === undefined || b === undefined) return false;
        return folderAncestor(a, bloomFolderLevels(this.bloomRange())) ===
          folderAncestor(b, bloomFolderLevels(this.bloomRange()));
      }
      case "cluster": {
        const a = this.clusterAssignments.get(track.path);
        const b = this.clusterAssignments.get(hovered.path);
        return a !== undefined && a === b;
      }
      case "tempo": {
        if (track.bpm == null || hovered.bpm == null) return false;
        if (!Number.isFinite(track.bpm) || !Number.isFinite(hovered.bpm))
          return false;
        const tol = (Math.abs(hovered.bpm) * bloomTempoPct(this.bloomRange())) / 100;
        return Math.abs(track.bpm - hovered.bpm) <= tol + 1e-9;
      }
      case "key": {
        const steps = camelotSteps(track, hovered);
        return steps !== null && steps <= bloomKeySteps(this.bloomRange());
      }
    }
  }

  /** Step the selection-bloom cursor through the colour tabs (**D** / **Shift+D**). */
  moveSelectionBloom(dir: 1 | -1 = 1): void {
    this.selectionBloom = stepColorMode(this.selectionBloom, dir);
  }

  /**
   * Paths in the hovered dot's bloom class, for **S** to isolate. Folder colour
   * isolates the whole subtree under the bloom's ancestor — the left-browser
   * folder-click behaviour, so **S** over a folder behaves like clicking it in the
   * tree. Every other variable isolates the range-aware bloom set.
   */
  bloomPaths(hovered: Track): Set<string> {
    if (this.selectionBloom === "folder") {
      const prefix = folderAncestor(
        hovered.folder ?? "",
        bloomFolderLevels(this.bloomRange()),
      );
      return this.hoverTargetPaths({ kind: "folder", prefix });
    }
    const out = new Set<string>();
    for (const t of this.allTracks) {
      if (this.trackBloomsWith(t, hovered)) out.add(t.path);
    }
    return out;
  }

  /** Known paths belonging to *target*, hidden or not (the solo/fit keep-set). */
  hoverTargetPaths(target: HoverTarget | null): Set<string> {
    const out = new Set<string>();
    if (!target) return out;
    for (const t of this.allTracks) {
      if (this.trackMatchesHoverTarget(t, target)) out.add(t.path);
    }
    return out;
  }

  /* ── Rule readouts (Solo / Excluded chips) ─────────────────── */

  /**
   * The rule a hover target filters with, as a chip readout: a folder in its colour
   * family, a cluster or key/BPM in its lens colour, a bare track, or a path-set's
   * own label. ``null`` when the target has no short name.
   */
  hoverTargetRule(target: HoverTarget | null): RuleChip | null {
    if (!target) return null;
    switch (target.kind) {
      case "folder": {
        const path = target.prefix;
        const display = displayFolderPath(path);
        return {
          value: display,
          icon: "folder",
          colors: dirTagColors(path),
          title: display,
        };
      }
      case "cluster": {
        const name = this.clusterNames.get(target.id);
        return {
          value: name ?? `Cluster ${target.id}`,
          icon: "tag",
          colors: [clusterColor(target.id)],
          title: name ? `Cluster ${target.id}` : undefined,
        };
      }
      case "key":
        return { value: `Key ${target.value}` };
      case "bpm":
        return { value: `${Math.round(target.value)} BPM ±${BPM_MATCH_TOLERANCE}` };
      case "track": {
        const t = this.trackByPath(target.path);
        return { value: t ? (t.title || "").trim() || t.filename : target.path };
      }
      case "paths":
        return {
          value:
            target.label ??
            `${target.paths.length} track${target.paths.length === 1 ? "" : "s"}`,
          icon: target.label ? "tag" : undefined,
          colors: target.color ? [target.color] : [],
        };
    }
  }

  /**
   * The rule the selection-bloom isolates at *hovered*: the bloom cursor's variable
   * at its current range, in the same words as the bloom HUD, coloured by the
   * feature lens. Only called when the hovered dot's bloom class is non-empty.
   */
  bloomRule(hovered: Track): RuleChip {
    const range = this.bloomRange();
    switch (this.selectionBloom) {
      case "folder": {
        const levels = bloomFolderLevels(range);
        const prefix = folderAncestor(hovered.folder ?? "", levels);
        return {
          value: displayFolderPath(prefix),
          icon: "folder",
          colors: dirTagColors(prefix),
          title: levels > 0 ? bloomRangeLabel("folder", range) ?? undefined : undefined,
        };
      }
      case "head": {
        const levels = bloomFolderLevels(range);
        const prefix = folderAncestor(
          this.folderPredictions.get(hovered.path) ?? "",
          levels,
        );
        const note = levels > 0 ? bloomRangeLabel("head", range) : null;
        return {
          value: displayFolderPath(prefix),
          icon: "folder",
          colors: dirTagColors(prefix),
          title: ["Projected folder", note].filter(Boolean).join(" · "),
        };
      }
      case "cluster": {
        const id = this.clusterAssignments.get(hovered.path);
        if (id === undefined) return { value: "Unclustered", icon: "tag" };
        const name = this.clusterNames.get(id);
        return {
          value: name ?? `Cluster ${id}`,
          icon: "tag",
          colors: [clusterColor(id)],
          title: name ? `Cluster ${id}` : undefined,
        };
      }
      case "tempo": {
        const bpm = hovered.bpm;
        const pct = bloomTempoPct(range);
        const tnorm = trackTempoNorm(hovered);
        return {
          value:
            bpm == null || !Number.isFinite(bpm)
              ? "Tempo"
              : `${Math.round(bpm)} BPM${pct ? ` ±${pct}%` : ""}`,
          colors: tnorm != null ? [tempoColor(tnorm)] : [],
        };
      }
      case "key": {
        const steps = bloomKeySteps(range);
        const af = hovered.audio_features;
        return {
          value: `Key ${hovered.musical_key ?? "?"}${
            steps ? ` ±${steps} step${steps === 1 ? "" : "s"}` : ""
          }`,
          colors: af ? [keyColor(af.key, af.mode)] : [],
        };
      }
    }
  }

  /**
   * The rule a selection isolates: the folders its tracks live in, capped to two names
   * with the rest counted (the full list is the chip's hover title). A single folder
   * keeps its colour family; a set spanning folders has no single colour.
   */
  selectionRule(): RuleChip | null {
    const folders = new Set<string>();
    for (const t of this.allTracks) {
      if (this.selected.has(t.path)) folders.add(t.folder ?? "");
    }
    if (folders.size === 0) return null;
    const sorted = [...folders].sort();
    const shown = sorted.slice(0, 2).map((f) => displayFolderPath(f));
    const extra = sorted.length - shown.length;
    return {
      value: extra > 0 ? `${shown.join(", ")} +${extra} more` : shown.join(", "),
      icon: "folder",
      colors: sorted.length === 1 ? dirTagColors(sorted[0]) : [],
      title: sorted.map((f) => displayFolderPath(f)).join("\n"),
    };
  }

  /** Replace the selection with every visible track belonging to *target*. */
  selectOnlyHoverTarget(target: HoverTarget | null): void {
    if (!target) return;
    this.selected.clear();
    for (const t of this.tracks) {
      if (this.trackMatchesHoverTarget(t, target)) this.selected.add(t.path);
    }
    this.emit("change");
  }

  /**
   * Select the in-view tracks among *paths*: replace the current selection, or
   * (with ``append``) add them to it. A bare call over the target whose tracks
   * are already the whole selection clears it instead — mirroring how **S**
   * toggles an already-isolated target back to show-all. No-ops when the
   * selection would not change, so a redundant **F** press does not re-render.
   */
  selectPaths(paths: Iterable<string>, append: boolean): void {
    const wanted = new Set(paths);
    const next = append ? new Set(this.selected) : new Set<string>();
    for (const t of this.tracks) {
      if (wanted.has(t.path)) next.add(t.path);
    }
    const same =
      next.size === this.selected.size &&
      [...next].every((p) => this.selected.has(p));
    if (!append && this.selected.size > 0 && same) {
      this.selected.clear();
      this.emit("change");
      return;
    }
    if (same) return;
    this.selected.clear();
    for (const p of next) this.selected.add(p);
    this.emit("change");
  }

  /**
   * At least one track in the view union lies under ``prefixNorm``, and every such
   * track is excluded from the canvas.
   */
  folderSubtreeFullyExcluded(prefixNorm: string): boolean {
    let anyUnder = false;
    for (const t of this.allTracks) {
      if (!this.trackInViewUnion(t)) continue;
      const tf = t.folder ?? "";
      const under =
        prefixNorm === "" ||
        tf === prefixNorm ||
        tf.startsWith(prefixNorm + "/");
      if (!under) continue;
      anyUnder = true;
      if (!this.isTrackHidden(t)) return false;
    }
    return anyUnder;
  }

  private _mutateRevealFolderSubtree(prefixNorm: string): void {
    for (const p of [...this.hiddenFolderPrefixes]) {
      if (
        p === prefixNorm ||
        p === "" ||
        (p !== "" && (prefixNorm === p || prefixNorm.startsWith(p + "/")))
      ) {
        this.hiddenFolderPrefixes.delete(p);
      }
    }
    const next = new Set(this.hiddenPaths);
    for (const path of this.hiddenPaths) {
      const t = this.trackByPath(path);
      if (!t) {
        next.delete(path);
        continue;
      }
      const tf = t.folder ?? "";
      const under =
        prefixNorm === "" ||
        tf === prefixNorm ||
        tf.startsWith(prefixNorm + "/");
      if (under) next.delete(path);
    }
    this.hiddenPaths = next;
  }

  /**
   * Bring ``prefixNorm`` back into the view: remove folder exclusions and per-track
   * exclusions under that subtree (including ancestor folder exclusions that hide it).
   */
  revealFolderSubtree(prefixNorm: string): void {
    this._mutateRevealFolderSubtree(prefixNorm);
    this._invalidateView();
    this.emit("change");
  }

  clearExclusions(): void {
    if (!this.hasExclusions) return;
    this.hiddenPaths.clear();
    this.hiddenFolderPrefixes.clear();
    this._invalidateView();
    this.emit("change");
  }

  /**
   * Bring back a fully excluded path set, then exclude everything outside it
   * (single update). Used when **s** (isolate) is pressed on a hovered target that is
   * already entirely excluded.
   */
  revealThenFocusPaths(paths: Iterable<string>): void {
    const keep = new Set(paths);
    this._mutateRevealPaths(keep);
    this._invalidateView();
    this.focusPaths(keep);
  }

  /** Un-hide *paths* (and any folder prefix covering one) without changing the rest. */
  revealPaths(paths: Iterable<string>): void {
    this._mutateRevealPaths(paths);
    this._invalidateView();
    this.emit("change");
  }

  /** Shared reveal mutation: un-hide *paths* and any folder prefix that covers one. */
  private _mutateRevealPaths(paths: Iterable<string>): void {
    const keep = new Set(paths);
    for (const p of [...this.hiddenFolderPrefixes]) {
      if (p === "") {
        this.hiddenFolderPrefixes.delete(p);
        continue;
      }
      for (const path of keep) {
        const tf = this.trackByPath(path)?.folder ?? "";
        if (tf === p || tf.startsWith(p + "/")) {
          this.hiddenFolderPrefixes.delete(p);
          break;
        }
      }
    }
    const next = new Set(this.hiddenPaths);
    for (const path of keep) next.delete(path);
    this.hiddenPaths = next;
  }

  toggleAxis(which: "axisX" | "axisY", tag: string): void {
    this[which] = this[which] === tag ? null : tag;
    this.emit("change");
  }

  setViewMode(mode: ViewMode): void {
    if (this.viewMode === mode) return;
    this.viewMode = mode;
    this.emit("change");
  }

  setFilterRange(tag: string, r: [number, number]): void {
    this.filterRanges[tag] = r;
    this.emit("change");
  }

  /* ── Selection ────────────────────────────────────────────── */

  selectAll(): void {
    for (const t of this.tracks) this.selected.add(t.path);
    this.emit("change");
  }

  clearSelection(): void {
    if (!this.selected.size) return;
    this.selected.clear();
    this.emit("change");
  }

  /**
   * Replace the whole selection with *paths*, dropping paths the model does not
   * know (e.g. a search hit for a file added since the last library load).
   * No-ops when the resulting set equals the current one.
   */
  replaceSelection(paths: Iterable<string>): void {
    const next = new Set<string>();
    for (const p of paths) {
      if (this.trackByPath(p)) next.add(p);
    }
    if (
      next.size === this.selected.size &&
      [...next].every((p) => this.selected.has(p))
    )
      return;
    this.selected.clear();
    for (const p of next) this.selected.add(p);
    this.emit("change");
  }

  selectInFolder(folderPath: string): void {
    const norm = !folderPath || folderPath === "." ? "" : folderPath;
    for (const t of this.allTracks) {
      if (!this.trackInViewUnion(t) || this.isTrackHidden(t)) continue;
      const tf = t.folder ?? "";
      if (norm === "" || tf === norm || tf.startsWith(norm + "/"))
        this.selected.add(t.path);
    }
    this.emit("change");
  }

  /**
   * Add every visible track under *any* of the folder roots (recursive) in one update.
   */
  addTracksUnderFolderPrefixes(paths: Iterable<string>): void {
    const norms = new Set<string>();
    for (const raw of paths) {
      norms.add(!raw || raw === "." ? "" : raw);
    }
    for (const t of this.allTracks) {
      if (!this.trackInViewUnion(t) || this.isTrackHidden(t)) continue;
      const tf = t.folder ?? "";
      for (const norm of norms) {
        if (norm === "" || tf === norm || tf.startsWith(norm + "/")) {
          this.selected.add(t.path);
          break;
        }
      }
    }
    this.emit("change");
  }

  /** Replace selection with every visible track in this folder (and subfolders). */
  selectOnlyInFolder(folderPath: string): void {
    const norm = !folderPath || folderPath === "." ? "" : folderPath;
    this.selectOnlyHoverTarget({ kind: "folder", prefix: norm });
  }

  /* ── Tags ─────────────────────────────────────────────────── */

  addKnownTag(name: string): void {
    if (!this._knownTags.includes(name)) {
      this._knownTags.push(name);
      this._knownTags.sort();
    }
  }

  renameTag(oldName: string, newName: string): void {
    if (isAudioFeatureAxisId(oldName)) return;
    // Update tag values in every track
    for (const t of this.allTracks) {
      if (oldName in t.tags) {
        t.tags[newName] = t.tags[oldName];
        delete t.tags[oldName];
      }
    }
    // Update known-tags list
    const idx = this._knownTags.indexOf(oldName);
    if (idx !== -1) this._knownTags[idx] = newName;
    if (!this._knownTags.includes(newName)) this._knownTags.push(newName);
    this._knownTags = [...new Set(this._knownTags)].sort();

    // Update axis assignments
    if (this.axisX === oldName) this.axisX = newName;
    if (this.axisY === oldName) this.axisY = newName;

    // Migrate filter range
    if (oldName in this.filterRanges) {
      this.filterRanges[newName] = this.filterRanges[oldName];
      delete this.filterRanges[oldName];
    }

    this._dirty();
  }

  /* ── Track mutations ──────────────────────────────────────── */

  applyFolderRename(oldRel: string, newRel: string): void {
    for (const t of this.allTracks) {
      if (t.folder === oldRel || t.folder.startsWith(oldRel + "/")) {
        t.folder = newRel + t.folder.slice(oldRel.length);
        t.path = (t.folder ? t.folder + "/" : "") + t.filename;
      }
    }
    const upd = (s: string) =>
      s === oldRel || s.startsWith(oldRel + "/")
        ? newRel + s.slice(oldRel.length)
        : s;
    if (
      this.folder === oldRel ||
      this.folder.startsWith(oldRel + "/")
    ) {
      this.folder = upd(this.folder);
    }
    this.hiddenFolderPrefixes = new Set(
      [...this.hiddenFolderPrefixes].map((p) => upd(p)),
    );
    const nh = new Set<string>();
    for (const p of this.hiddenPaths) {
      nh.add(
        p === oldRel || p.startsWith(oldRel + "/")
          ? newRel + p.slice(oldRel.length)
          : p,
      );
    }
    this.hiddenPaths = nh;
    this._dirty();
  }

  /** Update in-memory paths after a successful move (before library reload completes). */
  applyTracksMoved(oldPaths: string[], destFolder: string): void {
    const normDest =
      !destFolder || destFolder === "." ? "" : destFolder;
    for (const p of oldPaths) {
      const t = this.trackByPath(p);
      if (!t) continue;
      const fn = t.filename;
      const newPath = normDest ? `${normDest}/${fn}` : fn;
      if (this.hiddenPaths.has(p)) {
        this.hiddenPaths.delete(p);
        this.hiddenPaths.add(newPath);
      }
      t.folder = normDest;
      t.path = newPath;
    }
    this.selected.clear();
    this._dirty();
    this.emit("change");
  }

  /* ── Persistence ──────────────────────────────────────────── */

  saveLS(): void {
    // Eval mode is session-only: a mid-gallery save must never rewrite the
    // user's normal-mode preferences. In-memory state still changes for the
    // session; the next non-eval save persists the real values again.
    if (this.evalMode) return;
    localStorage.setItem(
      LS_KEY,
      JSON.stringify({
        folder: this.folder,
        recursive: this.recursive,
        axisX: this.axisX,
        axisY: this.axisY,
        filterRanges: this.filterRanges,
        hoverPreview: this.hoverPreview,
        previewVolume: this.previewVolume,
        lensId: this._lensId,
        lensPresets: this.lensPresets,
        scopeLensId: this.scopeLensId,
        clusterResolution: this.clusterResolution,
        colorMode: this.colorMode,
        selectionBloom: this.selectionBloom,
        bloomRanges: this.bloomRanges,
        headColorConfident: this.headColorConfident,
        clusterWeights: this.clusterWeights,
        genreWeights: this.genreWeights,
        trainFolders: this.trainFolders,
        showExternal: this.showExternal,
        externalInProjection: this.externalInProjection,
        rightPanelTab: this.rightPanelTab,
        view3d: this.view3d,
        tsneIteration: this.tsneIteration,
      }),
    );
    localStorage.setItem(LS_KEY + "_tags", JSON.stringify(this._knownTags));
  }

  loadLS(): void {
    // A fresh startup (no saved data) lands on the Genre map, the user-agreed
    // default; the saved-data block below overrides with a persisted / migrated
    // lens when one exists.
    this._lensId = "genre_full";
    try {
      const d = JSON.parse(localStorage.getItem(LS_KEY) ?? "null");
      if (d) {
        // Ignore legacy ``folder`` / ``viewFolders`` “library scope” — it only
        // hid tracks and is easy to get stuck via cached localStorage. Sidebar
        // focus resets to root until you click a folder label.
        this.folder = "";
        this.recursive = true;
        this.axisX = _migrateLibrosaAxisId(d.axisX ?? null);
        this.axisY = _migrateLibrosaAxisId(d.axisY ?? null);
        this.filterRanges = _migrateFilterRanges(d.filterRanges);
        this.hoverPreview = d.hoverPreview ?? false;
        // A stored 0 is indistinguishable from "never set" and reads as a broken,
        // silent preview, so it falls back to the default; mute is the M toggle.
        const rawPv = (d as { previewVolume?: unknown }).previewVolume;
        this.previewVolume =
          typeof rawPv === "number" && Number.isFinite(rawPv) && rawPv > 0
            ? Math.min(1, rawPv)
            : DEFAULT_PREVIEW_VOLUME;
        this._lensId =
          typeof d.lensId === "string"
            ? _migrateLegacyMapLensId(d.lensId)
            : // A fresh startup (no saved lens) lands on the Genre map, the
              // user-agreed default. A legacy ``viewMode: "tags"`` choice is
              // preserved; everything else defaults to the genre map and the
              // coordinator reconciles it to the combined fallback if the server
              // did not enable genre.
              d.viewMode === "tags"
              ? "tags"
              : "genre_full";
        this.lensPresets = _migrateLensPresets(
          (d as { lensPresets?: unknown }).lensPresets,
        );
        if (typeof (d as { scopeLensId?: unknown }).scopeLensId === "string") {
          this.scopeLensId = _migrateLegacyMapLensId(
            (d as { scopeLensId: string }).scopeLensId,
          );
        }
        const cr = Number((d as { clusterResolution?: number }).clusterResolution);
        this.clusterResolution = Number.isFinite(cr) && cr > 0 ? cr : 1.5;
        this.colorMode = _migrateColorMode(
          (d as { colorMode?: unknown }).colorMode,
          (d as { colorByCluster?: unknown }).colorByCluster,
        );
        this.selectionBloom = _migrateColorMode(
          (d as { selectionBloom?: unknown }).selectionBloom,
        );
        // Legacy ``bloomAmount`` (a visual scale) and the single ``bloomRange``
        // both predate the per-variable ranges; carry either over into every
        // slot so an existing value is not lost.
        this.bloomRanges = _migrateBloomRanges(
          (d as { bloomRanges?: unknown }).bloomRanges,
          (d as { bloomRange?: unknown }).bloomRange ??
            (d as { bloomAmount?: unknown }).bloomAmount,
        );
        this.headColorConfident =
          (d as { headColorConfident?: unknown }).headColorConfident !== false;
        this.clusterWeights = _migrateClusterWeights(
          (d as { clusterWeights?: unknown }).clusterWeights,
        );
        this.genreWeights = _migrateGenreWeights(
          (d as { genreWeights?: unknown }).genreWeights,
        );
        const tf = (d as { trainFolders?: unknown }).trainFolders;
        this.trainFolders = Array.isArray(tf)
          ? tf.filter((p): p is string => typeof p === "string")
          : [];
        this.showExternal = Boolean(
          (d as { showExternal?: boolean }).showExternal,
        );
        this.externalInProjection = Boolean(
          (d as { externalInProjection?: boolean }).externalInProjection,
        );
        const rpTab = (d as { rightPanelTab?: unknown }).rightPanelTab;
        this.rightPanelTab = RIGHT_PANEL_TABS.includes(rpTab as RightPanelTab)
          ? (rpTab as RightPanelTab)
          : "analysis";
        // A stored orbit is accepted only when it is fully valid: `on` must be
        // exactly true and both angles finite numbers. Anything else (missing
        // key, legacy shape, bad values) keeps the defaults — 3-D must never
        // come on by accident.
        const v3 = (d as { view3d?: unknown }).view3d;
        if (v3 && typeof v3 === "object") {
          const r = v3 as { on?: unknown; yaw?: unknown; pitch?: unknown };
          if (
            r.on === true &&
            typeof r.yaw === "number" &&
            Number.isFinite(r.yaw) &&
            typeof r.pitch === "number" &&
            Number.isFinite(r.pitch)
          ) {
            this.view3d = {
              on: true,
              yaw: r.yaw,
              pitch: Math.max(-MAX_PITCH, Math.min(MAX_PITCH, r.pitch)),
            };
          }
        }
        const ti = (d as { tsneIteration?: unknown }).tsneIteration;
        if (ti && typeof ti === "object") {
          // Legacy global override — a single ``{nIter, early}`` that used to apply
          // to every view. Its view is unknowable, so migrate it to the *active*
          // saved dimensionality only (2-D and 3-D keep separate budgets).
          // The per-dimensionality map is sanitised entry by entry.
          if ("nIter" in ti || "early" in ti) {
            const t = ti as { nIter?: unknown; early?: unknown };
            const nIter =
              typeof t.nIter === "number" && Number.isFinite(t.nIter)
                ? t.nIter
                : null;
            const early =
              typeof t.early === "number" && Number.isFinite(t.early)
                ? t.early
                : null;
            if (nIter !== null || early !== null) {
              this.tsneIteration[
                Model.tsneIterationKey(this.view3d.on ? 3 : 2)
              ] = { nIter, early };
            }
          } else {
            // Per-dimensionality map. ``d2``/``d3`` keys pass through; the older
            // per-view keys (``lensId:d2``/``lensId:d3``) are collapsed to one
            // representative per dimension: the active saved lens's own entry for
            // the active dimension, otherwise (and for the other dimension)
            // deterministically the genre lens, then the first sorted valid entry.
            const byDim: Record<2 | 3, Array<{ lens: string; nIter: number | null; early: number | null }>> =
              { 2: [], 3: [] };
            const entry = (
              v: unknown,
            ): { nIter: number | null; early: number | null } | null => {
              if (!v || typeof v !== "object") return null;
              const e = v as { nIter?: unknown; early?: unknown };
              const nIter =
                typeof e.nIter === "number" && Number.isFinite(e.nIter)
                  ? e.nIter
                  : null;
              const early =
                typeof e.early === "number" && Number.isFinite(e.early)
                  ? e.early
                  : null;
              return nIter === null && early === null ? null : { nIter, early };
            };
            for (const [key, val] of Object.entries(
              ti as Record<string, unknown>,
            )) {
              const direct = /^d[23]$/.exec(key);
              const legacy = /^(.+):d([23])$/.exec(key);
              const e = direct ? entry(val) : null;
              if (direct && e) {
                this.tsneIteration[key] = e;
                continue;
              }
              if (!legacy) continue;
              const v = entry(val);
              if (!v) continue;
              const dim = legacy[2] === "3" ? 3 : 2;
              byDim[dim].push({ lens: legacy[1], ...v });
            }
            const activeDim = this.view3d.on ? 3 : 2;
            const activeLens = this.activeBaseLensId();
            for (const dim of [2, 3] as const) {
              const list = byDim[dim];
              if (list.length === 0) continue;
              let chosen =
                dim === activeDim
                  ? list.find((e) => e.lens === activeLens)
                  : undefined;
              chosen ??= list.find((e) => e.lens === GENRE_FULL_LENS.id);
              chosen ??= [...list].sort((a, b) =>
                a.lens.localeCompare(b.lens),
              )[0];
              this.tsneIteration[Model.tsneIterationKey(dim)] = {
                nIter: chosen.nIter,
                early: chosen.early,
              };
            }
          }
        }
      }
    } catch {
      /* start fresh */
    }
    // The camera blend is transient: after a reload the view is fully 3-D or
    // fully 2-D — never mid-tween, which would look like a torn frame.
    this.view3dBlend = this.view3d.on ? 1 : 0;
    try {
      this._knownTags =
        JSON.parse(localStorage.getItem(LS_KEY + "_tags") ?? "null") || [];
    } catch {
      this._knownTags = [];
    }
  }
}
