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
 * Default per-source weights for clustering **and** the 2-D layout. Mirrors the
 * server's ``clustering.DEFAULT_WEIGHTS`` / ``projection_config.FROZEN_SOURCE_WEIGHTS``
 * — the map and the clusters must describe the same space, so one user-controlled
 * vector drives both. Persisted; "Reset" restores these values.
 */
export const DEFAULT_CLUSTER_WEIGHTS: Record<string, number> = {
  maest_logits: 0.45,
  maest: 0.25,
  rhythm: 0.15,
  clap: 0.1,
  supervised: 0.15,
  features: 0,
};

/** Slider labels, keyed by source name (insertion order = display order). */
export const CLUSTER_WEIGHT_LABELS: Record<string, string> = {
  maest_logits: "Genre",
  maest: "MAEST",
  rhythm: "Rhythm",
  clap: "CLAP",
  supervised: "Folders",
  features: "Audio",
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
};

function _migrateClusterWeights(raw: unknown): Record<string, number> {
  const out: Record<string, number> = { ...DEFAULT_CLUSTER_WEIGHTS };
  if (raw && typeof raw === "object") {
    for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
      if (!(k in out)) continue;
      const n = Number(v);
      if (Number.isFinite(n)) out[k] = Math.min(1, Math.max(0, n));
    }
  }
  return out;
}

/** Compact display / query-string form for a source weight (drops trailing zeros). */
export function formatSourceWeight(v: number): string {
  return Number(v.toFixed(4)).toString();
}

export function isAudioFeatureAxisId(tag: string): boolean {
  return _LIBROSA_FEATURE_SET.has(tag);
}

export type ViewMode = "tags" | "embeddings";

/** Hover-preview audition section; cycled with **D** and highlighted in the tooltip. */
export type PreviewSection = "intro" | "drop" | "break";

/**
 * A list-like navigation element under the pointer: a folder row, a cluster row, or a
 * card chip (key / BPM). One concept so the glow, **s** (solo), **h** (fit), **f**
 * (select) and **Shift+s** (exclude) affordances behave the same wherever the element
 * is rendered. See `design-patterns.md` → *Navigation targets*.
 */
export type HoverTarget =
  | { kind: "folder"; prefix: string }
  | { kind: "cluster"; id: number }
  | { kind: "key"; value: string }
  | { kind: "bpm"; value: number }
  | { kind: "track"; path: string };

/** BPM window that counts as "the same tempo" for the BPM chip's hover target. */
export const BPM_MATCH_TOLERANCE = 2;

export interface Viewport {
  ox: number;
  oy: number;
  zoom: number;
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
  const v = Math.max(0, Math.min(1, volume));
  return v * v;
}

// ─── Model ───────────────────────────────────────────────────
export class Model extends EventBus {
  allTracks: Track[] = [];
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
  vp: Viewport = { ox: 0, oy: 0, zoom: 1 };
  _knownTags: string[] = [];

  viewMode: ViewMode = "tags";
  embeddingPositions: Map<string, { x: number; y: number }> = new Map();
  embeddingsReady = false;
  embeddingsGenerating = false;
  embeddingProgress: {
    done: number;
    total: number;
    lastPath?: string;
    lastOk?: boolean;
    lastFailures?: string[];
  } | null = null;
  projectionPending = false;
  /**
   * In-flight phase of a layout request, streamed on `/api/embeddings/stream`.
   * The layout is a plain request rather than a queued job, so it carries its own
   * phase here instead of riding `queueStatus`.
   */
  layoutProgress: { phase: string; done: number; total: number } | null = null;
  /** True while the required models are not ready yet (embedding mode waits on `_pollModelReady`). */
  embeddingModelsLoading = false;
  /**
   * Hold **X** on the canvas: nudge dots by tag axes (embedding mode) or toward
   * embedding layout (tag mode). Strength 0.05–0.20 of a unit square chord.
   */
  explodeStrength = 0.1;
  /** While true, user is holding X (not persisted). */
  explodeHold = false;
  /** Hide explode offset during tag drag / transform so coordinates stay consistent. */
  suppressExplodeVisual = false;
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
   * selected resolution; ``clusterNames`` labels those ids. Colouring by
   * cluster is what makes the microgenre structure legible on the map.
   */
  clusterAssignments: Map<string, number> = new Map();
  clusterNames: Map<number, string> = new Map();
  clusterSizes: Map<number, number> = new Map();
  /** Available Leiden resolutions (coarse → fine), from the server. */
  clusterResolutions: number[] = [];
  /** Currently displayed resolution; persisted. */
  clusterResolution = 1.5;
  /** Colour dots by cluster instead of by folder; persisted. */
  colorByCluster = false;
  /**
   * Per-source weights for clustering and the layout; persisted. One vector drives
   * both pipelines so the map and cluster assignments describe the same space.
   */
  clusterWeights: Record<string, number> = { ...DEFAULT_CLUSTER_WEIGHTS };
  clusterRevision: string | null = null;
  clustersAvailable = false;
  /**
   * Virtual folders the next head training run draws labels from; persisted.
   * Empty means every root (the server's default).
   */
  trainFolders: string[] = [];

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

  private _cache: Track[] | null = null;
  private _pathMap: Map<string, Track> | null = null;

  /** Hidden by path (session-only). */
  hiddenPaths = new Set<string>();
  /** Hide every track in this folder and subfolders (normalised "", not "."). */
  hiddenFolderPrefixes = new Set<string>();
  /** Root ids whose role is "external" (not My library); from `/api/folders`. */
  externalRootIds = new Set<string>();
  /** Include external-root tracks in the view (persisted, default off). */
  showExternal = false;

  /* ── Computed ──────────────────────────────────────────────── */

  /**
   * True when the track belongs to the current view scope. External roots are
   * analysed and laid out like any other, but stay out of the default view until
   * "Show external" is on — so the map still shows where they would land.
   */
  trackInViewUnion(t: Track): boolean {
    return this.showExternal || !this.isExternalTrack(t);
  }

  /** True when *t* belongs to a root marked external (analysed, not My library). */
  isExternalTrack(t: Track): boolean {
    const folder = t.folder ?? "";
    const slash = folder.indexOf("/");
    const rootId = slash >= 0 ? folder.slice(0, slash) : folder;
    return this.externalRootIds.has(rootId);
  }

  isTrackHidden(t: Track): boolean {
    if (this.hiddenPaths.has(t.path)) return true;
    const tf = t.folder ?? "";
    for (const p of this.hiddenFolderPrefixes) {
      if (p === "") return true;
      if (tf === p || tf.startsWith(p + "/")) return true;
    }
    return false;
  }

  /** True while any track or folder subtree is excluded from the current view. */
  get hasExclusions(): boolean {
    return this.hiddenPaths.size > 0 || this.hiddenFolderPrefixes.size > 0;
  }

  /** Visible tracks under ``prefix`` (``""`` or ``"a/b"``), respecting view union + hides. */
  hasVisibleTracksUnderTreePrefix(treePrefix: string): boolean {
    for (const t of this.allTracks) {
      if (!this.trackInViewUnion(t) || this.isTrackHidden(t)) continue;
      const tf = t.folder ?? "";
      if (treePrefix === "" || tf === treePrefix || tf.startsWith(treePrefix + "/"))
        return true;
    }
    return false;
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

  passesFilter(track: Track): boolean {
    for (const [tag, [lo, hi]] of Object.entries(this.filterRanges)) {
      let v: number | undefined;
      if (isAudioFeatureAxisId(tag))
        v = this._audioFeatureScalar(track, tag) ?? track.tags[tag];
      else v = track.tags[tag];
      if (v === undefined) continue;
      if (v < lo || v > hi) return false;
    }
    return true;
  }

  /* ── Internal ─────────────────────────────────────────────── */

  private _dirty(): void {
    this._cache = null;
    this._pathMap = null;
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
    this._cache = null;
    this.selected.clear();
    this.emit("change");
  }

  hidePaths(paths: Iterable<string>): void {
    this.hiddenPaths = new Set([...this.hiddenPaths, ...paths]);
    this._cache = null;
    this._clearExclusionsIfNoneVisible();
    this.emit("change");
  }

  hideFolderTree(prefixNorm: string): void {
    this.hiddenFolderPrefixes.add(prefixNorm);
    this._cache = null;
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
    this._cache = null;
    this._clearExclusionsIfNoneVisible();
    this.emit("change");
  }

  setShowExternal(show: boolean): void {
    if (this.showExternal === show) return;
    this.showExternal = show;
    this._cache = null;
    this._clearExclusionsIfNoneVisible();
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
    this._cache = null;
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
    }
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
    this._cache = null;
    this.emit("change");
  }

  clearExclusions(): void {
    if (!this.hasExclusions) return;
    this.hiddenPaths.clear();
    this.hiddenFolderPrefixes.clear();
    this._cache = null;
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
    this._cache = null;
    this.focusPaths(keep);
  }

  /** Un-hide *paths* (and any folder prefix covering one) without changing the rest. */
  revealPaths(paths: Iterable<string>): void {
    this._mutateRevealPaths(paths);
    this._cache = null;
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
        viewMode: this.viewMode,
        explodeStrength: this.explodeStrength,
        clusterResolution: this.clusterResolution,
        colorByCluster: this.colorByCluster,
        clusterWeights: this.clusterWeights,
        trainFolders: this.trainFolders,
        showExternal: this.showExternal,
      }),
    );
    localStorage.setItem(LS_KEY + "_tags", JSON.stringify(this._knownTags));
  }

  loadLS(): void {
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
        this.viewMode = d.viewMode ?? "tags";
        const ex = Number((d as { explodeStrength?: number }).explodeStrength);
        this.explodeStrength = Number.isFinite(ex)
          ? Math.min(0.2, Math.max(0.05, ex))
          : 0.1;
        const cr = Number((d as { clusterResolution?: number }).clusterResolution);
        this.clusterResolution = Number.isFinite(cr) && cr > 0 ? cr : 1.5;
        this.colorByCluster = Boolean(
          (d as { colorByCluster?: boolean }).colorByCluster,
        );
        this.clusterWeights = _migrateClusterWeights(
          (d as { clusterWeights?: unknown }).clusterWeights,
        );
        const tf = (d as { trainFolders?: unknown }).trainFolders;
        this.trainFolders = Array.isArray(tf)
          ? tf.filter((p): p is string => typeof p === "string")
          : [];
        this.showExternal = Boolean(
          (d as { showExternal?: boolean }).showExternal,
        );
      }
    } catch {
      /* start fresh */
    }
    try {
      this._knownTags =
        JSON.parse(localStorage.getItem(LS_KEY + "_tags") ?? "null") || [];
    } catch {
      this._knownTags = [];
    }
  }
}
