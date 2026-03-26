export interface Track {
  path: string;
  filename: string;
  folder: string;
  tags: Record<string, number>;
  artist: string;
  title: string;
  fingerprint?: string | null;
}

export type ViewMode = "tags" | "embeddings";
export type ProjectionMethod = "umap" | "tsne";

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

// ─── Model ───────────────────────────────────────────────────
export class Model extends EventBus {
  allTracks: Track[] = [];
  folder = "";
  recursive = true;
  axisX: string | null = null;
  axisY: string | null = null;
  filterRanges: Record<string, [number, number]> = {};
  selected = new Set<string>();
  hoverPreview = false;
  vp: Viewport = { ox: 0, oy: 0, zoom: 1 };
  _knownTags: string[] = [];

  viewMode: ViewMode = "tags";
  projectionMethod: ProjectionMethod = "tsne";
  scaleByTags = true;
  scaleByFolders = true;
  useCLAP = true;
  useEffNet = false;
  /** Audio descriptors (librosa 6-vector); each toggles dims before projection. */
  useAudioFeatureTempo = false;
  useAudioFeatureKey = false;
  useAudioFeatureMode = false;
  useAudioFeatureEnergy = false;
  useAudioFeatureDance = false;
  /** Overall gain for the audio-feature block (matches server default). */
  featuresBlend = 0.42;
  /** Folder semantic re-weighting strength (server default 3). */
  folderContrastBoost = 3.0;
  /** Emphasis on deeper folder contrasts; 1=flat, 3=strong edge boost. */
  folderDepthBoost = 1.5;
  embeddingPositions: Map<string, { x: number; y: number }> = new Map();
  embeddingsReady = false;
  embeddingsGenerating = false;
  embeddingProgress: { done: number; total: number } | null = null;
  projectionPending = false;
  libraryLoadProgress: { done: number; total: number } | null = null;

  private _cache: Track[] | null = null;
  private _pathMap: Map<string, Track> | null = null;

  /* ── Computed ──────────────────────────────────────────────── */

  get tracks(): Track[] {
    if (this._cache) return this._cache;
    const f = this.folder;
    const result =
      f === "" && this.recursive
        ? this.allTracks
        : this.allTracks.filter((t) => {
            const tf = t.folder ?? "";
            return this.recursive
              ? tf === f || tf.startsWith(f + "/")
              : tf === f;
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
    return [...s].sort((a, b) => {
      const ha = withVals.has(a),
        hb = withVals.has(b);
      if (ha !== hb) return ha ? -1 : 1;
      return a < b ? -1 : a > b ? 1 : 0;
    });
  }

  tagHasValuesInView(tag: string): boolean {
    return this.tracks.some((t) => t.tags[tag] !== undefined);
  }

  /** Tag names for semantic weighting (all get full boost). */
  get contextTags(): string[] {
    return [...this.tags];
  }

  /** Unique folder paths for semantic weighting (depth determines boost). */
  get contextFolders(): string[] {
    const s = new Set<string>();
    for (const t of this.allTracks) {
      if (t.folder) s.add(t.folder);
    }
    return [...s];
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
      const v = track.tags[tag];
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

  setFolder(f: string): void {
    this.folder = f;
    this._cache = null;
    this.selected.clear();
    this.emit("change");
  }

  setRecursive(r: boolean): void {
    this.recursive = r;
    this._cache = null;
    this.selected.clear();
    this.emit("change");
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

  setProjectionMethod(method: ProjectionMethod): void {
    if (this.projectionMethod === method) return;
    this.projectionMethod = method;
    this.emit("change");
  }

  get anyAudioFeaturesEnabled(): boolean {
    return (
      this.useAudioFeatureTempo ||
      this.useAudioFeatureKey ||
      this.useAudioFeatureMode ||
      this.useAudioFeatureEnergy ||
      this.useAudioFeatureDance
    );
  }

  /** Six ``0``/``1`` chars: tempo, key_cos, key_sin, mode, energy, danceability. */
  audioFeatureMask(): string {
    const b = (x: boolean) => (x ? "1" : "0");
    return [
      b(this.useAudioFeatureTempo),
      b(this.useAudioFeatureKey),
      b(this.useAudioFeatureKey),
      b(this.useAudioFeatureMode),
      b(this.useAudioFeatureEnergy),
      b(this.useAudioFeatureDance),
    ].join("");
  }

  get activeSources(): string[] {
    const s: string[] = [];
    if (this.useCLAP) s.push("clap");
    if (this.useEffNet) s.push("effnet");
    if (this.anyAudioFeaturesEnabled) s.push("features");
    return s;
  }

  toggleSource(source: "clap" | "effnet"): void {
    const field = source === "clap" ? "useCLAP" : "useEffNet";
    const next = !this[field];
    if (!next && this.activeSources.length <= 1) return;
    (this as Record<string, unknown>)[field] = next;
    this.emit("change");
  }

  toggleAudioFeature(
    dim: "tempo" | "key" | "mode" | "energy" | "dance",
  ): void {
    const field =
      dim === "tempo"
        ? "useAudioFeatureTempo"
        : dim === "key"
          ? "useAudioFeatureKey"
          : dim === "mode"
            ? "useAudioFeatureMode"
            : dim === "energy"
              ? "useAudioFeatureEnergy"
              : "useAudioFeatureDance";
    const prev = this[field as keyof Model] as boolean;
    const next = !prev;
    if (!next && !this.useCLAP && !this.useEffNet && !this._anyAudioBesides(field)) {
      return;
    }
    (this as Record<string, unknown>)[field] = next;
    this.emit("change");
  }

  private _anyAudioBesides(
    field: "useAudioFeatureTempo" | "useAudioFeatureKey" | "useAudioFeatureMode" | "useAudioFeatureEnergy" | "useAudioFeatureDance",
  ): boolean {
    const m: Record<string, boolean> = {
      useAudioFeatureTempo: this.useAudioFeatureTempo,
      useAudioFeatureKey: this.useAudioFeatureKey,
      useAudioFeatureMode: this.useAudioFeatureMode,
      useAudioFeatureEnergy: this.useAudioFeatureEnergy,
      useAudioFeatureDance: this.useAudioFeatureDance,
    };
    m[field] = false;
    return Object.values(m).some(Boolean);
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

  selectInFolder(folderPath: string): void {
    const norm = !folderPath || folderPath === "." ? "" : folderPath;
    for (const t of this.tracks) {
      const tf = t.folder ?? "";
      if (norm === "" || tf === norm || tf.startsWith(norm + "/"))
        this.selected.add(t.path);
    }
    this.emit("change");
  }

  /* ── Tags ─────────────────────────────────────────────────── */

  addKnownTag(name: string): void {
    if (!this._knownTags.includes(name)) {
      this._knownTags.push(name);
      this._knownTags.sort();
    }
  }

  renameTag(oldName: string, newName: string): void {
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
    if (
      this.folder === oldRel ||
      this.folder.startsWith(oldRel + "/")
    ) {
      this.folder = newRel + this.folder.slice(oldRel.length);
    }
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
      t.folder = normDest;
      t.path = normDest ? `${normDest}/${fn}` : fn;
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
        viewMode: this.viewMode,
        projectionMethod: this.projectionMethod,
        scaleByTags: this.scaleByTags,
        scaleByFolders: this.scaleByFolders,
        useCLAP: this.useCLAP,
        useEffNet: this.useEffNet,
        useAudioFeatureTempo: this.useAudioFeatureTempo,
        useAudioFeatureKey: this.useAudioFeatureKey,
        useAudioFeatureMode: this.useAudioFeatureMode,
        useAudioFeatureEnergy: this.useAudioFeatureEnergy,
        useAudioFeatureDance: this.useAudioFeatureDance,
        featuresBlend: this.featuresBlend,
        folderContrastBoost: this.folderContrastBoost,
        folderDepthBoost: this.folderDepthBoost,
      }),
    );
    localStorage.setItem(LS_KEY + "_tags", JSON.stringify(this._knownTags));
  }

  loadLS(): void {
    try {
      const d = JSON.parse(localStorage.getItem(LS_KEY) ?? "null");
      if (d) {
        this.folder = d.folder ?? "";
        this.recursive = true;
        this.axisX = d.axisX ?? null;
        this.axisY = d.axisY ?? null;
        this.filterRanges = d.filterRanges ?? {};
        this.hoverPreview = d.hoverPreview ?? false;
        this.viewMode = d.viewMode ?? "tags";
        this.projectionMethod = d.projectionMethod ?? "tsne";
        this.scaleByTags = d.scaleByTags ?? true;
        this.scaleByFolders = d.scaleByFolders ?? true;
        this.useCLAP = d.useCLAP ?? true;
        this.useEffNet = d.useEffNet ?? false;
        const legacyAudio = (d as { useAudioFeatures?: boolean }).useAudioFeatures;
        this.useAudioFeatureTempo = d.useAudioFeatureTempo ?? legacyAudio ?? false;
        this.useAudioFeatureKey = d.useAudioFeatureKey ?? legacyAudio ?? false;
        this.useAudioFeatureMode = d.useAudioFeatureMode ?? legacyAudio ?? false;
        this.useAudioFeatureEnergy = d.useAudioFeatureEnergy ?? legacyAudio ?? false;
        this.useAudioFeatureDance = d.useAudioFeatureDance ?? legacyAudio ?? false;
        this.featuresBlend = d.featuresBlend ?? 0.42;
        this.folderContrastBoost = d.folderContrastBoost ?? 3.0;
        this.folderDepthBoost = d.folderDepthBoost ?? 1.5;
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
