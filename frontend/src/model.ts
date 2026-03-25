export interface Track {
  path: string;
  filename: string;
  folder: string;
  tags: Record<string, number>;
  artist: string;
  title: string;
}

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
