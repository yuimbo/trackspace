"use strict";

const LS_KEY = "trackspace";

// ─── Minimal event bus ───────────────────────────────────────
class EventBus {
  constructor() { this._h = {}; }
  on(e, fn)  { (this._h[e] ??= []).push(fn); }
  off(e, fn) { const a = this._h[e]; if (a) this._h[e] = a.filter(f => f !== fn); }
  emit(e, ...a) { for (const fn of this._h[e] ?? []) fn(...a); }
}

// ─── Model ───────────────────────────────────────────────────
export class Model extends EventBus {
  constructor() {
    super();
    this.allTracks    = [];
    this.folderTree   = null;
    this.folder       = "";
    this.recursive    = true;
    this.axisX        = null;
    this.axisY        = null;
    this.filterRanges = {};
    this.selected     = new Set();   // Set<trackPath>
    this.hoverPreview = false;
    this.vp           = { ox: 0, oy: 0, zoom: 1 };
    this._knownTags   = [];
    this._cache       = null;
    this._pathMap     = null;
  }

  /* ── Computed ──────────────────────────────────────────────── */

  get tracks() {
    if (this._cache) return this._cache;
    const f = this.folder;
    if (f === "" && this.recursive) {
      this._cache = this.allTracks;
    } else {
      this._cache = this.allTracks.filter(t => {
        const tf = t.folder ?? "";
        return this.recursive
          ? (tf === f || tf.startsWith(f + "/"))
          : tf === f;
      });
    }
    return this._cache;
  }

  get tags() {
    const s = new Set(this._knownTags);
    for (const t of this.allTracks)
      for (const k of Object.keys(t.tags)) s.add(k);
    const withVals = new Set();
    for (const t of this.tracks)
      for (const k of Object.keys(t.tags)) withVals.add(k);
    return [...s].sort((a, b) => {
      const ha = withVals.has(a), hb = withVals.has(b);
      if (ha !== hb) return ha ? -1 : 1;
      return a < b ? -1 : a > b ? 1 : 0;
    });
  }

  tagHasValuesInView(tag) {
    return this.tracks.some(t => t.tags[tag] !== undefined);
  }

  trackByPath(path) {
    if (!this._pathMap) {
      this._pathMap = new Map();
      for (const t of this.allTracks) this._pathMap.set(t.path, t);
    }
    return this._pathMap.get(path);
  }

  passesFilter(track) {
    for (const [tag, [lo, hi]] of Object.entries(this.filterRanges)) {
      const v = track.tags[tag];
      if (v === undefined) continue;
      if (v < lo || v > hi) return false;
    }
    return true;
  }

  /* ── Internal ─────────────────────────────────────────────── */

  _dirty() { this._cache = null; this._pathMap = null; }

  /* ── Mutators ─────────────────────────────────────────────── */

  setAllTracks(tracks) {
    this.allTracks = tracks;
    this._dirty();
    this.selected.clear();
    this.emit("change");
  }

  setFolderTree(tree) {
    this.folderTree = tree;
    this.emit("tree");
  }

  setFolder(f) {
    this.folder = f;
    this._cache = null;
    this.selected.clear();
    this.emit("change");
  }

  setRecursive(r) {
    this.recursive = r;
    this._cache = null;
    this.selected.clear();
    this.emit("change");
  }

  toggleAxis(which, tag) {
    this[which] = this[which] === tag ? null : tag;
    this.emit("change");
  }

  setFilterRange(tag, r) { this.filterRanges[tag] = r; this.emit("change"); }

  /* ── Selection ────────────────────────────────────────────── */

  selectAll() {
    for (const t of this.tracks) this.selected.add(t.path);
    this.emit("change");
  }

  clearSelection() {
    if (!this.selected.size) return;
    this.selected.clear();
    this.emit("change");
  }

  selectInFolder(folderPath) {
    const norm = (!folderPath || folderPath === ".") ? "" : folderPath;
    for (const t of this.tracks) {
      const tf = t.folder ?? "";
      if (norm === "" || tf === norm || tf.startsWith(norm + "/"))
        this.selected.add(t.path);
    }
    this.emit("change");
  }

  /* ── Tags ─────────────────────────────────────────────────── */

  addKnownTag(name) {
    if (!this._knownTags.includes(name)) {
      this._knownTags.push(name);
      this._knownTags.sort();
    }
  }

  /* ── Track mutations ──────────────────────────────────────── */

  applyFolderRename(oldRel, newRel) {
    for (const t of this.allTracks) {
      if (t.folder === oldRel || t.folder.startsWith(oldRel + "/")) {
        t.folder = newRel + t.folder.slice(oldRel.length);
        t.path = (t.folder ? t.folder + "/" : "") + t.filename;
      }
    }
    if (this.folder === oldRel || this.folder.startsWith(oldRel + "/")) {
      this.folder = newRel + this.folder.slice(oldRel.length);
    }
    this._dirty();
  }

  /* ── Persistence ──────────────────────────────────────────── */

  saveLS() {
    localStorage.setItem(LS_KEY, JSON.stringify({
      folder:       this.folder,
      recursive:    this.recursive,
      axisX:        this.axisX,
      axisY:        this.axisY,
      filterRanges: this.filterRanges,
      // vp intentionally not persisted – always starts at default zoom-out
      hoverPreview: this.hoverPreview,
    }));
    localStorage.setItem(LS_KEY + "_tags", JSON.stringify(this._knownTags));
  }

  loadLS() {
    try {
      const d = JSON.parse(localStorage.getItem(LS_KEY));
      if (d) {
        this.folder       = d.folder       ?? "";
        this.recursive    = true;
        this.axisX        = d.axisX        ?? null;
        this.axisY        = d.axisY        ?? null;
        this.filterRanges = d.filterRanges ?? {};
        // vp not restored from LS
        this.hoverPreview = d.hoverPreview ?? false;
      }
    } catch { /* start fresh */ }
    try {
      this._knownTags = JSON.parse(localStorage.getItem(LS_KEY + "_tags")) || [];
    } catch { this._knownTags = []; }
  }
}
