import type { Model, Track } from "../model";


export class PropertiesView {
  model: Model;
  $el: HTMLElement;
  private _pendingRestore = new Map<
    string,
    { timerId: number; track: Track | undefined }
  >();
  private _rangeAnchor: string | null = null;
  private _sortedPaths: string[] = [];
  private _listSel = new Set<string>();
  private _selHash = "";

  onDeselect: ((path: string) => void) | null = null;
  onRestore: ((path: string) => void) | null = null;
  onFocusRange: ((paths: Set<string>) => void) | null = null;
  onHoverTrack: ((path: string) => void) | null = null;
  onHoverEnd: (() => void) | null = null;

  constructor(model: Model, container: HTMLElement) {
    this.model = model;
    this.$el = container;
  }

  /* ── Sort helpers ───────────────────────────────────────── */

  private _sortKey(t: Track | undefined): string {
    if (!t) return "\x7f";
    const a = (t.artist || "").toLowerCase().trim();
    const ti = (t.title || t.filename || "").toLowerCase().trim();
    return a ? `${a}\x00${ti}` : `\x7f${ti}`;
  }

  private _sortedSelected(): string[] {
    const m = this.model;
    return [...m.selected]
      .map((p) => m.trackByPath(p))
      .filter((t): t is Track => Boolean(t))
      .sort((a, b) => (this._sortKey(a) < this._sortKey(b) ? -1 : 1))
      .map((t) => t.path);
  }

  /* ── Render ─────────────────────────────────────────────── */

  render(): void {
    this.$el.innerHTML = "";
    const m = this.model;
    const sorted = this._sortedSelected();
    this._sortedPaths = sorted;

    const hash = [...m.selected].sort().join("|");
    if (hash !== this._selHash) {
      this._selHash = hash;
      this._listSel.clear();
      this._rangeAnchor = null;
    }

    const ghosts = [...this._pendingRestore.entries()].filter(
      ([p]) => !m.selected.has(p),
    );

    if (sorted.length === 0 && ghosts.length === 0) {
      const empty = document.createElement("div");
      empty.className = "props-empty";
      empty.textContent = "No selection";
      this.$el.appendChild(empty);
      return;
    }

    // ── Active selection ────────────────────────────────────
    for (const path of sorted) {
      const track = m.trackByPath(path);
      const isListSel = this._listSel.has(path);
      const row = document.createElement("div");
      row.className =
        "props-track-row" + (isListSel ? " list-sel" : "");

      const info = document.createElement("div");
      info.className = "props-track-info";
      info.title = path;

      const primary = document.createElement("span");
      primary.className = "props-track-primary";
      primary.textContent = (track?.title || track?.filename) ?? path;
      info.appendChild(primary);

      if (track?.artist) {
        const secondary = document.createElement("span");
        secondary.className = "props-track-secondary";
        secondary.textContent = track.artist;
        info.appendChild(secondary);
      }

      info.addEventListener("mouseenter", () =>
        this.onHoverTrack?.(path),
      );
      info.addEventListener("mouseleave", () => this.onHoverEnd?.());

      info.addEventListener("click", (e) => {
        if (e.shiftKey && this._rangeAnchor && this._sortedPaths.length) {
          const ai = this._sortedPaths.indexOf(this._rangeAnchor);
          const bi = this._sortedPaths.indexOf(path);
          if (ai >= 0 && bi >= 0) {
            const lo = Math.min(ai, bi),
              hi = Math.max(ai, bi);
            for (const p of this._sortedPaths.slice(lo, hi + 1))
              this._listSel.add(p);
          }
        } else if (e.ctrlKey || e.metaKey) {
          if (this._listSel.has(path)) this._listSel.delete(path);
          else {
            this._listSel.add(path);
            this._rangeAnchor = path;
          }
        } else {
          this._listSel.clear();
          this._listSel.add(path);
          this._rangeAnchor = path;
        }
        this.render();
        e.stopPropagation();
      });

      row.appendChild(info);

      const focusBtn = document.createElement("button");
      focusBtn.className = "props-focus-btn";
      focusBtn.textContent = "⊙";
      focusBtn.title =
        "Focus select — keep only list-selected (or this track)";
      focusBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const targets =
          this._listSel.size > 0
            ? new Set(this._listSel)
            : new Set([path]);
        this._rangeAnchor = path;
        this.onFocusRange?.(targets);
      });
      row.appendChild(focusBtn);

      const delBtn = document.createElement("button");
      delBtn.className = "props-deselect-btn";
      delBtn.textContent = "×";
      delBtn.title = "Deselect (click ↩ ghost to restore)";
      delBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const t = m.trackByPath(path);
        const timerId = setTimeout(() => {
          this._pendingRestore.delete(path);
          this.render();
        }, 3000) as unknown as number;
        this._pendingRestore.set(path, { timerId, track: t });
        if (this._rangeAnchor === path) this._rangeAnchor = null;
        this._listSel.delete(path);
        this.onDeselect?.(path);
      });
      row.appendChild(delBtn);

      this.$el.appendChild(row);
    }

    // ── Ghost entries ────────────────────────────────────────
    if (ghosts.length > 0) {
      const sep = document.createElement("div");
      sep.className = "props-sep";
      this.$el.appendChild(sep);

      for (const [path, { track }] of ghosts) {
        const row = document.createElement("div");
        row.className = "props-ghost-row";
        row.title = "Click to restore to selection";

        const icon = document.createElement("span");
        icon.className = "props-ghost-icon";
        icon.textContent = "↩";
        row.appendChild(icon);

        const info = document.createElement("div");
        info.className = "props-ghost-info";

        const primary = document.createElement("span");
        primary.className = "props-track-primary";
        primary.textContent = (track?.title || track?.filename) ?? path;
        info.appendChild(primary);

        if (track?.artist) {
          const secondary = document.createElement("span");
          secondary.className = "props-track-secondary";
          secondary.textContent = track.artist;
          info.appendChild(secondary);
        }

        row.appendChild(info);
        row.addEventListener("click", () => {
          const entry = this._pendingRestore.get(path);
          if (entry) {
            clearTimeout(entry.timerId);
            this._pendingRestore.delete(path);
          }
          this.onRestore?.(path);
        });
        this.$el.appendChild(row);
      }
    }
  }
}
