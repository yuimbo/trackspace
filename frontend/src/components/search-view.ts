// ─── SearchView ─────────────────────────────────────────────
//
// The right-panel semantic-search surface: a query box plus a ranked result
// list. The query is encoded server-side with CLAP's text tower and matched
// against the cached audio embeddings.
//
// Views never mutate the model directly — the controller implements
// `onSearch` / `onFocusResult` / `onHoverTrack` / `onHoverEnd`.

import type { Model } from "../model";

/** Live-search delay: a burst of keystrokes collapses into one query. */
export const SEARCH_DEBOUNCE_MS = 300;

export interface SearchResult {
  path: string;
  score: number;
  /**
   * Best-matching CLAP segment centre as a fraction of the track duration, or
   * ``null`` for a pooled-only hit. The hover preview seeks here.
   */
  section?: number | null;
}

export interface SearchMeta {
  searched: number;
  totalTracks: number;
}

export class SearchView {
  model: Model;
  $el: HTMLElement;
  private $input: HTMLInputElement;
  private $results: HTMLElement;

  onSearch: ((query: string) => void) | null = null;
  onFocusResult: ((path: string) => void) | null = null;
  onHoverTrack: ((path: string) => void) | null = null;
  onHoverEnd: (() => void) | null = null;
  /** The box gained or lost focus — the canvas shows the search field only while active. */
  onActiveChange: ((active: boolean) => void) | null = null;

  /** Pending live-search timer, so a keystroke burst collapses into one query. */
  private _debounceTimer: ReturnType<typeof setTimeout> | undefined;

  constructor(model: Model, section: HTMLElement) {
    this.model = model;
    this.$el = section;
    this.$input = section.querySelector("#search-input") as HTMLInputElement;
    this.$results = section.querySelector("#search-results") as HTMLElement;

    this.$input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        clearTimeout(this._debounceTimer);
        this.onSearch?.(this.$input.value);
      } else if (e.key === "Escape") {
        e.preventDefault();
        clearTimeout(this._debounceTimer);
        this.$input.value = "";
        this.clear();
        // Esc leaves the input: the canvas search field hides with it.
        this.$input.blur();
      }
    });
    // Search while typing: the box no longer waits for Enter.
    this.$input.addEventListener("input", () => {
      clearTimeout(this._debounceTimer);
      const value = this.$input.value;
      if (!value.trim()) {
        this.onSearch?.("");
        return;
      }
      this._debounceTimer = setTimeout(
        () => this.onSearch?.(value),
        SEARCH_DEBOUNCE_MS,
      );
    });
    this.$input.addEventListener("focus", () => this._emitActive());
    this.$input.addEventListener("blur", () => this._emitActive());
  }

  /** Focus the query box (``/`` or ``⌘/Ctrl+F`` hotkey). */
  focusInput(): void {
    this.$input.focus();
    this.$input.select();
  }

  setBusy(busy: boolean): void {
    // The input stays editable: live search means the user keeps typing while a
    // query is in flight, and disabling/blurring it would drop the search-active
    // state (and hide the canvas field) mid-request. The coordinator drops a
    // response superseded by a newer query.
    this.$el.classList.toggle("searching", busy);
  }

  private _emitActive(): void {
    this.onActiveChange?.(document.activeElement === this.$input);
  }

  clear(): void {
    this.$results.innerHTML = "";
  }

  showError(message: string): void {
    this.$results.innerHTML = "";
    const err = document.createElement("div");
    err.className = "search-error";
    err.textContent = message;
    this.$results.appendChild(err);
  }

  renderResults(results: SearchResult[], meta: SearchMeta): void {
    this.$results.innerHTML = "";

    if (!results.length) {
      const empty = document.createElement("div");
      empty.className = "search-empty";
      empty.textContent =
        meta.searched === 0
          ? "No CLAP embeddings yet — analyse the library first."
          : "No matches.";
      this.$results.appendChild(empty);
      return;
    }

    const summary = document.createElement("div");
    summary.className = "search-summary";
    summary.textContent =
      `${results.length} of ${meta.searched} analysed` +
      (meta.totalTracks > meta.searched ? ` · ${meta.totalTracks} tracks` : "");
    this.$results.appendChild(summary);

    const top = Math.max(...results.map((r) => r.score), 1e-6);

    for (const r of results) {
      const t = this.model.trackByPath(r.path);
      const row = document.createElement("div");
      row.className = "search-result-row";
      row.title = `${r.path}\nscore ${r.score.toFixed(3)}`;

      const bar = document.createElement("div");
      bar.className = "search-score-bar";
      const fill = document.createElement("div");
      fill.className = "search-score-fill";
      fill.style.width = `${Math.max(2, Math.round((r.score / top) * 100))}%`;
      bar.appendChild(fill);
      row.appendChild(bar);

      const label = document.createElement("div");
      label.className = "search-result-label";
      const primary = document.createElement("span");
      primary.className = "search-result-primary";
      primary.textContent = (t?.title || t?.filename) ?? r.path;
      label.appendChild(primary);
      if (t?.artist) {
        const secondary = document.createElement("span");
        secondary.className = "search-result-secondary";
        secondary.textContent = t.artist;
        label.appendChild(secondary);
      }
      row.appendChild(label);

      row.addEventListener("mouseenter", () => this.onHoverTrack?.(r.path));
      row.addEventListener("mouseleave", () => this.onHoverEnd?.());
      row.addEventListener("click", () => this.onFocusResult?.(r.path));
      this.$results.appendChild(row);
    }
  }
}
