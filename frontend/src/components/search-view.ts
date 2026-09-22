// ─── SearchView ─────────────────────────────────────────────
//
// The right-panel semantic-search surface: a query box plus a ranked result
// list. The query is encoded server-side with CLAP's text tower and matched
// against the cached audio embeddings.
//
// Views never mutate the model directly — the controller implements
// `onSearch` / `onFocusResult` / `onHoverTrack` / `onHoverEnd`.

import type { Model } from "../model";
import { debounce } from "../lib/async";

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
  /** The search became active (popup opened) or was cleared. */
  onActiveChange: ((active: boolean) => void) | null = null;
  /** The popup should close (Enter, a result click, or a backdrop click). */
  onCommit: (() => void) | null = null;

  /** Pending live-search timer, so a keystroke burst collapses into one query. */
  private _searchDebounced = debounce(
    () => this.onSearch?.(this.$input.value),
    SEARCH_DEBOUNCE_MS,
  );

  constructor(model: Model, section: HTMLElement) {
    this.model = model;
    this.$el = section;
    this.$input = section.querySelector("#search-input") as HTMLInputElement;
    this.$results = section.querySelector("#search-results") as HTMLElement;

    this.$input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        this._searchDebounced.cancel();
        this.onSearch?.(this.$input.value);
        // Enter commits: the popup closes, the search stays live as a chip.
        this.onCommit?.();
      } else if (e.key === "Escape") {
        e.preventDefault();
        this._dismiss();
      }
    });
    // Search while typing: the box no longer waits for Enter.
    this.$input.addEventListener("input", () => {
      this._searchDebounced.cancel();
      if (!this.$input.value.trim()) {
        this.onSearch?.("");
        return;
      }
      this._searchDebounced();
    });
    // A click on the dimmed backdrop closes the popup without clearing.
    this.$el.addEventListener("mousedown", (e) => {
      if (e.target === this.$el) this.onCommit?.();
    });
  }

  /** Open the centered popup and focus the query box (``/`` or ``⌘/Ctrl+F``). */
  open(): void {
    this.$el.classList.remove("hidden");
    this.$input.focus();
    this.$input.select();
    this.onActiveChange?.(true);
  }

  /** Close the popup, leaving the live search active as a chip. */
  close(): void {
    this.$el.classList.add("hidden");
    this.$input.blur();
  }

  /** True while the popup is shown (Esc closes it before it pops a modifier). */
  isOpen(): boolean {
    return !this.$el.classList.contains("hidden");
  }

  /** Focus the query box (``/`` or ``⌘/Ctrl+F`` hotkey). */
  focusInput(): void {
    this.open();
  }

  /** Esc in the box: clear the search and close the popup. */
  private _dismiss(): void {
    this._searchDebounced.cancel();
    this.$input.value = "";
    this.onSearch?.("");
    this.onActiveChange?.(false);
    this.close();
  }

  /**
   * Run a query that did not come from typing (a Describe-tab term click).
   *
   * The popup opens and the box is filled, so the applied query is visible and
   * the canvas shows the search field.
   */
  setQuery(query: string): void {
    this._searchDebounced.cancel();
    this.$input.value = query;
    this.open();
    this.onSearch?.(query);
  }

  setBusy(busy: boolean): void {
    // The input stays editable: live search means the user keeps typing while a
    // query is in flight, and disabling/blurring it would drop the search-active
    // state (and hide the canvas field) mid-request. The coordinator drops a
    // response superseded by a newer query.
    this.$el.classList.toggle("searching", busy);
    // A visible "Searching…" row, so a click on a Describe term shows it is
    // working even before the ranked list arrives.
    const note = this.$results.querySelector(".search-busy");
    if (busy && !note) {
      const el = document.createElement("div");
      el.className = "search-busy";
      el.textContent = "Searching…";
      this.$results.prepend(el);
    } else if (!busy && note) {
      note.remove();
    }
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
      row.addEventListener("click", () => {
        this.onFocusResult?.(r.path);
        // A result click commits: close the popup, keep the search as a chip.
        this.onCommit?.();
      });
      this.$results.appendChild(row);
    }
  }
}
