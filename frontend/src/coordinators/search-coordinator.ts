// ─── SearchCoordinator ───────────────────────────────────────
//
// Owns semantic search: run the query against `/api/search`, turn the ranked
// hits into the canvas selection, and feed the right-panel result list. Extracted
// from `Controller` — the fetch/select/rank flow is independent of the DOM-heavy
// views, so the deps are structural and the coordinator is unit-testable.
//
// Search is whole-library: the backend ranks every track with a cached CLAP
// embedding, and the map shows the whole library too.

import { type Model } from "../model";
import type { SearchResult, SearchMeta } from "../components/search-view";
import type { SearchFieldPoint } from "../components/canvas-view";
import type { ViewModifierSink } from "./view-modifier-coordinator";

/** Ranked result list size requested from the backend (1–200). */
export const SEARCH_LIMIT = 50;

/** Shape of ``GET /api/search``. */
export interface SearchResponse {
  ok: boolean;
  query?: string;
  results?: SearchResult[];
  searched?: number;
  total_tracks?: number;
  /** Landmark sample for the CLAP-lens field (``field=1`` only). */
  field?: SearchFieldPoint[];
  error?: string;
}

/** The parts of `SearchView` this coordinator drives (structural for tests). */
export interface SearchViewLike {
  setBusy(busy: boolean): void;
  clear(): void;
  showError(message: string): void;
  renderResults(results: SearchResult[], meta: SearchMeta): void;
}

export interface SearchCoordinatorDeps {
  model: Model;
  view: SearchViewLike;
  /** The active-view-modifier stack: a live search is one chip. */
  viewMods: ViewModifierSink;
  /** Fetch ranked matches for a query (injectable so tests avoid `fetch`). */
  fetchResults(query: string, limit: number): Promise<SearchResponse>;
  /** Fit the canvas viewport to the given paths (result-row focus). */
  fitToPaths(paths: string[]): void;
  /** Publish the landmark field to the canvas (``null`` clears it). */
  setField(field: SearchFieldPoint[] | null): void;
  /** Publish the matched paths to the canvas (``null`` clears the fade). */
  setResults(paths: string[] | null): void;
}

export class SearchCoordinator {
  private _lastQuery = "";
  /** Id of the active `search` modifier, or null when no search is live. */
  private _modifierId: string | null = null;

  constructor(private deps: SearchCoordinatorDeps) {}

  /** Clear the result list (the box clears itself) and drop the modifier. */
  clear(): void {
    if (this._modifierId) {
      const id = this._modifierId;
      this._modifierId = null;
      this.deps.viewMods.remove(id);
    }
    this._clearState();
  }

  /** Revert the canvas + panel state without touching the modifier stack. */
  private _clearState(): void {
    this._lastQuery = "";
    this.deps.model.searchSections = new Map();
    this.deps.setField(null);
    this.deps.setResults(null);
    this.deps.view.clear();
  }

  /**
   * Encode *rawQuery* server-side and make the ranked hits the selection.
   *
   * A later query always wins: a response that arrives after a newer query was
   * submitted is dropped rather than overwriting the newer results.
   */
  async search(rawQuery: string): Promise<void> {
    const query = rawQuery.trim();
    this._lastQuery = query;
    if (!query) {
      this.clear();
      return;
    }

    this.deps.view.setBusy(true);
    let body: SearchResponse;
    try {
      body = await this.deps.fetchResults(query, SEARCH_LIMIT);
    } catch (err) {
      if (this._lastQuery !== query) return;
      this.deps.view.setBusy(false);
      this.deps.setField(null);
      this.deps.view.showError(err instanceof Error ? err.message : String(err));
      // No live search: drop the chip so the stack matches the canvas.
      if (this._modifierId) {
        const id = this._modifierId;
        this._modifierId = null;
        this.deps.viewMods.remove(id);
      }
      return;
    }
    if (this._lastQuery !== query) return;
    this.deps.view.setBusy(false);
    this.deps.setField(body.field ?? null);

    const m = this.deps.model;
    // Drop hits for files added since the last library load: the model cannot
    // resolve them, so they must not enter the selection.
    const results = (body.results ?? []).filter((r) => m.trackByPath(r.path));
    m.replaceSelection(results.map((r) => r.path));
    // Fade the non-matches in the map while this result set is active.
    this.deps.setResults(results.map((r) => r.path));
    // Remember where each hit matched so the preview auditions that section.
    m.searchSections = new Map(
      results
        .filter(
          (r): r is SearchResult & { section: number } =>
            typeof r.section === "number" && Number.isFinite(r.section),
        )
        .map((r) => [r.path, r.section]),
    );
    // A live search is an active view modifier: one chip, refreshed in place so a
    // keystroke burst does not stack chips.
    if (this._modifierId) {
      this.deps.viewMods.update(this._modifierId, { label: query });
    } else {
      this._modifierId = this.deps.viewMods.push(
        { kind: "search", label: query },
        () => {
          this._modifierId = null;
          this._clearState();
        },
      );
    }
    this.deps.view.renderResults(results, {
      searched: body.searched ?? 0,
      totalTracks: body.total_tracks ?? results.length,
    });
  }

  /** Focus one hit: select it alone and fit the viewport to it. */
  focusResult(path: string): void {
    this.deps.model.replaceSelection([path]);
    this.deps.fitToPaths([path]);
  }
}
