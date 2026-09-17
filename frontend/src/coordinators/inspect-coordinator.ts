// ─── InspectCoordinator ─────────────────────────────────────
//
// Owns the pinned inspect card's navigation actions: the "anti-track" (furthest
// neighbour in the embedding space), nearest neighbours, and soloing the
// neighbourhood. The card itself is rendered by the canvas (`lib/tooltip.ts`); this
// coordinator turns a button click into one fetch plus model mutations, so the
// canvas view stays free of API calls — same seam as `SearchCoordinator`.

import { type Model } from "../model";
import {
  ANTI_MORE_K,
  type InspectState,
  type TooltipResult,
} from "../lib/tooltip";

/** Neighbours requested for the "nearest" ranking. */
export const NEAR_K = 5;
/** Neighbours kept when soloing the neighbourhood. */
export const SOLO_K = 10;
/** The anti-track ranking: one on demand, expandable to the most distant list. */
export const ANTI_K = 1;

/** Shape of ``GET /api/tracks/similar``. */
export interface SimilarResponse {
  ok: boolean;
  error?: string;
  results?: TooltipResult[];
}

/** The parts of `CanvasView` this coordinator drives (structural for tests). */
export interface InspectViewLike {
  setInspect(patch: Partial<InspectState>): void;
  unpinInspect(): void;
}

export interface InspectCoordinatorDeps {
  model: Model;
  view: InspectViewLike;
  /** Fetch a ranking (injectable so tests avoid `fetch`). */
  fetchSimilar(
    path: string,
    mode: "near" | "far",
    k: number,
  ): Promise<SimilarResponse>;
  /** Fit the canvas viewport to the given paths. */
  fitToPaths(paths: string[]): void;
  /** Start the hover preview for a path (no-op when hover preview is off). */
  preview(path: string): void;
  /** Stop the hover preview (the pointer left the previewing element). */
  stopPreview(): void;
}

export class InspectCoordinator {
  constructor(private deps: InspectCoordinatorDeps) {}

  /** Run one inspect-card action (`data-inspect-action` or a result focus). */
  async action(action: string, path: string): Promise<void> {
    if (action === "close") {
      this.deps.view.unpinInspect();
      return;
    }
    if (action === "focus") {
      this.focus(path);
      return;
    }
    if (action === "hover") {
      // Hovering a result row auditions it straight away, like a canvas hover.
      this.deps.preview(path);
      return;
    }
    if (action === "unhover") {
      this.deps.stopPreview();
      return;
    }
    if (action === "solo") {
      await this.solo(path);
      return;
    }
    if (action === "anti-more") {
      // Expand the anti-track to the most distant list, in place.
      await this.rank(path, "far", ANTI_MORE_K);
      return;
    }
    if (action === "anti" || action === "near") {
      await this.rank(path, action === "anti" ? "far" : "near");
    }
  }

  /** Select one result alone, fit the view to it, and audition it. */
  private focus(path: string): void {
    if (!this.deps.model.trackByPath(path)) return;
    this.deps.model.replaceSelection([path]);
    this.deps.fitToPaths([path]);
    this.deps.preview(path);
  }

  private async rank(
    path: string,
    mode: "near" | "far",
    k?: number,
  ): Promise<void> {
    const limit = k ?? (mode === "far" ? ANTI_K : NEAR_K);
    this.deps.view.setInspect({ busy: true, message: null, mode, results: [] });
    const res = await this.fetch(path, mode, limit);
    if (!res.ok) {
      this.deps.view.setInspect({
        busy: false,
        message: res.error ?? "Similarity lookup failed",
        results: [],
      });
      return;
    }
    this.deps.view.setInspect({
      busy: false,
      message: null,
      mode,
      results: res.results ?? [],
    });
  }

  /** Isolate to the neighbourhood and fit the view to it. */
  private async solo(path: string): Promise<void> {
    this.deps.view.setInspect({ busy: true, message: null });
    const res = await this.fetch(path, "near", SOLO_K);
    if (!res.ok || !res.results?.length) {
      this.deps.view.setInspect({
        busy: false,
        message: res.error ?? "No neighbours to solo",
      });
      return;
    }
    const results = res.results;
    this.deps.view.setInspect({ busy: false, message: null, mode: "near", results });
    // Keep the inspected track itself in view: soloing a neighbourhood must not
    // hide the centre you are inspecting.
    const paths = [path, ...results.map((r) => r.path)];
    this.deps.model.focusPaths(new Set(paths));
    this.deps.fitToPaths(paths);
  }

  /** Fetch, turning a thrown transport error into a failed response. */
  private async fetch(
    path: string,
    mode: "near" | "far",
    k: number,
  ): Promise<SimilarResponse> {
    try {
      return await this.deps.fetchSimilar(path, mode, k);
    } catch (err) {
      return {
        ok: false,
        error: err instanceof Error ? err.message : "Similarity lookup failed",
      };
    }
  }
}
