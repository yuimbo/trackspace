// ─── EvaluationCoordinator ───────────────────────────────────
//
// Owns the read-only layout-evaluation gallery (`?layout_eval=1`): fetching the
// manifest + recipe artifacts, applying precomputed positions to the shared
// canvas, restricting the view to the artifact's tracks, and driving the toolbar
// view's state. It is deliberately *bounded*: it does not know about lenses,
// weights, scopes or analysis — the controller wires it to the embedding
// coordinator's one `applyPrecomputedPositions` entry point and never lets it
// reach the compute/analysis surface.
//
// Read-only is enforced centrally (see `lib/eval-readonly.ts` + the HTTP gate in
// `lib/api.ts`), so the coordinator itself never issues a write and never has to
// second-guess a panel.

import { type Model } from "../model";
import {
  blindLetter,
  computeEvalUniverse,
  evalExitUrl,
  isUnsupportedEvalSchema,
  parseEvalArtifact,
  parseEvalManifest,
  type EvalArtifact,
  type EvalManifest,
  type EvalManifestRecipe,
} from "../lib/layout-eval";
import {
  buildEvalExport,
  loadAllRatings,
  ratedCount,
  saveRating,
  type EvalRatingRecord,
} from "../lib/eval-ratings";
import { setEvalReadOnly } from "../lib/eval-readonly";
import { latestSeq, type Seq } from "../lib/async";
import type { EvalToolbarState, EvalToolbarView } from "../components/eval-toolbar-view";

/** Max artifact maps kept in memory; the least-recently-used one is evicted past this. */
const EVAL_ARTIFACT_CACHE_MAX = 8;

export interface EvaluationDeps {
  model: Model;
  view: EvalToolbarView;
  /** Apply a precomputed position map on the shared canvas animation frame. */
  applyPositions(positions: Map<string, { x: number; y: number }>): void;
  /** Move the camera to the default full-map view. */
  resetCamera(): void;
  fetchManifest(): Promise<unknown>;
  fetchArtifact(id: string): Promise<unknown>;
}

function msg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

export class EvaluationCoordinator {
  private seq: Seq = latestSeq();
  private manifest: EvalManifest | null = null;
  private maps: { recipeId: string; letter: string; label: string }[] = [];
  private currentIndex = 0;
  private artifacts = new Map<string, EvalArtifact>();
  private currentArtifact: EvalArtifact | null = null;
  private loaded = false;
  private revealed = false;
  private busy = false;
  private error: string | null = null;
  private fp = "";
  private eligible = 0;
  private total = 0;
  private unavailable = 0;
  private noteTimer = 0;

  constructor(private deps: EvaluationDeps) {
    this.deps.view.onSelect = (i) => void this._selectMap(i);
    this.deps.view.onPrev = () => void this._step(-1);
    this.deps.view.onNext = () => void this._step(1);
    this.deps.view.onResetCamera = () => this.deps.resetCamera();
    this.deps.view.onToggleReveal = (c) => this._toggleReveal(c);
    this.deps.view.onRate = (cat, v) => this._rate(cat, v);
    this.deps.view.onNote = (t) => this._note(t);
    this.deps.view.onExport = () => this._export();
    this.deps.view.onExit = () => this._exit();
  }

  /** True once a usable manifest has loaded (eval is active). */
  get active(): boolean {
    return this.manifest !== null;
  }

  /**
   * Enter eval mode and load the gallery. *initialRecipeId* (from ``&layout=<id>``)
   * selects a starting map; defaults to the first.
   */
  async start(initialRecipeId?: string): Promise<void> {
    const m = this.deps.model;
    m.setEvalMode(true);
    setEvalReadOnly(true);
    document.body.classList.add("eval-mode");
    this.deps.resetCamera();
    this._render();
    try {
      const raw = await this.deps.fetchManifest();
      // The route returns {ok:true, manifest}; tolerate a bare manifest too.
      const manifestPayload =
        raw && typeof raw === "object" && "manifest" in (raw as object)
          ? (raw as { manifest: unknown }).manifest
          : raw;
      const manifest = parseEvalManifest(manifestPayload);
      if (!manifest) {
        this._error(
          isUnsupportedEvalSchema(manifestPayload)
            ? "Unsupported layout-evaluation manifest schema — this build cannot display this gallery."
            : "No completed layout-evaluation generation found on this server.",
        );
        return;
      }
      this.manifest = manifest;
      this.fp = manifest.datasetFingerprint;
      this.maps = manifest.recipes.map((r, i) => ({
        recipeId: r.id,
        letter: blindLetter(i),
        label: r.label,
      }));
      let start = 0;
      if (initialRecipeId) {
        const idx = this.maps.findIndex((x) => x.recipeId === initialRecipeId);
        if (idx >= 0) start = idx;
      }
      this.currentIndex = start;
      await this._selectMap(start);
    } catch (e) {
      this._error(`Failed to load the layout-evaluation manifest: ${msg(e)}`);
    }
  }

  /** Call once the library stream has finished loading. */
  onLibraryLoaded(): void {
    this.loaded = true;
    if (this.currentArtifact) this._applyUniverse(this.currentArtifact);
    this._render();
  }

  /* ── Map selection ───────────────────────────────────────── */

  private async _step(dir: 1 | -1): Promise<void> {
    const next = this.currentIndex + dir;
    if (next < 0 || next >= this.maps.length) return;
    await this._selectMap(next);
  }

  private async _selectMap(index: number): Promise<void> {
    const map = this.maps[index];
    if (!map) return;
    this.currentIndex = index;
    const token = this.seq.next();
    this.busy = true;
    this._render();

    let artifact = this.artifacts.get(map.recipeId) ?? null;
    if (artifact) {
      // LRU: re-insert to move the hit to the most-recently-used end.
      this.artifacts.delete(map.recipeId);
      this.artifacts.set(map.recipeId, artifact);
    } else {
      try {
        const raw = await this.deps.fetchArtifact(map.recipeId);
        // A newer selection won the race: drop this stale response.
        if (!this.seq.isCurrent(token)) return;
        artifact = parseEvalArtifact(raw);
        if (!artifact) {
          this.busy = false;
          this._error(`Map ${map.letter} artifact is missing or invalid.`);
          return;
        }
        this._cacheArtifact(map.recipeId, artifact);
      } catch (e) {
        if (!this.seq.isCurrent(token)) return;
        this.busy = false;
        this._error(`Failed to load Map ${map.letter}: ${msg(e)}`);
        return;
      }
    }
    if (!this.seq.isCurrent(token)) return;

    this.currentArtifact = artifact;
    this.busy = false;
    this.deps.applyPositions(artifact.positions);
    if (this.loaded) this._applyUniverse(artifact);
    this._render();
  }

  /** Insert into the artifact cache, evicting the least-recently-used entry past the bound. */
  private _cacheArtifact(recipeId: string, artifact: EvalArtifact): void {
    this.artifacts.delete(recipeId);
    this.artifacts.set(recipeId, artifact);
    if (this.artifacts.size > EVAL_ARTIFACT_CACHE_MAX) {
      const oldest = this.artifacts.keys().next().value;
      if (oldest !== undefined) this.artifacts.delete(oldest);
    }
  }

  private _applyUniverse(artifact: EvalArtifact): void {
    const loaded = new Set(this.deps.model.allTracks.map((t) => t.path));
    const { eligible, unavailable } = computeEvalUniverse(
      new Set(artifact.positions.keys()),
      loaded,
    );
    this.eligible = eligible.size;
    this.total = artifact.positions.size;
    this.unavailable = unavailable;
    this.deps.model.setEvalUniverse(eligible);
  }

  /* ── Reveal / ratings ────────────────────────────────────── */

  private _toggleReveal(checked: boolean): void {
    this.revealed = checked;
    if (checked && this.fp) {
      const recipeId = this.maps[this.currentIndex]?.recipeId;
      const revision = this.currentArtifact?.revision;
      const label = this.maps[this.currentIndex]?.label ?? "";
      if (revision) saveRating(this.fp, revision, { label, recipeId });
    }
    this._render();
  }

  /** The stored record for the current map, looked up in a pre-parsed store. */
  private _currentRecord(
    all: Record<string, Record<string, EvalRatingRecord>>,
  ): EvalRatingRecord | null {
    const revision = this.currentArtifact?.revision;
    return this.fp && revision ? (all[this.fp]?.[revision] ?? null) : null;
  }

  private _rate(
    category: "neighbours" | "organisation" | "crowding" | "overall",
    value: number | null,
  ): void {
    const revision = this.currentArtifact?.revision;
    const recipeId = this.maps[this.currentIndex]?.recipeId ?? "";
    if (!this.fp || !revision) return;
    saveRating(this.fp, revision, {
      [category]: value,
      recipeId,
      label: this.revealed ? (this.maps[this.currentIndex]?.label ?? "") : "",
    });
    this._render();
  }

  private _note(text: string): void {
    const revision = this.currentArtifact?.revision;
    const recipeId = this.maps[this.currentIndex]?.recipeId ?? "";
    if (!this.fp || !revision) return;
    // Capture the target so a map switch mid-debounce lands the note on the
    // right record.
    const fp = this.fp;
    const rev = revision;
    const label = this.revealed ? (this.maps[this.currentIndex]?.label ?? "") : "";
    clearTimeout(this.noteTimer);
    this.noteTimer = setTimeout(() => {
      saveRating(fp, rev, { note: text, recipeId, label });
      this._render();
    }, 400) as unknown as number;
  }

  private _export(): void {
    if (!this.fp) return;
    const records = Object.values(loadAllRatings()[this.fp] ?? {});
    const json = buildEvalExport(this.fp, records);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `layout-eval-${this.fp.slice(0, 8)}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  private _exit(): void {
    window.location.href = evalExitUrl(window.location.href);
  }

  /* ── Render ──────────────────────────────────────────────── */

  private _error(message: string): void {
    this.error = message;
    this.busy = false;
    this._render();
  }

  private _timingLabel(meta: EvalManifestRecipe): string {
    const parts: string[] = [];
    for (const k of ["pca", "tsne", "umap", "alignment"]) {
      const v = meta.timings?.[k];
      if (typeof v === "number") parts.push(`${k} ${v.toFixed(1)}s`);
    }
    return parts.length ? parts.join(" · ") : meta.label;
  }

  private _render(): void {
    if (!this.manifest) {
      this.deps.view.render(this._emptyState());
      return;
    }
    const cur = this.maps[this.currentIndex];
    const meta = this.manifest.recipes.find((r) => r.id === cur?.recipeId);
    // Parse the ratings store once per render; per-map lookups are plain reads.
    const ratings = loadAllRatings();
    const fpRatings = this.fp ? (ratings[this.fp] ?? {}) : {};
    const record = this._currentRecord(ratings);
    const state: EvalToolbarState = {
      busy: this.busy,
      currentIndex: this.currentIndex,
      count: this.maps.length,
      maps: this.maps.map((m, i) => {
        const art = this.artifacts.get(m.recipeId);
        const rev = art?.revision ?? "";
        const rec = rev ? (fpRatings[rev] ?? null) : null;
        return {
          index: i,
          letter: m.letter,
          label: this.revealed ? m.label : undefined,
          rated: ratedCount(rec),
          revealed: this.revealed,
        };
      }),
      revealed: this.revealed,
      currentTiming: this.revealed && meta ? this._timingLabel(meta) : "",
      eligible: this.eligible,
      total: this.total,
      unavailable: this.unavailable,
      ratings: record,
      note: record?.note ?? "",
      datasetLabel: this.fp ? `dataset ${this.fp.slice(0, 8)}` : "",
      error: this.error,
    };
    this.deps.view.render(state);
  }

  private _emptyState(): EvalToolbarState {
    return {
      busy: this.busy,
      currentIndex: 0,
      count: 0,
      maps: [],
      revealed: false,
      currentTiming: "",
      eligible: 0,
      total: 0,
      unavailable: 0,
      ratings: null,
      note: "",
      datasetLabel: "",
      error: this.error,
    };
  }
}
