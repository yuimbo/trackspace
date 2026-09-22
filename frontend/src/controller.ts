import {
  type HoverTarget,
  type Model,
  type PreviewSection,
  type Track,
  type TrackStructure,
  isAudioFeatureAxisId,
  previewGain,
} from "./model";
import type {
  CanvasView,
  TagPanelView,
  PropertiesView,
  BatchView,
  StatusView,
  SearchView,
  ViewModifierView,
} from "./components";
import { toast, showLoad, hideLoad, setLoadBar, clearLoadBar } from "./lib/toast";
import { api, postJSON } from "./lib/api";
import { previewStartSeconds, sectionStartSeconds } from "./lib/structure";
import { cycle, clamp01 } from "./lib/util";
import { debounce, keyedInFlight } from "./lib/async";
import { bloomRangeLabel } from "./lib/bloom";
import {
  CommandManager,
  RenameTagCommand,
  RenameFolderCommand,
  MoveTracksCommand,
} from "./command";
import { HotkeyManager } from "./hotkeys";
import { AnalysisCoordinator } from "./coordinators/analysis-coordinator";
import { HeadCoordinator } from "./coordinators/head-coordinator";
import { EmbeddingCoordinator } from "./coordinators/embedding-coordinator";
import {
  FolderTreeCoordinator,
  type FolderCanvasAction,
} from "./coordinators/folder-tree-coordinator";
import { CanvasInteractionController } from "./coordinators/canvas-interaction";
import { OptionsPanelBinder } from "./coordinators/options-panel";
import { RightPanelBinder } from "./coordinators/right-panel";
import { ViewModifierCoordinator } from "./coordinators/view-modifier-coordinator";
import {
  SearchCoordinator,
  type SearchResponse,
} from "./coordinators/search-coordinator";
import {
  InspectCoordinator,
  PROXIMITY_ALL,
  type DescribeResponse,
  type SimilarResponse,
} from "./coordinators/inspect-coordinator";

// ─── Constants ───────────────────────────────────────────────
const FLUSH_MS = 500;
/** Range added per D + scroll step, at the gesture's clamped max (±0.4 → ±0.2). */
const BLOOM_RANGE_SCROLL_STEP = 0.5;

// ─── Internal types ──────────────────────────────────────────
interface PendingUpdate {
  path: string;
  tag: string;
  value: number | null;
}

// ═════════════════════════════════════════════════════════════
//  Controller
// ═════════════════════════════════════════════════════════════
export class Controller {
  private model: Model;
  private canvas: CanvasView;
  private tagPanel: TagPanelView;
  private featuresPanel: TagPanelView;
  private props: PropertiesView;
  private batch: BatchView;
  private status: StatusView;
  private searchView: SearchView;
  /** Corner stack of active view modifiers (search, proximity, scope, …). */
  private viewModifiers: ViewModifierView;

  private previewPath: string | null = null;
  /** Fraction to seek to once a freshly-started preview's metadata lands. */
  private pendingSeekFraction: number | null = null;
  /** Paths with an in-flight structure fetch, so a hover burst fetches once. */
  private structurePending = keyedInFlight<string, void>();
  /** Filter-tag → its chip id, so each active range filter has one chip. */
  private filterModifierIds = new Map<string, string>();
  private $audio: HTMLAudioElement;
  private pending = new Map<string, PendingUpdate>();
  private cmdMgr: CommandManager;
  readonly hotkeyMgr = new HotkeyManager();

  /** Queue stream, status polling, and the microgenre cluster panel. */
  private analysis: AnalysisCoordinator;

  /** Folder-supervised head panel: train/export/import/compare, imprint, undo. */
  private head: HeadCoordinator;

  /** Embedding-space view: layout cache, generation SSE, position animations. */
  private embedding: EmbeddingCoordinator;

  /** Sidebar folder tree: HTMX refresh, roots, highlights, selected-tracks overlay. */
  private folders: FolderTreeCoordinator;

  /** Canvas pointer gestures: selection, drag, transforms, pan/zoom, folder-drop. */
  private interaction: CanvasInteractionController;

  /** Right-column Options panel controls. */
  private options: OptionsPanelBinder;

  /** Right-panel tab strip (Analysis / Organise / Tracks / View). */
  private rightPanel: RightPanelBinder;

  /** Semantic search: query fetch, result selection, result panel. */
  private search: SearchCoordinator;

  /** The stack of active view modifiers, shown as corner chips. */
  private viewMods: ViewModifierCoordinator;

  /** Pinned inspect card: anti-track / nearest / solo-neighbourhood. */
  private inspect: InspectCoordinator;

  constructor(
    model: Model,
    canvasView: CanvasView,
    tagPanelView: TagPanelView,
    featuresPanelView: TagPanelView,
    propsView: PropertiesView,
    batchView: BatchView,
    statusView: StatusView,
    searchView: SearchView,
    viewModifierView: ViewModifierView,
  ) {
    this.model = model;
    this.canvas = canvasView;
    this.tagPanel = tagPanelView;
    this.featuresPanel = featuresPanelView;
    this.props = propsView;
    this.batch = batchView;
    this.status = statusView;
    this.searchView = searchView;
    this.viewModifiers = viewModifierView;

    this.viewMods = new ViewModifierCoordinator({
      model,
      view: viewModifierView,
    });

    this.cmdMgr = new CommandManager((effect) => {
      if (effect === "tags-dirty") {
        model.emit("tags-dirty");
      } else {
        model.emit("change");
      }
      if (effect === "change+htmx") {
        void this.folders.refresh();
      }
      model.saveLS();
    });

    this.$audio = document.getElementById("preview-audio") as HTMLAudioElement;

    this.head = new HeadCoordinator({
      model,
      // The head and its rows are a projection source, so the map must follow a
      // training run, an import, or a fan-out that moves tracks between folders.
      refreshProjection: () => this.embedding.refreshProjection(),
      refreshClusters: () => void this.analysis.fetchClusters(),
      refreshPredictions: () => this.analysis.refreshFolderPredictions(),
      // An imprint registers/removes a root, so the library and tree must
      // follow: the root set and the visible track set stay in step.
      onRootsChanged: () => this._reloadLibraryAndTree(),
    });

    this.rightPanel = new RightPanelBinder({ model });

    this.analysis = new AnalysisCoordinator({
      model,
      status: statusView,
      canvas: canvasView,
      scheduleDraw: () => this.canvas.scheduleDraw(),
      // Deferred: `embedding` is constructed just below. Applying weights
      // re-clusters on the server *and* refits the 2-D layout.
      refreshProjection: () => this.embedding.refreshProjection(),
      // Live slider drags use the cached source-weight Jacobian; the full
      // re-cluster + refit still runs on Apply.
      applyApproximateWeights: (weights) =>
        this.embedding.applyApproximateWeights(weights),
      fetchWeightJacobian: () => void this.embedding.fetchWeightJacobian(),
      // The queue owns Train's disabled state; the head panel renders it.
      head: this.head,
      // A broken-files row click selects the track and fits the view to it.
      focusTrack: (path) => {
        model.replaceSelection([path]);
        this.interaction.fitViewToPaths([path]);
      },
    });

    this.embedding = new EmbeddingCoordinator({
      model,
      status: statusView,
      canvas: canvasView,
      viewMods: this.viewMods,
    });

    this.interaction = new CanvasInteractionController({
      model,
      canvas: canvasView,
      status: statusView,
      props: propsView,
      batch: batchView,
      renderTagPanels: () => this._renderTagPanels(),
      queueTagWrite: (path, tag, value) =>
        this.pending.set(`${path}|${tag}`, { path, tag, value }),
      flushTagWrites: () => this._flushPending(),
      moveSelectionToFolder: (dest) => {
        const paths = [...model.selected];
        if (!paths.length) return;
        void this.cmdMgr.run(new MoveTracksCommand(paths, dest), model);
      },
      onHoverFolderChanged: () => this.folders.updateHighlights(),
      onHoverPreview: () => this._handlePreview(),
      stopPreview: () => this._stopPreview(),
      togglePreview: () => this._togglePreview(),
      adjustPreviewVolume: (factor) => this._adjustPreviewVolume(factor),
      commitPreviewVolume: () => this.model.saveLS(),
      // D owns the selection-bloom cursor (bare tap moves it, hold + scroll
      // adjusts the bloom range), so it is handled here, not by HotkeyManager.
      moveSelectionBloom: (dir) => this.analysis.moveSelectionBloom(dir),
      setSelectionBloom: (mode) => this.analysis.applySelectionBloom(mode),
      adjustBloomRange: (factor) => this._adjustBloomRange(factor),
      commitBloomRange: () => this.model.saveLS(),
      // Viewport moves (focus/homing) tween on the shared canvas frame.
      tweenViewport: (target) => this.embedding.tweenViewport(target),
      // The scope's lens is switched from its on-canvas tab bar.
      openScope: (scope, lensId, allowEmpty) =>
        void this.embedding.openScope(scope, lensId, allowEmpty),
      setScopeLens: (lensId) => void this.embedding.setScopeLens(lensId),
      resizeScope: (factor) => void this.embedding.resizeScope(factor),
      closeScope: () => this.embedding.closeScope(),
      // The persistent top tabbers switch the main lens / colour mode.
      switchLens: (lensId) => void this.embedding.switchToLens(lensId),
      setColorMode: (mode) => this.analysis.applyColorMode(mode, true),
    });

    this.search = new SearchCoordinator({
      model,
      view: searchView,
      viewMods: this.viewMods,
      fetchResults: (query, limit) =>
        api<SearchResponse>(
          `/api/search?q=${encodeURIComponent(query)}&limit=${limit}&field=1`,
          undefined,
          true,
        ),
      fitToPaths: (paths) => this.interaction.fitViewToPaths(paths),
      // The landmark field is purely visual: it lives on the canvas, not the model.
      // A live field *is* the search-active state, so clearing it also hides the
      // field even when the popup has already been committed closed.
      setField: (field) => {
        this.canvas.searchField = field;
        this.canvas.searchActive = field !== null;
        this.canvas.scheduleDraw();
      },
      // The matched paths drive the fade; purely visual, so on the canvas.
      setResults: (paths) => {
        this.canvas.searchResults = paths ? new Set(paths) : null;
        this.canvas.scheduleDraw();
      },
    });

    this.inspect = new InspectCoordinator({
      model,
      view: {
        setInspect: (patch) => canvasView.setInspect(patch),
        unpinInspect: () => canvasView.unpinInspect(),
      },
      viewMods: this.viewMods,
      fetchSimilar: (path, mode, k) =>
        api<SimilarResponse>(
          `/api/tracks/similar?path=${encodeURIComponent(path)}&mode=${mode}` +
            (k === PROXIMITY_ALL ? "&all=1" : `&k=${k}`),
          undefined,
          true,
        ),
      fetchDescribe: (path, limit) =>
        api<DescribeResponse>(
          `/api/tracks/describe?path=${encodeURIComponent(path)}&limit=${limit}`,
          undefined,
          true,
        ),
      fitToPaths: (paths) => this.interaction.fitViewToPaths(paths),
      preview: (path) => {
        if (!model.hoverPreview) return;
        const t = model.trackByPath(path);
        if (t) this._startPreview(t);
      },
      stopPreview: () => this._stopPreview(),
    });
    canvasView.onInspectAction = (action, path, value) => {
      // The card shows the head's projected folder, so load the predictions the
      // first time a card is pinned rather than waiting for that colour mode.
      if (action === "open" && !model.folderPredictionsLoaded)
        void this.analysis.fetchFolderPredictions();
      void this.inspect.action(action, path, value);
    };
    // A Describe-tab term is a semantic search: run it like a typed query.
    canvasView.onSearchTerm = (term) => this.searchView.setQuery(term);
    // Clicking or dragging the card's seekbar seeks the audition and resumes it.
    canvasView.onSeek = (path, fraction) => this._seekPreview(path, fraction);

    this.folders = new FolderTreeCoordinator({
      model,
      canvas: canvasView,
      isPointerBusy: () => this.interaction.isPointerBusy(),
      renameFolder: (path, newName) => {
        void this.cmdMgr.run(new RenameFolderCommand(path, newName), model);
      },
      renderTagPanels: () => this._renderTagPanels(),
      onRootsChanged: () => this._reloadLibraryAndTree(),
      refreshProjection: () => this.embedding.refreshProjection(),
      // "Imprint this root" on an external row prefills the head panel's target.
      onImprintRoot: (absPath) => this.head.prefillImprint(absPath),
      // The hover popout's buttons run the same canvas actions as the hotkeys.
      runCanvasAction: (action) => this._runCanvasAction(action),
    });

    this.options = new OptionsPanelBinder({
      model,
      stopPreview: () => this._stopPreview(),
      applyPreviewGain: (gain) => {
        this.$audio.volume = gain;
      },
    });

    this._wireModel();
    this._wireViewCallbacks();
    this.folders.bind();
    this.interaction.bind();
    this.options.bind();
    this.rightPanel.bind();
    this._bindTagControls();
    this.analysis.bindPanel();
    this.head.bindPanel();
    this._bindKeyboard();
    this._bindResize();
    this.options.sync();
    this.embedding.syncLensChrome();
    this._syncViewModifierChips();
  }

  private _renderTagPanels(): void {
    this.tagPanel.render();
    this.featuresPanel.render();
  }

  /* ── Model events ───────────────────────────────────────── */

  private _wireModel(): void {
    this.model.on("change", () => {
      this.canvas.scheduleDraw();
      this.status.update(this.model);
      this.props.render();
      this.batch.render();
      this._renderTagPanels();
      this.options.sync();
      this.folders.updateHighlights();
      this.folders.syncActiveLabels();
      this.folders.syncExternalVisibility();
      this._syncViewModifierChips();
      // ``m.tracks`` is rebuilt when hides/filters change; ``hoveredIdx`` must be
      // re-derived from the pointer after the next paint or hotkeys see a stale index.
      requestAnimationFrame(() => this.interaction.resyncHoverFromLastPointer());
    });
    this.model.on("tags-dirty", () => this.batch.renderDirty());
  }

  /* ── View callbacks ─────────────────────────────────────── */

  private _wireViewCallbacks(): void {
    const m = this.model;

    const onTagAxis = (which: string, tag: string) => {
      if (m.viewMode === "tags") {
        this.embedding.animateAxisChange(which as "axisX" | "axisY", tag);
      } else {
        m.toggleAxis(which as "axisX" | "axisY", tag);
        void this.embedding.switchToMode("tags");
      }
      this._renderTagPanels();
      m.saveLS();
    };
    this.tagPanel.onAxisChange = onTagAxis;
    this.featuresPanel.onAxisChange = onTagAxis;

    const onTagFilter = (tag: string, range: [number, number]) => {
      m.setFilterRange(tag, range);
      m.saveLS();
    };
    this.tagPanel.onFilterChange = onTagFilter;
    this.featuresPanel.onFilterChange = onTagFilter;

    this.tagPanel.onTagRename = (oldName, newName) => {
      void this.cmdMgr.run(new RenameTagCommand(oldName, newName), m);
    };

    this.props.onDeselect = (path) => {
      m.selected.delete(path);
      m.emit("change");
    };
    this.props.onRestore = (path) => {
      m.selected.add(path);
      m.emit("change");
    };
    this.props.onFocusRange = (keepPaths) => {
      for (const p of [...m.selected])
        if (!keepPaths.has(p)) m.selected.delete(p);
      m.emit("change");
    };
    // Hovering a selected track (Properties) or a search hit behaves like a
    // canvas hover: glow the dot, show the info card, and audition it.
    const onHoverTrack = (path: string) => {
      this.canvas.hoveredPath = path;
      this.canvas.hoverTarget = { kind: "track", path };
      this.canvas.scheduleDraw();
      if (!m.hoverPreview) return;
      const t = m.trackByPath(path);
      if (t) this._startPreview(t);
    };
    const onHoverEnd = () => {
      this.canvas.hoveredPath = null;
      if (this.canvas.hoverTarget?.kind === "track") {
        this.canvas.hoverTarget = null;
      }
      this.canvas.scheduleDraw();
      this._stopPreview();
    };
    this.props.onHoverTrack = onHoverTrack;
    this.props.onHoverEnd = onHoverEnd;
    this.searchView.onHoverTrack = onHoverTrack;
    this.searchView.onHoverEnd = onHoverEnd;

    this.searchView.onSearch = (query) => {
      void this.search.search(query);
    };
    this.searchView.onFocusResult = (path) => this.search.focusResult(path);
    // Enter or a result click commits: the popup closes, the search stays live
    // as a chip. The canvas field follows the search's active state.
    this.searchView.onCommit = () => this.searchView.close();
    this.searchView.onActiveChange = (active) => {
      this.canvas.searchActive = active;
      this.canvas.scheduleDraw();
    };

    // Active-view-modifier chips: × dismisses one (the coordinator runs its
    // clear callback), the inline controls drive the same model state the panels do.
    this.viewModifiers.onDismiss = (id) => this.viewMods.dismiss(id);
    this.viewModifiers.onProximityInput = (_id, percent) => {
      void this.inspect.setProximityFromChip(percent);
    };
    this.viewModifiers.onExternalRefit = (_id, checked) =>
      this._setExternalRefit(checked);
    this.viewModifiers.onExternalToggle = () => {
      m.setShowExternal(!m.showExternal);
      m.saveLS();
    };

    this.batch.onApply = (tag, updates) => {
      for (const [path, value] of updates) {
        const t = m.trackByPath(path);
        if (t) {
          t.tags[tag] = value;
          this.pending.set(`${path}|${tag}`, { path, tag, value });
        }
      }
      this.canvas.scheduleDraw();
      this._flushPending();
    };

    this.batch.onDragEnd = () => {
      m.emit("change");
    };

    this.batch.onClearTagFromSelection = (tag: string) => {
      void this._clearTagFromSelection(tag);
    };
  }

  /**
   * Mirror the visibility-filtering model state into the modifier chip stack:
   * exclusions, per-tag range filters, and shown external tracks.
   */
  private _syncViewModifierChips(): void {
    this._syncExclusionsModifier();
    this._syncFilterModifiers();
    this._syncExternalModifier();
  }

  private _syncExclusionsModifier(): void {
    // One chip per solo/exclude action, pushed where the action runs. When no
    // exclusions remain — Alt+S or a chip's × cleared them all — drop the rest.
    if (this.model.hasExclusions) return;
    for (const mod of this.viewMods.stack.filter(
      (x) => x.kind === "exclusions",
    )) {
      this.viewMods.remove(mod.id);
    }
  }

  private _syncFilterModifiers(): void {
    const m = this.model;
    const tags = Object.keys(m.filterRanges);
    for (const [tag, id] of [...this.filterModifierIds]) {
      if (tags.includes(tag)) continue;
      this.filterModifierIds.delete(tag);
      this.viewMods.remove(id);
    }
    for (const tag of tags) {
      const [lo, hi] = m.filterRanges[tag];
      const detail = `${Math.round(lo * 100)}–${Math.round(hi * 100)}%`;
      const id = this.filterModifierIds.get(tag);
      if (id) {
        this.viewMods.update(id, { detail });
      } else {
        this.filterModifierIds.set(
          tag,
          this.viewMods.push(
            { kind: "filters", label: m.axisDisplayName(tag), detail },
            () => {
              delete m.filterRanges[tag];
              m.emit("change");
            },
          ),
        );
      }
    }
  }

  private _syncExternalModifier(): void {
    const m = this.model;
    const active = this.viewMods.find("external");
    // A permanent two-state chip — Library + External (shown) or Library Only
    // (hidden) — with no ×: V toggles the visibility and R toggles the refit.
    const patch = {
      label: m.showExternal ? "Library + External" : "Library Only",
      detail: undefined,
      checked: m.showExternal ? m.externalInProjection : undefined,
      permanent: true,
    };
    if (active) {
      this.viewMods.update(active.id, patch);
    } else {
      this.viewMods.push({ kind: "external", ...patch }, () => {});
    }
  }

  /** Opt in/out of refitting the layout with external tracks (persisted). */
  private _setExternalRefit(included: boolean): void {
    if (this.model.externalInProjection === included) return;
    this.model.setExternalInProjection(included);
    this.model.saveLS();
    // The fit scope only matters while the externals are on the map.
    if (this.model.showExternal) void this.embedding.refreshProjection();
  }

  /* ── Data loading ───────────────────────────────────────── */

  // ── Background library loading state ──────────────────────
  private _libFlushTimer = 0;
  private _libPending: Track[] = [];

  private _loadLibrary(): Promise<void> {
    const $progress = document.getElementById("loading-progress");
    const m = this.model;
    m.setAllTracks([]);

    return new Promise((resolve) => {
      let firstFlushDone = false;
      let streamDone = false;

      const flush = (final = false) => {
        if (this._libPending.length) {
          const batch = this._libPending.splice(0);
          m.appendTracks(batch);
          this.canvas.scheduleDraw();
          this.status.update(m);
        }

        if (!firstFlushDone && (m.allTracks.length > 0 || final)) {
          firstFlushDone = true;
          if ($progress) $progress.textContent = "";
          clearLoadBar();
          resolve();
        }

        if (final) {
          clearInterval(this._libFlushTimer);
          this._libFlushTimer = 0;
          m.libraryLoadProgress = null;
          clearLoadBar();
          // Eligible paths / layout_revision depend on the track set; drop layout cache
          // so we never reuse coordinates after a full library rescan.
          this.embedding.invalidateLayoutCache();
          m.emit("change");
        }
      };

      // Flush buffered tracks to the canvas every 150 ms while stream is open.
      this._libFlushTimer = setInterval(() => {
        if (!streamDone) flush();
      }, 150) as unknown as number;

      const es = new EventSource("/api/library/stream?folder=&recursive=1");

      es.addEventListener("progress", ((e: MessageEvent) => {
        const data = JSON.parse(e.data) as {
          done: number;
          total: number;
          track?: Track;
          tracks?: Track[];
        };
        // The server chunks progress into batches of tracks; accept a single
        // `track` too so an older server still works.
        const batch = data.tracks ?? (data.track ? [data.track] : []);
        for (const t of batch) this._libPending.push(t);
        const tr = batch.length ? batch[batch.length - 1] : undefined;
        m.libraryLoadProgress = {
          done: data.done,
          total: data.total,
          lastPath: tr?.path,
          lastTitle: tr?.title?.trim() || undefined,
        };
        if ($progress) {
          $progress.textContent = `Loading library… ${data.done} / ${data.total}`;
        }
        setLoadBar(data.total > 0 ? data.done / data.total : null);
      }) as EventListener);

      es.addEventListener("done", (() => {
        es.close();
        streamDone = true;
        flush(true);
      }) as EventListener);

      es.onerror = () => {
        es.close();
        streamDone = true;
        if ($progress) $progress.textContent = "";
        flush(true);
      };
    });
  }

  async init(): Promise<void> {
    showLoad();
    setLoadBar(null);
    try {
      await Promise.all([this.folders.refresh(), this._loadLibrary()]);
    } finally {
      hideLoad();
    }
    this._renderTagPanels();
    this.props.render();
    this.batch.render();
    this.canvas.resize();
    this.canvas.scheduleDraw();
    this.status.update(this.model);

    // Queue stream + clusters are independent of view mode: analysis runs in
    // the background regardless of what the user is currently looking at.
    this.analysis.start();
    // The head panel checks for an imprint a closed browser left behind.
    this.head.start();

    if (this.model.viewMode === "embeddings") {
      void this.embedding.ensureAndProject();
    }
  }

  private async _reloadLibraryAndTree(): Promise<void> {
    showLoad();
    setLoadBar(null);
    try {
      await Promise.all([this.folders.refresh(), this._loadLibrary()]);
    } finally {
      hideLoad();
    }
    this._renderTagPanels();
    this.props.render();
    this.batch.render();
    this.canvas.scheduleDraw();
    this.status.update(this.model);
    if (this.model.viewMode === "embeddings") {
      void this.embedding.ensureAndProject();
    }
  }

  /* ── Debounced tag writes ───────────────────────────────── */

  private async _clearTagFromSelection(tag: string): Promise<void> {
    const m = this.model;
    if (isAudioFeatureAxisId(tag)) return;
    const updates: PendingUpdate[] = [];
    for (const path of m.selected) {
      const t = m.trackByPath(path);
      if (!t || t.tags[tag] === undefined) continue;
      delete t.tags[tag];
      updates.push({ path, tag, value: null });
      this.pending.delete(`${path}|${tag}`);
    }
    if (!updates.length) {
      m.emit("change");
      return;
    }
    this.canvas.scheduleDraw();
    try {
      await postJSON("/api/tracks/tags", { updates });
      toast(`Removed "${tag}" from ${updates.length} track(s)`, "ok");
    } catch {
      /* postJSON already toasted */
    }
    m.emit("change");
  }

  private _flushPending = debounce(async () => {
    const updates = [...this.pending.values()];
    this.pending.clear();
    if (!updates.length) return;
    try {
      const res = await postJSON<{
        count?: number;
        errors?: { path: string; error: string }[];
      }>("/api/tracks/tags", { updates });
      if (res?.errors?.length) {
        toast(
          `Saved ${res.count ?? updates.length} of ${updates.length} tag values; ${res.errors.length} failed`,
          "error",
        );
      } else {
        toast(`Saved ${updates.length} tag values`, "ok");
      }
    } catch {
      /* postJSON already toasted */
    }
  }, FLUSH_MS);

  /* ── Hover preview ──────────────────────────────────────── */

  /**
   * Bare **M** tap: toggle hover preview. Turning it off ends any audition;
   * turning it on auditions the track already under the pointer, so **M** works
   * without moving the mouse first.
   */
  private _togglePreview(): void {
    const m = this.model;
    m.hoverPreview = !m.hoverPreview;
    if (m.hoverPreview) this._handlePreview();
    else this._stopPreview();
    this.options.sync();
    m.saveLS();
  }

  /** **M + scroll**: scale the preview volume and mirror it on the slider. */
  private _adjustPreviewVolume(factor: number): void {
    const m = this.model;
    m.previewVolume = clamp01(m.previewVolume * factor);
    this.$audio.volume = previewGain(m.previewVolume);
    this.options.syncPreviewVolume();
  }

  /** **D + scroll**: widen/narrow the cursor variable's bloom range and redraw. */
  private _adjustBloomRange(factor: number): void {
    const m = this.model;
    // A variable with no range (cluster) has nothing to widen.
    if (bloomRangeLabel(m.selectionBloom, m.bloomRange()) === null) return;
    // Additive, not multiplicative: range 0 (exact class) is a real state a
    // scroll must be able to leave, unlike the volume's scaled taper.
    m.setBloomRange(
      m.bloomRange() + (factor - 1) * BLOOM_RANGE_SCROLL_STEP,
    );
    this.canvas.scheduleDraw();
  }

  private _handlePreview(): void {
    const m = this.model;
    const cv = this.canvas;
    if (!m.hoverPreview) {
      this._stopPreview();
      return;
    }
    if (cv.hoveredIdx < 0) {
      // The pinned card owns the audition while it is open: moving off a dot (or
      // across the gap to the card) must not end it. Closing the card does.
      if (!cv.inspect) this._stopPreview();
      return;
    }
    this._startPreview(m.tracks[cv.hoveredIdx]);
  }

  private _startPreview(track: Track): void {
    if (!track || this.previewPath === track.path) return;
    this.previewPath = track.path;
    this.model.previewTrackPath = track.path;
    this.model.previewSection = null;
    this.model.previewTime = 0;
    this.model.previewDuration = 0;
    // A search hit auditions the section that matched the query; otherwise the
    // preview seeks to the detected drop (structure lands asynchronously).
    const matchSection = this.model.searchSections.get(track.path);
    this.$audio.volume = previewGain(this.model.previewVolume);
    this.$audio.src = `/api/audio/${encodeURIComponent(track.path)}`;
    this.$audio.onloadedmetadata = () => {
      const s = this.model.trackStructure.get(track.path);
      this.model.previewDuration = Number.isFinite(this.$audio.duration)
        ? this.$audio.duration
        : 0;
      // A seekbar click seeks to its fraction; otherwise the preview seeks to the
      // detected drop (or the search hit's matching section).
      const pending = this.pendingSeekFraction;
      this.pendingSeekFraction = null;
      if (pending != null) {
        this.model.previewSection = null;
        this.model.previewTime = pending * this.$audio.duration;
        this.$audio.currentTime = this.model.previewTime;
      } else {
        this.model.previewSection = s?.peak_start != null ? "drop" : null;
        this.$audio.currentTime = previewStartSeconds(
          s,
          this.$audio.duration,
          matchSection,
        );
      }
      this.canvas.scheduleDraw();
    };
    // The seekbar fill follows playback, so every position tick redraws the card.
    this.$audio.ontimeupdate = () => {
      this.model.previewTime = this.$audio.currentTime;
      this.canvas.scheduleDraw();
    };
    this.$audio.play().catch(() => {});
    if (!this.model.trackStructure.has(track.path)) {
      void this._fetchStructure(track.path);
    }
  }

  /**
   * Seek the audition to *fraction* of the track and resume playback — a seekbar
   * click or drag. Seeking also turns the hover preview on, since the seekbar is
   * the hover-preview timeline. A different track is started first and seeks once
   * its metadata lands.
   */
  private _seekPreview(path: string, fraction: number): void {
    const m = this.model;
    if (!m.hoverPreview) {
      m.hoverPreview = true;
      this.options.sync();
      m.saveLS();
    }
    if (this.previewPath !== path) {
      const track = m.trackByPath(path);
      if (!track) return;
      this.pendingSeekFraction = fraction;
      this._startPreview(track);
      return;
    }
    const dur = this.$audio.duration;
    if (!Number.isFinite(dur) || dur <= 0) return;
    const t = Math.max(0, Math.min(dur - 0.05, fraction * dur));
    this.$audio.currentTime = t;
    m.previewTime = t;
    if (this.$audio.paused) this.$audio.play().catch(() => {});
    this.canvas.scheduleDraw();
  }

  /**
   * Cycle the hover preview between the sections the track actually has
   * (drop → break → intro) and re-seek. **W** forward, **Shift+W** backward.
   * No-op when nothing is auditioning.
   */
  private _cyclePreviewSection(dir: 1 | -1 = 1): void {
    const m = this.model;
    const path = this.previewPath;
    if (!path) return;
    const structure = m.trackStructure.get(path);
    const order: PreviewSection[] = ["drop", "break", "intro"];
    const present = order.filter((s) => sectionStartSeconds(structure, s) != null);
    if (!present.length) {
      toast("No cached structure for this track yet", "info");
      return;
    }
    // Forward from nothing starts at the first section, backward at the last.
    const fallback = dir > 0 ? present[0] : present[present.length - 1];
    const next = cycle(present, m.previewSection ?? fallback, dir);
    const start = sectionStartSeconds(structure, next);
    if (start == null) return;
    m.previewSection = next;
    if (this.$audio.duration > 0) {
      this.$audio.currentTime = start;
      m.previewTime = start;
    }
    this.canvas.scheduleDraw();
  }

  /**
   * Fetch one track's cached structure (intro/drop/break) and remember it.
   *
   * Silent and best-effort: the background ``structure`` stage may not have
   * reached the track yet, in which case the preview keeps its fixed position.
   * When the fetch lands while the track is still being auditioned, seek it to the
   * detected drop (unless the user already cycled to another section).
   */
  private async _fetchStructure(path: string): Promise<void> {
    if (this.model.trackStructure.has(path) || this.structurePending.has(path)) {
      return;
    }
    await this.structurePending.run(path, async () => {
      const res = await api<{
        tracks?: (TrackStructure & { path: string })[];
      }>(
        `/api/tracks/structure?paths=${encodeURIComponent(path)}`,
        undefined,
        true,
      );
      for (const entry of res?.tracks ?? []) {
        const { path: entryPath, ...structure } = entry;
        this.model.trackStructure.set(entryPath, structure);
      }
      const s = this.model.trackStructure.get(path);
      if (
        this.previewPath === path &&
        s?.peak_start != null &&
        this.$audio.duration > 0 &&
        !this.model.searchSections.has(path) &&
        (this.model.previewSection == null || this.model.previewSection === "drop")
      ) {
        this.model.previewSection = "drop";
        this.$audio.currentTime = previewStartSeconds(s, this.$audio.duration);
      }
      // Chips appear as soon as the structure lands, without moving the mouse.
      this.canvas.scheduleDraw();
    }).catch(() => {
      /* A missing row is the normal "not analysed yet" case; keep the fallback. */
    });
  }

  private _stopPreview(): void {
    if (!this.previewPath) return;
    this.previewPath = null;
    this.model.previewTrackPath = null;
    this.model.previewSection = null;
    this.model.previewTime = 0;
    this.model.previewDuration = 0;
    this.pendingSeekFraction = null;
    this.$audio.pause();
    this.$audio.removeAttribute("src");
    this.$audio.load();
    this.canvas.scheduleDraw();
  }

  /* ── Tag controls ───────────────────────────────────────── */

  private _bindTagControls(): void {
    const m = this.model;
    const $addT = document.getElementById("btn-add-tag")!;

    $addT.addEventListener("click", () => {
      let name = "new_tag";
      let n = 1;
      while (m.tags.includes(name) || isAudioFeatureAxisId(name)) {
        n++;
        name = `new_tag_${n}`;
      }
      m.addKnownTag(name);
      m.saveLS();
      this.tagPanel.scheduleRenameAfterRender(name);
      this._renderTagPanels();
    });

  }

  /* ── Keyboard shortcuts ─────────────────────────────────── */

  /**
   * Canvas actions shared by the global hotkeys and the folder-row popout.
   * Each acts on the *effective hover target* — the pinned card's navigation
   * element wins over the pointer, exactly as the hotkeys always did.
   */
  /** Push one visibility-filter chip (solo / exclude), with its own reverse. */
  private _pushExclusionChip(
    label: string,
    detail: string,
    clear: () => void,
  ): void {
    this.viewMods.push({ kind: "exclusions", label, detail }, clear);
  }

  /** A short readout of how many tracks a hover target covers. */
  private _targetDetail(target: HoverTarget): string {
    const n = this.model.hoverTargetPaths(target).size;
    return `${n} track${n === 1 ? "" : "s"}`;
  }

  private _runCanvasAction(action: FolderCanvasAction): void {
    const m = this.model;
    if (action === "fit-view") {
      this.interaction.fitView();
      return;
    }
    const target = this.interaction.effectiveHoverTarget();
    if (action === "isolate-selection") {
      // Solo is a visibility filter, so it pushes a chip that reveals again.
      const had = m.hasExclusions;
      this.interaction.solo();
      if (!had && m.hasExclusions) {
        const detail = target
          ? this._targetDetail(target)
          : `${m.selected.size} selected`;
        this._pushExclusionChip("Solo", detail, () => m.clearExclusions());
      }
      return;
    }
    if (!target) return;
    if (action === "focus-folder") {
      m.selectOnlyHoverTarget(target);
      m.saveLS();
      return;
    }
    // exclude-selection
    // A folder keeps its prefix-based exclusion so newly discovered files
    // under it are hidden too; the other kinds are path sets.
    if (target.kind === "folder") {
      if (m.folderSubtreeFullyExcluded(target.prefix)) {
        m.revealFolderSubtree(target.prefix);
      } else {
        m.hideFolderTree(target.prefix);
        this._pushExclusionChip("Excluded", this._targetDetail(target), () =>
          m.revealFolderSubtree(target.prefix),
        );
      }
    } else {
      const paths = m.hoverTargetPaths(target);
      if (paths.size === 0) return;
      if (m.pathsFullyHidden(paths)) {
        m.revealPaths(paths);
      } else {
        m.hidePaths(paths);
        this._pushExclusionChip("Excluded", this._targetDetail(target), () =>
          m.revealPaths(paths),
        );
      }
    }
    m.saveLS();
  }

  private _bindKeyboard(): void {
    const m = this.model;
    const hk = this.hotkeyMgr;

    hk.on("focus-search", () => this.searchView.focusInput());

    // M is owned by CanvasInteractionController (bare tap toggles, hold + scroll
    // adjusts the volume), so HotkeyManager lists it but does not dispatch it.

    hk.on("fit-view", () => this._runCanvasAction("fit-view"));

    hk.on("focus-folder", () => this._runCanvasAction("focus-folder"));

    hk.on("exclude-selection", () => this._runCanvasAction("exclude-selection"));

    hk.on("isolate-selection", () => this._runCanvasAction("isolate-selection"));

    hk.on("show-all-tracks", () => {
      m.clearExclusions();
      m.saveLS();
    });

    hk.on("cycle-preview-section", () => this._cyclePreviewSection(1));

    hk.on("cycle-preview-section-prev", () => this._cyclePreviewSection(-1));

    // D is owned by CanvasInteractionController (bare tap moves the selection-bloom
    // cursor, hold + scroll adjusts the bloom range), so HotkeyManager lists it
    // but does not dispatch it.

    hk.on("toggle-external", () => {
      // Hiding externals is a visibility filter, so the chip stack tracks it.
      m.setShowExternal(!m.showExternal);
      m.saveLS();
    });

    hk.on("toggle-external-refit", () =>
      this._setExternalRefit(!m.externalInProjection),
    );

    hk.on("select-all", () => m.selectAll());

    hk.on("undo", () => void this.cmdMgr.undo(m));

    hk.on("toggle-view", () => void this.embedding.cycleLens(1));

    hk.on("toggle-view-prev", () => void this.embedding.cycleLens(-1));

    hk.on("toggle-clusters", () => this.analysis.cycleColorMode());

    hk.on("toggle-clusters-prev", () => this.analysis.cycleColorMode(-1));

    hk.on("deselect", () => {
      // Transient surfaces first, then the newest view modifier, then selection.
      // (Esc inside the search box is handled by the box itself.)
      if (this.searchView.isOpen()) {
        this.searchView.close();
        return;
      }
      if (this.canvas.unpinInspect()) return;
      if (this.interaction.cancelActiveGesture()) return;
      if (this.viewMods.popNewest()) return;
      m.clearSelection();
    });
  }

  /* ── Window resize ──────────────────────────────────────── */

  private _bindResize(): void {
    window.addEventListener("resize", () => {
      this.canvas.resize();
      this.canvas.scheduleDraw();
    });
  }
}
