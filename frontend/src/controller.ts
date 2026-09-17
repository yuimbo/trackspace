import {
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
} from "./components";
import { toast, showLoad, hideLoad } from "./lib/toast";
import { api, postJSON } from "./lib/api";
import { previewStartSeconds, sectionStartSeconds } from "./lib/structure";
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
import { FolderTreeCoordinator } from "./coordinators/folder-tree-coordinator";
import { CanvasInteractionController } from "./coordinators/canvas-interaction";
import { OptionsPanelBinder } from "./coordinators/options-panel";
import {
  SearchCoordinator,
  type SearchResponse,
} from "./coordinators/search-coordinator";
import {
  InspectCoordinator,
  type SimilarResponse,
} from "./coordinators/inspect-coordinator";

// ─── Constants ───────────────────────────────────────────────
const FLUSH_MS = 500;

// ─── Utilities ───────────────────────────────────────────────
function debounce<Args extends unknown[]>(
  fn: (...args: Args) => void,
  ms: number,
): (...args: Args) => void {
  let t: ReturnType<typeof setTimeout>;
  return (...a) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...a), ms);
  };
}

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

  private previewPath: string | null = null;
  /** Paths with an in-flight structure fetch, so a hover burst fetches once. */
  private structurePending = new Set<string>();
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

  /** Sidebar folder tree: HTMX refresh, roots, highlights, selected-folders overlay. */
  private folders: FolderTreeCoordinator;

  /** Canvas pointer gestures: selection, drag, transforms, pan/zoom, folder-drop. */
  private interaction: CanvasInteractionController;

  /** Right-column Options panel controls. */
  private options: OptionsPanelBinder;

  /** Semantic search: query fetch, result selection, result panel. */
  private search: SearchCoordinator;

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
  ) {
    this.model = model;
    this.canvas = canvasView;
    this.tagPanel = tagPanelView;
    this.featuresPanel = featuresPanelView;
    this.props = propsView;
    this.batch = batchView;
    this.status = statusView;
    this.searchView = searchView;

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
      // An imprint registers/removes a root, so the library and tree must
      // follow: the root set and the visible track set stay in step.
      onRootsChanged: () => this._reloadLibraryAndTree(),
    });

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
    });

    this.embedding = new EmbeddingCoordinator({
      model,
      status: statusView,
      canvas: canvasView,
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
      // Viewport moves (focus/homing) tween on the shared canvas frame.
      tweenViewport: (target) => this.embedding.tweenViewport(target),
    });

    this.search = new SearchCoordinator({
      model,
      view: searchView,
      fetchResults: (query, limit) =>
        api<SearchResponse>(
          `/api/search?q=${encodeURIComponent(query)}&limit=${limit}`,
          undefined,
          true,
        ),
      fitToPaths: (paths) => this.interaction.fitViewToPaths(paths),
    });

    this.inspect = new InspectCoordinator({
      model,
      view: {
        setInspect: (patch) => canvasView.setInspect(patch),
        unpinInspect: () => canvasView.unpinInspect(),
      },
      fetchSimilar: (path, mode, k) =>
        api<SimilarResponse>(
          `/api/tracks/similar?path=${encodeURIComponent(path)}&mode=${mode}&k=${k}`,
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
    canvasView.onInspectAction = (action, path) =>
      void this.inspect.action(action, path);

    this.folders = new FolderTreeCoordinator({
      model,
      canvas: canvasView,
      isPointerBusy: () => this.interaction.isPointerBusy(),
      renameFolder: (path, newName) => {
        void this.cmdMgr.run(new RenameFolderCommand(path, newName), model);
      },
      renderTagPanels: () => this._renderTagPanels(),
      onRootsChanged: () => this._reloadLibraryAndTree(),
    });

    this.options = new OptionsPanelBinder({
      model,
      scheduleDraw: () => this.canvas.scheduleDraw(),
      toggleViewMode: () => void this.embedding.toggleViewMode(),
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
    this._bindTagControls();
    this.analysis.bindPanel();
    this.head.bindPanel();
    this._bindKeyboard();
    this._bindResize();
    this._bindViewportChrome();
    this.options.sync();
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
      this._updateClearHidesButton();
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
    // Hovering a selected track (Properties) or a search hit previews it.
    const onHoverTrack = (path: string) => {
      if (!m.hoverPreview) return;
      const t = m.trackByPath(path);
      if (t) this._startPreview(t);
    };
    const onHoverEnd = () => this._stopPreview();
    this.props.onHoverTrack = onHoverTrack;
    this.props.onHoverEnd = onHoverEnd;
    this.searchView.onHoverTrack = onHoverTrack;
    this.searchView.onHoverEnd = onHoverEnd;

    this.searchView.onSearch = (query) => {
      void this.search.search(query);
    };
    this.searchView.onFocusResult = (path) => this.search.focusResult(path);

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

  private _updateClearHidesButton(): void {
    const btn = document.getElementById("viewport-clear-hides");
    if (!btn) return;
    btn.classList.toggle("hidden", !this.model.hasExclusions);
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
          resolve();
        }

        if (final) {
          clearInterval(this._libFlushTimer);
          this._libFlushTimer = 0;
          m.libraryLoadProgress = null;
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
          track: Track;
        };
        this._libPending.push(data.track);
        const tr = data.track;
        m.libraryLoadProgress = {
          done: data.done,
          total: data.total,
          lastPath: tr.path,
          lastTitle: tr.title?.trim() || undefined,
        };
        if ($progress) {
          $progress.textContent = `Loading library… ${data.done} / ${data.total}`;
        }
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
    // A search hit auditions the section that matched the query; otherwise the
    // preview seeks to the detected drop (structure lands asynchronously).
    const matchSection = this.model.searchSections.get(track.path);
    this.$audio.volume = previewGain(this.model.previewVolume);
    this.$audio.src = `/api/audio/${encodeURIComponent(track.path)}`;
    this.$audio.onloadedmetadata = () => {
      const s = this.model.trackStructure.get(track.path);
      this.model.previewSection = s?.peak_start != null ? "drop" : null;
      this.$audio.currentTime = previewStartSeconds(
        s,
        this.$audio.duration,
        matchSection,
      );
      this.canvas.scheduleDraw();
    };
    this.$audio.play().catch(() => {});
    if (!this.model.trackStructure.has(track.path)) {
      void this._fetchStructure(track.path);
    }
  }

  /**
   * Cycle the hover preview between the sections the track actually has
   * (drop → break → intro) and re-seek. No-op when nothing is auditioning.
   */
  private _cyclePreviewSection(): void {
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
    const idx = m.previewSection ? present.indexOf(m.previewSection) : -1;
    const next = present[(idx + 1) % present.length];
    const start = sectionStartSeconds(structure, next);
    if (start == null) return;
    m.previewSection = next;
    if (this.$audio.duration > 0) this.$audio.currentTime = start;
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
    this.structurePending.add(path);
    try {
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
    } catch {
      /* A missing row is the normal "not analysed yet" case; keep the fallback. */
    } finally {
      this.structurePending.delete(path);
    }
  }

  private _stopPreview(): void {
    if (!this.previewPath) return;
    this.previewPath = null;
    this.model.previewTrackPath = null;
    this.model.previewSection = null;
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

  private _bindKeyboard(): void {
    const m = this.model;
    const hk = this.hotkeyMgr;

    hk.on("focus-search", () => this.searchView.focusInput());

    hk.on("toggle-preview", () => {
      m.hoverPreview = !m.hoverPreview;
      if (!m.hoverPreview) this._stopPreview();
      this.options.sync();
      m.saveLS();
    });

    hk.on("fit-view", () => this.interaction.fitView());

    hk.on("focus-folder", () => {
      const target = this.interaction.effectiveHoverTarget();
      if (!target) return;
      m.selectOnlyHoverTarget(target);
      m.saveLS();
    });

    hk.on("exclude-selection", () => {
      const target = this.interaction.effectiveHoverTarget();
      if (target) {
        // A folder keeps its prefix-based exclusion so newly discovered files
        // under it are hidden too; the other kinds are path sets.
        if (target.kind === "folder") {
          if (m.folderSubtreeFullyExcluded(target.prefix)) {
            m.revealFolderSubtree(target.prefix);
          } else {
            m.hideFolderTree(target.prefix);
          }
        } else {
          const paths = m.hoverTargetPaths(target);
          if (paths.size === 0) return;
          if (m.pathsFullyHidden(paths)) m.revealPaths(paths);
          else m.hidePaths(paths);
        }
        m.saveLS();
        return;
      }
      if (m.selected.size) {
        m.hidePaths([...m.selected]);
        m.saveLS();
      }
    });

    hk.on("isolate-selection", () => this.interaction.solo());

    hk.on("show-all-tracks", () => {
      m.clearExclusions();
      m.saveLS();
    });

    hk.on("cycle-preview-section", () => this._cyclePreviewSection());

    hk.on("select-all", () => m.selectAll());

    hk.on("undo", () => void this.cmdMgr.undo(m));

    hk.on("toggle-view", () => void this.embedding.toggleViewMode());

    hk.on("toggle-clusters", () => this.analysis.toggleColorByCluster());

    hk.on("deselect", () => {
      if (this.canvas.unpinInspect()) return;
      if (this.interaction.cancelActiveGesture()) return;
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

  private _bindViewportChrome(): void {
    document.getElementById("viewport-clear-hides")?.addEventListener("click", () => {
      this.model.clearExclusions();
      this.model.saveLS();
    });
  }
}
