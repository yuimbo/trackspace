import { type Model, type Track, isAudioFeatureAxisId } from "./model";
import type {
  CanvasView,
  TagPanelView,
  PropertiesView,
  BatchView,
  StatusView,
} from "./components";
import { toast, showLoad, hideLoad } from "./lib/toast";
import { postJSON } from "./lib/api";
import {
  CommandManager,
  RenameTagCommand,
  RenameFolderCommand,
  MoveTracksCommand,
} from "./command";
import { HotkeyManager } from "./hotkeys";
import { AnalysisCoordinator } from "./coordinators/analysis-coordinator";
import { EmbeddingCoordinator } from "./coordinators/embedding-coordinator";
import { FolderTreeCoordinator } from "./coordinators/folder-tree-coordinator";
import { CanvasInteractionController } from "./coordinators/canvas-interaction";
import { OptionsPanelBinder } from "./coordinators/options-panel";

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

  private previewPath: string | null = null;
  private $audio: HTMLAudioElement;
  private pending = new Map<string, PendingUpdate>();
  private cmdMgr: CommandManager;
  readonly hotkeyMgr = new HotkeyManager();

  /** Queue stream, status polling, and the microgenre cluster panel. */
  private analysis: AnalysisCoordinator;

  /** Embedding-space view: layout cache, generation SSE, position animations. */
  private embedding: EmbeddingCoordinator;

  /** Sidebar folder tree: HTMX refresh, roots, highlights, selected-folders overlay. */
  private folders: FolderTreeCoordinator;

  /** Canvas pointer gestures: selection, drag, transforms, pan/zoom, folder-drop. */
  private interaction: CanvasInteractionController;

  /** Right-column Options panel controls. */
  private options: OptionsPanelBinder;

  constructor(
    model: Model,
    canvasView: CanvasView,
    tagPanelView: TagPanelView,
    featuresPanelView: TagPanelView,
    propsView: PropertiesView,
    batchView: BatchView,
    statusView: StatusView,
  ) {
    this.model = model;
    this.canvas = canvasView;
    this.tagPanel = tagPanelView;
    this.featuresPanel = featuresPanelView;
    this.props = propsView;
    this.batch = batchView;
    this.status = statusView;

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

    this.analysis = new AnalysisCoordinator({
      model,
      status: statusView,
      scheduleDraw: () => this.canvas.scheduleDraw(),
      // Deferred: `embedding` is constructed just below. Applying weights
      // re-clusters on the server *and* refits the 2-D layout.
      refreshProjection: () => this.embedding.refreshProjection(),
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
    });

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
      applyPreviewVolume: (volume) => {
        this.$audio.volume = volume;
      },
    });

    this._wireModel();
    this._wireViewCallbacks();
    this.folders.bind();
    this.interaction.bind();
    this.options.bind();
    this._bindTagControls();
    this.analysis.bindPanel();
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
    this.props.onHoverTrack = (path) => {
      if (!m.hoverPreview) return;
      const t = m.trackByPath(path);
      if (t) this._startPreview(t);
    };
    this.props.onHoverEnd = () => {
      this._stopPreview();
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
    if (!m.hoverPreview || cv.hoveredIdx < 0) {
      this._stopPreview();
      return;
    }
    this._startPreview(m.tracks[cv.hoveredIdx]);
  }

  private _startPreview(track: Track): void {
    if (!track || this.previewPath === track.path) return;
    this.previewPath = track.path;
    this.$audio.volume = this.model.previewVolume;
    this.$audio.src = `/api/audio/${encodeURIComponent(track.path)}`;
    this.$audio.onloadedmetadata = () => {
      this.$audio.currentTime = this.$audio.duration * 0.4;
    };
    this.$audio.play().catch(() => {});
  }

  private _stopPreview(): void {
    if (!this.previewPath) return;
    this.previewPath = null;
    this.$audio.pause();
    this.$audio.removeAttribute("src");
    this.$audio.load();
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

  /** True if isolating would still exclude at least one currently visible track. */
  private _wouldIsolateExcludeMore(
    m: Model,
    kind: { type: "selection" } | { type: "folder"; prefix: string },
  ): boolean {
    for (const t of m.allTracks) {
      if (!m.trackInViewUnion(t) || m.isTrackHidden(t)) continue;
      if (kind.type === "selection") {
        if (!m.selected.has(t.path)) return true;
      } else {
        const tf = t.folder ?? "";
        const p = kind.prefix;
        const under =
          p === "" || tf === p || tf.startsWith(p + "/");
        if (!under) return true;
      }
    }
    return false;
  }

  private _bindKeyboard(): void {
    const m = this.model;
    const hk = this.hotkeyMgr;

    hk.on("toggle-preview", () => {
      m.hoverPreview = !m.hoverPreview;
      if (!m.hoverPreview) this._stopPreview();
      this.options.sync();
      m.saveLS();
    });

    hk.on("fit-view", () => this.interaction.fitView());

    hk.on("focus-folder", () => {
      const hp = this.interaction.effectiveHoveredFolderPrefix();
      if (hp === null) return;
      m.selectOnlyInFolder(hp);
      m.saveLS();
    });

    hk.on("exclude-selection", () => {
      const hp = this.interaction.effectiveHoveredFolderPrefix();
      if (hp !== null) {
        if (m.folderSubtreeFullyExcluded(hp)) {
          m.revealFolderSubtree(hp);
        } else {
          m.hideFolderTree(hp);
        }
        m.saveLS();
        return;
      }
      if (m.selected.size) {
        m.hidePaths([...m.selected]);
        m.saveLS();
      }
    });

    hk.on("isolate-selection", () => {
      const hp = this.interaction.effectiveHoveredFolderPrefix();
      if (hp !== null) {
        if (m.folderSubtreeFullyExcluded(hp)) {
          m.revealThenFocusFolderSubtree(hp);
        } else if (this._wouldIsolateExcludeMore(m, { type: "folder", prefix: hp })) {
          m.focusFolderSubtree(hp);
        } else {
          m.clearExclusions();
        }
        m.saveLS();
        return;
      }
      if (m.selected.size) {
        if (this._wouldIsolateExcludeMore(m, { type: "selection" })) {
          m.focusSelection();
        } else {
          m.clearExclusions();
        }
        m.saveLS();
      }
    });

    hk.on("show-all-tracks", () => {
      m.clearExclusions();
      m.saveLS();
    });

    hk.on("select-all", () => m.selectAll());

    hk.on("undo", () => void this.cmdMgr.undo(m));

    hk.on("toggle-view", () => void this.embedding.toggleViewMode());

    hk.on("toggle-clusters", () => this.analysis.toggleColorByCluster());

    hk.on("deselect", () => {
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
