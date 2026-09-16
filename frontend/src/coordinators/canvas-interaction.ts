// ─── CanvasInteractionController ──────────────────────────────
//
// Owns all canvas pointer gestures (click/box/lasso select, tag drag, transform
// handles, pan/zoom, folder-drop drag) and the explode-hold key. Extracted from
// `Controller` — gestures are a self-contained input layer whose outward effects are
// model mutations, tag writes (via the controller's debounced queue), and view
// repaints.
//
// Deps are structural interfaces so the gesture state machine is unit-testable with
// a fake canvas and without the DOM-heavy component classes.

import { type Model, type Track, isAudioFeatureAxisId } from "../model";
import type { CanvasView, TxHandle } from "../components";

const DRAG_THRESH = 4;
const ZOOM_FACTOR = 1.12;

const clamp = (v: number, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));

function axisWritable(axis: string | null): boolean {
  return !!(axis && !isAudioFeatureAxisId(axis));
}

function pointInPoly(
  px: number,
  py: number,
  poly: [number, number][],
): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i],
      [xj, yj] = poly[j];
    if (
      yi > py !== yj > py &&
      px < ((xj - xi) * (py - yi)) / (yj - yi) + xi
    )
      inside = !inside;
  }
  return inside;
}

// ─── Internal types ──────────────────────────────────────────
interface DragSnap {
  track: Track;
  x: number;
  y: number;
}

interface TxFormSnap {
  wx: number;
  wy: number;
  track: Track;
}

interface TxBounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

type MouseMode =
  | "idle"
  | "pan"
  | "lasso"
  | "boxsel"
  | "pending"
  | "drag"
  | "txform";

interface MouseState {
  mode: MouseMode;
  sx: number;
  sy: number;
  lx: number;
  ly: number;
  hitIdx: number;
  /** Was the hit track already in the selection when mousedown fired? */
  hitWasSelected: boolean;
  /** Did mousedown land inside the selection bounding box (but not on a dot)? */
  hitInSelectionBox: boolean;
  snap: Map<string, DragSnap> | null;
  draggingOutside: boolean;
  highlightedFolder: HTMLElement | null;
  txHandle: string | null;
  txSnap: Map<string, TxFormSnap> | null;
  txBounds: TxBounds | null;
  txStart: { wx: number; wy: number } | null;
}

export interface CanvasInteractionDeps {
  model: Model;
  canvas: CanvasView;
  status: { update(m: Model): void };
  props: { render(): void };
  batch: { render(): void };
  renderTagPanels(): void;
  /** Queue one optimistic tag write for the debounced batch flush. */
  queueTagWrite(path: string, tag: string, value: number): void;
  /** Flush queued tag writes to the server now. */
  flushTagWrites(): void;
  /** Move the current selection into a folder (undoable command). */
  moveSelectionToFolder(dest: string): void;
  /** Folder-tree glow follows the hovered track/folder. */
  onHoverFolderChanged(): void;
  /** Start/stop the hover audio preview for the hovered track. */
  onHoverPreview(): void;
  stopPreview(): void;
}

export class CanvasInteractionController {
  private mouse: MouseState;
  /** True while the pointer is inside the viewport canvas (for hover resync). */
  private _pointerOverCanvas = false;

  constructor(private deps: CanvasInteractionDeps) {
    this.mouse = {
      mode: "idle",
      sx: 0,
      sy: 0,
      lx: 0,
      ly: 0,
      hitIdx: -1,
      hitWasSelected: false,
      hitInSelectionBox: false,
      snap: null,
      draggingOutside: false,
      highlightedFolder: null,
      txHandle: null,
      txSnap: null,
      txBounds: null,
      txStart: null,
    };
  }

  /** Wire canvas pointer events and the explode-hold key. */
  bind(): void {
    this._bindCanvas();
    this._bindExplodeKeys();
    this._bindInspectKey();
  }

  /** True while a drag/transform gesture is in progress (hover clearing is suppressed). */
  isPointerBusy(): boolean {
    return this.mouse.mode === "drag" || this.mouse.mode === "txform";
  }

  /* ── Canvas mouse events ────────────────────────────────── */

  private _bindCanvas(): void {
    const $c = this.deps.canvas.$canvas;
    $c.addEventListener("mouseenter", () => {
      this._pointerOverCanvas = true;
    });
    $c.addEventListener("mousedown", (e) => this._onDown(e));
    $c.addEventListener("mousemove", (e) => this._onMove(e));
    $c.addEventListener("mouseup", (e) => this._onUp(e));
    $c.addEventListener("mouseleave", () => this._onLeave());
    $c.addEventListener("wheel", (e) => this._onWheel(e), {
      passive: false,
    });
    $c.addEventListener("contextmenu", (e) => e.preventDefault());
  }

  private _onDown(e: MouseEvent): void {
    const m = this.deps.model;
    const cv = this.deps.canvas;
    const mouse = this.mouse;
    const sx = e.offsetX,
      sy = e.offsetY;
    mouse.sx = sx;
    mouse.sy = sy;
    mouse.lx = sx;
    mouse.ly = sy;
    mouse.hitWasSelected = false;
    mouse.hitInSelectionBox = false;

    if (e.button === 1 || (e.button === 0 && e.altKey)) {
      mouse.mode = "pan";
      cv.$canvas.style.cursor = "grabbing";
      return;
    }
    if (e.button !== 0) return;

    // Resize handles always take priority (disabled in embedding mode).
    const txh =
      m.selected.size > 0 && m.viewMode !== "embeddings"
        ? cv.hitTestTransform(sx, sy)
        : null;
    if (txh && txh.type !== "move") {
      this._startTxform(txh.type, sx, sy);
      return;
    }

    const tracks = m.tracks;
    const hit = cv.hitTest(sx, sy);
    mouse.hitIdx = hit;

    // A plain click on empty background closes the pinned inspect card.
    if (hit < 0 && !e.shiftKey && !e.ctrlKey && !e.metaKey && !e.altKey)
      cv.unpinInspect();

    // Shift on empty space (no dot, not inside selection box) → lasso.
    if (e.shiftKey && hit < 0 && !(txh?.type === "move")) {
      mouse.mode = "lasso";
      cv.lassoPoints = [[sx, sy]];
      cv.intersectMode = e.ctrlKey || e.metaKey;
      return;
    }

    const additive = e.ctrlKey || e.metaKey || e.shiftKey;

    if (hit >= 0) {
      // Clicked on a track dot.
      const hitPath = tracks[hit].path;
      mouse.hitWasSelected = m.selected.has(hitPath);
      if (!mouse.hitWasSelected) {
        if (!additive) m.selected.clear();
        m.selected.add(hitPath);
      }
      // In embedding mode allow pending so a threshold drag can trigger folder-drop,
      // but skip the tag-position snap (embedding positions are used for ghosts instead).
      mouse.mode = "pending";
      if (m.viewMode !== "embeddings") {
        this._fillDragSnapFromSelection();
      }
    } else if (
      txh?.type === "move" &&
      m.viewMode !== "embeddings" &&
      (axisWritable(m.axisX) || axisWritable(m.axisY))
    ) {
      // Clicked inside the selection bounding box — same drag path as track drag.
      mouse.hitInSelectionBox = true;
      mouse.mode = "pending";
      this._fillDragSnapFromSelection();
    } else {
      // Clicked on empty background → box selection.
      mouse.mode = "boxsel";
      cv.boxSel = { x0: sx, y0: sy, x1: sx, y1: sy };
      cv.intersectMode = e.ctrlKey || e.metaKey;
    }

    cv.scheduleDraw();
    this.deps.status.update(m);
    this.deps.props.render();
    this.deps.batch.render();
  }

  /** Tag-drag snapshot for the current selection; used for canvas drag and folder-drop drag. */
  private _fillDragSnapFromSelection(): void {
    const m = this.deps.model;
    const mouse = this.mouse;
    mouse.snap = new Map();
    for (const path of m.selected) {
      const t = m.trackByPath(path);
      if (t)
        mouse.snap.set(path, {
          track: t,
          x: m.axisX ? m.axisScalar(t, m.axisX) : 0.5,
          y: m.axisY ? m.axisScalar(t, m.axisY) : 0.5,
        });
    }
  }

  /** Shared path: tag drag on canvas + drop on folder tree (from track pending or box-select). */
  private _startDocDragFromSelection(): void {
    const m = this.deps.model;
    const cv = this.deps.canvas;
    const mouse = this.mouse;
    m.suppressExplodeVisual = true;
    this._fillDragSnapFromSelection();
    mouse.mode = "drag";
    cv.$canvas.style.cursor = "move";
    const ghosts = new Map<string, { wx: number; wy: number; folder: string }>();
    for (const [path, snap] of mouse.snap!) {
      // In embedding mode use the embedding position for the ghost dot so it sits
      // on top of the dot the user clicked rather than at the (invisible) tag position.
      const ep = m.viewMode === "embeddings" ? m.embeddingPositions.get(path) : null;
      ghosts.set(path, {
        wx: ep ? ep.x : snap.x,
        wy: ep ? ep.y : snap.y,
        folder: snap.track.folder ?? "",
      });
    }
    cv.dragGhosts = ghosts;
    document.addEventListener("mousemove", this._onDocDragMove);
    document.addEventListener("mouseup", this._onDocDragUp);
  }

  private _startTxform(
    handleType: string,
    sx: number,
    sy: number,
  ): void {
    const m = this.deps.model;
    const cv = this.deps.canvas;
    const mouse = this.mouse;
    m.suppressExplodeVisual = true;
    const [wx, wy] = cv.s2w(sx, sy);

    mouse.mode = "txform";
    mouse.txHandle = handleType;
    mouse.txStart = { wx, wy };
    mouse.txBounds = cv._txWorld
      ? { ...cv._txWorld }
      : { minX: 0, maxX: 1, minY: 0, maxY: 1 };
    mouse.txSnap = new Map();

    for (const path of m.selected) {
      const t = m.trackByPath(path);
      if (t)
        mouse.txSnap.set(path, {
          wx: m.axisX ? m.axisScalar(t, m.axisX) : 0.5,
          wy: m.axisY ? m.axisScalar(t, m.axisY) : 0.5,
          track: t,
        });
    }

    const cursor =
      handleType === "move"
        ? "move"
        : handleType === "n" || handleType === "s"
          ? "ns-resize"
          : handleType === "e" || handleType === "w"
            ? "ew-resize"
            : handleType === "nw" || handleType === "se"
              ? "nwse-resize"
              : "nesw-resize";
    cv.$canvas.style.cursor = cursor;
  }

  private _applyTxform(wx: number, wy: number): void {
    const m = this.deps.model;
    const mouse = this.mouse;
    const h = mouse.txHandle!;
    const { minX, maxX, minY, maxY } = mouse.txBounds!;

    if (h === "move") {
      const dx = wx - mouse.txStart!.wx;
      const dy = wy - mouse.txStart!.wy;
      for (const [, snap] of mouse.txSnap!) {
        if (axisWritable(m.axisX))
          snap.track.tags[m.axisX!] = clamp(snap.wx + dx);
        if (axisWritable(m.axisY))
          snap.track.tags[m.axisY!] = clamp(snap.wy + dy);
      }
      m.emit("tags-dirty");
      return;
    }

    const affectsX = m.axisX && /e|w/.test(h);
    const affectsY = m.axisY && /n|s/.test(h);

    let scaleX = 1,
      axAnchor = minX;
    if (affectsX) {
      const isLeft = /w/.test(h);
      axAnchor = isLeft ? maxX : minX;
      const origEdge = isLeft ? minX : maxX;
      const edgeDist = origEdge - axAnchor;
      scaleX = edgeDist !== 0 ? (wx - axAnchor) / edgeDist : 1;
    }

    let scaleY = 1,
      ayAnchor = minY;
    if (affectsY) {
      const isTop = /n/.test(h);
      ayAnchor = isTop ? minY : maxY;
      const origEdge = isTop ? maxY : minY;
      const edgeDist = origEdge - ayAnchor;
      scaleY = edgeDist !== 0 ? (wy - ayAnchor) / edgeDist : 1;
    }

    for (const [, snap] of mouse.txSnap!) {
      const t = snap.track;
      if (affectsX && axisWritable(m.axisX))
        t.tags[m.axisX!] = clamp(axAnchor + (snap.wx - axAnchor) * scaleX);
      if (affectsY && axisWritable(m.axisY))
        t.tags[m.axisY!] = clamp(ayAnchor + (snap.wy - ayAnchor) * scaleY);
    }
    m.emit("tags-dirty");
  }

  /**
   * Recompute scatter hit/hover after ``tracks`` changed (indices are not stable).
   * Runs on the frame after ``scheduleDraw`` so ``positions`` matches the new list.
   */
  resyncHoverFromLastPointer(): void {
    if (!this._pointerOverCanvas || this.mouse.mode !== "idle") return;
    this._updateCanvasHoverAt(this.mouse.lx, this.mouse.ly);
  }

  /** Update hovered dot + sibling glow from canvas pixel coordinates. */
  private _updateCanvasHoverAt(sx: number, sy: number): void {
    const m = this.deps.model;
    const cv = this.deps.canvas;
    const prevIdx = cv.hoveredIdx;
    const prevPath =
      prevIdx >= 0 ? (m.tracks[prevIdx]?.path ?? null) : null;

    this.mouse.lx = sx;
    this.mouse.ly = sy;

    cv.hoveredIdx = cv.hitTest(sx, sy);
    const newPath =
      cv.hoveredIdx >= 0 ? (m.tracks[cv.hoveredIdx]?.path ?? null) : null;

    if (cv.hoveredIdx >= 0) {
      const ht = m.tracks[cv.hoveredIdx];
      cv.hoveredSiblingFolder = ht ? (ht.folder ?? "") : null;
    } else {
      cv.hoveredSiblingFolder = null;
    }

    if (cv.hoveredIdx === prevIdx && newPath === prevPath) return;

    clearTimeout(cv.tipTimer);
    cv.tipReady = false;
    cv.$tip.classList.add("hidden");
    if (cv.hoveredIdx >= 0)
      cv.tipTimer = setTimeout(() => {
        cv.tipReady = true;
        cv.scheduleDraw();
      }, 150) as unknown as number;
    cv.scheduleDraw();
    this.deps.onHoverPreview();
    this.deps.onHoverFolderChanged();
  }

  private _onMove(e: MouseEvent): void {
    const m = this.deps.model;
    const cv = this.deps.canvas;
    const mouse = this.mouse;
    const sx = e.offsetX,
      sy = e.offsetY;

    if (mouse.mode === "pan") {
      const [wx1, wy1] = cv.s2w(mouse.lx, mouse.ly);
      const [wx2, wy2] = cv.s2w(sx, sy);
      m.vp.ox -= wx2 - wx1;
      m.vp.oy -= wy2 - wy1;
      mouse.lx = sx;
      mouse.ly = sy;
      cv.scheduleDraw();
      return;
    }

    if (mouse.mode === "boxsel") {
      cv.boxSel = { x0: mouse.sx, y0: mouse.sy, x1: sx, y1: sy };
      cv.intersectMode = e.ctrlKey || e.metaKey;
      cv.scheduleDraw();
      return;
    }

    if (mouse.mode === "lasso") {
      cv.lassoPoints.push([sx, sy]);
      cv.intersectMode = e.ctrlKey || e.metaKey;
      cv.scheduleDraw();
      return;
    }

    if (mouse.mode === "txform") {
      const [wx, wy] = cv.s2w(sx, sy);
      this._applyTxform(wx, wy);
      cv.scheduleDraw();
      return;
    }

    if (mouse.mode === "pending") {
      if (
        Math.hypot(sx - mouse.sx, sy - mouse.sy) > DRAG_THRESH &&
        m.selected.size > 0 &&
        (m.viewMode === "embeddings" ||
          axisWritable(m.axisX) ||
          axisWritable(m.axisY))
      ) {
        this._startDocDragFromSelection();
      }
      return;
    }

    if (mouse.mode === "drag") {
      // In embedding mode the drag is only for folder-drop; skip tag mutations.
      if (m.viewMode !== "embeddings") {
        const [swx, swy] = cv.s2w(mouse.sx, mouse.sy);
        const [cwx, cwy] = cv.s2w(sx, sy);
        const dx = cwx - swx,
          dy = cwy - swy;
        for (const [, snap] of mouse.snap!) {
          if (axisWritable(m.axisX))
            snap.track.tags[m.axisX!] = clamp(snap.x + dx);
          if (axisWritable(m.axisY))
            snap.track.tags[m.axisY!] = clamp(snap.y + dy);
        }
        m.emit("tags-dirty");
      }
      cv.scheduleDraw();
      return;
    }

    // Idle hover
    this._updateCanvasHoverAt(sx, sy);

    const txh: TxHandle | { type: string; cursor: string } | null =
      m.selected.size > 0 ? cv.hitTestTransform(sx, sy) : null;
    if (txh) {
      cv.$canvas.style.cursor = txh.cursor;
    } else {
      cv.$canvas.style.cursor = "";
    }
  }

  private _onUp(e: MouseEvent): void {
    const m = this.deps.model;
    const cv = this.deps.canvas;
    const mouse = this.mouse;
    const tracks = m.tracks;

    if (mouse.mode === "boxsel") {
      const bs = cv.boxSel;
      cv.boxSel = null;
      cv.intersectMode = false;
      if (bs) {
        const bL = Math.min(bs.x0, bs.x1),
          bR = Math.max(bs.x0, bs.x1);
        const bT = Math.min(bs.y0, bs.y1),
          bB = Math.max(bs.y0, bs.y1);
        const isClick = bR - bL < 4 && bB - bT < 4;
        if (e.ctrlKey || e.metaKey) {
          // AND / intersect: keep only the existing selection that also falls inside the box
          if (!isClick) {
            const inBox = new Set<string>(
              cv.positions
                .filter((p) => p.sx >= bL && p.sx <= bR && p.sy >= bT && p.sy <= bB)
                .map((p) => tracks[p.idx].path),
            );
            for (const path of [...m.selected])
              if (!inBox.has(path)) m.selected.delete(path);
          }
        } else {
          m.selected.clear();
          if (!isClick) {
            for (const p of cv.positions)
              if (p.sx >= bL && p.sx <= bR && p.sy >= bT && p.sy <= bB)
                m.selected.add(tracks[p.idx].path);
          }
        }
      }
      this._finalizeSelection();
      return;
    }

    if (mouse.mode === "txform") {
      for (const path of m.selected) {
        const t = m.trackByPath(path);
        if (!t) continue;
        if (axisWritable(m.axisX) && t.tags[m.axisX!] !== undefined)
          this.deps.queueTagWrite(path, m.axisX!, t.tags[m.axisX!]);
        if (axisWritable(m.axisY) && t.tags[m.axisY!] !== undefined)
          this.deps.queueTagWrite(path, m.axisY!, t.tags[m.axisY!]);
      }
      this.deps.flushTagWrites();
      mouse.mode = "idle";
      mouse.txSnap = null;
      m.suppressExplodeVisual = false;
      cv.$canvas.style.cursor = "";
      cv.scheduleDraw();
      this.deps.status.update(m);
      this.deps.props.render();
      this.deps.batch.render();
      this.deps.renderTagPanels();
      return;
    }

    if (mouse.mode === "lasso") {
      const lassoPts = cv.lassoPoints;
      cv.lassoPoints = [];
      cv.intersectMode = false;
      const inLasso = new Set<string>(
        cv.positions
          .filter((p) => pointInPoly(p.sx, p.sy, lassoPts))
          .map((p) => tracks[p.idx].path),
      );
      if (e.ctrlKey || e.metaKey) {
        // AND / intersect: keep only the existing selection that also falls inside the lasso
        for (const path of [...m.selected])
          if (!inLasso.has(path)) m.selected.delete(path);
      } else {
        // Shift starts lasso, so treat shift as additive union
        if (!e.shiftKey) m.selected.clear();
        for (const path of inLasso) m.selected.add(path);
      }
    }

    if (mouse.mode === "pending") {
      const hitPath =
        mouse.hitIdx >= 0 ? tracks[mouse.hitIdx]?.path : null;
      const additive = e.ctrlKey || e.metaKey || e.shiftKey;

      if (mouse.hitInSelectionBox) {
        // Released inside the selection box without dragging: keep selection unchanged.
      } else if (additive) {
        if (hitPath && mouse.hitWasSelected) {
          // Toggle off a track that was already selected.
          m.selected.delete(hitPath);
        }
        // If !hitWasSelected the track was already added in _onDown, keep it.
      } else {
        // Plain click: narrow down to just this track (or clear if none).
        m.selected.clear();
        if (hitPath) m.selected.add(hitPath);
      }
    }

    if (mouse.mode !== "drag") {
      this._finalizeSelection();
    }
  }

  /** Shared post-selection cleanup: resets interaction state and emits
   *  "change" so all views (including folder highlights) update together. */
  private _finalizeSelection(): void {
    this.mouse.mode = "idle";
    this.mouse.snap = null;
    this.deps.canvas.$canvas.style.cursor = "";
    this.deps.model.suppressExplodeVisual = false;
    this.deps.model.emit("change");
  }

  private _onLeave(): void {
    const mouse = this.mouse;
    const cv = this.deps.canvas;
    if (mouse.mode === "drag" || mouse.mode === "txform") {
      mouse.draggingOutside = true;
      return;
    }
    if (mouse.mode !== "idle")
      this._onUp({
        offsetX: mouse.lx,
        offsetY: mouse.ly,
        ctrlKey: false,
        metaKey: false,
      } as MouseEvent);
    cv.boxSel = null;
    cv.intersectMode = false;
    this._pointerOverCanvas = false;
    cv.clearHover();
    cv.$canvas.style.cursor = "";
    this.deps.stopPreview();
    cv.scheduleDraw();
    this.deps.onHoverFolderChanged();
  }

  private _onWheel(e: WheelEvent): void {
    e.preventDefault();
    const m = this.deps.model;
    const cv = this.deps.canvas;
    const vp = m.vp;
    const sx = e.offsetX,
      sy = e.offsetY;

    const norm =
      e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? cv.$canvas.clientHeight : 1;
    const dx = e.deltaX * norm,
      dy = e.deltaY * norm;

    if (m.explodeHold && this._pointerOverCanvas) {
      const step =
        e.deltaMode === 1
          ? Math.sign(dy) * 0.01
          : e.deltaMode === 2
            ? Math.sign(dy) * 0.015
            : -dy * 0.0002;
      m.explodeStrength = clamp(m.explodeStrength + step, 0.05, 0.2);
      const $ex = document.getElementById(
        "explode-strength-range",
      ) as HTMLInputElement | null;
      if ($ex) $ex.value = String(Math.round(m.explodeStrength * 100) / 100);
      m.saveLS();
      cv.scheduleDraw();
      return;
    }

    if (e.ctrlKey) {
      const factor = Math.exp(-dy * 0.005);
      const [wx, wy] = cv.s2w(sx, sy);
      const zMin = cv.minZoom();
      vp.zoom = clamp(vp.zoom * factor, zMin, 50);
      const [wx2, wy2] = cv.s2w(sx, sy);
      vp.ox += wx - wx2;
      vp.oy += wy - wy2;
    } else {
      const isTrackpad =
        e.deltaMode === 0 && (Math.abs(dx) > 0 || Math.abs(dy) < 50);
      if (isTrackpad) {
        const [wx0, wy0] = cv.s2w(0, 0);
        const [wx1, wy1] = cv.s2w(dx, dy);
        vp.ox += wx1 - wx0;
        vp.oy += wy1 - wy0;
      } else {
        const f = dy < 0 ? ZOOM_FACTOR : 1 / ZOOM_FACTOR;
        const [wx, wy] = cv.s2w(sx, sy);
        vp.zoom = clamp(vp.zoom * f, cv.minZoom(), 50);
        const [wx2, wy2] = cv.s2w(sx, sy);
        vp.ox += wx - wx2;
        vp.oy += wy - wy2;
      }
    }
    cv.scheduleDraw();
  }

  /* ── Document-level drag (folder drop) ──────────────────── */

  private _cleanupDocDrag(): void {
    document.removeEventListener("mousemove", this._onDocDragMove);
    document.removeEventListener("mouseup", this._onDocDragUp);
    if (this.mouse.highlightedFolder) {
      this.mouse.highlightedFolder.classList.remove("drop-target");
      this.mouse.highlightedFolder = null;
    }
    this.mouse.draggingOutside = false;
  }

  private _onDocDragMove = (e: MouseEvent): void => {
    if (this.mouse.mode !== "drag") {
      this._cleanupDocDrag();
      return;
    }
    const el = document.elementFromPoint(e.clientX, e.clientY);
    const folderEl =
      el?.closest<HTMLElement>(".folder-label[data-path]") ?? null;
    if (this.mouse.highlightedFolder !== folderEl) {
      if (this.mouse.highlightedFolder)
        this.mouse.highlightedFolder.classList.remove("drop-target");
      this.mouse.highlightedFolder = folderEl;
      if (folderEl) folderEl.classList.add("drop-target");
    }
  };

  private _onDocDragUp = async (e: MouseEvent): Promise<void> => {
    if (this.mouse.mode !== "drag") {
      this._cleanupDocDrag();
      return;
    }
    const el = document.elementFromPoint(e.clientX, e.clientY);
    const folderEl =
      el?.closest<HTMLElement>(".folder-label[data-path]") ?? null;
    this._cleanupDocDrag();

    if (folderEl) {
      this._revertDragSnapToBaseline();
    } else {
      this._commitDrag();
    }

    this.deps.model.suppressExplodeVisual = false;

    // Reset interaction state immediately so that any canvas mousemove events
    // fired during the async network call below are ignored.
    this.mouse.mode = "idle";
    this.mouse.snap = null;
    this.deps.canvas.dragGhosts = null;
    this.deps.canvas.$canvas.style.cursor = "";
    this.deps.canvas.scheduleDraw();
    this.deps.status.update(this.deps.model);
    this.deps.props.render();
    this.deps.batch.render();
    this.deps.renderTagPanels();

    if (folderEl) {
      const dest =
        folderEl.dataset.path === "." ? "" : folderEl.dataset.path!;
      this.deps.moveSelectionToFolder(dest);
    }
  };

  /* ── Debounced tag writes ───────────────────────────────── */

  /** Undo in-viewport tag offsets when the drag ends with a folder drop instead of a canvas commit. */
  private _revertDragSnapToBaseline(): void {
    const m = this.deps.model;
    // In embedding mode no tag values were mutated during the drag, nothing to revert.
    if (m.viewMode === "embeddings") return;
    for (const [, snap] of this.mouse.snap ?? []) {
      if (axisWritable(m.axisX)) snap.track.tags[m.axisX!] = snap.x;
      if (axisWritable(m.axisY)) snap.track.tags[m.axisY!] = snap.y;
    }
    m.emit("tags-dirty");
  }

  private _commitDrag(): void {
    const m = this.deps.model;
    // In embedding mode no tag values were changed during the drag; nothing to persist.
    if (m.viewMode === "embeddings") return;
    for (const [path, snap] of this.mouse.snap ?? []) {
      if (axisWritable(m.axisX))
        this.deps.queueTagWrite(path, m.axisX!, snap.track.tags[m.axisX!]);
      if (axisWritable(m.axisY))
        this.deps.queueTagWrite(path, m.axisY!, snap.track.tags[m.axisY!]);
    }
    this.deps.flushTagWrites();
  }

  /* ── Fit view to visible track bounds ───────────────────── */

  private _vpFromWorldBounds(
    minX: number,
    maxX: number,
    minY: number,
    maxY: number,
  ): { ox: number; oy: number; zoom: number } {
    const midX = (minX + maxX) / 2,
      midY = (minY + maxY) / 2;
    const margin = 0.1;
    const zx =
      maxX - minX > 0.01 ? (1 - 2 * margin) / (maxX - minX) : 50;
    const zy =
      maxY - minY > 0.01 ? (1 - 2 * margin) / (maxY - minY) : 50;
    const zoom = Math.max(this.deps.canvas.minZoom(), Math.min(50, Math.min(zx, zy)));
    return {
      zoom,
      ox: midX - 1 / (2 * zoom),
      oy: midY - 1 / (2 * zoom),
    };
  }

  private _computeFitVP(): { ox: number; oy: number; zoom: number } {
    const m = this.deps.model;

    // In embedding mode positions are already normalised to [0,1], so the
    // default viewport shows everything.
    if (m.viewMode === "embeddings") return { ox: 0, oy: 0, zoom: 1 };

    const visible = m.tracks.filter((t) => m.passesFilter(t));
    if (visible.length === 0) return { ox: 0, oy: 0, zoom: 1 };

    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity;
    for (const t of visible) {
      const wx = m.axisX ? m.axisScalar(t, m.axisX) : 0.5;
      const wy = m.axisY ? m.axisScalar(t, m.axisY) : 0.5;
      if (wx < minX) minX = wx;
      if (wx > maxX) maxX = wx;
      if (wy < minY) minY = wy;
      if (wy > maxY) maxY = wy;
    }
    if (maxX - minX < 0.01 && maxY - minY < 0.01)
      return { ox: 0, oy: 0, zoom: 1 };

    return this._vpFromWorldBounds(minX, maxX, minY, maxY);
  }

  private _p80Radius(dists: number[]): number {
    if (!dists.length) return 0;
    const s = [...dists].sort((a, b) => a - b);
    const idx = Math.min(
      s.length - 1,
      Math.max(0, Math.ceil(0.8 * s.length) - 1),
    );
    return s[idx];
  }

  /** Fit viewport to an 80th-percentile disc around the centroid of *subset*. */
  private _fitViewToTracksPercentile(subset: Track[]): void {
    const m = this.deps.model;
    const pos: { wx: number; wy: number }[] = [];
    for (const t of subset) {
      const p = this.deps.canvas.worldPosForTrack(t);
      if (p) pos.push(p);
    }
    if (pos.length === 0) return;
    let cx = 0,
      cy = 0;
    for (const p of pos) {
      cx += p.wx;
      cy += p.wy;
    }
    cx /= pos.length;
    cy /= pos.length;
    const dists = pos.map((p) => Math.hypot(p.wx - cx, p.wy - cy));
    const rmax = this._p80Radius(dists);
    const inside = pos.filter(
      (p) => Math.hypot(p.wx - cx, p.wy - cy) <= rmax + 1e-9,
    );
    if (inside.length === 0) return;
    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity;
    for (const p of inside) {
      if (p.wx < minX) minX = p.wx;
      if (p.wx > maxX) maxX = p.wx;
      if (p.wy < minY) minY = p.wy;
      if (p.wy > maxY) maxY = p.wy;
    }
    if (maxX - minX < 0.01 && maxY - minY < 0.01) {
      Object.assign(m.vp, { ox: 0, oy: 0, zoom: 1 });
      return;
    }
    Object.assign(m.vp, this._vpFromWorldBounds(minX, maxX, minY, maxY));
  }

  /** Fit viewport to an 80th-percentile disc around the centroid (folder/track folder hover + H). */
  private _fitViewToFolderPercentile(prefixNorm: string): void {
    const m = this.deps.model;
    const subset: Track[] = [];
    for (const t of m.allTracks) {
      if (!m.trackInViewUnion(t) || m.isTrackHidden(t)) continue;
      if (!m.passesFilter(t)) continue;
      const tf = t.folder ?? "";
      if (prefixNorm !== "") {
        if (tf !== prefixNorm && !tf.startsWith(prefixNorm + "/")) continue;
      }
      subset.push(t);
    }
    this._fitViewToTracksPercentile(subset);
  }

  /** Fit viewport to the visible members of a hovered cluster (80th-percentile disc). */
  private _fitViewToClusterPercentile(cid: number): void {
    const m = this.deps.model;
    const keep = m.clusterPaths(cid);
    if (keep.size === 0) return;
    this._fitViewToTracksPercentile(
      m.tracks.filter((t) => keep.has(t.path) && m.passesFilter(t)),
    );
  }

  private _fitViewToTracks(): void {
    const target = this._computeFitVP();
    Object.assign(this.deps.model.vp, target);
  }

  /**
   * Folder prefix for H / S hotkeys: sidebar folder-row hover wins;
   * otherwise the hovered canvas track's folder (Library root = ``""``).
   */
  effectiveHoveredFolderPrefix(): string | null {
    const cv = this.deps.canvas;
    if (cv.hoveredFolderPrefix !== null) return cv.hoveredFolderPrefix;
    if (cv.hoveredIdx >= 0) {
      const t = this.deps.model.tracks[cv.hoveredIdx];
      if (t) return t.folder ?? "";
    }
    return null;
  }

  /** Fit-view hotkey (H): hovered cluster/folder 80th-percentile disc, else full view toggle. */
  fitView(): void {
    const m = this.deps.model;
    const cid = this.deps.canvas.hoveredClusterId;
    if (cid !== null && m.clusterPaths(cid).size > 0) {
      this._fitViewToClusterPercentile(cid);
      this.deps.canvas.scheduleDraw();
      m.saveLS();
      return;
    }
    const hp = this.effectiveHoveredFolderPrefix();
    if (hp !== null) {
      this._fitViewToFolderPercentile(hp);
      this.deps.canvas.scheduleDraw();
      m.saveLS();
      return;
    }
    const vp = m.vp;
    const fit = this._computeFitVP();
    const atFit =
      Math.abs(vp.ox - fit.ox) < 0.002 &&
      Math.abs(vp.oy - fit.oy) < 0.002 &&
      Math.abs(vp.zoom - fit.zoom) < 0.05;
    if (atFit) {
      vp.ox = 0;
      vp.oy = 0;
      vp.zoom = 1;
    } else {
      this._fitViewToTracks();
    }
    this.deps.canvas.scheduleDraw();
    m.saveLS();
  }

  /**
   * Isolate hotkey (S): hovered cluster → hovered folder → selection → everything.
   *
   * A second press on an already-isolated target clears the exclusions (show all);
   * with nothing hovered and no selection it also shows all.
   */
  solo(): void {
    const m = this.deps.model;

    const cid = this.deps.canvas.hoveredClusterId;
    if (cid !== null) {
      const keep = m.clusterPaths(cid);
      if (keep.size === 0) return;
      if (m.pathsFullyHidden(keep)) {
        m.revealThenFocusPaths(keep);
      } else if (m.hasVisibleTracksOutside(keep)) {
        m.focusPaths(keep);
      } else {
        m.clearExclusions();
      }
      m.saveLS();
      return;
    }

    const hp = this.effectiveHoveredFolderPrefix();
    if (hp !== null) {
      if (m.folderSubtreeFullyExcluded(hp)) {
        m.revealThenFocusFolderSubtree(hp);
      } else if (m.hasVisibleTracksOutside(m.pathsUnderPrefix(hp))) {
        m.focusFolderSubtree(hp);
      } else {
        m.clearExclusions();
      }
      m.saveLS();
      return;
    }

    if (m.selected.size) {
      if (m.hasVisibleTracksOutside(m.selected)) {
        m.focusSelection();
      } else {
        m.clearExclusions();
      }
      m.saveLS();
      return;
    }

    m.clearExclusions();
    m.saveLS();
  }

  /**
   * Fit the viewport to a specific set of track paths (semantic-search result
   * focus). Unlike the folder fit, a single track is zoomed to rather than reset
   * to the full view — focusing one hit is the whole point.
   */
  fitViewToPaths(paths: Iterable<string>): void {
    const m = this.deps.model;
    const wanted = new Set(paths);
    const pos: { wx: number; wy: number }[] = [];
    for (const t of m.allTracks) {
      if (!wanted.has(t.path)) continue;
      const p = this.deps.canvas.worldPosForTrack(t);
      if (p) pos.push(p);
    }
    if (pos.length === 0) return;
    let minX = Infinity,
      maxX = -Infinity,
      minY = Infinity,
      maxY = -Infinity;
    for (const p of pos) {
      if (p.wx < minX) minX = p.wx;
      if (p.wx > maxX) maxX = p.wx;
      if (p.wy < minY) minY = p.wy;
      if (p.wy > maxY) maxY = p.wy;
    }
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    // A single hit has zero extent; give it a readable window, not max zoom.
    const half = Math.max(maxX - minX, maxY - minY, 0.05) / 2;
    Object.assign(
      m.vp,
      this._vpFromWorldBounds(cx - half, cx + half, cy - half, cy + half),
    );
    this.deps.canvas.scheduleDraw();
    m.saveLS();
  }

  /**
   * Abort an in-flight box/lasso/drag gesture (the deselect hotkey).
   * Returns true when a gesture was cancelled; false means the caller should
   * fall through to clearing the selection.
   */
  cancelActiveGesture(): boolean {
    const m = this.deps.model;
    const cv = this.deps.canvas;
    const mouse = this.mouse;
    if (mouse.mode === "drag") {
      this._revertDragSnapToBaseline();
      this._cleanupDocDrag();
      m.suppressExplodeVisual = false;
      mouse.mode = "idle";
      mouse.snap = null;
      cv.dragGhosts = null;
      cv.$canvas.style.cursor = "";
      cv.scheduleDraw();
      this.deps.status.update(m);
      this.deps.props.render();
      this.deps.batch.render();
      return true;
    }
    if (mouse.mode === "boxsel") {
      cv.boxSel = null;
      cv.intersectMode = false;
      mouse.mode = "idle";
      cv.scheduleDraw();
      return true;
    }
    if (mouse.mode === "lasso") {
      cv.lassoPoints = [];
      cv.intersectMode = false;
      mouse.mode = "idle";
      cv.scheduleDraw();
      return true;
    }
    return false;
  }

  private _explodeKeysAllowed(target: EventTarget | null): boolean {
    const el = target as HTMLElement | null;
    if (!el) return true;
    if (el.tagName === "TEXTAREA") return false;
    if (el.tagName === "INPUT") {
      const type = (el as HTMLInputElement).type.toLowerCase();
      const isTextEntry =
        type === "" ||
        type === "text" ||
        type === "password" ||
        type === "email" ||
        type === "search" ||
        type === "url" ||
        type === "tel" ||
        type === "number" ||
        type === "date" ||
        type === "time" ||
        type === "datetime-local" ||
        type === "month" ||
        type === "week";
      if (isTextEntry) return false;
    }
    return true;
  }

  /** Hold **X** (outside text fields) for explode preview on the canvas. */
  private _bindExplodeKeys(): void {
    document.addEventListener("keydown", (e: KeyboardEvent) => {
      if (e.code !== "KeyX" || e.repeat) return;
      if (!this._explodeKeysAllowed(e.target)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      this.deps.model.explodeHold = true;
      this.deps.canvas.scheduleDraw();
    });
    document.addEventListener("keyup", (e: KeyboardEvent) => {
      if (e.code !== "KeyX") return;
      this.deps.model.explodeHold = false;
      this.deps.canvas.scheduleDraw();
    });
    window.addEventListener("blur", () => {
      if (!this.deps.model.explodeHold) return;
      this.deps.model.explodeHold = false;
      this.deps.canvas.scheduleDraw();
    });
  }

  /**
   * **Alt** over a hovered dot pins its inspect card open (and again closes it).
   * The card then survives leaving the canvas so it can be clicked.
   */
  private _bindInspectKey(): void {
    document.addEventListener("keydown", (e: KeyboardEvent) => {
      if (e.key !== "Alt" || e.repeat) return;
      if (!this._explodeKeysAllowed(e.target)) return;
      this.deps.canvas.toggleInspect();
    });
  }
}
