import type { Model } from "../model";


export class BatchView {
  model: Model;
  $el: HTMLElement;
  private _snapshots = new Map<string, Map<string, number>>();
  private _selHash = "";
  private _dragging = false;

  onApply: ((tag: string, updates: Map<string, number>) => void) | null =
    null;
  onDragEnd: (() => void) | null = null;

  constructor(model: Model, container: HTMLElement) {
    this.model = model;
    this.$el = container;
  }

  renderDirty(): void {
    if (this._dragging) return;
    this._snapshots.clear();
    this.render();
  }

  render(): void {
    if (this._dragging) return;

    const m = this.model;
    const hash = [...m.selected].sort().join("|");

    if (hash !== this._selHash) {
      this._selHash = hash;
      this._snapshots.clear();
    }

    this.$el.innerHTML = "";

    if (!m.selected.size) return;
    if (!m.tags.length) return;

    const grid = document.createElement("div");
    grid.className = "batch-tag-grid";

    for (const tag of m.tags) {
      const snap = this._getOrCreateSnapshot(tag);
      const vals = [...snap.values()];
      const hasVals = vals.length > 0;
      const origLo = hasVals ? Math.min(...vals) : 0.5;
      const origHi = hasVals ? Math.max(...vals) : 0.5;
      const origAvg = hasVals
        ? vals.reduce((a, b) => a + b, 0) / vals.length
        : 0.5;
      const narrow = !hasVals || origHi - origLo < 0.05;

      const label = document.createElement("span");
      label.className =
        "batch-tag-label" + (hasVals ? "" : " batch-tag-label--unset");
      label.textContent = tag;
      label.title = hasVals ? tag : tag + " (unset)";

      const slider = narrow
        ? this._makeSingleSlider(tag, origAvg, !hasVals)
        : this._makeTransformSlider(tag, origLo, origHi);

      grid.appendChild(label);
      grid.appendChild(slider);
    }

    this.$el.appendChild(grid);
  }

  /* ── Snapshot helpers ───────────────────────────────────── */

  private _getOrCreateSnapshot(tag: string): Map<string, number> {
    if (!this._snapshots.has(tag)) {
      const m = this.model;
      const snap = new Map<string, number>();
      for (const path of m.selected) {
        const t = m.trackByPath(path);
        if (t && t.tags[tag] !== undefined) snap.set(path, t.tags[tag]);
      }
      this._snapshots.set(tag, snap);
    }
    return this._snapshots.get(tag)!;
  }

  private _clearSnapshot(tag: string): void {
    this._snapshots.delete(tag);
  }

  /* ── Single translate slider (narrow / uniform range) ───── */

  private _makeSingleSlider(
    tag: string,
    initVal: number,
    isUnset = false,
  ): HTMLDivElement {
    const outer = document.createElement("div");
    outer.className = "txslider";

    const trackEl = document.createElement("div");
    trackEl.className = "txslider-track";
    outer.appendChild(trackEl);

    const knob = document.createElement("div");
    knob.className = "txslider-knob";
    if (isUnset) knob.style.opacity = "0.45";
    outer.appendChild(knob);

    let val = initVal;

    const update = () => {
      val = Math.max(0, Math.min(1, val));
      knob.style.left = `calc(${val * 100}% - 6px)`;
    };
    update();

    const fireTransform = () => {
      const snap = this._getOrCreateSnapshot(tag);
      const updates = new Map<string, number>();
      if (!snap.size) {
        for (const path of this.model.selected) updates.set(path, val);
      } else {
        const snapAvg =
          [...snap.values()].reduce((a, b) => a + b, 0) / snap.size;
        const delta = val - snapAvg;
        for (const [path, origVal] of snap)
          updates.set(path, Math.max(0, Math.min(1, origVal + delta)));
      }
      this.onApply?.(tag, updates);
    };

    knob.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      e.stopPropagation();
      e.preventDefault();
      knob.setPointerCapture(e.pointerId);
      this._dragging = true;

      const startX = e.clientX;
      const startVal = val;
      const W = outer.getBoundingClientRect().width || 100;

      knob.onpointermove = (e) => {
        val = Math.max(0, Math.min(1, startVal + (e.clientX - startX) / W));
        update();
        fireTransform();
      };
      knob.onpointerup = () => {
        knob.onpointermove = null;
        knob.onpointerup = null;
        this._dragging = false;
        this._clearSnapshot(tag);
        this.onDragEnd?.();
      };
    });

    return outer;
  }

  /* ── Transform slider (dual-knob scale/translate) ───────── */

  private _makeTransformSlider(
    tag: string,
    origLo: number,
    origHi: number,
  ): HTMLDivElement {
    const outer = document.createElement("div");
    outer.className = "txslider";

    const trackEl = document.createElement("div");
    trackEl.className = "txslider-track";
    outer.appendChild(trackEl);

    const fill = document.createElement("div");
    fill.className = "txslider-fill";
    trackEl.appendChild(fill);

    const loKnob = document.createElement("div");
    loKnob.className = "txslider-knob";
    const hiKnob = document.createElement("div");
    hiKnob.className = "txslider-knob";
    outer.appendChild(loKnob);
    outer.appendChild(hiKnob);

    let lo = origLo,
      hi = origHi;

    const update = () => {
      lo = Math.max(0, Math.min(1, lo));
      hi = Math.max(0, Math.min(1, hi));
      if (hi < lo) hi = lo;
      fill.style.left = lo * 100 + "%";
      fill.style.right = (1 - hi) * 100 + "%";
      loKnob.style.left = `calc(${lo * 100}% - 6px)`;
      hiKnob.style.left = `calc(${hi * 100}% - 6px)`;
    };
    update();

    const fireTransform = () => {
      const snap = this._getOrCreateSnapshot(tag);
      const vals = [...snap.values()];
      if (!vals.length) return;
      const sLo = Math.min(...vals);
      const sHi = Math.max(...vals);
      const range = sHi - sLo;
      const targetRange = hi - lo;

      const updates = new Map<string, number>();
      for (const [path, origVal] of snap) {
        const nv =
          range < 0.0001
            ? (lo + hi) / 2
            : lo + ((origVal - sLo) / range) * targetRange;
        updates.set(path, Math.max(0, Math.min(1, nv)));
      }
      this.onApply?.(tag, updates);
    };

    const startDrag = (
      el: HTMLElement,
      isKnob: boolean,
      isLo: boolean | null,
    ) => {
      el.addEventListener("pointerdown", (e) => {
        if (e.button !== 0) return;
        e.stopPropagation();
        e.preventDefault();
        el.setPointerCapture(e.pointerId);
        this._dragging = true;

        const startX = e.clientX;
        const startLo = lo,
          startHi = hi;
        const span = hi - lo;
        const W = outer.getBoundingClientRect().width || 100;

        el.onpointermove = (e) => {
          const delta = (e.clientX - startX) / W;
          if (!isKnob) {
            lo = Math.max(0, Math.min(1 - span, startLo + delta));
            hi = lo + span;
          } else if (isLo) {
            lo = Math.max(0, Math.min(hi, startLo + delta));
          } else {
            hi = Math.max(lo, Math.min(1, startHi + delta));
          }
          update();
          fireTransform();
        };

        el.onpointerup = () => {
          el.onpointermove = null;
          el.onpointerup = null;
          this._dragging = false;
          this._clearSnapshot(tag);
          this.onDragEnd?.();
        };
      });
    };

    fill.style.cursor = "ew-resize";
    startDrag(fill, false, null);
    startDrag(loKnob, true, true);
    startDrag(hiKnob, true, false);

    return outer;
  }
}
