import type { Model } from "../model";


export class TagPanelView {
  model: Model;
  $el: HTMLElement;

  onAxisChange: ((which: string, tag: string) => void) | null = null;
  onFilterChange: ((tag: string, range: [number, number]) => void) | null =
    null;
  onTagRename: ((oldName: string, newName: string) => void) | null = null;

  /** Tag name to auto-enter rename mode after next render. */
  private _pendingRename: string | null = null;
  private _interacting = false;

  constructor(model: Model, container: HTMLElement) {
    this.model = model;
    this.$el = container;
  }

  render(): void {
    if (this._interacting) return;
    this.$el.innerHTML = "";
    const m = this.model;
    for (const tag of m.tags) {
      const li = document.createElement("li");
      this._buildTagRow(li, tag);
      this.$el.appendChild(li);
    }
    if (this._pendingRename) {
      const tag = this._pendingRename;
      this._pendingRename = null;
      requestAnimationFrame(() => this.beginRename(tag));
    }
  }

  beginRename(tag: string): void {
    const nameEl = this.$el.querySelector(
      `[data-tag="${CSS.escape(tag)}"]`,
    ) as HTMLElement | null;
    if (!nameEl) return;

    const oldName = tag;
    const input = document.createElement("input");
    input.type = "text";
    input.className = "tag-rename-input";
    input.value = oldName;

    let done = false;
    nameEl.replaceWith(input);
    input.select();
    input.focus();

    const commit = () => {
      if (done) return;
      done = true;
      const raw = input.value.trim().toLowerCase().replace(/\s+/g, "_");
      if (!raw || raw === oldName) {
        input.replaceWith(nameEl);
        return;
      }
      if (this.model.tags.includes(raw) && raw !== oldName) {
        input.replaceWith(nameEl);
        return;
      }
      this.onTagRename?.(oldName, raw);
    };

    const cancel = () => {
      if (done) return;
      done = true;
      input.replaceWith(nameEl);
    };

    input.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") {
        ev.preventDefault();
        commit();
      }
      if (ev.key === "Escape") {
        ev.preventDefault();
        cancel();
      }
    });
    input.addEventListener("blur", () => commit());
  }

  scheduleRenameAfterRender(tag: string): void {
    this._pendingRename = tag;
  }

  private _buildTagRow(li: HTMLLIElement, tag: string): void {
    const m = this.model;
    if (!m.tagHasValuesInView(tag)) li.classList.add("tag-no-values");

    const name = document.createElement("span");
    name.className = "tag-name";
    name.dataset.tag = tag;
    name.textContent = tag;
    name.addEventListener("dblclick", (e) => {
      e.preventDefault();
      e.stopPropagation();
      this.beginRename(tag);
    });
    li.appendChild(name);

    (
      [
        ["X", "axisX"],
        ["Y", "axisY"],
      ] as const
    ).forEach(([lbl, key]) => {
      const btn = document.createElement("button");
      let cls = "axis-btn";
      if (m[key] === tag) cls += " active";
      if (m.viewMode !== "tags") cls += " mode-dimmed";
      btn.className = cls;
      btn.textContent = lbl;
      btn.addEventListener("click", () => this.onAxisChange?.(key, tag));
      li.appendChild(btn);
    });

    const rng = m.filterRanges[tag] || [0, 1];
    const { slider, resetBtn } = this._makeDualRangeSlider(
      rng[0],
      rng[1],
      (range) => this.onFilterChange?.(tag, range),
      () => this.onFilterChange?.(tag, [0, 1]),
    );
    li.appendChild(slider);
    li.appendChild(resetBtn);
  }

  private _makeDualRangeSlider(
    lo: number,
    hi: number,
    onChange: (range: [number, number]) => void,
    onReset: () => void,
  ): { slider: HTMLDivElement; resetBtn: HTMLButtonElement } {
    const wrap = document.createElement("div");
    wrap.className = "range-slider";

    const track = document.createElement("div");
    track.className = "range-track";
    wrap.appendChild(track);

    const fill = document.createElement("div");
    fill.className = "range-fill";
    track.appendChild(fill);

    const sLo = document.createElement("input") as HTMLInputElement;
    const sHi = document.createElement("input") as HTMLInputElement;
    for (const s of [sLo, sHi]) {
      Object.assign(s, { type: "range", min: "0", max: "1", step: "0.01" });
      s.className = "range-thumb";
      wrap.appendChild(s);
    }
    sLo.value = String(lo);
    sHi.value = String(hi);

    const sync = () => {
      const a = Math.min(+sLo.value, +sHi.value);
      const b = Math.max(+sLo.value, +sHi.value);
      fill.style.left = a * 100 + "%";
      fill.style.width = (b - a) * 100 + "%";
      sLo.style.zIndex = +sLo.value > +sHi.value ? "3" : "2";
    };
    sync();

    const fire = () =>
      onChange([
        Math.min(+sLo.value, +sHi.value),
        Math.max(+sLo.value, +sHi.value),
      ]);
    for (const s of [sLo, sHi]) {
      s.addEventListener("pointerdown", () => { this._interacting = true; });
      s.addEventListener("pointerup", () => { this._interacting = false; });
      s.addEventListener("input", () => { sync(); fire(); });
    }

    const resetBtn = document.createElement("button");
    resetBtn.className = "range-reset-btn";
    resetBtn.textContent = "↺";
    resetBtn.title = "Reset filter range";
    resetBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      sLo.value = "0";
      sHi.value = "1";
      sync();
      onReset();
    });

    return { slider: wrap, resetBtn };
  }
}
