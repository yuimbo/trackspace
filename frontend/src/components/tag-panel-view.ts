import type { Model } from "../model";


export class TagPanelView {
  model: Model;
  $el: HTMLElement;

  onAxisChange: ((which: string, tag: string) => void) | null = null;
  onFilterChange: ((tag: string, range: [number, number]) => void) | null =
    null;

  constructor(model: Model, container: HTMLElement) {
    this.model = model;
    this.$el = container;
  }

  render(): void {
    this.$el.innerHTML = "";
    const m = this.model;
    for (const tag of m.tags) {
      const li = document.createElement("li");
      this._buildTagRow(li, tag);
      this.$el.appendChild(li);
    }
  }

  private _buildTagRow(li: HTMLLIElement, tag: string): void {
    const m = this.model;
    if (!m.tagHasValuesInView(tag)) li.classList.add("tag-no-values");

    const name = document.createElement("span");
    name.className = "tag-name";
    name.textContent = tag;
    li.appendChild(name);

    (
      [
        ["X", "axisX"],
        ["Y", "axisY"],
      ] as const
    ).forEach(([lbl, key]) => {
      const btn = document.createElement("button");
      btn.className = "axis-btn" + (m[key] === tag ? " active" : "");
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
    sLo.addEventListener("input", () => {
      sync();
      fire();
    });
    sHi.addEventListener("input", () => {
      sync();
      fire();
    });

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
