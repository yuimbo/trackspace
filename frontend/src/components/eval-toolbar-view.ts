// ─── EvalToolbarView ─────────────────────────────────────────
//
// The compact toolbar overlay for the layout-evaluation gallery, rendered inside
// `#main` on top of the canvas. It is a dumb surface: it renders the state it is
// given and reports intent through callbacks the coordinator implements — the view
// never mutates the model or issues requests.

import { emptyRatings, type EvalRatings, type RatingCategory } from "../lib/eval-ratings";

/** One map entry in the select (blind letter until revealed). */
export interface EvalMapOption {
  index: number;
  letter: string;
  /** Algorithm/recipe label; only surfaced once the user reveals names. */
  label?: string;
  /** How many of the four categories already have a 1–5 score. */
  rated: number;
  revealed: boolean;
}

/** Everything the toolbar renders for one frame. */
export interface EvalToolbarState {
  busy: boolean;
  currentIndex: number;
  count: number;
  maps: EvalMapOption[];
  /** True once recipe names + timings are revealed. */
  revealed: boolean;
  /** Human-readable timing/recipe readout for the selected map (revealed only). */
  currentTiming: string;
  /** Eligible artifact tracks present in the loaded library. */
  eligible: number;
  /** Total artifact tracks (the map's own set). */
  total: number;
  /** Artifact tracks absent from the loaded library. */
  unavailable: number;
  ratings: EvalRatings | null;
  note: string;
  /** Short dataset identity, e.g. "dataset 1a2b3c4d". */
  datasetLabel: string;
  error: string | null;
}

export class EvalToolbarView {
  /** A map was selected in the dropdown (0-based). */
  onSelect: ((index: number) => void) | null = null;
  onPrev: (() => void) | null = null;
  onNext: (() => void) | null = null;
  onResetCamera: (() => void) | null = null;
  onToggleReveal: ((checked: boolean) => void) | null = null;
  onRate: ((category: RatingCategory, value: number | null) => void) | null = null;
  onNote: ((text: string) => void) | null = null;
  onExport: (() => void) | null = null;
  onExit: (() => void) | null = null;

  private $el: HTMLElement;

  constructor(el: HTMLElement) {
    this.$el = el;
    this._build();
    this._wire();
  }

  private _build(): void {
    this.$el.innerHTML = `
      <div class="ev-head">
        <span class="ev-title">Layout evaluation</span>
        <span class="ev-dataset" data-role="dataset"></span>
        <button type="button" class="ev-link" data-role="exit" title="Leave the eval gallery and return to normal mode">Exit ↺</button>
      </div>
      <div class="ev-row">
        <button type="button" class="ev-btn" data-role="prev" title="Previous map">‹</button>
        <select class="ev-select" data-role="select" aria-label="Select map"></select>
        <button type="button" class="ev-btn" data-role="next" title="Next map">›</button>
        <button type="button" class="ev-btn" data-role="reset" title="Reset the camera to fit the whole map">Reset view</button>
        <label class="ev-reveal" title="Show each map's recipe and its fit timings">
          <input type="checkbox" data-role="reveal" />
          <span>Reveal recipe &amp; timings</span>
        </label>
      </div>
      <div class="ev-status" data-role="status"></div>
      <div class="ev-ratings">
        <div class="ev-ratings-head">Ratings</div>
        <div data-role="ratings"></div>
        <label class="ev-note">
          <span>Note</span>
          <textarea data-role="note" rows="2" spellcheck="false" placeholder="Anything worth remembering about this map…"></textarea>
        </label>
        <button type="button" class="ev-btn ev-export" data-role="export" title="Download this dataset's ratings as JSON">Export ratings</button>
      </div>
      <div class="ev-error hidden" data-role="error"></div>
    `;
  }

  private _wire(): void {
    const $ = (sel: string) => this.$el.querySelector<HTMLElement>(sel)!;
    $('[data-role="prev"]').addEventListener("click", () => this.onPrev?.());
    $('[data-role="next"]').addEventListener("click", () => this.onNext?.());
    $('[data-role="reset"]').addEventListener("click", () => this.onResetCamera?.());
    $('[data-role="exit"]').addEventListener("click", () => this.onExit?.());
    $('[data-role="export"]').addEventListener("click", () => this.onExport?.());
    const select = $('[data-role="select"]') as HTMLSelectElement;
    select.addEventListener("change", () => {
      const idx = Number(select.value);
      if (Number.isInteger(idx)) this.onSelect?.(idx);
    });
    const reveal = $('[data-role="reveal"]') as HTMLInputElement;
    reveal.addEventListener("change", () => this.onToggleReveal?.(reveal.checked));
    const note = $('[data-role="note"]') as HTMLTextAreaElement;
    note.addEventListener("input", () => this.onNote?.(note.value));
  }

  /** Rebuild the whole toolbar from *state*; shows the toolbar on first render. */
  render(state: EvalToolbarState): void {
    this.$el.classList.remove("hidden");
    const $ = (sel: string) => this.$el.querySelector<HTMLElement>(sel)!;
    this.$el.classList.toggle("ev-busy", state.busy);

    $('[data-role="dataset"]').textContent = state.datasetLabel;

    const select = $('[data-role="select"]') as HTMLSelectElement;
    const prevSelected = Number(select.value);
    select.innerHTML = "";
    for (const m of state.maps) {
      const opt = document.createElement("option");
      opt.value = String(m.index);
      const rating = m.rated > 0 ? ` · ${m.rated}/4 rated` : "";
      opt.textContent = m.revealed && m.label
        ? `${m.letter}. ${m.label}${rating}`
        : `${m.letter}${rating}`;
      select.appendChild(opt);
    }
    if (Number.isInteger(prevSelected) && prevSelected < state.maps.length) {
      select.value = String(prevSelected);
    } else {
      select.value = String(state.currentIndex);
    }
    ($('[data-role="prev"]') as HTMLButtonElement).disabled = state.count <= 1;
    ($('[data-role="next"]') as HTMLButtonElement).disabled = state.count <= 1;

    (($('[data-role="reveal"]') as HTMLInputElement).checked = state.revealed);

    $('[data-role="status"]').textContent =
      `${state.eligible} / ${state.total} artifact tracks in library` +
      (state.unavailable > 0 ? ` · ${state.unavailable} unavailable` : "") +
      (state.revealed && state.currentTiming ? ` — ${state.currentTiming}` : "");

    this._renderRatings($('[data-role="ratings"]') as HTMLElement, state);

    const note = $('[data-role="note"]') as HTMLTextAreaElement;
    // Don't clobber in-progress typing: the note is debounced and saved, so a
    // mid-typing render (e.g. a rating click) must not reset the field.
    if (document.activeElement !== note && note.value !== state.note) {
      note.value = state.note;
    }

    const err = $('[data-role="error"]') as HTMLElement;
    err.classList.toggle("hidden", !state.error);
    err.textContent = state.error ?? "";
  }

  private _renderRatings($container: HTMLElement, state: EvalToolbarState): void {
    const r = state.ratings ?? emptyRatings();
    const cats: { key: RatingCategory; label: string }[] = [
      { key: "neighbours", label: "Musical neighbours" },
      { key: "organisation", label: "Global organisation" },
      { key: "crowding", label: "Crowding" },
      { key: "overall", label: "Overall" },
    ];
    $container.innerHTML = "";
    for (const c of cats) {
      const row = document.createElement("div");
      row.className = "ev-cat";
      const label = document.createElement("span");
      label.className = "ev-cat-label";
      label.textContent = c.label;
      const value = r[c.key];
      label.title = value != null ? `${value} / 5` : "Not rated";
      row.appendChild(label);

      const buttons = document.createElement("div");
      buttons.className = "ev-stars";
      for (let n = 1; n <= 5; n++) {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = String(n);
        b.className = value === n ? "on" : "";
        b.title = `${n} / 5`;
        b.addEventListener("click", () => {
          // Clicking the current value clears it (a quick re-rate).
          this.onRate?.(c.key, value === n ? null : n);
        });
        buttons.appendChild(b);
      }
      row.appendChild(buttons);
      $container.appendChild(row);
    }
  }
}
