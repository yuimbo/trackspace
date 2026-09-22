// ─── ViewModifierView ───────────────────────────────────────
//
// Renders the corner stack of **active view modifiers** as chips. The stack is
// view state read from `model.viewModifiers`; each chip carries a × that dismisses
// it, a `proximity` / `external` chip exposes its control inline so the modifier
// is adjustable without leaving the map, and a `permanent` chip is a view filter
// that is never dismissed.
//
// The view is a dumb surface: it renders what it is given and reports intent
// through callbacks the controller implements. Chips are **diffed by id**, so a live
// slider drag does not rebuild the input under the pointer.

import type { ViewModifier } from "../model";
import { hotkeyLabel } from "../hotkeys";
import { createKbd } from "../lib/dom";

export class ViewModifierView {
  private $el: HTMLElement;
  private _chips = new Map<string, HTMLElement>();

  /** A chip's × was pressed. */
  onDismiss: ((id: string) => void) | null = null;
  /** A proximity chip's slider moved (live). */
  onProximityInput: ((id: string, percent: number) => void) | null = null;
  /** An external chip's refit toggle flipped. */
  onExternalRefit: ((id: string, checked: boolean) => void) | null = null;
  /** An external chip's label was pressed: toggle showing external tracks. */
  onExternalToggle: ((id: string) => void) | null = null;

  constructor($el: HTMLElement) {
    this.$el = $el;
    this.$el.addEventListener("click", (e) => {
      const target = e.target as HTMLElement;
      const toggle = target.closest<HTMLElement>("[data-modifier-toggle]");
      if (toggle) {
        const id = toggle.closest<HTMLElement>("[data-modifier-id]")?.dataset
          .modifierId;
        if (id) this.onExternalToggle?.(id);
        return;
      }
      const btn = target.closest<HTMLElement>("[data-modifier-dismiss]");
      const id = btn?.closest<HTMLElement>("[data-modifier-id]")?.dataset
        .modifierId;
      if (id) this.onDismiss?.(id);
    });
    this.$el.addEventListener("input", (e) => {
      const range = (e.target as HTMLElement).closest<HTMLInputElement>(
        "[data-modifier-range]",
      );
      const id = range?.closest<HTMLElement>("[data-modifier-id]")?.dataset
        .modifierId;
      if (id && range) this.onProximityInput?.(id, Number(range.value));
    });
    this.$el.addEventListener("change", (e) => {
      const box = (e.target as HTMLElement).closest<HTMLInputElement>(
        "[data-modifier-refit]",
      );
      const id = box?.closest<HTMLElement>("[data-modifier-id]")?.dataset
        .modifierId;
      if (id && box) this.onExternalRefit?.(id, box.checked);
    });
  }

  /** Render the stack, reusing each existing chip by id. */
  render(modifiers: ViewModifier[]): void {
    this.$el.classList.toggle("hidden", modifiers.length === 0);
    const seen = new Set<string>();
    for (const mod of modifiers) {
      seen.add(mod.id);
      let chip = this._chips.get(mod.id);
      if (!chip) {
        chip = this._buildChip(mod);
        this._chips.set(mod.id, chip);
      }
      this._updateChip(chip, mod);
      this.$el.appendChild(chip);
    }
    for (const [id, chip] of this._chips) {
      if (seen.has(id)) continue;
      chip.remove();
      this._chips.delete(id);
    }
  }

  private _buildChip(mod: ViewModifier): HTMLElement {
    const chip = document.createElement("div");
    chip.className = "view-modifier";
    chip.dataset.modifierId = mod.id;
    chip.dataset.modifierKind = mod.kind;

    const label = document.createElement("span");
    label.className = "vm-label";

    if (mod.kind === "external") {
      // The permanent external chip: the label is the control (V toggles it), and
      // the refit checkbox sits on its own line below it (R).
      chip.classList.add("vm-external");
      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "vm-toggle";
      toggle.dataset.modifierToggle = "";
      toggle.title = "Show or hide external-root tracks";
      toggle.appendChild(label);
      const vKey = hotkeyLabel("toggle-external");
      if (vKey) toggle.appendChild(this._hotkey(vKey));
      chip.appendChild(toggle);
    } else {
      chip.appendChild(label);
    }

    const detail = document.createElement("span");
    detail.className = "vm-detail";
    chip.appendChild(detail);

    if (mod.kind === "proximity") {
      const range = document.createElement("input");
      range.type = "range";
      range.min = "0";
      range.max = "100";
      range.step = "1";
      range.className = "vm-range";
      range.dataset.modifierRange = "";
      range.title = "Neighbourhood size: keep the closest (or furthest) tracks";
      chip.appendChild(range);
    }
    if (mod.kind === "external") {
      // Only a *shown* external chip has anything to refit; `_updateChip` hides
      // the row while externals are off the map.
      const wrap = document.createElement("label");
      wrap.className = "vm-refit";
      wrap.title = "Refit the layout with external tracks included";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.dataset.modifierRefit = "";
      wrap.appendChild(box);
      const text = document.createElement("span");
      text.textContent = "Refit";
      wrap.appendChild(text);
      const rKey = hotkeyLabel("toggle-external-refit");
      if (rKey) wrap.appendChild(this._hotkey(rKey));
      chip.appendChild(wrap);
    }

    if (!mod.permanent) {
      const dismiss = document.createElement("button");
      dismiss.type = "button";
      dismiss.className = "vm-dismiss";
      dismiss.dataset.modifierDismiss = "";
      dismiss.title = "Clear this view modifier (Esc clears the newest)";
      dismiss.textContent = "×";
      chip.appendChild(dismiss);
    }
    return chip;
  }

  /** A `<kbd>` hotkey badge on a chip. */
  private _hotkey(label: string): HTMLElement {
    return createKbd(label, "vm-hotkey");
  }

  private _updateChip(chip: HTMLElement, mod: ViewModifier): void {
    (chip.querySelector(".vm-label") as HTMLElement).textContent = mod.label;
    const detail = chip.querySelector(".vm-detail") as HTMLElement;
    detail.textContent = mod.detail ?? "";
    detail.hidden = !mod.detail;
    const range = chip.querySelector<HTMLInputElement>("[data-modifier-range]");
    if (range) range.value = String(mod.value ?? 100);
    const box = chip.querySelector<HTMLInputElement>("[data-modifier-refit]");
    if (box) {
      box.checked = Boolean(mod.checked);
      box.closest<HTMLElement>(".vm-refit")!.hidden = mod.checked === undefined;
    }
  }
}
