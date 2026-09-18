// ─── OptionsPanelBinder ────────────────────────────────────────
//
// Owns the right-column Options panel controls (hover preview + volume, lens
// selector and presets, explode strength). Extracted from `Controller` — these are
// pure input bindings that mirror model state and delegate the actions that need
// other subsystems (lens switching, preset capture, preview playback).
//
// Deps are structural interfaces so the binder is unit-testable without the
// DOM-heavy component classes.

import { DEFAULT_PREVIEW_VOLUME, previewGain, type Model } from "../model";
import { BUILTIN_LENSES } from "../lib/lenses";

const clamp = (v: number, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));

export interface OptionsPanelDeps {
  model: Model;
  scheduleDraw(): void;
  /** Switch to a lens id (built-in or preset). */
  selectLens(id: string): void;
  /** Capture the current lens + display options as a new named preset. */
  saveLensPreset(): void;
  /** Delete the active preset; no-op when a built-in lens is active. */
  deleteLensPreset(): void;
  /** Stop hover-preview playback (when the toggle is turned off). */
  stopPreview(): void;
  /** Apply a new preview gain (already tapered) to the audio element. */
  applyPreviewGain(gain: number): void;
}

export class OptionsPanelBinder {
  constructor(private deps: OptionsPanelDeps) {}

  bind(): void {
    const m = this.deps.model;
    this._renderLensOptions();
    const $hov = document.getElementById(
      "hover-preview-toggle",
    ) as HTMLInputElement;

    $hov.checked = m.hoverPreview;
    $hov.addEventListener("change", () => {
      m.hoverPreview = $hov.checked;
      if (!m.hoverPreview) this.deps.stopPreview();
      m.saveLS();
    });

    const $pv = document.getElementById("hover-preview-volume") as HTMLInputElement | null;
    if ($pv) {
      $pv.value = String(m.previewVolume);
      this.deps.applyPreviewGain(previewGain(m.previewVolume));
      this._renderPreviewVolume();
      $pv.addEventListener("input", () => {
        m.previewVolume = clamp(parseFloat($pv.value) || 0, 0, 1);
        this._renderPreviewVolume();
        this.deps.applyPreviewGain(previewGain(m.previewVolume));
      });
      $pv.addEventListener("change", () => {
        m.previewVolume = clamp(parseFloat($pv.value) || 0, 0, 1);
        m.saveLS();
      });
    }

    const $lens = document.getElementById("lens-select") as HTMLSelectElement | null;
    if ($lens) $lens.addEventListener("change", () => this.deps.selectLens($lens.value));
    const $save = document.getElementById("lens-preset-save");
    if ($save) $save.addEventListener("click", () => this.deps.saveLensPreset());
    const $del = document.getElementById("lens-preset-delete");
    if ($del) $del.addEventListener("click", () => this.deps.deleteLensPreset());

    const $ex = document.getElementById("explode-strength-range") as HTMLInputElement | null;
    if ($ex) {
      $ex.value = String(m.explodeStrength);
      $ex.addEventListener("input", () => {
        m.explodeStrength = clamp(parseFloat($ex.value) || 0.1, 0.05, 0.2);
        this.deps.scheduleDraw();
      });
      $ex.addEventListener("change", () => {
        m.explodeStrength = clamp(parseFloat($ex.value) || 0.1, 0.05, 0.2);
        m.saveLS();
      });
    }
  }

  /** Sync all options controls to match current model state. */
  sync(): void {
    const m = this.deps.model;
    this._renderLensOptions();
    const $ex = document.getElementById("explode-strength-range") as HTMLInputElement | null;
    if ($ex) $ex.value = String(Math.round(m.explodeStrength * 100) / 100);
    const $pv = document.getElementById("hover-preview-volume") as HTMLInputElement | null;
    if ($pv) $pv.value = String(m.previewVolume);
    this._renderPreviewVolume();
    const $hov = document.getElementById("hover-preview-toggle") as HTMLInputElement | null;
    if ($hov) $hov.checked = m.hoverPreview;
  }

  /** Rebuild the lens `<select>` from the built-ins plus the saved presets. */
  private _renderLensOptions(): void {
    const m = this.deps.model;
    const sel = document.getElementById("lens-select") as HTMLSelectElement | null;
    if (sel) {
      // Built with DOM APIs, not innerHTML: a preset name is user input.
      sel.replaceChildren();
      const groups: [string, { id: string; label: string }[]][] = [
        ["Lenses", BUILTIN_LENSES.map((l) => ({ id: l.id, label: l.label }))],
      ];
      if (m.lensPresets.length) {
        groups.push([
          "Presets",
          m.lensPresets.map((p) => ({ id: p.id, label: p.name })),
        ]);
      }
      for (const [label, items] of groups) {
        const og = document.createElement("optgroup");
        og.label = label;
        for (const item of items) {
          const opt = document.createElement("option");
          opt.value = item.id;
          opt.textContent = item.label;
          og.appendChild(opt);
        }
        sel.appendChild(og);
      }
      sel.value = m.lensId;
    }
    const del = document.getElementById(
      "lens-preset-delete",
    ) as HTMLButtonElement | null;
    if (del) del.disabled = !m.lensPresets.some((p) => p.id === m.lensId);
    const save = document.getElementById(
      "lens-preset-save",
    ) as HTMLButtonElement | null;
    // The tag lens has no map display options to capture.
    if (save) save.disabled = m.lens().kind === "tags";
  }

  /** Show the slider position as a percentage. */
  private _renderPreviewVolume(): void {
    const out = document.getElementById("hover-preview-volume-value");
    if (!out) return;
    const pct = Math.round(this.deps.model.previewVolume * 100);
    out.textContent = `${pct}%`;
  }
}
