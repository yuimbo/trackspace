// ─── OptionsPanelBinder ────────────────────────────────────────
//
// Owns the status-bar **hover preview** control (toggle + volume). The lens and
// colour-mode controls were removed: the persistent top tabbers are their single
// surface, so this binder no longer mirrors a lens `<select>` or a preset UI.
//
// Deps are structural interfaces so the binder is unit-testable without the
// DOM-heavy component classes.

import { previewGain, type Model } from "../model";

const clamp = (v: number, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));

export interface OptionsPanelDeps {
  model: Model;
  /** Stop hover-preview playback (when the toggle is turned off). */
  stopPreview(): void;
  /** Apply a new preview gain (already tapered) to the audio element. */
  applyPreviewGain(gain: number): void;
}

export class OptionsPanelBinder {
  constructor(private deps: OptionsPanelDeps) {}

  bind(): void {
    const m = this.deps.model;
    const $hov = document.getElementById(
      "hover-preview-toggle",
    ) as HTMLInputElement | null;
    if ($hov) {
      $hov.checked = m.hoverPreview;
      $hov.addEventListener("change", () => {
        m.hoverPreview = $hov.checked;
        if (!m.hoverPreview) this.deps.stopPreview();
        m.saveLS();
      });
    }

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
  }

  /** Sync the hover-preview controls to match current model state. */
  sync(): void {
    const m = this.deps.model;
    const $pv = document.getElementById("hover-preview-volume") as HTMLInputElement | null;
    if ($pv) $pv.value = String(m.previewVolume);
    this._renderPreviewVolume();
    const $hov = document.getElementById("hover-preview-toggle") as HTMLInputElement | null;
    if ($hov) $hov.checked = m.hoverPreview;
  }

  /** Mirror the preview volume onto the slider + readout (live M + scroll). */
  syncPreviewVolume(): void {
    const $pv = document.getElementById("hover-preview-volume") as HTMLInputElement | null;
    if ($pv) $pv.value = String(this.deps.model.previewVolume);
    this._renderPreviewVolume();
  }

  /** Show the slider position as a percentage. */
  private _renderPreviewVolume(): void {
    const out = document.getElementById("hover-preview-volume-value");
    if (!out) return;
    const pct = Math.round(this.deps.model.previewVolume * 100);
    out.textContent = `${pct}%`;
  }
}
