// ─── OptionsPanelBinder ────────────────────────────────────────
//
// Owns the status-bar **hover preview** control (toggle + volume). The lens and
// colour-mode controls were removed: the persistent top tabbers are their single
// surface, so this binder no longer mirrors a lens `<select>` or a preset UI.
//
// Deps are structural interfaces so the binder is unit-testable without the
// DOM-heavy component classes.

import { previewGain, type Model } from "../model";
import { MAX_PERSPECTIVE } from "../lib/camera3d";
import { clamp } from "../lib/util";

export interface OptionsPanelDeps {
  model: Model;
  /** Stop hover-preview playback (when the toggle is turned off). */
  stopPreview(): void;
  /** Apply a new preview gain (already tapered) to the audio element. */
  applyPreviewGain(gain: number): void;
  /** Apply a new 3-D perspective strength to the canvas. */
  setPerspective(persp: number): void;
  /** Redraw the canvas (after a shading tweak). */
  scheduleDraw(): void;
}

/**
 * Owns the View-options 3-D controls (perspective strength, cel shading) and the
 * status-bar hover-preview control (toggle + volume). The lens and colour-mode
 * controls were removed: the persistent top tabbers are their single surface.
 *
 * Deps are structural interfaces so the binder is unit-testable without the
 * DOM-heavy component classes.
 */
export class OptionsPanelBinder {
  constructor(private deps: OptionsPanelDeps) {}

  bind(): void {
    const m = this.deps.model;
    const $persp = document.getElementById(
      "view-perspective",
    ) as HTMLInputElement | null;
    if ($persp) {
      $persp.value = String(m.view3d.persp);
      this._renderPerspective();
      $persp.addEventListener("input", () => {
        m.view3d.persp = clamp(
          parseFloat($persp.value) || 0,
          0,
          MAX_PERSPECTIVE,
        );
        this._renderPerspective();
        this.deps.setPerspective(m.view3d.persp);
      });
      $persp.addEventListener("change", () => {
        m.view3d.persp = clamp(
          parseFloat($persp.value) || 0,
          0,
          MAX_PERSPECTIVE,
        );
        m.saveLS();
      });
    }

    this._bindShading();

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
    const $persp = document.getElementById(
      "view-perspective",
    ) as HTMLInputElement | null;
    if ($persp) $persp.value = String(m.view3d.persp);
    this._renderPerspective();
    this._writeShading();
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

  /** Show the 3-D perspective strength as a percentage (0 % = orthographic). */
  private _renderPerspective(): void {
    const out = document.getElementById("view-perspective-value");
    if (!out) return;
    out.textContent = `${Math.round(this.deps.model.view3d.persp * 100)}%`;
  }

  /**
   * The cel-shading sliders. Each entry is `[element id, model.shading field,
   * clamp ceiling]`; one list drives bind, sync and the readouts so the three
   * cannot drift. Values are session-only (the model field is transient).
   */
  private _bindShading(): void {
    const m = this.deps.model;
    const sliders: [HTMLInputElement, keyof typeof m.shading, number][] = [];
    for (const [id, field, max] of SHADING_SLIDERS) {
      const el = document.getElementById(id) as HTMLInputElement | null;
      // Partial markup (tests, a stripped page): leave all four unbound rather
      // than half-wire the panel.
      if (!el) return;
      sliders.push([el, field, max]);
    }
    this._writeShading();
    const apply = (): void => {
      for (const [el, field, max] of sliders) {
        m.shading[field] = clamp(parseFloat(el.value) || 0, 0, max);
      }
      this._writeShading();
      this.deps.scheduleDraw();
    };
    for (const [el] of sliders) {
      el.addEventListener("input", apply);
      el.addEventListener("change", apply);
    }
  }

  /** Mirror `model.shading` onto the sliders and their readouts. */
  private _writeShading(): void {
    const shading = this.deps.model.shading;
    for (const [id, field] of SHADING_SLIDERS) {
      const el = document.getElementById(id) as HTMLInputElement | null;
      if (el) el.value = String(shading[field]);
      const out = document.getElementById(`${id}-value`);
      if (!out) continue;
      out.textContent =
        field === "azimuth" || field === "elevation"
          ? `${Math.round(shading[field])}°`
          : `${Math.round(shading[field] * 100)}%`;
    }
  }
}

/** id → model field → clamp ceiling, for the four cel-shading sliders. */
const SHADING_SLIDERS = [
  ["shade-sun-az", "azimuth", 360],
  ["shade-sun-el", "elevation", 90],
  ["shade-shadow-edge", "shadowEdge", 1],
  ["shade-shadow-tone", "shadowTone", 1],
] as const;
