// ─── OptionsPanelBinder ────────────────────────────────────────
//
// Owns the right-column Options panel controls (hover preview + volume, embedding
// view mode, explode strength). Extracted from `Controller` — these are pure
// input bindings that mirror model state and delegate the two actions that need
// other subsystems (view-mode toggle, preview playback).
//
// Deps are structural interfaces so the binder is unit-testable without the
// DOM-heavy component classes.

import { type Model } from "../model";

const clamp = (v: number, lo = 0, hi = 1) => Math.max(lo, Math.min(hi, v));

export interface OptionsPanelDeps {
  model: Model;
  scheduleDraw(): void;
  /** Flip between tags and embedding-space view. */
  toggleViewMode(): void;
  /** Stop hover-preview playback (when the toggle is turned off). */
  stopPreview(): void;
  /** Apply a new preview volume to the audio element. */
  applyPreviewVolume(volume: number): void;
}

export class OptionsPanelBinder {
  constructor(private deps: OptionsPanelDeps) {}

  bind(): void {
    const m = this.deps.model;
    const $hov = document.getElementById(
      "hover-preview-toggle",
    ) as HTMLInputElement;
    const $vm = document.getElementById(
      "view-mode-toggle",
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
      this.deps.applyPreviewVolume(m.previewVolume);
      $pv.addEventListener("input", () => {
        m.previewVolume = clamp(parseFloat($pv.value) || 0, 0, 1);
        this.deps.applyPreviewVolume(m.previewVolume);
      });
      $pv.addEventListener("change", () => {
        m.previewVolume = clamp(parseFloat($pv.value) || 0, 0, 1);
        m.saveLS();
      });
    }

    $vm.checked = m.viewMode === "embeddings";
    $vm.addEventListener("change", () => this.deps.toggleViewMode());

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
    const isEmbed = m.viewMode === "embeddings";
    const $vm = document.getElementById("view-mode-toggle") as HTMLInputElement | null;
    if ($vm) $vm.checked = isEmbed;
    const $ex = document.getElementById("explode-strength-range") as HTMLInputElement | null;
    if ($ex) $ex.value = String(Math.round(m.explodeStrength * 100) / 100);
    const $pv = document.getElementById("hover-preview-volume") as HTMLInputElement | null;
    if ($pv) $pv.value = String(m.previewVolume);
    const $hov = document.getElementById("hover-preview-toggle") as HTMLInputElement | null;
    if ($hov) $hov.checked = m.hoverPreview;
  }
}
