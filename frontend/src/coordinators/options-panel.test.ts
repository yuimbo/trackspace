import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Model, previewGain } from "../model";
import { OptionsPanelBinder } from "./options-panel";

const OPTIONS_HTML = `
  <input id="hover-preview-toggle" type="checkbox" />
  <input id="hover-preview-volume" type="range" min="0" max="1" step="0.01" />
  <input id="view-mode-toggle" type="checkbox" />
  <input id="explode-strength-range" type="range" min="0.05" max="0.2" step="0.01" />
`;

interface Harness {
  model: Model;
  scheduleDraw: ReturnType<typeof vi.fn>;
  toggleViewMode: ReturnType<typeof vi.fn>;
  stopPreview: ReturnType<typeof vi.fn>;
  applyPreviewGain: ReturnType<typeof vi.fn>;
  binder: OptionsPanelBinder;
}

function makeHarness(): Harness {
  const model = new Model();
  const scheduleDraw = vi.fn();
  const toggleViewMode = vi.fn();
  const stopPreview = vi.fn();
  const applyPreviewGain = vi.fn();
  const binder = new OptionsPanelBinder({
    model,
    scheduleDraw,
    toggleViewMode,
    stopPreview,
    applyPreviewGain,
  });
  binder.bind();
  return {
    model,
    scheduleDraw,
    toggleViewMode,
    stopPreview,
    applyPreviewGain,
    binder,
  };
}

const el = <T extends HTMLElement>(id: string) =>
  document.getElementById(id) as T;

function change(id: string): void {
  el(id).dispatchEvent(new Event("change", { bubbles: true }));
}

function input(id: string): void {
  el(id).dispatchEvent(new Event("input", { bubbles: true }));
}

beforeEach(() => {
  document.body.innerHTML = OPTIONS_HTML;
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("OptionsPanelBinder", () => {
  it("reflects the model into the controls on bind", () => {
    const { model, applyPreviewGain } = makeHarness();
    expect(el<HTMLInputElement>("hover-preview-toggle").checked).toBe(
      model.hoverPreview,
    );
    expect(el<HTMLInputElement>("view-mode-toggle").checked).toBe(false);
    expect(applyPreviewGain).toHaveBeenCalledWith(
      previewGain(model.previewVolume),
    );
  });

  it("toggles hover preview and stops playback when disabled", () => {
    const { model, stopPreview } = makeHarness();
    const $hov = el<HTMLInputElement>("hover-preview-toggle");
    model.hoverPreview = true;
    $hov.checked = false;
    change("hover-preview-toggle");
    expect(model.hoverPreview).toBe(false);
    expect(stopPreview).toHaveBeenCalled();
  });

  it("applies the perceptual taper of the preview volume", () => {
    const { model, applyPreviewGain } = makeHarness();
    const $pv = el<HTMLInputElement>("hover-preview-volume");
    $pv.value = "0.5";
    input("hover-preview-volume");
    expect(model.previewVolume).toBeCloseTo(0.5, 5);
    // Half the slider is a quarter of full scale.
    expect(applyPreviewGain).toHaveBeenCalledWith(0.25);
  });

  it("delegates the view-mode toggle", () => {
    const { toggleViewMode } = makeHarness();
    change("view-mode-toggle");
    expect(toggleViewMode).toHaveBeenCalledTimes(1);
  });

  it("updates explode strength and repaints", () => {
    const { model, scheduleDraw } = makeHarness();
    const $ex = el<HTMLInputElement>("explode-strength-range");
    $ex.value = "0.15";
    input("explode-strength-range");
    expect(model.explodeStrength).toBeCloseTo(0.15, 5);
    expect(scheduleDraw).toHaveBeenCalled();
  });

  it("clamps an out-of-range explode strength", () => {
    const { model } = makeHarness();
    const $ex = el<HTMLInputElement>("explode-strength-range");
    $ex.value = "9";
    input("explode-strength-range");
    expect(model.explodeStrength).toBe(0.2);
  });

  it("syncs control state from the model", () => {
    const { model, binder } = makeHarness();
    model.viewMode = "embeddings";
    model.explodeStrength = 0.12;
    model.previewVolume = 0.3;
    model.hoverPreview = false;
    binder.sync();
    expect(el<HTMLInputElement>("view-mode-toggle").checked).toBe(true);
    expect(el<HTMLInputElement>("explode-strength-range").value).toBe("0.12");
    expect(el<HTMLInputElement>("hover-preview-volume").value).toBe("0.3");
    expect(el<HTMLInputElement>("hover-preview-toggle").checked).toBe(false);
  });
});
