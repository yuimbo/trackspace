import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Model, previewGain } from "../model";
import { OptionsPanelBinder } from "./options-panel";

const OPTIONS_HTML = `
  <input id="hover-preview-toggle" type="checkbox" />
  <input id="hover-preview-volume" type="range" min="0" max="1" step="0.01" />
  <select id="lens-select"></select>
  <button id="lens-preset-save" type="button"></button>
  <button id="lens-preset-delete" type="button"></button>
`;

interface Harness {
  model: Model;
  selectLens: ReturnType<typeof vi.fn>;
  saveLensPreset: ReturnType<typeof vi.fn>;
  deleteLensPreset: ReturnType<typeof vi.fn>;
  stopPreview: ReturnType<typeof vi.fn>;
  applyPreviewGain: ReturnType<typeof vi.fn>;
  binder: OptionsPanelBinder;
}

function makeHarness(): Harness {
  const model = new Model();
  const selectLens = vi.fn();
  const saveLensPreset = vi.fn();
  const deleteLensPreset = vi.fn();
  const stopPreview = vi.fn();
  const applyPreviewGain = vi.fn();
  const binder = new OptionsPanelBinder({
    model,
    selectLens,
    saveLensPreset,
    deleteLensPreset,
    stopPreview,
    applyPreviewGain,
  });
  binder.bind();
  return {
    model,
    selectLens,
    saveLensPreset,
    deleteLensPreset,
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

function click(id: string): void {
  el(id).dispatchEvent(new MouseEvent("click", { bubbles: true }));
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
    expect(el<HTMLSelectElement>("lens-select").value).toBe(model.lensId);
    expect(applyPreviewGain).toHaveBeenCalledWith(
      previewGain(model.previewVolume),
    );
  });

  it("lists the built-in lenses and the saved presets", () => {
    const { model, binder } = makeHarness();
    const sel = el<HTMLSelectElement>("lens-select");
    expect([...sel.options].map((o) => o.value)).toContain("combined");
    expect([...sel.options].map((o) => o.value)).toContain("camelot");
    model.lensPresets = [
      {
        id: "preset:1",
        name: "My view",
        baseLens: "combined",
        weights: model.clusterWeights,
        colorByCluster: false,
        clusterResolution: 1.5,
      },
    ];
    binder.sync();
    expect([...sel.options].map((o) => o.value)).toContain("preset:1");
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

  it("delegates the lens selector", () => {
    const { selectLens } = makeHarness();
    const sel = el<HTMLSelectElement>("lens-select");
    sel.value = "clap";
    change("lens-select");
    expect(selectLens).toHaveBeenCalledWith("clap");
  });

  it("delegates preset save and delete", () => {
    const { model, binder, saveLensPreset, deleteLensPreset } = makeHarness();
    // Save is disabled on the tag lens (no map options to capture).
    model.lensId = "clap";
    binder.sync();
    click("lens-preset-save");
    expect(saveLensPreset).toHaveBeenCalledTimes(1);
    // Delete is disabled while a built-in lens is active, so enable a preset.
    model.lensPresets = [
      {
        id: "preset:1",
        name: "My view",
        baseLens: "combined",
        weights: model.clusterWeights,
        colorByCluster: false,
        clusterResolution: 1.5,
      },
    ];
    model.lensId = "preset:1";
    const del = el<HTMLButtonElement>("lens-preset-delete");
    del.disabled = false;
    click("lens-preset-delete");
    expect(deleteLensPreset).toHaveBeenCalledTimes(1);
  });

  it("enables preset delete only for an active preset", () => {
    const { model, binder } = makeHarness();
    const del = el<HTMLButtonElement>("lens-preset-delete");
    expect(del.disabled).toBe(true);
    model.lensPresets = [
      {
        id: "preset:1",
        name: "My view",
        baseLens: "combined",
        weights: model.clusterWeights,
        colorByCluster: false,
        clusterResolution: 1.5,
      },
    ];
    model.lensId = "preset:1";
    binder.sync();
    expect(del.disabled).toBe(false);
  });

  it("syncs control state from the model", () => {
    const { model, binder } = makeHarness();
    model.lensId = "clap";
    model.previewVolume = 0.3;
    model.hoverPreview = false;
    binder.sync();
    expect(el<HTMLSelectElement>("lens-select").value).toBe("clap");
    expect(el<HTMLInputElement>("hover-preview-volume").value).toBe("0.3");
    expect(el<HTMLInputElement>("hover-preview-toggle").checked).toBe(false);
  });
});
