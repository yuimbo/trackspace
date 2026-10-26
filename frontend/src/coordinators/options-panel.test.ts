import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Model, previewGain } from "../model";
import { MAX_PERSPECTIVE } from "../lib/camera3d";
import { OptionsPanelBinder } from "./options-panel";

const OPTIONS_HTML = `
  <input id="view-perspective" type="range" min="0" max="0.6" step="0.01" />
  <span id="view-perspective-value"></span>
  <input id="shade-sun-az" type="range" min="0" max="360" step="1" />
  <span id="shade-sun-az-value"></span>
  <input id="shade-sun-el" type="range" min="0" max="90" step="1" />
  <span id="shade-sun-el-value"></span>
  <input id="shade-shadow-edge" type="range" min="0" max="1" step="0.01" />
  <span id="shade-shadow-edge-value"></span>
  <input id="shade-shadow-tone" type="range" min="0" max="1" step="0.01" />
  <span id="shade-shadow-tone-value"></span>
  <input id="hover-preview-toggle" type="checkbox" />
  <input id="hover-preview-volume" type="range" min="0" max="1" step="0.01" />
  <span id="hover-preview-volume-value"></span>
`;

interface Harness {
  model: Model;
  stopPreview: ReturnType<typeof vi.fn>;
  applyPreviewGain: ReturnType<typeof vi.fn>;
  setPerspective: ReturnType<typeof vi.fn>;
  scheduleDraw: ReturnType<typeof vi.fn>;
  binder: OptionsPanelBinder;
}

function makeHarness(): Harness {
  const model = new Model();
  const stopPreview = vi.fn();
  const applyPreviewGain = vi.fn();
  const setPerspective = vi.fn();
  const scheduleDraw = vi.fn();
  const binder = new OptionsPanelBinder({
    model,
    stopPreview,
    applyPreviewGain,
    setPerspective,
    scheduleDraw,
  });
  binder.bind();
  return {
    model,
    stopPreview,
    applyPreviewGain,
    setPerspective,
    scheduleDraw,
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
    expect(applyPreviewGain).toHaveBeenCalledWith(
      previewGain(model.previewVolume),
    );
    expect(el<HTMLInputElement>("view-perspective").value).toBe(
      String(model.view3d.persp),
    );
    expect(el("view-perspective-value").textContent).toBe(
      `${Math.round(model.view3d.persp * 100)}%`,
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
    expect(el("hover-preview-volume-value").textContent).toBe("50%");
  });

  it("drags the perspective slider into the model and the canvas", () => {
    const { model, setPerspective } = makeHarness();
    const $persp = el<HTMLInputElement>("view-perspective");
    $persp.value = "0.25";
    input("view-perspective");
    expect(model.view3d.persp).toBeCloseTo(0.25, 5);
    expect(setPerspective).toHaveBeenCalledWith(0.25);
    expect(el("view-perspective-value").textContent).toBe("25%");
  });

  it("clamps a dragged perspective to [0, MAX_PERSPECTIVE] and persists on change", () => {
    const { model } = makeHarness();
    const $persp = el<HTMLInputElement>("view-perspective");
    $persp.value = String(MAX_PERSPECTIVE + 1);
    change("view-perspective");
    expect(model.view3d.persp).toBe(MAX_PERSPECTIVE);
    expect(JSON.parse(localStorage.getItem("trackspace")!).view3d.persp).toBe(
      MAX_PERSPECTIVE,
    );
  });

  it("binds the cel-shading sliders to the model and redraws", () => {
    const { model, scheduleDraw } = makeHarness();
    // The controls start on the model's shipped defaults.
    expect(el<HTMLInputElement>("shade-sun-az").value).toBe(
      String(model.shading.azimuth),
    );
    expect(el<HTMLInputElement>("shade-shadow-tone").value).toBe(
      String(model.shading.shadowTone),
    );
    el<HTMLInputElement>("shade-sun-az").value = "200";
    el<HTMLInputElement>("shade-sun-el").value = "30";
    el<HTMLInputElement>("shade-shadow-edge").value = "0.3";
    el<HTMLInputElement>("shade-shadow-tone").value = "0.7";
    input("shade-shadow-tone");
    expect(model.shading).toEqual({
      azimuth: 200,
      elevation: 30,
      shadowEdge: 0.3,
      shadowTone: 0.7,
    });
    expect(scheduleDraw).toHaveBeenCalled();
    expect(el("shade-sun-az-value").textContent).toBe("200°");
    expect(el("shade-shadow-edge-value").textContent).toBe("30%");
    expect(el("shade-shadow-tone-value").textContent).toBe("70%");
  });

  it("clamps an out-of-range shading value", () => {
    const { model } = makeHarness();
    el<HTMLInputElement>("shade-sun-el").value = "120";
    change("shade-sun-el");
    expect(model.shading.elevation).toBe(90);
  });

  it("mirrors a live volume change onto the slider and readout", () => {    const { model, binder } = makeHarness();
    model.previewVolume = 0.3;
    binder.syncPreviewVolume();
    expect(el<HTMLInputElement>("hover-preview-volume").value).toBe("0.3");
    expect(el("hover-preview-volume-value").textContent).toBe("30%");
  });

  it("syncs control state from the model", () => {
    const { model, binder } = makeHarness();
    model.previewVolume = 0.3;
    model.hoverPreview = false;
    model.view3d.persp = 0.12;
    model.shading = {
      azimuth: 200,
      elevation: 20,
      shadowEdge: 0.25,
      shadowTone: 0.5,
    };
    binder.sync();
    expect(el<HTMLInputElement>("hover-preview-volume").value).toBe("0.3");
    expect(el<HTMLInputElement>("hover-preview-toggle").checked).toBe(false);
    expect(el<HTMLInputElement>("view-perspective").value).toBe("0.12");
    expect(el("view-perspective-value").textContent).toBe("12%");
    expect(el<HTMLInputElement>("shade-sun-az").value).toBe("200");
    expect(el("shade-sun-el-value").textContent).toBe("20°");
    expect(el("shade-shadow-edge-value").textContent).toBe("25%");
  });
});
