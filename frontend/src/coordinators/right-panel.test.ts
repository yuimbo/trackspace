import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { Model } from "../model";
import { RightPanelBinder } from "./right-panel";

const PANEL_HTML = `
  <aside id="right-panel">
    <section id="search-section"></section>
    <section id="properties-section"></section>
    <nav id="rp-tabs">
      <button type="button" class="rp-tab" data-rp-tab="analysis"></button>
      <button type="button" class="rp-tab" data-rp-tab="organise"></button>
      <button type="button" class="rp-tab" data-rp-tab="tracks"></button>
      <button type="button" class="rp-tab" data-rp-tab="view"></button>
    </nav>
    <section id="analysis-section" data-rp-panel="analysis"></section>
    <section id="head-section" data-rp-panel="organise"></section>
    <section id="batch-section" data-rp-panel="tracks"></section>
    <section id="options-section" data-rp-panel="view"></section>
  </aside>
`;

function panel(name: string): HTMLElement {
  return document.querySelector<HTMLElement>(`[data-rp-panel="${name}"]`)!;
}

function tab(name: string): HTMLElement {
  return document.querySelector<HTMLElement>(`[data-rp-tab="${name}"]`)!;
}

beforeEach(() => {
  document.body.innerHTML = PANEL_HTML;
  localStorage.clear();
});

afterEach(() => {
  localStorage.clear();
});

describe("RightPanelBinder", () => {
  it("shows the model's tab on bind and hides the others", () => {
    const model = new Model();
    model.rightPanelTab = "tracks";
    new RightPanelBinder({ model }).bind();

    expect(panel("tracks").hidden).toBe(false);
    expect(panel("analysis").hidden).toBe(true);
    expect(panel("organise").hidden).toBe(true);
    expect(panel("view").hidden).toBe(true);
    expect(tab("tracks").classList.contains("active")).toBe(true);
    expect(tab("tracks").getAttribute("aria-selected")).toBe("true");
  });

  it("switches tabs on click and persists the choice", () => {
    const model = new Model();
    const binder = new RightPanelBinder({ model });
    binder.bind();

    tab("view").click();

    expect(model.rightPanelTab).toBe("view");
    expect(panel("view").hidden).toBe(false);
    expect(panel("analysis").hidden).toBe(true);
    expect(tab("view").classList.contains("active")).toBe(true);
    expect(tab("analysis").classList.contains("active")).toBe(false);

    // Persisted, so a fresh model reads the same tab back.
    const reloaded = new Model();
    reloaded.loadLS();
    expect(reloaded.rightPanelTab).toBe("view");
  });

  it("ignores an unknown persisted tab", () => {
    localStorage.setItem(
      "trackspace",
      JSON.stringify({ rightPanelTab: "bogus" }),
    );
    const model = new Model();
    model.loadLS();
    expect(model.rightPanelTab).toBe("analysis");
  });
});
