import { describe, expect, it, vi } from "vitest";
import { Model, type ViewModifier } from "../model";
import { ViewModifierCoordinator } from "./view-modifier-coordinator";

function makeHarness() {
  const model = new Model();
  const render = vi.fn<(mods: ViewModifier[]) => void>();
  const coord = new ViewModifierCoordinator({ model, view: { render } });
  return { model, render, coord };
}

describe("ViewModifierCoordinator", () => {
  it("pushes a modifier on top of the stack and renders it", () => {
    const { coord, model, render } = makeHarness();
    const id = coord.push({ kind: "search", label: "techno" }, () => {});
    expect(model.viewModifiers).toEqual([
      { id, kind: "search", label: "techno" },
    ]);
    expect(render).toHaveBeenLastCalledWith(model.viewModifiers);
  });

  it("keeps push order oldest → newest", () => {
    const { coord, model } = makeHarness();
    coord.push({ kind: "search", label: "a" }, () => {});
    coord.push({ kind: "scope", label: "Camelot" }, () => {});
    expect(model.viewModifiers.map((m) => m.kind)).toEqual(["search", "scope"]);
  });

  it("updates a modifier's readout in place", () => {
    const { coord, model } = makeHarness();
    const id = coord.push({ kind: "proximity", label: "a.mp3" }, () => {});
    coord.update(id, { detail: "nearest 40%", value: 40 });
    expect(model.viewModifiers).toEqual([
      { id, kind: "proximity", label: "a.mp3", detail: "nearest 40%", value: 40 },
    ]);
  });

  it("dismisses one modifier by running its clear callback", () => {
    const { coord, model } = makeHarness();
    const clear = vi.fn();
    const id = coord.push({ kind: "search", label: "x" }, clear);
    coord.dismiss(id);
    expect(model.viewModifiers).toEqual([]);
    expect(clear).toHaveBeenCalledTimes(1);
  });

  it("pops the newest modifier on Esc, not the oldest", () => {
    const { coord, model } = makeHarness();
    const clearOld = vi.fn();
    const clearNew = vi.fn();
    coord.push({ kind: "search", label: "old" }, clearOld);
    coord.push({ kind: "scope", label: "new" }, clearNew);
    expect(coord.popNewest()).toBe(true);
    expect(clearNew).toHaveBeenCalledTimes(1);
    expect(clearOld).not.toHaveBeenCalled();
    expect(model.viewModifiers.map((m) => m.label)).toEqual(["old"]);
  });

  it("reports false when Esc finds no modifier to pop", () => {
    const { coord } = makeHarness();
    expect(coord.popNewest()).toBe(false);
  });

  it("removes an entry whose state was cleared directly, with no callback", () => {
    const { coord, model } = makeHarness();
    const clear = vi.fn();
    const id = coord.push({ kind: "scope", label: "Camelot" }, clear);
    coord.remove(id);
    expect(model.viewModifiers).toEqual([]);
    expect(clear).not.toHaveBeenCalled();
  });

  it("finds the active modifier of a kind", () => {
    const { coord } = makeHarness();
    coord.push({ kind: "search", label: "x" }, () => {});
    const id = coord.push({ kind: "filters", label: "Tempo" }, () => {});
    expect(coord.find("filters")?.id).toBe(id);
    expect(coord.find("proximity")).toBeUndefined();
  });

  it("ignores a dismiss for an id that is no longer on the stack", () => {
    const { coord } = makeHarness();
    expect(() => coord.dismiss("vm-999")).not.toThrow();
  });

  it("never dismisses a permanent modifier", () => {
    const { coord, model } = makeHarness();
    const clear = vi.fn();
    const id = coord.push(
      { kind: "external", label: "Library Only", permanent: true },
      clear,
    );
    coord.dismiss(id);
    expect(model.viewModifiers).toHaveLength(1);
    expect(clear).not.toHaveBeenCalled();
  });

  it("pops the newest dismissible modifier, skipping permanent ones", () => {
    const { coord, model } = makeHarness();
    coord.push(
      { kind: "external", label: "Library Only", permanent: true },
      () => {},
    );
    const clearSearch = vi.fn();
    coord.push({ kind: "search", label: "x" }, clearSearch);
    expect(coord.popNewest()).toBe(true);
    expect(clearSearch).toHaveBeenCalledTimes(1);
    expect(model.viewModifiers.map((m) => m.kind)).toEqual(["external"]);
  });

  it("reports false when only permanent modifiers remain", () => {
    const { coord } = makeHarness();
    coord.push(
      { kind: "external", label: "Library Only", permanent: true },
      () => {},
    );
    expect(coord.popNewest()).toBe(false);
  });
});
