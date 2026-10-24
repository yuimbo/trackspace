import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { assertWritable, isEvalReadOnly, setEvalReadOnly } from "./eval-readonly";

beforeEach(() => setEvalReadOnly(false));
afterEach(() => setEvalReadOnly(false));

describe("eval-readonly guard", () => {
  it("is inert by default", () => {
    expect(isEvalReadOnly()).toBe(false);
    expect(() => assertWritable("edit tags")).not.toThrow();
  });

  it("throws once enabled", () => {
    setEvalReadOnly(true);
    expect(isEvalReadOnly()).toBe(true);
    expect(() => assertWritable("edit tags")).toThrow(/read-only layout evaluation/i);
  });
});

describe("owned HTTP layer gate", () => {
  it("blocks POSTs while eval read-only is active, without calling fetch", async () => {
    setEvalReadOnly(true);
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);
    const { postJSON } = await import("./api");
    await expect(postJSON("/api/tracks/tags", { updates: [] })).rejects.toThrow(
      /read-only layout evaluation/i,
    );
    expect(fetchSpy).not.toHaveBeenCalled();
    vi.unstubAllGlobals();
  });

  it("still allows GETs while eval read-only is active", async () => {
    setEvalReadOnly(true);
    vi.stubGlobal("fetch", vi.fn(() => Promise.resolve({ ok: true, json: () => Promise.resolve({}) })));
    const { api } = await import("./api");
    await expect(api("/api/layout-evaluation", undefined, true)).resolves.toEqual({});
    vi.unstubAllGlobals();
  });
});
