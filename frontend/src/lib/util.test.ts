import { describe, expect, it } from "vitest";
import { clamp, clamp01, cycle, escapeHtml } from "./util";

describe("clamp", () => {
  it("clamps to [0,1] by default", () => {
    expect(clamp(-1)).toBe(0);
    expect(clamp(0.4)).toBe(0.4);
    expect(clamp(2)).toBe(1);
  });

  it("honours explicit bounds", () => {
    expect(clamp(150, 0, 100)).toBe(100);
    expect(clamp(-5, 0, 100)).toBe(0);
    expect(clamp(40, 0, 100)).toBe(40);
  });

  it("clamp01 matches clamp(v, 0, 1)", () => {
    for (const v of [-2, -0.1, 0, 0.3, 1, 1.7]) {
      expect(clamp01(v)).toBe(clamp(v, 0, 1));
    }
  });
});

describe("escapeHtml", () => {
  it("escapes the five HTML-significant characters", () => {
    expect(escapeHtml(`<a href="x">&'</a>`)).toBe(
      "&lt;a href=&quot;x&quot;&gt;&amp;&#39;&lt;/a&gt;",
    );
  });

  it("leaves plain text unchanged", () => {
    expect(escapeHtml("Kick drum")).toBe("Kick drum");
  });
});

describe("cycle", () => {
  it("steps forward and wraps", () => {
    expect(cycle(["a", "b", "c"], "a")).toBe("b");
    expect(cycle(["a", "b", "c"], "c")).toBe("a");
  });

  it("steps backward and wraps", () => {
    expect(cycle(["a", "b", "c"], "a", -1)).toBe("c");
    expect(cycle(["a", "b", "c"], "b", -1)).toBe("a");
  });

  it("returns the fallback when the current value is absent", () => {
    expect(cycle(["a", "b", "c"], "z" as string)).toBe("a");
    expect(cycle(["a", "b", "c"], "z" as string, -1, "b")).toBe("b");
  });

  it("throws on an empty list", () => {
    expect(() => cycle([], "a")).toThrow("cycle: empty list");
  });
});
