import { describe, expect, it } from "vitest";
import {
  blindLetter,
  computeEvalUniverse,
  evalExitUrl,
  isUnsupportedEvalSchema,
  parseEvalArtifact,
  parseEvalManifest,
  EVAL_SCHEMA_VERSION,
} from "./layout-eval";

const MANIFEST = {
  schema_version: 1,
  status: "complete",
  dataset_fingerprint: "abcd1234",
  recipes: {
    baseline: {
      status: "complete",
      label: "PCA50 → openTSNE",
      tracks: 3,
      timings: { pca: 0.2, tsne: 1.1, alignment: 0 },
      aligned_to: null,
    },
    pca2: {
      status: "complete",
      label: "PCA2 → openTSNE",
      tracks: 3,
      timings: { pca: 0.1, tsne: 0.9, alignment: 0.01 },
      aligned_to: "baseline",
    },
    error_recipe: { status: "error", reason: "too few tracks" },
  },
};

describe("parseEvalManifest", () => {
  it("lists only completed recipes, preserving order", () => {
    const m = parseEvalManifest(MANIFEST);
    expect(m).not.toBeNull();
    expect(m!.datasetFingerprint).toBe("abcd1234");
    expect(m!.recipes.map((r) => r.id)).toEqual(["baseline", "pca2"]);
    expect(m!.recipes[1].alignedTo).toBe("baseline");
  });

  it("returns null when no completed recipe remains", () => {
    const m = parseEvalManifest({
      ...MANIFEST,
      recipes: { a: { status: "error", reason: "x" } },
    });
    expect(m).toBeNull();
  });

  it("rejects a manifest with a different schema version", () => {
    expect(parseEvalManifest({ ...MANIFEST, schema_version: 2 })).toBeNull();
    expect(parseEvalManifest({ ...MANIFEST, schema_version: 0 })).toBeNull();
  });

  it("requires the current schema version", () => {
    const { schema_version: _dropped, ...noVersion } = MANIFEST as Record<
      string,
      unknown
    >;
    expect(parseEvalManifest(noVersion)).toBeNull();
  });

  it("accepts the current schema version and reports it", () => {
    const m = parseEvalManifest({ ...MANIFEST, schema_version: EVAL_SCHEMA_VERSION });
    expect(m).not.toBeNull();
    expect(m!.schemaVersion).toBe(EVAL_SCHEMA_VERSION);
  });

  it("returns null for garbage", () => {
    expect(parseEvalManifest(null)).toBeNull();
    expect(parseEvalManifest("nope")).toBeNull();
    expect(parseEvalManifest({})).toBeNull();
  });

  it("defaults legacy entries to 2-D with the id as the group", () => {
    const m = parseEvalManifest(MANIFEST)!;
    expect(m.recipes[0]).toMatchObject({ id: "baseline", dims: 2, group: "baseline" });
    expect(m.recipes[1]).toMatchObject({ id: "pca2", dims: 2, group: "pca2" });
  });

  it("parses paired dims/group metadata", () => {
    const m = parseEvalManifest({
      ...MANIFEST,
      recipes: {
        base: { status: "complete", label: "Base", dims: 2, group: "base" },
        base_3d: { status: "complete", label: "Base (3D)", dims: 3, group: "base" },
      },
    })!;
    expect(m.recipes.map((r) => [r.id, r.dims, r.group])).toEqual([
      ["base", 2, "base"],
      ["base_3d", 3, "base"],
    ]);
  });

  it("rejects an entry with an unsupported dims value", () => {
    const m = parseEvalManifest({
      ...MANIFEST,
      recipes: {
        odd: { status: "complete", label: "Odd", dims: 4, group: "odd" },
        ok: { status: "complete", label: "Ok", dims: 2, group: "ok" },
      },
    });
    expect(m!.recipes.map((r) => r.id)).toEqual(["ok"]);
  });
});

describe("isUnsupportedEvalSchema", () => {
  it("flags only a manifest carrying a different schema", () => {
    expect(isUnsupportedEvalSchema({ ...MANIFEST, schema_version: 2 })).toBe(true);
    expect(isUnsupportedEvalSchema({ schema_version: 0 })).toBe(true);
  });

  it("passes the current schema and non-manifest payloads", () => {
    expect(isUnsupportedEvalSchema(MANIFEST)).toBe(false);
    expect(isUnsupportedEvalSchema(null)).toBe(false);
    expect(isUnsupportedEvalSchema("nope")).toBe(false);
    expect(isUnsupportedEvalSchema({})).toBe(false);
  });
});

describe("parseEvalArtifact", () => {
  const artifact = {
    id: "baseline",
    label: "PCA50 → openTSNE",
    revision: "rev-1",
    dataset_fingerprint: "abcd1234",
    tracks: 2,
    aligned_to: null,
    timings: { pca: 0.2, tsne: 1.1 },
    positions: [
      { path: "a.mp3", x: 0.1, y: 0.2 },
      { path: "b.mp3", x: 0.8, y: 0.9 },
    ],
  };

  it("parses a valid artifact into a positions map", () => {
    const a = parseEvalArtifact(artifact);
    expect(a).not.toBeNull();
    expect(a!.positions.size).toBe(2);
    expect(a!.positions.get("a.mp3")).toEqual({ x: 0.1, y: 0.2 });
    expect(a!.revision).toBe("rev-1");
  });

  it("rejects non-finite coordinates", () => {
    const bad = parseEvalArtifact({
      ...artifact,
      positions: [
        { path: "a.mp3", x: Infinity, y: 0.2 },
        { path: "b.mp3", x: 0.8, y: 0.9 },
      ],
    });
    expect(bad).toBeNull();
  });

  it("rejects an empty or non-array positions set", () => {
    expect(parseEvalArtifact({ ...artifact, positions: [] })).toBeNull();
    expect(parseEvalArtifact({ ...artifact, positions: "nope" })).toBeNull();
  });

  it("defaults a legacy artifact to 2-D and the id as the group", () => {
    const a = parseEvalArtifact(artifact)!;
    expect(a.dims).toBe(2);
    expect(a.group).toBe("baseline");
    expect(a.positions.get("a.mp3")).toEqual({ x: 0.1, y: 0.2 });
  });

  it("parses a paired 3-D artifact with finite z", () => {
    const a = parseEvalArtifact({
      ...artifact,
      dims: 3,
      group: "baseline",
      positions: [
        { path: "a.mp3", x: 0.1, y: 0.2, z: 0.3 },
        { path: "b.mp3", x: 0.8, y: 0.9, z: 0.4 },
      ],
    })!;
    expect(a.dims).toBe(3);
    expect(a.group).toBe("baseline");
    expect(a.positions.get("a.mp3")).toEqual({ x: 0.1, y: 0.2, z: 0.3 });
  });

  it("rejects a 3-D artifact whose positions lack a finite z", () => {
    const base = { ...artifact, dims: 3, group: "baseline" };
    expect(
      parseEvalArtifact({
        ...base,
        positions: [{ path: "a.mp3", x: 0.1, y: 0.2 }],
      }),
    ).toBeNull();
    expect(
      parseEvalArtifact({
        ...base,
        positions: [{ path: "a.mp3", x: 0.1, y: 0.2, z: Infinity }],
      }),
    ).toBeNull();
  });

  it("rejects an unsupported artifact dimensionality", () => {
    expect(parseEvalArtifact({ ...artifact, dims: 4 })).toBeNull();
  });
});

describe("blindLetter", () => {
  it("maps 0-based indices to A, B, …", () => {
    expect(blindLetter(0)).toBe("A");
    expect(blindLetter(1)).toBe("B");
    expect(blindLetter(7)).toBe("H");
  });
});

describe("evalExitUrl", () => {
  it("strips the eval-only params and preserves unrelated ones", () => {
    const out = evalExitUrl(
      "http://host/app?layout_eval=1&set=budget&layout=baseline&theme=dark&foo=bar",
    );
    const url = new URL(out);
    expect(url.searchParams.has("layout_eval")).toBe(false);
    expect(url.searchParams.has("layout")).toBe(false);
    expect(url.searchParams.has("set")).toBe(false);
    expect(url.searchParams.get("theme")).toBe("dark");
    expect(url.searchParams.get("foo")).toBe("bar");
  });

  it("leaves a plain URL untouched", () => {
    expect(evalExitUrl("http://host/app")).toBe("http://host/app");
  });
});

describe("computeEvalUniverse", () => {
  const artifactPaths = new Set(["a.mp3", "b.mp3", "c.mp3"]);

  it("keeps artifact paths present in the library", () => {
    const loaded = new Set(["a.mp3", "b.mp3", "d.mp3"]);
    const { eligible, unavailable } = computeEvalUniverse(artifactPaths, loaded);
    expect([...eligible].sort()).toEqual(["a.mp3", "b.mp3"]);
    expect(unavailable).toBe(1); // c.mp3
  });

  it("reports everything unavailable when the library is empty", () => {
    const { eligible, unavailable } = computeEvalUniverse(artifactPaths, new Set());
    expect(eligible.size).toBe(0);
    expect(unavailable).toBe(3);
  });
});
