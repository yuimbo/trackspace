import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  RATING_CATEGORIES,
  buildEvalExport,
  emptyRatings,
  loadRating,
  ratedCount,
  saveRating,
  type EvalRatingRecord,
} from "./eval-ratings";

beforeEach(() => localStorage.clear());
afterEach(() => localStorage.clear());

describe("eval-ratings storage", () => {
  it("stores and reloads a rating keyed by fingerprint + revision", () => {
    const fp = "fp1";
    const rev = "rev-A";
    saveRating(fp, rev, { neighbours: 4, organisation: 3, recipeId: "baseline" });
    const rec = loadRating(fp, rev);
    expect(rec).not.toBeNull();
    expect(rec!.neighbours).toBe(4);
    expect(rec!.organisation).toBe(3);
    expect(rec!.crowding).toBeNull();
    expect(rec!.recipeId).toBe("baseline");
  });

  it("isolates ratings across revisions of the same dataset", () => {
    const fp = "fp1";
    saveRating(fp, "rev-A", { overall: 5, recipeId: "baseline" });
    saveRating(fp, "rev-B", { overall: 2, recipeId: "pca2" });
    expect(loadRating(fp, "rev-A")!.overall).toBe(5);
    expect(loadRating(fp, "rev-B")!.overall).toBe(2);
  });

  it("isolates ratings across different datasets", () => {
    saveRating("fp1", "rev-A", { overall: 5 });
    saveRating("fp2", "rev-A", { overall: 1 });
    expect(loadRating("fp1", "rev-A")!.overall).toBe(5);
    expect(loadRating("fp2", "rev-A")!.overall).toBe(1);
    expect(loadRating("fp1", "missing") ?? null).toBeNull();
  });

  it("merges patches into one record and ignores out-of-range values", () => {
    const fp = "fp1";
    const rev = "rev-A";
    saveRating(fp, rev, { neighbours: 2, note: "sparse" });
    saveRating(fp, rev, { crowding: 9, overall: 4 });
    const rec = loadRating(fp, rev)!;
    expect(rec.neighbours).toBe(2);
    expect(rec.overall).toBe(4);
    expect(rec.crowding).toBeNull(); // 9 is out of 1–5
    expect(rec.note).toBe("sparse");
  });
});

describe("ratedCount", () => {
  it("counts filled categories", () => {
    expect(ratedCount(null)).toBe(0);
    const r = emptyRatings();
    expect(ratedCount(r)).toBe(0);
    expect(ratedCount({ ...r, neighbours: 3, overall: 1 })).toBe(2);
  });
});

describe("buildEvalExport", () => {
  it("serialises all records with the dataset fingerprint", () => {
    const fp = "fp1";
    const rec: EvalRatingRecord = {
      ...emptyRatings(),
      recipeId: "baseline",
      revision: "rev-A",
      label: "PCA50 → openTSNE",
      neighbours: 4,
      note: "nice neighbours",
      updatedAt: 123,
    };
    const json = JSON.parse(buildEvalExport(fp, [rec]));
    expect(json.dataset_fingerprint).toBe(fp);
    expect(json.maps).toHaveLength(1);
    expect(json.maps[0].label).toBe("PCA50 → openTSNE");
    expect(json.maps[0].neighbours).toBe(4);
  });

  it("sorts rows by updatedAt", () => {
    const mk = (rev: string, at: number): EvalRatingRecord => ({
      ...emptyRatings(),
      recipeId: rev,
      revision: rev,
      label: "",
      updatedAt: at,
    });
    const json = JSON.parse(
      buildEvalExport("fp1", [mk("b", 200), mk("a", 100)]),
    );
    expect(json.maps.map((m: { recipeId: string }) => m.recipeId)).toEqual(["a", "b"]);
  });

  it("declares the exact category set", () => {
    expect(RATING_CATEGORIES).toEqual([
      "neighbours",
      "organisation",
      "crowding",
      "overall",
    ]);
  });
});
