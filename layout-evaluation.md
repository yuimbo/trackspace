# Layout evaluation (offline precomputed maps)

Generates reproducible 2-D maps — plus explicitly registered 3-D siblings in the
paired sets — over the *same* cached library embeddings for the browser eval
gallery. It is **read-only** over the production caches: it gathers
the composite matrix exactly like the interactive projection, fits its own
openTSNE/UMAP models, and writes JSON artifacts under `data/layout-evaluation/<set>/`
(gitignored with the rest of `data/`). It never writes to the production layout
cache, the t-SNE fit cache, or MP3 tags. The projection machinery it reuses is in
`audio_pipeline.md` → *Projection methods* / *Composite vector construction*.

**Baseline.** `baseline` is the original gallery control: openTSNE over PCA50 with
the production perplexity/seed, the frozen default source weights, feature mask and
blend, at a **frozen 500/250 budget** — the original openTSNE library default. It is
fully **unsupervised**: it fits the raw composite matrix with no folder/tag
supervision, and the other base recipes are fitted over that same fixed input and
aligned onto the baseline by an orthogonal rotation/reflection only (no warp). The
label says "production default" because recipes and labels are frozen artifact
identity (`label` is hashed into the revision); the current production fit is the
multilevel chain described in `audio_pipeline.md` → *Projection methods*. Treat the
500/250 baseline as a historical control, not a production measurement.

Each output artifact holds the `{path, x, y}` positions for the whole evaluated
cohort; an artifact whose recipe is 3-D adds a finite `z` to every position. Paired
groups carry a 2-D base and an explicit `_3d` sibling with `dims` / `group`
metadata; the two are aligned onto **dimension-specific** references, so their x/y
are not guaranteed to coincide.

**Cohort.** The built-in generator (`precompute_layout_evaluation`) uses
`load_infos(scope="library")`: the configured **library roots only**, external
roots excluded (the same cohort the interactive library-only fit uses). On this
library that is **7,153 tracks** — a frozen historical cohort, not the all-roots
production set. The all-roots production benchmark measures a different,
**15,571-track** eligible set (15,574 fingerprinted); see *Production projection
benchmark* below. Do not compare counts across the two without noting the cohort.
Artifacts are keyed by their own dataset fingerprint/revision, so a regenerated set
with a changed cohort is a new rating identity.

**The sets.** A set is a directory of artifacts generated with one recipe list;
the gallery serves only allowlisted names (`base`, `budget`, `scale`, `early`,
`mixes`, `ablation`, `hierarchy`, `hierarchy-budget`, `hierarchy-depth`,
`hierarchy-tuning`, `hierarchy-distribution`, `hierarchy-heuristic`,
`hierarchy-fractions`, `hierarchy-density`, `hierarchy-gamma`,
`hierarchy-combination`, `genre`, `genre-directed`, `genre-logits-ab`,
`folder-separation`, `folder-cheap`):

| Set | Recipes | Tests |
|-----|---------|-------|
| `base` (default) | baseline, pca2, pca25, tsne_budget, umap2_n15, umap2_n60, umap_tsne, tsne_umap_init | the original eight-map gallery: dimensionality, iteration budget, UMAP variants |
| `budget` | baseline + tsne_budget100, tsne_budget150, tsne_budget | total t-SNE iteration count; the 500/250 baseline is the reference |
| `scale` | tsne_budget100 (control) + tsne_p60, tsne_p120, tsne_multiscale30_120 | the perplexity axis |
| `early` | tsne_budget100 (control) + tsne_early150, tsne_early100 | early-exaggeration phase length at the frozen 100-iteration control budget |
| `mixes` | tsne_budget100 (control) + tsne_rhythm_forward, tsne_texture_style | source weights: rhythm-led vs timbre/semantics-led |
| `ablation` | tsne_budget100 (control) + tsne_maest_supervised, tsne_maest_only | dropping whole blocks (rhythm+CLAP, then supervised) |
| `hierarchy` | tsne_early100 (control) + tsne50_plain + six `tsne_sample*` maps | two-level sample-initialisation: coarse fit on a sample, then a full refinement from it |
| `hierarchy-budget` | tsne_early100 (control) + tsne50_plain + tsne_sample1000_early0 (anchor) + seven coarse-budget `tsne_sample1000_*` maps | two-level sample init with the coarse budget swept (sample1000, final 100/0) |
| `hierarchy-depth` | tsne_early100 (control) + tsne50_plain + tsne_sample1000_early0 (anchor) + two `tsne_sample*_early0` + three `tsne_chain*` maps | how deep the sample hierarchy can go: two-level at small samples plus nested multilevel chains |
| `hierarchy-tuning` | tsne_chain50_250_1000 (control/reference) + four `tsne_chain50_250_1000_*` variants | isolate the global-shape budget around the shallow 50→250→1000 chain: cheaper/final-only coarse and a very long final fit |
| `hierarchy-distribution` | tsne_chain50_250_1000_final50 (control/reference) + four `tsne_chain50_250_1000_*_final50` variants | where the normal-iteration budget is spent across the chain's levels: a fixed +900 spread over one level or evenly (final 50/0) |
| `hierarchy-heuristic` | tsne_chain50_250_1000_final50 (reference) + six `tsne_geo*_*` geometric chains | exponential level sizes with a sqrt normal-iteration schedule and per-level perplexities, isolating the level-growth and perplexity axes |
| `hierarchy-fractions` | tsne_geo8_combined (reference) + three `tsne_fraction_g*` maps | size-adaptive fraction chains: level sizes as fractions of n, auto sqrt normal budget, flat perplexity 30 |
| `hierarchy-density` | tsne_geo8_combined (reference) + `tsne_density2/4/8/16` + `tsne_continuous16` | how deep a fixed 541-normal fraction chain can go (2→16 levels) plus a 1-iteration-per-stage stress |
| `hierarchy-gamma` | `tsne_density4` (flat-30 control) + `tsne_gamma0/05/1/2/4` | per-level perplexity schedule (gamma exponent) over the density4 geometry |
| `hierarchy-combination` | `tsne_continuous16` (2-D reference) + `tsne_continuous16_wide` + `tsne_continuous16_wide_gamma4` and the three explicit `_3d` siblings | continuous 16-stage winner vs a wider 1/256→1/16 span vs the same span with a gamma-4 (1/256-anchored) schedule; paired 2-D/3-D groups |
| `genre`, `genre-directed`, `genre-logits-ab` | genre-space maps (`backend.experiments.genre_space maps --gallery-output …`) | genre representations over the offline cohort (schema-1, virtual paths); see README → *Offline genre benchmark* |
| `folder-separation` | `folder_on` / `folder_off`, each paired with an explicit `_3d` sibling (`backend.benchmark_production_projection --gallery-output …`) | historical production `genre_full` 2-D/3-D map with semantic folder weighting on vs off, frozen at full basis / folder boost 6 / depth 3, before the centroid-only 2.4 production adoption |
| `folder-cheap` | six groups (`folder_off`, `centroids`, `approx`, `off_exag15`, `off_exag2`, `centroids_exag15`), each paired with an explicit `_3d` sibling | cheap folder-basis comparison: unweighted baseline vs centroid-only and randomized-SVD bases, plus late-exaggeration 1.5/2 variants; all scored against the same unweighted PCA reference |

Each set's **first recipe is its alignment reference** (the CLI derives that from
`GALLERY_SETS`): `budget` → the 500/250 `baseline`; `scale`/`early`/`mixes`/`ablation`
→ `tsne_budget100`; `hierarchy`/`hierarchy-budget`/`hierarchy-depth` → `tsne_early100`;
`hierarchy-tuning` → `tsne_chain50_250_1000`; `hierarchy-distribution`/`-heuristic` →
`tsne_chain50_250_1000_final50`; `hierarchy-fractions`/`-density` → `tsne_geo8_combined`;
`hierarchy-gamma` → `tsne_density4`; `hierarchy-combination` → `tsne_continuous16`.
Recipe parameters (weights, perplexities, budgets) live in
`backend/embeddings/layout_evaluation.py`; judge the maps by ear and ratings, not by
which looks more spread out.

### Generate

`--set` selects a built-in set's recipes, reference and default output directory
from `GALLERY_SETS`; `--recipes` remains available for an ad-hoc list. The two are
mutually exclusive, and `--output` / `--reference` override the set defaults.

```bash
# Base eight-map gallery → <data-dir>/layout-evaluation (the default the server serves)
python -m backend.precompute_layout_evaluation --set base

# Every other built-in set → <data-dir>/layout-evaluation/<set>
python -m backend.precompute_layout_evaluation --set budget
python -m backend.precompute_layout_evaluation --set scale
python -m backend.precompute_layout_evaluation --set early
python -m backend.precompute_layout_evaluation --set mixes
python -m backend.precompute_layout_evaluation --set ablation
python -m backend.precompute_layout_evaluation --set hierarchy
python -m backend.precompute_layout_evaluation --set hierarchy-budget
python -m backend.precompute_layout_evaluation --set hierarchy-depth
python -m backend.precompute_layout_evaluation --set hierarchy-tuning
python -m backend.precompute_layout_evaluation --set hierarchy-distribution
python -m backend.precompute_layout_evaluation --set hierarchy-heuristic
python -m backend.precompute_layout_evaluation --set hierarchy-fractions
python -m backend.precompute_layout_evaluation --set hierarchy-density
python -m backend.precompute_layout_evaluation --set hierarchy-gamma
python -m backend.precompute_layout_evaluation --set hierarchy-combination

# Common options
python -m backend.precompute_layout_evaluation --set base --data-dir data \
  --root /path/to/music --limit 200 --seed 42 --threads 8 -v
```

`run_generation` writes an `in_progress` manifest **before** the first fit (so an
older `complete` manifest cannot masquerade as this run), rewrites it after each
recipe, and ends `complete` / `partial` / `failed`. Per-recipe `status`/`reason`
and the dataset fingerprint, seed, source/version environment and timings are in
the manifest and each artifact, so an interrupted or partial run is inspectable.

Other options: `--data-dir` is the cache dir (`TRACKSPACE_DATA_DIR` or `data/`);
`--root` overrides `roots.json` with one library root; `--limit` caps tracks for a
smoke run; `--sources` selects a source subset (single-lens weights, not the frozen
default mix); `--reference` overrides the set's alignment reference; `--seed` is
the fit/PCA/order seed (default 42); `--threads` is `n_jobs` (-1 = all cores).

The `genre`/`genre-directed`/`genre-logits-ab` and `folder-separation`/`folder-cheap`
sets are **externally generated**: `--set <name>` is accepted by the gallery route
but rejected by this CLI with a pointer to the owning tool. Use:

```bash
# Genre-space maps (see README → Offline genre benchmark for cohort/extract/score)
python -m backend.experiments.genre_space maps \
  --manifest data/genre-space/first/cohort.json \
  --artifacts data/genre-space/first/vectors \
  --gallery-output data/layout-evaluation/genre-directed \
  --output data/genre-space/first/directed-maps \
  --recipe-set directed

# Folder on/off at today's benchmark defaults (exact-SVD `full` basis at the
# current production folder boost 2.4 / depth 3; not the current centroid path or
# the historical boost-6 artifacts, so this regenerates a new-revision gallery)
python -m backend.benchmark_production_projection \
  --data-dir data --scope all --recipe genre_full \
  --cases native,folders-off --dims 2,3 \
  --gallery-output data/layout-evaluation/folder-separation

# Cheap folder-basis comparison (all six cases scored against the same
# unweighted PCA reference; no exact-SVD full basis)
python -m backend.benchmark_production_projection \
  --data-dir data --scope all --recipe genre_full \
  --cases folders-off,centroids,approx,off-exag15,off-exag2,centroids-exag15 \
  --dims 2,3 --repeats 3 --metric-reference unweighted \
  --gallery-output data/layout-evaluation/folder-cheap \
  --output data/layout-evaluation/folder-cheap-timings.json
```


**The `folder-separation` set** is a historical comparison of the production
`genre_full` map with semantic folder weighting applied against the same composite
with folder scaling off. Its artifacts were fitted with the **legacy exact-SVD
`full` folder basis at folder boost 6 / depth 3**, before production adopted the
centroid-only basis at boost 2.4. Both modes were prepared once over the identical
all-roots cohort (external-root tracks included) and fitted at 2-D and 3-D with the
production multilevel budget (final 50/0); each off map is aligned onto the on map
**of its own dimension**. The gallery has two groups (on/off), and **3** switches a
group between its 2-D and 3-D sibling.

**Provenance vs regeneration.** The command above runs today's benchmark defaults:
`native` still selects the exact-SVD `full` basis, but production folder weights are
now boost 2.4 / depth 3 (`FROZEN_FOLDER_BOOST`), so it produces a **new** gallery
with different labels/revisions, not the frozen boost-6 artifacts. Keep the old
artifacts to inspect the historical comparison; regenerate only when you mean the
current configuration. There is no CLI knob for a historical boost — adding one is
not wanted. The recorded prepare timing pairs the weighted production prepare with
the warm unweighted gather diagnostic — an estimate, not a cold on/off comparison.
Folder weighting is a soft semantic basis
(`backend/embeddings/semantic_weighting.py`), not a hard folder boundary and not a
`T_n` partition choice, so this set neither tunes nor depends on how that basis is
built. Each artifact's timings come from the first measured fit in its first
measured round (warmups excluded), recorded in the manifest and the artifact; the
manifest `timings.total` counts each shared prepare/PCA once plus every fit and
alignment once per artifact, and the manifest dataset also records the CLI
cohort selectors (`scope` / `limit`). `seed` is the CLI sampling/order/metric
seed; `fit_seed` is the production optimizer seed (42), which `--seed` never
changes.

**The `folder-cheap` set** compares the unweighted baseline with the two cheap
folder-basis solvers (`centroids` = hierarchical contrast directions only,
`approx` = the same within-folder PCs from a seeded randomized SVD) and three
late-exaggeration variants (off ×1.5, off ×2, centroids ×1.5); the
un-suffixed cases use the production canonical late exaggeration (1.25; the
manifest records both the raw value and its effective value). The artifact was also
measured at the historical boost 6 (recorded in its `recipe_params`); the
**relative** basis comparison is what it supports, not a claim about current
production at 2.4. The legacy exact-SVD `full` basis is deliberately absent — it is
the expensive path this comparison is meant to avoid. Cases are prepared once each
per library size: `centroids` is cold, `approx` and `folders-off` are warm gathers.
All six maps are scored against **the same unweighted PCA matrix**
(`--metric-reference unweighted`; the old default stays each case's own PCA input),
so recall and Spearman numbers are comparable across bases. The `folders-off` map
is the alignment reference of its own dimension; the manifest records each case's
`folder_basis_mode`, `final_exaggeration`, and prepare timing/warm flag. Its
fidelity scores live in the `--output` JSON, not the gallery manifest.

Caveat: prepare/PCA timings are measured **once per basis mode, not per
repeat**. `centroids` is prepared first and cold; `approx` and `folders-off` are
warm, so their prepare times are not an apples-to-apples cold comparison. The
fits themselves run three measured rounds in a seeded shuffled order, and the
quality numbers use the common unweighted reference above.

**The `hierarchy` set** is an exploratory speed/fidelity probe of **two-level
sample initialisation**: each sampled map fits a coarse t-SNE on a deterministic
uniform sample of rows, expands every full-data row onto its nearest landmark's
coarse position (plus a tiny deterministic jitter), then refines a full-data
t-SNE from that initialisation. The reference is the plain 100-normal/100-early
control `tsne_early100` (an existing recipe). Budgets are written **normal/early**
— the order is unambiguous:

| Recipe | sample `n` | coarse budget | final (refine) budget |
|--------|-----------|---------------|-----------------------|
| `tsne_early100` (control) | — (plain) | — | 100 / 100 |
| `tsne50_plain` | — (plain) | — | 50 / 50 |
| `tsne_sample500` | 500 | 100 / 100 | 100 / 100 |
| `tsne_sample1000` | 1000 | 100 / 100 | 100 / 100 |
| `tsne_sample2000` | 2000 | 100 / 100 | 100 / 100 |
| `tsne_sample1000_early0` | 1000 | 100 / 100 | 100 / **0** (no early phase) |
| `tsne_sample1000_final50` | 1000 | 100 / 100 | 50 / 50 |
| `tsne_sample1000_coarse50` | 1000 | 50 / 50 | 100 / 100 |

This compares sample size (500/1000/2000) and where the compute budget is spent
(coarse vs final phase) against the plain controls. It is exploratory: judge by
ear and ratings, not by which looks more spread out. A sampled map's artifact
timings record the whole pipeline under `tsne` plus informational
`tsne_coarse` / `tsne_expand` / `tsne_refine` keys (the latter are **not** summed
again into `total`).

**The `hierarchy-budget` set** sweeps the **coarse** budget of the two-level
sample init while the final refinement is held fixed. Every map is sample1000
with the final budget 100 normal / **0 early** — the `tsne_sample1000_early0`
anchor geometry — and only the coarse budget changes. The ids write the coarse
budget as `coarse{normal}e{early}`. The reference is the plain 100/100 control
`tsne_early100`; `tsne50_plain` and the anchor `tsne_sample1000_early0` round out
the controls. Budgets are **normal/early**:

| Recipe | sample `n` | coarse budget | final (refine) budget |
|--------|-----------|---------------|-----------------------|
| `tsne_early100` (control) | — (plain) | — | 100 / 100 |
| `tsne50_plain` | — (plain) | — | 50 / 50 |
| `tsne_sample1000_early0` (anchor) | 1000 | 100 / 100 | 100 / **0** |
| `tsne_sample1000_coarse10e100` | 1000 | **10** / 100 | 100 / **0** |
| `tsne_sample1000_coarse25e100` | 1000 | **25** / 100 | 100 / **0** |
| `tsne_sample1000_coarse50e100` | 1000 | **50** / 100 | 100 / **0** |
| `tsne_sample1000_coarse500e100` | 1000 | **500** / 100 | 100 / **0** |
| `tsne_sample1000_coarse100e0` | 1000 | 100 / **0** | 100 / **0** |
| `tsne_sample1000_coarse100e25` | 1000 | 100 / **25** | 100 / **0** |
| `tsne_sample1000_coarse100e50` | 1000 | 100 / **50** | 100 / **0** |

This isolates *where* the coarse budget is spent: how few coarse normal iterations
(or how short the coarse early phase) still seed landmarks that survive the full
refinement. It uses only the existing two-level fields — no new recipe field.

**The `hierarchy-depth` set** pushes the sample hierarchy as far down as it can
go: two new **two-level** maps at small samples (100 / 250) plus three
**multilevel chain** recipes (method `tsne_multilevel_init`). A multilevel chain
builds a single seeded nested sample hierarchy — the coarse fit runs on the
smallest level, then each larger level expands the previous coords onto its own
rows and refines a full t-SNE at that size — ending in a full-data fit at the
recipe's own final budget. The chain ids spell the level sizes (`100→500→2000`),
and the levels (sizes + per-level `normal/early` budgets) are stored in each
artifact's `recipe` record. The reference is `tsne_early100`; `tsne50_plain` and
the anchor `tsne_sample1000_early0` are the controls. Budgets are **normal/early**:

| Recipe | levels (size → size → …) | coarse budget | intermediate levels | final budget |
|--------|--------------------------|---------------|---------------------|--------------|
| `tsne_early100` (control) | — (plain) | — | — | 100 / 100 |
| `tsne50_plain` | — (plain) | — | — | 50 / 50 |
| `tsne_sample1000_early0` (anchor) | — (two-level, sample1000) | 100 / 100 | — | 100 / **0** |
| `tsne_sample100_early0` | — (two-level, sample100) | 100 / 100 | — | 100 / **0** |
| `tsne_sample250_early0` | — (two-level, sample250) | 100 / 100 | — | 100 / **0** |
| `tsne_chain100_500_2000_final100` | 100 → 500 → 2000 | 100 / 100 | 100 / 0 | 100 / **0** |
| `tsne_chain100_500_2000_final50` | 100 → 500 → 2000 | 100 / 100 | 100 / 0 | 50 / **0** |
| `tsne_chain50_250_1000` | 50 → 250 → 1000 | 100 / 100 | 100 / 0 | 100 / **0** |

This asks whether going coarse → coarse → … → full (each refinement seeded by the
previous, cheaper level) matches a plain or two-level fit, and how little coarse
data is enough. A chain's artifact timings record the whole pipeline under `tsne`
plus per-level `tsne_level0`/`tsne_level1`/…, `tsne_expand` and `tsne_final` keys
(informational; **not** summed again into `total`).

**The `hierarchy-tuning` set** isolates the *global-shape* budget around the
shallow `50→250→1000` chain: the reference `tsne_chain50_250_1000` (coarse
100/100, intermediates 100/0, final 100/0) plus four variants that move only one
axis, so the difference you see is attributable to that axis alone. Every map is
PCA50, seed 42, references `tsne_chain50_250_1000`, and uses only the existing
multilevel fields. Budgets are **normal/early**:

| Recipe | levels (size → size → …) | coarse budget | intermediate levels | final budget |
|--------|--------------------------|---------------|---------------------|--------------|
| `tsne_chain50_250_1000` (control/reference) | 50 → 250 → 1000 | 100 / 100 | 100 / 0 | 100 / **0** |
| `tsne_chain50_250_1000_final50` | 50 → 250 → 1000 | 100 / 100 | 100 / 0 | 50 / **0** |
| `tsne_chain50_250_1000_coarse10` | 50 → 250 → 1000 | **10** / 100 | 100 / 0 | 100 / **0** |
| `tsne_chain50_250_1000_coarse10_final50` | 50 → 250 → 1000 | **10** / 100 | 100 / 0 | 50 / **0** |
| `tsne_chain50_250_1000_final2000` | 50 → 250 → 1000 | 100 / 100 | 100 / 0 | **2000** / **0** |

The `final*` variants touch **only** the final full-data fit (its `normal` count;
the early phase stays 0) — the coarse and intermediate levels are byte-identical
to the reference, so a `final2000` result isolates whether a much longer final
fit alone changes the global shape, and a `final50` the opposite. The `coarse10`
variants move only the coarse budget (10 normal / 100 early at the 50-row level).
This is exploratory: judge by ear and ratings, not spread.

**The `hierarchy-distribution` set** holds the *final-fit geometry* fixed (50
normal / 0 early) and asks **where** the normal-iteration budget is spent across
the shallow `50→250→1000` chain's levels. The reference is
`tsne_chain50_250_1000_final50` (the shared final50 tuning variant: coarse 100/100,
intermediates 100/0, final 50/0). Every variant adds exactly **900** normal
iterations on top of the reference's 300 level-normal total (so all total 1200),
keeping the early phases unchanged (100 on the first level, 0 elsewhere) and the
final fit byte-identical. Budgets are **normal/early**, spelled per level as
`level{size} {normal}/{early}` so a level size (50/250/1000) is never confused
with a normal-iteration count:

| Recipe | levels (size → size → …) | level budgets (normal/early) | final budget |
|--------|--------------------------|------------------------------|--------------|
| `tsne_chain50_250_1000_final50` (control/reference) | 50 → 250 → 1000 | 100 / 100, 100 / 0, 100 / 0 | 50 / **0** |
| `tsne_chain50_250_1000_early1000_final50` | 50 → 250 → 1000 | **1000** / 100, 100 / 0, 100 / 0 | 50 / **0** |
| `tsne_chain50_250_1000_middle1000_final50` | 50 → 250 → 1000 | 100 / 100, **1000** / 0, 100 / 0 | 50 / **0** |
| `tsne_chain50_250_1000_late1000_final50` | 50 → 250 → 1000 | 100 / 100, 100 / 0, **1000** / 0 | 50 / **0** |
| `tsne_chain50_250_1000_spread400_final50` | 50 → 250 → 1000 | **400** / 100, **400** / 0, **400** / 0 | 50 / **0** |

The `early`/`middle`/`late` variants dump the whole 900-iteration surplus into the
first / second / third level (at level sizes 50 / 250 / 1000 respectively); the
`spread400` variant spreads it evenly (400 per level). This isolates whether it is
the *total* normal budget or *where* it is concentrated that changes the global
shape, all at a fixed cheap final fit. This is exploratory: judge by ear and
ratings, not spread.

**The `hierarchy-heuristic` set** probes **geometric** multilevel chains: level
sizes grow **exponentially** (×4 or ×8) so each stage is a *separate* optimisation
that rebuilds its affinities only once, at a level whose pairwise cost grows with
size — unlike a long single fit, which keeps one fixed affinity matrix across
every gradient iteration. The cost driver is therefore the number and size of
discrete stages, not the iteration count within one stage. Every map is PCA50,
seed 42, final fit **50 normal / 0 early at an explicit perplexity 30**, and
references the existing fast `tsne_chain50_250_1000_final50` (unchanged, production
perplexity rule). The `tsne_geo4_*` level sizes are 256 → 1024 → 4096; `geo8` is
256 → 2048. The per-level normal-iteration counts are the **source-of-truth**
schedule **`round(50·√(7153/size))`** for the frozen 7,153-track library-root
cohort — written as explicit constants, so the schedule never depends on a
runtime `n`. The `geo4` values are 264/132/66 (the `round` of
`50·√(7153/size)` at sizes 256/1024/4096); `geo8` shares the 256 and 2048 rows →
264/93; `combined_double` doubles the geo4 counts → 528/264/132:

| Recipe | levels (size → …) | level budgets (normal/early) | per-level perplexities | final budget |
|--------|-------------------|------------------------------|------------------------|--------------|
| `tsne_chain50_250_1000_final50` (reference) | 50 → 250 → 1000 | 100/100, 100/0, 100/0 | production | 50 / 0, p30 (production) |
| `tsne_geo4_fixed` | 256 → 1024 → 4096 | 100/100, 100/0, 100/0 | 30, 30, 30 | 50 / 0, p30 |
| `tsne_geo4_sqrt` | 256 → 1024 → 4096 | 264/100, 132/0, 66/0 | 30, 30, 30 | 50 / 0, p30 |
| `tsne_geo4_perplexity` | 256 → 1024 → 4096 | 100/100, 100/0, 100/0 | **80, 40, 30** | 50 / 0, p30 |
| `tsne_geo4_combined` | 256 → 1024 → 4096 | 264/100, 132/0, 66/0 | **80, 40, 30** | 50 / 0, p30 |
| `tsne_geo8_combined` | 256 → 2048 | 264/100, 93/0 | **80, 30** | 50 / 0, p30 |
| `tsne_geo4_combined_double` | 256 → 1024 → 4096 | **528**/100, **264**/0, **132**/0 | **80, 40, 30** | 50 / 0, p30 |

The 80 perplexity on the 256-row first level is valid (80 < 256, and the existing
perplexity clamp guards smaller datasets), so no clamp fires on the first stage.
`tsne_geo4_fixed` is the axis baseline: `tsne_geo4_sqrt` isolates the
normal-iteration schedule (vs fixed 100s), `tsne_geo4_perplexity` isolates the
per-level perplexity axis (vs all-30), and `tsne_geo4_combined` moves both at once.
`tsne_geo8_combined` asks whether *fewer, bigger* stages (×8 instead of ×4) behave
differently; `tsne_geo4_combined_double` doubles the normal counts (final stays
50/0, not doubled) to separate the total budget from the schedule. Requested
per-level perplexities are stored per recipe (identity); each artifact's
`recipe` record carries both the requested `tsne_level_perplexities` and the final
`perplexities`.

**The `hierarchy-fractions` set** makes the multilevel chain **size-adaptive**:
instead of fixed absolute level sizes (the `hierarchy-heuristic` recipes), each
recipe names level sizes as **fractions of the actual dataset** — `round(n·f)` at
runtime — so the same recipe adapts to any library size while keeping its
count-axis geometry. Every map is PCA50, seed 42, final fit **50 normal / 0 early
at an explicit perplexity 30**, flat per-level perplexity 30, and references the
existing fixed `tsne_geo8_combined` (256→2048, perps 80/30). The per-level normal
budgets come from the automatic **`round(50·√(n/count))`** formula (no fixed total
budget), so the normal-iteration axis is deliberately **sqrt-linked** to the count
axis:

| Recipe | fractions (of n) | counts @7153 | normal budgets @7153 | per-level perps | final |
|--------|------------------|--------------|----------------------|-----------------|-------|
| `tsne_geo8_combined` (reference) | fixed 256→2048 | 256, 2048 | 264/100, 93/0 | 80, 30 | 50 / 0, p30 |
| `tsne_fraction_g4` | 1/16 → 1/4 | 447, 1788 | 200/100, 100/0 | 30, 30 | 50 / 0, p30 |
| `tsne_fraction_g8` | 1/64 → 1/8 | 112, 894 | 400/100, 141/0 | 30, 30 | 50 / 0, p30 |
| `tsne_fraction_g16` | 1/256 → 1/16 | 28, 447 | 799/100, 200/0 | **9**, 30 | 50 / 0, p30 |

`g4`/`g8`/`g16` compare the fraction galleries to each other and to the fixed
`tsne_geo8_combined` reference (descriptively — the reference's 80/30 perps are
not an isolated control). `tsne_fraction_g16`'s 28-row first level cannot hold
perplexity 30, so its recorded first-level perp is **9** (the backend clamp), disclosed
here rather than hidden. Compare the three fraction chains to each other and to
the favorite `tsne_geo8_combined` by ear and ratings, not spread.

**The `hierarchy-density` set** asks **how deep a fixed-budget fraction chain can
go**. Every map is PCA50, seed 42, final fit 50/0 at explicit perplexity 30, flat
per-level perplexity 30, and shares a fixed **TOTAL normal budget of 541** across
all coarse levels — split by **inverse-sqrt-of-fraction weight** (largest
remainder, at least 1 iteration per level) — the **same allocation policy at every
depth**. `tsne_density2` (1/64→1/8, 2 levels) is the 2-level baseline and reproduces
the auto `tsne_fraction_g8` budget (400/141); `density4/8/16` log-space the same
1/64→1/8 range into 4/8/16 levels; `continuous16` keeps density16's 16 fractions
but gives the whole chain a total-normal budget of 16 (so **1 normal iteration per
stage**, early 100 on the first) to isolate what a *dense rebuild* of many tiny
stages costs vs the full-budget `density16`:

| Recipe | fractions (of n) | levels | normal budgets @7153 (sum 541) | per-level perps | final |
|--------|------------------|--------|-------------------------------|-----------------|-------|
| `tsne_geo8_combined` (reference) | fixed 256→2048 | 2 | 264, 93 | 80, 30 | 50 / 0, p30 |
| `tsne_density2` | 1/64 → 1/8 | 2 | 400, 141 | 30, 30 | 50 / 0, p30 |
| `tsne_density4` | 1/64 → 1/8 | 4 | 211, 149, 106, 75 | 30 ×4 | 50 / 0, p30 |
| `tsne_density8` | 1/64 → 1/8 | 8 | 107, 93, 80, 69, 59, 51, 44, 38 | 30 ×8 | 50 / 0, p30 |
| `tsne_density16` | 1/64 → 1/8 | 16 | 54, 50, 47, 44, 41, 38, 36, 33, 31, 29, 27, 25, 24, 22, 21, 19 | 30 ×16 | 50 / 0, p30 |
| `tsne_continuous16` | 1/64 → 1/8 | 16 | 1 per stage (budget 16) | 30 ×16 | 50 / 0, p30 |

The counts/normals above are for the frozen 7,153-track library-root cohort; the
resolver recomputes them from the actual `n`, so the budget allocation "moves
slightly" with the rounded counts. The depth comparison is `density2` vs `4` vs
`8` vs `16` under the same 541-allocation policy; `continuous16` **stresses** the
per-stage affinity-rebuild overhead of a dense chain (many tiny separate stages)
versus `density16` — it is NOT a literal continuous (per-iteration affinity
update) fit. This is exploratory: judge by ear, ratings, and the scaling
benchmark, not spread.

> **Why fractions?** A size-adaptive chain keeps a recipe's *relative* geometry
> stable across library sizes, so a single recipe can be benchmarked and compared
> at many sizes without a rewrite per `n`. The two fraction fields and the
> `tsne_perplexity_gamma` field are the building blocks: gamma *schedules* the
> per-level perplexity, and the `hierarchy-gamma` set below varies it over the
> `tsne_density4` geometry.

**The `hierarchy-gamma` set** isolates the **per-level perplexity schedule** (the
`tsne_perplexity_gamma` exponent) on an otherwise-identical chain. Every map is
the `tsne_density4` geometry — PCA50, seed 42, final fit **50/0 at explicit
perplexity 30**, fractions 1/64→1/8 ×4 (counts @7153: 112/224/447/894), fixed
**TOTAL normal budget 541** (211/149/106/75, first level early 100). The only
difference is the coarse per-level perplexity schedule

```
p(t) = 30 + 50·(1 − t)^gamma        t = log(count/(n/64)) / log(64)
```

where `t` is the level's log-scaled position between `n/64` and `n` (clamped
0..1), so the last coarse level (n/8) always sits at `t = ½` → `p = 55` for gamma
1, and every schedule converges on the final fit's perplexity 30. The existing
`tsne_density4` (gamma `None` = flat 30 per level) is the **flat-30 control** and
is listed first as the reference:

| Recipe | fractions (of n) | per-level perps @7153 (approx) | final |
|--------|------------------|-------------------------------|-------|
| `tsne_density4` (reference, gamma `None`) | 1/64 → 1/8 | 30, 30, 30, 30 | 50 / 0, p30 |
| `tsne_gamma0` | 1/64 → 1/8 | 37, 74.3, 80, 80 (clipped high) | 50 / 0, p30 |
| `tsne_gamma05` | 1/64 → 1/8 | 37, 74.3, 71, 65 | 50 / 0, p30 |
| `tsne_gamma1` | 1/64 → 1/8 | 37, 72, 63, 55 | 50 / 0, p30 |
| `tsne_gamma2` | 1/64 → 1/8 | 37, 65, 52, 42 | 50 / 0, p30 |
| `tsne_gamma4` | 1/64 → 1/8 | 37, 54, 40, 33 | 50 / 0, p30 |

All five gamma variants share the same 541-iteration coarse budget and final
50/0 fit. Gamma 0 requests 80 throughout the coarse stages; higher gamma
drops the middle/late requested perplexities faster. openTSNE caps them at
`(count - 1) / 3`, so all gamma variants start at effective perplexity 37
on the 112-track level. These are clipped gamma curves, not unconstrained
80-to-30 curves. Compare them against the flat-30 `tsne_density4` control.

**The `hierarchy-combination` set** pairs a continuous-stress chain with a
**wider fraction span** and a gamma-4 schedule on that same span, then asks
whether adding a third dimension costs or preserves quality. All three 2-D
recipes keep the 16-level, 1-normal-per-stage continuous-stress budget, the
**final 50 normal / 0 early fit at perplexity 30**, and PCA50 / seed 42. The
wide recipes move the levels from 1/64→1/8 to **1/256→1/16**; `wide_gamma4`
additionally schedules the coarse perplexity with
`p(t) = 30 + 50·(1−t)^gamma`, anchoring `t` at `n/256` (log span 256) instead
of the hard-coded `n/64`. Each base has an explicitly registered `_3d` sibling
(`dims=3`, `group=<base id>`); everything except identity/dimension/group is
inherited, so a pair differs only by output dimension:

| Recipe | fractions (of n) | counts @7153 | normal budgets | per-level perps @7153 | final |
|--------|------------------|--------------|----------------|-----------------------|-------|
| `tsne_continuous16` | 1/64 → 1/8 | 112 … 894 | 1 per stage (budget 16) | requested 30 ×16; effective 30 ×16 | 50 / 0, p30 |
| `tsne_continuous16_wide` | 1/256 → 1/16 | 28 … 447 | 1 per stage (budget 16) | requested 30 ×16; effective 9, 11, 13, 16, 19.3, 23, 28, then 30 ×9 | 50 / 0, p30 |
| `tsne_continuous16_wide_gamma4` | 1/256 → 1/16 | 28 … 447 | 1 per stage (budget 16) | requested `30+50(1−t)^4`; effective 9 … 42.0 … 33.1 | 50 / 0, p30 |

The labels name the **requested** schedule (e.g. `perps 30+50(1−t)^4`); openTSNE
clips low-count levels at `(count − 1) / 3`, so the **effective** values are
lower on the small wide levels. The artifact's `effective_schedules` records the
exact `(count, normal, early, perp)` stages per size — read that, not the label,
when the schedule matters. The 3-D siblings fit the same input matrix and are
aligned onto `tsne_continuous16_3d`, so their x/y need not coincide with the
2-D maps. Compare pairs by ear, ratings and the scaling benchmark below.

> **Measured finding (descriptive, not a cross-genre proof).** On this library's
> frozen default-recipe composite vectors, the 16-level `wide_gamma4` chain with the
> 50/0 final fit was the visual winner of this gallery, and the scaling benchmark's
> wall-clock numbers below are for that same frozen mixed-source input. The
> schedule depends only on the runtime row count and output dimensionality, never
> on the source weights, so it was adopted for the production default *and*
> `genre_full` on that weight-agnostic basis. That adoption decision did **not**
> rate or benchmark the genre vectors themselves; later production benchmarks
> measure the `genre_full` pipeline's wall-clock and fidelity, not whether genre
> representations beat another representation. Reading the adoption as evidence of
> genre-lens quality would still be an extrapolation. No further selection
> experiment is planned.

### Scaling benchmark (`backend/benchmark_layout_scaling.py`)

A reproducible **wall-clock** benchmark for how `fit_recipe` scales with library
size, from a small subset up to the full eligible track set — reusing the same
canonical composite input the interactive projection uses (frozen
sources/weights/mask/blend, `prepare_projection_input`). It writes no layout,
fit-cache, gallery, or MP3 data (the cache constructors may still create
schema/metadata/WAL) and writes one JSON artifact. Its cohort is the
**library-root** selection (`load_infos` scope `library`, external roots
excluded — 7,153 tracks here); it is **not** the all-roots production cohort.
**Below ~10k tracks** openTSNE auto-selects its Barnes-Hut engine, so the whole
`500 → full-library` range runs BH — no gradient-method or FFT branch is
implemented, so a library-root cohort of `>= 10k` eligible tracks is **refused**
rather than measured on the wrong engine. That refusal does not make the
15,571-track all-roots production set unmeasurable: use
`backend/benchmark_production_projection.py` for it. The numbers are timings on
this machine/this data, not cross-library speedup claims.

```bash
# Default recipes × sizes 500,1000,2000,4000 (+ the full eligible set, always last)
.venv/bin/python -m backend.benchmark_layout_scaling \
  --data-dir data --sizes 500,1000,2000,4000 --repeats 3 --threads 8 --seed 42
# → data/layout-evaluation/hierarchy-scaling.json

# The five hierarchy-tuning chain recipes over the full eligible set (always
# appended/deduped, so --sizes 7153 collapses to it on this library).
.venv/bin/python -m backend.benchmark_layout_scaling \
  --data-dir data --sizes 7153 \
  --recipes tsne_chain50_250_1000,tsne_chain50_250_1000_final50,tsne_chain50_250_1000_coarse10,tsne_chain50_250_1000_coarse10_final50,tsne_chain50_250_1000_final2000 \
  --repeats 3 --threads 8 --seed 42 \
  --output data/layout-evaluation/hierarchy-tuning-timings.json

# The five hierarchy-distribution chain recipes (reference tsne_chain50_250_1000_final50
# + four normal-iteration distribution variants) over the full eligible set.
.venv/bin/python -m backend.benchmark_layout_scaling \
  --data-dir data --sizes 7153 \
  --recipes tsne_chain50_250_1000_final50,tsne_chain50_250_1000_early1000_final50,tsne_chain50_250_1000_middle1000_final50,tsne_chain50_250_1000_late1000_final50,tsne_chain50_250_1000_spread400_final50 \
  --repeats 3 --threads 8 --seed 42 \
  --output data/layout-evaluation/hierarchy-distribution-timings.json

# The seven hierarchy-heuristic geometric chains over the full eligible set
# (the reference tsne_chain50_250_1000_final50 + the six tsne_geo* recipes).
.venv/bin/python -m backend.benchmark_layout_scaling \
  --data-dir data --sizes 7153 \
  --recipes tsne_chain50_250_1000_final50,tsne_geo4_fixed,tsne_geo4_sqrt,tsne_geo4_perplexity,tsne_geo4_combined,tsne_geo8_combined,tsne_geo4_combined_double \
  --repeats 3 --threads 8 --seed 42 \
  --output data/layout-evaluation/hierarchy-heuristic-timings.json

# The four hierarchy-fractions size-adaptive chains over the full eligible set
# (the fixed tsne_geo8_combined reference + the three fraction recipes).
.venv/bin/python -m backend.benchmark_layout_scaling \
  --data-dir data --sizes 7153 \
  --recipes tsne_geo8_combined,tsne_fraction_g4,tsne_fraction_g8,tsne_fraction_g16 \
  --repeats 3 --threads 8 --seed 42 \
  --output data/layout-evaluation/hierarchy-fractions-timings.json

# The six hierarchy-density chains over the full eligible set (the reference plus
# density2/4/8/16 and the 1-iteration-per-stage continuous16 stress).
.venv/bin/python -m backend.benchmark_layout_scaling \
  --data-dir data --sizes 7153 \
  --recipes tsne_geo8_combined,tsne_density2,tsne_density4,tsne_density8,tsne_density16,tsne_continuous16 \
  --repeats 3 --threads 8 --seed 42 \
  --output data/layout-evaluation/hierarchy-density-timings.json

# The six hierarchy-gamma chains over the full eligible set (the flat-30
# tsne_density4 reference + the five gamma-schedule variants on the same geometry).
.venv/bin/python -m backend.benchmark_layout_scaling \
  --data-dir data --sizes 7153 \
  --recipes tsne_density4,tsne_gamma0,tsne_gamma05,tsne_gamma1,tsne_gamma2,tsne_gamma4 \
  --repeats 3 --threads 8 --seed 42 \
  --output data/layout-evaluation/hierarchy-gamma-timings.json

# The combined fraction/density/gamma ramp run (14 recipes, one artifact; this
# is the scored report the TODO/follow-up refers to).
.venv/bin/python -m backend.benchmark_layout_scaling \
  --data-dir data --sizes 7153 \
  --recipes tsne_geo8_combined,tsne_fraction_g4,tsne_fraction_g8,tsne_fraction_g16,tsne_density2,tsne_density4,tsne_density8,tsne_density16,tsne_continuous16,tsne_gamma0,tsne_gamma05,tsne_gamma1,tsne_gamma2,tsne_gamma4 \
  --repeats 3 --threads 8 --seed 42 \
  --output data/layout-evaluation/hierarchy-ramps-timings.json

# The paired hierarchy-combination run: three 2-D bases + their three explicit
# 3-D siblings, same sizes/seed/threads, over 1k → 2k → 4k → full. 7153 collapses
# to the full eligible set on this library (always appended/deduped).
.venv/bin/python -m backend.benchmark_layout_scaling \
  --data-dir data --sizes 1000,2000,4000,7153 \
  --recipes tsne_continuous16,tsne_continuous16_wide,tsne_continuous16_wide_gamma4,tsne_continuous16_3d,tsne_continuous16_wide_3d,tsne_continuous16_wide_gamma4_3d \
  --repeats 3 --threads 8 --seed 42 \
  --output data/layout-evaluation/hierarchy-combination-timings.json
```

Each size is a seeded nested prefix of one row permutation (a size's tracks are a
superset of every smaller size's). Per size, every recipe gets one **warmup** fit
(excluded from timings and scoring), then `--repeats` measured **rounds** — each
round visits every recipe in its own deterministic shuffle, so no recipe
systematically runs first/last. Because the benchmark gathers one canonical
PCA50 t-SNE input, it **accepts only** recipes with `pca_dim=50`, no source mix,
and a `tsne` / `tsne_sample_init` / `tsne_multilevel_init` method — a UMAP or
source-mix recipe would fit a different input geometry and is rejected up front,
before the gather. Output dimension is **not** a compatibility axis: a 2-D
recipe and its explicit 3-D sibling fit the same matrix and are scored by the
same metrics, so a paired run lists both. The numerical BLAS/OpenMP threads are bound
with **threadpoolctl** around each size's whole fit block, and each fit's
wall-clock is taken *inside* that context immediately before the call, excluding
threadpoolctl setup. The JSON records the actual per-round recipe orders, raw
wall-clock samples of the whole `fit_recipe` (its own PCA + t-SNE) plus the
per-stage keys, median/min/max summaries, and three **untimed** quality metrics
for the first measured map of each recipe: neighbourhood recall @10 (the
back-compat `fidelity` field) and @100 (`fidelity100`, both Euclidean over the
canonical PCA50), and a broad-distance **Spearman** rank correlation
(`broad_distance_spearman`, 2048 seeded random pair distances — PCA50 Euclidean
vs the map's low-dimensional Euclidean distance, 2-D or 3-D, the same pairs for
every recipe at a size). The Spearman is a rank-correlation **proxy** of whether
the layout keeps broad pairwise distances ranked like the high-dim space — NOT
proof that t-SNE distances are globally meaningful; a degenerate layout records
it as a null score rather than a fabricated zero. These are indicative, not proof
of global quality. The envelope records each recipe's params + revision so a run
is reproducible by id alone would not be. For size-adaptive multilevel recipes
each size's entry also records an `effective_schedules` map: the exact
`(count, normal, early, perp)` coarse stages the recipe resolved at that size
(resolved from the actual subset size, not the full library), so a run tracks
what each measured fit actually did. The JSON is written with `complete: false`
**before** the expensive gather (so an older completed report cannot masquerade
as this run) and checkpointed again after each size, so a bounded run keeps its
partial results.

**Paired 2-D/3-D report.** Run a base and its registered `_3d` sibling in the
same invocation so they share sizes, seed, threads, distance pairs and the
warmup/round protocol. Then pair them by `group` (never by string suffix) and
report the speed ratio plus the same-size matched metrics:

```bash
.venv/bin/python - <<'PY'
import json
from backend.embeddings.layout_evaluation import RECIPES_BY_ID

pairs = {}
for rid, recipe in RECIPES_BY_ID.items():
    pairs.setdefault(recipe.group or rid, {})[recipe.dims or 2] = rid

report = json.load(open("data/layout-evaluation/hierarchy-combination-timings.json"))
for size, block in report["results"].items():
    for group, by_dims in sorted(pairs.items()):
        if 2 not in by_dims or 3 not in by_dims:
            continue
        if by_dims[2] not in block["recipes"] or by_dims[3] not in block["recipes"]:
            continue
        two, three = (block["recipes"][by_dims[d]] for d in (2, 3))
        ratio = three["timings"]["total"]["median"] / two["timings"]["total"]["median"]
        print(
            f"{size:>5} {group:<34} 3D/2D speed {ratio:.2f}x  "
            f"recall@10 {two['fidelity']:.3f}/{three['fidelity']:.3f}  "
            f"broad rho {two['broad_distance_spearman']:.3f}/"
            f"{three['broad_distance_spearman']:.3f}"
        )
PY
```

The 2-D recipe's params omit `dims`/`group` (`None` means 2-D / its own id); the
3-D sibling records `dims: 3` and `group: <base id>`. A ratio is only meaningful
when both rows come from the **same** benchmark invocation, size and seed.

Requires **openTSNE** (the t-SNE recipes); the four UMAP recipes need
**umap-learn**. **threadpoolctl** (already installed via scikit-learn) is declared
as a direct dependency for the thread-cap. All are in `requirements.txt`. Any
missing dependency is reported as a clear per-recipe error status, never silently
skipped.

### Production projection benchmark (`backend/benchmark_production_projection.py`)

Bounded benchmark of the actual production path at any library size: canonical
recipe → composite gather + folder-semantic weighting → production PCA50 →
production multilevel `_fit_tsne`. It records weighted/unweighted prepare,
`basis_estimate` (a difference estimate, not an isolated timer), PCA, per-stage
fit timings, resolved engines, threadpool settings and fidelity. It writes no
MP3 tags, production layouts, or in-flight fits; cache constructors may create
schema/metadata/WAL. `pipeline_estimate = prepare + PCA + fit` excludes HTTP,
queue wait, Nyström placement, the weight Jacobian, and the 3-D reference fit.

```bash
.venv/bin/python -m backend.benchmark_production_projection \
  --cases native,folders-off,bh,theta07,theta09,exag2,exag4 \
  --dims 2,3 --scope all --recipe genre_full --repeats 3 \
  --output data/layout-evaluation/production-axes.json
```

Full-scope measurements on this library (15,574 fingerprinted → **15,571
eligible**, genre_full, all roots) are recorded in
`data/layout-evaluation/production-axes.json`; the earlier pilot artifact
`data/layout-evaluation/production-benchmark.json` keeps its raw measurements
and old double-counted `end_to_end_cold` field. These artifacts timed the
benchmark's `native` case: the **legacy exact-SVD `full` folder basis at folder
boost 6 / depth 3**, recorded in their `recipe_params`. They are historical
full-basis numbers, not the current centroid-only 2.4 production prepare. In that
historical configuration the weighted prepare dominated: **32.584 s** of a
**34.49 s** 2-D estimate (94.5%) and a **38.72 s** 3-D estimate (84.2%),
excluding the HTTP/queue/2-D-reference overhead above. Production's centroid
prepare is cheaper (see `audio_pipeline.md` → *Projection methods*); no current
full-scope cold centroid prepare is recorded in that artifact, so do not quote
32.584 s as today's production cost. The 2-D full-data level resolves to FFT
above 10k rows (coarse levels stay BH); 3-D pins BH.

The server serves the same directory it stores eval artifacts in:
`<data-dir>/layout-evaluation` (default `data/layout-evaluation`). Keep the
generator's `--output` in step with the server's `TRACKSPACE_DATA_DIR` so the
gallery reads what you generated.

Read-only GET routes (no write/upload surface):

- `GET /api/layout-evaluation` — the manifest: dataset fingerprint, sources/weights,
  versions, backend, and every recipe's status, timing, `dims` and paired `group`.
- `GET /api/layout-evaluation/<id>` — one recipe's `positions` (`{path, x, y}`, plus a
  finite `z` on 3-D artifacts), revision, `dims`/`group`, and timing. Only recipes the
  current manifest marks `complete` are served; traversal and unknown ids are rejected,
  and missing/corrupt/non-finite artifacts (including a 3-D artifact without a finite
  `z` on every position) return a JSON error.
- `?set=<name>` on either route selects a **named set**: artifacts under a fixed
  subdirectory of the base (`budget` → `<data-dir>/layout-evaluation/budget`,
  `scale` → `<data-dir>/layout-evaluation/scale`,
  `early` → `<data-dir>/layout-evaluation/early`,
  `mixes` → `<data-dir>/layout-evaluation/mixes`,
  `ablation` → `<data-dir>/layout-evaluation/ablation`,
  `hierarchy` → `<data-dir>/layout-evaluation/hierarchy`,
  `hierarchy-budget` → `<data-dir>/layout-evaluation/hierarchy-budget`,
  `hierarchy-depth` → `<data-dir>/layout-evaluation/hierarchy-depth`,
  `hierarchy-tuning` → `<data-dir>/layout-evaluation/hierarchy-tuning`,
  `hierarchy-distribution` → `<data-dir>/layout-evaluation/hierarchy-distribution`,
  `hierarchy-heuristic` → `<data-dir>/layout-evaluation/hierarchy-heuristic`,
  `hierarchy-fractions` → `<data-dir>/layout-evaluation/hierarchy-fractions`,
  `hierarchy-density` → `<data-dir>/layout-evaluation/hierarchy-density`,
  `hierarchy-gamma` → `<data-dir>/layout-evaluation/hierarchy-gamma`,
  `hierarchy-combination` → `<data-dir>/layout-evaluation/hierarchy-combination`,
  `genre` → `<data-dir>/layout-evaluation/genre`,
  `genre-directed` → `<data-dir>/layout-evaluation/genre-directed`,
  `genre-logits-ab` → `<data-dir>/layout-evaluation/genre-logits-ab`,
  `folder-separation` → `<data-dir>/layout-evaluation/folder-separation`,
  `folder-cheap` → `<data-dir>/layout-evaluation/folder-cheap`). Without it, the base
  (eight-map) gallery is served. Only allowlisted names (`base`, `budget`,
  `scale`, `early`, `mixes`, `ablation`, `hierarchy`, `hierarchy-budget`,
  `hierarchy-depth`, `hierarchy-tuning`, `hierarchy-distribution`,
  `hierarchy-heuristic`, `hierarchy-fractions`, `hierarchy-density`,
  `hierarchy-gamma`, `hierarchy-combination`, `genre`, `genre-directed`,
  `genre-logits-ab`, `folder-separation`, `folder-cheap`) are
  accepted — anything else is a 400.

### Browser gallery (blind eval)

Run the app normally (dev or production), then open it with the eval query param:

```
http://localhost:5173/?layout_eval=1          # dev (Vite)
http://127.0.0.1:5111/?layout_eval=1          # production build
http://localhost:5173/?layout_eval=1&layout=pca2   # open on a specific map
http://localhost:5173/?layout_eval=1&set=budget    # the four-map budget set
http://localhost:5173/?layout_eval=1&set=scale     # the four-map scale set
http://localhost:5173/?layout_eval=1&set=early     # the three-map early-phase set
http://localhost:5173/?layout_eval=1&set=mixes     # the three-map source-weight set
http://localhost:5173/?layout_eval=1&set=ablation  # the three-map dropped-block set
http://localhost:5173/?layout_eval=1&set=hierarchy # the eight-map sample-initialisation set
http://localhost:5173/?layout_eval=1&set=hierarchy-budget # the ten-map coarse-budget sweep set
http://localhost:5173/?layout_eval=1&set=hierarchy-depth # the eight-map multilevel chain set
http://localhost:5173/?layout_eval=1&set=hierarchy-tuning # the five-map chain-budget tuning set
http://localhost:5173/?layout_eval=1&set=hierarchy-distribution # the five-map normal-iteration distribution set
http://localhost:5173/?layout_eval=1&set=hierarchy-heuristic # the seven-map geometric chain set
http://localhost:5173/?layout_eval=1&set=hierarchy-fractions # the four-map size-adaptive fraction set
http://localhost:5173/?layout_eval=1&set=hierarchy-density # the six-map fixed-budget depth set
http://localhost:5111/?layout_eval=1&set=hierarchy-gamma # gamma-curve comparison
http://localhost:5111/?layout_eval=1&set=hierarchy-combination # paired 2-D/3-D continuous comparison
http://localhost:5111/?layout_eval=1&set=folder-separation # production folder weighting on/off (paired 2-D/3-D)
http://localhost:5111/?layout_eval=1&set=folder-cheap # six-group cheap folder-basis comparison (paired 2-D/3-D)
http://localhost:5173/?layout_eval=1&set=genre     # the genre-space maps gallery
```

The gallery is a **read-only, single-bundle** mode: the same controller, canvas and
hover-audio path run, with a compact toolbar overlay. Maps are shown as **Map A, Map B,
…** (blind — the recipe name stays hidden until you toggle *Reveal recipe & timings*,
so first impressions are not coloured by the algorithm). A **navigation group** gets
one letter: a 2-D recipe and its explicit 3-D sibling share the letter (ratings stay
per artifact), so a paired set exposes one step per recipe rather than one per
dimensionality. Letters follow the **manifest order of a group's first recipe** (the
API serialises the manifest's JSON keys sorted, so `baseline` is always Map A). The
view is restricted to the artifact's own tracks and nothing is computed on demand:
no projection fetch, no analysis SSE, no lens/scope/weight work.

From the toolbar you can:

- **Select / ‹ ›** — switch groups; positions animate on the shared canvas frame and
  selection, colour and filters stay stable.
- **3** — for a paired group, switch between its 2-D and 3-D artifact. The 3-D map
  rotates on the existing camera blend; the toolbar reads *2D/3D · 3 to toggle*.
  An unpaired group stays 2-D and the key is a no-op.
- **Reset view** — return the camera to the full map.
- **Reveal recipe & timings** — surface each group's base algorithm name and the
  selected artifact's fit timings (PCA/t-SNE/UMAP/alignment seconds) from the
  manifest.
- **Ratings** — per **artifact** (the 2-D and 3-D sibling are rated separately), four
  1–5 scores (musical neighbours, global organisation, crowding, overall) plus a
  free-text note, persisted in `localStorage` keyed by the **dataset fingerprint +
  artifact revision** so re-runs and re-reveals stay isolated. Clicking the current
  score clears it.
- **Export ratings** — downloads this dataset's ratings as a JSON file
  (`layout-eval-<first 8 chars of the dataset fingerprint>.json`). Records keep the
  **artifact** recipe id, not the group id. Keep the
  exports under `data/layout-evaluation/ratings-exports/` if you collect them
  with the artifacts.
- **Exit ↺** — reloads the app without the eval query, returning to normal mode.

Eval mode is **session-only and separate from production settings**: it never
persists view settings (`Model.saveLS` is a no-op while it is active) and the
ratings live under their own localStorage key, so browsing the gallery cannot
touch normal-mode preferences.

Audio works exactly as in the main app (hover preview → `/api/audio`). Because the
browser blocks autoplay until a trusted gesture, click anywhere in the toolbar or map
once before the first hover playback. The mode is enforced read-only: every owned
write path (`lib/api.ts` POST/PATCH/DELETE/PUT) and every lens/scope/weight compute
entry point is inert while it is active, so no write can reach the server.

> Regenerating the maps (e.g. after a cache change or a new recipe set) is the CLI
> above — the gallery reads whatever `<data-dir>/layout-evaluation` currently holds.
> Ratings are keyed by fingerprint + revision, so a regenerated dataset gets a fresh
> rating sheet even under the same maps.
