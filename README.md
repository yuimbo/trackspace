# Trackspace

Organise music in a 2D tag space. Tags are stored as private ID3 frames (`TXXX:trackspace:*`) directly in your mp3 files.

## Setup

```bash
cd tools/trackspace

# Python deps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Frontend deps
cd frontend && npm install && cd ..
```

The UI is **Vite + TypeScript** with **Alpine.js** (e.g. shortcuts modal) and **HTMX** (folder tree HTML from Flask `GET /partials/folder-tree`).

## Dev mode (hot-reload)

```bash
cd frontend
npm run dev:all
```

Opens Vite at [http://localhost:5173](http://localhost:5173). `/api` and `/partials` are proxied to Flask on port 5111.

## Production

```bash
# Build frontend
cd frontend && npm run build && cd ..

# Run server
source .venv/bin/activate
python -m backend.app
```

**`python -m backend.app` serves the prebuilt `frontend/dist`.** A change to any
`frontend/src` file is not picked up until `npm run build` runs again — including a
change to the shape of a wire payload the backend emits, which will break the served
bundle until it is rebuilt. For frontend work use `npm run dev:all` above, which serves
the source directly.

Then open [http://127.0.0.1:5111](http://127.0.0.1:5111).

**Options:**

```
python -m backend.app /path/to/music # optional initial music root
python -m backend.app -p 8080        # custom port
python -m backend.app --host 0.0.0.0 # expose on LAN
```

**Genre is the default (the Genre map).** The app opens on the **Genre** lens — the
`genre_full` projection: a canonical 7-source recipe (logits .40 / maest .20 /
layer-7 .25 / MERT-L19 .15; the other three at zero) fitted by the production
multilevel t-SNE chain (see `audio_pipeline.md` → *Projection methods*) with a 50/0
final-fit budget (the gallery 500/250 is an explicit override, not the default).
MAEST layer-7 is a **default-on** analysis block; the independent MERT encoder
runs by default too, but only once you pass `--trust-remote-code` (see the flags below).
The legacy Combined / MAEST / Folder-head maps are replaced by the Genre map
(CLAP and Rhythm lenses remain). Nothing downloads or executes MERT by itself — see the
flags below.

Run the full genre blocks (MAEST layer-7 + MERT) with the default start; add
`--trust-remote-code` to enable MERT:

```bash
.venv/bin/python -m backend.app /path/to/music \
  --trust-remote-code
```

That is **local-only**: it never downloads MERT weights (weights are fetched only with
the separate `--allow-download` flag) and never runs untrusted remote code. MERT's
weights are **CC-BY-NC 4.0 (~2.5 GB)** and it ships pinned repo-specific modeling code;
the CC-BY-NC licence is accepted by default for this **personal project**, but running
MERT always requires the explicit `--trust-remote-code` flag.

- `--no-genre-full` — disable the genre_full blocks (Genre is the default): no MAEST
  layer-7 and no MERT worker, and the legacy Combined map replaces the Genre map. The
  legacy `default` recipe is retained internally for compatibility.
- `--no-mert` — keep layer-7 but disable MERT (a MAEST-only genre recipe; faster, no
  download).
- `--trust-remote-code` — **required to run MERT**: allow MERT's pinned repo-specific
  modeling code.
- `--allow-download` — **separate, optional**: fetch uncached MERT weights. Without it
  the server keeps `local_files_only` and refuses to download.

See *Genre lens & Genre-full rebuild* below for how the Genre view relates to clustering.

## Genre lens & Genre-full rebuild

The **Genre lens** is the default `genre_full` projection described above. The
server serves the genre recipe by default; `--no-genre-full`
disables its layer-7/MERT analysis blocks (the recipe id itself is always served, and a
disabled server's frontend falls back to the Combined map).

**Switching to the Genre lens is a view only.** It changes the *projection* (where
tracks sit on the canvas), never the cluster assignments — a cluster's membership and
the Microgenres panel are unchanged by which lens you are looking at. Re-clustering
into the genre space is a **separate, explicit** action: the Analysis panel's rebuild
sends `POST /api/clusters/rebuild` with `recipe=genre_full` (also gated on the server's
genre_full state). The plain rebuild button is a re-run, not a reset; only an explicit
`recipe=genre_full` switches the clustering weights to the genre vector.

**Genre map weights preview.** While the Genre map is active, its seven tunable sources
(logits / maest / layer-7 / MERT / rhythm / clap / supervised) live in the **View
options** panel. Dragging a slider previews the map through the in-memory weight
Jacobian (a draft is never written to disk and never triggers a request on its own);
**Apply** commits the vector — a cluster rebuild + exact re-layout — and **Reset**
restores the exact applied baseline without a round-trip. A source whose server stage is
off (or with no cached vectors yet) has a disabled slider.

**openTSNE iteration presets & overrides.** The view-options panel exposes the
full-fit `tsne_n_iter` and early-exaggeration `tsne_early_exaggeration_iter` as
named presets plus free numeric fields, persisted **per dimensionality** (keyed
by `d2`/`d3`, so every 2-D embedding-map lens shares the 2-D budget and 3-D keeps
its own separate budget):

- Presets: *default* (the active recipe's production multilevel budget — no
  override sent), *Fast* (100/100), *Balanced* (250/150), *Gallery* (500/250).
- Ranges: `tsne_n_iter` 50–2000, `tsne_early_exaggeration_iter` 0–1000 (0 disables
  early exaggeration). The iteration controls do not expose the canonical late
  exaggeration on the final fit (1.25); it is always applied unless a benchmark
  passes an explicit comparison value.
- A deviation from the active recipe's default is part of the **layout revision and
  the fit-cache key**, so a user-chosen budget never shares a fit with the recipe
  default (and `early=0` is treated as a real override, not silently restored to the
  default). Switching lens or 2-D↔3-D resyncs the control to that dimensionality's
  stored budget.

**Production projection.** Every map — 2-D and 3-D, default and `genre_full` — uses
the one production multilevel sample-init t-SNE chain. Its schedule, the per-axis
display normalisation and the cache version are single-sourced in
`audio_pipeline.md` → *Projection methods*; this README does not repeat those
constants. Fits are cached and persisted, so a cold fit is paid once and later
additions are placed with `TSNEEmbedding.transform` rather than a refit.

**Production folder weighting** uses the **centroid-only** semantic basis
(hierarchical contrast directions, no within-folder PC SVD) with depth emphasis.
The exact mode/amplitude live in `audio_pipeline.md` → *Semantic weighting*; the
centroid basis is cheap, so a cold first run no longer pays the exact-SVD
full-basis cost.

## Docs

- `AGENTS.md` — entry point and core invariants (start here for contributing).
- `architecture.md` — system structure, frontend MVC, backend packages, HTTP API.
- `ui_conventions.md` — frontend interaction rules (canvas, tags, folders, hotkeys).
- `audio_pipeline.md` — fingerprinting → embeddings → projection → clustering, and the job queue.
- `embedding-next-steps.md` — the embedding/analysis implementation plans (selected
  `genre_full` adoption, optimization, MuQ evaluation, validation).
- `design-patterns.md` — shared interaction patterns (navigation targets, hover state, coordinators, input controls).
- `trackspace-projection-methods.md` — research notes on parametric/kernel projection candidates (background, not normative).
- `layout-evaluation.md` — offline eval recipe catalogue, CLI, gallery and benchmark protocols.
- `TODO.md` — deferred work and the product roadmap.

## Layout evaluation

Offline, reproducible gallery maps over the *same* cached embeddings, generated by
`backend/precompute_layout_evaluation.py` and served read-only by the app. It never
writes the production layout cache, the fit cache, or MP3 tags.

```bash
# Base eight-map gallery → <data-dir>/layout-evaluation
python -m backend.precompute_layout_evaluation --set base

# Any other built-in set → <data-dir>/layout-evaluation/<set>
python -m backend.precompute_layout_evaluation --set hierarchy-gamma
python -m backend.precompute_layout_evaluation --set hierarchy-combination
```

`--set` takes its recipes, alignment reference and output directory from
`GALLERY_SETS`; `--recipes` remains available for an ad-hoc list. Then open the
gallery:

```
http://localhost:5173/?layout_eval=1&set=hierarchy-gamma   # dev
http://127.0.0.1:5111/?layout_eval=1&set=hierarchy-gamma   # production build
```

The built-in generator uses the **library-root cohort only** (7,153 tracks here,
external roots excluded). The all-roots production benchmark covers a different
**15,571-track** eligible set; keep the cohorts distinct when comparing numbers.
The `genre*` and `folder-*` sets are generated by their own tools, not `--set`.

Full recipe catalogue, every CLI option, the blind-gallery/ratings workflow, and
the scaling and production-benchmark protocols and caveats live in
`layout-evaluation.md`. Production projection policy (multilevel schedule, folder
basis, perplexity) is single-sourced in `audio_pipeline.md` → *Projection methods*.

## Offline genre benchmark (`data/genre-space/`)

Reproducible, read-only comparison of genre representations — MAEST versus MERT — over a
**frozen, labelled cohort** split by artist/release identity. It exists because broad-genre
purity cannot measure microgenre separation (see `TODO.md` → *Offline genre benchmark*).
All artifacts live under `data/genre-space/<set>/` (gitignored): `cohort.json`,
`vectors/`, `report/`, plus `directed-analysis.json`/`.md`, `layer7-performance.json`,
`logits-ab-report/`.

**Design (read before trusting a number):**

- **Folders are the weak labels.** A track's label is the first `--label-depth` path
  components under `--root` (depth 1 = broad genres; point `--root` at a subgenre folder
  for subgenre labels). The benchmark can only claim distinctions the labels encode.
- **Identity-disjoint 70/30 split** by artist / `artist::album`; no artist or release
  spans a split. Release needs the artist+album pair, which the real cache does not
  store, so on a real run only artist identity is available.
- **Frozen windows, deliberately different from production.** Extraction re-decodes the
  manifest's exact windows — **production cache outputs are never used** — so every model
  hears identical audio.
- **Complete-case scoring** on the intersection of tracks every recipe extracted; no
  zero-filling. Partial extraction coverage is reported; invalid provenance/checksums are
  rejected.
- **Dev-only tuning, retrieval eval.** PCA fitted / recipes selected on `dev` only;
  heldout tracks are the *queries*, candidates the whole cohort — retrieval, not a heldout
  classifier. A same-artist/release exclusion variant is scored for sanity.
- **`--limit 20` is compatibility-only** — a smoke check of plumbing, not quality; a real
  comparison uses a fresh scale-500 cohort and fresh directories.

```bash
# 1. Cohort — frozen manifest; read-only over the track cache (tracks must already be
#    scanned into data/track_cache.db and unchanged since; the cache is opened read-only)
python -m backend.experiments.genre_space cohort \
  --root /path/to/GENRES \
  --track-db data/track_cache.db \
  --output data/genre-space/smoke/cohort.json \
  --limit 20 --seed 42 --label-depth 1

# 2. Extract — re-decodes the manifest's exact windows, per model
python -m backend.experiments.genre_space extract \
  --manifest data/genre-space/smoke/cohort.json \
  --output data/genre-space/smoke/vectors \
  --model maest

# 2b. MERT — downloads/executes the pinned remote modeling code only with explicit flags
#     (trust_remote_code). Weights ~2.53 GB, CC-BY-NC 4.0; never auto-downloaded.
python -m backend.experiments.genre_space extract \
  --manifest data/genre-space/smoke/cohort.json \
  --output data/genre-space/smoke/vectors \
  --model mert --revision 0c0c8c6f6e04dcc39d96f94e0be9978a74848906 \
  --trust-remote-code --allow-download

# 3. Score — retrieval metrics over the complete-case intersection
python -m backend.experiments.genre_space score \
  --manifest data/genre-space/smoke/cohort.json \
  --artifacts data/genre-space/smoke/vectors \
  --output data/genre-space/smoke/report

# 4. (optional) Maps — one PCA50 -> openTSNE chart per --recipes id, for visual
#    inspection only. Transductive PCA fits ALL cohort rows (unlike score, which fits
#    PCA on dev rows only), so maps are display-only, not another scoring signal.
#    Writes coords_<id>.json; pass --gallery-output for a schema-1 blind-eval gallery
#    under data/layout-evaluation/genre (paths via --roots-file, default data/roots.json).
python -m backend.experiments.genre_space maps \
  --manifest data/genre-space/smoke/cohort.json \
  --artifacts data/genre-space/smoke/vectors \
  --output data/genre-space/smoke/maps \
  --gallery-output data/layout-evaluation/genre \
  --recipes 'maest_logits,maest_logits+maest@643-357'

#    Then open the gallery (production build):
#    http://127.0.0.1:5111/?layout_eval=1&set=genre
```

**Recipe sets.** `score` and `maps` take `--recipe-set baseline|directed|logits-ab`
(default `baseline`). `baseline` is the fixed 13-recipe sweep above. The two fixed presets
reuse existing cached vectors (no new inference), require every source and never produce a
partial set:

```bash
# Directed — all five contrastive options (logits vs final hidden vs layer-7 vs MERT19),
# one PCA50 -> openTSNE map each; fits all five ids when --recipes is omitted.
# Blind-eval UI: set=genre-directed.
python -m backend.experiments.genre_space maps \
  --manifest data/genre-space/first/cohort.json \
  --artifacts data/genre-space/first/vectors \
  --output data/genre-space/first/directed-maps \
  --gallery-output data/layout-evaluation/genre-directed \
  --recipe-set directed

# Logits A/B — genre_full vs the same recipe with the logits removed; reuses the
# genre_full id/weights from the directed set (one source of truth). set=genre-logits-ab.
python -m backend.experiments.genre_space maps \
  --manifest data/genre-space/first/cohort.json \
  --artifacts data/genre-space/first/vectors \
  --output data/genre-space/first/logits-ab-maps \
  --gallery-output data/layout-evaluation/genre-logits-ab \
  --recipe-set logits-ab
```

The recipe weights/tables, quantitative results and recommended presets are single-sourced
in `audio_pipeline.md` → *Genre-space evaluation conclusions (October 2026)*. **A map is a
qualitative view of the chosen representation, not an equivalent of the quantitative
neighbourhood-retrieval scores** — for the numbers use `score --recipe-set directed` and
`data/genre-space/first/directed-analysis.json` / `.md`.

**Operational notes.** MAEST runs on MPS as in production; **MERT ran on MPS too** — both
models completed the 499/500 extract on MPS — CPU/CUDA remain the alternative. Licenses:
`maest-infer` is AGPL; MERT weights are CC-BY-NC 4.0 (non-commercial). Offline tools do
not change production embedding defaults.
