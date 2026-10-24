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

## Docs

- `AGENTS.md` — entry point and core invariants (start here for contributing).
- `architecture.md` — system structure, frontend MVC, backend packages, HTTP API.
- `ui_conventions.md` — frontend interaction rules (canvas, tags, folders, hotkeys).
- `audio_pipeline.md` — fingerprinting → embeddings → projection → clustering, and the job queue.
- `design-patterns.md` — shared interaction patterns (navigation targets, hover state, coordinators, input controls).
- `trackspace-projection-methods.md` — research notes on parametric/kernel projection candidates (background, not normative).
- `TODO.md` — deferred work and the product roadmap.

## Layout evaluation (offline precomputed maps)

Generates reproducible 2-D maps over the *same* cached library embeddings for the
browser eval gallery. It is **read-only** over the production caches: it gathers
the composite matrix exactly like the interactive projection, fits its own
openTSNE/UMAP models, and writes JSON artifacts under `data/layout-evaluation/<set>/`
(gitignored with the rest of `data/`). It never writes to the production layout
cache, the t-SNE fit cache, or MP3 tags. The projection machinery it reuses is in
`audio_pipeline.md` → *Projection methods* / *Composite vector construction*.

**Baseline.** The `baseline` recipe mirrors the production default lens fit —
openTSNE over PCA50 with the production perplexity/seed, the frozen default source
weights, feature mask and blend. It is fully **unsupervised**: it fits the raw
composite matrix with no folder/tag supervision, and every other recipe is fitted
and compared over that same fixed unsupervised input, aligned onto the baseline by
an orthogonal rotation/reflection only (no warp). Each output artifact holds the
`{path, x, y}` positions for the whole track set (~7k files on this library).

> **Historical control.** The eval baseline is a **frozen 500/250 control**
> (`tsne_n_iter=500`, early-exaggeration unchanged at 250) — the original openTSNE
> library default. Production fits its own budget (`OPENTSNE_N_ITER=100`,
> `OPENTSNE_EARLY_EXAGGERATION_ITER=100`, see `audio_pipeline.md` → *Projection
> methods*), so the 500/250 baseline is a historical control, not a production
> measurement. The eval recipes and their artifacts/labels/revisions are frozen.

**The sets.** A set is a directory of artifacts generated with one recipe list;
the gallery serves only allowlisted names (`base`, `budget`, `scale`, `early`,
`mixes`, `ablation`):

| Set | Recipes | Tests |
|-----|---------|-------|
| `base` (default) | baseline, pca2, pca25, tsne_budget, umap2_n15, umap2_n60, umap_tsne, tsne_umap_init | the original eight-map gallery: dimensionality, iteration budget, UMAP variants |
| `budget` | baseline + tsne_budget100, tsne_budget150, tsne_budget | total t-SNE iteration count (early phase unchanged) |
| `scale` | tsne_budget100 (control) + tsne_p60, tsne_p120, tsne_multiscale30_120 | the perplexity axis |
| `early` | tsne_budget100 (control) + tsne_early150, tsne_early100 | early-exaggeration phase length at production's n_iter=100 |
| `mixes` | tsne_budget100 (control) + tsne_rhythm_forward, tsne_texture_style | source weights: rhythm-led vs timbre/semantics-led |
| `ablation` | tsne_budget100 (control) + tsne_maest_supervised, tsne_maest_only | dropping whole blocks (rhythm+CLAP, then supervised) |

Each non-base set's control (`tsne_budget100`) is its alignment reference, so no
500-iteration baseline map is fitted for them. Recipe parameters (weights,
perplexities, budgets) live in `backend/embeddings/layout_evaluation.py`; judge the
maps by ear and ratings, not by which looks more spread out.

```bash
# All eight recipes → data/layout-evaluation (the default the server serves)
python -m backend.precompute_layout_evaluation

# The four-map "budget" iteration set → data/layout-evaluation/budget
# (baseline + tsne_budget100/150/tsne_budget; the 250-iteration id is `tsne_budget`)
python -m backend.precompute_layout_evaluation \
  --output data/layout-evaluation/budget \
  --recipes baseline,tsne_budget100,tsne_budget150,tsne_budget

# The four-map "scale" perplexity set → data/layout-evaluation/scale
# (control tsne_budget100 + tsne_p60/tsne_p120/tsne_multiscale30_120)
python -m backend.precompute_layout_evaluation \
  --output data/layout-evaluation/scale \
  --recipes tsne_budget100,tsne_p60,tsne_p120,tsne_multiscale30_120 \
  --reference tsne_budget100

# The three-map "early" set → data/layout-evaluation/early
python -m backend.precompute_layout_evaluation \
  --data-dir data --output data/layout-evaluation/early \
  --recipes tsne_budget100,tsne_early150,tsne_early100 \
  --reference tsne_budget100 --seed 42 --threads 8

# The three-map "mixes" source-weight set → data/layout-evaluation/mixes
python -m backend.precompute_layout_evaluation \
  --output data/layout-evaluation/mixes \
  --recipes tsne_budget100,tsne_rhythm_forward,tsne_texture_style \
  --reference tsne_budget100

# The three-map "ablation" set → data/layout-evaluation/ablation
python -m backend.precompute_layout_evaluation \
  --data-dir data --output data/layout-evaluation/ablation \
  --recipes tsne_budget100,tsne_maest_supervised,tsne_maest_only \
  --reference tsne_budget100 --seed 42 --threads 8

# A quick smoke run on a subset; a single library root; custom cache/output dirs
python -m backend.precompute_layout_evaluation --limit 200 --recipes baseline,pca2
python -m backend.precompute_layout_evaluation --root /path/to/music
python -m backend.precompute_layout_evaluation --data-dir /tmp/ts --output /tmp/ts/layout-evaluation
```

Requires **openTSNE** (the t-SNE recipes); the four UMAP recipes need
**umap-learn** (both are in `requirements.txt`). Any missing dependency is
reported as a clear per-recipe error status, never silently skipped.

The server serves the same directory it stores eval artifacts in:
`<data-dir>/layout-evaluation` (default `data/layout-evaluation`). Keep the
generator's `--output` in step with the server's `TRACKSPACE_DATA_DIR` so the
gallery reads what you generated.

Read-only GET routes (no write/upload surface):

- `GET /api/layout-evaluation` — the manifest: dataset fingerprint, sources/weights,
  versions, backend, and every recipe's status and timing.
- `GET /api/layout-evaluation/<id>` — one recipe's `positions` (`{path, x, y}`),
  revision, and timing. Only recipes the current manifest marks `complete` are
  served; traversal and unknown ids are rejected, and missing/corrupt/non-finite
  artifacts return a JSON error.
- `?set=<name>` on either route selects a **named set**: artifacts under a fixed
  subdirectory of the base (`budget` → `<data-dir>/layout-evaluation/budget`,
  `scale` → `<data-dir>/layout-evaluation/scale`,
  `early` → `<data-dir>/layout-evaluation/early`,
  `mixes` → `<data-dir>/layout-evaluation/mixes`,
  `ablation` → `<data-dir>/layout-evaluation/ablation`). Without it, the base
  (eight-map) gallery is served. Only allowlisted names (`base`, `budget`,
  `scale`, `early`, `mixes`, `ablation`) are accepted — anything else is a 400.

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
```

The gallery is a **read-only, single-bundle** mode: the same controller, canvas and
hover-audio path run, with a compact toolbar overlay. Maps are shown as **Map A, Map B,
…** (blind — the recipe name stays hidden until you toggle *Reveal recipe & timings*,
so first impressions are not coloured by the algorithm). Letters follow the
**alphabetical order of recipe ids**: the API serialises the manifest's JSON keys
sorted, so `baseline` is always Map A. The view is restricted to the
artifact's own tracks and nothing is computed on demand: no projection fetch, no
analysis SSE, no lens/scope/weight work.

From the toolbar you can:

- **Select / ‹ ›** — switch maps; positions animate on the shared canvas frame and
  selection, colour and filters stay stable.
- **Reset view** — return the camera to the full map.
- **Reveal recipe & timings** — surface each map's algorithm name and its fit timings
  (PCA/t-SNE/UMAP/alignment seconds) from the manifest.
- **Ratings** — per map, four 1–5 scores (musical neighbours, global organisation,
  crowding, overall) plus a free-text note, persisted in `localStorage` keyed by the
  **dataset fingerprint + artifact revision** so re-runs and re-reveals stay isolated.
  Clicking the current score clears it.
- **Export ratings** — downloads this dataset's ratings as a JSON file
  (`layout-eval-<first 8 chars of the dataset fingerprint>.json`). Keep the
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
> below — the gallery reads whatever `<data-dir>/layout-evaluation` currently holds.
> Ratings are keyed by fingerprint + revision, so a regenerated dataset gets a fresh
> rating sheet even under the same maps.
