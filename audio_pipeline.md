# Trackspace — Audio & ML pipeline

Everything from fingerprinting through embeddings, projection, and microgenre
clustering, plus the durable job queue that runs it. System shape lives in
`architecture.md`; UI behaviour in `ui_conventions.md`.

---

## Overview

Trackspace supports content-based audio similarity via **chromaprint** fingerprinting
and several embedding models (**CLAP**, **MAEST**, **rhythm**, **librosa
audio features**). Expensive analysis runs through a durable SQLite-backed job queue.

```
scan      → ID3 tags + Chromaprint fingerprint
features  → classical 6-D + rhythm 14-D
maest     → MAEST embedding + 519 style logits + per-excerpt vectors
clap      → CLAP audio embedding (the column span text directions project onto)
supervise → folder-supervised 128-D projection head over the cached vectors
cluster   → weighted blocks → PCA → cosine kNN → multi-resolution Leiden
```

---

## Fingerprinting (`backend/fingerprint.py`)

- Calls `fpcalc` (chromaprint CLI) and SHA-256-hashes the raw fingerprint.
- System dependency: `brew install chromaprint` (provides `fpcalc`).
- `fpcalc` is resolved via `shutil.which` at import time, with a fallback to common
  homebrew/system paths.  npm/concurrently child processes sometimes have a stripped
  `PATH` that omits `/opt/homebrew/bin`, so bare `fpcalc` may not be found.
- Integrated into `_cached_read_all()` — fingerprints are computed once per file and
  stored alongside ID3 data in the track cache.

**Cache poisoning guard:** `_cached_read_all` checks `cached.get("fingerprint")` (truthy),
not just `"fingerprint" in cached`.  If a previous run stored `null` (e.g. fpcalc was
missing), the next run with fpcalc available will retry rather than serving the stale null.
Never cache a `None` fingerprint and treat it as a permanent result.

---

## Audio decoding (`backend/decode_stderr.py`)

All audio decode goes through `librosa_load` / `librosa_get_duration`, which do two
things bare `librosa.load` does not:

- **Silence native stderr.** libsndfile / libmpg123 print ID3, Xing/VBR and resync
  diagnostics straight to fd 2, bypassing Python logging (`redirect_stderr` does not
  help), so fd 2 is `dup2`-ed to `/dev/null` (or `TRACKSPACE_DECODE_STDERR_LOG`) for
  the duration of the call.
- **Serialise the native MP3 open.** libsndfile's `mpeg_decoder_init` calls the global,
  non-thread-safe `mpg123_init()` on *every* `sf_open` — libsndfile's own source carries
  a `*** FIXME - Threading issues ***` note there. Two workers opening MP3s at once (the
  `features` and `maest` stages decode in parallel) race in mpg123's static layer-table
  init and call through a half-written function pointer, killing the process with
  **SIGBUS** (`mpeg_init` → `mpeg_open` → `psf_open_file` → `sf_open`).

`_DECODE_LOCK` is held only across the open: reads on an already-open handle use
per-handle state, so the expensive decode still runs in parallel. `librosa_load` hoists
the open out of `librosa.load` (which accepts a pre-opened `soundfile.SoundFile`) to
achieve this; `librosa_get_duration` takes the lock for the whole (cheap) call.

The lock is **per process**, so it does not serialise the `features` stage's process
pool (each child has its own). It does mean one process's `librosa_get_duration` calls
are serial with each other — for a VBR MP3 with no Xing header libsndfile scans the
whole file on the first probe (measured ~1.2 s on a 6-minute track, cached by librosa
afterwards), so that probe is worth avoiding per track.

A one-time single-threaded warmup does **not** fix it — the init runs on every open.
Reproduced standalone against the real library: 10/10 crashes unlocked, 0/10 with the open
serialised. `tests/test_decode_stderr.py` pins the contract.

**Why this regressed:** the durable job queue replaced a single generation thread with one
worker thread per stage, so decode that was serial by construction now overlaps.

---

## Two-cache architecture

| Cache | File | Key | Stores |
|-------|------|-----|--------|
| `TrackCache` | `data/track_cache.db` | `(path, mtime)` | ID3 tags, metadata, chromaprint fingerprint hash |
| `FeatureCache` | `data/audio_features.db` | `fingerprint` (SHA-256 of chromaprint) | one row per source: CLAP/MAEST/rhythm/features vectors + version |

The split means identical audio content (same recording, different filenames) shares one
embedding entry. `TrackCache` still invalidates on mtime change; `FeatureCache` is
content-addressed and versioned.

---

## Embedding sources

The source set is fixed (`FROZEN_SOURCES`); the Options panel exposes one weight
slider per source, and **Apply** sends the vector to both the map and the clusters.

| Source | Module | Dimensions | Model |
|--------|--------|-----------|-------|
| MAEST logits | `backend/embeddings/maest.py` | 519 | `discogs-maest-30s-pw-129e-519l` (primary genre signal) |
| MAEST embedding | `backend/embeddings/maest.py` | 768 | same model pass |
| Rhythm | `backend/embeddings/rhythm_features.py` | 14 | `librosa` DSP |
| CLAP | `backend/embeddings/clap.py` | 512 | `laion/larger_clap_music` (HuggingFace) |
| Audio features | `backend/embeddings/librosa_audio_features.py` | 6 | `librosa` (+ **madmom** RNN tempo when installed) |
| Supervised | `backend/embeddings/supervised.py` | 128 | folder-labelled projection head over the cached blocks (no encoder) |

### Independent versioning per source

| Source | Version constant | Cache columns |
|--------|-----------------|---------------|
| CLAP | `EMBEDDING_VERSION` (`embeddings/clap.py`) | `clap_embedding`, `version` |
| Audio features | `FEATURES_VERSION` (`embeddings/librosa_audio_features.py`) | `features`, `features_version` |
| MAEST | `MAEST_VERSION` (`embeddings/maest.py`) | `maest_embedding` + `maest_logits` + `maest_excerpts`, `maest_version` |
| Rhythm | `RHYTHM_VERSION` (`embeddings/rhythm_features.py`) | `rhythm`, `rhythm_version` |
| Supervised | training-run id (see *Weak supervision*) | `supervised_embedding`, `supervised_version` |

Bumping one source's version invalidates only that source's cached data.
MAEST's embedding and logits share one version column because a single model
pass produces both — `put_maest()` writes them in one statement.

New columns are added via the `ALTER TABLE` migration loop in
`FeatureCache._init_db`, so existing databases migrate in place (verified
against the real 9.4k-row cache with no data loss). All sources share one
generic implementation driven by the `_SOURCES` registry; the public API is
source-parameterised (`get` / `put` / `has` / `get_all` / `count` /
`purge_old_versions(source, version)`), and `get_all` chunks its
`WHERE fingerprint IN (...)` at 500 ids so a large library cannot exceed
SQLite's variable limit.

**Purge writes empty blobs, not NULL.** `clap_embedding` is declared `NOT NULL`,
so a purge that set it to NULL raised `IntegrityError` — which the broad
`except Exception` swallowed, silently making CLAP purges no-ops. Purge now
overwrites each blob with an empty blob, and every read path uses
`length(blob) > 0` as its presence test. Any new cache read must use the same
empty-blob-as-absent rule or it will resurrect purged rows.

---

## CLAP embeddings (`backend/embeddings/clap.py`)

- Model: `laion/larger_clap_music` loaded in a background thread at server startup
  (does not block Flask from accepting requests).
- **Inference lock:** `clap.inference_lock` is the single process-wide accelerator
  lock. It serialises every forward pass that touches MPS/CUDA: CLAP audio, CLAP
  text, MAEST's transformer (which imports the same lock rather than defining its
  own), and mlx-vis (`UMAP` / `TSNE`) on Apple Metal. MPS/CUDA are not safe for
  concurrent dispatches from multiple threads; without one shared lock, a CLAP worker
  and the MAEST worker dispatching to MPS at the same time, or a Flask request
  running semantic-weighting text inference while a worker produces audio
  embeddings, would crash or hang (IOGPU `commit command buffer with uncommitted
  encoder` assertions).
- **Rule for new GPU code paths:** If a path dispatches work to MPS/Metal/CUDA
  from request threads, either (1) guard it with `clap.inference_lock`, or
  (2) prove the backend is thread-safe under mixed-framework load before
  allowing concurrent execution. Do not define a second lock.
- **Rule for env-toggle tests:** Backend enable flags are cached lazily; after
  changing `TRACKSPACE_CUML` / `TRACKSPACE_MLX_VIS` in tests, call
  `reset_projection_backend_cache()` before running a projection.
- Uses MPS (Apple Silicon), CUDA, or CPU — whichever is available.
- Audio is loaded via librosa at 48 kHz.  Three 15 s segments (at 20 %/50 %/80 %
  through the file) are embedded independently and averaged for robustness.
  Short tracks (≤ 22.5 s) use the whole file as a single segment.
- **The queue stage.** `KIND_CLAP` (`backend/jobs/kinds.py`) drains one job per
  track through `make_clap_handler`. `clap.analyze_batch` decodes per track on the
  worker thread, then runs one forward pass per batch of segments (the feature
  extractor pads each segment independently, so variable-length segments batch
  safely) — one `inference_lock` acquisition per batch instead of one per segment.
  Progress is streamed via the queue's SSE feed (`/api/jobs/stream`), so tracks
  appear incrementally. The frontend sends `priority_paths` so tracks currently
  visible on screen are processed first. A CLAP drain triggers a re-cluster, so the
  map and clusters pick up the new block without a manual rebuild.
- **The block is load-bearing.** The audio embedding is not just a low-weight
  texture signal: it is the column span that `layout`'s semantic weighting projects
  tag/folder-name text directions onto. With no cached CLAP rows, the block is absent
  and tag-based weighting is silently inert. `--no-clap` therefore skips the stage
  entirely (the enqueue filter and worker gate both key off it) rather than queueing
  jobs behind a model that will never load.

### Embedding versioning

`EMBEDDING_VERSION` (int) in `backend/embeddings/clap.py` encodes the current CLAP
embedding strategy. All `FeatureCache` reads filter by version, so bumping the
constant auto-invalidates stale entries. Old-version rows are purged on server startup.

| Version | Strategy |
|---------|----------|
| 0 | Legacy (pre-versioning) |
| 1 | Single 30 s middle-of-track chunk |
| 2 | Multi-segment: 3 × 15 s at 20 %/50 %/80 %, averaged |

When changing how embeddings are produced (model, chunk strategy, post-processing),
bump `EMBEDDING_VERSION` and add a row to the table above.

---

## MAEST model management

MAEST (`discogs-maest-30s-pw-129e-519l`) is the **primary genre model**: it was
trained on a Discogs style taxonomy, so its 519 logits already describe a track
as a fuzzy point in genre space — frequently more useful for microgenre
clustering than the hidden embedding. Checkpoints auto-download to
`~/.cache/torch/hub/checkpoints/` on first use.

**The mel front end must run on CPU.** `torchaudio`'s STFT requires input and
window on the same device, and `model.to("mps")` drags the window onto the GPU,
producing `stft input and window must be on the same device`. `load_model()`
therefore detaches `model.melspectrogram`, forces it back to CPU, and sets the
attribute to `None` so no code path can re-enter a GPU mel. Inference then runs
`model(mel, melspectrogram_input=True)`.

Batching matters: mel-on-CPU + batched transformer inference measured
**~0.11 s/excerpt** on MPS versus ~1.0 s/excerpt unbatched on CPU (~9×).

Three 30 s excerpts at 20/50/80 % are embedded as one batch and mean-pooled.
Per-excerpt vectors are stored alongside the pooled pair (the `maest_excerpts`
blob) so weak supervision can mine hard negatives and detect genre-spanning
tracks without re-running inference.

---

## Audio feature extraction (`librosa` + optional `madmom`)

**Audio feature vector (6D):**
`[tempo_norm, key_cos, key_sin, mode, energy_norm, danceability]`

- Tempo: **madmom** (`RNNBeatProcessor` + comb `TempoEstimationProcessor`) when importable,
  else `librosa.feature.tempo()` (same audio clip; `FEATURES_VERSION` 6+).
  Classical-feature excerpts stay at **22.05 kHz** for librosa; madmom receives a
  **44.1 kHz** resample of that buffer only. Optional env
  **`TRACKSPACE_MADMOM_FAST=1`** uses one BLSTM instead of the default
  eight-network ensemble for a large speedup with slightly less robust tempo (pick one
  setting per library and keep it stable, or bump `FEATURES_VERSION` / purge features
  when toggling). Profile locally: `.venv/bin/python tests/profile_madmom_tempo.py`.
- Key: Krumhansl-Schmuckler algorithm on `librosa.feature.chroma_cqt()`
- Energy: `librosa.feature.rms()` (log-scaled)
- Danceability: onset strength autocorrelation regularity

Key is encoded on the **circle of fifths as a unit circle** in 2D:
`key_cos = cos(2π · fifths_pos / 12)`, `key_sin = sin(2π · fifths_pos / 12)`.
This preserves harmonic topology — keys a fifth apart are geometrically close,
and the circular wrap is seamless.  Mode (major=1, minor=0) is a 3rd dimension.

**Rhythm vector (14-D)** — `onset_density`, `onset_strength_mean/std`,
`beat_confidence`, `pulse_clarity`, `syncopation`, `kick_periodicity`,
`bass_ratio`, `spectral_centroid_norm`, `spectral_rolloff_norm`,
`spectral_flux`, `dynamic_range`, `loudness`, `rhythmic_complexity`. All
normalised to [0,1]; the STFT / onset envelope / beat track are computed once
and reused across dimensions. This block is what separates rhythmically
distinct neighbouring genres.

---

## Composite vector construction

`_gather_composite_vecs` in `embeddings/layout.py`:

Each enabled source block is **L2-normalized per row** before concatenation so
that 768-D MAEST does not dominate 14-D rhythm through dimensionality, then
scaled by `sqrt(weight)` from `FROZEN_SOURCE_WEIGHTS` so relative influence is
set explicitly (same algebra as the clustering pipeline — see *Microgenre
clustering*).

A track is included when it has every **required** source (`REQUIRED_SOURCES`,
i.e. MAEST); optional sources it lacks contribute a zero row. Requiring *all*
enabled sources would hide every track until the slowest model had run over the
whole library.

**Semantic weighting with composite vectors:** Folder centroid decomposition uses
the **full** composite matrix.  CLAP text directions (tags, folder-name fallbacks)
are placed in the **CLAP column span** (`clap_col_lo : clap_col_lo + clap_dim`),
which depends on *sources* order — `mat[:, :clap_dim]` was wrong when CLAP was
not the first block.

---

## Projection methods

`compute_projection()` projects cached embeddings to 2D. The backends live in
`embeddings/projection_backends.py`. (The frontend currently always requests
`method=tsne`; `pca` is used internally for the live preview during generation.)

| Method | Backend function | Notes |
|--------|-----------------|-------|
| `"tsne"` (default) | `_project_tsne` (openTSNE → cuML → mlx-vis → scikit-learn) | Best local cluster preservation; slower |
| `"pca"` | `_project_pca` (numpy SVD) | Instant; used for live incremental updates during generation |

**Incremental t-SNE (openTSNE, prototype).** With `TRACKSPACE_OPENTSNE=1`
and `openTSNE` importable, `_project_tsne` routes to `_project_tsne_incremental`,
which fits an embedding once per *vector config* and then places only newly added
tracks into it via `TSNEEmbedding.transform` instead of refitting the library.
Key facts:

- The fit cache keys on `_composite_config_token` (sources, mask, blend, weights,
  versions) plus the semantic-basis params — **not** the track set, so adding tracks
  reuses the fit. Changing sources/mask/blend/weights/context refits.
- Tracks already in the fit are sliced from the cached coordinates, so they never
  move between incremental updates. New tracks are PCA-reduced with the *fit's* PCA
  basis before `transform` (not the current matrix's basis).
- Additions above `TRACKSPACE_OPENTSNE_REFIT_FRACTION` (default 0.3) of the fitted
  set trigger a full refit for quality; `TRACKSPACE_OPENTSNE_MAX_POINTS` (default
  20000) falls back to the non-incremental chain above the cap.
- `TSNEEmbedding.transform` recentres the fitted embedding in place, so the cached
  coords are re-read from the embedding after every transform.
- CPU-only, so it does **not** take `inference_lock` (unlike mlx-vis).

Off by default: the cold fit is slower than mlx-vis on Apple Silicon (~20.3 s vs
~12.7 s at 9.5k × 50), while a +500 transform is ~0.93 s (~14× faster than a
full refit). Flipping the default is a product decision.

PCA is never shown in the UI toggle — it is only used internally for fast intermediate
projections while embeddings are being generated.

### Projection caching

Projection results are cached at two levels:

**Server-side (revision-keyed LRU):**
`_revision_layout_cache` in `embeddings/layout.py` is an `OrderedDict[str, list[dict]]`
(capped at 32 entries). The key is `layout_revision` — a deterministic SHA-256
computed by `compute_layout_revision()` from sorted eligible paths, method,
sources, feature mask/blend, folder/tag context, and cache version constants.
Because the key is content-addressed, row order in the input data does not
affect cache identity.

**Client-side (query-string → {revision, positions} map):**
`_layoutCache` in `coordinators/embedding-coordinator.ts` maps
`_projectionQueryString()` → `{ revision, positions }`, capped at 8 entries with
LRU eviction. Before fetching `/api/embeddings/projection`, the coordinator checks
whether a cached entry's `revision` matches the server's current `layout_revision`
(obtained from `/api/embeddings/status`). If it matches, cached positions
are applied directly — no projection round-trip. This handles A→B→A cycles
(e.g. toggling TSNE↔UMAP↔TSNE) without redundant server work.

**openTSNE fit cache (incremental path only):**
`_tsne_fit_cache` in `embeddings/projection_backends.py` holds up to 4 fitted
embeddings keyed by vector config (not track set). It sits *below* the
revision-keyed LRU: the revision changes when tracks are added, so the LRU misses,
and the fit cache turns that miss into a `transform` of only the new rows. The two
caches answer different questions — revision = "same exact view?", fit config =
"same vector space?".

Both caches are invalidated when new embeddings are generated (server: implicitly
via new revision; client: `invalidateLayoutCache()` on SSE `done`).

**Deterministic ordering:** `_build_track_infos` sorts `infos` by path after the
`as_completed` gather loop. This is critical — `as_completed` returns results in
nondeterministic order, which would produce different composite cache keys, different
t-SNE/UMAP row ordering (and thus different layouts), and revision cache misses.
All new code that builds track info lists must maintain this sort invariant.

---

## Semantic weighting (context-aware projection)

Before projection, the embedding space can be **re-weighted** so directions
reflecting the user's folder organisation and tag names receive higher contrast.

**Two strategies — data-driven folders, text-based tags:**

| Source | Method | Rationale |
|--------|--------|-----------|
| Folders | Hierarchical centroid decomposition from actual track embeddings | Folders are manual curation — their tracks ARE the ground truth |
| Tags | CLAP text embedding of tag names | Tags are labels, not track collections |

**Hierarchical centroid decomposition** (`_build_folder_directions` in
`embeddings/semantic_weighting.py`):

For each folder node the direction is
`centroid(all tracks under node) − centroid(all tracks under parent)`.
The parent centroid *is* the weighted mean of its children's centroids, so
subtracting it naturally strips the shared component among siblings — each
child retains only its unique contrast.  This decomposes the embedding space
hierarchically: broad genre distinctions live at shallow levels, subtle
microgenre differences at deeper levels.

**Depth-increasing boost:**  Deeper levels receive *more* weight
(`boost × depth_boost^(depth−1)`, default `depth_boost = 1.5`) because broad
genre axes are already well-separated in CLAP space — it is the fine sibling
distinctions that dimensionality reduction tends to collapse and that benefit
most from amplification.

**Fallback:** Folders with fewer than 2 embedded tracks cannot produce a
reliable centroid; these fall back to CLAP text embedding of the folder's
leaf name.

**Flow:**
1. The frontend collects `model.contextTags` (tag names) and `model.contextFolders`
   (unique full folder paths from all tracks).
2. These are sent as `context_tags` and `context_folders` query params on
   `GET /api/embeddings/projection` (legacy alias: `/api/embeddings/umap`).
3. `_build_folder_directions()` infers the folder hierarchy from the leaf paths,
   computes centroids per node (including all descendants), and derives one
   normalised contrast direction per node.  Intermediate parent nodes that were
   not in the original folder set are inferred automatically.
4. `_apply_semantic_weighting()` combines data-driven folder directions, text
   fallbacks, and tag text directions into a single matrix `T_n` with weights
   `w`, then applies `X_out = X @ (I + T_n^T diag(w) T_n)`.

Weighting is transparent — when no context is provided, raw embeddings are
projected unmodified.

---

## Weak supervision (folder-supervised projection head)

`backend/embeddings/supervised.py` trains a small MLP over the **cached** vectors
and exposes its 128-D output as an ordinary source block (`supervised`), so the map
and the clusters consume it like any other signal. Nothing here re-runs an encoder:
training reads only `FeatureCache`.

**Folders are the labels.** The leaf folder of each virtual path is a class; folders
with fewer than `MIN_FOLDER_LABEL_SIZE` (8) tracks are noise and their tracks are
unlabelled. Classes are sorted so the class list is stable across runs.

**The input recipe is stateless.** The head's input is the per-source blocks
L2-normalised per row and scaled by `sqrt(weight)` (`SUPERVISED_INPUT_WEIGHTS`),
with **no column standardisation** — unlike the clustering pipeline. Column statistics
depend on the current track set, so a stateful recipe would make a single new track's
projection drift as the library grows. The head's own output is excluded from its
input (`SUPERVISED_INPUT_WEIGHTS` drops `supervised`), so the block cannot feed
back into itself.

**Loss.** Supervised contrastive (SupCon) with class-balanced batches, plus a
hard-negative repulsion term. Positives are all same-folder rows in a batch;
early stopping on a per-class validation split returns the best weights.

**Excerpts refine the supervision** (`maest_excerpts`, written by the MAEST stage):

- *Hard-negative mining.* Per-track vectors are excerpt centroids (or the pooled
  MAEST embedding when excerpts are absent — both live in the same 768-D MAEST
  space). The hardest cross-folder pairs are mined and pushed apart explicitly.
- *Label confidence.* A track whose excerpts disagree (intro vs drop) is a noisy
  single-label example; its loss weight decays with intra-track dispersion.

**Training is CPU-only.** At library scale the head is a few seconds of CPU matmuls.
This is deliberate: the GPU paths are serialised by `clap.inference_lock`, and a
training run holding that lock would stall all embedding generation for no benefit.

**The run id is the cache version.** A retrain rewrites every `supervised` row in
place, so rows are stored under `version = run_id` (a monotonic counter persisted in
the head sidecar and mirrored into the `feature_meta` table). Layout and cluster
revisions include the run id, so a retrain invalidates cached coordinates; the
openTSNE fit and composite caches include it via `_source_versions_token`.

**A changed input space needs a retrain.** The head's `input_sources` and `in_dim`
are recorded at training time; `project_supervised_missing` refuses to project new
tracks when the gathered input space no longer matches (it logs "input space changed"
and returns 0). Any change to `FROZEN_SOURCES` — e.g. adding the `clap` block or
removing a source — therefore requires one manual retrain (the *Train folder view*
button) before new tracks get supervised rows again.

**Triggers and staleness.** Training is a library-wide singleton (`"library"` dedupe
key). It is queued manually (the *Train folder view* button) or automatically once
after the first MAEST pass drains while no head exists. New tracks analysed later are
projected through the current head at the top of every cluster pass (a cheap matmul), so
they get a row without a retrain. A retrain is an explicit user action — folder
reorganisation is invisible to the filesystem watcher.

---

## Analysis pipeline: durable job queue

All expensive analysis runs through a **durable SQLite-backed job queue**
(`backend/jobs/`). This replaced an in-process `embed_status` dict guarded by a
global "running" flag. The change matters for correctness, not just tidiness:

| Old behaviour | Queue behaviour |
|---|---|
| Progress lived in RAM — a restart lost it | State is SQL rows; restart resumes |
| One global `running` flag; concurrent requests were dropped | Per-job leases; work is never silently discarded |
| A crash left `running = True` forever | Expired leases are reclaimed to `pending` |
| Duplicate triggers queued duplicate work | `(kind, dedupe_key)` is UNIQUE — enqueue is idempotent |
| A transient decode error lost the track | Retry budget (`max_attempts`, default 3) |
| Counts were hand-maintained counters that could drift | `stats()` aggregates the table — cannot drift |

### Job kinds (pipeline order)

```
scan      → ID3 tags + Chromaprint fingerprint   (CPU, batch 8)
features  → classical 6-D + rhythm 14-D          (CPU, N processes, batch = N)
maest     → MAEST embedding + 519 style logits   (GPU, batch 4, gated on model)
clap      → CLAP audio embedding                 (GPU, batch 4, gated on model)
supervise → folder-supervised projection head    (CPU, singleton, whole-library)
cluster   → kNN graph + multi-resolution Leiden  (whole-library singleton)
```

Names live in `backend/jobs/kinds.py` so enqueuers, workers, and routes cannot
drift apart.

`clap` sits after `maest` so the primary genre signal lands first and the map
renders while the (decode-dominated) CLAP backfill continues; its drain triggers a
re-cluster like any other per-track stage.

**The `features` stage parallelises across processes, not threads.**
`backend/services/features_worker.py` is the pool child; `make_features_handler`
runs each claimed track there and the parent keeps the cache write and the
decode-warning relay. madmom's BLSTM forward pass is GIL-bound: several features
*threads* do not run in parallel and actively convoy — measured on one track,
0.8 s at 1 thread, 2.5 s at 2, 10.5 s at 4 — while a process pool scales
near-linearly (51 → 134 → 140 tracks/min at 1/4/8 processes on the real
library). `TRACKSPACE_FEATURES_WORKERS` sets the process count (default 4).
The pool uses `spawn`, not `fork`: the parent is multi-threaded (Flask, workers,
watchdog) and forking it would hand children locks held by other threads.
Consequently `_DECODE_LOCK` is per-process, and the madmom processors are
per-thread inside each process (`backend/embeddings/madmom_tempo.py`) — the BLSTM
layers carry hidden state across frames, so a shared processor corrupts tempo
values. `warm()` in the child module builds each child's processors on start.

### Rules when adding a stage

1. **Never raise for one bad track.** Return `JobResult.failure(...)`; the rest
   of the batch still commits and the queue owns the retry decision.
2. **Use `retry=False` for permanent errors** (missing file, unsupported codec)
   so a dead track does not consume three attempts *and* three model runs.
3. **Batch only at the GPU boundary.** MAEST and CLAP batch across tracks. The
   CPU `features` stage claims a batch only to keep its process pool fed; each track
   is decoded and computed in one child, so a slow decode does not stall the pool.
4. **Filter against the cache before enqueueing**, so the queue reflects
   *outstanding* work. This is what makes the progress numbers meaningful.
5. **Gate on model readiness** with `JobWorker(gate=...)` rather than letting
   the worker claim jobs and fail them while a model is still loading.
6. **Library-wide work uses one dedupe key** (`"library"` for clustering), so a
   burst of filesystem events collapses into at most one pending re-run.
7. **A `--no-*` flag must also skip its enqueue**, not just eager loading:
   a stage whose model never loads would otherwise leave jobs pending forever
   (the `clap_enabled` gate on `enqueue_library_work`).

`--no-maest` / `--no-clap` skip eager model loading. Flask runs
with `use_reloader=False`: the reloader would fork a second set of worker
threads competing for the same queue.

---

## Microgenre clustering

Pipeline (`backend/embeddings/clustering.py`):

```
per-source blocks
   ↓  standardize + L2-normalise per block, scale by sqrt(weight)
weighted matrix
   ↓  PCA → ≤128 dims
reduced
   ↓  cosine kNN (k=20), bridged into one connected component
graph
   ↓  Leiden at 6 resolutions
multi-resolution cluster hierarchy
```

### Three decisions that are easy to get wrong

**Cluster in feature space, never in t-SNE/UMAP space.** 2-D projections
distort global geometry; they are for *looking at*, not grouping by.

**`sqrt(weight)` per block, not blind concatenation.** Squared Euclidean
distance is additive over concatenated blocks, so scaling a block by `sqrt(w)`
makes it contribute exactly `w ×` its distance. The weighted-distance
formula therefore falls out of ordinary PCA/kNN — no custom metric needed.

**The kNN graph must be connected.** Leiden cannot merge vertices with no path
between them, so a fragmented graph pins the cluster count to the component
count and makes the resolution parameter *silently inert* — the whole
multi-resolution hierarchy collapses to one granularity. `build_knn_graph`
bridges components via their strongest cross-pair edge. This was found by
testing: with tight clusters and k=20 the graph fragmented into 9 components
and every resolution returned the same 9 clusters.

### Weights — `DEFAULT_WEIGHTS` / `FROZEN_SOURCE_WEIGHTS`

| Source | Weight | Role |
|---|---|---|
| `maest_logits` | 0.45 | Discogs genre/style space — the primary signal |
| `maest` | 0.25 | Audio semantics from the hidden embedding |
| `rhythm` | 0.15 | Groove/BPM distinctions between neighbouring genres |
| `clap` | 0.10 | Complementary texture **+ the only text encoder** |
| `supervised` | 0.15 | Learned folder-organisation signal (absent until trained) |
| `features` | 0.00 | Off by default (subsumed by `rhythm`) |

`projection_config.FROZEN_SOURCE_WEIGHTS` mirrors `clustering.DEFAULT_WEIGHTS`
deliberately: a 2-D map that disagreed with the cluster assignments would be
actively misleading. **Keep them in sync.**

### Interactive weights

The defaults above are overridable per request. The Options panel exposes one slider
per source; **Apply** sends the vector to both pipelines, so the map and the clusters
keep describing the same space:

- `POST /api/clusters/rebuild` with `{"weights": {...}}` stores the requested
  vector on `ClusterStore` and queues a re-cluster. Weights live on the store rather
  than in the job payload because re-clusters also fire automatically when an analysis
  stage drains — a per-job payload would let those runs silently revert to the defaults.
  An absent `weights` field keeps the current setting (defaults on a fresh server), so
  the plain "Rebuild clusters" button is a re-run, not a reset.
- `GET /api/embeddings/projection?w=maest_logits:0.45,maest:0.25,…` applies the
  same vector to the layout. Unlisted sources fall back to their frozen default;
  unknown sources and out-of-range values are a structured 400.
- **Source weights are part of `compute_layout_revision`.** Without that, two weight
  configurations would hash identically and the revision-keyed position caches would
  serve one configuration's coordinates for the other. The `w=` param is omitted when
  the weights equal the defaults, so default query keys stay stable.
- Applying a new vector costs a full t-SNE refit (~12–20 s at 9.5k tracks), so the UI
  commits on an explicit Apply rather than on every slider release; the cluster
  re-run itself is a few seconds.

CLAP is retained at low weight because semantic folder/tag weighting in
`layout.py` projects *text* directions into the CLAP column span — dropping
CLAP would remove that capability.

### Required vs optional sources

`REQUIRED_SOURCES = ("maest_logits", "maest")`. A track needs MAEST to be
placed; optional sources — including `supervised` — contribute a neutral zero row
when missing, so a partially analysed (or not-yet-trained) library still renders
instead of hiding tracks until every model has run. The frontend gates the embedding
view on **MAEST only** — gating on CLAP left the canvas stuck on "loading
models…" under `--no-clap`.

### Resolutions

`DEFAULT_RESOLUTIONS = (0.15, 0.4, 0.8, 1.5, 3.0, 6.0)`, calibrated against the
real library (~9.5k tracks, k=20) → roughly 2 / 3 / 6 / 12 / 25 / 60 clusters.
Below ~0.1 everything collapses to one cluster. **Validate resolution ranges
against real embeddings, not synthetic blobs** — synthetic Gaussians are far
more separable than real music and make the sweep look broken.

### Cluster naming

`name_clusters` names a cluster by *contrast*: its mean logit per style minus
the library-wide mean. Without the subtraction every cluster in an electronic
library is named "Electronic---Techno"; with it, clusters come out as
"Drum n Bass / Jungle", "Dubstep / Grime", "Psy-Trance / Progressive Trance".

### Validated behaviour

On 150 tracks drawn from 6 hand-curated genre folders, resolution 1.5 produced
6 clusters matching the folders at 148/150, with correct generated names, and
the clusters appear as spatially separated regions in the t-SNE layout.
