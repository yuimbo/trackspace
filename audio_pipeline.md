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
structure → intro/drop/break timestamps (loudness envelope, no model)
maest     → MAEST embedding + 519 style logits + per-excerpt vectors
clap      → CLAP audio embedding (the column span text directions project onto)
supervise → folder-supervised 128-D projection head over the cached vectors
cluster   → weighted blocks → PCA → cosine kNN → multi-resolution Leiden
layout    → warm the per-lens 2-D projections (background, after cluster)
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

**Whole-track reads can fail on valid MP3s.** libsndfile's *full* read can raise
("Unspecified internal error") on an MP3 whose container metadata lies about the frame
count, while a *windowed* read (`offset` + `duration`) of the same file succeeds.
`audio_decode.load_resilient_full_mono` tries the full read, then falls back to decoding
fixed windows up to the metadata duration and concatenating them. The `structure` stage uses it
(the segment loader cannot serve a whole-track loudness envelope); without the fallback those
files were permanent structure gaps while `features`/`maest`, which decode segments, succeeded.

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
A projection request may select a **subset** of the sources (`sources=`), which is how
the UI's per-source lenses project one block at a time; the combined lens omits the
param. Each source set derives its own required sources (a single-source lens requires
only itself) and is part of `layout_revision`.

| Source | Module | Dimensions | Model |
|--------|--------|-----------|-------|
| MAEST logits | `backend/embeddings/maest.py` | 519 | `discogs-maest-30s-pw-129e-519l` (primary genre signal) |
| MAEST embedding | `backend/embeddings/maest.py` | 768 | same model pass |
| Rhythm | `backend/embeddings/rhythm_features.py` | 14 | `librosa` DSP |
| CLAP | `backend/embeddings/clap.py` | 512 | `laion/larger_clap_music_and_speech` (HuggingFace) |
| Audio features | `backend/embeddings/feature_encoding.py` | 6 | perceptual re-encoding of `librosa_audio_features` (Camelot + tempo spiral) |
| Supervised | `backend/embeddings/supervised.py` | 128 | folder-labelled projection head over the cached blocks (no encoder) |

`structure` is cached alongside the vector sources but is **not one of them**: it is
per-track metadata (six timestamps), never a projection or clustering block, so it
is absent from `FROZEN_SOURCES` and from the source-weight sliders.

### Independent versioning per source

| Source | Version constant | Cache columns |
|--------|-----------------|---------------|
| CLAP | `EMBEDDING_VERSION` (`embeddings/clap.py`) | `clap_embedding`, `version` |
| Audio features | `FEATURES_VERSION` (`embeddings/librosa_audio_features.py`) | `features`, `features_version` |
| MAEST | `MAEST_VERSION` (`embeddings/maest.py`) | `maest_embedding` + `maest_logits` + `maest_excerpts`, `maest_version` |
| Rhythm | `RHYTHM_VERSION` (`embeddings/rhythm_features.py`) | `rhythm`, `rhythm_version` |
| Supervised | training-run id (see *Weak supervision*) | `supervised_embedding`, `supervised_version` |
| Structure | `STRUCTURE_VERSION` (`embeddings/structure.py`) | `structure`, `structure_version` |

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

- Model: `laion/larger_clap_music_and_speech` loaded in a background thread at server startup
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
- **Per-segment vectors are cached too.** `analyze_batch_segments` returns each
  segment's vector and its centre fraction; `make_clap_handler` stores them as the
  `clap_segments` / `clap_segment_pos` blobs (sharing the CLAP `version` column)
  alongside the pooled mean. Text search ranks against them so a hit auditions the
  best-matching section (see *Text search* below). `analyze_batch` remains the
  mean-pooled view for layout/clustering. A pooled row without segments is
  re-queued once to backfill them.
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
| 3 | Model switched `laion/larger_clap_music` → `laion/larger_clap_music_and_speech` (see below) |

When changing how embeddings are produced (model, chunk strategy, post-processing),
bump `EMBEDDING_VERSION` and add a row to the table above.

**The `laion/larger_clap_music` HF checkpoint is broken — do not use it.** Its
converted weights are degenerate: both towers collapse to near-constant vectors
(text-vs-text cosine ≈ 0.999, audio-vs-audio ≈ 0.95–0.996, text-vs-audio ≈ 0.00), so
the canonical HF pipeline returns a uniform softmax (0.0834 across 12 labels on a techno
track; white noise, silence and a sine tone give *identical* outputs). Reproduced on CPU and
MPS and under transformers 4.57.1 and 5.3.0, with exact weight loading and no missing
checkpoint keys — it is the checkpoint, not our decode or the transformers version (HF
discussion #2 on that repo, open since Dec 2023). `laion/larger_clap_music_and_speech` is the
music-trained conversion that works (real-track check: a techno track ranks "techno" 0.59 over
"deep house" 0.36; a deep-house track ranks "deep house" 0.55). Switching it invalidated
every cached `clap` row (`EMBEDDING_VERSION` 2 → 3), so a one-off CLAP re-analysis is
required; the supervised head also consumes the `clap` block and should be retrained afterwards.

### Text search (`backend/routes/api_search.py`)

`GET /api/search?q=…&limit=50` encodes the query with CLAP's text tower and ranks
the cached per-segment CLAP audio vectors by cosine similarity — the retrieval use of
the same text encoder that semantic weighting already projects tag/folder names with.
Each track scores on its **best-matching segment** and the hit carries that segment's
centre as a `section` fraction, so the hover preview auditions the part that actually
matched the query rather than the track's drop. Tracks analysed before per-segment
vectors existed fall back to the pooled mean (no `section`). It is **read-only over
`FeatureCache`** (no encoder re-runs over the library) and returns the top-N hits plus
`searched` / `total_tracks` coverage counts. The text forward pass runs on the request
thread under `clap.inference_lock`, exactly like semantic weighting during a projection
request. The frontend turns the hits into the canvas selection and lists them (with score
bars) in the Search panel; clicking a result selects it alone and fits the viewport to
it.

`field=1` additionally returns a ~128-point **landmark** sample
(`field: [{path, score}]`) for the canvas search field. The landmarks are the
representative tracks the layout's own weight chart already selects
(`landmark_projection._select_landmarks` over the L2-normalised pooled CLAP
matrix), so the sample is a property of the cache, not of the query; the selection
is cached by a CLAP-content token because the k-means is ~0.3 s at 10k tracks.
The client has every track's 2-D map position, so it splats the sample into the
"where does my query match" glow without any further backend work. The field is a
local-density view, not a calibrated similarity contour (t-SNE preserves only local
neighbourhoods).

Retrieval quality depends on the CLAP checkpoint above — with the working
`larger_clap_music_and_speech` conversion the query ranking is meaningful; with the
old `larger_clap_music` conversion it was query-insensitive.

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

**Inference mode, and why MAEST is not compiled.** Every inference forward
(MAEST, CLAP audio/text, the supervised head's validation and projection) runs
under `torch.inference_mode()`, not `no_grad()`: the outputs go straight to numpy,
so the weaker autograd guarantee costs nothing. Measured, it is neutral — an
in-process, interleaved A/B of the same shapes gave 0.96× (batch 3) and 0.97×
(batch 12), inside the noise floor — so it is kept for correctness of intent, not
for speed.

`torch.compile` on the MAEST transformer was measured twice and **rejected**.
Interleaved, same-process comparisons (mel precomputed, no decode, no shape churn,
alternating order, 6 rounds) reproduce the same slowdown:

| measurement | eager | compiled | slowdown |
|---|---|---|---|
| forward, 3 excerpts | 388 ms | 890 ms | 2.29× |
| forward, 12 excerpts | 1643 ms | 3875 ms | 2.36× |
| `analyze_batch`, 8 real tracks | 4.3 s | 7.7 s | 1.78× |

`dynamic=True` was no better (896 / 3927 ms), the queue harness agreed (compiled
~1.4–1.6 s/track versus 0.4–0.8 eager on the 48-track sample, plus the ~15 s compile
in run 1), and the compiled outputs match eager to ~3e-6 — a kernel-speed regression, not
a numerical one. The compiled model's *best* forward (643 ms) is still slower than eager's
*median* (388 ms), so the ordering is robust to the run-to-run spread. An earlier
"promising" smoke reading was an artefact — the fixture yields one excerpt per track, so the
apparent batch-12 was really a batch of 4, and `warmup` had hidden the recompile. Retry only
on CUDA, and only with the paired measurement below.

Decide any future accelerator change on an idle machine, comparing best-run numbers (the
harness does **not** call `warmup`, so run 1 includes first-call/compile cost and run 2+
is steady state):

```sh
.venv/bin/python tests/profile_pipeline.py --root /path/to/music \
    --sample 48 --stages maest --repeat 3
```

---

## Audio feature extraction (`librosa` + optional `madmom`)

**Audio feature vector (6D):**
`[tempo_norm, key_cos, key_sin, mode, energy_norm, danceability]`

- Tempo: **madmom** (`RNNBeatProcessor` + comb `TempoEstimationProcessor`) when importable,
  else `librosa.feature.tempo()` (same audio clip; `FEATURES_VERSION` 6+).
  Normalised to `[0, 1]` over **60–250 BPM** (`BPM_LO`/`BPM_HI` in `backend/tuning.py`);
  the upper end matches madmom's `MAX_BPM`, so the estimator's whole range survives
  normalisation instead of saturating at a lower ceiling.
  Classical-feature excerpts stay at **22.05 kHz** for librosa; madmom receives a
  **44.1 kHz** resample of that buffer only. The default is **one BLSTM**
  (`_madmom_fast_enabled()`), measured ~4× faster than the 8-network ensemble
  (1032 → 246 ms/track) and within 1 BPM on 26/29 real tracks; on an exact
  half/double disagreement with the librosa estimate, `reconcile_tempo_octave()`
  prefers librosa. Set **`TRACKSPACE_MADMOM_FAST=0`** for the ensemble (pick one
  setting per library and keep it stable, or bump `FEATURES_VERSION` / purge features
  when toggling). Profile locally: `.venv/bin/python tests/profile_madmom_tempo.py`.
- Key: Krumhansl-Schmuckler algorithm on `librosa.feature.chroma_cqt()`
- Energy: `librosa.feature.rms()` (log-scaled)
- Danceability: onset strength autocorrelation regularity

Key is encoded on the **circle of fifths as a unit circle** in 2D:
`key_cos = cos(2π · fifths_pos / 12)`, `key_sin = sin(2π · fifths_pos / 12)`.
This preserves harmonic topology — keys a fifth apart are geometrically close,
and the circular wrap is seamless.  Mode (major=1, minor=0) is a 3rd dimension.

### Perceptual re-encoding (`embeddings/feature_encoding.py`)

The *cached* vector above is not perceptually uniform, so the composite
`features` block is re-encoded from it (a pure function, no re-analysis) before
masking:

- **Camelot wheel.** Major keys sit on an outer ring at their circle-of-fifths
  angle; minor keys on an inner ring rotated three fifths (90°) so a relative
  major/minor pair (C major / A minor — the same key signature) shares a radius.
  This replaces the cos/sin + separate mode bit, which put a relative pair as far
  apart as two unrelated keys.
- **Tempo spiral.** One turn per octave: the angle is the fractional octave of the
  BPM and the radius grows with log-BPM, so 60/120/240 BPM land on the same ray.
  `tempo_norm` is linear in BPM, which put half/double-time siblings far apart.

The block keeps six columns — `[camelot_x, camelot_y, spiral_x, spiral_y,
energy, danceability]` — so the composite gather, PCA and t-SNE are unchanged.
`FEATURE_ENCODING_VERSION` is part of the features row-cache key and the
projection recipe is bumped; the raw cache row is untouched, so this needs **no
re-analysis**. The default `features` weight is 0, so the combined map is unchanged
until the *Audio* slider is raised.

**Rhythm vector (14-D)** — `onset_density`, `onset_strength_mean/std`,
`beat_confidence`, `pulse_clarity`, `syncopation`, `kick_periodicity`,
`bass_ratio`, `spectral_centroid_norm`, `spectral_rolloff_norm`,
`spectral_flux`, `dynamic_range`, `loudness`, `rhythmic_complexity`. All
normalised to [0,1]; the STFT / onset envelope / beat track are computed once
and reused across dimensions. This block is what separates rhythmically
distinct neighbouring genres.

---

## Track structure (`backend/embeddings/structure.py`)

Cheap DSP, not a model: a whole-track mono decode at **2205 Hz** yields an RMS
envelope (~4 Hz frames, ~3 s moving average), and four coarse sections are read
off it by thresholding against the track's own 95th-percentile level:

- **intro start** — first *sustained* crossing of 5 % of the peak level (≥ 1 s), so
  lead-in silence is skipped; 0 when the track starts immediately. The hover
  preview seeks here when cycling to the intro.
- **intro end** — first *sustained* crossing of 80 % of the peak level (0 when the
  track starts loud).
- **drop (peak)** — the start of the highest-energy 30 s window; this is what the
  hover preview seeks to.
- **break** — the longest sustained dip below 60 % of the peak level (≥ 8 s) that
  returns to full energy afterwards. A quiet opening or trailing fade touches an end
  and is not a break.

The detector (`detect_sections`) is a pure function over the envelope, so it is
unit-tested without audio. Boundaries are compensated for the half-window smoothing
lag, and the first window at the maximum energy wins (not an arbitrary later
equal-energy window). Absent sections are stored as `NaN` and surface as `null` over
the API. `STRUCTURE_VERSION` is **2** (was 1) since `intro_start` was added; old
blobs are purged on startup and refilled by the queue.

**Decode dominates the cost, and it is cache-backed.** A six-minute MP3 is ~0.3–0.4 s
warm; the stage runs in a **thread** pool (`TRACKSPACE_STRUCTURE_WORKERS`, default
4) rather than a process pool — `soundfile` reads release the GIL, so threads scale
(~2.1× at 4 workers measured) and no spawned child pays the backend re-import the
`features` process pool pays. A one-off whole-library backfill is ~an hour of decode.

**Genre-spanning flag.** The per-excerpt MAEST vectors (already cached for weak
supervision) give a second, free signal: `supervised.is_genre_spanning` flags a track
whose excerpts disagree (mean pairwise cosine dispersion ≥
`SPANNING_DISPERSION_THRESHOLD`), i.e. intro and drop sit in different corners of
the space. The route `/api/tracks/structure` returns the six timestamps plus this
flag, read-only over `FeatureCache` — no audio is decoded on a request thread.

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

**Weights are not comparable across sources.** The blocks have very different global
geometry — a 519-D near-one-hot logit block vs a 14-D DSP block vs a 512-D CLAP
embedding — so a weight of `0.5` does not mean the same thing per source; a slider
can look inert because it is moving a block whose distances are already tiny or huge
relative to the others. The clustering path standardises each column first
(`clustering.build_weighted_matrix`); the layout deliberately does not, because column
statistics depend on the current track set and a *stateful* recipe would make a track's
projection drift as the library grows (see the `layout.py` docstring). Matching the
distributions without that drift needs a transform fitted once on a **frozen reference
set** and cached in a chart artifact — an open item in `TODO.md`, together with the
`chart_revision` split the landmark extension needs.

**Semantic weighting with composite vectors:** Folder centroid decomposition uses
the **full** composite matrix.  CLAP text directions (tags, folder-name fallbacks)
are placed in the **CLAP column span** (`clap_col_lo : clap_col_lo + clap_dim`),
which depends on *sources* order — `mat[:, :clap_dim]` was wrong when CLAP was
not the first block.

---

## Projection methods

`compute_projection()` projects cached embeddings to 2D. The backends live in
`embeddings/projection_backends.py`. (The frontend requests `method=tsne`; `pca` is
used internally for the live preview during generation.) The request carries the
lens's `sources=` subset and optional `w=` weights; both are part of
`layout_revision`, so each lens caches its own coordinates.

| Method | Backend function | Notes |
|--------|-----------------|-------|
| `"tsne"` (default) | `_project_tsne` (openTSNE → cuML → mlx-vis → scikit-learn) | Best local cluster preservation; slower |
| `"pca"` | `_project_pca` (numpy SVD) | Instant; used for live incremental updates during generation |

**Incremental t-SNE (openTSNE).** On by default (`TRACKSPACE_OPENTSNE=0`
disables it), `_project_tsne` routes to `_project_tsne_incremental`, which fits
an embedding once per *vector config* and then places only newly added tracks into
it via `TSNEEmbedding.transform` instead of refitting the library. The production
fit uses the openTSNE-only iteration budget in `projection_config`
(`OPENTSNE_N_ITER=100`, `OPENTSNE_EARLY_EXAGGERATION_ITER=100`); the fallback
cuML / mlx-vis / sklearn backends keep their own iteration defaults and are
untouched. Measured on this machine (7,153 tracks, warm OS cache, single runs),
the production 100/100 fit is **~3.5 s** (6.0 s at 100/250, 12.4 s at 500/250).
The transform-vs-refit evidence below was measured on the real library
(n=14,163, 50-D) under the old 500/250 iteration budget: a cold fit is 23.8 s, a
`.transform` of 708 new tracks is **1.28 s (19× faster)**, and placed-point
neighbourhood recall@10 is 0.313 vs the base layout's own 0.360 — a ~9% fidelity
cost. Its cold fit matches scikit-learn's and its full-fit recall is slightly
*better* (0.360 vs 0.342), so it is also the better-quality backend. Key facts:

- The fit cache keys on `_composite_config_token` (sources, mask, blend, weights,
  versions) plus the semantic-basis params — **not the track set**, so adding tracks
  reuses the fit. Changing sources/mask/blend/weights/context refits.
- The cache key is namespaced with the projection recipe version
  (`|opentsne_recipe=<PROJECTION_RECIPE_VERSION>`), so bumping
  `PROJECTION_RECIPE_VERSION` (e.g. when the production iteration budget changes)
  invalidates every fitted embedding — in memory and persisted — and refits rather
  than reusing a stale one. Direct callers always pass the original config key, so
  the namespace is added exactly once.
- The semantic-basis param key (`_semantic_basis_param_key`) keys on the folder
  *set*, not the full path set, so a track added to an existing folder reuses the
  chart (and is transformed into it) while a new *folder* is a new direction and
  refits. The basis cache key deliberately uses `_composite_config_token`, never
  `_composite_param_key`, or the full path set would leak back in.
- Tracks already in the fit are sliced from the cached coordinates, so they never
  move between incremental updates. New tracks are PCA-reduced with the *fit's* PCA
  basis before `transform` (not the current matrix's basis).
- Additions above `TRACKSPACE_OPENTSNE_REFIT_FRACTION` (default 0.3) of the fitted
  set trigger a full refit for quality; `TRACKSPACE_OPENTSNE_MAX_POINTS` (default
  20000) falls back to the non-incremental chain above the cap.
- `TSNEEmbedding.transform` recentres the fitted embedding in place, so the cached
  coords are re-read from the embedding after every transform.
- CPU-only, so it does **not** take `inference_lock` (unlike mlx-vis).
- The fitted embedding is **persisted** (`tsne_fit_cache` table, keyed by the same
  config, capped at 8) so a restart transforms rather than refits. The blob is
  magic-prefixed and unpickled defensively: an unreadable or older blob reads as a
  miss and simply refits, so a library upgrade can never load an incompatible pickle.
- The in-memory fit cache holds 8 entries, one per lens, so a full five-lens cycle
  never evicts a fit it will need again.

PCA is never shown in the UI toggle — it is only used internally for fast intermediate
projections while embeddings are being generated.

**scikit-learn fallback pre-reduction.** The final fallback (cuML absent, library above
`TRACKSPACE_MLX_TSNE_MAX_POINTS`) runs scikit-learn t-SNE, which does not reduce
dimensions itself. Raw composite vectors are ~2,000-D and Barnes-Hut on that directly is
minutes-slow at library scale — long enough that the real layout never lands and the map
stays on the PCA preview. The fallback therefore PCA-reduces to
`TRACKSPACE_SKLEARN_TSNE_PCA_DIM` (default 50) first, mirroring the openTSNE path;
set it to 0 to disable. cuML and mlx-vis reduce internally and are untouched.

### Lens gauge (why a lens switch does not flip the map)

t-SNE's objective is invariant to rotation, reflection and uniform scale, so two
lenses over the same library land in arbitrary, unrelated frames — switching lens
looked like the whole map flipped even though neighbourhoods were unchanged.
Measured on the real library, **6 of 10 lens pairs differed by a reflection**.

`embeddings/lens_gauge.py` fits a Procrustes (Kabsch) **similarity** transform —
rotation, reflection, uniform scale, translation — from each new layout onto a
reference layout, using the tracks the two share. `compute_projection` calls
`gauge_layout()` just before `_normalise_coords`, so the gauged coordinates are what
gets normalised, cached and persisted; a restart therefore replays already-gauged
layouts and re-anchors consistently.

Measured end-to-end on the five real cached lenses: **mean pairwise RMSE 0.518 →
0.305 (−41%), every pair improved**, two reflections corrected. The transform is
shape-preserving, so this changes only the gauge and never the structure —
distance-ratio spread 2.8e-16 and 100% nearest-neighbour preservation.

The family key is `source-versions | track_set_key(paths)`: layouts are only
comparable over the same track set and cache generation. It deliberately excludes
sources/mask/weights — keying on anything lens-varying would give every lens its own
anchor and the alignment would silently never fire (`track_set_key` exists so that
coupling is explicit rather than an accident of a hash slice, and
`test_track_set_key_is_lens_independent` pins it).

Two alternatives were measured and **rejected**:

- **Anchor-free canonical gauge** (rotate each lens onto its own PCA axes with a
  skew-based sign rule) reached only 0.479 and made 4 of 10 pairs *worse* — the
  principal axes are unstable when the cloud is near-isotropic.
- **Axis flip/swap only** (no rotation/scale) was worse than Procrustes on every
  lens.

The reference lens barely matters (mean RMSE 0.299–0.316 across all five choices),
so it is not configurable: whichever layout is registered first for a family wins.
Fitting is robust to partial overlap — a fit on 5% of the tracks scored 0.3756 vs
0.3723 on all of them — so a lens with a smaller eligible set still aligns.

This generalises the older `_pca_ref_cache` trick in `projection_backends.py`, which
already did exactly this for the PCA preview so successive previews would not spin.

### Projection caching

Projection results are cached at three levels:

**Server-side (revision-keyed LRU + disk):**
`_revision_layout_cache` in `embeddings/layout.py` is an `OrderedDict[str, list[dict]]`
(capped at 32 entries); on a miss it reads the persisted copy from the
`layout_cache` table in `audio_features.db` (`FeatureCache.get_layout` /
`put_layout`, capped at 32 revisions). The in-process LRU dies with the process, so
without the disk tier every restart re-fits all five lenses (seconds each —
measured; see *Projection methods*); the disk
tier turns that into a ~5 ms read (~356 KB for 14k tracks). The key is
`layout_revision` — a deterministic SHA-256 computed by
`compute_layout_revision()` from sorted eligible paths, method, sources, feature
mask/blend, folder/tag context, and cache version constants. Because the key is
content-addressed, row order in the input data does not affect cache identity, and it
is stable across restarts.

**Coverage and revision reads are presence-only.**
`batch_fetch_maps` returns `{fingerprint: True}` sets from
`FeatureCache.presence()` — one query with a boolean column per source — rather
than decoding every blob (~328 MB per status request). Only
`_gather_composite_vecs` reads the vectors themselves, through its own
`_fetch_source_map`. `presence()` results are memoised in-process keyed on the
cache's write epoch plus the source versions and a fingerprint-set digest, so
repeated status polls and projection requests within one epoch skip the scan.

**Measured cache work (this machine, 7,153 tracks).** The full six-source
`FeatureCache.get_all` read is 401 ms cold → 28 ms on a warm repeat, bounded by
the in-memory LRU's byte budget (`LIBRARY_MEMORY_BYTES` = 384 MiB, entry cap
`LIBRARY_MEMORY_ENTRIES`; both in `feature_cache.py`, applied by
`trackspace_state.py` and the eval CLI). The `presence()` memo above is
92.8 ms → 0.7 ms. A composite cold re-gather fell from 43.7 s to ~0.35 s
(cache-warm measurement) when the per-row supervised-version SQLite read was
replaced by one snapshot per gather — `layout.py` reads `supervised_version`
once and reuses it for every source row and key. The prepared source-row LRU in
`layout.py` is bounded by bytes (`_SOURCE_ROW_CACHE_BYTES_MAX` = 256 MiB)
rather than an entry count.

**Client-side (query-string → {revision, positions} map):**
`_layoutCache` in `coordinators/embedding-coordinator.ts` maps
`_projectionQueryString()` → `{ revision, positions }`, capped at 8 entries with
LRU eviction. Before fetching `/api/embeddings/projection`, the coordinator checks
whether a cached entry's `revision` matches the server's current `layout_revision`
(obtained from `/api/embeddings/status`). If it matches, cached positions
are applied directly — no projection round-trip. This handles A→B→A cycles
(e.g. toggling TSNE↔UMAP↔TSNE) without redundant server work.

**openTSNE fit cache (incremental path):**
`_tsne_fit_cache` in `embeddings/projection_backends.py` holds up to 8 fitted
embeddings keyed by vector config (not track set), with a disk tier behind it
(`tsne_fit_cache` table). It sits *below* the revision-keyed LRU: the revision
changes when tracks are added, so the LRU misses, and the fit cache turns that miss
into a `transform` of only the new rows. The two caches answer different
questions — revision = "same exact view?", fit config = "same vector space?".
Both the in-memory and persisted keys are namespaced with the projection recipe
version, so a recipe change (e.g. the production `n_iter` budget) makes every old
fit a miss and refits it under the new key.
The fit cache is guarded by `_fit_cache_lock` (it is mutated from request threads
and the prefetch worker) and is only written for cached fits, so the Jacobian's
throwaway perturbation fits never evict or persist the live fit.

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

**Rank-deficient blocks and `gesdd`.** A folder whose tracks are nearly collinear in the
composite space gives a within-folder PCA block that is finite but numerically
rank-deficient. `np.linalg.svd` drives LAPACK's `gesdd`, which can raise *"SVD did not
converge"* on such a block and 500 the whole projection request. All SVDs in the layout /
clustering path go through `embeddings/linalg_utils.svd_singular`, which retries with scipy's
`gesvd` driver — it converges on the same input — and the folder PC helper also skips a
non-finite block outright.

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
openTSNE fit and composite caches include it via `_source_versions_token`. The run
id is read from the cache **once per gather** (never per source row) and that single
snapshot drives the composite/row keys and the versions token, so a retrain still
invalidates everything coherently on the next gather without a metadata read for
every row.

**A changed input space needs a retrain.** The head's `input_sources` and `in_dim`
are recorded at training time; `project_supervised_missing` refuses to project new
tracks when the gathered input space no longer matches (it logs "input space changed"
and returns 0). Any change to `FROZEN_SOURCES` — e.g. adding the `clap` block or
removing a source — therefore requires one manual retrain (the *Train*
button in the Analysis panel's *Folder head* section) before new tracks get
supervised rows again.

**Training scope is selectable.** `POST /api/supervised/train` takes an optional
`folders` list of virtual folders (a root id or a subtree). Only tracks under those
folders become training examples, so a head can be learned from one root while still
being applied to the whole library. An omitted/empty list trains on every root. The
selection rides on the singleton job's payload and is recorded in the head meta
(`train_folders`).

**Triggers and staleness.** Training is a library-wide singleton (`"library"` dedupe
key). It is queued manually (the *Train* button in the Analysis panel's *Folder
head* section) or automatically once after the first MAEST pass drains while no
head exists. New tracks analysed later are
projected through the current head at the top of every cluster pass (a cheap matmul), so
they get a row without a retrain. A retrain is an explicit user action — folder
reorganisation is invisible to the filesystem watcher.

### Head bundles: export, import, compare, fan-out

The head is a **portable artifact**, not just local state. Training also computes
per-folder **centroids** in the 128-D supervised space (`compute_centroids`) and stores
them in the head payload, so a nearest-centroid lookup can assign a track to a folder
without a retrain (`predict_folders`).

- **Export** (`GET /api/supervised/export`) writes a self-describing bundle
  (`HEAD_BUNDLE_FORMAT`, versioned) holding the weights, the meta sidecar and the
  centroids.
- **Import** (`POST /api/supervised/import`) installs an uploaded bundle under a
  **fresh local run id**. The run id is the cache version for every `supervised`
  row, so reusing the source library's id would make the imported head collide with
  rows already cached here; a fresh id forces a clean re-projection. A bundle whose
  recorded `source_versions` differ from the running encoders is **refused** with a
  409 and a per-source `source_mismatches` map, because it would project the new
  library into the wrong space; the client can retry with `force=1` once the user has
  accepted the mismatch.
- **Compare** (`POST /api/supervised/compare`) scores shared folders by centroid
  cosine. It reports `compatible: false` — and no similarity — when the two heads
  differ in `in_dim`, `input_sources` or `source_versions`, because a cosine across
  two different encoder spaces is meaningless.
- **Fan-out** (`POST /api/supervised/fanout`) predicts a folder per track by
  nearest centroid and moves the files into subfolders of the target folder. Class
  names are virtual paths that include the *training* root id, so
  `folder_subpath` strips the first component before using it as a subfolder; a
  class that is a root itself maps to the target root and is skipped. It is
  **dry-run by default** — `apply` is the only step that touches disk — and a name
  collision gets a ` (n)` suffix rather than overwriting a file.

**Only place a track that is close enough.** Nearest-centroid alone forces *every*
track into *some* folder, even one that resembles none of the learned folders.
Training therefore also stores a per-class **match floor**
(`compute_match_thresholds`): the 5th percentile of that class's own training rows'
cosine to its centroid. At fan-out, a track moves only when its winning similarity
clears its class's floor; otherwise it is reported `unmatched` and left where it is.
The floor is calibrated to how tightly the head packs each folder, so a diffuse folder
gets a lower floor than a tight one, and an old head without floors (or a class with no
training row) matches everything, preserving the previous behaviour.

**The same prediction drives the canvas colouring.** `GET /api/supervised/predictions`
returns the nearest learned folder per track (path → class name) plus the subset that clears
its match floor, read straight from the cached `supervised` rows — no projection, no encoder
work. The *Predicted folder* colour mode (`model.colorMode === "head"`) uses it so the map
shows where the head *thinks* each track belongs; the client refetches it when the head changes
(a drained supervise job or an import). The View tab's *Predicted* select chooses between
*Confident only* (the floor-gated subset) and *All nearest*.

**Unmatched tracks can still be filed by genre.** With the imprint's *Sort unmatched
into Generic/…* toggle on, a track that clears no learned folder's floor is filed under
its top Discogs label as `Generic/<Genre>/<Style>` instead of being left loose. MAEST's
519 `Genre---Style` logits — already gathered for the projection — are the label source;
`generic_subpath` mirrors the taxonomy's depth and sanitises each component, and returns
nothing when the label table is unavailable, in which case the track stays `unmatched`.

**Fan-out is journalled and undoable.** Every applied move is written to the
`fanout_journal` table (source, target, batch id) *before* the file is moved, so
`POST /api/supervised/fanout/undo` can restore the previous layout. The journal
records the *actual* target, so a ` (n)` collision suffix is undone back to the
original name; a batch is undoable once (`GET` on the same path reports whether one is
available).

**Recursive fan-out, origin memory, and the empty-dir sweep.** With `recursive`, nested
files are gathered and lifted into the learned folders directly below the target root. Each
moved track's pre-move relative folder is remembered in the `origin_folders` table,
keyed by fingerprint, so the original layout stays inspectable after the move journal is
pruned. On `apply`, the emptied old subdirectories are swept deepest-first, bounded to
the fan-out root (undo recreates the directories it needs, but does not re-sweep).

**Imprinting an external folder.** A fan-out normally targets a folder inside the
library. *Imprint* extends it to a folder elsewhere on disk: the target is registered as
an **ephemeral root** so the scan/analysis/fan-out paths can resolve it, its tracks are
analysed for exactly the sources the head consumes (`SUPERVISED_INPUT_SOURCES`:
`maest`, `maest_logits`, `rhythm`, `clap`), and the plan/apply then runs the ordinary
fan-out against that root. `POST /api/supervised/imprint/analyze` registers the root and
queues analysis, `GET /api/supervised/imprint/status` reports readiness, and
`POST /api/supervised/imprint/finish` removes the root again. The ephemeral root is
persisted by path in `roots.json` (`temp_roots`) so a restart mid-imprint can still
finish and clean it up, it is marked **external** (it is a projection target, not the
user's library), and a target that contains an existing library root is refused
rather than silently unregistering that root. `GET /api/supervised/imprint/temp_roots`
lists the unfinished imprints a closed browser left behind and
`POST /api/supervised/imprint/discard` drops them without touching the files.

### Source-weight Jacobian (live slider drag)

The layout is a non-linear function of the source weights, so a slider drag cannot
re-project without a full t-SNE refit. `compute_projection(..., with_jacobian=True)`
returns a first-order finite-difference Jacobian `d(position)/d(weight)` around the
current weights (`_weight_jacobian`: one projection per source). The *Source weights*
panel fetches it once per weight vector when opened and applies it linearly while a
slider moves; the full re-cluster + refit still runs on **Apply**, and the base layout
is fetched and drawn first so the Jacobian's S+1 fits never contend with it.

The perturbation fits are uncached (`cache_fit=False`): their one-off weight configs
would otherwise evict the live fit the base layout reuses, turning the *next* layout
into a full refit. `compute_projection` also reports phases (`on_progress`): the base
layout first, then the Jacobian per source, streamed as `layout_progress` (the layout is
a plain request, not a queued job) so the canvas pill and status bar show progress.

The approximation is only valid for the small deltas of a drag — the sign and direction
are right, the magnitude only to first order — and a source already at weight 0 has no
derivative, so raising it from 0 is a step the linearisation cannot describe.

**Why the refit finite difference is not a derivative (measured).** t-SNE is
non-convex and its init is data-dependent, so a small weight change lands in a
*different local optimum*: on synthetic data the refit-FD had `cos ≈ −0.17` with a
step, and PCA's own Jacobian `cos ≈ 0.9998`. The refit-FD is therefore not a
derivative, and applying it pins dots to the [0,1] corners.

### Smooth landmark extension (weight-aware preview)

`backend/embeddings/landmark_projection.py` replaces that finite difference with a
*fixed, smooth* function of the weights, differentiated in closed form. Landmarks
`l` are actual tracks; with their unweighted per-source blocks `u_ls` and cached
squared distances `Delta_ils = ||u_is − u_ls||^2`:

```
D2_il(w) = sum_s w_s Delta_ils        (the same additive metric)
a_il(w)  = softmax_l(−D2_il(w) / tau) (normalised affinities)
g_i(w)   = sum_l a_il(w) c_l          (landmark coordinates)
dg_i/dw_s = −(1/tau) sum_l a_il Delta_ils (c_l − g_i)
```

- `c_l` is fitted once, at `w0`, against the existing base t-SNE layout (ridge
  least squares) and cached per `layout_revision` in `landmark_model_for`; refitting
  it per drag would be the "recompute a fitted quantity" mistake. The chart is
  persisted beyond the layout LRU so an eviction cannot change an "identical" request.
- The derivative is **exact for this nonlinear `g`** (not a finite difference) and
  avoids differentiating `sqrt(w_s)`: the squared distances are already affine in `w`.
  Verified against central differences of the same `g` (relative error < 1e-3).
- The preview is the residual form `y~_i(w) = y_i^0 + g_i(w) − g_i(w0)`, so it
  reproduces the base layout *exactly* at `w = w0`. The frontend's `y^0 + J dw` is
  its first-order truncation.
- **The translation + uniform-zoom component is projected out** of each source's Jacobian
  (`_remove_gauge`): fit `J_i ≈ α + β·(y_i^0 − centroid)` and keep the residual. A
  Jacobian that only offsets or scales the whole map carries no shape information — the
  canvas refits the view anyway — so it reads as "nothing happened". Only the shape change
  is returned, and the extremes of the map define the frame.
- The frontend **re-normalises every preview state to the unit square**, matching the
  server's min-max normalisation of every layout, so a drag cannot drift or zoom the map
  out of frame.
- Cost is `O(N*M*S)` after the `Delta` cache (128 landmarks: ~51 MB at 20k
  tracks). The route uses it for `?jacobian=1` and falls back to the refit-FD when
  the chart cannot be fit (too few tracks).
- **Limit:** a frozen reference chart cannot reorganise its own geometry to match a
  radically different metric, so insertion and chart rebuilding stay different
  operations. The 2-D fidelity across the weight range is an empirical question, not
  established by the construction.

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

`stats()` / `stats_by_kind()` / `failed_keys_by_kind()` are memoised per queue
epoch with a 1 s TTL (`_STATS_MEMO_TTL`): the epoch only advances for this
process's own mutations, and the TTL caps the staleness when another process
writes the same queue.

### Job kinds (pipeline order)

```
scan      → ID3 tags + Chromaprint fingerprint   (CPU, batch 8)
features  → classical 6-D + rhythm 14-D          (CPU, N processes, batch = N)
structure → intro/drop/break timestamps          (CPU, N threads, batch = N)
maest     → MAEST embedding + 519 style logits   (GPU, batch 4, gated on model)
clap      → CLAP audio embedding                 (GPU, batch 4, gated on model)
supervise → folder-supervised projection head    (CPU, singleton, whole-library)
cluster   → kNN graph + multi-resolution Leiden  (whole-library singleton)
layout    → warm each lens's 2-D projection      (CPU/GPU, one job per lens)
```

Names live in `backend/jobs/kinds.py` so enqueuers, workers, and routes cannot
drift apart.

`clap` sits after `maest` so the primary genre signal lands first and the map
renders while the CLAP backfill continues; its drain triggers a re-cluster like any
other per-track stage. (CLAP is not the long pole: measured ~0.14 s/track, one of the
fastest stages — see *Per-stage throughput*.)

**Layouts are pre-warmed behind the scenes.** A t-SNE fit is seconds long at library
scale (measured; see *Projection methods*), and the UI's lenses (combined + one per
source) each have their own
`layout_revision`, so the first switch to a lens would otherwise wait for a fit. Once
the `cluster` pass drains — the last consumer of the cached vectors — the queue
enqueues one low-priority `layout` job per lens (`PROJECTION_LENSES` in
`projection_config.py`, kept in step with `frontend/src/lib/lenses.ts`). Each job
calls the same `compute_lens_projection` service the HTTP route uses, so it computes
the *same* `layout_revision` and populates the revision-keyed position cache the
client will later read. A lens whose revision is already cached is a no-op, so the
sweep after every cluster pass costs a revision check, not a refit. The status bar
shows it as **Preparing views** (one job per lens, driven by the stage step count).

**A user projection preempts the prefetch.** A fit is seconds long and holds the shared
accelerator lock, so without a gate a lens switch could wait behind several prefetched fits
(the "never-ending computing layout" case). `services/projection_priority.py` serialises
projection fits and lets a waiting user request run before any further prefetch starts: the
prefetch yields the slot whenever a user request is active or waiting, and a user request
waits at most the one fit already in flight. The route calls
`compute_lens_projection` with `user=True`; the prefetch passes `user=False`.

**The layout phase is correlated with its request.** `layout_progress` echoes the request's
query string, and the client ignores a phase whose query is not the request it is currently
awaiting — with two overlapping requests, a stale phase must not overwrite the newer one's
progress. The canvas pill is labelled with the *target* lens (`Loading CLAP timbre …`) plus an
elapsed readout, because the map keeps showing the previous lens while the new one fits. The
speculative `layout` prefetch is background work: the status bar reports it, but the canvas
hides it.

**Progress is reported per job and per phase.** Each completed per-track job emits a
`progress` event on `/api/jobs/stream`; the library-wide singletons instead
report named phases as `stage_progress` events — `supervise` emits `gather`,
`mining`, `training (epoch i/N)`, `projecting`, `cluster` emits `vectors`,
`pca`, `graph`, `leiden (i/N)`, and `layout` emits `layout <lens>`. The latest phase
is echoed in `/api/jobs/status`
(`kind.stage`) so a reload mid-run still renders it. The frontend reduces the
whole snapshot to one summary — overall `finished/total` plus the running stage —
in `frontend/src/lib/pipeline.ts`, shown both in the status bar and as the
on-canvas indicator.

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

### Per-stage throughput

`tests/profile_pipeline.py` drives the real handlers through the durable queue over a
deterministic sample and reports s/track and tracks/min per stage, so changes here are
comparable. It shares one feature cache across the stages of a repeat (so `supervise` /
`cluster` see the per-track rows) and rebuilds it per repeat (so each stage redoes its
work), while the `features` process pool and the MAEST/CLAP models survive across
repeats. Run it on an **idle machine**: under load the run-to-run spread is larger than
the effect being measured.

Indicative best-run numbers (64-track sample, Apple Silicon; the machine was busy, so treat
them as relative, not absolute):

| Stage | s/track | tracks/min | Notes |
|---|---|---|---|
| scan | 0.156 | 385 | ID3 + fpcalc |
| features | 0.387 | 155 | 4-process pool, warm |
| structure | 0.117 | 512 | 4-thread pool, whole-track decode |
| maest | 0.388–0.43 | 139–155 | **slowest per-track stage**; no `torch.compile` (2.4× slower on MPS) |
| clap | 0.135 | 446 | 3 × 15 s decode + one batched forward |
| supervise | — | — | whole-library, seconds (skips without MAEST rows) |
| cluster | — | — | whole-library, seconds |

A later 48-track run on the same loaded machine gave maest 0.595 / clap 0.162
(best of 3) — the spread between the two sessions is the load, which is why the
`torch.compile` question was settled by same-process, interleaved forward medians
(*MAEST model management*) rather than by cross-session wall-clock.

Two conclusions worth keeping in mind before optimising:

- **Decode is cheap.** A 15 s / 48 kHz segment decodes in ~14 ms and a whole 6-minute
  track at 2205 Hz in ~300–450 ms, so `_DECODE_LOCK` contention is not a material
  fraction of any stage, and CLAP's "decode-dominated" label was wrong — CLAP is one of
  the *fast* stages. The decode lock's cost is the SIGBUS it prevents, not throughput.
- **MAEST is the per-track bottleneck** (~0.4 s/track, ≈4× CLAP). Its transformer
  forward, not decode, is where the end-to-end time goes, so the batch-size default is the
  remaining lever; `inference_mode()` is kept but measured neutral, and `torch.compile` is
  a measured regression on MPS (*MAEST model management*).

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
8. **Revive `done` rows on automatic enqueues, `failed` rows only on explicit
   retries.** A version bump blanks stale cache rows while the durable queue still
   calls that work `done`, so the backfill must revive those rows. A *failed* row is
   a broken file: reviving it on every keep-warm enqueue retries it on every page
   load, forever. Only explicit retriggers (a stage, a track, or `force`) and the
   startup version-purge backfill set `requeue_failed`.

**Terminal failures are not pending work.** Coverage reports
`pending = fingerprinted − analysed − terminally-failed` (per source, deduped by
fingerprint so a failure already covered via a duplicate path is not a gap). The map's
"analysis ready" gate keys on `tracks_pending_maest/rhythm`: if a permanently broken
track counted as pending, the gate would never open and the frontend would stay on the
fast PCA preview instead of fetching the real t-SNE layout. The broken-files panel is
the actionable list behind the `tracks_failed_*` counts.

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

**When it runs.** A re-cluster is queued whenever a per-track stage drains
(`_after_cluster_inputs_changed`), on `POST /api/clusters/rebuild`, and **once at
startup** (`app.main`). The startup run exists because `ClusterStore` is in-memory:
without it, restarting with an already-analysed library leaves nothing to drain, so
`/api/clusters` would report nothing until the user pressed "Rebuild clusters". The
queue's single `"library"` dedupe key collapses the startup run with any drain or
manual request, and a library with no MAEST rows is a cheap no-op.

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

**Source ablation: keep the full blend.** In the user's blind ratings (single
rater, eval `ablation` set), the full default blend ranked first, MAEST + logits
+ supervised second, and MAEST + logits alone third — dropping rhythm/CLAP, and
then the supervised head, did not win.

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
- Applying a new vector costs a full t-SNE refit (seconds — measured; see
  *Projection methods*), so the UI
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
