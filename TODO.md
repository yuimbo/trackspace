# Trackspace — engineering TODO

Concrete, actionable work and deferred items. Design rules live in the docs:
`architecture.md`, `ui_conventions.md`, `audio_pipeline.md`.

## User feedback

- ✅ **Intro/drop/break detection, and a meaningful hover preview.** The `structure`
  stage derives intro end, drop, and break timestamps from a whole-track loudness
  envelope (no model) and caches them; the hover preview now seeks to the detected
  drop instead of a fixed 40 %, and the tooltip shows the sections. See
  `audio_pipeline.md` → *Track structure*.

- [ ] Whacko idea? Add a 3d view for the embedding space?

## Deferred / optional

### Frontend

- [ ] **Projection refit progress.** The t-SNE/UMAP refit is still opaque: the
      canvas shows a static "Computing layout…" pill (`canvas-view.ts`,
      `_embeddingWaitLabel`) while `/api/embeddings/projection` runs for seconds.
      openTSNE exposes iteration callbacks; a phase/percent on the same
      `stage_progress` channel would cover it. Only worth it if the refit is a
      visible fraction of a first run.

- [ ] **Optional: split `coordinators/canvas-interaction.ts`** (~1040 lines, the
      largest frontend file). The ~200 lines of fit-view math (`_vpFromWorldBounds`,
      `_computeFitVP`, `_p80Radius`, `_fitViewToFolderPercentile`, `_fitViewToTracks`,
      `fitView`, `effectiveHoveredFolderPrefix`) are the natural seam into a
      `canvas-fit-view.ts`.
- [ ] **Optional: `lib/event-bus.ts`, `model/persistence.ts`, `api/types.ts`.**
      Extract the EventBus base, localStorage persistence, and shared API response
      types out of `model.ts` / `lib/api.ts`.
- [ ] **Nit:** `clamp` is duplicated in `coordinators/options-panel.ts`,
      `coordinators/canvas-interaction.ts`, and `coordinators/analysis-coordinator.ts`.
- [ ] **Structure preview follow-ups.** The preview always seeks to the drop;
      cycling peak → break → intro on repeated hovers of the same track was
      deferred (product decision). `_fetchStructure` (`controller.ts`) fetches one
      path per hover — a batched request for the visible tracks would cut the
      round-trips. A small section timeline in the tooltip (or a track inspector)
      would make the break/peak markers legible, not just textual.

### Backend

- [ ] **`services/analysis_pipeline.py` — split if it grows again.** Handlers are
      already one factory function per stage; the cluster store and the enqueue helpers
      are the natural next seams.
- [ ] **Drop the dead generation helpers.** `coverage.build_generation_work` and
      `clap.batch_ensure_embeddings` have no callers since the durable queue took
      over (the queue's `enqueue_library_work` filter supersedes them); remove them
      and their re-exports in `embeddings/__init__.py` / `layout.py`.
- [ ] **Cheaper loudness envelope for `structure`.** `analyze_structure` decodes the
      whole file a second time, but the `scan` stage's `fpcalc` already decodes it
      and its raw fingerprint is discarded (only the hash is kept). If chromaprint can
      expose its 16-band frame image, the envelope could ride the scan pass and the
      structure stage would need no decode at all.
- [ ] **Structure on damaged files.** `analyze_structure` calls `librosa_load`
      directly; the resilient probe in `audio_decode.load_resilient_audio_segments`
      tolerates bad MP3 metadata and broken seeks. Reuse it if damaged files show up
      as structure gaps.
- [ ] **Structure coverage in `/api/embeddings/status`.** Deliberately omitted (the
      queue's per-kind stats already report structure progress, and `SourceVersions`
      feeds the layout revision hash, where a metadata-only source must not appear).
      Add a separate count if the status UI ever needs it.

## Product roadmap (analysis)

Ordered roughly by expected value. Each is additive to the cached feature store —
none require re-running an encoder over the library.

- ✅ **Weak supervision.** Folder labels now train a small 128-D projection head over
  the cached vectors, exposed as the `supervised` source for both the map and the
  clusters. Per-excerpt MAEST vectors are cached (`maest_excerpts`) and drive
  hard-negative mining plus per-track label-confidence weighting. See
  `audio_pipeline.md` → *Weak supervision*.
- **Playlist labels.** Playlists do not exist in the product yet; when they do, they
  are another noisy label source for the same head (the training recipe already takes
  a label vector, so this is a data-plumbing change, not a new model).
- **Per-excerpt analysis.** Excerpt vectors are cached and used for mining, and the
  *user-facing* genre-spanning flag (intro vs drop) now rides `/api/tracks/structure`
  and the tooltip (`supervised.is_genre_spanning`).
- ✅ **Text search.** `GET /api/search?q=…&limit=` encodes the query with CLAP's text
  tower (`generate_text_embeddings`) and ranks cached CLAP audio embeddings by cosine
  similarity; the Search panel turns the hits into the canvas selection and clicking a
  result fits the viewport to it. Unblocked by the CLAP checkpoint fix below (re-run the
  CLAP stage first).
- **Search follow-ups.** Folder-scoped search (`folder` param), search-as-you-type
  debounce, and a minimum-score threshold / "select all" affordance. Fit-on-focus is done
  (`CanvasInteractionController.fitViewToPaths`). Perf: `/api/search` calls the *uncached*
  `build_track_infos_fn` (`backend/routes/api_search.py`), so every query re-reads the
  whole library — use `track_infos_cached` (with `library_mtime_signature`) or resolve only
  the candidate paths, as `/api/tracks/structure` now does.
- **MERT-v2.** Deliberately deferred until MAEST shows a measured blind spot — on the
  6-genre validation it reached 96.6% purity.
- **Persist cluster results.** `ClusterStore` is in-memory; a restart recomputes in
  seconds at current library size. Persist only if that stops being true.

## Analysis performance

Making the pipeline faster. Ordered roughly by expected value; measure before and after.

### Parallelism

- ✅ **Scale the `features` stage horizontally.** Done, but *not* with threads:
  the work is CPU-bound and independent per track, yet madmom's BLSTM forward pass
  is GIL-bound, so more features *threads* are slower, not faster (one track:
  0.8 s at 1 thread → 2.5 s at 2 → 10.5 s at 4; the decode is ~10 ms, so it is
  not the cause). The stage now runs each track in a **process** pool
  (`backend/services/features_worker.py`; `TRACKSPACE_FEATURES_WORKERS`, default
  4). Measured on a 64-track real sample: 51 → 91 → 134 → 140 tracks/min at
  1/2/4/8 processes (`tests/profile_features_workers.py`). The pool uses `spawn`
  (not `fork`) because the parent is multi-threaded, and each child re-imports
  `backend.app` (~3.5 s, once, parallel) before `warm()` builds its processors.
- [x] **`_DECODE_LOCK` contention is not material — measured.** Decode is ~14 ms
  per 15 s / 48 kHz segment and ~300–450 ms for a whole 6-minute track at 2205 Hz,
  so even a fully serialised open is a small fraction of every stage (MAEST is
  ~0.4 s/track, CLAP ~0.14). Do not instrument it; the lock's cost is the SIGBUS
  it prevents, not throughput. See `audio_pipeline.md` → *Per-stage throughput*.
- [ ] **Features process pool: cut child startup.** Each spawned child re-imports
  `backend.app` (hence torch, ~3.5 s) before `warm()` builds its madmom
  processors. A `forkserver` pool with a lightweight preload, or a dedicated
  worker entrypoint, would remove that; only worth it if the server is restarted
  often relative to library size.
- [ ] **Tune `scan`.** `batch_size=8` with `TRACKSPACE_FPCALC_CONCURRENCY`
      (default `min(8, cpus)`). Measure whether fpcalc subprocesses or the SQLite
      writes are the limit and raise only the one that is.
- [ ] **Structure pool sizing.** The `structure` stage runs 4 threads by default
      (`TRACKSPACE_STRUCTURE_WORKERS`; measured ~2.1× at 4 on a 16-track sample).
      Sweep the worker count on the real library and confirm threads stay ahead of a
      process pool as the library (and decode mix) grows.
- [ ] **Parallelise within a stage.** `features` and `maest` decode per track
      sequentially; a small per-stage thread pool (decode+compute) could overlap the
      locked open with DSP/GPU work without adding workers.

### Acceleration

- [ ] **Stop using libsndfile for MP3 (correctness, not speed).** It is the SIGBUS
      source (`audio_pipeline.md` → *Audio decoding*) and forces the open lock.
      **The original `torchaudio` plan is stale:** on torchaudio 2.11 the sox/ffmpeg
      file backends are gone — `torchaudio.load` raises `ImportError: TorchCodec is
      required` (torchcodec is not installed). So this now means adding torchcodec +
      FFmpeg shared libs, or shelling out to the `ffmpeg` already on PATH.
      Measured: an `ffmpeg` subprocess is *not* faster (equal on whole tracks,
      ~2.7× slower per 15 s segment from process spawn), so the only win is
      dropping `_DECODE_LOCK` for the SIGBUS invariant.
- [ ] **MAEST: `torch.compile` / `torch.inference_mode` — now the top item.** The
      batched transformer is the measured per-track bottleneck (~0.4 s/track vs CLAP's
      ~0.14; see *Per-stage throughput*). Try `inference_mode()` instead of
      `no_grad()` (safe, tiny) and `torch.compile` on the model, measuring
      cold-start vs steady-state on MPS on a quiet machine before enabling.
- [ ] **MAEST mel front end.** The mel runs on CPU per excerpt; check whether the
      three excerpts can share one batched `torchaudio` transform instead of three
      separate calls.
- [ ] **madmom tempo.** `TRACKSPACE_MADMOM_FAST=1` (one BLSTM) is ~4× faster but
      opt-in. Benchmark the tempo error on a labelled sample and consider making it the
      default (bump `FEATURES_VERSION` if the vector changes).
- [ ] **Projection backends.** `TRACKSPACE_CUML` / `TRACKSPACE_MLX_VIS` are chosen by
      import success; re-benchmark mlx-vis vs openTSNE vs sklearn after any
      torch/librosa bump, and revisit `TRACKSPACE_OPENTSNE` being off by default.

### Dependencies

- [x] **CLAP text/audio alignment was near-degenerate — fixed by switching the
      checkpoint.** The `laion/larger_clap_music` HF conversion is broken: both towers
      collapse to near-constant vectors (text-vs-text cosine ≈ 0.999, audio-vs-audio
      ≈ 0.95–0.996, text-vs-audio ≈ 0.00), so the canonical HF
      `zero-shot-audio-classification` pipeline returns a uniform softmax (0.0834 across
      12 labels on a techno track; white noise, silence and a sine tone give *identical*
      outputs). Confirmed to be the checkpoint, not us: exact weight loading, correct
      `audio=` processor keyword, valid log-mel, CPU=MPS, transformers 4.57=5.3, and an
      open upstream report (HF discussion #2, Dec 2023). `laion/clap-htsat-unfused` and
      `laion/larger_clap_music_and_speech` both discriminate; the latter is the
      music-trained one and is now `CLAP_MODEL_ID` (`EMBEDDING_VERSION` 2 → 3).
      Real-track check: a techno track ranks "techno" 0.59 > "deep house" 0.36; a
      deep-house track ranks "deep house" 0.55 > "ambient" 0.41 > "techno" 0.25.
- [ ] **Re-run the CLAP stage and retrain the supervised head.** The checkpoint switch
      bumped `EMBEDDING_VERSION` 2 → 3, so every cached `clap` row is purged on next
      startup and must be regenerated (decode-dominated — hours on the real library). The
      supervised head consumes the `clap` block (`SUPERVISED_INPUT_WEIGHTS`), so its
      weights are stale against the new vectors — retrain via *Train folder view* once the
      CLAP backfill drains, or its `supervised` block is misleading.
- [ ] **`transformers` is installed at 5.3** against a `>=4.40` floor — a major bump.
      Pin a tested floor and smoke-test the CLAP text/audio paths. (The
      `larger_clap_music` checkpoint bug above reproduces on 4.57.1, so it is a broken
      conversion, not a 5.x regression.)
- [ ] **`madmom` is unmaintained** (last release 2020) and already emits
      `pkg_resources` (removal slated) and NumPy 2.4 `align=0` deprecations. It will
      break on a future setuptools/NumPy. Evaluate replacing the tempo/key estimators
      or vendoring the two processors we use.
- [ ] **`libsndfile`/mpg123 thread safety.** Track the upstream fix for the
      `mpeg_decoder_init` FIXME and drop `_DECODE_LOCK` when it lands.
- [ ] **Pin the GPU/ML stack.** `torch` 2.11 / `mlx-vis` 0.7 are
      unpinned floors; a breaking change would land silently. Add upper bounds for the
      ones that affect embeddings.

### Profiling & benchmarks

- ✅ **Per-stage throughput harness.** `tests/profile_pipeline.py` drives the real
      handlers through the durable queue over a deterministic sample and reports
      s/track + tracks/min per stage, with a best-run summary, skip reasons, and a
      non-destructive scratch-cache design. Baseline (busy machine, relative only):
      scan 0.16, features 0.39, structure 0.12, **maest 0.39–0.43**, clap 0.14
      s/track. Recorded in `audio_pipeline.md` → *Per-stage throughput*.
- [ ] **CLAP `max_batch` sweep.** Measured ~0.14 s/track (~446 tracks/min) at the
      default `batch_size=4` → 12 segments, so CLAP is *not* decode-dominated and
      is one of the fastest stages. Low priority; sweep `max_batch` vs MPS memory
      only if a profile ever puts CLAP near the top.
- [ ] **Purge leftover EffNet cache rows.** `data/audio_features.db` still holds
      `effnet_embedding` / `effnet_version` columns and rows from before the source
      was deleted (~48 MB at 9.5k tracks), and `data/models/discogs-effnet-*.onnx`
      is still on disk. `data/` is gitignored and nothing reads them, so this is
      only worth doing if the disk footprint matters.
- [ ] **Cold vs warm end-to-end run.** The harness gives the warm per-track sum
      (~1.18 s/track across scan+features+structure+maest+clap ≈ 51 tracks/min, so
      ~3.5 h for 10.8k tracks), but a full `force` re-analysis wall-clock and a
      no-op re-run are still unrecorded; capture both.
- [ ] **MAEST batch-size sweep — the top per-track lever.** MAEST is the slowest
      stage (~0.4 s/track), so its batch/compile choices matter most. A sweep on a
      **quiet** machine was inconclusive under load (batch 1/2/4/8/16 spread
      ~70–150 tracks/min with no clean winner; the loaded machine's run-to-run
      spread exceeded the effect). Re-run
      `tests/profile_pipeline.py --stages maest --maest-batch N --repeat 3` when idle
      and pair it with the `inference_mode` / `torch.compile` item under
      *Acceleration*.
- [ ] **Peak RSS / GPU memory.** No memory baseline exists; capture peak RSS and MPS
      allocation per stage before adding workers or batches.
- ✅ **Decode cost breakdown.** ~14 ms per 15 s/48 kHz segment and ~300–450 ms for
      a whole 6-minute track at 2205 Hz; an `ffmpeg` subprocess is equal on whole
      tracks but ~2.7× slower per segment (process spawn). Decode is not the
      bottleneck. See *Per-stage throughput* in `audio_pipeline.md`.
- [ ] **Cluster/projection timing at scale.** `DEFAULT_RESOLUTIONS` and k=20 were
      calibrated at ~9.5k tracks; re-measure once the library doubles.
