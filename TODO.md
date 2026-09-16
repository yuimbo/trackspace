# Trackspace — engineering TODO

Concrete, actionable work and deferred items. Design rules live in the docs:
`architecture.md`, `ui_conventions.md`, `audio_pipeline.md`.

## Deferred / optional

### Frontend

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

### Backend

- [ ] **`services/analysis_pipeline.py` — split if it grows again.** Handlers are
      already one factory function per stage; the cluster store and the enqueue helpers
      are the natural next seams.
- [ ] **Drop the dead generation helpers.** `coverage.build_generation_work` and
      `clap.batch_ensure_embeddings` have no callers since the durable queue took
      over (the queue's `enqueue_library_work` filter supersedes them); remove them
      and their re-exports in `embeddings/__init__.py` / `layout.py`.

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
- **Per-excerpt analysis.** Excerpt vectors are now cached and used for mining, but
  the *user-facing* genre-spanning flag (intro vs drop) is still unused — it needs a
  track-inspection surface and a dispersion threshold, no new inference.
- **Text search.** CLAP's text encoder already produces the semantic directions used
  for weighting; a query box could reuse it for retrieval.
- **MERT-v2.** Deliberately deferred until MAEST shows a measured blind spot — on the
  6-genre validation it reached 96.6% purity.
- **Persist cluster results.** `ClusterStore` is in-memory; a restart recomputes in
  seconds at current library size. Persist only if that stops being true.

## Analysis performance

Making the pipeline faster. Ordered roughly by expected value; measure before and after.

### Parallelism

- [ ] **Scale the `features` stage horizontally.** One worker, `batch_size=2`
      (`backend/app.py`), and the work is CPU-bound (librosa + madmom) and
      independent per track. Run N workers for the kind and measure the wall-clock
      curve; the durable queue already leases per job, so this is a wiring change
      plus a check that concurrent SQLite WAL writers do not contend.
- [ ] **Measure `_DECODE_LOCK` contention.** The decode lock serialises every
      native MP3 open, and now also covers CLAP's 48 kHz decodes alongside
      `features` / `maest`. Instrument the wait time in
      `decode_stderr.librosa_load`; if it is a material fraction of the stage wall
      time, prioritise the torchaudio-decode item under *Acceleration*.
- [ ] **Tune `scan`.** `batch_size=8` with `TRACKSPACE_FPCALC_CONCURRENCY`
      (default `min(8, cpus)`). Measure whether fpcalc subprocesses or the SQLite
      writes are the limit and raise only the one that is.
- [ ] **Parallelise within a stage.** `features` and `maest` decode per track
      sequentially; a small per-stage thread pool (decode+compute) could overlap the
      locked open with DSP/GPU work without adding workers.

### Acceleration

- [ ] **Stop using libsndfile for MP3.** It is the SIGBUS source
      (`audio_pipeline.md` → *Audio decoding*) and forces the open lock. `torchaudio`
      is already a dependency and decodes MP3 via ffmpeg/sox without mpg123's global
      init, which would let opens run unlocked. Prototype behind an env flag, A/B the
      embeddings on a fixed sample, then flip the default and drop `_DECODE_LOCK`.
- [ ] **MAEST: `torch.compile` / `torch.inference_mode`.** The batched transformer
      is the per-track GPU bottleneck. Try `torch.compile` on the model (or at least
      `inference_mode()` instead of `no_grad()`) and measure cold-start vs steady-state
      on MPS before enabling.
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

- [ ] **`transformers` is installed at 5.3** against a `>=4.40` floor — a major bump.
      Pin a tested floor and smoke-test the CLAP text/audio paths.
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

- [ ] **Per-stage throughput harness.** `tests/test_profiling_embeddings.py` and
      `tests/profile_madmom_tempo.py` are ad-hoc. Add one script that reports
      tracks/min and s/track per stage (scan/features/maest/clap/supervise/cluster)
      on a fixed sample, so every change here is comparable.
- [ ] **CLAP stage throughput.** The stage is decode-dominated (3 × 15 s at 48 kHz)
      with one batched forward per track. Measure tracks/min on the real library and
      sweep `max_batch` (the handler passes `batch_size=4` → 12 segments) against
      MPS memory, then pick a default the way MAEST did.
- [ ] **Purge leftover EffNet cache rows.** `data/audio_features.db` still holds
      `effnet_embedding` / `effnet_version` columns and rows from before the source
      was deleted (~48 MB at 9.5k tracks), and `data/models/discogs-effnet-*.onnx`
      is still on disk. `data/` is gitignored and nothing reads them, so this is
      only worth doing if the disk footprint matters.
- [ ] **Cold vs warm end-to-end run.** Record wall-clock for a full `force` re-analysis
      and a no-op re-run; that is the number the parallelism work must move.
- [ ] **MAEST batch-size sweep.** The handler passes `max_batch=12` (4 tracks × 3
      excerpts). Sweep batch vs throughput vs MPS memory and pick a default.
- [ ] **Peak RSS / GPU memory.** No memory baseline exists; capture peak RSS and MPS
      allocation per stage before adding workers or batches.
- [ ] **Decode cost breakdown.** Time open vs read vs resample inside `librosa_load`;
      this decides whether the decode lock actually costs anything.
- [ ] **Cluster/projection timing at scale.** `DEFAULT_RESOLUTIONS` and k=20 were
      calibrated at ~9.5k tracks; re-measure once the library doubles.
