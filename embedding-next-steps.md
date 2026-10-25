# Embedding next steps — implementation plans

Single source of detailed, hand-off ready plans for the embedding/analysis work after
the offline genre benchmark. Findings and *why* live in `audio_pipeline.md`; the
benchmark recipe/CLI live in the README (*Offline genre benchmark*) and
`backend/experiments/genre_space/`; **this file is where the *plans* live**.

Status: the user has **selected `genre_full`** as the preferred genre-block recipe, but
**nothing is wired into production yet** — no app-wide source-weight change has been
made and no new encoder weights were deployed. The four plans below are ranked by
expected value and are deliberately decoupled so WS3 can run in parallel with WS2.

---

## Ranking

| # | Workstream | Why now | Est. (engineering) |
|---|-----------|---------|--------------------|
| 1 | Selected `genre_full` production adoption + single-pass L7 | User selected it; currently the only path is a two-pass, non-production extract | 2–3 days |
| 2 | Optimize analysis / model-centric scheduler + progressive draft map | MAEST is the per-track bottleneck; coordinated-scheduler latency/fairness is unmeasured | ~3 days |
| 3 | MuQ-MuLan audio+text evaluation vs CLAP (needs semantic UI) | Read-only research; can run in parallel with WS2 | 2–3 days |
| 4 | Validation and reproducibility (quality/safety/performance) | Only viable where the harness already exists; **no new model grid**; playlists deferred until the product has them | 1–2 days |

Estimates are engineering effort, **not** runtime-throughput promises. No plan changes
the app-wide default source weights without explicit user approval.

---

## Global constraints (apply to every plan)

- **No app-wide source-weight change.** The genre-block recipe is a separate,
  opt-in recipe id; `FROZEN_SOURCE_WEIGHTS` / `clustering.DEFAULT_WEIGHTS` are the
  app defaults and stay put. A genre-block budget is decided *relative to* the
  existing sources (rhythm, CLAP, supervised), never by overwriting them.
- **One accelerator lock.** `clap.inference_lock` is the single process-wide
  serialisation point for every PyTorch/MLX dispatch. Do **not** define a second
  accelerator lock and do **not** add concurrent GPU dispatch from separate threads —
  MPS/CUDA are not safe for concurrent forwards, and mixed-framework concurrency
  produces IOGPU `commit command buffer with uncommitted encoder` assertions. Any new
  GPU code path either takes `inference_lock` or proves the backend thread-safe under
  mixed-framework load first. This is **distinct** from `_DECODE_LOCK` (below), which
  guards the native MP3 open, not GPU dispatch — two locks for two different failures.
- **Safe native decode.** All MP3 decode goes through
  `backend/decode_stderr.py` / `load_resilient_audio_segments*` (serial `_DECODE_LOCK`
  across the native `sf_open`; whole-track fallback for lying MP3 headers). Do not open
  native audio from a new thread pool without that lock. The **SIGBUS** this prevents is a
  concurrent native MP3 open (`mpg123` layer-table init) — a different failure from
  `inference_lock`'s IOGPU guard above; do not conflate the two.
- **Cache invariants.** Every source lives in `FeatureCache._SOURCES` with its own
  version column, added by the `ALTER TABLE` migration loop; all reads use
  empty-blob-as-absent and purges write empty blobs. Bumping a source's version makes
  its old rows read as **absent** (blanked) and they are recomputed at the new version by
  the startup backfill; **other sources are unaffected** — tests assert old-version rows
  are absent, not that they survive a purge.
- **Durable queue.** Per-track work returns `JobResult.failure(...)` (never raises);
  permanent errors use `retry=False`; model readiness gates on the worker, not the
  handler. Version purges revive `done` rows; only explicit retriggers revive `failed`.
  The queue **already runs one worker per model kind** (MAEST, CLAP), each with its
  model **loaded once at startup and warmed** — resident-once is current behaviour to
  preserve, not a plan assumption.
- **Real data stays local.** No cloud upload of audio, embeddings, or queries. Model
  weights download to the local cache only and never auto-run remote code without an
  explicit flag.
- **Artifacts are preserved, not deleted.** Old gallery exports (`data/layout-evaluation/…`)
  and local benchmark data (~253 MB) and the downloaded HF model cache (~2.4 GB) are
  reported, never silently deleted.

---

# WS 1 — Selected `genre_full` production adoption + single-pass L7

## Goal
Wire the user-selected `genre_full` recipe (logits .40 / maest .20 / layer7 .25 /
mert19 .15) into production as an **opt-in recipe id** with each new block (layer-7,
MERT-L19) having its own cache version, queue jobs, and GPU guard — **without changing
the app-wide default recipe** — and make layer-7 / MERT-L19 extraction **single-pass** so
`genre_full` stops being the ~99 ms/3-excerpt two-pass cost recorded in the benchmark.

## Current verified state
- `genre_full` is **selected by the user** as preferred; production is **not wired**.
  The benchmark presets and measured costs are in `audio_pipeline.md` →
  *Genre-space evaluation conclusions*.
- The user-selected `genre_full` weights (logits .40 / maest .20 / layer7 .25 /
  mert19 .15) are exactly the `genre_full` recipe already tracked in
  `backend/experiments/genre_space/scoring.py` (`maest_logits` .40, `maest` .20,
  `maest_layer7` .25, `mert_l19` .15) — the production opt-in recipe must mirror that
  existing config, not a new set of weights.
- Layer-7 is native 2304-D `(cls + dist + patch-mean)` at **block 6** (= one-based
  layer 7 of 12 blocks), **not** the Essentia tokenpool-768 variant. Today extraction
  runs **two passes**; capturing layer-7 during the MAEST full forward is the
  unimplemented priority.
- MAEST produces `embedding` (768), `logits` (519), `excerpt_embeddings` in one pass
  (`backend/embeddings/maest.py`, `MaestAnalysis`), cached as `maest` /
  `maest_logits` / `maest_excerpts` sharing one version column via
  `FeatureCache.put_maest`. `make_maest_handler` runs under `clap.inference_lock`.
- MERT is an **independent audio encoder** (`m-a-p/MERT-v2-30s`), **not** a MAEST
  extension. Genre-space tooling names its sources `mert_l19` / `mert_l23`
  (1024-dim, `hidden_states[18]` / `[22]`); layer-7 is `maest_layer7` (2304-dim).
  Extraction is pinned to a 40-hex revision and gated on explicit
  `--trust-remote-code` / `--allow-download`.
- No layer-7 or MERT cache columns exist yet. MERT weights are CC-BY-NC 4.0 and are
  never auto-downloaded (benchmark uses explicit `--trust-remote-code --allow-download`).
- Layer-7 gets its **own cache version column** but is produced by the **same MAEST
  forward** as the embedding/logits, so its version is **paired with / dependent on the
  MAEST checkpoint version** via a checkpoint manifest fingerprint — a checkpoint change
  must invalidate layer-7 and maest together, never one alone.

## Handoff scope (opt-in vs app defaults are separate goals)
This plan wires `genre_full` as an **opt-in recipe id** only. Promoting it to the
**app-wide default** is a distinct, later decision that needs explicit approval — it is
out of scope here and would change `FROZEN_SOURCE_WEIGHTS` / `clustering.DEFAULT_WEIGHTS`.

## Concrete files / interfaces
- `backend/embeddings/maest.py` — single-pass L7 capture inside `analyze_batch`'s
  existing full forward; extend `MaestAnalysis` with `layer7` (MERT is **not** a
  MaestAnalysis field — it is a separate encoder).
- `backend/embeddings/mert.py` — **new, independent** module for MERT-L19 extraction.
- `backend/embeddings/feature_cache.py` — `_SOURCES`: add `maest_layer7` (2304-D, own
  version column) and `mert_l19` (1024-D, own version column) via the `ALTER TABLE`
  loop; empty-blob-as-absent reads; `purge_old_versions`.
- `backend/services/analysis_pipeline.py` — `make_maest_handler` writes the
  `maest_layer7` pair **in the same MAEST job** (alongside embedding/logits/excerpts,
  its own version column); a **separate MERT handler** (`make_mert_handler`) writes
  `mert_l19`; each gates on its own model. **No MAEST-stage flag for MERT.**
- `backend/jobs/kinds.py` — add a **`mert` kind** (MERT is a separate worker; no
  reusing the `maest` kind).
- `backend/app.py` — worker wiring + enqueue filter for the `mert` kind.
- `backend/embeddings/projection_config.py` — add an **opt-in** `genre_full` recipe id;
  `FROZEN_SOURCE_WEIGHTS` unchanged. `clustering.DEFAULT_WEIGHTS` untouched.
- `backend/embeddings/layout.py` / `clustering.py` — optional-block handling for the new
  sources in the composite gather (zero-fill) and `REQUIRED_SOURCES`.
- `backend/embeddings/supervised.py` + `analysis_pipeline.py` — decide whether
  `SUPERVISED_INPUT_WEIGHTS` consumes the new blocks (see Risks) and retrain coherence.

## Dependencies
- Single-pass L7 needs the forward-hook / model API to be valid for the pinned
  `discogs-maest-30s-pw-129e-519l` checkpoint.
- MERT-L19 needs its own pinned revision + explicit licence (CC-BY-NC) acceptance and a
  `trust_remote_code` gate.

## Sequence (slices)
1. **Single-pass L7 in `maest.analyze_batch`.** A scoped forward hook capturing the
   exact `cls`/`dist`/`patch-mean` of **block 6** during the existing full forward
   (which still returns final hidden + logits), removed in a `finally`. Add a parity
   test: single-pass vs the current two-pass path is **not expected to be bit-exact** —
   assert `np.allclose` with an acceptable, **recorded tolerance** (deterministic per
   excerpt and pooling; record what the tolerance has to be). No cache change yet.
2. **`maest_layer7` cache source, written by the same MAEST job.** Registry entry +
   migration + its own version column; write it alongside the embedding/logits/excerpts
   in `make_maest_handler`; surface in coverage/status reads.
3. **MERT-L19 in a new `backend/embeddings/mert.py`.** Independent encoder — **L19
   only** (1024-dim, `hidden_states[18]`); do not retain the 24 hidden layers or an L23
   that nothing consumes, when a safe hook can derive L19. Pinned revision, CC-BY-NC
   gate, `inference_lock`-guarded, never auto-downloaded, `--trust-remote-code` gate.
4. **`mert_l19` cache source (own version column) + opt-in `genre_full` recipe.**
   Registry/migration/version; the `genre_full` recipe id in `projection_config` (not
   the default); composite gather + clustering optional-block handling; supervised input
   decision. Wire the separate `mert` kind + worker + enqueue filter.
5. **Rollout.** Opt-in recipe id (no replacing old data), feature-gated, version +
   checkpoint-hash purge semantics, durable jobs returning `JobResult.failure(...)`,
   old artifacts preserved and reported.

## Acceptance checks
- Parity test passes (single-pass layer-7 == two-pass layer-7).
- `maest_layer7` and `mert_l19` rows read/write; a version bump makes old rows **absent**
  and they recompute at the new version (tests assert old absent; other sources
  unaffected).
- `genre_full` projects and clusters as an opt-in recipe; the default recipe and
  app-wide weights are byte-identical to before.
- Coverage/status shows the new blocks; a `--no-mert`-style flag skips the `mert` kind's
  enqueue (its own kind, not a maest-stage flag).
- Old benchmark artifacts and gallery exports are untouched.

## Risks
- **Layer-index off-by-one** (block 6 == layer 7 confirmed, but a hook on the wrong
  submodule silently captures the wrong tensor — guard with a shape assertion).
- **Numeric parity** between single-pass and two-pass (determinism per excerpt/pooling;
  cuda vs MPS may differ numerically — pin the parity check to one device).
- **MERT licence/checkpoint** drift and the `trust_remote_code` surface (an independent
  encoder with its own pinned revision).
- **Supervised input drift**: adding blocks to `SUPERVISED_INPUT_WEIGHTS` changes the
  head's input space → requires a retrain and invalidates cached `supervised` rows;
  prefer keeping the head's input unchanged until the genre recipe is stable.

## Constraints
No app-wide weight change; one `inference_lock`; CC-BY-NC gate for MERT; opt-in recipe
id only; new blocks independent cache versions; MERT is its own kind/worker, never a
MAEST-stage flag.

## Measurements
- Layer-7 single-pass vs two-pass wall-clock and GPU time (the ~99 ms/3-excerpt
  removal).
- Cache footprint (2304-D layer-7 ~9 KB pooled + ~27 KB for 3 excerpts **total**; MERT-L19
  1024-D per vector), peak RSS/GPU memory.
- Coverage/failure counts for the new stages.

## Rollback / feature gates
- Opt-in recipe id means old cached data and defaults are untouched; flip an env flag
  to disable `genre_full` / MERT.
- A version bump blanks only the new sources' old rows (other sources unaffected); MERT
  download requires explicit flags.

**Estimate: ~2–3 days.** Implementation is deferred until explicitly started — this
plan is the hand-off, not the wiring.

---

# WS 2 — Optimize analysis / model-centric scheduler + progressive draft map

## Goal
Cut end-to-end time-to-first-useful-analysis and keep the map usable **progressively**
while analysis drains — without new encoders and without risky concurrency. Includes
the user's batching/scheduling ideas (fixed microbatch, bounded decode prefetch,
model-window priority scheduling) as **measured** candidates, never assertions.

## Current verified state
- One worker thread per job kind, **8 kinds** (`scan`, `features`, `structure`,
  `maest`, `clap`, `supervise`, `cluster`, `layout`). Batch sizes: scan 8; features =
  `max(2, features_workers)` (default 4-process pool); structure = `max(2,
  structure_workers)` (default 4-thread pool); **MAEST batch 4** (inference
  `max_batch=12` excerpts); **CLAP batch 8** (`max_batch=24` segments); supervise /
  cluster / layout are singletons (batch 1).
- Models are loaded once at startup and warmed; no per-track reload.
- MAEST is the per-track bottleneck (~0.4 s/track, ≈4× CLAP); `torch.compile` was a
  measured regression on MPS; `inference_mode()` is kept but neutral.
- A partial map **already renders**: the MAEST pair is required and optional sources
  contribute zero rows, the frontend gates the embedding view on MAEST only, and the
  queue streams SSE per completed job — so tracks appear progressively as they land.
  A drained per-track stage triggers a re-cluster then a layout prefetch, which can delay
  the early first draft; measure that first-draft latency and the drain-prefetch before
  changing anything.
- Existing records: layer-7/MERT cached-MPS timings and per-extract peak RSS are recorded
  in the benchmark tooling; the **coordinated scheduler's p50/p95 latency and cross-stage
  fairness are not** — that is what the baseline harness must establish. Decode cost is
  whatever the baseline measures, not an assumed constant.

## Concrete files / interfaces
- `backend/jobs/queue.py`, `backend/jobs/worker.py` — priority/model-window
  scheduling, fixed microbatch, backpressure.
- `backend/app.py` — worker wiring, batch sizes, `priority_paths`, enqueue filters.
- `backend/services/analysis_pipeline.py` — handlers, decode/infer split, per-stage
  metrics.
- `backend/embeddings/maest.py`, `backend/embeddings/clap.py` — fixed microbatch,
  decode-ahead.
- `backend/decode_stderr.py` — bounded decode prefetch (CPU, safe-open serialisation).
- `backend/services/projection_priority.py` — extend user-vs-prefetch projection
  priority to the model-window scheduler so interactive queries are never starved.
- `frontend/src/coordinators/embedding-coordinator.ts`, the SSE handler, and
  `frontend/src/lib/pipeline.ts` — progressive draft refinement, explicit per-track
  readiness, throttled first-N animation on the existing `EmbeddingCoordinator` frame.

## Dependencies
- WS2 is gated on a **baseline harness** first (slice 1). No scheduler/batching change
  is made without numbers.

## Sequence (slices)
1. **Baseline harness.** Full-throughput + per-stage breakdown; capture
   first-10/100 time to MAEST-ready / draft / final; per-stage queue wait, GPU-lock
   wait, decode/infer/cache-write, model warm-up cost, peak RSS/GPU, fairness and
   latency; **freeze the cache** (full cold/warm) and artifact versions so every repeat
   is comparable. This is the gate: scheduler/batching work is only justified if the
   harness shows a real gap.
2. **Model-centric scheduler.** A homogeneous batch scheduler reserves **one shared
   accelerator slot across all model kinds** (MAEST, CLAP, later MERT) and runs them as
   sequential time windows with priority + fairness — selection order **phase 1:
   MAEST**, then CLAP/MERT — so it does **not** drain an entire model and leave the
   library unresponsive. Interactive queries win via the existing priority path.
   **Same-GPU parallel inference across models is NOT a default** — it is only a future
   experiment once proven safe; the one shared slot stays sequential. **Encoder batches
   are one-forward over the whole excerpt** — LLM-style token-level continuous batching
   does not apply (vary-length windows can be length-bucketed into bounded batches, not
   interleaved steps); that is a research direction, not a blanket prohibition. A
   separate future experiment may investigate **CUDA streams / multiple workers on the
   same model**, but only to prove parity, thread-safety, memory and fairness **with a
   single owner** before any guard change. **No promise of automatic improvement** — the
   harness decides. User ordering is a **scheduling benefit** (GPU locality/priority),
   never asserted as a swap-cost win.
3. **Fixed microbatch + bounded decode prefetch.** Keep the current microbatch
   (MAEST batch 4 → ≤12 excerpts; CLAP batch 8 → ≤24 segments) and decode ahead on CPU,
   bounded by a **byte semaphore** (backpressure, not unlimited decode). Do **not**
   add concurrent GPU threads; the single `inference_lock` stays.
4. **Progressive draft refinement (on the existing SSE readiness).** The partial map
   already renders; add explicit per-track source-level partial/final coverage, throttle
   active-vector/weight/label re-projection to a first-N milestone (not every track),
   animate through the existing `EmbeddingCoordinator` frame, select a usable provisional
   MAEST recipe + version, and let incremental MERT results upgrade **without forcing
   missing zeros**; normalize weight availability / mark coverage and use the existing
   policy when deciding production (high stakes). Measure first-draft latency and the
   drain-triggered cluster/layout delay first — do not build on an unverified assumption.
5. **Rollout + measurement.** Re-run the harness, compare, gate by env flag, document.
   Inspect the drain-triggered cluster/layout delay before claiming the first draft is
   already fast.

## Acceptance checks
- Measured throughput / time-to-first-draft improvement over baseline.
- No SIGBUS/regression; interactive queries never starved; memory bounded by the
  semaphore.
- Progressive draft renders and stays responsive while MAEST/CLAP drain.
- No concurrent GPU dispatch; single `inference_lock` unchanged.

## Risks
- Decode-prefetch buffer bloat (mitigate with byte semaphore).
- Scheduler starvation of interactive work (mitigate with the priority path).
- One shared accelerator slot serialises all model stages — throughput gain depends on
  the harness showing a real gap; cuda-vs-MPS timing differences.
- Assuming the first draft is already fast — it is gated by the drain-triggered
  cluster/layout; verify, don't assert.

## Constraints
Single `inference_lock`; **one shared accelerator slot (sequential across model kinds)**;
no concurrent GPU threads / same-model multi-worker until proven safe with one owner; no
per-track model reload; user ordering = scheduling only; byte-bounded backpressure.

## Measurements
Baseline harness: throughput, first-10/100, per-stage queue/GPU-lock wait,
decode/infer/cache, warm-up, RSS/GPU peak, p50/p95 latency, fairness.

## Rollback / feature gates
Env-gated scheduler and batching; frozen-cache harness; single-lock invariant is the
backstop.

**Estimate: ~3 days.**

---

# WS 3 — MuQ-MuLan audio+text evaluation vs CLAP (needs semantic UI)

## Goal
Evaluate **MuQ-MuLan** (an audio+text model) as a possible replacement for CLAP for
semantic search / describe / weighting — **offline and read-only**, in parallel with
WS2. It needs a small semantic-comparison UI (a 3-model selector reusing the Search
panel) but **does not rewire production CLAP**.

## Current verified state
- CLAP (`laion/larger_clap_music_and_speech`, `EMBEDDING_VERSION=3`, sr 48 kHz,
  3×15 s segments) is the only text encoder and drives `/api/search` text search,
  `clap_describe` (FACET_BANK/AXES), and `semantic_weighting` (folder/tag text
  directions projected into the CLAP column span). All text paths route through
  `clap.generate_text_embeddings`.
- Text component files to discover references (do not rewire): `backend/routes/api_search.py`,
  `backend/embeddings/semantic_weighting.py`, `backend/embeddings/clap_describe.py`.
- Genre-space benchmark tooling (`backend/experiments/genre_space/`, `scoring.py`)
  is a ready template for a read-only model comparison.

## Stage 0 (mandatory, before any code)
- **Licence: CC BY-NC 4.0.** Record it and gate any use on the licence.
- **Checkpoint pin** from the **official model card revision** (first source of truth);
  do **not** assume every checkpoint in the family is interchangeable.
- **Public-training difference** (verify at implementation): the audio encoder is the
  **open MuQ public checkpoint**, trained on **MSD**, distinct from the **larger model in
  the paper** (proprietary-scale training). MuQ-MuLan (the ~700M audio+text pair) and a
  **separate SSL ~310M audio-only** estimate are **different variants** — do not equate
  them; decide which is being evaluated and pin it.
- **sr 24 kHz official** (verify at implementation). **Verify current docs at
  implementation** — do **not** assume the model exposes a standard/CLAP-like API;
  confirm loading, the audio/text entry points, and the checkpoint revision.
- Download-to-cache only; never auto-download; never auto-run remote code.

## Concrete files / interfaces (proposed — verify before coding)
- `backend/experiments/muq/` — offline eval harness (cohort/extract/score mirroring
  `genre_space`).
- `backend/embeddings/muq.py` — **new, independently named** module; `clap.py` source
  and version stay untouched.
- A per-model audio+text **segment pool** with the same windows (fair mode) plus a
  native MuQ mode. Query-audio embeddings use the **same encoder** checks.
- **Do not align CLAP text with MuQ audio** (different spaces; no cross-model text/audio
  matching).
- Semantic-comparison UI: a **small 3-model selector** (CLAP / MuQ / hybrid) reusing
  the Search panel, reading **offline precomputed retrieval results** — read-only, a
  dedicated manifest, **not** a 2-D contour/map.

## Dependencies
None on production. Model download cached locally; a separate segment pool per model.

## Sequence (slices)
0. **Licence + checkpoint pin** + download-to-cache + verify sr/docs/API at
   implementation. No production change without user approval.
1. **Offline extraction**: MuQ audio+text embeddings over the same frozen query-manifest
   windows as CLAP (fair + native modes), same encoder checks for audio queries; CLAP
   source/version preserved as-is.
2. **Query manifest**: 30–50 real intents across genre, instrumentation, mood, texture,
   including hard negatives / neighbour microgenres; **graded multi-relevance labels**
   with **segment audition** for judging.
3. **Retrieval eval**: NDCG@10, P@5; **recall only when judged positives are known**;
   annotate the 3 alternatives (CLAP, MuQ, hybrid) with the **genre space fixed**; a
   dedicated manifest, not a 2-D contour.
4. **Compare query-result UI + latency**: reuse the Search panel with the 3-model
   selector; measure query latency cold/warm (p50/p95), memory, coverage, failures;
   prototype a rerank **only if** it measures as complementary.
5. **Decision**: replace / search-only / keep baseline (CLAP). Report.

## Acceptance checks
- CLAP source/version byte-unchanged; eval is read-only and offline.
- NDCG@10 / P@5 / recall (where judged positives are known) produced for the 3
  alternatives on the same frozen manifest.
- Latency/memory/coverage/failure measured, not guessed.
- No production path changed; no cloud upload.

## Risks
Licence; model-card/API drift (do not assume a standard API or that all checkpoints are
interchangeable); sr 24k vs CLAP 48k; text/audio space mismatch; MuQ-MuLan 700M-class
hardware cost; data privacy (real data stays local).

## Constraints
Read-only, offline, preserves CLAP, no production rewire, verify docs at
implementation, real data local.

## Measurements
NDCG@10 / P@5 / recall; latency cold/warm (p50/p95); memory; coverage; failures.

## Rollback / feature gates
Eval harness flags; nothing in production changes, so rollback is a no-op.

**Estimate: 2–3 days, parallel with WS2.**

---

# WS 4 — Validation and reproducibility (quality/safety/performance)

## Goal
Firm up reproducibility, safety, and quality validation of the **existing** pipeline and
offline benchmark tooling. **Only viable where the harness already exists — this is not
a new model grid.**

## Current verified state
- Genre-space tooling exists with CI bootstrap for the L7/MERT **pair** only; there is
  no generic CI scoring.
- Small real microgenre cohort (33 depth-2 parents, often nq=1) → low confidence.
- Cluster/projection timing at scale and a cold/warm end-to-end run are unmeasured.
- **Playlists do not exist in the product yet**, so playlist-label work is contingent and
  deferred (see below), not near-term.

## Concrete files / interfaces
- `backend/experiments/genre_space/scoring.py` — generalise to report CIs for **any**
  recipe comparison.
- `backend/embeddings/layout_evaluation.py` / `tests/profile_pipeline.py` — scale and
  cold/warm baselines.
- `backend/services/analysis_pipeline.py` / `supervised.py` — external-root validation.

## Sequence (slices)
1. **Generic confidence-interval scoring.** Generalise `scoring.py` so any recipe
   comparison reports CIs without a bespoke analysis pass.
2. **Reproducibility + safety.** Freeze the full cache cold/warm artifact versions;
   verify no cloud uploads; surface model-download cache + licence; **report** (not
   delete) preserved old gallery exports/data (~253 MB local, ~2.4 GB HF cache).
3. **Quality validation.** Larger real microgenre validation **if viable**; external-root
   validation; supervised retrain after a CLAP backfill.
4. **Targeted performance.** Cluster/projection timing at scale; cold/warm end-to-end run.
   (Peak RSS/GPU is partly covered by WS2's baseline harness; keep this to scale and
   cold/warm.)

## Deferred (contingent — do NOT schedule near-term)
- **Playlist labels.** Playlists do not exist in the product yet; when they do, they are
  another noisy label source for the same supervised head (data-plumbing only, no new
  model). Revisit only once playlist data exists.

## Acceptance checks
- Generic CI scoring; reproducible frozen-cache runs; no cloud upload; artifacts
  reported not deleted; measured (not assumed) scale numbers.

## Risks
Low-confidence labels; scale beyond calibrated ranges.

## Constraints
No new model grid; real data local; frozen-cache reproducibility.

## Measurements
CI coverage of comparisons; scale timings; cold/warm end-to-end.

## Rollback / feature gates
All additive/tooling; no production weight change.

**Estimate: ~1–2 days (near-term slices only; playlists deferred until the data exists).**
