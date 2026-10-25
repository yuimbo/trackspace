"""MAEST analysis and rhythm descriptors against the real fixture track.

MAEST tests are skipped when the checkpoint is unavailable (offline CI) rather
than failing, but when the model *is* present the assertions are real: shapes,
finiteness, determinism, and that decode failures degrade to ``None``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from backend.embeddings import maest as maest_mod
from backend.embeddings import rhythm_features as rf

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "test_track.mp3")

def _maest_ready() -> bool:
    """Load once at collection time; skip the suite if the checkpoint is absent."""
    if not maest_mod.is_model_ready():
        maest_mod.load_model()
    return maest_mod.is_model_ready()


maest_available = pytest.mark.skipif(
    not _maest_ready(),
    reason=f"MAEST checkpoint unavailable ({maest_mod.load_error()})",
)


# ── rhythm features ───────────────────────────────────────────


def test_rhythm_vector_shape_and_range() -> None:
    vec = rf.extract_rhythm_features(FIXTURE)
    assert vec is not None
    assert vec.shape == (rf.RHYTHM_FEATURE_DIM,)
    assert vec.dtype == np.float32
    assert np.isfinite(vec).all()
    # Every dimension is normalised so no single feature can dominate the block.
    assert (vec >= 0.0).all() and (vec <= 1.0).all()


def test_rhythm_names_match_dimension() -> None:
    assert len(rf.RHYTHM_FEATURE_NAMES) == rf.RHYTHM_FEATURE_DIM


def test_rhythm_is_deterministic() -> None:
    a = rf.extract_rhythm_features(FIXTURE)
    b = rf.extract_rhythm_features(FIXTURE)
    assert a is not None and b is not None
    assert np.allclose(a, b)


def test_rhythm_missing_file_returns_none() -> None:
    assert rf.extract_rhythm_features("/definitely/not/here.mp3") is None


def test_rhythm_batch_reports_per_path() -> None:
    out = rf.generate_rhythm_features_batch([FIXTURE, "/nope/x.mp3"])
    assert out[FIXTURE] is not None
    assert out["/nope/x.mp3"] is None


def test_rhythm_summary_is_labelled() -> None:
    vec = rf.extract_rhythm_features(FIXTURE)
    summary = rf.rhythm_feature_summary(vec)
    assert set(summary) == set(rf.RHYTHM_FEATURE_NAMES)
    assert rf.rhythm_feature_summary(None) == {}


# ── MAEST ─────────────────────────────────────────────────────


@maest_available
def test_maest_analysis_shapes() -> None:
    result = maest_mod.analyze(FIXTURE)
    assert result is not None
    assert result.embedding.shape == (maest_mod.MAEST_EMBED_DIM,)
    assert result.logits.shape == (maest_mod.MAEST_LOGIT_DIM,)
    assert result.num_excerpts >= 1
    assert result.excerpt_embeddings.shape == (
        result.num_excerpts,
        maest_mod.MAEST_EMBED_DIM,
    )
    assert np.isfinite(result.embedding).all()
    assert np.isfinite(result.logits).all()


@maest_available
def test_maest_pooled_embedding_matches_excerpt_mean() -> None:
    result = maest_mod.analyze(FIXTURE)
    assert result is not None
    expected = result.excerpt_embeddings.mean(axis=0)
    assert np.allclose(result.embedding, expected, atol=1e-5)


@maest_available
def test_maest_is_deterministic() -> None:
    a = maest_mod.analyze(FIXTURE)
    b = maest_mod.analyze(FIXTURE)
    assert a is not None and b is not None
    assert np.allclose(a.embedding, b.embedding, atol=1e-4)


@maest_available
def test_maest_missing_file_returns_none() -> None:
    assert maest_mod.analyze("/definitely/not/here.mp3") is None


@maest_available
def test_maest_batch_keys_every_requested_path() -> None:
    out = maest_mod.analyze_batch([FIXTURE, "/nope/x.mp3"])
    assert set(out) == {FIXTURE, "/nope/x.mp3"}
    assert out[FIXTURE] is not None
    assert out["/nope/x.mp3"] is None


@maest_available
def test_maest_layer7_present_and_pooled_shape() -> None:
    result = maest_mod.analyze(FIXTURE)
    assert result is not None
    assert result.layer7 is not None
    assert result.layer7.shape == (maest_mod.MAEST_LAYER7_DIM,)
    assert result.layer7.dtype == np.float32
    assert np.isfinite(result.layer7).all()


@maest_available
def test_maest_layer7_is_deterministic() -> None:
    a = maest_mod.analyze(FIXTURE)
    b = maest_mod.analyze(FIXTURE)
    assert a is not None and b is not None
    assert a.layer7 is not None and b.layer7 is not None
    assert np.allclose(a.layer7, b.layer7, atol=1e-4, rtol=1e-4)


@maest_available
def test_maest_layer7_parity_singlepass_vs_twopass() -> None:
    """Single-pass block-6 capture == the benchmark's two-pass ``transformer_block=6``.

    Both runs consume the same deterministic mel input, so block 6 is the same
    computation; this pins the production single-pass capture against the reference
    two-pass path ``_model(..., transformer_block=6)`` that extraction uses today.
    """
    import torch

    mod = maest_mod
    assert mod._model is not None and mod._mel is not None
    target = mod.MAEST_SR * mod.MAEST_SEGMENT_SECONDS
    # Deterministic 2-excerpt batch, distinct rows so stacking is exercised.
    t = np.linspace(-1.0, 1.0, target, dtype=np.float32)
    batch = np.stack([t, np.roll(t, 3)]).astype(np.float32)
    wave = torch.from_numpy(batch)
    mel = mod._mel(wave)
    mel_dev = mel.to(mod._device)

    with mod.inference_lock:
        with torch.inference_mode():
            # Reference: the existing two-pass path (returns ``None, layer7``).
            _, ref = mod._model(
                mel_dev,
                melspectrogram_input=True,
                transformer_block=mod.MAEST_LAYER7_BLOCK,
            )
            ref = ref.detach().to("cpu").numpy().astype(np.float32)
            # Single-pass: the production scoped hook captures layer-7 during the full forward.
            with mod._capture_layer7(mod._model) as captured:
                mod._model(mel_dev, melspectrogram_input=True)
            single = captured["layer7"].detach().to("cpu").numpy().astype(np.float32)

    assert single.shape == ref.shape == (2, mod.MAEST_LAYER7_DIM)
    # Recorded tolerance: on MPS single-pass vs two-pass is bit-exact (measured max abs
    # diff 0.0). Kept loose enough to absorb a different accelerator's rounding (CUDA).
    assert np.allclose(single, ref, atol=1e-4, rtol=1e-4)


@maest_available
def test_top_styles_are_ranked_discogs_labels() -> None:
    result = maest_mod.analyze(FIXTURE)
    assert result is not None
    styles = maest_mod.top_styles(result.logits, k=5)
    assert len(styles) == 5
    scores = [s for _, s in styles]
    assert scores == sorted(scores, reverse=True)
    # Labels come from the Discogs taxonomy, formatted "Genre---Style".
    assert all("---" in name for name, _ in styles)


def test_discogs_label_table_size() -> None:
    labels = maest_mod.discogs_style_labels()
    if labels:
        assert len(labels) == maest_mod.MAEST_LOGIT_DIM


def test_empty_logits_yield_no_styles() -> None:
    assert maest_mod.top_styles(np.array([], dtype=np.float32)) == []


# ── layer-7 hook hygiene ──────────────────────────────────────
# The single-pass capture must never leak a forward hook onto a later call, and it
# must raise loudly (not silently pool garbage) if the block layout is unexpected.


class _L7Block:
    def __init__(self, out_shape):
        self._shape = out_shape
        self._hooks = {}

    def register_forward_hook(self, fn):
        handle = _L7Handle(self, fn)
        self._hooks[handle] = fn
        return handle

    def __call__(self, *args):
        import torch

        output = torch.zeros(*self._shape, dtype=torch.float32)
        for fn in list(self._hooks.values()):
            out = fn(self, args, output)
            if out is not None:
                output = out
        return output


class _L7Handle:
    def __init__(self, block, fn):
        self._block = block
        self._fn = fn
        self.removed = False

    def remove(self):
        self.removed = True
        self._block._hooks.pop(self, None)


class _FakeBlocks:
    def __init__(self, n, out_shape=(2, 3, 768)):
        self.blocks = [_L7Block(out_shape) for _ in range(n)]


def test_capture_layer7_removes_hook_after_forward() -> None:
    model = _FakeBlocks(7)
    handle = model.blocks[maest_mod.MAEST_LAYER7_BLOCK]._hooks
    captured = {}
    with maest_mod._capture_layer7(model) as cap:
        model.blocks[maest_mod.MAEST_LAYER7_BLOCK](None)
        captured["layer7"] = cap["layer7"]
    assert not handle  # hook removed in the context's finally
    assert captured["layer7"].shape == (2, maest_mod.MAEST_LAYER7_DIM)


def test_capture_layer7_removes_hook_on_forward_error() -> None:
    model = _FakeBlocks(7)
    with pytest.raises(RuntimeError):
        with maest_mod._capture_layer7(model):
            raise RuntimeError("forward failed")
    assert not model.blocks[maest_mod.MAEST_LAYER7_BLOCK]._hooks


def test_capture_layer7_raises_loudly_on_wrong_shape() -> None:
    # A wrong submodule/layout emits a different shape; the hook must raise rather
    # than silently capture garbage, and still clean up.
    model = _FakeBlocks(7, out_shape=(2, 2, 768))  # needs cls+dist+feats (>=3)
    with pytest.raises(RuntimeError):
        with maest_mod._capture_layer7(model):
            model.blocks[maest_mod.MAEST_LAYER7_BLOCK](None)
    assert not model.blocks[maest_mod.MAEST_LAYER7_BLOCK]._hooks


def test_reconcile_checkpoint_couples_maest_family(monkeypatch, tmp_path) -> None:
    """A changed checkpoint purges the MAEST family together, unrelated blocks kept."""
    from backend.embeddings.feature_cache import FeatureCache

    fc = FeatureCache(str(tmp_path / "features.db"))
    fc.put_maest(
        "a", np.ones(8, np.float32), np.ones(519, np.float32),
        version=1, layer7=np.ones(2304, np.float32),
    )
    fc.put_maest("b", np.ones(8, np.float32), np.ones(519, np.float32), version=1)
    fc.put("clap", "a", np.ones(4, np.float32), version=1)
    fc.put("rhythm", "b", np.ones(3, np.float32), version=1)

    monkeypatch.setattr(maest_mod, "checkpoint_fingerprint", lambda: "sha-AAA")
    assert maest_mod.reconcile_checkpoint(fc) is True
    assert fc.has("maest", "a", version=1) is False
    assert fc.has("maest_layer7", "a", version=1) is False
    assert fc.has("maest_logits", "b", version=1) is False
    assert fc.has("maest_excerpts", "b", version=1) is False
    # Unrelated blocks are preserved.
    assert fc.has("clap", "a", version=1)
    assert fc.has("rhythm", "b", version=1)
    assert fc.get_meta("maest_checkpoint_sha256") == "sha-AAA"

    # No change -> no purge.
    assert maest_mod.reconcile_checkpoint(fc) is False
    # A changed checkpoint purges again.
    monkeypatch.setattr(maest_mod, "checkpoint_fingerprint", lambda: "sha-BBB")
    assert maest_mod.reconcile_checkpoint(fc) is True


def test_reconcile_checkpoint_noop_when_fingerprint_unknown(monkeypatch, tmp_path) -> None:
    from backend.embeddings.feature_cache import FeatureCache

    fc = FeatureCache(str(tmp_path / "features.db"))
    fc.put_maest("a", np.ones(8, np.float32), np.ones(519, np.float32), version=1)
    monkeypatch.setattr(maest_mod, "checkpoint_fingerprint", lambda: None)
    assert maest_mod.reconcile_checkpoint(fc) is False
    assert fc.has("maest", "a", version=1)
