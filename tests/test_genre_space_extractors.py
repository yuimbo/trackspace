"""Genre-space extractor tests — mocks only: no real models, no downloads, no inference.

Exercises validation, decode failures, pooling/masking math, layer arguments,
native sample rates, atomic output writing, and metadata against tiny fakes.
"""

from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from backend.experiments.genre_space import dataset as ds
from backend.experiments.genre_space import extractors as ex

REV = "0c0c8c6f6e04dcc39d96f94e0be9978a74848906"
CKPT_URL = "https://example.invalid/discogs-maest-30s-pw-129e-519l-swa.ckpt"
CKPT_SHA = "d" * 64


# ── fakes ────────────────────────────────────────────────────────────────────


def make_track(tmp_path, name, windows=None, fp=None):
    p = tmp_path / name
    if not p.exists():
        p.write_bytes(b"\x00" * 1024)
    st = os.stat(p)
    return {
        "fp": fp or (f"fp-{name}" + "x" * 64)[:64],
        "path": str(p),
        "mtime_ns": st.st_mtime_ns,
        "size": st.st_size,
        "label": "test/rock",
        "split": "dev",
        "group": "G0000",
        "windows": windows if windows is not None else [{"start": 5.0, "duration": 30.0}],
    }


def make_manifest(tracks):
    """A *valid* manifest: proper schema/root/label/split/group/mtime/size/fp + computed id."""
    m = {"schema_version": ds.SCHEMA_VERSION, "root": "/music", "dataset_id": "", "tracks": tracks}
    m["dataset_id"] = ds.dataset_id(m)
    return m


def fake_librosa_load(path, sr=None, mono=None, offset=None, duration=None):
    """Full-length decode except for tracks whose filename says empty/trunc."""
    if "empty" in os.path.basename(path):
        return np.array([], dtype=np.float32), sr
    if "trunc" in os.path.basename(path):
        return np.zeros(int(duration * sr * 0.5), dtype=np.float32), sr
    return np.zeros(int(duration * sr), dtype=np.float32), sr


class FakeMel:
    def __init__(self):
        self.shapes = []

    def __call__(self, x):
        self.shapes.append(tuple(x.shape))
        return torch.zeros(x.shape[0], 128, 1875)

    def to(self, device):
        return self


class FakeMaest:
    def __init__(self, zero=False):
        self.melspectrogram = FakeMel()
        self.mel = self.melspectrogram  # stable handle: extractor clears model.melspectrogram
        self.calls = []
        self.zero = zero

    def eval(self):
        return self

    def init_melspectrogram(self):
        self.melspectrogram = self.mel  # restore handle; extractor sets model.melspectrogram = None after load
        return None

    def to(self, device):
        return self

    def __call__(self, mel, melspectrogram_input=False, transformer_block=-1):
        b = mel.shape[0]
        self.calls.append({"melspectrogram_input": melspectrogram_input, "transformer_block": transformer_block})
        fill = torch.zeros if self.zero else torch.ones
        if transformer_block == -1:
            return fill(b, ex.MAEST_LOGIT_DIM), fill(b, ex.MAEST_EMBED_DIM)
        return None, fill(b, ex.MAEST_LAYER7_DIM)


class FakeMertExtractor:
    def __init__(self):
        self.calls = []

    def __call__(self, waveforms, sampling_rate=None, return_tensors=None, **kwargs):
        self.calls.append({
            "sampling_rate": sampling_rate,
            "return_tensors": return_tensors,
            "padding": kwargs.get("padding"),
            "return_attention_mask": kwargs.get("return_attention_mask"),
            "do_normalize": kwargs.get("do_normalize"),
        })
        lengths = [len(w) for w in waveforms]
        n, t_max = len(waveforms), max(lengths)
        mask = torch.ones(n, t_max)
        for i, ln in enumerate(lengths):
            mask[i, ln:] = 0.0
        return {"input_values": torch.zeros(n, t_max), "attention_mask": mask}


class FakeMertModel:
    def __init__(self, n_hidden=25):
        self.calls = []
        self.n_hidden = n_hidden

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, input_values=None, attention_mask=None, output_hidden_states=False):
        n, t = input_values.shape
        self.calls.append({"attention_mask_given": attention_mask is not None, "output_hidden_states": output_hidden_states})
        mask = attention_mask if attention_mask is not None else torch.ones(n, t)
        hidden = tuple((mask[..., None]).repeat(1, 1, ex.MERT_EMBED_DIM) for _ in range(self.n_hidden))
        return SimpleNamespace(hidden_states=hidden, feature_attention_mask=mask)


def patch_maest(monkeypatch, tmp_path, model=None):
    ckpt = tmp_path / "ckpt.ckpt"
    ckpt.write_bytes(b"x")
    monkeypatch.setattr(ex, "_maest_checkpoint", lambda: (CKPT_URL, CKPT_SHA, ckpt))
    model = model or FakeMaest()
    monkeypatch.setattr("maest_infer.get_maest", lambda arch=None: model)
    return model


def patch_mert(monkeypatch, model=None, extractor=None):
    """Patch transformers; returns (model, extractor, load_kwargs) where load_kwargs
    collects every ``from_pretrained`` call's keyword arguments."""
    import transformers

    model = model or FakeMertModel()
    extractor = extractor or FakeMertExtractor()
    load_kwargs = []

    def _fe(*a, **k):
        load_kwargs.append(k)
        return extractor

    def _m(*a, **k):
        load_kwargs.append(k)
        return model

    monkeypatch.setattr(transformers, "AutoFeatureExtractor", type("_FE", (), {"from_pretrained": staticmethod(_fe)}))
    monkeypatch.setattr(transformers, "AutoModel", type("_M", (), {"from_pretrained": staticmethod(_m)}))
    return model, extractor, load_kwargs


def patch_decode(monkeypatch, fn=fake_librosa_load):
    monkeypatch.setattr("backend.decode_stderr.librosa_load", fn)
    return fn


# ── validation ───────────────────────────────────────────────────────────────


def test_manifest_validation(tmp_path):
    good = make_track(tmp_path, "a.mp3")
    assert ds.validate_manifest(make_manifest([good])) == []
    bad_cases = [
        (dict(make_manifest([good]), root=""), "root"),
        (dict(make_manifest([good]), schema_version=2), "schema_version"),
        (dict(make_manifest([good]), dataset_id="ds-x"), "dataset_id"),
        (make_manifest([dict(good, fp="")]), "fp"),
        (make_manifest([dict(good, path="relative.mp3")]), "path"),
        (make_manifest([dict(good, label="")]), "label"),
        (make_manifest([dict(good, split="train")]), "split"),
        (make_manifest([dict(good, group=7)]), "group"),
        (make_manifest([dict(good, mtime_ns="x")]), "mtime_ns"),
        (make_manifest([dict(good, size=-1)]), "size"),
        (make_manifest([dict(good, windows=[])]), "windows"),
        (make_manifest([dict(good, windows=[{"start": 0.0, "duration": 30.0}] * 4)]), "windows"),
        (make_manifest([dict(good, windows=[{"start": -1.0, "duration": 10.0}])]), "start"),
        (make_manifest([dict(good, windows=[{"start": 0.0, "duration": 0.0}])]), "duration"),
    ]
    for manifest, needle in bad_cases:
        problems = ds.validate_manifest(manifest)
        assert any(needle in p for p in problems), (manifest, problems)


def test_extract_rejects_invalid_manifest_before_any_output(tmp_path):
    bad = make_manifest([make_track(tmp_path, "a.mp3", windows=[{"start": 0.0, "duration": 30.0}] * 4)])
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="windows must be a list of 1..3"):
        ex.extract(bad, out, device="cpu")
    assert not out.exists()  # validation precedes output creation and model load


def test_extract_rejects_stale_dataset_id_before_any_output(tmp_path, monkeypatch):
    # The public extract() must enforce the same dataset-id invariant the CLI
    # checks: the supplied manifest's dataset_id must equal the canonical hash.
    patch_decode(monkeypatch)
    patch_maest(monkeypatch, tmp_path)
    manifest = make_manifest([make_track(tmp_path, "a.mp3")])
    manifest["dataset_id"] = "0" * 64  # valid hex, but not the canonical id -> stale/edited
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="edited or is stale"):
        ex.extract(manifest, out, device="cpu")
    assert not out.exists()  # fails before output creation or model load


def test_model_and_device_validation(tmp_path):
    with pytest.raises(ValueError, match="model must be"):
        ex.extract(make_manifest([make_track(tmp_path, "a.mp3")]), tmp_path / "out", model="clap")
    with pytest.raises(ValueError, match="device"):
        ex._pick_device("gpu")
    assert ex._pick_device("auto") in ("cpu", "cuda", "mps")
    assert ex._pick_device("cpu") == "cpu"


def test_mert_revision_validation(tmp_path):
    out = tmp_path / "out"
    for revision in (None, "main", REV[:-1], "Z" + REV[1:]):
        with pytest.raises(ValueError, match="40 hex"):
            ex.extract(make_manifest([make_track(tmp_path, "a.mp3")]), out, model="mert",
                       revision=revision, allow_download=True, trust_remote_code=True)


def test_mert_trust_remote_code_and_download_flags(tmp_path, monkeypatch):
    track = make_track(tmp_path, "a.mp3")
    # trust_remote_code is mandatory even when downloads are allowed
    with pytest.raises(RuntimeError, match="trust_remote_code=True"):
        ex.extract(make_manifest([track]), tmp_path / "out", model="mert",
                   revision=REV, allow_download=True)
    # trust without allow_download refuses when the pinned revision is not cached
    # (the real revision is cached locally, so force the "not cached" branch)
    monkeypatch.setattr(ex, "_mert_revision_cached", lambda rev: False)
    with pytest.raises(RuntimeError, match="allow_download=True"):
        ex.extract(make_manifest([make_track(tmp_path, "b.mp3")]), tmp_path / "out2", model="mert",
                   revision=REV, allow_download=False, trust_remote_code=True)
    # a cached revision loads with local_files_only=True (allow_download only gates fetching)
    patch_decode(monkeypatch)
    _, _, load_kwargs = patch_mert(monkeypatch)
    monkeypatch.setattr(ex, "_mert_revision_cached", lambda rev: True)
    ex.extract(make_manifest([make_track(tmp_path, "c.mp3")]), tmp_path / "out3", model="mert",
               revision=REV, trust_remote_code=True, device="cpu")
    assert load_kwargs and all(
        k["trust_remote_code"] is True and k["local_files_only"] is True and k["revision"] == REV
        for k in load_kwargs
    )


def test_maest_missing_checkpoint_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "_maest_checkpoint", lambda: (CKPT_URL, CKPT_SHA, tmp_path / "missing.ckpt"))
    with pytest.raises(RuntimeError, match="torch.hub cache"):
        ex.extract(make_manifest([make_track(tmp_path, "a.mp3")]), tmp_path / "out", model="maest")


def test_refuse_overwrite(tmp_path, monkeypatch):
    patch_decode(monkeypatch)
    patch_maest(monkeypatch, tmp_path)
    out = tmp_path / "out"
    ex.extract(make_manifest([make_track(tmp_path, "a.mp3")]), out, device="cpu")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ex.extract(make_manifest([make_track(tmp_path, "b.mp3")]), out, device="cpu")


# ── MAEST happy path, layer args, native SR, excerpts ────────────────────────


def test_maest_extract_happy_path(tmp_path, monkeypatch):
    patch_decode(monkeypatch)
    model = patch_maest(monkeypatch, tmp_path)
    tracks = [
        make_track(tmp_path, "a1.mp3", windows=[{"start": 0.0, "duration": 30.0}], fp="f" * 64),
        make_track(tmp_path, "a2.mp3", windows=[{"start": 0.0, "duration": 20.0}, {"start": 40.0, "duration": 10.0}], fp="g" * 64),
    ]
    meta = ex.extract(make_manifest(tracks), tmp_path / "out", device="cpu")
    npz = np.load(tmp_path / "out" / "maest.npz")
    assert list(npz["fingerprints"]) == ["f" * 64, "g" * 64]
    assert npz["maest"].shape == (2, ex.MAEST_EMBED_DIM)
    assert npz["maest_logits"].shape == (2, ex.MAEST_LOGIT_DIM)
    assert npz["maest_layer7"].shape == (2, ex.MAEST_LAYER7_DIM)
    assert npz["maest_excerpts"].shape == (2, 3, ex.MAEST_EMBED_DIM)
    assert npz["excerpt_counts"].tolist() == [1, 2]
    # second track: slot 2 zero-padded, real slots carry the pooled values
    assert np.allclose(npz["maest_excerpts"][1, 2], 0.0)
    assert np.allclose(npz["maest_excerpts"][1, 0], 1.0)
    assert np.allclose(npz["maest"][1], 1.0)
    # final call + intermediate block-6 call, both melspectrogram_input
    assert [c["melspectrogram_input"] for c in model.calls] == [True, True, True, True]
    assert [c["transformer_block"] for c in model.calls] == [-1, 6, -1, 6]
    # 30 s pad: both windows of track 2 land in one (2, 480000) batch
    assert model.mel.shapes == [(1, 480000), (2, 480000)]
    assert meta["coverage"] == {"total": 2, "ok": 2, "failed": 0}


def test_native_sample_rates_and_frozen_windows(tmp_path, monkeypatch):
    calls = []

    def recorder(path, sr=None, mono=None, offset=None, duration=None):
        calls.append((sr, mono, offset, duration))
        return np.zeros(int(duration * sr), dtype=np.float32), sr

    patch_decode(monkeypatch, recorder)
    patch_maest(monkeypatch, tmp_path)
    ex.extract(
        make_manifest([make_track(tmp_path, "m.mp3", windows=[{"start": 10.0, "duration": 30.0}])]),
        tmp_path / "out_m", device="cpu",
    )
    assert calls == [(16000, True, 10.0, 30.0)]
    calls.clear()
    patch_mert(monkeypatch)
    ex.extract(
        make_manifest([make_track(tmp_path, "r.mp3", windows=[{"start": 2.0, "duration": 10.0}, {"start": 20.0, "duration": 5.0}])]),
        tmp_path / "out_r", model="mert", revision=REV, allow_download=True, trust_remote_code=True, device="cpu",
    )
    assert calls == [(24000, True, 2.0, 10.0), (24000, True, 20.0, 5.0)]


def test_per_track_failures(tmp_path, monkeypatch):
    patch_decode(monkeypatch)
    patch_maest(monkeypatch, tmp_path)
    tracks = [
        make_track(tmp_path, "ok.mp3", fp="a" * 64),
        make_track(tmp_path, "b_empty.mp3", fp="b" * 64),
        make_track(tmp_path, "c_trunc.mp3", fp="c" * 64),
        dict(make_track(tmp_path, "d.mp3", fp="d" * 64), mtime_ns=1),  # changed since manifest
    ]
    meta = ex.extract(make_manifest(tracks), tmp_path / "out", device="cpu")
    assert meta["coverage"] == {"total": 4, "ok": 1, "failed": 3}
    by_fp = {f["fp"]: f for f in meta["failures"]}
    assert "empty decode" in by_fp["b" * 64]["error"]
    assert by_fp["b" * 64]["stage"] == "decode"
    assert "truncated decode" in by_fp["c" * 64]["error"]
    assert "changed since manifest" in by_fp["d" * 64]["error"]
    assert by_fp["d" * 64]["stage"] == "stat"
    npz = np.load(tmp_path / "out" / "maest.npz")
    assert list(npz["fingerprints"]) == ["a" * 64]  # successful tracks only, manifest order


def test_zero_pooled_vector_fails(tmp_path, monkeypatch):
    patch_decode(monkeypatch)
    patch_maest(monkeypatch, tmp_path, model=FakeMaest(zero=True))
    out = tmp_path / "out"
    with pytest.raises(RuntimeError, match="no tracks extracted"):
        ex.extract(make_manifest([make_track(tmp_path, "a.mp3")]), out, device="cpu")
    assert not (out / "maest.npz").exists()
    assert not list(out.glob("*.tmp"))


# ── MERT ─────────────────────────────────────────────────────────────────────


def test_mert_extract_masked_mean(tmp_path, monkeypatch):
    patch_decode(monkeypatch)
    model, extractor, _ = patch_mert(monkeypatch)
    tracks = [make_track(tmp_path, "a.mp3", windows=[{"start": 0.0, "duration": 5.0}, {"start": 6.0, "duration": 2.0}], fp="f" * 64)]
    meta = ex.extract(make_manifest(tracks), tmp_path / "out", model="mert",
                      revision=REV, allow_download=True, trust_remote_code=True, device="cpu")
    npz = np.load(tmp_path / "out" / "mert.npz")
    assert npz["mert_l19"].shape == (1, ex.MERT_EMBED_DIM)
    assert npz["mert_l23"].shape == (1, ex.MERT_EMBED_DIM)
    assert npz["mert_l19_excerpts"].shape == (1, 3, ex.MERT_EMBED_DIM)
    assert npz["excerpt_counts"].tolist() == [2]
    # hidden = 1.0 on valid frames, 0.0 on padded: masked mean is exactly 1.0
    assert np.allclose(npz["mert_l19"][0], 1.0)
    assert np.allclose(npz["mert_l23"][0], 1.0)
    assert np.allclose(npz["mert_l19_excerpts"][0, 1], 1.0)
    assert np.allclose(npz["mert_l19_excerpts"][0, 2], 0.0)  # zero-padded slot
    assert model.calls[0]["attention_mask_given"] is True
    assert model.calls[0]["output_hidden_states"] is True
    call = extractor.calls[0]
    assert call["sampling_rate"] == 24000
    assert call["return_tensors"] == "pt"
    assert call["padding"] is True
    assert call["return_attention_mask"] is True
    assert call["do_normalize"] is False
    assert meta["sample_rate_hz"] == ex.MERT_SR
    assert meta["checkpoint"]["trusted_remote_code"] is True
    assert meta["checkpoint"]["custom_code_acknowledged"] is True
    assert meta["checkpoint"]["revision"] == REV
    assert meta["checkpoint"]["local_files_only"] is False


def test_mert_too_few_hidden_states(tmp_path, monkeypatch):
    patch_decode(monkeypatch)
    patch_mert(monkeypatch, model=FakeMertModel(n_hidden=10))
    with pytest.raises(RuntimeError, match="no tracks extracted"):
        ex.extract(make_manifest([make_track(tmp_path, "a.mp3")]), tmp_path / "out", model="mert",
                   revision=REV, allow_download=True, trust_remote_code=True, device="cpu")


# ── mask_pool (pure) ─────────────────────────────────────────────────────────


def test_mask_pool_math():
    h = np.array([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    out = ex.mask_pool(h, np.array([[1, 1, 0]]))
    assert out.shape == (1, 2)
    assert np.allclose(out, [[2.0, 3.0]])  # mean of frames 0 and 1 only


def test_mask_pool_torch_inputs():
    h = torch.tensor([[[1.0], [3.0], [5.0]]])
    out = ex.mask_pool(h, torch.tensor([[1, 0, 1]]))
    assert np.allclose(out, [[3.0]])


def test_mask_pool_single_excerpt():
    h = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    assert np.allclose(ex.mask_pool(h, np.array([1, 1, 0])), [1.5, 15.0])


def test_mask_pool_rejects_invalid():
    with pytest.raises(ValueError, match="temporal dims"):
        ex.mask_pool(np.zeros((3, 2)), np.zeros(4))
    with pytest.raises(ValueError, match="binary"):
        ex.mask_pool(np.zeros((2, 2)), np.array([0, 2]))
    with pytest.raises(ValueError, match="no valid frame"):
        ex.mask_pool(np.ones((2, 2)), np.array([0, 0]))
    with pytest.raises(ValueError, match="non-finite"):
        ex.mask_pool(np.array([[np.nan, 1.0], [1.0, 1.0]]), np.array([1, 1]))


# ── metadata, json sidecar, progress, hygiene ────────────────────────────────


def test_metadata_fields_and_json_sidecar(tmp_path, monkeypatch):
    patch_decode(monkeypatch)
    patch_maest(monkeypatch, tmp_path)
    tracks = [
        make_track(tmp_path, "ok.mp3", fp="a" * 64),
        make_track(tmp_path, "bad_empty.mp3", fp="b" * 64),
    ]
    manifest = make_manifest(tracks)
    meta = ex.extract(manifest, tmp_path / "out", device="cpu")
    assert meta["dataset_id"] == manifest["dataset_id"] == ds.dataset_id(manifest)
    assert meta["manifest_dataset_id"] == manifest["dataset_id"]
    assert meta["sample_rate_hz"] == ex.MAEST_SR
    assert meta["model"] == "maest"
    assert meta["checkpoint"]["sha256"] == hashlib.sha256((tmp_path / "ckpt.ckpt").read_bytes()).hexdigest()
    assert meta["checkpoint"]["sha256_expected"] == CKPT_SHA
    assert meta["checkpoint"]["arch"] == ex.MAEST_ARCH
    assert "NOT the Essentia" in meta["layers"]["maest_layer7"]
    assert meta["dtype"] == "float32"
    assert meta["device"] == "cpu"
    assert meta["wall_seconds"] > 0
    assert set(meta["per_track_timing"]) == {"a" * 64}
    assert meta["coverage"] == {"total": 2, "ok": 1, "failed": 1}
    assert len(meta["failures"]) == 1
    assert len(meta["artifact"]["sha256"]) == 64
    assert meta["rss"]["peak_rss_bytes"] > 0
    assert meta["software"]["python"].startswith("3")
    for key in ("maest-infer", "torchaudio", "librosa", "soundfile"):
        assert key in meta["software"]
    assert json.loads((tmp_path / "out" / "maest.json").read_text()) == meta
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["maest.json", "maest.npz"]


def test_progress_callback_and_flush_print(tmp_path, monkeypatch, capsys):
    patch_decode(monkeypatch)
    patch_maest(monkeypatch, tmp_path)
    msgs = []
    ex.extract(
        make_manifest([make_track(tmp_path, "ok.mp3"), make_track(tmp_path, "x_empty.mp3")]),
        tmp_path / "out", device="cpu", on_progress=msgs.append,
    )
    assert any("[1/2]" in m for m in msgs)
    assert any("FAILED" in m for m in msgs)
    capsys.readouterr()
    ex.extract(make_manifest([make_track(tmp_path, "ok2.mp3")]), tmp_path / "out2", device="cpu")
    captured = capsys.readouterr().out
    assert "[1/1]" in captured and "wrote maest.npz" in captured
