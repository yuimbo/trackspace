"""MERT module: licence/trust gates, masked pooling, and L19 extraction.

Uses no model and no download: the load-gate tests return before ``transformers``
is imported (the gates are checked first), and the pooling/extraction tests run
against numpy and a deterministic fake that mirrors the pinned ``modeling_mert2.py``
forward (per-length-group ``_encode`` + full-batch assembly of ``hidden_states``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from backend.embeddings import mert as mert_mod


def _reset(monkeypatch, *, accepted: bool, trusted: bool) -> None:
    monkeypatch.setattr(mert_mod, "_loaded", False)
    monkeypatch.setattr(mert_mod, "_load_failed", None)
    monkeypatch.setattr(mert_mod, "_accept_noncommercial", accepted)
    monkeypatch.setattr(mert_mod, "_trust_remote_code", trusted)
    monkeypatch.setattr(mert_mod, "_allow_download", False)


# ── security gates ────────────────────────────────────────────


def test_load_requires_noncommercial_licence(monkeypatch):
    _reset(monkeypatch, accepted=False, trusted=False)
    mert_mod.load_model()
    assert not mert_mod.is_model_ready()
    assert mert_mod.load_error() is not None
    assert "CC-BY-NC" in mert_mod.load_error()


def test_load_requires_trust_remote_code(monkeypatch):
    _reset(monkeypatch, accepted=True, trusted=False)
    mert_mod.load_model()
    assert not mert_mod.is_model_ready()
    assert mert_mod.load_error() is not None
    assert "trust-remote-code" in mert_mod.load_error()


def test_gates_default_off():
    assert mert_mod.licence_acceptance() is False
    assert mert_mod._trust_remote_code is False
    assert mert_mod._allow_download is False


# ── off-by-one pin ────────────────────────────────────────────


def test_l19_off_by_one_pin():
    # Pinned config num_hidden_layers=24, hidden_size=1024; modeling_mert2.py
    # `_encode` appends one state per layer so hidden_states[0]=layers[0]=block 1
    # and L19 (one-based block 19) is index 18.
    assert mert_mod.MERT_L19_BLOCK == 19
    assert mert_mod.MERT_L19_INDEX == 18
    assert mert_mod.MERT_EMBED_DIM == 1024
    assert len(mert_mod.MERT_REVISION) == 40


# ── masked temporal pooling ───────────────────────────────────


def test_mask_pool_masked_mean():
    # 2 frames valid of 4 -> masked mean over the valid frames only.
    hidden = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    mask = np.array([1, 1, 0, 0])
    pooled = mert_mod.mask_pool(hidden, mask)
    np.testing.assert_allclose(pooled, np.array([1.5, 15.0]))
    assert pooled.dtype == np.float32


def test_mask_pool_batched():
    hidden = np.array([[[1.0, 1.0], [3.0, 3.0]], [[5.0, 5.0], [7.0, 7.0]]])
    mask = np.array([[1, 0], [1, 1]])
    pooled = mert_mod.mask_pool(hidden, mask)
    np.testing.assert_allclose(pooled, np.array([[1.0, 1.0], [6.0, 6.0]]))


def test_mask_pool_validates():
    with pytest.raises(ValueError, match="temporal dims differ"):
        mert_mod.mask_pool(np.ones((4, 8)), np.ones(5))
    with pytest.raises(ValueError, match="binary"):
        mert_mod.mask_pool(np.ones((4, 8)), np.array([0, 1, 2, 0]))
    with pytest.raises(ValueError, match="non-finite"):
        mert_mod.mask_pool(
            np.array([[1.0, np.nan], [2.0, 3.0]]), np.ones(2)
        )
    with pytest.raises(ValueError, match="no valid frame"):
        mert_mod.mask_pool(np.ones((4, 8)), np.zeros(4))


def test_mask_pool_accepts_tensors():
    pooled = mert_mod.mask_pool(
        torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]),
        torch.tensor([1, 1, 0, 0]),
    )
    np.testing.assert_allclose(pooled, np.array([1.5, 15.0]))


# ── L19 extraction (faithful fake of the pinned MERT2 forward) ──


class _FakeMert2Output:
    def __init__(self, last_hidden_state, hidden_states, feature_mask):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states
        self.feature_attention_mask = feature_mask


class _FakeHandle:
    def __init__(self, block, fn):
        self._block = block
        self._fn = fn
        self.removed = False

    def remove(self):
        self.removed = True
        self._block._hooks.pop(self, None)


class _FakeBlock:
    """Hookable stand-in for a ConformerBlock: returns ``hidden + index``, fires hooks."""

    def __init__(self, index):
        self.index = index
        self._hooks = {}

    def register_forward_hook(self, fn):
        handle = _FakeHandle(self, fn)
        self._hooks[handle] = fn
        return handle

    def forward(self, hidden, positions=None):
        out = hidden + self.index  # deterministic, layer-observable
        for fn in list(self._hooks.values()):
            fn(self, (hidden, positions), out)
        return out


class _FakeConfig:
    def __init__(self, hidden, ratio):
        self.hidden_size = hidden
        self.inputs_to_logits_ratio = ratio


class _FakeMert2:
    """Deterministic, hookable stand-in for the pinned MERT2Model forward.

    Mirrors modeling_mert2.py: each distinct raw waveform length is encoded
    separately (``_encode`` per unique ``attention_mask.sum`` length) and the
    per-group layer states are zero-padded and ``index_copy``'d into the full
    batch (forward lines 343-356). When ``output_hidden_states`` is False no states
    tuple is built — only ``last_hidden_state`` and ``feature_attention_mask`` are
    returned, and per-layer output is reachable only through the ``layers[i]``
    forward hooks. A layer's output encodes (per-example seed, layer index, frame
    index), so parity to ``hidden_states[18]`` is observable.
    """

    def __init__(self, n_layers=24, hidden=1024, ratio=960):
        self.n_layers = n_layers
        self.hidden = hidden
        self.ratio = ratio
        self.layers = [_FakeBlock(i) for i in range(n_layers)]
        self.config = _FakeConfig(hidden, ratio)

    def _encode(self, input_values):
        g, samples = input_values.shape
        feat = samples // self.ratio
        seed = input_values[:, 0]  # (G,) per-example identity
        frame = torch.arange(feat, dtype=torch.float32)  # (feat,)
        hidden = (seed[:, None] + frame[None, :])[:, :, None].expand(g, feat, self.hidden).float()
        states = []
        for block in self.layers:
            hidden = block.forward(hidden)
            states.append(hidden)
        return hidden, states

    def __call__(self, input_values, attention_mask=None, output_hidden_states=False):
        batch, samples = input_values.shape
        if attention_mask is not None:
            lengths = attention_mask.bool().sum(dim=1)
            feature_lengths = lengths // self.ratio
            maximum = int(feature_lengths.max().item())
            feature_mask = torch.arange(maximum)[None, :] < feature_lengths[:, None]
            hidden, state_values = None, None
            for length in torch.unique(lengths, sorted=True).tolist():
                indices = torch.where(lengths == length)[0]
                gh, gs = self._encode(input_values.index_select(0, indices)[:, :length])
                pad = (0, 0, 0, maximum - gh.shape[1])
                if hidden is None:
                    hidden = gh.new_zeros(batch, maximum, self.hidden)
                    if output_hidden_states:
                        state_values = [torch.zeros_like(hidden) for _ in self.layers]
                hidden = hidden.index_copy(0, indices, torch.nn.functional.pad(gh, pad))
                if state_values is not None:
                    for i, v in enumerate(gs):
                        state_values[i] = state_values[i].index_copy(
                            0, indices, torch.nn.functional.pad(v, pad)
                        )
            states = tuple(state_values) if state_values is not None else None
        else:
            hidden, states = self._encode(input_values)
            feature_mask = torch.ones(hidden.shape[:2], dtype=torch.bool)
            states = tuple(states) if output_hidden_states else None
        return _FakeMert2Output(hidden, states, feature_mask)


class _DoubleFireBlock(_FakeBlock):
    """Fires each hook twice per forward, simulating a changed layout."""

    def forward(self, hidden, positions=None):
        out = hidden + self.index
        fns = list(self._hooks.values())
        for _ in range(2):
            for fn in fns:
                fn(self, (hidden, positions), out)
        return out


class _DoubleFireMert(_FakeMert2):
    def __init__(self):
        super().__init__(n_layers=24)
        self.layers = [_DoubleFireBlock(i) for i in range(24)]


def _heterogeneous_batch():
    """Two tracks, different lengths: track0 4 feature frames, track1 2."""
    ratio = 960
    lengths = [4 * ratio, 2 * ratio]  # 3840, 1920 samples
    samples = max(lengths)
    input_values = torch.zeros(2, samples)
    input_values[0, 0] = 2.0  # track0 seed
    input_values[1, 0] = 5.0  # track1 seed
    attention_mask = torch.zeros(2, samples, dtype=torch.int64)
    for i, length in enumerate(lengths):
        attention_mask[i, :length] = 1
    return input_values, attention_mask


def test_extract_l19_equals_hidden_states_18_homogeneous():
    # A single unpadded track -> L19 (from the layers[18] hook) is exactly
    # hidden_states[18] from a reference output_hidden_states=True forward.
    model = _FakeMert2()
    samples = 4 * 960
    input_values = torch.zeros(1, samples)
    input_values[0, 0] = 2.0
    l19, mask = mert_mod._extract_l19(model, input_values, None, batch_len=1)
    assert tuple(l19.shape) == (1, 4, 1024)
    assert tuple(mask.shape) == (1, 4)
    out = model(input_values, None, output_hidden_states=True)
    assert torch.equal(l19, out.hidden_states[mert_mod.MERT_L19_INDEX])
    assert not model.layers[mert_mod.MERT_L19_INDEX]._hooks  # hook cleaned up
    # Pooled layer-19 masked mean over valid frames. Each block adds its index to
    # the running hidden, so block 18 contributes triangular(18) = 171:
    # seed 2 + mean(frame 0..3)=1.5 + 171 = 174.5.
    pooled = mert_mod.mask_pool(l19.numpy()[0], mask.numpy()[0])
    np.testing.assert_allclose(pooled, np.full(1024, 174.5))


def test_extract_l19_heterogeneous_batch_parity():
    # Different lengths -> one forward per homogeneous subgroup via the layers[18]
    # hook, then zero-padded assembly. Parity: equals output_hidden_states[18].
    model = _FakeMert2()
    input_values, attention_mask = _heterogeneous_batch()
    l19, mask = mert_mod._extract_l19(model, input_values, attention_mask, batch_len=2)
    assert tuple(l19.shape) == (2, 4, 1024)
    assert tuple(mask.shape) == (2, 4)
    assert mask.tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    out = model(input_values, attention_mask, output_hidden_states=True)
    assert torch.equal(l19, out.hidden_states[mert_mod.MERT_L19_INDEX])
    assert not model.layers[mert_mod.MERT_L19_INDEX]._hooks  # hook cleaned up
    # Masked means over each track's valid frames only (block-18 contribution 171):
    #   track0 seed 2, F=4 -> 2 + mean(0..3)=1.5 + 171 = 174.5
    #   track1 seed 5, F=2 -> 5 + mean(0..1)=0.5 + 171 = 176.5
    pooled = mert_mod.mask_pool(l19.numpy(), mask.numpy())
    np.testing.assert_allclose(pooled[0], np.full(1024, 174.5))
    np.testing.assert_allclose(pooled[1], np.full(1024, 176.5))


def test_extract_l19_does_not_request_output_hidden_states():
    # Production must run the forward WITHOUT output_hidden_states=True so the
    # model never builds the 24-state tuple; the L19 hook is then the only L19
    # source. Wrap the fake's __call__ to record what the production path requests.
    model = _FakeMert2()
    input_values, attention_mask = _heterogeneous_batch()
    seen = {"flags": []}

    orig = _FakeMert2.__call__

    def _wrapped(self, input_values, attention_mask=None, output_hidden_states=False):
        seen["flags"].append(output_hidden_states)
        return orig(
            self, input_values, attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

    _FakeMert2.__call__ = _wrapped
    try:
        l19, _mask = mert_mod._extract_l19(model, input_values, attention_mask, batch_len=2)
    finally:
        _FakeMert2.__call__ = orig
    assert seen["flags"] and all(f is False for f in seen["flags"])
    # Reference forward still yields the same L19 from hidden_states[18].
    out = model(input_values, attention_mask, output_hidden_states=True)
    assert torch.equal(l19, out.hidden_states[mert_mod.MERT_L19_INDEX])


def test_extract_l19_returns_cpu():
    model = _FakeMert2()
    input_values, attention_mask = _heterogeneous_batch()
    l19, mask = mert_mod._extract_l19(model, input_values, attention_mask, batch_len=2)
    assert l19.device.type == "cpu"
    assert l19.dtype == torch.float32
    assert mask.device.type == "cpu"


def test_extract_l19_validates_too_few_layers():
    model = _FakeMert2(n_layers=10)  # no layer index 18
    input_values, attention_mask = _heterogeneous_batch()
    with pytest.raises(ValueError, match="more than"):
        mert_mod._extract_l19(model, input_values, attention_mask, batch_len=2)


def test_extract_l19_validates_embed_dim():
    model = _FakeMert2(hidden=768)  # wrong embed dim -> hook raises loudly
    input_values, attention_mask = _heterogeneous_batch()
    with pytest.raises(RuntimeError, match="unexpected"):
        mert_mod._extract_l19(model, input_values, attention_mask, batch_len=2)


def test_extract_l19_validates_batch_len():
    model = _FakeMert2()
    input_values, attention_mask = _heterogeneous_batch()
    with pytest.raises(ValueError, match="batch mismatch"):
        mert_mod._extract_l19(model, input_values, attention_mask, batch_len=3)


def test_run_forward_l19_asserts_single_hook_fire():
    # A layer that fires its hook more than once per homogeneous subgroup means the
    # layout changed; the helper must raise rather than pool a partial capture.
    model = _DoubleFireMert()
    input_values = torch.zeros(1, 4 * 960)
    input_values[0, 0] = 1.0
    with pytest.raises(RuntimeError, match="fired 2 times"):
        mert_mod._extract_l19(model, input_values, None, batch_len=1)
    assert not model.layers[mert_mod.MERT_L19_INDEX]._hooks  # hook cleaned up