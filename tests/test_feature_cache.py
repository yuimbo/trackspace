"""Unit tests for the source-parameterised FeatureCache."""

import numpy as np
import pytest

from backend.embeddings.feature_cache import _SOURCES, FeatureCache


def _mk(tmp_path, max_memory: int = 2000) -> FeatureCache:
    return FeatureCache(str(tmp_path / "features.db"), max_memory=max_memory)


def test_roundtrip_each_source(tmp_path):
    fc = _mk(tmp_path)
    for i, source in enumerate(_SOURCES):
        fp = f"fp-{i}"
        vec = (np.arange(4 + i, dtype=np.float32) + 1.0)

        assert fc.get(source, fp) is None
        assert fc.has(source, fp) is False

        fc.put(source, fp, vec)

        assert fc.has(source, fp) is True
        got = fc.get(source, fp)
        assert got is not None
        np.testing.assert_allclose(got, vec)
        assert fc.count(source) == 1

        # The memory LRU is version-blind (as in the original cache); SQL reads
        # apply the version filter.
        fc._mem.clear()
        assert fc.get(source, fp, version=999) is None
        assert fc.has(source, fp, version=999) is False


def test_get_all_ignores_version_in_mem_and_filters_in_sql(tmp_path):
    fc = _mk(tmp_path)
    fc.put("clap", "a", np.ones(4, np.float32))
    fc.put("clap", "b", np.ones(4, np.float32), version=2)

    assert set(fc.get_all("clap", ["a", "b", "missing"])) == {"a", "b"}
    fc._mem.clear()
    assert set(fc.get_all("clap", ["a", "b"], version=0)) == {"a"}
    fc._mem.clear()
    assert set(fc.get_all("clap", ["a", "b"], version=2)) == {"b"}


def test_put_maest_writes_pair_and_count(tmp_path):
    fc = _mk(tmp_path)
    emb = np.arange(8, dtype=np.float32)
    logits = np.arange(519, dtype=np.float32)
    fc.put_maest("fp", emb, logits, version=1)

    assert fc.has("maest", "fp", version=1)
    assert fc.has("maest_logits", "fp", version=1)
    got_emb = fc.get("maest", "fp", version=1)
    got_logits = fc.get("maest_logits", "fp", version=1)
    assert got_emb is not None and got_logits is not None
    np.testing.assert_allclose(got_emb, emb)
    np.testing.assert_allclose(got_logits, logits)
    assert fc.count("maest") == 1


def test_lru_eviction_across_mixed_prefixes(tmp_path):
    fc = _mk(tmp_path, max_memory=3)
    fc.put("clap", "a", np.ones(2, np.float32))
    fc.put("effnet", "b", np.ones(2, np.float32))
    fc.put("rhythm", "c", np.ones(2, np.float32))
    assert set(fc._mem) == {"clap:a", "effnet:b", "rhythm:c"}

    fc.put("maest", "d", np.ones(2, np.float32))
    assert "clap:a" not in fc._mem
    assert len(fc._mem) == 3
    # Evicted from memory but still readable from SQLite.
    assert fc.get("clap", "a") is not None


def test_purge_old_versions_evicts_clap_mem(tmp_path):
    fc = _mk(tmp_path)
    fc.put("clap", "fp", np.ones(4, np.float32), version=2)
    assert "clap:fp" in fc._mem

    assert fc.purge_old_versions("clap", 3) == 1
    assert "clap:fp" not in fc._mem
    assert fc.get("clap", "fp", version=2) is None


def test_purge_old_versions_nulls_maest_pair(tmp_path):
    fc = _mk(tmp_path)
    fc.put_maest("fp", np.ones(4, np.float32), np.ones(519, np.float32), version=2)
    assert fc.has("maest", "fp", version=2)
    assert fc.has("maest_logits", "fp", version=2)

    assert fc.purge_old_versions("maest", 3) == 1
    assert fc.get("maest", "fp", version=2) is None
    assert fc.get("maest_logits", "fp", version=2) is None
    assert not any(k.startswith(("maest:", "maestlog:")) for k in fc._mem)


def test_get_all_chunks_large_batches(tmp_path):
    fc = _mk(tmp_path)
    fps = [f"fp{i}" for i in range(550)]
    for fp in fps:
        fc.put("rhythm", fp, np.ones(3, np.float32))
    fc._mem.clear()

    got = fc.get_all("rhythm", fps)
    assert len(got) == 550


def test_write_epoch_bumps_on_write_and_purge(tmp_path):
    fc = _mk(tmp_path)
    e0 = fc.write_epoch()
    fc.put("clap", "fp", np.ones(2, np.float32))
    e1 = fc.write_epoch()
    assert e1 > e0
    assert fc.purge_old_versions("clap", 1) == 1
    assert fc.write_epoch() > e1


def test_unknown_source_raises(tmp_path):
    fc = _mk(tmp_path)
    with pytest.raises(ValueError):
        fc.get("nope", "fp")
    with pytest.raises(ValueError):
        fc.put("nope", "fp", np.ones(2, np.float32))


# ── Prefixed-key LRU behaviour ────────────────────────────────


def test_lru_keys_use_documented_source_prefixes(tmp_path):
    """Every source maps to its own prefix, so mixed-source keys cannot collide."""
    fc = _mk(tmp_path, max_memory=16)
    for i, source in enumerate(_SOURCES):
        fc.put(source, f"fp{i}", np.ones(2, np.float32))

    expected = {f"{_SOURCES[s].lru_prefix}fp{i}" for i, s in enumerate(_SOURCES)}
    assert set(fc._mem) == expected


def test_lru_get_promotes_recency(tmp_path):
    fc = _mk(tmp_path, max_memory=2)
    fc.put("clap", "a", np.ones(2, np.float32))
    fc.put("effnet", "b", np.ones(2, np.float32))

    # Touching the oldest key makes it the most recently used.
    assert fc.get("clap", "a") is not None
    fc.put("rhythm", "c", np.ones(2, np.float32))

    assert "effnet:b" not in fc._mem  # evicted: it was the least recently used
    assert set(fc._mem) == {"clap:a", "rhythm:c"}


def test_lru_get_all_promotes_recency(tmp_path):
    fc = _mk(tmp_path, max_memory=2)
    fc.put("clap", "a", np.ones(2, np.float32))
    fc.put("clap", "b", np.ones(2, np.float32))

    # get_all promotes the keys it returns, so "a" survives and "b" is next out.
    fc.get_all("clap", ["a"])
    fc.put("rhythm", "c", np.ones(2, np.float32))

    assert "clap:b" not in fc._mem
    assert set(fc._mem) == {"clap:a", "rhythm:c"}


def test_purge_maest_evicts_only_maest_prefixes(tmp_path):
    fc = _mk(tmp_path, max_memory=16)
    fc.put("clap", "c", np.ones(2, np.float32))
    fc.put("effnet", "e", np.ones(2, np.float32))
    fc.put("rhythm", "r", np.ones(2, np.float32))
    fc.put_maest("m", np.ones(4, np.float32), np.ones(8, np.float32), version=1)
    assert {"maest:m", "maestlog:m"} <= set(fc._mem)

    assert fc.purge_old_versions("maest", 2) == 1
    assert not any(k.startswith(("maest:", "maestlog:")) for k in fc._mem)
    assert {"clap:c", "effnet:e", "rhythm:r"} <= set(fc._mem)
