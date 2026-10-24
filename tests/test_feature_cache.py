"""Unit tests for the source-parameterised FeatureCache."""

import numpy as np
import pytest

from backend.embeddings.feature_cache import _SOURCES, FeatureCache
from tests.conftest import install_counting_conn


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
        fc.clear_memory()
        assert fc.get(source, fp, version=999) is None
        assert fc.has(source, fp, version=999) is False


def test_get_all_ignores_version_in_mem_and_filters_in_sql(tmp_path):
    fc = _mk(tmp_path)
    fc.put("clap", "a", np.ones(4, np.float32))
    fc.put("clap", "b", np.ones(4, np.float32), version=2)

    assert set(fc.get_all("clap", ["a", "b", "missing"])) == {"a", "b"}
    fc.clear_memory()
    assert set(fc.get_all("clap", ["a", "b"], version=0)) == {"a"}
    fc.clear_memory()
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
    fc.put("features", "b", np.ones(2, np.float32))
    fc.put("rhythm", "c", np.ones(2, np.float32))
    assert set(fc._mem) == {"clap:a", "feat:b", "rhythm:c"}

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
    fc.clear_memory()

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
    fc.put("features", "b", np.ones(2, np.float32))

    # Touching the oldest key makes it the most recently used.
    assert fc.get("clap", "a") is not None
    fc.put("rhythm", "c", np.ones(2, np.float32))

    assert "feat:b" not in fc._mem  # evicted: it was the least recently used
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
    fc.put("features", "e", np.ones(2, np.float32))
    fc.put("rhythm", "r", np.ones(2, np.float32))
    fc.put_maest("m", np.ones(4, np.float32), np.ones(8, np.float32), version=1)
    assert {"maest:m", "maestlog:m"} <= set(fc._mem)

    assert fc.purge_old_versions("maest", 2) == 1
    assert not any(k.startswith(("maest:", "maestlog:")) for k in fc._mem)
    assert {"clap:c", "feat:e", "rhythm:r"} <= set(fc._mem)


def test_put_maest_writes_excerpts(tmp_path):
    fc = _mk(tmp_path)
    excerpts = np.arange(3 * 8, dtype=np.float32).reshape(3, 8)
    fc.put_maest(
        "fp", np.ones(8, np.float32), np.ones(519, np.float32),
        excerpts=excerpts, version=1,
    )
    assert fc.has("maest_excerpts", "fp", version=1)
    got = fc.get("maest_excerpts", "fp", version=1)
    assert got is not None
    np.testing.assert_allclose(got, excerpts.reshape(-1))


def test_put_maest_without_excerpts_preserves_existing(tmp_path):
    fc = _mk(tmp_path)
    excerpts = np.ones((2, 4), np.float32)
    fc.put_maest(
        "fp", np.ones(8, np.float32), np.ones(519, np.float32),
        excerpts=excerpts, version=1,
    )
    # A caller with only the pooled pair must not erase mined vectors.
    fc.put_maest("fp", np.full(8, 2.0, np.float32), np.ones(519, np.float32), version=1)
    assert fc.has("maest_excerpts", "fp", version=1)


def test_purge_maest_clears_excerpts(tmp_path):
    fc = _mk(tmp_path)
    fc.put_maest(
        "fp", np.ones(4, np.float32), np.ones(8, np.float32),
        excerpts=np.ones((2, 4), np.float32), version=1,
    )
    assert fc.purge_old_versions("maest", 2) == 1
    assert fc.get("maest_excerpts", "fp", version=1) is None
    assert not any(k.startswith("maestexc:") for k in fc._mem)


def test_supervised_source_roundtrip_and_versioning(tmp_path):
    fc = _mk(tmp_path)
    v1 = np.ones(4, np.float32)
    v2 = np.full(4, 2.0, np.float32)
    fc.put("supervised", "fp", v1, version=1)
    fc.clear_memory()
    assert fc.has("supervised", "fp", version=1)
    assert fc.has("supervised", "fp", version=2) is False

    # A retrain rewrites the same row under the new run id.
    fc.put("supervised", "fp", v2, version=2)
    fc.clear_memory()
    assert fc.get("supervised", "fp", version=1) is None
    np.testing.assert_allclose(fc.get("supervised", "fp", version=2), v2)


def test_meta_roundtrip_and_missing(tmp_path):
    fc = _mk(tmp_path)
    assert fc.get_meta("supervised_run_id") is None
    fc.set_meta("supervised_run_id", 3)
    assert fc.get_meta("supervised_run_id") == "3"
    fc.set_meta("supervised_run_id", 4)
    assert fc.get_meta("supervised_run_id") == "4"


# ── Byte budget (max_memory_bytes) ────────────────────────────


def _mk_bytes(tmp_path, budget: int, max_memory: int = 1_000_000) -> FeatureCache:
    return FeatureCache(
        str(tmp_path / "features.db"),
        max_memory=max_memory,
        max_memory_bytes=budget,
    )


def test_memory_bytes_tracks_puts_and_overwrites(tmp_path):
    fc = _mk(tmp_path)
    assert fc.memory_bytes() == 0

    vec = np.ones(10, np.float32)
    fc.put("clap", "a", vec)
    assert fc.memory_bytes() == int(vec.nbytes)

    # Overwrite: the old array's bytes are subtracted before the new counted.
    bigger = np.ones(20, np.float32)
    fc.put("clap", "a", bigger)
    assert fc.memory_bytes() == int(bigger.nbytes)

    smaller = np.ones(4, np.float32)
    fc.put("clap", "a", smaller)
    assert fc.memory_bytes() == int(smaller.nbytes)


def test_byte_budget_evicts_lru_entries(tmp_path):
    fc = _mk_bytes(tmp_path, budget=64)
    fc.put("clap", "a", np.ones(8, np.float32))  # 32 B
    fc.put("clap", "b", np.ones(8, np.float32))  # 64 B total — at budget
    assert set(fc._mem) == {"clap:a", "clap:b"}

    fc.put("clap", "c", np.ones(8, np.float32))  # 96 B > 64 → evict oldest
    assert "clap:a" not in fc._mem
    assert set(fc._mem) == {"clap:b", "clap:c"}
    assert fc.memory_bytes() == 64


def test_entry_cap_still_applies_with_byte_budget(tmp_path):
    # Entry cap and byte budget are independent limits; either may trigger.
    fc = FeatureCache(
        str(tmp_path / "features.db"),
        max_memory=2,
        max_memory_bytes=1_000_000,
    )
    for i in range(3):
        fc.put("clap", str(i), np.ones(8, np.float32))
    assert len(fc._mem) == 2
    assert "clap:0" not in fc._mem


def test_no_byte_eviction_when_budget_none(tmp_path):
    fc = _mk(tmp_path, max_memory=2)
    fc.put("clap", "a", np.ones(1024, np.float32))
    fc.put("clap", "b", np.ones(1024, np.float32))
    # 8 KiB across two entries; the entry cap (not bytes) governs eviction.
    assert set(fc._mem) == {"clap:a", "clap:b"}
    assert fc.memory_bytes() == 2 * int(np.ones(1024, np.float32).nbytes)
    fc.put("clap", "c", np.ones(1024, np.float32))
    assert "clap:a" not in fc._mem


def test_prefix_eviction_subtracts_bytes(tmp_path):
    fc = _mk(tmp_path)
    emb = np.ones(8, np.float32)
    logits = np.ones(519, np.float32)
    clap = np.ones(4, np.float32)
    fc.put_maest("m", emb, logits, version=1)
    fc.put("clap", "c", clap)
    assert fc.memory_bytes() == emb.nbytes + logits.nbytes + clap.nbytes

    with fc._lock:
        fc._evict_mem_keys_with_prefix_unlocked("maest:")
    assert fc.memory_bytes() == logits.nbytes + clap.nbytes
    with fc._lock:
        fc._evict_mem_keys_with_prefix_unlocked("maestlog:")
    assert fc.memory_bytes() == clap.nbytes
    assert set(fc._mem) == {"clap:c"}


def test_purge_old_versions_subtracts_bytes(tmp_path):
    fc = _mk(tmp_path)
    emb = np.ones(8, np.float32)
    logits = np.ones(519, np.float32)
    clap = np.ones(4, np.float32)
    fc.put_maest("m", emb, logits, version=1)
    fc.put("clap", "c", clap)

    assert fc.purge_old_versions("maest", 2) == 1
    assert fc.memory_bytes() == clap.nbytes
    assert set(fc._mem) == {"clap:c"}


def test_get_all_refreshes_recency_under_byte_eviction(tmp_path):
    fc = _mk_bytes(tmp_path, budget=96)
    fc.put("clap", "a", np.ones(8, np.float32))  # 32 B
    fc.put("clap", "b", np.ones(8, np.float32))  # 64 B
    # Touching "a" makes "b" the least recently used.
    fc.get_all("clap", ["a"])
    fc.put("clap", "c", np.ones(8, np.float32))  # 96 B — at budget
    assert set(fc._mem) == {"clap:a", "clap:b", "clap:c"}
    fc.put("clap", "d", np.ones(8, np.float32))  # 128 B > 96 → evict "b"
    assert "clap:b" not in fc._mem
    assert set(fc._mem) == {"clap:a", "clap:c", "clap:d"}


def test_oversized_entry_survives_own_put_but_is_lru_evictable(tmp_path):
    fc = _mk_bytes(tmp_path, budget=16)
    big = np.ones(1024, np.float32)  # 4 KiB ≫ budget
    fc.put("clap", "a", big)
    assert set(fc._mem) == {"clap:a"}
    assert fc.memory_bytes() == int(big.nbytes)

    # A later put evicts it like any other LRU entry; the guard only stops its
    # own put from looping the cache empty.
    fc.put("clap", "b", np.ones(2, np.float32))
    assert set(fc._mem) == {"clap:b"}
    assert fc.memory_bytes() == 8


# ── presence() memo ───────────────────────────────────────────


def test_presence_memo_hit_avoids_second_sql(tmp_path):
    fc = _mk(tmp_path)
    for src in ("clap", "maest"):
        fc.put(src, "a", np.ones(4, np.float32), version=1)
    fc.clear_memory()  # force the SQL path
    counter = [0]
    install_counting_conn(fc, counter)

    versions = {"clap": 1, "maest": 1}
    first = fc.presence(versions, ["a"])
    assert first["clap"] == {"a": True}
    assert first["maest"] == {"a": True}
    n = counter[0]
    assert n > 0

    assert fc.presence(versions, ["a"]) == first
    assert counter[0] == n  # memo hit: no second scan


def test_presence_memo_put_and_purge_invalidate(tmp_path):
    fc = _mk(tmp_path)
    fc.put("clap", "a", np.ones(4, np.float32), version=1)
    fc.clear_memory()
    counter = [0]
    install_counting_conn(fc, counter)

    versions = {"clap": 1}
    assert fc.presence(versions, ["a"])["clap"] == {"a": True}
    n0 = counter[0]

    fc.put("clap", "b", np.ones(4, np.float32), version=1)  # epoch bump
    assert fc.presence(versions, ["a", "b"])["clap"] == {"a": True, "b": True}
    assert counter[0] > n0  # miss after put
    n1 = counter[0]
    assert fc.presence(versions, ["a", "b"])["clap"] == {"a": True, "b": True}
    assert counter[0] == n1  # hit again

    assert fc.purge_old_versions("clap", 2) == 2  # epoch bump (n > 0)
    assert fc.presence(versions, ["a", "b"])["clap"] == {}
    assert counter[0] > n1  # miss after purge

    # A no-op purge (n == 0) must NOT invalidate the memo.
    assert fc.purge_old_versions("clap", 2) == 0
    n3 = counter[0]  # the no-op UPDATE itself is counted
    assert fc.presence(versions, ["a", "b"])["clap"] == {}
    assert counter[0] == n3  # still a memo hit


def test_presence_memo_different_fps_and_versions_miss(tmp_path):
    fc = _mk(tmp_path)
    fc.put("clap", "a", np.ones(4, np.float32), version=1)
    fc.clear_memory()
    counter = [0]
    install_counting_conn(fc, counter)

    assert fc.presence({"clap": 1}, ["a"])["clap"] == {"a": True}
    n0 = counter[0]

    # Different fingerprint set → miss.
    assert fc.presence({"clap": 1}, ["a", "b"])["clap"] == {"a": True}
    assert counter[0] > n0
    n1 = counter[0]

    # Same set in a different order → hit (digest sorts the fingerprints).
    assert fc.presence({"clap": 1}, ["b", "a"])["clap"] == {"a": True}
    assert counter[0] == n1

    # Different source versions (e.g. a supervised retrain) → miss.
    assert fc.presence({"clap": 2}, ["a"])["clap"] == {}
    assert counter[0] > n1


def test_presence_memo_result_isolated_from_memo(tmp_path):
    fc = _mk(tmp_path)
    fc.put("clap", "a", np.ones(4, np.float32), version=1)
    fc.clear_memory()
    counter = [0]
    install_counting_conn(fc, counter)

    versions = {"clap": 1}
    first = fc.presence(versions, ["a"])
    n0 = counter[0]
    # Aggressively mutate the returned maps.
    first["clap"]["b"] = True
    first["maest"] = {"x": True}

    assert fc.presence(versions, ["a"]) == {"clap": {"a": True}}
    assert counter[0] == n0  # memo untouched by the caller's mutations
