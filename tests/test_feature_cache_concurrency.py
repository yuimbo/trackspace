"""Thread-contention tests for FeatureCache.

The real usage pattern is concurrent reads and writes while analysis runs in the
background. These tests hammer one cache from several threads and assert the lock
discipline and per-call SQLite connection strategy hold: no exceptions, the memory
LRU stays bounded, ``write_epoch`` is monotonic, and every non-purged write is
readable afterwards.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from backend.embeddings.feature_cache import _SOURCES, FeatureCache

#: Sources written directly; the MAEST pair goes through ``put_maest``.
_DIRECT_SOURCES = [s for s in _SOURCES if s not in ("maest", "maest_logits")]


def _mk(tmp_path, max_memory: int) -> FeatureCache:
    return FeatureCache(str(tmp_path / "features.db"), max_memory=max_memory)


def test_concurrent_reads_writes_and_purges(tmp_path):
    max_memory = 32
    fc = _mk(tmp_path, max_memory=max_memory)
    version = 7
    n_writers = 4
    per_writer = 40
    parties = n_writers + 3  # readers + purger + epoch sampler
    barrier = threading.Barrier(parties)

    def writer(wid: int) -> None:
        barrier.wait()
        for i in range(per_writer):
            fp = f"w{wid}-{i}"
            for source in _DIRECT_SOURCES:
                fc.put(
                    source,
                    fp,
                    np.full(4, float(wid) + i, np.float32),
                    version=version,
                )
            fc.put_maest(
                fp,
                np.ones(8, np.float32),
                np.ones(519, np.float32),
                version=version,
            )

    def reader() -> None:
        barrier.wait()
        for i in range(per_writer * n_writers):
            fp = f"w{i % n_writers}-{i % per_writer}"
            fc.get("clap", fp, version=version)
            fc.get_all("features", [fp], version=version)
            fc.has("rhythm", fp, version=version)
            fc.count("maest", version=version)

    def purger() -> None:
        barrier.wait()
        for _ in range(10):
            # Purges nothing: every row carries the current version.
            fc.purge_old_versions("clap", version)

    epochs: list[int] = []

    def sampler() -> None:
        barrier.wait()
        for _ in range(200):
            epochs.append(fc.write_epoch())

    with ThreadPoolExecutor(max_workers=parties) as pool:
        futures = [pool.submit(writer, w) for w in range(n_writers)]
        futures.append(pool.submit(reader))
        futures.append(pool.submit(purger))
        futures.append(pool.submit(sampler))
        for fut in futures:
            fut.result()  # re-raises any thread exception

    for wid in range(n_writers):
        for i in range(per_writer):
            fp = f"w{wid}-{i}"
            assert fc.has("clap", fp, version=version)
            assert fc.has("maest", fp, version=version)
            assert fc.has("maest_logits", fp, version=version)

    assert fc.count("clap", version=version) == n_writers * per_writer

    # The memory LRU never grew past its bound.
    assert len(fc._mem) <= max_memory

    # write_epoch is monotonic and advanced.
    assert epochs == sorted(epochs)
    assert fc.write_epoch() > 0


def test_concurrent_purge_removes_stale_rows(tmp_path):
    fc = _mk(tmp_path, max_memory=16)
    total = 200
    for i in range(total):
        fc.put("clap", f"fp{i}", np.ones(4, np.float32), version=1)

    barrier = threading.Barrier(4)

    def purger() -> None:
        barrier.wait()
        fc.purge_old_versions("clap", 2)

    def reader() -> None:
        barrier.wait()
        for i in range(total):
            fc.get("clap", f"fp{i}", version=1)

    def counter() -> None:
        barrier.wait()
        for _ in range(50):
            fc.count("clap", version=1)

    def sampler() -> None:
        barrier.wait()
        for _ in range(50):
            fc.write_epoch()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(purger),
            pool.submit(reader),
            pool.submit(counter),
            pool.submit(sampler),
        ]
        for fut in futures:
            fut.result()

    # Stale rows are gone from both SQLite and the memory LRU.
    assert fc.count("clap", version=1) == 0
    assert not any(k.startswith("clap:") for k in fc._mem)
