"""Process resource limits — run early from ``app`` before Torch / MPS load."""

from __future__ import annotations

import os
import sys


def raise_nofile_limit() -> None:
    """Raise the soft RLIMIT_NOFILE cap before HuggingFace shards and parallel I/O.

    macOS shells often default to 256 open files; model weights, SQLite WAL,
    and ``ThreadPoolExecutor`` library scans can exhaust that and make Metal
    fail with ``Too many open files`` when opening ``default.metallib``.

    Set ``TRACKSPACE_RLIMIT_NOFILE`` to the minimum soft limit wanted (still
    capped by the hard limit).  No-op on Windows.
    """
    if sys.platform == "win32":
        return
    try:
        import resource
    except ImportError:
        return
    raw = os.environ.get("TRACKSPACE_RLIMIT_NOFILE", "").strip()
    try:
        min_desired = int(raw) if raw else 10_240
    except ValueError:
        min_desired = 10_240
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (ValueError, OSError):
        return
    if soft >= min_desired:
        return
    target = min_desired
    if hard != resource.RLIM_INFINITY:
        target = min(target, hard)
    if target <= soft:
        return
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (ValueError, OSError):
        pass
