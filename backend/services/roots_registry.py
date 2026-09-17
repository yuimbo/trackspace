"""Library roots: stable IDs, nested-folder merge rules, and roots.json persistence."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, MutableMapping, MutableSet
from typing import Any


def stable_root_id(abs_path: str, roots: Mapping[str, str]) -> str:
    """Short hash-based id, uniquified if the same prefix collides with another path."""
    base = hashlib.sha1(abs_path.encode("utf-8")).hexdigest()[:8]
    rid = base
    i = 1
    while rid in roots and roots[rid] != abs_path:
        rid = f"{base[:6]}{i:02d}"
        i += 1
    return rid


def add_root(abs_path: str, roots: MutableMapping[str, str]) -> tuple[bool, str]:
    """Insert a root folder and drop redundant descendants.

    Returns ``(False, existing_id)`` if *abs_path* is already covered by a root.
    """
    p = os.path.abspath(abs_path)
    if not os.path.isdir(p):
        raise FileNotFoundError(p)
    for rid, root_abs in list(roots.items()):
        if p == root_abs or p.startswith(root_abs.rstrip(os.sep) + os.sep):
            return False, rid
    for rid, root_abs in list(roots.items()):
        if root_abs.startswith(p.rstrip(os.sep) + os.sep):
            del roots[rid]
    rid = stable_root_id(p, roots)
    roots[rid] = p
    return True, rid


def remove_root(root_id: str, roots: MutableMapping[str, str]) -> bool:
    if root_id in roots:
        del roots[root_id]
        return True
    return False


def save_roots_state(
    roots: Mapping[str, str],
    state_path: str,
    *,
    temp_roots: Iterable[str] = (),
) -> None:
    """Write root paths to JSON (ids are not persisted; recomputed on load).

    *temp_roots* are ids of ephemeral imprint roots; they are persisted by path so
    a restart mid-imprint can still finish and remove them.
    """
    payload = {
        "roots": list(roots.values()),
        "temp_roots": [
            roots[rid] for rid in sorted(temp_roots) if rid in roots
        ],
    }
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def load_roots_state(
    roots: MutableMapping[str, str],
    state_path: str,
    *,
    default_root: str | None,
    temp_roots: MutableSet[str] | None = None,
) -> None:
    """Load paths from JSON; optional *default_root* when file is empty or missing.

    Non-directories are skipped. After load, state is normalised and written back
    (merges nested paths, drops stale entries). Persisted ephemeral-root paths are
    mapped back to their ids so an interrupted imprint stays removable.
    """
    roots.clear()
    if temp_roots is not None:
        temp_roots.clear()
    loaded_any = False
    temp_paths: list[str] = []
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        for raw in data.get("roots", []):
            p = os.path.abspath(str(raw))
            if os.path.isdir(p):
                add_root(p, roots)
                loaded_any = True
        temp_paths = [
            os.path.abspath(str(raw)) for raw in data.get("temp_roots", [])
        ]
    except Exception:
        loaded_any = False

    if not loaded_any and default_root is not None:
        add_root(default_root, roots)

    if temp_roots is not None:
        by_path = {root_abs: rid for rid, root_abs in roots.items()}
        for p in temp_paths:
            rid = by_path.get(p)
            if rid is not None:
                temp_roots.add(rid)

    save_roots_state(roots, state_path, temp_roots=temp_roots or ())
