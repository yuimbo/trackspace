"""Library roots: stable IDs, nested-folder merge rules, and roots.json persistence."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, MutableMapping, MutableSet
from typing import Any

#: A root the user considers their own library (default).
ROLE_LIBRARY = "library"
#: A root analysed and laid out, but kept out of the default view and the head's
#: training labels — e.g. downloads or reference material.
ROLE_EXTERNAL = "external"
#: Every accepted role, for request validation.
ROLES = (ROLE_LIBRARY, ROLE_EXTERNAL)


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


def root_role(root_id: str, root_roles: Mapping[str, str]) -> str:
    """Role of *root_id*, defaulting to :data:`ROLE_LIBRARY` when unrecorded."""
    return root_roles.get(root_id, ROLE_LIBRARY)


def set_root_role(
    root_id: str, role: str, root_roles: MutableMapping[str, str]
) -> None:
    """Record *role* for *root_id*; the caller validates the role value."""
    root_roles[root_id] = role


def library_root_ids(
    roots: Mapping[str, str], root_roles: Mapping[str, str]
) -> list[str]:
    """Ids of roots that are part of "my library" (not external)."""
    return [
        rid
        for rid in roots
        if root_role(rid, root_roles) != ROLE_EXTERNAL
    ]


def save_roots_state(
    roots: Mapping[str, str],
    state_path: str,
    *,
    temp_roots: Iterable[str] = (),
    root_roles: Mapping[str, str] | None = None,
) -> None:
    """Write root paths to JSON (ids are not persisted; recomputed on load).

    *temp_roots* are ids of ephemeral imprint roots; they are persisted by path so
    a restart mid-imprint can still finish and remove them. *root_roles* is persisted
    the same way, as the list of paths whose role is external (library is the
    default, so it needs no key and old files stay valid).
    """
    roles = root_roles or {}
    payload = {
        "roots": list(roots.values()),
        "external_roots": [
            roots[rid]
            for rid in sorted(roots)
            if roles.get(rid) == ROLE_EXTERNAL
        ],
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
    root_roles: MutableMapping[str, str] | None = None,
) -> None:
    """Load paths from JSON; optional *default_root* when file is empty or missing.

    Non-directories are skipped. After load, state is normalised and written back
    (merges nested paths, drops stale entries). Persisted ephemeral-root and
    external-root paths are mapped back to their ids so an interrupted imprint
    stays removable and the view keeps its sections.
    """
    roots.clear()
    if temp_roots is not None:
        temp_roots.clear()
    if root_roles is not None:
        root_roles.clear()
    loaded_any = False
    temp_paths: list[str] = []
    external_paths: list[str] = []
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
        external_paths = [
            os.path.abspath(str(raw)) for raw in data.get("external_roots", [])
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

    if root_roles is not None:
        external = set(external_paths)
        for rid, root_abs in roots.items():
            root_roles[rid] = (
                ROLE_EXTERNAL if root_abs in external else ROLE_LIBRARY
            )

    save_roots_state(
        roots,
        state_path,
        temp_roots=temp_roots or (),
        root_roles=root_roles or {},
    )
