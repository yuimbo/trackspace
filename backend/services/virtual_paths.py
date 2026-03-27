from __future__ import annotations

import os
from collections.abc import Mapping
from werkzeug.exceptions import BadRequest, Forbidden, NotFound


def root_and_rel_from_virtual(vpath: str, roots: Mapping[str, str]) -> tuple[str, str]:
    raw = (vpath or "").strip().lstrip("/")
    if not raw:
        raise BadRequest()
    parts = raw.split("/", 1)
    rid = parts[0]
    rel = parts[1] if len(parts) > 1 else ""
    root_abs = roots.get(rid)
    if not root_abs:
        raise NotFound()
    return root_abs, rel


def virtual_from_abs(abs_path: str, roots: Mapping[str, str]) -> str:
    best_rid = None
    best_root = None
    for rid, root in roots.items():
        if abs_path == root or abs_path.startswith(root.rstrip(os.sep) + os.sep):
            if best_root is None or len(root) > len(best_root):
                best_rid = rid
                best_root = root
    if not best_rid or not best_root:
        raise BadRequest()
    rel = os.path.relpath(abs_path, best_root)
    return best_rid if rel == "." else f"{best_rid}/{rel}"


def resolve_virtual_path(rel: str, roots: Mapping[str, str]) -> str:
    """Resolve a client-supplied virtual path against active roots safely."""
    root_abs, rel_inside = root_and_rel_from_virtual(rel, roots)
    joined = os.path.normpath(os.path.join(root_abs, rel_inside))
    if not (joined == root_abs or joined.startswith(root_abs.rstrip(os.sep) + os.sep)):
        raise Forbidden()
    return joined


def is_mp3(name: str) -> bool:
    return name.lower().endswith(".mp3")


def list_mp3s_under_abs(folder: str, recursive: bool = False) -> list[str]:
    """Return absolute paths of mp3 files in one absolute folder."""
    results = []
    if recursive:
        for dirpath, _, filenames in os.walk(folder):
            for filename in filenames:
                if is_mp3(filename):
                    results.append(os.path.join(dirpath, filename))
    else:
        for filename in os.listdir(folder):
            full = os.path.join(folder, filename)
            if os.path.isfile(full) and is_mp3(filename):
                results.append(full)
    return sorted(results)


def list_mp3s(folder_vpath: str, roots: Mapping[str, str], recursive: bool = False) -> list[str]:
    """Return absolute mp3 paths for one virtual folder, or all roots when empty."""
    if folder_vpath in ("", "."):
        results: list[str] = []
        for root_abs in roots.values():
            results.extend(list_mp3s_under_abs(root_abs, recursive=True))
        return sorted(results)
    return list_mp3s_under_abs(resolve_virtual_path(folder_vpath, roots), recursive)
