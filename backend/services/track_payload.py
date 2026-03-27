from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any


def track_dict_from_read(
    path: str,
    info: dict[str, Any],
    *,
    virtual_from_abs: Callable[[str], str],
    display_bpm_key: Callable[[Any], tuple[int | None, str | None]],
    get_audio_features: Callable[[str], Any | None],
    feat_by_fp: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One API track object from a filesystem path and _cached_read_all payload."""
    rel_path = virtual_from_abs(path)
    folder_rel = virtual_from_abs(os.path.dirname(path))
    fp = info.get("fingerprint")
    bpm: int | None = None
    musical_key: str | None = None
    if isinstance(fp, str) and fp:
        arr = feat_by_fp.get(fp) if feat_by_fp is not None else get_audio_features(fp)
        if arr is not None:
            bpm, musical_key = display_bpm_key(arr)
    return {
        "path": rel_path,
        "filename": os.path.basename(path),
        "folder": folder_rel,
        "tags": info.get("tags", {}),
        "artist": info.get("artist", ""),
        "title": info.get("title", ""),
        "fingerprint": info.get("fingerprint"),
        "bpm": bpm,
        "musical_key": musical_key,
    }


def build_track_list(
    paths: list[str],
    results: dict[str, dict[str, Any]],
    *,
    track_dict_builder: Callable[..., dict[str, Any]],
    get_all_audio_features: Callable[[list[str]], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert path->data mapping into the track dicts the frontend expects."""
    fps_ordered: list[str] = []
    seen: set[str] = set()
    for path in paths:
        fp = results.get(path, {}).get("fingerprint")
        if not isinstance(fp, str) or not fp or fp in seen:
            continue
        seen.add(fp)
        fps_ordered.append(fp)

    feat_map = get_all_audio_features(fps_ordered) if fps_ordered else {}
    return [track_dict_builder(path, results.get(path, {}), feat_by_fp=feat_map) for path in paths]
