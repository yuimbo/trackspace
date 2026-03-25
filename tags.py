"""Read / write trackspace tags stored as TXXX private frames in ID3v2.

Each tag is stored as:
    TXXX:trackspace:<tagname>  →  float string "0.0" .. "1.0"

Only tags with an assigned value are stored; removing a value deletes the frame.
"""

import os
from mutagen.id3 import ID3, TXXX, ID3NoHeaderError

TAG_PREFIX = "trackspace:"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _txxx_key(tagname: str) -> str:
    """Return the TXXX description used inside ID3 for a given tag name."""
    return f"{TAG_PREFIX}{tagname}"


def _open_id3(path: str) -> ID3:
    """Open (or create) an ID3 tag object for *path*."""
    try:
        return ID3(path)
    except ID3NoHeaderError:
        tag = ID3()
        tag.save(path)
        return ID3(path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def read_tags(path: str) -> dict[str, float]:
    """Return ``{tagname: value}`` for every trackspace tag on *path*."""
    try:
        id3 = ID3(path)
    except Exception:
        return {}
    result = {}
    for key, frame in id3.items():
        if isinstance(frame, TXXX) and frame.desc.startswith(TAG_PREFIX):
            tagname = frame.desc[len(TAG_PREFIX):]
            try:
                result[tagname] = float(frame.text[0])
            except (IndexError, ValueError):
                pass
    return result


def write_tag(path: str, tagname: str, value: float) -> None:
    """Set a single trackspace tag to *value* (clamped 0-1)."""
    value = max(0.0, min(1.0, float(value)))
    id3 = _open_id3(path)
    desc = _txxx_key(tagname)
    id3.delall("TXXX:" + desc)
    id3.add(TXXX(encoding=3, desc=desc, text=[str(value)]))
    id3.save(path)


def delete_tag(path: str, tagname: str) -> None:
    """Remove a trackspace tag from *path* if it exists."""
    try:
        id3 = ID3(path)
    except Exception:
        return
    desc = _txxx_key(tagname)
    id3.delall("TXXX:" + desc)
    id3.save(path)


def rename_tag(path: str, old_name: str, new_name: str) -> bool:
    """Rename a trackspace tag on *path*.  Returns True if the tag existed."""
    try:
        id3 = ID3(path)
    except Exception:
        return False
    old_desc = _txxx_key(old_name)
    value = None
    for frame in id3.values():
        if isinstance(frame, TXXX) and frame.desc == old_desc:
            value = frame.text[0]
            break
    if value is None:
        return False
    id3.delall("TXXX:" + old_desc)
    id3.add(TXXX(encoding=3, desc=_txxx_key(new_name), text=[value]))
    id3.save(path)
    return True


def read_metadata(path: str) -> dict[str, str]:
    """Return standard ID3 metadata keys: 'title', 'artist' (when present)."""
    try:
        id3 = ID3(path)
    except Exception:
        return {}
    meta: dict[str, str] = {}
    if "TIT2" in id3:
        v = str(id3["TIT2"].text[0]).strip()
        if v:
            meta["title"] = v
    if "TPE1" in id3:
        v = str(id3["TPE1"].text[0]).strip()
        if v:
            meta["artist"] = v
    return meta


def batch_read(paths: list[str]) -> dict[str, dict[str, float]]:
    """Return ``{path: {tagname: value, …}, …}`` for every path."""
    return {p: read_tags(p) for p in paths}


def collect_tag_names(paths: list[str]) -> list[str]:
    """Return sorted unique tag names across all *paths*."""
    names: set[str] = set()
    for p in paths:
        names.update(read_tags(p).keys())
    return sorted(names)
