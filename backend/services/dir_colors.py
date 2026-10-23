"""Folder swatch colours for the server-rendered folder tree.

Mirrors ``frontend/src/lib/colors.ts`` (``dirColor``) exactly: the sidebar partial
paints swatches server-side while the canvas paints the same tracks client-side, and
they must agree. Both test suites assert the same golden vectors in
``frontend/src/lib/dir-color-vectors.json``.

A folder is painted as a tint of its *anchor* — the nearest ancestor whose hue it
shares. The default anchor is structural: a root or top-level folder is its own
anchor, and a deeper folder inherits its top-level ancestor's hue. ``anchors``
overrides that per folder with the semantic anchor from
``backend/services/folder_anchors.py`` (a sub-genre anchors to its genre; an
unrelated folder is its own anchor), so ``GENRES/<genre>/<subgenre>`` reads as one
family while uncorrelated folders stay distinct. With no override the result is
byte-identical to the structural rule, which keeps the golden vectors valid.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

_HASH_MASK = 0x3FFFF


def _hash_path(s: str, mult: int, seed: int) -> int:
    """Original 31-rolling path hash; *mult*/*seed* salt the offset channels."""
    h = seed & _HASH_MASK
    for ch in s:
        h = (h * mult + ord(ch)) & _HASH_MASK
    return h


def _signed_unit(h: int) -> float:
    """Deterministic value in [-1, 1] from a hash."""
    return (h % 2001) / 1000 - 1


def dir_color(folder_path: str, anchors: Mapping[str, str] | None = None) -> str:
    if not folder_path:
        return "hsl(350,60%,55%)"
    segments = folder_path.split("/")
    depth = len(segments) - 1
    anchor = anchors.get(folder_path) if anchors else None
    if not anchor:
        anchor = folder_path if depth <= 1 else f"{segments[0]}/{segments[1]}"
    below = depth - anchor.count("/")
    if below <= 0:
        return f"hsl({_hash_path(anchor, 31, 0) % 360},65%,60%)"

    hue = _hash_path(anchor, 31, 0) % 360
    # One level below the anchor is the widest tint; each further level halves the
    # deviation. The amplitude is deliberately wide (lightness ±30, saturation ±18
    # one level below the anchor) so sibling subgenres stay separable on the map.
    f = 0.5 ** (below - 1)
    dl = _signed_unit(_hash_path(folder_path, 7, 11)) * 30 * f
    ds = _signed_unit(_hash_path(folder_path, 13, 23)) * 18 * f
    light = math.floor(60 + dl + 0.5)
    sat = math.floor(65 + ds + 0.5)
    return f"hsl({hue},{sat}%,{light}%)"
