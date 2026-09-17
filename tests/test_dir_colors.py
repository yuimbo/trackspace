"""Golden-vector and property tests for folder swatch colours.

The vectors live in ``frontend/src/lib/dir-color-vectors.json`` and are asserted by
the frontend suite too (``frontend/src/lib/colors.test.ts``), so the two
implementations cannot drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.services.dir_colors import dir_color

_VECTORS_PATH = (
    Path(__file__).resolve().parents[1]
    / "frontend"
    / "src"
    / "lib"
    / "dir-color-vectors.json"
)
_VECTORS: dict[str, str] = json.loads(_VECTORS_PATH.read_text(encoding="utf-8"))


def _legacy_dir_color(folder_path: str) -> str:
    if not folder_path:
        return "hsl(350,60%,55%)"
    h = 0
    for ch in folder_path:
        h = (h * 31 + ord(ch)) & 0x3FFFF
    return f"hsl({h % 360},65%,60%)"


def _lightness(color: str) -> int:
    return int(color.split(",")[2].rstrip("%)"))


def test_matches_shared_golden_vectors() -> None:
    for path, expected in _VECTORS.items():
        assert dir_color(path) == expected, path


def test_empty_path_fallback() -> None:
    assert dir_color("") == "hsl(350,60%,55%)"


def test_roots_and_top_level_unchanged() -> None:
    for path in ["", "abc12345", "abc12345/GENRES", "def67890", "def67890/to_sort"]:
        assert dir_color(path) == _legacy_dir_color(path), path


def test_top_level_family_shares_one_hue() -> None:
    for path in [
        "abc12345/GENRES/Rock",
        "abc12345/GENRES/Rock/70s",
        "abc12345/GENRES/Rock/70s/Live",
        "abc12345/GENRES/Jazz",
    ]:
        assert dir_color(path).startswith("hsl(286,"), path


def test_lightness_deviation_halves_per_level() -> None:
    devs = [
        abs(_lightness(dir_color(p)) - 60)
        for p in [
            "abc12345/GENRES/Rock",
            "abc12345/GENRES/Rock/70s",
            "abc12345/GENRES/Rock/70s/Live",
        ]
    ]
    assert devs[0] <= 30
    assert devs[1] <= 15
    assert devs[2] <= 8


def test_sibling_subfolders_get_distinct_tints() -> None:
    assert dir_color("abc12345/GENRES/Rock") != dir_color("abc12345/GENRES/Jazz")


def test_deterministic() -> None:
    path = "abc12345/GENRES/Rock/70s"
    assert dir_color(path) == dir_color(path)
