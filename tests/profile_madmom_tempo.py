"""Profile madmom tempo estimation (cProfile).

Run from repo root::

    .venv/bin/python tests/profile_madmom_tempo.py
    TRACKSPACE_MADMOM_FAST=1 .venv/bin/python tests/profile_madmom_tempo.py
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import librosa  # noqa: E402

from backend.embeddings.madmom_tempo import estimate_tempo_bpm, warmup_madmom_tempo  # noqa: E402


def main() -> None:
    fixture = ROOT / "tests" / "fixtures" / "test_track.mp3"
    audio, sr = librosa.load(str(fixture), sr=22050, mono=True)
    max_s = 20.0
    nmax = int(sr * max_s)
    if len(audio) > nmax:
        audio = audio[:nmax]

    warmup_madmom_tempo()

    pr = cProfile.Profile()
    repeats = 5
    pr.enable()
    for _ in range(repeats):
        estimate_tempo_bpm(audio, sr)
    pr.disable()

    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(35)
    print(s.getvalue())


if __name__ == "__main__":
    main()
