"""``compute_fingerprint`` subprocess handling, without touching real audio.

The real-world case these pin: fpcalc exits 2/3 on an MP3 it can only partially
decode, but still writes a ``FINGERPRINT=`` line from the readable part. That line is
a stable content hash, so the track must stay analysable; only a file with no
fingerprint at all is a genuine failure.
"""

from __future__ import annotations

import hashlib

from backend import fingerprint


class _Proc:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _stub_run(monkeypatch, proc: _Proc) -> None:
    monkeypatch.setattr(fingerprint, "_FPCALC", "/usr/bin/fpcalc")
    monkeypatch.setattr(fingerprint.subprocess, "run", lambda *a, **k: proc)


def test_partial_decode_still_fingerprints(monkeypatch, tmp_path):
    _stub_run(
        monkeypatch,
        _Proc(3, "DURATION=100\nFINGERPRINT=1,2,3\n", "ERROR: Invalid data"),
    )
    fp = fingerprint.compute_fingerprint(str(tmp_path / "damaged.mp3"))
    assert fp == hashlib.sha256(b"1,2,3").hexdigest()


def test_no_fingerprint_line_is_none(monkeypatch, tmp_path):
    _stub_run(monkeypatch, _Proc(2, "", "ERROR: Invalid data found"))
    assert fingerprint.compute_fingerprint(str(tmp_path / "broken.mp3")) is None


def test_clean_run_fingerprints(monkeypatch, tmp_path):
    _stub_run(monkeypatch, _Proc(0, "DURATION=100\nFINGERPRINT=9,8,7\n"))
    fp = fingerprint.compute_fingerprint(str(tmp_path / "ok.mp3"))
    assert fp == hashlib.sha256(b"9,8,7").hexdigest()
