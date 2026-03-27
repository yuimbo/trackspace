"""Chromaprint audio fingerprinting.

Generates a content-based hash from the audio signal so that the same
recording always maps to the same key, regardless of filename or metadata.

Requires the ``fpcalc`` CLI (part of chromaprint).
Install:  brew install chromaprint   (macOS)
          apt install libchromaprint-tools  (Debian/Ubuntu)
"""

import errno
import hashlib
import logging
import os
import shutil
import subprocess
import threading
import time

log = logging.getLogger(__name__)


def _fpcalc_slot_count() -> int:
    """Max concurrent ``fpcalc`` subprocesses (each adds pipe FDs; fork may hit EMFILE)."""
    raw = os.environ.get("TRACKSPACE_FPCALC_CONCURRENCY", "").strip()
    if raw:
        try:
            return max(1, min(64, int(raw)))
        except ValueError:
            pass
    return min(8, max(2, (os.cpu_count() or 4)))


# Serialises subprocess spawns so parallel library scans do not run dozens of
# fpcalc children at once (common source of Errno 24 after dev reload / heavy I/O).
_FPCALC_SLOTS = threading.BoundedSemaphore(_fpcalc_slot_count())

_EMFILE_RETRIES = 3

# Resolve fpcalc binary at import time.  npm/concurrently child processes
# sometimes have a stripped PATH that doesn't include /opt/homebrew/bin.
_FPCALC: str | None = (
    shutil.which("fpcalc")
    or shutil.which("fpcalc", path="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin")
)
if _FPCALC:
    log.info("fpcalc found at %s", _FPCALC)
else:
    log.warning("fpcalc not found — audio fingerprinting will be disabled")

_warned_once = False


def compute_fingerprint(path: str, duration: int = 120) -> str | None:
    """Return a hex SHA-256 of the chromaprint fingerprint for *path*.

    *duration* caps how many seconds of audio fpcalc analyses (default 120).
    Returns ``None`` if fpcalc is missing or the file can't be decoded.
    """
    global _warned_once
    if not _FPCALC:
        if not _warned_once:
            log.warning("Skipping fingerprint — fpcalc not found")
            _warned_once = True
        return None
    for attempt in range(_EMFILE_RETRIES):
        try:
            with _FPCALC_SLOTS:
                proc = subprocess.run(
                    [_FPCALC, "-raw", "-length", str(duration), path],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    stdin=subprocess.DEVNULL,
                )
            if proc.returncode != 0:
                log.debug(
                    "fpcalc failed (rc=%d) for %s: %s",
                    proc.returncode,
                    path,
                    proc.stderr.strip(),
                )
                return None
            fp_line = ""
            for line in proc.stdout.splitlines():
                if line.startswith("FINGERPRINT="):
                    fp_line = line[len("FINGERPRINT="):]
                    break
            if not fp_line:
                log.debug("fpcalc produced no FINGERPRINT for %s", path)
                return None
            return hashlib.sha256(fp_line.encode()).hexdigest()
        except subprocess.TimeoutExpired:
            log.warning("fpcalc timed out for %s", path)
            return None
        except OSError as e:
            if getattr(e, "errno", None) == errno.EMFILE and attempt + 1 < _EMFILE_RETRIES:
                time.sleep(0.08 * (attempt + 1))
                continue
            log.warning("fpcalc OSError for %s: %s", path, e)
            return None
