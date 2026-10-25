"""Frozen genre-benchmark cohorts, built offline from the SQLite track cache.

``create_cohort`` selects a deterministic, balanced subset of already-analysed
tracks from one music root, groups them by artist/release identity, and splits
them into ``dev``/``heldout`` so no identity crosses a split. Read-only: the
cache DB is opened in ``mode=ro`` (never through :class:`TrackCache`, whose
connection runs the WAL write pragma), no ID3 tag is read or written, and
nothing is re-fingerprinted. All metadata comes from the cached JSON blob;
missing metadata is reported, not fetched.

Excerpts are not decoded here. Each selected track gets one lazy
``librosa_get_duration`` probe so the manifest can carry exact 30 s window
positions; a track whose probe is invalid is excluded and reported in
diagnostics (never aborted). The extractor must re-stat the frozen file before
decoding and surface any change rather than silently re-reading.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sqlite3
import stat
from collections import defaultdict
from pathlib import Path

from backend.sqlite_conn import ClosingConnection

SCHEMA_VERSION = 1
EXCERPT_SECONDS = 30.0
#: Mirror of backend.embeddings.audio_decode.SEGMENT_FRACTIONS, kept local so
#: importing this module never pulls in librosa (see excerpt_policy).
EXCERPT_FRACTIONS = (0.2, 0.5, 0.8)
SHORT_MAX_SECONDS = EXCERPT_SECONDS * 1.5  # 45 s
DEV_FRACTION = 0.7
SPLITS = ("dev", "heldout")

EXCERPT_POLICY = (
    "30s excerpt windows: duration<=30 -> one whole-track window; "
    "30<duration<=45 -> one centred 30s window (production truncates from the "
    "start; cohorts centre instead); duration>45 -> three 30s windows centred "
    "at 0.2/0.5/0.8, clamped to [0, duration-30]. Durations come from one lazy "
    "librosa_get_duration probe per selected track; a track whose probe is "
    "invalid is excluded from the manifest and reported in diagnostics rather "
    "than aborting the cohort."
)


def librosa_get_duration(**kwargs):
    """Duration probe with a lazy import so CLI --help never loads librosa."""
    from backend.decode_stderr import librosa_get_duration as _real

    return _real(**kwargs)


class _DisjointSet:
    """Union-find over hashable keys (path-compressed)."""

    __slots__ = ("_parent",)

    def __init__(self) -> None:
        self._parent: dict = {}

    def find(self, x):
        parent = self._parent
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def _windows_for(duration: float) -> list[dict] | None:
    """Exact window positions for *duration*; ``None`` when the duration is unusable."""
    if not math.isfinite(duration) or duration <= 0:
        return None
    if duration <= EXCERPT_SECONDS:
        return [{"start": 0.0, "duration": round(duration, 6)}]
    if duration <= SHORT_MAX_SECONDS:
        return [{"start": round((duration - EXCERPT_SECONDS) / 2, 6), "duration": EXCERPT_SECONDS}]
    out = []
    for frac in EXCERPT_FRACTIONS:
        start = min(max(duration * frac - EXCERPT_SECONDS / 2, 0.0), max(duration - EXCERPT_SECONDS, 0.0))
        out.append({"start": round(start, 6), "duration": EXCERPT_SECONDS})
    return out


def dataset_id(manifest: dict) -> str:
    """SHA-256 over the manifest's canonical JSON, excluding its own id.

    ``allow_nan=False`` and no ``default`` coercion: malformed values raise
    instead of being silently masked into the hash.
    """
    body = {k: v for k, v in manifest.items() if k != "dataset_id"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_manifest(manifest: dict) -> list[str]:
    """Return a list of problems; empty means the manifest is usable.

    Labels may be edited freely; the CLI must recompute ``dataset_id`` via
    :func:`dataset_id` and verify it — this does not re-hash. Identity checks
    are structural: no known artist or release identity may cross splits, even
    if group ids were forged.
    """
    problems = []
    if not isinstance(manifest, dict):
        return ["manifest must be a dict"]
    if not isinstance(manifest.get("root"), str) or not manifest.get("root"):
        problems.append("root must be a non-empty str")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version != {SCHEMA_VERSION}")
    did = manifest.get("dataset_id")
    if not (isinstance(did, str) and len(did) == 64 and all(c in "0123456789abcdef" for c in did)):
        problems.append("dataset_id must be a 64-char lowercase hex sha256")
    tracks = manifest.get("tracks")
    if not isinstance(tracks, list):
        return problems + ["tracks must be a list"]
    seen_fp: dict[str, tuple[str, str]] = {}
    seen_paths: set[str] = set()
    group_split: dict[str, str] = {}
    uf = _DisjointSet()

    for i, t in enumerate(tracks):
        tag = f"tracks[{i}]"
        if not isinstance(t, dict):
            problems.append(f"{tag}: not a dict")
            continue
        fp, path = t.get("fp"), t.get("path")
        fp_ok = isinstance(fp, str) and bool(fp)
        path_ok = isinstance(path, str) and os.path.isabs(path)
        if not fp_ok:
            problems.append(f"{tag}: fp must be a non-empty str")
        if not path_ok:
            problems.append(f"{tag}: path must be an absolute str")
        if path_ok:  # hashable key only; malformed values are flagged above
            if path in seen_paths:
                problems.append(f"{tag}: duplicate path {path}")
            seen_paths.add(path)
        if fp_ok:
            identity = (t.get("artist", ""), t.get("release", ""))
            if fp in seen_fp:
                problems.append(f"{tag}: duplicate fingerprint {fp}")
                if seen_fp[fp] != identity:
                    problems.append(f"{tag}: fingerprint {fp} has conflicting artist/release identity")
            seen_fp[fp] = identity
        label = t.get("label")
        if not isinstance(label, str) or not label:
            problems.append(f"{tag}: label must be a non-empty str")
        for field in ("artist", "release"):
            if t.get(field) is not None and not isinstance(t.get(field), str):
                problems.append(f"{tag}: {field} must be a str")
        if not (isinstance(t.get("mtime_ns"), int) and not isinstance(t.get("mtime_ns"), bool)):
            problems.append(f"{tag}: mtime_ns must be an int")
        size = t.get("size")
        if not (isinstance(size, int) and not isinstance(size, bool) and size >= 0):
            problems.append(f"{tag}: size must be a non-negative int")
        split = t.get("split")
        if split not in SPLITS:
            problems.append(f"{tag}: split {split!r} not in {SPLITS}")
        gid = t.get("group")
        if gid is not None:
            if not isinstance(gid, str):
                problems.append(f"{tag}: group must be a str")
            else:  # hashable key only; malformed values are flagged above
                if gid in group_split and group_split[gid] != split:
                    problems.append(f"{tag}: group {gid} spans splits")
                group_split[gid] = split
        # Identity invariants: a shared known artist or release must share a split.
        if isinstance(t.get("artist"), str) and t["artist"]:
            uf.union(i, ("A", t["artist"]))
        if isinstance(t.get("release"), str) and t["release"]:
            uf.union(i, ("R", t["release"]))
        windows = t.get("windows")
        if not isinstance(windows, list) or not 1 <= len(windows) <= 3:
            problems.append(f"{tag}: windows must be a list of 1..3 entries")
            continue
        for w in windows:
            if not isinstance(w, dict):
                problems.append(f"{tag}: window entries must be dicts")
                continue
            start, dur = w.get("start"), w.get("duration")
            if not (isinstance(start, (int, float)) and math.isfinite(start) and start >= 0):
                problems.append(f"{tag}: window start invalid {start!r}")
            if not (isinstance(dur, (int, float)) and math.isfinite(dur) and 0 < dur <= EXCERPT_SECONDS):
                problems.append(f"{tag}: window duration invalid {dur!r}")

    # No known identity may cross splits, regardless of the group-id fields.
    comps: dict[tuple, list[int]] = defaultdict(list)
    for i in range(len(tracks)):
        if isinstance(tracks[i], dict):
            comps[uf.find(i)].append(i)
    for idxs in comps.values():
        splits = {
            tracks[i].get("split")
            for i in idxs
            if isinstance(tracks[i].get("split"), str)
        }
        if len(splits) > 1:
            problems.append(
                f"identity group spans splits: tracks {sorted(idxs)} "
                f"have splits {sorted(splits)}"
            )
    return problems


def create_cohort(
    root: Path | str,
    track_db: Path | str,
    *,
    limit: int = 500,
    seed: int = 42,
    label_depth: int = 1,
) -> dict:
    """Build a frozen, labelled, dev/heldout-split cohort manifest.

    Keeps only existing ``.mp3`` regular files under *root* whose mtime still
    matches a ``track_cache`` row in *track_db* (read via ``mode=ro`` URI).
    Never writes the cache, tags, or any file under *root*.
    """
    root = Path(root).resolve()
    if label_depth < 1:
        raise ValueError("label_depth must be >= 1")
    if limit < 0:
        raise ValueError("limit must be >= 0")

    uri = Path(track_db).resolve().as_uri() + "?mode=ro"
    with ClosingConnection(sqlite3.connect(uri, uri=True)) as conn:
        rows = conn.execute("SELECT path, mtime, data FROM track_cache").fetchall()

    diag: dict = defaultdict(int)
    conflict_paths: list[str] = []
    candidates: list[dict] = []
    for path, mtime, blob in rows:
        p = Path(path)
        if not p.is_relative_to(root):
            diag["skipped_outside_root"] += 1
            continue
        try:
            st = os.stat(p)
        except OSError:
            diag["skipped_not_on_disk"] += 1
            continue
        if not stat.S_ISREG(st.st_mode):
            diag["skipped_not_regular"] += 1
            continue
        if p.suffix.lower() != ".mp3":
            diag["skipped_not_mp3"] += 1
            continue
        if st.st_mtime != mtime:
            diag["skipped_mtime_mismatch"] += 1
            continue
        try:
            data = json.loads(blob)
        except (TypeError, ValueError):
            diag["skipped_bad_cache_blob"] += 1
            continue
        if not isinstance(data, dict):
            diag["skipped_bad_cache_blob"] += 1
            continue
        fp = data.get("fingerprint")
        if not isinstance(fp, str) or not fp:
            diag["skipped_missing_fingerprint"] += 1
            continue
        rel = p.relative_to(root)
        parent_parts = rel.parent.parts
        if not parent_parts:
            diag["skipped_no_label"] += 1
            continue
        artist = str(data.get("artist") or "").strip().casefold()
        album = str(data.get("album") or data.get("release") or "").strip().casefold()
        release = f"{artist}::{album}" if artist and album else ""
        if not artist:
            diag["missing_metadata_artist"] += 1
        if not release:
            diag["missing_metadata_release"] += 1
        candidates.append({
            "path": str(p),
            "relative_path": rel.as_posix(),
            "fp": fp,
            "artist": artist,
            "release": release,
            "label": "/".join(parent_parts[:label_depth]),
            "mtime_ns": st.st_mtime_ns,
            "size": st.st_size,
        })
    diag["total_cache_rows"] = len(rows)

    # Fingerprint dedup: conflicting folder labels exclude the whole fingerprint;
    # same-label duplicates keep the lexicographically smallest path.
    by_fp: dict[str, list[dict]] = defaultdict(list)
    for t in candidates:
        by_fp[t["fp"]].append(t)
    pool: list[dict] = []
    for fp, group in by_fp.items():
        labels = {t["label"] for t in group}
        if len(labels) > 1:
            diag["fingerprint_conflicts"] += 1
            conflict_paths.extend(t["path"] for t in group)
            continue
        group.sort(key=lambda t: t["path"])
        pool.append(group[0])
        diag["fp_duplicates_dropped"] += len(group) - 1
    pool.sort(key=lambda t: (t["label"], t["path"]))

    # Union-find over artist / release identities. Release identity requires the
    # artist+album pair (``artist::album``); tracks with neither are their own
    # group (fingerprints are unique after dedup).
    uf = _DisjointSet()
    for t in pool:
        key = ("T", t["fp"])
        if t["artist"]:
            uf.union(key, ("A", t["artist"]))
        if t["release"]:
            uf.union(key, ("R", t["release"]))

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for t in pool:
        groups[uf.find(("T", t["fp"]))].append(t)

    # Seeded ~70/30 split; groups stay whole and keep their assignment.
    ordered = sorted(groups, key=lambda gid: min(t["path"] for t in groups[gid]))
    gid_names = {gid: f"G{i:04d}" for i, gid in enumerate(ordered)}
    rng = random.Random(seed)
    rng.shuffle(ordered)
    dev_n = held_n = dev_groups = held_groups = 0
    for gid in ordered:
        n = len(groups[gid])
        split = "dev" if dev_n < DEV_FRACTION * (dev_n + held_n + n) else "heldout"
        for t in groups[gid]:
            t["group"] = gid_names[gid]
            t["split"] = split
        if split == "dev":
            dev_n += n
            dev_groups += 1
        else:
            held_n += n
            held_groups += 1

    # Seeded balanced round-robin over labels, up to limit: one track per label
    # per round, each pick targeting a ~70/30 dev/heldout mix per label (seeded
    # first draw, then a 0.7 greedy) with fallback to the other split when one
    # is empty. Buckets are shuffled (seeded), so the sample is not biased to
    # lexically-first paths. Groups were already split whole; sampling never
    # moves a track across splits.
    by_label: dict[str, dict[str, list[dict]]] = defaultdict(lambda: {"dev": [], "heldout": []})
    for t in pool:
        by_label[t["label"]][t["split"]].append(t)
    sample_rng = random.Random(f"{seed}:sample")
    for buckets in by_label.values():
        for lst in buckets.values():
            lst.sort(key=lambda t: t["path"])
            sample_rng.shuffle(lst)
    labels = list(by_label)
    sample_rng.shuffle(labels)
    sel_count: dict[tuple, int] = defaultdict(int)
    selected: list[dict] = []
    while len(selected) < limit:
        progressed = False
        for label in labels:
            if len(selected) >= limit:
                break
            buckets = by_label[label]
            if not buckets["dev"] and not buckets["heldout"]:
                continue
            dev_n, held_n = sel_count[(label, "dev")], sel_count[(label, "heldout")]
            if dev_n + held_n == 0:
                split = "dev" if sample_rng.random() < DEV_FRACTION else "heldout"
            else:
                split = "dev" if dev_n < DEV_FRACTION * (dev_n + held_n + 1) else "heldout"
            if not buckets[split]:
                split = "heldout" if split == "dev" else "dev"
            if not buckets[split]:
                continue
            selected.append(buckets[split].pop(0))
            sel_count[(label, split)] += 1
            progressed = True
        if not progressed:
            break

    # Lazy duration probe on selected tracks only; exact window positions. A
    # track whose probe is invalid is excluded (reported, never aborted), so
    # every manifest track carries 1..3 windows.
    duration_failures: list[str] = []
    kept: list[dict] = []
    for t in selected:
        try:
            duration = float(librosa_get_duration(path=t["path"]))
        except Exception:
            duration = float("nan")
        windows = _windows_for(duration)
        if windows is None:
            duration_failures.append(t["relative_path"])
            continue
        t["duration_seconds"] = duration
        t["windows"] = windows
        kept.append(t)
    selected = kept
    diag["duration_probe_failed"] = len(duration_failures)
    diag["duration_probe_failed_paths"] = duration_failures

    # Per-label diagnostics over the selected cohort.
    label_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"dev": 0, "heldout": 0})
    selected_counts = {"dev": 0, "heldout": 0}
    for t in selected:
        label_counts[t["label"]][t["split"]] += 1
        selected_counts[t["split"]] += 1
    labels_insufficient = {
        label: [s for s in SPLITS if counts[s] == 0]
        for label, counts in label_counts.items()
        if any(counts[s] == 0 for s in SPLITS)
    }

    selected.sort(key=lambda t: (t["split"], t["label"], t["path"]))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "",
        "root": str(root),
        "track_db": str(Path(track_db).resolve()),
        "seed": seed,
        "label_depth": label_depth,
        "limit": limit,
        "excerpt_policy": EXCERPT_POLICY,
        "diagnostics": {
            **dict(diag),
            "fingerprint_conflict_paths": conflict_paths,
            "groups": {"total": len(groups), "dev": dev_groups, "heldout": held_groups},
            "selected": selected_counts,
            "labels": dict(label_counts),
            "labels_insufficient": labels_insufficient,
        },
        "tracks": selected,
    }
    manifest["dataset_id"] = dataset_id(manifest)
    return manifest
