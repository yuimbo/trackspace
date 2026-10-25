"""Offline genre-benchmark cohort builder.

Synthetic only: dummy mp3 files (never decoded), a hand-built SQLite
``track_cache``, and a monkeypatched ``librosa_get_duration``.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from pathlib import Path

import pytest

from backend.experiments.genre_space import dataset


def _track(root: Path, rel: str, mtime: float, data: dict) -> tuple[str, float, str]:
    """Write a dummy mp3 and return a matching (path, mtime, data) cache row."""
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"dummy-mp3")
    os.utime(p, (mtime, mtime))
    st = os.stat(p)  # cache mtime must equal exactly what stat reports
    return str(p), st.st_mtime, json.dumps(data)


def _make_db(path: Path, rows: list) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE IF NOT EXISTS track_cache "
                 "(path TEXT NOT NULL, mtime REAL NOT NULL, data TEXT NOT NULL, "
                 "PRIMARY KEY (path, mtime))")
    conn.executemany(
        "INSERT OR REPLACE INTO track_cache (path, mtime, data) VALUES (?,?,?)", rows
    )
    conn.commit()
    conn.close()


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "genres"
    root.mkdir()
    db = tmp_path / "cache.db"
    rows: list = []
    state = {"root": root, "db": db, "rows": rows}

    def add(rel: str, mtime: float = 1000.0, data: dict | None = None) -> str:
        data = data or {"artist": "X", "title": rel, "fingerprint": "fp-" + rel}
        rows.append(_track(root, rel, mtime, data))
        return str(root / rel)

    def durations(mapping: dict[str, float]) -> None:
        monkeypatch.setattr(
            dataset,
            "librosa_get_duration",
            lambda path: mapping.get(
                Path(path).relative_to(root).as_posix(), 240.0
            ),
        )

    def write_db() -> None:
        _make_db(db, rows)

    def build(**kw) -> dict:
        write_db()
        return dataset.create_cohort(root, db, **kw)

    state["add"] = add
    state["durations"] = durations
    state["write_db"] = write_db
    state["build"] = build
    return state


def _defaults(data: dict) -> dict:
    """Fill artist/title/fingerprint defaults like ``env.add`` does."""
    return {"artist": "X", "title": "", "fingerprint": "fp", **data}


# ---------------------------------------------------------------------------
# Manifest shape, windows, labels
# ---------------------------------------------------------------------------


def test_basic_manifest_and_window_policy(env):
    env["add"]("Rock/AlbumA/01.mp3", data=_defaults({"artist": "The Rocks", "fingerprint": "fp1"}))
    env["add"]("Rock/AlbumA/02.mp3", data=_defaults({"artist": "The Rocks", "fingerprint": "fp2"}))
    env["add"]("Jazz/AlbumB/01.mp3", data=_defaults({"artist": "Miles", "fingerprint": "fp3"}))
    env["durations"]({"Rock/AlbumA/01.mp3": 25.0, "Rock/AlbumA/02.mp3": 40.0, "Jazz/AlbumB/01.mp3": 240.0})
    cohort = env["build"](limit=10)

    assert cohort["schema_version"] == 1
    assert cohort["root"] == str(env["root"].resolve())
    assert cohort["seed"] == 42 and cohort["label_depth"] == 1
    assert len(cohort["dataset_id"]) == 64
    assert cohort["dataset_id"] == dataset.dataset_id(cohort)
    assert dataset.validate_manifest(cohort) == []
    assert "excerpt_policy" in cohort
    assert {t["label"] for t in cohort["tracks"]} == {"Rock", "Jazz"}

    by_rel = {t["relative_path"]: t for t in cohort["tracks"]}
    assert by_rel["Rock/AlbumA/01.mp3"]["artist"] == "the rocks"  # normalized
    # <=30s: one whole-track window
    assert by_rel["Rock/AlbumA/01.mp3"]["windows"] == [{"start": 0.0, "duration": 25.0}]
    # 30-45s: one centred 30s window (not production's start-truncation)
    assert by_rel["Rock/AlbumA/02.mp3"]["windows"] == [{"start": 5.0, "duration": 30.0}]
    # >45s: three 30s windows centred at .2/.5/.8
    w3 = by_rel["Jazz/AlbumB/01.mp3"]["windows"]
    assert len(w3) == 3
    assert [w["start"] for w in w3] == [
        pytest.approx(0.2 * 240 - 15),
        pytest.approx(0.5 * 240 - 15),
        pytest.approx(0.8 * 240 - 15),
    ]
    assert all(w["duration"] == 30.0 for w in w3)
    assert by_rel["Jazz/AlbumB/01.mp3"]["duration_seconds"] == 240.0


def test_label_depth_uses_relative_parent_components(env):
    env["add"]("Rock/AlbumA/01.mp3", data=_defaults({"fingerprint": "fp1"}))
    env["add"]("Rock/AlbumA/02.mp3", data=_defaults({"fingerprint": "fp2"}))
    env["durations"]({})
    cohort = env["build"](label_depth=2)
    assert {t["label"] for t in cohort["tracks"]} == {"Rock/AlbumA"}


def test_file_directly_in_root_has_no_label_and_is_skipped(env):
    env["add"]("loose.mp3", data=_defaults({"fingerprint": "fp1"}))
    env["add"]("Rock/01.mp3", data=_defaults({"fingerprint": "fp2"}))
    env["durations"]({})
    cohort = env["build"]()
    assert [t["relative_path"] for t in cohort["tracks"]] == ["Rock/01.mp3"]
    assert cohort["diagnostics"]["skipped_no_label"] == 1


# ---------------------------------------------------------------------------
# Cache / file filtering
# ---------------------------------------------------------------------------


def test_mtime_mismatch_and_non_mp3_excluded(env, tmp_path):
    env["add"]("Rock/01.mp3", data=_defaults({"fingerprint": "fp1"}))
    # stale cache row: file on disk but with a different mtime
    stale = tmp_path / "genres" / "Rock" / "02.mp3"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"x")
    os.utime(stale, (2000.0, 2000.0))
    env["rows"].append((str(stale), 1000.0, json.dumps(_defaults({"fingerprint": "fp2"}))))
    # non-mp3 file with a cache row
    txt = tmp_path / "genres" / "Rock" / "notes.txt"
    txt.write_bytes(b"n")
    env["rows"].append((str(txt), os.stat(txt).st_mtime, json.dumps(_defaults({"fingerprint": "fp3"}))))
    env["durations"]({})
    cohort = env["build"]()

    assert [t["relative_path"] for t in cohort["tracks"]] == ["Rock/01.mp3"]
    diag = cohort["diagnostics"]
    assert diag["skipped_mtime_mismatch"] == 1
    assert diag["skipped_not_mp3"] == 1


def test_cache_row_for_missing_file_excluded(env, tmp_path):
    env["add"]("Rock/01.mp3", data=_defaults({"fingerprint": "fp1"}))
    missing = tmp_path / "genres" / "Rock" / "gone.mp3"
    env["rows"].append((str(missing), 1000.0, json.dumps(_defaults({"fingerprint": "fp2"}))))
    env["durations"]({})
    cohort = env["build"]()
    assert [t["relative_path"] for t in cohort["tracks"]] == ["Rock/01.mp3"]
    assert cohort["diagnostics"]["skipped_not_on_disk"] == 1


def test_row_outside_root_excluded(env, tmp_path):
    env["add"]("Rock/01.mp3", data=_defaults({"fingerprint": "fp1"}))
    outside = tmp_path / "elsewhere" / "x.mp3"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_bytes(b"x")
    env["rows"].append((str(outside), os.stat(outside).st_mtime, json.dumps(_defaults({"fingerprint": "fp2"}))))
    env["durations"]({})
    cohort = env["build"]()
    assert [t["relative_path"] for t in cohort["tracks"]] == ["Rock/01.mp3"]
    assert cohort["diagnostics"]["skipped_outside_root"] == 1


# ---------------------------------------------------------------------------
# Fingerprint dedup
# ---------------------------------------------------------------------------


def test_fingerprint_conflict_excludes_whole_fp(env):
    env["add"]("Rock/x.mp3", data=_defaults({"fingerprint": "fpX"}))
    env["add"]("Jazz/y.mp3", data=_defaults({"fingerprint": "fpX"}))
    env["add"]("Rock/ok.mp3", data=_defaults({"fingerprint": "fpOK"}))
    env["durations"]({})
    cohort = env["build"]()

    assert [t["relative_path"] for t in cohort["tracks"]] == ["Rock/ok.mp3"]
    diag = cohort["diagnostics"]
    assert diag["fingerprint_conflicts"] == 1
    assert len(diag["fingerprint_conflict_paths"]) == 2


def test_same_label_duplicate_keeps_lexical_min_path(env):
    env["add"]("Rock/b.mp3", data=_defaults({"fingerprint": "fpX"}))
    env["add"]("Rock/a.mp3", data=_defaults({"fingerprint": "fpX"}))
    env["durations"]({})
    cohort = env["build"]()

    assert [t["relative_path"] for t in cohort["tracks"]] == ["Rock/a.mp3"]
    assert cohort["diagnostics"]["fp_duplicates_dropped"] == 1


# ---------------------------------------------------------------------------
# Identity grouping and splits
# ---------------------------------------------------------------------------


def test_artist_group_stays_in_one_split(env):
    env["add"]("Rock/01.mp3", data=_defaults({"artist": "The Band", "fingerprint": "fp1"}))
    env["add"]("Jazz/02.mp3", data=_defaults({"artist": "The Band", "fingerprint": "fp2"}))
    env["add"]("Pop/03.mp3", data=_defaults({"artist": "Solo", "fingerprint": "fp3"}))
    env["durations"]({})
    cohort = env["build"]()

    band = [t for t in cohort["tracks"] if t["artist"] == "the band"]
    assert len(band) == 2
    assert band[0]["group"] == band[1]["group"]
    assert band[0]["split"] == band[1]["split"]


def test_release_pair_identity_requires_artist_and_album(env):
    # Same album name but different artists must NOT merge...
    env["add"]("Rock/01.mp3", data=_defaults({"artist": "Singer A", "album": "Same Album", "fingerprint": "fp1"}))
    env["add"]("Jazz/02.mp3", data=_defaults({"artist": "Singer B", "album": "Same Album", "fingerprint": "fp2"}))
    # ...while the same artist+album pair must merge.
    env["add"]("Rock/03.mp3", data=_defaults({"artist": "Singer A", "album": "Same Album", "fingerprint": "fp3"}))
    # Artist known but album missing: groups by artist alone, no release identity.
    env["add"]("Pop/04.mp3", data=_defaults({"artist": "Singer A", "fingerprint": "fp4"}))
    env["durations"]({})
    cohort = env["build"]()

    by_fp = {t["fp"]: t for t in cohort["tracks"]}
    assert by_fp["fp1"]["group"] != by_fp["fp2"]["group"]  # album name collision avoided
    assert by_fp["fp1"]["group"] == by_fp["fp3"]["group"]  # artist+album pair merges
    assert by_fp["fp4"]["group"] == by_fp["fp1"]["group"]  # shared artist still merges
    assert by_fp["fp1"]["release"] == "singer a::same album"


def test_unknown_artist_and_album_get_unique_groups(env):
    env["add"]("Rock/01.mp3", data=_defaults({"artist": "", "fingerprint": "fp1"}))
    env["add"]("Rock/02.mp3", data=_defaults({"artist": "", "fingerprint": "fp2"}))
    env["durations"]({})
    cohort = env["build"]()

    tracks = cohort["tracks"]
    assert len(tracks) == 2
    assert tracks[0]["group"] != tracks[1]["group"]
    assert cohort["diagnostics"]["missing_metadata_artist"] == 2


def test_splits_are_deterministic_and_reproducible(env):
    for i in range(8):
        env["add"](f"Rock/a{i}.mp3", data=_defaults({"artist": f"A{i}", "fingerprint": f"fp{i}"}))
    env["durations"]({})
    first = env["build"](seed=7)
    second = env["build"](seed=7)

    assert first == second
    assert first["dataset_id"] == second["dataset_id"]


def test_limit_and_balanced_round_robin(env):
    # 8 groups -> each label spans both splits; limit 4 picks one track per
    # (split, label) bucket.
    for i in range(4):
        env["add"](f"Rock/a{i}.mp3", data=_defaults({"artist": f"RA{i}", "fingerprint": f"r{i}"}))
        env["add"](f"Jazz/b{i}.mp3", data=_defaults({"artist": f"JA{i}", "fingerprint": f"j{i}"}))
    env["durations"]({})
    cohort = env["build"](limit=4)

    assert len(cohort["tracks"]) == 4
    counts = {label: 0 for label in ("Rock", "Jazz")}
    for t in cohort["tracks"]:
        counts[t["label"]] += 1
    assert counts == {"Rock": 2, "Jazz": 2}
    assert {t["split"] for t in cohort["tracks"]} == {"dev", "heldout"}


# ---------------------------------------------------------------------------
# Excerpts: lazy probes, invalid durations
# ---------------------------------------------------------------------------


def test_duration_probed_lazily_on_selected_tracks_only(env):
    for i in range(6):
        env["add"](f"Rock/a{i}.mp3", data=_defaults({"fingerprint": f"fp{i}"}))
    probed: list[str] = []

    def counting_duration(path):
        rel = Path(path).relative_to(env["root"]).as_posix()
        probed.append(rel)
        return 240.0

    import backend.experiments.genre_space.dataset as ds

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ds, "librosa_get_duration", counting_duration)
    try:
        cohort = env["build"](limit=4)
    finally:
        monkeypatch.undo()

    assert len(cohort["tracks"]) == 4
    assert len(probed) == 4
    assert sorted(probed) == sorted(t["relative_path"] for t in cohort["tracks"])


def test_invalid_duration_track_excluded_not_abort(env):
    env["add"]("Rock/01.mp3", data=_defaults({"fingerprint": "fp1"}))
    env["add"]("Rock/02.mp3", data=_defaults({"fingerprint": "fp2"}))
    env["add"]("Rock/03.mp3", data=_defaults({"fingerprint": "fp3"}))
    env["durations"]({"Rock/01.mp3": float("nan")})
    cohort = env["build"]()

    assert [t["relative_path"] for t in cohort["tracks"]] == ["Rock/02.mp3", "Rock/03.mp3"]
    assert cohort["diagnostics"]["duration_probe_failed"] == 1
    assert cohort["diagnostics"]["duration_probe_failed_paths"] == ["Rock/01.mp3"]
    assert all(1 <= len(t["windows"]) <= 3 for t in cohort["tracks"])
    assert dataset.validate_manifest(cohort) == []


# ---------------------------------------------------------------------------
# dataset_id / validate_manifest
# ---------------------------------------------------------------------------


def test_dataset_id_excludes_itself_but_hashes_labels(env):
    env["add"]("Rock/01.mp3", data=_defaults({"fingerprint": "fp1"}))
    env["durations"]({})
    cohort = env["build"]()

    assert dataset.dataset_id(cohort) == cohort["dataset_id"]
    assert dataset.dataset_id({k: v for k, v in cohort.items() if k != "dataset_id"}) == cohort["dataset_id"]

    edited = json.loads(json.dumps(cohort))
    edited["tracks"][0]["label"] = "Edited"
    assert dataset.dataset_id(edited) != cohort["dataset_id"]
    edited["dataset_id"] = dataset.dataset_id(edited)
    assert dataset.validate_manifest(edited) == []


def test_validate_manifest_flags_problems(env):
    env["add"]("Rock/01.mp3", data=_defaults({"artist": "A", "fingerprint": "fp1"}))
    env["add"]("Rock/02.mp3", data=_defaults({"artist": "B", "fingerprint": "fp2"}))
    env["add"]("Rock/03.mp3", data=_defaults({"artist": "A", "fingerprint": "fp3"}))
    env["durations"]({})
    cohort = env["build"]()
    assert dataset.validate_manifest(cohort) == []
    t0 = cohort["tracks"][0]

    def mutate(**changes) -> list[str]:
        m = json.loads(json.dumps(cohort))
        m["tracks"] = [dict(t, **changes) for t in m["tracks"]]
        return dataset.validate_manifest(m)

    # Structural field checks.
    assert any("duplicate path" in p for p in mutate(path=t0["path"]))
    assert any("path must be an absolute str" in p for p in mutate(path="Rock/01.mp3"))
    assert any("label must be a non-empty str" in p for p in mutate(label=""))
    assert any("split 'train'" in p for p in mutate(split="train"))
    assert any("mtime_ns must be an int" in p for p in mutate(mtime_ns=1.5))
    assert any("size must be a non-negative int" in p for p in mutate(size="12"))
    m = json.loads(json.dumps(cohort))
    m["tracks"] = ["garbage", m["tracks"][1], m["tracks"][2]]
    assert any("not a dict" in p for p in dataset.validate_manifest(m))

    # Windows must be 1..3 valid entries; empty is not silently accepted.
    assert any("windows must be a list of 1..3" in p for p in mutate(windows=[]))
    assert any("window duration invalid" in p for p in mutate(windows=[{"start": 0.0, "duration": 31.0}]))
    assert any("window duration invalid" in p for p in mutate(windows=[{"start": 0.0, "duration": float("nan")}]))
    assert any("window start invalid" in p for p in mutate(windows=[{"start": -1.0, "duration": 10.0}]))

    # Fingerprint uniqueness and identity consistency.
    m = json.loads(json.dumps(cohort))
    m["tracks"][1]["fp"] = t0["fp"]  # different artist -> conflicting identity
    probs = dataset.validate_manifest(m)
    assert any("duplicate fingerprint" in p for p in probs)
    assert any("conflicting artist/release identity" in p for p in probs)
    m = json.loads(json.dumps(cohort))
    m["tracks"][2]["fp"] = m["tracks"][0]["fp"]  # same artist -> duplicate only
    probs = dataset.validate_manifest(m)
    assert any("duplicate fingerprint" in p for p in probs)
    assert not any("conflicting artist/release identity" in p for p in probs)

    # Same group id but different splits.
    m = json.loads(json.dumps(cohort))
    m["tracks"][0]["group"] = m["tracks"][1]["group"]
    m["tracks"][0]["split"] = "heldout" if m["tracks"][1]["split"] == "dev" else "dev"
    assert any("spans splits" in p for p in dataset.validate_manifest(m))

    # Identity invariant survives forged (unique) group ids: a shared artist
    # must never cross splits.
    m = json.loads(json.dumps(cohort))
    pair = [t for t in m["tracks"] if t["artist"] == "a"]
    assert len(pair) == 2
    pair[0]["group"] = "FORGED0"
    pair[1]["group"] = "FORGED1"
    pair[0]["split"] = "heldout" if pair[1]["split"] == "dev" else "dev"
    assert any("identity group spans splits" in p for p in dataset.validate_manifest(m))


@pytest.mark.parametrize(
    "field, value, expected",
    [
        ("path", ["not", "a", "str"], "path must be an absolute str"),
        ("path", 42, "path must be an absolute str"),
        ("fp", ["x"], "fp must be a non-empty str"),
        ("fp", 5, "fp must be a non-empty str"),
        ("group", ["G"], "group must be a str"),
        ("artist", ["A"], "artist must be a str"),
        ("release", ["R"], "release must be a str"),
        ("mtime_ns", True, "mtime_ns must be an int"),
        ("size", True, "size must be a non-negative int"),
        ("size", -1, "size must be a non-negative int"),
        ("windows", [], "windows must be a list of 1..3"),
        ("windows", "x", "windows must be a list of 1..3"),
        ("windows", [{"start": 0.0, "duration": 0.0}], "window duration invalid"),
        ("windows", [{"start": float("nan"), "duration": 10.0}], "window start invalid"),
        ("split", None, "split None not in"),
    ],
)
def test_validate_manifest_malformed_field_returns_problem_not_crash(env, field, value, expected):
    env["add"]("Rock/01.mp3", data=_defaults({"artist": "A", "fingerprint": "fp1"}))
    env["add"]("Rock/02.mp3", data=_defaults({"artist": "B", "fingerprint": "fp2"}))
    env["add"]("Rock/03.mp3", data=_defaults({"artist": "A", "fingerprint": "fp3"}))
    env["durations"]({})
    cohort = env["build"]()
    m = json.loads(json.dumps(cohort))
    m["tracks"][0][field] = value
    problems = dataset.validate_manifest(m)  # must not raise on unhashable/mixed values
    assert any(expected in p for p in problems)


def test_validate_manifest_malformed_top_level(env):
    assert dataset.validate_manifest(["not", "a", "dict"]) == ["manifest must be a dict"]
    env["add"]("Rock/01.mp3", data=_defaults({"fingerprint": "fp1"}))
    env["durations"]({})
    cohort = env["build"]()
    for bad_root in ({"a": 1}, ""):
        m = json.loads(json.dumps(cohort))
        m["root"] = bad_root
        assert any("root must be a non-empty str" in p for p in dataset.validate_manifest(m))
    m = json.loads(json.dumps(cohort))
    m["tracks"] = "nope"
    assert any("tracks must be a list" in p for p in dataset.validate_manifest(m))


def test_cache_blob_non_dict_skipped_not_crash(env, tmp_path):
    env["add"]("Rock/01.mp3", data=_defaults({"fingerprint": "fp1"}))
    p = tmp_path / "genres" / "Rock" / "listblob.mp3"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    os.utime(p, (1000.0, 1000.0))
    st = os.stat(p)
    env["rows"].append((str(p), st.st_mtime, json.dumps(["not", "a", "dict"])))
    env["durations"]({})
    cohort = env["build"]()

    assert [t["relative_path"] for t in cohort["tracks"]] == ["Rock/01.mp3"]
    assert cohort["diagnostics"]["skipped_bad_cache_blob"] == 1


def test_default_limit_is_500(env):
    # Approved API default is 500; the 20-track smoke passes --limit 20 at
    # runtime instead of relying on the default.
    for i in range(600):
        env["add"](f"Rock/a{i:03d}.mp3", data=_defaults({"artist": f"A{i}", "fingerprint": f"fp{i}"}))
    env["durations"]({})
    cohort = env["build"]()
    assert len(cohort["tracks"]) == 500


def test_limit_20_spans_both_splits(env):
    # 4 labels x 8 artist-groups: the ~70/30 split puts every label in both
    # splits, so the default 20-track sample keeps both represented and each
    # label contributes ~5 tracks.
    for i in range(8):
        for label in ("Rock", "Jazz", "Pop", "Metal"):
            env["add"](f"{label}/a{i}.mp3",
                       data=_defaults({"artist": f"{label}A{i}", "fingerprint": f"{label}-{i}"}))
    env["durations"]({})
    cohort = env["build"](limit=20)

    assert len(cohort["tracks"]) == 20
    assert {t["split"] for t in cohort["tracks"]} == {"dev", "heldout"}
    diag = cohort["diagnostics"]["selected"]
    assert diag["dev"] + diag["heldout"] == 20
    for label in ("Rock", "Jazz", "Pop", "Metal"):
        assert sum(1 for t in cohort["tracks"] if t["label"] == label) == 5


# ---------------------------------------------------------------------------
# Read-only guarantee
# ---------------------------------------------------------------------------


def test_cohort_creation_writes_nothing(env):
    env["add"]("Rock/01.mp3", data=_defaults({"fingerprint": "fp1"}))
    env["add"]("Rock/02.mp3", data=_defaults({"fingerprint": "fp2"}))
    env["durations"]({})
    db = env["db"]
    env["write_db"]()
    db_mtime_before = db.stat().st_mtime_ns
    mp3_mtimes_before = {p: p.stat().st_mtime_ns for p in env["root"].rglob("*.mp3")}
    files_before = {str(p) for p in env["root"].rglob("*")}

    dataset.create_cohort(env["root"], db)

    assert not (db.parent / (db.name + "-wal")).exists()
    assert not (db.parent / (db.name + "-shm")).exists()
    assert db.stat().st_mtime_ns == db_mtime_before
    assert {p: p.stat().st_mtime_ns for p in env["root"].rglob("*.mp3")} == mp3_mtimes_before
    assert {str(p) for p in env["root"].rglob("*")} == files_before


def test_insufficient_split_diagnostic(env):
    # 10 Rock artist-groups + 1 Solo group: the ~70/30 split leaves >=2 of the
    # Rock groups in each split (so Rock is never flagged) while Solo's single
    # group can only ever land in one split.
    for i in range(10):
        env["add"](f"Rock/a{i}.mp3", data=_defaults({"artist": f"A{i}", "fingerprint": f"r{i}"}))
    env["add"]("Solo/01.mp3", data=_defaults({"artist": "Only", "fingerprint": "s0"}))
    env["durations"]({})
    cohort = env["build"]()

    assert "Solo" in cohort["diagnostics"]["labels_insufficient"]
    assert "Rock" not in cohort["diagnostics"]["labels_insufficient"]
    labels = cohort["diagnostics"]["labels"]
    assert sum(labels[l][s] for l in labels for s in ("dev", "heldout")) == len(cohort["tracks"])
