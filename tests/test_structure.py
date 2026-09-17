"""Track-structure detection, its cache contract, and the structure job stage.

The detector is exercised on synthetic envelopes (no audio), plus one real decode of a
generated wav. The handler is exercised with a fake compute function so the test pins
the cache/queue contract, not the DSP.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.embeddings import structure as structure_mod
from backend.embeddings import supervised as supervised_mod
from backend.embeddings.feature_cache import FeatureCache
from backend.embeddings.structure import (
    FRAME_RATE,
    STRUCTURE_FIELDS,
    TrackStructure,
    analyze_structure,
    decode_structure,
    detect_sections,
    encode_structure,
)
from backend.jobs import (
    JOB_KINDS,
    KIND_STRUCTURE,
    Job,
    JobQueue,
    kind_label,
)
from backend.services.analysis_pipeline import (
    AnalysisDeps,
    enqueue_library_work,
    make_structure_handler,
)


def _envelope(parts: list[tuple[float, float]]) -> np.ndarray:
    """Concatenate constant-level spans into an envelope at ``FRAME_RATE``."""
    return np.concatenate([np.full(int(s * FRAME_RATE), v) for s, v in parts])


# ── registry ─────────────────────────────────────────────────


def test_structure_kind_is_registered_and_labelled():
    assert KIND_STRUCTURE in JOB_KINDS
    assert kind_label(KIND_STRUCTURE) == "Track structure"


# ── detector (pure, synthetic envelopes) ──────────────────────


def test_detects_intro_drop_and_break():
    # quiet intro → loud body → quiet break → loud outro
    env = _envelope([(30, 0.05), (60, 0.9), (20, 0.08), (90, 0.85)])
    s = detect_sections(env, FRAME_RATE)

    assert s.intro_start == pytest.approx(0, abs=2)
    assert s.intro_end == pytest.approx(30, abs=4)
    assert s.peak_start == pytest.approx(30, abs=8)
    assert s.break_start is not None and s.break_end is not None
    assert 88 <= s.break_start <= 92
    assert 106 <= s.break_end <= 112


def test_intro_start_skips_lead_in_silence():
    # 12 s of near-silence before the music: the intro starts there, not at 0.
    env = _envelope([(12, 0.0005), (40, 0.8), (20, 0.01), (40, 0.75)])
    s = detect_sections(env, FRAME_RATE)

    assert s.intro_start == pytest.approx(12, abs=3)
    assert s.intro_end is not None
    assert s.intro_start is not None and s.intro_start <= s.intro_end


def test_intro_start_is_zero_when_the_track_starts_immediately():
    s = detect_sections(np.full(100, 0.5), FRAME_RATE)
    assert s.intro_start == 0.0


def test_track_with_no_drop_has_no_break():
    # Steady energy: no intro, no interior dip, peak is the whole track.
    s = detect_sections(np.full(100, 0.5), FRAME_RATE)
    assert s.intro_start == 0.0
    assert s.intro_end == 0.0
    assert s.peak_start == 0.0
    assert s.peak_end == pytest.approx(25.0)
    assert s.break_start is None and s.break_end is None


def test_silence_has_no_sections():
    s = detect_sections(np.zeros(100), FRAME_RATE)
    assert s == TrackStructure()


def test_empty_envelope_has_no_sections():
    assert detect_sections(np.zeros(0), FRAME_RATE) == TrackStructure()


def test_trailing_fade_is_not_a_break():
    # A quiet tail touches the end, so it is an outro, not a break.
    env = _envelope([(20, 0.9), (40, 0.9), (40, 0.02)])
    s = detect_sections(env, FRAME_RATE)
    assert s.break_start is None and s.break_end is None


# ── blob round-trip ──────────────────────────────────────────


def test_blob_round_trip_preserves_values_and_absences():
    s = TrackStructure(
        intro_start=5.0,
        intro_end=12.5,
        peak_start=40.0,
        peak_end=70.0,
        break_start=None,
        break_end=None,
    )
    blob = encode_structure(s)
    assert blob.dtype == np.float32
    assert blob.shape == (len(STRUCTURE_FIELDS),)
    assert decode_structure(blob) == {
        "intro_start": 5.0,
        "intro_end": 12.5,
        "peak_start": 40.0,
        "peak_end": 70.0,
        "break_start": None,
        "break_end": None,
    }


def test_analyze_structure_decodes_a_real_wav(tmp_path):
    import soundfile as sf

    sr = 22050
    sig = np.concatenate(
        [
            np.full(int(5 * sr), 0.01, np.float32),
            np.full(int(10 * sr), 0.5, np.float32),
            np.full(int(5 * sr), 0.01, np.float32),
            np.full(int(10 * sr), 0.45, np.float32),
        ]
    )
    path = tmp_path / "tone.wav"
    sf.write(str(path), sig, sr)

    s = analyze_structure(str(path))
    assert s is not None
    assert s.peak_start is not None and s.peak_end is not None
    assert s.peak_start < s.peak_end


def test_analyze_structure_returns_none_for_missing_file(tmp_path):
    assert analyze_structure(str(tmp_path / "nope.mp3")) is None


# ── genre spanning ───────────────────────────────────────────


def test_genre_spanning_flags_divergent_excerpts():
    a = np.zeros(768, np.float32)
    a[0] = 1.0
    b = np.zeros(768, np.float32)
    b[1] = 1.0  # orthogonal to a
    assert supervised_mod.is_genre_spanning(np.stack([a, b])) is True


def test_genre_spanning_ignores_similar_excerpts():
    a = np.zeros(768, np.float32)
    a[0] = 1.0
    assert supervised_mod.is_genre_spanning(np.stack([a, a * 0.99])) is False


def test_genre_spanning_is_false_without_two_excerpts():
    assert supervised_mod.is_genre_spanning(None) is False
    assert supervised_mod.is_genre_spanning(np.zeros((1, 768), np.float32)) is False


# ── handler ─────────────────────────────────────────────────


def _deps(tmp_path, fc: FeatureCache, *, compute=None) -> AnalysisDeps:
    track = tmp_path / "track.mp3"
    track.write_bytes(b"ID3\x00fake")
    return AnalysisDeps(
        feature_cache=fc,
        track_cache=None,
        cached_read_all=lambda _p: {"fingerprint": "fp"},
        resolve_virtual_path=lambda _v: str(track),
        virtual_from_abs=lambda _a: "track.mp3",
        list_mp3s=lambda _f, _r: [str(track)],
        maest_mod=None,
        rhythm_mod=None,
        audio_features_mod=None,
        maest_version=1,
        rhythm_version=1,
        features_version=1,
        structure_version=1,
    )


def _job(job_id: int = 1) -> Job:
    return Job(
        id=job_id,
        kind=KIND_STRUCTURE,
        dedupe_key="track.mp3",
        payload={"path": "track.mp3"},
        attempts=0,
        priority=105,
    )


def test_structure_handler_writes_blob_then_serves_cache(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    calls: list[str] = []

    def compute(path: str) -> dict:
        calls.append(path)
        return {
            "structure": encode_structure(
                TrackStructure(intro_end=1.0, peak_start=2.0, peak_end=3.0)
            ),
            "error": None,
        }

    handler = make_structure_handler(_deps(tmp_path, fc), compute_fn=compute)
    out = handler([_job()])
    assert out[1].ok is True
    assert fc.has("structure", "fp", version=1)
    assert decode_structure(fc.get("structure", "fp", version=1))["peak_start"] == 2.0

    again = handler([_job(2)])
    assert again[2].ok is True
    assert again[2].detail.get("cached") is True
    assert len(calls) == 1


def test_structure_handler_reports_compute_failure(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))

    def compute(_path: str) -> dict:
        return {"structure": None, "error": "decode failed"}

    handler = make_structure_handler(_deps(tmp_path, fc), compute_fn=compute)
    out = handler([_job()])
    assert out[1].ok is False
    assert "decode failed" in (out[1].error or "")
    assert not fc.has("structure", "fp", version=1)


def test_structure_handler_missing_file_is_permanent(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    deps = _deps(tmp_path, fc)
    deps.resolve_virtual_path = lambda _v: str(tmp_path / "missing.mp3")
    handler = make_structure_handler(deps)

    out = handler([_job()])
    assert out[1].ok is False
    assert out[1].retry is False


def test_structure_handler_no_fingerprint_is_retryable(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    deps = _deps(tmp_path, fc)
    deps.cached_read_all = lambda _p: {"fingerprint": None}
    handler = make_structure_handler(deps)

    out = handler([_job()])
    assert out[1].ok is False
    assert out[1].retry is True


def test_structure_handler_uses_thread_pool(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    seen: list[str] = []

    def compute(path: str) -> dict:
        seen.append(path)
        return {"structure": encode_structure(TrackStructure(peak_start=5.0)), "error": None}

    handler = make_structure_handler(
        _deps(tmp_path, fc), pool_size=2, compute_fn=compute
    )
    out = handler([_job(1), _job(2)])
    assert all(r.ok for r in out.values())
    assert sorted(seen) == sorted([str(tmp_path / "track.mp3")] * 2)


# ── enqueue filter ──────────────────────────────────────────


def test_enqueue_structure_queues_only_missing_rows(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    fc.put("structure", "fp1", encode_structure(TrackStructure(1.0)), version=1)
    queue = JobQueue(str(tmp_path / "jobs.db"))
    infos = [
        {"path": "a.mp3", "fingerprint": "fp1"},
        {"path": "b.mp3", "fingerprint": "fp2"},
    ]
    queued = enqueue_library_work(
        queue, infos, deps=_deps(tmp_path, fc), kinds=(KIND_STRUCTURE,)
    )
    assert queued[KIND_STRUCTURE] == 1


def test_enqueue_structure_revives_a_terminal_job_when_the_cache_is_missing(tmp_path):
    """A version bump blanks the cache row but leaves its queue row terminal.

    The cache is the source of truth, so the job must be revived — otherwise the
    backfill never runs and the Structure retrigger appears to do nothing.
    """
    fc = FeatureCache(str(tmp_path / "features.db"))
    queue = JobQueue(str(tmp_path / "jobs.db"))
    infos = [{"path": "a.mp3", "fingerprint": "fp1"}]
    deps = _deps(tmp_path, fc)

    first = enqueue_library_work(queue, infos, deps=deps, kinds=(KIND_STRUCTURE,))
    assert first[KIND_STRUCTURE] == 1
    queue.complete(queue.claim(KIND_STRUCTURE)[0])
    assert queue.stats(KIND_STRUCTURE).done == 1

    # The cache row is absent (as after a version purge), so it must re-queue.
    assert not fc.has("structure", "fp1", version=deps.structure_version)
    again = enqueue_library_work(queue, infos, deps=deps, kinds=(KIND_STRUCTURE,))
    assert again[KIND_STRUCTURE] == 1
    assert queue.stats(KIND_STRUCTURE).pending == 1
    assert queue.stats(KIND_STRUCTURE).done == 0


# ── cache source ────────────────────────────────────────────


def test_structure_cache_round_trip_and_version_purge(tmp_path):
    fc = FeatureCache(str(tmp_path / "features.db"))
    blob = encode_structure(TrackStructure(1.0, 2.0, 3.0, 4.0, 5.0))
    fc.put("structure", "fp", blob, version=2)

    assert fc.count("structure", version=2) == 1
    assert fc.count("structure", version=1) == 0

    purged = fc.purge_old_versions("structure", 3)
    assert purged == 1
    assert not fc.has("structure", "fp", version=2)
