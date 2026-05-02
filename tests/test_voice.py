"""Phase 41: voice capture — recording, transcription, ingestion.

Tests don't open a real microphone or run faster-whisper. Instead we
inject deterministic samples + stub the transcriber so the pipeline
plumbing is verified without hardware or model downloads.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

# ============================== imports ===============================

def test_voice_module_imports_without_extra():
    """Importing the module shouldn't require sounddevice — that gets
    raised lazily inside record_with_vad. Test that the import itself
    works on a vanilla install."""
    import secondbrain.voice  # noqa: F401


# ===================== virtual path generation ========================

def test_virtual_path_format():
    from secondbrain.voice import _virtual_path_for_capture

    vp = _virtual_path_for_capture()
    assert vp.startswith("voice://")
    # Format YYYY-MM-DD-HHMMSS — 17 chars after the scheme
    suffix = vp.split("//", 1)[1]
    assert len(suffix) == 17
    assert suffix[4] == "-" and suffix[7] == "-" and suffix[10] == "-"


# ============== record_with_vad missing-dependency path ==============

def test_record_raises_when_sounddevice_missing(monkeypatch):
    """When sounddevice isn't installed, record_with_vad should raise
    the friendly VoiceCaptureUnavailable rather than a confusing
    ImportError mid-stream."""
    import sys

    from secondbrain.voice import VoiceCaptureUnavailable, record_with_vad

    # Force ImportError by removing/poisoning the modules.
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(VoiceCaptureUnavailable, match="\\[voice\\]"):
        record_with_vad()


# ===================== save_audio writes a wav =========================

def test_save_audio_round_trip(tmp_path, tmp_cfg):
    """Generate fake samples, save → read back, verify wav metadata."""
    np = pytest.importorskip("numpy")
    import wave

    from secondbrain.voice import save_audio

    cfg = replace(tmp_cfg, data_dir=tmp_path)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    samples = np.linspace(-0.3, 0.3, 16000, dtype="float32")
    out_path = save_audio(cfg, samples, 16000, "voice://2026-04-15-100000")
    assert out_path.endswith("2026-04-15-100000.wav")
    with wave.open(out_path, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2  # 16-bit
        assert wf.getframerate() == 16000
        assert wf.getnframes() == 16000


def test_save_audio_clips_out_of_range_samples(tmp_path, tmp_cfg):
    """Float samples > 1.0 / < -1.0 must clip rather than wrap."""
    np = pytest.importorskip("numpy")
    import wave

    from secondbrain.voice import save_audio

    cfg = replace(tmp_cfg, data_dir=tmp_path)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    samples = np.array([3.0, -3.0, 0.0, 0.5], dtype="float32")
    out_path = save_audio(cfg, samples, 16000, "voice://x")
    with wave.open(out_path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
    int16 = np.frombuffer(frames, dtype="<i2")
    # Clipped to int16 boundaries.
    assert int16[0] == 32767
    assert int16[1] == -32767
    assert int16[2] == 0


# ============================ transcribe_samples ========================

def test_transcribe_samples_empty_returns_empty_string(tmp_cfg):
    """No samples → empty transcript, no Whisper call attempted."""
    np = pytest.importorskip("numpy")

    from secondbrain.voice import transcribe_samples

    out = transcribe_samples(tmp_cfg, np.zeros(0, dtype="float32"), 16000)
    assert out == ""


def test_transcribe_samples_returns_what_transcriber_says(
    tmp_cfg, monkeypatch,
):
    np = pytest.importorskip("numpy")

    from secondbrain import voice as voice_mod

    class FakeTranscriber:
        name = "fake"
        def transcribe(self, path):
            return "  hello brain  "

    monkeypatch.setattr(voice_mod, "make_transcriber", lambda cfg: FakeTranscriber())
    samples = np.full(16000, 0.1, dtype="float32")  # 1s of fake audio
    out = voice_mod.transcribe_samples(tmp_cfg, samples, 16000)
    assert out == "hello brain"


def test_transcribe_samples_handles_missing_whisper_extra(tmp_cfg, monkeypatch):
    """When the [whisper] extra isn't installed, raise the clear error."""
    np = pytest.importorskip("numpy")

    from secondbrain import voice as voice_mod
    from secondbrain.voice import VoiceCaptureUnavailable

    def boom(cfg):
        raise ImportError("faster-whisper not installed")

    monkeypatch.setattr(voice_mod, "make_transcriber", boom)
    with pytest.raises(VoiceCaptureUnavailable, match="\\[voice\\]"):
        voice_mod.transcribe_samples(
            tmp_cfg, np.full(16000, 0.1, dtype="float32"), 16000,
        )


def test_transcribe_samples_returns_empty_when_transcriber_disabled(
    tmp_cfg, monkeypatch,
):
    """If make_transcriber returns None (e.g. transcribe_enabled=false),
    don't pretend to transcribe — return ''."""
    np = pytest.importorskip("numpy")

    from secondbrain import voice as voice_mod

    monkeypatch.setattr(voice_mod, "make_transcriber", lambda cfg: None)
    samples = np.full(16000, 0.1, dtype="float32")
    out = voice_mod.transcribe_samples(tmp_cfg, samples, 16000)
    assert out == ""


# ============================ end-to-end =============================

def test_capture_end_to_end_indexes_transcript(
    fresh_db, tmp_cfg, fake_embedder, monkeypatch, tmp_path,
):
    """Stub the recorder + transcriber; verify capture writes a doc to
    the brain that's then findable via the index."""
    np = pytest.importorskip("numpy")

    from secondbrain import voice as voice_mod
    from secondbrain.voice import capture

    cfg = replace(tmp_cfg, data_dir=tmp_path, voice_save_audio=False)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    fake_samples = np.full(32000, 0.05, dtype="float32")  # 2s of "audio"

    def fake_record(*a, **kw):
        return fake_samples, 16000, 2.0

    monkeypatch.setattr(voice_mod, "record_with_vad", fake_record)

    class FakeTranscriber:
        name = "fake"
        def transcribe(self, path):
            return "voyage rate limits are 8M tokens per minute"

    monkeypatch.setattr(voice_mod, "make_transcriber", lambda cfg: FakeTranscriber())

    result = capture(cfg, fresh_db, fake_embedder)
    assert result.transcript == "voyage rate limits are 8M tokens per minute"
    assert result.duration_seconds == 2.0
    assert result.virtual_path.startswith("voice://")
    assert result.audio_path is None  # voice_save_audio=False
    assert result.chunks_indexed >= 1
    # The doc should now be in the brain.
    row = fresh_db.execute(
        "SELECT * FROM files WHERE path = ?", (result.virtual_path,),
    ).fetchone()
    assert row is not None


def test_capture_silence_returns_empty_without_indexing(
    fresh_db, tmp_cfg, fake_embedder, monkeypatch,
):
    """Sub-half-second of audio → no transcript → no row in the brain."""
    np = pytest.importorskip("numpy")
    from secondbrain import voice as voice_mod
    from secondbrain.voice import capture

    def fake_record(*a, **kw):
        return np.zeros(0, dtype="float32"), 16000, 0.0

    monkeypatch.setattr(voice_mod, "record_with_vad", fake_record)

    before = fresh_db.execute(
        "SELECT COUNT(*) AS n FROM files",
    ).fetchone()["n"]
    result = capture(tmp_cfg, fresh_db, fake_embedder)
    after = fresh_db.execute(
        "SELECT COUNT(*) AS n FROM files",
    ).fetchone()["n"]
    assert result.transcript == ""
    assert result.chunks_indexed == 0
    assert before == after, "silent capture must not write to the brain"


def test_capture_blank_transcript_skipped(
    fresh_db, tmp_cfg, fake_embedder, monkeypatch,
):
    """Non-trivial recording but Whisper returned empty (e.g. background
    hum) → still skipped. Don't pollute the brain with empty docs."""
    np = pytest.importorskip("numpy")
    from secondbrain import voice as voice_mod
    from secondbrain.voice import capture

    def fake_record(*a, **kw):
        return np.full(16000, 0.05, dtype="float32"), 16000, 1.0

    class BlankTranscriber:
        name = "x"
        def transcribe(self, path):
            return "   "  # whitespace only

    monkeypatch.setattr(voice_mod, "record_with_vad", fake_record)
    monkeypatch.setattr(voice_mod, "make_transcriber",
                        lambda cfg: BlankTranscriber())
    result = capture(tmp_cfg, fresh_db, fake_embedder)
    assert result.transcript == ""
    assert result.chunks_indexed == 0


def test_capture_saves_audio_when_enabled(
    fresh_db, tmp_cfg, fake_embedder, monkeypatch, tmp_path,
):
    np = pytest.importorskip("numpy")
    from secondbrain import voice as voice_mod
    from secondbrain.voice import capture

    cfg = replace(tmp_cfg, data_dir=tmp_path, voice_save_audio=True)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    def fake_record(*a, **kw):
        return np.full(16000, 0.1, dtype="float32"), 16000, 1.0

    monkeypatch.setattr(voice_mod, "record_with_vad", fake_record)

    class FakeTranscriber:
        name = "x"
        def transcribe(self, path):
            return "test transcript"

    monkeypatch.setattr(voice_mod, "make_transcriber", lambda cfg: FakeTranscriber())

    result = capture(cfg, fresh_db, fake_embedder)
    assert result.audio_path is not None
    from pathlib import Path
    assert Path(result.audio_path).exists()
    assert (Path(result.audio_path).parent.name) == "voice"
