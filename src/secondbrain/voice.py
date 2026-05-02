"""Voice capture — speak into the brain instead of typing.

The use case: walking from class, idea hits, you don't want to pull out
the laptop and type a note. Run ``secondbrain capture``, speak, stop
talking, transcript appears, ingested.

Auto-VAD stop: we record continuously and watch the rolling RMS volume.
After ``cfg.voice_silence_seconds`` (default 1.5s) below the silence
threshold, recording stops automatically. So the friction is just
"start command, speak, stop talking" — no hotkey, no second command.

Setup: install the [voice] extra:

    pip install -e .[voice]

This pulls ``sounddevice`` (cross-platform audio I/O) and
``faster-whisper`` (the transcriber, already used by the file watcher
for audio/video files).

Storage:
- Each capture gets a virtual_path of ``voice://YYYY-MM-DD-HHMMSS``.
- Transcript stored as a regular brain document (kind="document",
  source="voice"). Searchable via the same hybrid retrieval as
  everything else; chat agents can find your voice notes via
  ``search_brain``.
- Optionally also writes the raw .wav to ``cfg.data_dir / "voice"``
  for keepsake — toggle via ``cfg.voice_save_audio``.

Failure modes:
- Missing ``[voice]`` extra → CLI prints install hint and exits.
- No microphone / sounddevice can't open default device → caught,
  surfaced as a friendly error.
- Whisper fails → transcript is empty; we don't ingest a blank doc.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import Config
from .embedder import Embedder
from .indexer import index_text
from .transcriber import make_transcriber

log = logging.getLogger(__name__)

# Hard caps so we never accidentally record forever (e.g. silence
# threshold mistuned). 5 minutes is more than enough for any one capture.
_MAX_RECORD_SECONDS = 300
_DEFAULT_SAMPLE_RATE = 16_000  # Whisper expects 16kHz
_DEFAULT_SILENCE_RMS = 0.012   # tuned on a quiet room with built-in mic
_DEFAULT_SILENCE_SECONDS = 1.5
_MIN_RECORD_SECONDS = 0.5      # don't transcribe sub-half-second blips


@dataclass
class CaptureResult:
    """Outcome of one capture session."""
    transcript: str
    duration_seconds: float
    virtual_path: str
    audio_path: str | None     # set when cfg.voice_save_audio is True
    chunks_indexed: int = 0


class VoiceCaptureUnavailable(RuntimeError):
    """Raised when [voice] dependencies aren't installed or the mic is unreachable."""


def record_with_vad(
    sample_rate: int = _DEFAULT_SAMPLE_RATE,
    silence_rms: float = _DEFAULT_SILENCE_RMS,
    silence_seconds: float = _DEFAULT_SILENCE_SECONDS,
    max_seconds: int = _MAX_RECORD_SECONDS,
    on_status: callable | None = None,  # type: ignore[valid-type]
):
    """Record from the default mic until ``silence_seconds`` of quiet.

    Returns ``(samples, sample_rate, duration_seconds)``. ``samples`` is a
    1-D numpy float32 array in [-1, 1].

    Reads ~0.1s blocks and tracks RMS. We stop after enough consecutive
    sub-threshold blocks. ``on_status`` fires with one of:
      ('start', None)              — recording started
      ('volume', rms_value)        — periodic volume update for UI
      ('silence_grace', remaining) — counting down silence-stop
      ('stop', reason_str)         — recording stopped
    """
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as e:
        raise VoiceCaptureUnavailable(
            "Voice capture needs the [voice] extra: "
            "pip install -e \".[voice]\""
        ) from e

    block_seconds = 0.1
    block_samples = int(sample_rate * block_seconds)
    buf: list = []
    silence_blocks_needed = max(1, int(silence_seconds / block_seconds))
    silence_blocks = 0
    started_at = time.time()

    if on_status:
        on_status("start", None)

    try:
        with sd.InputStream(
            channels=1, samplerate=sample_rate, dtype="float32",
        ) as stream:
            while True:
                if time.time() - started_at >= max_seconds:
                    if on_status:
                        on_status("stop", "max_duration")
                    break
                block, _overflow = stream.read(block_samples)
                buf.append(block.flatten().copy())
                rms = float(np.sqrt(np.mean(block ** 2)))
                if on_status:
                    on_status("volume", rms)
                if rms < silence_rms:
                    silence_blocks += 1
                    if on_status:
                        on_status("silence_grace",
                                  silence_blocks_needed - silence_blocks)
                    # Don't trigger silence-stop until we've recorded at
                    # least the minimum useful duration. Otherwise an
                    # initial pause kills the capture.
                    if (
                        silence_blocks >= silence_blocks_needed
                        and (time.time() - started_at) >= _MIN_RECORD_SECONDS
                    ):
                        if on_status:
                            on_status("stop", "silence")
                        break
                else:
                    silence_blocks = 0
    except sd.PortAudioError as e:  # type: ignore[attr-defined]
        raise VoiceCaptureUnavailable(
            f"Couldn't open microphone: {e}. "
            "Check your default input device + permissions."
        ) from e

    if not buf:
        return np.zeros(0, dtype="float32"), sample_rate, 0.0
    samples = np.concatenate(buf)
    duration = len(samples) / sample_rate
    return samples, sample_rate, duration


def transcribe_samples(
    cfg: Config, samples, sample_rate: int,
) -> str:
    """Run faster-whisper over a numpy float32 sample array.

    Writes a tempfile because faster-whisper's API takes a path. Cleans
    up on its own. Returns the joined transcript text (segments concat'd
    with spaces).
    """
    import tempfile
    import wave

    # Whisper wants either int16 PCM or float32 in [-1, 1]. We write
    # int16 PCM via wave, which is universally supported.
    import numpy as np

    if len(samples) == 0:
        return ""

    int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype("int16")
    with tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False, prefix="sb-voice-",
    ) as tmp:
        path = Path(tmp.name)
    try:
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(int16.tobytes())
        try:
            transcriber = make_transcriber(cfg)
        except ImportError as e:
            raise VoiceCaptureUnavailable(
                "Transcription needs the [voice] extra (faster-whisper): "
                "pip install -e \".[voice]\""
            ) from e
        if transcriber is None:
            return ""
        return transcriber.transcribe(path).strip()
    finally:
        try:
            path.unlink()
        except OSError:
            pass


def save_audio(
    cfg: Config, samples, sample_rate: int, virtual_path: str,
) -> str:
    """Persist the raw .wav to ``cfg.data_dir / 'voice'`` for keepsake.

    Returns the saved path string. The filename matches the transcript's
    virtual_path so they're easy to correlate later.
    """
    import wave

    import numpy as np

    voice_dir = cfg.data_dir / "voice"
    voice_dir.mkdir(parents=True, exist_ok=True)
    # virtual_path looks like 'voice://2026-04-15-143022'.
    name = virtual_path.split("//", 1)[-1] + ".wav"
    out = voice_dir / name
    int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype("int16")
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16.tobytes())
    return str(out)


def capture(
    cfg: Config,
    conn,
    embedder: Embedder,
    *,
    silence_rms: float | None = None,
    silence_seconds: float | None = None,
    max_seconds: int = _MAX_RECORD_SECONDS,
    on_status: callable | None = None,  # type: ignore[valid-type]
    note_label: str = "voice note",
) -> CaptureResult:
    """End-to-end capture: record → transcribe → ingest.

    Returns a ``CaptureResult`` with the transcript + ingestion result.
    Raises ``VoiceCaptureUnavailable`` if the [voice] extra isn't
    installed or the mic can't be opened.
    """
    sr = int(getattr(cfg, "voice_sample_rate", _DEFAULT_SAMPLE_RATE) or _DEFAULT_SAMPLE_RATE)
    sil_rms = (
        silence_rms if silence_rms is not None
        else float(getattr(cfg, "voice_silence_rms", _DEFAULT_SILENCE_RMS))
    )
    sil_secs = (
        silence_seconds if silence_seconds is not None
        else float(getattr(cfg, "voice_silence_seconds", _DEFAULT_SILENCE_SECONDS))
    )

    samples, sample_rate, duration = record_with_vad(
        sample_rate=sr, silence_rms=sil_rms, silence_seconds=sil_secs,
        max_seconds=max_seconds, on_status=on_status,
    )
    if duration < _MIN_RECORD_SECONDS:
        return CaptureResult(
            transcript="", duration_seconds=duration,
            virtual_path="", audio_path=None, chunks_indexed=0,
        )

    transcript = transcribe_samples(cfg, samples, sample_rate)
    if not transcript.strip():
        return CaptureResult(
            transcript="", duration_seconds=duration,
            virtual_path="", audio_path=None, chunks_indexed=0,
        )

    virtual_path = _virtual_path_for_capture()
    audio_path: str | None = None
    if getattr(cfg, "voice_save_audio", True):
        try:
            audio_path = save_audio(cfg, samples, sample_rate, virtual_path)
        except OSError as e:
            log.warning("voice: couldn't save audio: %s", e)

    title = f"{note_label} · {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    body_lines = [f"# {title}", "", transcript]
    if audio_path:
        body_lines += ["", f"[Audio file]({audio_path})"]
    body = "\n".join(body_lines)

    result = index_text(
        conn=conn, embedder=embedder, cfg=cfg,
        virtual_path=virtual_path, title=title,
        content=body, mtime=time.time(),
        kind="document", source="voice",
    )
    chunks = result.chunks if result.status == "indexed" else 0
    return CaptureResult(
        transcript=transcript,
        duration_seconds=duration,
        virtual_path=virtual_path,
        audio_path=audio_path,
        chunks_indexed=chunks,
    )


def _virtual_path_for_capture() -> str:
    """Stable ID + readable timestamp. Re-running within one second is
    rare for human voice capture; collisions are tolerable (index_text
    upserts on virtual_path)."""
    return "voice://" + datetime.now().strftime("%Y-%m-%d-%H%M%S")
