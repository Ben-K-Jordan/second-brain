"""Audio/video transcription. Default: faster-whisper, runs locally on CPU."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from .config import Config


@runtime_checkable
class Transcriber(Protocol):
    name: str

    def transcribe(self, path: Path) -> str: ...


class LocalWhisperTranscriber:
    """faster-whisper based transcriber.

    Models live at ~/.cache/huggingface or wherever HF caches. First use of
    each model size downloads it (small=244MB, medium=769MB, large=1550MB).
    PyAV (bundled with faster-whisper) handles audio extraction from video,
    so no separate ffmpeg install is required.
    """

    def __init__(self, model_size: str = "small", device: str = "cpu"):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "Whisper transcription requires the [whisper] extra. "
                "Install with: pip install -e .[whisper]"
            ) from e

        # int8 on CPU is the right default for personal use: ~5x faster than
        # float32 with negligible quality loss for most content. We default to
        # device="cpu" rather than "auto" because faster-whisper's "auto" tries
        # CUDA first and fails noisily on machines without the cuBLAS DLLs even
        # when CTranslate2 itself was built without GPU support.
        compute_type = "int8" if device == "cpu" else "float16"
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.name = f"faster-whisper-{model_size}"

    def transcribe(self, path: Path) -> str:
        segments, _info = self._model.transcribe(
            str(path),
            beam_size=1,
            vad_filter=True,  # skip silence; big speedup on real-world audio
        )
        parts: list[str] = []
        for seg in segments:
            text = seg.text.strip()
            if text:
                parts.append(text)
        return " ".join(parts)


def make_transcriber(cfg: Config) -> Transcriber | None:
    """Return a transcriber if enabled in config, else None.

    The transcriber is lazy-instantiated (model loaded eagerly on construction)
    so only the indexer pays the load cost and only when it actually needs to
    transcribe something.
    """
    if not cfg.transcribe_enabled:
        return None
    return LocalWhisperTranscriber(model_size=cfg.whisper_model_size)
