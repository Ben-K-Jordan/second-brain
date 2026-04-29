"""Image OCR. Currently uses Tesseract via pytesseract.

Tesseract binary must be installed separately:
- Windows: winget install UB-Mannheim.TesseractOCR
- macOS:   brew install tesseract
- Linux:   apt install tesseract-ocr (or distro equivalent)

A future addition will plug in CLIP / voyage-multimodal-3 for semantic image
content (the "diagram of system architecture" use case). Both surfaces will
flow into the same SearchResult shape so callers don't need to care.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Protocol, runtime_checkable

from .config import Config

log = logging.getLogger(__name__)


@runtime_checkable
class OCREngine(Protocol):
    name: str

    def ocr(self, path: Path) -> str: ...


class TesseractOCR:
    def __init__(self, lang: str = "eng", binary_path: str | None = None):
        try:
            import pytesseract
        except ImportError as e:
            raise ImportError(
                "OCR requires the [ocr] extra. Install with: pip install -e .[ocr]"
            ) from e

        self._pytesseract = pytesseract
        if binary_path:
            pytesseract.pytesseract.tesseract_cmd = binary_path
        elif shutil.which("tesseract") is None:
            # Common Windows install location from UB-Mannheim package
            for candidate in (
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            ):
                if Path(candidate).exists():
                    pytesseract.pytesseract.tesseract_cmd = candidate
                    break
        self._lang = lang
        self.name = f"tesseract-{lang}"

    def ocr(self, path: Path) -> str:
        from PIL import Image

        try:
            img = Image.open(path)
        except Exception as e:
            log.warning("PIL could not open %s: %s", path, e)
            return ""
        try:
            return self._pytesseract.image_to_string(img, lang=self._lang).strip()
        except Exception as e:
            log.warning("tesseract failed on %s: %s", path, e)
            return ""


def make_ocr_engine(cfg: Config) -> OCREngine | None:
    """Construct an OCR engine if enabled in config, else None."""
    if not cfg.ocr_enabled:
        return None
    return TesseractOCR(lang=cfg.ocr_lang)
