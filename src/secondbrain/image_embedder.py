"""Multimodal embeddings for semantic image search.

Default backend: Voyage's voyage-multimodal-3 (1024-dim shared text/image space).
Lets you search for images by text query: "the diagram showing system
architecture", "screenshot with airline confirmation", "photo of my whiteboard".

Stored separately from text embeddings (vec_images vs vec_chunks) so the
existing text index keeps working unchanged. Image OCR via Tesseract still
runs in parallel - the two paths complement each other (OCR for in-image
text, multimodal for visual content).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

from tenacity import retry, stop_after_attempt, wait_exponential

from .config import Config

log = logging.getLogger(__name__)


_VOYAGE_MM_DIMS: dict[str, int] = {
    "voyage-multimodal-3": 1024,
}


@runtime_checkable
class ImageEmbedder(Protocol):
    name: str
    dim: int

    def embed_image(self, path: Path) -> list[float]: ...

    def embed_text_query(self, text: str) -> list[float]: ...


class VoyageMultimodalEmbedder:
    """voyage-multimodal-3: text and images share the same 1024-dim vector space."""

    def __init__(self, api_key: str, model: str = "voyage-multimodal-3"):
        import voyageai

        if model not in _VOYAGE_MM_DIMS:
            raise ValueError(
                f"Unknown Voyage multimodal model: {model}. Known: {list(_VOYAGE_MM_DIMS)}"
            )
        self._client = voyageai.Client(api_key=api_key)
        self.name = model
        self.dim = _VOYAGE_MM_DIMS[model]

    def _open_image(self, path: Path):
        from PIL import Image

        # Voyage's SDK accepts PIL.Image objects directly. Convert to RGB to
        # avoid issues with palette / RGBA images.
        img = Image.open(path)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        return img

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def embed_image(self, path: Path) -> list[float]:
        img = self._open_image(path)
        result = self._client.multimodal_embed(
            inputs=[[img]],
            model=self.name,
            input_type="document",
        )
        return list(result.embeddings[0])

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def embed_text_query(self, text: str) -> list[float]:
        result = self._client.multimodal_embed(
            inputs=[[text]],
            model=self.name,
            input_type="query",
        )
        return list(result.embeddings[0])


def make_image_embedder(cfg: Config) -> ImageEmbedder | None:
    """Construct the multimodal embedder if enabled and credentials are present."""
    if not cfg.image_embed_enabled:
        return None
    if not cfg.voyage_api_key:
        return None
    return VoyageMultimodalEmbedder(cfg.voyage_api_key, model=cfg.multimodal_model)
