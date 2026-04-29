"""Pluggable embedders. API-first (Voyage) with local fallback (sentence-transformers)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tenacity import retry, stop_after_attempt, wait_exponential

from .budget import check_budget, record_usage
from .config import Config


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


_VOYAGE_DIMS: dict[str, int] = {
    "voyage-3": 1024,
    "voyage-3-lite": 512,
    "voyage-3-large": 1024,
    "voyage-code-3": 1024,
}


class VoyageEmbedder:
    def __init__(self, api_key: str, model: str = "voyage-3", cfg: Config | None = None):
        import voyageai

        if model not in _VOYAGE_DIMS:
            raise ValueError(f"Unknown Voyage model: {model}. Known: {list(_VOYAGE_DIMS)}")
        self._client = voyageai.Client(api_key=api_key)
        self.name = model
        self.dim = _VOYAGE_DIMS[model]
        self._cfg = cfg

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        if not texts:
            return []
        if self._cfg is not None:
            check_budget(self._cfg, "voyage")
        result = self._client.embed(texts, model=self.name, input_type=input_type)
        if self._cfg is not None:
            record_usage(
                self._cfg, "voyage", self.name,
                input_tokens=getattr(result, "total_tokens", 0),
                note=f"embed/{input_type}",
            )
        return [list(e) for e in result.embeddings]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, input_type="document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], input_type="query")[0]


class LocalEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Local embedder requires sentence-transformers. "
                "Install with: pip install second-brain[local]"
            ) from e
        self._model = SentenceTransformer(model_name)
        self.name = model_name
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        arr = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return arr.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def make_embedder(cfg: Config) -> Embedder:
    """Pick an embedder based on config and available credentials."""
    provider = cfg.embedder_provider
    if provider == "auto":
        provider = "voyage" if cfg.voyage_api_key else "local"

    if provider == "voyage":
        if not cfg.voyage_api_key:
            raise RuntimeError(
                "embedder_provider='voyage' but VOYAGE_API_KEY is not set. "
                "Export it or set embedder_provider='local'."
            )
        return VoyageEmbedder(cfg.voyage_api_key, model=cfg.voyage_model, cfg=cfg)

    if provider == "local":
        return LocalEmbedder(cfg.local_model)

    raise ValueError(f"Unknown embedder_provider: {provider}")
