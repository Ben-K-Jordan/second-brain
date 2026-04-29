"""Cross-encoder reranking. Voyage rerank-2 turns a noisy hybrid top-50 into a sharp top-k."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tenacity import retry, stop_after_attempt, wait_exponential

from .budget import check_budget, record_usage
from .config import Config


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(
        self, query: str, documents: list[str], top_k: int
    ) -> list[tuple[int, float]]:
        """Return [(original_index, relevance_score)] sorted by score desc."""
        ...


class VoyageReranker:
    def __init__(self, api_key: str, model: str = "rerank-2-lite", cfg: Config | None = None):
        import voyageai

        self._client = voyageai.Client(api_key=api_key)
        self.name = model
        self._model = model
        self._cfg = cfg

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def rerank(
        self, query: str, documents: list[str], top_k: int
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        top_k = min(top_k, len(documents))
        if self._cfg is not None:
            check_budget(self._cfg, "voyage")
        result = self._client.rerank(
            query=query,
            documents=documents,
            model=self._model,
            top_k=top_k,
        )
        if self._cfg is not None:
            record_usage(
                self._cfg, "voyage", self._model,
                input_tokens=getattr(result, "total_tokens", 0),
                note=f"rerank/{len(documents)}docs",
            )
        return [(r.index, r.relevance_score) for r in result.results]


def make_reranker(cfg: Config) -> Reranker | None:
    """Construct a reranker if enabled and credentials are available, else None."""
    if not cfg.rerank_enabled:
        return None
    if not cfg.voyage_api_key:
        return None
    return VoyageReranker(cfg.voyage_api_key, model=cfg.rerank_model, cfg=cfg)
