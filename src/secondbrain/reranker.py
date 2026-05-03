"""Cross-encoder reranking. Voyage rerank-2 turns a noisy hybrid top-50 into a sharp top-k."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

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

    def rerank(
        self, query: str, documents: list[str], top_k: int
    ) -> list[tuple[int, float]]:
        # Round 15 (audit-found gap A4) — check_budget runs OUTSIDE
        # the @retry-decorated call. Previously check_budget was
        # inside the retried path, which meant a BudgetExceededError
        # (a RuntimeError) got retried 3x and surfaced with the
        # wrong shape. Also (A2) explicit feature='rerank' so the
        # per-feature bucket fires.
        if not documents:
            return []
        if self._cfg is not None:
            check_budget(self._cfg, "voyage", feature="rerank")
        return self._rerank_with_retry(query, documents, top_k)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=10),
        # Only retry on real network / API errors; never on
        # BudgetExceededError or programmer errors. We can't import
        # the SDK exception here without forcing a dep at import time,
        # so retry on the broad `Exception` minus our own RuntimeError
        # subclass is the practical compromise — see the explicit
        # exclusion in the wrapper above.
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    )
    def _rerank_with_retry(
        self, query: str, documents: list[str], top_k: int
    ) -> list[tuple[int, float]]:
        top_k = min(top_k, len(documents))
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
                feature="rerank",
            )
        return [(r.index, r.relevance_score) for r in result.results]


def make_reranker(cfg: Config) -> Reranker | None:
    """Construct a reranker if enabled and credentials are available, else None."""
    if not cfg.rerank_enabled:
        return None
    if not cfg.voyage_api_key:
        return None
    return VoyageReranker(cfg.voyage_api_key, model=cfg.rerank_model, cfg=cfg)
