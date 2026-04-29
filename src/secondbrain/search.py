"""Hybrid retrieval: vector + BM25 fused via Reciprocal Rank Fusion.

Beyond vanilla RRF this module layers in two quality features:

- **Adaptive alpha**: per-query tuning of vector vs. keyword weight based on
  cheap heuristics (capitalization ratio, ID-like tokens, query length).
- **Time-decay**: a gentle recency bonus that nudges fresh files up without
  drowning out long-lived reference material. Configurable half-life.
"""

from __future__ import annotations

import math
import re
import sqlite3
import time
from dataclasses import dataclass

from .db import serialize_f32
from .embedder import Embedder
from .reranker import Reranker

RRF_K = 60  # RRF constant; 60 is the original paper's value


@dataclass
class SearchResult:
    chunk_id: int
    file_path: str
    chunk_index: int
    text: str
    score: float
    sources: tuple[str, ...]  # which retrievers matched: ("vector",), ("fts",), or both
    mtime: float | None = None
    reranked: bool = False


_ID_LIKE = re.compile(r"\d|[A-Z]{2,}")


def adaptive_alpha(query: str, default: float = 0.5) -> float:
    """Pick a per-query alpha in [0, 1] (0=BM25 only, 1=vector only).

    Rules of thumb:
      - **Long prose (>= 7 tokens) wins**, even if it sprinkles in acronyms.
        Prose is rarely word-for-word in the source; vector helps most.
      - Otherwise, capital-heavy or ID-bearing short queries -> BM25 lean
        (exact-token matching wins on names, IDs, ticker symbols).
    """
    tokens = query.split()
    if not tokens:
        return default
    long_prose = len(tokens) >= 7
    if long_prose:
        return min(0.8, default + 0.2)
    cap_ratio = sum(1 for t in tokens if t and t[0].isupper()) / len(tokens)
    has_idlike = any(_ID_LIKE.search(t) for t in tokens)
    if has_idlike or cap_ratio >= 0.5:
        return max(0.2, default - 0.3)
    return default


def _time_decay_factor(mtime: float, half_life_days: float, now: float | None = None) -> float:
    """Exponential decay in [0, 1]: 1 for now, 0.5 at one half-life, ~0 deep past."""
    now = now if now is not None else time.time()
    age_days = max(0.0, (now - mtime) / 86400.0)
    return math.exp(-math.log(2) * age_days / max(1e-6, half_life_days))


def _vector_search(
    conn: sqlite3.Connection, query_embedding: list[float], k: int
) -> list[tuple[int, float]]:
    """Return [(chunk_id, distance)] from sqlite-vec, lowest distance first."""
    rows = conn.execute(
        "SELECT chunk_id, distance FROM vec_chunks "
        "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (serialize_f32(query_embedding), k),
    ).fetchall()
    return [(r["chunk_id"], r["distance"]) for r in rows]


def _fts_search(conn: sqlite3.Connection, query: str, k: int) -> list[tuple[int, float]]:
    """Return [(chunk_id, bm25_score)] from FTS5."""
    fts_query = _sanitize_fts_query(query)
    if not fts_query:
        return []
    rows = conn.execute(
        "SELECT rowid, bm25(fts_chunks) AS score FROM fts_chunks "
        "WHERE fts_chunks MATCH ? ORDER BY score LIMIT ?",
        (fts_query, k),
    ).fetchall()
    return [(r["rowid"], r["score"]) for r in rows]


def _sanitize_fts_query(query: str) -> str:
    """Make a user query safe for FTS5 MATCH. We OR the terms for recall."""
    tokens = [t for t in (w.strip(".,?!:;\"'()[]{}") for w in query.split()) if t]
    tokens = [t.replace('"', "") for t in tokens]
    if not tokens:
        return ""
    quoted = [f'"{t}"' for t in tokens]
    return " OR ".join(quoted)


def _rrf_merge(
    vec_results: list[tuple[int, float]],
    fts_results: list[tuple[int, float]],
    alpha: float,
) -> dict[int, tuple[float, set[str]]]:
    """Fuse two ranked lists via Reciprocal Rank Fusion, weighted by alpha.

    alpha=1.0 -> vector only, alpha=0.0 -> keyword only. Returns
    {chunk_id: (combined_score, {source_names})}.
    """
    scores: dict[int, tuple[float, set[str]]] = {}
    for rank, (cid, _) in enumerate(vec_results):
        s = alpha / (RRF_K + rank + 1)
        prev = scores.get(cid, (0.0, set()))
        scores[cid] = (prev[0] + s, prev[1] | {"vector"})
    for rank, (cid, _) in enumerate(fts_results):
        s = (1 - alpha) / (RRF_K + rank + 1)
        prev = scores.get(cid, (0.0, set()))
        scores[cid] = (prev[0] + s, prev[1] | {"fts"})
    return scores


def _hydrate(
    conn: sqlite3.Connection, chunk_ids: list[int]
) -> dict[int, tuple[str, int, str, float]]:
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT c.id, c.chunk_index, c.text, f.path, f.mtime "
        f"FROM chunks c JOIN files f ON f.id = c.file_id "
        f"WHERE c.id IN ({placeholders})",
        chunk_ids,
    ).fetchall()
    return {
        r["id"]: (r["path"], r["chunk_index"], r["text"], r["mtime"]) for r in rows
    }


def hybrid_search(
    conn: sqlite3.Connection,
    embedder: Embedder,
    query: str,
    k: int = 10,
    alpha: float | None = 0.5,
    reranker: Reranker | None = None,
    rerank_overfetch: int = 50,
    use_adaptive_alpha: bool = False,
    time_decay_weight: float = 0.0,
    time_decay_half_life_days: float = 365.0,
) -> list[SearchResult]:
    """Run hybrid search and return up to k merged results.

    - When ``use_adaptive_alpha`` is set, ``alpha`` is treated as the *default*
      and per-query overrides may push it up (conceptual prose) or down
      (proper-noun / ID-bearing queries).
    - When ``time_decay_weight > 0``, ranking blends in a recency bonus with
      the given exponential half-life.
    - When ``reranker`` is supplied, the top ``rerank_overfetch`` candidates
      from RRF are reranked by the cross-encoder before truncating to k.
    """
    effective_alpha = alpha if alpha is not None else 0.5
    if use_adaptive_alpha:
        effective_alpha = adaptive_alpha(query, default=effective_alpha)

    candidate_count = max(rerank_overfetch if reranker else k * 3, 30)
    q_emb = embedder.embed_query(query)
    vec = _vector_search(conn, q_emb, candidate_count)
    fts = _fts_search(conn, query, candidate_count)
    fused = _rrf_merge(vec, fts, alpha=effective_alpha)

    candidates = sorted(fused.items(), key=lambda kv: -kv[1][0])[:candidate_count]
    chunk_ids = [cid for cid, _ in candidates]
    hydrated = _hydrate(conn, chunk_ids)

    # Apply time-decay to the RRF score before reranking. We blend by weight w:
    #   blended = (1 - w) * normalized_rrf + w * recency
    # Normalising RRF to [0, 1] within this candidate set keeps the weights
    # intuitive (w=0.1 means ~10% recency influence relative to retrieval).
    if time_decay_weight > 0 and candidates:
        max_rrf = max(s for _, (s, _) in candidates) or 1.0
        decayed: list[tuple[int, float, set[str]]] = []
        now = time.time()
        for cid, (rrf_score, sources) in candidates:
            if cid not in hydrated:
                continue
            mtime = hydrated[cid][3]
            recency = _time_decay_factor(mtime, time_decay_half_life_days, now=now)
            normalized = rrf_score / max_rrf
            blended = (1 - time_decay_weight) * normalized + time_decay_weight * recency
            decayed.append((cid, blended, sources))
        decayed.sort(key=lambda x: -x[1])
        ordered_candidates = decayed
    else:
        ordered_candidates = [(cid, s, srcs) for cid, (s, srcs) in candidates]

    if reranker and len(ordered_candidates) > 1:
        cids: list[int] = []
        docs: list[str] = []
        for cid, _s, _src in ordered_candidates:
            if cid in hydrated:
                cids.append(cid)
                docs.append(hydrated[cid][2])
        rerank_pairs = reranker.rerank(query, docs, top_k=k)
        results: list[SearchResult] = []
        for orig_idx, score in rerank_pairs:
            cid = cids[orig_idx]
            path, idx, text, mtime = hydrated[cid]
            _, sources = fused[cid]
            results.append(
                SearchResult(
                    chunk_id=cid,
                    file_path=path,
                    chunk_index=idx,
                    text=text,
                    score=score,
                    sources=tuple(sorted(sources)),
                    mtime=mtime,
                    reranked=True,
                )
            )
        return results

    results = []
    for cid, score, sources in ordered_candidates[:k]:
        if cid not in hydrated:
            continue
        path, idx, text, mtime = hydrated[cid]
        results.append(
            SearchResult(
                chunk_id=cid,
                file_path=path,
                chunk_index=idx,
                text=text,
                score=score,
                sources=tuple(sorted(sources)),
                mtime=mtime,
            )
        )
    return results


def vector_only(
    conn: sqlite3.Connection,
    embedder: Embedder,
    query: str,
    k: int = 10,
    reranker: Reranker | None = None,
    rerank_overfetch: int = 50,
) -> list[SearchResult]:
    return hybrid_search(
        conn, embedder, query, k=k, alpha=1.0,
        reranker=reranker, rerank_overfetch=rerank_overfetch,
    )


def keyword_only(
    conn: sqlite3.Connection,
    embedder: Embedder,
    query: str,
    k: int = 10,
    reranker: Reranker | None = None,
    rerank_overfetch: int = 50,
) -> list[SearchResult]:
    return hybrid_search(
        conn, embedder, query, k=k, alpha=0.0,
        reranker=reranker, rerank_overfetch=rerank_overfetch,
    )
