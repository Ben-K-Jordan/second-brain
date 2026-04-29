"""Hybrid retrieval: vector + BM25 fused via Reciprocal Rank Fusion."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from .db import serialize_f32
from .embedder import Embedder

RRF_K = 60  # RRF constant; 60 is the original paper's value


@dataclass
class SearchResult:
    chunk_id: int
    file_path: str
    chunk_index: int
    text: str
    score: float
    sources: tuple[str, ...]  # which retrievers matched: ("vector",), ("fts",), or both


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
) -> dict[int, tuple[str, int, str]]:
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT c.id, c.chunk_index, c.text, f.path "
        f"FROM chunks c JOIN files f ON f.id = c.file_id "
        f"WHERE c.id IN ({placeholders})",
        chunk_ids,
    ).fetchall()
    return {r["id"]: (r["path"], r["chunk_index"], r["text"]) for r in rows}


def hybrid_search(
    conn: sqlite3.Connection,
    embedder: Embedder,
    query: str,
    k: int = 10,
    alpha: float = 0.5,
) -> list[SearchResult]:
    """Run hybrid search and return up to k merged results."""
    over_fetch = max(k * 3, 30)
    q_emb = embedder.embed_query(query)
    vec = _vector_search(conn, q_emb, over_fetch)
    fts = _fts_search(conn, query, over_fetch)
    fused = _rrf_merge(vec, fts, alpha=alpha)

    top = sorted(fused.items(), key=lambda kv: -kv[1][0])[:k]
    chunk_ids = [cid for cid, _ in top]
    hydrated = _hydrate(conn, chunk_ids)

    results: list[SearchResult] = []
    for cid, (score, sources) in top:
        if cid not in hydrated:
            continue
        path, idx, text = hydrated[cid]
        results.append(
            SearchResult(
                chunk_id=cid,
                file_path=path,
                chunk_index=idx,
                text=text,
                score=score,
                sources=tuple(sorted(sources)),
            )
        )
    return results


def vector_only(
    conn: sqlite3.Connection, embedder: Embedder, query: str, k: int = 10
) -> list[SearchResult]:
    return hybrid_search(conn, embedder, query, k=k, alpha=1.0)


def keyword_only(
    conn: sqlite3.Connection, embedder: Embedder, query: str, k: int = 10
) -> list[SearchResult]:
    return hybrid_search(conn, embedder, query, k=k, alpha=0.0)
