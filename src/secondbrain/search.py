"""Hybrid retrieval: vector + BM25 fused via Reciprocal Rank Fusion.

Beyond vanilla RRF this module layers in several quality features:

- **Adaptive alpha**: per-query tuning of vector vs. keyword weight based on
  cheap heuristics (capitalization ratio, ID-like tokens, query length).
- **Time-decay**: a gentle recency bonus that nudges fresh files up without
  drowning out long-lived reference material. Configurable half-life.
- **HyDE**: for vague conceptual queries, ask Claude to draft a hypothetical
  answer and embed *that* — the embedding lives in the same neighborhood as
  real answers, so vector recall jumps. Falls back gracefully if no
  ANTHROPIC_API_KEY.
- **Source-aware boost**: lift personal-content paths (notes, journals)
  above passive downloads. Config-driven path patterns.
"""

from __future__ import annotations

import logging
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass

from .budget import BudgetExceededError, check_budget, record_usage
from .config import Config
from .db import serialize_f32
from .embedder import Embedder
from .reranker import Reranker

log = logging.getLogger(__name__)


_HYDE_SYSTEM_PROMPT = """\
The user is searching their personal knowledge base. Draft a short, plausible \
answer to their question — the kind of paragraph that would appear in a \
document, transcript, or note in their files if the answer existed there. \
Use natural language and the terminology you'd expect a real source document \
to use. Do not preface ("Here's a hypothetical answer"). Do not hedge \
("the answer might be"). Just write 2-4 sentences as though excerpting from \
a real source. Keep it grounded and specific even though it's hypothetical."""


def hyde_rewrite(
    cfg: Config,
    query: str,
    model: str = "claude-haiku-4-5",
    max_tokens: int = 256,
) -> str:
    """Generate a hypothetical answer for the query, suitable for embedding.

    Returns the original query unchanged if the Anthropic SDK isn't installed,
    ANTHROPIC_API_KEY isn't set, or the daily Anthropic cap is hit — HyDE is
    a quality-bump, never a hard dependency. Errors are logged but never
    raised; a vague hypothetical is better than a search failure.

    Every call goes through ``check_budget`` and ``record_usage`` so the
    spend ledger reflects HyDE traffic. Without this, a search loop with
    HyDE enabled was silently uncapped and unaccounted-for.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return query
    try:
        import anthropic
    except ImportError:
        return query

    try:
        check_budget(cfg, "anthropic")
    except BudgetExceededError as e:
        log.warning("HyDE skipped: %s", e)
        return query

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=_HYDE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": query}],
        )
        text = "\n".join(b.text for b in response.content if b.type == "text").strip()
        try:
            record_usage(
                cfg, "anthropic", model,
                input_tokens=getattr(response.usage, "input_tokens", 0),
                output_tokens=getattr(response.usage, "output_tokens", 0),
                note="hyde",
            )
        except Exception as e:  # noqa: BLE001
            log.warning("HyDE usage recording failed: %s", e)
        return text or query
    except Exception as e:
        log.warning("HyDE rewrite failed, falling back to raw query: %s", e)
        return query


def should_use_hyde(query: str) -> bool:
    """Heuristic: HyDE helps most on conceptual / vague queries; it adds
    latency + cost on simple keyword lookups where it doesn't help.

    Apply when:
      - query is at least 4 words (gives Claude something to work with), AND
      - query contains a question mark OR a "what/how/why/when/who/where" lead
        OR is at least 8 words long (likely descriptive/conceptual).
    """
    tokens = query.split()
    if len(tokens) < 4:
        return False
    q_low = query.lower().strip()
    if "?" in query:
        return True
    if any(q_low.startswith(w + " ") for w in ("what", "how", "why", "when", "who", "where")):
        return True
    return len(tokens) >= 8

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
    start_offset: int | None = None  # byte offset of this chunk in the original file
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


def _path_score_multiplier(
    path: str,
    personal_prefixes: tuple[str, ...],
    personal_boost: float,
    download_prefixes: tuple[str, ...],
    download_demote: float,
) -> float:
    """Boost or demote a result based on where its source file lives.

    Files in user-curated locations (Documents, notes folders) usually
    contain higher-signal-per-token than passively-downloaded content.
    Multiplier is applied to the blended RRF/recency score; reranker
    runs after, so this only nudges which candidates the reranker sees.
    """
    if not path:
        return 1.0
    p_low = path.replace("\\", "/").lower()
    for prefix in personal_prefixes:
        if prefix and prefix.lower() in p_low:
            return personal_boost
    for prefix in download_prefixes:
        if prefix and prefix.lower() in p_low:
            return download_demote
    return 1.0


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
) -> dict[int, tuple[str, int, str, float, int | None]]:
    """Returns {chunk_id: (path, chunk_index, text, mtime, start_offset)}."""
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" * len(chunk_ids))
    rows = conn.execute(
        f"SELECT c.id, c.chunk_index, c.text, c.start_offset, f.path, f.mtime "
        f"FROM chunks c JOIN files f ON f.id = c.file_id "
        f"WHERE c.id IN ({placeholders})",
        chunk_ids,
    ).fetchall()
    return {
        r["id"]: (r["path"], r["chunk_index"], r["text"], r["mtime"], r["start_offset"])
        for r in rows
    }


def _eligible_chunk_ids(
    conn: sqlite3.Connection,
    path_prefix: str | None,
    kind: str | None,
    since_days: int | None,
) -> set[int] | None:
    """Pre-compute the set of chunk_ids matching the filter constraints.

    Returns None if no filter is applied (callers should skip the filter step).
    """
    where: list[str] = []
    params: list = []
    if path_prefix:
        where.append("REPLACE(f.path, '\\', '/') LIKE ?")
        params.append(path_prefix.replace("\\", "/").rstrip("/") + "%")
    if kind:
        where.append("f.kind = ?")
        params.append(kind)
    if since_days is not None:
        where.append("f.mtime >= ?")
        params.append(time.time() - since_days * 86400)
    if not where:
        return None
    sql = (
        "SELECT c.id FROM chunks c JOIN files f ON f.id = c.file_id "
        f"WHERE {' AND '.join(where)}"
    )
    return {row["id"] for row in conn.execute(sql, params).fetchall()}


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
    path_prefix: str | None = None,
    kind: str | None = None,
    since_days: int | None = None,
    use_hyde: bool = False,
    hyde_model: str = "claude-haiku-4-5",
    personal_prefixes: tuple[str, ...] = (),
    personal_boost: float = 1.0,
    download_prefixes: tuple[str, ...] = (),
    download_demote: float = 1.0,
    cfg: Config | None = None,
) -> list[SearchResult]:
    """Run hybrid search and return up to k merged results.

    - ``path_prefix`` / ``kind`` / ``since_days`` filter results to a folder,
      file kind ('document' / 'code' / 'audio_video' / 'image' / 'url'), or
      a recency window (in days). When any filter is set we over-fetch to
      compensate for what the filter drops.
    - When ``use_adaptive_alpha`` is set, ``alpha`` is the default and may be
      overridden per-query (long prose -> vector, IDs -> BM25).
    - When ``time_decay_weight > 0``, ranking blends in a recency bonus.
    - ``use_hyde`` rewrites the query into a hypothetical answer via Claude
      Haiku before embedding for the vector path. BM25 still uses the raw
      query (term-frequency benefits from real terms, not synthetic ones).
    - ``personal_prefixes`` / ``download_prefixes`` boost or demote results
      whose source path matches the patterns. Applied to the blended score
      before reranking.
    - When ``reranker`` is supplied, the top ``rerank_overfetch`` candidates
      are reranked by a cross-encoder before truncating to k.
    """
    effective_alpha = alpha if alpha is not None else 0.5
    if use_adaptive_alpha:
        effective_alpha = adaptive_alpha(query, default=effective_alpha)

    eligible = _eligible_chunk_ids(conn, path_prefix, kind, since_days)
    has_filter = eligible is not None
    over_factor = 5 if has_filter else 1
    candidate_count = max(
        (rerank_overfetch if reranker else k * 3) * over_factor, 30
    )

    # HyDE: embed a hypothetical answer instead of the raw query when the
    # query is conceptual enough to benefit. BM25 always uses raw query.
    # Requires ``cfg`` so the call goes through the budget cap; if cfg is
    # missing (legacy callers), we silently skip HyDE rather than risk an
    # uncapped Anthropic call.
    if use_hyde and cfg is not None and should_use_hyde(query):
        hypothetical = hyde_rewrite(cfg, query, model=hyde_model)
        q_emb = embedder.embed_query(hypothetical)
    else:
        q_emb = embedder.embed_query(query)
    vec = _vector_search(conn, q_emb, candidate_count)
    fts = _fts_search(conn, query, candidate_count)

    if eligible is not None:
        vec = [(cid, d) for cid, d in vec if cid in eligible]
        fts = [(cid, s) for cid, s in fts if cid in eligible]

    fused = _rrf_merge(vec, fts, alpha=effective_alpha)

    candidates = sorted(fused.items(), key=lambda kv: -kv[1][0])[:candidate_count]
    chunk_ids = [cid for cid, _ in candidates]
    hydrated = _hydrate(conn, chunk_ids)

    # Apply time-decay to the RRF score before reranking. We blend by weight w:
    #   blended = (1 - w) * normalized_rrf + w * recency
    # Normalising RRF to [0, 1] within this candidate set keeps the weights
    # intuitive (w=0.1 means ~10% recency influence relative to retrieval).
    apply_path_boost = (
        (personal_prefixes and personal_boost != 1.0)
        or (download_prefixes and download_demote != 1.0)
    )
    if time_decay_weight > 0 and candidates:
        max_rrf = max(s for _, (s, _) in candidates) or 1.0
        decayed: list[tuple[int, float, set[str]]] = []
        now = time.time()
        for cid, (rrf_score, sources) in candidates:
            if cid not in hydrated:
                continue
            path, _, _, mtime, _ = hydrated[cid]
            recency = _time_decay_factor(mtime, time_decay_half_life_days, now=now)
            normalized = rrf_score / max_rrf
            blended = (1 - time_decay_weight) * normalized + time_decay_weight * recency
            if apply_path_boost:
                blended *= _path_score_multiplier(
                    path, personal_prefixes, personal_boost,
                    download_prefixes, download_demote,
                )
            decayed.append((cid, blended, sources))
        decayed.sort(key=lambda x: -x[1])
        ordered_candidates = decayed
    elif apply_path_boost and candidates:
        boosted: list[tuple[int, float, set[str]]] = []
        for cid, (rrf_score, sources) in candidates:
            if cid not in hydrated:
                continue
            path = hydrated[cid][0]
            mult = _path_score_multiplier(
                path, personal_prefixes, personal_boost,
                download_prefixes, download_demote,
            )
            boosted.append((cid, rrf_score * mult, sources))
        boosted.sort(key=lambda x: -x[1])
        ordered_candidates = boosted
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
            # Defend against an out-of-range index from the reranker (a
            # truncated `documents` list, an SDK bug, or a stale cached
            # response): silently skip rather than crash the search.
            if orig_idx < 0 or orig_idx >= len(cids):
                continue
            cid = cids[orig_idx]
            if cid not in hydrated:
                continue
            path, idx, text, mtime, start_offset = hydrated[cid]
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
                    start_offset=start_offset,
                    reranked=True,
                )
            )
        return results

    results = []
    for cid, score, sources in ordered_candidates[:k]:
        if cid not in hydrated:
            continue
        path, idx, text, mtime, start_offset = hydrated[cid]
        results.append(
            SearchResult(
                chunk_id=cid,
                file_path=path,
                chunk_index=idx,
                text=text,
                score=score,
                sources=tuple(sorted(sources)),
                mtime=mtime,
                start_offset=start_offset,
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
