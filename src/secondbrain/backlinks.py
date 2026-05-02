"""Phase 52: auto-backlinks between docs — the Obsidian-style brain.

Search finds what you're looking for; backlinks find what you weren't.
After a doc lands in the index, we compute its top-K most-similar
existing docs by chunk-embedding nearness, and store the pairs in
``backlinks``. The ``/file/<path>`` dashboard view + a CLI command
surface them as "see also" context.

Why this matters more than search:

- You search when you have a question. Backlinks fire passively while
  you're already reading a doc — "here's the related stuff you forgot
  you wrote".
- They turn the index from a query box into something you wander.
- They're symmetric: linking new → old also lets old → new, so
  re-reading an ancient note surfaces today's fresh context.

How it works:

1. After ``replace_chunks`` finishes for a file (indexer + connector
   syncs both go through there), ``link_doc`` runs.
2. Pull every chunk-embedding for that file. Use each as a query
   against ``vec_chunks`` to fetch the K nearest *other-file* chunks.
3. Aggregate by destination ``file_id``: keep the BEST (lowest
   distance) match per candidate file. This is more robust than
   averaging — a long doc has many chunks, and the strongest match is
   what makes the link feel right.
4. Take the top K by distance, threshold-filter, and store both
   directions (src → dst AND dst → src) so retrieval is one-sided.

Cost: one vec query per chunk on the new doc. For a typical 5-chunk
doc that's 5 sub-millisecond queries. For a giant 200-chunk PDF we
cap the chunk fan-out so the cost stays bounded.

Bulk recompute is supported via ``rebuild_all`` for migrating an
existing brain forward.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass

from .db import serialize_f32

log = logging.getLogger(__name__)


# ---- Tunables ---------------------------------------------------------

# How many neighbours to keep per source file. Higher = more "see also"
# but lower precision. Five is the sweet spot for a personal brain.
_DEFAULT_K = 5
# Distance threshold. sqlite-vec returns L2 distance for unit-normalised
# vectors; for embeddings produced by Voyage / minilm both, this caps
# at sqrt(2) ≈ 1.414 (anti-correlated pair). Empirically docs that
# share a topic land below ~0.9; we set the threshold at 1.0 so we
# keep middling matches but cut the bottom 20% noise.
_DEFAULT_MAX_DISTANCE = 1.0
# Per-chunk fan-out: how many neighbours each chunk pulls before the
# file-level aggregation. Higher = more candidate diversity, more cost.
# Set to 50 because a long source doc's own chunks dominate the head
# of the nearest-neighbour list (distance 0 against themselves) — we
# need enough headroom so non-self matches survive the in-Python filter.
_PER_CHUNK_FETCH = 50
# Cap chunks scanned per source doc. A 500-chunk PDF doesn't need every
# chunk to find its neighbours; the first ~30 give plenty of signal and
# bound the cost.
_MAX_CHUNKS_PER_SOURCE = 30


@dataclass
class Backlink:
    """One direction of a similarity link."""
    src_file_id: int
    dst_file_id: int
    score: float           # sqlite-vec distance (LOWER is more similar)


@dataclass
class BacklinkView:
    """Hydrated form of a backlink — what the dashboard / CLI shows."""
    file_id: int
    path: str
    title: str             # H1 of the first chunk, falling back to path
    score: float           # raw distance
    similarity: float      # 1 / (1 + score) — handy for "0..1" UI

    @property
    def percent(self) -> int:
        """0..100 cosmetic score so users have a feel for relatedness."""
        return max(0, min(100, int(self.similarity * 100)))


# ============================ computation =============================

def link_doc(
    conn: sqlite3.Connection,
    file_id: int,
    *,
    k: int = _DEFAULT_K,
    max_distance: float = _DEFAULT_MAX_DISTANCE,
) -> int:
    """Compute neighbours for ``file_id`` and persist both directions.

    Idempotent: existing rows for (src, dst) and (dst, src) are
    overwritten with the new score. Returns the count of pair-rows
    written (so for K neighbours, expect 2K).

    Failures are caught and logged so a backlink computation issue
    can never take down an ingest.
    """
    try:
        neighbours = compute_neighbors(
            conn, file_id, k=k, max_distance=max_distance,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("backlinks: compute failed for file_id=%s: %s", file_id, e)
        return 0
    if not neighbours:
        return 0
    return record_backlinks(conn, file_id, neighbours)


def compute_neighbors(
    conn: sqlite3.Connection,
    file_id: int,
    *,
    k: int = _DEFAULT_K,
    max_distance: float = _DEFAULT_MAX_DISTANCE,
) -> list[tuple[int, float]]:
    """Return up to ``k`` nearest other-file ids with their best
    distance. Empty list when the file has no chunks (which can happen
    for empty docs / image-only files)."""
    embeddings = _source_chunk_embeddings(conn, file_id)
    if not embeddings:
        return []
    # Aggregate: per candidate file_id, keep the lowest distance seen.
    # Two-step query (vec MATCH alone, then chunks JOIN) mirrors the
    # search.py pattern — sqlite-vec's MATCH plays best as a leaf op.
    best: dict[int, float] = {}
    for emb in embeddings:
        vec_rows = conn.execute(
            "SELECT chunk_id, distance FROM vec_chunks "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (serialize_f32(emb), _PER_CHUNK_FETCH),
        ).fetchall()
        if not vec_rows:
            continue
        chunk_ids = [int(r["chunk_id"]) for r in vec_rows]
        # Map chunk_id → distance for the in-Python merge.
        dist_by_chunk = {
            int(r["chunk_id"]): float(r["distance"]) for r in vec_rows
        }
        placeholders = ",".join("?" * len(chunk_ids))
        chunk_rows = conn.execute(
            f"SELECT id, file_id FROM chunks WHERE id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        for cr in chunk_rows:
            other = int(cr["file_id"])
            if other == file_id:
                continue
            dist = dist_by_chunk.get(int(cr["id"]))
            if dist is None:
                continue
            cur = best.get(other)
            if cur is None or dist < cur:
                best[other] = dist
    if not best:
        return []
    # Sort ascending (lowest distance = most similar) then threshold.
    pairs = sorted(best.items(), key=lambda kv: kv[1])
    pairs = [(fid, d) for fid, d in pairs if d <= max_distance]
    return pairs[:k]


def _source_chunk_embeddings(
    conn: sqlite3.Connection, file_id: int,
) -> list[list[float]]:
    """Pull up to N chunk embeddings for the source file. We unpack
    sqlite-vec's float32 blob back to a Python list since vec0's MATCH
    expects the same encoding it emits.

    Capped by ``_MAX_CHUNKS_PER_SOURCE`` so very long docs don't fan
    out into thousands of vec queries.
    """
    rows = conn.execute(
        "SELECT v.embedding FROM vec_chunks v "
        "JOIN chunks c ON c.id = v.chunk_id "
        "WHERE c.file_id = ? "
        "ORDER BY c.chunk_index ASC LIMIT ?",
        (file_id, _MAX_CHUNKS_PER_SOURCE),
    ).fetchall()
    out: list[list[float]] = []
    for r in rows:
        blob = r["embedding"]
        if not blob:
            continue
        # sqlite-vec stores as little-endian float32. Each float = 4 bytes.
        n = len(blob) // 4
        if n == 0:
            continue
        import struct
        out.append(list(struct.unpack(f"<{n}f", blob)))
    return out


# ============================ persistence =============================

def record_backlinks(
    conn: sqlite3.Connection,
    file_id: int,
    neighbours: list[tuple[int, float]],
) -> int:
    """Persist forward + reverse pairs.

    For each (dst, score) in ``neighbours``:

      INSERT OR REPLACE INTO backlinks (src=file_id, dst=dst, score)
      INSERT OR REPLACE INTO backlinks (src=dst, dst=file_id, score)

    REPLACE so re-running on a re-ingest refreshes the score (e.g.
    when chunks change after editing the source doc).
    """
    n = time.time()
    rowcount = 0
    for dst_id, score in neighbours:
        # Forward.
        conn.execute(
            "INSERT INTO backlinks(src_file_id, dst_file_id, score, created_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(src_file_id, dst_file_id) DO UPDATE SET "
            "  score = excluded.score, created_at = excluded.created_at",
            (file_id, dst_id, score, n),
        )
        rowcount += 1
        # Reverse — make the link discoverable from either side.
        conn.execute(
            "INSERT INTO backlinks(src_file_id, dst_file_id, score, created_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(src_file_id, dst_file_id) DO UPDATE SET "
            "  score = excluded.score, created_at = excluded.created_at",
            (dst_id, file_id, score, n),
        )
        rowcount += 1
    conn.commit()
    return rowcount


def clear_backlinks_for(conn: sqlite3.Connection, file_id: int) -> int:
    """Delete every link touching ``file_id`` (both directions). Used
    by ``rebuild_all`` and called manually when a doc changes radically
    enough that old neighbours are wrong."""
    cur = conn.execute(
        "DELETE FROM backlinks WHERE src_file_id = ? OR dst_file_id = ?",
        (file_id, file_id),
    )
    conn.commit()
    return cur.rowcount


# ============================ retrieval ===============================

def get_backlinks(
    conn: sqlite3.Connection, file_id: int, limit: int = 10,
) -> list[BacklinkView]:
    """Return hydrated views of the top neighbours for a file. The
    dashboard's `/file/<path>` view + the CLI use this."""
    rows = conn.execute(
        "SELECT b.dst_file_id, b.score, f.path "
        "FROM backlinks b JOIN files f ON f.id = b.dst_file_id "
        "WHERE b.src_file_id = ? "
        "ORDER BY b.score ASC LIMIT ?",
        (file_id, limit),
    ).fetchall()
    out: list[BacklinkView] = []
    for r in rows:
        title = _doc_title(conn, int(r["dst_file_id"]), r["path"])
        score = float(r["score"])
        out.append(BacklinkView(
            file_id=int(r["dst_file_id"]),
            path=r["path"],
            title=title,
            score=score,
            similarity=1.0 / (1.0 + score),
        ))
    return out


def get_backlinks_for_path(
    conn: sqlite3.Connection, path: str, limit: int = 10,
) -> list[BacklinkView]:
    """Convenience for callers that have a path, not an id."""
    row = conn.execute(
        "SELECT id FROM files WHERE path = ?", (path,),
    ).fetchone()
    if row is None:
        return []
    return get_backlinks(conn, int(row["id"]), limit=limit)


def _doc_title(
    conn: sqlite3.Connection, file_id: int, path: str,
) -> str:
    """Best-effort H1 for the dashboard rendering. Same shape as
    daily_brief / tasks helpers but lives here so backlinks doesn't
    create a circular dependency."""
    row = conn.execute(
        "SELECT text FROM chunks WHERE file_id = ? "
        "ORDER BY chunk_index ASC LIMIT 1",
        (file_id,),
    ).fetchone()
    if row is None:
        return path
    for line in (row["text"] or "").splitlines():
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip() or path
    return path


# ========================= bulk recompute =============================

def rebuild_all(
    conn: sqlite3.Connection,
    *,
    k: int = _DEFAULT_K,
    max_distance: float = _DEFAULT_MAX_DISTANCE,
    on_progress=None,
) -> int:
    """Recompute every file's neighbours from scratch. Drops existing
    backlinks first so stale pairs (from before a docs schema change /
    embedder switch) get cleaned up.

    ``on_progress(done, total)`` fires once per file so a CLI / TUI
    can render a progress bar. Returns the total pair-rows written.
    """
    conn.execute("DELETE FROM backlinks")
    conn.commit()
    rows = conn.execute(
        "SELECT id FROM files ORDER BY id ASC",
    ).fetchall()
    total = len(rows)
    written = 0
    for i, r in enumerate(rows, 1):
        written += link_doc(
            conn, int(r["id"]), k=k, max_distance=max_distance,
        )
        if on_progress is not None:
            try:
                on_progress(i, total)
            except Exception:  # noqa: BLE001
                pass
    return written
