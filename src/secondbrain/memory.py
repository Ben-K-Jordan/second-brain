"""Phase 86 + 87: cross-conversation memory + temporal queries.

Phase 86 — Cross-conversation memory
  Today's chat agent has memory only within a single conversation.
  This module persists distilled facts ("Ben works on a personal
  knowledge base", "prefers Sonnet 4.6 for tool use", "uses Voyage
  embeddings") so the next conversation starts informed.

  Two extraction paths:
    1. **Auto-distill** at the end of a conversation: the agent
       summarises the conversation into 0-N memory candidates.
    2. **Explicit teach**: ``secondbrain memory remember "<fact>"``
       lets the user pin context manually.

  Recall: ``most_relevant_memories(query, k)`` does keyword-overlap
  scoring over keys + content. Cheap (no embedding cost) and good
  enough for the small N (typically dozens-to-hundreds of memories).

Phase 87 — Temporal queries
  "What did I know about transformers 6 months ago?" — answers from
  the index *as it existed then*, not as it is now.

  Implementation: weekly index snapshots persist the set of file_ids
  present at each cutoff. A temporal query finds the closest
  preceding snapshot and filters its result set to files that
  existed at that point.

  Limitation: this is a *file-existence* time machine, not a content
  one. If a file was edited since, you see today's content with
  yesterday's eligibility. For full content-history we'd need to
  preserve chunk versions per snapshot — much more storage. The
  file-existence model is the 90/10 sweet spot.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)


# ---- Tunables ---------------------------------------------------------

# Snapshots taken weekly is plenty for "what did I know N months ago"
# queries — finer granularity is wasted space.
_SNAPSHOT_INTERVAL_DAYS = 7
# Cap memories returned to chat to keep prompt token usage bounded.
_MAX_MEMORIES_IN_PROMPT = 20
# Memories under this confidence get filtered out of recall (not
# deleted — user might still tweak them).
_MIN_RECALL_CONFIDENCE = 0.3


# ---- Data shapes -----------------------------------------------------

@dataclass
class Memory:
    id: int
    key: str
    content: str
    kind: str
    source_conversation_id: int | None
    created_at: float
    last_referenced_at: float | None
    reference_count: int
    confidence: float


@dataclass
class Snapshot:
    id: int
    taken_at: float
    file_ids: set[int]
    label: str | None
    n_files: int


# ============================ Phase 86: memories ======================

def remember(
    conn: sqlite3.Connection,
    key: str, content: str,
    *,
    kind: str = "fact",
    source_conversation_id: int | None = None,
    confidence: float = 0.9,
) -> int:
    """Persist a memory. Idempotent on key — same key updates content
    + bumps confidence.

    Args:
        key: short topic anchor; lowercase preferred.
        content: the actual fact to remember.
        kind: 'fact' | 'preference' | 'context'.
        confidence: 0..1. Manual `remember` calls default high (0.9);
            auto-distillation may pass lower values (e.g. 0.6).

    Returns the memory id.
    """
    key = (key or "").strip().lower()
    content = (content or "").strip()
    if not key or not content:
        raise ValueError("key + content must be non-empty")
    if kind not in ("fact", "preference", "context"):
        raise ValueError(f"unknown kind {kind!r}")
    confidence = max(0.0, min(1.0, float(confidence)))
    n = time.time()
    cur = conn.execute(
        "INSERT INTO chat_memories"
        "(key, content, kind, source_conversation_id, "
        " created_at, confidence) "
        "VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET "
        "  content = excluded.content, "
        "  kind = excluded.kind, "
        "  confidence = MAX(confidence, excluded.confidence) "
        "RETURNING id",
        (key, content, kind, source_conversation_id, n, confidence),
    )
    rid = int(cur.fetchone()["id"])
    conn.commit()
    return rid


def forget(conn: sqlite3.Connection, key: str) -> bool:
    """Hard-delete a memory by key."""
    cur = conn.execute(
        "DELETE FROM chat_memories WHERE key = ?",
        (key.strip().lower(),),
    )
    conn.commit()
    return cur.rowcount > 0


def get_memory(
    conn: sqlite3.Connection, key: str,
) -> Memory | None:
    row = conn.execute(
        "SELECT * FROM chat_memories WHERE key = ?",
        (key.strip().lower(),),
    ).fetchone()
    return _row_to_memory(row) if row else None


def list_memories(
    conn: sqlite3.Connection, *,
    kind: str | None = None, limit: int = 100,
) -> list[Memory]:
    if kind:
        rows = conn.execute(
            "SELECT * FROM chat_memories WHERE kind = ? "
            "ORDER BY confidence DESC, created_at DESC LIMIT ?",
            (kind, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM chat_memories "
            "ORDER BY confidence DESC, created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_memory(r) for r in rows]


# Token tokeniser: lowercase alphanumeric runs. Cheap; close enough
# to what's useful for memory keyword matching.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def most_relevant_memories(
    conn: sqlite3.Connection, query: str,
    *, k: int = 5,
) -> list[Memory]:
    """Pull the k memories most relevant to a query. Scoring:
    fraction of query tokens that appear in the memory's key + content.

    Cheap because the memory table stays small (typically <1000 rows).
    Promotes most-recently-referenced on ties.
    """
    q_tokens = set(_TOKEN_RE.findall((query or "").lower()))
    if not q_tokens:
        return []
    rows = conn.execute(
        "SELECT * FROM chat_memories WHERE confidence >= ?",
        (_MIN_RECALL_CONFIDENCE,),
    ).fetchall()
    scored: list[tuple[float, float, sqlite3.Row]] = []
    for r in rows:
        haystack = f"{r['key']} {r['content']}".lower()
        h_tokens = set(_TOKEN_RE.findall(haystack))
        if not h_tokens:
            continue
        overlap = len(q_tokens & h_tokens)
        if overlap == 0:
            continue
        score = overlap / len(q_tokens)
        # Tie-break: more recent reference first.
        recency = r["last_referenced_at"] or 0.0
        scored.append((score, recency, r))
    scored.sort(key=lambda t: (-t[0], -t[1]))
    return [_row_to_memory(r) for _score, _r, r in scored[:k]]


def mark_referenced(
    conn: sqlite3.Connection, memory_ids: list[int],
) -> int:
    """Bump the reference count + last_referenced_at for memories
    that the chat agent just used. Helps prioritise active context
    in subsequent recalls."""
    if not memory_ids:
        return 0
    placeholders = ",".join("?" * len(memory_ids))
    cur = conn.execute(
        f"UPDATE chat_memories SET "
        f"  last_referenced_at = ?, "
        f"  reference_count = reference_count + 1 "
        f"WHERE id IN ({placeholders})",
        [time.time(), *memory_ids],
    )
    conn.commit()
    return cur.rowcount


def render_memories_for_prompt(
    memories: list[Memory], *, header: str = "Things to remember about the user",
) -> str:
    """Format a list of memories as a system-prompt block. Empty list
    → empty string (so the chat path can safely concatenate)."""
    if not memories:
        return ""
    lines = [f"## {header}", ""]
    for m in memories[:_MAX_MEMORIES_IN_PROMPT]:
        prefix = {
            "fact": "·", "preference": "★", "context": "◊",
        }.get(m.kind, "·")
        lines.append(f"{prefix} {m.content}")
    return "\n".join(lines)


def _row_to_memory(row: sqlite3.Row) -> Memory:
    return Memory(
        id=int(row["id"]),
        key=row["key"],
        content=row["content"],
        kind=row["kind"],
        source_conversation_id=row["source_conversation_id"],
        created_at=row["created_at"],
        last_referenced_at=row["last_referenced_at"],
        reference_count=int(row["reference_count"]),
        confidence=float(row["confidence"]),
    )


# ============================ Phase 87: snapshots =====================

def take_snapshot(
    conn: sqlite3.Connection, *, label: str | None = None,
) -> int:
    """Capture today's set of file_ids. Returns the snapshot id."""
    rows = conn.execute(
        "SELECT id FROM files",
    ).fetchall()
    file_ids = sorted(int(r["id"]) for r in rows)
    cur = conn.execute(
        "INSERT INTO index_snapshots(taken_at, file_ids_json, label, n_files) "
        "VALUES (?, ?, ?, ?) RETURNING id",
        (time.time(), json.dumps(file_ids), label, len(file_ids)),
    )
    rid = int(cur.fetchone()["id"])
    conn.commit()
    log.info("memory: snapshot #%s captured (%d files)", rid, len(file_ids))
    return rid


def needs_snapshot(
    conn: sqlite3.Connection, *,
    interval_days: int = _SNAPSHOT_INTERVAL_DAYS,
) -> bool:
    """True iff no snapshot in the last ``interval_days``."""
    cutoff = time.time() - interval_days * 86400
    row = conn.execute(
        "SELECT 1 FROM index_snapshots WHERE taken_at >= ? LIMIT 1",
        (cutoff,),
    ).fetchone()
    return row is None


def list_snapshots(
    conn: sqlite3.Connection, *, limit: int = 20,
) -> list[Snapshot]:
    rows = conn.execute(
        "SELECT * FROM index_snapshots ORDER BY taken_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_row_to_snapshot(r) for r in rows]


def snapshot_at(
    conn: sqlite3.Connection, when: float,
) -> Snapshot | None:
    """Return the most recent snapshot taken at-or-before ``when``.
    Returns None when no snapshot exists at that time horizon."""
    row = conn.execute(
        "SELECT * FROM index_snapshots WHERE taken_at <= ? "
        "ORDER BY taken_at DESC LIMIT 1",
        (when,),
    ).fetchone()
    return _row_to_snapshot(row) if row else None


def filter_to_snapshot(
    conn: sqlite3.Connection, file_ids: list[int], *, when: float,
) -> list[int]:
    """Restrict a file_id list to those that existed in the snapshot
    closest to ``when``. Returns input unchanged when no snapshot
    covers the requested time."""
    snap = snapshot_at(conn, when)
    if snap is None:
        return file_ids
    snap_set = snap.file_ids
    return [fid for fid in file_ids if fid in snap_set]


def _row_to_snapshot(row: sqlite3.Row) -> Snapshot:
    try:
        ids = json.loads(row["file_ids_json"] or "[]")
    except (ValueError, TypeError):
        ids = []
    return Snapshot(
        id=int(row["id"]),
        taken_at=row["taken_at"],
        file_ids=set(int(x) for x in ids if isinstance(x, int)),
        label=row["label"],
        n_files=int(row["n_files"]),
    )


# ============================ daemon hooks ============================

def take_snapshot_if_due(
    cfg, conn: sqlite3.Connection,
) -> bool:
    """Daemon entrypoint. Take a snapshot if none in the last week."""
    if not needs_snapshot(conn):
        return False
    take_snapshot(conn)
    return True
