"""Round 10 (#6) — single audit log for every AI action.

Every time an LLM call modifies user-visible state (a draft gets
generated, a task gets extracted, an email gets classified, an
insight gets surfaced), one row lands here. Then the user has a
single place to ask "why did the AI do X?" — both for debugging
and for trust.

Schema is deliberately wide (many nullable columns) so each feature
can record what's relevant without forcing a single shape. Cost is
recorded in cents to match the budget module's conventions; ``status``
covers success / fallback-to-local / failed.

Surfaces:
  - ``record_action`` — the helper every LLM call site invokes.
  - ``recent`` / ``by_kind`` — list helpers for the dashboard + CLI.
  - ``rollup_today`` — counts + cost grouped by feature, used by
    the spend transparency view (round-10 #3).

Bounded: ``trim_old`` keeps 30 days of rows so the table doesn't
grow without bound. Daemon calls it nightly.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import weakref as _weakref
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


# Schema-init cache, mirrors email_assist.
_SCHEMA_INITIALIZED: _weakref.WeakSet = _weakref.WeakSet()

# Round 13 fix (audit-found concurrency hazard) — every
# ``record_action`` call commits its own row, but the writer conn
# is shared across the daemon worker thread, dashboard worker
# threads, and the MCP server thread. Without serialisation, an
# audit ``commit()`` from thread A could commit another transaction
# that thread B had open, surprise-flushing partial writes.
#
# This RLock serialises ``record_action``'s INSERT + commit so the
# audit log can never accidentally flush an unrelated transaction.
# Reentrant because future code might end up calling record_action
# from within an audit-emitting LLM helper.
_AUDIT_WRITE_LOCK = threading.RLock()

# Default retention. 30 days of audit rows keeps the table small
# (sub-MB) while preserving enough history for "why did this draft
# get generated last week?" debugging.
_DEFAULT_RETENTION_DAYS = 30


def _ensure_schema(conn: sqlite3.Connection) -> None:
    try:
        if conn in _SCHEMA_INITIALIZED:
            return
    except TypeError:
        pass
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS ai_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            kind TEXT NOT NULL,            -- 'draft' | 'analyze' | 'classify'
                                           -- | 'thanks_draft' | 'extract_promise'
                                           -- | 'voice_profile' | 'voice_critique'
                                           -- | 'summary' | 'briefing' | 'tag'
                                           -- | 'chat' | 'fallback_chat'
            feature TEXT NOT NULL,         -- shorter slug for budget bucket
            model TEXT,                    -- 'claude-sonnet-4-6' / 'llama3.1' / etc
            status TEXT NOT NULL,          -- 'success' | 'fallback_local'
                                           -- | 'budget_exceeded' | 'api_error'
                                           -- | 'no_provider' | 'parse_error'
            file_id INTEGER,               -- which file triggered (nullable FK-less)
            person_id INTEGER,             -- when applicable
            draft_id INTEGER,              -- when the action wrote a draft row
            prompt_chars INTEGER NOT NULL DEFAULT 0,
            response_chars INTEGER NOT NULL DEFAULT 0,
            cents REAL NOT NULL DEFAULT 0,
            summary TEXT,                  -- one-line human-readable
                                           --  ('drafted reply to Sarah', etc)
            error TEXT,                    -- non-null on status='*_error'
            extra_json TEXT                -- feature-specific extras
        );
        CREATE INDEX IF NOT EXISTS idx_ai_actions_ts
            ON ai_actions(ts DESC);
        CREATE INDEX IF NOT EXISTS idx_ai_actions_kind_ts
            ON ai_actions(kind, ts DESC);
        CREATE INDEX IF NOT EXISTS idx_ai_actions_file_id
            ON ai_actions(file_id);
    """)
    conn.commit()
    try:
        _SCHEMA_INITIALIZED.add(conn)
    except TypeError:
        pass


@dataclass
class AIAction:
    id: int
    ts: float
    kind: str
    feature: str
    model: str
    status: str
    file_id: int | None
    person_id: int | None
    draft_id: int | None
    prompt_chars: int
    response_chars: int
    cents: float
    summary: str
    error: str
    extra: dict = field(default_factory=dict)


def record_action(
    conn: sqlite3.Connection,
    *,
    kind: str,
    feature: str,
    model: str = "",
    status: str = "success",
    file_id: int | None = None,
    person_id: int | None = None,
    draft_id: int | None = None,
    prompt_chars: int = 0,
    response_chars: int = 0,
    cents: float = 0.0,
    summary: str = "",
    error: str = "",
    extra: dict | None = None,
) -> int:
    """Persist one audit row. Best-effort — a logging failure should
    NEVER take down the calling LLM pipeline, so all errors are
    swallowed + logged at debug level.

    Round 13 fix — guarded by ``_AUDIT_WRITE_LOCK`` so the per-row
    commit can't accidentally flush an unrelated transaction
    held by another thread on the shared writer conn.

    Returns the new row id, or 0 on failure."""
    try:
        with _AUDIT_WRITE_LOCK:
            _ensure_schema(conn)
            cur = conn.execute(
                "INSERT INTO ai_actions"
                "(ts, kind, feature, model, status, file_id, person_id, "
                " draft_id, prompt_chars, response_chars, cents, summary, "
                " error, extra_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "RETURNING id",
                (
                    time.time(), kind, feature, model or "", status,
                    file_id, person_id, draft_id,
                    int(prompt_chars), int(response_chars), float(cents),
                    summary[:500] if summary else "",
                    error[:500] if error else "",
                    json.dumps(extra) if extra else None,
                ),
            )
            row = cur.fetchone()
            conn.commit()
            return int(row["id"]) if row else 0
    except Exception as e:  # noqa: BLE001
        log.debug("ai_audit: record failed: %s", e)
        return 0


def recent(
    conn: sqlite3.Connection,
    *,
    limit: int = 50,
    kind: str | None = None,
) -> list[AIAction]:
    """List most-recent actions, optionally filtered by kind."""
    _ensure_schema(conn)
    if kind:
        rows = conn.execute(
            "SELECT * FROM ai_actions WHERE kind = ? "
            "ORDER BY ts DESC LIMIT ?",
            (kind, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM ai_actions ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_action(r) for r in rows]


def by_kind(
    conn: sqlite3.Connection,
    *,
    days: int = 7,
) -> dict[str, int]:
    """Counts grouped by kind over the last N days. Powers the
    audit-page summary chips."""
    _ensure_schema(conn)
    cutoff = time.time() - days * 86400
    rows = conn.execute(
        "SELECT kind, COUNT(*) AS n FROM ai_actions "
        "WHERE ts >= ? GROUP BY kind ORDER BY n DESC",
        (cutoff,),
    ).fetchall()
    return {r["kind"]: int(r["n"]) for r in rows}


def rollup_today(
    conn: sqlite3.Connection,
) -> dict[str, dict]:
    """Per-feature roll-up of counts + cents over the last 24h.
    Used by the spend transparency view (#3)."""
    _ensure_schema(conn)
    cutoff = time.time() - 24 * 3600
    rows = conn.execute(
        "SELECT feature, COUNT(*) AS n, SUM(cents) AS cents, "
        "       SUM(prompt_chars) AS pc, SUM(response_chars) AS rc "
        "FROM ai_actions WHERE ts >= ? "
        "GROUP BY feature ORDER BY cents DESC",
        (cutoff,),
    ).fetchall()
    return {
        r["feature"]: {
            "n": int(r["n"]),
            "cents": float(r["cents"] or 0),
            "prompt_chars": int(r["pc"] or 0),
            "response_chars": int(r["rc"] or 0),
        }
        for r in rows
    }


def cost_for_window(
    conn: sqlite3.Connection,
    *,
    file_id: int,
    around_ts: float,
    window_seconds: float = 90.0,
) -> dict:
    """Round 10 (#3) — sum AI cost + call count for a specific
    file in a time window around ``around_ts``. Used to compute
    per-draft cost on the /drafts page (a draft is the result of
    several LLM calls — analyze + draft + critique + maybe regen
    — all keyed to the same file_id within a few seconds).
    """
    _ensure_schema(conn)
    lo = around_ts - window_seconds
    hi = around_ts + window_seconds
    row = conn.execute(
        "SELECT COUNT(*) AS n, SUM(cents) AS cents "
        "FROM ai_actions "
        "WHERE file_id = ? AND ts BETWEEN ? AND ?",
        (file_id, lo, hi),
    ).fetchone()
    return {
        "n": int(row["n"] or 0),
        "cents": float(row["cents"] or 0),
    }


def trim_old(
    conn: sqlite3.Connection,
    *,
    keep_days: int = _DEFAULT_RETENTION_DAYS,
) -> int:
    """Daemon hook: drop rows older than ``keep_days``. Returns the
    count of rows removed."""
    _ensure_schema(conn)
    cutoff = time.time() - keep_days * 86400
    cur = conn.execute("DELETE FROM ai_actions WHERE ts < ?", (cutoff,))
    conn.commit()
    return cur.rowcount


def _row_to_action(row) -> AIAction:
    extra: dict = {}
    try:
        if row["extra_json"]:
            parsed = json.loads(row["extra_json"])
            if isinstance(parsed, dict):
                extra = parsed
    except (TypeError, ValueError):
        extra = {}
    return AIAction(
        id=int(row["id"]),
        ts=float(row["ts"]),
        kind=row["kind"],
        feature=row["feature"],
        model=row["model"] or "",
        status=row["status"],
        file_id=row["file_id"],
        person_id=row["person_id"],
        draft_id=row["draft_id"],
        prompt_chars=int(row["prompt_chars"] or 0),
        response_chars=int(row["response_chars"] or 0),
        cents=float(row["cents"] or 0),
        summary=row["summary"] or "",
        error=row["error"] or "",
        extra=extra,
    )
