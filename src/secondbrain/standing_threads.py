"""Round 19 (Phase EA-8) — standing-thread tracker.

A "standing thread" is a long-running conversation/topic between
the user and one or more people: an ongoing project, a recurring
client, a research topic that's spanned weeks. An EA holds a
running summary of each one in their head.

This module identifies standing threads by detecting clusters of
correspondence around the same set of people and entities, and —
optionally — summarizes each via Haiku.

Detection heuristic (cheap, no LLM):
  - Group emails / messages by the set of (people_mentioned).
  - A "thread" is a group with ≥ ``MIN_MESSAGES`` and a span
    of ≥ ``MIN_DAYS`` between first and last.
  - Tag with the dominant entity (most frequent ORG/PROJECT/PRODUCT).

Summarization (one Haiku call per thread, on demand):
  - Concatenate the most-recent N message previews, run through
    Sonnet for: 2-sentence summary, list of decisions, list of
    open questions.

Persisted in ``standing_threads`` so we don't re-detect/re-summarize.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import weakref as _weakref
from dataclasses import asdict, dataclass

from .config import Config

log = logging.getLogger(__name__)

_SCHEMA_INITIALIZED: _weakref.WeakSet = _weakref.WeakSet()

MIN_MESSAGES = 5
MIN_DAYS = 14
_SUMMARY_MODEL = "claude-haiku-4-5"
_SUMMARY_MAX_INPUT_CHARS = 15000


@dataclass
class StandingThread:
    id: int
    topic: str           # the dominant entity / "with X & Y"
    person_ids: list[int]
    summary_md: str
    n_messages: int
    first_message_at: float
    last_message_at: float
    last_summarized_at: float | None
    decisions: list[str]
    open_questions: list[str]
    file_ids: list[int]   # the constituent docs

    def to_dict(self) -> dict:
        return asdict(self)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    try:
        if conn in _SCHEMA_INITIALIZED:
            return
    except TypeError:
        pass
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS standing_threads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            person_ids_json TEXT NOT NULL DEFAULT '[]',
            summary_md TEXT NOT NULL DEFAULT '',
            n_messages INTEGER NOT NULL DEFAULT 0,
            first_message_at REAL,
            last_message_at REAL,
            last_summarized_at REAL,
            decisions_json TEXT NOT NULL DEFAULT '[]',
            open_questions_json TEXT NOT NULL DEFAULT '[]',
            file_ids_json TEXT NOT NULL DEFAULT '[]',
            -- Idempotency: same (sorted person_ids + topic) → same row.
            dedup_key TEXT NOT NULL UNIQUE
        );
        CREATE INDEX IF NOT EXISTS idx_standing_threads_last_msg
            ON standing_threads(last_message_at DESC);
    """)
    conn.commit()
    try:
        _SCHEMA_INITIALIZED.add(conn)
    except TypeError:
        pass


def _row_to_thread(row) -> StandingThread:
    return StandingThread(
        id=int(row["id"]),
        topic=row["topic"] or "",
        person_ids=json.loads(row["person_ids_json"] or "[]"),
        summary_md=row["summary_md"] or "",
        n_messages=int(row["n_messages"]),
        first_message_at=float(row["first_message_at"] or 0),
        last_message_at=float(row["last_message_at"] or 0),
        last_summarized_at=(
            float(row["last_summarized_at"])
            if row["last_summarized_at"] else None
        ),
        decisions=json.loads(row["decisions_json"] or "[]"),
        open_questions=json.loads(row["open_questions_json"] or "[]"),
        file_ids=json.loads(row["file_ids_json"] or "[]"),
    )


def _redact(text: str) -> str:
    try:
        from .safety import redact_text
        return redact_text(text or "")
    except ImportError:
        return text or ""


# ============================ detection =============================


def detect_threads(
    conn: sqlite3.Connection,
    *,
    days: int = 60,
    min_messages: int = MIN_MESSAGES,
    min_days: int = MIN_DAYS,
) -> int:
    """Walk recent emails / messages, group by (frozenset of person
    ids mentioned), and persist clusters that look like standing
    threads. Returns the number of threads created/updated."""
    _ensure_schema(conn)
    cutoff = time.time() - days * 86400
    try:
        rows = conn.execute(
            "SELECT pm.person_id, pm.file_id, pm.mtime "
            "FROM person_mentions pm "
            "JOIN files f ON f.id = pm.file_id "
            "WHERE f.indexed_at >= ? "
            "  AND (f.kind = 'email' OR f.kind = 'message') "
            "ORDER BY pm.file_id",
            (cutoff,),
        ).fetchall()
    except sqlite3.OperationalError:
        return 0
    # Group by file_id → set of person_ids.
    file_people: dict[int, set[int]] = {}
    file_mtime: dict[int, float] = {}
    for r in rows:
        fid = int(r["file_id"])
        file_people.setdefault(fid, set()).add(int(r["person_id"]))
        file_mtime[fid] = float(r["mtime"])
    # Group files by frozenset of person_ids — files with the same
    # exact set form a "thread".
    cluster: dict[frozenset, list[int]] = {}
    for fid, pids in file_people.items():
        if not pids:
            continue
        cluster.setdefault(frozenset(pids), []).append(fid)
    n_persisted = 0
    for pids_fs, fids in cluster.items():
        if len(fids) < min_messages:
            continue
        mtimes = [file_mtime[f] for f in fids]
        first_ts = min(mtimes)
        last_ts = max(mtimes)
        span_days = (last_ts - first_ts) / 86400.0
        if span_days < min_days:
            continue
        # Topic: dominant entity across these files (best-effort).
        topic = _dominant_entity(conn, fids) or _participants_topic(
            conn, pids_fs,
        )
        person_ids = sorted(pids_fs)
        dedup = _dedup_key(topic, person_ids)
        try:
            cur = conn.execute(
                "INSERT INTO standing_threads"
                "(topic, person_ids_json, n_messages, first_message_at, "
                " last_message_at, file_ids_json, dedup_key) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    _redact(topic),
                    json.dumps(person_ids),
                    len(fids), first_ts, last_ts,
                    json.dumps(sorted(fids)),
                    dedup,
                ),
            )
            conn.commit()
            if cur.rowcount > 0:
                n_persisted += 1
        except sqlite3.IntegrityError:
            # Already exists — refresh n_messages, last_message_at, file_ids.
            conn.execute(
                "UPDATE standing_threads SET "
                "n_messages = ?, last_message_at = ?, "
                "first_message_at = ?, file_ids_json = ? "
                "WHERE dedup_key = ?",
                (
                    len(fids), last_ts, first_ts,
                    json.dumps(sorted(fids)), dedup,
                ),
            )
            conn.commit()
    return n_persisted


def _dominant_entity(
    conn: sqlite3.Connection, file_ids: list[int],
) -> str | None:
    if not file_ids:
        return None
    placeholders = ",".join("?" for _ in file_ids)
    try:
        row = conn.execute(
            f"SELECT MIN(e.text) AS text, COUNT(*) AS n "
            f"FROM entities e "
            f"JOIN chunks c ON c.id = e.chunk_id "
            f"WHERE c.file_id IN ({placeholders}) "
            f"  AND e.label IN "
            f"      ('ORG','PRODUCT','WORK_OF_ART','EVENT','PROJECT','GPE') "
            f"GROUP BY e.text_lower "
            f"ORDER BY n DESC LIMIT 1",
            tuple(file_ids),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    return row["text"] if row and row["n"] >= 2 else None


def _participants_topic(
    conn: sqlite3.Connection, person_ids: frozenset[int],
) -> str:
    if not person_ids:
        return "(unknown)"
    placeholders = ",".join("?" for _ in person_ids)
    rows = conn.execute(
        f"SELECT display_name FROM people "
        f"WHERE id IN ({placeholders}) "
        f"ORDER BY mention_count DESC LIMIT 3",
        tuple(sorted(person_ids)),
    ).fetchall()
    names = [r["display_name"] for r in rows if r["display_name"]]
    if not names:
        return "(unknown thread)"
    if len(names) == 1:
        return f"with {names[0]}"
    return f"with {', '.join(names[:-1])} & {names[-1]}"


def _dedup_key(topic: str, person_ids: list[int]) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(topic.lower().strip().encode("utf-8"))
    h.update(b"|")
    h.update(",".join(str(p) for p in sorted(person_ids)).encode("utf-8"))
    return h.hexdigest()[:32]


# ============================ summarization =========================


_SUMMARY_SYSTEM = """\
You are an executive assistant summarizing a long-running conversation
thread (multiple emails / messages over weeks) for the user. The user
gives you the most recent excerpts. You return a JSON object with:

  summary: 2-3 sentences, factual, no fluff
  decisions: array of decisions made across the thread
  open_questions: array of unanswered questions from the thread

Be conservative. Empty arrays are fine.
Return ONLY the JSON object.
"""


def summarize(
    conn: sqlite3.Connection, cfg: Config, thread_id: int,
    *, force: bool = False,
) -> StandingThread | None:
    """Run the LLM summarizer on one thread. Re-runs only if forced
    or if last_summarized_at is older than 7 days."""
    _ensure_schema(conn)
    row = conn.execute(
        "SELECT * FROM standing_threads WHERE id = ?", (thread_id,),
    ).fetchone()
    if not row:
        return None
    last_sum = row["last_summarized_at"]
    if not force and last_sum is not None:
        age_days = (time.time() - float(last_sum)) / 86400.0
        if age_days < 7:
            return _row_to_thread(row)

    file_ids = json.loads(row["file_ids_json"] or "[]")
    if not file_ids:
        return _row_to_thread(row)
    placeholders = ",".join("?" for _ in file_ids)
    chunks = conn.execute(
        f"SELECT f.path, f.indexed_at, "
        f"       SUBSTR(c.text, 1, 800) AS preview "
        f"FROM files f "
        f"JOIN chunks c ON c.file_id = f.id AND c.chunk_index = 0 "
        f"WHERE f.id IN ({placeholders}) "
        f"ORDER BY f.indexed_at DESC LIMIT 20",
        tuple(file_ids),
    ).fetchall()
    if not chunks:
        return _row_to_thread(row)
    body = "\n\n---\n\n".join(
        f"[{r['path']}]\n{r['preview']}" for r in chunks
    )
    try:
        from .email_assist import _safe_for_prompt
        body = _safe_for_prompt(body, max_chars=_SUMMARY_MAX_INPUT_CHARS)
    except ImportError:
        body = body[:_SUMMARY_MAX_INPUT_CHARS]

    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _row_to_thread(row)
    try:
        import anthropic

        from .budget import (
            check_budget,
            record_usage,
        )
        check_budget(cfg, "anthropic", feature="standing_threads")
    except Exception:  # noqa: BLE001
        return _row_to_thread(row)
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=_SUMMARY_MODEL,
            max_tokens=800,
            system=_SUMMARY_SYSTEM,
            messages=[{
                "role": "user",
                "content": (
                    f"Topic: {row['topic']}\n\nRecent excerpts:\n{body}"
                ),
            }],
        )
    except anthropic.APIError as e:
        log.warning("standing_threads: API error: %s", e)
        return _row_to_thread(row)
    in_tok = getattr(resp.usage, "input_tokens", 0)
    out_tok = getattr(resp.usage, "output_tokens", 0)
    try:
        record_usage(
            cfg, "anthropic", _SUMMARY_MODEL,
            input_tokens=in_tok, output_tokens=out_tok,
            note=f"standing_thread/{thread_id}",
            feature="standing_threads",
        )
    except Exception:  # noqa: BLE001
        pass
    raw = "\n".join(
        b.text for b in resp.content
        if getattr(b, "type", "") == "text"
    ).strip()
    if raw.startswith("```"):
        import re
        raw = re.sub(r"^```\w*\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return _row_to_thread(row)
    if not isinstance(parsed, dict):
        return _row_to_thread(row)
    summary = _redact(str(parsed.get("summary") or ""))[:1500]
    decisions = [
        _redact(str(d))[:300]
        for d in parsed.get("decisions") or []
    ][:10]
    open_qs = [
        _redact(str(q))[:300]
        for q in parsed.get("open_questions") or []
    ][:10]
    conn.execute(
        "UPDATE standing_threads SET "
        "summary_md = ?, decisions_json = ?, "
        "open_questions_json = ?, last_summarized_at = ? "
        "WHERE id = ?",
        (
            summary, json.dumps(decisions), json.dumps(open_qs),
            time.time(), thread_id,
        ),
    )
    conn.commit()
    return get_thread(conn, thread_id)


# ============================ queries ===============================


def get_thread(
    conn: sqlite3.Connection, thread_id: int,
) -> StandingThread | None:
    _ensure_schema(conn)
    row = conn.execute(
        "SELECT * FROM standing_threads WHERE id = ?", (thread_id,),
    ).fetchone()
    return _row_to_thread(row) if row else None


def list_threads(
    conn: sqlite3.Connection, *, limit: int = 30,
) -> list[StandingThread]:
    _ensure_schema(conn)
    rows = conn.execute(
        "SELECT * FROM standing_threads "
        "ORDER BY last_message_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_row_to_thread(r) for r in rows]
