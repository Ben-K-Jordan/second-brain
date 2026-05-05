"""Round 19 (Phase EA-1) — bidirectional commitment tracker.

The single biggest "feels like an EA" gap in the brain pre-round-19.
A real EA holds two running lists in their head:

  - **Outgoing** ("you owe them"): things you committed to other
    people. "Send Sarah the deck by Friday." "Get back to John
    re: contract terms." If these slip, your reputation slips.
  - **Incoming** ("they owe you"): things others committed to
    you. "John will review the proposal." "Sarah will introduce
    me to her CTO." If these go silent, you have to nudge.

This module extracts both directions from emails/journal/meeting
transcripts via Claude Haiku and tracks them in the ``followups``
table with status, due-by hint, and source provenance. The daemon
runs the extractor periodically; the dashboard's `/followups` view
shows open items grouped by direction × person.

Action items extracted by ``meeting_capture`` flow through the
same table so a single "open threads" view spans every promise the
user has on either side.

## Design notes

- **Idempotent extraction** — keyed on (source_kind, source_file_id,
  description_hash) so re-running the extractor on the same input
  doesn't double-create rows.
- **Confidence scoring** — LLM tags each extraction with a 0-1
  confidence; below ``_MIN_CONFIDENCE`` we drop. Personal-scope
  preference: false negatives (missed commitment) hurt less than
  false positives (nagging about something the user never said).
- **Lifecycle** — status ∈ {open, resolved, dismissed}. Resolution
  is manual from the dashboard for now; a follow-up round can add
  auto-resolution by detecting "I just sent it" in your sent mail.
- **Privacy** — every extracted ``description`` and ``source_excerpt``
  passes through ``redact_text`` before persistence. The LLM call
  itself goes through ``_safe_for_prompt`` (round 13 invariant).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
import weakref as _weakref
from collections.abc import Iterable
from dataclasses import asdict, dataclass

from .config import Config

log = logging.getLogger(__name__)

_SCHEMA_INITIALIZED: _weakref.WeakSet = _weakref.WeakSet()

# Round 21 fix (audit-found gap F1) — write-lock for the followups
# table. Shared with followups_ops via that module's import-and-use.
# Serialises concurrent writers from the daemon (extractor +
# auto-resolve) and the dashboard.
_WRITE_LOCK = threading.RLock()

_MIN_CONFIDENCE = 0.55
_EXTRACTOR_MODEL = "claude-haiku-4-5"
_EXTRACTOR_MAX_BODY_CHARS = 8000


@dataclass
class Followup:
    id: int
    direction: str           # 'outgoing' or 'incoming'
    person_id: int | None
    person_name: str
    topic: str
    description: str
    source_kind: str         # 'email', 'meeting', 'journal', 'manual'
    source_file_id: int | None
    source_excerpt: str
    status: str              # 'open', 'resolved', 'dismissed'
    due_at: float | None
    promised_at: float | None
    resolved_at: float | None
    created_at: float
    updated_at: float
    confidence: float
    extracted_by: str        # 'llm' or 'manual'

    def to_dict(self) -> dict:
        return asdict(self)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    try:
        if conn in _SCHEMA_INITIALIZED:
            return
    except TypeError:
        pass
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS followups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            direction TEXT NOT NULL CHECK(direction IN
                ('outgoing', 'incoming')),
            person_id INTEGER REFERENCES people(id) ON DELETE SET NULL,
            person_name TEXT NOT NULL DEFAULT '',
            topic TEXT NOT NULL,
            description TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            source_file_id INTEGER REFERENCES files(id) ON DELETE SET NULL,
            source_excerpt TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'open' CHECK(status IN
                ('open', 'resolved', 'dismissed')),
            due_at REAL,
            promised_at REAL,
            resolved_at REAL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            confidence REAL NOT NULL DEFAULT 1.0,
            extracted_by TEXT NOT NULL DEFAULT 'manual',
            -- Idempotency key: same (kind, file, description-hash)
            -- can't land twice. Lets the extractor be re-run safely
            -- on the same source input.
            dedup_key TEXT NOT NULL UNIQUE
        );
        CREATE INDEX IF NOT EXISTS idx_followups_status
            ON followups(status, direction);
        CREATE INDEX IF NOT EXISTS idx_followups_person
            ON followups(person_id);
        CREATE INDEX IF NOT EXISTS idx_followups_source
            ON followups(source_file_id);
        CREATE INDEX IF NOT EXISTS idx_followups_due
            ON followups(due_at) WHERE due_at IS NOT NULL;
    """)
    conn.commit()
    try:
        _SCHEMA_INITIALIZED.add(conn)
    except TypeError:
        pass


def _row_to_followup(row) -> Followup:
    return Followup(
        id=int(row["id"]),
        direction=row["direction"],
        person_id=(int(row["person_id"])
                   if row["person_id"] is not None else None),
        person_name=row["person_name"] or "",
        topic=row["topic"] or "",
        description=row["description"] or "",
        source_kind=row["source_kind"] or "",
        source_file_id=(int(row["source_file_id"])
                        if row["source_file_id"] is not None else None),
        source_excerpt=row["source_excerpt"] or "",
        status=row["status"],
        due_at=row["due_at"],
        promised_at=row["promised_at"],
        resolved_at=row["resolved_at"],
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
        confidence=float(row["confidence"]),
        extracted_by=row["extracted_by"],
    )


# ============================ persistence ===========================


def _dedup_key(
    *, source_kind: str, source_file_id: int | None,
    direction: str, description: str,
) -> str:
    """Stable hash for idempotency. Same (kind, file, direction,
    description) → same key → INSERT OR IGNORE no-ops."""
    h = hashlib.sha256()
    h.update(source_kind.encode("utf-8"))
    h.update(b"|")
    h.update(str(source_file_id or "").encode("utf-8"))
    h.update(b"|")
    h.update(direction.encode("utf-8"))
    h.update(b"|")
    # Lowercased + whitespace-collapsed so trivial variations dedupe.
    h.update(" ".join(description.lower().split()).encode("utf-8"))
    return h.hexdigest()[:32]


def add_followup_with_status(
    conn: sqlite3.Connection,
    *,
    direction: str,
    topic: str,
    description: str,
    person_id: int | None = None,
    person_name: str = "",
    source_kind: str = "manual",
    source_file_id: int | None = None,
    source_excerpt: str = "",
    due_at: float | None = None,
    promised_at: float | None = None,
    confidence: float = 1.0,
    extracted_by: str = "manual",
) -> tuple[int, bool]:
    """Round 21 — idempotent insert returning (row_id, was_new).

    The "was_new" flag comes directly from ``cur.rowcount`` AFTER the
    INSERT OR IGNORE, BEFORE any commit. Replaces the round-19 racy
    5-second time-window heuristic. Returns (0, False) on failure.
    """
    if direction not in ("outgoing", "incoming"):
        raise ValueError(
            f"direction must be outgoing|incoming; got {direction!r}",
        )
    key = _dedup_key(
        source_kind=source_kind, source_file_id=source_file_id,
        direction=direction, description=description,
    )
    # Round 13/14 invariant — redact before persisting.
    try:
        from .safety import redact_text
        description = redact_text(description)
        source_excerpt = redact_text(source_excerpt)
        topic = redact_text(topic)
        person_name = redact_text(person_name)
    except ImportError:
        pass
    now = time.time()
    try:
        with _WRITE_LOCK:
            # Round 21 — schema init also under the lock; otherwise
            # the first writer thread can race against a concurrent
            # one on the executescript + commit path even though the
            # WeakSet guard tries to short-circuit subsequent calls.
            _ensure_schema(conn)
            cur = conn.execute(
                "INSERT OR IGNORE INTO followups"
                "(direction, person_id, person_name, topic, description, "
                " source_kind, source_file_id, source_excerpt, status, "
                " due_at, promised_at, created_at, updated_at, "
                " confidence, extracted_by, dedup_key) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?, ?, ?, ?, ?)",
                (
                    direction, person_id, person_name, topic, description,
                    source_kind, source_file_id, source_excerpt,
                    due_at, promised_at, now, now,
                    float(confidence), extracted_by, key,
                ),
            )
            was_new = cur.rowcount > 0
            try:
                conn.commit()
            except sqlite3.OperationalError:
                pass  # INSERT OR IGNORE hit dedup → no transaction
            existing = conn.execute(
                "SELECT id FROM followups WHERE dedup_key = ?", (key,),
            ).fetchone()
            return (
                int(existing["id"]) if existing else 0,
                was_new,
            )
    except Exception as e:  # noqa: BLE001
        log.warning("followups.add: failed: %s", e)
        return 0, False


def add_followup(
    conn: sqlite3.Connection,
    *,
    direction: str,
    topic: str,
    description: str,
    person_id: int | None = None,
    person_name: str = "",
    source_kind: str = "manual",
    source_file_id: int | None = None,
    source_excerpt: str = "",
    due_at: float | None = None,
    promised_at: float | None = None,
    confidence: float = 1.0,
    extracted_by: str = "manual",
) -> int:
    """Idempotent insert. Returns the new (or existing) row id, or 0
    on failure. See ``add_followup_with_status`` if you also need
    to know whether the row was new vs. existing."""
    row_id, _was_new = add_followup_with_status(
        conn,
        direction=direction, topic=topic, description=description,
        person_id=person_id, person_name=person_name,
        source_kind=source_kind, source_file_id=source_file_id,
        source_excerpt=source_excerpt, due_at=due_at,
        promised_at=promised_at, confidence=confidence,
        extracted_by=extracted_by,
    )
    return row_id


def mark_resolved(conn: sqlite3.Connection, followup_id: int) -> bool:
    _ensure_schema(conn)
    with _WRITE_LOCK:
        cur = conn.execute(
            "UPDATE followups SET status='resolved', resolved_at=?, "
            "updated_at=? WHERE id = ? AND status = 'open'",
            (time.time(), time.time(), followup_id),
        )
        conn.commit()
        return cur.rowcount > 0


def mark_dismissed(conn: sqlite3.Connection, followup_id: int) -> bool:
    _ensure_schema(conn)
    with _WRITE_LOCK:
        cur = conn.execute(
            "UPDATE followups SET status='dismissed', resolved_at=?, "
            "updated_at=? WHERE id = ? AND status = 'open'",
            (time.time(), time.time(), followup_id),
        )
        conn.commit()
        return cur.rowcount > 0


# ============================ queries ===============================


def list_open(
    conn: sqlite3.Connection,
    *,
    direction: str | None = None,
    person_id: int | None = None,
    limit: int = 200,
) -> list[Followup]:
    _ensure_schema(conn)
    sql = "SELECT * FROM followups WHERE status = 'open'"
    params: list = []
    if direction:
        sql += " AND direction = ?"
        params.append(direction)
    if person_id is not None:
        sql += " AND person_id = ?"
        params.append(person_id)
    sql += (
        " ORDER BY "
        "  CASE WHEN due_at IS NULL THEN 1 ELSE 0 END, "
        "  due_at ASC, "
        "  promised_at DESC, "
        "  created_at DESC "
        "LIMIT ?"
    )
    params.append(limit)
    rows = conn.execute(sql, tuple(params)).fetchall()
    return [_row_to_followup(r) for r in rows]


def list_overdue(
    conn: sqlite3.Connection,
    *,
    grace_seconds: float = 0.0,
    limit: int = 100,
) -> list[Followup]:
    """Open followups whose ``due_at`` has passed."""
    _ensure_schema(conn)
    cutoff = time.time() - grace_seconds
    rows = conn.execute(
        "SELECT * FROM followups WHERE status = 'open' "
        "AND due_at IS NOT NULL AND due_at <= ? "
        "ORDER BY due_at ASC LIMIT ?",
        (cutoff, limit),
    ).fetchall()
    return [_row_to_followup(r) for r in rows]


def count_open(
    conn: sqlite3.Connection, *, direction: str | None = None,
) -> int:
    _ensure_schema(conn)
    sql = "SELECT COUNT(*) AS n FROM followups WHERE status = 'open'"
    params: list = []
    if direction:
        sql += " AND direction = ?"
        params.append(direction)
    row = conn.execute(sql, tuple(params)).fetchone()
    return int(row["n"]) if row else 0


def get(conn: sqlite3.Connection, followup_id: int) -> Followup | None:
    _ensure_schema(conn)
    row = conn.execute(
        "SELECT * FROM followups WHERE id = ?", (followup_id,),
    ).fetchone()
    return _row_to_followup(row) if row else None


# ============================ extraction ============================


_EXTRACT_SYSTEM = """\
You read a piece of correspondence (email, journal entry, meeting transcript)
and extract bidirectional commitments. Return ONLY a JSON array.

Each commitment object has:
  direction: "outgoing" if THE USER promised something TO someone,
             "incoming" if SOMEONE promised something TO the user
  person:    who the commitment involves (their display name; "unknown"
             if unclear)
  topic:     2-6 word summary
  description: 1 sentence explicit statement of what was promised
  due_hint:  ISO date "YYYY-MM-DD" if explicitly mentioned, else null
  confidence: 0.0-1.0; 1.0 = literal "I will send you X by Y",
             0.5 = strong implication, < 0.5 = drop it
  excerpt:   the source sentence(s) that establish the commitment

Rules:
  - Only extract EXPLICIT commitments. Polite phrases like "let me know",
    "happy to help", "we should chat" are NOT commitments unless paired
    with a specific deliverable.
  - "I'll think about it" is NOT a commitment.
  - "Per our conversation, sending X" → outgoing commitment from user.
  - Don't fabricate names. If unclear who's involved, use "unknown".
  - Don't extract anything from headers/signatures/auto-replies.
  - If nothing qualifies, return [].
"""


def extract_from_text(
    cfg: Config,
    *,
    text: str,
    user_name: str,
    source_kind: str,
    source_file_id: int | None,
    source_label: str = "",
) -> list[dict]:
    """LLM extraction. Returns a list of raw dicts (not yet persisted).

    The caller decides which to persist (e.g. dropping low-confidence
    or post-filtering by person link). Returns [] on any error so a
    bad input never breaks the daemon job.
    """
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return []
    try:
        import anthropic
    except ImportError:
        return []
    try:
        from .budget import check_budget, record_usage
        check_budget(cfg, "anthropic", feature="followups")
    except Exception:  # noqa: BLE001 — also catches BudgetExceededError
        return []
    # Round 13/14 invariant — redact before send.
    try:
        from .email_assist import _safe_for_prompt
        body_clip = _safe_for_prompt(
            text, max_chars=_EXTRACTOR_MAX_BODY_CHARS,
        )
    except ImportError:
        body_clip = (text or "")[:_EXTRACTOR_MAX_BODY_CHARS]
    if not body_clip.strip():
        return []
    user_prompt = (
        f"User's own name: {user_name}\n"
        f"Source kind: {source_kind}\n"
        + (f"Source label: {source_label}\n" if source_label else "")
        + f"\n---\n{body_clip}\n---"
    )
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=_EXTRACTOR_MODEL,
            max_tokens=1500,
            system=_EXTRACT_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )
        try:
            record_usage(
                cfg, "anthropic", _EXTRACTOR_MODEL,
                input_tokens=getattr(
                    response.usage, "input_tokens", 0,
                ),
                output_tokens=getattr(
                    response.usage, "output_tokens", 0,
                ),
                note=f"followups/{source_kind}",
                feature="followups",
            )
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001
        log.warning("followups.extract: API error: %s", e)
        return []
    try:
        raw = "\n".join(
            b.text for b in response.content
            if getattr(b, "type", "") == "text"
        ).strip()
    except Exception:  # noqa: BLE001
        return []
    # Strip optional ```json fences.
    if raw.startswith("```"):
        import re
        raw = re.sub(r"^```\w*\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        items = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning("followups.extract: bad JSON: %s", e)
        return []
    if not isinstance(items, list):
        return []
    return [it for it in items if isinstance(it, dict)]


def extract_and_persist(
    conn: sqlite3.Connection,
    cfg: Config,
    *,
    text: str,
    user_name: str,
    source_kind: str,
    source_file_id: int | None,
    source_label: str = "",
) -> int:
    """End-to-end: extract via LLM, filter low-confidence, resolve
    person names to IDs, and persist. Returns count of new rows
    landed (idempotent — re-extracting the same source returns 0).
    """
    items = extract_from_text(
        cfg, text=text, user_name=user_name,
        source_kind=source_kind,
        source_file_id=source_file_id,
        source_label=source_label,
    )
    if not items:
        return 0
    n_added = 0
    # Lazy import to avoid circular deps.
    try:
        from . import people as people_mod
    except ImportError:
        people_mod = None  # type: ignore[assignment]
    for it in items:
        try:
            direction = str(it.get("direction") or "").lower()
            if direction not in ("outgoing", "incoming"):
                continue
            confidence = float(it.get("confidence") or 0.0)
            if confidence < _MIN_CONFIDENCE:
                continue
            description = str(it.get("description") or "").strip()
            if not description:
                continue
            topic = str(it.get("topic") or "").strip() or description[:60]
            person_name = str(it.get("person") or "").strip()
            person_id: int | None = None
            if (
                people_mod is not None and person_name
                and person_name.lower() != "unknown"
            ):
                try:
                    p = people_mod.find_person_by_name(conn, person_name)
                    if p is not None:
                        person_id = int(p.id)
                except Exception:  # noqa: BLE001
                    pass
            due_at: float | None = None
            due_hint = it.get("due_hint")
            if due_hint:
                try:
                    from datetime import date, datetime
                    d = date.fromisoformat(str(due_hint))
                    due_at = datetime(d.year, d.month, d.day).timestamp()
                except (ValueError, TypeError):
                    due_at = None
            excerpt = str(it.get("excerpt") or "")[:1000]
            new_id, was_new = add_followup_with_status(
                conn,
                direction=direction,
                topic=topic,
                description=description,
                person_id=person_id,
                person_name=person_name,
                source_kind=source_kind,
                source_file_id=source_file_id,
                source_excerpt=excerpt,
                due_at=due_at,
                promised_at=time.time(),
                confidence=confidence,
                extracted_by="llm",
            )
            # Round 21 fix (audit-found gap A3) — use the
            # ``was_new`` flag directly from ``cur.rowcount`` instead
            # of a 5-second wall-clock window. The earlier heuristic
            # was racy with concurrent extractor runs and broken on
            # slow LLMs.
            if new_id and was_new:
                n_added += 1
        except Exception as e:  # noqa: BLE001
            log.debug("followups: skipping malformed item: %s", e)
            continue
    return n_added


def extract_from_recent_inputs(
    conn: sqlite3.Connection,
    cfg: Config,
    *,
    hours: int = 24,
    user_name: str | None = None,
    max_files: int = 30,
) -> dict:
    """Daemon entry point. Walk the last ``hours`` of newly-indexed
    files of the right kind (email / meeting / journal) and run
    extraction on each. Idempotent — files already extracted from
    are tracked in ``followups_extracted`` so re-runs no-op.
    """
    _ensure_schema(conn)
    # Bookkeeping table — one row per (file_id, run_at) so we can
    # tell what's been processed.
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS followups_extracted (
            file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
            extracted_at REAL NOT NULL,
            n_added INTEGER NOT NULL DEFAULT 0
        );
    """)
    conn.commit()
    cutoff = time.time() - hours * 3600
    if user_name is None:
        user_name = getattr(cfg, "user_name", None) or "User"
    # Round 24 fix (audit-found systemic bug) — production stores
    # Gmail/IMAP/transcripts as ``kind='url'`` with virtual-path
    # prefixes (``imap://``, ``gmail://``, ``transcript://``,
    # ``voice://``, ``journal://``). The earlier ``kind`` set
    # filter only accepted iMessage's ``kind='message'``, so
    # extraction never ran on Gmail/IMAP for any user.
    eligible_kinds = {
        "email", "message", "transcript", "journal", "voice",
    }
    eligible_path_prefixes = (
        "imap://", "gmail://", "transcript://",
        "voice://", "journal://",
    )
    rows = conn.execute(
        "SELECT f.id, f.path, f.kind, f.indexed_at FROM files f "
        "LEFT JOIN followups_extracted fe ON fe.file_id = f.id "
        "WHERE f.indexed_at >= ? AND fe.file_id IS NULL "
        "ORDER BY f.indexed_at DESC LIMIT ?",
        (cutoff, max_files),
    ).fetchall()
    n_files = 0
    n_followups = 0
    for r in rows:
        path = (r["path"] or "")
        if (
            r["kind"] not in eligible_kinds
            and not any(
                path.startswith(p) for p in eligible_path_prefixes
            )
        ):
            continue
        # Pull body text from chunks (concat the first N for size cap).
        chunk_rows = conn.execute(
            "SELECT text FROM chunks WHERE file_id = ? "
            "ORDER BY chunk_index LIMIT 30",
            (int(r["id"]),),
        ).fetchall()
        if not chunk_rows:
            continue
        text = "\n".join(c["text"] for c in chunk_rows)
        added = extract_and_persist(
            conn, cfg,
            text=text, user_name=user_name,
            source_kind=_normalise_kind(r["kind"]),
            source_file_id=int(r["id"]),
            source_label=str(r["path"] or ""),
        )
        n_files += 1
        n_followups += added
        try:
            conn.execute(
                "INSERT OR REPLACE INTO followups_extracted"
                "(file_id, extracted_at, n_added) VALUES (?, ?, ?)",
                (int(r["id"]), time.time(), added),
            )
            conn.commit()
        except Exception:  # noqa: BLE001
            pass
    return {"files_scanned": n_files, "followups_added": n_followups}


def _normalise_kind(kind: str) -> str:
    if kind == "transcript":
        return "meeting"
    if kind == "voice":
        return "journal"
    if kind == "message":
        return "email"
    return kind


# ============================ rendering helpers =====================


def serialise_for_brief(rows: Iterable[Followup]) -> list[dict]:
    """Lightweight serialisation suitable for daily-brief / weekly-
    letter consumption. No source_excerpt (size; we don't surface
    raw text in the brief). Sorted by direction first, then due."""
    return [
        {
            "id": r.id,
            "direction": r.direction,
            "person": r.person_name,
            "topic": r.topic,
            "due_at": r.due_at,
            "promised_at": r.promised_at,
            "age_days": (
                (time.time() - r.promised_at) / 86400.0
                if r.promised_at else None
            ),
        }
        for r in rows
    ]
