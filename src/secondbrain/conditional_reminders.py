"""Round 19 (Phase EA-10) — conditional reminders.

"Remind me to nudge Sarah if she hasn't responded by Friday."
"Tell me when the contract is back from John."
"Ping me about my Bali trip when Q3 starts."

These are reminders gated on a *condition* rather than a simple
scheduled time. The daemon polls the conditions periodically and
fires a notification when the condition transitions from false →
true (or, for date-passed conditions, when the date passes).

Supported condition kinds (extensible):

  - ``no_reply_from``: fire if no email has arrived from
    ``email`` since ``since_ts``. Optionally with ``due_at``
    (fire even if reply arrives after due).
  - ``date_passed``: fire when ``fire_after`` ts is reached.
  - ``followup_unresolved``: fire if a specific followup id is
    still status='open' at ``fire_after``.

Lifecycle: pending → fired (terminal). Cancelled state is for
user-dismissed without firing.

Persisted in ``conditional_reminders`` so daemon restart preserves
state.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import weakref as _weakref
from dataclasses import asdict, dataclass

log = logging.getLogger(__name__)

_SCHEMA_INITIALIZED: _weakref.WeakSet = _weakref.WeakSet()


@dataclass
class ConditionalReminder:
    id: int
    description: str
    condition_kind: str
    condition: dict
    fire_after: float | None
    status: str            # 'pending', 'fired', 'cancelled'
    created_at: float
    fired_at: float | None
    notification_key: str  # used when firing → notifications.enqueue

    def to_dict(self) -> dict:
        return asdict(self)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    try:
        if conn in _SCHEMA_INITIALIZED:
            return
    except TypeError:
        pass
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS conditional_reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT NOT NULL,
            condition_kind TEXT NOT NULL,
            condition_json TEXT NOT NULL DEFAULT '{}',
            fire_after REAL,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending', 'fired', 'cancelled')),
            created_at REAL NOT NULL,
            fired_at REAL,
            notification_key TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_conditional_reminders_status
            ON conditional_reminders(status, fire_after);
    """)
    conn.commit()
    try:
        _SCHEMA_INITIALIZED.add(conn)
    except TypeError:
        pass


def _row_to_reminder(row) -> ConditionalReminder:
    return ConditionalReminder(
        id=int(row["id"]),
        description=row["description"] or "",
        condition_kind=row["condition_kind"] or "",
        condition=json.loads(row["condition_json"] or "{}"),
        fire_after=row["fire_after"],
        status=row["status"],
        created_at=float(row["created_at"]),
        fired_at=row["fired_at"],
        notification_key=row["notification_key"] or "",
    )


# ============================ create ================================


def add_reminder(
    conn: sqlite3.Connection,
    *,
    description: str,
    condition_kind: str,
    condition: dict,
    fire_after: float | None = None,
) -> int:
    """Create a new conditional reminder. Returns the new row id."""
    _ensure_schema(conn)
    if condition_kind not in (
        "no_reply_from", "date_passed", "followup_unresolved",
    ):
        raise ValueError(
            f"unsupported condition_kind: {condition_kind!r}",
        )
    # Round 13 invariant — redact persisted text.
    try:
        from .safety import redact_text
        description = redact_text(description)
    except ImportError:
        pass
    notification_key = (
        f"cond:{condition_kind}:{int(time.time() * 1000)}"
    )
    cur = conn.execute(
        "INSERT INTO conditional_reminders"
        "(description, condition_kind, condition_json, fire_after, "
        " created_at, notification_key) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            description, condition_kind,
            json.dumps(condition or {}),
            fire_after, time.time(), notification_key,
        ),
    )
    conn.commit()
    return int(cur.lastrowid or 0)


def cancel(conn: sqlite3.Connection, reminder_id: int) -> bool:
    _ensure_schema(conn)
    cur = conn.execute(
        "UPDATE conditional_reminders SET status='cancelled' "
        "WHERE id = ? AND status = 'pending'",
        (reminder_id,),
    )
    conn.commit()
    return cur.rowcount > 0


# ============================ check + fire ==========================


def _condition_met(
    conn: sqlite3.Connection, r: ConditionalReminder,
) -> bool:
    """Evaluate one reminder's condition. Returns True if it should
    fire now."""
    now = time.time()
    if r.fire_after is not None and now < r.fire_after:
        # Not yet eligible to fire.
        return False
    cond = r.condition or {}
    if r.condition_kind == "date_passed":
        # If we got past the fire_after gate above, this fires.
        return r.fire_after is not None
    if r.condition_kind == "no_reply_from":
        email = (cond.get("email") or "").strip().lower()
        since = float(cond.get("since_ts") or 0.0)
        if not email or not since:
            return False
        # Look for any inbound email from this address since `since`.
        # Round 24 fix (audit-found systemic bug) — production
        # Gmail/IMAP land as ``kind='url'``; the earlier filter
        # never matched, so the existence check always returned
        # ``row is None`` → ``no_reply`` reminders ALWAYS fired,
        # even when the user HAD received a reply. Use the shared
        # email-kind helper.
        from .db import EMAIL_KIND_SQL
        try:
            row = conn.execute(
                "SELECT 1 FROM files f "
                "JOIN chunks c ON c.file_id = f.id AND c.chunk_index = 0 "
                "WHERE f.indexed_at >= ? "
                f"  AND {EMAIL_KIND_SQL} "
                "  AND LOWER(c.text) LIKE ? "
                "LIMIT 1",
                (since, f"%from: %{email}%"),
            ).fetchone()
        except sqlite3.OperationalError:
            row = None
        # "no_reply" → fire only if NO matching email exists yet AND
        # the fire_after gate has passed.
        return row is None
    if r.condition_kind == "followup_unresolved":
        fid = cond.get("followup_id")
        if fid is None:
            return False
        try:
            row = conn.execute(
                "SELECT status FROM followups WHERE id = ?",
                (int(fid),),
            ).fetchone()
        except sqlite3.OperationalError:
            return False
        return bool(row) and row["status"] == "open"
    return False


def check_and_fire(conn: sqlite3.Connection) -> int:
    """Evaluate every pending reminder. Returns the number fired."""
    _ensure_schema(conn)
    rows = conn.execute(
        "SELECT * FROM conditional_reminders WHERE status = 'pending'",
    ).fetchall()
    n_fired = 0
    try:
        from . import notifications
    except ImportError:
        notifications = None  # type: ignore[assignment]
    for r in rows:
        rem = _row_to_reminder(r)
        try:
            if not _condition_met(conn, rem):
                continue
        except Exception as e:  # noqa: BLE001
            log.warning(
                "conditional_reminders: eval failed for #%d: %s",
                rem.id, e,
            )
            continue
        # Fire: enqueue a notification + flip status.
        if notifications is not None:
            try:
                notifications.enqueue(
                    conn,
                    key=rem.notification_key,
                    kind="conditional",
                    urgency="med",
                    title=rem.description[:120],
                    body=f"Triggered by: {rem.condition_kind}",
                    href="/notifications",
                    payload={
                        "reminder_id": rem.id,
                        "condition_kind": rem.condition_kind,
                    },
                )
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "conditional_reminders: notify failed for #%d: %s",
                    rem.id, e,
                )
        conn.execute(
            "UPDATE conditional_reminders SET status='fired', "
            "fired_at = ? WHERE id = ?",
            (time.time(), rem.id),
        )
        conn.commit()
        n_fired += 1
    return n_fired


# ============================ queries ===============================


def list_pending(
    conn: sqlite3.Connection, *, limit: int = 100,
) -> list[ConditionalReminder]:
    _ensure_schema(conn)
    rows = conn.execute(
        "SELECT * FROM conditional_reminders "
        "WHERE status = 'pending' "
        "ORDER BY COALESCE(fire_after, created_at) ASC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_row_to_reminder(r) for r in rows]


def list_recent(
    conn: sqlite3.Connection, *, limit: int = 50,
) -> list[ConditionalReminder]:
    _ensure_schema(conn)
    rows = conn.execute(
        "SELECT * FROM conditional_reminders "
        "ORDER BY COALESCE(fired_at, created_at) DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_row_to_reminder(r) for r in rows]


def get(
    conn: sqlite3.Connection, reminder_id: int,
) -> ConditionalReminder | None:
    _ensure_schema(conn)
    row = conn.execute(
        "SELECT * FROM conditional_reminders WHERE id = ?",
        (reminder_id,),
    ).fetchone()
    return _row_to_reminder(row) if row else None
