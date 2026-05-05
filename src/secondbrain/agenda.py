"""Round 19 (Phase EA-2) — 1:1 agenda builder.

When the user has a 1:1 with a specific person coming up, an EA's
job is to pull together everything that's been simmering between
them so the user walks in prepared:

  - **Last 1:1 notes** — what was discussed last time (from prior
    meeting captures with this person)
  - **Open threads from email** — recent unresolved threads
  - **Open follow-ups** — both directions (you owe them / they owe you)
  - **Journal mentions** — anything the user wrote about wanting
    to discuss with this person
  - **Recent shared topics** — entities both of you have been
    around in the last 14 days

Output: a structured ``Agenda`` dataclass + a Markdown rendering
suitable for the dashboard or a copy-paste into a notes app.

This module deliberately does NOT call an LLM — it's a fast
aggregation built from existing tables. The LLM-pitched version
(narrative agenda email) can be done client-side via chat with
this data as context.

## Design notes

- **No persistence**: agendas are fresh per-call, computed from
  current state. Caching is per-page-load only.
- **Person-anchored**: agenda input is a person_id. Calendar event
  lookup ("agenda for my next 1:1 with Sarah") happens upstream.
- **Privacy**: every text snippet that gets surfaced passes through
  ``redact_text`` to maintain the round-13 invariant.
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

_SCHEMA_INITIALIZED: _weakref.WeakSet = _weakref.WeakSet()
# Round 21 fix (audit-found gap F1) — write lock for agenda_notes.
# Daemon doesn't write here, but dashboard worker threads do.
_WRITE_LOCK = threading.RLock()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Round 20 — agenda persistence + per-person notes bucket.

    ``agenda_notes`` is a free-form per-person "things to bring up
    next time" stash the user can append to anytime. Round-19's
    aggregation queries are still real-time; this just adds the
    user-curated layer on top.
    """
    try:
        if conn in _SCHEMA_INITIALIZED:
            return
    except TypeError:
        pass
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS agenda_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL REFERENCES people(id)
                ON DELETE CASCADE,
            text TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending', 'discussed', 'dropped')),
            created_at REAL NOT NULL,
            discussed_at REAL,
            tags_json TEXT NOT NULL DEFAULT '[]'
        );
        CREATE INDEX IF NOT EXISTS idx_agenda_notes_person
            ON agenda_notes(person_id, status);
    """)
    conn.commit()
    try:
        _SCHEMA_INITIALIZED.add(conn)
    except TypeError:
        pass


@dataclass
class AgendaNote:
    id: int
    person_id: int
    text: str
    status: str
    created_at: float
    discussed_at: float | None
    tags: list[str]


def add_note(
    conn: sqlite3.Connection, person_id: int, text: str,
    *, tags: list[str] | None = None,
) -> int:
    """Add a "want to bring up with X" note. Idempotent? No — duplicates
    are allowed since a user might genuinely want to surface the same
    topic twice."""
    _ensure_schema(conn)
    try:
        from .safety import redact_text
        text = redact_text(text)
    except ImportError:
        pass
    with _WRITE_LOCK:
        cur = conn.execute(
            "INSERT INTO agenda_notes(person_id, text, created_at, tags_json) "
            "VALUES (?, ?, ?, ?)",
            (person_id, text, time.time(), json.dumps(tags or [])),
        )
        conn.commit()
        return int(cur.lastrowid or 0)


def list_notes(
    conn: sqlite3.Connection, person_id: int,
    *,
    status: str = "pending", limit: int = 50,
) -> list[AgendaNote]:
    _ensure_schema(conn)
    rows = conn.execute(
        "SELECT * FROM agenda_notes "
        "WHERE person_id = ? AND status = ? "
        "ORDER BY created_at DESC LIMIT ?",
        (person_id, status, limit),
    ).fetchall()
    return [
        AgendaNote(
            id=int(r["id"]),
            person_id=int(r["person_id"]),
            text=r["text"] or "",
            status=r["status"],
            created_at=float(r["created_at"]),
            discussed_at=r["discussed_at"],
            tags=json.loads(r["tags_json"] or "[]"),
        )
        for r in rows
    ]


def mark_discussed(conn: sqlite3.Connection, note_id: int) -> bool:
    _ensure_schema(conn)
    with _WRITE_LOCK:
        cur = conn.execute(
            "UPDATE agenda_notes SET status='discussed', discussed_at=? "
            "WHERE id = ? AND status = 'pending'",
            (time.time(), note_id),
        )
        conn.commit()
        return cur.rowcount > 0


def drop_note(conn: sqlite3.Connection, note_id: int) -> bool:
    _ensure_schema(conn)
    with _WRITE_LOCK:
        cur = conn.execute(
            "UPDATE agenda_notes SET status='dropped', discussed_at=? "
            "WHERE id = ? AND status = 'pending'",
            (time.time(), note_id),
        )
        conn.commit()
        return cur.rowcount > 0


@dataclass
class AgendaItem:
    kind: str           # 'last_meeting', 'open_email', 'followup_out',
                        # 'followup_in', 'journal_note', 'shared_topic'
    title: str
    detail: str = ""
    href: str = ""
    age_days: float | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class Agenda:
    person_id: int
    person_name: str
    last_contact_at: float | None
    days_since_contact: int | None
    last_meeting: AgendaItem | None
    open_followups_outgoing: list[AgendaItem]   # you owe them
    open_followups_incoming: list[AgendaItem]   # they owe you
    open_email_threads: list[AgendaItem]
    journal_notes: list[AgendaItem]
    shared_topics: list[AgendaItem]
    generated_at: float
    # Round 20 — user-curated agenda notes ("things to bring up
    # next time we meet") for this person.
    user_notes: list[AgendaItem] = field(default_factory=list)

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    @property
    def total_items(self) -> int:
        return (
            (1 if self.last_meeting else 0)
            + len(self.open_followups_outgoing)
            + len(self.open_followups_incoming)
            + len(self.open_email_threads)
            + len(self.journal_notes)
            + len(self.shared_topics)
            + len(self.user_notes)
        )


def _redact(text: str) -> str:
    try:
        from .safety import redact_text
        return redact_text(text or "")
    except ImportError:
        return text or ""


def _last_meeting_with(
    conn: sqlite3.Connection, person_id: int,
) -> AgendaItem | None:
    """The most recent meeting capture where this person was an
    attendee or owner. We use entity-link as the heuristic: if the
    person's name appears in the meeting transcript chunks, count
    it as "this person was in the room"."""
    try:
        row = conn.execute(
            "SELECT mc.* FROM meeting_captures mc "
            "JOIN person_mentions pm ON pm.file_id = mc.file_id "
            "WHERE pm.person_id = ? "
            "ORDER BY mc.captured_at DESC LIMIT 1",
            (person_id,),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    if not row:
        return None
    age_days = max(0.0, (time.time() - float(row["captured_at"])) / 86400.0)
    decisions_summary = ""
    try:
        import json
        decisions = json.loads(row["decisions_json"] or "[]")
        if decisions:
            decisions_summary = " · ".join(
                _redact(d.get("text", "")[:80]) for d in decisions[:3]
            )
    except Exception:  # noqa: BLE001
        pass
    return AgendaItem(
        kind="last_meeting",
        title=_redact(row["title"] or "(untitled meeting)"),
        detail=decisions_summary,
        href=f"/file?file_id={row['file_id']}",
        age_days=age_days,
        extra={"file_id": int(row["file_id"])},
    )


def _open_followups_for(
    conn: sqlite3.Connection, person_id: int,
) -> tuple[list[AgendaItem], list[AgendaItem]]:
    """Returns (outgoing, incoming) AgendaItems for this person."""
    try:
        from . import followups
        out_rows = followups.list_open(
            conn, direction="outgoing", person_id=person_id, limit=10,
        )
        in_rows = followups.list_open(
            conn, direction="incoming", person_id=person_id, limit=10,
        )
    except Exception:  # noqa: BLE001
        return [], []

    def _to_item(f, kind: str) -> AgendaItem:
        age_days = (
            (time.time() - f.promised_at) / 86400.0
            if f.promised_at else None
        )
        detail = ""
        if f.due_at:
            from datetime import date
            d = date.fromtimestamp(f.due_at).isoformat()
            detail = f"due {d}"
        return AgendaItem(
            kind=kind,
            title=_redact(f.topic),
            detail=detail,
            href=f"/followups#fu{f.id}",
            age_days=age_days,
            extra={
                "followup_id": f.id,
                "description": _redact(f.description),
            },
        )

    return (
        [_to_item(f, "followup_out") for f in out_rows],
        [_to_item(f, "followup_in") for f in in_rows],
    )


def _open_email_threads_with(
    conn: sqlite3.Connection, person_id: int,
    *, days: int = 30, limit: int = 5,
) -> list[AgendaItem]:
    """Recent email files mentioning this person without a
    classification of 'replied' or 'archived'. Best-effort — we
    don't have full thread state, but stale unanswered emails are
    a common 1:1 topic."""
    cutoff = time.time() - days * 86400
    try:
        rows = conn.execute(
            "SELECT DISTINCT f.id, f.path, f.indexed_at, "
            "       SUBSTR(c.text, 1, 200) AS preview "
            "FROM person_mentions pm "
            "JOIN files f ON f.id = pm.file_id "
            "JOIN chunks c ON c.file_id = f.id AND c.chunk_index = 0 "
            "WHERE pm.person_id = ? "
            "  AND f.indexed_at >= ? "
            "  AND (f.kind = 'email' OR f.kind = 'message') "
            "ORDER BY f.indexed_at DESC LIMIT ?",
            (person_id, cutoff, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    out: list[AgendaItem] = []
    for r in rows:
        age = (time.time() - float(r["indexed_at"])) / 86400.0
        out.append(AgendaItem(
            kind="open_email",
            title=_redact((r["path"] or "")[:100]),
            detail=_redact((r["preview"] or "")[:120]),
            href=f"/file?file_id={r['id']}",
            age_days=age,
            extra={"file_id": int(r["id"])},
        ))
    return out


def _journal_notes_about(
    conn: sqlite3.Connection, person_name: str,
    *, days: int = 60, limit: int = 5,
) -> list[AgendaItem]:
    """Journal entries from the last ``days`` mentioning this person
    by name. Lightweight LIKE-based — works without person_mentions
    on the journal entries table."""
    cutoff = time.time() - days * 86400
    if not person_name.strip():
        return []
    try:
        rows = conn.execute(
            "SELECT date, text FROM journal_entries "
            "WHERE created_at >= ? "
            "  AND LOWER(text) LIKE ? "
            "ORDER BY date DESC LIMIT ?",
            (cutoff, f"%{person_name.lower()}%", limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    out: list[AgendaItem] = []
    for r in rows:
        # Pull the sentence containing the name.
        text = r["text"] or ""
        idx = text.lower().find(person_name.lower())
        if idx == -1:
            snippet = text[:160]
        else:
            start = max(0, idx - 60)
            end = min(len(text), idx + len(person_name) + 100)
            snippet = ("…" if start > 0 else "") + text[start:end] + (
                "…" if end < len(text) else ""
            )
        try:
            from datetime import datetime
            iso = r["date"]
            age = (datetime.now() - datetime.fromisoformat(iso)).days
        except Exception:  # noqa: BLE001
            age = None
        out.append(AgendaItem(
            kind="journal_note",
            title=f"Journal {r['date']}",
            detail=_redact(snippet),
            href=f"/journal?date={r['date']}",
            age_days=float(age) if age is not None else None,
            extra={"date": r["date"]},
        ))
    return out


def _shared_topics(
    conn: sqlite3.Connection, person_id: int,
    *, days: int = 14, limit: int = 5,
) -> list[AgendaItem]:
    """Entities (orgs, projects) that have appeared in docs both
    with this person AND in the user's recent activity. Heuristic:
    take entities that co-occur with the person in the last ``days``
    of indexed content."""
    cutoff = time.time() - days * 86400
    try:
        rows = conn.execute(
            "SELECT MIN(e.text) AS text, e.label, "
            "       COUNT(DISTINCT c.file_id) AS n "
            "FROM person_mentions pm "
            "JOIN chunks c ON c.file_id = pm.file_id "
            "JOIN entities e ON e.chunk_id = c.id "
            "JOIN files f ON f.id = pm.file_id "
            "WHERE pm.person_id = ? "
            "  AND f.indexed_at >= ? "
            "  AND e.label IN "
            "      ('ORG','PRODUCT','WORK_OF_ART','EVENT','GPE','PROJECT') "
            "GROUP BY e.text_lower, e.label "
            "ORDER BY n DESC LIMIT ?",
            (person_id, cutoff, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        AgendaItem(
            kind="shared_topic",
            title=_redact(r["text"] or ""),
            detail=f"{r['label']} · {r['n']} doc(s)",
            extra={"label": r["label"], "n_docs": int(r["n"])},
        )
        for r in rows
    ]


# ============================ public ==============================


def build_agenda(
    conn: sqlite3.Connection,
    person_id: int,
    *,
    email_days: int = 30,
    journal_days: int = 60,
    shared_topic_days: int = 14,
) -> Agenda | None:
    """Compute a fresh agenda for ``person_id``. Returns None if the
    person doesn't exist."""
    from . import people as people_mod
    p = people_mod.get_person(conn, person_id)
    if p is None:
        return None
    last_contact_at = p.last_contact_at
    days_since = None
    if last_contact_at:
        days_since = int((time.time() - last_contact_at) / 86400.0)
    last_meeting = _last_meeting_with(conn, person_id)
    out_followups, in_followups = _open_followups_for(conn, person_id)
    open_emails = _open_email_threads_with(
        conn, person_id, days=email_days,
    )
    journal = _journal_notes_about(
        conn, p.display_name, days=journal_days,
    )
    shared = _shared_topics(
        conn, person_id, days=shared_topic_days,
    )
    # Round 20 — user-curated notes.
    user_notes_items: list[AgendaItem] = []
    try:
        for n in list_notes(conn, person_id, status="pending"):
            age = max(0, int((time.time() - n.created_at) / 86400.0))
            user_notes_items.append(AgendaItem(
                kind="user_note",
                title=_redact(n.text),
                detail=", ".join(n.tags) if n.tags else "",
                age_days=float(age),
                extra={"note_id": n.id},
            ))
    except Exception:  # noqa: BLE001
        pass
    return Agenda(
        person_id=p.id,
        person_name=p.display_name,
        last_contact_at=last_contact_at,
        days_since_contact=days_since,
        last_meeting=last_meeting,
        open_followups_outgoing=out_followups,
        open_followups_incoming=in_followups,
        open_email_threads=open_emails,
        journal_notes=journal,
        shared_topics=shared,
        generated_at=time.time(),
        user_notes=user_notes_items,
    )


def render_markdown(agenda: Agenda) -> str:
    """Plaintext-Markdown rendering — suitable for copy-paste into
    a notes app or pre-meeting review."""
    lines: list[str] = []
    lines.append(f"# 1:1 with {agenda.person_name}")
    if agenda.days_since_contact is not None:
        lines.append(f"_{agenda.days_since_contact} day(s) since last contact_\n")
    else:
        lines.append("_No prior contact recorded_\n")
    if agenda.total_items == 0:
        lines.append("_(nothing pending — clean slate)_")
        return "\n".join(lines)

    # User-curated "things to bring up" come FIRST — this is what
    # the user explicitly flagged for next-time. The rest is just
    # context.
    if agenda.user_notes:
        lines.append("## Things to bring up")
        for n in agenda.user_notes:
            tag_str = f"  _[{n.detail}]_" if n.detail else ""
            age_str = (
                f"  · {int(n.age_days)}d"
                if n.age_days is not None and n.age_days >= 1
                else ""
            )
            lines.append(f"- {n.title}{tag_str}{age_str}")
        lines.append("")

    if agenda.last_meeting:
        lines.append("## Last meeting")
        lm = agenda.last_meeting
        age = (
            f"{int(lm.age_days)}d ago"
            if lm.age_days is not None else ""
        )
        lines.append(f"- **{lm.title}** {age}")
        if lm.detail:
            lines.append(f"  - {lm.detail}")
        lines.append("")

    if agenda.open_followups_outgoing:
        lines.append("## You owe them")
        for f in agenda.open_followups_outgoing:
            extra = (f"  ({f.detail})" if f.detail else "")
            age = (
                f"  · {int(f.age_days)}d open"
                if f.age_days is not None else ""
            )
            lines.append(f"- {f.title}{extra}{age}")
        lines.append("")

    if agenda.open_followups_incoming:
        lines.append("## They owe you")
        for f in agenda.open_followups_incoming:
            extra = (f"  ({f.detail})" if f.detail else "")
            age = (
                f"  · {int(f.age_days)}d open"
                if f.age_days is not None else ""
            )
            lines.append(f"- {f.title}{extra}{age}")
        lines.append("")

    if agenda.open_email_threads:
        lines.append("## Recent email threads")
        for e in agenda.open_email_threads:
            age = (
                f"  ({int(e.age_days)}d ago)"
                if e.age_days is not None else ""
            )
            lines.append(f"- {e.title}{age}")
            if e.detail:
                lines.append(f"  - {e.detail}")
        lines.append("")

    if agenda.journal_notes:
        lines.append("## Journal mentions")
        for j in agenda.journal_notes:
            lines.append(f"- {j.title}: {j.detail}")
        lines.append("")

    if agenda.shared_topics:
        lines.append("## Shared topics this fortnight")
        for s in agenda.shared_topics:
            lines.append(f"- {s.title} ({s.detail})")
        lines.append("")

    return "\n".join(lines).rstrip()
