"""Round 16 (Phase G) — chronological timeline of everything.

A unified read-only view that merges every dated signal in the brain
into one stream. Lets you ask "what was happening Oct 12?" and just
look. Sources:

  - Files indexed (any kind)
  - Tasks created and completed
  - Habits checked in
  - Journal entries
  - Health metric daily summaries
  - Email triage classifications (urgent/fyi/etc.)
  - Meeting transcripts
  - Notifications fired
  - Weekly letters generated
  - Insights surfaced
  - Synthesis runs

Each event is a uniform shape so the dashboard can render them all in
the same row format.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta

log = logging.getLogger(__name__)


@dataclass
class TimelineEvent:
    ts: float
    kind: str        # 'file_indexed' | 'task_done' | 'task_created' | ...
    title: str
    detail: str = ""
    href: str = ""
    icon: str = ""

    @property
    def date_str(self) -> str:
        return datetime.fromtimestamp(self.ts).strftime("%Y-%m-%d")

    @property
    def time_str(self) -> str:
        return datetime.fromtimestamp(self.ts).strftime("%H:%M")


def _redact(text: str | None) -> str:
    if not text:
        return ""
    try:
        from .safety import redact_text
        return redact_text(text)
    except ImportError:
        return text


def _events_files(
    conn: sqlite3.Connection, since_ts: float, until_ts: float,
) -> list[TimelineEvent]:
    out = []
    try:
        rows = conn.execute(
            "SELECT id, path, kind, indexed_at FROM files "
            "WHERE indexed_at >= ? AND indexed_at < ? "
            "ORDER BY indexed_at DESC LIMIT 500",
            (since_ts, until_ts),
        ).fetchall()
    except sqlite3.OperationalError:
        return out
    icon_for = {
        "document": "📄", "code": "📝", "audio_video": "🎙",
        "image": "🖼", "url": "🔗", "email": "✉",
        "transcript": "💬", "review": "📅", "message": "💬",
    }
    for r in rows:
        title = (r["path"] or "").rsplit("/", 1)[-1] or "(unnamed)"
        kind = r["kind"] or "file"
        out.append(TimelineEvent(
            ts=float(r["indexed_at"]),
            kind=f"file:{kind}",
            title=_redact(title),
            detail=_redact(r["path"] or ""),
            href=f"/file?path={r['path']}",
            icon=icon_for.get(kind, "📁"),
        ))
    return out


def _events_tasks(
    conn: sqlite3.Connection, since_ts: float, until_ts: float,
) -> list[TimelineEvent]:
    out = []
    try:
        rows = conn.execute(
            "SELECT id, text, status, created_at, completed_at "
            "FROM tasks "
            "WHERE (created_at >= ? AND created_at < ?) "
            "   OR (completed_at >= ? AND completed_at < ?) "
            "ORDER BY created_at DESC LIMIT 200",
            (since_ts, until_ts, since_ts, until_ts),
        ).fetchall()
    except sqlite3.OperationalError:
        return out
    for r in rows:
        if r["completed_at"] and since_ts <= r["completed_at"] < until_ts:
            out.append(TimelineEvent(
                ts=float(r["completed_at"]),
                kind="task_done",
                title=f"Completed: {_redact(r['text'])}",
                href="/tasks",
                icon="✓",
            ))
        if r["created_at"] and since_ts <= r["created_at"] < until_ts:
            out.append(TimelineEvent(
                ts=float(r["created_at"]),
                kind="task_created",
                title=f"Added task: {_redact(r['text'])}",
                href="/tasks",
                icon="+",
            ))
    return out


def _events_habits(
    conn: sqlite3.Connection, since_ts: float, until_ts: float,
) -> list[TimelineEvent]:
    out = []
    try:
        rows = conn.execute(
            "SELECT hc.checked_at, hc.note, h.name "
            "FROM habit_checkins hc JOIN habits h ON h.id = hc.habit_id "
            "WHERE hc.checked_at >= ? AND hc.checked_at < ? "
            "ORDER BY hc.checked_at DESC",
            (since_ts, until_ts),
        ).fetchall()
    except sqlite3.OperationalError:
        return out
    for r in rows:
        out.append(TimelineEvent(
            ts=float(r["checked_at"]),
            kind="habit_checkin",
            title=f"Habit: {r['name']}",
            detail=_redact(r["note"] or ""),
            href="/habits",
            icon="●",
        ))
    return out


def _events_journal(
    conn: sqlite3.Connection, since_ts: float, until_ts: float,
) -> list[TimelineEvent]:
    out = []
    try:
        rows = conn.execute(
            "SELECT date, mood, text, created_at FROM journal_entries "
            "WHERE created_at >= ? AND created_at < ? "
            "ORDER BY created_at DESC",
            (since_ts, until_ts),
        ).fetchall()
    except sqlite3.OperationalError:
        return out
    for r in rows:
        snippet = _redact((r["text"] or "")[:120])
        mood_str = f"{r['mood']}/5" if r["mood"] else "—"
        out.append(TimelineEvent(
            ts=float(r["created_at"]),
            kind="journal",
            title=f"Journal ({mood_str})",
            detail=snippet,
            href="/journal",
            icon="📓",
        ))
    return out


def _events_health(
    conn: sqlite3.Connection, since_ts: float, until_ts: float,
) -> list[TimelineEvent]:
    """Daily health snapshot: one event per (date, sleep_score) row."""
    out = []
    try:
        rows = conn.execute(
            "SELECT date, value FROM health_metrics "
            "WHERE metric = 'sleep_score' "
            "  AND date >= ? AND date < ? "
            "ORDER BY date DESC",
            (date.fromtimestamp(since_ts).isoformat(),
             date.fromtimestamp(until_ts).isoformat()),
        ).fetchall()
    except (sqlite3.OperationalError, AttributeError):
        return out
    for r in rows:
        try:
            d = date.fromisoformat(r["date"])
        except (ValueError, TypeError):
            continue
        ts = datetime(d.year, d.month, d.day, 9, 0).timestamp()
        out.append(TimelineEvent(
            ts=ts,
            kind="health_daily",
            title=f"Sleep score {r['value']:.0f}",
            href="/health",
            icon="💤",
        ))
    return out


def _events_email_triage(
    conn: sqlite3.Connection, since_ts: float, until_ts: float,
) -> list[TimelineEvent]:
    out = []
    try:
        rows = conn.execute(
            "SELECT ec.label, ec.confidence, f.path, f.indexed_at "
            "FROM email_classifications ec "
            "JOIN files f ON f.id = ec.file_id "
            "WHERE f.indexed_at >= ? AND f.indexed_at < ? "
            "ORDER BY f.indexed_at DESC LIMIT 100",
            (since_ts, until_ts),
        ).fetchall()
    except sqlite3.OperationalError:
        return out
    icon_for = {"urgent": "🔴", "important": "🟠", "fyi": "🔵",
                "social": "🟢", "promo": "⚪"}
    for r in rows:
        title = (r["path"] or "").rsplit("/", 1)[-1] or "(email)"
        out.append(TimelineEvent(
            ts=float(r["indexed_at"]),
            kind=f"email:{r['label']}",
            title=f"Email ({r['label']}): {_redact(title)[:60]}",
            href=f"/file?path={r['path']}",
            icon=icon_for.get(r["label"], "✉"),
        ))
    return out


def _events_notifications(
    conn: sqlite3.Connection, since_ts: float, until_ts: float,
) -> list[TimelineEvent]:
    out = []
    try:
        rows = conn.execute(
            "SELECT created_at, kind, title, body, href "
            "FROM notifications "
            "WHERE created_at >= ? AND created_at < ? "
            "ORDER BY created_at DESC",
            (since_ts, until_ts),
        ).fetchall()
    except sqlite3.OperationalError:
        return out
    for r in rows:
        out.append(TimelineEvent(
            ts=float(r["created_at"]),
            kind=f"notif:{r['kind']}",
            # Round 18 fix (audit-found gap M4) — every other event
            # source applies _redact() to title; this one passed it
            # through raw. Notification titles built from email
            # subjects ("Re: API key for ...") could leak the
            # secret-shaped substrings the round-13 redaction
            # invariant was supposed to mask uniformly.
            title=_redact(r["title"] or ""),
            detail=_redact(r["body"] or ""),
            href=r["href"] or "/notifications",
            icon="🔔",
        ))
    return out


def _events_weekly_letters(
    conn: sqlite3.Connection, since_ts: float, until_ts: float,
) -> list[TimelineEvent]:
    out = []
    try:
        rows = conn.execute(
            "SELECT week_end, generated_at FROM weekly_letters "
            "WHERE generated_at >= ? AND generated_at < ? "
            "ORDER BY generated_at DESC",
            (since_ts, until_ts),
        ).fetchall()
    except sqlite3.OperationalError:
        return out
    for r in rows:
        out.append(TimelineEvent(
            ts=float(r["generated_at"]),
            kind="weekly_letter",
            title=f"Weekly letter: {r['week_end']}",
            href=f"/review?week_end={r['week_end']}",
            icon="📅",
        ))
    return out


def assemble(
    conn: sqlite3.Connection,
    since_ts: float,
    until_ts: float,
    *,
    kinds: set[str] | None = None,
) -> list[TimelineEvent]:
    """Pull every signal in the [since, until) window. Returned
    events are sorted by ts descending (newest first).

    ``kinds`` is an optional set of category prefixes to include.
    Each event has a category prefix like 'file', 'task', 'journal',
    'habit', 'health', 'email', 'notif', 'weekly'. Pass e.g.
    ``{'task', 'journal'}`` to filter.
    """
    sources = [
        ("file", _events_files),
        ("task", _events_tasks),
        ("habit", _events_habits),
        ("journal", _events_journal),
        ("health", _events_health),
        ("email", _events_email_triage),
        ("notif", _events_notifications),
        ("weekly", _events_weekly_letters),
    ]
    out: list[TimelineEvent] = []
    for prefix, fn in sources:
        if kinds is not None and prefix not in kinds:
            continue
        try:
            out.extend(fn(conn, since_ts, until_ts))
        except Exception:  # noqa: BLE001
            log.exception("timeline source %s failed", prefix)
    out.sort(key=lambda e: e.ts, reverse=True)
    return out


def parse_window(
    when: str | None = None,
    days: int = 1,
) -> tuple[float, float]:
    """Return (since_ts, until_ts) for a date window.

    ``when='2025-04-12'`` → that day; ``when=None`` → today.
    ``days=N`` widens to N days ending at ``when`` (inclusive).

    Round 17 fix (audit-found gap A-window) — clamp ``days`` to
    >= 1. Previously ``days=0`` silently produced a 1-day window
    (start = end), and ``days=-3`` yielded a future-shifted window
    via the negative timedelta. Now both raise ValueError so a CLI /
    MCP caller passing nonsense gets immediate feedback.
    """
    if days < 1:
        raise ValueError(
            f"days must be >= 1; got {days}",
        )
    if when:
        try:
            end_d = date.fromisoformat(when)
        except ValueError:
            end_d = date.today()
    else:
        end_d = date.today()
    start_d = end_d - timedelta(days=days - 1)
    since = datetime(start_d.year, start_d.month, start_d.day).timestamp()
    until = (
        datetime(end_d.year, end_d.month, end_d.day) + timedelta(days=1)
    ).timestamp()
    return since, until


def group_by_date(
    events: list[TimelineEvent],
) -> dict[str, list[TimelineEvent]]:
    """Bucket events by 'YYYY-MM-DD'. Insertion order = newest day first
    because the input list is sorted desc."""
    out: dict[str, list[TimelineEvent]] = {}
    for e in events:
        out.setdefault(e.date_str, []).append(e)
    return out
