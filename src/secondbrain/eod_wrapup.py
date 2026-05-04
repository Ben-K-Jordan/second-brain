"""Round 19 (Phase EA-9) — end-of-day wrap-up.

Companion to the morning ``daily_brief``. At ~6 PM local the daemon
generates a short retrospective:

  - **Today**: tasks completed, drafts sent/persisted, journals
    written, habits checked-in, emails triaged, meetings captured
  - **Slipped**: tasks that aged past today's intent, follow-ups
    that aged another day
  - **Tomorrow**: calendar events, due-by-tomorrow follow-ups,
    overdue cadence-contacts that need a touch

Output: a structured ``EodWrapup`` dataclass + a Markdown rendering
suitable for a notification, a dashboard view, or an email digest.

No LLM call required for the basic version (this is fast aggregation
over existing tables). Optional Sonnet "narrative" version can be
called via the same MCP tool with a flag.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime
from datetime import time as _time

log = logging.getLogger(__name__)


@dataclass
class EodMetric:
    label: str
    value: int
    detail: str = ""


@dataclass
class EodSlip:
    kind: str           # 'task', 'followup'
    title: str
    age_days: int
    href: str = ""


@dataclass
class EodTomorrowItem:
    kind: str           # 'meeting', 'followup_due', 'overdue_contact'
    title: str
    when: str = ""
    href: str = ""


@dataclass
class EodWrapup:
    date: str           # ISO
    today_metrics: list[EodMetric]
    slipped: list[EodSlip]
    tomorrow: list[EodTomorrowItem]
    generated_at: float

    def to_dict(self) -> dict:
        return asdict(self)


# ============================ aggregation ===========================


def _today_bounds() -> tuple[float, float]:
    today = date.today()
    start = datetime.combine(today, _time(0, 0)).timestamp()
    end = (
        datetime.combine(today, _time(0, 0)).timestamp() + 86400.0
    )
    return start, end


def _today_metrics(conn: sqlite3.Connection) -> list[EodMetric]:
    start, end = _today_bounds()
    out: list[EodMetric] = []

    # Tasks done.
    try:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM tasks "
            "WHERE status = 'done' AND completed_at >= ? AND completed_at < ?",
            (start, end),
        ).fetchone()["n"]
        out.append(EodMetric("tasks done", int(n)))
    except sqlite3.OperationalError:
        pass

    # Email drafts persisted today.
    try:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM email_drafts "
            "WHERE created_at >= ? AND created_at < ?",
            (start, end),
        ).fetchone()["n"]
        out.append(EodMetric("drafts written", int(n)))
    except sqlite3.OperationalError:
        pass

    # Drafts marked sent today.
    try:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM email_drafts "
            "WHERE status = 'sent' AND sent_at >= ? AND sent_at < ?",
            (start, end),
        ).fetchone()["n"]
        out.append(EodMetric("drafts sent", int(n)))
    except sqlite3.OperationalError:
        pass

    # Journal entry today.
    try:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM journal_entries "
            "WHERE date = ?",
            (date.today().isoformat(),),
        ).fetchone()["n"]
        if n:
            out.append(EodMetric("journal", int(n), "written"))
    except sqlite3.OperationalError:
        pass

    # Habit check-ins.
    try:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM habit_checkins "
            "WHERE checked_at >= ? AND checked_at < ?",
            (start, end),
        ).fetchone()["n"]
        out.append(EodMetric("habits checked-in", int(n)))
    except sqlite3.OperationalError:
        pass

    # Emails triaged.
    try:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM email_classifications "
            "WHERE classified_at >= ? AND classified_at < ?",
            (start, end),
        ).fetchone()["n"]
        out.append(EodMetric("emails triaged", int(n)))
    except sqlite3.OperationalError:
        pass

    # Meetings captured.
    try:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM meeting_captures "
            "WHERE captured_at >= ? AND captured_at < ?",
            (start, end),
        ).fetchone()["n"]
        if n:
            out.append(EodMetric("meetings captured", int(n)))
    except sqlite3.OperationalError:
        pass

    # Followups added today.
    try:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM followups "
            "WHERE created_at >= ? AND created_at < ?",
            (start, end),
        ).fetchone()["n"]
        if n:
            out.append(EodMetric("followups added", int(n)))
    except sqlite3.OperationalError:
        pass

    # Followups resolved today.
    try:
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM followups "
            "WHERE status = 'resolved' "
            "  AND resolved_at >= ? AND resolved_at < ?",
            (start, end),
        ).fetchone()["n"]
        if n:
            out.append(EodMetric("followups resolved", int(n)))
    except sqlite3.OperationalError:
        pass

    return out


def _slipped(conn: sqlite3.Connection, *, limit: int = 8) -> list[EodSlip]:
    out: list[EodSlip] = []
    # Open tasks > 7 days old.
    try:
        rows = conn.execute(
            "SELECT id, text, created_at FROM tasks "
            "WHERE status = 'open' "
            "  AND created_at < ? "
            "ORDER BY created_at ASC LIMIT ?",
            (time.time() - 7 * 86400, limit),
        ).fetchall()
        for r in rows:
            age = int((time.time() - float(r["created_at"])) / 86400.0)
            out.append(EodSlip(
                kind="task",
                title=(r["text"] or "")[:100],
                age_days=age,
                href="/tasks",
            ))
    except sqlite3.OperationalError:
        pass
    # Open followups > 7 days old.
    try:
        rows = conn.execute(
            "SELECT id, topic, promised_at, direction FROM followups "
            "WHERE status = 'open' "
            "  AND COALESCE(promised_at, created_at) < ? "
            "ORDER BY COALESCE(promised_at, created_at) ASC LIMIT ?",
            (time.time() - 7 * 86400, limit),
        ).fetchall()
        for r in rows:
            age = int(
                (time.time() - float(r["promised_at"] or 0))
                / 86400.0,
            )
            label = "owed" if r["direction"] == "outgoing" else "waiting on"
            out.append(EodSlip(
                kind="followup",
                title=f"{label}: {r['topic']}",
                age_days=age,
                href=f"/followups#fu{r['id']}",
            ))
    except sqlite3.OperationalError:
        pass
    out.sort(key=lambda s: s.age_days, reverse=True)
    return out[:limit]


def _tomorrow(
    conn: sqlite3.Connection,
    *,
    limit: int = 10,
) -> list[EodTomorrowItem]:
    out: list[EodTomorrowItem] = []
    tomorrow = date.today().toordinal() + 1
    tom_start = datetime.fromordinal(tomorrow).timestamp()
    tom_end = tom_start + 86400.0
    # Followups due tomorrow.
    try:
        rows = conn.execute(
            "SELECT id, topic, direction, due_at FROM followups "
            "WHERE status = 'open' AND due_at >= ? AND due_at < ? "
            "ORDER BY due_at ASC LIMIT ?",
            (tom_start, tom_end, limit),
        ).fetchall()
        for r in rows:
            label = "owe" if r["direction"] == "outgoing" else "owed"
            out.append(EodTomorrowItem(
                kind="followup_due",
                title=f"({label}) {r['topic']}",
                when="tomorrow",
                href=f"/followups#fu{r['id']}",
            ))
    except sqlite3.OperationalError:
        pass
    # Overdue cadence contacts.
    try:
        from . import people as people_mod
        overdue = people_mod.list_overdue_contacts(
            conn, limit=5, tier_filter=["vip"],
        )
        for o in overdue:
            out.append(EodTomorrowItem(
                kind="overdue_contact",
                title=f"reach out to {o.person.display_name}",
                when=f"{o.days_since_contact}d since contact",
                href=f"/person?id={o.person.id}",
            ))
    except Exception:  # noqa: BLE001
        pass
    return out[:limit]


# ============================ public ==============================


def build_wrapup(conn: sqlite3.Connection) -> EodWrapup:
    """Compute the EOD wrap-up. Cheap; no LLM call."""
    return EodWrapup(
        date=date.today().isoformat(),
        today_metrics=_today_metrics(conn),
        slipped=_slipped(conn),
        tomorrow=_tomorrow(conn),
        generated_at=time.time(),
    )


def render_markdown(w: EodWrapup) -> str:
    lines: list[str] = [f"# End of day · {w.date}\n"]
    if w.today_metrics:
        lines.append("## Today")
        for m in w.today_metrics:
            extra = f" ({m.detail})" if m.detail else ""
            lines.append(f"- {m.label}: {m.value}{extra}")
        lines.append("")
    else:
        lines.append("## Today\n_(no recorded activity)_\n")
    if w.slipped:
        lines.append("## Slipped (>7d open)")
        for s in w.slipped:
            lines.append(f"- {s.title} · {s.age_days}d open")
        lines.append("")
    if w.tomorrow:
        lines.append("## Tomorrow")
        for t in w.tomorrow:
            extra = f" · {t.when}" if t.when else ""
            lines.append(f"- {t.title}{extra}")
        lines.append("")
    return "\n".join(lines).rstrip()


def daemon_post_eod_notification(conn: sqlite3.Connection) -> bool:
    """Daemon hook: build wrapup, surface it as a notification.
    Returns True if a notification was actually enqueued.

    Idempotent on date — repeated calls on the same day no-op.
    """
    today_iso = date.today().isoformat()
    try:
        from . import notifications
        wrapup = build_wrapup(conn)
        # Compose a one-line title + multi-line body.
        n_today = sum(m.value for m in wrapup.today_metrics)
        n_slipped = len(wrapup.slipped)
        n_tomorrow = len(wrapup.tomorrow)
        title = (
            f"End of day · {n_today} action(s) today"
            + (f" · {n_slipped} slipping" if n_slipped else "")
        )
        body_lines = []
        if wrapup.tomorrow:
            body_lines.append(
                f"{n_tomorrow} item(s) on tomorrow's plate"
            )
        if wrapup.slipped:
            body_lines.append(
                f"{n_slipped} item(s) older than a week"
            )
        body = " · ".join(body_lines) or "Nothing flagged."
        return notifications.enqueue(
            conn,
            key=f"eod:{today_iso}",
            kind="eod_wrapup",
            urgency="low",
            title=title,
            body=body,
            href="/eod",
            payload={"date": today_iso},
        )
    except Exception as e:  # noqa: BLE001
        log.warning("eod_wrapup: notification failed: %s", e)
        return False
