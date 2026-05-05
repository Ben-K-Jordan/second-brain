"""Round 22 — the EA-shaped "morning desk" assembly.

Goal: a single page (``/today``) that reads like an executive
assistant walking into your office, not like a database. Same
underlying data as the rest of the brain — different framing.

Key differences from ``daily_brief``:
  - ``daily_brief`` is comprehensive: every habit, every metric,
    every section. Designed for an emailable digest.
  - ``today`` is *prioritised*: top 3-5 decisions, top 3 calendar
    items, 1-2 contextual notes. Designed for "what do I need to
    do right now?".

Composition:
  1. **Decisions** — things requiring a yes/no/snooze answer right
     now. Pulled from triage_queue (urgent emails), followups
     (you-owe entries with explicit due_at), and notifications
     (RSVPs, contract sign-offs flagged via email_classifications).
  2. **Today's events** — the calendar slice for the next ~9 hours.
  3. **Worth knowing** — the "I noticed..." surface. Cadence-
     overdue VIPs, health anomalies, journal nudges.

Each surfaced item carries a one-line ``why`` so the user always
sees the reasoning ("Sarah hasn't replied in 4 days") rather than
just data ("Sarah · 4d"). Round-21's audit hammered this point as
the #1 trust signal.

Time-of-day mode shifts the framing:
  - **morning** (5-11am): "Good morning — here's the day"
  - **midday** (11am-2pm): "Quick afternoon update"
  - **afternoon** (2-5pm): "Wrap-up window"
  - **evening** (5-9pm): EOD-style "what got done"
  - **night** (9pm-5am): "Tomorrow looks like..."

Same data, different prompt + ordering — real EAs don't give you
9am-style stand-ups at 7pm.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date, datetime

from .config import Config

log = logging.getLogger(__name__)


# ============================ data classes ===========================


@dataclass
class Action:
    """One button on a Decision card. ``method`` is 'GET' or 'POST'.

    'POST' implies the dashboard renders a form pointing at ``href``
    (with same-origin guard); 'GET' just renders an anchor."""
    label: str
    href: str
    method: str = "GET"
    style: str = "default"  # 'primary' | 'default' | 'subtle'


@dataclass
class Decision:
    """One row in the "decisions for you this morning" surface."""
    kind: str          # 'triage_email' | 'followup_owed' | 'rsvp' | 'meeting_invite' | 'task_due'
    title: str         # one-liner
    why: str           # round-21 trust signal — "Sarah replied 4d ago"
    primary: Action
    secondary: list[Action] = field(default_factory=list)
    icon: str = "▸"
    item_id: int | None = None
    age_days: float | None = None


@dataclass
class WorthKnowing:
    """One row in the contextual "I noticed..." surface."""
    title: str
    why: str
    kind: str  # 'cadence' | 'health' | 'journal' | 'birthday' | 'stale_thread'
    action: Action | None = None
    icon: str = "·"


@dataclass
class CalendarSlice:
    """Light view of an upcoming calendar item. ``prep_href`` only
    set when meeting prep is available."""
    when: str      # "11am" / "2pm"
    title: str
    detail: str = ""   # attendees, location, etc.
    prep_href: str | None = None


@dataclass
class TodayDesk:
    """Everything ``/today`` needs to render in one shot."""
    greeting: str
    mode: str             # one of {morning, midday, afternoon, evening, night}
    decisions: list[Decision] = field(default_factory=list)
    upcoming: list[CalendarSlice] = field(default_factory=list)
    worth_knowing: list[WorthKnowing] = field(default_factory=list)
    quiet_message: str | None = None
    generated_at: float = 0.0

    def is_quiet(self) -> bool:
        """No decisions + no events = a "take the win" quiet day."""
        return (
            not self.decisions
            and not self.upcoming
            and not self.worth_knowing
        )


# ============================ time-of-day ============================


def time_of_day_mode(now: datetime | None = None) -> str:
    """Round 22 — pick the framing mode by hour-of-day. Real EAs
    pace differently across the day; this matches that."""
    h = (now or datetime.now()).hour
    if 5 <= h < 11:
        return "morning"
    if 11 <= h < 14:
        return "midday"
    if 14 <= h < 17:
        return "afternoon"
    if 17 <= h < 21:
        return "evening"
    return "night"


_GREETING_BY_MODE = {
    "morning":    "Good morning",
    "midday":     "Quick check-in",
    "afternoon":  "Afternoon snapshot",
    "evening":    "Wrapping up the day",
    "night":      "Tomorrow looks like",
}


def greeting_for(
    user_name: str, mode: str, now: datetime | None = None,
) -> str:
    """Compose the greeting line. e.g.
    "Good morning, Ben — Tuesday, May 5"."""
    now = now or datetime.now()
    base = _GREETING_BY_MODE.get(mode, "Hello")
    name_part = f", {user_name}" if user_name else ""
    weekday = now.strftime("%A")
    # Windows strftime doesn't honor %-d (POSIX no-pad). Use the
    # platform fallbacks: try POSIX → Windows → manual.
    try:
        date_part = now.strftime("%B %-d")
    except ValueError:
        try:
            date_part = now.strftime("%B %#d")
        except ValueError:
            date_part = f"{now.strftime('%B')} {now.day}"
    return f"{base}{name_part} — {weekday}, {date_part}"


# ============================ decision builders ======================


def _decisions_from_followups(
    conn: sqlite3.Connection, *, limit: int = 3,
) -> list[Decision]:
    """Pull the most urgent open follow-ups (you-owe direction,
    sorted by overdue-first, then due_at, then promised_at)."""
    try:
        from . import followups_ops
        rows = followups_ops.list_visible_open(
            conn, direction="outgoing", limit=10,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("today: followups query failed: %s", e)
        return []
    now = time.time()
    sorted_rows = sorted(
        rows,
        key=lambda f: (
            0 if (f.due_at and f.due_at < now) else 1,
            f.due_at or float("inf"),
            -(f.promised_at or 0),
        ),
    )[:limit]
    out = []
    for f in sorted_rows:
        why = _followup_why(f)
        title = (
            f"{f.topic} → {f.person_name}"
            if f.person_name else f.topic
        )
        out.append(Decision(
            kind="followup_owed",
            title=title,
            why=why,
            primary=Action(
                label="Mark done",
                href=f"/followups/{f.id}/resolve",
                method="POST", style="primary",
            ),
            secondary=[
                Action(
                    label="Snooze 1d",
                    href=f"/followups/{f.id}/snooze?days=1",
                    method="POST", style="subtle",
                ),
                Action(
                    label="Edit",
                    href=f"/followups#fu{f.id}",
                ),
            ],
            icon="📌",
            item_id=f.id,
            age_days=(
                (now - f.promised_at) / 86400.0
                if f.promised_at else None
            ),
        ))
    return out


def _followup_why(f) -> str:
    """One-line explanation. Round 21's audit nailed why-lines as
    the trust signal."""
    now = time.time()
    if f.due_at and f.due_at < now:
        days_late = max(1, int((now - f.due_at) / 86400.0))
        return f"Past due by {days_late}d"
    if f.due_at:
        days_left = max(0, int((f.due_at - now) / 86400.0))
        return f"Due in {days_left}d"
    if f.promised_at:
        days_old = max(1, int((now - f.promised_at) / 86400.0))
        return f"Promised {days_old}d ago"
    return "No specific due date"


def _decisions_from_triage(
    conn: sqlite3.Connection, *, limit: int = 3,
) -> list[Decision]:
    """Top emails awaiting decision today. Uses triage_queue's
    ranked output."""
    try:
        from . import triage_queue
        items = triage_queue.build_queue(
            conn, hours=48, max_items=limit + 5,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("today: triage query failed: %s", e)
        return []
    out = []
    for it in items[:limit]:
        sender = it.from_display or it.from_email or "(unknown)"
        title = (
            f"{it.subject[:60]}"
            if it.subject else
            f"Email from {sender}"
        )
        why = _triage_why(it)
        actions: list[Action] = []
        primary: Action
        if it.draft_id:
            primary = Action(
                label="Send draft",
                href=f"/drafts/{it.draft_id}/sent",
                method="POST", style="primary",
            )
            actions.append(Action(
                label="Edit draft",
                href=f"/drafts#d{it.draft_id}",
            ))
        else:
            primary = Action(
                label="Open",
                href=f"/file?file_id={it.file_id}",
            )
        actions.append(Action(
            label="Snooze 1d",
            href=f"/triage/{it.file_id}/snooze?hours=24",
            method="POST", style="subtle",
        ))
        actions.append(Action(
            label="Skip",
            href=f"/triage/{it.file_id}/skip",
            method="POST", style="subtle",
        ))
        out.append(Decision(
            kind="triage_email",
            title=f"{sender}: {title}",
            why=why,
            primary=primary,
            secondary=actions,
            icon="✉️",
            item_id=it.file_id,
        ))
    return out


def _triage_why(it) -> str:
    bits = []
    if it.is_vip:
        bits.append("VIP sender")
    label = (it.label or "").lower()
    if label == "urgent":
        bits.append("flagged urgent")
    elif label == "follow_up":
        bits.append("needs reply")
    age_h = max(1, int(getattr(it, "age_hours", 0) or 0))
    if age_h < 24:
        bits.append(f"{age_h}h old")
    else:
        bits.append(f"{int(age_h / 24)}d old")
    return " · ".join(bits) or "in your inbox"


# ============================ worth-knowing ==========================


def _worth_knowing_cadence(
    conn: sqlite3.Connection, *, limit: int = 2,
) -> list[WorthKnowing]:
    """VIPs past their cadence target."""
    try:
        from . import people as people_mod
        overdue = people_mod.list_overdue_contacts(
            conn, limit=limit, tier_filter=["vip"],
        )
    except Exception as e:  # noqa: BLE001
        log.warning("today: cadence query failed: %s", e)
        return []
    out = []
    for o in overdue:
        weeks = int(o.days_since_contact / 7)
        target = o.person.cadence_days or 0
        target_str = (
            f"; usual cadence is {target}d"
            if target else ""
        )
        out.append(WorthKnowing(
            kind="cadence",
            title=f"Reach out to {o.person.display_name}",
            why=f"{weeks}wk since contact{target_str}",
            action=Action(
                label="View",
                href=f"/person?id={o.person.id}",
            ),
            icon="👋",
        ))
    return out


def _worth_knowing_health(
    conn: sqlite3.Connection,
) -> list[WorthKnowing]:
    """Surface a health anomaly if one exists. Cheap query off
    health_metrics; doesn't trigger a sync."""
    try:
        from . import health as health_mod
        snap = health_mod.snapshot(conn)
    except Exception:  # noqa: BLE001
        return []
    if snap is None or not snap.metrics:
        return []
    # Find the most-extreme metric where delta_pct exceeds threshold.
    flagged = [
        m for m in snap.metrics
        if abs(getattr(m, "delta_pct", 0) or 0) >= 15
    ]
    if not flagged:
        return []
    worst = max(flagged, key=lambda m: abs(m.delta_pct))
    direction = "dropped" if worst.delta_pct < 0 else "up"
    pct = abs(int(worst.delta_pct))
    return [WorthKnowing(
        kind="health",
        title=f"{worst.label} {direction} {pct}%",
        why=f"vs. recent average ({worst.window_days}d window)",
        action=Action(label="View", href="/health"),
        icon="❤️",
    )]


def _worth_knowing_journal(
    conn: sqlite3.Connection,
) -> list[WorthKnowing]:
    """Suggest a journal nudge if it's been >3 days."""
    try:
        last_ts = conn.execute(
            "SELECT MAX(created_at) AS ts FROM journal_entries",
        ).fetchone()
        last = float(last_ts["ts"] or 0) if last_ts else 0
    except sqlite3.OperationalError:
        return []
    if not last:
        return []
    days_since = (time.time() - last) / 86400.0
    if days_since < 3:
        return []
    return [WorthKnowing(
        kind="journal",
        title="No journal entry in a few days",
        why=f"last entry {int(days_since)}d ago",
        action=Action(label="Add entry", href="/journal"),
        icon="📓",
    )]


def _worth_knowing_birthdays(
    conn: sqlite3.Connection, *, days_window: int = 7,
) -> list[WorthKnowing]:
    """Birthdays within the next ``days_window`` days."""
    try:
        rows = conn.execute(
            "SELECT id, display_name, birthday FROM people "
            "WHERE birthday IS NOT NULL AND birthday != '' "
            "LIMIT 200",
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    today = date.today()
    upcoming: list[tuple[int, str, str, int]] = []
    for r in rows:
        bday = (r["birthday"] or "").strip()
        try:
            parts = bday.split("-")
            mm, dd = (
                (int(parts[1]), int(parts[2]))
                if len(parts) == 3 else
                (int(parts[0]), int(parts[1]))
            )
        except (ValueError, IndexError):
            continue
        # Use round-17's safe-date helper (handles Feb 29 → Feb 28).
        try:
            from .notifications import _safe_date_in_year
            this_year = _safe_date_in_year(today.year, mm, dd)
            if this_year is None:
                continue
            if this_year < today:
                this_year = _safe_date_in_year(today.year + 1, mm, dd)
                if this_year is None:
                    continue
        except Exception:  # noqa: BLE001
            continue
        days_until = (this_year - today).days
        if 0 <= days_until <= days_window:
            upcoming.append(
                (int(r["id"]), r["display_name"] or "—", bday, days_until),
            )
    upcoming.sort(key=lambda x: x[3])
    out = []
    for pid, name, _bday, days_until in upcoming[:2]:
        when = (
            "today" if days_until == 0 else
            "tomorrow" if days_until == 1 else
            f"in {days_until} days"
        )
        out.append(WorthKnowing(
            kind="birthday",
            title=f"{name}'s birthday {when}",
            why="want to send a note?",
            action=Action(
                label="View",
                href=f"/person?id={pid}",
            ),
            icon="🎂",
        ))
    return out


# ============================ calendar ===============================


def _today_calendar(
    cfg: Config, *, max_items: int = 5,
) -> list[CalendarSlice]:
    """Pull today's events from the configured calendars. Cheap and
    forgiving — if calendar isn't configured, returns []."""
    try:
        from . import calendar_view
    except Exception:  # noqa: BLE001
        return []
    try:
        events = calendar_view.list_events_today(cfg)
    except Exception as e:  # noqa: BLE001
        log.warning("today: calendar fetch failed: %s", e)
        return []
    out = []
    for e in events[:max_items]:
        when = ""
        try:
            when = e.start_local.strftime("%-I:%M%p").lower()
        except (AttributeError, ValueError):
            try:
                when = e.start_local.strftime("%#I:%M%p").lower()
            except (AttributeError, ValueError):
                when = ""
        attendees = e.attendees if hasattr(e, "attendees") else []
        detail = ""
        if attendees:
            detail = (
                f"with {attendees[0]}"
                if len(attendees) == 1 else
                f"{len(attendees)} attendees"
            )
        out.append(CalendarSlice(
            when=when,
            title=e.title,
            detail=detail,
            prep_href=getattr(e, "prep_href", None),
        ))
    return out


# ============================ assembly ===============================


def assemble_today(
    cfg: Config, conn: sqlite3.Connection,
    *,
    user_name: str | None = None,
    now: datetime | None = None,
    max_decisions: int = 5,
    max_worth_knowing: int = 3,
) -> TodayDesk:
    """The single entry point. Collect everything, rank, return."""
    now = now or datetime.now()
    user_name = user_name or getattr(cfg, "user_name", "") or ""
    mode = time_of_day_mode(now)
    greeting = greeting_for(user_name, mode, now)

    # Decisions: blend triage + followups, cap to max_decisions.
    decisions: list[Decision] = []
    decisions.extend(_decisions_from_triage(conn, limit=3))
    decisions.extend(_decisions_from_followups(conn, limit=3))
    # Sort by urgency: overdue followups + age beats fresh triage.
    decisions = sorted(
        decisions,
        key=lambda d: (
            0 if (d.age_days and d.age_days >= 1) else 1,
            -(d.age_days or 0),
        ),
    )[:max_decisions]

    # Calendar slice for today.
    upcoming = _today_calendar(cfg, max_items=5)
    # In evening/night mode, surface tomorrow instead.
    if mode in ("evening", "night"):
        try:
            from . import calendar_view
            tomorrow_events = calendar_view.list_events_tomorrow(cfg)
            upcoming = []
            for e in tomorrow_events[:3]:
                when = ""
                try:
                    when = (
                        f"tom {e.start_local.strftime('%-I:%M%p').lower()}"
                    )
                except (AttributeError, ValueError):
                    try:
                        when = (
                            "tom "
                            + e.start_local.strftime('%#I:%M%p').lower()
                        )
                    except (AttributeError, ValueError):
                        when = "tom"
                upcoming.append(CalendarSlice(
                    when=when, title=e.title,
                    detail="", prep_href=None,
                ))
        except Exception:  # noqa: BLE001
            pass

    # Worth knowing.
    wk: list[WorthKnowing] = []
    wk.extend(_worth_knowing_cadence(conn, limit=2))
    wk.extend(_worth_knowing_health(conn))
    wk.extend(_worth_knowing_journal(conn))
    wk.extend(_worth_knowing_birthdays(conn))
    wk = wk[:max_worth_knowing]

    # Quiet-day message?
    quiet = None
    if not decisions and not upcoming and not wk:
        quiet = _quiet_message_for(mode)

    return TodayDesk(
        greeting=greeting,
        mode=mode,
        decisions=decisions,
        upcoming=upcoming,
        worth_knowing=wk,
        quiet_message=quiet,
        generated_at=time.time(),
    )


_QUIET_BY_MODE = {
    "morning":   "Quiet morning. Coffee on me.",
    "midday":    "Nothing on the table. Smooth so far.",
    "afternoon": "All clear. Take the focus block.",
    "evening":   "Day's wrapped. Well done.",
    "night":     "Light tomorrow — same shape, less to do.",
}


def _quiet_message_for(mode: str) -> str:
    return _QUIET_BY_MODE.get(mode, "Nothing for you right now.")


# ============================ render =================================


def render_markdown(desk: TodayDesk) -> str:
    """Markdown rendering — used by the dashboard's /today page +
    optional emailable variant. Mirrors the EA-tone layout from the
    round-22 design."""
    lines = [
        f"# {desk.greeting}",
        "",
    ]
    if desk.is_quiet() and desk.quiet_message:
        lines.append(desk.quiet_message)
        return "\n".join(lines)
    if desk.decisions:
        verb = (
            "decisions for you this morning"
            if desk.mode == "morning" else
            "things waiting on you" if desk.mode == "midday" else
            "still open" if desk.mode == "afternoon" else
            "loose ends from today" if desk.mode == "evening" else
            "open from yesterday"
        )
        lines.append(f"**{len(desk.decisions)} {verb}**")
        lines.append("")
        for d in desk.decisions:
            lines.append(f"{d.icon} **{d.title}**")
            lines.append(f"  _{d.why}_")
            lines.append("")
    if desk.upcoming:
        header = (
            "Coming up today"
            if desk.mode in ("morning", "midday", "afternoon") else
            "Tomorrow"
        )
        lines.append(f"**{header}**")
        lines.append("")
        for ev in desk.upcoming:
            prep = (
                f" ([prep ready]({ev.prep_href}))"
                if ev.prep_href else ""
            )
            detail = f" — {ev.detail}" if ev.detail else ""
            lines.append(f"- {ev.when} — {ev.title}{detail}{prep}")
        lines.append("")
    if desk.worth_knowing:
        lines.append("**Worth knowing**")
        lines.append("")
        for w in desk.worth_knowing:
            lines.append(f"{w.icon} {w.title}")
            lines.append(f"  _{w.why}_")
            lines.append("")
    lines.append("_That's it for now._")
    return "\n".join(lines)
