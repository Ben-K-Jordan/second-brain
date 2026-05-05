"""Round 22 — EA-toned UX copy bank.

A central place for the user-facing strings that should sound like
an executive assistant, not a database. Centralised so the tone
stays consistent across the dashboard / brief / weekly letter, and
so future tone passes are a single-file change.

Two main surfaces:
  - ``EMPTY_STATES``: per-route empty messages. The current dashboard
    has 30+ "No X." strings; this swaps them for warmer copy.
  - ``WHY``: rendering helpers for the round-22 "always show why"
    trust-signal pattern (round-21 audit's #1 finding).

Usage:
  from .ux_copy import empty_state, why_phrase
  ...
  return empty_state("triage")  # → "Inbox at zero. Nicely done."
"""

from __future__ import annotations

import time
from datetime import datetime

# ============================ empty states ===========================


EMPTY_STATES: dict[str, str] = {
    # Triage / inbox
    "triage":            "Inbox at zero. Nicely done.",
    "triage_morning":    "Inbox at zero this morning. Coffee on me.",
    # Followups
    "followups_outgoing":  "Nothing on your plate. Take the win.",
    "followups_incoming":  "Everyone's caught up to you.",
    "followups_all":       "No open follow-ups. Clean slate.",
    "followups_history":   "Nothing resolved yet — give it a week.",
    "followups_snoozed":   "No snoozed items.",
    # Tasks
    "tasks":              "All clear on the task list.",
    "tasks_done_today":   "Nothing checked off yet today — first one?",
    # Notifications
    "notifications":          "Quiet morning. Coffee on me.",
    "notifications_evening":  "Quiet evening. You're caught up.",
    # Drafts / thanks
    "drafts":             "No drafts pending. Inbox stays calm.",
    "thanks":             "No thanks-yous queued.",
    # Calendar / agenda
    "agenda_no_person":    "No people in the brain yet. Add one to start agendas.",
    "agenda_no_notes":     "_(no pending topics — add one above)_",
    "calendar_today":      "No events today. Good day for deep work.",
    "calendar_tomorrow":   "Light day tomorrow.",
    # Health / journal / habits
    "health":              "No Oura data yet — connect at /health/system.",
    "journal_today":       "Nothing journaled today yet.",
    "habits":              "No habits configured. Add one at /habits.",
    "goals":               "No goals tracked. Define one at /goals.",
    # Capture / scheduling
    "capture":             "No meeting captures yet. Index a transcript and they'll appear here.",
    "scheduling":          "No scheduling proposals logged. Use chat to find time.",
    # Search / generic
    "search":              "No matches.",
    "people":              "No people indexed yet — sync a connector first.",
    # Today desk
    "today_quiet":         "Quiet day. Take the win.",
    "today_evening":       "Day's wrapped. Well done.",
}


def empty_state(key: str, fallback: str = "Nothing here.") -> str:
    """Return the EA-toned empty message for ``key``.

    Falls back to ``fallback`` (or "Nothing here.") if the key isn't
    in the bank. Lookup is permissive so callers can pass route names
    directly without worrying about typos breaking pages.
    """
    return EMPTY_STATES.get(key, fallback)


# ============================ time-of-day adaptive ==================


def adaptive_empty(base_key: str, now: datetime | None = None) -> str:
    """Return the time-of-day-flavoured empty message when one is
    available, falling back to the base key.

    e.g. ``adaptive_empty("notifications")`` returns
    ``"Quiet evening. You're caught up."`` after 5pm and the regular
    quiet message before."""
    h = (now or datetime.now()).hour
    if 17 <= h < 21:
        evening_key = f"{base_key}_evening"
        if evening_key in EMPTY_STATES:
            return EMPTY_STATES[evening_key]
    if 5 <= h < 11:
        morning_key = f"{base_key}_morning"
        if morning_key in EMPTY_STATES:
            return EMPTY_STATES[morning_key]
    return EMPTY_STATES.get(base_key, "Nothing here.")


# ============================ why-line builders ======================


def days_ago_phrase(ts: float | None) -> str:
    """"3 days ago" / "yesterday" / "today" — humanised time delta."""
    if not ts:
        return ""
    delta = max(0, time.time() - float(ts))
    days = int(delta / 86400.0)
    if days == 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"{days} days ago"
    weeks = days // 7
    if weeks < 8:
        return f"{weeks}wk ago" if weeks == 1 else f"{weeks}wks ago"
    months = days // 30
    return f"{months}mo ago"


def overdue_phrase(due_ts: float | None) -> str:
    """"Past due 3d" / "due today" / "due in 5d"."""
    if not due_ts:
        return ""
    now = time.time()
    if due_ts < now:
        days = max(1, int((now - due_ts) / 86400.0))
        return f"past due {days}d"
    days = int((due_ts - now) / 86400.0)
    if days == 0:
        return "due today"
    if days == 1:
        return "due tomorrow"
    return f"due in {days}d"
