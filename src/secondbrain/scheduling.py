"""Round 19 (Phase EA-4) — proactive scheduling helper.

A real EA's biggest daily verb is *coordinating time*: "find 30 min
with Sarah next week, prefer Tuesday afternoon, draft the invite."

The brain has Google Calendar read access via MCP. This module
provides the *logic* for the find-time + draft-invite flow. The
actual calendar API call is delegated to the existing MCP tools
(``mcp__calendar__list_events`` etc.) so we don't re-implement OAuth.

What this module does:

  1. ``find_open_slots`` — given the user's busy events for a date
     window + a duration + preferences (days-of-week, hours-of-day,
     buffer-around-existing-meetings), return a ranked list of
     candidate slots.
  2. ``draft_proposal_email`` — compose a "here are 3 times that
     work for me" email in the user's voice (round-13 voice profile)
     for the user to send to the other party.
  3. ``parse_busy_blocks`` — adapter from the MCP calendar tool's
     event-list shape to our internal ``BusyBlock`` shape.

The dashboard / chat surface calls these helpers; the user picks
the slot and confirms the invite via their email client (or the
calendar's own UI).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

log = logging.getLogger(__name__)


@dataclass
class BusyBlock:
    start: datetime
    end: datetime
    title: str = ""


@dataclass
class TimeSlot:
    start: datetime
    end: datetime
    rank: float = 0.0    # higher = better fit for user preferences

    @property
    def duration_minutes(self) -> int:
        return int((self.end - self.start).total_seconds() / 60)


@dataclass
class SchedulingPrefs:
    """Optional user preferences that shape slot ranking.

    All times in user's local TZ (no DST math here — caller handles
    that). Lists of weekdays use Python's Monday=0..Sunday=6.
    """
    duration_minutes: int = 30
    earliest_hour: int = 9            # don't propose before 9am local
    latest_hour: int = 17             # don't propose after 5pm local
    preferred_weekdays: list[int] | None = None  # e.g. [1, 2, 3] = Tue-Thu
    preferred_hours: list[int] | None = None     # e.g. [10, 14, 15]
    avoid_weekdays: list[int] | None = None      # e.g. [0, 4] = Mon, Fri
    buffer_minutes: int = 15          # padding around existing meetings
    max_slots: int = 6


def find_open_slots(
    busy: list[BusyBlock],
    *,
    window_start: date,
    window_end: date,
    prefs: SchedulingPrefs | None = None,
) -> list[TimeSlot]:
    """Return ranked candidate slots in [window_start, window_end].

    ``busy`` is the user's existing calendar (already filtered to
    the right calendar). The function:

      1. Walks each day in the window.
      2. Within ``[earliest_hour, latest_hour]`` builds candidate
         slots of ``duration_minutes`` length, stepped by 30 min.
      3. Drops slots that overlap any busy block (with buffer).
      4. Drops slots whose weekday is in ``avoid_weekdays``.
      5. Ranks remaining slots by preference matching.

    Returns up to ``prefs.max_slots`` slots, ordered best-first.
    """
    if prefs is None:
        prefs = SchedulingPrefs()
    if window_end < window_start:
        return []

    # Inflate busy blocks by buffer.
    buf = timedelta(minutes=max(0, prefs.buffer_minutes))
    inflated = [
        (b.start - buf, b.end + buf) for b in busy
    ]

    candidates: list[TimeSlot] = []
    cur_day = window_start
    while cur_day <= window_end:
        weekday = cur_day.weekday()
        if (
            prefs.avoid_weekdays
            and weekday in set(prefs.avoid_weekdays)
        ):
            cur_day = cur_day + timedelta(days=1)
            continue
        # Iterate 30-min step starts within the workday.
        h = prefs.earliest_hour
        m = 0
        while h < prefs.latest_hour:
            slot_start = datetime.combine(cur_day, time(h, m))
            slot_end = slot_start + timedelta(
                minutes=prefs.duration_minutes,
            )
            if slot_end.hour > prefs.latest_hour or (
                slot_end.hour == prefs.latest_hour
                and slot_end.minute > 0
            ):
                break
            # Reject if it overlaps any busy block.
            ok = True
            for bs, be in inflated:
                if slot_start < be and slot_end > bs:
                    ok = False
                    break
            if ok:
                candidates.append(TimeSlot(start=slot_start, end=slot_end))
            # Advance 30 min.
            m += 30
            if m >= 60:
                h += 1
                m = 0
        cur_day = cur_day + timedelta(days=1)

    if not candidates:
        return []

    # Rank: preferred weekday + preferred hour boost.
    pref_wd = set(prefs.preferred_weekdays or [])
    pref_hr = set(prefs.preferred_hours or [])
    for c in candidates:
        score = 0.0
        wd = c.start.weekday()
        if pref_wd and wd in pref_wd:
            score += 10.0
        if pref_hr and c.start.hour in pref_hr:
            score += 6.0
        # Slight preference for not-Monday-morning, not-Friday-late.
        if wd == 0 and c.start.hour < 11:
            score -= 2.0
        if wd == 4 and c.start.hour >= 15:
            score -= 2.0
        # Prefer earlier in the window (sooner is more useful).
        days_out = (c.start.date() - window_start).days
        score -= 0.3 * days_out
        c.rank = score

    candidates.sort(key=lambda s: s.rank, reverse=True)
    return candidates[: prefs.max_slots]


def parse_busy_blocks(events: list[dict]) -> list[BusyBlock]:
    """Adapter from the Google Calendar MCP / list_events shape.

    Each event has at least: ``start`` and ``end`` (each with
    ``dateTime`` ISO string OR ``date`` for all-day). We treat
    all-day events as "busy 9am-5pm" since all-day blocks usually
    mean OOO/conference/etc. — caller can override if needed.
    """
    out: list[BusyBlock] = []
    for ev in events or []:
        try:
            title = (ev.get("summary") or "").strip()
            start_obj = ev.get("start") or {}
            end_obj = ev.get("end") or {}
            if "dateTime" in start_obj and "dateTime" in end_obj:
                # Strip TZ for naive comparison; caller is in local.
                s = _parse_iso_naive(start_obj["dateTime"])
                e = _parse_iso_naive(end_obj["dateTime"])
            elif "date" in start_obj and "date" in end_obj:
                # All-day → 9am-5pm of each day in range.
                d_start = date.fromisoformat(start_obj["date"])
                d_end = date.fromisoformat(end_obj["date"])
                cur = d_start
                while cur < d_end:
                    out.append(BusyBlock(
                        start=datetime.combine(cur, time(9, 0)),
                        end=datetime.combine(cur, time(17, 0)),
                        title=title or "(all-day)",
                    ))
                    cur = cur + timedelta(days=1)
                continue
            else:
                continue
            out.append(BusyBlock(start=s, end=e, title=title))
        except Exception as e:  # noqa: BLE001
            log.debug("scheduling: skipping malformed event: %s", e)
            continue
    return out


def _parse_iso_naive(iso: str) -> datetime:
    """Parse an ISO timestamp, dropping any TZ info to compare in
    local-naive terms. The Google Calendar MCP returns user-local
    times in the user's calendar TZ, so naive is fine here."""
    s = iso.strip()
    # Strip trailing Z or ±HH:MM.
    if s.endswith("Z"):
        s = s[:-1]
    elif len(s) >= 6 and (s[-6] in "+-") and s[-3] == ":":
        s = s[:-6]
    return datetime.fromisoformat(s)


# ============================ proposal drafter =====================


def draft_proposal_email(
    *,
    recipient_name: str,
    slots: list[TimeSlot],
    user_greeting: str = "Hi",
    user_signoff: str = "Best",
    user_name: str = "",
    purpose: str = "",
    timezone_label: str = "",
) -> str:
    """Compose a "here are some times that work" email body.

    Round-13 voice-profile style: greeting + brief framing + bullet
    list of slots + sign-off. Uses the user's own greeting/sign-off
    so it doesn't sound canned. No LLM call — deterministic
    formatting (the LLM-generated variant lives in the chat tool).
    """
    name_safe = (recipient_name or "").strip() or "there"
    greeting = f"{user_greeting} {name_safe},"
    framing = (
        f"Wanted to find time {('to ' + purpose) if purpose else 'to chat'}"
        f" — here are a few slots that work for me"
        f"{(' (' + timezone_label + ')') if timezone_label else ''}:"
    )
    slot_lines = []
    for s in slots:
        # "Tuesday, Apr 23 · 2:00–2:30 PM"
        dow = s.start.strftime("%A")
        # Portable %-d: format with %d then strip leading zero.
        date_str = s.start.strftime("%b %d").replace(" 0", " ")
        time_str = (
            f"{_fmt_time(s.start)}–{_fmt_time(s.end)}"
        )
        slot_lines.append(f"  · {dow}, {date_str} · {time_str}")
    slots_block = "\n".join(slot_lines) if slot_lines else "  · (no slots)"
    closing = (
        "Let me know which works (or none if I should try another week)."
    )
    sig = (
        f"{user_signoff},\n{user_name}" if user_name
        else user_signoff
    )
    return (
        f"{greeting}\n\n"
        f"{framing}\n{slots_block}\n\n"
        f"{closing}\n\n"
        f"{sig}"
    )


def _fmt_time(dt: datetime) -> str:
    """'2:00 PM' / '10:30 AM' formatting that's portable across
    Windows/Linux (where %-I behavior differs)."""
    hr = dt.hour % 12 or 12
    minute = dt.minute
    suffix = "AM" if dt.hour < 12 else "PM"
    if minute == 0:
        return f"{hr} {suffix}"
    return f"{hr}:{minute:02d} {suffix}"
