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

import json
import logging
import sqlite3
import threading
import time as _time_mod
import weakref as _weakref
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

log = logging.getLogger(__name__)

_SCHEMA_INITIALIZED: _weakref.WeakSet = _weakref.WeakSet()
# Round 21 fix (audit-found gap F1) — serialise writes from
# dashboard threads + chat tools.
_WRITE_LOCK = threading.RLock()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Round 20 — per-person scheduling preferences + outcome log.

    ``scheduling_prefs`` stores per-person overrides ("Sarah prefers
    Tue afternoons"). ``scheduling_proposals`` logs proposals we've
    drafted for outcome tracking.
    """
    try:
        if conn in _SCHEMA_INITIALIZED:
            return
    except TypeError:
        pass
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS scheduling_prefs (
            person_id INTEGER PRIMARY KEY REFERENCES people(id)
                ON DELETE CASCADE,
            preferred_weekdays_json TEXT NOT NULL DEFAULT '[]',
            preferred_hours_json TEXT NOT NULL DEFAULT '[]',
            avoid_weekdays_json TEXT NOT NULL DEFAULT '[]',
            duration_minutes INTEGER,
            buffer_minutes INTEGER,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS scheduling_proposals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER REFERENCES people(id) ON DELETE SET NULL,
            person_name TEXT NOT NULL DEFAULT '',
            slots_json TEXT NOT NULL DEFAULT '[]',
            email_body TEXT NOT NULL DEFAULT '',
            proposed_at REAL NOT NULL,
            chosen_slot_iso TEXT,        -- which the recipient picked
            outcome TEXT NOT NULL DEFAULT 'pending'
                CHECK(outcome IN ('pending', 'scheduled',
                                   'declined', 'expired'))
        );
        CREATE INDEX IF NOT EXISTS idx_scheduling_proposals_at
            ON scheduling_proposals(proposed_at DESC);
    """)
    conn.commit()
    try:
        _SCHEMA_INITIALIZED.add(conn)
    except TypeError:
        pass


@dataclass
class PersonPrefs:
    person_id: int
    preferred_weekdays: list[int]
    preferred_hours: list[int]
    avoid_weekdays: list[int]
    duration_minutes: int | None
    buffer_minutes: int | None


def get_person_prefs(
    conn: sqlite3.Connection, person_id: int,
) -> PersonPrefs | None:
    _ensure_schema(conn)
    row = conn.execute(
        "SELECT * FROM scheduling_prefs WHERE person_id = ?",
        (person_id,),
    ).fetchone()
    if not row:
        return None
    return PersonPrefs(
        person_id=int(row["person_id"]),
        preferred_weekdays=json.loads(
            row["preferred_weekdays_json"] or "[]",
        ),
        preferred_hours=json.loads(
            row["preferred_hours_json"] or "[]",
        ),
        avoid_weekdays=json.loads(row["avoid_weekdays_json"] or "[]"),
        duration_minutes=row["duration_minutes"],
        buffer_minutes=row["buffer_minutes"],
    )


def set_person_prefs(
    conn: sqlite3.Connection,
    person_id: int,
    *,
    preferred_weekdays: list[int] | None = None,
    preferred_hours: list[int] | None = None,
    avoid_weekdays: list[int] | None = None,
    duration_minutes: int | None = None,
    buffer_minutes: int | None = None,
) -> None:
    """Upsert prefs. Lists default to [] when explicitly set; ints
    are stored as-is (None means no override)."""
    _ensure_schema(conn)
    existing = get_person_prefs(conn, person_id)
    pw = (
        preferred_weekdays
        if preferred_weekdays is not None
        else (existing.preferred_weekdays if existing else [])
    )
    ph = (
        preferred_hours
        if preferred_hours is not None
        else (existing.preferred_hours if existing else [])
    )
    aw = (
        avoid_weekdays
        if avoid_weekdays is not None
        else (existing.avoid_weekdays if existing else [])
    )
    dm = (
        duration_minutes
        if duration_minutes is not None
        else (existing.duration_minutes if existing else None)
    )
    bm = (
        buffer_minutes
        if buffer_minutes is not None
        else (existing.buffer_minutes if existing else None)
    )
    with _WRITE_LOCK:
        conn.execute(
            "INSERT OR REPLACE INTO scheduling_prefs"
            "(person_id, preferred_weekdays_json, preferred_hours_json, "
            " avoid_weekdays_json, duration_minutes, buffer_minutes, "
            " updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                person_id, json.dumps(pw), json.dumps(ph),
                json.dumps(aw), dm, bm, _time_mod.time(),
            ),
        )
        conn.commit()


def merge_with_global_prefs(
    cfg, person_prefs: PersonPrefs | None,
    *,
    duration_minutes: int = 30,
) -> SchedulingPrefs:
    """Combine per-person overrides with global cfg.* defaults.

    Falls back to round-19 defaults when nothing is configured.
    """
    earliest = int(getattr(cfg, "scheduling_earliest_hour", 9))
    latest = int(getattr(cfg, "scheduling_latest_hour", 17))
    default_buffer = int(getattr(cfg, "scheduling_buffer_minutes", 15))
    p = person_prefs
    return SchedulingPrefs(
        duration_minutes=(
            p.duration_minutes if p and p.duration_minutes
            else duration_minutes
        ),
        earliest_hour=earliest,
        latest_hour=latest,
        preferred_weekdays=(p.preferred_weekdays if p else None) or None,
        preferred_hours=(p.preferred_hours if p else None) or None,
        avoid_weekdays=(p.avoid_weekdays if p else None) or None,
        buffer_minutes=(
            p.buffer_minutes if p and p.buffer_minutes
            else default_buffer
        ),
    )


def log_proposal(
    conn: sqlite3.Connection,
    *,
    person_id: int | None,
    person_name: str,
    slots: list,
    email_body: str,
) -> int:
    _ensure_schema(conn)
    try:
        from .safety import redact_text
        person_name = redact_text(person_name)
        email_body = redact_text(email_body)
    except ImportError:
        pass
    slots_json = json.dumps([
        {
            "start_iso": s.start.isoformat(),
            "end_iso": s.end.isoformat(),
            "rank": s.rank,
        } for s in slots
    ])
    with _WRITE_LOCK:
        cur = conn.execute(
            "INSERT INTO scheduling_proposals"
            "(person_id, person_name, slots_json, email_body, proposed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                person_id, person_name, slots_json, email_body,
                _time_mod.time(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid or 0)


def list_recent_proposals(
    conn: sqlite3.Connection, *, limit: int = 30,
) -> list[dict]:
    _ensure_schema(conn)
    rows = conn.execute(
        "SELECT * FROM scheduling_proposals "
        "ORDER BY proposed_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [
        {
            "id": int(r["id"]),
            "person_id": r["person_id"],
            "person_name": r["person_name"],
            "slots": json.loads(r["slots_json"] or "[]"),
            "email_body": r["email_body"],
            "proposed_at": float(r["proposed_at"]),
            "outcome": r["outcome"],
            "chosen_slot_iso": r["chosen_slot_iso"],
        }
        for r in rows
    ]


def mark_proposal_outcome(
    conn: sqlite3.Connection, proposal_id: int,
    outcome: str, chosen_slot_iso: str = "",
) -> bool:
    if outcome not in ("scheduled", "declined", "expired"):
        raise ValueError(f"bad outcome: {outcome!r}")
    _ensure_schema(conn)
    with _WRITE_LOCK:
        cur = conn.execute(
            "UPDATE scheduling_proposals SET outcome = ?, "
            "chosen_slot_iso = ? WHERE id = ?",
            (outcome, chosen_slot_iso or None, proposal_id),
        )
        conn.commit()
        return cur.rowcount > 0


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
