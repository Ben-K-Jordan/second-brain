"""Round 9-A — brain-grounded prep for upcoming external meetings.

The existing ``event_briefing.py`` runs Claude with web_search to give
"who is this person, what's their company doing" pre-meeting briefings.
That's *external* prep. This module is the complement: *internal* prep
based purely on what the user's brain already knows — prior emails
with this person, meeting transcripts mentioning them, open tasks
where they're the recipient, topics that came up the last time we
talked.

Pipeline:

  1. ``iter_upcoming_external_meetings`` — calendar events in the next
     N hours with at least one non-internal attendee.
  2. For each meeting + each external attendee, ``people.gather_full_context``
     pulls the per-person context block (round-9 shared helper).
  3. ``build_prep`` assembles a structured prep doc — pure aggregation,
     no LLM needed for assembly. The result is markdown ready for
     CLI / dashboard / morning-brief consumption.
  4. Daemon job pre-generates prep for the next ~12h of meetings so
     the user doesn't pay LLM latency on demand. Output is cached
     keyed by event_id; rebuilds when the meeting moves.

No persisted schema — prep is computed-on-read from the existing
people / files / tasks / entities tables. Cheap enough that we don't
need to cache rows in SQLite; a process-level dict is fine for the
daemon's pre-fetch.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config

log = logging.getLogger(__name__)


# ---- Tunables -------------------------------------------------------

# Look this far ahead for upcoming meetings. 24h covers tomorrow's
# day; longer horizons risk pre-fetching prep that becomes stale by
# the time the meeting happens.
_UPCOMING_LOOKAHEAD_SECONDS = 24 * 3600
# Max external attendees to surface per meeting. Bigger meetings get
# truncated rather than blowing up the prep doc.
_MAX_ATTENDEES_PER_MEETING = 4
# How long the cached prep is good for (refresh on every daemon tick
# anyway, but don't re-render mid-tick).
_CACHE_TTL_SECONDS = 30 * 60


# ---- Data shapes ----------------------------------------------------

@dataclass
class AttendeePrep:
    """Per-person prep block for one attendee of an upcoming meeting.

    Each field is ``None`` / empty when the brain has nothing on this
    person — the renderer collapses empty fields rather than printing
    a wall of "(no prior context)" placeholders.
    """
    name: str
    email: str
    days_since_seen: int          # 0 = met today, big number = stale
    n_prior_emails: int
    n_open_tasks: int
    open_task_lines: list[str] = field(default_factory=list)
    recent_mention_paths: list[str] = field(default_factory=list)
    co_topics: list[str] = field(default_factory=list)


@dataclass
class MeetingPrep:
    """Round 9-A — prep for one upcoming meeting."""
    event_id: str
    title: str
    starts_at: float
    when_str: str                 # human-readable "Tomorrow 2:00 PM"
    duration_minutes: int
    location: str
    organizer: str
    attendees: list[AttendeePrep] = field(default_factory=list)


# ---- Pipeline -------------------------------------------------------

def iter_upcoming_external_meetings(
    cfg: Config, *, lookahead_seconds: int = _UPCOMING_LOOKAHEAD_SECONDS,
) -> list:
    """Reuse the existing event_briefing iterator + filter to ones
    with at least one external attendee. Skip-pattern filtering
    mirrors meeting_thanks (we don't prep for daily standups).
    """
    try:
        from .event_briefing import iter_upcoming_events
    except ImportError:
        return []
    try:
        from .meeting_thanks import (
            _classify_attendees,
            _looks_skippable,
            _own_email_domains,
        )
    except ImportError:
        return []

    own = _own_email_domains(cfg)
    out = []
    try:
        events = list(iter_upcoming_events(cfg, lookahead_seconds))
    except Exception as e:  # noqa: BLE001
        log.warning("meeting_prep: calendar fetch failed: %s", e)
        return []
    for ev in events:
        external, _internal = _classify_attendees(
            ev.attendees, own, ev.organizer_email,
        )
        if not external:
            continue
        if _looks_skippable(ev.title, ev.duration_seconds or 0):
            continue
        out.append(ev)
    out.sort(key=lambda e: e.starts_at)
    return out


def build_prep(
    conn: sqlite3.Connection,
    cfg: Config,
    event,
    *,
    max_attendees: int = _MAX_ATTENDEES_PER_MEETING,
) -> MeetingPrep:
    """Assemble the prep block for one upcoming meeting.

    Pure aggregation across people / files / tasks / entities — no
    LLM call. Each external attendee gets one ``AttendeePrep`` slot
    populated from ``people.gather_full_context_by_alias`` when we
    know them, or a minimal "first time we're seeing this person"
    block when we don't.
    """
    from . import people as people_mod
    from .meeting_thanks import _classify_attendees, _own_email_domains

    own = _own_email_domains(cfg)
    external, _internal = _classify_attendees(
        event.attendees, own, event.organizer_email,
    )
    duration_min = (event.duration_seconds or 0) // 60
    when_str = _format_when(event.starts_at)
    attendees: list[AttendeePrep] = []
    for email in external[:max_attendees]:
        ctx = people_mod.gather_full_context_by_alias(conn, email)
        if ctx is None:
            # New contact — best we can do is the email handle.
            attendees.append(AttendeePrep(
                name=_email_to_display_name(email),
                email=email,
                days_since_seen=0,
                n_prior_emails=0,
                n_open_tasks=0,
            ))
            continue
        attendees.append(AttendeePrep(
            name=ctx.person.display_name or ctx.person.canonical_name,
            email=email,
            days_since_seen=ctx.days_since_seen,
            n_prior_emails=len(ctx.prior_emails),
            n_open_tasks=len(ctx.open_tasks),
            open_task_lines=[
                _truncate(text, 120) for _tid, text in ctx.open_tasks[:5]
            ],
            recent_mention_paths=[
                m.file_path for m in ctx.recent_mentions[:5]
            ],
            co_topics=[t for t, _n in ctx.co_topics[:6]],
        ))
    return MeetingPrep(
        event_id=event.event_id,
        title=event.title,
        starts_at=event.starts_at,
        when_str=when_str,
        duration_minutes=int(duration_min),
        location=event.location or "",
        organizer=event.organizer_email or "",
        attendees=attendees,
    )


def _format_when(starts_at: float) -> str:
    """Human-readable 'Tomorrow 2:00 PM' / 'Today 14:00' label.

    Uses local time. The prep block is read by humans in their TZ;
    UTC would be confusing on a morning brief."""
    from datetime import date, datetime, timedelta
    dt = datetime.fromtimestamp(starts_at)
    today = date.today()
    if dt.date() == today:
        prefix = "Today"
    elif dt.date() == today + timedelta(days=1):
        prefix = "Tomorrow"
    else:
        prefix = dt.strftime("%a %b %d")
    return f"{prefix} {dt.strftime('%H:%M')}"


def _email_to_display_name(email: str) -> str:
    """Cheap fallback when we have no people-table profile.
    'sarah.chen@x.com' → 'Sarah Chen'. Best-effort cosmetic."""
    handle = email.split("@", 1)[0] if "@" in email else email
    parts = handle.replace(".", " ").replace("_", " ").replace("-", " ").split()
    return " ".join(p.capitalize() for p in parts) or email


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rstrip() + "…"


# ---- Markdown render ------------------------------------------------

def render_prep_markdown(prep: MeetingPrep) -> str:
    """Render a single MeetingPrep as scannable markdown. Sections
    fold up gracefully when empty — a first-time contact shows a
    minimal 'we don't know them yet' block rather than a wall of
    placeholders."""
    lines: list[str] = []
    lines.append(f"# {prep.title}")
    bits = [prep.when_str]
    if prep.duration_minutes:
        bits.append(f"{prep.duration_minutes} min")
    if prep.location:
        bits.append(prep.location)
    lines.append("_" + " · ".join(bits) + "_")
    lines.append("")
    if not prep.attendees:
        lines.append(
            "_(no external attendees; this is an internal meeting "
            "and prep was skipped)_",
        )
        return "\n".join(lines)
    for a in prep.attendees:
        lines.append(f"## {a.name} `<{a.email}>`")
        if a.days_since_seen == 0 and a.n_prior_emails == 0:
            lines.append(
                "_First time you're seeing this person in your brain._",
            )
            lines.append("")
            continue
        meta = []
        if a.days_since_seen:
            meta.append(f"{a.days_since_seen}d since last seen")
        if a.n_prior_emails:
            meta.append(f"{a.n_prior_emails} prior email(s)")
        if a.n_open_tasks:
            meta.append(f"{a.n_open_tasks} open task(s)")
        if meta:
            lines.append("_" + " · ".join(meta) + "_")
        if a.open_task_lines:
            lines.append("")
            lines.append("**Open with this person:**")
            for t in a.open_task_lines:
                lines.append(f"- [ ] {t}")
        if a.co_topics:
            lines.append("")
            lines.append("**Topics that come up:** " + ", ".join(a.co_topics))
        if a.recent_mention_paths:
            lines.append("")
            lines.append("**Recent context:**")
            for p in a.recent_mention_paths[:3]:
                lines.append(f"- `{p}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---- Daemon-cached pre-fetch ----------------------------------------

# Process-level cache keyed by event_id — we don't persist this since
# upstream data (mentions / emails / tasks) is the source of truth and
# rebuilds are cheap. Refreshed every daemon tick that hits this job.
_PREP_CACHE: dict[str, tuple[float, MeetingPrep]] = {}


def get_prep_cached(
    conn: sqlite3.Connection, cfg: Config, event,
) -> MeetingPrep:
    """Cache-aware variant. Returns the cached prep when fresh,
    otherwise rebuilds. Used by the daemon's pre-fetch + the
    on-demand /prep dashboard view."""
    cached = _PREP_CACHE.get(event.event_id)
    now = time.time()
    if cached and (now - cached[0]) < _CACHE_TTL_SECONDS:
        return cached[1]
    prep = build_prep(conn, cfg, event)
    _PREP_CACHE[event.event_id] = (now, prep)
    return prep


def upcoming_preps(
    conn: sqlite3.Connection, cfg: Config,
    *, lookahead_seconds: int = _UPCOMING_LOOKAHEAD_SECONDS,
) -> list[MeetingPrep]:
    """Build prep for every upcoming external meeting in the window.
    Cached; safe to call multiple times per tick."""
    events = iter_upcoming_external_meetings(
        cfg, lookahead_seconds=lookahead_seconds,
    )
    return [get_prep_cached(conn, cfg, ev) for ev in events]


def prefetch_upcoming(
    conn: sqlite3.Connection, cfg: Config,
) -> int:
    """Daemon entrypoint — warm the cache for upcoming meetings so
    on-demand reads (CLI / dashboard) are instant. Returns count
    of meetings prepped."""
    preps = upcoming_preps(conn, cfg)
    if preps:
        log.info("meeting_prep: prefetched %d upcoming meeting(s)", len(preps))
    return len(preps)
