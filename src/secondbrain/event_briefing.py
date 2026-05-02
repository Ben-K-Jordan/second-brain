"""Pre-event briefings: 'before each meeting / class, tell me what I should know'.

How it works:

  1. The daemon polls the user's calendar(s) every minute for events
     starting within ``cfg.briefing_lookahead_minutes`` (default 30).
     Sources: Google Calendar API (when authorized), and the iCal feed
     at ``CALENDAR_ICS_URL`` (when set). Both are queried directly — we
     don't rely on the brain's last sync, so a briefing 10 min before
     the event reflects the event's current state.

  2. For each event without an existing briefing, we call ``ask_brain``
     with a structured prompt. The agent uses the same tools as the
     chat — ``search_brain`` for what's already in the brain about the
     topic + attendees, ``web_search`` for public facts (LinkedIn,
     company news) when the brain doesn't already have the answer.

  3. The result is persisted to ``event_briefings``. UNIQUE on
     ``(event_id, event_source)`` so a re-run replaces rather than
     accumulating stale rows. A tray notification fires so you actually
     see it.

  4. The dashboard ``/briefings`` page surfaces upcoming events + their
     briefing status. CLI: ``secondbrain brief next``.

Cost: ~$0.05-0.15 per briefing depending on how many web searches the
model decides it needs. The Anthropic budget cap still applies.

Failures (network, budget exceeded, calendar API down) are caught,
logged, and stored as rows with ``error`` populated so the dashboard
can show what went wrong.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import requests

from .budget import BudgetExceededError, daily_spend_cents
from .chat import ask_brain
from .config import Config
from .db import (
    event_briefing_get,
    event_briefing_save,
)
from .embedder import Embedder
from .notify import notify
from .reranker import Reranker

log = logging.getLogger(__name__)


@dataclass
class CalendarEvent:
    """Unified shape across Google Calendar API + ICS feed."""
    event_id: str
    source: str                 # 'google_calendar' or 'ics'
    starts_at: float            # epoch seconds
    title: str
    url: str = ""
    location: str = ""
    description: str = ""
    organizer_email: str = ""
    attendees: list[str] = field(default_factory=list)
    calendar_name: str = ""
    duration_seconds: int = 0


# ============================ event sources ============================

def iter_upcoming_events(
    cfg: Config, lookahead_seconds: int,
) -> Iterator[CalendarEvent]:
    """Yield events from every configured calendar source that start in
    [now, now + lookahead_seconds]. Order is unspecified — caller sorts
    if it cares."""
    now = time.time()
    horizon = now + lookahead_seconds
    yield from _iter_google_calendar(cfg, now, horizon)
    yield from _iter_ics(cfg, now, horizon)


def _iter_google_calendar(
    cfg: Config, time_min_ts: float, time_max_ts: float,
) -> Iterator[CalendarEvent]:
    """Hit Google Calendar's API directly so the briefing reflects the
    current state, not the last sync. Uses the existing OAuth scaffold."""
    try:
        from .connectors._google_oauth import (
            GoogleAuthError,
            authorized_session,
            is_authorized,
        )
        from .connectors.google_calendar import GOOGLE_CALENDAR_SCOPES
    except ImportError:
        return
    if not is_authorized(cfg, GOOGLE_CALENDAR_SCOPES):
        return
    try:
        s = authorized_session(cfg, GOOGLE_CALENDAR_SCOPES)
    except GoogleAuthError as e:
        log.warning("event briefing: google calendar auth failed: %s", e)
        return
    if s is None:
        return
    iso_min = datetime.fromtimestamp(time_min_ts, tz=UTC).isoformat()
    iso_max = datetime.fromtimestamp(time_max_ts, tz=UTC).isoformat()
    api = "https://www.googleapis.com/calendar/v3"
    try:
        cal_list_resp = s.get(f"{api}/users/me/calendarList", timeout=15)
    except requests.RequestException as e:
        log.warning("event briefing: calendar list fetch failed: %s", type(e).__name__)
        s.close()
        return
    if cal_list_resp.status_code != 200:
        log.warning("event briefing: calendar list HTTP %s", cal_list_resp.status_code)
        s.close()
        return
    try:
        cals = cal_list_resp.json().get("items") or []
    except ValueError:
        cals = []
    try:
        for cal in cals:
            if cal.get("hidden"):
                continue
            cal_id = cal.get("id")
            cal_name = cal.get("summary", cal_id)
            if not cal_id:
                continue
            try:
                ev_resp = s.get(
                    f"{api}/calendars/{requests.utils.quote(cal_id, safe='')}/events",
                    params={
                        "timeMin": iso_min,
                        "timeMax": iso_max,
                        "singleEvents": "true",
                        "orderBy": "startTime",
                        "maxResults": 50,
                    },
                    timeout=15,
                )
            except requests.RequestException as e:
                log.warning(
                    "event briefing: events fetch %s failed: %s",
                    cal_name, type(e).__name__,
                )
                continue
            if ev_resp.status_code != 200:
                continue
            try:
                items = ev_resp.json().get("items") or []
            except ValueError:
                continue
            for ev in items:
                ce = _normalize_gcal_event(ev, cal_id, cal_name)
                if ce is not None:
                    yield ce
    finally:
        s.close()


def _normalize_gcal_event(
    ev: dict, cal_id: str, cal_name: str,
) -> CalendarEvent | None:
    if ev.get("status") == "cancelled":
        return None
    eid = ev.get("id")
    if not eid:
        return None
    start = ev.get("start") or {}
    starts_at = _gcal_start_ts(start)
    if starts_at == 0.0:
        return None
    end = ev.get("end") or {}
    duration = max(0, int(_gcal_start_ts(end) - starts_at))
    organizer = (ev.get("organizer") or {}).get("email") or ""
    attendees: list[str] = []
    for a in ev.get("attendees") or []:
        addr = (a.get("email") or "").strip()
        if addr and addr not in attendees and addr.lower() != organizer.lower():
            attendees.append(addr)
    return CalendarEvent(
        event_id=f"{cal_id}/{eid}",
        source="google_calendar",
        starts_at=starts_at,
        title=ev.get("summary") or "(no title)",
        url=ev.get("htmlLink") or "",
        location=ev.get("location") or "",
        description=ev.get("description") or "",
        organizer_email=organizer,
        attendees=attendees,
        calendar_name=cal_name,
        duration_seconds=duration,
    )


def _gcal_start_ts(value: dict) -> float:
    if not value:
        return 0.0
    if "dateTime" in value:
        try:
            return datetime.fromisoformat(
                value["dateTime"].replace("Z", "+00:00"),
            ).timestamp()
        except ValueError:
            return 0.0
    if "date" in value:
        try:
            d = datetime.fromisoformat(value["date"]).replace(tzinfo=UTC)
            return d.timestamp()
        except ValueError:
            return 0.0
    return 0.0


def _iter_ics(
    cfg: Config, time_min_ts: float, time_max_ts: float,
) -> Iterator[CalendarEvent]:
    """Pull events from the configured ICS feed, if any."""
    import os

    ics_url = (os.environ.get("CALENDAR_ICS_URL") or "").strip()
    if not ics_url:
        return
    try:
        r = requests.get(
            ics_url, timeout=30, allow_redirects=True,
            headers={"User-Agent": "second-brain/0.0.1"},
        )
        r.raise_for_status()
    except requests.RequestException as e:
        log.warning("event briefing: ICS fetch failed: %s", type(e).__name__)
        return
    try:
        from .connectors.calendar import _parse_events
    except ImportError:
        return
    seen: set[tuple[str, str]] = set()
    for ev in _parse_events(r.text):
        uid = ev.get("uid", "")
        if not uid:
            continue
        recurrence_id = ev.get("recurrence-id", "")
        key = (uid, recurrence_id)
        if key in seen:
            continue
        seen.add(key)
        starts_at = float(ev.get("dtstart_ts") or 0.0)
        if starts_at < time_min_ts or starts_at > time_max_ts:
            continue
        ends = float(ev.get("dtend_ts") or 0.0)
        duration = max(0, int(ends - starts_at)) if ends else 0
        attendees: list[str] = []
        for raw_att in ev.get("attendee", "").split(","):
            cand = raw_att.strip()
            if cand:
                attendees.append(cand)
        yield CalendarEvent(
            event_id=uid + ("#" + recurrence_id if recurrence_id else ""),
            source="ics",
            starts_at=starts_at,
            title=ev.get("summary") or "(no title)",
            url=ev.get("url", ""),
            location=ev.get("location", ""),
            description=ev.get("description", ""),
            organizer_email=ev.get("organizer", "").lower().lstrip("mailto:"),
            attendees=attendees,
            duration_seconds=duration,
        )


# =========================== prompt building ==========================

def build_prompt(
    event: CalendarEvent, conflicts: list[CalendarEvent] | None = None,
) -> str:
    """Frame the event for the chat agent to brief on.

    When ``conflicts`` is non-empty, the prompt includes a "calendar
    conflicts" section so the model knows to mention the overlap in the
    "Anything urgent" output. Detection lives in
    ``find_overlapping_events``; we just report what was found.
    """
    lines = [
        "Generate a pre-event briefing the user can read in 30 seconds before "
        "walking in. Be tight, specific, and grounded.",
        "",
        "EVENT:",
        f"  Title: {event.title}",
        f"  When: {_human_when(event.starts_at, event.duration_seconds)}",
    ]
    if event.calendar_name:
        lines.append(f"  Calendar: {event.calendar_name}")
    if event.location:
        lines.append(f"  Location: {event.location}")
    if event.url:
        lines.append(f"  Calendar link: {event.url}")
    if event.organizer_email:
        lines.append(f"  Organizer: {event.organizer_email}")
    if event.attendees:
        lines.append(f"  Attendees: {', '.join(event.attendees[:12])}")
    if event.description:
        # Cap description so we don't blow the context window on long invites.
        d = event.description
        if len(d) > 1500:
            d = d[:1500] + "\n[...truncated]"
        lines += ["", "Description:", d]

    if conflicts:
        lines += [
            "",
            "CALENDAR CONFLICTS — these other events overlap in time. Call "
            "this out clearly under 'Anything urgent':",
            _format_conflicts(conflicts),
        ]

    lines += [
        "",
        "Please structure the briefing as:",
        "",
        "**Quick context** — when/where/who in one sentence.",
        "**What you should know** — the key 2-4 things, drawn from search_brain "
        "(the user's notes, emails, past chats, GitHub/Slack/Linear/Canvas data) "
        "and web_search for public facts about attendees, companies, or topics "
        "the brain doesn't already cover.",
        "**Suggested questions or talking points** — 2-3 specific ones, "
        "tailored to what you found.",
        "**Anything urgent** — deadlines, prep work, related items "
        "(e.g. assignments due today if it's a class; recent application "
        "movement if it's a recruiting call). If there are CALENDAR CONFLICTS "
        "above, name them explicitly.",
        "",
        "Always cite sources inline. For each attendee, briefly note who they "
        "are if findable. If the user has had past meetings or threads with "
        "them, surface a one-line summary. Don't fabricate.",
    ]
    return "\n".join(lines)


def _human_when(ts: float, duration_seconds: int) -> str:
    """e.g. 'Tue 2026-04-15 14:00 (in 12 minutes, ~30 min)'."""
    when = datetime.fromtimestamp(ts).strftime("%a %Y-%m-%d %H:%M")
    delta_seconds = ts - time.time()
    if delta_seconds < 0:
        rel = f"started {int(abs(delta_seconds) // 60)} min ago"
    elif delta_seconds < 3600:
        rel = f"in {int(delta_seconds // 60)} minutes"
    else:
        rel = f"in {round(delta_seconds / 3600, 1)} hours"
    pieces = [when, f"({rel}"]
    if duration_seconds > 0:
        pieces.append(f"~{duration_seconds // 60} min")
    return " · ".join(pieces) + ")"


# ============================== generation ============================

def generate_for_event(
    cfg: Config, conn, embedder: Embedder, reranker: Reranker | None,
    event: CalendarEvent,
    conflicts: list[CalendarEvent] | None = None,
) -> dict:
    """Run the agent + persist the result. Returns a dict summarising the
    outcome so the daemon / dashboard can react.

    ``conflicts`` is the list of overlapping events the caller already
    found (the scheduler precomputes this once per tick rather than
    re-walking the calendar per event). Pass None to skip the
    conflicts-block in the prompt.
    """
    spend_before = 0.0
    try:
        spend_before = daily_spend_cents(cfg, "anthropic")
    except Exception:  # noqa: BLE001
        pass

    prompt = build_prompt(event, conflicts=conflicts)
    payload_json = _serialize_event(event)
    try:
        response = ask_brain(cfg, conn, embedder, reranker, prompt)
    except BudgetExceededError as e:
        log.warning("event briefing: budget exceeded for %s: %s", event.title, e)
        event_briefing_save(
            conn,
            event_id=event.event_id, event_source=event.source,
            event_starts_at=event.starts_at, event_title=event.title,
            event_url=event.url, event_payload_json=payload_json,
            briefing_text=None,
            error=f"budget exceeded: {e}",
        )
        return {"ok": False, "error": "budget"}
    except Exception as e:  # noqa: BLE001
        log.warning("event briefing: failed for %s: %s", event.title, e)
        event_briefing_save(
            conn,
            event_id=event.event_id, event_source=event.source,
            event_starts_at=event.starts_at, event_title=event.title,
            event_url=event.url, event_payload_json=payload_json,
            briefing_text=None,
            error=str(e)[:500],
        )
        return {"ok": False, "error": str(e)[:200]}

    cents_spent: float | None = None
    try:
        cents_spent = max(0.0, daily_spend_cents(cfg, "anthropic") - spend_before)
    except Exception:  # noqa: BLE001
        pass

    cites_payload = [
        {
            "kind": c.kind,
            "file_path": c.file_path,
            "url": c.url,
            "page_title": c.page_title,
            "chunk_index": c.chunk_index,
            "score": round(c.score, 4),
            "text": c.text if len(c.text) <= 600 else c.text[:600] + "…",
        }
        for c in response.citations
    ]
    event_briefing_save(
        conn,
        event_id=event.event_id, event_source=event.source,
        event_starts_at=event.starts_at, event_title=event.title,
        event_url=event.url, event_payload_json=payload_json,
        briefing_text=response.text,
        citations_json=json.dumps(cites_payload),
        cents_spent=cents_spent,
    )
    return {"ok": True, "text": response.text, "cents": cents_spent or 0.0}


def _serialize_event(event: CalendarEvent) -> str:
    return json.dumps({
        "event_id": event.event_id,
        "source": event.source,
        "starts_at": event.starts_at,
        "title": event.title,
        "url": event.url,
        "location": event.location,
        "description": event.description,
        "organizer_email": event.organizer_email,
        "attendees": event.attendees,
        "calendar_name": event.calendar_name,
        "duration_seconds": event.duration_seconds,
    })


# ============================== scheduler =============================

def run_briefings_if_due(
    cfg: Config, conn, embedder: Embedder, reranker: Reranker | None,
) -> int:
    """Daemon hook: scan for upcoming events and brief on any without
    a briefing yet. Returns the number of briefings generated this call.

    Polled once a minute by the daemon — same cadence as watchlists.
    """
    if not getattr(cfg, "event_briefing_enabled", True):
        return 0
    lookahead_min = int(getattr(cfg, "briefing_lookahead_minutes", 30) or 30)
    cap_per_run = int(getattr(cfg, "briefing_max_per_run", 5) or 5)

    # Pull a wider conflict window once (next 24h) so we can detect
    # overlaps for any due event without re-walking calendars per-event.
    conflict_pool = _gather_due_events(cfg, 24 * 3600)

    generated = 0
    for event in _gather_due_events(cfg, lookahead_min * 60):
        if generated >= cap_per_run:
            log.info("event briefing: hit per-run cap (%d); deferring rest", cap_per_run)
            break
        existing = event_briefing_get(conn, event.event_id, event.source)
        # Skip if we already have a successful briefing. Errored briefings
        # get retried — likely transient (network, budget that's since reset).
        if existing is not None and not existing["error"]:
            continue
        conflicts = find_overlapping_events(event, conflict_pool)
        log.info(
            "event briefing: generating for %r (%s)%s",
            event.title, event.source,
            f" — {len(conflicts)} conflict(s)" if conflicts else "",
        )
        result = generate_for_event(
            cfg, conn, embedder, reranker, event, conflicts=conflicts,
        )
        generated += 1
        if result.get("ok"):
            try:
                mins = max(1, int((event.starts_at - time.time()) // 60))
                notify(
                    f"second-brain: brief ready for '{event.title}'",
                    f"Starts in {mins} min — open the dashboard to read it.",
                )
            except Exception as e:  # noqa: BLE001
                log.warning("event briefing notify failed: %s", e)
    return generated


def _gather_due_events(cfg: Config, lookahead_seconds: int) -> list[CalendarEvent]:
    """Materialise + sort upcoming events from every source."""
    events = list(iter_upcoming_events(cfg, lookahead_seconds))
    events.sort(key=lambda e: e.starts_at)
    return events


def find_overlapping_events(
    target: CalendarEvent, candidates: list[CalendarEvent],
) -> list[CalendarEvent]:
    """Return events that overlap with ``target`` in time, excluding the
    target itself. Used by the briefing pipeline to surface "you also have
    X scheduled at the same time."

    Two events overlap when target_start < other_end AND other_start <
    target_end. When duration is unknown (0), we treat the event as a
    single point and only flag exact-time clashes.
    """
    target_end = target.starts_at + (target.duration_seconds or 0)
    out: list[CalendarEvent] = []
    for ev in candidates:
        if ev.event_id == target.event_id and ev.source == target.source:
            continue
        ev_end = ev.starts_at + (ev.duration_seconds or 0)
        # Strict overlap when both have duration; point-overlap (==) when
        # neither does. The "or" handles either side being a point.
        if target.duration_seconds == 0 and ev.duration_seconds == 0:
            if target.starts_at == ev.starts_at:
                out.append(ev)
            continue
        if target.starts_at < ev_end and ev.starts_at < target_end:
            out.append(ev)
    return out


def _format_conflicts(conflicts: list[CalendarEvent]) -> str:
    """One-line summary of overlapping events for the briefing prompt."""
    if not conflicts:
        return ""
    pieces = []
    for c in conflicts[:5]:
        when = datetime.fromtimestamp(c.starts_at).strftime("%H:%M")
        dur = (
            f" (~{c.duration_seconds // 60} min)"
            if c.duration_seconds else ""
        )
        cal = f" · {c.calendar_name}" if c.calendar_name else ""
        pieces.append(f"{when}{dur} — {c.title}{cal}")
    extra = (
        f" + {len(conflicts) - 5} more"
        if len(conflicts) > 5 else ""
    )
    return "\n".join(pieces) + extra


# ============================== ad-hoc API ============================

def manual_event(
    title: str,
    starts_at_iso: str,
    description: str = "",
    attendees: list[str] | None = None,
    location: str = "",
) -> CalendarEvent:
    """Build a CalendarEvent from raw inputs (for the dashboard's
    'brief me on this ad-hoc event' form, or the CLI)."""
    try:
        starts_at = datetime.fromisoformat(
            starts_at_iso.replace("Z", "+00:00"),
        ).timestamp()
    except ValueError as e:
        raise ValueError(f"can't parse starts_at {starts_at_iso!r}: {e}") from e
    eid = f"manual-{int(starts_at)}-{abs(hash(title))}"
    return CalendarEvent(
        event_id=eid, source="manual",
        starts_at=starts_at,
        title=title or "(untitled event)",
        description=description,
        attendees=list(attendees or []),
        location=location,
    )


# `timedelta` referenced indirectly via datetime arithmetic; keep on graph.
_ = timedelta
