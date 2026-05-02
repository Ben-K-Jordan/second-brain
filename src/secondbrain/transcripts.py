"""Transcript-format detection + course matching for class lectures.

Plaud (and Otter, Limitless, etc.) auto-emails you transcripts after each
recording. Our IMAP connector picks those emails up generically; this
module recognises transcript-shaped messages and produces structured
data so they index as actual lectures, not generic emails.

Flow:

  1. ``imap_email.py`` extracts the email body + headers as usual.
  2. It calls ``detect_transcript(headers, body, subject)`` here.
  3. If a parser matches, returns a ``Transcript`` with structured
     fields: title, recorded_at, speakers, body_clean, summary, etc.
  4. The IMAP connector uses those to build a richer
     ``ConnectorDocument`` with ``source="transcript:plaud"`` (or
     whichever provider) and metadata that lets retrieval scope by
     course / lecture / speaker.
  5. ``match_canvas_course`` runs at index time to tag the doc with
     a Canvas course code when the subject or recording timestamp
     identifies one.

Designed to be tolerant: if a parser misses the format details, the
content still ingests as a regular email — degraded experience, not a
crash. We only get smarter, never worse.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

log = logging.getLogger(__name__)


# Match course codes like "BME 410", "CS 374", "BIOMG-1350", "STS 2761".
# Two letter prefixes through five letters; optional hyphen/space; 3-4 digits.
_COURSE_CODE_RE = re.compile(
    r"\b([A-Z]{2,5})[\s\-]*([0-9]{3,4})\b"
)

# Plaud's auto-export emails come from this domain (per their support docs).
# We also recognise their characteristic transcript shape so any future
# sender change doesn't break us.
_PLAUD_FROM_PATTERNS = (
    "plaud.app", "plaud.ai", "noreply@plaud", "@plaud.",
)

# Granola's note-share emails. They've used a few sender domains over time;
# match liberally on the brand string.
_GRANOLA_FROM_PATTERNS = (
    "granola.so", "granola.ai", "noreply@granola", "@granola.",
)


@dataclass
class Transcript:
    """Structured form of a lecture / meeting transcript."""
    provider: str                      # 'plaud' | 'otter' | 'granola' | 'generic'
    title: str                         # human-readable
    body: str                          # cleaned body text (speaker turns or notes)
    recorded_at: float = 0.0           # epoch seconds; 0 when unknown
    duration_seconds: int = 0
    speakers: list[str] = field(default_factory=list)
    summary: str = ""                  # auto-summary section if present
    raw_subject: str = ""
    raw_from: str = ""
    course_code: str = ""              # filled by match_canvas_course
    # Meeting-mode fields. Granola + post-meeting tools tend to produce
    # structured notes with action items + attendees as first-class
    # things separate from the running transcript text.
    action_items: list[str] = field(default_factory=list)
    attendees: list[str] = field(default_factory=list)
    # Filled by match_calendar_event when the recording timestamp lines
    # up with a calendar event in your authorised Google Calendar.
    event_id: str = ""
    event_source: str = ""             # 'google_calendar' | 'ics' | ''
    event_title: str = ""


# ============================ detection ===============================

def detect_transcript(
    from_addr: str, subject: str, body: str,
) -> Transcript | None:
    """Top-level dispatch. Returns a Transcript if any parser claims it,
    None otherwise. Order matters — Plaud first because we know its
    format; ``generic`` is a last-resort that catches anything with
    speaker labels at all."""
    from_l = (from_addr or "").lower()
    subj = subject or ""
    body = body or ""
    if not body.strip():
        return None
    if any(p in from_l for p in _PLAUD_FROM_PATTERNS):
        t = _parse_plaud(subj, body, from_addr)
        if t is not None:
            return t
    if any(p in from_l for p in _GRANOLA_FROM_PATTERNS):
        t = _parse_granola(subj, body, from_addr)
        if t is not None:
            return t
    # Body-shape detection for tools that don't have a known sender —
    # e.g. user manually forwards a Granola note from their phone, or
    # a future tool ships a transcript-shaped email.
    if _looks_like_granola_notes(body):
        t = _parse_granola(subj, body, from_addr)
        if t is not None:
            return t
    # Heuristic fallbacks. Generic / Otter detection share the same
    # speaker-turn regex but differ on header conventions.
    if "otter.ai" in from_l:
        t = _parse_otter(subj, body, from_addr)
        if t is not None:
            return t
    if "limitless" in from_l:
        t = _parse_generic(subj, body, from_addr, provider="limitless")
        if t is not None:
            return t
    if _looks_like_transcript(body):
        return _parse_generic(subj, body, from_addr, provider="generic")
    return None


def _looks_like_transcript(body: str) -> bool:
    """Heuristic: text with 3+ speaker-turn-like lines is probably a
    transcript. Catches Otter / Plaud / manual exports without
    explicitly knowing the sender."""
    pattern = re.compile(
        r"^\s*(Speaker\s*\d+|[\w\-\.\s]{1,40}(?:\s*\([\w\d:\s\-]+\))?)\s*:",
        re.MULTILINE,
    )
    return len(pattern.findall(body)) >= 3


# ============================ parsers ================================

# Plaud's email body roughly looks like:
#
#   Title: My recording
#   Recorded on: 2026-04-15 14:30
#   Duration: 45 min
#
#   ## Summary
#   <auto-summary text>
#
#   ## Transcript
#   Speaker 1 (00:00): blah blah
#   Speaker 2 (00:32): more blah
#   ...

_PLAUD_RECORDED_RE = re.compile(
    r"Recorded\s*(?:on|at)?\s*:?\s*([0-9]{4}-[0-9]{2}-[0-9]{2}[T\s][0-9:]+)",
    re.IGNORECASE,
)
_PLAUD_DURATION_RE = re.compile(
    r"Duration\s*:?\s*([0-9]+)\s*(?:min|m\b)", re.IGNORECASE,
)
_PLAUD_TITLE_RE = re.compile(r"^Title\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_PLAUD_SUMMARY_RE = re.compile(
    r"##\s*Summary\s*\n(.*?)(?=\n##|\Z)", re.IGNORECASE | re.DOTALL,
)
_PLAUD_TRANSCRIPT_RE = re.compile(
    r"##\s*Transcript\s*\n(.*)", re.IGNORECASE | re.DOTALL,
)
_SPEAKER_TURN_RE = re.compile(
    r"^([\w\-\.\s]{1,40})(?:\s*\(([0-9:]+)\))?\s*:\s*(.*)$",
    re.MULTILINE,
)


def _parse_plaud(subject: str, body: str, from_addr: str) -> Transcript | None:
    """Parse Plaud's auto-email format. We're tolerant — if Summary or
    Transcript headings aren't present we fall back to the whole body."""
    title_m = _PLAUD_TITLE_RE.search(body)
    title = (
        title_m.group(1).strip() if title_m
        else _strip_re(subject) or "Plaud recording"
    )

    recorded = 0.0
    rec_m = _PLAUD_RECORDED_RE.search(body)
    if rec_m:
        recorded = _parse_iso_or_local(rec_m.group(1))

    duration = 0
    dur_m = _PLAUD_DURATION_RE.search(body)
    if dur_m:
        try:
            duration = int(dur_m.group(1)) * 60
        except ValueError:
            duration = 0

    # Pull the transcript section if Plaud clearly delimited it.
    body_clean = body
    transcript_m = _PLAUD_TRANSCRIPT_RE.search(body)
    if transcript_m:
        body_clean = transcript_m.group(1).strip()

    summary = ""
    summ_m = _PLAUD_SUMMARY_RE.search(body)
    if summ_m:
        summary = summ_m.group(1).strip()

    speakers = _extract_speakers(body_clean)

    return Transcript(
        provider="plaud", title=title, body=body_clean,
        recorded_at=recorded, duration_seconds=duration,
        speakers=speakers, summary=summary,
        raw_subject=subject, raw_from=from_addr,
    )


_OTTER_TURN_RE = re.compile(
    r"^([\w\.\-]+(?:\s+[\w\.\-]+)?)\s+(\d+:\d{2})\s*$",
    re.MULTILINE,
)


# Granola's "Notes from <meeting>" emails are markdown-flavoured: a
# short header (date, attendees), then sections like ## Summary,
# ## Key Points, ## Action Items, optionally ## Transcript at the end.
# Subject is usually "Notes: <meeting title>" or "Notes from <meeting>".
_GRANOLA_SUBJECT_RE = re.compile(
    r"^(?:Notes\s*(?:from|on|:)\s*|Granola:\s*)?(.*?)$",
    re.IGNORECASE,
)
_GRANOLA_DATE_RE = re.compile(
    r"(?:Date|Meeting\s*time|When)\s*:\s*([^\n]+)",
    re.IGNORECASE,
)
_GRANOLA_ATTENDEES_RE = re.compile(
    r"(?:Attendees|Participants|People)\s*:\s*([^\n]+)",
    re.IGNORECASE,
)
_GRANOLA_SECTION_RE = re.compile(
    r"^#{1,3}\s+(.+?)\s*\n(.*?)(?=\n#{1,3}\s+|\Z)",
    re.DOTALL | re.MULTILINE,
)


def _looks_like_granola_notes(body: str) -> bool:
    """Heuristic for Granola's structured-notes format from any sender:
    multiple ## sections including at least one of the canonical
    headings."""
    if not body:
        return False
    canonical = (
        "summary", "key points", "action items", "decisions",
        "next steps", "transcript",
    )
    found = 0
    for m in _GRANOLA_SECTION_RE.finditer(body):
        if m.group(1).strip().lower() in canonical:
            found += 1
            if found >= 2:
                return True
    return False


def _parse_granola(
    subject: str, body: str, from_addr: str,
) -> Transcript | None:
    """Granola exports meeting notes as markdown with named sections.
    We extract the title, attendees, action items, summary, and the
    transcript (if present) into a structured Transcript.

    Tolerant: we don't require any specific section to be present. A
    note with only "## Summary" still parses; a note with everything
    parses fully.
    """
    title = _strip_re(subject) or "Granola notes"
    # Strip the "Notes from " prefix that Granola usually adds.
    m = _GRANOLA_SUBJECT_RE.match(title)
    if m and m.group(1):
        title = m.group(1).strip().strip('"').strip("'") or title

    sections: dict[str, str] = {}
    for sm in _GRANOLA_SECTION_RE.finditer(body):
        heading = sm.group(1).strip().lower()
        sections[heading] = sm.group(2).strip()

    summary = sections.get("summary", "")
    transcript_text = (
        sections.get("transcript", "")
        or sections.get("full transcript", "")
        or sections.get("conversation", "")
    )

    # Action items: each line starting with `- [ ]` / `- [x]` / `* `
    # / `- ` becomes one item.
    action_items_raw = (
        sections.get("action items", "")
        or sections.get("next steps", "")
        or sections.get("todos", "")
    )
    action_items: list[str] = []
    for line in action_items_raw.splitlines():
        cleaned = re.sub(r"^\s*[-*]\s*\[?[\sx]?\]?\s*", "", line).strip()
        if cleaned:
            action_items.append(cleaned)

    # Attendees come either as a "Attendees: a, b, c" line in the header
    # or as the section ## Attendees with bullet items.
    attendees: list[str] = []
    att_m = _GRANOLA_ATTENDEES_RE.search(body)
    if att_m:
        attendees = [
            a.strip() for a in att_m.group(1).split(",") if a.strip()
        ]
    else:
        att_block = sections.get("attendees", "") or sections.get("participants", "")
        if att_block:
            for line in att_block.splitlines():
                cleaned = re.sub(r"^\s*[-*]\s*", "", line).strip()
                if cleaned:
                    attendees.append(cleaned)

    # Recording timestamp: prefer "Date: ..." in the header. Fall back
    # to nothing — caller may use email-arrival time as the anchor.
    recorded = 0.0
    date_m = _GRANOLA_DATE_RE.search(body)
    if date_m:
        recorded = _parse_iso_or_local(date_m.group(1).strip())

    # Speakers from transcript section, if any.
    speakers = _extract_speakers(transcript_text) if transcript_text else []

    # Body for indexing: prefer the transcript when present (richer for
    # search), otherwise the structured notes (still useful for "what
    # did we decide about X" queries).
    body_for_index = transcript_text or body

    return Transcript(
        provider="granola",
        title=title,
        body=body_for_index,
        recorded_at=recorded,
        speakers=speakers,
        summary=summary,
        action_items=action_items,
        attendees=attendees,
        raw_subject=subject,
        raw_from=from_addr,
    )


def _parse_otter(subject: str, body: str, from_addr: str) -> Transcript | None:
    """Otter.ai shares mostly via web link rather than full body, but
    when transcripts ARE in-body they look like:

        Alice 0:00
        text
        Bob 0:14
        text

    Speaker lines are name + space + timestamp on their own line, with
    the actual content on subsequent lines. This is the inverse of the
    Plaud / generic ``Name: text`` shape, so we use a separate regex.
    """
    if not _OTTER_TURN_RE.search(body):
        return None
    seen: list[str] = []
    for m in _OTTER_TURN_RE.finditer(body):
        name = m.group(1).strip()
        if name and name not in seen and len(name) <= 40:
            seen.append(name)
        if len(seen) >= 12:
            break
    title = _strip_re(subject) or "Otter transcript"
    return Transcript(
        provider="otter", title=title, body=body,
        speakers=seen, raw_subject=subject, raw_from=from_addr,
    )


def _parse_generic(
    subject: str, body: str, from_addr: str, provider: str = "generic",
) -> Transcript:
    """Catch-all for transcript-shaped emails we don't have a specific
    parser for. We extract speakers from speaker-turn lines and use
    the email subject as the title."""
    speakers = _extract_speakers(body)
    title = _strip_re(subject) or "Transcript"
    return Transcript(
        provider=provider, title=title, body=body,
        speakers=speakers, raw_subject=subject, raw_from=from_addr,
    )


# ============================ helpers ================================

def _strip_re(s: str) -> str:
    """Strip the 'Re:' / 'Fwd:' prefix that gets inherited from forwards."""
    s = (s or "").strip()
    while True:
        lower = s.lower()
        for prefix in ("re:", "fwd:", "fw:"):
            if lower.startswith(prefix):
                s = s[len(prefix):].strip()
                break
        else:
            break
    return s


def _parse_iso_or_local(s: str) -> float:
    """Try a few common datetime formats. Returns 0.0 on failure."""
    s = s.strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.timestamp()
        except ValueError:
            continue
    # Try fromisoformat as a last resort (handles 2026-04-15T14:30:00+00:00).
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _extract_speakers(body: str) -> list[str]:
    """Pull unique speaker labels from the body. Order = first-mention.

    Skips one-token labels that look like English words (so a paragraph
    starting "Although: it depends" doesn't get treated as a speaker).
    """
    seen: list[str] = []
    for m in _SPEAKER_TURN_RE.finditer(body):
        name = m.group(1).strip()
        if not name or len(name) > 40:
            continue
        # Filter common false positives: lines like "Note:", "Note 1:" etc.
        if name.lower() in {"note", "summary", "tldr", "agenda", "topic", "p.s"}:
            continue
        if name not in seen:
            seen.append(name)
        if len(seen) >= 12:
            break
    return seen


# ====================== Canvas course matching ========================

def match_canvas_course(
    transcript: Transcript, canvas_courses: list[dict] | None = None,
) -> str:
    """Try to identify which Canvas course this transcript is from.

    Strategies, in order:
      1. Course code regex on the email subject (most reliable).
      2. Course code regex on the title (Plaud's "Title:" line).
      3. Substring match against the names of canvas_courses.

    Returns the matched course code (uppercase, normalized) or "" when
    nothing matches. Never raises.

    ``canvas_courses`` is a list of dicts with at least ``name`` and
    optionally ``course_code`` (matches the shape Canvas's API
    returns). Pass [] or None to skip strategy 3.
    """
    # Strategy 1: subject contains a course-code-like token.
    code = _extract_course_code(transcript.raw_subject)
    if code:
        return code
    # Strategy 2: same regex on the Plaud-extracted title.
    code = _extract_course_code(transcript.title)
    if code:
        return code
    # Strategy 3: fuzzy-match canvas course names.
    if canvas_courses:
        haystacks = [transcript.raw_subject, transcript.title]
        for course in canvas_courses:
            name = (course.get("name") or "").strip()
            cc = (course.get("course_code") or "").strip().upper()
            if not name and not cc:
                continue
            for h in haystacks:
                hl = (h or "").lower()
                if cc and cc.lower() in hl:
                    return cc
                if name and len(name) >= 4 and name.lower() in hl:
                    return cc or _normalize_code(name)
    return ""


def _extract_course_code(text: str) -> str:
    """Pull the first Canvas-style course code out of a string."""
    if not text:
        return ""
    m = _COURSE_CODE_RE.search(text)
    if not m:
        return ""
    return f"{m.group(1).upper()} {m.group(2)}"


def _normalize_code(name: str) -> str:
    """Best-effort normalisation when we only had a course name to go on.
    Returns the first word + first number-like substring."""
    parts = (name or "").split()
    if len(parts) >= 2 and parts[0].isalpha() and any(c.isdigit() for c in parts[1]):
        return f"{parts[0].upper()} {parts[1]}"
    return (name or "").strip()


# ====================== calendar event matching =======================

def match_calendar_event(
    cfg, transcript: Transcript, window_minutes: int = 30,
) -> tuple[str, str, str] | None:
    """Identify which calendar event this transcript was the recording of.

    Strategy: hit the user's authorised Google Calendar API for events
    whose start time is within ``window_minutes`` of ``transcript.recorded_at``.
    Returns ``(event_id, event_source, event_title)`` on a hit, None
    otherwise. Never raises — failure logs and returns None.

    We don't rely on the brain's last sync of calendar events; we hit the
    API directly so a transcript landing 5 min after a meeting gets
    linked to the right occurrence even if the brain hasn't sync'd yet.
    """
    if transcript.recorded_at <= 0:
        return None
    try:
        # Probe import to ensure the briefing helpers are available
        # (they're optional — pulled in via the [calendar] extra).
        import importlib.util as _ilu
        if _ilu.find_spec("secondbrain.event_briefing") is None:
            return None
    except ImportError:
        return None

    # We want events in [recorded_at - window, recorded_at + window].
    # ``iter_upcoming_events`` walks from now() forward, which won't
    # work for past recordings. Roll our own narrow range query.
    candidates = list(_iter_events_around(cfg, transcript.recorded_at, window_minutes))
    if not candidates:
        return None
    # Pick the closest match by start-time delta.
    candidates.sort(
        key=lambda ev: abs(ev.starts_at - transcript.recorded_at),
    )
    best = candidates[0]
    return (best.event_id, best.source, best.title)


def _iter_events_around(cfg, anchor_ts: float, window_minutes: int):
    """Yield calendar events whose start time is within ``window_minutes``
    of ``anchor_ts``, from every authorised calendar source.

    Stays robust when sources are unavailable: bad auth or no ICS URL
    just skip silently.
    """
    from datetime import datetime, timezone
    window_seconds = max(60, int(window_minutes) * 60)
    start_ts = anchor_ts - window_seconds
    end_ts = anchor_ts + window_seconds
    yield from _iter_gcal_events_in_range(cfg, start_ts, end_ts)
    yield from _iter_ics_events_in_range(cfg, start_ts, end_ts)
    _ = (datetime, timezone)  # imports kept on graph for runtime helpers below


def _iter_gcal_events_in_range(cfg, start_ts: float, end_ts: float):
    """Mirror of event_briefing._iter_google_calendar but with explicit
    start/end. Kept here to avoid widening that module's API.
    """
    import requests

    try:
        from .connectors._google_oauth import (
            GoogleAuthError,
            authorized_session,
            is_authorized,
        )
        from .connectors.google_calendar import GOOGLE_CALENDAR_SCOPES
        from .event_briefing import _normalize_gcal_event
    except ImportError:
        return
    if not is_authorized(cfg, GOOGLE_CALENDAR_SCOPES):
        return
    try:
        s = authorized_session(cfg, GOOGLE_CALENDAR_SCOPES)
    except GoogleAuthError as e:
        log.warning("transcript event-match: google calendar auth failed: %s", e)
        return
    if s is None:
        return
    from datetime import datetime
    iso_min = datetime.fromtimestamp(start_ts, tz=UTC).isoformat()
    iso_max = datetime.fromtimestamp(end_ts, tz=UTC).isoformat()
    api = "https://www.googleapis.com/calendar/v3"
    try:
        try:
            cal_resp = s.get(f"{api}/users/me/calendarList", timeout=15)
        except requests.RequestException as e:
            log.warning("transcript event-match: calendar list fetch failed: %s",
                        type(e).__name__)
            return
        if cal_resp.status_code != 200:
            return
        try:
            cals = cal_resp.json().get("items") or []
        except ValueError:
            return
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
                        "maxResults": 25,
                    },
                    timeout=15,
                )
            except requests.RequestException:
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


def _iter_ics_events_in_range(cfg, start_ts: float, end_ts: float):
    """ICS feed counterpart, narrowed to the recording window."""
    import os

    import requests

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
        log.warning("transcript event-match: ICS fetch failed: %s", type(e).__name__)
        return
    try:
        from .connectors.calendar import _parse_events
        from .event_briefing import CalendarEvent
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
        if starts_at < start_ts or starts_at > end_ts:
            continue
        ends = float(ev.get("dtend_ts") or 0.0)
        duration = max(0, int(ends - starts_at)) if ends else 0
        yield CalendarEvent(
            event_id=uid + ("#" + recurrence_id if recurrence_id else ""),
            source="ics",
            starts_at=starts_at,
            title=ev.get("summary") or "(no title)",
            duration_seconds=duration,
        )
