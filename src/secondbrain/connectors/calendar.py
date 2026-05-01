"""Calendar connector — ingests events from a public/secret ICS URL.

Google Calendar, Apple iCloud Calendar, Outlook, Notion, etc. all expose
ICS feeds. Get the URL from your calendar's sharing settings (e.g. Google
Calendar → Settings → "Secret address in iCal format") and set it as
``CALENDAR_ICS_URL`` in your environment.

Events become time-stamped documents searchable like anything else.

Caveat: if you run *both* this ICS connector and the Google Calendar API
connector against the same Google account, the same events get indexed
twice (different virtual_paths, identical content). The hash-dedup salts
by source so they stay distinct rows in `files`; pick one or the other to
avoid the noise.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from datetime import datetime, timezone

import requests

from ..config import Config
from . import ConnectorDocument

log = logging.getLogger(__name__)


def _parse_ics_dt(value: str) -> float:
    """Parse an ICS DTSTART/DTEND value to Unix timestamp.

    Handles Z-suffixed UTC, naive datetimes (treated as UTC), and date-only
    values. Returns now() on parse failure rather than crashing the feed.
    """
    if not value:
        return time.time()
    try:
        if "T" in value:
            if value.endswith("Z"):
                dt = datetime.strptime(value, "%Y%m%dT%H%M%SZ")
                return dt.replace(tzinfo=timezone.utc).timestamp()
            dt = datetime.strptime(value, "%Y%m%dT%H%M%S")
            return dt.replace(tzinfo=timezone.utc).timestamp()
        dt = datetime.strptime(value, "%Y%m%d")
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return time.time()


def _unfold(text: str) -> Iterator[str]:
    """ICS folds long lines with CRLF + space. Unfold to logical lines."""
    buf: list[str] = []
    for raw in text.splitlines():
        if raw.startswith((" ", "\t")) and buf:
            buf[-1] += raw[1:]
        else:
            buf.append(raw)
    yield from buf


def _unescape(value: str) -> str:
    return (
        value.replace("\\n", "\n")
             .replace("\\N", "\n")
             .replace("\\,", ",")
             .replace("\\;", ";")
             .replace("\\\\", "\\")
    )


def _parse_events(ics_text: str) -> Iterator[dict]:
    current: dict | None = None
    for line in _unfold(ics_text):
        if line == "BEGIN:VEVENT":
            current = {}
            continue
        if line == "END:VEVENT":
            if current:
                yield current
            current = None
            continue
        if current is None or ":" not in line:
            continue
        key_part, _, value = line.partition(":")
        # Strip parameters like "DTSTART;TZID=America/New_York"
        key = key_part.split(";", 1)[0].lower()
        current[key] = _unescape(value)
        if key == "dtstart":
            current["dtstart_ts"] = _parse_ics_dt(value)
        elif key == "dtend":
            current["dtend_ts"] = _parse_ics_dt(value)


class CalendarConnector:
    name = "calendar"

    def is_enabled(self, cfg: Config) -> bool:
        return bool(os.environ.get("CALENDAR_ICS_URL"))

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]:
        url = os.environ["CALENDAR_ICS_URL"]
        try:
            r = requests.get(
                url, timeout=60, allow_redirects=True,
                headers={"User-Agent": "second-brain/0.0.1"},
            )
            r.raise_for_status()
        except requests.RequestException as e:
            # Don't log `e` directly - the request URL contains the secret
            # iCal feed token and `RequestException.__str__` may include it
            # on DNS/timeout failures.
            log.warning("calendar fetch failed: %s", type(e).__name__)
            return

        text = r.text
        seen_keys: set[tuple[str, str]] = set()
        for ev in _parse_events(text):
            uid = ev.get("uid", "")
            if not uid:
                continue
            # ICS uses (UID, RECURRENCE-ID) for occurrence overrides; deduping
            # on UID alone drops legitimate recurrence exceptions ("the team
            # meeting on the 5th was rescheduled to 4pm").
            # The ICS field name is RECURRENCE-ID; our generic parser
            # lowercases the key as-is, so it lands as "recurrence-id".
            key = (uid, ev.get("recurrence-id", ""))
            if key in seen_keys:
                continue
            seen_keys.add(key)

            summary = ev.get("summary", "(no title)")
            mtime = ev.get("dtstart_ts", time.time())

            lines = [f"# {summary}"]
            if "dtstart" in ev:
                lines.append(f"Start: {ev['dtstart']}")
            if "dtend" in ev:
                lines.append(f"End: {ev['dtend']}")
            if ev.get("location"):
                lines.append(f"Location: {ev['location']}")
            if ev.get("organizer"):
                lines.append(f"Organizer: {ev['organizer']}")
            description = ev.get("description")
            if description:
                lines.append("")
                lines.append(description)

            yield ConnectorDocument(
                source="calendar",
                virtual_path=f"calendar://{uid}",
                title=summary,
                content="\n".join(lines),
                mtime=mtime,
                metadata={
                    "uid": uid,
                    "dtstart": ev.get("dtstart", ""),
                    "dtend": ev.get("dtend", ""),
                },
            )
