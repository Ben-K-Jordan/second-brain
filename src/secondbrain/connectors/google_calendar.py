"""Google Calendar connector — pulls events via the Calendar API.

Uses the shared Google OAuth scaffold. Compared to the ICS connector, this
one gets:
  - Event metadata (attendees, organizer, location, conference link)
  - All calendars the user has access to (not just one feed)
  - Proper recurring event expansion via the API's singleEvents=true

Defaults: ±90 days from now (recent past + near future). Adjust via
``SB_CALENDAR_RANGE_DAYS``.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone

import requests

from ..config import Config
from . import ConnectorDocument
from ._google_oauth import authorized_session, is_authorized

log = logging.getLogger(__name__)

GOOGLE_CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

_API = "https://www.googleapis.com/calendar/v3"
_DEFAULT_RANGE_DAYS = 90


def _parse_event_time(value: dict | None) -> tuple[float, str]:
    """Return (timestamp, display_string) for an event start/end. Handles
    all-day (date) and timed (dateTime) variants."""
    if not value:
        return time.time(), ""
    if "dateTime" in value:
        try:
            dt = datetime.fromisoformat(value["dateTime"].replace("Z", "+00:00"))
            return dt.timestamp(), dt.strftime("%Y-%m-%d %H:%M %Z")
        except ValueError:
            return time.time(), value["dateTime"]
    if "date" in value:
        try:
            dt = datetime.fromisoformat(value["date"]).replace(tzinfo=timezone.utc)
            return dt.timestamp(), value["date"] + " (all day)"
        except ValueError:
            return time.time(), value["date"]
    return time.time(), ""


class GoogleCalendarConnector:
    name = "google_calendar"

    def is_enabled(self, cfg: Config) -> bool:
        return is_authorized(cfg, GOOGLE_CALENDAR_SCOPES)

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]:
        s = authorized_session(cfg, GOOGLE_CALENDAR_SCOPES)
        if s is None:
            log.warning("Google Calendar: no credentials. Run `secondbrain auth google`.")
            return

        days = int(os.environ.get("SB_CALENDAR_RANGE_DAYS", _DEFAULT_RANGE_DAYS))
        now = datetime.now(timezone.utc)
        time_min = (now - timedelta(days=days)).isoformat()
        time_max = (now + timedelta(days=days)).isoformat()

        try:
            for cal in self._list_calendars(s):
                yield from self._iter_events(s, cal, time_min, time_max)
        finally:
            s.close()

    # --- helpers --------------------------------------------------------

    def _list_calendars(self, s: requests.Session) -> Iterator[dict]:
        r = s.get(f"{_API}/users/me/calendarList", timeout=30)
        if r.status_code != 200:
            log.warning("Calendar list failed: %s %s", r.status_code, r.text[:200])
            return
        for cal in r.json().get("items") or []:
            # Skip calendars the user has explicitly hidden
            if cal.get("hidden"):
                continue
            yield cal

    def _iter_events(
        self,
        s: requests.Session,
        cal: dict,
        time_min: str,
        time_max: str,
    ) -> Iterator[ConnectorDocument]:
        cal_id = cal["id"]
        cal_summary = cal.get("summary", cal_id)
        page_token: str | None = None
        while True:
            params: dict[str, str | bool] = {
                "timeMin": time_min,
                "timeMax": time_max,
                "singleEvents": "true",
                "orderBy": "startTime",
                "maxResults": 250,
            }
            if page_token:
                params["pageToken"] = page_token
            r = s.get(
                f"{_API}/calendars/{requests.utils.quote(cal_id, safe='')}/events",
                params=params, timeout=30,
            )
            if r.status_code != 200:
                log.warning(
                    "Calendar events fetch failed for %s: %s %s",
                    cal_id, r.status_code, r.text[:200],
                )
                return
            data = r.json()
            for ev in data.get("items") or []:
                doc = self._render_event(cal_summary, ev)
                if doc is not None:
                    yield doc
            page_token = data.get("nextPageToken")
            if not page_token:
                return

    def _render_event(self, cal_name: str, ev: dict) -> ConnectorDocument | None:
        ev_id = ev.get("id")
        if not ev_id or ev.get("status") == "cancelled":
            return None

        summary = ev.get("summary") or "(no title)"
        start_ts, start_str = _parse_event_time(ev.get("start"))
        _, end_str = _parse_event_time(ev.get("end"))
        location = ev.get("location") or ""
        organizer = (ev.get("organizer") or {}).get("email") or ""
        creator = (ev.get("creator") or {}).get("email") or ""
        attendees_list = ev.get("attendees") or []
        attendees = ", ".join(
            (a.get("email") or a.get("displayName") or "") for a in attendees_list
        )
        description = ev.get("description") or ""
        meet_link = (ev.get("hangoutLink") or "").strip()
        html_link = ev.get("htmlLink") or ""

        lines = [f"# {summary}", "", f"Calendar: {cal_name}"]
        if start_str:
            lines.append(f"Start: {start_str}")
        if end_str:
            lines.append(f"End: {end_str}")
        if location:
            lines.append(f"Location: {location}")
        if organizer:
            lines.append(f"Organizer: {organizer}")
        elif creator:
            lines.append(f"Creator: {creator}")
        if attendees:
            lines.append(f"Attendees: {attendees}")
        if meet_link:
            lines.append(f"Meet link: {meet_link}")
        if description:
            lines.append("")
            lines.append(description)

        return ConnectorDocument(
            source="google_calendar",
            virtual_path=f"google_calendar://{ev_id}",
            title=summary,
            content="\n".join(lines),
            mtime=start_ts,
            metadata={
                "calendar": cal_name,
                "start": start_str,
                "end": end_str,
                "location": location,
                "organizer": organizer,
                "html_link": html_link,
                "attendees": [
                    a.get("email") for a in attendees_list if a.get("email")
                ],
            },
        )
