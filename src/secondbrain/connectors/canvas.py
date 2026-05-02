"""Canvas LMS connector — assignments, announcements, syllabi, grades.

Beats the "subscribe Google Calendar to Canvas iCal" route because it
gives you the full assignment description (rubrics, submission types,
attachments, link to the Canvas page) plus your current submission
status and grade — not just a date on a calendar.

Setup:
  1. In Canvas: Account → Settings → Approved Integrations → "+ New
     Access Token". Copy the token.
  2. Find your Canvas root URL (e.g. ``https://canvas.illinois.edu`` or
     ``https://<school>.instructure.com``).
  3. Set env vars::

        CANVAS_BASE_URL=https://canvas.illinois.edu
        CANVAS_TOKEN=<your token>

Optional config (in ``~/.secondbrain/config.toml``):

    canvas_window_days = 60   # how far back/forward to ingest events

What it ingests, per active course:
  - One doc per assignment (description + due date + points + status).
    The mtime is the due_at, so time-decay surfaces assignments that
    are due soon.
  - One doc per announcement (the prof's recent posts).
  - One doc for the course syllabus (when the syllabus body is set).

The chat agent can then answer "what's due tomorrow in CS 374?",
"what did the prof say about the midterm in BME 410?", or "list all
assignments worth more than 10 points due this week" using the same
``search_brain`` plumbing as everything else.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from html import unescape

import requests

from ..config import Config
from . import USER_AGENT, ConnectorDocument, respect_retry_after

log = logging.getLogger(__name__)

_TIMEOUT = 30
_PAGE_SIZE = 100
_DEFAULT_WINDOW_DAYS = 60
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(html: str) -> str:
    if not html:
        return ""
    text = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</li>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</h[1-6]>", "\n\n", text, flags=re.IGNORECASE)
    text = _TAG_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return unescape(text).strip()


def _iso_to_ts(s: str | None) -> float:
    if not s:
        return 0.0
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _format_when(ts: float) -> str:
    if not ts:
        return ""
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")


def _required_env() -> tuple[str, str] | None:
    base = (os.environ.get("CANVAS_BASE_URL") or "").strip().rstrip("/")
    token = (os.environ.get("CANVAS_TOKEN") or "").strip()
    if not base or not token:
        return None
    if not base.startswith(("http://", "https://")):
        base = "https://" + base
    return base, token


class CanvasConnector:
    name = "canvas"

    def is_enabled(self, cfg: Config) -> bool:
        return _required_env() is not None

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]:
        creds = _required_env()
        if creds is None:
            return
        base, token = creds
        window = int(getattr(cfg, "canvas_window_days", _DEFAULT_WINDOW_DAYS))

        s = requests.Session()
        s.headers.update({
            "Authorization": f"Bearer {token}",
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        })
        try:
            for course in self._list_active_courses(s, base):
                yield from self._fetch_course(s, base, course, window)
        finally:
            s.close()

    # --- helpers --------------------------------------------------------

    def _paged_get(self, s: requests.Session, url: str) -> Iterator[dict]:
        """Walk a Canvas paginated endpoint via the Link: rel="next" header.

        Canvas rate-limits at ~60 RPS per token; we honor 429 + Retry-After.
        Fewer requests = fewer headaches; we ask for the max page size.
        """
        sep = "&" if "?" in url else "?"
        next_url: str | None = f"{url}{sep}per_page={_PAGE_SIZE}"
        while next_url:
            for _ in range(3):
                try:
                    r = s.get(next_url, timeout=_TIMEOUT)
                except requests.RequestException as e:
                    log.warning("Canvas %s fetch failed: %s", next_url, type(e).__name__)
                    return
                if respect_retry_after(r):
                    continue
                break
            else:
                return
            if r.status_code == 401:
                log.warning("Canvas: token rejected (HTTP 401)")
                return
            if r.status_code == 403:
                log.warning("Canvas: access forbidden for %s (HTTP 403)", next_url)
                return
            if r.status_code != 200:
                log.warning("Canvas %s HTTP %s", next_url, r.status_code)
                return
            try:
                items = r.json() or []
            except ValueError:
                return
            if isinstance(items, dict):
                # A few endpoints return a single object. Yield it once and stop.
                yield items
                return
            yield from items
            next_url = _next_link(r.headers.get("Link", ""))

    def _list_active_courses(
        self, s: requests.Session, base: str,
    ) -> Iterator[dict]:
        """Active enrollments only — finished/archived courses are noise."""
        url = (
            f"{base}/api/v1/courses"
            "?enrollment_state=active&include[]=syllabus_body&include[]=term"
        )
        yield from self._paged_get(s, url)

    def _fetch_course(
        self, s: requests.Session, base: str, course: dict, window_days: int,
    ) -> Iterator[ConnectorDocument]:
        cid = course.get("id")
        if not cid:
            return
        course_name = (course.get("name") or course.get("course_code")
                       or f"Canvas course {cid}")
        course_code = course.get("course_code") or course_name

        # Syllabus: one doc per course when populated.
        syllabus_doc = self._render_syllabus(base, cid, course_name, course)
        if syllabus_doc is not None:
            yield syllabus_doc

        # Assignments — the headline content. Bounded by window so we don't
        # ingest the entire history of a year-long course.
        cutoff_recent = time.time() - window_days * 86400
        cutoff_future = time.time() + window_days * 86400
        for asgn in self._paged_get(
            s,
            f"{base}/api/v1/courses/{cid}/assignments?include[]=submission",
        ):
            doc = self._render_assignment(base, cid, course_name, course_code, asgn)
            if doc is None:
                continue
            # Window filter: keep anything due in [-N, +N] days of now,
            # plus anything without a due_at (project descriptions etc).
            due = _iso_to_ts(asgn.get("due_at"))
            if due and (due < cutoff_recent or due > cutoff_future):
                continue
            yield doc

        # Announcements — recent prof posts. Limit to the same window.
        # Canvas announcements live under the discussion-topics endpoint
        # with only_announcements=true.
        for ann in self._paged_get(
            s,
            f"{base}/api/v1/courses/{cid}/discussion_topics?only_announcements=true",
        ):
            doc = self._render_announcement(base, cid, course_name, course_code, ann)
            if doc is None:
                continue
            posted = _iso_to_ts(
                ann.get("posted_at") or ann.get("delayed_post_at")
                or ann.get("last_reply_at"),
            )
            if posted and posted < cutoff_recent:
                continue
            yield doc

    def _render_syllabus(
        self, base: str, cid: int, course_name: str, course: dict,
    ) -> ConnectorDocument | None:
        body = _strip_html(course.get("syllabus_body") or "")
        if not body:
            return None
        term = ((course.get("term") or {}).get("name")) or ""
        url = f"{base}/courses/{cid}"
        lines = [f"# Syllabus: {course_name}"]
        if term:
            lines.append(f"Term: {term}")
        lines.append(f"Link: {url}")
        lines.append("")
        lines.append(body)
        return ConnectorDocument(
            source="canvas",
            virtual_path=f"canvas://syllabus/{cid}",
            title=f"[{course_name}] Syllabus",
            content="\n".join(lines),
            # Use now() so syllabi don't get demoted by time-decay.
            mtime=time.time(),
            metadata={
                "kind": "syllabus",
                "course_id": cid,
                "course_name": course_name,
                "term": term,
                "url": url,
            },
        )

    def _render_assignment(
        self, base: str, cid: int, course_name: str, course_code: str,
        asgn: dict,
    ) -> ConnectorDocument | None:
        aid = asgn.get("id")
        if not aid:
            return None
        title = asgn.get("name") or "(untitled)"
        due_at = asgn.get("due_at")
        unlock_at = asgn.get("unlock_at")
        lock_at = asgn.get("lock_at")
        points = asgn.get("points_possible")
        sub_types = ", ".join(asgn.get("submission_types") or [])
        html_url = asgn.get("html_url") or f"{base}/courses/{cid}/assignments/{aid}"
        body = _strip_html(asgn.get("description") or "")

        # Submission summary (what the student has done about it).
        sub = asgn.get("submission") or {}
        sub_state = sub.get("workflow_state") or ""
        sub_score = sub.get("score")
        sub_grade = sub.get("grade")
        sub_submitted_at = sub.get("submitted_at")

        lines = [f"# {title}", "", f"Course: {course_name} ({course_code})"]
        if due_at:
            lines.append(f"Due: {_format_when(_iso_to_ts(due_at))}")
        if unlock_at:
            lines.append(f"Available from: {_format_when(_iso_to_ts(unlock_at))}")
        if lock_at:
            lines.append(f"Locks: {_format_when(_iso_to_ts(lock_at))}")
        if points is not None:
            lines.append(f"Points: {points}")
        if sub_types:
            lines.append(f"Submit via: {sub_types}")
        lines.append(f"Link: {html_url}")
        if sub_state:
            status_bits = [sub_state]
            if sub_grade is not None:
                status_bits.append(f"grade {sub_grade}")
            if sub_score is not None:
                status_bits.append(f"score {sub_score}")
            if sub_submitted_at:
                status_bits.append(f"submitted {sub_submitted_at}")
            lines.append("Status: " + " · ".join(status_bits))
        if body:
            lines.append("")
            lines.append(body)

        # mtime = due_at so time-decay weighting bubbles upcoming things up.
        # If there's no due date (e.g. project description), use now() so
        # it doesn't sink to the bottom of search results.
        mtime = _iso_to_ts(due_at) or time.time()

        return ConnectorDocument(
            source="canvas",
            virtual_path=f"canvas://assignment/{cid}/{aid}",
            title=f"[{course_code}] {title}",
            content="\n".join(lines),
            mtime=mtime,
            metadata={
                "kind": "assignment",
                "course_id": cid,
                "course_name": course_name,
                "course_code": course_code,
                "due_at": due_at or "",
                "points": points,
                "submission_state": sub_state,
                "submission_grade": sub_grade,
                "submission_score": sub_score,
                "url": html_url,
            },
        )

    def _render_announcement(
        self, base: str, cid: int, course_name: str, course_code: str,
        ann: dict,
    ) -> ConnectorDocument | None:
        ann_id = ann.get("id")
        if not ann_id:
            return None
        title = ann.get("title") or "(announcement)"
        body = _strip_html(ann.get("message") or "")
        if not body:
            return None
        author = ((ann.get("author") or {}).get("display_name")) or ""
        posted = ann.get("posted_at") or ann.get("delayed_post_at")
        url = ann.get("html_url") or f"{base}/courses/{cid}/discussion_topics/{ann_id}"

        lines = [f"# {title}", "", f"Course: {course_name} ({course_code})"]
        if author:
            lines.append(f"Author: {author}")
        if posted:
            lines.append(f"Posted: {_format_when(_iso_to_ts(posted))}")
        lines.append(f"Link: {url}")
        lines.append("")
        lines.append(body)

        return ConnectorDocument(
            source="canvas",
            virtual_path=f"canvas://announcement/{cid}/{ann_id}",
            title=f"[{course_code}] {title}",
            content="\n".join(lines),
            mtime=_iso_to_ts(posted) or time.time(),
            metadata={
                "kind": "announcement",
                "course_id": cid,
                "course_name": course_name,
                "course_code": course_code,
                "author": author,
                "posted_at": posted or "",
                "url": url,
            },
        )


# ----------------------- Link-header pagination ------------------------

_LINK_RE = re.compile(r'<([^>]+)>\s*;\s*rel\s*=\s*"([^"]+)"')


def _next_link(link_header: str) -> str | None:
    """Parse the Canvas-style Link: header and return the rel="next" URL.

    Canvas (and most REST APIs that paginate this way) ships a header like:
        Link: <https://.../page=2>; rel="next", <...>; rel="last"
    """
    if not link_header:
        return None
    for match in _LINK_RE.finditer(link_header):
        url, rel = match.group(1), match.group(2)
        if rel == "next":
            return url
    return None
