"""Canvas LMS connector — assignments, announcements, syllabi, grades.

Beats the "subscribe Google Calendar to Canvas iCal" route because it
gives you the full assignment description (rubrics, submission types,
attachments, link to the Canvas page) plus your current submission
status and grade — not just a date on a calendar.

How auth actually works (re: SSO + Duo)
---------------------------------------
The Canvas web UI requires your school's SSO + Duo each time. The API
does NOT. You generate a *personal access token* once (which itself
requires one SSO+Duo login, since the token-generation page lives in
Canvas), and from then on the connector authenticates with just
``Authorization: Bearer <token>``. No SSO. No Duo. No browser dance.

Easiest setup (recommended):

    secondbrain auth canvas

That command opens the token-generation page, walks you through it,
verifies the token works, and saves it to
``~/.secondbrain/canvas_credentials.json``.

Manual setup (if you prefer env vars):
  1. In Canvas: Account → Settings → Approved Integrations
     → "+ New Access Token". Leave "Expires" blank for a non-expiring
     token. Copy the value (it's only shown once).
  2. Set env vars::

        CANVAS_BASE_URL=https://canvas.<school>.edu  (or *.instructure.com)
        CANVAS_TOKEN=<your token>

If your school disables personal access tokens (rare; Cornell + most
US universities allow them), fall back to the iCal route: in Canvas
go to Calendar → "Calendar Feed", copy the URL, and set
``CALENDAR_ICS_URL=<that url>`` so the existing ICS calendar
connector picks up assignment due dates as plain calendar events
(without rubrics / grades / submissions).

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


# --- Credential plumbing ---------------------------------------------

import json as _json  # noqa: E402  (kept local-scope feel; module is small)
from pathlib import Path as _Path  # noqa: E402


def _credentials_path(cfg: Config) -> _Path:
    """Where ``secondbrain auth canvas`` writes the saved token."""
    return cfg.data_dir / "canvas_credentials.json"


def save_canvas_credentials(cfg: Config, base_url: str, token: str) -> None:
    """Persist a Canvas base URL + token to a 0600 file in the data dir.

    Mirrors the Google OAuth scaffold's storage style. The token is the
    only secret in the file; we don't include anything PII-heavy.
    """
    base_url = base_url.strip().rstrip("/")
    if base_url and not base_url.startswith(("http://", "https://")):
        base_url = "https://" + base_url
    path = _credentials_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        _json.dump({"base_url": base_url, "token": token}, f)
    try:
        path.chmod(0o600)
    except OSError:
        # Windows ignores the bits anyway; fine.
        pass


def load_canvas_credentials(cfg: Config) -> tuple[str, str] | None:
    """Read the saved (base_url, token) tuple, or None if missing/corrupt."""
    path = _credentials_path(cfg)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = _json.load(f)
    except (OSError, _json.JSONDecodeError):
        return None
    base = (data.get("base_url") or "").strip().rstrip("/")
    token = (data.get("token") or "").strip()
    if not base or not token:
        return None
    return base, token


def verify_canvas_token(base_url: str, token: str) -> dict | None:
    """Probe ``/api/v1/users/self`` to confirm the token works.

    Returns the user JSON on success, None on any failure (bad token,
    base URL typo, network error). Cheap — one quick GET.
    """
    base_url = base_url.strip().rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        base_url = "https://" + base_url
    try:
        r = requests.get(
            f"{base_url}/api/v1/users/self",
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            },
            timeout=15,
        )
    except requests.RequestException as e:
        log.warning("Canvas token verify failed: %s", type(e).__name__)
        return None
    if r.status_code != 200:
        log.warning("Canvas token verify HTTP %s", r.status_code)
        return None
    try:
        return r.json()
    except ValueError:
        return None


def _required_env(cfg: Config | None = None) -> tuple[str, str] | None:
    """Resolve the (base_url, token) pair from env vars or saved creds.

    Env vars win over the credentials file, so you can override on the
    command line for one-off testing. ``cfg`` is required to read the
    saved-credentials file; passed as None means "env only".
    """
    base = (os.environ.get("CANVAS_BASE_URL") or "").strip().rstrip("/")
    token = (os.environ.get("CANVAS_TOKEN") or "").strip()
    if base and token:
        if not base.startswith(("http://", "https://")):
            base = "https://" + base
        return base, token
    if cfg is not None:
        saved = load_canvas_credentials(cfg)
        if saved is not None:
            return saved
    return None


class CanvasConnector:
    name = "canvas"

    def is_enabled(self, cfg: Config) -> bool:
        return _required_env(cfg) is not None

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]:
        creds = _required_env(cfg)
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
