"""Jobs connector — pulls live listings from companies you specifically watch.

Bypasses the usual "scrape LinkedIn / Indeed" headache by talking directly
to the public ATS APIs that most tech companies use to host their boards:

  - Greenhouse: ``boards-api.greenhouse.io/v1/boards/<token>/jobs``
  - Lever:      ``api.lever.co/v0/postings/<company>``
  - Ashby:      ``api.ashbyhq.com/posting-api/job-board/<org>``

The user lists companies in config:

    [jobs]
    greenhouse = ["anthropic", "openai", "stripe"]
    lever      = ["mistral", "github"]
    ashby      = ["modal", "rippling"]

Each open posting becomes one ConnectorDocument keyed by
``jobs://<provider>/<company>/<posting-id>``. Re-running sync upserts based
on the ATS's ``updated_at``; postings that disappear from the board are
NOT auto-removed (the ATS API doesn't expose a "list of removed" endpoint;
they just stop appearing). A future ``secondbrain prune jobs`` could
reconcile if needed.

Each provider's API is unauthenticated and polite, but we still respect
429s via the shared ``respect_retry_after`` helper.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterator
from datetime import datetime
from html import unescape

import requests

from ..config import Config
from . import USER_AGENT, ConnectorDocument, respect_retry_after

log = logging.getLogger(__name__)

_TIMEOUT = 30
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
        return time.time()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return time.time()


def _get(s: requests.Session, url: str) -> dict | list | None:
    """GET with one retry on 429. Returns the parsed JSON body, or None on
    any other failure."""
    for _ in range(3):
        try:
            r = s.get(url, timeout=_TIMEOUT)
        except requests.RequestException as e:
            log.warning("jobs fetch %s failed: %s", url, type(e).__name__)
            return None
        if respect_retry_after(r):
            continue
        if r.status_code == 404:
            log.warning("jobs board not found: %s", url)
            return None
        if r.status_code != 200:
            log.warning("jobs fetch %s HTTP %s", url, r.status_code)
            return None
        try:
            return r.json()
        except ValueError:
            log.warning("jobs fetch %s: non-JSON response", url)
            return None
    return None


def _config_companies(cfg: Config, provider: str) -> list[str]:
    """Read the configured company list for one provider. Tolerant of
    missing config (returns empty list)."""
    raw = getattr(cfg, f"jobs_{provider}", None) or ()
    out: list[str] = []
    for c in raw:
        c = (c or "").strip().lower()
        if c and c not in out:
            out.append(c)
    return out


# ----------------------------- Greenhouse -----------------------------

def _fetch_greenhouse(s: requests.Session, token: str) -> Iterator[ConnectorDocument]:
    """Greenhouse boards live at boards-api.greenhouse.io/v1/boards/<token>.

    The ``content=true`` flag tells the API to include the HTML description.
    """
    url = f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true"
    data = _get(s, url)
    if not isinstance(data, dict):
        return
    for j in data.get("jobs") or []:
        try:
            yield _render_greenhouse_job(token, j)
        except Exception as e:  # noqa: BLE001
            log.warning("greenhouse render failed for %s: %s", token, e)


def _render_greenhouse_job(company: str, j: dict) -> ConnectorDocument | None:
    jid = j.get("id")
    if not jid:
        return None
    title = j.get("title") or "(untitled)"
    html_body = j.get("content") or ""
    body = _strip_html(html_body)
    location = ((j.get("location") or {}).get("name")) or ""
    departments = ", ".join(d.get("name", "") for d in j.get("departments") or [])
    offices = ", ".join(o.get("name", "") for o in j.get("offices") or [])
    abs_url = j.get("absolute_url") or ""
    updated = _iso_to_ts(j.get("updated_at"))

    lines = [f"# {title}", "", f"Company: {company} (Greenhouse)"]
    if location:    lines.append(f"Location: {location}")
    if offices:     lines.append(f"Offices: {offices}")
    if departments: lines.append(f"Departments: {departments}")
    if abs_url:     lines.append(f"Apply: {abs_url}")
    if body:
        lines.append("")
        lines.append(body)
    return ConnectorDocument(
        source="jobs",
        virtual_path=f"jobs://greenhouse/{company}/{jid}",
        title=f"[{company}] {title}",
        content="\n".join(lines),
        mtime=updated,
        metadata={
            "provider": "greenhouse",
            "company": company,
            "title": title,
            "location": location,
            "url": abs_url,
            "departments": departments,
            "id": jid,
        },
    )


# -------------------------------- Lever -------------------------------

def _fetch_lever(s: requests.Session, company: str) -> Iterator[ConnectorDocument]:
    """Lever's public postings API: api.lever.co/v0/postings/<company>?mode=json."""
    url = f"https://api.lever.co/v0/postings/{company}?mode=json"
    data = _get(s, url)
    if not isinstance(data, list):
        return
    for j in data:
        try:
            yield _render_lever_job(company, j)
        except Exception as e:  # noqa: BLE001
            log.warning("lever render failed for %s: %s", company, e)


def _render_lever_job(company: str, j: dict) -> ConnectorDocument | None:
    jid = j.get("id")
    if not jid:
        return None
    title = j.get("text") or "(untitled)"
    desc = j.get("descriptionPlain") or _strip_html(j.get("description") or "")
    lists = j.get("lists") or []
    bullets = "\n".join(
        f"## {bl.get('text', '')}\n{_strip_html(bl.get('content', ''))}"
        for bl in lists
    )
    location = ((j.get("categories") or {}).get("location")) or ""
    team = ((j.get("categories") or {}).get("team")) or ""
    commitment = ((j.get("categories") or {}).get("commitment")) or ""
    apply_url = j.get("applyUrl") or j.get("hostedUrl") or ""
    # Lever's createdAt is epoch milliseconds.
    try:
        created = float(j.get("createdAt") or 0) / 1000.0
    except (TypeError, ValueError):
        created = 0.0

    lines = [f"# {title}", "", f"Company: {company} (Lever)"]
    if location:    lines.append(f"Location: {location}")
    if team:        lines.append(f"Team: {team}")
    if commitment:  lines.append(f"Commitment: {commitment}")
    if apply_url:   lines.append(f"Apply: {apply_url}")
    if desc:
        lines.append("")
        lines.append(desc)
    if bullets:
        lines.append("")
        lines.append(bullets)
    return ConnectorDocument(
        source="jobs",
        virtual_path=f"jobs://lever/{company}/{jid}",
        title=f"[{company}] {title}",
        content="\n".join(lines),
        mtime=created or time.time(),
        metadata={
            "provider": "lever",
            "company": company,
            "title": title,
            "location": location,
            "team": team,
            "url": apply_url,
            "id": jid,
        },
    )


# -------------------------------- Ashby -------------------------------

def _fetch_ashby(s: requests.Session, org: str) -> Iterator[ConnectorDocument]:
    """Ashby's public job-board API. ``includeCompensation=true`` is harmless
    even when the company hasn't elected to publish ranges."""
    url = (
        f"https://api.ashbyhq.com/posting-api/job-board/{org}"
        "?includeCompensation=true"
    )
    data = _get(s, url)
    if not isinstance(data, dict):
        return
    for j in data.get("jobs") or []:
        try:
            yield _render_ashby_job(org, j)
        except Exception as e:  # noqa: BLE001
            log.warning("ashby render failed for %s: %s", org, e)


def _render_ashby_job(org: str, j: dict) -> ConnectorDocument | None:
    jid = j.get("id") or j.get("jobId")
    if not jid:
        return None
    title = j.get("title") or "(untitled)"
    desc = _strip_html(j.get("descriptionHtml") or "") or (j.get("descriptionPlain") or "")
    location = j.get("location") or ""
    department = j.get("department") or ""
    employment_type = j.get("employmentType") or ""
    apply_url = j.get("jobUrl") or j.get("applyUrl") or ""
    updated = _iso_to_ts(j.get("publishedAt") or j.get("updatedAt"))
    comp = j.get("compensation") or {}
    comp_summary = comp.get("compensationTierSummary") or ""

    lines = [f"# {title}", "", f"Company: {org} (Ashby)"]
    if location:        lines.append(f"Location: {location}")
    if department:      lines.append(f"Department: {department}")
    if employment_type: lines.append(f"Type: {employment_type}")
    if comp_summary:    lines.append(f"Compensation: {comp_summary}")
    if apply_url:       lines.append(f"Apply: {apply_url}")
    if desc:
        lines.append("")
        lines.append(desc)
    return ConnectorDocument(
        source="jobs",
        virtual_path=f"jobs://ashby/{org}/{jid}",
        title=f"[{org}] {title}",
        content="\n".join(lines),
        mtime=updated,
        metadata={
            "provider": "ashby",
            "company": org,
            "title": title,
            "location": location,
            "department": department,
            "employment_type": employment_type,
            "url": apply_url,
            "id": jid,
        },
    )


# ---------------------------- Connector class --------------------------

class JobsConnector:
    """Pulls open postings from Greenhouse / Lever / Ashby boards.

    Configure the company lists in ``~/.secondbrain/config.toml``::

        jobs_greenhouse = ["anthropic", "openai", "stripe"]
        jobs_lever      = ["mistral"]
        jobs_ashby      = ["modal"]

    The connector is enabled when at least one of those lists is non-empty.
    """

    name = "jobs"

    def is_enabled(self, cfg: Config) -> bool:
        return bool(
            _config_companies(cfg, "greenhouse")
            or _config_companies(cfg, "lever")
            or _config_companies(cfg, "ashby")
        )

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]:
        s = requests.Session()
        s.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        })
        try:
            for token in _config_companies(cfg, "greenhouse"):
                yield from _fetch_greenhouse(s, token)
            for company in _config_companies(cfg, "lever"):
                yield from _fetch_lever(s, company)
            for org in _config_companies(cfg, "ashby"):
                yield from _fetch_ashby(s, org)
        finally:
            s.close()
