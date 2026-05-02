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


# ----------------------------- SmartRecruiters ------------------------

def _fetch_smartrecruiters(s: requests.Session, org: str) -> Iterator[ConnectorDocument]:
    """SmartRecruiters' public postings API:
    api.smartrecruiters.com/v1/companies/<company>/postings.

    Listings come back in a paginated envelope; ``offset`` + ``limit``
    walk pages. Each posting needs a second fetch for the description.
    """
    base = f"https://api.smartrecruiters.com/v1/companies/{org}/postings"
    offset = 0
    page_size = 100
    while True:
        data = _get(s, f"{base}?limit={page_size}&offset={offset}")
        if not isinstance(data, dict):
            return
        items = data.get("content") or []
        if not items:
            return
        for j in items:
            try:
                doc = _render_smartrecruiters_summary(org, j)
                if doc is not None:
                    yield doc
            except Exception as e:  # noqa: BLE001
                log.warning("smartrecruiters render failed for %s: %s", org, e)
        if len(items) < page_size:
            return
        offset += page_size


def _render_smartrecruiters_summary(org: str, j: dict) -> ConnectorDocument | None:
    """Render the summary endpoint's payload. The summary already includes
    title/location/department/jobAd which is enough for retrieval; we don't
    do the extra per-posting GET unless someone really wants the full
    description."""
    jid = j.get("id")
    if not jid:
        return None
    title = j.get("name") or "(untitled)"
    location = j.get("location") or {}
    loc_str = ", ".join(filter(None, [
        location.get("city"), location.get("region"), location.get("country"),
    ]))
    department = ((j.get("department") or {}).get("label")) or ""
    employment_type = ((j.get("typeOfEmployment") or {}).get("label")) or ""
    industry = ((j.get("industry") or {}).get("label")) or ""
    company = ((j.get("company") or {}).get("name")) or org
    apply_url = ((j.get("applyUrl")) or "")
    posting_url = ((j.get("ref")) or "")
    if not apply_url:
        # Fallback: SmartRecruiters' canonical job page.
        apply_url = f"https://jobs.smartrecruiters.com/{org}/{jid}"
    job_ad = j.get("jobAd") or {}
    sections = (job_ad.get("sections") or {})
    parts: list[str] = []
    for key in ("companyDescription", "jobDescription", "qualifications", "additionalInformation"):
        sec = sections.get(key) or {}
        text = _strip_html(sec.get("text") or "")
        if text:
            parts.append(f"## {sec.get('title', key)}\n{text}")
    desc = "\n\n".join(parts)
    updated = _iso_to_ts(j.get("releasedDate") or j.get("createdOn"))

    lines = [f"# {title}", "", f"Company: {company} (SmartRecruiters)"]
    if loc_str:        lines.append(f"Location: {loc_str}")
    if department:     lines.append(f"Department: {department}")
    if employment_type:lines.append(f"Type: {employment_type}")
    if industry:       lines.append(f"Industry: {industry}")
    if apply_url:      lines.append(f"Apply: {apply_url}")
    if desc:
        lines.append("")
        lines.append(desc)
    return ConnectorDocument(
        source="jobs",
        virtual_path=f"jobs://smartrecruiters/{org}/{jid}",
        title=f"[{company}] {title}",
        content="\n".join(lines),
        mtime=updated,
        metadata={
            "provider": "smartrecruiters",
            "company": company,
            "title": title,
            "location": loc_str,
            "department": department,
            "employment_type": employment_type,
            "url": apply_url,
            "ref_url": posting_url,
            "id": jid,
        },
    )


# ------------------------------- Recruitee ----------------------------

def _fetch_recruitee(s: requests.Session, slug: str) -> Iterator[ConnectorDocument]:
    """Recruitee's careers API: <slug>.recruitee.com/api/offers/."""
    url = f"https://{slug}.recruitee.com/api/offers/"
    data = _get(s, url)
    if not isinstance(data, dict):
        return
    for j in data.get("offers") or []:
        try:
            doc = _render_recruitee_offer(slug, j)
            if doc is not None:
                yield doc
        except Exception as e:  # noqa: BLE001
            log.warning("recruitee render failed for %s: %s", slug, e)


def _render_recruitee_offer(slug: str, j: dict) -> ConnectorDocument | None:
    jid = j.get("id")
    if not jid:
        return None
    title = j.get("title") or "(untitled)"
    location = j.get("location") or ""
    department = j.get("department") or ""
    employment_type = j.get("employment_type_code") or j.get("employment_type") or ""
    description = _strip_html(j.get("description") or "") or (j.get("description_plain") or "")
    requirements = _strip_html(j.get("requirements") or "") or (j.get("requirements_plain") or "")
    apply_url = j.get("careers_url") or j.get("careers_apply_url") or ""
    updated = _iso_to_ts(j.get("created_at") or j.get("updated_at"))

    lines = [f"# {title}", "", f"Company: {slug} (Recruitee)"]
    if location:        lines.append(f"Location: {location}")
    if department:      lines.append(f"Department: {department}")
    if employment_type: lines.append(f"Type: {employment_type}")
    if apply_url:       lines.append(f"Apply: {apply_url}")
    if description:
        lines.append("")
        lines.append(description)
    if requirements:
        lines.append("")
        lines.append("## Requirements")
        lines.append(requirements)
    return ConnectorDocument(
        source="jobs",
        virtual_path=f"jobs://recruitee/{slug}/{jid}",
        title=f"[{slug}] {title}",
        content="\n".join(lines),
        mtime=updated,
        metadata={
            "provider": "recruitee",
            "company": slug,
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
    """Pulls open postings from Greenhouse / Lever / Ashby /
    SmartRecruiters / Recruitee boards.

    Configure the company lists in ``~/.secondbrain/config.toml``::

        jobs_greenhouse      = ["anthropic", "openai", "stripe"]
        jobs_lever           = ["mistral"]
        jobs_ashby           = ["modal"]
        jobs_smartrecruiters = ["bosch", "publicis-groupe"]
        jobs_recruitee       = ["someco"]

    The connector is enabled when at least one of those lists is non-empty.

    Workday is intentionally not included: every Workday tenant lives at
    a per-company subdomain (e.g. microsoft.wd5.myworkdayjobs.com) and
    requires CSRF tokens for the search endpoint, which would mean a
    headless-browser dependency. For Workday-hosted companies, prefer
    ``--preset jobs`` web search, or scrape via the Indeed/LinkedIn
    aggregators that re-list those postings.
    """

    name = "jobs"

    def is_enabled(self, cfg: Config) -> bool:
        return bool(
            _config_companies(cfg, "greenhouse")
            or _config_companies(cfg, "lever")
            or _config_companies(cfg, "ashby")
            or _config_companies(cfg, "smartrecruiters")
            or _config_companies(cfg, "recruitee")
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
            for org in _config_companies(cfg, "smartrecruiters"):
                yield from _fetch_smartrecruiters(s, org)
            for slug in _config_companies(cfg, "recruitee"):
                yield from _fetch_recruitee(s, slug)
        finally:
            s.close()
