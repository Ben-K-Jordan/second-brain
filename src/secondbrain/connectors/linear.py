"""Linear connector — pulls issues + comments + projects via the GraphQL API.

Auth: personal API key in ``LINEAR_API_KEY``. Generate at
https://linear.app/settings/account/security → "Personal API keys".

Defaults: 500 issues most-recently-updated, with their comments inlined.
Tunable via ``SB_LINEAR_MAX_ISSUES``. Each issue + each comment thread
becomes a single ConnectorDocument keyed by Linear's stable issue ID.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from datetime import datetime

import requests

from ..config import Config
from . import ConnectorDocument

log = logging.getLogger(__name__)

_API = "https://api.linear.app/graphql"
_DEFAULT_MAX_ISSUES = 500
_PAGE_SIZE = 50


def _iso_to_ts(s: str | None) -> float:
    if not s:
        return time.time()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return time.time()


_ISSUES_QUERY = """\
query Issues($first: Int!, $after: String) {
  issues(first: $first, after: $after, orderBy: updatedAt) {
    pageInfo { hasNextPage endCursor }
    nodes {
      id
      identifier
      title
      description
      url
      priority
      state { name }
      assignee { name email }
      creator { name email }
      team { key name }
      project { name }
      labels { nodes { name } }
      createdAt
      updatedAt
      comments(first: 50) {
        nodes {
          body
          user { name }
          createdAt
        }
      }
    }
  }
}
"""


class LinearConnector:
    name = "linear"

    def is_enabled(self, cfg: Config) -> bool:
        return bool(os.environ.get("LINEAR_API_KEY"))

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]:
        token = os.environ["LINEAR_API_KEY"]
        s = requests.Session()
        s.headers.update({
            "Authorization": token,
            "Content-Type": "application/json",
            "User-Agent": "second-brain/0.0.1",
        })
        cap = int(os.environ.get("SB_LINEAR_MAX_ISSUES", _DEFAULT_MAX_ISSUES))
        try:
            yield from self._iter_issues(s, cap)
        finally:
            s.close()

    # --- helpers ----------------------------------------------------------

    def _iter_issues(self, s: requests.Session, cap: int) -> Iterator[ConnectorDocument]:
        cursor: str | None = None
        emitted = 0
        while emitted < cap:
            page = min(_PAGE_SIZE, cap - emitted)
            r = s.post(
                _API,
                json={
                    "query": _ISSUES_QUERY,
                    "variables": {"first": page, "after": cursor},
                },
                timeout=60,
            )
            if r.status_code != 200:
                log.warning("Linear issues fetch failed: %s %s", r.status_code, r.text[:200])
                return
            data = r.json()
            if "errors" in data:
                log.warning("Linear GraphQL error: %s", data["errors"])
                return
            issues = (data.get("data") or {}).get("issues") or {}
            for issue in issues.get("nodes") or []:
                doc = self._render_issue(issue)
                if doc is not None:
                    yield doc
                    emitted += 1
                    if emitted >= cap:
                        return
            page_info = issues.get("pageInfo") or {}
            if not page_info.get("hasNextPage"):
                return
            cursor = page_info.get("endCursor")

    def _render_issue(self, issue: dict) -> ConnectorDocument | None:
        ident = issue.get("identifier") or issue.get("id")
        title = issue.get("title") or "(no title)"
        if not ident:
            return None

        team = (issue.get("team") or {}).get("name") or ""
        project = (issue.get("project") or {}).get("name") or ""
        state = (issue.get("state") or {}).get("name") or ""
        assignee = (issue.get("assignee") or {}).get("name") or ""
        creator = (issue.get("creator") or {}).get("name") or ""
        labels = ", ".join(
            (lab.get("name") or "")
            for lab in ((issue.get("labels") or {}).get("nodes") or [])
        )
        priority = issue.get("priority") or 0
        url = issue.get("url") or ""

        lines: list[str] = [f"# [{ident}] {title}"]
        meta_bits = []
        if team:     meta_bits.append(f"Team: {team}")
        if project:  meta_bits.append(f"Project: {project}")
        if state:    meta_bits.append(f"State: {state}")
        if assignee: meta_bits.append(f"Assignee: {assignee}")
        if creator:  meta_bits.append(f"Creator: {creator}")
        if labels:   meta_bits.append(f"Labels: {labels}")
        if priority: meta_bits.append(f"Priority: {priority}")
        if meta_bits:
            lines.extend(meta_bits)
            lines.append("")

        body = issue.get("description") or ""
        if body:
            lines.append(body)
            lines.append("")

        comments = (issue.get("comments") or {}).get("nodes") or []
        if comments:
            lines.append("## Comments")
            for c in comments:
                user = (c.get("user") or {}).get("name") or "?"
                ts = c.get("createdAt") or ""
                lines.append("")
                lines.append(f"**{user}** — {ts}")
                lines.append(c.get("body") or "")

        return ConnectorDocument(
            source="linear",
            virtual_path=f"linear://issue/{ident}",
            title=f"[{ident}] {title}",
            content="\n".join(lines),
            mtime=_iso_to_ts(issue.get("updatedAt") or issue.get("createdAt")),
            metadata={
                "identifier": ident,
                "url": url,
                "team": team, "project": project, "state": state,
                "assignee": assignee, "labels": labels,
            },
        )
