"""GitHub connector — pulls READMEs and recent issues/PRs from your repos.

Auth: Personal Access Token in ``GITHUB_TOKEN``. Fine-grained or classic both
work; only ``Contents: Read`` and ``Issues: Read`` / ``Pull requests: Read``
scopes are needed. Repos are paginated; for each repo we grab the README
plus the 50 most recently updated issues and PRs.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from collections.abc import Iterator
from datetime import datetime, timezone

import requests

from ..config import Config
from . import ConnectorDocument

log = logging.getLogger(__name__)

_API = "https://api.github.com"
_PER_PAGE = 100
_ISSUES_PER_REPO = 50


def _iso_to_ts(s: str | None) -> float:
    if not s:
        return time.time()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return time.time()


class GitHubConnector:
    name = "github"

    def is_enabled(self, cfg: Config) -> bool:
        return bool(os.environ.get("GITHUB_TOKEN"))

    def _session(self) -> requests.Session:
        token = os.environ["GITHUB_TOKEN"]
        s = requests.Session()
        s.headers.update({
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "second-brain/0.0.1",
        })
        return s

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]:
        s = self._session()
        try:
            for repo in self._iter_repos(s):
                yield from self._fetch_repo(s, repo)
        finally:
            s.close()

    # --- helpers ----------------------------------------------------------

    def _iter_repos(self, s: requests.Session) -> Iterator[dict]:
        page = 1
        while True:
            r = s.get(
                f"{_API}/user/repos",
                params={"per_page": _PER_PAGE, "sort": "updated", "page": page},
                timeout=30,
            )
            if r.status_code != 200:
                log.warning("GitHub /user/repos failed: %s %s", r.status_code, r.text[:200])
                return
            repos = r.json()
            if not repos:
                return
            yield from repos
            if len(repos) < _PER_PAGE:
                return
            page += 1

    def _fetch_repo(self, s: requests.Session, repo: dict) -> Iterator[ConnectorDocument]:
        full_name = repo["full_name"]
        repo_mtime = _iso_to_ts(repo.get("updated_at"))

        # README
        try:
            r = s.get(f"{_API}/repos/{full_name}/readme", timeout=30)
            if r.status_code == 200:
                data = r.json()
                content_b64 = data.get("content", "")
                # GitHub returns base64 with newlines
                content_b64 = content_b64.replace("\n", "")
                if content_b64:
                    raw = base64.b64decode(content_b64).decode("utf-8", errors="replace")
                    yield ConnectorDocument(
                        source="github",
                        virtual_path=f"github://{full_name}/README",
                        title=f"{full_name} — README",
                        content=raw,
                        mtime=repo_mtime,
                    )
        except Exception as e:
            log.warning("README fetch failed for %s: %s", full_name, e)

        # Issues + PRs (issues endpoint returns both; we partition for clarity)
        try:
            r = s.get(
                f"{_API}/repos/{full_name}/issues",
                params={"state": "all", "per_page": _ISSUES_PER_REPO, "sort": "updated"},
                timeout=30,
            )
            if r.status_code == 200:
                for item in r.json():
                    is_pr = "pull_request" in item
                    n = item["number"]
                    title = item.get("title") or f"#{n}"
                    body = item.get("body") or ""
                    state = item.get("state", "")
                    author = (item.get("user") or {}).get("login", "?")
                    created = item.get("created_at", "")
                    kind = "pr" if is_pr else "issue"
                    text = (
                        f"# {title}\n\n"
                        f"Repo: {full_name}\n"
                        f"Type: {kind}\n"
                        f"State: {state}\n"
                        f"Author: {author}\n"
                        f"Created: {created}\n\n"
                        f"{body}"
                    )
                    yield ConnectorDocument(
                        source="github",
                        virtual_path=f"github://{full_name}/{kind}/{n}",
                        title=f"[{full_name}] {kind} #{n}: {title}",
                        content=text,
                        mtime=_iso_to_ts(item.get("updated_at")),
                    )
        except Exception as e:
            log.warning("issues fetch failed for %s: %s", full_name, e)
