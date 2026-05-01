"""Reddit connector — pulls your saved posts/comments and your own activity.

Uses Reddit's "script app" OAuth2 flow, which is the right pattern for a
single-user personal tool — no redirect server needed.

Setup:
  1. Go to https://www.reddit.com/prefs/apps -> Create another app...
  2. Pick "script", give any name, redirect URI = http://localhost.
  3. Note the client ID (under the app name) and the secret.
  4. Set env vars:
        REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET,
        REDDIT_USERNAME,  REDDIT_PASSWORD
     (yes, password — script-app grant requires it; only used to mint a
     token against your *own* account, never sent anywhere else)

What it ingests:
  - Posts and comments you've saved (the "saved" button)
  - Your own submissions and comments (the last 100 of each)
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

_AUTH_URL = "https://www.reddit.com/api/v1/access_token"
_API = "https://oauth.reddit.com"
# Reddit asks for a descriptive UA per https://github.com/reddit-archive/reddit/wiki/API
_USER_AGENT = "second-brain (personal indexer; https://github.com/Ben-K-Jordan/second-brain)"
_DEFAULT_LIMIT = 500


def _required_env() -> tuple[str, str, str, str] | None:
    cid = os.environ.get("REDDIT_CLIENT_ID")
    secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user = os.environ.get("REDDIT_USERNAME")
    password = os.environ.get("REDDIT_PASSWORD")
    if not all([cid, secret, user, password]):
        return None
    return cid, secret, user, password  # type: ignore[return-value]


def _ts(value: float | int | None) -> float:
    return float(value) if value is not None else time.time()


def _format_when(epoch: float) -> str:
    if not epoch:
        return ""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


class RedditConnector:
    name = "reddit"

    def is_enabled(self, cfg: Config) -> bool:
        return _required_env() is not None

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]:
        creds = _required_env()
        if creds is None:
            return
        cid, secret, user, password = creds

        # Mint an access token via the password grant. Reddit only allows this
        # for "script" apps where the developer == the end user, which is
        # exactly our case for a personal indexer.
        token_resp = requests.post(
            _AUTH_URL,
            auth=(cid, secret),
            data={"grant_type": "password", "username": user, "password": password},
            headers={"User-Agent": _USER_AGENT},
            timeout=30,
        )
        if token_resp.status_code != 200:
            # Never log token_resp.text - the password-grant endpoint can echo
            # form fields (including the password) into error responses.
            log.warning(
                "Reddit token fetch failed: HTTP %s "
                "(check REDDIT_CLIENT_ID/SECRET/USERNAME/PASSWORD)",
                token_resp.status_code,
            )
            return
        token = (token_resp.json() or {}).get("access_token")
        if not token:
            log.warning("Reddit returned no access_token")
            return

        s = requests.Session()
        s.headers.update({
            "Authorization": f"bearer {token}",
            "User-Agent": _USER_AGENT,
        })

        cap = int(os.environ.get("SB_REDDIT_MAX", _DEFAULT_LIMIT))
        try:
            yield from self._iter_listing(s, f"/user/{user}/saved", source_label="saved", limit=cap)
            yield from self._iter_listing(s, f"/user/{user}/submitted", source_label="own_post", limit=cap)
            yield from self._iter_listing(s, f"/user/{user}/comments", source_label="own_comment", limit=cap)
        finally:
            s.close()

    # --- helpers --------------------------------------------------------

    def _iter_listing(
        self, s: requests.Session, path: str, source_label: str, limit: int = _DEFAULT_LIMIT
    ) -> Iterator[ConnectorDocument]:
        after: str | None = None
        emitted = 0
        while emitted < limit:
            params: dict[str, str | int] = {"limit": min(100, limit - emitted), "raw_json": 1}
            if after:
                params["after"] = after
            r = s.get(f"{_API}{path}", params=params, timeout=30)
            if r.status_code == 429:
                from . import respect_retry_after
                if respect_retry_after(r):
                    continue
            if r.status_code != 200:
                log.warning("Reddit %s failed: HTTP %s", path, r.status_code)
                return
            data = (r.json() or {}).get("data") or {}
            children = data.get("children") or []
            if not children:
                return
            for c in children:
                doc = self._render_thing(c, source_label)
                if doc is not None:
                    yield doc
                    emitted += 1
                    if emitted >= limit:
                        return
            after = data.get("after")
            if not after:
                return

    def _render_thing(self, child: dict, source_label: str) -> ConnectorDocument | None:
        kind = child.get("kind")  # "t1" = comment, "t3" = post
        d = child.get("data") or {}
        if kind == "t3":
            return self._render_post(d, source_label)
        if kind == "t1":
            return self._render_comment(d, source_label)
        return None

    def _render_post(self, d: dict, source_label: str) -> ConnectorDocument | None:
        post_id = d.get("id")
        if not post_id:
            return None
        title = d.get("title") or "(no title)"
        sub = d.get("subreddit") or ""
        author = d.get("author") or ""
        when = _format_when(_ts(d.get("created_utc")))
        score = d.get("score", 0)
        permalink = d.get("permalink") or ""
        url = d.get("url") or ""
        body = d.get("selftext") or ""

        lines = [
            f"# {title}",
            "",
            f"Subreddit: r/{sub}",
            f"Author: u/{author}",
            f"Posted: {when}",
            f"Score: {score}",
        ]
        if url and url != permalink:
            lines.append(f"Link: {url}")
        if body:
            lines.append("")
            lines.append(body)
        return ConnectorDocument(
            source="reddit",
            virtual_path=f"reddit://post/{post_id}",
            title=title,
            content="\n".join(lines),
            mtime=_ts(d.get("created_utc")),
            metadata={
                "subreddit": sub, "author": author,
                "score": score, "kind": source_label,
                "permalink": f"https://reddit.com{permalink}" if permalink else "",
            },
        )

    def _render_comment(self, d: dict, source_label: str) -> ConnectorDocument | None:
        cid = d.get("id")
        if not cid:
            return None
        sub = d.get("subreddit") or ""
        author = d.get("author") or ""
        when = _format_when(_ts(d.get("created_utc")))
        score = d.get("score", 0)
        body = d.get("body") or ""
        link_title = d.get("link_title") or ""
        permalink = d.get("permalink") or ""
        if not body.strip():
            return None
        title = link_title or f"Comment in r/{sub}"
        lines = [
            f"# Comment: {title}",
            "",
            f"Subreddit: r/{sub}",
            f"Author: u/{author}",
            f"Posted: {when}",
            f"Score: {score}",
            "",
            body,
        ]
        return ConnectorDocument(
            source="reddit",
            virtual_path=f"reddit://comment/{cid}",
            title=title,
            content="\n".join(lines),
            mtime=_ts(d.get("created_utc")),
            metadata={
                "subreddit": sub, "author": author,
                "score": score, "kind": source_label,
                "permalink": f"https://reddit.com{permalink}" if permalink else "",
            },
        )
