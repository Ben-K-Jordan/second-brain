"""Bluesky connector — pulls your posts via the public AT Protocol API.

No auth required for public profiles: the AppView at ``api.bsky.app`` exposes
``app.bsky.feed.getAuthorFeed`` which returns your posts/replies/reposts.
For private accounts you'd need an authenticated PDS session, which we skip
for v1 since most personal accounts on Bluesky are public.

Setup:
  - ``BLUESKY_HANDLE`` env var: your handle, e.g. ``alice.bsky.social`` or
    ``alice.example.com``.

What it ingests:
  - Posts you've authored (latest ``SB_BLUESKY_MAX``, default 500).
  - Reply-text and the parent's text (when reachable in the feed payload)
    so threaded context comes along.

Each post becomes one ConnectorDocument keyed by the post's stable AT-URI.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from datetime import datetime

import requests

from ..config import Config
from . import USER_AGENT, ConnectorDocument, respect_retry_after

log = logging.getLogger(__name__)

_API = "https://public.api.bsky.app/xrpc"
_TIMEOUT = 30
_DEFAULT_MAX = 500


def _iso_to_ts(s: str | None) -> float:
    if not s:
        return time.time()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return time.time()


class BlueskyConnector:
    name = "bluesky"

    def is_enabled(self, cfg: Config) -> bool:
        return bool(os.environ.get("BLUESKY_HANDLE"))

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]:
        handle = os.environ["BLUESKY_HANDLE"]
        cap = int(os.environ.get("SB_BLUESKY_MAX", _DEFAULT_MAX))
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            yield from self._iter_author_feed(s, handle, cap)
        finally:
            s.close()

    # --- helpers --------------------------------------------------------

    def _iter_author_feed(
        self, s: requests.Session, handle: str, cap: int
    ) -> Iterator[ConnectorDocument]:
        cursor: str | None = None
        emitted = 0
        # Bluesky paginates 100 per page; cap protects us if the user has
        # tens of thousands of posts.
        while emitted < cap:
            params: dict[str, str | int] = {
                "actor": handle,
                "limit": min(100, cap - emitted),
                "filter": "posts_with_replies",
            }
            if cursor:
                params["cursor"] = cursor
            for _ in range(3):
                try:
                    r = s.get(
                        f"{_API}/app.bsky.feed.getAuthorFeed",
                        params=params, timeout=_TIMEOUT,
                    )
                except requests.RequestException as e:
                    log.warning("Bluesky feed fetch failed: %s", type(e).__name__)
                    return
                if respect_retry_after(r):
                    continue
                break
            else:
                return
            if r.status_code == 400:
                # Most often: handle didn't resolve. Return cleanly.
                log.warning("Bluesky: handle %r could not be resolved", handle)
                return
            if r.status_code != 200:
                log.warning("Bluesky getAuthorFeed HTTP %s", r.status_code)
                return
            data = r.json() or {}
            for fp in data.get("feed") or []:
                doc = self._render_feed_item(fp)
                if doc is not None:
                    yield doc
                    emitted += 1
                    if emitted >= cap:
                        return
            cursor = data.get("cursor")
            if not cursor:
                return

    def _render_feed_item(self, fp: dict) -> ConnectorDocument | None:
        post = fp.get("post") or {}
        record = post.get("record") or {}
        uri = post.get("uri") or ""
        if not uri:
            return None
        author = (post.get("author") or {})
        handle = author.get("handle") or "?"
        display = author.get("displayName") or handle
        text = (record.get("text") or "").strip()
        if not text:
            return None
        created = _iso_to_ts(record.get("createdAt"))
        like_count = post.get("likeCount", 0)
        repost_count = post.get("repostCount", 0)
        reply_count = post.get("replyCount", 0)

        # Reply context: the parent post's text if Bluesky shipped it.
        reply_ctx = ""
        reply = fp.get("reply") or {}
        parent = reply.get("parent") or {}
        if parent:
            parent_record = parent.get("record") or {}
            parent_text = (parent_record.get("text") or "").strip()
            parent_handle = ((parent.get("author") or {}).get("handle")) or "?"
            if parent_text:
                reply_ctx = (
                    f"\n\n> **Replying to @{parent_handle}**:\n> "
                    + parent_text.replace("\n", "\n> ")
                )

        # https URL Bluesky uses to render the post; nice for "open in browser".
        # at:// URIs aren't clickable; convert to /profile/<handle>/post/<rkey>.
        rkey = uri.rsplit("/", 1)[-1]
        web_url = f"https://bsky.app/profile/{handle}/post/{rkey}"

        lines = [
            f"# Post by @{handle}" + (f" ({display})" if display != handle else ""),
            "",
            text + reply_ctx,
            "",
            f"Likes: {like_count}  Reposts: {repost_count}  Replies: {reply_count}",
            f"Posted: {datetime.fromtimestamp(created).isoformat() if created else '?'}",
            f"Link: {web_url}",
        ]
        return ConnectorDocument(
            source="bluesky",
            virtual_path=f"bsky://post/{handle}/{rkey}",
            title=text[:80] + ("…" if len(text) > 80 else ""),
            content="\n".join(lines),
            mtime=created,
            metadata={
                "handle": handle,
                "display_name": display,
                "uri": uri,
                "url": web_url,
                "likes": like_count,
                "reposts": repost_count,
                "replies": reply_count,
                "is_reply": bool(reply_ctx),
            },
        )
