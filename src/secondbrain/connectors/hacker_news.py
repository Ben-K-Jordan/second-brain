"""Hacker News connector — your favorites + your own posts/comments.

No auth needed: HN's Firebase API is fully public. We just need your
username, since favorites and submissions are exposed per-user.

Setup:
  ``[Environment]::SetEnvironmentVariable("HN_USERNAME", "yourname", "User")``

What it ingests:
  - Stories and comments you've favorited (parsed from your HN
    ``/favorites`` page, which the API doesn't expose directly)
  - Your last ``SB_HN_MAX`` (default 500) submitted items, fetched via
    ``/v0/user/<name>.json`` → ``/v0/item/<id>.json``

Each story or comment becomes one ConnectorDocument keyed by HN's stable
numeric item id, so re-runs upsert cleanly.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Iterator
from html import unescape

import requests

from ..config import Config
from . import ConnectorDocument

log = logging.getLogger(__name__)

_API = "https://hacker-news.firebaseio.com/v0"
_FAVORITES_URL = "https://news.ycombinator.com/favorites"
_DEFAULT_MAX = 500


# `<p>` blocks in HN comment HTML get rendered as paragraph breaks; everything
# else just needs tag-stripping to be readable plain text.
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(html: str) -> str:
    if not html:
        return ""
    text = html.replace("<p>", "\n\n").replace("</p>", "")
    text = _TAG_RE.sub("", text)
    return unescape(text).strip()


class HackerNewsConnector:
    name = "hacker_news"

    def is_enabled(self, cfg: Config) -> bool:
        return bool(os.environ.get("HN_USERNAME"))

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]:
        from . import USER_AGENT

        user = os.environ["HN_USERNAME"]
        cap = int(os.environ.get("SB_HN_MAX", _DEFAULT_MAX))
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT})

        try:
            # Build the union of label sets per item id first, so a single
            # item that's both a favorite AND own submission gets `["favorite",
            # "own"]` instead of nondeterministically flipping between the
            # two on re-runs (and overwriting via upsert).
            labels: dict[int, list[str]] = {}
            for item_id in self._iter_favorites(s, user, cap):
                labels.setdefault(item_id, []).append("favorite")
            for item_id in self._iter_user_submissions(s, user, cap):
                lst = labels.setdefault(item_id, [])
                if "own" not in lst:
                    lst.append("own")

            for item_id, label_list in labels.items():
                doc = self._fetch_item(s, item_id, labels=label_list)
                if doc is not None:
                    yield doc
        finally:
            s.close()

    # --- helpers ----------------------------------------------------------

    def _iter_favorites(self, s: requests.Session, user: str, cap: int) -> Iterator[int]:
        """Scrape `/favorites?id=user` since the API doesn't expose this list.

        HN paginates with `?p=N`; each page lists item IDs in `<tr class="athing"
        id="...">` rows. We grab the IDs out of those rows.
        """
        page = 1
        emitted = 0
        # The "id" attribute on athing rows is the item id. Match defensively.
        row_re = re.compile(r'class="athing[^"]*"\s+id="(\d+)"')
        while emitted < cap:
            try:
                r = s.get(
                    _FAVORITES_URL,
                    params={"id": user, "p": page},
                    timeout=30,
                )
            except requests.RequestException as e:
                log.warning("HN favorites fetch failed: %s", e)
                return
            if r.status_code != 200:
                log.warning("HN favorites fetch returned %s", r.status_code)
                return
            ids = [int(m) for m in row_re.findall(r.text)]
            if not ids:
                return
            for iid in ids:
                yield iid
                emitted += 1
                if emitted >= cap:
                    return
            page += 1
            # HN serves the same page when you go past the end; bail if a
            # page returned fewer than the typical 30 rows.
            if len(ids) < 30:
                return

    def _iter_user_submissions(
        self, s: requests.Session, user: str, cap: int
    ) -> Iterator[int]:
        try:
            r = s.get(f"{_API}/user/{user}.json", timeout=30)
        except requests.RequestException as e:
            log.warning("HN user fetch failed: %s", e)
            return
        if r.status_code != 200:
            log.warning("HN /v0/user/%s returned %s", user, r.status_code)
            return
        data = r.json() or {}
        submitted = data.get("submitted") or []
        for iid in submitted[:cap]:
            yield iid

    def _fetch_item(
        self, s: requests.Session, item_id: int, labels: list[str]
    ) -> ConnectorDocument | None:
        try:
            r = s.get(f"{_API}/item/{item_id}.json", timeout=30)
        except requests.RequestException as e:
            log.warning("HN item %s fetch failed: %s", item_id, type(e).__name__)
            return None
        if r.status_code != 200:
            return None
        item = r.json() or {}
        if item.get("deleted") or item.get("dead"):
            return None
        kind = item.get("type") or "story"  # story | comment | poll | job
        author = item.get("by") or "?"
        ts = float(item.get("time") or 0)
        score = item.get("score") or 0
        title = item.get("title") or ""
        url = item.get("url") or ""
        text = _strip_html(item.get("text") or "")

        if kind == "comment":
            doc_title = f"HN comment by {author}"
            lines = [
                f"# {doc_title}",
                "",
                f"Author: {author}",
                f"Posted: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(ts)) if ts else '?'}",
                "",
                text,
            ]
        else:
            doc_title = title or f"HN {kind} {item_id}"
            lines = [
                f"# {doc_title}",
                "",
                f"Author: {author}",
                f"Score: {score}",
                f"Posted: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime(ts)) if ts else '?'}",
            ]
            if url:
                lines.append(f"Link: {url}")
            if text:
                lines.append("")
                lines.append(text)

        return ConnectorDocument(
            source="hacker_news",
            virtual_path=f"hn://item/{item_id}",
            title=doc_title,
            content="\n".join(lines),
            mtime=ts or time.time(),
            metadata={
                "kind": kind,
                # `labels` is the union of relationships - "favorite" and/or
                # "own" - so re-runs don't flip a single value back and forth.
                "labels": list(labels),
                "author": author,
                "score": score,
                "url": url,
                "permalink": f"https://news.ycombinator.com/item?id={item_id}",
            },
        )
