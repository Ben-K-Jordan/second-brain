"""Pocket connector — pulls articles you've saved to read later.

Auth: two-step OAuth that mints a permanent access token.
  1. Create a "consumer" at https://getpocket.com/developer/apps/new
     with Retrieve permission. Note the consumer key.
  2. Set ``POCKET_CONSUMER_KEY`` once, then run::
         secondbrain auth pocket
     which prints a URL to authorize, then writes the resulting access
     token back to your env.
  Or, if you already have a token, set ``POCKET_ACCESS_TOKEN`` directly.

What it ingests:
  - Every saved item from ``/v3/get`` (paginated)
  - URL, title, excerpt, time saved, time read, tags, word count

Pocket itself doesn't host the article body; the excerpt + title gets you
solid retrieval, and storing the URL means downstream tools can fetch the
full content if they want it.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator

import requests

from ..config import Config
from . import ConnectorDocument

log = logging.getLogger(__name__)

_API = "https://getpocket.com/v3/get"
_PAGE_SIZE = 200
_MAX_ITEMS = 5000


class PocketConnector:
    name = "pocket"

    def is_enabled(self, cfg: Config) -> bool:
        return bool(
            os.environ.get("POCKET_CONSUMER_KEY")
            and os.environ.get("POCKET_ACCESS_TOKEN")
        )

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]:
        consumer = os.environ["POCKET_CONSUMER_KEY"]
        token = os.environ["POCKET_ACCESS_TOKEN"]
        offset = 0
        emitted = 0
        s = requests.Session()
        s.headers.update({
            "Content-Type": "application/json; charset=UTF-8",
            "X-Accept": "application/json",
            "User-Agent": "second-brain/0.0.1",
        })
        try:
            while emitted < _MAX_ITEMS:
                payload = {
                    "consumer_key": consumer,
                    "access_token": token,
                    "state": "all",       # archived + unread, to be thorough
                    "detailType": "complete",
                    "sort": "newest",
                    "count": _PAGE_SIZE,
                    "offset": offset,
                }
                try:
                    r = s.post(_API, json=payload, timeout=30)
                except requests.RequestException as e:
                    log.warning("Pocket request failed: %s", e)
                    return
                if r.status_code != 200:
                    # Never log r.text - Pocket echoes the request payload
                    # (consumer_key + access_token) in some error responses.
                    log.warning(
                        "Pocket /v3/get failed: HTTP %s "
                        "(check POCKET_CONSUMER_KEY / POCKET_ACCESS_TOKEN)",
                        r.status_code,
                    )
                    return
                data = r.json() or {}
                # /v3/get returns {"list": {"<item_id>": {...}}}; an empty
                # response can be `[]` instead of `{}`, so be defensive.
                # Distinguish "missing list" (probably auth or backend
                # issue) from "empty list" (you have no items) so the user
                # gets a log line instead of silent zero-results.
                if "list" not in data:
                    log.warning(
                        "Pocket: unexpected payload shape (keys=%s); auth or backend issue?",
                        list(data)[:5],
                    )
                    return
                items_obj = data.get("list")
                if isinstance(items_obj, dict):
                    items = list(items_obj.values())
                else:
                    items = []
                if not items:
                    return
                for item in items:
                    doc = self._render_item(item)
                    if doc is not None:
                        yield doc
                        emitted += 1
                        if emitted >= _MAX_ITEMS:
                            return
                if len(items) < _PAGE_SIZE:
                    return
                offset += _PAGE_SIZE
        finally:
            s.close()

    # --- helpers ----------------------------------------------------------

    def _render_item(self, item: dict) -> ConnectorDocument | None:
        item_id = item.get("item_id")
        if not item_id:
            return None
        title = (
            item.get("resolved_title")
            or item.get("given_title")
            or item.get("resolved_url")
            or f"Pocket item {item_id}"
        )
        url = item.get("resolved_url") or item.get("given_url") or ""
        excerpt = item.get("excerpt") or ""
        word_count = item.get("word_count") or "?"
        try:
            time_added = float(item.get("time_added") or 0)
        except (TypeError, ValueError):
            time_added = 0.0
        try:
            time_read = float(item.get("time_read") or 0)
        except (TypeError, ValueError):
            time_read = 0.0
        status = item.get("status")  # "0" unread, "1" archived, "2" deleted
        tags = ", ".join((item.get("tags") or {}).keys())
        authors = ", ".join(
            (a.get("name") or "")
            for a in (item.get("authors") or {}).values()
        )

        lines = [f"# {title}", "", url, "", excerpt]
        meta_lines = []
        if authors: meta_lines.append(f"Author(s): {authors}")
        if tags:    meta_lines.append(f"Tags: {tags}")
        meta_lines.append(f"Word count: {word_count}")
        if time_added:
            meta_lines.append(
                f"Saved: {time.strftime('%Y-%m-%d', time.gmtime(time_added))}"
            )
        if time_read:
            meta_lines.append(
                f"Read: {time.strftime('%Y-%m-%d', time.gmtime(time_read))}"
            )
        if status == "1":
            meta_lines.append("Status: archived")
        elif status == "0":
            meta_lines.append("Status: unread")
        if meta_lines:
            lines.append("")
            lines.extend(meta_lines)

        return ConnectorDocument(
            source="pocket",
            virtual_path=f"pocket://item/{item_id}",
            title=title,
            content="\n".join(lines),
            mtime=time_added or time.time(),
            metadata={
                "url": url,
                "tags": tags,
                "status": status,
                "word_count": word_count,
                "authors": authors,
            },
        )
