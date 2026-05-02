"""Phase 78: Readwise highlights connector.

Readwise aggregates highlights across Kindle, Apple Books, web
clippers, podcasts, etc. — wherever you "save" something, it lands
there. That's a high-signal "I cared about this" marker that
otherwise lives outside the brain.

This connector pulls every highlight via Readwise's V2 export API,
groups them by source book/article, and emits one
``ConnectorDocument`` per source so the brain has a navigable view
of "what I've highlighted from <book>" rather than 200 separate
chunk-sized highlights.

Setup:
    READWISE_TOKEN=<from https://readwise.io/access_token>
    cfg.readwise_window_days = 365  # how far back to pull
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import requests

from ..config import Config
from . import USER_AGENT, ConnectorDocument

log = logging.getLogger(__name__)


_API_BASE = "https://readwise.io/api/v2"
_TIMEOUT = 30
_DEFAULT_WINDOW_DAYS = 365


def _resolve_token() -> str | None:
    return (os.environ.get("READWISE_TOKEN") or "").strip() or None


class ReadwiseConnector:
    name = "readwise"

    def is_enabled(self, cfg: Config) -> bool:
        return _resolve_token() is not None

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]:
        token = _resolve_token()
        if not token:
            return
        window_days = int(
            getattr(cfg, "readwise_window_days", _DEFAULT_WINDOW_DAYS),
        )
        updated_after = (
            datetime.now(tz=UTC) - timedelta(days=window_days)
        ).isoformat()
        s = requests.Session()
        s.headers.update({
            "Authorization": f"Token {token}",
            "User-Agent": USER_AGENT,
        })
        try:
            books_by_id = self._fetch_books(s)
            highlights_by_book = self._fetch_highlights(
                s, updated_after, books_by_id,
            )
        except requests.RequestException as e:
            log.warning("readwise: fetch failed: %s", type(e).__name__)
            s.close()
            return
        finally:
            s.close()

        for book_id, hls in highlights_by_book.items():
            book = books_by_id.get(book_id) or {}
            doc = self._render(book_id, book, hls)
            if doc is not None:
                yield doc

    def _fetch_books(self, s: requests.Session) -> dict[int, dict]:
        """Return {book_id: book_dict} across paginated /books."""
        out: dict[int, dict] = {}
        url = f"{_API_BASE}/books/?page_size=1000"
        while url:
            try:
                r = s.get(url, timeout=_TIMEOUT)
            except requests.RequestException:
                return out
            if r.status_code != 200:
                log.warning("readwise: books HTTP %s", r.status_code)
                return out
            try:
                payload = r.json()
            except ValueError:
                return out
            for b in payload.get("results") or []:
                bid = b.get("id")
                if bid is not None:
                    out[int(bid)] = b
            url = payload.get("next")
        return out

    def _fetch_highlights(
        self, s: requests.Session, updated_after: str,
        books_by_id: dict[int, dict],
    ) -> dict[int, list[dict]]:
        """Return {book_id: [highlight, ...]} for highlights updated
        after the cutoff. Skips highlights with no parent book."""
        out: dict[int, list[dict]] = defaultdict(list)
        url = (
            f"{_API_BASE}/highlights/?page_size=1000"
            f"&updated__gt={updated_after}"
        )
        while url:
            try:
                r = s.get(url, timeout=_TIMEOUT)
            except requests.RequestException:
                return out
            if r.status_code != 200:
                log.warning("readwise: highlights HTTP %s", r.status_code)
                return out
            try:
                payload = r.json()
            except ValueError:
                return out
            for h in payload.get("results") or []:
                book_id = h.get("book_id")
                if book_id is None:
                    continue
                out[int(book_id)].append(h)
            url = payload.get("next")
        return out

    def _render(
        self, book_id: int, book: dict, highlights: list[dict],
    ) -> ConnectorDocument | None:
        if not highlights:
            return None
        title = book.get("title") or f"Book {book_id}"
        author = book.get("author") or ""
        category = book.get("category") or "book"
        source = book.get("source") or ""
        url = book.get("source_url") or ""

        lines: list[str] = [f"# {title}", ""]
        meta_bits: list[str] = []
        if author:
            meta_bits.append(f"by {author}")
        if category:
            meta_bits.append(category)
        if source:
            meta_bits.append(f"via {source}")
        if meta_bits:
            lines.append(" · ".join(meta_bits))
        if url:
            lines.append(f"Link: {url}")
        lines.append("")
        lines.append(f"**{len(highlights)} highlight(s)**")
        lines.append("")

        # Sort by location (best ordering signal Readwise gives us)
        # then by created date as fallback.
        def _sort_key(h):
            return (
                h.get("location") or 0,
                h.get("highlighted_at") or h.get("created_at") or "",
            )
        highlights = sorted(highlights, key=_sort_key)
        latest_ts = 0.0
        for h in highlights:
            text = (h.get("text") or "").strip()
            if not text:
                continue
            note = (h.get("note") or "").strip()
            tags = ", ".join(
                t.get("name", "") for t in (h.get("tags") or [])
            )
            lines.append(f"> {text}")
            if note:
                lines.append(f"  — _{note}_")
            if tags:
                lines.append(f"  _tags: {tags}_")
            lines.append("")
            ts_str = (
                h.get("highlighted_at") or h.get("created_at") or ""
            )
            if ts_str:
                try:
                    ts = datetime.fromisoformat(
                        ts_str.replace("Z", "+00:00"),
                    ).timestamp()
                    if ts > latest_ts:
                        latest_ts = ts
                except ValueError:
                    pass

        mtime = latest_ts or time.time()
        return ConnectorDocument(
            source="readwise",
            virtual_path=f"readwise://{category}/{book_id}",
            title=f"[highlights] {title}",
            content="\n".join(lines),
            mtime=mtime,
            kind="url",
            metadata={
                "book_id": book_id,
                "author": author,
                "category": category,
                "source": source,
                "url": url,
                "highlight_count": len(highlights),
            },
        )
