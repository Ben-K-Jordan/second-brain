"""News connector — keyword-driven article ingestion via NewsAPI.org.

Setup:
  1. Get a free API key at https://newsapi.org (100 requests/day on free).
  2. ``[Environment]::SetEnvironmentVariable("NEWSAPI_KEY", "<key>", "User")``
  3. Configure topics in ``~/.secondbrain/config.toml``::

        news_topics  = ["artificial intelligence", "anthropic", "openai"]
        news_sources = ["techcrunch", "the-verge"]   # optional source allowlist
        news_window_days = 1                         # how far back to pull

Each article (title + description + URL + published_at) becomes one
ConnectorDocument keyed by URL. Re-running the sync just upserts; deleted
or aged-out articles are not actively pruned (they fall off the recency
window naturally).

The free tier of NewsAPI doesn't return full article bodies — only title +
description + URL. That's still very useful for retrieval ("what was that
thing I read about X yesterday") and the URL lets the user / chat agent
fetch the full content on demand.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from urllib.parse import quote

import requests

from ..config import Config
from . import USER_AGENT, ConnectorDocument, respect_retry_after

log = logging.getLogger(__name__)

_API = "https://newsapi.org/v2"
_TIMEOUT = 30
_PAGE_SIZE = 100
_DEFAULT_WINDOW_DAYS = 1


def _iso_to_ts(s: str | None) -> float:
    if not s:
        return time.time()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return time.time()


def _topics(cfg: Config) -> list[str]:
    return [t.strip() for t in (getattr(cfg, "news_topics", ()) or ()) if t.strip()]


def _sources(cfg: Config) -> str | None:
    """NewsAPI takes 'sources' as a comma-separated list of source ids."""
    src = [s.strip() for s in (getattr(cfg, "news_sources", ()) or ()) if s.strip()]
    return ",".join(src) if src else None


def _window_days(cfg: Config) -> int:
    raw = getattr(cfg, "news_window_days", _DEFAULT_WINDOW_DAYS)
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return _DEFAULT_WINDOW_DAYS


class NewsConnector:
    """Pulls keyword-matched articles from NewsAPI.org.

    Enabled when ``NEWSAPI_KEY`` is set AND ``news_topics`` is non-empty.
    """

    name = "news"

    def is_enabled(self, cfg: Config) -> bool:
        return bool(os.environ.get("NEWSAPI_KEY")) and bool(_topics(cfg))

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]:
        api_key = os.environ["NEWSAPI_KEY"]
        topics = _topics(cfg)
        sources = _sources(cfg)
        window = _window_days(cfg)
        cutoff = datetime.now(UTC).replace(microsecond=0)
        from_date = cutoff.fromtimestamp(time.time() - window * 86400, tz=UTC)
        from_str = from_date.strftime("%Y-%m-%dT%H:%M:%S")

        s = requests.Session()
        s.headers.update({
            "User-Agent": USER_AGENT,
            "X-Api-Key": api_key,
            "Accept": "application/json",
        })
        seen_urls: set[str] = set()
        try:
            for topic in topics:
                yield from self._fetch_topic(s, topic, sources, from_str, seen_urls)
        finally:
            s.close()

    # --- helpers --------------------------------------------------------

    def _fetch_topic(
        self,
        s: requests.Session,
        topic: str,
        sources: str | None,
        from_str: str,
        seen_urls: set[str],
    ) -> Iterator[ConnectorDocument]:
        # Use /everything for full keyword search; /top-headlines is too
        # narrow once topics get specific.
        params: dict[str, str | int] = {
            "q": topic,
            "from": from_str,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": _PAGE_SIZE,
        }
        if sources:
            params["sources"] = sources
        url = (
            f"{_API}/everything?"
            + "&".join(f"{k}={quote(str(v), safe='')}" for k, v in params.items())
        )
        for _ in range(3):
            try:
                r = s.get(url, timeout=_TIMEOUT)
            except requests.RequestException as e:
                log.warning("news fetch failed: %s", type(e).__name__)
                return
            if respect_retry_after(r):
                continue
            break
        else:
            return
        if r.status_code == 401:
            log.warning("news: NEWSAPI_KEY rejected (HTTP 401)")
            return
        if r.status_code == 426:
            log.warning(
                "news: NEWSAPI requires HTTPS upgrade or a paid plan for "
                "this query (HTTP 426). Try fewer/older articles."
            )
            return
        if r.status_code != 200:
            log.warning("news /everything HTTP %s for topic %r", r.status_code, topic)
            return
        try:
            data = r.json()
        except ValueError:
            log.warning("news: non-JSON response")
            return
        if data.get("status") != "ok":
            log.warning("news error: %s", data.get("message", "?"))
            return
        for art in data.get("articles") or []:
            doc = self._render_article(art, topic)
            if doc is None:
                continue
            if doc.virtual_path in seen_urls:
                continue
            seen_urls.add(doc.virtual_path)
            yield doc

    def _render_article(self, art: dict, topic: str) -> ConnectorDocument | None:
        url = (art.get("url") or "").strip()
        title = (art.get("title") or "").strip()
        if not url or not title:
            return None
        # NewsAPI sometimes returns "[Removed]" placeholder articles when
        # the source has retracted the piece. Skip those.
        if title == "[Removed]":
            return None
        description = (art.get("description") or "").strip()
        content = (art.get("content") or "").strip()
        author = (art.get("author") or "").strip()
        source_name = ((art.get("source") or {}).get("name")) or ""
        published = _iso_to_ts(art.get("publishedAt"))

        lines = [f"# {title}", ""]
        if source_name:  lines.append(f"Source: {source_name}")
        if author:       lines.append(f"Author: {author}")
        if published:
            lines.append(
                "Published: "
                + datetime.fromtimestamp(published, tz=UTC)
                .strftime("%Y-%m-%d %H:%M UTC")
            )
        lines.append(f"Topic: {topic}")
        lines.append(f"Link: {url}")
        if description:
            lines.append("")
            lines.append(description)
        if content and content != description:
            lines.append("")
            lines.append(content)

        return ConnectorDocument(
            source="news",
            virtual_path=url,
            title=title,
            content="\n".join(lines),
            mtime=published or time.time(),
            metadata={
                "topic": topic,
                "source_name": source_name,
                "author": author,
                "url": url,
            },
        )
