"""Substack connector — pulls posts from publications you follow.

Substack doesn't have a documented user API, but every publication exposes
an Atom/RSS feed at ``<publication-url>/feed`` (e.g.
``https://stratechery.com/feed``). This connector reads a list of feed URLs
from your config and pulls the latest posts from each.

Setup:
  - ``SUBSTACK_FEEDS`` env var: comma-separated list of feed URLs, e.g.
    ``https://stratechery.com/feed,https://www.platformer.news/feed``
  - Or set ``substack_feeds`` in config.toml as a TOML array (preferred —
    keeps the URLs out of your shell history).

What it ingests:
  - Title, author, publication, posted date
  - Full HTML body of the post (stripped to readable text)
  - Permalink

The body is whatever Substack chose to put in the feed. For paid posts
that's usually just the lede; that's still useful for retrieval ("what was
that thing I read about X"). For free posts you typically get the full text.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Iterator
from html import unescape
from xml.etree import ElementTree as ET

import requests

from ..config import Config
from . import USER_AGENT, ConnectorDocument, respect_retry_after

log = logging.getLogger(__name__)

_TIMEOUT = 30
_PER_FEED_CAP = 200

# Atom/RSS namespaces — feed parsers without lxml have to spell these out.
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "content": "http://purl.org/rss/1.0/modules/content/",
    "dc": "http://purl.org/dc/elements/1.1/",
}

_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(html: str) -> str:
    if not html:
        return ""
    text = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
    text = _TAG_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return unescape(text).strip()


def _parse_feed_urls(cfg: Config) -> list[str]:
    """Resolve the list of Substack feed URLs from config or env."""
    urls = list(getattr(cfg, "substack_feeds", ()) or ())
    raw = os.environ.get("SUBSTACK_FEEDS", "")
    for chunk in raw.split(","):
        u = chunk.strip()
        if u:
            urls.append(u)
    # De-dup while keeping order.
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def _rss_to_iso(value: str | None) -> float:
    """Best-effort: parse an RFC-822 date string (RSS) or ISO-8601 (Atom)."""
    if not value:
        return time.time()
    # Try RFC 822 first (RSS is more common on Substack feeds).
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
    ):
        try:
            from datetime import datetime
            return datetime.strptime(value, fmt).timestamp()
        except ValueError:
            continue
    return time.time()


class SubstackConnector:
    name = "substack"

    def is_enabled(self, cfg: Config) -> bool:
        return bool(_parse_feed_urls(cfg))

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]:
        feeds = _parse_feed_urls(cfg)
        s = requests.Session()
        s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/atom+xml,application/rss+xml,text/xml"})
        try:
            for url in feeds:
                yield from self._fetch_feed(s, url)
        finally:
            s.close()

    # --- helpers --------------------------------------------------------

    def _fetch_feed(self, s: requests.Session, url: str) -> Iterator[ConnectorDocument]:
        for _ in range(3):
            try:
                r = s.get(url, timeout=_TIMEOUT)
            except requests.RequestException as e:
                log.warning("Substack feed fetch %s failed: %s", url, type(e).__name__)
                return
            if respect_retry_after(r):
                continue
            break
        else:
            return
        if r.status_code != 200:
            log.warning("Substack feed %s returned HTTP %s", url, r.status_code)
            return

        try:
            root = ET.fromstring(r.text)
        except ET.ParseError as e:
            log.warning("Substack feed %s parse error: %s", url, e)
            return

        # Walk both RSS (channel/item) and Atom (feed/entry) shapes.
        items: list[ET.Element] = list(root.findall(".//item"))
        if not items:
            items = list(root.findall("atom:entry", _NS))
        publication = self._publication_title(root)
        emitted = 0
        for item in items:
            doc = self._render_item(item, publication, fallback_feed=url)
            if doc is None:
                continue
            yield doc
            emitted += 1
            if emitted >= _PER_FEED_CAP:
                return

    def _publication_title(self, root: ET.Element) -> str:
        # Prefer the channel/feed title over the feed URL slug.
        ch = root.find("channel")
        if ch is not None:
            t = ch.findtext("title")
            if t:
                return t.strip()
        t = root.findtext("atom:title", default=None, namespaces=_NS)
        return (t or "").strip()

    def _render_item(
        self, item: ET.Element, publication: str, fallback_feed: str,
    ) -> ConnectorDocument | None:
        # RSS shape
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        guid = (item.findtext("guid") or "").strip()
        pub_date = item.findtext("pubDate") or item.findtext("dc:date", namespaces=_NS)
        # Atom fallback
        if not title:
            title = (item.findtext("atom:title", default="", namespaces=_NS) or "").strip()
        if not link:
            link_el = item.find("atom:link", _NS)
            if link_el is not None:
                link = link_el.attrib.get("href", "")
        if not guid:
            guid = (item.findtext("atom:id", default="", namespaces=_NS) or "").strip()
        if not pub_date:
            pub_date = (
                item.findtext("atom:published", namespaces=_NS)
                or item.findtext("atom:updated", namespaces=_NS)
            )

        # Body: prefer content:encoded (full HTML), then description, then atom content.
        html_body = item.findtext("content:encoded", namespaces=_NS) or ""
        if not html_body:
            html_body = item.findtext("description") or ""
        if not html_body:
            html_body = item.findtext("atom:content", default="", namespaces=_NS) or ""
        body = _strip_html(html_body)

        author = (item.findtext("dc:creator", namespaces=_NS) or "").strip()
        if not author:
            atom_author = item.find("atom:author/atom:name", _NS)
            if atom_author is not None and atom_author.text:
                author = atom_author.text.strip()

        if not (title or body):
            return None

        identifier = guid or link or f"{fallback_feed}#{title[:60]}"
        when = _rss_to_iso(pub_date)

        lines = [f"# {title or '(untitled)'}", ""]
        if publication:
            lines.append(f"Publication: {publication}")
        if author:
            lines.append(f"Author: {author}")
        if pub_date:
            lines.append(f"Posted: {pub_date}")
        if link:
            lines.append(f"Link: {link}")
        if body:
            lines.append("")
            lines.append(body)

        # Use the canonical post URL as the virtual_path when available - it's
        # already globally unique. Fall back to guid if the feed only ships ids.
        vp = link or f"substack://item/{identifier}"
        return ConnectorDocument(
            source="substack",
            virtual_path=vp,
            title=title or "(untitled)",
            content="\n".join(lines),
            mtime=when,
            metadata={
                "publication": publication,
                "author": author,
                "url": link,
                "guid": guid,
            },
        )
