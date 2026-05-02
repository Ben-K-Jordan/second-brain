"""Generic RSS / Atom feed connector.

Substack already has its own connector (substack-specific UX); this is the
catch-all for everything else: Indeed RSS (where it works), Google Alerts
(every alert exposes an RSS feed), HN-style aggregators, blog feeds you
follow, your own LinkedIn job-alert RSS (saved searches export RSS), etc.

Setup:
  - ``RSS_FEEDS`` env var: comma-separated feed URLs. Or set
    ``rss_feeds`` in config.toml as a TOML array (preferred — keeps URLs
    out of your shell history; some Google Alert URLs contain a token).

Each item becomes one ConnectorDocument keyed by the item link (or guid
when there's no link). Re-running the sync just upserts.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Iterator
from datetime import datetime
from html import unescape
from xml.etree import ElementTree as ET

import requests

from ..config import Config
from . import USER_AGENT, ConnectorDocument, respect_retry_after

log = logging.getLogger(__name__)

_TIMEOUT = 30
_PER_FEED_CAP = 200
_TAG_RE = re.compile(r"<[^>]+>")
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "content": "http://purl.org/rss/1.0/modules/content/",
    "dc": "http://purl.org/dc/elements/1.1/",
}


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


def _parse_date(value: str | None) -> float:
    """RSS uses RFC 822 dates; Atom uses ISO-8601. Try both, fall back to now."""
    if not value:
        return time.time()
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
    ):
        try:
            return datetime.strptime(value, fmt).timestamp()
        except ValueError:
            continue
    return time.time()


def _feeds(cfg: Config) -> list[str]:
    out: list[str] = []
    for u in getattr(cfg, "rss_feeds", ()) or ():
        u = (u or "").strip()
        if u and u not in out:
            out.append(u)
    raw = os.environ.get("RSS_FEEDS", "")
    for chunk in raw.split(","):
        u = chunk.strip()
        if u and u not in out:
            out.append(u)
    return out


class RSSConnector:
    """Pulls items from a configured list of arbitrary RSS / Atom feeds."""

    name = "rss"

    def is_enabled(self, cfg: Config) -> bool:
        return bool(_feeds(cfg))

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]:
        s = requests.Session()
        s.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "application/atom+xml,application/rss+xml,text/xml;q=0.9,*/*;q=0.5",
        })
        try:
            for url in _feeds(cfg):
                yield from self._fetch_feed(s, url)
        finally:
            s.close()

    # --- helpers --------------------------------------------------------

    def _fetch_feed(self, s: requests.Session, url: str) -> Iterator[ConnectorDocument]:
        for _ in range(3):
            try:
                r = s.get(url, timeout=_TIMEOUT)
            except requests.RequestException as e:
                log.warning("rss fetch %s failed: %s", url, type(e).__name__)
                return
            if respect_retry_after(r):
                continue
            break
        else:
            return
        if r.status_code != 200:
            log.warning("rss feed %s HTTP %s", url, r.status_code)
            return
        try:
            root = ET.fromstring(r.text)
        except ET.ParseError as e:
            log.warning("rss feed %s parse error: %s", url, e)
            return
        feed_title = self._feed_title(root) or url

        items = list(root.findall(".//item"))
        if not items:
            items = list(root.findall("atom:entry", _NS))
        emitted = 0
        for item in items:
            doc = self._render_item(item, feed_url=url, feed_title=feed_title)
            if doc is None:
                continue
            yield doc
            emitted += 1
            if emitted >= _PER_FEED_CAP:
                return

    def _feed_title(self, root: ET.Element) -> str:
        ch = root.find("channel")
        if ch is not None:
            t = ch.findtext("title")
            if t:
                return t.strip()
        t = root.findtext("atom:title", default=None, namespaces=_NS)
        return (t or "").strip()

    def _render_item(
        self, item: ET.Element, feed_url: str, feed_title: str,
    ) -> ConnectorDocument | None:
        title = (item.findtext("title") or "").strip() or (
            item.findtext("atom:title", default="", namespaces=_NS) or ""
        ).strip()
        link = (item.findtext("link") or "").strip()
        if not link:
            link_el = item.find("atom:link", _NS)
            if link_el is not None:
                link = link_el.attrib.get("href", "")
        guid = (item.findtext("guid") or "").strip() or (
            item.findtext("atom:id", default="", namespaces=_NS) or ""
        ).strip()
        pub = (
            item.findtext("pubDate")
            or item.findtext("dc:date", namespaces=_NS)
            or item.findtext("atom:published", namespaces=_NS)
            or item.findtext("atom:updated", namespaces=_NS)
        )
        author = (item.findtext("dc:creator", namespaces=_NS) or "").strip()
        if not author:
            ae = item.find("atom:author/atom:name", _NS)
            if ae is not None and ae.text:
                author = ae.text.strip()

        html_body = item.findtext("content:encoded", namespaces=_NS) or ""
        if not html_body:
            html_body = item.findtext("description") or ""
        if not html_body:
            html_body = item.findtext("atom:content", default="", namespaces=_NS) or ""
        body = _strip_html(html_body)

        if not (title or body):
            return None

        identifier = link or guid or f"{feed_url}#{title[:60]}"
        when = _parse_date(pub)

        lines = [f"# {title or '(untitled)'}", ""]
        if feed_title and feed_title != title:
            lines.append(f"Feed: {feed_title}")
        if author:
            lines.append(f"Author: {author}")
        if pub:
            lines.append(f"Posted: {pub}")
        if link:
            lines.append(f"Link: {link}")
        if body:
            lines.append("")
            lines.append(body)

        # Canonical URL is preferred for the virtual_path so re-runs upsert.
        vp = link if link else f"rss://{identifier}"
        return ConnectorDocument(
            source="rss",
            virtual_path=vp,
            title=title or "(untitled RSS item)",
            content="\n".join(lines),
            mtime=when,
            metadata={
                "feed_title": feed_title,
                "feed_url": feed_url,
                "url": link,
                "author": author,
                "guid": guid,
            },
        )
