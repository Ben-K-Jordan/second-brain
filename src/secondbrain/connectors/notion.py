"""Notion connector — pulls all pages your integration has access to.

Auth: Internal integration token in ``NOTION_TOKEN``. Create one at
https://www.notion.so/my-integrations, then share specific pages or your
whole workspace with the integration. Without sharing, this fetches nothing.

Each page becomes a ConnectorDocument with rendered Markdown-ish text from
its blocks. Nested pages, databases, and child blocks beyond the first
level are not recursed in v1; they show up as their own top-level pages
when shared.
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

_API = "https://api.notion.com/v1"
_NOTION_VERSION = "2022-06-28"


def _iso_to_ts(s: str | None) -> float:
    if not s:
        return time.time()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return time.time()


def _rich_text(rich: list[dict]) -> str:
    """Flatten Notion's rich_text array into plain text."""
    return "".join(t.get("plain_text", "") for t in rich)


class NotionConnector:
    name = "notion"

    def is_enabled(self, cfg: Config) -> bool:
        return bool(os.environ.get("NOTION_TOKEN"))

    def _session(self) -> requests.Session:
        s = requests.Session()
        s.headers.update({
            "Authorization": f"Bearer {os.environ['NOTION_TOKEN']}",
            "Notion-Version": _NOTION_VERSION,
            "Content-Type": "application/json",
            "User-Agent": "second-brain/0.0.1",
        })
        return s

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]:
        s = self._session()
        try:
            for page in self._iter_pages(s):
                doc = self._fetch_page(s, page)
                if doc is not None:
                    yield doc
        finally:
            s.close()

    # --- helpers ----------------------------------------------------------

    def _iter_pages(self, s: requests.Session) -> Iterator[dict]:
        cursor: str | None = None
        while True:
            body: dict = {
                "filter": {"property": "object", "value": "page"},
                "page_size": 100,
            }
            if cursor:
                body["start_cursor"] = cursor
            r = s.post(f"{_API}/search", json=body, timeout=30)
            if r.status_code != 200:
                log.warning("Notion /search failed: %s %s", r.status_code, r.text[:200])
                return
            data = r.json()
            for page in data.get("results", []):
                yield page
            if not data.get("has_more"):
                return
            cursor = data.get("next_cursor")

    def _fetch_page(self, s: requests.Session, page: dict) -> ConnectorDocument | None:
        page_id = page["id"]
        title = self._extract_title(page)
        try:
            blocks = self._fetch_blocks(s, page_id)
        except Exception as e:
            log.warning("Notion blocks fetch failed for %s: %s", page_id, e)
            return None

        body = self._render_blocks(blocks)
        if not body.strip() and not title.strip():
            return None

        text = f"# {title or 'Untitled'}\n\n{body}"
        url = page.get("url") or f"https://www.notion.so/{page_id.replace('-', '')}"

        return ConnectorDocument(
            source="notion",
            virtual_path=f"notion://{page_id}",
            title=title or "Untitled Notion page",
            content=text,
            mtime=_iso_to_ts(page.get("last_edited_time")),
            metadata={"notion_url": url},
        )

    def _fetch_blocks(self, s: requests.Session, block_id: str) -> list[dict]:
        out: list[dict] = []
        cursor: str | None = None
        while True:
            params = {"page_size": 100}
            if cursor:
                params["start_cursor"] = cursor
            r = s.get(f"{_API}/blocks/{block_id}/children", params=params, timeout=30)
            if r.status_code != 200:
                break
            data = r.json()
            out.extend(data.get("results", []))
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
        return out

    def _extract_title(self, page: dict) -> str:
        # Pages either expose `properties.<title-prop>.title` or `properties.title.title`
        for _, prop in (page.get("properties") or {}).items():
            if prop.get("type") == "title":
                t = _rich_text(prop.get("title", []))
                if t:
                    return t
        return ""

    def _render_blocks(self, blocks: list[dict]) -> str:
        lines: list[str] = []
        for b in blocks:
            t = b.get("type")
            data = b.get(t, {}) if t else {}
            text = _rich_text(data.get("rich_text", []))
            if t == "heading_1":
                lines.append(f"# {text}")
            elif t == "heading_2":
                lines.append(f"## {text}")
            elif t == "heading_3":
                lines.append(f"### {text}")
            elif t == "bulleted_list_item":
                lines.append(f"- {text}")
            elif t == "numbered_list_item":
                lines.append(f"1. {text}")
            elif t == "to_do":
                box = "x" if data.get("checked") else " "
                lines.append(f"- [{box}] {text}")
            elif t == "quote":
                lines.append(f"> {text}")
            elif t == "callout":
                lines.append(f"> {text}")
            elif t == "code":
                lang = data.get("language", "")
                lines.append(f"```{lang}\n{text}\n```")
            elif t == "divider":
                lines.append("---")
            elif t == "paragraph":
                if text:
                    lines.append(text)
            elif text:
                lines.append(text)
        return "\n\n".join(line for line in lines if line)
