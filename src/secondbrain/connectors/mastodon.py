"""Mastodon connector — pulls your statuses + favourites + bookmarks.

Auth: a personal access token from your home instance. Setup:
  1. Go to https://<your-instance>/settings/applications
  2. New application; scopes ``read`` is enough.
  3. Copy "Your access token".
  4. ``[Environment]::SetEnvironmentVariable("MASTODON_INSTANCE", "https://hachyderm.io", "User")``
  5. ``[Environment]::SetEnvironmentVariable("MASTODON_ACCESS_TOKEN", "<token>", "User")``

What it ingests:
  - Statuses you've posted (via /api/v1/accounts/verify_credentials → id →
    /api/v1/accounts/<id>/statuses)
  - Statuses you've favourited (/api/v1/favourites)
  - Statuses you've bookmarked (/api/v1/bookmarks)

Reblogs/boosts aren't ingested separately — the boosted post still surfaces
through /api/v1/accounts/{id}/statuses with reblog=True; we render it with a
note so search distinguishes "I posted this" from "I boosted this".
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Iterator
from datetime import datetime
from html import unescape

import requests

from ..config import Config
from . import USER_AGENT, ConnectorDocument, respect_retry_after

log = logging.getLogger(__name__)

_TIMEOUT = 30
_DEFAULT_PER_KIND = 500
_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(html: str) -> str:
    if not html:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
    text = _TAG_RE.sub("", text)
    return unescape(text).strip()


def _iso_to_ts(s: str | None) -> float:
    if not s:
        return time.time()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return time.time()


class MastodonConnector:
    name = "mastodon"

    def is_enabled(self, cfg: Config) -> bool:
        return bool(
            os.environ.get("MASTODON_INSTANCE")
            and os.environ.get("MASTODON_ACCESS_TOKEN")
        )

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]:
        instance = os.environ["MASTODON_INSTANCE"].rstrip("/")
        token = os.environ["MASTODON_ACCESS_TOKEN"]
        cap = int(os.environ.get("SB_MASTODON_MAX", _DEFAULT_PER_KIND))
        s = requests.Session()
        s.headers.update({
            "Authorization": f"Bearer {token}",
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        })
        try:
            account = self._fetch_account(s, instance)
            if account is None:
                log.warning("Mastodon: could not resolve account from token")
                return
            uid = account.get("id")
            handle = account.get("acct") or account.get("username") or "?"

            seen: set[str] = set()
            # Own statuses (includes boosts; we tag them as such).
            yield from self._iter_paged(
                s, instance, f"/api/v1/accounts/{uid}/statuses",
                cap, label="own", handle=handle, seen=seen,
            )
            # Favourites.
            yield from self._iter_paged(
                s, instance, "/api/v1/favourites",
                cap, label="favourite", handle=handle, seen=seen,
            )
            # Bookmarks.
            yield from self._iter_paged(
                s, instance, "/api/v1/bookmarks",
                cap, label="bookmark", handle=handle, seen=seen,
            )
        finally:
            s.close()

    # --- helpers --------------------------------------------------------

    def _fetch_account(self, s: requests.Session, instance: str) -> dict | None:
        try:
            r = s.get(f"{instance}/api/v1/accounts/verify_credentials", timeout=_TIMEOUT)
        except requests.RequestException as e:
            log.warning("Mastodon verify_credentials failed: %s", type(e).__name__)
            return None
        if r.status_code != 200:
            log.warning("Mastodon verify_credentials HTTP %s", r.status_code)
            return None
        return r.json()

    def _iter_paged(
        self, s: requests.Session, instance: str, path: str,
        cap: int, label: str, handle: str, seen: set[str],
    ) -> Iterator[ConnectorDocument]:
        url = f"{instance}{path}"
        params: dict[str, int | str] = {"limit": 40}
        emitted = 0
        # Mastodon's paging uses Link: header rels; chase max_id manually.
        max_id: str | None = None
        while emitted < cap:
            qp = dict(params)
            if max_id:
                qp["max_id"] = max_id
            for _ in range(3):
                try:
                    r = s.get(url, params=qp, timeout=_TIMEOUT)
                except requests.RequestException as e:
                    log.warning("Mastodon %s fetch failed: %s", path, type(e).__name__)
                    return
                if respect_retry_after(r):
                    continue
                break
            else:
                return
            if r.status_code != 200:
                log.warning("Mastodon %s HTTP %s", path, r.status_code)
                return
            statuses = r.json() or []
            if not statuses:
                return
            for st in statuses:
                doc = self._render_status(st, label=label, owner_handle=handle)
                if doc is None:
                    continue
                # De-dup across favourite/bookmark/own pulls when the same
                # status shows up via multiple endpoints.
                if doc.virtual_path in seen:
                    continue
                seen.add(doc.virtual_path)
                yield doc
                emitted += 1
                if emitted >= cap:
                    return
            max_id = str(statuses[-1].get("id") or "")
            if not max_id:
                return

    def _render_status(
        self, st: dict, label: str, owner_handle: str
    ) -> ConnectorDocument | None:
        sid = st.get("id")
        if not sid:
            return None
        # If this is a boost, the meaningful content is in `reblog`.
        reblog = st.get("reblog")
        if reblog:
            inner = reblog
            is_boost = True
        else:
            inner = st
            is_boost = False

        text = _strip_html(inner.get("content") or "")
        if not text:
            return None
        author = ((inner.get("account") or {}).get("acct")) or "?"
        when = _iso_to_ts(inner.get("created_at"))
        url = inner.get("url") or ""

        favs = inner.get("favourites_count", 0)
        boosts = inner.get("reblogs_count", 0)
        replies = inner.get("replies_count", 0)

        kind_line = "boosted" if is_boost else label
        title_text = text.replace("\n", " ")
        title = title_text[:80] + ("…" if len(title_text) > 80 else "")
        lines = [
            f"# Mastodon {kind_line} · @{author}",
            "",
            text,
            "",
            f"Favs: {favs}  Boosts: {boosts}  Replies: {replies}",
            f"Posted: {datetime.fromtimestamp(when).isoformat() if when else '?'}",
            f"Link: {url}" if url else "",
        ]
        return ConnectorDocument(
            source="mastodon",
            # Use the inner status id so a boost and the original boosted
            # status converge on a single virtual_path.
            virtual_path=f"mastodon://status/{author}/{inner.get('id')}",
            title=title or f"Mastodon status {inner.get('id')}",
            content="\n".join(line for line in lines if line is not None),
            mtime=when,
            metadata={
                "label": label,
                "is_boost": is_boost,
                "author": author,
                "owner_handle": owner_handle,
                "url": url,
                "favourites": favs,
                "reblogs": boosts,
                "replies": replies,
            },
        )
