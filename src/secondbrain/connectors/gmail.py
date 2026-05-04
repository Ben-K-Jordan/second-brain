"""Gmail connector — pulls recent messages via the Gmail API.

Uses the shared Google OAuth scaffold (``_google_oauth.py``). Enabled once
you've run ``secondbrain auth google`` with Gmail in scope.

Defaults:
- Last 30 days of messages
- Excludes Promotions / Social / Updates categories (these are mostly
  marketing noise; configurable via env)
- Caps at 500 messages per sync to keep first runs bounded
- Skips drafts and spam

Each message becomes a ConnectorDocument with subject + sender + recipients
+ plain-text body. HTML-only messages are converted with a basic strip.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import time
from collections.abc import Iterator
from email.utils import parsedate_to_datetime

import requests

from ..config import Config
from . import ConnectorDocument
from ._google_oauth import (
    GoogleAuthError,
    ScopeMissing,
    authorized_session,
    is_authorized,
)

log = logging.getLogger(__name__)

# Read-only Gmail scope. We never modify user mail.
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

_API = "https://gmail.googleapis.com/gmail/v1"
_DEFAULT_QUERY = "newer_than:30d -category:promotions -category:social -category:updates -in:spam -in:trash"
_DEFAULT_MAX = 500


def _strip_html(html: str) -> str:
    """Convert HTML email body to readable plain text. Not perfect but good
    enough for retrieval — Gmail viewers do something similar by default."""
    text = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Preserve preformatted code blocks - the catch-all <[^>]+> below would
    # otherwise join code lines with no whitespace, killing recall on
    # code-in-email (alerts from CI, code review notifications, etc).
    text = re.sub(r"<pre[^>]*>", "\n```\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</pre>", "\n```\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    # Decode common HTML entities without pulling in html.parser overhead
    text = (
        text.replace("&nbsp;", " ").replace("&amp;", "&")
            .replace("&lt;", "<").replace("&gt;", ">")
            .replace("&quot;", '"').replace("&#39;", "'")
    )
    # Collapse runs of blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _decode_body(data: str) -> str:
    """Gmail returns body data as URL-safe base64. Decode to text."""
    if not data:
        return ""
    try:
        raw = base64.urlsafe_b64decode(data + "==")  # tolerate missing padding
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_body(payload: dict) -> str:
    """Walk a Gmail message payload (potentially nested multipart) and pull
    out the best body text we can find.

    Preference order: plain-text part > HTML part (stripped). For multipart,
    we recurse into all parts and concatenate plain text; if no plain text
    is found we fall back to the first HTML part."""
    plain_parts: list[str] = []
    html_parts: list[str] = []

    def walk(part: dict) -> None:
        mime = (part.get("mimeType") or "").lower()
        body = part.get("body") or {}
        data = body.get("data")
        if mime == "text/plain" and data:
            plain_parts.append(_decode_body(data))
        elif mime == "text/html" and data:
            html_parts.append(_decode_body(data))
        for sub in part.get("parts", []) or []:
            walk(sub)

    walk(payload)
    if plain_parts:
        return "\n\n".join(p for p in plain_parts if p).strip()
    if html_parts:
        return _strip_html(html_parts[0])
    return ""


def _header(headers: list[dict], name: str) -> str:
    name_l = name.lower()
    for h in headers:
        if (h.get("name") or "").lower() == name_l:
            return h.get("value", "") or ""
    return ""


def _date_to_ts(date_header: str) -> float:
    if not date_header:
        return time.time()
    try:
        dt = parsedate_to_datetime(date_header)
        return dt.timestamp()
    except (TypeError, ValueError):
        return time.time()


class GmailConnector:
    name = "gmail"

    def is_enabled(self, cfg: Config) -> bool:
        # Enabled if a Google client_secret + valid creds exist for Gmail scope.
        return is_authorized(cfg, GMAIL_SCOPES)

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]:
        try:
            s = authorized_session(cfg, GMAIL_SCOPES)
        except ScopeMissing as e:
            log.warning(
                "Gmail: %s. Re-run `secondbrain auth google` to grant the missing scope.",
                e,
            )
            return
        except GoogleAuthError as e:
            log.warning("Gmail: auth error: %s", e)
            return
        if s is None:
            log.warning("Gmail: no Google credentials. Run `secondbrain auth google`.")
            return

        query = os.environ.get("SB_GMAIL_QUERY", _DEFAULT_QUERY)
        max_messages = int(os.environ.get("SB_GMAIL_MAX", _DEFAULT_MAX))

        try:
            for msg_id in self._iter_message_ids(s, query, max_messages):
                doc = self._fetch_message(s, msg_id)
                if doc is not None:
                    yield doc
        finally:
            s.close()

    # --- helpers --------------------------------------------------------

    def _iter_message_ids(
        self, s: requests.Session, query: str, cap: int
    ) -> Iterator[str]:
        # Round 18 fix (audit-found gap M5) — honor 429 Retry-After.
        # Gmail's per-user quota is 250 quota units / second; without
        # backoff a busy mailbox sync gets throttled and truncates.
        from . import respect_retry_after
        page_token: str | None = None
        emitted = 0
        while emitted < cap:
            params = {"q": query, "maxResults": min(100, cap - emitted)}
            if page_token:
                params["pageToken"] = page_token
            r = s.get(f"{_API}/users/me/messages", params=params, timeout=30)
            if respect_retry_after(r):
                continue  # re-issue same page
            if r.status_code != 200:
                log.warning("Gmail list failed: %s %s", r.status_code, r.text[:200])
                return
            data = r.json()
            for item in data.get("messages") or []:
                yield item["id"]
                emitted += 1
                if emitted >= cap:
                    return
            page_token = data.get("nextPageToken")
            if not page_token:
                return

    def _fetch_message(self, s: requests.Session, msg_id: str) -> ConnectorDocument | None:
        from . import respect_retry_after
        url = f"{_API}/users/me/messages/{msg_id}"
        params = {"format": "full"}
        r = s.get(url, params=params, timeout=30)
        if respect_retry_after(r):
            r = s.get(url, params=params, timeout=30)
        if r.status_code != 200:
            log.warning("Gmail fetch %s failed: %s", msg_id, r.status_code)
            return None
        msg = r.json()
        payload = msg.get("payload", {})
        headers = payload.get("headers", []) or []

        subject = _header(headers, "Subject") or "(no subject)"
        from_ = _header(headers, "From")
        to = _header(headers, "To")
        cc = _header(headers, "Cc")
        date = _header(headers, "Date")
        snippet = msg.get("snippet", "")
        body = _extract_body(payload) or snippet
        if not body.strip() and not subject.strip():
            return None

        thread_id = msg.get("threadId", msg_id)
        labels = msg.get("labelIds", []) or []

        meta_lines = [
            f"Subject: {subject}",
            f"From: {from_}",
            f"To: {to}",
        ]
        if cc:
            meta_lines.append(f"Cc: {cc}")
        if date:
            meta_lines.append(f"Date: {date}")
        if labels:
            meta_lines.append(f"Labels: {', '.join(labels)}")
        text = "# " + subject + "\n\n" + "\n".join(meta_lines) + "\n\n" + body

        return ConnectorDocument(
            source="gmail",
            virtual_path=f"gmail://thread/{thread_id}/message/{msg_id}",
            title=subject,
            content=text,
            mtime=_date_to_ts(date),
            metadata={
                "from": from_, "to": to, "cc": cc,
                "date": date, "thread_id": thread_id, "labels": labels,
            },
        )
