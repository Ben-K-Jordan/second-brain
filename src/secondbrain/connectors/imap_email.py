"""IMAP email connector — ingest emails matching configured criteria.

This is the universal interface for everything platform-x doesn't expose
an API for. LinkedIn, Indeed, Handshake, and basically every recruiting
platform supports email alerts; Google Alerts is email-first; most
newsletters arrive by email. Pointing this connector at a Gmail label
("LinkedIn") or an IMAP folder turns all of those into searchable text.

Setup (Gmail example):
  1. Create a filter that labels relevant mail (e.g. ``label:LinkedIn``).
  2. Enable IMAP in Gmail settings; create an app password
     (https://myaccount.google.com/apppasswords).
  3. Configure in config.toml::

        [imap]
        host        = "imap.gmail.com"
        port        = 993
        username    = "you@gmail.com"
        # The folder/label to scan. For Gmail labels:
        folders     = ["LinkedIn", "JobAlerts", "Substack"]
        # Days back to ingest on each sync.
        window_days = 14

     Plus the env var:
        SECONDBRAIN_IMAP_PASSWORD=<your app password>

Each email becomes one ConnectorDocument keyed by ``imap://<folder>/<uid>``
or by Message-ID when present.

Read-only — we never modify or delete messages on the server.
"""

from __future__ import annotations

import email
import imaplib
import logging
import os
import re
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from email.message import Message
from email.utils import parsedate_to_datetime
from html import unescape

from ..config import Config
from . import ConnectorDocument

log = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_DEFAULT_WINDOW_DAYS = 14
_DEFAULT_MAX_PER_FOLDER = 1000


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


def _decode_part(part: Message) -> str:
    """Decode a MIME part to text, handling charset + transfer encoding."""
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""
    if isinstance(payload, bytes):
        charset = part.get_content_charset() or "utf-8"
        try:
            return payload.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            return payload.decode("utf-8", errors="replace")
    return str(payload)


def _extract_body(msg: Message) -> str:
    """Walk the MIME tree to extract a readable body. Prefer text/plain;
    fall back to text/html with tags stripped."""
    plain_parts: list[str] = []
    html_parts: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            ct = (part.get_content_type() or "").lower()
            if ct == "text/plain":
                plain_parts.append(_decode_part(part))
            elif ct == "text/html":
                html_parts.append(_decode_part(part))
    else:
        ct = (msg.get_content_type() or "").lower()
        if ct == "text/plain":
            plain_parts.append(_decode_part(msg))
        elif ct == "text/html":
            html_parts.append(_decode_part(msg))
    if plain_parts:
        return "\n\n".join(p for p in plain_parts if p).strip()
    if html_parts:
        return _strip_html(html_parts[0])
    return ""


def _config(cfg: Config) -> dict | None:
    """Read IMAP config from the ``imap`` block in config.toml.

    Returns None when the connector isn't configured (host or username
    missing). Tolerant of partial config.
    """
    host = (getattr(cfg, "imap_host", "") or "").strip()
    user = (getattr(cfg, "imap_username", "") or "").strip()
    if not host or not user:
        return None
    folders = list(getattr(cfg, "imap_folders", ()) or ())
    if not folders:
        return None
    return {
        "host": host,
        "port": int(getattr(cfg, "imap_port", 993) or 993),
        "username": user,
        "folders": folders,
        "window_days": int(getattr(cfg, "imap_window_days", _DEFAULT_WINDOW_DAYS)),
        "max_per_folder": int(
            getattr(cfg, "imap_max_per_folder", _DEFAULT_MAX_PER_FOLDER)
        ),
    }


class ImapEmailConnector:
    """Read-only IMAP scanner. Pulls every message in configured folders
    within ``imap_window_days`` and emits one ConnectorDocument per email.
    """

    name = "imap"

    def is_enabled(self, cfg: Config) -> bool:
        if _config(cfg) is None:
            return False
        return bool(os.environ.get("SECONDBRAIN_IMAP_PASSWORD"))

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]:
        config = _config(cfg)
        if config is None:
            return
        password = os.environ.get("SECONDBRAIN_IMAP_PASSWORD", "")
        if not password:
            return

        try:
            imap = imaplib.IMAP4_SSL(config["host"], config["port"], timeout=30)
        except (OSError, imaplib.IMAP4.error) as e:
            log.warning("IMAP connect to %s failed: %s", config["host"], e)
            return

        try:
            try:
                imap.login(config["username"], password)
            except imaplib.IMAP4.error as e:
                log.warning("IMAP login failed (check SECONDBRAIN_IMAP_PASSWORD): %s", e)
                return
            for folder in config["folders"]:
                yield from self._fetch_folder(
                    imap, folder, config["window_days"], config["max_per_folder"],
                )
        finally:
            try:
                imap.logout()
            except imaplib.IMAP4.error:
                pass

    # --- helpers --------------------------------------------------------

    def _fetch_folder(
        self, imap: imaplib.IMAP4_SSL, folder: str, window_days: int, cap: int,
    ) -> Iterator[ConnectorDocument]:
        # Gmail labels with spaces/special chars need quoted folder names.
        select_arg = f'"{folder}"' if (" " in folder or "/" in folder) else folder
        try:
            status, _ = imap.select(select_arg, readonly=True)
        except imaplib.IMAP4.error as e:
            log.warning("IMAP select %r failed: %s", folder, e)
            return
        if status != "OK":
            log.warning("IMAP folder %r doesn't exist or is unreadable", folder)
            return

        since = (datetime.now(UTC) - _td(window_days)).strftime("%d-%b-%Y")
        try:
            status, data = imap.search(None, "SINCE", since)
        except imaplib.IMAP4.error as e:
            log.warning("IMAP search %r failed: %s", folder, e)
            return
        if status != "OK" or not data or not data[0]:
            return
        uids = data[0].split()
        # Most-recent first, capped.
        uids = list(reversed(uids))[:cap]
        for uid in uids:
            doc = self._fetch_one(imap, folder, uid)
            if doc is not None:
                yield doc

    def _fetch_one(
        self, imap: imaplib.IMAP4_SSL, folder: str, uid: bytes,
    ) -> ConnectorDocument | None:
        try:
            status, data = imap.fetch(uid, "(RFC822)")
        except imaplib.IMAP4.error as e:
            log.warning("IMAP fetch %s/%s failed: %s", folder, uid, e)
            return None
        if status != "OK" or not data or not data[0]:
            return None
        # data[0] is a tuple; the second element is the raw message bytes.
        raw = data[0][1] if isinstance(data[0], tuple) else None
        if not raw:
            return None
        msg = email.message_from_bytes(raw)

        from_ = (msg.get("From") or "").strip()
        to_ = (msg.get("To") or "").strip()
        subject = (msg.get("Subject") or "").strip() or "(no subject)"
        date_hdr = msg.get("Date") or ""
        msg_id = (msg.get("Message-ID") or "").strip().strip("<>")
        try:
            when_dt = parsedate_to_datetime(date_hdr) if date_hdr else None
            mtime = when_dt.timestamp() if when_dt else time.time()
        except (TypeError, ValueError):
            mtime = time.time()

        body = _extract_body(msg) or "(no body)"
        # Cap body so a giant 1MB email doesn't blow up an embedding batch.
        if len(body) > 60_000:
            body = body[:60_000] + "\n[...truncated]"

        lines = [f"# {subject}", ""]
        if from_:    lines.append(f"From: {from_}")
        if to_:      lines.append(f"To: {to_}")
        if date_hdr: lines.append(f"Date: {date_hdr}")
        lines.append(f"Folder: {folder}")
        lines.append("")
        lines.append(body)

        # virtual_path uses Message-ID when present (globally stable),
        # falls back to imap://folder/uid.
        if msg_id:
            vp = f"imap://msgid/{msg_id}"
        else:
            uid_s = uid.decode("ascii", errors="replace")
            vp = f"imap://{folder}/{uid_s}"
        return ConnectorDocument(
            source="imap",
            virtual_path=vp,
            title=subject,
            content="\n".join(lines),
            mtime=mtime,
            metadata={
                "from": from_,
                "to": to_,
                "folder": folder,
                "message_id": msg_id,
                "date": date_hdr,
            },
        )


def _td(days: int):
    """timedelta(days=days), inlined to avoid an import for one call."""
    from datetime import timedelta
    return timedelta(days=max(0, int(days)))
