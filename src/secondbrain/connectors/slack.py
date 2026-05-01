"""Slack connector — pulls recent messages from channels you're a member of.

Auth: user OAuth token (xoxp-...) in ``SLACK_USER_TOKEN``. Setup:
  1. Go to https://api.slack.com/apps → Create New App → "From scratch".
  2. Add it to your workspace.
  3. Under "OAuth & Permissions" → User Token Scopes, add:
       channels:history, channels:read,
       groups:history, groups:read,
       im:history, im:read,
       mpim:history, mpim:read,
       users:read.
  4. Install to workspace; copy the User OAuth Token (starts with xoxp-).
  5. ``[Environment]::SetEnvironmentVariable("SLACK_USER_TOKEN", "xoxp-...", "User")``.

What it ingests:
  - Channels you're a member of (public + private)
  - DMs and group DMs you're a participant in
  - Messages from the last ``SB_SLACK_DAYS`` days (default 30)
  - Resolved usernames so the AI can reason about who said what

Each channel becomes one ConnectorDocument with the conversation rendered
as readable markdown — chronological, one block per message, with author
and timestamp.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from datetime import datetime, timezone

import requests

from ..config import Config
from . import ConnectorDocument

log = logging.getLogger(__name__)

_API = "https://slack.com/api"
_DEFAULT_DAYS = 30
_MESSAGES_PER_CHANNEL = 500


class SlackConnector:
    name = "slack"

    def is_enabled(self, cfg: Config) -> bool:
        return bool(os.environ.get("SLACK_USER_TOKEN"))

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]:
        token = os.environ["SLACK_USER_TOKEN"]
        s = requests.Session()
        s.headers.update({
            "Authorization": f"Bearer {token}",
            "User-Agent": "second-brain/0.0.1",
        })
        days = int(os.environ.get("SB_SLACK_DAYS", _DEFAULT_DAYS))
        oldest = time.time() - days * 86400

        try:
            users = self._fetch_user_directory(s)
            for ch in self._iter_conversations(s):
                doc = self._fetch_channel(s, ch, oldest, users)
                if doc is not None:
                    yield doc
        finally:
            s.close()

    # --- helpers ----------------------------------------------------------

    def _slack_get(self, s: requests.Session, method: str, params: dict) -> dict:
        """Call a Slack Web API method, return the parsed body. Logs and
        returns {} on error so the iteration continues."""
        try:
            r = s.get(f"{_API}/{method}", params=params, timeout=30)
            data = r.json()
            if not data.get("ok"):
                log.warning("Slack %s failed: %s", method, data.get("error"))
                return {}
            return data
        except (requests.RequestException, ValueError) as e:
            log.warning("Slack %s exception: %s", method, e)
            return {}

    def _fetch_user_directory(self, s: requests.Session) -> dict[str, str]:
        """Map user_id -> display name. Cached for the duration of a sync."""
        out: dict[str, str] = {}
        cursor: str | None = None
        while True:
            params: dict[str, str | int] = {"limit": 200}
            if cursor:
                params["cursor"] = cursor
            data = self._slack_get(s, "users.list", params)
            if not data:
                return out
            for u in data.get("members") or []:
                uid = u.get("id")
                if not uid:
                    continue
                profile = u.get("profile") or {}
                name = (
                    profile.get("display_name")
                    or profile.get("real_name")
                    or u.get("name")
                    or uid
                )
                out[uid] = name
            cursor = (data.get("response_metadata") or {}).get("next_cursor") or ""
            if not cursor:
                return out

    def _iter_conversations(self, s: requests.Session) -> Iterator[dict]:
        """Yield each conversation the user is a member of."""
        cursor: str | None = None
        while True:
            params: dict[str, str | int] = {
                "types": "public_channel,private_channel,mpim,im",
                "exclude_archived": "true",
                "limit": 100,
            }
            if cursor:
                params["cursor"] = cursor
            data = self._slack_get(s, "users.conversations", params)
            if not data:
                return
            for ch in data.get("channels") or []:
                yield ch
            cursor = (data.get("response_metadata") or {}).get("next_cursor") or ""
            if not cursor:
                return

    def _fetch_channel(
        self, s: requests.Session, ch: dict, oldest: float, users: dict[str, str]
    ) -> ConnectorDocument | None:
        ch_id = ch.get("id")
        if not ch_id:
            return None
        is_im = bool(ch.get("is_im"))
        is_mpim = bool(ch.get("is_mpim"))
        if is_im:
            other = users.get(ch.get("user", ""), "?")
            ch_name = f"DM with {other}"
        elif is_mpim:
            ch_name = ch.get("name", "(group DM)")
        else:
            ch_name = "#" + (ch.get("name") or "(unnamed)")

        messages = self._fetch_messages(s, ch_id, oldest)
        if not messages:
            return None

        lines: list[str] = [f"# {ch_name}", "", f"Channel ID: {ch_id}", ""]
        latest_ts = oldest
        # Render oldest -> newest so context reads naturally.
        for msg in reversed(messages):
            ts_str = msg.get("ts") or "0"
            try:
                ts_num = float(ts_str)
            except ValueError:
                ts_num = 0.0
            latest_ts = max(latest_ts, ts_num)
            uid = msg.get("user") or msg.get("bot_id") or "?"
            user = users.get(uid, uid)
            text = msg.get("text") or ""
            # Replace user mentions <@U123> with names where we can
            for m_uid, m_name in users.items():
                text = text.replace(f"<@{m_uid}>", f"@{m_name}")
            when = datetime.fromtimestamp(ts_num, tz=timezone.utc).strftime("%Y-%m-%d %H:%M") if ts_num else ""
            lines.append(f"**{user}** — {when}")
            lines.append(text)
            lines.append("")

        return ConnectorDocument(
            source="slack",
            virtual_path=f"slack://channel/{ch_id}",
            title=ch_name,
            content="\n".join(lines),
            mtime=latest_ts or time.time(),
            metadata={
                "channel_id": ch_id,
                "channel_name": ch_name,
                "is_im": is_im, "is_mpim": is_mpim,
                "message_count": len(messages),
            },
        )

    def _fetch_messages(
        self, s: requests.Session, channel: str, oldest: float
    ) -> list[dict]:
        out: list[dict] = []
        cursor: str | None = None
        while len(out) < _MESSAGES_PER_CHANNEL:
            params: dict[str, str | int | float] = {
                "channel": channel,
                "limit": min(200, _MESSAGES_PER_CHANNEL - len(out)),
                "oldest": f"{oldest:.6f}",
            }
            if cursor:
                params["cursor"] = cursor
            data = self._slack_get(s, "conversations.history", params)
            if not data:
                return out
            out.extend(data.get("messages") or [])
            if not data.get("has_more"):
                return out
            cursor = (data.get("response_metadata") or {}).get("next_cursor") or ""
            if not cursor:
                return out
        return out
