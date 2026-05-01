"""X (Twitter) archive connector — parses the zip you download from X.

X's API is paywalled at $100+/mo and the read-only tier doesn't cover the
endpoints we'd need (your own tweets, bookmarks, likes). The pragmatic answer:
they let you download a full archive of your account from
Settings → Your account → Download an archive of your data.

Setup:
  1. Request your archive on X. They email a download link in 24-48h.
  2. Unzip it somewhere stable, e.g. ``~/twitter-archive/``.
  3. ``[Environment]::SetEnvironmentVariable("X_ARCHIVE_PATH", "C:\\path\\to\\twitter-archive", "User")``.

What it ingests:
  - Tweets you've posted (data/tweets.js)
  - Tweets you've bookmarked (data/bookmark.js or data/bookmarks.js)
  - Tweets you've liked (data/like.js or data/likes.js)
  - Direct messages (data/direct-messages.js) — optional, off by default
    because DMs are sensitive; opt in with ``SB_X_INCLUDE_DMS=1``.

The archive files are JS-wrapped JSON. They look like:
    window.YTD.tweets.part0 = [ { "tweet": {...} }, ... ]
We strip the prefix to get plain JSON. Re-running on a refreshed archive
just updates entries by their stable tweet id — virtual_paths collide and
``index_text`` upserts.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path

from ..config import Config
from . import ConnectorDocument

log = logging.getLogger(__name__)

# The wrapper looks like: `window.YTD.<name>.part<N> = `
_PREFIX_RE = re.compile(r"^\s*window\.YTD\.[A-Za-z0-9_]+\.part\d+\s*=\s*", re.DOTALL)


def _twitter_ts_to_epoch(s: str | None) -> float:
    """Parse Twitter's archive date format: 'Wed Oct 10 20:19:24 +0000 2018'."""
    if not s:
        return 0.0
    try:
        return datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y").timestamp()
    except ValueError:
        return 0.0


def _load_js_array(path: Path) -> list[dict]:
    """Load one of the archive's `.js` files and return the wrapped array."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("Cannot read %s: %s", path, e)
        return []
    body = _PREFIX_RE.sub("", raw, count=1)
    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        log.warning("Cannot parse %s as JSON: %s", path, e)
        return []
    if not isinstance(data, list):
        return []
    return data


def _first_existing(base: Path, names: Iterable[str]) -> Path | None:
    for n in names:
        p = base / n
        if p.exists():
            return p
    return None


class XArchiveConnector:
    name = "x_archive"

    def is_enabled(self, cfg: Config) -> bool:
        path = os.environ.get("X_ARCHIVE_PATH")
        if not path:
            return False
        return Path(path).expanduser().is_dir()

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]:
        root = Path(os.environ["X_ARCHIVE_PATH"]).expanduser()
        # Some archives put files directly under root, others under ``data/``.
        data_dir = root / "data"
        base = data_dir if data_dir.is_dir() else root

        # Tweets
        tweets_path = _first_existing(base, ("tweets.js", "tweet.js"))
        if tweets_path:
            yield from self._iter_tweets(tweets_path)

        # Bookmarks
        bookmarks_path = _first_existing(base, ("bookmark.js", "bookmarks.js"))
        if bookmarks_path:
            yield from self._iter_bookmarks(bookmarks_path)

        # Likes
        likes_path = _first_existing(base, ("like.js", "likes.js"))
        if likes_path:
            yield from self._iter_likes(likes_path)

        # DMs (opt-in)
        if os.environ.get("SB_X_INCLUDE_DMS") == "1":
            dm_path = _first_existing(
                base, ("direct-messages.js", "direct_messages.js")
            )
            if dm_path:
                yield from self._iter_dms(dm_path)

    # --- iterators --------------------------------------------------------

    def _iter_tweets(self, path: Path) -> Iterator[ConnectorDocument]:
        for entry in _load_js_array(path):
            t = entry.get("tweet") if isinstance(entry, dict) else None
            if not t:
                continue
            tid = t.get("id_str") or t.get("id")
            if not tid:
                continue
            text = t.get("full_text") or t.get("text") or ""
            created = _twitter_ts_to_epoch(t.get("created_at"))
            faves = t.get("favorite_count") or 0
            rts = t.get("retweet_count") or 0
            in_reply = t.get("in_reply_to_screen_name") or ""
            lines = [
                "# Tweet" + (f" → @{in_reply}" if in_reply else ""),
                "",
                text,
                "",
                f"Posted: {datetime.fromtimestamp(created).isoformat() if created else '?'}",
                f"Likes: {faves}  Retweets: {rts}",
            ]
            yield ConnectorDocument(
                source="x_archive",
                virtual_path=f"x://tweet/{tid}",
                title=(text[:80] + "…") if len(text) > 80 else (text or f"Tweet {tid}"),
                content="\n".join(lines),
                mtime=created or 0.0,
                metadata={
                    "kind": "own_tweet",
                    "favorites": faves,
                    "retweets": rts,
                    "in_reply_to": in_reply,
                    "url": f"https://x.com/i/web/status/{tid}",
                },
            )

    def _iter_bookmarks(self, path: Path) -> Iterator[ConnectorDocument]:
        for entry in _load_js_array(path):
            b = entry.get("tweet") if isinstance(entry, dict) else None
            # Bookmark records sometimes nest under "bookmark" instead.
            if not b and isinstance(entry, dict):
                b = entry.get("bookmark")
            if not b:
                continue
            tid = b.get("tweetId") or b.get("id_str") or b.get("id")
            if not tid:
                continue
            text = b.get("fullText") or b.get("full_text") or b.get("text") or ""
            yield ConnectorDocument(
                source="x_archive",
                virtual_path=f"x://bookmark/{tid}",
                title=(text[:80] + "…") if len(text) > 80 else (text or f"Bookmark {tid}"),
                content=f"# Bookmarked tweet\n\n{text}\n\nhttps://x.com/i/web/status/{tid}",
                mtime=0.0,  # archive often lacks bookmark timestamps
                metadata={"kind": "bookmark", "url": f"https://x.com/i/web/status/{tid}"},
            )

    def _iter_likes(self, path: Path) -> Iterator[ConnectorDocument]:
        for entry in _load_js_array(path):
            like = entry.get("like") if isinstance(entry, dict) else None
            if not like:
                continue
            tid = like.get("tweetId")
            if not tid:
                continue
            text = like.get("fullText") or ""
            yield ConnectorDocument(
                source="x_archive",
                virtual_path=f"x://like/{tid}",
                title=(text[:80] + "…") if len(text) > 80 else (text or f"Liked {tid}"),
                content=f"# Liked tweet\n\n{text}\n\nhttps://x.com/i/web/status/{tid}",
                mtime=0.0,
                metadata={"kind": "like", "url": f"https://x.com/i/web/status/{tid}"},
            )

    def _iter_dms(self, path: Path) -> Iterator[ConnectorDocument]:
        # DMs are nested: list of {"dmConversation": {"conversationId": ..., "messages": [...]}}
        convs = _load_js_array(path)
        for entry in convs:
            conv = entry.get("dmConversation") if isinstance(entry, dict) else None
            if not conv:
                continue
            cid = conv.get("conversationId") or "?"
            messages = conv.get("messages") or []
            if not messages:
                continue
            lines = [f"# DM conversation {cid}", ""]
            latest = 0.0
            for m in messages:
                msg = m.get("messageCreate") if isinstance(m, dict) else None
                if not msg:
                    continue
                sender = msg.get("senderId") or "?"
                created = msg.get("createdAt") or ""
                text = msg.get("text") or ""
                # createdAt is ISO-8601 here, not the Twitter format.
                try:
                    ts = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
                except ValueError:
                    ts = 0.0
                latest = max(latest, ts)
                lines.append(f"**{sender}** — {created}")
                lines.append(text)
                lines.append("")
            yield ConnectorDocument(
                source="x_archive",
                virtual_path=f"x://dm/{cid}",
                title=f"DM conversation {cid}",
                content="\n".join(lines),
                mtime=latest,
                metadata={"kind": "dm", "message_count": len(messages)},
            )
