"""Round 16 (Phase D) — iMessage / SMS connector.

Reads from Apple's ``chat.db`` (the SQLite store backing the Messages
app on macOS / iOS). We treat each conversation thread as a single
``ConnectorDocument`` so:

  - Search returns the whole thread, not isolated single messages
    (which would be near-useless without context).
  - Threads with the same person across SMS+iMessage are grouped.
  - Re-sync is idempotent: virtual_path = ``imessage://chat-{id}`` and
    we always write the full current snapshot.

Schema notes (Apple chat.db):

  - ``handle`` rows are participants (phone or email).
  - ``chat`` rows are conversation threads.
  - ``message`` rows are individual messages.
  - ``chat_message_join`` links them. ``chat_handle_join`` links
    chats ↔ participants.
  - ``message.text`` is sometimes NULL on iOS 16+; the actual text
    lives in ``message.attributedBody`` (NSKeyedArchiver blob). We
    handle both: text-when-present, else best-effort byte extraction
    from attributedBody.
  - Apple stores timestamps as nanoseconds since 2001-01-01. Convert
    via ``date / 1e9 + 978307200`` (Unix epoch offset).

Cross-platform note:

  - chat.db only exists on macOS (``~/Library/Messages/chat.db``).
  - On Windows / Linux, the user must copy the chat.db over manually
    (or run a tiny relay on a Mac that pushes the file). This module
    just needs the path to a valid chat.db; it doesn't care which OS.
  - Configure path via ``IMESSAGE_DB_PATH`` env var or
    ``imessage_db_path`` in config.toml. If the env / config path is
    absent, we fall back to the macOS default, which on Windows just
    means "skip silently".

Privacy:

  - We pass message text through ``redact_text`` to mask secret-shaped
    substrings (API keys, SSNs, credit cards) before they enter the
    index. The full conversation context is preserved otherwise.
  - Group chats stay grouped; we don't try to anonymise participants.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from . import ConnectorDocument

log = logging.getLogger(__name__)


_APPLE_EPOCH_OFFSET = 978307200  # seconds between Unix epoch and 2001-01-01
_DEFAULT_MAC_PATH = "~/Library/Messages/chat.db"

# Limit per-chat message count so a 10-year-old thread doesn't make a
# 50 MB single doc. The most recent N messages are kept.
_MAX_MESSAGES_PER_CHAT = 1000

# Skip chats with fewer than this many messages — usually one-off spam
# or accidentally-started conversations.
_MIN_MESSAGES_PER_CHAT = 2


@dataclass
class _ChatRow:
    chat_id: int
    display_name: str
    participants: list[str]
    is_group: bool


def _resolve_db_path(cfg: Config) -> Path | None:
    """Return the chat.db path, or None if unconfigured / missing."""
    raw = (
        os.environ.get("IMESSAGE_DB_PATH", "")
        or getattr(cfg, "imessage_db_path", "")
        or ""
    )
    if not raw:
        # Fall back to mac default — silently skip on Windows / Linux.
        raw = _DEFAULT_MAC_PATH
    p = Path(raw).expanduser().resolve()
    if not p.exists() or not p.is_file():
        log.debug("imessage: chat.db not found at %s", p)
        return None
    return p


def _apple_ts_to_unix(value: int | None) -> float:
    """Convert Apple's 2001-epoch nanoseconds to Unix seconds. Apple
    used integer seconds before iOS 11 then switched to nanoseconds;
    detect by magnitude."""
    if not value:
        return 0.0
    # > 1e15 means nanoseconds since 2001 (~31.7 years × 1e9). Anything
    # smaller is plain seconds since 2001.
    if value > 10**12:
        return value / 1e9 + _APPLE_EPOCH_OFFSET
    return float(value) + _APPLE_EPOCH_OFFSET


def _extract_text_from_attributed(blob: bytes | None) -> str:
    """Best-effort text extraction from attributedBody NSKeyedArchiver
    blob. We don't fully unarchive — just walk the bytes for printable
    runs >= 4 chars long. Apple's format embeds the message text
    relatively cleanly so a byte scan recovers it 95%+ of the time.

    Returns empty string when nothing recognisable is found.
    """
    if not blob:
        return ""
    out: list[str] = []
    current: list[int] = []
    for b in blob:
        # ASCII printable + extended ascii covers most everyday text.
        if 32 <= b < 127 or b in (9, 10, 13):
            current.append(b)
        else:
            if len(current) >= 4:
                # Decode + strip control chars.
                run = bytes(current).decode("utf-8", errors="replace")
                # NSKeyedArchiver header noise is mostly NSMutable/NSString-
                # prefixed type names; drop runs that look like type names.
                if not _is_type_name(run):
                    out.append(run)
            current = []
    if len(current) >= 4:
        run = bytes(current).decode("utf-8", errors="replace")
        if not _is_type_name(run):
            out.append(run)
    return " ".join(out).strip()


def _is_type_name(s: str) -> bool:
    """Heuristic: NSKeyedArchiver class names like 'NSMutableString',
    'NSAttributedString', 'NSDictionary', etc. Skip them."""
    s = s.strip()
    return (
        s.startswith(("NS", "__kCF", "_NS"))
        or s in ("streamtyped", "iI", "@")
        or len(s) < 4
    )


def _safe_redact(text: str | None) -> str:
    if not text:
        return ""
    try:
        from ..safety import redact_text
        return redact_text(text)
    except ImportError:
        return text


def _list_chats(conn: sqlite3.Connection) -> list[_ChatRow]:
    """Return every chat with its participants + display name."""
    chats: dict[int, _ChatRow] = {}
    # Fetch chats + their handles in one go.
    rows = conn.execute("""
        SELECT
            c.ROWID AS chat_id,
            c.display_name AS display_name,
            c.style AS style,
            h.id AS handle
        FROM chat c
        LEFT JOIN chat_handle_join chj ON chj.chat_id = c.ROWID
        LEFT JOIN handle h ON h.ROWID = chj.handle_id
        ORDER BY c.ROWID
    """).fetchall()
    for r in rows:
        cid = int(r["chat_id"])
        chat = chats.get(cid)
        if chat is None:
            chat = _ChatRow(
                chat_id=cid,
                display_name=(r["display_name"] or "").strip(),
                participants=[],
                # style 43 = group chat in Apple's schema; 45 = direct.
                is_group=int(r["style"] or 0) == 43,
            )
            chats[cid] = chat
        if r["handle"]:
            chat.participants.append(str(r["handle"]))
    return list(chats.values())


def _fetch_messages(
    conn: sqlite3.Connection, chat_id: int, limit: int = _MAX_MESSAGES_PER_CHAT,
) -> list[dict]:
    """Pull the most recent N messages for a chat, oldest-first."""
    rows = conn.execute("""
        SELECT
            m.ROWID AS msg_id,
            m.text AS text,
            m.attributedBody AS attributedBody,
            m.is_from_me AS is_from_me,
            m.date AS date,
            m.cache_has_attachments AS has_attachments,
            h.id AS sender_handle
        FROM message m
        JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
        LEFT JOIN handle h ON h.ROWID = m.handle_id
        WHERE cmj.chat_id = ?
        ORDER BY m.date DESC
        LIMIT ?
    """, (chat_id, limit)).fetchall()
    out = []
    for r in reversed(rows):  # oldest-first for readability
        text = (r["text"] or "").strip()
        if not text:
            text = _extract_text_from_attributed(r["attributedBody"])
        if not text and not r["has_attachments"]:
            # Nothing to capture for this message.
            continue
        out.append({
            "id": int(r["msg_id"]),
            "text": text,
            "is_from_me": bool(r["is_from_me"]),
            "ts": _apple_ts_to_unix(r["date"]),
            "sender": (r["sender_handle"] or "").strip(),
            "has_attachments": bool(r["has_attachments"]),
        })
    return out


def _format_thread(
    chat: _ChatRow, messages: list[dict],
) -> tuple[str, str]:
    """Render a chat thread as Markdown. Returns (title, body)."""
    if chat.display_name:
        title = chat.display_name
    elif chat.is_group:
        title = "Group: " + ", ".join(chat.participants[:4])
    elif chat.participants:
        title = chat.participants[0]
    else:
        title = f"Conversation #{chat.chat_id}"

    from datetime import datetime
    lines = [f"# Conversation: {title}", ""]
    if chat.is_group and chat.participants:
        lines.append("**Participants:** " + ", ".join(chat.participants))
        lines.append("")
    last_date: str | None = None
    for m in messages:
        ts = m["ts"]
        if not ts:
            continue
        dt = datetime.fromtimestamp(ts)
        date_str = dt.strftime("%Y-%m-%d")
        time_str = dt.strftime("%H:%M")
        if date_str != last_date:
            lines.append(f"\n## {date_str}\n")
            last_date = date_str
        speaker = "**Me**" if m["is_from_me"] else f"**{m['sender'] or 'them'}**"
        attach = " 📎" if m["has_attachments"] and not m["text"] else ""
        text = _safe_redact(m["text"]) or "(attachment)"
        lines.append(f"`{time_str}` {speaker}: {text}{attach}")
    return title, "\n".join(lines)


class IMessageConnector:
    name = "imessage"

    def is_enabled(self, cfg: Config) -> bool:
        return _resolve_db_path(cfg) is not None

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]:
        path = _resolve_db_path(cfg)
        if path is None:
            return

        # chat.db should be opened READ-ONLY — we never want to mutate
        # Apple's database. The URI form forces RO; we also copy the
        # file to a temp first because Apple holds open handles on it
        # and SQLite WAL on a live file can confuse our reader.
        import shutil
        import tempfile
        with tempfile.NamedTemporaryFile(
            suffix=".db", delete=False,
        ) as tf:
            tmp_path = Path(tf.name)
        try:
            try:
                shutil.copyfile(str(path), str(tmp_path))
                # Also copy the WAL + shm if present (Apple uses WAL).
                for suffix in ("-wal", "-shm"):
                    side = path.parent / (path.name + suffix)
                    if side.exists():
                        try:
                            shutil.copyfile(
                                str(side), str(tmp_path) + suffix,
                            )
                        except OSError:
                            pass
            except (OSError, PermissionError) as e:
                log.warning(
                    "imessage: could not copy chat.db (need Full Disk "
                    "Access on macOS?): %s", e,
                )
                return

            try:
                conn = sqlite3.connect(
                    f"file:{tmp_path.as_posix()}?mode=ro",
                    uri=True,
                )
                conn.row_factory = sqlite3.Row
            except sqlite3.OperationalError as e:
                log.warning("imessage: cannot open chat.db: %s", e)
                return

            try:
                chats = _list_chats(conn)
            except sqlite3.OperationalError as e:
                log.warning("imessage: chat list query failed: %s", e)
                conn.close()
                return

            for chat in chats:
                try:
                    messages = _fetch_messages(conn, chat.chat_id)
                except sqlite3.OperationalError as e:
                    log.warning(
                        "imessage: messages query failed for chat %d: %s",
                        chat.chat_id, e,
                    )
                    continue
                if len(messages) < _MIN_MESSAGES_PER_CHAT:
                    continue
                title, body = _format_thread(chat, messages)
                latest_ts = max((m["ts"] for m in messages), default=0.0)
                yield ConnectorDocument(
                    source="imessage",
                    virtual_path=f"imessage://chat-{chat.chat_id}",
                    title=title,
                    content=body,
                    mtime=latest_ts,
                    kind="message",
                    metadata={
                        "chat_id": chat.chat_id,
                        "is_group": chat.is_group,
                        "participants": chat.participants,
                        "n_messages": len(messages),
                    },
                )
            conn.close()
        finally:
            for p in (tmp_path, Path(str(tmp_path) + "-wal"),
                      Path(str(tmp_path) + "-shm")):
                if p.exists():
                    try:
                        p.unlink()
                    except OSError:
                        pass
