"""Chat-history connector — makes past chat conversations searchable.

The chat module already persists every conversation to the SQLite index
(``chat_conversations`` + ``chat_messages``). This connector turns each
conversation into one ConnectorDocument so ``secondbrain sync chat_history``
folds it into the search pipeline.

Why bother:
  - The chat agent itself can now find its own past answers via
    ``search_brain``. Ask "what did we conclude about voyage rate limits
    last week?" and the model will pull up the prior conversation.
  - The dashboard's /search page surfaces past conversations alongside
    files and connector docs.
  - The graph + entities + tagging passes now apply to chat content too.

Each conversation becomes one document keyed by ``chat://{conversation_id}``;
re-running sync upserts based on ``updated_at``. The body is the
chronological transcript with role-labelled turns and (for assistant
turns) a short citations footer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

from ..config import Config
from ..db import (
    chat_get_messages,
    chat_list_conversations,
    connect_readonly,
)
from . import ConnectorDocument

log = logging.getLogger(__name__)


def _render_user_content(raw: object) -> str:
    """User messages are stored as JSON-encoded strings (one user turn = one
    string). Defensive: also accept legacy list-of-blocks shape."""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return "\n".join(
            b.get("text", "") for b in raw
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def _render_assistant_content(raw: object) -> str:
    """Assistant content is a list of Anthropic content blocks. We surface
    the text blocks; tool_use/tool_result are skipped because the
    surrounding context already makes their effect visible."""
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, list):
        return ""
    out: list[str] = []
    for b in raw:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            t = b.get("text") or ""
            if t.strip():
                out.append(t)
    return "\n".join(out)


class ChatHistoryConnector:
    """Index every saved chat conversation as one searchable document."""

    name = "chat_history"

    def is_enabled(self, cfg: Config) -> bool:
        # Always enabled - if there are no conversations yet, fetch() just
        # yields nothing. No env vars or external services to configure.
        return True

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]:
        # Open a read-only connection so we don't fight the daemon for the
        # writer lock during a bulk sync. The chat tables live in the same
        # DB as everything else.
        try:
            conn = connect_readonly(cfg.db_path)
        except FileNotFoundError:
            # No index yet → no chats. Nothing to do.
            return
        try:
            for conv in chat_list_conversations(conn, limit=10_000):
                doc = self._render_conversation(conn, conv)
                if doc is not None:
                    yield doc
        finally:
            conn.close()

    # --- helpers --------------------------------------------------------

    def _render_conversation(self, conn, conv) -> ConnectorDocument | None:
        cid = conv["id"]
        title = conv["title"] or f"Chat {cid}"
        msgs = chat_get_messages(conn, cid)
        if not msgs:
            return None

        lines: list[str] = [f"# {title}", ""]
        cite_paths: list[str] = []
        for m in msgs:
            try:
                content = json.loads(m["content_json"])
            except (json.JSONDecodeError, TypeError):
                continue
            role = m["role"]
            if role == "user":
                text = _render_user_content(content).strip()
                if text:
                    lines.append(f"**You:** {text}")
                    lines.append("")
            elif role == "assistant":
                text = _render_assistant_content(content).strip()
                if text:
                    lines.append(f"**Brain:** {text}")
                    lines.append("")
                if m["citations_json"]:
                    try:
                        cites = json.loads(m["citations_json"])
                    except json.JSONDecodeError:
                        cites = []
                    for c in cites or []:
                        p = c.get("file_path", "")
                        if p and p not in cite_paths:
                            cite_paths.append(p)

        if cite_paths:
            lines.append("---")
            lines.append("Sources cited in this conversation:")
            for p in cite_paths:
                lines.append(f"- {p}")

        body = "\n".join(lines).strip()
        if not body:
            return None

        return ConnectorDocument(
            source="chat",
            virtual_path=f"chat://{cid}",
            title=title,
            content=body,
            mtime=float(conv["updated_at"]),
            metadata={
                "conversation_id": cid,
                "message_count": int(conv["n_messages"]),
                "cited_paths": cite_paths,
            },
        )
