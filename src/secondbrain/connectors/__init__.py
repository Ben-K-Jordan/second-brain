"""Connectors pull data from cloud sources (GitHub, Notion, Calendar, etc.)
into the same pipeline used for filesystem ingest.

Each connector implements a small protocol: name, is_enabled(cfg), and
fetch(cfg) yielding ConnectorDocument objects. The CLI `secondbrain sync`
command runs them through ``index_text`` so the resulting docs become
searchable, entity-extracted, and dedup'd alongside everything else.

Connectors are opt-in via env vars (GITHUB_TOKEN, NOTION_TOKEN, etc.) so
adding a new source is just exporting a token. No OAuth dance required for
these — Gmail / Google Drive will live in a separate module that handles
the OAuth flow.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .. import __version__ as _SECONDBRAIN_VERSION
from ..config import Config

# Shared HTTP defaults so every connector is consistent. Override per-call
# only when there's a clear reason (large file downloads, GraphQL endpoints
# that aggregate a lot of data, etc).
DEFAULT_TIMEOUT = 30
USER_AGENT = f"second-brain/{_SECONDBRAIN_VERSION}"

_log = logging.getLogger(__name__)


def respect_retry_after(response, max_wait: float = 60.0) -> bool:
    """If ``response`` is a 429, sleep up to ``max_wait`` seconds and return True.

    Caller should re-issue the same request after we return True. Honors the
    ``Retry-After`` header when present (Google, GitHub, Pocket, Reddit all
    set it). Without this, every connector treats a 429 identically to a 5xx
    and silently truncates the sync.
    """
    try:
        if response.status_code != 429:
            return False
    except AttributeError:
        return False
    raw = response.headers.get("Retry-After", "5") if hasattr(response, "headers") else "5"
    try:
        wait = float(raw)
    except (TypeError, ValueError):
        wait = 5.0
    wait = max(0.5, min(max_wait, wait))
    _log.info("rate limited (429); sleeping %.1fs before retry", wait)
    time.sleep(wait)
    return True


@dataclass
class ConnectorDocument:
    """A single document fetched from a connector source.

    The ``virtual_path`` becomes the file's path in the index — must be
    globally unique. Convention: ``<source>://<stable-identifier>``, e.g.
    ``github://owner/repo/issues/42`` or ``notion://<page-uuid>``.
    """

    source: str
    virtual_path: str
    title: str
    content: str
    mtime: float
    kind: str = "url"
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class Connector(Protocol):
    name: str

    def is_enabled(self, cfg: Config) -> bool: ...

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]: ...


def all_connectors() -> list[type[Connector]]:
    """Registry of connector classes. Order = run order for `sync all`.

    Cheap/local connectors first (browser is just a SQLite read), then
    network-bound ones. Gmail + Drive last because they're the slowest
    on first sync.
    """
    from .browser import BrowserHistoryConnector
    from .calendar import CalendarConnector
    from .github import GitHubConnector
    from .gmail import GmailConnector
    from .google_calendar import GoogleCalendarConnector
    from .google_drive import GoogleDriveConnector
    from .hacker_news import HackerNewsConnector
    from .linear import LinearConnector
    from .notion import NotionConnector
    from .pocket import PocketConnector
    from .reddit import RedditConnector
    from .slack import SlackConnector
    from .x_archive import XArchiveConnector

    return [
        BrowserHistoryConnector,
        XArchiveConnector,        # local zip read; fast
        GitHubConnector,
        LinearConnector,
        NotionConnector,
        SlackConnector,
        RedditConnector,
        HackerNewsConnector,
        PocketConnector,
        CalendarConnector,
        GoogleCalendarConnector,
        GmailConnector,
        GoogleDriveConnector,
    ]


def get_connector(name: str) -> type[Connector] | None:
    """Look up a connector class by its `.name` attribute."""
    for cls in all_connectors():
        if cls().name == name:
            return cls
    return None
