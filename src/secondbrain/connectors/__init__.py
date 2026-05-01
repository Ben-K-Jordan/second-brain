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

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..config import Config


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
    """Registry of connector classes. Order = run order for `sync all`."""
    from .browser import BrowserHistoryConnector
    from .calendar import CalendarConnector
    from .github import GitHubConnector
    from .notion import NotionConnector

    return [
        GitHubConnector,
        NotionConnector,
        BrowserHistoryConnector,
        CalendarConnector,
    ]


def get_connector(name: str) -> type[Connector] | None:
    """Look up a connector class by its `.name` attribute."""
    for cls in all_connectors():
        if cls().name == name:
            return cls
    return None
