"""Browser history connector — reads Chrome / Edge SQLite history files.

Both Chromium-based browsers store visited URLs + titles in a SQLite file
called `History` under their user-data dir. The DB is locked while the
browser is running, so we copy it to a temp location before reading.

Times are stored as microseconds since the WebKit epoch (1601-01-01 UTC).
We convert to Unix seconds for our index.

No auth needed — just file-system access. Limited to the default profile
in v1; multi-profile support can come later if anyone needs it.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from ..config import Config
from . import ConnectorDocument

log = logging.getLogger(__name__)

# Microseconds between 1601-01-01 (WebKit epoch) and 1970-01-01 (Unix epoch).
_WEBKIT_OFFSET_US = 11_644_473_600_000_000

_MAX_URLS = 5000  # cap so we don't index a lifetime of clutter on first run


def _webkit_to_unix(value: int) -> float:
    if value <= 0:
        return 0.0
    return (value - _WEBKIT_OFFSET_US) / 1_000_000


def _profile_paths() -> list[tuple[str, Path]]:
    """Return [(label, path-to-History-db)] for known browsers on this machine."""
    home = Path.home()
    candidates = [
        ("chrome", home / "AppData/Local/Google/Chrome/User Data/Default/History"),
        ("edge",   home / "AppData/Local/Microsoft/Edge/User Data/Default/History"),
        ("brave",  home / "AppData/Local/BraveSoftware/Brave-Browser/User Data/Default/History"),
        # macOS / Linux paths could be added here later
    ]
    return [(label, p) for label, p in candidates if p.exists()]


class BrowserHistoryConnector:
    name = "browser"

    def is_enabled(self, cfg: Config) -> bool:
        return bool(_profile_paths())

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]:
        for label, src in _profile_paths():
            yield from self._read_profile(label, src)

    # --- helpers ----------------------------------------------------------

    def _read_profile(self, label: str, src: Path) -> Iterator[ConnectorDocument]:
        # Browser holds a write lock on the live DB; copy first.
        with tempfile.NamedTemporaryFile(
            prefix=f"sb-history-{label}-", suffix=".db", delete=False
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            try:
                shutil.copy2(src, tmp_path)
            except PermissionError:
                log.warning("could not copy %s history (browser running?)", label)
                return

            try:
                conn = sqlite3.connect(str(tmp_path))
                try:
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(
                        "SELECT url, title, visit_count, last_visit_time "
                        "FROM urls "
                        "WHERE last_visit_time > 0 AND title != '' "
                        "ORDER BY last_visit_time DESC "
                        "LIMIT ?",
                        (_MAX_URLS,),
                    ).fetchall()
                finally:
                    # On Windows the temp file stays locked until the conn is
                    # closed; without try/finally a DatabaseError mid-query
                    # would leak the lock and the unlink in the outer finally
                    # would silently fail.
                    conn.close()
            except sqlite3.DatabaseError as e:
                log.warning("could not read %s history db: %s", label, e)
                return

            for r in rows:
                url = r["url"]
                title = (r["title"] or "").strip()
                if not title or not url:
                    continue
                visits = int(r["visit_count"] or 0)
                mtime = _webkit_to_unix(int(r["last_visit_time"] or 0))
                # Strip query strings + fragments from the indexed path.
                # Otherwise reset-password URLs ("?token=...") and other
                # query-string secrets land verbatim in `files.path`,
                # the search palette, MCP responses, and the dashboard.
                try:
                    parts = urlsplit(url)
                    safe_url = urlunsplit(parts._replace(query="", fragment=""))
                except ValueError:
                    safe_url = url
                content = (
                    f"# {title}\n\n"
                    f"URL: {safe_url}\n"
                    f"Browser: {label}\n"
                    f"Visit count: {visits}\n"
                )
                yield ConnectorDocument(
                    source="browser",
                    virtual_path=f"browser://{label}/{safe_url}",
                    title=title,
                    content=content,
                    mtime=mtime,
                    metadata={"url": safe_url, "visits": visits, "browser": label},
                )
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass
