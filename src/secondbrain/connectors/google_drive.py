"""Google Drive connector — pulls Docs / Sheets / Slides as searchable text.

Reuses the shared Google OAuth scaffold. Adds the drive.readonly scope on
top of Gmail/Calendar so a single ``secondbrain auth google`` flow covers
everything.

What it ingests:
  - Google Docs    -> exported as text/plain
  - Google Sheets  -> exported as text/csv (each sheet becomes one doc)
  - Google Slides  -> exported as text/plain
  - Plain text / Markdown stored in Drive -> downloaded as-is

Skips: PDFs (those are usually downloads of files you have locally; index
those via the filesystem), images (CLIP path handles those), and binaries.

Defaults: 500-file cap to keep first runs bounded. Tunable via
``SB_DRIVE_MAX``. Also skips files in Trash automatically.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Iterator
from datetime import datetime

import requests

from ..config import Config
from . import ConnectorDocument
from ._google_oauth import authorized_session, is_authorized

log = logging.getLogger(__name__)

GOOGLE_DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

_API = "https://www.googleapis.com/drive/v3"
_DEFAULT_MAX = 500

# Drive mime types we know how to handle, mapped to their export format.
# Plain text / markdown stored in Drive (mime not under google-apps.*) is
# downloaded directly via files/{id}?alt=media.
_EXPORT_FORMAT = {
    "application/vnd.google-apps.document":     "text/plain",
    "application/vnd.google-apps.spreadsheet":  "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}

_DIRECT_DOWNLOAD_PREFIXES = ("text/", "application/json", "application/xml")


def _iso_to_ts(s: str | None) -> float:
    if not s:
        return time.time()
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return time.time()


class GoogleDriveConnector:
    name = "google_drive"

    def is_enabled(self, cfg: Config) -> bool:
        return is_authorized(cfg, GOOGLE_DRIVE_SCOPES)

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]:
        s = authorized_session(cfg, GOOGLE_DRIVE_SCOPES)
        if s is None:
            log.warning("Drive: no credentials. Run `secondbrain auth google`.")
            return

        cap = int(os.environ.get("SB_DRIVE_MAX", _DEFAULT_MAX))
        try:
            for f in self._iter_files(s, cap):
                doc = self._fetch_file(s, f)
                if doc is not None:
                    yield doc
        finally:
            s.close()

    # --- helpers ----------------------------------------------------------

    def _iter_files(self, s: requests.Session, cap: int) -> Iterator[dict]:
        # Drive query: skip trash, limit to types we can extract text from.
        # Order by recently-modified for the most useful subset on first run.
        types = list(_EXPORT_FORMAT) + [
            "text/plain", "text/markdown", "application/json",
        ]
        type_clause = " or ".join(f"mimeType = '{t}'" for t in types)
        q = f"trashed = false and ({type_clause})"

        page_token: str | None = None
        emitted = 0
        while emitted < cap:
            params: dict[str, str | int] = {
                "q": q,
                "pageSize": min(100, cap - emitted),
                "fields": (
                    "nextPageToken,"
                    "files(id,name,mimeType,modifiedTime,webViewLink,size,owners)"
                ),
                "orderBy": "modifiedTime desc",
            }
            if page_token:
                params["pageToken"] = page_token
            r = s.get(f"{_API}/files", params=params, timeout=30)
            if r.status_code != 200:
                log.warning("Drive list failed: %s %s", r.status_code, r.text[:200])
                return
            data = r.json()
            for f in data.get("files") or []:
                yield f
                emitted += 1
                if emitted >= cap:
                    return
            page_token = data.get("nextPageToken")
            if not page_token:
                return

    def _fetch_file(self, s: requests.Session, f: dict) -> ConnectorDocument | None:
        file_id = f["id"]
        name = f.get("name", "(untitled)")
        mime = f.get("mimeType", "")
        modified = f.get("modifiedTime", "")

        text: str | None = None
        if mime in _EXPORT_FORMAT:
            export_mime = _EXPORT_FORMAT[mime]
            r = s.get(
                f"{_API}/files/{file_id}/export",
                params={"mimeType": export_mime}, timeout=60,
            )
            if r.status_code != 200:
                log.warning("Drive export %s failed: %s", name, r.status_code)
                return None
            text = r.text
        elif any(mime.startswith(p) for p in _DIRECT_DOWNLOAD_PREFIXES):
            r = s.get(
                f"{_API}/files/{file_id}",
                params={"alt": "media"}, timeout=60,
            )
            if r.status_code != 200:
                return None
            try:
                text = r.content.decode("utf-8", errors="replace")
            except Exception:
                return None
        else:
            return None

        if not text or not text.strip():
            return None

        web_link = f.get("webViewLink") or f"https://drive.google.com/file/d/{file_id}"
        owners = ", ".join((o.get("displayName") or "") for o in (f.get("owners") or []))

        header_lines = [f"# {name}", "", f"Drive ID: {file_id}", f"Type: {mime}"]
        if owners:
            header_lines.append(f"Owner(s): {owners}")
        if modified:
            header_lines.append(f"Modified: {modified}")
        body = "\n".join(header_lines) + "\n\n" + text

        return ConnectorDocument(
            source="google_drive",
            virtual_path=f"google_drive://{file_id}",
            title=name,
            content=body,
            mtime=_iso_to_ts(modified),
            metadata={
                "drive_id": file_id, "mime": mime,
                "web_link": web_link, "owners": owners,
            },
        )
