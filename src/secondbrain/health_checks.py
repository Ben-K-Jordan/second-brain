"""Round 10 (#9 + #1) — health checks for fragile external integrations.

Many of the second-brain's flagship features depend on third-party
state that can silently break:

  - Google Calendar OAuth refresh token (Google rotates / revokes
    these for unverified personal apps; refresh failures take down
    meeting prep, meeting thanks, today's events in the brief).
  - IMAP authentication (app passwords get rotated, sessions get
    invalidated).
  - Voyage / Anthropic API keys (the user can revoke them).
  - The local LLM (Ollama isn't running).
  - Photo capture folder existence.
  - Watched folders existing.

This module runs cheap pings against each integration + persists
the result. The daily brief surfaces "X has been broken for Y days"
when something's been failing. The new ``secondbrain doctor`` CLI
runs the same checks on demand for ad-hoc debugging.

Checks intentionally don't block on failure — a slow / down third
party shouldn't take down the daemon scheduler.
"""

from __future__ import annotations

import logging
import sqlite3
import time
import weakref as _weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Config

log = logging.getLogger(__name__)


_SCHEMA_INITIALIZED: _weakref.WeakSet = _weakref.WeakSet()

# How often the daemon retries each check.
_CHECK_INTERVAL_SECONDS = 6 * 3600    # every 6 hours
# Threshold for "this has been broken a while" — surface in brief.
_STALE_AFTER_SECONDS = 24 * 3600       # 24h


def _ensure_schema(conn: sqlite3.Connection) -> None:
    try:
        if conn in _SCHEMA_INITIALIZED:
            return
    except TypeError:
        pass
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS health_checks (
            name TEXT PRIMARY KEY,         -- 'google_calendar' | 'imap' | etc
            last_checked_at REAL NOT NULL,
            last_ok_at REAL,               -- null if never succeeded
            ok INTEGER NOT NULL,           -- 0/1 of last result
            error TEXT,
            extra_json TEXT
        );
    """)
    # Round 11 — first_checked_at column for "configured but broken
    # from the start" stale detection. The original is_stale rule
    # silently skipped checks that had never succeeded; this fixes it.
    try:
        conn.execute(
            "ALTER TABLE health_checks "
            "ADD COLUMN first_checked_at REAL",
        )
        conn.commit()
    except sqlite3.OperationalError as e:
        if "duplicate column" not in str(e).lower():
            log.debug(
                "health_checks: ALTER add first_checked_at skipped: %s", e,
            )
    conn.commit()
    try:
        _SCHEMA_INITIALIZED.add(conn)
    except TypeError:
        pass


@dataclass
class HealthStatus:
    name: str
    ok: bool
    last_checked_at: float
    last_ok_at: float | None
    error: str
    extra: dict
    first_checked_at: float | None = None  # round 11

    @property
    def days_since_ok(self) -> int | None:
        """Days since the last success. None when we've never seen
        a success (unconfigured) or when ok=True now."""
        if self.ok or self.last_ok_at is None:
            return None
        return max(0, int((time.time() - self.last_ok_at) // 86400))

    @property
    def is_stale(self) -> bool:
        """True when this check has been failing long enough that the
        brief should nudge the user.

        Round 11 fix (audit-found semantic bug): we used to skip
        ``last_ok_at is None`` entirely under the rationale "never
        configured — don't spam". But that hides the case where the
        user DID configure something (e.g. set ANTHROPIC_API_KEY)
        and it's failing on every check (wrong shape, revoked key) —
        the daily brief silently dropped that nudge.

        New rule: if the check has been running for 24h+ since
        ``first_checked_at`` and never succeeded, treat that as stale
        too. Distinguishes "configured but broken since the start"
        from "ran once five minutes ago, give it time".
        """
        if self.ok:
            return False
        now = time.time()
        if self.last_ok_at is not None:
            return (now - self.last_ok_at) > _STALE_AFTER_SECONDS
        # Never succeeded. Stale if it's been observed (and failing)
        # for at least the threshold.
        if self.first_checked_at is not None:
            return (now - self.first_checked_at) > _STALE_AFTER_SECONDS
        # Old-schema row without first_checked_at — assume not stale
        # (the next check pass will populate the column).
        return False


# ---- Individual check functions -------------------------------------

def check_google_calendar(cfg: Config) -> tuple[bool, str, dict]:
    """Ping the calendar list endpoint with the existing OAuth scaffold.
    Returns ``(ok, error, extra)``."""
    try:
        from .connectors._google_oauth import (
            authorized_session,
            is_authorized,
        )
        from .connectors.google_calendar import GOOGLE_CALENDAR_SCOPES
    except ImportError as e:
        return False, f"google connector unavailable: {e}", {}
    if not is_authorized(cfg, GOOGLE_CALENDAR_SCOPES):
        return False, "not authorized (run: secondbrain auth google)", {}
    try:
        s = authorized_session(cfg, GOOGLE_CALENDAR_SCOPES)
    except Exception as e:  # noqa: BLE001
        return False, f"oauth refresh failed: {e}", {}
    if s is None:
        return False, "oauth session returned None", {}
    try:
        r = s.get(
            "https://www.googleapis.com/calendar/v3/users/me/calendarList",
            timeout=10,
        )
    except Exception as e:  # noqa: BLE001
        return False, f"calendar API request failed: {e}", {}
    finally:
        try:
            s.close()
        except Exception:  # noqa: BLE001
            pass
    if r.status_code != 200:
        return False, f"calendar API HTTP {r.status_code}", {
            "status_code": r.status_code,
        }
    try:
        n_cals = len(r.json().get("items") or [])
    except Exception:  # noqa: BLE001
        n_cals = 0
    return True, "", {"n_calendars": n_cals}


def _key_fingerprint(key: str) -> str:
    """Round 14 (audit-found gap M2) — return a non-reversible
    fingerprint of an API key for the health-check ``extras`` dict.

    The previous version stored ``key[:12]`` (Anthropic) / ``key[:8]``
    (Voyage). Those substrings are mostly the well-known prefix
    (``sk-ant-api01-``, ``pa-``) but include a few real bytes of
    secret entropy that ended up rendered to the dashboard ``/health``
    page and so could leak via screenshots or shoulder-surfing.

    We hash with SHA-256 and keep the first 8 hex chars — enough to
    distinguish "did the key change?" across health checks without
    revealing any of the underlying key material.
    """
    import hashlib
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


def check_anthropic_key(cfg: Config) -> tuple[bool, str, dict]:
    """Verify the Anthropic key is set + parseable. Doesn't make a
    paid call — just shape-checks. Saves $0.0001/check vs round-trip."""
    import os
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return False, "ANTHROPIC_API_KEY not set", {}
    if not key.startswith("sk-ant-"):
        return False, "key shape unexpected (does not start with sk-ant-)", {}
    return True, "", {"key_fingerprint": _key_fingerprint(key)}


def check_voyage_key(cfg: Config) -> tuple[bool, str, dict]:
    """Same shape-check for Voyage. The embedder will fail loudly
    if the key is wrong; this catches "key was never set" cleanly."""
    import os
    key = os.environ.get("VOYAGE_API_KEY", "") or (
        getattr(cfg, "voyage_api_key", "") or ""
    )
    if not key:
        return False, "VOYAGE_API_KEY not set", {}
    if not key.startswith("pa-"):
        return False, "key shape unexpected (does not start with pa-)", {}
    return True, "", {"key_fingerprint": _key_fingerprint(key)}


def check_local_llm(cfg: Config) -> tuple[bool, str, dict]:
    """Ping the configured Ollama host. Fast — single HTTP GET to
    /api/tags with a 0.5s timeout."""
    try:
        from . import local_llm
    except ImportError as e:
        return False, f"local_llm import failed: {e}", {}
    if not local_llm.is_available(cfg):
        host = getattr(cfg, "local_llm_host", "http://localhost:11434")
        return False, f"Ollama not reachable at {host}", {}
    models = local_llm.list_models(cfg)
    return True, "", {"models": models[:10]}


def check_imap(
    cfg: Config, *, network: bool = False,
) -> tuple[bool, str, dict]:
    """Round 11 — IMAP credential check. Shape-only by default
    (verifies host / user / password env-var are all set) so the
    hourly daemon doesn't write a login attempt to the IMAP
    server's auth logs every cycle.

    ``network=True`` does the real login; used by ``secondbrain
    doctor`` for ad-hoc triage. The hourly daemon job uses the
    cheap shape-check to detect "user changed app password and
    forgot to update env var" without spamming the server.
    """
    import os
    host = (getattr(cfg, "imap_host", "") or "").strip()
    user = (getattr(cfg, "imap_username", "") or "").strip()
    pwd = os.environ.get("SECONDBRAIN_IMAP_PASSWORD", "")
    if not host:
        return True, "(imap not configured)", {"configured": False}
    if not user:
        return False, "imap_username empty in config", {"configured": True}
    if not pwd:
        return False, (
            "SECONDBRAIN_IMAP_PASSWORD env var not set"
        ), {"configured": True}
    if not network:
        # Shape-only path: all three are set, assume good. The next
        # IMAP sync (which does its own login) will surface a real
        # auth failure via the connector's error path if creds are bad.
        return True, "", {
            "host": host, "user": user, "verified": "shape-only",
        }
    try:
        import imaplib
        with imaplib.IMAP4_SSL(host, getattr(cfg, "imap_port", 993),
                               timeout=15) as M:
            M.login(user, pwd)
    except Exception as e:  # noqa: BLE001
        return False, f"imap login failed: {type(e).__name__}: {e}", {}
    return True, "", {"host": host, "user": user, "verified": "live-login"}


def check_watched_folders(cfg: Config) -> tuple[bool, str, dict]:
    """Every watched folder must exist. Surfaces the case where a
    drive got unmounted or a folder was renamed."""
    from pathlib import Path
    folders = list(getattr(cfg, "watched_folders", []) or [])
    if not folders:
        return True, "(no watched folders configured)", {"configured": False}
    missing = []
    for f in folders:
        p = Path(f).expanduser()
        if not p.is_dir():
            missing.append(str(p))
    if missing:
        return False, f"missing folders: {missing}", {"missing": missing}
    return True, "", {"n_folders": len(folders)}


# ---- Registry + runner ----------------------------------------------

_CHECKS: dict = {
    "google_calendar": check_google_calendar,
    "anthropic": check_anthropic_key,
    "voyage": check_voyage_key,
    "local_llm": check_local_llm,
    "imap": check_imap,
    "watched_folders": check_watched_folders,
}


def run_all(
    conn: sqlite3.Connection, cfg: Config,
    *, network: bool = False,
) -> dict[str, HealthStatus]:
    """Run every registered check + persist results. Returns the
    full status dict.

    ``network=True`` opts into checks that make real network calls
    (currently: live IMAP login). Used by ``secondbrain doctor``
    for ad-hoc triage but NOT by the hourly daemon job — that runs
    shape-only to avoid auth-log spam on the IMAP server.
    """
    _ensure_schema(conn)
    out: dict[str, HealthStatus] = {}
    for name, fn in _CHECKS.items():
        try:
            # Pass network kwarg only to checks that accept it.
            try:
                ok, err, extra = fn(cfg, network=network)
            except TypeError:
                ok, err, extra = fn(cfg)
        except Exception as e:  # noqa: BLE001
            ok, err, extra = False, f"check raised: {e}", {}
        _persist(conn, name=name, ok=ok, error=err, extra=extra)
        # Read back so days_since_ok reflects persisted state.
        st = get_status(conn, name)
        if st is not None:
            out[name] = st
    return out


def _persist(
    conn: sqlite3.Connection, *,
    name: str, ok: bool, error: str, extra: dict,
) -> None:
    """Upsert a check result, preserving last_ok_at + first_checked_at
    across runs (round 11)."""
    import json
    now = time.time()
    row = conn.execute(
        "SELECT last_ok_at, first_checked_at FROM health_checks "
        "WHERE name = ?",
        (name,),
    ).fetchone()
    if row and ok:
        last_ok_at = now
    elif row:
        last_ok_at = row["last_ok_at"]
    else:
        last_ok_at = now if ok else None
    # Preserve first_checked_at if we've seen this name before.
    first_checked_at = (
        row["first_checked_at"] if row and row["first_checked_at"]
        else now
    )
    conn.execute(
        "INSERT OR REPLACE INTO health_checks"
        "(name, last_checked_at, last_ok_at, ok, error, "
        " extra_json, first_checked_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            name, now, last_ok_at, 1 if ok else 0,
            error[:500] if error else "",
            json.dumps(extra) if extra else None,
            first_checked_at,
        ),
    )
    conn.commit()


def get_status(
    conn: sqlite3.Connection, name: str,
) -> HealthStatus | None:
    _ensure_schema(conn)
    row = conn.execute(
        "SELECT * FROM health_checks WHERE name = ?", (name,),
    ).fetchone()
    if row is None:
        return None
    return _row_to_status(row)


def list_status(conn: sqlite3.Connection) -> list[HealthStatus]:
    _ensure_schema(conn)
    rows = conn.execute(
        "SELECT * FROM health_checks ORDER BY name",
    ).fetchall()
    return [_row_to_status(r) for r in rows]


def stale_failures(conn: sqlite3.Connection) -> list[HealthStatus]:
    """Checks that have been failing for over the stale threshold —
    these are the ones the brief should nudge about."""
    return [s for s in list_status(conn) if s.is_stale]


def _row_to_status(row) -> HealthStatus:
    import json
    extra: dict = {}
    try:
        if row["extra_json"]:
            parsed = json.loads(row["extra_json"])
            if isinstance(parsed, dict):
                extra = parsed
    except (TypeError, ValueError):
        extra = {}
    # Round 11 — first_checked_at column may not exist on legacy
    # databases (round-10 schema). Guard the access.
    try:
        first_checked = row["first_checked_at"]
        first_checked_at = float(first_checked) if first_checked else None
    except (IndexError, KeyError):
        first_checked_at = None
    return HealthStatus(
        name=row["name"],
        ok=bool(row["ok"]),
        last_checked_at=float(row["last_checked_at"]),
        last_ok_at=float(row["last_ok_at"]) if row["last_ok_at"] else None,
        error=row["error"] or "",
        extra=extra,
        first_checked_at=first_checked_at,
    )


def run_if_due(conn: sqlite3.Connection, cfg: Config) -> int:
    """Daemon entrypoint — runs every check at most every
    ``_CHECK_INTERVAL_SECONDS``. Returns count of checks executed."""
    _ensure_schema(conn)
    cutoff = time.time() - _CHECK_INTERVAL_SECONDS
    row = conn.execute(
        "SELECT MAX(last_checked_at) AS last FROM health_checks",
    ).fetchone()
    last = row["last"] if row else None
    if last and last > cutoff:
        return 0
    out = run_all(conn, cfg)
    if out:
        log.info("health_checks: ran %d check(s)", len(out))
    return len(out)
