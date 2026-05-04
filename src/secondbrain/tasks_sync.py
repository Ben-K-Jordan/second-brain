"""Phase 76: bidirectional task sync with external services.

The ``tasks`` table (Phase 47) reserved ``external_id`` /
``external_provider`` columns for exactly this hookup. This module
fills them in via Todoist's REST API (the easiest cross-platform
option — Apple Reminders is macOS-only AppleScript).

How it works:

  1. **Push**: open tasks without ``external_id`` get created on the
     remote. We store the returned id back in our row so subsequent
     syncs know we own that remote task.

  2. **Pull**: for tasks we own (have an ``external_id``), check the
     remote's status. If the user closed it on Todoist, we mark it
     done locally — keeping the daily brief honest.

  3. **No deletes**: we don't push our task deletions to the remote
     and we don't mirror remote deletes locally. Easier reasoning
     about a one-way "soft-delete via cancel" model than chasing
     race conditions where both sides delete + re-create.

Auth: ``TODOIST_TOKEN`` env var. Token comes from
https://app.todoist.com/app/settings/integrations/developer.

Cost: Todoist's free tier has rate limits (450 requests / 15 min)
which we stay well under — one sync pass at most O(20) tasks.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass

import requests

from . import tasks as tasks_mod
from .config import Config

log = logging.getLogger(__name__)


_TODOIST_API = "https://api.todoist.com/rest/v2"
_TIMEOUT = 30
# Cap one sync pass to keep API quota usage bounded.
_MAX_PUSH_PER_RUN = 20
_MAX_PULL_PER_RUN = 50


@dataclass
class SyncResult:
    pushed: int = 0          # newly created on Todoist
    pulled_done: int = 0     # marked done locally because remote closed
    errors: int = 0


def _resolve_token() -> str | None:
    return (os.environ.get("TODOIST_TOKEN") or "").strip() or None


def is_enabled() -> bool:
    return _resolve_token() is not None


def _session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    })
    return s


def push_open_tasks(
    conn: sqlite3.Connection, *, max_per_run: int = _MAX_PUSH_PER_RUN,
) -> tuple[int, int]:
    """Create open tasks without external_id on Todoist.

    Returns (pushed, errors)."""
    token = _resolve_token()
    if not token:
        return 0, 0
    rows = conn.execute(
        "SELECT id, text FROM tasks "
        "WHERE status = 'open' AND external_id = '' "
        "ORDER BY created_at ASC LIMIT ?",
        (max_per_run,),
    ).fetchall()
    if not rows:
        return 0, 0
    s = _session(token)
    pushed = 0
    errors = 0
    try:
        for r in rows:
            try:
                resp = s.post(
                    f"{_TODOIST_API}/tasks",
                    json={"content": r["text"]},
                    timeout=_TIMEOUT,
                )
            except requests.RequestException as e:
                log.warning("todoist: push failed for #%s: %s",
                            r["id"], type(e).__name__)
                errors += 1
                continue
            # Round 18 fix (audit-found gap M6) — honor 429
            # Retry-After. Todoist enforces 450 req / 15 min;
            # without backoff we hammer the API while it's
            # asking us to slow down.
            from .connectors import respect_retry_after
            if respect_retry_after(resp):
                try:
                    resp = s.post(
                        f"{_TODOIST_API}/tasks",
                        json={"content": r["text"]},
                        timeout=_TIMEOUT,
                    )
                except requests.RequestException as e:
                    log.warning("todoist: push retry failed for #%s: %s",
                                r["id"], type(e).__name__)
                    errors += 1
                    continue
            if resp.status_code not in (200, 204):
                log.warning("todoist: push HTTP %s for #%s",
                            resp.status_code, r["id"])
                errors += 1
                continue
            try:
                remote_id = str(resp.json().get("id") or "")
            except ValueError:
                errors += 1
                continue
            if not remote_id:
                errors += 1
                continue
            conn.execute(
                "UPDATE tasks SET external_id = ?, external_provider = 'todoist' "
                "WHERE id = ?",
                (remote_id, r["id"]),
            )
            pushed += 1
    finally:
        s.close()
    if pushed:
        conn.commit()
    return pushed, errors


def pull_remote_completions(
    conn: sqlite3.Connection, *, max_per_run: int = _MAX_PULL_PER_RUN,
) -> tuple[int, int]:
    """Check our owned remote tasks; if the remote closed them, mark
    done locally.

    Todoist's API returns 200 + `is_completed: true` for closed tasks.
    A 404 means the user deleted on the remote — we treat that as
    cancelled locally to keep state in sync without chasing edge
    cases.

    Returns (pulled_done, errors).
    """
    token = _resolve_token()
    if not token:
        return 0, 0
    rows = conn.execute(
        "SELECT id, external_id FROM tasks "
        "WHERE status = 'open' AND external_provider = 'todoist' "
        "  AND external_id <> '' "
        "ORDER BY created_at DESC LIMIT ?",
        (max_per_run,),
    ).fetchall()
    if not rows:
        return 0, 0
    s = _session(token)
    pulled = 0
    errors = 0
    try:
        for r in rows:
            try:
                resp = s.get(
                    f"{_TODOIST_API}/tasks/{r['external_id']}",
                    timeout=_TIMEOUT,
                )
            except requests.RequestException as e:
                log.warning("todoist: pull failed for #%s: %s",
                            r["id"], type(e).__name__)
                errors += 1
                continue
            # Round 18 fix (audit-found gap M6) — honor 429.
            from .connectors import respect_retry_after
            if respect_retry_after(resp):
                try:
                    resp = s.get(
                        f"{_TODOIST_API}/tasks/{r['external_id']}",
                        timeout=_TIMEOUT,
                    )
                except requests.RequestException as e:
                    log.warning("todoist: pull retry failed for #%s: %s",
                                r["id"], type(e).__name__)
                    errors += 1
                    continue
            if resp.status_code == 404:
                # Remote deleted → cancel locally.
                tasks_mod.mark_cancelled(conn, r["id"])
                continue
            if resp.status_code != 200:
                errors += 1
                continue
            try:
                payload = resp.json()
            except ValueError:
                errors += 1
                continue
            if (
                payload.get("is_completed")
                and tasks_mod.mark_done(conn, r["id"])
            ):
                pulled += 1
    finally:
        s.close()
    return pulled, errors


def sync(conn: sqlite3.Connection) -> SyncResult:
    """One pass: push new + pull completions. Idempotent."""
    pushed, push_err = push_open_tasks(conn)
    pulled, pull_err = pull_remote_completions(conn)
    return SyncResult(
        pushed=pushed,
        pulled_done=pulled,
        errors=push_err + pull_err,
    )


def run_if_due(cfg: Config, conn: sqlite3.Connection) -> bool:
    """Daemon entrypoint. The Scheduler bounds the cadence; this just
    no-ops when the user doesn't have Todoist configured."""
    if not is_enabled():
        return False
    result = sync(conn)
    log.info(
        "todoist sync: pushed=%d pulled=%d errors=%d",
        result.pushed, result.pulled_done, result.errors,
    )
    return True
