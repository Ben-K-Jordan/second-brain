"""Round 16 (Phase C) — smart desktop notifications.

A notification queue + rule-based detector + tray surfacer. The
philosophy is "the brain proactively tells you genuinely time-sensitive
things — and ONLY those". Notification spam is what made every other
PIM tool annoying; we err on the side of "fewer, better" by default.

Categories:
  - **email_urgent**: A new email triaged as 'urgent' arrived.
  - **birthday**: A contact's birthday is in the next 3 days.
  - **journal_nudge**: You haven't journaled in N+ days (default: 3).
  - **stale_health**: A health check failed and hasn't recovered (e.g.
    Anthropic key invalid).
  - **review_ready**: A new weekly letter was generated.
  - **draft_pending**: Email drafts are piling up unreviewed (>=5).
  - **task_overdue**: A task with a due date passed.

Storage:
  ``notifications`` table. Each row has a stable ``key`` so detectors
  can re-fire idempotently (e.g. "email_urgent:file_id=123" only fires
  once per email). ``status`` is 'pending' / 'shown' / 'dismissed'.

Detection:
  ``detect_all(conn)`` runs every detector and INSERT-OR-IGNOREs new
  rows. Daemon job runs hourly. Tray + dashboard pull from the same
  queue.

Surfacing:
  - **Tray**: a daemon job pops pending notifications via
    ``icon.notify(...)``. Marked 'shown' so we don't re-pop.
  - **Dashboard**: ``/notifications`` page lists pending + recent. JSON
    endpoint ``/api/notifications`` for the badge.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
import weakref as _weakref
from dataclasses import dataclass
from datetime import date, timedelta

log = logging.getLogger(__name__)

_SCHEMA_INITIALIZED: _weakref.WeakSet = _weakref.WeakSet()


# Round 17 fix (audit-found gap E1) — serialise writes to the
# notifications table. The tray notification thread, the scheduler
# thread, and FastAPI dashboard request workers all share the same
# writer connection (Round 1 design choice). sqlite3 connections are
# not thread-safe by default; concurrent commits can interleave the
# cursor state and corrupt fetches. Round 13 added an identical lock
# for ai_audit; this is the equivalent for notifications.
_WRITE_LOCK = threading.RLock()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    try:
        if conn in _SCHEMA_INITIALIZED:
            return
    except TypeError:
        pass
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            -- ``key`` lets detectors re-fire idempotently (UNIQUE).
            key TEXT NOT NULL UNIQUE,
            kind TEXT NOT NULL,
            -- urgency: 'low' (badge only) / 'med' (badge + dashboard
            -- callout) / 'high' (tray pop). The dashboard renders
            -- 'high' with a louder color.
            urgency TEXT NOT NULL DEFAULT 'med',
            title TEXT NOT NULL,
            body TEXT NOT NULL DEFAULT '',
            href TEXT NOT NULL DEFAULT '',
            payload_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'pending',
                -- 'pending' | 'shown' | 'dismissed'
            created_at REAL NOT NULL,
            shown_at REAL,
            dismissed_at REAL
        );
        CREATE INDEX IF NOT EXISTS idx_notif_status_created
            ON notifications(status, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_notif_kind
            ON notifications(kind);
    """)
    conn.commit()
    try:
        _SCHEMA_INITIALIZED.add(conn)
    except TypeError:
        pass


# ============================ data class ==============================


@dataclass
class Notification:
    id: int
    key: str
    kind: str
    urgency: str
    title: str
    body: str
    href: str
    payload_json: str
    status: str
    created_at: float
    shown_at: float | None
    dismissed_at: float | None


def _row_to_notification(row) -> Notification:
    return Notification(
        id=int(row["id"]),
        key=row["key"],
        kind=row["kind"],
        urgency=row["urgency"],
        title=row["title"],
        body=row["body"] or "",
        href=row["href"] or "",
        payload_json=row["payload_json"] or "{}",
        status=row["status"],
        created_at=float(row["created_at"]),
        shown_at=float(row["shown_at"]) if row["shown_at"] else None,
        dismissed_at=(
            float(row["dismissed_at"]) if row["dismissed_at"] else None
        ),
    )


# ============================ enqueue / fetch =========================


def enqueue(
    conn: sqlite3.Connection,
    *,
    key: str,
    kind: str,
    title: str,
    body: str = "",
    urgency: str = "med",
    href: str = "",
    payload: dict | None = None,
) -> bool:
    """Insert a notification. Returns True iff the row was new
    (UNIQUE on key means re-fires no-op).
    """
    import json
    # Round 17 fix: serialise schema-init AND INSERT + commit so
    # concurrent writers (tray thread, scheduler, dashboard workers)
    # can't interleave OR race on the lazy schema setup. Mirrors the
    # round-13 _AUDIT_WRITE_LOCK pattern in ai_audit.
    with _WRITE_LOCK:
        try:
            _ensure_schema(conn)
            cur = conn.execute(
                "INSERT OR IGNORE INTO notifications"
                "(key, kind, urgency, title, body, href, payload_json, "
                " status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
                (key, kind, urgency, title, body, href,
                 json.dumps(payload or {}), time.time()),
            )
            try:
                conn.commit()
            except sqlite3.OperationalError:
                # INSERT OR IGNORE hit UNIQUE conflict → no
                # transaction was opened → commit() raises. Safe to
                # ignore; the row already exists from a prior call.
                pass
            return cur.rowcount > 0
        except Exception as e:  # noqa: BLE001
            log.debug("notifications: enqueue failed: %s", e)
            return False


def list_pending(
    conn: sqlite3.Connection, limit: int = 20,
) -> list[Notification]:
    _ensure_schema(conn)
    rows = conn.execute(
        "SELECT * FROM notifications WHERE status = 'pending' "
        "ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_row_to_notification(r) for r in rows]


def list_recent(
    conn: sqlite3.Connection, limit: int = 50,
) -> list[Notification]:
    _ensure_schema(conn)
    rows = conn.execute(
        "SELECT * FROM notifications "
        "ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_row_to_notification(r) for r in rows]


def count_pending(conn: sqlite3.Connection) -> int:
    _ensure_schema(conn)
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM notifications WHERE status = 'pending'",
    ).fetchone()
    return int(row["n"] or 0)


def mark_shown(conn: sqlite3.Connection, notification_id: int) -> None:
    _ensure_schema(conn)
    with _WRITE_LOCK:
        try:
            conn.execute(
                "UPDATE notifications SET status = 'shown', shown_at = ? "
                "WHERE id = ? AND status = 'pending'",
                (time.time(), notification_id),
            )
            try:
                conn.commit()
            except sqlite3.OperationalError:
                pass  # nothing to commit (no row matched)
        except Exception as e:  # noqa: BLE001
            log.debug("notifications: mark_shown failed: %s", e)


def mark_dismissed(conn: sqlite3.Connection, notification_id: int) -> None:
    _ensure_schema(conn)
    with _WRITE_LOCK:
        try:
            conn.execute(
                "UPDATE notifications SET status = 'dismissed', "
                "dismissed_at = ? WHERE id = ?",
                (time.time(), notification_id),
            )
            try:
                conn.commit()
            except sqlite3.OperationalError:
                pass
        except Exception as e:  # noqa: BLE001
            log.debug("notifications: mark_dismissed failed: %s", e)


def dismiss_all(conn: sqlite3.Connection) -> int:
    """Round 17 fix (audit-found gap D2) — only dismisses 'pending'.

    The earlier version flipped both 'pending' AND 'shown' rows to
    'dismissed', overwriting the ``shown_at`` history of items the
    user had already seen via tray pop. The user-facing semantics of
    'Dismiss all' is "clear my pending inbox" — already-shown items
    aren't pending and shouldn't be touched.
    """
    _ensure_schema(conn)
    with _WRITE_LOCK:
        try:
            cur = conn.execute(
                "UPDATE notifications SET status = 'dismissed', "
                "dismissed_at = ? WHERE status = 'pending'",
                (time.time(),),
            )
            try:
                conn.commit()
            except sqlite3.OperationalError:
                pass
            return cur.rowcount
        except Exception as e:  # noqa: BLE001
            log.debug("notifications: dismiss_all failed: %s", e)
            return 0


# ============================ detectors ===============================


def _safe_date_in_year(year: int, month: int, day: int) -> date | None:
    """Round 17 fix (audit-found gap H3) — handle Feb 29 in non-leap
    years by remapping to Feb 28. Without this, anyone born on Feb 29
    silently gets zero birthday notifications in non-leap years
    (because ``date(2025, 2, 29)`` raises ``ValueError`` and the
    ``except`` swallows the entire person).

    Returns the resolved date or None if month/day is fundamentally
    invalid (e.g. day=99).
    """
    try:
        return date(year, month, day)
    except ValueError:
        # Most common failure: Feb 29 in non-leap year.
        if month == 2 and day == 29:
            try:
                return date(year, 2, 28)
            except ValueError:
                return None
        return None


def _detect_email_urgent(conn: sqlite3.Connection) -> int:
    """Round 16 — every email classified 'urgent' in the last 24h
    fires one notification (idempotent by file_id)."""
    cutoff = time.time() - 24 * 3600
    n = 0
    try:
        rows = conn.execute(
            "SELECT ec.file_id, f.path "
            "FROM email_classifications ec "
            "JOIN files f ON f.id = ec.file_id "
            "WHERE ec.label = 'urgent' AND f.indexed_at >= ?",
            (cutoff,),
        ).fetchall()
    except sqlite3.OperationalError:
        return 0
    # Round 25 fix (audit-found gap M2) — use file_id as the URL
    # parameter (round-25 fix H1 makes /file accept it). The
    # earlier ``?path=...`` interpolation didn't URL-encode and
    # broke for paths containing ``<``, ``>``, ``@``, ``?``, ``&``,
    # ``+``, ``#`` (e.g. imap://msgid/<id@host>). file_id is just
    # a digit, no escaping concerns.
    # Round 26 fix (audit-found gap M6) — date-bucket the key so
    # the same urgent email re-fires once per local day if the user
    # hasn't acted on it. Previously a single notification would
    # never resurface even if the email sat triaged-urgent for days.
    today = date.today().isoformat()
    for r in rows:
        title = (r["path"] or "").rsplit("/", 1)[-1] or "(email)"
        if enqueue(
            conn,
            key=f"email_urgent:file_id={r['file_id']}:{today}",
            kind="email_urgent", urgency="high",
            title=f"Urgent email: {title[:60]}",
            body="Triaged as urgent. Open the drafts page to respond.",
            href=f"/file?file_id={int(r['file_id'])}",
            payload={"file_id": int(r["file_id"])},
        ):
            n += 1
    return n


def _detect_birthdays(conn: sqlite3.Connection) -> int:
    """Notify 3 days ahead of any birthday in your contacts."""
    n = 0
    today = date.today()
    horizon = today + timedelta(days=3)
    try:
        rows = conn.execute(
            "SELECT id, display_name, birthday FROM people "
            "WHERE birthday IS NOT NULL AND birthday != ''",
        ).fetchall()
    except sqlite3.OperationalError:
        return 0
    for r in rows:
        bday_str = (r["birthday"] or "").strip()
        if not bday_str:
            continue
        try:
            # Birthdays often stored as 'MM-DD' or 'YYYY-MM-DD'.
            parts = bday_str.split("-")
            if len(parts) == 3:
                mm, dd = int(parts[1]), int(parts[2])
            elif len(parts) == 2:
                mm, dd = int(parts[0]), int(parts[1])
            else:
                continue
            this_year = _safe_date_in_year(today.year, mm, dd)
            if this_year is None:
                continue
            if this_year < today:
                this_year = _safe_date_in_year(today.year + 1, mm, dd)
                if this_year is None:
                    continue
        except (ValueError, IndexError):
            continue
        if today <= this_year <= horizon:
            days_until = (this_year - today).days
            label = (
                "today" if days_until == 0 else
                "tomorrow" if days_until == 1 else
                f"in {days_until} days"
            )
            if enqueue(
                conn,
                key=f"birthday:{r['id']}:{this_year.isoformat()}",
                kind="birthday", urgency="med",
                title=f"{r['display_name']}'s birthday {label}",
                body="Want to send a quick note?",
                href=f"/person?id={r['id']}",
                payload={
                    "person_id": int(r["id"]),
                    "birthday": this_year.isoformat(),
                },
            ):
                n += 1
    return n


def _detect_journal_nudge(conn: sqlite3.Connection) -> int:
    """If you haven't journaled in 3+ days, gentle nudge (idempotent
    per stretch)."""
    try:
        row = conn.execute(
            "SELECT date FROM journal_entries ORDER BY date DESC LIMIT 1",
        ).fetchone()
    except sqlite3.OperationalError:
        return 0
    today = date.today()
    if row is None:
        last = today - timedelta(days=999)
    else:
        try:
            last = date.fromisoformat(row["date"])
        except ValueError:
            return 0
    gap = (today - last).days
    if gap < 3:
        return 0
    # Bucket by week so we re-fire weekly, not daily.
    week_key = today.isocalendar()
    key = f"journal_nudge:{week_key[0]}-W{week_key[1]:02d}"
    return int(enqueue(
        conn,
        key=key,
        kind="journal_nudge", urgency="low",
        title=f"You haven't journaled in {gap} days",
        body="Want to capture today? Even one line counts.",
        href="/journal",
        payload={"days_since_last": gap},
    ))


def _detect_stale_health(conn: sqlite3.Connection) -> int:
    """Health checks (Anthropic key invalid, IMAP down, etc.) failing
    >24h get a high-urgency notification."""
    try:
        from . import health_checks
        stale = health_checks.stale_failures(conn)
    except Exception:  # noqa: BLE001
        return 0
    n = 0
    for status in stale:
        # Bucket by day so we re-fire daily until fixed.
        today = date.today().isoformat()
        if enqueue(
            conn,
            key=f"stale_health:{status.name}:{today}",
            kind="stale_health", urgency="high",
            title=f"Health check failing: {status.name}",
            body=status.error[:200] if status.error else "See diagnostics.",
            href="/health/system",
            payload={"check": status.name},
        ):
            n += 1
    return n


def _detect_review_ready(conn: sqlite3.Connection) -> int:
    """When a new weekly letter lands, surface it ONCE."""
    try:
        from . import weekly_letter
        letter = weekly_letter.latest_letter(conn)
    except Exception:  # noqa: BLE001
        return 0
    if letter is None:
        return 0
    return int(enqueue(
        conn,
        key=f"review_ready:{letter.week_end}",
        kind="review_ready", urgency="med",
        title=f"Weekly letter ready ({letter.week_end})",
        body="Your synthesis of the week is in.",
        href=f"/review?week_end={letter.week_end}",
        payload={"week_end": letter.week_end},
    ))


def _detect_draft_pending(conn: sqlite3.Connection) -> int:
    """≥5 pending drafts → one nudge per day."""
    try:
        n_drafts = int(conn.execute(
            "SELECT COUNT(*) AS n FROM email_drafts WHERE status = 'pending'",
        ).fetchone()["n"] or 0)
    except sqlite3.OperationalError:
        return 0
    if n_drafts < 5:
        return 0
    today = date.today().isoformat()
    return int(enqueue(
        conn,
        key=f"draft_pending:{today}",
        kind="draft_pending", urgency="low",
        title=f"{n_drafts} email drafts waiting for review",
        body="Open /drafts to send or discard.",
        href="/drafts",
        payload={"count": n_drafts},
    ))


def _detect_task_overdue(conn: sqlite3.Connection) -> int:
    """Tasks with a due_hint that's parseable as a past date."""
    n = 0
    try:
        rows = conn.execute(
            "SELECT id, text, due_hint FROM tasks "
            "WHERE status = 'open' AND due_hint IS NOT NULL "
            "AND due_hint != ''",
        ).fetchall()
    except sqlite3.OperationalError:
        return 0
    today = date.today()
    for r in rows:
        due = _parse_due_hint(r["due_hint"], today)
        if due is None or due > today:
            continue
        # Re-fire weekly if still overdue.
        wk = today.isocalendar()
        key = f"task_overdue:{r['id']}:{wk[0]}-W{wk[1]:02d}"
        days_late = (today - due).days
        if enqueue(
            conn,
            key=key,
            kind="task_overdue", urgency="med",
            title=f"Task overdue ({days_late}d): {r['text'][:60]}",
            body=f"Was due {due.isoformat()}.",
            href="/tasks",
            payload={
                "task_id": int(r["id"]), "due": due.isoformat(),
            },
        ):
            n += 1
    return n


def _parse_due_hint(hint: str, today: date) -> date | None:
    """Best-effort due-date parsing. Handles 'YYYY-MM-DD', 'tomorrow',
    'next monday', 'in 3 days'. Returns None if unparseable."""
    s = (hint or "").strip().lower()
    if not s:
        return None
    if s == "today":
        return today
    if s in ("tomorrow", "tmrw"):
        return today + timedelta(days=1)
    # ISO date
    try:
        return date.fromisoformat(s)
    except ValueError:
        pass
    # 'in N days'
    import re
    m = re.match(r"in (\d+) days?$", s)
    if m:
        return today + timedelta(days=int(m.group(1)))
    # 'next monday' etc — too brittle, skip.
    return None


def _detect_followup_overdue(conn: sqlite3.Connection) -> int:
    """Round 20 — followups whose due_at has passed.

    Idempotent key: ``followup_overdue:<id>`` — fires once per
    overdue followup. If the user resolves and a new followup
    re-uses the same id (impossible in practice), the key still
    matches and we no-op."""
    # Make sure the round-20 ``snooze_until`` column exists before
    # we reference it. The followups schema is round 19; the
    # snooze_until extension is round 20 in followups_ops.
    try:
        from . import followups_ops
        followups_ops._ensure_extended_schema(conn)
    except Exception:  # noqa: BLE001
        pass
    try:
        rows = conn.execute(
            "SELECT id, topic, person_name, due_at, direction "
            "FROM followups "
            "WHERE status = 'open' AND due_at IS NOT NULL "
            "  AND due_at < ? "
            "  AND (snooze_until IS NULL OR snooze_until <= ?) "
            "LIMIT 50",
            (time.time(), time.time()),
        ).fetchall()
    except sqlite3.OperationalError:
        return 0
    n = 0
    # Round 26 fix (audit-found gap M6) — date-bucket the key so a
    # still-overdue followup re-surfaces every local day until the
    # user resolves, snoozes, or dismisses it. Earlier the unique
    # key fired exactly once per followup id, so an item that sat
    # overdue for two weeks only ever pinged on day 1.
    today = date.today().isoformat()
    for r in rows:
        urgency = "high" if r["direction"] == "outgoing" else "med"
        action = "send" if r["direction"] == "outgoing" else "nudge"
        body = (
            f"Past due — {action} reply to "
            f"{r['person_name'] or '(unknown)'}"
        )
        if enqueue(
            conn,
            key=f"followup_overdue:{r['id']}:{today}",
            kind="followup_overdue",
            urgency=urgency,
            title=f"Overdue: {r['topic']}",
            body=body,
            href=f"/followups#fu{r['id']}",
            payload={
                "followup_id": int(r["id"]),
                "direction": r["direction"],
            },
        ):
            n += 1
    return n


def _detect_followup_stale(conn: sqlite3.Connection) -> int:
    """Round 20 — incoming followups that have been pending too long.

    "Sarah said she'd review by Tuesday" — if no resolution after
    14 days, surface a "want to nudge?" notification.
    """
    try:
        from . import followups_ops
        followups_ops._ensure_extended_schema(conn)
    except Exception:  # noqa: BLE001
        pass
    cutoff = time.time() - 14 * 86400
    try:
        rows = conn.execute(
            "SELECT id, topic, person_name, promised_at FROM followups "
            "WHERE status = 'open' AND direction = 'incoming' "
            "  AND COALESCE(promised_at, created_at) < ? "
            "  AND (snooze_until IS NULL OR snooze_until <= ?) "
            "LIMIT 30",
            (cutoff, time.time()),
        ).fetchall()
    except sqlite3.OperationalError:
        return 0
    n = 0
    # Round 26 fix (audit-found gap M6) — week-bucket the key so a
    # stale incoming followup quietly re-surfaces once per ISO week
    # until the user resolves or nudges. A weekly cadence avoids
    # nagging the user daily while still keeping aging items in view.
    iso_year, iso_week, _ = date.today().isocalendar()
    week_bucket = f"{iso_year}W{iso_week:02d}"
    for r in rows:
        days = int(
            (time.time() - float(r["promised_at"] or 0)) / 86400.0,
        )
        if enqueue(
            conn,
            key=f"followup_stale:{r['id']}:{week_bucket}",
            kind="followup_stale",
            urgency="low",
            title=f"Stale: {r['topic']}",
            body=(
                f"{r['person_name'] or 'they'} owed this "
                f"{days}d ago. Want to nudge?"
            ),
            href=f"/followups#fu{r['id']}",
            payload={"followup_id": int(r["id"])},
        ):
            n += 1
    return n


def _detect_cadence_overdue(conn: sqlite3.Connection) -> int:
    """Round 20 — VIPs the user hasn't reached out to within their
    cadence target."""
    try:
        from . import people as people_mod
        overdue = people_mod.list_overdue_contacts(
            conn, limit=20, tier_filter=["vip"],
        )
    except Exception:  # noqa: BLE001
        return 0
    # Round 21 fix (audit-found gap A6) — use local-tz date for the
    # "once per day per person" key. Earlier ``int(time.time()/86400)``
    # was UTC days, so users west of UTC had the boundary flip mid-
    # afternoon → either two notifications in one local day or none
    # for two local days.
    today_iso = date.today().isoformat()
    n = 0
    for o in overdue:
        if enqueue(
            conn,
            key=f"cadence_overdue:{o.person.id}:{today_iso}",
            kind="cadence_overdue",
            urgency="med",
            title=f"Reach out: {o.person.display_name}",
            body=(
                f"VIP, {o.days_since_contact}d since contact "
                f"({o.days_overdue}d past target)"
            ),
            href=f"/person?id={o.person.id}",
            payload={"person_id": int(o.person.id)},
        ):
            n += 1
    return n


def detect_all(conn: sqlite3.Connection) -> dict:
    """Run every detector. Returns a per-detector count of newly-enqueued
    notifications. Safe to call repeatedly — each detector uses
    idempotent keys."""
    out = {
        "email_urgent": _detect_email_urgent(conn),
        "birthday": _detect_birthdays(conn),
        "journal_nudge": _detect_journal_nudge(conn),
        "stale_health": _detect_stale_health(conn),
        "review_ready": _detect_review_ready(conn),
        "draft_pending": _detect_draft_pending(conn),
        "task_overdue": _detect_task_overdue(conn),
        # Round 20 detectors —
        "followup_overdue": _detect_followup_overdue(conn),
        "followup_stale": _detect_followup_stale(conn),
        "cadence_overdue": _detect_cadence_overdue(conn),
    }
    total = sum(out.values())
    if total:
        log.info("notifications: detected %d new (%s)", total, out)
    return out


# Round 17 fix (audit-found gap B-throttle) — the dashboard
# /notifications page used to call detect_all() on every load,
# which means a full people-table scan + email scan per page hit.
# Throttle to once per ``_DETECT_THROTTLE_SEC`` so refreshing the
# page doesn't hammer the DB. The hourly scheduler still fires
# unconditionally for guaranteed freshness.
_DETECT_THROTTLE_SEC = 60.0
_last_detect_ts: float = 0.0
_detect_throttle_lock = threading.Lock()


def detect_all_throttled(conn: sqlite3.Connection) -> dict | None:
    """Same as ``detect_all`` but no-ops when called more often than
    every ``_DETECT_THROTTLE_SEC`` seconds. Returns the detector
    summary dict, or None when throttled."""
    global _last_detect_ts
    now = time.time()
    with _detect_throttle_lock:
        if now - _last_detect_ts < _DETECT_THROTTLE_SEC:
            return None
        _last_detect_ts = now
    return detect_all(conn)


# ============================ tray surfacer ===========================


def pop_to_tray(
    conn: sqlite3.Connection, tray_icon, *, max_per_tick: int = 3,
) -> int:
    """Pop the oldest pending high/med urgency notifications via the
    pystray icon. Marks them 'shown' so they don't re-pop. Skips low
    urgency — those are dashboard-badge only."""
    if tray_icon is None:
        return 0
    _ensure_schema(conn)
    rows = conn.execute(
        "SELECT * FROM notifications "
        "WHERE status = 'pending' AND urgency IN ('high', 'med') "
        "ORDER BY "
        "  CASE urgency WHEN 'high' THEN 0 ELSE 1 END, created_at ASC "
        "LIMIT ?",
        (max_per_tick,),
    ).fetchall()
    n = 0
    for r in rows:
        notif = _row_to_notification(r)
        try:
            tray_icon.notify(notif.body or notif.title, notif.title)
        except Exception as e:  # noqa: BLE001
            log.warning("notifications: tray notify failed: %s", e)
            continue
        mark_shown(conn, notif.id)
        n += 1
    return n


# ============================ daemon entry ============================


def run_detectors_if_due(conn: sqlite3.Connection) -> int:
    """Daemon entry — run all detectors, return total enqueued count."""
    try:
        out = detect_all(conn)
    except Exception:  # noqa: BLE001
        log.exception("notifications: detector run crashed")
        return 0
    return sum(out.values())
