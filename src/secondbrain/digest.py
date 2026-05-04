"""Daily email digest of watchlist activity.

Renders a single email summarizing every watchlist that's run since the
last successful digest (or the last 24h if there's no prior digest). HTML
+ plain-text alternative parts for clients that don't render HTML.

Configuration lives in ``Config`` (digest_enabled / digest_to /
digest_send_time / digest_smtp_*). The SMTP password is taken from the
``SECONDBRAIN_SMTP_PASSWORD`` env var so the user never needs to put it in
config.toml.

Two callers:
- ``secondbrain digest send`` — one-shot, useful for testing or "give me
  the digest right now."
- ``run_digest_if_due(cfg, conn)`` — called by the daemon every minute;
  fires when the local time crosses ``digest_send_time`` AND the last
  digest is older than 12h (so a clock skew or daemon restart at 8:01am
  doesn't double-send).

Failures are caught + logged; we record a row in ``digest_runs`` either
way so the dashboard can show "last sent" / "last failure".
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import time
from datetime import datetime
from email.message import EmailMessage
from html import escape

from .config import Config
from .db import watchlist_list, watchlist_runs

log = logging.getLogger(__name__)


def _safe(s: str | None) -> str:
    """Phase 88 — apply sensitive-content redaction before any string
    leaves this module via the SMTP relay. Cheap (regex-only) and
    idempotent. ``None`` collapses to empty string so renderers can
    safely concatenate."""
    if not s:
        return ""
    try:
        from .safety import redact_text
    except ImportError:
        return s
    return redact_text(s)


# --------------------- digest_runs schema (ad-hoc) ---------------------
# Tiny side table; not worth its own helpers in db.py.

def _ensure_digest_runs_table(conn) -> None:
    """Create the digest_runs table on first use (idempotent)."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS digest_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sent_at REAL NOT NULL,
            success INTEGER NOT NULL,
            error TEXT,
            recipients TEXT,
            watchlists_summarized INTEGER NOT NULL DEFAULT 0,
            new_items_total INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_digest_runs_sent_at
            ON digest_runs(sent_at DESC);
    """)
    conn.commit()


def last_digest_sent_at(conn) -> float | None:
    """Return the most recent successful digest's send time, or None."""
    _ensure_digest_runs_table(conn)
    row = conn.execute(
        "SELECT sent_at FROM digest_runs WHERE success = 1 "
        "ORDER BY sent_at DESC LIMIT 1",
    ).fetchone()
    return float(row["sent_at"]) if row else None


def _record_run(conn, success: bool, error: str | None,
                recipients: str, n_wls: int, n_new: int) -> None:
    _ensure_digest_runs_table(conn)
    conn.execute(
        "INSERT INTO digest_runs"
        "(sent_at, success, error, recipients, watchlists_summarized, new_items_total) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (time.time(), 1 if success else 0, error, recipients, n_wls, n_new),
    )
    conn.commit()


# ----------------------------- rendering -------------------------------

def _gather(conn, since_ts: float | None) -> list[dict]:
    """For each watchlist, collect runs since ``since_ts`` (or last 24h)."""
    cutoff = since_ts if since_ts is not None else time.time() - 24 * 3600
    out: list[dict] = []
    for wl in watchlist_list(conn):
        runs = [
            r for r in watchlist_runs(conn, wl["id"], limit=20)
            if r["started_at"] >= cutoff and r["finished_at"] is not None
        ]
        if not runs:
            continue
        # Most recent run is the headline; collect all "new" items since cutoff.
        seen_paths: set[str] = set()
        all_new_paths: list[str] = []
        for r in runs:
            if r["error"]:
                continue
            try:
                np = json.loads(r["new_paths_json"]) if r["new_paths_json"] else []
            except (json.JSONDecodeError, TypeError):
                np = []
            for p in np:
                if p in seen_paths:
                    continue
                seen_paths.add(p)
                all_new_paths.append(p)
        latest = runs[0]
        out.append({
            "watchlist": dict(wl),
            "latest_answer": latest["answer"] or "",
            "latest_started_at": latest["started_at"],
            "latest_error": latest["error"],
            "all_new_paths": all_new_paths,
            "run_count": len(runs),
        })
    return out


def _render_html(rows: list[dict], since_ts: float | None) -> str:
    """Render the digest as an HTML email body (no external CSS)."""
    if not rows:
        body = (
            '<p>No watchlist activity in the window.</p>'
        )
    else:
        sections: list[str] = []
        for r in rows:
            wl = r["watchlist"]
            sched = wl["schedule_minutes"]
            new_paths = r["all_new_paths"]
            new_section = ""
            if new_paths:
                items = "".join(
                    f'<li><a href="{escape(_safe(p))}" '
                    f'style="color:#2c7a2c;">{escape(_safe(p))}</a></li>'
                    for p in new_paths[:30]
                )
                more = (
                    f'<li><i>… and {len(new_paths) - 30} more</i></li>'
                    if len(new_paths) > 30 else ""
                )
                new_section = (
                    f'<p style="margin:8px 0 4px 0;font-weight:600;">'
                    f'New since last digest ({len(new_paths)})</p>'
                    f'<ul style="padding-left:20px;margin:4px 0;">{items}{more}</ul>'
                )
            answer = _safe(r["latest_answer"])
            answer_html = ""
            if answer:
                # naive line→<br> rendering; the model's bullets stay readable.
                answer_html = (
                    '<div style="background:#f7f7f4;border-left:3px solid '
                    '#4abe4a;padding:10px 14px;margin:6px 0;">'
                    + escape(answer).replace("\n", "<br>")
                    + '</div>'
                )
            err_html = ""
            if r["latest_error"]:
                err_html = (
                    f'<p style="color:#a33;"><i>last run errored: '
                    f'{escape(_safe(r["latest_error"]))}</i></p>'
                )
            sections.append(
                f'<section style="margin:24px 0;padding-top:12px;'
                f'border-top:1px solid #ddd;">'
                f'<h3 style="margin:0 0 4px 0;">{escape(wl["name"])} '
                f'<span style="font-weight:400;color:#888;font-size:0.85em;">'
                f'· {r["run_count"]} run(s) · every {sched}m</span></h3>'
                f'<p style="color:#666;font-style:italic;margin:0 0 8px 0;">'
                f'"{escape(_safe(wl["query"]))}"</p>'
                f'{new_section}{answer_html}{err_html}'
                f'</section>'
            )
        body = "".join(sections)

    when = (
        datetime.fromtimestamp(since_ts).strftime("%Y-%m-%d %H:%M")
        if since_ts else "(last 24h)"
    )
    return (
        '<html><body style="font-family:system-ui,-apple-system,sans-serif;'
        'max-width:680px;margin:24px auto;color:#222;">'
        '<h1 style="color:#2c7a2c;">second-brain digest</h1>'
        f'<p style="color:#888;margin-top:-12px;">since {escape(when)}</p>'
        + body
        + '<hr style="border:none;border-top:1px solid #eee;margin:32px 0 12px 0;">'
        '<p style="color:#aaa;font-size:0.85em;">'
        'Generated by your local second-brain daemon.'
        '</p></body></html>'
    )


def _render_text(rows: list[dict], since_ts: float | None) -> str:
    """Plain-text alternative for clients without HTML rendering."""
    when = (
        datetime.fromtimestamp(since_ts).strftime("%Y-%m-%d %H:%M")
        if since_ts else "(last 24h)"
    )
    lines = [f"second-brain digest (since {when})", ""]
    if not rows:
        lines.append("(no watchlist activity)")
        return "\n".join(lines)
    for r in rows:
        wl = r["watchlist"]
        lines.append(f"## {wl['name']}")
        lines.append(f'   "{_safe(wl["query"])}"')
        if r["all_new_paths"]:
            lines.append(f"   New since last digest ({len(r['all_new_paths'])}):")
            for p in r["all_new_paths"][:30]:
                lines.append(f"     - {_safe(p)}")
            if len(r["all_new_paths"]) > 30:
                lines.append(f"     ... and {len(r['all_new_paths']) - 30} more")
        if r["latest_error"]:
            lines.append(f"   ! last run errored: {_safe(r['latest_error'])}")
        elif r["latest_answer"]:
            lines.append("")
            for ln in _safe(r["latest_answer"]).splitlines():
                lines.append(f"   {ln}")
        lines.append("")
    return "\n".join(lines)


# --------------------------- send + schedule ---------------------------

def build_email(cfg: Config, conn, since_ts: float | None) -> tuple[EmailMessage, int, int]:
    """Build the digest message. Returns (msg, n_watchlists, n_new_items)."""
    rows = _gather(conn, since_ts)
    n_new = sum(len(r["all_new_paths"]) for r in rows)
    msg = EmailMessage()
    msg["From"] = cfg.digest_smtp_from or cfg.digest_smtp_user
    msg["To"] = cfg.digest_to
    when_lbl = (
        datetime.fromtimestamp(since_ts).strftime("%Y-%m-%d")
        if since_ts else datetime.now().strftime("%Y-%m-%d")
    )
    new_marker = f" ({n_new} new)" if n_new else ""
    msg["Subject"] = f"second-brain digest · {when_lbl}{new_marker}"
    msg.set_content(_render_text(rows, since_ts))
    msg.add_alternative(_render_html(rows, since_ts), subtype="html")
    return msg, len(rows), n_new


def send_digest(cfg: Config, conn) -> tuple[bool, str]:
    """Build and send the digest now. Returns (success, message-or-error)."""
    if not cfg.digest_enabled:
        return False, "digest_enabled is false in config"
    if not cfg.digest_to:
        return False, "digest_to is empty"
    password = os.environ.get("SECONDBRAIN_SMTP_PASSWORD", "")
    if not password:
        return False, "SECONDBRAIN_SMTP_PASSWORD env var not set"

    since = last_digest_sent_at(conn)
    msg, n_wl, n_new = build_email(cfg, conn, since)

    try:
        with smtplib.SMTP(cfg.digest_smtp_host, cfg.digest_smtp_port, timeout=30) as s:
            s.ehlo()
            s.starttls()
            s.ehlo()
            s.login(cfg.digest_smtp_user, password)
            s.send_message(msg)
    except (smtplib.SMTPException, OSError, ValueError) as e:
        # Round 18 fix (audit-found gap L13) — also catch ValueError.
        # ``send_message`` on a malformed ``EmailMessage`` (e.g. a
        # comma-separated To: line that fails RFC 5322 strict
        # parsing) raises ValueError, which the prior except tuple
        # missed → digest job crashed instead of recording a
        # non-fatal failure.
        err = f"{type(e).__name__}: {e}"
        log.warning("digest send failed: %s", err)
        _record_run(conn, success=False, error=err,
                    recipients=cfg.digest_to,
                    n_wls=n_wl, n_new=n_new)
        return False, err
    log.info("digest sent to %s (%d watchlists, %d new items)",
             cfg.digest_to, n_wl, n_new)
    _record_run(conn, success=True, error=None,
                recipients=cfg.digest_to,
                n_wls=n_wl, n_new=n_new)
    return True, f"sent to {cfg.digest_to} ({n_new} new items)"


def run_digest_if_due(cfg: Config, conn) -> bool:
    """Daemon hook: send the digest if local time has crossed
    ``digest_send_time`` and the last digest was > 12h ago.

    Returns True iff a send actually happened (success or failure).
    Called once a minute by the daemon loop.
    """
    if not cfg.digest_enabled or not cfg.digest_to:
        return False
    try:
        hh, mm = cfg.digest_send_time.split(":")
        target_h, target_m = int(hh), int(mm)
    except (ValueError, AttributeError):
        log.warning(
            "digest_send_time %r isn't HH:MM; skipping",
            cfg.digest_send_time,
        )
        return False

    now = datetime.now()
    # Build today's send target in local time. If `now` is past it AND
    # the last digest is more than 12h old, fire.
    target = now.replace(
        hour=target_h, minute=target_m, second=0, microsecond=0,
    )
    if now < target:
        return False
    last = last_digest_sent_at(conn)
    if last is not None and (time.time() - last) < 12 * 3600:
        return False
    success, info = send_digest(cfg, conn)
    log.info("digest auto-fire: success=%s msg=%s", success, info)
    return True
