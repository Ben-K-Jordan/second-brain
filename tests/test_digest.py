"""Phase 32: email digest rendering + scheduler."""

from __future__ import annotations

import json
import time
from dataclasses import replace

from secondbrain.db import (
    watchlist_create,
    watchlist_run_record_finish,
    watchlist_run_record_start,
)
from secondbrain.digest import (
    _ensure_digest_runs_table,
    _gather,
    _render_html,
    _render_text,
    build_email,
    last_digest_sent_at,
    run_digest_if_due,
    send_digest,
)


def _seed_run(conn, wid, paths, answer="(ok)", new_paths=None, started_at=None):
    rid = watchlist_run_record_start(conn, wid)
    if started_at is not None:
        conn.execute(
            "UPDATE watchlist_runs SET started_at = ? WHERE id = ?",
            (started_at, rid),
        )
        conn.commit()
    cites = [{"file_path": p} for p in paths]
    np = list(new_paths) if new_paths is not None else list(paths)
    watchlist_run_record_finish(
        conn, rid,
        answer=answer,
        citations_json=json.dumps(cites),
        new_paths_json=json.dumps(np) if np else None,
        new_count=len(np),
    )


# ----------------------------- gather --------------------------------

def test_gather_empty_returns_empty(fresh_db):
    """No watchlists → empty list, no crash."""
    assert _gather(fresh_db, None) == []


def test_gather_includes_recent_runs(fresh_db):
    wid = watchlist_create(fresh_db, "pm", "q")
    _seed_run(fresh_db, wid, ["https://a", "https://b"])
    rows = _gather(fresh_db, None)
    assert len(rows) == 1
    assert rows[0]["watchlist"]["name"] == "pm"
    assert set(rows[0]["all_new_paths"]) == {"https://a", "https://b"}


def test_gather_filters_by_since(fresh_db):
    """Runs older than the cutoff are excluded."""
    wid = watchlist_create(fresh_db, "pm", "q")
    _seed_run(
        fresh_db, wid, ["https://old"],
        started_at=time.time() - 48 * 3600,
    )
    _seed_run(fresh_db, wid, ["https://fresh"])
    rows = _gather(fresh_db, since_ts=time.time() - 24 * 3600)
    assert len(rows) == 1
    assert rows[0]["all_new_paths"] == ["https://fresh"]


def test_gather_dedups_new_paths_across_runs(fresh_db):
    """If two runs in the window both reported the same new path, it
    should appear once in the gathered new_paths list."""
    wid = watchlist_create(fresh_db, "pm", "q")
    _seed_run(fresh_db, wid, ["https://a"], new_paths=["https://a"])
    _seed_run(fresh_db, wid, ["https://a", "https://b"],
              new_paths=["https://a", "https://b"])  # bug-by-design: a "appears" twice
    rows = _gather(fresh_db, since_ts=time.time() - 3600)
    assert rows[0]["all_new_paths"].count("https://a") == 1


# ---------------------------- rendering -------------------------------

def test_render_text_no_activity():
    out = _render_text([], None)
    assert "no watchlist activity" in out.lower()


def test_render_text_includes_query_and_new_paths():
    rows = [{
        "watchlist": {"name": "pm", "query": "PM internships",
                      "schedule_minutes": 1440},
        "latest_answer": "Found 2 new postings.",
        "latest_started_at": 0,
        "latest_error": None,
        "all_new_paths": ["https://x.com/job/1", "https://x.com/job/2"],
        "run_count": 1,
    }]
    out = _render_text(rows, None)
    assert "## pm" in out
    assert "PM internships" in out
    assert "https://x.com/job/1" in out
    assert "Found 2 new postings" in out


def test_render_text_truncates_long_lists():
    paths = [f"https://x.com/job/{i}" for i in range(60)]
    rows = [{
        "watchlist": {"name": "pm", "query": "q", "schedule_minutes": 1440},
        "latest_answer": "", "latest_started_at": 0, "latest_error": None,
        "all_new_paths": paths, "run_count": 1,
    }]
    out = _render_text(rows, None)
    assert "and 30 more" in out  # 60 paths, top-30 shown, 30 truncated


def test_render_html_includes_watchlist_section():
    rows = [{
        "watchlist": {"name": "pm", "query": "PM internships",
                      "schedule_minutes": 1440},
        "latest_answer": "Found things.",
        "latest_started_at": 0, "latest_error": None,
        "all_new_paths": ["https://example.com/job/123"],
        "run_count": 1,
    }]
    html = _render_html(rows, None)
    assert "<h3" in html
    assert "pm" in html
    assert "https://example.com/job/123" in html


def test_render_html_escapes_user_data():
    """Watchlist name + query come from the user; we must escape HTML
    so a query like '<script>' can't inject into the email body."""
    rows = [{
        "watchlist": {"name": "<script>alert(1)</script>",
                      "query": "<img src=x>",
                      "schedule_minutes": 60},
        "latest_answer": "<b>boom</b>",
        "latest_started_at": 0, "latest_error": None,
        "all_new_paths": [], "run_count": 1,
    }]
    html = _render_html(rows, None)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;b&gt;boom&lt;/b&gt;" in html


def test_render_html_shows_error():
    rows = [{
        "watchlist": {"name": "pm", "query": "q", "schedule_minutes": 60},
        "latest_answer": "", "latest_started_at": 0,
        "latest_error": "rate limited",
        "all_new_paths": [], "run_count": 1,
    }]
    html = _render_html(rows, None)
    assert "rate limited" in html
    assert "errored" in html


# ------------------------------ build ---------------------------------

def test_build_email_subject_format(fresh_db, tmp_cfg):
    cfg = replace(tmp_cfg, digest_enabled=True,
                  digest_to="user@example.com",
                  digest_smtp_user="me@example.com")
    wid = watchlist_create(fresh_db, "pm", "q")
    _seed_run(fresh_db, wid, ["https://a"])
    msg, n_wl, n_new = build_email(cfg, fresh_db, None)
    assert msg["To"] == "user@example.com"
    assert msg["From"] == "me@example.com"
    assert "second-brain digest" in msg["Subject"]
    assert "(1 new)" in msg["Subject"]
    assert n_wl == 1 and n_new == 1
    # Should have both text and html parts.
    parts = list(msg.iter_parts())
    types = {p.get_content_type() for p in parts} or {msg.get_content_type()}
    assert "text/html" in types or "multipart/alternative" in (msg.get_content_type(),)


def test_build_email_no_new_omits_count(fresh_db, tmp_cfg):
    cfg = replace(tmp_cfg, digest_enabled=True,
                  digest_to="user@example.com",
                  digest_smtp_user="me@example.com")
    msg, n_wl, n_new = build_email(cfg, fresh_db, None)
    assert n_new == 0
    assert "(0 new)" not in msg["Subject"]
    # Subject should be stable when there's nothing to report.
    assert "second-brain digest" in msg["Subject"]


# ----------------------------- send_digest ----------------------------

def test_send_digest_disabled(fresh_db, tmp_cfg):
    """digest_enabled=false should refuse to send."""
    cfg = replace(tmp_cfg, digest_enabled=False, digest_to="x@y.com")
    ok, msg = send_digest(cfg, fresh_db)
    assert ok is False
    assert "digest_enabled" in msg


def test_send_digest_no_recipients(fresh_db, tmp_cfg):
    cfg = replace(tmp_cfg, digest_enabled=True, digest_to="")
    ok, msg = send_digest(cfg, fresh_db)
    assert ok is False
    assert "digest_to" in msg


def test_send_digest_no_password(fresh_db, tmp_cfg, monkeypatch):
    monkeypatch.delenv("SECONDBRAIN_SMTP_PASSWORD", raising=False)
    cfg = replace(tmp_cfg, digest_enabled=True,
                  digest_to="x@y.com", digest_smtp_user="me@y.com")
    ok, msg = send_digest(cfg, fresh_db)
    assert ok is False
    assert "SECONDBRAIN_SMTP_PASSWORD" in msg


def test_send_digest_records_failure(fresh_db, tmp_cfg, monkeypatch):
    """A failed SMTP send still records a row in digest_runs so the
    history view shows it."""
    monkeypatch.setenv("SECONDBRAIN_SMTP_PASSWORD", "fake")

    # Stub smtplib.SMTP to raise a connection error.
    class FakeSMTP:
        def __init__(self, *a, **kw):
            raise OSError("network unreachable")

    import secondbrain.digest as dig
    monkeypatch.setattr(dig.smtplib, "SMTP", FakeSMTP)

    cfg = replace(tmp_cfg, digest_enabled=True,
                  digest_to="x@y.com", digest_smtp_user="me@y.com",
                  digest_smtp_host="smtp.example.com")
    wid = watchlist_create(fresh_db, "pm", "q")
    _seed_run(fresh_db, wid, ["https://a"])
    ok, _msg = send_digest(cfg, fresh_db)
    assert ok is False
    # Failure row should be persisted.
    rows = fresh_db.execute(
        "SELECT * FROM digest_runs WHERE success = 0",
    ).fetchall()
    assert len(rows) == 1
    assert "network unreachable" in (rows[0]["error"] or "")


def test_send_digest_success_path(fresh_db, tmp_cfg, monkeypatch):
    """Stub SMTP to "succeed"; verify a success row + last_digest_sent_at
    advances."""
    monkeypatch.setenv("SECONDBRAIN_SMTP_PASSWORD", "fake")

    sent: list = []

    class FakeSMTP:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def ehlo(self): pass
        def starttls(self): pass
        def login(self, *a, **kw): pass
        def send_message(self, m): sent.append(m)

    import secondbrain.digest as dig
    monkeypatch.setattr(dig.smtplib, "SMTP", FakeSMTP)

    cfg = replace(tmp_cfg, digest_enabled=True,
                  digest_to="x@y.com", digest_smtp_user="me@y.com")
    wid = watchlist_create(fresh_db, "pm", "q")
    _seed_run(fresh_db, wid, ["https://a"])
    before_send = time.time()
    ok, _msg = send_digest(cfg, fresh_db)
    assert ok is True
    assert len(sent) == 1
    assert sent[0]["To"] == "x@y.com"
    last_sent = last_digest_sent_at(fresh_db)
    assert last_sent is not None and last_sent >= before_send


# ----------------------------- scheduler ------------------------------

def test_run_digest_if_due_returns_false_when_disabled(fresh_db, tmp_cfg):
    cfg = replace(tmp_cfg, digest_enabled=False)
    assert run_digest_if_due(cfg, fresh_db) is False


def test_run_digest_if_due_skips_if_recent(fresh_db, tmp_cfg, monkeypatch):
    """If a digest sent successfully < 12h ago, don't re-fire even past
    digest_send_time."""
    cfg = replace(tmp_cfg, digest_enabled=True, digest_to="x@y.com",
                  digest_send_time="00:00")  # always past
    _ensure_digest_runs_table(fresh_db)
    fresh_db.execute(
        "INSERT INTO digest_runs(sent_at, success, recipients, "
        "watchlists_summarized, new_items_total) "
        "VALUES (?, 1, ?, 0, 0)",
        (time.time() - 3600, "x@y.com"),
    )
    fresh_db.commit()
    # Should be a no-op because last digest was 1h ago.
    sent: list = []
    monkeypatch.setattr(
        "secondbrain.digest.send_digest", lambda *a, **kw: sent.append("X") or (True, "ok"),
    )
    run_digest_if_due(cfg, fresh_db)
    assert sent == [], "shouldn't re-send within 12h"


def test_run_digest_if_due_invalid_time_format(fresh_db, tmp_cfg):
    """Bad digest_send_time logs + returns False rather than crashing."""
    cfg = replace(tmp_cfg, digest_enabled=True, digest_to="x@y.com",
                  digest_send_time="garbage")
    assert run_digest_if_due(cfg, fresh_db) is False
