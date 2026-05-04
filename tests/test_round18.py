"""Round 18 — fixes for the post-round-17 comprehensive audit.

Each test maps to a finding from the audit:
  - HIGH H1: reflected XSS via /timeline ?kinds= query param
  - HIGH H2: /timeline navigation broken (date vs date_str param)
  - HIGH H3: /api/click missing same-origin CSRF guard
  - MED M4:  notification title not redacted in timeline
  - MED M5:  GitHub/Gmail/Linear/Pocket/Readwise no Retry-After backoff
  - MED M6:  Todoist sync no backoff on 429
  - MED M7:  people.canonicalize doesn't NFC-normalize
  - MED M8:  Config() creates real ~/second-brain/ in tests
  - LOW L9:  AppleScript escape uses '' instead of \\"
  - LOW L10: iMessage connector leaks sqlite conn on exception
  - LOW L12: synthesis _walk_cluster set ordering
  - LOW L13: digest.py SMTP missing ValueError catch
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

import pytest

# ============================ H1 — XSS on /timeline =================


def test_timeline_kinds_xss_payload_does_not_reflect(
    monkeypatch, tmp_path, fake_embedder,
):
    """Round 18 fix: an attacker-supplied ?kinds= with HTML/JS must
    NOT reflect into href attributes."""
    from fastapi.testclient import TestClient

    from secondbrain.config import Config
    from secondbrain.dashboard import create_app

    cfg = Config(data_dir=tmp_path / "data")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("secondbrain.dashboard.load_config", lambda: cfg)
    monkeypatch.setattr(
        "secondbrain.dashboard.make_embedder", lambda c: fake_embedder,
    )
    monkeypatch.setattr(
        "secondbrain.dashboard.make_reranker", lambda c: None,
    )
    app = create_app()
    client = TestClient(app)
    payload = '"><script>alert(1)</script>'
    r = client.get(f"/timeline?date=2025-04-12&kinds={payload}")
    assert r.status_code == 200
    # The raw payload must not appear in the response — the kinds
    # filter is closed-vocabulary, so the bad token gets dropped.
    assert payload not in r.text
    assert "<script>alert(1)</script>" not in r.text


def test_timeline_kinds_whitelist_keeps_valid_tokens(
    monkeypatch, tmp_path, fake_embedder,
):
    """Valid kind tokens still pass through and are reflected."""
    from fastapi.testclient import TestClient

    from secondbrain.config import Config
    from secondbrain.dashboard import create_app

    cfg = Config(data_dir=tmp_path / "data")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("secondbrain.dashboard.load_config", lambda: cfg)
    monkeypatch.setattr(
        "secondbrain.dashboard.make_embedder", lambda c: fake_embedder,
    )
    monkeypatch.setattr(
        "secondbrain.dashboard.make_reranker", lambda c: None,
    )
    app = create_app()
    client = TestClient(app)
    r = client.get("/timeline?date=2025-04-12&kinds=task,journal")
    assert r.status_code == 200
    # The reflected query string in nav links contains only the
    # whitelisted tokens.
    assert "kinds=journal,task" in r.text or "kinds=task,journal" in r.text


# ============================ H2 — /timeline ?date= param ===========


def test_timeline_date_query_param_binds(
    monkeypatch, tmp_path, fake_embedder,
):
    """Round 18 fix: Query(alias='date') makes ?date=... work.
    Previously the param was named date_str so FastAPI never bound
    it — every nav link silently snapped back to today."""
    from fastapi.testclient import TestClient

    from secondbrain.config import Config
    from secondbrain.dashboard import create_app

    cfg = Config(data_dir=tmp_path / "data")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("secondbrain.dashboard.load_config", lambda: cfg)
    monkeypatch.setattr(
        "secondbrain.dashboard.make_embedder", lambda c: fake_embedder,
    )
    monkeypatch.setattr(
        "secondbrain.dashboard.make_reranker", lambda c: None,
    )
    app = create_app()
    client = TestClient(app)
    # A specific past date must be acknowledged in the rendered
    # anchor span.
    r = client.get("/timeline?date=2024-01-15&days=7")
    assert r.status_code == 200
    assert "2024-01-15" in r.text


# ============================ H3 — /api/click CSRF =================


def test_api_click_rejects_cross_origin(
    monkeypatch, tmp_path, fake_embedder,
):
    """Round 18 fix: /api/click is now guarded by same-origin like
    every other state-mutating POST."""
    from fastapi.testclient import TestClient

    from secondbrain.config import Config
    from secondbrain.dashboard import create_app

    cfg = Config(data_dir=tmp_path / "data")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("secondbrain.dashboard.load_config", lambda: cfg)
    monkeypatch.setattr(
        "secondbrain.dashboard.make_embedder", lambda c: fake_embedder,
    )
    monkeypatch.setattr(
        "secondbrain.dashboard.make_reranker", lambda c: None,
    )
    app = create_app()
    client = TestClient(app)
    r = client.post(
        "/api/click",
        json={"path": "/some/path", "source": "evil"},
        headers={"origin": "https://evil.com"},
    )
    assert r.status_code == 403


def test_api_click_accepts_same_origin(
    monkeypatch, tmp_path, fake_embedder,
):
    """Localhost referer (the dashboard's own JS) still works."""
    from fastapi.testclient import TestClient

    from secondbrain.config import Config
    from secondbrain.dashboard import create_app

    cfg = Config(data_dir=tmp_path / "data")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("secondbrain.dashboard.load_config", lambda: cfg)
    monkeypatch.setattr(
        "secondbrain.dashboard.make_embedder", lambda c: fake_embedder,
    )
    monkeypatch.setattr(
        "secondbrain.dashboard.make_reranker", lambda c: None,
    )
    app = create_app()
    client = TestClient(app)
    r = client.post(
        "/api/click",
        json={"path": "/foo", "source": "search"},
        headers={"referer": "http://127.0.0.1:8765/"},
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True}


# ============================ M4 — notif title redacted =============


def test_timeline_notification_title_is_redacted(fresh_db):
    """Round 18 fix: notification.title in the timeline now passes
    through _redact, matching every other event source."""
    from secondbrain import notifications, timeline

    secret_title = (
        "Re: API key sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    )
    notifications.enqueue(
        fresh_db,
        key="t1", kind="email_urgent",
        title=secret_title,
        body="(unused body)",
    )
    since_ts, until_ts = timeline.parse_window(None, days=1)
    events = timeline.assemble(fresh_db, since_ts, until_ts)
    notif_events = [e for e in events if e.kind.startswith("notif")]
    assert notif_events, "expected at least one notif timeline event"
    assert "sk-ant-api03-AAAA" not in notif_events[0].title
    assert "[REDACTED:anthropic_key]" in notif_events[0].title


# ============================ M5/M6 — connector backoff =============


def test_github_iter_repos_retries_on_429(monkeypatch):
    """Round 18 fix: github connector now honors 429 Retry-After
    via respect_retry_after."""
    from secondbrain.connectors import github as gh

    sleeps: list[float] = []
    monkeypatch.setattr(gh, "time", MagicMock(sleep=sleeps.append))

    call_log = []

    class _FakeSess:
        def get(self, url, **kw):
            call_log.append(url)
            r = MagicMock()
            r.headers = {"Retry-After": "1"}
            r.status_code = 429 if len(call_log) == 1 else 200
            r.json = lambda: [] if r.status_code == 200 else None
            return r

    # Monkeypatch the helper directly so we don't actually sleep.
    import secondbrain.connectors as conn_pkg
    monkeypatch.setattr(conn_pkg.time, "sleep", sleeps.append)
    s = _FakeSess()
    list(gh.GitHubConnector()._iter_repos(s))
    # First call returned 429 → respect_retry_after slept → retry.
    assert len(call_log) >= 2


def test_gmail_iter_message_ids_retries_on_429(monkeypatch):
    """Same for Gmail."""
    import secondbrain.connectors as conn_pkg
    from secondbrain.connectors import gmail
    sleeps: list[float] = []
    monkeypatch.setattr(conn_pkg.time, "sleep", sleeps.append)

    call_log = []

    class _FakeSess:
        def get(self, url, **kw):
            call_log.append(url)
            r = MagicMock()
            r.headers = {"Retry-After": "1"}
            r.status_code = 429 if len(call_log) == 1 else 200
            r.json = lambda: {"messages": [], "nextPageToken": None}
            return r

    s = _FakeSess()
    list(gmail.GmailConnector()._iter_message_ids(s, "is:unread", 10))
    assert len(call_log) >= 2


def test_linear_iter_issues_retries_on_429(monkeypatch):
    """Same for Linear."""
    import secondbrain.connectors as conn_pkg
    from secondbrain.connectors import linear
    sleeps: list[float] = []
    monkeypatch.setattr(conn_pkg.time, "sleep", sleeps.append)

    call_log = []

    class _FakeSess:
        def post(self, url, **kw):
            call_log.append(url)
            r = MagicMock()
            r.headers = {"Retry-After": "1"}
            r.status_code = 429 if len(call_log) == 1 else 200
            r.json = lambda: {
                "data": {
                    "issues": {
                        "nodes": [],
                        "pageInfo": {"hasNextPage": False},
                    },
                },
            }
            return r

    s = _FakeSess()
    list(linear.LinearConnector()._iter_issues(s, cap=5))
    assert len(call_log) >= 2


# ============================ M7 — NFC normalization ================


def test_canonicalize_normalizes_unicode_nfc():
    """Round 18 fix: same name in NFC vs NFD forms must canonicalize
    identically so people-table dedup works across input sources."""
    from secondbrain.people import canonicalize
    nfc = "José"   # José as single codepoint
    nfd = "José"  # José as e + combining acute
    assert canonicalize(nfc) == canonicalize(nfd)
    # And casefold handles ß correctly.
    assert canonicalize("STRAßE") == canonicalize("strasse")


def test_alias_dedup_across_nfc_nfd(fresh_db):
    """Round 18 fix: add_alias uses canonicalize so adding the same
    name in NFC and NFD forms doesn't create two rows."""
    from secondbrain import people as pm

    pid = pm.upsert_person(fresh_db, display_name="José")
    pm.add_alias(fresh_db, pid, "José")  # NFC
    pm.add_alias(fresh_db, pid, "José")  # NFD — same person
    n = fresh_db.execute(
        "SELECT COUNT(*) AS n FROM person_aliases WHERE person_id = ?",
        (pid,),
    ).fetchone()["n"]
    assert n == 1


# ============================ M8 — Config() no real dir =============


def test_config_does_not_create_user_data_dir_on_instantiation(
    tmp_path, monkeypatch,
):
    """Round 18 fix: bare Config() must NOT call mkdir on the real
    user data directory. Tests that don't pass an explicit data_dir
    were silently materializing ~/second-brain/."""
    import secondbrain.config as cfg_mod

    # Spy on platformdirs.user_data_path to confirm we don't pass
    # ensure_exists=True.
    captured: list[dict] = []
    real_user_data_path = cfg_mod.user_data_path

    def spy_user_data_path(*args, **kwargs):
        captured.append({"args": args, "kwargs": kwargs})
        return real_user_data_path(*args, **kwargs)

    monkeypatch.setattr(cfg_mod, "user_data_path", spy_user_data_path)
    cfg_mod.app_data_dir()
    assert captured, "user_data_path was not called"
    assert captured[0]["kwargs"].get("ensure_exists") is False, (
        f"ensure_exists must be False; got {captured[0]['kwargs']!r}"
    )


# ============================ L9 — AppleScript escape ==============


def test_applescript_escape_handles_quotes_and_backslashes():
    """Round 18 fix: \\" not '' for AppleScript double-quote escape."""
    from secondbrain.notify import _applescript_escape

    # `"` becomes `\"` (literal backslash + quote).
    assert _applescript_escape('hello "world"') == r'hello \"world\"'
    # `\` becomes `\\`.
    assert _applescript_escape("a\\b") == "a\\\\b"
    # And both together — backslashes escaped first, then quotes.
    assert _applescript_escape('a"\\b') == 'a\\"\\\\b'


# ============================ L10 — iMessage conn leak =============


def test_imessage_connector_closes_conn_on_consumer_exception(
    tmp_path, monkeypatch,
):
    """Round 18 fix: a downstream exception during yield must not
    leak the SQLite connection."""
    import sqlite3 as sq

    from secondbrain.config import Config
    from secondbrain.connectors.imessage import IMessageConnector

    cfg = Config(data_dir=tmp_path / "data")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    chat_db = tmp_path / "fake_chat.db"
    conn = sq.connect(str(chat_db))
    conn.executescript("""
        CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
        CREATE TABLE chat (
            ROWID INTEGER PRIMARY KEY, display_name TEXT, style INTEGER
        );
        CREATE TABLE chat_handle_join (chat_id INTEGER, handle_id INTEGER);
        CREATE TABLE message (
            ROWID INTEGER PRIMARY KEY, text TEXT, attributedBody BLOB,
            is_from_me INTEGER, date INTEGER,
            cache_has_attachments INTEGER, handle_id INTEGER
        );
        CREATE TABLE chat_message_join (
            chat_id INTEGER, message_id INTEGER
        );
        INSERT INTO chat (ROWID, display_name, style)
            VALUES (1, 'test', 43);
        INSERT INTO message (
            ROWID, text, is_from_me, date, cache_has_attachments,
            handle_id
        ) VALUES (1, 'hi there', 0, 0, 0, NULL);
        INSERT INTO chat_message_join (chat_id, message_id) VALUES (1, 1);
    """)
    conn.commit()
    conn.close()
    cfg.imessage_db_path = str(chat_db)

    # Track sqlite3.connect calls so we can verify close().
    real_connect = sqlite3.connect
    opened_conns: list = []

    def tracking_connect(*args, **kwargs):
        c = real_connect(*args, **kwargs)
        opened_conns.append(c)
        return c

    monkeypatch.setattr(
        "secondbrain.connectors.imessage.sqlite3.connect",
        tracking_connect,
    )

    # Drain just one item then break — simulating a downstream abort.
    gen = IMessageConnector().fetch(cfg)
    try:
        next(gen)
    except StopIteration:
        pass
    gen.close()  # forces finalization

    # The temp-copy connection must be closed.
    # On Windows, calling .execute on a closed conn raises ProgrammingError;
    # on Linux it raises OperationalError. Either confirms close.
    for c in opened_conns:
        with pytest.raises(sqlite3.Error):
            c.execute("SELECT 1")


# ============================ L12 — synthesis set ordering ==========


def test_walk_cluster_member_ids_list_materialized():
    """Round 18 fix: source check that _walk_cluster materialises
    the member_ids set into a list once before passing twice. The
    earlier code expanded the set twice in one expression which is
    safe-by-coincidence in CPython but brittle to refactor."""
    import inspect

    from secondbrain import synthesis
    src = inspect.getsource(synthesis._walk_cluster)
    assert "member_ids_list" in src, src
    # Both expansions reference the list, not the set.
    assert "*member_ids_list, *member_ids_list" in src


# ============================ L13 — digest ValueError =============


def test_digest_send_handles_value_error(tmp_cfg, monkeypatch):
    """Round 18 fix: ValueError from a malformed EmailMessage must
    be caught + recorded as a non-fatal failure, not crash the job."""
    from secondbrain import digest as dg

    monkeypatch.setenv("SECONDBRAIN_SMTP_PASSWORD", "x")
    tmp_cfg.digest_enabled = True
    tmp_cfg.digest_smtp_user = "u@example.com"
    tmp_cfg.digest_smtp_host = "smtp.example.com"
    tmp_cfg.digest_smtp_port = 587
    tmp_cfg.digest_to = "ben@example.com"

    monkeypatch.setattr(
        dg, "build_email", lambda *a, **kw: (MagicMock(), 1, 2),
    )
    monkeypatch.setattr(dg, "last_digest_sent_at", lambda c: None)
    monkeypatch.setattr(dg, "_record_run", lambda *a, **kw: None)

    class _FakeSMTP:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def ehlo(self):
            pass

        def starttls(self):
            pass

        def login(self, u, p):
            pass

        def send_message(self, msg):
            raise ValueError("comma-separated To: fails RFC 5322")

    monkeypatch.setattr(dg.smtplib, "SMTP", _FakeSMTP)
    fake_conn = MagicMock()
    ok, err = dg.send_digest(tmp_cfg, fake_conn)
    assert ok is False
    assert "ValueError" in err


# ============================ smoke ================================


def test_round18_touched_modules_still_import():
    """Belt-and-suspenders: all the touched modules import cleanly."""
    from secondbrain import digest, notifications, notify, people, synthesis, timeline  # noqa: F401
    from secondbrain.connectors import (  # noqa: F401
        github,
        gmail,
        imessage,
        linear,
        pocket,
        readwise,
    )
