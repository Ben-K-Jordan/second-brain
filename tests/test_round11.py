"""Round 11 — hardening + UX polish on the round-10 audit work.

Specific fixes covered:
  - Classifier audit-logs every run (gap in #6)
  - Cold-start voice fallback applies to meeting_thanks too (#5 gap)
  - IMAP health check is shape-only by default (#9 over-eager)
  - /audit rows link to source/draft when applicable (#6 UX)
  - Nav has urgent-style badge for stale health failures (#9 visibility)
  - /tasks page surfaces recipient + due_hint from round-9-C extractor
  - /health/system layout slug renamed to 'diagnostics' to avoid
    collision with Personal/Health (Oura) nav entry
"""

from __future__ import annotations

import json
import time
from unittest.mock import patch

from secondbrain import ai_audit, email_assist, health_checks, meeting_thanks

# ============================ Classifier audit hook ==================

def test_classifier_audit_logs_on_local_fallback(fresh_db, tmp_cfg, monkeypatch):
    """Every classifier run writes an ai_actions row, even when only
    the local-LLM path runs."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from secondbrain import local_llm
    fake = local_llm.LocalCompletion(
        text="urgent", model="llama3.1",
        prompt_tokens=100, completion_tokens=2,
    )
    with patch.object(local_llm, "is_available", return_value=True), \
         patch.object(local_llm, "complete", return_value=fake):
        result = email_assist._default_classifier(
            "boss@x.com", "URGENT: budget", "We need to talk Q3.",
            tmp_cfg, conn=fresh_db, file_id=42,
        )
    assert result.get("label") == "urgent"
    actions = ai_audit.recent(fresh_db, kind="classify")
    assert len(actions) == 1
    a = actions[0]
    assert a.feature == "email_triage"
    assert a.status == "fallback_local"
    assert a.file_id == 42
    assert "urgent" in a.summary


def test_classifier_audit_logs_when_no_provider_available(
    fresh_db, tmp_cfg, monkeypatch,
):
    """Classifier with neither Anthropic nor local available logs
    a 'no_provider' / similar status row + returns {}."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from secondbrain import local_llm
    with patch.object(local_llm, "is_available", return_value=False):
        result = email_assist._default_classifier(
            "x@y", "subject", "body of email",
            tmp_cfg, conn=fresh_db, file_id=1,
        )
    assert result == {}
    actions = ai_audit.recent(fresh_db, kind="classify")
    assert len(actions) == 1
    # Status reflects "couldn't classify".
    assert actions[0].status in (
        "no_provider", "api_error", "budget_exceeded", "parse_error",
    )


def test_classify_one_threads_conn_to_classifier(fresh_db, tmp_cfg):
    """classify_one passes conn + file_id to the classifier so the
    audit log gets file linkage."""
    captured: dict = {}

    def _stub(from_, subject, body, cfg, *, conn=None, file_id=None):
        captured["conn"] = conn
        captured["file_id"] = file_id
        return {"label": "urgent", "confidence": 0.9}

    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at) "
        "VALUES ('imap://msgid/x', ?, 1, 'url', ?)",
        (time.time(), time.time()),
    )
    fid = fresh_db.execute("SELECT id FROM files").fetchone()["id"]
    fresh_db.execute(
        "INSERT INTO chunks(file_id, chunk_index, text) "
        "VALUES (?, 0, '# subject\\nFrom: boss\\nFolder: INBOX\\n\\nbody')",
        (fid,),
    )
    fresh_db.commit()
    email_assist.classify_one(
        fresh_db, fid, cfg=tmp_cfg, classifier=_stub,
    )
    assert captured["conn"] is fresh_db
    assert captured["file_id"] == fid


def test_classify_one_falls_back_to_legacy_signature(fresh_db, tmp_cfg):
    """Old-style classifier stubs (no conn/file_id kwargs) still
    work via the TypeError fallback."""
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at) "
        "VALUES ('imap://msgid/legacy', ?, 1, 'url', ?)",
        (time.time(), time.time()),
    )
    fid = fresh_db.execute("SELECT id FROM files").fetchone()["id"]
    fresh_db.execute(
        "INSERT INTO chunks(file_id, chunk_index, text) "
        "VALUES (?, 0, '# subject\\nFrom: x\\nFolder: INBOX\\n\\nbody')",
        (fid,),
    )
    fresh_db.commit()
    classification = email_assist.classify_one(
        fresh_db, fid, cfg=tmp_cfg,
        classifier=lambda f, s, b, c: {"label": "newsletter"},
    )
    assert classification is not None
    assert classification.label == "newsletter"


# ============================ Cold-start voice for thanks ============

def test_meeting_thanks_uses_default_voice_when_none_set(
    fresh_db, tmp_cfg,
):
    """Thanks drafter on a fresh install should use the curated
    bootstrap profile, not fall through to neutral-tone."""
    captured: dict = {}

    def _drafter(*, prompt, cfg):
        captured["prompt"] = prompt
        return {
            "primary": "Thanks Sarah! — Ben",
            "alternative": "",
            "reasoning": "", "confidence": 0.7, "open_questions": [],
        }

    # Seed minimal meeting_thanks row in 'ready' state with user context.
    meeting_thanks._ensure_schema(fresh_db)
    fresh_db.execute(
        "INSERT INTO meeting_thanks"
        "(event_id, event_title, starts_at, ends_at, attendees_json, "
        " transcript_path, user_context, status, created_at, updated_at) "
        "VALUES ('e1', 'Coffee w/ Sarah', ?, ?, ?, NULL, ?, 'ready', ?, ?)",
        (
            time.time() - 3600, time.time() - 1800,
            json.dumps(["sarah@external.com"]),
            "Talked about onboarding.",
            time.time(), time.time(),
        ),
    )
    fresh_db.commit()
    mt_row = fresh_db.execute(
        "SELECT id FROM meeting_thanks",
    ).fetchone()
    mt_id = int(mt_row["id"])
    # No voice profile saved yet → bootstrap default should kick in.
    out = meeting_thanks.generate_thanks_draft(
        fresh_db, tmp_cfg, mt_id, drafter=_drafter,
    )
    assert out is not None
    # Bootstrap profile's avoided_phrases should appear in the prompt
    # (proving the cold-start fallback ran rather than the
    # "no voice profile yet" placeholder).
    assert "I hope this email finds you well" in captured["prompt"]
    assert "Best regards," in captured["prompt"]


# ============================ IMAP health check shape-only ===========

def test_imap_check_shape_only_no_network_call(tmp_cfg, monkeypatch):
    """Shape-only path: with all three (host / user / pwd) set,
    returns ok=True without trying to log in over the network."""
    monkeypatch.setenv("SECONDBRAIN_IMAP_PASSWORD", "x")
    tmp_cfg.imap_host = "imap.example.com"
    tmp_cfg.imap_username = "me@example.com"
    # Sanity: if it tried to actually connect, it would fail or hang.
    # We patch IMAP4_SSL just in case to detect a regression.
    import imaplib
    with patch.object(imaplib, "IMAP4_SSL") as mock_imap:
        ok, _err, extra = health_checks.check_imap(tmp_cfg)
    assert ok is True
    assert extra["verified"] == "shape-only"
    mock_imap.assert_not_called()


def test_imap_check_network_path_actually_connects(tmp_cfg, monkeypatch):
    """network=True path makes the real connection. Stub IMAP4_SSL
    to verify we DO call it."""
    monkeypatch.setenv("SECONDBRAIN_IMAP_PASSWORD", "x")
    tmp_cfg.imap_host = "imap.example.com"
    tmp_cfg.imap_username = "me@example.com"
    import imaplib
    mock_inst = type("M", (), {
        "__enter__": lambda self: self,
        "__exit__": lambda self, *a: None,
        "login": lambda self, u, p: None,
    })()
    with patch.object(imaplib, "IMAP4_SSL", return_value=mock_inst) as m:
        ok, _err, extra = health_checks.check_imap(
            tmp_cfg, network=True,
        )
    assert ok is True
    assert extra["verified"] == "live-login"
    m.assert_called_once()


def test_imap_check_unconfigured_is_ok(tmp_cfg):
    """No host = unconfigured = not failing. (Don't nag about
    integrations the user hasn't set up.)"""
    tmp_cfg.imap_host = ""
    ok, _err, extra = health_checks.check_imap(tmp_cfg)
    assert ok is True
    assert extra["configured"] is False


def test_run_all_uses_shape_only_by_default(fresh_db, tmp_cfg, monkeypatch):
    """The hourly daemon call path must NOT trigger live network
    calls (auth-log spam)."""
    monkeypatch.setenv("SECONDBRAIN_IMAP_PASSWORD", "x")
    tmp_cfg.imap_host = "imap.example.com"
    tmp_cfg.imap_username = "me@example.com"
    import imaplib
    with patch.object(imaplib, "IMAP4_SSL") as mock_imap:
        health_checks.run_all(fresh_db, tmp_cfg)
    mock_imap.assert_not_called()


def test_run_all_network_true_does_live_check(fresh_db, tmp_cfg, monkeypatch):
    monkeypatch.setenv("SECONDBRAIN_IMAP_PASSWORD", "x")
    tmp_cfg.imap_host = "imap.example.com"
    tmp_cfg.imap_username = "me@example.com"
    import imaplib
    mock_inst = type("M", (), {
        "__enter__": lambda self: self,
        "__exit__": lambda self, *a: None,
        "login": lambda self, u, p: None,
    })()
    with patch.object(imaplib, "IMAP4_SSL", return_value=mock_inst) as m:
        health_checks.run_all(fresh_db, tmp_cfg, network=True)
    m.assert_called_once()


# ============================ Nav-counts: health badge ===============

def test_api_nav_counts_includes_health_when_stale(
    monkeypatch, tmp_path, fake_embedder,
):
    """When a health check is failing, /api/nav-counts returns
    health > 0 + urgent.health=True."""
    # Seed a stale-failure row directly so we don't need to run
    # the full check pipeline (which depends on env).
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
    # Open the read-only conn the dashboard caches by hitting any page.
    client.get("/")
    # Now seed a failing health row directly via the writer conn.
    from secondbrain.db import connect, init_schema
    conn = connect(cfg.db_path)
    init_schema(conn, fake_embedder.dim, fake_embedder.name)
    health_checks._ensure_schema(conn)
    conn.execute(
        "INSERT INTO health_checks"
        "(name, last_checked_at, last_ok_at, ok, error) "
        "VALUES ('google_calendar', ?, ?, 0, 'token revoked')",
        (time.time(), time.time() - 5 * 86400),
    )
    conn.commit()
    conn.close()
    # Need a fresh client to bust the read-only conn cache.
    app2 = create_app()
    client2 = TestClient(app2)
    r = client2.get("/api/nav-counts")
    assert r.status_code == 200
    data = r.json()
    assert data["health"] >= 1
    assert data["urgent"]["health"] is True


# ============================ Diagnostics page renders ===============

def test_diagnostics_page_renders_without_crashing(
    monkeypatch, tmp_path, fake_embedder,
):
    """The /health/system page should render even on an empty DB."""
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
    r = client.get("/health/system")
    assert r.status_code == 200
    # Title in H1 should say "Diagnostics" (round-13 rename).
    assert "Diagnostics" in r.text


# ============================ Tasks page recipient surface ===========

def test_tasks_page_renders_recipient_when_known(
    monkeypatch, tmp_path, fake_embedder,
):
    """A task with recipient_person_id → renders the person link."""
    from fastapi.testclient import TestClient

    from secondbrain import people as people_mod
    from secondbrain import tasks
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
    create_app()  # warm config / paths; we seed via fresh conn below
    # Seed via a fresh writer conn (the dashboard's cached one isn't
    # used until a route hits it).
    from secondbrain.db import connect, init_schema
    conn = connect(cfg.db_path)
    init_schema(conn, fake_embedder.dim, fake_embedder.name)
    pid = people_mod.upsert_person(
        conn, display_name="Sarah Chen",
    )
    tasks._ensure_round9c_columns(conn)
    conn.execute(
        "INSERT INTO tasks(text, text_lower, source_path, source_title, "
        " status, created_at, recipient_person_id, due_hint) "
        "VALUES (?, ?, 'manual', '(typed)', 'open', ?, ?, ?)",
        ("send Sarah the deck", "send sarah the deck", time.time(),
         pid, "Friday"),
    )
    conn.commit()
    conn.close()
    # Fresh app to bust the cached conn.
    app2 = create_app()
    client2 = TestClient(app2)
    r = client2.get("/tasks")
    assert r.status_code == 200
    assert "Sarah Chen" in r.text
    assert "Friday" in r.text
    # Person link present.
    assert f"/person?id={pid}" in r.text
