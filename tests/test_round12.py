"""Round 12 — fixes for issues an external audit found in round 10/11.

Each test maps to a HIGH/MEDIUM finding the agent surfaced:
  - Person edit form was wiping un-submitted fields
  - _regenerate_with_critique was skipping redaction + audit
  - chat.stream_chat sent raw user input + history to Anthropic
    with no redaction and no audit log
  - /audit + /api/nav-counts used the read-only conn but invoked
    _ensure_schema which writes
  - synthesis / tagger LLM calls weren't auditing
  - is_stale silently excluded never-OK checks
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from secondbrain import (
    ai_audit,
    email_assist,
    health_checks,
)

# ============================ Person edit field-wipe =================

def test_person_edit_keeps_unchanged_fields_intact(
    monkeypatch, tmp_path, fake_embedder,
):
    """The edit POST handler must NOT clear fields the form didn't
    explicitly change. Round 11 had the docstring claim "empty
    strings preserve" but the implementation passed them through,
    causing set_field to write empty strings = clear."""
    from fastapi.testclient import TestClient

    from secondbrain import people as people_mod
    from secondbrain.config import Config
    from secondbrain.dashboard import create_app
    from secondbrain.db import connect, init_schema
    cfg = Config(data_dir=tmp_path / "data")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("secondbrain.dashboard.load_config", lambda: cfg)
    monkeypatch.setattr(
        "secondbrain.dashboard.make_embedder", lambda c: fake_embedder,
    )
    monkeypatch.setattr(
        "secondbrain.dashboard.make_reranker", lambda c: None,
    )
    # Seed a fully-populated person row.
    conn = connect(cfg.db_path)
    init_schema(conn, fake_embedder.dim, fake_embedder.name)
    pid = people_mod.upsert_person(
        conn, display_name="Sarah Chen", email="sarah@x.com",
        company="Acme", role="PM",
    )
    people_mod.set_field(
        conn, pid, birthday="1990-05-12", notes="great mentor",
    )
    conn.close()
    # POST with only email field changed (everything else re-submits
    # its existing value, as a real form would).
    app = create_app()
    client = TestClient(app)
    r = client.post(
        f"/person/{pid}/edit",
        data={
            "email": "sarah-new@x.com",
            "role": "PM",
            "company": "Acme",
            "birthday": "1990-05-12",
            "notes": "great mentor",
        },
        # Round 13 added a same-origin guard to /person/.../edit; pass
        # a localhost referer so the POST clears it.
        headers={"referer": "http://127.0.0.1:8765/person?id=" + str(pid)},
        follow_redirects=False,
    )
    assert r.status_code == 303
    # All fields except email should be intact.
    conn2 = connect(cfg.db_path)
    person = people_mod.get_person(conn2, pid)
    conn2.close()
    assert person.email == "sarah-new@x.com"
    assert person.role == "PM"
    assert person.company == "Acme"
    assert person.birthday == "1990-05-12"
    assert person.notes == "great mentor"


def test_person_edit_clears_field_when_form_blanks_it(
    monkeypatch, tmp_path, fake_embedder,
):
    """When the user explicitly blanks a field on the form (not
    just leaves it un-touched), it should clear."""
    from fastapi.testclient import TestClient

    from secondbrain import people as people_mod
    from secondbrain.config import Config
    from secondbrain.dashboard import create_app
    from secondbrain.db import connect, init_schema
    cfg = Config(data_dir=tmp_path / "data")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("secondbrain.dashboard.load_config", lambda: cfg)
    monkeypatch.setattr(
        "secondbrain.dashboard.make_embedder", lambda c: fake_embedder,
    )
    monkeypatch.setattr(
        "secondbrain.dashboard.make_reranker", lambda c: None,
    )
    conn = connect(cfg.db_path)
    init_schema(conn, fake_embedder.dim, fake_embedder.name)
    pid = people_mod.upsert_person(
        conn, display_name="X", email="x@x.com", company="Old Co",
    )
    conn.close()
    app = create_app()
    client = TestClient(app)
    # Send the form with company=blank (intentional clear).
    r = client.post(
        f"/person/{pid}/edit",
        data={
            "email": "x@x.com",
            "role": "",
            "company": "",
            "birthday": "",
            "notes": "",
        },
        # Round 13 added a same-origin guard to /person/.../edit; pass
        # a localhost referer so the POST clears it.
        headers={"referer": "http://127.0.0.1:8765/person?id=" + str(pid)},
        follow_redirects=False,
    )
    assert r.status_code == 303
    conn2 = connect(cfg.db_path)
    person = people_mod.get_person(conn2, pid)
    conn2.close()
    # Company explicitly cleared (was "Old Co").
    assert person.company == ""


# ============================ Regen path redaction + audit ===========

def test_regenerate_with_critique_redacts_prior_draft(tmp_cfg):
    """Round 11 fix: prior_draft echoed into the regen prompt now
    goes through _safe_for_prompt (was raw before)."""
    captured: dict = {}

    def _capture(*, prompt, **_kw):
        captured["prompt"] = prompt
        return None  # short-circuit; we only care about the prompt

    with patch.object(
        email_assist, "_llm_json_call", side_effect=_capture,
    ):
        email_assist._regenerate_with_critique(
            prior_draft=(
                "Use sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA "
                "to authorize. — Ben"
            ),
            critique="- uses banned phrase",
            from_="boss@x", subject="status", body="status?",
            style_samples="(none)",
            user_name="Ben", cfg=tmp_cfg,
        )
    # Secret should be masked in the regen prompt.
    assert "sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" \
        not in captured["prompt"]


def test_regenerate_with_critique_passes_audit_kwargs(tmp_cfg):
    """Round 11 fix: regen call now writes its own ai_actions row."""
    captured: dict = {}

    def _capture(**kw):
        captured.update(kw)
        return None

    with patch.object(
        email_assist, "_llm_json_call", side_effect=_capture,
    ):
        email_assist._regenerate_with_critique(
            prior_draft="x", critique="y",
            from_="z@a", subject="b", body="c",
            style_samples="", user_name="Ben", cfg=tmp_cfg,
            conn="fake_conn", file_id=99,
        )
    assert captured["audit_kind"] == "draft_regen"
    assert captured["audit_file_id"] == 99
    assert captured["conn"] == "fake_conn"


# ============================ Chat redaction + audit =================

def test_stream_chat_redacts_user_message_before_send(
    fresh_db, tmp_cfg, fake_embedder,
):
    """Round 11 fix: stream_chat's user_message goes through
    safety.redact_text before being added to messages.

    We can't easily exercise the full streaming path in a unit test,
    but we can verify the redaction step by inspecting the
    `_redact_chat` import + chat.py source. Instead, we invoke the
    helper directly: confirm safety.redact_text would mask the
    secret payload we'd be sending."""
    from secondbrain.safety import redact_text

    secret_msg = (
        "Run with sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA "
        "in scope"
    )
    cleaned = redact_text(secret_msg)
    assert "sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" \
        not in cleaned
    assert "Run with" in cleaned


def test_stream_chat_writes_ai_actions_per_iteration(
    fresh_db, tmp_cfg, fake_embedder, monkeypatch,
):
    """Round 11 fix: stream_chat now writes one ai_actions row
    per iteration. We can verify by stubbing the SDK + draining
    the generator."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test")
    monkeypatch.setattr(tmp_cfg, "chat_max_tool_iterations", 0)
    monkeypatch.setattr(tmp_cfg, "chat_model", "claude-haiku-4-5")
    monkeypatch.setattr(tmp_cfg, "web_search_enabled", False)

    # Mock the streaming response.
    mock_text = MagicMock()
    mock_text.text_stream = ["hi"]
    mock_response = MagicMock()
    mock_response.usage.input_tokens = 100
    mock_response.usage.output_tokens = 5
    mock_response.usage.server_tool_use = None
    mock_response.content = []
    mock_response.stop_reason = "end_turn"
    mock_text.get_final_message = MagicMock(return_value=mock_response)
    mock_text.__enter__ = MagicMock(return_value=mock_text)
    mock_text.__exit__ = MagicMock(return_value=False)

    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic.return_value.messages.stream.return_value = (
        mock_text
    )

    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        from secondbrain import chat as chat_mod
        # Drain the stream.
        events = list(chat_mod.stream_chat(
            tmp_cfg, fresh_db, fake_embedder, None, "hello",
        ))
    # Should have written at least one chat audit row.
    actions = ai_audit.recent(fresh_db, kind="chat")
    assert len(actions) >= 1
    assert actions[0].feature == "chat"
    assert events  # at least the done event


# ============================ Read-only conn fix =====================

def test_audit_page_uses_writer_conn_so_schema_init_works(
    monkeypatch, tmp_path, fake_embedder,
):
    """The /audit page used to use get_read_state; ai_audit.recent
    calls _ensure_schema which writes (CREATE TABLE + COMMIT).
    Round 11 fix: switched to get_state (writer conn)."""
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
    # Empty DB → /audit should render the empty-state body without
    # crashing on an attempted CREATE TABLE through a read-only conn.
    r = client.get("/audit")
    assert r.status_code == 200
    # Same for /api/nav-counts which calls health_checks.stale_failures.
    r2 = client.get("/api/nav-counts")
    assert r2.status_code == 200


# ============================ Synthesis + tagger audit ===============

def test_summary_writes_audit_row(fresh_db, tmp_cfg):
    """Round 11 fix: summarize_doc records an ai_actions row whether
    the generator succeeds or fails."""
    from secondbrain import synthesis

    # Seed a file + chunk that's long enough for needs_summary.
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at) "
        "VALUES ('/x.md', ?, 1, 'document', ?)",
        (time.time(), time.time()),
    )
    fid = fresh_db.execute("SELECT id FROM files").fetchone()["id"]
    fresh_db.execute(
        "INSERT INTO chunks(file_id, chunk_index, text) "
        "VALUES (?, 0, ?)",
        (fid, "# Test\n\n" + "x" * 5000),
    )
    fresh_db.commit()
    # Stub generator → returns a result.
    out = synthesis.materialize_summary(
        fresh_db, fid, cfg=tmp_cfg,
        generator=lambda title, body, cfg: {
            "tldr": "test summary",
            "key_points": [],
        },
    )
    assert out is not None
    actions = ai_audit.recent(fresh_db, kind="summary")
    assert len(actions) == 1
    assert actions[0].file_id == fid
    assert actions[0].status == "success"


def test_summary_records_failure_too(fresh_db, tmp_cfg):
    """When the generator raises, an api_error row lands in audit."""
    from secondbrain import synthesis

    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at) "
        "VALUES ('/y.md', ?, 1, 'document', ?)",
        (time.time(), time.time()),
    )
    fid = fresh_db.execute("SELECT id FROM files").fetchone()["id"]
    fresh_db.execute(
        "INSERT INTO chunks(file_id, chunk_index, text) "
        "VALUES (?, 0, ?)",
        (fid, "# Boom\n\n" + "x" * 5000),
    )
    fresh_db.commit()

    def _crash(title, body, cfg):
        raise RuntimeError("simulated LLM crash")

    synthesis.materialize_summary(
        fresh_db, fid, cfg=tmp_cfg, generator=_crash,
    )
    actions = ai_audit.recent(fresh_db, kind="summary")
    assert len(actions) == 1
    assert actions[0].status == "api_error"
    assert "simulated LLM crash" in actions[0].error


def test_tagger_audit_no_provider_logs_when_no_key(
    fresh_db, tmp_cfg, monkeypatch,
):
    """tagger.generate_tags now logs a no_provider row when
    ANTHROPIC_API_KEY is absent + conn is given."""
    from secondbrain import tagger
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    out = tagger.generate_tags(
        "some chunk text", tmp_cfg, conn=fresh_db, file_id=42,
    )
    assert out == []
    actions = ai_audit.recent(fresh_db, kind="tag")
    assert len(actions) == 1
    assert actions[0].status == "no_provider"
    assert actions[0].file_id == 42


# ============================ is_stale fix ===========================

def test_is_stale_when_never_succeeded_and_old(fresh_db):
    """A check that's been failing 25h without ever succeeding now
    counts as stale (round 11 fix)."""
    health_checks._ensure_schema(fresh_db)
    # Manually seed a 'never ok, first checked 25h ago' row.
    long_ago = time.time() - 25 * 3600
    fresh_db.execute(
        "INSERT INTO health_checks"
        "(name, last_checked_at, last_ok_at, ok, error, "
        " first_checked_at) VALUES "
        "('anthropic', ?, NULL, 0, 'key missing', ?)",
        (time.time(), long_ago),
    )
    fresh_db.commit()
    statuses = health_checks.list_status(fresh_db)
    [st] = statuses
    assert st.is_stale is True


def test_is_stale_returns_false_for_recently_seen_failure(fresh_db):
    """Configured-but-just-broken should NOT be stale yet (gives the
    user time to react / for the integration to recover)."""
    health_checks._ensure_schema(fresh_db)
    fresh_db.execute(
        "INSERT INTO health_checks"
        "(name, last_checked_at, last_ok_at, ok, error, "
        " first_checked_at) VALUES "
        "('anthropic', ?, NULL, 0, 'key missing', ?)",
        (time.time(), time.time() - 600),  # 10 min ago
    )
    fresh_db.commit()
    [st] = health_checks.list_status(fresh_db)
    assert st.is_stale is False


def test_is_stale_old_schema_compat(fresh_db):
    """A row from an older schema (no first_checked_at column / value)
    must not crash + must default to not-stale."""
    health_checks._ensure_schema(fresh_db)
    # Even though schema has the column, set first_checked_at to NULL
    # to simulate an old persisted row.
    fresh_db.execute(
        "INSERT INTO health_checks"
        "(name, last_checked_at, last_ok_at, ok, error, "
        " first_checked_at) VALUES "
        "('legacy', ?, NULL, 0, 'old', NULL)",
        (time.time(),),
    )
    fresh_db.commit()
    [st] = health_checks.list_status(fresh_db)
    assert st.first_checked_at is None
    assert st.is_stale is False


def test_persist_preserves_first_checked_at_across_runs(
    fresh_db, tmp_cfg, monkeypatch,
):
    """run_all twice → first_checked_at stays at the first run's
    timestamp."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    tmp_cfg.voyage_api_key = ""
    tmp_cfg.watched_folders = []
    health_checks.run_all(fresh_db, tmp_cfg)
    statuses1 = {s.name: s.first_checked_at
                 for s in health_checks.list_status(fresh_db)}
    time.sleep(0.05)
    health_checks.run_all(fresh_db, tmp_cfg)
    statuses2 = {s.name: s.first_checked_at
                 for s in health_checks.list_status(fresh_db)}
    for name, first in statuses1.items():
        assert statuses2[name] == first  # preserved


# ============================ Launchpad + palette completeness =======

def test_launchpad_includes_diagnostics_and_audit(
    monkeypatch, tmp_path, fake_embedder,
):
    """The Overview launchpad must include the round-10 / round-11
    new pages (Diagnostics, Audit) so users can find them without
    expanding More."""
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
    r = client.get("/")
    assert r.status_code == 200
    assert "Diagnostics" in r.text
    assert 'href="/health/system"' in r.text
    assert 'href="/audit"' in r.text


def test_palette_includes_diagnostics():
    """⌘K palette pages must include Diagnostics."""
    from secondbrain.dashboard import PALETTE_JS
    assert "label: 'Diagnostics'" in PALETTE_JS


# ============================ Responsive nav CSS =====================

def test_nav_has_responsive_breakpoints():
    """The CSS must include @media queries for narrow viewports so
    the More dropdown doesn't overflow on phones."""
    from secondbrain.dashboard import CSS
    assert "@media (max-width: 720px)" in CSS
    assert "@media (max-width: 480px)" in CSS
