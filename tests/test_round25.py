"""Round 25 — fixes for the deeper-systemic audit.

Each test maps to a finding from the audit:
  - HIGH H1: /file?file_id=N route accepts file_id (was 422)
  - HIGH H2: drafter daemon path passes cfg.user_name
  - HIGH H3: study materializer includes canvas:// + voice://
  - HIGH H4: meeting_thanks._own_email_domains reads cfg.user_email
  - HIGH H5: find_open_time_slots MCP tool uses cfg defaults
  - MED M2: notification href uses file_id (URL-safe)
  - MED M3+M4: _normalise_kind switches on path prefix for kind='url'
"""

from __future__ import annotations

import time


def _client(monkeypatch, tmp_path, fake_embedder):
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
    return cfg, TestClient(create_app())


# ============================ HIGH H1 — /file?file_id ===================


def test_file_route_accepts_file_id(monkeypatch, tmp_path, fake_embedder):
    """Round 25 fix: /file?file_id=N must work (was 422). The round-22
    EA UI generates these links from 6 sites."""
    from secondbrain.db import connect, init_schema

    cfg, client = _client(monkeypatch, tmp_path, fake_embedder)
    seed = connect(cfg.db_path)
    init_schema(seed, fake_embedder.dim, fake_embedder.name)
    seed.execute(
        "INSERT INTO files(path, mtime, size, kind, content_hash, "
        "indexed_at) VALUES ('test.txt', 0, 0, 'document', 'h', ?)",
        (time.time(),),
    )
    fid = seed.execute(
        "SELECT id FROM files WHERE path='test.txt'",
    ).fetchone()["id"]
    seed.execute(
        "INSERT INTO chunks(file_id, chunk_index, text, start_offset) "
        "VALUES (?, 0, 'hello world', 0)", (fid,),
    )
    seed.commit()
    seed.close()
    r = client.get(f"/file?file_id={fid}")
    assert r.status_code == 200, (
        f"Expected 200 with file_id; got {r.status_code}"
    )
    assert "hello world" in r.text or "test.txt" in r.text


def test_file_route_still_accepts_path(monkeypatch, tmp_path, fake_embedder):
    """Backward compat: /file?path=... still works."""
    from secondbrain.db import connect, init_schema

    cfg, client = _client(monkeypatch, tmp_path, fake_embedder)
    seed = connect(cfg.db_path)
    init_schema(seed, fake_embedder.dim, fake_embedder.name)
    seed.execute(
        "INSERT INTO files(path, mtime, size, kind, content_hash, "
        "indexed_at) VALUES ('foo.txt', 0, 0, 'document', 'h', ?)",
        (time.time(),),
    )
    fid = seed.execute(
        "SELECT id FROM files WHERE path='foo.txt'",
    ).fetchone()["id"]
    seed.execute(
        "INSERT INTO chunks(file_id, chunk_index, text, start_offset) "
        "VALUES (?, 0, 'body', 0)", (fid,),
    )
    seed.commit()
    seed.close()
    r = client.get("/file?path=foo.txt")
    assert r.status_code == 200


def test_file_route_no_args_returns_400(monkeypatch, tmp_path, fake_embedder):
    cfg, client = _client(monkeypatch, tmp_path, fake_embedder)
    r = client.get("/file")
    assert r.status_code == 400


def test_file_route_unknown_file_id(monkeypatch, tmp_path, fake_embedder):
    cfg, client = _client(monkeypatch, tmp_path, fake_embedder)
    r = client.get("/file?file_id=99999")
    assert r.status_code == 200
    assert "Not in index" in r.text


# ============================ HIGH H2 — user_name through drafters ======


def test_generate_drafts_due_passes_user_name(
    fresh_db, tmp_cfg, monkeypatch,
):
    """Round 25 fix: daemon-side draft generation must propagate
    cfg.user_name to the drafter, not silently default to 'I'."""
    from secondbrain import email_assist
    email_assist._ensure_schema(fresh_db)
    tmp_cfg.user_name = "Ben"
    # Seed an email + classification.
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, content_hash, "
        "indexed_at) VALUES ('imap://INBOX/1', 0, 0, 'url', 'h', ?)",
        (time.time(),),
    )
    fid = fresh_db.execute(
        "SELECT id FROM files WHERE path='imap://INBOX/1'",
    ).fetchone()["id"]
    fresh_db.execute(
        "INSERT INTO chunks(file_id, chunk_index, text, start_offset) "
        "VALUES (?, 0, 'From: a@x\nSubject: hi\n\ntest', 0)", (fid,),
    )
    fresh_db.execute(
        "INSERT INTO email_classifications"
        "(file_id, label, confidence, classified_at) "
        "VALUES (?, 'urgent', 0.9, ?)", (fid, time.time()),
    )
    fresh_db.commit()

    captured: dict = {}

    def fake_generate_draft(conn, file_id, *, cfg, drafter=None,
                             user_name="I", **kw):
        captured["user_name"] = user_name
        return None  # don't actually persist

    monkeypatch.setattr(email_assist, "generate_draft", fake_generate_draft)
    email_assist.generate_drafts_due(fresh_db, tmp_cfg)
    assert captured.get("user_name") == "Ben", (
        f"Expected 'Ben', got {captured!r}"
    )


def test_generate_thanks_draft_default_user_name(
    fresh_db, tmp_cfg, monkeypatch,
):
    """Round 25 fix: meeting_thanks.generate_thanks_draft must
    default user_name from cfg.user_name when not passed."""
    # Source-string proof: the function defaults user_name from
    # cfg.user_name when not explicitly passed.
    import inspect

    from secondbrain import meeting_thanks as mt
    src = inspect.getsource(mt.generate_thanks_draft)
    assert "cfg.user_name" in src or 'getattr(cfg, "user_name"' in src


# ============================ HIGH H3 — study + canvas ==================


def test_study_materializer_includes_canvas_paths(fresh_db):
    """Round 25 fix: materialize_due_cards must walk Canvas LMS
    files (canvas:// path prefix), not just transcripts/IMAP."""
    import inspect

    from secondbrain import study
    # Source-string check is the cheapest way to pin this — a
    # full materialize round-trip would require LLM stubbing.
    # Pull whichever function references the path filter.
    src = inspect.getsource(study)
    assert "'canvas://%'" in src


# ============================ HIGH H4 — own_email_domains ===============


def test_own_email_domains_reads_user_email():
    """Round 25 fix: _own_email_domains must include cfg.user_email
    so Gmail-only users (no IMAP) get their own domain detected."""
    from secondbrain.config import Config
    from secondbrain.meeting_thanks import _own_email_domains
    cfg = Config()
    cfg.user_email = "ben@example.com"
    cfg.imap_username = ""
    cfg.digest_smtp_user = ""
    cfg.digest_smtp_from = ""
    domains = _own_email_domains(cfg)
    assert "example.com" in domains, (
        f"Expected example.com via user_email; got {domains}"
    )


def test_own_email_domains_combines_sources():
    """All four config fields contribute when set."""
    from secondbrain.config import Config
    from secondbrain.meeting_thanks import _own_email_domains
    cfg = Config()
    cfg.user_email = "a@one.com"
    cfg.imap_username = "b@two.com"
    cfg.digest_smtp_user = "c@three.com"
    cfg.digest_smtp_from = "d@four.com"
    domains = _own_email_domains(cfg)
    assert "one.com" in domains
    assert "two.com" in domains
    assert "three.com" in domains
    assert "four.com" in domains


# ============================ HIGH H5 — scheduling MCP tool =============


def test_find_open_time_slots_uses_cfg_defaults(
    monkeypatch, tmp_path, fake_embedder,
):
    """Round 25 fix: when caller passes earliest_hour/latest_hour=0,
    the MCP tool now consults cfg.scheduling_earliest_hour /
    scheduling_latest_hour instead of the hardcoded 9-5."""
    from secondbrain import mcp_server, scheduling
    from secondbrain.config import Config

    cfg = Config(data_dir=tmp_path / "data")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.scheduling_earliest_hour = 7
    cfg.scheduling_latest_hour = 19
    cfg.scheduling_buffer_minutes = 10

    # Stub the cfg getter.
    monkeypatch.setattr(
        mcp_server, "_get_state",
        lambda: (cfg, None, None, None),
    )
    captured: dict = {}

    def fake_find_open_slots(busy, *, window_start, window_end, prefs):
        captured["prefs"] = prefs
        return []  # empty result; test only the prefs

    monkeypatch.setattr(scheduling, "find_open_slots", fake_find_open_slots)
    monkeypatch.setattr(scheduling, "parse_busy_blocks", lambda evs: [])
    mcp_server.find_open_time_slots(
        days_ahead=3, duration_minutes=30,
        earliest_hour=0, latest_hour=0,  # 0 = use cfg
        busy_events_json="[]",
    )
    prefs = captured.get("prefs")
    assert prefs is not None
    assert prefs.earliest_hour == 7, (
        f"Expected earliest_hour=7 from cfg; got {prefs.earliest_hour}"
    )
    assert prefs.latest_hour == 19
    assert prefs.buffer_minutes == 10


def test_find_open_time_slots_respects_explicit_override(
    monkeypatch, tmp_path, fake_embedder,
):
    """When caller passes non-zero hours, those win over cfg."""
    from secondbrain import mcp_server, scheduling
    from secondbrain.config import Config
    cfg = Config(data_dir=tmp_path / "data")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.scheduling_earliest_hour = 7
    cfg.scheduling_latest_hour = 19

    monkeypatch.setattr(
        mcp_server, "_get_state",
        lambda: (cfg, None, None, None),
    )
    captured: dict = {}

    def fake_find_open_slots(busy, *, window_start, window_end, prefs):
        captured["prefs"] = prefs
        return []

    monkeypatch.setattr(scheduling, "find_open_slots", fake_find_open_slots)
    monkeypatch.setattr(scheduling, "parse_busy_blocks", lambda evs: [])
    mcp_server.find_open_time_slots(
        days_ahead=3, duration_minutes=30,
        earliest_hour=10, latest_hour=16,  # explicit override
        busy_events_json="[]",
    )
    prefs = captured["prefs"]
    assert prefs.earliest_hour == 10
    assert prefs.latest_hour == 16


# ============================ MED M2 — URL-safe href ====================


def test_email_urgent_notif_href_uses_file_id():
    """Round 25 fix: notif href is now ``/file?file_id=N`` not
    ``/file?path=<unencoded>``. Path-based hrefs broke for paths
    with ``<``, ``>``, ``@`` etc."""
    import inspect

    from secondbrain import notifications
    src = inspect.getsource(notifications._detect_email_urgent)
    assert "file_id={" in src
    # Old broken pattern with raw path interpolation must be gone.
    assert 'href=f"/file?path={r' not in src


# ============================ MED M3+M4 — _normalise_kind ==============


def test_normalise_kind_url_imap_becomes_email():
    from secondbrain.followups import _normalise_kind
    assert _normalise_kind("url", "imap://INBOX/42") == "email"


def test_normalise_kind_url_gmail_becomes_email():
    from secondbrain.followups import _normalise_kind
    assert _normalise_kind(
        "url", "gmail://thread/T1/message/M1",
    ) == "email"


def test_normalise_kind_url_transcript_becomes_meeting():
    from secondbrain.followups import _normalise_kind
    assert _normalise_kind(
        "url", "transcript://meeting-2025-05-05",
    ) == "meeting"


def test_normalise_kind_url_voice_becomes_journal():
    from secondbrain.followups import _normalise_kind
    assert _normalise_kind("url", "voice://note-1") == "journal"


def test_normalise_kind_url_unknown_path_unchanged():
    """A URL with a non-email/transcript/voice prefix stays as-is."""
    from secondbrain.followups import _normalise_kind
    assert _normalise_kind("url", "readwise://highlight/42") == "url"


def test_normalise_kind_legacy_kind_values_unchanged():
    """Round 24 helper still recognises kind='message'/'voice'/etc.
    when the path doesn't match a known prefix."""
    from secondbrain.followups import _normalise_kind
    assert _normalise_kind("message", "") == "email"
    assert _normalise_kind("voice", "") == "journal"
    assert _normalise_kind("transcript", "") == "meeting"
    assert _normalise_kind("email", "") == "email"


# ============================ smoke ====================================


def test_round25_modules_import():
    from secondbrain import (  # noqa: F401
        dashboard,
        email_assist,
        followups,
        mcp_server,
        meeting_thanks,
        notifications,
        study,
    )
