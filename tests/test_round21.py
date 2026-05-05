"""Round 21 — fixes for the round-20 audit.

Each test maps to a finding from the audit:
  - HIGH A1: auto_resolve must filter to user-authored content
  - HIGH A2: validate evidence_file_id is in candidate set
  - HIGH A3: add_followup_with_status returns (id, was_new) tuple;
             extract_and_persist no longer uses time-window
  - HIGH F1: write locks on the 5 new round-20 modules
  - MED A5: triage queue SQL defensive parens
  - MED A6: cadence_overdue uses local-day key
  - MED A7: cadence_user_set flag preserves user clears
  - MED A8: cadence inference more stable (min_contacts raised)
  - MED A9: _followups_section limit alignment
  - MED C1+C2: redact LLM prompts in followups_ops
  - MED I1: dashboard nav for EA pages
  - MED A-config: load_config wires new fields
  - LOW I2: agenda detail page renders add-note UI
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

# ============================ HIGH A1 — sender filter ===================


def test_auto_resolve_skips_incoming_email(fresh_db, tmp_cfg, monkeypatch):
    """A reply from someone else mentioning the topic must NOT
    resolve the user's outgoing followup."""
    from secondbrain import followups, followups_ops

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test")
    tmp_cfg.user_email = "ben@x"

    fid = followups.add_followup(
        fresh_db, direction="outgoing",
        topic="Send Q3 deck",
        description="Send Sarah the Q3 numbers deck",
        person_name="Sarah",
        promised_at=time.time() - 3600,
    )
    # An INCOMING reply from Sarah, not a sent email.
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, content_hash, "
        "indexed_at) VALUES ('reply.eml', 0, 0, 'email', 'h1', ?)",
        (time.time() - 100,),
    )
    rid = fresh_db.execute(
        "SELECT id FROM files WHERE path='reply.eml'",
    ).fetchone()["id"]
    fresh_db.execute(
        "INSERT INTO chunks(file_id, chunk_index, text, start_offset) "
        "VALUES (?, 0, "
        "'From: sarah@y\nTo: ben@x\nSubject: Re: Q3 deck\n\n"
        "Looks great, thanks for the Q3 deck!', 0)",
        (rid,),
    )
    fresh_db.commit()

    # The LLM stub would say "resolved", but we should never reach
    # it because the filter drops the incoming mail.
    monkeypatch.setattr(
        followups_ops, "_llm_check_resolution",
        lambda cfg, **kw: pytest_unreachable(kw),
    )
    n = followups_ops.auto_resolve_from_sent_mail(
        fresh_db, tmp_cfg, hours=24,
    )
    assert n == 0
    assert followups.get(fresh_db, fid).status == "open"


def pytest_unreachable(kw):
    raise AssertionError("LLM should not have been called")


def test_auto_resolve_accepts_journal(fresh_db, tmp_cfg, monkeypatch):
    """Journal entries are always user-authored (no From: filter)."""
    from secondbrain import followups, followups_ops

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test")
    fid = followups.add_followup(
        fresh_db, direction="outgoing",
        topic="Send Q3 deck",
        description="Send Sarah the Q3 numbers deck",
        person_name="Sarah",
        promised_at=time.time() - 3600,
    )
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, content_hash, "
        "indexed_at) VALUES ('j.txt', 0, 0, 'journal', 'h', ?)",
        (time.time() - 100,),
    )
    jid = fresh_db.execute(
        "SELECT id FROM files WHERE path='j.txt'",
    ).fetchone()["id"]
    fresh_db.execute(
        "INSERT INTO chunks(file_id, chunk_index, text, start_offset) "
        "VALUES (?, 0, 'Sent the Q3 deck to Sarah this morning. "
        "Done!', 0)", (jid,),
    )
    fresh_db.commit()

    monkeypatch.setattr(
        followups_ops, "_llm_check_resolution",
        lambda cfg, **kw: {
            "resolved": True, "confidence": 0.9,
            "evidence_file_id": jid,
            "evidence": "Journal entry confirms the user sent it.",
        },
    )
    n = followups_ops.auto_resolve_from_sent_mail(
        fresh_db, tmp_cfg, hours=24,
    )
    assert n == 1
    assert followups.get(fresh_db, fid).status == "resolved"


# ============================ HIGH A2 — fid validation ==================


def test_auto_resolve_rejects_hallucinated_fid(
    fresh_db, tmp_cfg, monkeypatch,
):
    """LLM returning a non-candidate evidence_file_id must be coerced
    to None rather than persisted."""
    from secondbrain import followups, followups_ops

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test")
    tmp_cfg.user_email = "ben@x"
    fid = followups.add_followup(
        fresh_db, direction="outgoing",
        topic="Send deck", description="Send Sarah the deck",
        person_name="Sarah", promised_at=time.time() - 3600,
    )
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, content_hash, "
        "indexed_at) VALUES ('sent.eml', 0, 0, 'email', 'h', ?)",
        (time.time() - 100,),
    )
    sid = fresh_db.execute(
        "SELECT id FROM files WHERE path='sent.eml'",
    ).fetchone()["id"]
    fresh_db.execute(
        "INSERT INTO chunks(file_id, chunk_index, text, start_offset) "
        "VALUES (?, 0, 'From: ben@x\nSubject: deck\n\n"
        "Sent the deck to Sarah today.', 0)", (sid,),
    )
    fresh_db.commit()

    # LLM hallucinates a non-existent file_id.
    monkeypatch.setattr(
        followups_ops, "_llm_check_resolution",
        lambda cfg, **kw: {
            "resolved": True, "confidence": 0.9,
            "evidence_file_id": 99999,  # not in candidate set
            "evidence": "User sent the deck",
        },
    )
    followups_ops.auto_resolve_from_sent_mail(
        fresh_db, tmp_cfg, hours=24,
    )
    # The followup IS resolved, but evidence_file_id is None
    # (the bad fid was coerced).
    row = fresh_db.execute(
        "SELECT evidence_file_id FROM followup_resolutions "
        "WHERE followup_id = ?",
        (fid,),
    ).fetchone()
    assert row is not None
    assert row["evidence_file_id"] is None


# ============================ HIGH A3 — was_new tuple ===================


def test_add_followup_with_status_returns_was_new(fresh_db):
    from secondbrain import followups

    rid1, was_new1 = followups.add_followup_with_status(
        fresh_db, direction="outgoing",
        topic="t", description="d",
    )
    assert rid1 > 0
    assert was_new1 is True
    # Same dedup_key → row exists, was_new must be False.
    rid2, was_new2 = followups.add_followup_with_status(
        fresh_db, direction="outgoing",
        topic="t", description="d",
    )
    assert rid2 == rid1
    assert was_new2 is False


def test_extract_and_persist_uses_was_new_not_time_window(
    fresh_db, tmp_cfg, monkeypatch,
):
    """Round 21 fix: re-running extract on the same source must
    return n=0 (no double-count) even if the LLM is slow enough
    that the original row is now >5s old."""
    from secondbrain import followups

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test")
    # Need a real file row for the FK on followups.source_file_id.
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, content_hash, "
        "indexed_at) VALUES ('e.eml', 0, 0, 'email', 'h', ?)",
        (time.time(),),
    )
    fid = fresh_db.execute(
        "SELECT id FROM files WHERE path='e.eml'",
    ).fetchone()["id"]
    fresh_db.commit()

    fake_items = [{
        "direction": "outgoing", "person": "Sarah",
        "topic": "Send deck", "description": "Send Q3 deck",
        "confidence": 0.9, "due_hint": None, "excerpt": "...",
    }]
    monkeypatch.setattr(
        followups, "extract_from_text",
        lambda *a, **kw: fake_items,
    )
    n1 = followups.extract_and_persist(
        fresh_db, tmp_cfg,
        text="...", user_name="Ben",
        source_kind="email", source_file_id=fid,
    )
    assert n1 == 1
    # Second call on the same source: dedup hits, was_new=False,
    # so n_added=0. Earlier code would still count it as new if
    # the wall-clock window was small enough.
    n2 = followups.extract_and_persist(
        fresh_db, tmp_cfg,
        text="...", user_name="Ben",
        source_kind="email", source_file_id=fid,
    )
    assert n2 == 0


# ============================ HIGH F1 — write locks =====================


def test_followups_module_has_write_lock():
    from secondbrain import followups, followups_ops
    assert hasattr(followups, "_WRITE_LOCK")
    # followups_ops imports the same lock instance.
    assert followups_ops._WRITE_LOCK is followups._WRITE_LOCK


def test_each_round20_module_has_write_lock():
    from secondbrain import (
        agenda,
        meeting_capture,
        scheduling,
        triage_queue,
    )
    for m in (agenda, scheduling, triage_queue, meeting_capture):
        assert hasattr(m, "_WRITE_LOCK"), m.__name__


def test_concurrent_followup_writes_serialised(fresh_db):
    from secondbrain import followups

    # Init schema before threads start so they don't race on the
    # CREATE TABLE statements (the write lock guards INSERT/UPDATE
    # only; schema-init runs before).
    followups._ensure_schema(fresh_db)

    n_threads = 6
    n_per = 25
    errors: list = []

    def writer(t: int):
        try:
            for i in range(n_per):
                followups.add_followup(
                    fresh_db, direction="outgoing",
                    topic=f"t{t}-{i}", description=f"d{t}-{i}",
                )
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [
        threading.Thread(target=writer, args=(t,))
        for t in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [], f"threads raised: {errors}"
    rows = followups.list_open(
        fresh_db, limit=n_threads * n_per + 10,
    )
    # Each (topic, description) is unique → all 150 rows land.
    assert len(rows) == n_threads * n_per


# ============================ MED A6 — local-day key ====================


def test_cadence_overdue_key_uses_local_date(fresh_db, monkeypatch):
    """Round 21 fix: notification key should use the local-tz
    date, not UTC days-since-epoch. Verified by source-string
    inspection since stubbing local TZ portably is brittle."""
    import inspect

    from secondbrain import notifications
    src = inspect.getsource(notifications._detect_cadence_overdue)
    # New key uses today_iso = date.today().isoformat().
    assert "today_iso" in src
    assert "date.today" in src
    # Old broken pattern (UTC days) must be gone.
    assert "time.time() / 86400" not in src


# ============================ MED A7 — user-set flag ====================


def test_set_field_flips_cadence_user_set(fresh_db):
    from secondbrain import people as pm

    pid = pm.upsert_person(fresh_db, display_name="Sarah")
    # Default: cadence_user_set = 0.
    row = fresh_db.execute(
        "SELECT cadence_user_set FROM people WHERE id = ?", (pid,),
    ).fetchone()
    assert row["cadence_user_set"] == 0
    # Setting cadence_days flips the flag.
    pm.set_field(fresh_db, pid, cadence_days=14)
    row = fresh_db.execute(
        "SELECT cadence_user_set FROM people WHERE id = ?", (pid,),
    ).fetchone()
    assert row["cadence_user_set"] == 1
    # Clearing (cadence_days=0 → NULL) ALSO keeps the flag at 1.
    pm.set_field(fresh_db, pid, cadence_days=0)
    row = fresh_db.execute(
        "SELECT cadence_days, cadence_user_set FROM people WHERE id = ?",
        (pid,),
    ).fetchone()
    assert row["cadence_days"] is None
    assert row["cadence_user_set"] == 1


def test_auto_apply_inferred_skips_user_cleared(fresh_db, monkeypatch):
    """A user who explicitly cleared cadence_days back to NULL
    should NOT have it re-inferred."""
    from secondbrain import people as pm

    pid = pm.upsert_person(fresh_db, display_name="Sarah")
    pm.set_field(fresh_db, pid, tier="vip", cadence_days=0)
    # Force inference to return 30 if asked.
    monkeypatch.setattr(
        pm, "infer_cadence_for_person",
        lambda conn, pid, **kw: 30,
    )
    n = pm.auto_apply_inferred_cadence(fresh_db)
    p = pm.get_person(fresh_db, pid)
    assert p.cadence_days is None  # not auto-overwritten
    assert n == 0


# ============================ MED A8 — cadence stability ================


def test_infer_cadence_min_contacts_raised(fresh_db):
    from secondbrain import people as pm

    pid = pm.upsert_person(fresh_db, display_name="Sarah")
    # Seed only 5 mentions — below the new floor of 6.
    base = time.time() - 60 * 86400
    for i in range(5):
        ts = base + i * 14 * 86400
        fresh_db.execute(
            "INSERT INTO files(path, mtime, size, kind, content_hash, "
            "indexed_at) VALUES (?, 0, 0, 'document', ?, ?)",
            (f"f{i}.txt", f"h{i}", ts),
        )
        fid = fresh_db.execute(
            "SELECT id FROM files WHERE path = ?", (f"f{i}.txt",),
        ).fetchone()["id"]
        fresh_db.execute(
            "INSERT INTO chunks(file_id, chunk_index, text, start_offset) "
            "VALUES (?, 0, 't', 0)", (fid,),
        )
        cid = fresh_db.execute(
            "SELECT id FROM chunks WHERE file_id = ?", (fid,),
        ).fetchone()["id"]
        fresh_db.execute(
            "INSERT INTO person_mentions"
            "(person_id, chunk_id, file_id, mtime) "
            "VALUES (?, ?, ?, ?)", (pid, cid, fid, ts),
        )
    fresh_db.commit()
    assert pm.infer_cadence_for_person(fresh_db, pid) is None


# ============================ MED C1+C2 — redaction =====================


def test_llm_check_resolution_redacts_candidate_preview(
    fresh_db, tmp_cfg, monkeypatch,
):
    """The candidate preview must pass through redact_text before
    going into the LLM prompt."""
    from secondbrain import followups, followups_ops

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test")
    tmp_cfg.user_email = "ben@x"
    followups.add_followup(
        fresh_db, direction="outgoing",
        topic="Send key", description="Send the API key Sarah needs",
        person_name="Sarah", promised_at=time.time() - 3600,
    )
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, content_hash, "
        "indexed_at) VALUES ('sent.eml', 0, 0, 'email', 'h', ?)",
        (time.time() - 100,),
    )
    sid = fresh_db.execute(
        "SELECT id FROM files WHERE path='sent.eml'",
    ).fetchone()["id"]
    secret_body = (
        "From: ben@x\nSubject: Send key\n\n"
        "Here's the key Sarah: "
        "sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    )
    fresh_db.execute(
        "INSERT INTO chunks(file_id, chunk_index, text, start_offset) "
        "VALUES (?, 0, ?, 0)", (sid, secret_body),
    )
    fresh_db.commit()

    captured: dict = {}
    fake_block = MagicMock()
    fake_block.type = "text"
    fake_block.text = '{"resolved": false, "confidence": 0.0}'
    fake_resp = MagicMock()
    fake_resp.content = [fake_block]
    fake_resp.usage.input_tokens = 50
    fake_resp.usage.output_tokens = 5

    class _Anth:
        APIError = Exception

        def Anthropic(*a, **kw):  # noqa: N802 — match SDK shape
            client = MagicMock()

            def create(**kwargs):
                captured["messages"] = kwargs["messages"]
                return fake_resp
            client.messages.create = create
            return client

    with patch.dict("sys.modules", {"anthropic": _Anth}):
        followups_ops.auto_resolve_from_sent_mail(
            fresh_db, tmp_cfg, hours=24,
        )
    payload = captured.get("messages", [])
    assert payload, "LLM should have been called"
    user_msg = payload[0]["content"]
    # The raw secret must not have been sent.
    assert "sk-ant-api03-AAAA" not in user_msg
    assert "[REDACTED:anthropic_key]" in user_msg


# ============================ MED I1 — nav ==============================


def test_ea_pages_in_nav():
    from secondbrain.dashboard import _NAV_GROUPS
    flattened = {
        href for _, items in _NAV_GROUPS for _, href in items
    }
    for path in (
        "/followups", "/agenda", "/triage", "/capture",
        "/scheduling", "/eod",
    ):
        assert path in flattened, f"{path} not in _NAV_GROUPS"


# ============================ MED A-config — load_config ================


def test_load_config_reads_user_name_and_scheduling(tmp_path):
    """Round 21 fix: load_config now reads the new EA-shaped fields
    instead of leaving them at defaults."""
    from secondbrain.config import load_config

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'user_name = "Ben"\n'
        'user_email = "ben@example.com"\n'
        'scheduling_earliest_hour = 8\n'
        'scheduling_latest_hour = 18\n'
        'scheduling_buffer_minutes = 10\n'
        'eod_send_time = "17:30"\n',
    )
    cfg = load_config(path=config_path)
    assert cfg.user_name == "Ben"
    assert cfg.user_email == "ben@example.com"
    assert cfg.scheduling_earliest_hour == 8
    assert cfg.scheduling_latest_hour == 18
    assert cfg.scheduling_buffer_minutes == 10
    assert cfg.eod_send_time == "17:30"


# ============================ MED A9 — brief limit ======================


def test_followups_section_pulls_8_per_side(fresh_db):
    """Source check that the brief now pulls 8 per side instead
    of 6 (with the 10-cap maintained)."""
    import inspect

    from secondbrain import daily_brief
    src = inspect.getsource(daily_brief._followups_section)
    assert 'limit=8' in src, "should pull 8 per side"


# ============================ LOW I2 — agenda UI ========================


def test_agenda_detail_renders_add_note_form(
    monkeypatch, tmp_path, fake_embedder,
):
    from fastapi.testclient import TestClient

    from secondbrain import people as pm
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
    seed = connect(cfg.db_path)
    init_schema(seed, fake_embedder.dim, fake_embedder.name)
    pid = pm.upsert_person(seed, display_name="Sarah")
    seed.close()
    client = TestClient(create_app())
    r = client.get(f"/agenda?id={pid}")
    assert r.status_code == 200
    # Form to add a new note must be present.
    assert f'action="/agenda/{pid}/note"' in r.text
    assert 'Add a topic' in r.text or 'Add a topic' in r.text.lower() or "Add" in r.text


# ============================ smoke =====================================


def test_round21_modules_import():
    from secondbrain import (  # noqa: F401
        agenda,
        config,
        daily_brief,
        followups,
        followups_ops,
        meeting_capture,
        notifications,
        people,
        scheduling,
        triage_queue,
    )
