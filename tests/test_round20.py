"""Round 20 — full Tier 1+2 EA build-out tests.

  - Followups: auto-resolution, snooze/edit/bulk, nudge drafter, stats
  - Notifications: overdue, stale, cadence-overdue detectors
  - Daily brief: followups + triage sections
  - Meeting capture: attendees, edit, recap-sent
  - Agenda: user notes
  - Scheduling: per-person prefs + proposal log
  - Triage queue: state (snooze/done) + queue filtering
  - People: cadence inference
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

# ============================ followups_ops ===========================


def test_followup_snooze_hides_from_visible(fresh_db):
    from secondbrain import followups, followups_ops

    fid = followups.add_followup(
        fresh_db, direction="outgoing",
        topic="t", description="d",
    )
    assert followups_ops.snooze(fresh_db, fid, days=3)
    visible = followups_ops.list_visible_open(fresh_db)
    assert fid not in {f.id for f in visible}
    # With include_snoozed=True, it shows.
    incl = followups_ops.list_visible_open(
        fresh_db, include_snoozed=True,
    )
    assert fid in {f.id for f in incl}


def test_followup_unsnooze(fresh_db):
    from secondbrain import followups, followups_ops

    fid = followups.add_followup(
        fresh_db, direction="outgoing",
        topic="t", description="d",
    )
    followups_ops.snooze(fresh_db, fid, days=7)
    assert followups_ops.unsnooze(fresh_db, fid)
    visible = followups_ops.list_visible_open(fresh_db)
    assert fid in {f.id for f in visible}


def test_followup_edit_changes_fields(fresh_db):
    from secondbrain import followups, followups_ops

    fid = followups.add_followup(
        fresh_db, direction="outgoing",
        topic="old topic", description="old desc",
    )
    assert followups_ops.edit(
        fresh_db, fid,
        topic="new topic",
        description="new desc",
    )
    f = followups.get(fresh_db, fid)
    assert f.topic == "new topic"
    assert f.description == "new desc"


def test_followup_edit_due_at_clears_with_none(fresh_db):
    from secondbrain import followups, followups_ops

    fid = followups.add_followup(
        fresh_db, direction="outgoing",
        topic="t", description="d",
        due_at=time.time() + 86400,
    )
    f = followups.get(fresh_db, fid)
    assert f.due_at is not None
    followups_ops.edit(fresh_db, fid, due_at=None)
    f = followups.get(fresh_db, fid)
    assert f.due_at is None


def test_bulk_dismiss_by_person(fresh_db):
    from secondbrain import followups, followups_ops
    from secondbrain import people as pm

    pid = pm.upsert_person(fresh_db, display_name="Sarah")
    pid2 = pm.upsert_person(fresh_db, display_name="Bob")
    followups.add_followup(
        fresh_db, direction="outgoing",
        topic="t1", description="d1", person_id=pid,
    )
    followups.add_followup(
        fresh_db, direction="outgoing",
        topic="t2", description="d2", person_id=pid,
    )
    followups.add_followup(
        fresh_db, direction="outgoing",
        topic="t3", description="d3", person_id=pid2,
    )
    n = followups_ops.bulk_dismiss(fresh_db, person_id=pid)
    assert n == 2
    # Bob's row still open.
    rows = followups.list_open(fresh_db)
    assert len(rows) == 1
    assert rows[0].person_id == pid2


def test_bulk_dismiss_overdue_only(fresh_db):
    from secondbrain import followups, followups_ops

    yesterday = time.time() - 86400
    tomorrow = time.time() + 86400
    followups.add_followup(
        fresh_db, direction="outgoing",
        topic="overdue", description="o", due_at=yesterday,
    )
    followups.add_followup(
        fresh_db, direction="outgoing",
        topic="future", description="f", due_at=tomorrow,
    )
    n = followups_ops.bulk_dismiss(fresh_db, overdue_only=True)
    assert n == 1
    rows = followups.list_open(fresh_db)
    assert rows[0].topic == "future"


def test_compute_stats_counts_correctly(fresh_db):
    from secondbrain import followups, followups_ops

    yesterday = time.time() - 86400
    followups.add_followup(
        fresh_db, direction="outgoing", topic="o1",
        description="d", due_at=yesterday,
    )
    fid = followups.add_followup(
        fresh_db, direction="incoming", topic="i1", description="d",
    )
    followups_ops.snooze(fresh_db, fid, days=5)
    s = followups_ops.compute_stats(fresh_db)
    assert s.open_outgoing == 1
    assert s.open_incoming == 0  # snoozed → not counted as visible
    assert s.overdue_count == 1
    assert s.snoozed_count == 1


# ============================ auto-resolution =========================


def test_auto_resolve_marks_followup_resolved(
    fresh_db, tmp_cfg, monkeypatch,
):
    from secondbrain import followups, followups_ops

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test")
    fid = followups.add_followup(
        fresh_db, direction="outgoing",
        topic="Send Q3 deck",
        description="Send Sarah the Q3 numbers deck",
        person_name="Sarah",
        promised_at=time.time() - 3600,
    )
    # Seed a "sent email" file with overlap.
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, content_hash, "
        "indexed_at) VALUES ('sent.eml', 0, 0, 'email', 'h', ?)",
        (time.time() - 100,),
    )
    sent_id = fresh_db.execute(
        "SELECT id FROM files WHERE path='sent.eml'",
    ).fetchone()["id"]
    fresh_db.execute(
        "INSERT INTO chunks(file_id, chunk_index, text, start_offset) "
        "VALUES (?, 0, ?, 0)",
        (sent_id, "From: ben@x\nTo: sarah@y\nSubject: Q3 deck\n\n"
                  "Hi Sarah, here's the Q3 deck attached.\n"),
    )
    fresh_db.commit()

    # Stub the LLM verdict to "yes resolved".
    monkeypatch.setattr(
        followups_ops, "_llm_check_resolution",
        lambda cfg, **kw: {
            "resolved": True, "confidence": 0.9,
            "evidence_file_id": sent_id,
            "evidence": "User sent Sarah the Q3 deck.",
        },
    )
    n = followups_ops.auto_resolve_from_sent_mail(
        fresh_db, tmp_cfg, hours=24,
    )
    assert n == 1
    f = followups.get(fresh_db, fid)
    assert f.status == "resolved"


def test_auto_resolve_skips_low_confidence(
    fresh_db, tmp_cfg, monkeypatch,
):
    from secondbrain import followups, followups_ops

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test")
    fid = followups.add_followup(
        fresh_db, direction="outgoing",
        topic="Send Q3 deck", description="Send Sarah Q3 deck",
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
        "VALUES (?, 0, 'From: ben\nSubject: Q3 deck\n\n"
        "Will send tomorrow.', 0)",
        (sid,),
    )
    fresh_db.commit()
    monkeypatch.setattr(
        followups_ops, "_llm_check_resolution",
        lambda cfg, **kw: {"resolved": True, "confidence": 0.3},
    )
    n = followups_ops.auto_resolve_from_sent_mail(
        fresh_db, tmp_cfg, hours=24,
    )
    assert n == 0
    assert followups.get(fresh_db, fid).status == "open"


# ============================ nudge drafter ===========================


def test_draft_nudge_returns_subject_and_body(
    fresh_db, tmp_cfg, monkeypatch,
):
    from secondbrain import followups, followups_ops

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test")
    fid = followups.add_followup(
        fresh_db, direction="incoming",
        topic="Contract review", description="John reviews proposal",
        person_name="John", promised_at=time.time() - 14 * 86400,
    )

    fake_block = MagicMock()
    fake_block.type = "text"
    fake_block.text = (
        '{"subject": "Re: Contract review", '
        '"body": "Wanted to circle back on the contract review — '
        'any updates?"}'
    )
    fake_resp = MagicMock()
    fake_resp.content = [fake_block]
    fake_resp.usage.input_tokens = 50
    fake_resp.usage.output_tokens = 20
    mock_anth = MagicMock()
    mock_anth.Anthropic.return_value.messages.create.return_value = (
        fake_resp
    )
    mock_anth.APIError = Exception
    with patch.dict("sys.modules", {"anthropic": mock_anth}):
        draft = followups_ops.draft_nudge(fresh_db, tmp_cfg, fid)
    assert draft is not None
    assert "Re: Contract review" in draft["subject"]
    assert "circle back" in draft["body"]


def test_draft_nudge_rejects_outgoing_followup(fresh_db, tmp_cfg, monkeypatch):
    from secondbrain import followups, followups_ops
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test")
    fid = followups.add_followup(
        fresh_db, direction="outgoing",
        topic="t", description="d",
    )
    out = followups_ops.draft_nudge(fresh_db, tmp_cfg, fid)
    assert out is None  # outgoing followups don't get nudged


# ============================ notifications ==========================


def test_notification_followup_overdue(fresh_db):
    from secondbrain import followups, notifications

    yesterday = time.time() - 86400
    fid = followups.add_followup(
        fresh_db, direction="outgoing",
        topic="overdue!", description="d", due_at=yesterday,
        person_name="Sarah",
    )
    n = notifications._detect_followup_overdue(fresh_db)
    assert n == 1
    pending = notifications.list_pending(fresh_db)
    assert any("overdue!" in p.title for p in pending)
    assert fid > 0  # smoke: followup id valid


def test_notification_followup_stale(fresh_db):
    from secondbrain import followups, notifications

    followups.add_followup(
        fresh_db, direction="incoming",
        topic="stale", description="d",
        person_name="John",
        promised_at=time.time() - 30 * 86400,
    )
    n = notifications._detect_followup_stale(fresh_db)
    assert n == 1


def test_notification_cadence_overdue(fresh_db):
    from secondbrain import notifications
    from secondbrain import people as pm

    pid = pm.upsert_person(fresh_db, display_name="Sarah")
    pm.set_field(fresh_db, pid, tier="vip", cadence_days=7)
    fresh_db.execute(
        "UPDATE people SET last_contact_at = ? WHERE id = ?",
        (time.time() - 30 * 86400, pid),
    )
    fresh_db.commit()
    n = notifications._detect_cadence_overdue(fresh_db)
    assert n >= 1


# ============================ daily brief ============================


def test_daily_brief_followups_section(fresh_db):
    from secondbrain import daily_brief, followups

    followups.add_followup(
        fresh_db, direction="outgoing",
        topic="brief test", description="d", person_name="Sarah",
    )
    items = daily_brief._followups_section(fresh_db)
    assert len(items) == 1
    assert items[0]["topic"] == "brief test"
    assert items[0]["direction"] == "outgoing"


def test_daily_brief_triage_section(fresh_db):
    from secondbrain import daily_brief, email_assist

    email_assist._ensure_schema(fresh_db)
    now = time.time()
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, content_hash, "
        "indexed_at) VALUES ('e.eml', 0, 0, 'email', 'h', ?)",
        (now - 3600,),
    )
    fid = fresh_db.execute(
        "SELECT id FROM files WHERE path='e.eml'",
    ).fetchone()["id"]
    fresh_db.execute(
        "INSERT INTO chunks(file_id, chunk_index, text, start_offset) "
        "VALUES (?, 0, 'From: alice@x\nSubject: URGENT', 0)",
        (fid,),
    )
    fresh_db.execute(
        "INSERT INTO email_classifications"
        "(file_id, label, confidence, classified_at) "
        "VALUES (?, 'urgent', 0.9, ?)", (fid, now),
    )
    fresh_db.commit()
    triage = daily_brief._triage_section(fresh_db)
    assert triage["count"] >= 1


# ============================ meeting capture =========================


def test_meeting_capture_attendees_extracted(
    fresh_db, tmp_cfg, monkeypatch,
):
    from secondbrain import meeting_capture

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test")
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, content_hash, "
        "indexed_at) VALUES ('m.txt', 0, 100, 'transcript', 'h', ?)",
        (time.time(),),
    )
    fid = fresh_db.execute(
        "SELECT id FROM files WHERE path='m.txt'",
    ).fetchone()["id"]
    fresh_db.execute(
        "INSERT INTO chunks(file_id, chunk_index, text, start_offset) "
        "VALUES (?, 0, 'Ben: hi. Sarah: ok.', 0)", (fid,),
    )
    fresh_db.commit()

    fake = {
        "title": "test", "decisions": [], "actions": [],
        "open_questions": [], "recap_draft": "",
        "attendees": ["Ben", "Sarah", "Maria"],
    }
    monkeypatch.setattr(
        meeting_capture, "_extract_via_llm",
        lambda cfg, transcript, user_name: (fake, "x", 0.0),
    )
    cap = meeting_capture.capture(fresh_db, tmp_cfg, fid, user_name="Ben")
    assert cap is not None
    assert "Ben" in cap.attendees
    assert "Sarah" in cap.attendees


def test_meeting_capture_edit(fresh_db):
    from secondbrain import meeting_capture
    meeting_capture._ensure_schema(fresh_db)
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, content_hash, "
        "indexed_at) VALUES ('m.txt', 0, 0, 'transcript', 'h', ?)",
        (time.time(),),
    )
    fid = fresh_db.execute(
        "SELECT id FROM files WHERE path='m.txt'",
    ).fetchone()["id"]
    fresh_db.execute(
        "INSERT INTO meeting_captures"
        "(file_id, title, captured_at) VALUES (?, 'old', ?)",
        (fid, time.time()),
    )
    fresh_db.commit()
    assert meeting_capture.edit_capture(
        fresh_db, fid, title="new title",
    )
    cap = meeting_capture.get_capture(fresh_db, fid)
    assert cap.title == "new title"
    assert cap.user_edited is True


def test_meeting_capture_recap_sent_marker(fresh_db):
    from secondbrain import meeting_capture
    meeting_capture._ensure_schema(fresh_db)
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, content_hash, "
        "indexed_at) VALUES ('m.txt', 0, 0, 'transcript', 'h', ?)",
        (time.time(),),
    )
    fid = fresh_db.execute(
        "SELECT id FROM files WHERE path='m.txt'",
    ).fetchone()["id"]
    fresh_db.execute(
        "INSERT INTO meeting_captures"
        "(file_id, title, captured_at) VALUES (?, 't', ?)",
        (fid, time.time()),
    )
    fresh_db.commit()
    assert meeting_capture.mark_recap_sent(fresh_db, fid)
    cap = meeting_capture.get_capture(fresh_db, fid)
    assert cap.recap_sent_at is not None


# ============================ agenda notes ===========================


def test_agenda_notes_lifecycle(fresh_db):
    from secondbrain import agenda
    from secondbrain import people as pm

    pid = pm.upsert_person(fresh_db, display_name="Sarah")
    nid = agenda.add_note(
        fresh_db, pid, "Ask about Q3 numbers",
    )
    assert nid > 0
    notes = agenda.list_notes(fresh_db, pid)
    assert len(notes) == 1
    assert notes[0].text == "Ask about Q3 numbers"
    assert agenda.mark_discussed(fresh_db, nid)
    notes = agenda.list_notes(fresh_db, pid, status="pending")
    assert notes == []


def test_agenda_includes_user_notes(fresh_db):
    from secondbrain import agenda
    from secondbrain import people as pm

    pid = pm.upsert_person(fresh_db, display_name="Sarah")
    agenda.add_note(fresh_db, pid, "Test note 1")
    agenda.add_note(fresh_db, pid, "Test note 2")
    a = agenda.build_agenda(fresh_db, pid)
    assert a is not None
    assert len(a.user_notes) == 2
    md = agenda.render_markdown(a)
    assert "Things to bring up" in md
    assert "Test note 1" in md


# ============================ scheduling prefs =======================


def test_scheduling_prefs_persist_and_merge(fresh_db, tmp_cfg):
    from secondbrain import people as pm
    from secondbrain import scheduling

    pid = pm.upsert_person(fresh_db, display_name="Sarah")
    scheduling.set_person_prefs(
        fresh_db, pid,
        preferred_weekdays=[1, 2, 3],
        preferred_hours=[14, 15],
        duration_minutes=45,
    )
    p = scheduling.get_person_prefs(fresh_db, pid)
    assert p is not None
    assert p.preferred_weekdays == [1, 2, 3]
    assert p.duration_minutes == 45

    merged = scheduling.merge_with_global_prefs(tmp_cfg, p)
    assert merged.duration_minutes == 45
    assert merged.preferred_hours == [14, 15]


def test_scheduling_proposal_log_and_outcome(fresh_db):
    from secondbrain import people as pm
    from secondbrain import scheduling

    pid = pm.upsert_person(fresh_db, display_name="Sarah")
    from datetime import datetime
    slot = scheduling.TimeSlot(
        start=datetime(2025, 5, 1, 14, 0),
        end=datetime(2025, 5, 1, 14, 30),
        rank=10.0,
    )
    proposal_id = scheduling.log_proposal(
        fresh_db, person_id=pid, person_name="Sarah",
        slots=[slot], email_body="Hi Sarah, ...",
    )
    assert proposal_id > 0
    proposals = scheduling.list_recent_proposals(fresh_db)
    assert len(proposals) == 1
    assert proposals[0]["outcome"] == "pending"
    assert scheduling.mark_proposal_outcome(
        fresh_db, proposal_id, "scheduled",
        chosen_slot_iso="2025-05-01T14:00:00",
    )
    proposals = scheduling.list_recent_proposals(fresh_db)
    assert proposals[0]["outcome"] == "scheduled"


# ============================ triage queue state =====================


def test_triage_done_excludes_from_queue(fresh_db, tmp_cfg):
    from secondbrain import email_assist, triage_queue

    email_assist._ensure_schema(fresh_db)
    triage_queue._ensure_schema(fresh_db)
    now = time.time()
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, content_hash, "
        "indexed_at) VALUES ('e.eml', 0, 0, 'email', 'h', ?)",
        (now - 3600,),
    )
    fid = fresh_db.execute(
        "SELECT id FROM files WHERE path='e.eml'",
    ).fetchone()["id"]
    fresh_db.execute(
        "INSERT INTO chunks(file_id, chunk_index, text, start_offset) "
        "VALUES (?, 0, 'From: alice@x\nSubject: URGENT', 0)",
        (fid,),
    )
    fresh_db.execute(
        "INSERT INTO email_classifications"
        "(file_id, label, confidence, classified_at) "
        "VALUES (?, 'urgent', 0.9, ?)", (fid, now),
    )
    fresh_db.commit()
    queue = triage_queue.build_queue(fresh_db, hours=24)
    assert any(it.file_id == fid for it in queue)
    triage_queue.mark_done(fresh_db, fid)
    queue2 = triage_queue.build_queue(fresh_db, hours=24)
    assert not any(it.file_id == fid for it in queue2)


def test_triage_snooze_hides_until_passes(fresh_db):
    from secondbrain import email_assist, triage_queue
    email_assist._ensure_schema(fresh_db)
    triage_queue._ensure_schema(fresh_db)
    now = time.time()
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, content_hash, "
        "indexed_at) VALUES ('e.eml', 0, 0, 'email', 'h', ?)",
        (now - 3600,),
    )
    fid = fresh_db.execute(
        "SELECT id FROM files WHERE path='e.eml'",
    ).fetchone()["id"]
    fresh_db.execute(
        "INSERT INTO chunks(file_id, chunk_index, text, start_offset) "
        "VALUES (?, 0, 'From: alice@x\nSubject: URGENT', 0)",
        (fid,),
    )
    fresh_db.execute(
        "INSERT INTO email_classifications"
        "(file_id, label, confidence, classified_at) "
        "VALUES (?, 'urgent', 0.9, ?)", (fid, now),
    )
    fresh_db.commit()
    triage_queue.snooze(fresh_db, fid, hours=24)
    queue = triage_queue.build_queue(fresh_db, hours=24)
    assert not any(it.file_id == fid for it in queue)


def test_triage_done_count_today(fresh_db):
    from secondbrain import triage_queue
    triage_queue._ensure_schema(fresh_db)
    # The triage_state.file_id has a FK to files(id), so we need
    # actual file rows to point at.
    for path in ("a", "b"):
        fresh_db.execute(
            "INSERT INTO files(path, mtime, size, kind, content_hash, "
            "indexed_at) VALUES (?, 0, 0, 'email', ?, ?)",
            (path, f"h_{path}", time.time()),
        )
    fids = [
        r["id"] for r in fresh_db.execute(
            "SELECT id FROM files WHERE path IN ('a','b')",
        ).fetchall()
    ]
    fresh_db.commit()
    triage_queue.mark_done(fresh_db, fids[0])
    triage_queue.mark_done(fresh_db, fids[1])
    assert triage_queue.done_count_today(fresh_db) == 2


# ============================ cadence inference ======================


def test_infer_cadence_from_mention_history(fresh_db):
    from secondbrain import people as pm

    pid = pm.upsert_person(fresh_db, display_name="Sarah")
    # Seed 13 mentions roughly every 14 days for 6 months. Each
    # mention needs a unique chunk_id (UNIQUE on person_id+chunk_id).
    base = time.time() - 180 * 86400
    for i in range(13):
        ts = base + i * 14 * 86400
        fresh_db.execute(
            "INSERT INTO files(path, mtime, size, kind, "
            "content_hash, indexed_at) "
            "VALUES (?, 0, 0, 'document', ?, ?)",
            (f"f{i}.txt", f"h{i}", ts),
        )
        fid = fresh_db.execute(
            "SELECT id FROM files WHERE path = ?", (f"f{i}.txt",),
        ).fetchone()["id"]
        fresh_db.execute(
            "INSERT INTO chunks(file_id, chunk_index, text, "
            "start_offset) VALUES (?, 0, 't', 0)", (fid,),
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
    suggested = pm.infer_cadence_for_person(fresh_db, pid)
    assert suggested in (7, 14)  # close to 14d


def test_infer_cadence_too_few_signals_returns_none(fresh_db):
    from secondbrain import people as pm
    pid = pm.upsert_person(fresh_db, display_name="Sarah")
    # No mentions → None.
    assert pm.infer_cadence_for_person(fresh_db, pid) is None


def test_auto_apply_inferred_cadence_only_unset(fresh_db):
    from secondbrain import people as pm
    pid = pm.upsert_person(fresh_db, display_name="Sarah")
    # Pre-set cadence — should NOT be overwritten.
    pm.set_field(fresh_db, pid, cadence_days=30, tier="vip")
    n = pm.auto_apply_inferred_cadence(fresh_db)
    p = pm.get_person(fresh_db, pid)
    assert p.cadence_days == 30  # preserved
    # n could be 0 or 1; semantics is "didn't override existing"
    assert n == 0


# ============================ dashboard routes =======================


def _client_with_dashboard(monkeypatch, tmp_path, fake_embedder):
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


def test_followups_page_with_filter(monkeypatch, tmp_path, fake_embedder):
    from secondbrain import followups
    from secondbrain import people as pm
    from secondbrain.db import connect, init_schema

    cfg, client = _client_with_dashboard(monkeypatch, tmp_path, fake_embedder)
    seed = connect(cfg.db_path)
    init_schema(seed, fake_embedder.dim, fake_embedder.name)
    pid = pm.upsert_person(seed, display_name="Sarah")
    followups.add_followup(
        seed, direction="outgoing", topic="for sarah", description="d",
        person_id=pid,
    )
    followups.add_followup(
        seed, direction="outgoing", topic="for nobody", description="d",
    )
    seed.close()
    r = client.get(f"/followups?person_id={pid}")
    assert r.status_code == 200
    assert "for sarah" in r.text
    assert "for nobody" not in r.text


def test_followups_history_view(monkeypatch, tmp_path, fake_embedder):
    from secondbrain import followups
    from secondbrain.db import connect, init_schema

    cfg, client = _client_with_dashboard(monkeypatch, tmp_path, fake_embedder)
    seed = connect(cfg.db_path)
    init_schema(seed, fake_embedder.dim, fake_embedder.name)
    fid = followups.add_followup(
        seed, direction="outgoing", topic="resolved-thing", description="d",
    )
    followups.mark_resolved(seed, fid)
    seed.close()
    r = client.get("/followups?show=history")
    assert r.status_code == 200
    assert "resolved-thing" in r.text


def test_scheduling_page_renders(monkeypatch, tmp_path, fake_embedder):
    cfg, client = _client_with_dashboard(monkeypatch, tmp_path, fake_embedder)
    r = client.get("/scheduling")
    assert r.status_code == 200
    assert "Scheduling" in r.text


def test_capture_index_renders(monkeypatch, tmp_path, fake_embedder):
    cfg, client = _client_with_dashboard(monkeypatch, tmp_path, fake_embedder)
    r = client.get("/capture")
    assert r.status_code == 200
    assert ("Meeting captures" in r.text or "captures" in r.text.lower())


def test_followup_snooze_csrf_guard(monkeypatch, tmp_path, fake_embedder):
    cfg, client = _client_with_dashboard(monkeypatch, tmp_path, fake_embedder)
    r = client.post(
        "/followups/1/snooze",
        data={"days": "7"},
        headers={"origin": "https://evil.com"},
    )
    assert r.status_code == 403


def test_person_set_tier_csrf_guard(monkeypatch, tmp_path, fake_embedder):
    cfg, client = _client_with_dashboard(monkeypatch, tmp_path, fake_embedder)
    r = client.post(
        "/person/1/tier",
        data={"tier": "vip"},
        headers={"origin": "https://evil.com"},
    )
    assert r.status_code == 403


def test_round20_modules_import_clean():
    from secondbrain import (  # noqa: F401
        agenda,
        daily_brief,
        followups_ops,
        meeting_capture,
        notifications,
        people,
        scheduling,
        triage_queue,
    )
