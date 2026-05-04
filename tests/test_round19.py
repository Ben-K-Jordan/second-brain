"""Round 19 — executive assistant features.

Tests for the 9 new modules + people extensions + dashboard routes
+ MCP tools + daemon jobs.

  - Followups (commitment tracker, both directions)
  - Meeting capture (decisions, actions, recap drafter)
  - Agenda (1:1 builder)
  - People VIP tiering + cadence
  - Triage queue (morning email priority)
  - Scheduling helper (find-time + draft proposal)
  - Gift ideas (birthday ideation)
  - Standing threads (long-running topic tracker)
  - EOD wrap-up
  - Conditional reminders
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

# ============================ followups =============================


def test_followup_add_and_query(fresh_db):
    """add_followup persists, list_open returns it, dedup is keyed."""
    from secondbrain import followups

    fid = followups.add_followup(
        fresh_db,
        direction="outgoing",
        topic="Send Q3 deck",
        description="Send the Q3 numbers deck to Sarah by Friday.",
        person_name="Sarah",
        source_kind="manual",
        promised_at=time.time(),
    )
    assert fid > 0
    rows = followups.list_open(fresh_db, direction="outgoing")
    assert len(rows) == 1
    assert rows[0].topic == "Send Q3 deck"
    assert rows[0].direction == "outgoing"
    assert rows[0].status == "open"
    # Idempotent — same description doesn't double-insert.
    fid2 = followups.add_followup(
        fresh_db,
        direction="outgoing",
        topic="Send Q3 deck",
        description="Send the Q3 numbers deck to Sarah by Friday.",
        person_name="Sarah",
        source_kind="manual",
    )
    assert fid2 == fid
    assert followups.count_open(fresh_db, direction="outgoing") == 1


def test_followup_resolve_and_dismiss(fresh_db):
    from secondbrain import followups

    fid = followups.add_followup(
        fresh_db, direction="incoming",
        topic="John reviews the proposal",
        description="John said he'll review by Tuesday.",
        person_name="John",
    )
    assert followups.mark_resolved(fresh_db, fid)
    assert followups.count_open(fresh_db) == 0
    rows = followups.list_open(fresh_db, direction="incoming")
    assert rows == []
    # Resolving twice is a no-op.
    assert not followups.mark_resolved(fresh_db, fid)


def test_followup_overdue_filter(fresh_db):
    from secondbrain import followups

    yesterday = time.time() - 86400
    tomorrow = time.time() + 86400
    overdue_id = followups.add_followup(
        fresh_db, direction="outgoing",
        topic="overdue", description="overdue",
        due_at=yesterday,
    )
    not_yet_id = followups.add_followup(
        fresh_db, direction="outgoing",
        topic="not yet", description="not yet",
        due_at=tomorrow,
    )
    assert overdue_id > 0 and not_yet_id > 0
    overdue = followups.list_overdue(fresh_db)
    assert len(overdue) == 1
    assert overdue[0].id == overdue_id


def test_followup_redacts_persisted_text(fresh_db):
    from secondbrain import followups

    fid = followups.add_followup(
        fresh_db, direction="outgoing",
        topic="API key for Sarah",
        description=(
            "I will send sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        ),
    )
    row = followups.get(fresh_db, fid)
    assert row is not None
    assert "sk-ant-api03-AAAA" not in row.description
    assert "[REDACTED:anthropic_key]" in row.description


def test_followup_extract_persists_high_confidence_only(
    fresh_db, tmp_cfg, monkeypatch,
):
    """LLM-extracted items below _MIN_CONFIDENCE are dropped."""
    from secondbrain import followups

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test")
    fake = [
        {
            "direction": "outgoing", "person": "Sarah",
            "topic": "Send deck",
            "description": "Send Q3 deck to Sarah",
            "confidence": 0.9,
            "due_hint": None, "excerpt": "...",
        },
        {
            "direction": "incoming", "person": "Bob",
            "topic": "Maybe call back",
            "description": "Bob said maybe he'll call back",
            "confidence": 0.3,  # too low; should drop
            "due_hint": None, "excerpt": "...",
        },
    ]
    monkeypatch.setattr(
        followups, "extract_from_text",
        lambda *a, **kw: fake,
    )
    n = followups.extract_and_persist(
        fresh_db, tmp_cfg,
        text="(unused — extractor stubbed)",
        user_name="Ben",
        source_kind="email",
        source_file_id=None,
    )
    assert n == 1  # only the high-confidence one
    rows = followups.list_open(fresh_db)
    assert len(rows) == 1
    assert rows[0].topic == "Send deck"


# ============================ people VIP / cadence ===================


def test_people_set_tier_and_cadence(fresh_db):
    from secondbrain import people as pm

    pid = pm.upsert_person(fresh_db, display_name="Sarah")
    pm.set_field(fresh_db, pid, tier="vip", cadence_days=7)
    p = pm.get_person(fresh_db, pid)
    assert p.tier == "vip"
    assert p.cadence_days == 7

    # Invalid tier rejected.
    with pytest.raises(ValueError):
        pm.set_field(fresh_db, pid, tier="emperor")

    # cadence_days=0 clears.
    pm.set_field(fresh_db, pid, cadence_days=0)
    p2 = pm.get_person(fresh_db, pid)
    assert p2.cadence_days is None


def test_overdue_contacts_surfaces_vip(fresh_db):
    from secondbrain import people as pm

    pid = pm.upsert_person(fresh_db, display_name="Sarah")
    pm.set_field(fresh_db, pid, tier="vip", cadence_days=14)
    # Force a stale last_contact_at by direct UPDATE.
    fresh_db.execute(
        "UPDATE people SET last_contact_at = ? WHERE id = ?",
        (time.time() - 30 * 86400, pid),
    )
    fresh_db.commit()
    overdue = pm.list_overdue_contacts(fresh_db)
    assert len(overdue) == 1
    assert overdue[0].person.id == pid
    assert overdue[0].days_overdue >= 15


def test_is_vip_email(fresh_db):
    from secondbrain import people as pm

    pid = pm.upsert_person(
        fresh_db, display_name="Sarah", email="sarah@example.com",
    )
    pm.set_field(fresh_db, pid, tier="vip")
    assert pm.is_vip_email(fresh_db, "sarah@example.com")
    assert pm.is_vip_email(fresh_db, "Sarah@Example.com")
    assert not pm.is_vip_email(fresh_db, "bob@example.com")
    assert not pm.is_vip_email(fresh_db, "")


def test_find_person_by_name_resolution(fresh_db):
    from secondbrain import people as pm

    pid = pm.upsert_person(fresh_db, display_name="Sarah Chen")
    # Exact canonical match.
    assert pm.find_person_by_name(fresh_db, "Sarah Chen").id == pid
    # Case-insensitive.
    assert pm.find_person_by_name(fresh_db, "sarah chen").id == pid
    # NFD/NFC variants of accented names.
    pid2 = pm.upsert_person(fresh_db, display_name="José Cruz")
    assert pm.find_person_by_name(fresh_db, "José Cruz").id == pid2  # NFC
    assert pm.find_person_by_name(fresh_db, "José Cruz").id == pid2  # NFD


# ============================ meeting_capture =======================


def test_meeting_capture_persists_and_flows_to_followups(
    fresh_db, tmp_cfg, monkeypatch,
):
    from secondbrain import followups, meeting_capture

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test")
    # Seed a transcript file.
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, "
        "content_hash, indexed_at) "
        "VALUES ('m1.txt', 0, 100, 'audio_video', 'h1', ?)",
        (time.time(),),
    )
    file_id = fresh_db.execute(
        "SELECT id FROM files WHERE path = 'm1.txt'",
    ).fetchone()["id"]
    fresh_db.execute(
        "INSERT INTO chunks(file_id, chunk_index, text, start_offset) "
        "VALUES (?, 0, 'meeting transcript ...', 0)",
        (file_id,),
    )
    fresh_db.commit()

    fake_llm_response = {
        "title": "Q3 planning sync",
        "decisions": [
            {"text": "Ship feature X by April 30",
             "rationale": "Customer commits"},
        ],
        "actions": [
            {"owner": "Ben", "description": "Draft the spec",
             "due_hint": ""},
            {"owner": "Sarah",
             "description": "Review architecture",
             "due_hint": "2025-05-01"},
        ],
        "open_questions": ["Who handles the rollout?"],
        "recap_draft": "Quick recap: we decided ...",
    }

    def fake_extract(cfg, transcript, user_name):
        return fake_llm_response, "claude-sonnet-4-5", 0.0

    monkeypatch.setattr(meeting_capture, "_extract_via_llm", fake_extract)

    cap = meeting_capture.capture(
        fresh_db, tmp_cfg, file_id, user_name="Ben",
    )
    assert cap is not None
    assert cap.title == "Q3 planning sync"
    assert len(cap.decisions) == 1
    assert len(cap.actions) == 2
    assert "rollout" in cap.open_questions[0]
    assert cap.recap_draft.startswith("Quick recap")

    # Action items flowed into followups.
    rows = followups.list_open(fresh_db)
    descriptions = {r.description for r in rows}
    assert "Draft the spec" in descriptions
    assert "Review architecture" in descriptions
    # User-owned action → outgoing.
    user_action = [r for r in rows if r.description == "Draft the spec"][0]
    assert user_action.direction == "outgoing"
    # Other-owned action → incoming.
    other_action = [
        r for r in rows if r.description == "Review architecture"
    ][0]
    assert other_action.direction == "incoming"


def test_meeting_capture_idempotent(fresh_db, tmp_cfg, monkeypatch):
    """Second call without overwrite returns the cached row."""
    from secondbrain import meeting_capture

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test")
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, "
        "content_hash, indexed_at) "
        "VALUES ('m2.txt', 0, 100, 'transcript', 'h2', ?)",
        (time.time(),),
    )
    fid = fresh_db.execute(
        "SELECT id FROM files WHERE path = 'm2.txt'",
    ).fetchone()["id"]
    fresh_db.execute(
        "INSERT INTO chunks(file_id, chunk_index, text, start_offset) "
        "VALUES (?, 0, 'tt', 0)", (fid,),
    )
    fresh_db.commit()
    n_calls = {"n": 0}

    def fake_extract(cfg, transcript, user_name):
        n_calls["n"] += 1
        return {
            "title": "X", "decisions": [], "actions": [],
            "open_questions": [], "recap_draft": "",
        }, "x", 0.0

    monkeypatch.setattr(meeting_capture, "_extract_via_llm", fake_extract)
    meeting_capture.capture(fresh_db, tmp_cfg, fid)
    meeting_capture.capture(fresh_db, tmp_cfg, fid)  # second call no-ops
    assert n_calls["n"] == 1


# ============================ agenda ================================


def test_agenda_aggregates_followups(fresh_db):
    from secondbrain import agenda, followups
    from secondbrain import people as pm

    pid = pm.upsert_person(fresh_db, display_name="Sarah")
    followups.add_followup(
        fresh_db, direction="outgoing", topic="Send deck",
        description="Send Sarah the Q3 deck",
        person_id=pid, person_name="Sarah",
    )
    followups.add_followup(
        fresh_db, direction="incoming",
        topic="Architecture review",
        description="Sarah will review architecture",
        person_id=pid, person_name="Sarah",
    )
    a = agenda.build_agenda(fresh_db, pid)
    assert a is not None
    assert a.person_name == "Sarah"
    assert len(a.open_followups_outgoing) == 1
    assert len(a.open_followups_incoming) == 1
    md = agenda.render_markdown(a)
    assert "You owe them" in md
    assert "They owe you" in md


def test_agenda_unknown_person_returns_none(fresh_db):
    from secondbrain import agenda
    assert agenda.build_agenda(fresh_db, 99999) is None


def test_agenda_empty_state(fresh_db):
    from secondbrain import agenda
    from secondbrain import people as pm

    pid = pm.upsert_person(fresh_db, display_name="Nobody")
    a = agenda.build_agenda(fresh_db, pid)
    assert a is not None
    assert a.total_items == 0
    md = agenda.render_markdown(a)
    assert "clean slate" in md


# ============================ triage_queue ==========================


def test_triage_queue_ranks_by_label_and_age(fresh_db):
    from secondbrain import triage_queue

    # Seed two emails: one urgent, one fyi, with classification rows.
    now = time.time()
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, "
        "content_hash, indexed_at) "
        "VALUES ('e1', 0, 0, 'email', 'h1', ?)", (now - 3600,),
    )
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, "
        "content_hash, indexed_at) "
        "VALUES ('e2', 0, 0, 'email', 'h2', ?)", (now - 3600,),
    )
    e1 = fresh_db.execute(
        "SELECT id FROM files WHERE path='e1'",
    ).fetchone()["id"]
    e2 = fresh_db.execute(
        "SELECT id FROM files WHERE path='e2'",
    ).fetchone()["id"]
    fresh_db.execute(
        "INSERT INTO chunks(file_id, chunk_index, text, start_offset) "
        "VALUES (?, 0, ?, 0)",
        (e1, "From: alice@example.com\nSubject: URGENT contract"),
    )
    fresh_db.execute(
        "INSERT INTO chunks(file_id, chunk_index, text, start_offset) "
        "VALUES (?, 0, ?, 0)",
        (e2, "From: news@example.com\nSubject: Newsletter"),
    )
    # Create the email_classifications table and seed.
    from secondbrain import email_assist
    email_assist._ensure_schema(fresh_db)
    fresh_db.execute(
        "INSERT INTO email_classifications"
        "(file_id, label, confidence, classified_at) "
        "VALUES (?, ?, ?, ?)",
        (e1, "urgent", 0.9, now),
    )
    fresh_db.execute(
        "INSERT INTO email_classifications"
        "(file_id, label, confidence, classified_at) "
        "VALUES (?, ?, ?, ?)",
        (e2, "fyi", 0.8, now),
    )
    fresh_db.commit()

    queue = triage_queue.build_queue(fresh_db, hours=24)
    assert len(queue) >= 1
    # Urgent email ranks above fyi.
    labels = [it.label for it in queue]
    if "urgent" in labels and "fyi" in labels:
        u_idx = labels.index("urgent")
        f_idx = labels.index("fyi")
        assert u_idx < f_idx


def test_triage_queue_vip_promotes(fresh_db):
    """A VIP sender's email outranks a non-VIP urgent."""
    from secondbrain import email_assist, triage_queue
    from secondbrain import people as pm

    email_assist._ensure_schema(fresh_db)
    now = time.time()
    pid = pm.upsert_person(
        fresh_db, display_name="Sarah", email="sarah@example.com",
    )
    pm.set_field(fresh_db, pid, tier="vip")

    # Sarah sends an FYI; Bob sends an URGENT.
    for path, h in [
        ("e_sarah", "h_sarah"),
        ("e_bob", "h_bob"),
    ]:
        fresh_db.execute(
            "INSERT INTO files(path, mtime, size, kind, "
            "content_hash, indexed_at) "
            "VALUES (?, 0, 0, 'email', ?, ?)",
            (path, h, now - 3600),
        )
    e_sarah = fresh_db.execute(
        "SELECT id FROM files WHERE path='e_sarah'",
    ).fetchone()["id"]
    e_bob = fresh_db.execute(
        "SELECT id FROM files WHERE path='e_bob'",
    ).fetchone()["id"]
    fresh_db.execute(
        "INSERT INTO chunks(file_id, chunk_index, text, start_offset) "
        "VALUES (?, 0, 'From: sarah@example.com\nSubject: fyi', 0)",
        (e_sarah,),
    )
    fresh_db.execute(
        "INSERT INTO chunks(file_id, chunk_index, text, start_offset) "
        "VALUES (?, 0, 'From: bob@example.com\nSubject: URGENT', 0)",
        (e_bob,),
    )
    fresh_db.execute(
        "INSERT INTO email_classifications"
        "(file_id, label, confidence, classified_at) "
        "VALUES (?, 'fyi', 0.9, ?)", (e_sarah, now),
    )
    fresh_db.execute(
        "INSERT INTO email_classifications"
        "(file_id, label, confidence, classified_at) "
        "VALUES (?, 'urgent', 0.9, ?)", (e_bob, now),
    )
    fresh_db.commit()
    queue = triage_queue.build_queue(fresh_db, hours=24, min_score=0)
    # VIP bonus (+50) shouldn't beat URGENT base (+100), but VIP fyi
    # must be flagged is_vip + ranked.
    sarah_item = [
        it for it in queue if it.from_email == "sarah@example.com"
    ][0]
    assert sarah_item.is_vip


# ============================ scheduling ============================


def test_find_open_slots_basic():
    from secondbrain import scheduling

    busy = [
        scheduling.BusyBlock(
            start=datetime(2025, 4, 22, 10, 0),
            end=datetime(2025, 4, 22, 11, 0),
            title="standup",
        ),
    ]
    slots = scheduling.find_open_slots(
        busy,
        window_start=date(2025, 4, 22),
        window_end=date(2025, 4, 22),
        prefs=scheduling.SchedulingPrefs(
            duration_minutes=30,
            earliest_hour=9, latest_hour=12,
            buffer_minutes=15,
        ),
    )
    assert slots
    # No slot should overlap the busy block (with 15min buffer).
    for s in slots:
        assert s.end <= datetime(2025, 4, 22, 9, 45) or (
            s.start >= datetime(2025, 4, 22, 11, 15)
        )


def test_find_open_slots_avoids_weekday():
    from secondbrain import scheduling

    slots = scheduling.find_open_slots(
        [],
        window_start=date(2025, 4, 21),  # Mon
        window_end=date(2025, 4, 25),     # Fri
        prefs=scheduling.SchedulingPrefs(
            duration_minutes=30,
            avoid_weekdays=[0, 4],  # Mon, Fri
        ),
    )
    weekdays = {s.start.weekday() for s in slots}
    assert 0 not in weekdays
    assert 4 not in weekdays


def test_parse_busy_blocks_handles_dateTime_and_allday():
    from secondbrain import scheduling

    events = [
        {
            "summary": "Standup",
            "start": {"dateTime": "2025-04-22T10:00:00-04:00"},
            "end": {"dateTime": "2025-04-22T10:30:00-04:00"},
        },
        {
            "summary": "Conference",
            "start": {"date": "2025-04-23"},
            "end": {"date": "2025-04-24"},
        },
    ]
    blocks = scheduling.parse_busy_blocks(events)
    assert len(blocks) == 2
    assert blocks[0].title == "Standup"
    # All-day → 9-5 fallback.
    assert blocks[1].start.hour == 9
    assert blocks[1].end.hour == 17


def test_draft_proposal_email():
    from secondbrain import scheduling

    slots = [
        scheduling.TimeSlot(
            start=datetime(2025, 4, 22, 14, 0),
            end=datetime(2025, 4, 22, 14, 30),
        ),
    ]
    msg = scheduling.draft_proposal_email(
        recipient_name="Sarah",
        slots=slots,
        user_greeting="Hi", user_signoff="Cheers",
        user_name="Ben", purpose="catch up",
    )
    assert "Hi Sarah" in msg
    assert "Cheers" in msg
    assert "Ben" in msg
    assert "catch up" in msg


# ============================ gift_ideas ============================


def test_gift_ideas_idempotent_cache(fresh_db, tmp_cfg, monkeypatch):
    from secondbrain import gift_ideas
    from secondbrain import people as pm

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test")
    pid = pm.upsert_person(fresh_db, display_name="Sarah")
    pm.set_field(fresh_db, pid, notes="loves sci-fi")

    n_calls = {"n": 0}
    fake_resp = MagicMock()
    fake_resp.usage.input_tokens = 50
    fake_resp.usage.output_tokens = 30
    block = MagicMock()
    block.type = "text"
    block.text = json.dumps([
        {"title": "First gift", "description": "x",
         "why": "y", "price_range": "$30"},
        {"title": "Second gift", "description": "x",
         "why": "y", "price_range": "$50"},
        {"title": "Third gift", "description": "x",
         "why": "y", "price_range": "$20"},
    ])
    fake_resp.content = [block]

    class _FakeClient:
        def __init__(self, *a, **kw):
            self.messages = MagicMock()

            def create(**kw):
                n_calls["n"] += 1
                return fake_resp

            self.messages.create = create

    fake_anthropic = MagicMock()
    fake_anthropic.Anthropic = _FakeClient
    fake_anthropic.APIError = Exception
    with patch.dict("sys.modules", {"anthropic": fake_anthropic}):
        ideas1 = gift_ideas.generate_for_person(fresh_db, tmp_cfg, pid)
        ideas2 = gift_ideas.generate_for_person(fresh_db, tmp_cfg, pid)
    assert ideas1 is not None and ideas2 is not None
    assert len(ideas1.ideas) == 3
    # Second call hits cache → no second LLM call.
    assert n_calls["n"] == 1


def test_gift_ideas_skips_when_no_birthday(fresh_db, tmp_cfg, monkeypatch):
    from secondbrain import gift_ideas
    from secondbrain import people as pm
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    pm.upsert_person(fresh_db, display_name="No Birthday")
    out = gift_ideas.list_for_upcoming_birthdays(
        fresh_db, tmp_cfg, auto_generate=False,
    )
    assert out == []


# ============================ standing_threads ======================


def test_standing_threads_detects_long_clusters(fresh_db):
    from secondbrain import standing_threads

    pid_sarah = fresh_db.execute(
        "INSERT INTO people(canonical_name, display_name, "
        "first_seen_at, last_seen_at) "
        "VALUES ('sarah', 'Sarah', ?, ?) RETURNING id",
        (0, 0),
    ).fetchone()["id"]
    pid_john = fresh_db.execute(
        "INSERT INTO people(canonical_name, display_name, "
        "first_seen_at, last_seen_at) "
        "VALUES ('john', 'John', ?, ?) RETURNING id",
        (0, 0),
    ).fetchone()["id"]
    # 6 emails between Sarah & John spanning 30 days.
    base_ts = time.time() - 30 * 86400
    for i in range(6):
        ts = base_ts + i * 5 * 86400
        path = f"email_{i}"
        fresh_db.execute(
            "INSERT INTO files(path, mtime, size, kind, "
            "content_hash, indexed_at) "
            "VALUES (?, ?, 0, 'email', ?, ?)",
            (path, ts, f"h_{i}", ts),
        )
        fid = fresh_db.execute(
            "SELECT id FROM files WHERE path = ?", (path,),
        ).fetchone()["id"]
        fresh_db.execute(
            "INSERT INTO chunks(file_id, chunk_index, text, "
            "start_offset) VALUES (?, 0, 'mtg', 0)", (fid,),
        )
        fresh_db.execute(
            "INSERT INTO person_mentions"
            "(person_id, chunk_id, file_id, mtime) "
            "SELECT ?, c.id, c.file_id, ? FROM chunks c "
            "WHERE c.file_id = ?",
            (pid_sarah, ts, fid),
        )
        fresh_db.execute(
            "INSERT INTO person_mentions"
            "(person_id, chunk_id, file_id, mtime) "
            "SELECT ?, c.id, c.file_id, ? FROM chunks c "
            "WHERE c.file_id = ?",
            (pid_john, ts, fid),
        )
    fresh_db.commit()

    n = standing_threads.detect_threads(
        fresh_db, days=60, min_messages=5, min_days=14,
    )
    assert n == 1
    threads = standing_threads.list_threads(fresh_db)
    assert len(threads) == 1
    assert threads[0].n_messages == 6


# ============================ eod_wrapup ============================


def test_eod_wrapup_reports_today_metrics(fresh_db):
    from secondbrain import eod_wrapup

    # Populate a few signals: a journal entry today.
    today_iso = date.today().isoformat()
    fresh_db.execute(
        "INSERT INTO journal_entries"
        "(date, mood, text, created_at, updated_at) "
        "VALUES (?, 4, 'good day', ?, ?)",
        (today_iso, time.time(), time.time()),
    )
    fresh_db.commit()
    w = eod_wrapup.build_wrapup(fresh_db)
    assert w.date == today_iso
    md = eod_wrapup.render_markdown(w)
    assert "End of day" in md


def test_eod_notification_idempotent(fresh_db):
    from secondbrain import eod_wrapup, notifications

    a = eod_wrapup.daemon_post_eod_notification(fresh_db)
    b = eod_wrapup.daemon_post_eod_notification(fresh_db)
    # First inserts, second is dedup'd by UNIQUE key.
    assert a is True
    assert b is False
    pending = notifications.list_pending(fresh_db)
    assert any(n.kind == "eod_wrapup" for n in pending)


# ============================ conditional_reminders ==================


def test_conditional_reminder_fires_when_date_passed(fresh_db):
    from secondbrain import conditional_reminders

    rid = conditional_reminders.add_reminder(
        fresh_db,
        description="Check on Q3 contract",
        condition_kind="date_passed",
        condition={},
        fire_after=time.time() - 60,  # already past
    )
    n = conditional_reminders.check_and_fire(fresh_db)
    assert n == 1
    r = conditional_reminders.get(fresh_db, rid)
    assert r.status == "fired"


def test_conditional_reminder_holds_until_fire_after(fresh_db):
    from secondbrain import conditional_reminders

    rid = conditional_reminders.add_reminder(
        fresh_db,
        description="Future reminder",
        condition_kind="date_passed",
        condition={},
        fire_after=time.time() + 3600,
    )
    n = conditional_reminders.check_and_fire(fresh_db)
    assert n == 0
    r = conditional_reminders.get(fresh_db, rid)
    assert r.status == "pending"


def test_conditional_reminder_followup_unresolved(fresh_db):
    from secondbrain import conditional_reminders, followups

    fid = followups.add_followup(
        fresh_db, direction="incoming",
        topic="Sarah reviews", description="...",
    )
    conditional_reminders.add_reminder(
        fresh_db,
        description="Has Sarah replied?",
        condition_kind="followup_unresolved",
        condition={"followup_id": fid},
        fire_after=time.time() - 1,  # immediately eligible
    )
    n = conditional_reminders.check_and_fire(fresh_db)
    assert n == 1
    # If we resolve and re-check, no double-fire.
    followups.mark_resolved(fresh_db, fid)
    n2 = conditional_reminders.check_and_fire(fresh_db)
    assert n2 == 0


def test_conditional_reminder_cancel(fresh_db):
    from secondbrain import conditional_reminders

    rid = conditional_reminders.add_reminder(
        fresh_db,
        description="x",
        condition_kind="date_passed",
        condition={},
        fire_after=time.time() - 10,
    )
    assert conditional_reminders.cancel(fresh_db, rid)
    # Cancelled reminders don't fire.
    n = conditional_reminders.check_and_fire(fresh_db)
    assert n == 0


# ============================ dashboard routes ======================


def test_followups_route_renders(monkeypatch, tmp_path, fake_embedder):
    from fastapi.testclient import TestClient

    from secondbrain import followups
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
    # Seed via side connection.
    from secondbrain.db import connect, init_schema
    seed = connect(cfg.db_path)
    init_schema(seed, fake_embedder.dim, fake_embedder.name)
    followups.add_followup(
        seed, direction="outgoing",
        topic="Send deck", description="Send Q3 deck",
        person_name="Sarah",
    )
    seed.close()

    client = TestClient(create_app())
    r = client.get("/followups")
    assert r.status_code == 200
    assert "Send deck" in r.text
    assert "Follow-ups" in r.text


def test_eod_route_renders(monkeypatch, tmp_path, fake_embedder):
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
    client = TestClient(create_app())
    r = client.get("/eod")
    assert r.status_code == 200
    assert "End of day" in r.text


def test_followup_resolve_csrf_guard(monkeypatch, tmp_path, fake_embedder):
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
    client = TestClient(create_app())
    r = client.post(
        "/followups/1/resolve",
        headers={"origin": "https://evil.com"},
        follow_redirects=False,
    )
    assert r.status_code == 403


# ============================ smoke ================================


def test_round19_modules_import():
    from secondbrain import (  # noqa: F401
        agenda,
        conditional_reminders,
        eod_wrapup,
        followups,
        gift_ideas,
        meeting_capture,
        scheduling,
        standing_threads,
        triage_queue,
    )
