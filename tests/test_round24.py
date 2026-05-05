"""Round 24 — fix systemic ``kind`` filter bug.

The audit found ~10 features that LOOK implemented but didn't work
for any Gmail / IMAP user. Production stores Gmail docs as
``kind='url'`` with ``gmail://thread/...`` paths and IMAP as
``kind='url'`` with ``imap://...`` paths. ~7 surfaces filtered by
``kind='email'`` / ``kind='message'`` and silently returned nothing.

These tests pin the fix by inserting files with the PRODUCTION
shape (``kind='url'`` + virtual paths) and asserting each affected
surface returns them.
"""

from __future__ import annotations

import time


def _seed_gmail_email(
    conn, *, file_id_path="gmail://thread/T1/message/M1",
    sender="alice@example.com", subject="urgent question",
    body=None, indexed_at=None,
):
    """Insert an email with the PRODUCTION Gmail shape."""
    indexed_at = indexed_at if indexed_at is not None else time.time()
    body = body or (
        f"From: {sender}\nSubject: {subject}\n\nNeed your input."
    )
    conn.execute(
        "INSERT INTO files(path, mtime, size, kind, content_hash, "
        "indexed_at) VALUES (?, 0, 0, 'url', ?, ?)",
        (file_id_path, f"h:{file_id_path}", indexed_at),
    )
    fid = conn.execute(
        "SELECT id FROM files WHERE path = ?", (file_id_path,),
    ).fetchone()["id"]
    conn.execute(
        "INSERT INTO chunks(file_id, chunk_index, text, start_offset) "
        "VALUES (?, 0, ?, 0)",
        (fid, body),
    )
    conn.commit()
    return fid


def _seed_imap_email(
    conn, *, file_id_path="imap://INBOX/42",
    sender="bob@example.com", subject="contract",
    body=None, indexed_at=None,
):
    """Insert an email with the PRODUCTION IMAP shape."""
    indexed_at = indexed_at if indexed_at is not None else time.time()
    body = body or (
        f"From: {sender}\nSubject: {subject}\nFolder: INBOX\n\n"
        "Please review."
    )
    conn.execute(
        "INSERT INTO files(path, mtime, size, kind, content_hash, "
        "indexed_at) VALUES (?, 0, 0, 'url', ?, ?)",
        (file_id_path, f"h:{file_id_path}", indexed_at),
    )
    fid = conn.execute(
        "SELECT id FROM files WHERE path = ?", (file_id_path,),
    ).fetchone()["id"]
    conn.execute(
        "INSERT INTO chunks(file_id, chunk_index, text, start_offset) "
        "VALUES (?, 0, ?, 0)",
        (fid, body),
    )
    conn.commit()
    return fid


# ============================ HIGH 1 — triage queue =====================


def test_triage_queue_finds_gmail_emails(fresh_db):
    """Round 24 fix: build_queue must surface Gmail-shaped files
    (kind='url', gmail:// path), not just kind='email' test data."""
    from secondbrain import email_assist, triage_queue
    email_assist._ensure_schema(fresh_db)
    fid = _seed_gmail_email(fresh_db)
    fresh_db.execute(
        "INSERT INTO email_classifications"
        "(file_id, label, confidence, classified_at) "
        "VALUES (?, 'urgent', 0.9, ?)", (fid, time.time()),
    )
    fresh_db.commit()
    queue = triage_queue.build_queue(fresh_db, hours=24)
    assert any(it.file_id == fid for it in queue), (
        f"Gmail email not found in queue; got "
        f"{[(i.file_id, i.path) for i in queue]}"
    )


def test_triage_queue_finds_imap_emails(fresh_db):
    from secondbrain import email_assist, triage_queue
    email_assist._ensure_schema(fresh_db)
    fid = _seed_imap_email(fresh_db)
    fresh_db.execute(
        "INSERT INTO email_classifications"
        "(file_id, label, confidence, classified_at) "
        "VALUES (?, 'urgent', 0.9, ?)", (fid, time.time()),
    )
    fresh_db.commit()
    queue = triage_queue.build_queue(fresh_db, hours=24)
    assert any(it.file_id == fid for it in queue)


# ============================ HIGH 3 — classify_due =====================


def test_classify_due_picks_up_gmail(fresh_db, monkeypatch):
    """classify_due must include both gmail:// and imap:// paths."""
    from secondbrain import email_assist
    email_assist._ensure_schema(fresh_db)
    gmail_fid = _seed_gmail_email(
        fresh_db,
        file_id_path="gmail://thread/T1/message/M1",
    )
    imap_fid = _seed_imap_email(
        fresh_db, file_id_path="imap://INBOX/42",
    )

    captured: list[int] = []

    def fake_classify_one(conn, file_id, *, cfg, classifier=None):
        captured.append(int(file_id))
        return True

    monkeypatch.setattr(email_assist, "classify_one", fake_classify_one)
    cfg = MagicMock()
    cfg.email_classify_max_per_tick = 100
    cfg.email_classify_age_days = 30
    email_assist.classify_due(fresh_db, cfg=cfg)
    assert gmail_fid in captured, (
        f"Gmail fid not classified; got {captured}"
    )
    assert imap_fid in captured, (
        f"IMAP fid not classified; got {captured}"
    )


from unittest.mock import MagicMock  # noqa: E402  (used in tests above)

# ============================ HIGH 4 — followups extractor ==============


def test_followups_extractor_picks_up_gmail(
    fresh_db, tmp_cfg, monkeypatch,
):
    """extract_from_recent_inputs must walk Gmail-shaped emails."""
    from secondbrain import followups
    fid = _seed_gmail_email(
        fresh_db, body="From: ben\nI'll send Sarah the deck Friday.",
    )
    captured: list[int] = []

    def fake_extract(conn, cfg, *, text, user_name,
                     source_kind, source_file_id, **kw):
        captured.append(int(source_file_id))
        return 0  # we just verify the call happened

    monkeypatch.setattr(
        followups, "extract_and_persist", fake_extract,
    )
    out = followups.extract_from_recent_inputs(
        fresh_db, tmp_cfg, hours=24, max_files=10,
    )
    assert fid in captured, (
        f"Gmail fid not visited by extractor; got {captured}"
    )
    assert out["files_scanned"] >= 1


# ============================ HIGH 5 — meeting_capture ==================


def test_meeting_capture_sees_imap_transcripts(fresh_db, tmp_cfg, monkeypatch):
    """daemon_capture_recent must walk IMAP-ingested transcripts
    (Granola/Otter pattern), not just local audio_video files."""
    from secondbrain import meeting_capture
    meeting_capture._ensure_schema(fresh_db)
    # Granola transcripts arrive via IMAP with a "transcript"
    # folder convention.
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, content_hash, "
        "indexed_at) VALUES (?, 0, 0, 'url', ?, ?)",
        (
            "imap://Transcripts/123",
            "h:granola1",
            time.time(),
        ),
    )
    fresh_db.commit()
    captured_ids: list[int] = []

    def fake_capture(conn, cfg, file_id, *, user_name, overwrite=False):
        captured_ids.append(int(file_id))
        return MagicMock()

    monkeypatch.setattr(meeting_capture, "capture", fake_capture)
    n = meeting_capture.daemon_capture_recent(
        fresh_db, tmp_cfg, max_per_run=5,
    )
    assert n >= 1
    assert len(captured_ids) >= 1


def test_meeting_capture_sees_transcript_paths(fresh_db, tmp_cfg, monkeypatch):
    """``transcript://`` virtual paths also match."""
    from secondbrain import meeting_capture
    meeting_capture._ensure_schema(fresh_db)
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, content_hash, "
        "indexed_at) VALUES (?, 0, 0, 'url', ?, ?)",
        (
            "transcript://meeting-2025-05-05",
            "h:t1",
            time.time(),
        ),
    )
    fresh_db.commit()
    captured: list[int] = []
    monkeypatch.setattr(
        meeting_capture, "capture",
        lambda conn, cfg, fid, **kw: (
            captured.append(int(fid)), MagicMock(),
        )[1],
    )
    n = meeting_capture.daemon_capture_recent(
        fresh_db, tmp_cfg, max_per_run=5,
    )
    assert n >= 1


# ============================ MED 6 — standing threads ================


def test_standing_threads_detect_walks_gmail(fresh_db):
    """detect_threads must walk gmail:// + imap:// rows. We don't
    have enough fixture machinery for the cluster heuristic, so
    this test is more about: does the SQL query EVEN return them?
    """
    from secondbrain.db import EMAIL_KIND_SQL
    _seed_gmail_email(fresh_db)
    _seed_imap_email(fresh_db)
    rows = fresh_db.execute(
        f"SELECT path FROM files f WHERE {EMAIL_KIND_SQL}",
    ).fetchall()
    paths = {r["path"] for r in rows}
    assert any(p.startswith("gmail://") for p in paths)
    assert any(p.startswith("imap://") for p in paths)


# ============================ MED 7 — agenda emails =====================


def test_agenda_email_threads_finds_gmail(fresh_db):
    """The 1:1 agenda's open-emails section must surface Gmail."""
    from secondbrain import agenda
    from secondbrain import people as pm
    pid = pm.upsert_person(fresh_db, display_name="Sarah", email="sarah@x")
    fid = _seed_gmail_email(
        fresh_db,
        sender="sarah@x",
        body="From: sarah@x\nSubject: Project update\n\nHey...",
    )
    # Wire the person to the file via person_mentions.
    chunk = fresh_db.execute(
        "SELECT id FROM chunks WHERE file_id = ?", (fid,),
    ).fetchone()
    fresh_db.execute(
        "INSERT INTO person_mentions(person_id, chunk_id, file_id, mtime) "
        "VALUES (?, ?, ?, ?)",
        (pid, chunk["id"], fid, time.time()),
    )
    fresh_db.commit()
    items = agenda._open_email_threads_with(fresh_db, pid)
    assert len(items) == 1, (
        f"Expected 1 email thread; got {items}"
    )


# ============================ MED 8 — auto-resolve candidates ===========


def test_auto_resolve_finds_gmail_sent_mail(
    fresh_db, tmp_cfg, monkeypatch,
):
    """Auto-resolve candidate query must surface gmail://-shaped
    sent items so the round-20 'I sent it' detector works."""
    from secondbrain import followups, followups_ops
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test")
    tmp_cfg.user_email = "ben@example.com"
    fid = followups.add_followup(
        fresh_db, direction="outgoing",
        topic="Send Q3 deck",
        description="Send Sarah the Q3 numbers deck",
        person_name="Sarah",
        promised_at=time.time() - 3600,
    )
    _seed_gmail_email(
        fresh_db,
        sender="ben@example.com",  # user is sender
        subject="Q3 deck",
        body=(
            "From: ben@example.com\nTo: sarah@x\nSubject: Q3 deck\n\n"
            "Hi Sarah, here's the Q3 deck you asked for."
        ),
    )
    monkeypatch.setattr(
        followups_ops, "_llm_check_resolution",
        lambda cfg, **kw: {
            "resolved": True, "confidence": 0.9,
            "evidence_file_id": kw["candidates"][0]["fid"],
            "evidence": "User sent the deck.",
        },
    )
    n = followups_ops.auto_resolve_from_sent_mail(
        fresh_db, tmp_cfg, hours=24,
    )
    assert n == 1
    assert followups.get(fresh_db, fid).status == "resolved"


# ============================ MED 9 — weekly letter stat ==============


def test_weekly_letter_emails_count_gmail(fresh_db):
    """The week's "emails" stat must count gmail:// + imap://."""
    from secondbrain import weekly_letter
    _seed_gmail_email(
        fresh_db, file_id_path="gmail://thread/A/message/1",
    )
    _seed_imap_email(
        fresh_db, file_id_path="imap://INBOX/100",
    )
    sigs = weekly_letter.assemble_signals(fresh_db)
    counts = sigs.counts
    assert counts.get("emails", 0) >= 2, (
        f"Expected >= 2 emails counted; got {counts}"
    )


# ============================ MED 10 — conditional reminders ==========


def test_no_reply_reminder_sees_inbound_gmail(fresh_db):
    """A ``no_reply_from`` conditional reminder must NOT fire when
    a reply HAS arrived via Gmail. Round-22 audit found this fired
    incorrectly because the kind filter never matched Gmail."""
    from secondbrain import conditional_reminders as cr
    cr._ensure_schema(fresh_db)
    sarah_email = "sarah@x"
    rid = cr.add_reminder(
        fresh_db,
        description="Nudge Sarah if she doesn't reply",
        condition_kind="no_reply_from",
        condition={"email": sarah_email, "since": time.time() - 86400},
        fire_after=time.time() - 1,  # already past fire window
    )
    # Seed an inbound Gmail from Sarah AFTER the since cutoff.
    _seed_gmail_email(
        fresh_db,
        sender=sarah_email,
        body=f"From: {sarah_email}\nSubject: Re: thing\n\nHere you go.",
    )
    # The reminder should NOT fire because Sarah HAS replied.
    fired = cr.check_and_fire(fresh_db)
    assert fired == 0, (
        "Reminder fired even though Sarah replied — kind filter "
        "didn't see the Gmail row"
    )
    r = cr.get(fresh_db, rid)
    assert r.status == "pending"  # not fired


# ============================ LOW 11 — actionable detection ============


def test_followups_open_makes_brief_actionable(fresh_db):
    """Round 24 fix: a brief with open follow-ups but no other
    sections is no longer classified as 'quiet'."""
    from secondbrain.daily_brief import DailyBrief, _has_actionable_content
    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[], assignments_due_soon=[],
        open_action_items=[], queue_top=[],
        watchlist_highlights=[],
        followups_open=[{
            "topic": "Send deck", "person": "Sarah",
            "direction": "outgoing", "age_days": 3.0,
        }],
        triage_today={},
    )
    assert _has_actionable_content(brief) is True


def test_triage_today_makes_brief_actionable(fresh_db):
    from secondbrain.daily_brief import DailyBrief, _has_actionable_content
    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[], assignments_due_soon=[],
        open_action_items=[], queue_top=[],
        watchlist_highlights=[],
        followups_open=[],
        triage_today={"count": 5, "top_senders": ["alice"]},
    )
    assert _has_actionable_content(brief) is True


def test_truly_quiet_brief_still_quiet(fresh_db):
    """A brief with no actionable content of any kind is still
    classified as quiet (so the revisit-suggestions path can fire)."""
    from secondbrain.daily_brief import DailyBrief, _has_actionable_content
    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[], assignments_due_soon=[],
        open_action_items=[], queue_top=[],
        watchlist_highlights=[],
        followups_open=[],
        triage_today={},
    )
    assert _has_actionable_content(brief) is False


# ============================ shared helper ============================


def test_email_kind_sql_constant_shape():
    """Sanity: the helper SQL contains the production prefixes."""
    from secondbrain.db import EMAIL_KIND_SQL
    assert "imap://" in EMAIL_KIND_SQL
    assert "gmail://" in EMAIL_KIND_SQL
    # Backward compat for legacy fixture data.
    assert "'email'" in EMAIL_KIND_SQL
    assert "'message'" in EMAIL_KIND_SQL


def test_transcript_kind_sql_constant_shape():
    from secondbrain.db import TRANSCRIPT_KIND_SQL
    assert "transcript" in TRANSCRIPT_KIND_SQL
    assert "audio_video" in TRANSCRIPT_KIND_SQL


def test_round24_modules_import():
    from secondbrain import (  # noqa: F401
        agenda,
        conditional_reminders,
        daily_brief,
        db,
        email_assist,
        followups,
        followups_ops,
        meeting_capture,
        standing_threads,
        triage_queue,
        weekly_letter,
    )
