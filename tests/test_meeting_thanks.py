"""Round 8 — auto thank-you emails after meetings.

Tests cover the deterministic plumbing (schema, attendee
classification, skip heuristics, transcript matching by title-token
overlap + mtime) plus the end-to-end pipeline from registration
through draft persistence with a stubbed LLM.
"""

from __future__ import annotations

import time
from unittest.mock import patch

from secondbrain import meeting_thanks

# ============================ helpers =================================

def _seed_transcript(
    conn, *, path, title, body, mtime,
):
    """Insert a transcript:// file + first chunk for matching tests."""
    cur = conn.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at) "
        "VALUES (?, ?, ?, 'document', ?)",
        (path, mtime, len(body), mtime),
    )
    fid = cur.lastrowid
    full = f"# {title}\n\n{body}"
    conn.execute(
        "INSERT INTO chunks(file_id, chunk_index, text) VALUES (?, 0, ?)",
        (fid, full),
    )
    conn.commit()
    return fid


class _FakeEvent:
    """Minimal CalendarEvent stand-in for tests — avoids the real
    google_calendar import path."""

    def __init__(
        self, *, event_id, title, starts_at, duration_seconds,
        attendees, organizer_email="",
    ):
        self.event_id = event_id
        self.title = title
        self.starts_at = starts_at
        self.duration_seconds = duration_seconds
        self.attendees = attendees
        self.organizer_email = organizer_email
        self.source = "test"
        self.url = ""
        self.location = ""
        self.description = ""
        self.calendar_name = ""


# ============================ schema ==================================

def test_schema_creates_meeting_thanks_table(fresh_db):
    meeting_thanks._ensure_schema(fresh_db)
    rows = fresh_db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name = 'meeting_thanks'",
    ).fetchall()
    assert any(r["name"] == "meeting_thanks" for r in rows)


# ============================ Attendee classification =================

def test_classify_attendees_external_only(tmp_cfg):
    own = {"acme.com"}
    external, internal = meeting_thanks._classify_attendees(
        ["alice@external.com", "bob@acme.com", "carol@another.com"],
        own_domains=own, organizer="me@acme.com",
    )
    assert "alice@external.com" in external
    assert "carol@another.com" in external
    assert "bob@acme.com" in internal
    assert "me@acme.com" not in (external + internal)


def test_classify_attendees_no_own_domains_treats_all_as_external():
    """Without configured own-domains, every attendee is external —
    the safer default than silently dropping the meeting."""
    external, internal = meeting_thanks._classify_attendees(
        ["alice@x.com", "bob@y.com"],
        own_domains=set(), organizer="",
    )
    assert sorted(external) == ["alice@x.com", "bob@y.com"]
    assert internal == []


def test_classify_attendees_skips_invalid():
    external, _ = meeting_thanks._classify_attendees(
        ["not-an-email", "", "real@x.com"],
        own_domains=set(), organizer="",
    )
    assert external == ["real@x.com"]


# ============================ Skip heuristics =========================

def test_looks_skippable_recurring_team_meetings():
    assert meeting_thanks._looks_skippable("Daily standup", 30 * 60)
    assert meeting_thanks._looks_skippable("Weekly team sync", 60 * 60)
    assert meeting_thanks._looks_skippable("ENG All Hands", 60 * 60)
    assert meeting_thanks._looks_skippable("Office hours", 60 * 60)


def test_looks_skippable_blocks():
    assert meeting_thanks._looks_skippable("focus block", 90 * 60)
    assert meeting_thanks._looks_skippable("OOO", 8 * 3600)


def test_looks_skippable_too_short():
    assert meeting_thanks._looks_skippable("intro chat", 5 * 60) is True


def test_looks_skippable_too_long():
    assert meeting_thanks._looks_skippable("workshop", 6 * 3600) is True


def test_does_not_skip_real_coffee_chat():
    assert meeting_thanks._looks_skippable("Coffee w/ Sarah", 30 * 60) is False
    assert meeting_thanks._looks_skippable(
        "Interview - Anthropic - Ben", 45 * 60,
    ) is False


# ============================ Title overlap ===========================

def test_normalise_title_strips_prefixes():
    assert meeting_thanks._normalise_title("[meeting] Coffee w/ Sarah") == "coffee w/ sarah"
    assert meeting_thanks._normalise_title("Re: Q3 review") == "q3 review"


def test_title_overlap_perfect_match():
    score = meeting_thanks._title_overlap(
        "Coffee chat with Sarah",
        "Coffee chat with Sarah",
    )
    assert score == 1.0


def test_title_overlap_partial():
    score = meeting_thanks._title_overlap(
        "Coffee w/ Sarah",
        "Sarah <> Ben sync",
    )
    # "sarah" overlaps; jaccard = 1/4 = 0.25 (under threshold)
    assert 0.0 < score < 0.5


def test_title_overlap_unrelated():
    assert meeting_thanks._title_overlap(
        "Standup", "Quarterly review with the board",
    ) == 0.0


# ============================ Transcript matching =====================

def test_find_transcript_matches_on_mtime_and_title(fresh_db):
    """A transcript whose mtime is within 2h of meeting end AND
    whose title overlaps the event title gets matched."""
    ends_at = time.time() - 30 * 60  # 30 min ago
    _seed_transcript(
        fresh_db,
        path="transcript://granola/abc",
        title="Coffee chat with Sarah Chen",
        body="we talked about onboarding and her recent move to product",
        mtime=ends_at + 60,  # transcript dropped a minute after meeting end
    )
    path = meeting_thanks._find_transcript_for_event(
        fresh_db, title="Coffee chat with Sarah Chen", ends_at=ends_at,
    )
    assert path == "transcript://granola/abc"


def test_find_transcript_skips_when_mtime_too_far(fresh_db):
    """Transcript from 5 hours ago is not the one for a meeting
    that just ended."""
    ends_at = time.time()
    _seed_transcript(
        fresh_db,
        path="transcript://granola/old",
        title="Coffee chat with Sarah Chen",
        body="...",
        mtime=ends_at - 5 * 3600,
    )
    path = meeting_thanks._find_transcript_for_event(
        fresh_db, title="Coffee chat with Sarah Chen", ends_at=ends_at,
    )
    assert path is None


def test_find_transcript_skips_when_title_doesnt_overlap(fresh_db):
    """Same time, completely different topic → no match."""
    ends_at = time.time()
    _seed_transcript(
        fresh_db,
        path="transcript://granola/different",
        title="Engineering all-hands",
        body="...",
        mtime=ends_at,
    )
    path = meeting_thanks._find_transcript_for_event(
        fresh_db, title="Coffee chat with Sarah", ends_at=ends_at,
    )
    assert path is None


def test_find_transcript_picks_best_among_candidates(fresh_db):
    """When two transcripts match the time window, pick the one
    with higher title overlap."""
    ends_at = time.time()
    _seed_transcript(
        fresh_db, path="transcript://granola/poor",
        title="Engineering sync", body="...",
        mtime=ends_at - 10 * 60,
    )
    _seed_transcript(
        fresh_db, path="transcript://granola/good",
        title="Coffee chat with Sarah",
        body="...", mtime=ends_at - 30 * 60,
    )
    path = meeting_thanks._find_transcript_for_event(
        fresh_db, title="Coffee with Sarah", ends_at=ends_at,
    )
    assert path == "transcript://granola/good"


# ============================ Registration ============================

def test_register_pending_thanks_inserts_row(fresh_db, tmp_cfg):
    """Calendar event with an external attendee → new pending row."""
    ev = _FakeEvent(
        event_id="evt-1",
        title="Coffee w/ Sarah",
        starts_at=time.time() - 3600,
        duration_seconds=30 * 60,
        attendees=["sarah@external.com"],
    )
    with patch.object(
        meeting_thanks, "iter_recent_events", return_value=[ev],
        create=True,
    ), patch(
        "secondbrain.event_briefing.iter_recent_events",
        return_value=[ev],
    ):
        n = meeting_thanks.register_pending_thanks(fresh_db, tmp_cfg)
    assert n == 1
    pending = meeting_thanks.list_pending(fresh_db)
    assert len(pending) == 1
    assert pending[0].event_title == "Coffee w/ Sarah"
    assert "sarah@external.com" in pending[0].attendees


def test_register_pending_thanks_skips_internal_only(fresh_db, tmp_cfg):
    """Pure-internal meeting (everyone @user-domain) gets dropped."""
    tmp_cfg.imap_username = "me@acme.com"
    ev = _FakeEvent(
        event_id="evt-internal",
        title="Team sync",
        starts_at=time.time() - 3600,
        duration_seconds=30 * 60,
        attendees=["bob@acme.com", "alice@acme.com"],
        organizer_email="me@acme.com",
    )
    with patch(
        "secondbrain.event_briefing.iter_recent_events",
        return_value=[ev],
    ):
        n = meeting_thanks.register_pending_thanks(fresh_db, tmp_cfg)
    assert n == 0


def test_register_pending_thanks_marks_skip_for_standup(fresh_db, tmp_cfg):
    """Recurring internal-style meetings get auto-skipped (not dropped
    — we still record them so the user can see the decision)."""
    ev = _FakeEvent(
        event_id="evt-standup",
        title="Daily standup",
        starts_at=time.time() - 3600,
        duration_seconds=15 * 60,
        attendees=["external@vendor.com"],  # has external but title skippable
    )
    with patch(
        "secondbrain.event_briefing.iter_recent_events",
        return_value=[ev],
    ):
        meeting_thanks.register_pending_thanks(fresh_db, tmp_cfg)
    all_rows = meeting_thanks.list_all(fresh_db)
    assert len(all_rows) == 1
    assert all_rows[0].status == "skipped"


def test_register_pending_thanks_idempotent(fresh_db, tmp_cfg):
    ev = _FakeEvent(
        event_id="evt-dup",
        title="Coffee w/ Anna",
        starts_at=time.time() - 3600,
        duration_seconds=30 * 60,
        attendees=["anna@external.com"],
    )
    with patch(
        "secondbrain.event_briefing.iter_recent_events",
        return_value=[ev],
    ):
        n1 = meeting_thanks.register_pending_thanks(fresh_db, tmp_cfg)
        n2 = meeting_thanks.register_pending_thanks(fresh_db, tmp_cfg)
    assert n1 == 1
    assert n2 == 0


def test_register_promotes_to_ready_when_transcript_present(
    fresh_db, tmp_cfg,
):
    """A meeting with a matching transcript should land directly in
    'ready' status, not 'pending_context'."""
    ends_at = time.time() - 3600
    _seed_transcript(
        fresh_db, path="transcript://granola/coffee",
        title="Coffee chat with Sarah",
        body="onboarding talk",
        mtime=ends_at + 60,
    )
    ev = _FakeEvent(
        event_id="evt-ready",
        title="Coffee chat with Sarah",
        starts_at=ends_at - 30 * 60,
        duration_seconds=30 * 60,
        attendees=["sarah@external.com"],
    )
    with patch(
        "secondbrain.event_briefing.iter_recent_events",
        return_value=[ev],
    ):
        meeting_thanks.register_pending_thanks(fresh_db, tmp_cfg)
    rows = meeting_thanks.list_pending(fresh_db)
    assert len(rows) == 1
    assert rows[0].status == "ready"
    assert rows[0].transcript_path == "transcript://granola/coffee"


# ============================ Rematch =================================

def test_rematch_transcripts_promotes_pending_to_ready(fresh_db, tmp_cfg):
    """Pending row + later-arriving transcript should promote on
    rematch."""
    ends_at = time.time() - 3600
    ev = _FakeEvent(
        event_id="evt-rematch",
        title="Catch-up with Marcus",
        starts_at=ends_at - 30 * 60,
        duration_seconds=30 * 60,
        attendees=["marcus@external.com"],
    )
    with patch(
        "secondbrain.event_briefing.iter_recent_events",
        return_value=[ev],
    ):
        meeting_thanks.register_pending_thanks(fresh_db, tmp_cfg)
    # No transcript yet → status is pending_context.
    rows = meeting_thanks.list_pending(fresh_db)
    assert rows[0].status == "pending_context"
    # Transcript shows up after the fact.
    _seed_transcript(
        fresh_db, path="transcript://otter/123",
        title="Catch-up with Marcus",
        body="we discussed the side project",
        mtime=ends_at + 120,
    )
    n = meeting_thanks.rematch_transcripts(fresh_db)
    assert n == 1
    rows = meeting_thanks.list_pending(fresh_db)
    assert rows[0].status == "ready"
    assert rows[0].transcript_path == "transcript://otter/123"


# ============================ Set context / skip ======================

def test_set_context_promotes_to_ready(fresh_db, tmp_cfg):
    ev = _FakeEvent(
        event_id="evt-ctx", title="Mentor chat", starts_at=time.time(),
        duration_seconds=45 * 60,
        attendees=["mentor@external.com"],
    )
    with patch(
        "secondbrain.event_briefing.iter_recent_events",
        return_value=[ev],
    ):
        meeting_thanks.register_pending_thanks(fresh_db, tmp_cfg)
    [row] = meeting_thanks.list_pending(fresh_db)
    ok = meeting_thanks.set_context(
        fresh_db, row.id,
        "Talked about my career growth + her recent role change.",
    )
    assert ok is True
    refreshed = meeting_thanks.get(fresh_db, row.id)
    assert refreshed is not None
    assert refreshed.status == "ready"
    assert "career growth" in refreshed.user_context


def test_set_context_empty_text_is_no_op(fresh_db, tmp_cfg):
    ev = _FakeEvent(
        event_id="evt-empty", title="x", starts_at=time.time(),
        duration_seconds=30 * 60, attendees=["x@external.com"],
    )
    with patch(
        "secondbrain.event_briefing.iter_recent_events",
        return_value=[ev],
    ):
        meeting_thanks.register_pending_thanks(fresh_db, tmp_cfg)
    [row] = meeting_thanks.list_pending(fresh_db)
    assert meeting_thanks.set_context(fresh_db, row.id, "   ") is False
    assert meeting_thanks.set_context(fresh_db, row.id, "") is False


def test_mark_skipped_flips_status(fresh_db, tmp_cfg):
    ev = _FakeEvent(
        event_id="evt-skip", title="One-off chat", starts_at=time.time(),
        duration_seconds=30 * 60, attendees=["x@external.com"],
    )
    with patch(
        "secondbrain.event_briefing.iter_recent_events",
        return_value=[ev],
    ):
        meeting_thanks.register_pending_thanks(fresh_db, tmp_cfg)
    [row] = meeting_thanks.list_pending(fresh_db)
    assert meeting_thanks.mark_skipped(fresh_db, row.id) is True
    # No longer in pending.
    assert meeting_thanks.list_pending(fresh_db) == []
    refreshed = meeting_thanks.get(fresh_db, row.id)
    assert refreshed.status == "skipped"


# ============================ Drafting ================================

def test_generate_thanks_draft_persists_with_voice_metadata(
    fresh_db, tmp_cfg,
):
    """End-to-end with stubbed LLM. Verify the draft lands in
    email_drafts with kind='meeting_thanks' in metadata."""
    import json as _json

    from secondbrain import email_assist

    # Schema the email tables.
    email_assist._ensure_schema(fresh_db)

    # Persist a voice profile so the critique loop has data.
    profile = email_assist.VoiceProfile(
        greetings=["hi {name},"], sign_offs=["—Ben"],
        avg_sentence_words=8, avg_reply_chars=180,
        contraction_rate=0.85, exclamation_rate=0.1, emoji_rate=0,
        common_openers=[], common_closers=[],
        avoided_phrases=["Best regards,"],
        register_notes="warm casual.", n_samples=10,
    )
    email_assist._save_voice_profile(fresh_db, profile)

    # Seed a meeting w/ user_context.
    ends_at = time.time() - 3600
    ev = _FakeEvent(
        event_id="evt-coffee",
        title="Coffee w/ Sarah",
        starts_at=ends_at - 30 * 60,
        duration_seconds=30 * 60,
        attendees=["sarah@external.com"],
    )
    with patch(
        "secondbrain.event_briefing.iter_recent_events",
        return_value=[ev],
    ):
        meeting_thanks.register_pending_thanks(fresh_db, tmp_cfg)
    [row] = meeting_thanks.list_pending(fresh_db)
    meeting_thanks.set_context(
        fresh_db, row.id,
        "Talked about her move to product, I committed to "
        "introducing her to Marcus.",
    )

    # Stub the drafter LLM to return a structured response.
    def _drafter(*, prompt, cfg):
        return {
            "primary": (
                "Hi Sarah, great chatting today — I'll follow up "
                "with the Marcus intro this week. — Ben"
            ),
            "alternative": (
                "Sarah, thanks for the time today. I'll send the "
                "Marcus intro shortly. Best, Ben"
            ),
            "reasoning": "Casual coffee chat, voice match.",
            "confidence": 0.9,
            "open_questions": [],
        }
    # Stub the critique to return OK so we don't go down the
    # regenerate path — that's a separate code path.
    with patch.object(
        email_assist, "critique_draft_against_voice",
        return_value="OK",
    ):
        draft_id = meeting_thanks.generate_thanks_draft(
            fresh_db, tmp_cfg, row.id,
            user_name="Ben", drafter=_drafter,
        )
    assert draft_id is not None

    # Status should be drafted now.
    refreshed = meeting_thanks.get(fresh_db, row.id)
    assert refreshed.status == "drafted"
    assert refreshed.draft_id == draft_id

    # Draft should be discoverable through email_assist surfaces.
    drafts = email_assist.list_unsent_drafts(fresh_db)
    assert len(drafts) == 1
    d = drafts[0]
    assert "Marcus intro" in d.draft_text
    # Metadata should mark this as a meeting_thanks kind.
    raw = fresh_db.execute(
        "SELECT metadata_json FROM email_drafts WHERE id = ?",
        (draft_id,),
    ).fetchone()["metadata_json"]
    meta = _json.loads(raw)
    assert meta["kind"] == "meeting_thanks"
    assert meta["meeting_thanks_id"] == row.id
    assert meta["meeting_event_title"] == "Coffee w/ Sarah"


def test_generate_thanks_draft_skips_pending_status(
    fresh_db, tmp_cfg,
):
    """Status='pending_context' is not draftable until context shows up."""
    ev = _FakeEvent(
        event_id="evt-pending", title="x", starts_at=time.time(),
        duration_seconds=30 * 60, attendees=["x@external.com"],
    )
    with patch(
        "secondbrain.event_briefing.iter_recent_events",
        return_value=[ev],
    ):
        meeting_thanks.register_pending_thanks(fresh_db, tmp_cfg)
    [row] = meeting_thanks.list_pending(fresh_db)
    assert row.status == "pending_context"
    out = meeting_thanks.generate_thanks_draft(fresh_db, tmp_cfg, row.id)
    assert out is None


def test_generate_thanks_draft_returns_none_when_meeting_missing(
    fresh_db, tmp_cfg,
):
    out = meeting_thanks.generate_thanks_draft(fresh_db, tmp_cfg, 99999)
    assert out is None


# ============================ mark_sent_for_draft =====================

def test_mark_sent_for_draft_flips_status(fresh_db, tmp_cfg):
    """When the user marks the linked draft sent, the meeting_thanks
    row should auto-flip to 'sent' too."""
    from secondbrain import email_assist

    email_assist._ensure_schema(fresh_db)
    ev = _FakeEvent(
        event_id="evt-sent", title="Lunch w/ Anna",
        starts_at=time.time() - 3600,
        duration_seconds=60 * 60,
        attendees=["anna@external.com"],
    )
    with patch(
        "secondbrain.event_briefing.iter_recent_events",
        return_value=[ev],
    ):
        meeting_thanks.register_pending_thanks(fresh_db, tmp_cfg)
    [row] = meeting_thanks.list_pending(fresh_db)
    meeting_thanks.set_context(
        fresh_db, row.id, "talked about her travel plans",
    )
    with patch.object(
        email_assist, "critique_draft_against_voice", return_value="OK",
    ):
        draft_id = meeting_thanks.generate_thanks_draft(
            fresh_db, tmp_cfg, row.id, user_name="Ben",
            drafter=lambda **kw: {
                "primary": "Hi Anna! Safe travels — Ben",
                "alternative": "",
                "reasoning": "", "confidence": 0.7, "open_questions": [],
            },
        )
    assert draft_id is not None
    # User marks the draft sent (via /drafts/<id>/sent).
    email_assist.mark_draft_sent(fresh_db, draft_id)
    flipped = meeting_thanks.mark_sent_for_draft(fresh_db, draft_id)
    assert flipped is True
    refreshed = meeting_thanks.get(fresh_db, row.id)
    assert refreshed.status == "sent"


# ============================ Daemon scheduler =======================

def test_daemon_registers_meeting_thanks_job(fresh_db, tmp_cfg):
    from secondbrain.daemon import _build_daemon_scheduler

    sched = _build_daemon_scheduler(
        tmp_cfg, fresh_db, embedder=None, reranker=None,
    )
    assert "meeting_thanks" in set(sched.names())
