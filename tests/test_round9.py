"""Round 9 tests — meeting prep (A), stale connections (B), and the
structured task-promise extractor (C).

The shared person-context helper feeds A and C, so it gets the most
coverage. Each feature also has wiring tests for daemon registration
+ render-block output.
"""

from __future__ import annotations

import time
from unittest.mock import patch

# ============================ helpers =================================

def _seed_person(conn, *, display_name, email="", company="",
                 last_seen_at=None, mention_count=0):
    """Insert a row directly so we can pin last_seen_at + mention_count
    deterministically (the public upsert_person bumps mtime to now)."""
    from secondbrain import people as people_mod
    pid = people_mod.upsert_person(
        conn, display_name=display_name, email=email, company=company,
    )
    n = last_seen_at if last_seen_at is not None else time.time()
    conn.execute(
        "UPDATE people SET last_seen_at = ?, mention_count = ? "
        "WHERE id = ?",
        (n, mention_count, pid),
    )
    conn.commit()
    return pid


def _seed_chunk(conn, *, path, mtime, text, kind="document"):
    cur = conn.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (path, mtime, len(text), kind, mtime),
    )
    fid = cur.lastrowid
    conn.execute(
        "INSERT INTO chunks(file_id, chunk_index, text) VALUES (?, 0, ?)",
        (fid, text),
    )
    conn.commit()
    return fid


# ============================ Shared helper (gather_full_context) ====

def test_gather_full_context_basic_profile(fresh_db):
    """Returns the person + days-since stats even when no other data."""
    from secondbrain import people as people_mod

    pid = _seed_person(
        fresh_db, display_name="Sarah Chen",
        email="sarah@example.com",
        last_seen_at=time.time() - 30 * 86400,
        mention_count=5,
    )
    ctx = people_mod.gather_full_context(fresh_db, pid)
    assert ctx is not None
    assert ctx.person.display_name == "Sarah Chen"
    assert ctx.days_since_seen == 30
    assert ctx.recent_mentions == []
    assert ctx.prior_emails == []
    assert ctx.open_tasks == []


def test_gather_full_context_picks_up_emails(fresh_db):
    """A file with 'From: <alias>' becomes a prior email."""
    from secondbrain import people as people_mod

    pid = _seed_person(fresh_db, display_name="Sarah Chen", mention_count=2)
    _seed_chunk(
        fresh_db, path="imap://msgid/abc",
        mtime=time.time() - 86400,
        text="# subject\nFrom: Sarah Chen <s@x.com>\nFolder: INBOX\n\nbody",
    )
    ctx = people_mod.gather_full_context(fresh_db, pid)
    assert ctx is not None
    assert any(
        p == "imap://msgid/abc"
        for p, _t in ctx.prior_emails
    )


def test_gather_full_context_picks_up_open_tasks(fresh_db):
    """A task whose text mentions the person's alias surfaces as
    open_tasks. Heuristic substring match — round-9-C will replace
    this with FK matching, but the helper handles both."""
    from secondbrain import people as people_mod

    pid = _seed_person(
        fresh_db, display_name="Sarah Chen", mention_count=3,
    )
    fresh_db.execute(
        "INSERT INTO tasks"
        "(text, text_lower, source_path, source_title, status, created_at) "
        "VALUES ('email Sarah Chen the deck', "
        "'email sarah chen the deck', 't://x', 'meeting', 'open', ?)",
        (time.time(),),
    )
    fresh_db.commit()
    ctx = people_mod.gather_full_context(fresh_db, pid)
    assert any("Sarah Chen" in t for _id, t in ctx.open_tasks)


def test_gather_full_context_by_alias_resolves(fresh_db):
    """Alias lookup wrapper hits the same person."""
    from secondbrain import people as people_mod

    _seed_person(fresh_db, display_name="Marcus Hill", mention_count=2)
    ctx = people_mod.gather_full_context_by_alias(fresh_db, "Marcus Hill")
    assert ctx is not None
    assert ctx.person.display_name == "Marcus Hill"


def test_gather_full_context_unknown_alias_returns_none(fresh_db):
    from secondbrain import people as people_mod
    assert people_mod.gather_full_context_by_alias(
        fresh_db, "nobody@nowhere.com",
    ) is None


# ============================ A: meeting_prep =========================

class _FakeEvent:
    """Stand-in for CalendarEvent used by meeting_prep."""

    def __init__(self, *, event_id, title, starts_at, duration_seconds,
                 attendees, organizer_email="", location=""):
        self.event_id = event_id
        self.title = title
        self.starts_at = starts_at
        self.duration_seconds = duration_seconds
        self.attendees = attendees
        self.organizer_email = organizer_email
        self.location = location
        self.url = ""
        self.description = ""
        self.calendar_name = ""
        self.source = "test"


def test_meeting_prep_filters_internal_only(fresh_db, tmp_cfg):
    """Pure-internal meetings get dropped before prep is even built."""
    from secondbrain import meeting_prep
    tmp_cfg.imap_username = "me@acme.com"
    ev = _FakeEvent(
        event_id="e1", title="Team sync",
        starts_at=time.time() + 3600,
        duration_seconds=30 * 60,
        attendees=["bob@acme.com", "alice@acme.com"],
        organizer_email="me@acme.com",
    )
    with patch(
        "secondbrain.event_briefing.iter_upcoming_events",
        return_value=[ev],
    ):
        out = meeting_prep.iter_upcoming_external_meetings(tmp_cfg)
    assert out == []


def test_meeting_prep_skips_recurring_patterns(fresh_db, tmp_cfg):
    """Recurring patterns (standup) get filtered even with externals."""
    from secondbrain import meeting_prep
    ev = _FakeEvent(
        event_id="e2", title="Daily standup",
        starts_at=time.time() + 3600,
        duration_seconds=30 * 60,
        attendees=["external@vendor.com"],
    )
    with patch(
        "secondbrain.event_briefing.iter_upcoming_events",
        return_value=[ev],
    ):
        out = meeting_prep.iter_upcoming_external_meetings(tmp_cfg)
    assert out == []


def test_meeting_prep_keeps_real_coffee_chat(fresh_db, tmp_cfg):
    from secondbrain import meeting_prep
    ev = _FakeEvent(
        event_id="e3", title="Coffee w/ Sarah",
        starts_at=time.time() + 3600,
        duration_seconds=30 * 60,
        attendees=["sarah@external.com"],
    )
    with patch(
        "secondbrain.event_briefing.iter_upcoming_events",
        return_value=[ev],
    ):
        out = meeting_prep.iter_upcoming_external_meetings(tmp_cfg)
    assert len(out) == 1


def test_build_prep_first_time_attendee(fresh_db, tmp_cfg):
    """Unknown attendee → empty context, but display name derived
    from the email handle."""
    from secondbrain import meeting_prep
    ev = _FakeEvent(
        event_id="e4", title="Intro call",
        starts_at=time.time() + 3600,
        duration_seconds=30 * 60,
        attendees=["new.contact@bigco.com"],
    )
    prep = meeting_prep.build_prep(fresh_db, tmp_cfg, ev)
    assert prep.title == "Intro call"
    assert len(prep.attendees) == 1
    a = prep.attendees[0]
    assert a.email == "new.contact@bigco.com"
    assert a.name == "New Contact"
    assert a.n_prior_emails == 0
    assert a.n_open_tasks == 0


def test_build_prep_pulls_known_person_context(fresh_db, tmp_cfg):
    """Calendar events carry attendee emails; the prep builder
    alias-resolves them against the people table."""
    from secondbrain import meeting_prep
    from secondbrain import people as people_mod
    pid = _seed_person(
        fresh_db, display_name="Sarah Chen",
        email="sarah@example.com",
        last_seen_at=time.time() - 60 * 86400,
        mention_count=10,
    )
    # Add the email as an alias so find_by_alias can resolve it.
    people_mod.add_alias(fresh_db, pid, "sarah@example.com")
    people_mod.clear_alias_cache()
    ev = _FakeEvent(
        event_id="e5", title="Catch up with Sarah",
        starts_at=time.time() + 3600,
        duration_seconds=30 * 60,
        attendees=["sarah@example.com"],
    )
    prep = meeting_prep.build_prep(fresh_db, tmp_cfg, ev)
    assert len(prep.attendees) == 1
    a = prep.attendees[0]
    assert a.name == "Sarah Chen"
    assert a.days_since_seen == 60


def test_render_prep_markdown_first_time_block():
    from secondbrain.meeting_prep import (
        AttendeePrep,
        MeetingPrep,
        render_prep_markdown,
    )
    p = MeetingPrep(
        event_id="e6", title="Intro w/ Anna",
        starts_at=time.time() + 3600,
        when_str="Today 14:00", duration_minutes=30,
        location="", organizer="",
        attendees=[
            AttendeePrep(
                name="Anna Doe", email="anna@x.com",
                days_since_seen=0, n_prior_emails=0, n_open_tasks=0,
            ),
        ],
    )
    md = render_prep_markdown(p)
    assert "Intro w/ Anna" in md
    assert "Anna Doe" in md
    assert "First time" in md


def test_render_prep_markdown_known_person_blocks():
    from secondbrain.meeting_prep import (
        AttendeePrep,
        MeetingPrep,
        render_prep_markdown,
    )
    p = MeetingPrep(
        event_id="e7", title="Sync with Sarah",
        starts_at=time.time() + 3600,
        when_str="Tomorrow 10:00", duration_minutes=45,
        location="Zoom", organizer="me@x.com",
        attendees=[
            AttendeePrep(
                name="Sarah Chen", email="sarah@x.com",
                days_since_seen=42, n_prior_emails=3, n_open_tasks=2,
                open_task_lines=["send Sarah the design doc"],
                co_topics=["onboarding", "Q3 roadmap"],
                recent_mention_paths=["transcript://granola/abc"],
            ),
        ],
    )
    md = render_prep_markdown(p)
    assert "Sarah Chen" in md
    assert "42d since last seen" in md
    assert "send Sarah the design doc" in md
    assert "onboarding" in md
    assert "transcript://granola/abc" in md


def test_daemon_registers_meeting_prep_prefetch(fresh_db, tmp_cfg):
    from secondbrain.daemon import _build_daemon_scheduler
    sched = _build_daemon_scheduler(
        tmp_cfg, fresh_db, embedder=None, reranker=None,
    )
    assert "meeting_prep_prefetch" in set(sched.names())


# ============================ B: stale connections ===================

def test_find_stale_connections_excludes_recent(fresh_db):
    """Someone seen 30 days ago is below the 60-day threshold."""
    from secondbrain import connections
    _seed_person(
        fresh_db, display_name="Recent Friend",
        last_seen_at=time.time() - 30 * 86400, mention_count=20,
    )
    out = connections.find_stale_connections(fresh_db)
    assert all(c.name != "Recent Friend" for c in out)


def test_find_stale_connections_excludes_low_mentions(fresh_db):
    """Need at least 3 mentions before someone's a 'real' connection."""
    from secondbrain import connections
    _seed_person(
        fresh_db, display_name="Stranger",
        last_seen_at=time.time() - 200 * 86400, mention_count=1,
    )
    out = connections.find_stale_connections(fresh_db)
    assert all(c.name != "Stranger" for c in out)


def test_find_stale_connections_ranks_by_score(fresh_db):
    """Higher mention_count + longer silence → higher score → ranks
    above lower-mention more-recent contacts."""
    from secondbrain import connections
    _seed_person(
        fresh_db, display_name="Long Gone Boss",
        last_seen_at=time.time() - 180 * 86400, mention_count=50,
    )
    _seed_person(
        fresh_db, display_name="Casual Acquaintance",
        last_seen_at=time.time() - 70 * 86400, mention_count=4,
    )
    out = connections.find_stale_connections(fresh_db)
    assert len(out) == 2
    # The higher-mention longer-gone person should rank first.
    assert out[0].name == "Long Gone Boss"


def test_find_stale_connections_caps_at_max_age(fresh_db):
    """Past max_age (365d), the silence bonus saturates so a 5-yr-gone
    person doesn't get an unbounded score."""
    from secondbrain import connections
    _seed_person(
        fresh_db, display_name="Five Years Gone",
        last_seen_at=time.time() - 5 * 365 * 86400, mention_count=10,
    )
    _seed_person(
        fresh_db, display_name="Year Gone",
        last_seen_at=time.time() - 365 * 86400, mention_count=10,
    )
    out = connections.find_stale_connections(fresh_db)
    # Both should appear, with similar scores (saturated). 5-year
    # shouldn't be wildly higher than 1-year.
    by_name = {c.name: c.score for c in out}
    assert "Five Years Gone" in by_name
    assert "Year Gone" in by_name
    assert abs(by_name["Five Years Gone"] - by_name["Year Gone"]) < 0.1


def test_render_stale_block_empty_returns_empty():
    from secondbrain.connections import render_stale_block
    assert render_stale_block([]) == ""


def test_render_stale_block_includes_meta():
    from secondbrain.connections import StaleConnection, render_stale_block
    c = StaleConnection(
        person_id=1, name="Sarah Chen", email="s@x.com",
        company="Acme", role="PM",
        days_since_seen=120, mention_count=15, score=10.0,
    )
    out = render_stale_block([c])
    assert "Sarah Chen" in out
    assert "4 months" in out
    assert "Acme" in out
    assert "PM" in out


def test_stale_connection_in_daily_brief(fresh_db, tmp_cfg):
    """End-to-end: brief assembly should pull stale connections when
    candidates exist."""
    from secondbrain.daily_brief import assemble_brief
    _seed_person(
        fresh_db, display_name="Old Friend",
        last_seen_at=time.time() - 100 * 86400, mention_count=15,
    )
    # Stub calendar fetch to avoid network.
    with patch(
        "secondbrain.daily_brief._today_events", return_value=[],
    ), patch(
        "secondbrain.daily_brief._upcoming_preps_section", return_value=[],
    ):
        brief = assemble_brief(tmp_cfg, fresh_db)
    assert any(c.name == "Old Friend" for c in brief.stale_connections)


# ============================ C: promise extractor ===================

def test_extract_promises_from_text_returns_structured(tmp_cfg):
    """Stub the LLM call and verify we parse the JSON shape."""
    from secondbrain import tasks
    fake = {
        "promises": [
            {
                "text": "send Sarah the design doc",
                "recipient": "Sarah",
                "due_hint": "Friday",
            },
            {
                "text": "follow up with Marcus",
                "recipient": "Marcus",
                "due_hint": "",
            },
        ],
    }
    with patch(
        "secondbrain.email_assist._llm_json_call",
        return_value=fake,
    ):
        out = tasks.extract_promises_from_text(
            "long enough to clear the min-length filter " * 5,
            cfg=tmp_cfg,
        )
    assert len(out) == 2
    assert out[0]["recipient"] == "Sarah"
    assert out[0]["due_hint"] == "Friday"


def test_extract_promises_from_text_short_returns_empty(tmp_cfg):
    """Inputs under 50 chars don't even hit the LLM."""
    from secondbrain import tasks
    out = tasks.extract_promises_from_text("hi", cfg=tmp_cfg)
    assert out == []


def test_extract_promises_no_cfg_returns_empty():
    from secondbrain import tasks
    out = tasks.extract_promises_from_text("plenty of text " * 20)
    assert out == []


def test_extract_promises_handles_llm_failure(tmp_cfg):
    from secondbrain import tasks
    with patch(
        "secondbrain.email_assist._llm_json_call",
        return_value=None,
    ):
        out = tasks.extract_promises_from_text(
            "lots of text " * 30, cfg=tmp_cfg,
        )
    assert out == []


def test_extract_promises_filters_invalid_entries(tmp_cfg):
    """Entries with no ``text`` field get dropped on the floor."""
    from secondbrain import tasks
    fake = {
        "promises": [
            {"text": "", "recipient": "x"},
            {"recipient": "no text key"},
            {"text": "send the link", "recipient": "Sarah"},
        ],
    }
    with patch(
        "secondbrain.email_assist._llm_json_call",
        return_value=fake,
    ):
        out = tasks.extract_promises_from_text(
            "lots of text " * 30, cfg=tmp_cfg,
        )
    assert len(out) == 1
    assert out[0]["text"] == "send the link"


def test_materialize_promises_persists_with_recipient(fresh_db, tmp_cfg):
    """End-to-end: seed a transcript, stub the extractor, verify the
    new task has the recipient_person_id matched."""
    from secondbrain import tasks
    pid = _seed_person(fresh_db, display_name="Sarah Chen", mention_count=3)
    _seed_chunk(
        fresh_db, path="transcript://granola/r9c",
        mtime=time.time(),
        text="# Coffee with Sarah\n\n"
             "We talked about onboarding. "
             "I'll send Sarah the design doc by Friday.",
    )
    fake_promises = [
        {
            "text": "send Sarah the design doc",
            "recipient": "Sarah Chen",
            "due_hint": "Friday",
        },
    ]
    with patch.object(
        tasks, "extract_promises_from_text",
        return_value=fake_promises,
    ):
        n = tasks.materialize_promises_from_transcripts(fresh_db, tmp_cfg)
    assert n == 1
    rows = tasks.list_open(fresh_db)
    assert len(rows) == 1
    assert rows[0].text == "send Sarah the design doc"
    assert rows[0].recipient_person_id == pid
    assert rows[0].due_hint == "Friday"
    # State table should record the run.
    runs = fresh_db.execute(
        "SELECT n_promises FROM task_promise_runs",
    ).fetchall()
    assert runs[0]["n_promises"] == 1


def test_materialize_promises_idempotent(fresh_db, tmp_cfg):
    """Re-running over the same transcript shouldn't double-extract."""
    from secondbrain import tasks
    _seed_chunk(
        fresh_db, path="transcript://granola/dup",
        mtime=time.time(),
        text="One transcript with one promise inside it.",
    )
    fake = [{"text": "send the deck", "recipient": "", "due_hint": ""}]
    with patch.object(
        tasks, "extract_promises_from_text", return_value=fake,
    ) as mock_call:
        n1 = tasks.materialize_promises_from_transcripts(fresh_db, tmp_cfg)
        n2 = tasks.materialize_promises_from_transcripts(fresh_db, tmp_cfg)
    assert n1 == 1
    # Second run: the file is in task_promise_runs so no re-call.
    assert n2 == 0
    assert mock_call.call_count == 1


def test_materialize_promises_unmatched_recipient_persists_anyway(
    fresh_db, tmp_cfg,
):
    """Even when the recipient name doesn't match anyone in people,
    the task still persists (just with recipient_person_id=NULL)."""
    from secondbrain import tasks
    _seed_chunk(
        fresh_db, path="transcript://granola/unknown",
        mtime=time.time(),
        text="A transcript referencing someone we don't know yet.",
    )
    fake = [
        {
            "text": "follow up with Mystery Person",
            "recipient": "Mystery Person",
            "due_hint": "",
        },
    ]
    with patch.object(
        tasks, "extract_promises_from_text", return_value=fake,
    ):
        n = tasks.materialize_promises_from_transcripts(
            fresh_db, tmp_cfg,
        )
    assert n == 1
    rows = tasks.list_open(fresh_db)
    assert rows[0].recipient_person_id is None


def test_daemon_registers_task_promises_job(fresh_db, tmp_cfg):
    from secondbrain.daemon import _build_daemon_scheduler
    sched = _build_daemon_scheduler(
        tmp_cfg, fresh_db, embedder=None, reranker=None,
    )
    assert "task_promises" in set(sched.names())


# ============================ format_task_line shows due_hint ========

def test_format_task_line_shows_due_hint(fresh_db):
    from secondbrain.tasks import Task, format_task_line
    t = Task(
        id=42, text="email Sarah", source_path="t://x",
        source_title="Coffee", status="open",
        created_at=time.time(), completed_at=None,
        due_at=None, recipient_person_id=1, due_hint="Friday",
    )
    line = format_task_line(t)
    assert "#42" in line
    assert "email Sarah" in line
    assert "Friday" in line
