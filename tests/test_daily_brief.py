"""Phase 44: daily brief aggregator tests.

The aggregator pulls from five sources (calendar / Canvas / transcripts /
reading queue / watchlist runs). Tests stub the calendar fetch (network
+ OAuth scaffold) but exercise the SQL paths against a real ``fresh_db``
because the queries are the part most likely to silently regress.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from secondbrain.daily_brief import (
    ActionItem,
    Assignment,
    CompletedTask,
    DailyBrief,
    HealthMetricLine,
    HealthSnapshot,
    QueueItem,
    WatchlistHighlight,
    _open_action_items,
    _queue_top,
    _watchlist_highlights,
    assemble_brief,
    format_markdown,
)


# Lightweight CalendarEvent stand-in matching the shape the renderer uses.
# Avoids importing the real one (which pulls in event_briefing → requests).
@dataclass
class FakeEvent:
    starts_at: float
    title: str
    location: str = ""
    attendees: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.attendees is None:
            self.attendees = []


# ---- helpers to seed fixtures -----------------------------------------

def _insert_file(conn, path: str, *, mtime: float | None = None,
                 indexed_at: float | None = None, kind: str = "url") -> int:
    """Insert a row in ``files`` and return its id. Tiny helper because
    every section's test wants to seed a couple of files."""
    n = time.time()
    cur = conn.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at, content_hash) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (path, mtime if mtime is not None else n, 1,
         kind, indexed_at if indexed_at is not None else n, None),
    )
    conn.commit()
    return cur.lastrowid


def _insert_chunk(conn, file_id: int, chunk_index: int, text: str) -> int:
    cur = conn.execute(
        "INSERT INTO chunks(file_id, chunk_index, text) VALUES (?, ?, ?)",
        (file_id, chunk_index, text),
    )
    conn.commit()
    return cur.lastrowid


# =================== assignments due soon ============================

def test_assignments_due_soon_returns_only_within_window(fresh_db):
    """Canvas connector stores due_at as mtime. Anything outside [now,
    now+72h] should be excluded from the brief."""
    from secondbrain.daily_brief import _assignments_due_soon

    now = time.time()
    # Within window — due tomorrow.
    fid_close = _insert_file(
        fresh_db, "canvas://assignment/1/100", mtime=now + 24 * 3600,
    )
    _insert_chunk(
        fresh_db, fid_close, 0,
        "# Problem set 3\n\nCourse: BME 410 (BME 410)\n"
        "Due: tomorrow\nLink: https://canvas.example/courses/1/a/100\n",
    )
    # Way out of window — due in 30 days.
    fid_far = _insert_file(
        fresh_db, "canvas://assignment/1/200", mtime=now + 30 * 86400,
    )
    _insert_chunk(fresh_db, fid_far, 0, "# Final project\nLink: x\n")
    # In the past — already due. Should also drop.
    fid_past = _insert_file(
        fresh_db, "canvas://assignment/1/300", mtime=now - 86400,
    )
    _insert_chunk(fresh_db, fid_past, 0, "# Old hw\nLink: y\n")

    out = _assignments_due_soon(fresh_db)
    paths = [a.path for a in out]
    assert "canvas://assignment/1/100" in paths
    assert "canvas://assignment/1/200" not in paths
    assert "canvas://assignment/1/300" not in paths


def test_assignments_pull_title_and_url_from_chunk(fresh_db):
    from secondbrain.daily_brief import _assignments_due_soon

    fid = _insert_file(
        fresh_db, "canvas://assignment/2/50", mtime=time.time() + 12 * 3600,
    )
    _insert_chunk(
        fresh_db, fid, 0,
        "# Lab 4 writeup\n\nCourse: CS 374 (CS 374)\n"
        "Due: Friday\nLink: https://canvas.example/courses/2/assignments/50\n"
        "rest of body...",
    )
    out = _assignments_due_soon(fresh_db)
    assert len(out) == 1
    a = out[0]
    assert a.title == "Lab 4 writeup"
    assert a.url == "https://canvas.example/courses/2/assignments/50"


def test_assignments_falls_back_to_path_when_no_chunk(fresh_db):
    """Chunkless rows shouldn't crash — show the path so the user can
    still find the row."""
    from secondbrain.daily_brief import _assignments_due_soon

    _insert_file(
        fresh_db, "canvas://assignment/3/9", mtime=time.time() + 3600,
    )
    out = _assignments_due_soon(fresh_db)
    assert len(out) == 1
    assert out[0].title == "canvas://assignment/3/9"
    assert out[0].url == ""


# ===================== open action items ==============================

def test_action_items_extracts_open_checkboxes(fresh_db):
    fid = _insert_file(
        fresh_db, "transcript://granola/abc",
        indexed_at=time.time() - 3600,
    )
    _insert_chunk(
        fresh_db, fid, 0,
        "# [meeting] Sprint planning\n\n"
        "## Action items\n"
        "- [ ] Email Sarah about the API contract\n"
        "- [ ] Draft the migration plan\n"
        "- [x] Already-done thing\n"
        "- [ ] Review the resume scorer PR\n",
    )
    out = _open_action_items(fresh_db)
    texts = [a.text for a in out]
    assert "Email Sarah about the API contract" in texts
    assert "Draft the migration plan" in texts
    assert "Review the resume scorer PR" in texts
    # Closed checkbox is filtered.
    assert "Already-done thing" not in texts


def test_action_items_skips_old_transcripts(fresh_db):
    """The window is 14 days. Older transcripts shouldn't pollute the
    brief with stale unticked promises."""
    fid_old = _insert_file(
        fresh_db, "transcript://granola/old",
        indexed_at=time.time() - 30 * 86400,
    )
    _insert_chunk(
        fresh_db, fid_old, 0,
        "# Meeting\n- [ ] An ancient promise nobody kept\n",
    )
    out = _open_action_items(fresh_db)
    texts = [a.text for a in out]
    assert "An ancient promise nobody kept" not in texts


def test_action_items_dedupes_repeats_within_one_doc(fresh_db):
    """Same item text in two chunks of the same doc — only surface once.
    Otherwise a meeting that quotes the action items list twice (once in
    'transcript', once in 'summary') would appear twice in the brief."""
    fid = _insert_file(
        fresh_db, "transcript://granola/dupe",
        indexed_at=time.time() - 60,
    )
    _insert_chunk(fresh_db, fid, 0, "- [ ] Talk to Alice")
    _insert_chunk(fresh_db, fid, 1, "- [ ] Talk to Alice")
    out = _open_action_items(fresh_db)
    assert sum(1 for a in out if a.text == "Talk to Alice") == 1


def test_action_items_caps_at_max(fresh_db):
    """Don't let one chatty meeting fill the entire section."""
    fid = _insert_file(
        fresh_db, "transcript://granola/big",
        indexed_at=time.time() - 60,
    )
    body = "\n".join(f"- [ ] Item number {i}" for i in range(30))
    _insert_chunk(fresh_db, fid, 0, body)
    out = _open_action_items(fresh_db)
    assert len(out) == 10  # _ACTION_ITEM_MAX


def test_action_items_ignores_non_transcript_paths(fresh_db):
    """We only scan transcript:// docs. A regular Markdown todo list
    that happens to live under a different path shouldn't get scraped —
    that's a separate use case (Phase 47 tasks module)."""
    fid = _insert_file(
        fresh_db, "C:\\notes\\todos.md", indexed_at=time.time() - 60,
    )
    _insert_chunk(fresh_db, fid, 0, "- [ ] Buy milk\n- [ ] Pay bills\n")
    out = _open_action_items(fresh_db)
    assert out == []


# ========================= reading queue ==============================

def test_queue_top_pulls_unread_with_summaries(fresh_db):
    from secondbrain.db import (
        reading_queue_enqueue,
        reading_queue_set_summary,
    )

    rid1 = reading_queue_enqueue(
        fresh_db, url="https://example.com/a", title="Article A",
        source="manual",
    )
    reading_queue_set_summary(fresh_db, rid1, summary="A 60-second precis.")
    reading_queue_enqueue(
        fresh_db, url="https://example.com/b", title="Article B",
        source="manual",
    )

    out = _queue_top(fresh_db)
    assert len(out) == 2
    by_url = {q.url: q for q in out}
    assert by_url["https://example.com/a"].summary == "A 60-second precis."
    assert by_url["https://example.com/b"].summary == ""


def test_queue_top_excludes_read_or_skipped(fresh_db):
    from secondbrain.db import (
        reading_queue_enqueue,
        reading_queue_mark_read,
        reading_queue_mark_skipped,
    )

    rid1 = reading_queue_enqueue(
        fresh_db, url="https://example.com/x", title="X", source="manual",
    )
    rid2 = reading_queue_enqueue(
        fresh_db, url="https://example.com/y", title="Y", source="manual",
    )
    reading_queue_enqueue(
        fresh_db, url="https://example.com/z", title="Z", source="manual",
    )
    reading_queue_mark_read(fresh_db, rid1)
    reading_queue_mark_skipped(fresh_db, rid2)

    out = _queue_top(fresh_db)
    assert [q.url for q in out] == ["https://example.com/z"]


# ===================== watchlist highlights ===========================

def test_watchlist_highlights_only_recent_with_new_items(fresh_db):
    """24h window, ``finished_at`` not null, ``new_count`` > 0."""
    fresh_db.execute(
        "INSERT INTO watchlists(name, query, schedule_minutes, created_at) "
        "VALUES (?, ?, ?, ?)",
        ("AI news", "what came out today", 60, time.time()),
    )
    fresh_db.commit()
    wl_id = fresh_db.execute(
        "SELECT id FROM watchlists",
    ).fetchone()["id"]

    now = time.time()
    # Recent + has new items → should appear.
    fresh_db.execute(
        "INSERT INTO watchlist_runs"
        "(watchlist_id, started_at, finished_at, new_paths_json, new_count) "
        "VALUES (?, ?, ?, ?, ?)",
        (wl_id, now - 1800, now - 1700,
         json.dumps(["https://x.com/a", "https://y.com/b"]), 2),
    )
    # Recent but no new items → should NOT appear.
    fresh_db.execute(
        "INSERT INTO watchlist_runs"
        "(watchlist_id, started_at, finished_at, new_paths_json, new_count) "
        "VALUES (?, ?, ?, ?, ?)",
        (wl_id, now - 600, now - 500, json.dumps([]), 0),
    )
    # Old run with new items → should NOT appear.
    fresh_db.execute(
        "INSERT INTO watchlist_runs"
        "(watchlist_id, started_at, finished_at, new_paths_json, new_count) "
        "VALUES (?, ?, ?, ?, ?)",
        (wl_id, now - 50 * 3600, now - 50 * 3600 + 60,
         json.dumps(["https://stale.com"]), 1),
    )
    fresh_db.commit()

    out = _watchlist_highlights(fresh_db)
    assert len(out) == 1
    assert out[0].name == "AI news"
    assert out[0].new_count == 2
    assert "https://x.com/a" in out[0].sample_paths


def test_watchlist_highlights_skips_failed_runs(fresh_db):
    """Don't surface runs that errored — the new_count is meaningless."""
    fresh_db.execute(
        "INSERT INTO watchlists(name, query, schedule_minutes, created_at) "
        "VALUES (?, ?, ?, ?)",
        ("News", "q", 60, time.time()),
    )
    fresh_db.commit()
    wl_id = fresh_db.execute(
        "SELECT id FROM watchlists",
    ).fetchone()["id"]
    now = time.time()
    fresh_db.execute(
        "INSERT INTO watchlist_runs"
        "(watchlist_id, started_at, finished_at, new_paths_json, "
        " new_count, error) VALUES (?, ?, ?, ?, ?, ?)",
        (wl_id, now - 600, now - 500, json.dumps(["x"]), 1,
         "rate limited"),
    )
    fresh_db.commit()
    assert _watchlist_highlights(fresh_db) == []


def test_watchlist_highlights_tolerates_malformed_json(fresh_db):
    """Bad new_paths_json shouldn't crash — show the count, no samples."""
    fresh_db.execute(
        "INSERT INTO watchlists(name, query, schedule_minutes, created_at) "
        "VALUES (?, ?, ?, ?)",
        ("X", "q", 60, time.time()),
    )
    fresh_db.commit()
    wl_id = fresh_db.execute(
        "SELECT id FROM watchlists",
    ).fetchone()["id"]
    now = time.time()
    fresh_db.execute(
        "INSERT INTO watchlist_runs"
        "(watchlist_id, started_at, finished_at, new_paths_json, new_count) "
        "VALUES (?, ?, ?, ?, ?)",
        (wl_id, now - 600, now - 500, "not-json{", 3),
    )
    fresh_db.commit()
    out = _watchlist_highlights(fresh_db)
    assert len(out) == 1
    assert out[0].new_count == 3
    assert out[0].sample_paths == []


# ============================ rendering ===============================

def test_render_skips_empty_sections():
    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[],
        assignments_due_soon=[],
        open_action_items=[],
        queue_top=[],
        watchlist_highlights=[],
    )
    md = format_markdown(brief, header_date="Friday, May 1")
    assert "Daily brief — Friday, May 1" in md
    # No section headers.
    assert "## " not in md
    # Quiet-day fallback.
    assert "Quiet day" in md


def test_render_includes_sections_in_order():
    """The render should emit calendar → assignments → action items →
    queue → watchlist, in that order. Front-loads the time-sensitive
    stuff."""
    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[FakeEvent(starts_at=time.time() + 3600,
                                title="1:1 with manager")],
        assignments_due_soon=[Assignment(
            title="[BME 410] Pset 3", due_at=time.time() + 6 * 3600,
            url="https://canvas.example/a/1", path="canvas://assignment/1/1",
        )],
        open_action_items=[ActionItem(
            text="Email Sarah", source_path="transcript://granola/x",
            source_title="Sprint planning",
        )],
        queue_top=[QueueItem(
            queue_id=1, url="https://example.com/post",
            title="A great post", summary="Worth reading.",
        )],
        watchlist_highlights=[WatchlistHighlight(
            name="AI news", new_count=4,
            sample_paths=["https://x.com/a"], finished_at=time.time(),
        )],
    )
    md = format_markdown(brief, header_date="today")
    # Order check: each section header appears, in order.
    pos_cal = md.find("## Today on the calendar")
    pos_asgn = md.find("## Class — due in the next 72h")
    pos_ai = md.find("## Open action items")
    pos_q = md.find("## Reading queue")
    pos_w = md.find("## Watchlists")
    assert -1 < pos_cal < pos_asgn < pos_ai < pos_q < pos_w


def test_render_event_with_attendees_truncates_long_lists():
    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[FakeEvent(
            starts_at=time.time() + 3600, title="All-hands",
            attendees=["a@x.co", "b@x.co", "c@x.co", "d@x.co", "e@x.co"],
        )],
        assignments_due_soon=[],
        open_action_items=[],
        queue_top=[],
        watchlist_highlights=[],
    )
    md = format_markdown(brief, header_date="today")
    assert "+2" in md  # 5 attendees, first 3 named, +2 hidden
    assert "a@x.co" in md
    assert "e@x.co" not in md


def test_render_action_item_uses_unticked_checkbox():
    """User should be able to copy a line straight into Obsidian and
    tick it off — so we render as `- [ ]` not as `-`."""
    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[],
        assignments_due_soon=[],
        open_action_items=[ActionItem(
            text="Reply to recruiter",
            source_path="transcript://granola/x",
            source_title="Career chat",
        )],
        queue_top=[],
        watchlist_highlights=[],
    )
    md = format_markdown(brief, header_date="today")
    assert "- [ ] Reply to recruiter" in md


def test_render_queue_summary_quoted_so_markdown_doesnt_break():
    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[],
        assignments_due_soon=[],
        open_action_items=[],
        queue_top=[QueueItem(
            queue_id=1, url="https://example.com",
            title="Long article",
            summary="Line 1\nLine 2\nLine 3",
        )],
        watchlist_highlights=[],
    )
    md = format_markdown(brief, header_date="today")
    assert "  > Line 1" in md
    assert "  > Line 2" in md
    assert "  > Line 3" in md


# ========================= integration ================================

def test_assemble_brief_returns_filled_object_with_no_calendar(
    fresh_db, tmp_cfg, monkeypatch,
):
    """Wire the whole assembly. Calendar source is stubbed to empty so
    we don't hit the network; everything else exercises real SQL."""
    import secondbrain.daily_brief as db_mod

    monkeypatch.setattr(db_mod, "_today_events", lambda cfg: [])

    # Seed an assignment.
    fid = _insert_file(
        fresh_db, "canvas://assignment/1/1", mtime=time.time() + 3600,
    )
    _insert_chunk(
        fresh_db, fid, 0,
        "# Pset\nCourse: x\nLink: https://c\n",
    )
    # Seed an action item.
    tfid = _insert_file(
        fresh_db, "transcript://granola/x", indexed_at=time.time() - 60,
    )
    _insert_chunk(fresh_db, tfid, 0, "- [ ] Send the deck")
    # Seed a queue item.
    from secondbrain.db import reading_queue_enqueue
    reading_queue_enqueue(
        fresh_db, url="https://r.com", title="R", source="manual",
    )

    brief = assemble_brief(tmp_cfg, fresh_db)
    assert brief.today_events == []
    assert len(brief.assignments_due_soon) == 1
    assert len(brief.open_action_items) == 1
    assert len(brief.queue_top) == 1
    md = format_markdown(brief, header_date="today")
    assert "Pset" in md
    assert "Send the deck" in md
    assert "R" in md


def test_today_events_swallows_calendar_failure(tmp_cfg, monkeypatch):
    """A calendar exception shouldn't take down the brief."""
    import secondbrain.daily_brief as db_mod
    import secondbrain.event_briefing as eb

    def boom(cfg, lookahead_seconds):
        raise RuntimeError("network exploded")

    monkeypatch.setattr(eb, "iter_upcoming_events", boom)
    out = db_mod._today_events(tmp_cfg)
    assert out == []


# ============== polish-pass extensions (Phase 44 v2) ==================

def test_action_items_carry_task_id_and_age(fresh_db):
    """Each ActionItem should expose task.id + age_days so the brief
    can render `tasks done <id>` actionable + 'this has been open
    forever' nudges."""
    from secondbrain import tasks as tasks_mod
    from secondbrain.daily_brief import _open_action_items

    fid = _insert_file(
        fresh_db, "transcript://granola/x", indexed_at=time.time() - 60,
    )
    _insert_chunk(
        fresh_db, fid, 0,
        "## Action items\n- Email Sarah\n",
    )
    out = _open_action_items(fresh_db)
    assert len(out) == 1
    item = out[0]
    assert item.task_id > 0
    # Sanity: the id should round-trip through the tasks module.
    t = tasks_mod.get(fresh_db, item.task_id)
    assert t.text == "Email Sarah"


def test_render_action_items_includes_task_id_for_done_command():
    """Rendered markdown must show the id so the user knows what to
    type into `tasks done`."""
    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[],
        assignments_due_soon=[],
        open_action_items=[ActionItem(
            text="Email Sarah",
            source_path="transcript://granola/x",
            source_title="Sprint planning",
            task_id=42,
            age_days=2,
        )],
        queue_top=[],
        watchlist_highlights=[],
    )
    md = format_markdown(brief, header_date="today")
    assert "#42" in md
    assert "(2d)" in md
    assert "tasks done" in md


def test_render_skips_age_for_brand_new_tasks():
    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[],
        assignments_due_soon=[],
        open_action_items=[ActionItem(
            text="Just added", source_path="manual",
            source_title="(typed)", task_id=1, age_days=0,
        )],
        queue_top=[],
        watchlist_highlights=[],
    )
    md = format_markdown(brief, header_date="today")
    # Don't render `(0d)` — that's noise on freshly-added tasks.
    assert "(0d)" not in md


def test_health_snapshot_assembles_when_oura_data_present(fresh_db):
    """The brief should include sleep/readiness/activity values from
    the health_metrics table (Phase 56) so the user sees this morning's
    physical context alongside their schedule."""
    from secondbrain import health
    from secondbrain.connectors.oura import DailySummary
    from secondbrain.daily_brief import _health_snapshot

    summaries = [
        DailySummary(
            date="2026-04-13", sleep_score=70, readiness_score=70,
            activity_score=80,
        ),
        DailySummary(
            date="2026-04-14", sleep_score=80, readiness_score=80,
            activity_score=85,
        ),
        DailySummary(
            date="2026-04-15", sleep_score=90, readiness_score=85,
            activity_score=95,
        ),
    ]
    health.ingest_summaries(fresh_db, summaries)
    snap = _health_snapshot(fresh_db)
    assert snap is not None
    metrics_by_name = {m.metric: m for m in snap.metrics}
    assert "sleep_score" in metrics_by_name
    sleep = metrics_by_name["sleep_score"]
    assert sleep.label == "Sleep"
    assert sleep.latest == 90.0
    assert sleep.latest_date == "2026-04-15"
    # Average of [70, 80, 90] is 80.0 → delta = (90-80)/80 * 100 = +12.5%
    assert sleep.average == 80.0
    assert abs(sleep.delta_pct - 12.5) < 0.01


def test_health_snapshot_returns_none_when_no_data(fresh_db):
    from secondbrain.daily_brief import _health_snapshot
    assert _health_snapshot(fresh_db) is None


def test_render_health_marks_significant_drops(fresh_db):
    """A 5%+ drop should get a ↓ arrow so the user clocks the dip
    without parsing percentages."""
    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[],
        assignments_due_soon=[],
        open_action_items=[],
        queue_top=[],
        watchlist_highlights=[],
        health=HealthSnapshot(metrics=[HealthMetricLine(
            metric="sleep_score", label="Sleep",
            latest=70, latest_date="2026-04-15",
            average=85, delta_pct=-17.6,
        )]),
    )
    md = format_markdown(brief, header_date="today")
    assert "## Health (Oura)" in md
    assert "↓" in md
    assert "Sleep" in md
    assert "70" in md


def test_render_health_marks_significant_jumps():
    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[], assignments_due_soon=[],
        open_action_items=[], queue_top=[],
        watchlist_highlights=[],
        health=HealthSnapshot(metrics=[HealthMetricLine(
            metric="readiness_score", label="Readiness",
            latest=92, latest_date="2026-04-15",
            average=80, delta_pct=15.0,
        )]),
    )
    md = format_markdown(brief, header_date="today")
    assert "↑" in md


def test_render_health_omits_arrow_when_in_band():
    """Within ±5% of the average — no arrow, just the number."""
    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[], assignments_due_soon=[],
        open_action_items=[], queue_top=[],
        watchlist_highlights=[],
        health=HealthSnapshot(metrics=[HealthMetricLine(
            metric="sleep_score", label="Sleep",
            latest=82, latest_date="2026-04-15",
            average=80, delta_pct=2.5,
        )]),
    )
    md = format_markdown(brief, header_date="today")
    assert "↓" not in md
    assert "↑" not in md


def test_yesterday_done_pulls_recent_completions(fresh_db):
    """Tasks completed in the last 36h appear in 'Recently done'."""
    from secondbrain import tasks as tasks_mod
    from secondbrain.daily_brief import _yesterday_done

    tid = tasks_mod.add_manual(fresh_db, "Send the report")
    tasks_mod.mark_done(fresh_db, tid)
    out = _yesterday_done(fresh_db)
    assert len(out) == 1
    assert out[0].text == "Send the report"


def test_yesterday_done_excludes_old_completions(fresh_db):
    """Tasks done 5 days ago shouldn't drop into today's brief."""
    from secondbrain.daily_brief import _yesterday_done

    fresh_db.execute(
        "INSERT INTO tasks(text, text_lower, source_path, source_title, "
        " status, created_at, completed_at) "
        "VALUES (?, ?, 'manual', '(typed)', 'done', ?, ?)",
        ("ancient win", "ancient win",
         time.time() - 6 * 86400, time.time() - 5 * 86400),
    )
    fresh_db.commit()
    assert _yesterday_done(fresh_db) == []


def test_revisit_suggestions_only_fire_on_quiet_days(fresh_db):
    """If there's actionable content, the brief should NOT pull
    revisit suggestions — they'd push the time-sensitive stuff
    further down the page."""
    from secondbrain.daily_brief import _has_actionable_content

    brief_with_events = DailyBrief(
        generated_at=time.time(),
        today_events=[FakeEvent(starts_at=time.time(), title="X")],
        assignments_due_soon=[], open_action_items=[],
        queue_top=[], watchlist_highlights=[],
    )
    assert _has_actionable_content(brief_with_events) is True

    quiet_brief = DailyBrief(
        generated_at=time.time(),
        today_events=[], assignments_due_soon=[],
        open_action_items=[], queue_top=[],
        watchlist_highlights=[],
    )
    assert _has_actionable_content(quiet_brief) is False


def test_revisit_suggestions_skip_when_brain_too_small(fresh_db):
    """A brain with fewer than the threshold files shouldn't surface
    revisit suggestions — there's not enough archive yet."""
    from secondbrain.daily_brief import _revisit_suggestions
    # Empty brain.
    assert _revisit_suggestions(fresh_db) == []


def test_render_quiet_day_shows_banner_when_no_actionable(fresh_db):
    """When the only sections are passive (health/done/revisit), the
    brief should still emit the 'quiet day' note + show the passive
    sections so the user has something to look at."""
    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[], assignments_due_soon=[],
        open_action_items=[], queue_top=[],
        watchlist_highlights=[],
        yesterday_done=[CompletedTask(text="x", completed_at=time.time())],
    )
    md = format_markdown(brief, header_date="today")
    assert "Quiet day" in md
    assert "Recently done" in md


def test_render_full_day_does_not_show_quiet_banner():
    """A brief with calendar events shouldn't carry the quiet-day
    banner — that'd be misleading."""
    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[FakeEvent(
            starts_at=time.time() + 3600, title="Standup",
        )],
        assignments_due_soon=[], open_action_items=[],
        queue_top=[], watchlist_highlights=[],
    )
    md = format_markdown(brief, header_date="today")
    assert "Quiet day" not in md


# ===================== scheduler / SMTP send ==========================

def test_last_brief_sent_at_returns_none_initially(fresh_db):
    from secondbrain.daily_brief import last_brief_sent_at
    assert last_brief_sent_at(fresh_db) is None


def test_run_brief_if_due_skips_when_disabled(fresh_db, tmp_cfg):
    """Disabled in config → daemon must not even attempt the send."""
    from secondbrain.daily_brief import run_brief_if_due
    tmp_cfg.daily_brief_enabled = False
    tmp_cfg.digest_to = "test@example.com"
    assert run_brief_if_due(tmp_cfg, fresh_db) is False


def test_run_brief_if_due_skips_without_recipient(fresh_db, tmp_cfg):
    """Enabled but no recipient → still skip."""
    from secondbrain.daily_brief import run_brief_if_due
    tmp_cfg.daily_brief_enabled = True
    tmp_cfg.digest_to = ""
    assert run_brief_if_due(tmp_cfg, fresh_db) is False


def test_run_brief_if_due_skips_when_too_recent(fresh_db, tmp_cfg, monkeypatch):
    """If we sent within the cooldown, the daemon must NOT re-send."""
    import secondbrain.daily_brief as db_mod

    # Pretend a brief was sent 1h ago.
    db_mod._ensure_brief_runs_table(fresh_db)
    fresh_db.execute(
        "INSERT INTO daily_brief_runs(sent_at, success, error, recipients) "
        "VALUES (?, ?, ?, ?)",
        (time.time() - 3600, 1, None, "test@example.com"),
    )
    fresh_db.commit()

    tmp_cfg.daily_brief_enabled = True
    tmp_cfg.digest_to = "test@example.com"
    tmp_cfg.daily_brief_send_time = "00:00"  # always past
    # Should not trigger send (cooldown).
    sent = []
    monkeypatch.setattr(
        db_mod, "send_brief",
        lambda *a, **kw: (sent.append(True), (True, "ok"))[1],
    )
    assert db_mod.run_brief_if_due(tmp_cfg, fresh_db) is False
    assert sent == []


def test_run_brief_if_due_handles_malformed_send_time(fresh_db, tmp_cfg):
    """Garbage in `daily_brief_send_time` should log + skip, not crash."""
    from secondbrain.daily_brief import run_brief_if_due
    tmp_cfg.daily_brief_enabled = True
    tmp_cfg.digest_to = "test@example.com"
    tmp_cfg.daily_brief_send_time = "not-a-time"
    assert run_brief_if_due(tmp_cfg, fresh_db) is False


def test_send_brief_returns_failure_without_password(fresh_db, tmp_cfg, monkeypatch):
    from secondbrain.daily_brief import send_brief
    monkeypatch.delenv("SECONDBRAIN_SMTP_PASSWORD", raising=False)
    tmp_cfg.daily_brief_enabled = True
    tmp_cfg.digest_to = "test@example.com"
    success, msg = send_brief(tmp_cfg, fresh_db)
    assert success is False
    assert "SMTP_PASSWORD" in msg


def test_send_brief_records_failure_in_runs_table(fresh_db, tmp_cfg, monkeypatch):
    """Failed sends must persist to daily_brief_runs so the dashboard
    / status command can show 'last failure' details."""
    from secondbrain.daily_brief import _ensure_brief_runs_table, send_brief
    monkeypatch.delenv("SECONDBRAIN_SMTP_PASSWORD", raising=False)
    tmp_cfg.daily_brief_enabled = True
    tmp_cfg.digest_to = "test@example.com"
    send_brief(tmp_cfg, fresh_db)
    # Even though we returned early on missing password, that's a
    # config issue not a send attempt — the table SHOULDN'T have a
    # failure row for missing-password (we never tried SMTP).
    _ensure_brief_runs_table(fresh_db)
    n = fresh_db.execute(
        "SELECT COUNT(*) AS n FROM daily_brief_runs",
    ).fetchone()["n"]
    assert n == 0  # didn't even try


def test_minimal_md_to_html_renders_headings_and_bullets():
    """The HTML alternative for the email needs to handle H1-3,
    bullets, blockquotes — the brief uses all three."""
    from secondbrain.daily_brief import _minimal_md_to_html
    md = (
        "# Daily brief\n\n"
        "## Section\n\n"
        "- bullet one\n"
        "- bullet two\n\n"
        "> a quote\n"
    )
    html = _minimal_md_to_html(md)
    assert "<h1>Daily brief</h1>" in html
    assert "<h2>Section</h2>" in html
    assert "<li>bullet one</li>" in html
    assert "<li>bullet two</li>" in html
    assert "<blockquote" in html and "a quote" in html


def test_minimal_md_to_html_escapes_special_characters():
    """User content can include `<` / `>` / `&` — those must escape
    so they don't break the email's HTML rendering."""
    from secondbrain.daily_brief import _minimal_md_to_html
    html = _minimal_md_to_html("- a < b > c & d")
    assert "&lt;" in html
    assert "&gt;" in html
    assert "&amp;" in html
    # Raw special chars should NOT have escaped through.
    assert "a < b" not in html


# ============== Polish v3: integration with new phases ===============

def test_brief_renders_habits_section(fresh_db):
    """Phase 79 hookup — habits with current streak appear in the brief."""
    from datetime import date, timedelta

    from secondbrain import personal
    from secondbrain.daily_brief import _habits_section

    hid = personal.add_habit(fresh_db, "Workout", cadence="daily")
    today = date.today()
    for i in range(3):
        personal.checkin(
            fresh_db, hid, when=today - timedelta(days=i),
        )
    out = _habits_section(fresh_db)
    assert len(out) == 1
    assert out[0].name == "Workout"
    assert out[0].streak_days == 3


def test_brief_renders_goals_section(fresh_db):
    from secondbrain import personal
    from secondbrain.daily_brief import _goals_section

    personal.add_goal(
        fresh_db, "Apply to jobs", target_per_week=5,
    )
    out = _goals_section(fresh_db)
    assert len(out) == 1
    assert out[0].name == "Apply to jobs"
    assert out[0].target_per_week == 5
    assert out[0].on_track is False  # 0/5


def test_brief_renders_email_section(fresh_db):
    """Phase 82 hookup — triage counts surface as a brief section."""
    from secondbrain import email_assist
    from secondbrain.daily_brief import _email_section

    # Seed classifications directly (bypassing classify_one which
    # needs chunks). The brief only reads from email_classifications,
    # so this is the unit under test.
    email_assist._ensure_schema(fresh_db)
    for i, label in enumerate(["urgent", "urgent", "newsletter"]):
        cur = fresh_db.execute(
            "INSERT INTO files(path, mtime, size, kind, indexed_at) "
            "VALUES (?, ?, 1, 'url', ?)",
            (f"imap://msgid/{i}", time.time(), time.time()),
        )
        fid = cur.lastrowid
        fresh_db.execute(
            "INSERT INTO email_classifications"
            "(file_id, label, classified_at) VALUES (?, ?, ?)",
            (fid, label, time.time()),
        )
    fresh_db.commit()
    out = _email_section(fresh_db)
    assert out is not None
    assert out.urgent == 2
    assert out.other == 1


def test_brief_renders_knowledge_gaps(fresh_db):
    """Phase 68 hookup — open gaps surface as study targets."""
    from secondbrain import study
    from secondbrain.daily_brief import _gaps_section

    study.log_gap(
        fresh_db, "what is X?", n_results=0, top_score=None,
    )
    out = _gaps_section(fresh_db)
    assert len(out) == 1
    assert "what is X" in out[0].question


def test_brief_pending_drafts_count(fresh_db):
    """Phase 83 hookup — pending drafts count surfaces."""
    from secondbrain.daily_brief import _pending_drafts_count

    # No drafts yet.
    assert _pending_drafts_count(fresh_db) == 0
    # Seed a classified email + a draft.
    cur = fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at) "
        "VALUES (?, ?, 1, 'url', ?)",
        ("imap://msgid/x", time.time(), time.time()),
    )
    fid = cur.lastrowid
    fresh_db.execute(
        "INSERT INTO email_drafts(file_id, draft_text, generated_at) "
        "VALUES (?, ?, ?)",
        (fid, "draft text", time.time()),
    )
    fresh_db.commit()
    assert _pending_drafts_count(fresh_db) == 1


def test_render_includes_insights_at_top(fresh_db):
    """Insights should render BEFORE calendar — user should clock
    them first."""
    from secondbrain.daily_brief import (
        DailyBrief,
        InsightLine,
        format_markdown,
    )

    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[FakeEvent(starts_at=time.time(), title="Standup")],
        assignments_due_soon=[], open_action_items=[],
        queue_top=[], watchlist_highlights=[],
        insights=[InsightLine(
            headline="You've referenced 'voyage' 5x this week",
            detail="2× jump vs prior 3 weeks. Synthesise?",
        )],
    )
    md = format_markdown(brief, header_date="today")
    pos_insights = md.find("## Worth noticing")
    pos_calendar = md.find("## Today on the calendar")
    assert -1 < pos_insights < pos_calendar


def test_render_habits_streaks_get_fire_emoji():
    """30+ day streaks get 🔥. 100+ get 🏔. <7 stay plain."""
    from secondbrain.daily_brief import (
        DailyBrief,
        HabitLine,
        format_markdown,
    )

    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[FakeEvent(starts_at=time.time(), title="X")],
        assignments_due_soon=[], open_action_items=[],
        queue_top=[], watchlist_highlights=[],
        habits=[
            HabitLine(name="Plain", streak_days=3,
                      expected_30d=30, actual_30d=20),
            HabitLine(name="Sparkle", streak_days=10,
                      expected_30d=30, actual_30d=25),
            HabitLine(name="Fire", streak_days=45,
                      expected_30d=30, actual_30d=29),
            HabitLine(name="Mountain", streak_days=200,
                      expected_30d=30, actual_30d=30),
        ],
    )
    md = format_markdown(brief, header_date="today")
    # No marker on plain.
    assert "Plain** — 3d streak" in md
    assert "✨" in md
    assert "🔥" in md
    assert "🏔" in md


def test_render_goals_marks_off_track():
    from secondbrain.daily_brief import (
        DailyBrief,
        GoalLine,
        format_markdown,
    )

    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[FakeEvent(starts_at=time.time(), title="X")],
        assignments_due_soon=[], open_action_items=[],
        queue_top=[], watchlist_highlights=[],
        goals=[
            GoalLine(
                name="Apply to jobs", target_per_week=5,
                progress_this_week=2, on_track=False,
            ),
        ],
    )
    md = format_markdown(brief, header_date="today")
    assert "Apply to jobs" in md
    assert "2/5" in md
    # Off-track marker is `[·]`, on-track is `[✓]`.
    assert "[·]" in md


def test_pending_drafts_in_email_section():
    """When the email section has pending drafts, render the count
    + how to review them."""
    from secondbrain.daily_brief import (
        DailyBrief,
        EmailTriageLine,
        format_markdown,
    )

    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[FakeEvent(starts_at=time.time(), title="X")],
        assignments_due_soon=[], open_action_items=[],
        queue_top=[], watchlist_highlights=[],
        email=EmailTriageLine(urgent=2, response=3, other=10),
        pending_email_drafts=4,
    )
    md = format_markdown(brief, header_date="today")
    assert "## Email" in md
    assert "2 urgent" in md
    assert "4 draft" in md
    assert "secondbrain drafts" in md  # CLI hint


def test_actionable_content_includes_insights_email_drafts_gaps():
    """The 'quiet day' classifier needs to weight the new signals."""
    from secondbrain.daily_brief import (
        DailyBrief,
        EmailTriageLine,
        GapLine,
        InsightLine,
        _has_actionable_content,
    )

    base = DailyBrief(
        generated_at=time.time(),
        today_events=[], assignments_due_soon=[],
        open_action_items=[], queue_top=[],
        watchlist_highlights=[],
    )
    assert _has_actionable_content(base) is False

    with_insight = DailyBrief(
        generated_at=time.time(),
        today_events=[], assignments_due_soon=[],
        open_action_items=[], queue_top=[],
        watchlist_highlights=[],
        insights=[InsightLine(headline="x", detail="y")],
    )
    assert _has_actionable_content(with_insight) is True

    with_drafts = DailyBrief(
        generated_at=time.time(),
        today_events=[], assignments_due_soon=[],
        open_action_items=[], queue_top=[],
        watchlist_highlights=[],
        pending_email_drafts=2,
    )
    assert _has_actionable_content(with_drafts) is True

    with_urgent = DailyBrief(
        generated_at=time.time(),
        today_events=[], assignments_due_soon=[],
        open_action_items=[], queue_top=[],
        watchlist_highlights=[],
        email=EmailTriageLine(urgent=3),
    )
    assert _has_actionable_content(with_urgent) is True

    with_gaps = DailyBrief(
        generated_at=time.time(),
        today_events=[], assignments_due_soon=[],
        open_action_items=[], queue_top=[],
        watchlist_highlights=[],
        knowledge_gaps=[GapLine(gap_id=1, question="q?")],
    )
    assert _has_actionable_content(with_gaps) is True
