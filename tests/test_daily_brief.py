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
    DailyBrief,
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
