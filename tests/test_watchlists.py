"""Watchlists schema + scheduler logic + web-search citation routing."""

from __future__ import annotations

import json
import time

from secondbrain.db import (
    watchlist_create,
    watchlist_delete,
    watchlist_due,
    watchlist_get,
    watchlist_latest_run,
    watchlist_list,
    watchlist_run_record_finish,
    watchlist_run_record_start,
    watchlist_runs,
    watchlist_set_enabled,
)

# ---------------------------- CRUD ------------------------------------

def test_watchlist_lifecycle(fresh_db):
    wid = watchlist_create(fresh_db, "pm-internships", "what came out today", schedule_minutes=60)
    rows = watchlist_list(fresh_db)
    assert len(rows) == 1
    assert rows[0]["name"] == "pm-internships"
    assert rows[0]["schedule_minutes"] == 60
    assert rows[0]["enabled"] == 1
    assert rows[0]["last_run_at"] is None

    watchlist_set_enabled(fresh_db, wid, False)
    assert watchlist_get(fresh_db, wid)["enabled"] == 0
    watchlist_set_enabled(fresh_db, wid, True)
    assert watchlist_get(fresh_db, wid)["enabled"] == 1

    watchlist_delete(fresh_db, wid)
    assert watchlist_get(fresh_db, wid) is None
    assert watchlist_list(fresh_db) == []


def test_watchlist_create_clamps_minimum_schedule(fresh_db):
    """Schedules under 5 minutes get clamped to 5."""
    wid = watchlist_create(fresh_db, "noisy", "q", schedule_minutes=1)
    assert watchlist_get(fresh_db, wid)["schedule_minutes"] == 5


# --------------------------- Scheduling -------------------------------

def test_watchlist_due_includes_never_run(fresh_db):
    wid = watchlist_create(fresh_db, "fresh", "q", schedule_minutes=60)
    due = watchlist_due(fresh_db)
    assert any(r["id"] == wid for r in due), "fresh watchlist should be due"


def test_watchlist_due_excludes_recently_run(fresh_db):
    wid = watchlist_create(fresh_db, "recent", "q", schedule_minutes=1440)
    # Manually mark it as just run.
    fresh_db.execute(
        "UPDATE watchlists SET last_run_at = ? WHERE id = ?",
        (time.time(), wid),
    )
    fresh_db.commit()
    due = watchlist_due(fresh_db)
    assert not any(r["id"] == wid for r in due)


def test_watchlist_due_excludes_disabled(fresh_db):
    wid = watchlist_create(fresh_db, "off", "q", schedule_minutes=60)
    watchlist_set_enabled(fresh_db, wid, False)
    assert not any(r["id"] == wid for r in watchlist_due(fresh_db))


def test_watchlist_due_includes_overdue(fresh_db):
    """A watchlist last run > schedule_minutes ago should be due again."""
    wid = watchlist_create(fresh_db, "overdue", "q", schedule_minutes=60)
    long_ago = time.time() - 2 * 3600  # 2 hours ago
    fresh_db.execute(
        "UPDATE watchlists SET last_run_at = ? WHERE id = ?",
        (long_ago, wid),
    )
    fresh_db.commit()
    assert any(r["id"] == wid for r in watchlist_due(fresh_db))


# ----------------------------- Runs -----------------------------------

def test_run_record_lifecycle(fresh_db):
    wid = watchlist_create(fresh_db, "test", "q")
    rid = watchlist_run_record_start(fresh_db, wid)
    runs = watchlist_runs(fresh_db, wid)
    assert len(runs) == 1
    assert runs[0]["finished_at"] is None
    assert runs[0]["started_at"] > 0

    watchlist_run_record_finish(
        fresh_db, rid,
        answer="found 3 new postings",
        citations_json='[{"url": "https://example.com"}]',
        cents_spent=2.5,
    )
    runs = watchlist_runs(fresh_db, wid)
    assert runs[0]["finished_at"] is not None
    assert runs[0]["answer"] == "found 3 new postings"
    assert runs[0]["cents_spent"] == 2.5

    # last_run_at on parent should now equal the run's started_at
    parent = watchlist_get(fresh_db, wid)
    assert parent["last_run_at"] == runs[0]["started_at"]


def test_run_record_failure_path(fresh_db):
    wid = watchlist_create(fresh_db, "fail", "q")
    rid = watchlist_run_record_start(fresh_db, wid)
    watchlist_run_record_finish(fresh_db, rid, error="rate limited")
    latest = watchlist_latest_run(fresh_db, wid)
    assert latest["error"] == "rate limited"
    assert latest["answer"] is None


def test_run_cascade_drops_history(fresh_db):
    wid = watchlist_create(fresh_db, "cascade", "q")
    rid = watchlist_run_record_start(fresh_db, wid)
    watchlist_run_record_finish(fresh_db, rid, answer="x")
    assert len(watchlist_runs(fresh_db, wid)) == 1
    watchlist_delete(fresh_db, wid)
    # cascade should remove the run row
    rows = fresh_db.execute(
        "SELECT * FROM watchlist_runs WHERE watchlist_id = ?", (wid,),
    ).fetchall()
    assert rows == []


# ---------------- Watchlist runner _build_prompt shape ----------------

def test_build_prompt_first_run_uses_ever():
    from secondbrain.watchlist import _build_prompt
    p = _build_prompt("PM internships", None)
    assert "Watchlist query: PM internships" in p
    assert "since ever" in p


def test_build_prompt_returning_run_includes_iso_timestamp():
    from secondbrain.watchlist import _build_prompt
    p = _build_prompt("PM internships", 1735689600.0)  # 2025-01-01 00:00 UTC
    assert "since 2025-01-01" in p


# ----------- web-search citation extraction (Phase 25) ---------------

def test_extract_web_search_rows_object_shape():
    from secondbrain.chat import _extract_web_search_rows

    class FakeItem:
        def __init__(self, url, title):
            self.url = url
            self.title = title

    class FakeBlock:
        type = "web_search_tool_result"
        content = [FakeItem("https://a", "A"), FakeItem("https://b", "B")]

    rows = _extract_web_search_rows(FakeBlock())
    assert rows == [
        {"url": "https://a", "title": "A"},
        {"url": "https://b", "title": "B"},
    ]


def test_extract_web_search_rows_dict_fallback():
    """The SDK can return dicts in some surfaces; we handle both."""
    from secondbrain.chat import _extract_web_search_rows

    class FakeBlock:
        content = [{"url": "https://x", "title": "X"}, {"title": "no url"}]

    rows = _extract_web_search_rows(FakeBlock())
    assert rows == [{"url": "https://x", "title": "X"}]  # row without url skipped


def test_extract_web_search_rows_string_content_returns_empty():
    """Error responses surface as a string content; we shouldn't crash."""
    from secondbrain.chat import _extract_web_search_rows

    class FakeBlock:
        content = "rate limit exceeded"

    assert _extract_web_search_rows(FakeBlock()) == []


def test_citation_dataclass_kind_default():
    from secondbrain.chat import Citation
    c = Citation(chunk_id=1, file_path="/x", chunk_index=0, text="t", score=0.5)
    assert c.kind == "brain"
    assert c.url == ""
    assert c.page_title == ""


def test_citation_can_be_constructed_as_web():
    from secondbrain.chat import Citation
    c = Citation(
        chunk_id=-99, file_path="https://x", chunk_index=0, text="snippet",
        score=1.0, kind="web", url="https://x", page_title="X title",
    )
    assert c.kind == "web"
    assert c.page_title == "X title"


# Check that the watchlist runner correctly converts a Citation back into
# the JSON shape stored in watchlist_runs.citations_json - we verify this
# end-to-end in test_db.py via the round-trip; here we just spot-check the
# format doesn't drop the kind/url fields.
def test_run_record_citations_json_round_trip(fresh_db):
    wid = watchlist_create(fresh_db, "rt", "q")
    rid = watchlist_run_record_start(fresh_db, wid)
    payload = [
        {"kind": "web", "file_path": "https://x", "url": "https://x",
         "page_title": "X", "chunk_index": 0, "score": 1.0, "text": "snippet"},
        {"kind": "brain", "file_path": "/notes/a.md", "url": "",
         "page_title": "", "chunk_index": 3, "score": 0.92, "text": "from brain"},
    ]
    watchlist_run_record_finish(
        fresh_db, rid, answer="ok", citations_json=json.dumps(payload),
    )
    latest = watchlist_latest_run(fresh_db, wid)
    parsed = json.loads(latest["citations_json"])
    assert parsed[0]["kind"] == "web"
    assert parsed[1]["kind"] == "brain"
    assert parsed[0]["url"] == "https://x"


# Sanity: the latest_summary helper hands back a parsed dict.
def test_latest_summary_parses_citations(fresh_db):
    from secondbrain.watchlist import latest_summary

    wid = watchlist_create(fresh_db, "sum", "q")
    rid = watchlist_run_record_start(fresh_db, wid)
    watchlist_run_record_finish(
        fresh_db, rid, answer="hi",
        citations_json=json.dumps([{"kind": "web", "url": "https://x"}]),
        cents_spent=1.5,
    )
    s = latest_summary(fresh_db, wid)
    assert s is not None
    assert s["answer"] == "hi"
    assert s["citations"] == [{"kind": "web", "url": "https://x"}]
    assert s["cents_spent"] == 1.5
    assert s["error"] is None


def test_latest_summary_returns_none_for_no_runs(fresh_db):
    from secondbrain.watchlist import latest_summary

    wid = watchlist_create(fresh_db, "empty", "q")
    assert latest_summary(fresh_db, wid) is None
