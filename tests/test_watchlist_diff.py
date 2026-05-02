"""Phase 30+31: watchlist diff (new since last run) + notifications."""

from __future__ import annotations

import json

from secondbrain.db import (
    watchlist_create,
    watchlist_get,
    watchlist_latest_run,
    watchlist_previous_run,
    watchlist_run_record_finish,
    watchlist_run_record_start,
)
from secondbrain.watchlist import _compute_new_paths, latest_summary


def _finish(conn, wid: int, paths: list[str]):
    """Helper: start + finish a successful run with the given citation paths."""
    rid = watchlist_run_record_start(conn, wid)
    cites = [{"file_path": p} for p in paths]
    new_paths, new_count = _compute_new_paths(conn, wid, rid, cites)
    watchlist_run_record_finish(
        conn, rid,
        answer="ok",
        citations_json=json.dumps(cites),
        new_paths_json=json.dumps(new_paths) if new_paths else None,
        new_count=new_count,
    )
    return rid, new_paths


# --------------------------- diff math --------------------------------

def test_first_run_has_no_prior_so_everything_is_new(fresh_db):
    """The first run has nothing to diff against; every citation is 'new'."""
    wid = watchlist_create(fresh_db, "fresh", "q")
    _, new_paths = _finish(fresh_db, wid, ["https://a", "https://b"])
    assert set(new_paths) == {"https://a", "https://b"}


def test_no_diff_when_paths_unchanged(fresh_db):
    wid = watchlist_create(fresh_db, "x", "q")
    _finish(fresh_db, wid, ["https://a", "https://b"])
    _, new_paths = _finish(fresh_db, wid, ["https://a", "https://b"])
    assert new_paths == []


def test_only_new_paths_returned(fresh_db):
    wid = watchlist_create(fresh_db, "x", "q")
    _finish(fresh_db, wid, ["https://a", "https://b"])
    _, new_paths = _finish(fresh_db, wid, ["https://a", "https://c", "https://d"])
    # b dropped (we don't care; just removed coverage); c+d are new.
    assert set(new_paths) == {"https://c", "https://d"}


def test_diff_skips_failed_runs(fresh_db):
    """Failed runs (error != null) shouldn't anchor the diff. The diff
    should compare against the most recent SUCCESSFUL run."""
    wid = watchlist_create(fresh_db, "x", "q")
    _finish(fresh_db, wid, ["https://a", "https://b"])
    # A failed run between the two successful ones.
    rid_bad = watchlist_run_record_start(fresh_db, wid)
    watchlist_run_record_finish(fresh_db, rid_bad, error="rate limited")
    # The next run should still diff against the FIRST run, not the
    # failed run (which has no citations).
    _, new_paths = _finish(fresh_db, wid, ["https://b", "https://c"])
    assert new_paths == ["https://c"], "diff should ignore failed runs"


def test_diff_handles_empty_citations(fresh_db):
    """A run with zero citations shouldn't blow up the diff for the next run."""
    wid = watchlist_create(fresh_db, "x", "q")
    rid1 = watchlist_run_record_start(fresh_db, wid)
    watchlist_run_record_finish(fresh_db, rid1, answer="nothing found",
                                citations_json="[]", new_count=0)
    _, new_paths = _finish(fresh_db, wid, ["https://a"])
    # Prior run had zero paths, so everything in this run is new.
    assert new_paths == ["https://a"]


# ---------------------- watchlist_previous_run helper -------------------

def test_previous_run_finds_only_successful(fresh_db):
    wid = watchlist_create(fresh_db, "x", "q")
    rid_ok = watchlist_run_record_start(fresh_db, wid)
    watchlist_run_record_finish(fresh_db, rid_ok, answer="good",
                                citations_json="[]", new_count=0)
    rid_bad = watchlist_run_record_start(fresh_db, wid)
    watchlist_run_record_finish(fresh_db, rid_bad, error="boom")
    rid_now = watchlist_run_record_start(fresh_db, wid)
    prev = watchlist_previous_run(fresh_db, wid, rid_now)
    assert prev is not None
    assert prev["id"] == rid_ok, "should skip the failed rid_bad"


def test_previous_run_returns_none_when_no_history(fresh_db):
    wid = watchlist_create(fresh_db, "x", "q")
    rid_now = watchlist_run_record_start(fresh_db, wid)
    assert watchlist_previous_run(fresh_db, wid, rid_now) is None


# -------------------- latest_summary surfaces new_count -----------------

def test_latest_summary_includes_diff_fields(fresh_db):
    wid = watchlist_create(fresh_db, "x", "q")
    _finish(fresh_db, wid, ["https://a"])
    _finish(fresh_db, wid, ["https://a", "https://b", "https://c"])
    s = latest_summary(fresh_db, wid)
    assert s is not None
    assert s["new_count"] == 2
    assert set(s["new_paths"]) == {"https://b", "https://c"}


def test_latest_summary_empty_when_no_new_paths(fresh_db):
    wid = watchlist_create(fresh_db, "x", "q")
    _finish(fresh_db, wid, ["https://a"])
    _finish(fresh_db, wid, ["https://a"])
    s = latest_summary(fresh_db, wid)
    assert s["new_count"] == 0
    assert s["new_paths"] == []


# -------------------- run_watchlist persists diff data ------------------

def test_run_watchlist_persists_new_count(fresh_db, tmp_cfg, monkeypatch):
    """When run_watchlist finishes, the new_count + new_paths_json should
    land in watchlist_runs."""
    from secondbrain import watchlist as wl_mod
    from secondbrain.chat import ChatResponse, Citation

    def fake_ask(cfg, conn, embedder, reranker, prompt, **kwargs):
        return ChatResponse(text="answer", citations=[
            Citation(chunk_id=-1, file_path="https://x.com/job/1",
                     chunk_index=0, text="...", score=1.0,
                     kind="web", url="https://x.com/job/1", page_title="J1"),
            Citation(chunk_id=-2, file_path="https://x.com/job/2",
                     chunk_index=0, text="...", score=1.0,
                     kind="web", url="https://x.com/job/2", page_title="J2"),
        ], iterations=1)

    monkeypatch.setattr(wl_mod, "ask_brain", fake_ask)
    # Suppress notify so the test doesn't try to spawn powershell.
    monkeypatch.setattr(wl_mod, "notify", lambda *a, **kw: True)

    wid = watchlist_create(fresh_db, "jobs", "q")
    # First run — both should be "new" (no prior run).
    wl_mod.run_watchlist(tmp_cfg, fresh_db, None, None, wid, "q", None)
    latest = watchlist_latest_run(fresh_db, wid)
    assert latest["new_count"] == 2

    # Second run with same citations — nothing new.
    wl_mod.run_watchlist(tmp_cfg, fresh_db, None, None, wid, "q",
                         watchlist_get(fresh_db, wid)["last_run_at"])
    latest = watchlist_latest_run(fresh_db, wid)
    assert latest["new_count"] == 0


def test_run_watchlist_notifies_only_when_new_and_has_prior(
    fresh_db, tmp_cfg, monkeypatch,
):
    """We don't want a notification on the very first run (everything is
    "new" but the user just created the watchlist). We DO want one on
    subsequent runs when something changed."""
    from secondbrain import watchlist as wl_mod
    from secondbrain.chat import ChatResponse, Citation

    fired: list[tuple[str, str]] = []

    def fake_notify(title, message, url=None):
        fired.append((title, message))
        return True

    monkeypatch.setattr(wl_mod, "notify", fake_notify)

    citations: list[Citation] = []

    def fake_ask(cfg, conn, embedder, reranker, prompt, **kwargs):
        return ChatResponse(text="ok", citations=list(citations), iterations=0)

    monkeypatch.setattr(wl_mod, "ask_brain", fake_ask)

    wid = watchlist_create(fresh_db, "jobs", "q")

    # Run 1: 2 results, no prior -> NO notification.
    citations[:] = [
        Citation(chunk_id=-1, file_path="https://x/1", chunk_index=0,
                 text="", score=1.0, kind="web"),
    ]
    wl_mod.run_watchlist(tmp_cfg, fresh_db, None, None, wid, "q", None)
    assert fired == [], "no notification on first run"

    # Run 2: same paths -> no diff, no notification.
    wl_mod.run_watchlist(tmp_cfg, fresh_db, None, None, wid, "q", None)
    assert fired == []

    # Run 3: a new path -> notification fires.
    citations.append(Citation(
        chunk_id=-2, file_path="https://x/2", chunk_index=0,
        text="", score=1.0, kind="web",
    ))
    wl_mod.run_watchlist(tmp_cfg, fresh_db, None, None, wid, "q", None)
    assert len(fired) == 1
    assert "1 new" in fired[0][0]


# ----------------------- notify module sanity -------------------------

def test_notify_returns_false_for_empty_message():
    from secondbrain.notify import notify
    assert notify("title", "") is False


def test_notify_xml_escape():
    """XAML toast template needs entity escaping. Spot-check it doesn't
    blow up on awkward characters - we don't actually invoke powershell."""
    from secondbrain.notify import _xml_escape

    assert _xml_escape("a & b < c > d") == "a &amp; b &lt; c &gt; d"
    assert _xml_escape('quote "hi"') == "quote &quot;hi&quot;"
