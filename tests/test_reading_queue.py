"""Phase 40: reading queue + auto-summaries."""

from __future__ import annotations

from dataclasses import replace

from secondbrain.db import (
    reading_queue_enqueue,
    reading_queue_get,
    reading_queue_history,
    reading_queue_mark_read,
    reading_queue_mark_skipped,
    reading_queue_pending_summary,
    reading_queue_set_summary,
    reading_queue_unread,
    reading_queue_unread_count,
)
from secondbrain.reading_queue import (
    _is_http_url,
    enqueue_from_watchlist_run,
    summarise_pending,
)

# ============================ DB lifecycle ============================

def test_enqueue_returns_id_and_lists_unread(fresh_db):
    rid = reading_queue_enqueue(
        fresh_db, url="https://x.com/job/1", title="Job 1",
        source="watchlist:1", fit_label="great fit", fit_score=0.71,
    )
    assert rid is not None
    rows = reading_queue_unread(fresh_db)
    assert len(rows) == 1
    assert rows[0]["url"] == "https://x.com/job/1"
    assert rows[0]["fit_label"] == "great fit"
    assert reading_queue_unread_count(fresh_db) == 1


def test_enqueue_duplicate_url_returns_none(fresh_db):
    """UNIQUE on url means re-queueing the same URL is a no-op."""
    rid1 = reading_queue_enqueue(
        fresh_db, url="https://a.com/x", title="t", source="manual",
    )
    rid2 = reading_queue_enqueue(
        fresh_db, url="https://a.com/x", title="different", source="manual",
    )
    assert rid1 is not None
    assert rid2 is None
    rows = reading_queue_unread(fresh_db)
    # Original row preserved.
    assert len(rows) == 1
    assert rows[0]["title"] == "t"


def test_enqueue_empty_url_returns_none(fresh_db):
    assert reading_queue_enqueue(fresh_db, url="", title="t", source="x") is None


def test_set_summary_round_trip(fresh_db):
    rid = reading_queue_enqueue(
        fresh_db, url="https://x", title="t", source="manual",
    )
    reading_queue_set_summary(fresh_db, rid, summary="60-second pre-read")
    row = reading_queue_get(fresh_db, rid)
    assert row["summary"] == "60-second pre-read"
    assert row["summary_error"] is None
    assert row["summary_generated_at"] is not None


def test_set_summary_error(fresh_db):
    rid = reading_queue_enqueue(
        fresh_db, url="https://x", title="t", source="manual",
    )
    reading_queue_set_summary(
        fresh_db, rid, summary=None, error="404 not found",
    )
    row = reading_queue_get(fresh_db, rid)
    assert row["summary"] is None
    assert row["summary_error"] == "404 not found"


def test_pending_summary_excludes_summarised_and_errored(fresh_db):
    """summarise_pending should only return rows where summary AND
    summary_error are both null."""
    a = reading_queue_enqueue(fresh_db, url="https://a", title="a", source="m")
    b = reading_queue_enqueue(fresh_db, url="https://b", title="b", source="m")
    c = reading_queue_enqueue(fresh_db, url="https://c", title="c", source="m")
    reading_queue_set_summary(fresh_db, a, summary="done")
    reading_queue_set_summary(fresh_db, b, summary=None, error="rate limit")
    pending = reading_queue_pending_summary(fresh_db)
    pending_ids = {p["id"] for p in pending}
    assert a not in pending_ids
    assert b not in pending_ids
    assert c in pending_ids


def test_pending_summary_excludes_read_and_skipped(fresh_db):
    """If the user marks something read/skipped before the summariser
    gets to it, we should skip it — they don't care anymore."""
    a = reading_queue_enqueue(fresh_db, url="https://a", title="a", source="m")
    b = reading_queue_enqueue(fresh_db, url="https://b", title="b", source="m")
    reading_queue_mark_read(fresh_db, a)
    reading_queue_mark_skipped(fresh_db, b)
    assert reading_queue_pending_summary(fresh_db) == []


def test_unread_excludes_read_and_skipped(fresh_db):
    a = reading_queue_enqueue(fresh_db, url="https://a", title="a", source="m")
    b = reading_queue_enqueue(fresh_db, url="https://b", title="b", source="m")
    c = reading_queue_enqueue(fresh_db, url="https://c", title="c", source="m")
    reading_queue_mark_read(fresh_db, a)
    reading_queue_mark_skipped(fresh_db, b)
    rows = reading_queue_unread(fresh_db)
    assert {r["id"] for r in rows} == {c}


def test_history_includes_read_and_skipped(fresh_db):
    a = reading_queue_enqueue(fresh_db, url="https://a", title="a", source="m")
    b = reading_queue_enqueue(fresh_db, url="https://b", title="b", source="m")
    reading_queue_mark_read(fresh_db, a)
    reading_queue_mark_skipped(fresh_db, b)
    rows = reading_queue_history(fresh_db)
    assert {r["id"] for r in rows} == {a, b}


def test_unread_count_matches_list(fresh_db):
    for i in range(5):
        reading_queue_enqueue(fresh_db, url=f"https://x/{i}", title="t", source="m")
    assert reading_queue_unread_count(fresh_db) == 5


# ============================== filtering =============================

def test_is_http_url_accepts_http_https():
    assert _is_http_url("https://example.com/")
    assert _is_http_url("http://example.com/path?q=1")


def test_is_http_url_rejects_other_schemes():
    """Brain virtual_paths like reddit:// shouldn't go in the queue —
    they're not clickable URLs."""
    assert not _is_http_url("reddit://post/abc")
    assert not _is_http_url("file:///home/x.md")
    assert not _is_http_url("")
    assert not _is_http_url("just a string")


# ============== watchlist → queue: filtering by preset ===============

def test_enqueue_from_watchlist_news_preset_takes_everything(fresh_db):
    """News-flavoured watchlists pipe every new item into the queue."""
    new_items = [
        {"file_path": "https://nyt.com/a", "page_title": "A"},
        {"file_path": "https://wsj.com/b", "page_title": "B"},
    ]
    n = enqueue_from_watchlist_run(
        fresh_db, watchlist_id=42, watchlist_preset="news",
        new_items=new_items,
    )
    assert n == 2
    urls = {r["url"] for r in reading_queue_unread(fresh_db)}
    assert urls == {"https://nyt.com/a", "https://wsj.com/b"}


def test_enqueue_from_watchlist_jobs_preset_filters_by_fit(fresh_db):
    """Job-flavoured watchlists only enqueue 'great fit' items."""
    new_items = [
        {"file_path": "https://x/great", "page_title": "G",
         "fit_label": "great fit", "fit_score": 0.7},
        {"file_path": "https://x/decent", "page_title": "D",
         "fit_label": "decent fit", "fit_score": 0.6},
        {"file_path": "https://x/stretch", "page_title": "S",
         "fit_label": "stretch", "fit_score": 0.5},
    ]
    n = enqueue_from_watchlist_run(
        fresh_db, watchlist_id=1, watchlist_preset="jobs",
        new_items=new_items,
    )
    assert n == 1
    urls = {r["url"] for r in reading_queue_unread(fresh_db)}
    assert urls == {"https://x/great"}


def test_enqueue_from_watchlist_no_preset_filters_by_fit(fresh_db):
    """Watchlists without a recognised preset behave like jobs (only
    great-fit) so we don't pollute the queue with random web hits."""
    new_items = [
        {"file_path": "https://x/great", "fit_label": "great fit"},
        {"file_path": "https://x/decent", "fit_label": "decent fit"},
    ]
    n = enqueue_from_watchlist_run(
        fresh_db, watchlist_id=1, watchlist_preset=None, new_items=new_items,
    )
    assert n == 1


def test_enqueue_from_watchlist_skips_non_http_urls(fresh_db):
    """Brain virtual_paths (reddit://, etc.) shouldn't end up in the queue."""
    new_items = [
        {"file_path": "reddit://post/abc", "fit_label": "great fit"},
        {"file_path": "https://x/job", "fit_label": "great fit"},
    ]
    n = enqueue_from_watchlist_run(
        fresh_db, watchlist_id=1, watchlist_preset="jobs", new_items=new_items,
    )
    assert n == 1


def test_enqueue_from_watchlist_dedups_by_url(fresh_db):
    """Same URL surfacing twice across runs queues once."""
    new_items = [{"file_path": "https://x", "fit_label": "great fit"}]
    enqueue_from_watchlist_run(fresh_db, 1, "jobs", new_items)
    enqueue_from_watchlist_run(fresh_db, 1, "jobs", new_items)
    rows = reading_queue_unread(fresh_db)
    assert len(rows) == 1


def test_enqueue_from_watchlist_empty_returns_zero(fresh_db):
    assert enqueue_from_watchlist_run(fresh_db, 1, "news", []) == 0


def test_enqueue_from_watchlist_records_source_and_fit(fresh_db):
    new_items = [{
        "file_path": "https://anthropic.com/jobs/1",
        "page_title": "PM Intern", "fit_label": "great fit",
        "fit_score": 0.78,
    }]
    enqueue_from_watchlist_run(fresh_db, 7, "jobs", new_items)
    row = reading_queue_unread(fresh_db)[0]
    assert row["source"] == "watchlist:7"
    assert row["fit_label"] == "great fit"
    assert row["fit_score"] == 0.78


# ============================== summariser ============================

def test_summarise_pending_disabled(fresh_db, tmp_cfg, fake_embedder):
    cfg = replace(tmp_cfg, read_queue_enabled=False)
    reading_queue_enqueue(fresh_db, url="https://x", title="t", source="m")
    assert summarise_pending(cfg, fresh_db, fake_embedder, None) == 0


def test_summarise_pending_writes_summary(
    fresh_db, tmp_cfg, fake_embedder, monkeypatch,
):
    """Stub ask_brain to return a deterministic summary; verify it
    lands on the row."""
    from secondbrain import reading_queue as rq
    from secondbrain.chat import ChatResponse

    monkeypatch.setattr(
        rq, "ask_brain",
        lambda *a, **kw: ChatResponse(text="60-second precis", citations=[]),
    )
    rid = reading_queue_enqueue(
        fresh_db, url="https://x", title="t", source="m",
    )
    n = summarise_pending(tmp_cfg, fresh_db, fake_embedder, None)
    assert n == 1
    row = reading_queue_get(fresh_db, rid)
    assert row["summary"] == "60-second precis"
    assert row["summary_error"] is None


def test_summarise_pending_records_budget_failure(
    fresh_db, tmp_cfg, fake_embedder, monkeypatch,
):
    from secondbrain import reading_queue as rq
    from secondbrain.budget import BudgetExceededError

    def boom(*a, **kw):
        raise BudgetExceededError("anthropic", 1000.0, 500.0)

    monkeypatch.setattr(rq, "ask_brain", boom)
    rid = reading_queue_enqueue(fresh_db, url="https://x", title="t", source="m")
    n = summarise_pending(tmp_cfg, fresh_db, fake_embedder, None)
    assert n == 0
    row = reading_queue_get(fresh_db, rid)
    assert row["summary"] is None
    assert "budget" in (row["summary_error"] or "").lower()


def test_summarise_pending_records_unexpected_failure(
    fresh_db, tmp_cfg, fake_embedder, monkeypatch,
):
    from secondbrain import reading_queue as rq

    def boom(*a, **kw):
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(rq, "ask_brain", boom)
    rid = reading_queue_enqueue(fresh_db, url="https://x", title="t", source="m")
    summarise_pending(tmp_cfg, fresh_db, fake_embedder, None)
    row = reading_queue_get(fresh_db, rid)
    assert "network unreachable" in (row["summary_error"] or "")


def test_summarise_pending_skips_already_summarised(
    fresh_db, tmp_cfg, fake_embedder, monkeypatch,
):
    """Once summarised, an item shouldn't be re-summarised on the next
    daemon tick."""
    from secondbrain import reading_queue as rq

    calls: list = []

    def fake_ask(*a, **kw):
        calls.append(1)
        from secondbrain.chat import ChatResponse
        return ChatResponse(text="x", citations=[])

    monkeypatch.setattr(rq, "ask_brain", fake_ask)
    rid = reading_queue_enqueue(fresh_db, url="https://x", title="t", source="m")
    summarise_pending(tmp_cfg, fresh_db, fake_embedder, None)
    summarise_pending(tmp_cfg, fresh_db, fake_embedder, None)
    assert len(calls) == 1, "second tick shouldn't re-summarise"
    assert reading_queue_get(fresh_db, rid)["summary"] == "x"


def test_summarise_pending_respects_per_run_cap(
    fresh_db, tmp_cfg, fake_embedder, monkeypatch,
):
    from secondbrain import reading_queue as rq

    def fake_ask(*a, **kw):
        from secondbrain.chat import ChatResponse
        return ChatResponse(text="x", citations=[])

    monkeypatch.setattr(rq, "ask_brain", fake_ask)
    for i in range(10):
        reading_queue_enqueue(fresh_db, url=f"https://x/{i}", title="t", source="m")
    cfg = replace(tmp_cfg, read_queue_summarise_per_run=3)
    n = summarise_pending(cfg, fresh_db, fake_embedder, None)
    assert n == 3


# =================== watchlist preset inference ======================

def test_watchlist_preset_for_returns_matching_preset(fresh_db):
    """If a watchlist's saved domains match one of the named presets,
    watchlist_preset_for returns that preset name."""
    from secondbrain.db import watchlist_create
    from secondbrain.presets import PRESETS
    from secondbrain.reading_queue import watchlist_preset_for

    wid = watchlist_create(
        fresh_db, "n", "q",
        allowed_domains=PRESETS["news"],
    )
    assert watchlist_preset_for(fresh_db, wid) == "news"


def test_watchlist_preset_for_returns_none_for_custom_domains(fresh_db):
    """Custom or partial domain lists don't match any preset."""
    from secondbrain.db import watchlist_create
    from secondbrain.reading_queue import watchlist_preset_for

    wid = watchlist_create(
        fresh_db, "x", "q", allowed_domains=["custom.com"],
    )
    assert watchlist_preset_for(fresh_db, wid) is None


def test_watchlist_preset_for_returns_none_when_no_domains(fresh_db):
    from secondbrain.db import watchlist_create
    from secondbrain.reading_queue import watchlist_preset_for

    wid = watchlist_create(fresh_db, "x", "q")
    assert watchlist_preset_for(fresh_db, wid) is None
