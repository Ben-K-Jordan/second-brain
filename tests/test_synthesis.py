"""Phase 72 + 73 + 74 + 75: synthesis tests.

Coverage:
  - Phase 74 auto-summary: needs_summary, materialize_summary,
    materialize_summaries_due
  - Phase 73 smart projects: cluster detection from backlinks
  - Phase 75 insights: topic spikes + health drift + dedup
  - Phase 72 weekly review: assemble + format + idempotent index
"""

from __future__ import annotations

import time

from secondbrain import synthesis

# ============================ helpers =================================

def _seed_doc(conn, *, path, title, body, indexed_at=None):
    n = indexed_at or time.time()
    cur = conn.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at, content_hash) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (path, n, len(body), "document", n, None),
    )
    fid = cur.lastrowid
    full = f"# {title}\n\n{body}"
    conn.execute(
        "INSERT INTO chunks(file_id, chunk_index, text) VALUES (?, ?, ?)",
        (fid, 0, full),
    )
    conn.commit()
    return fid


def _stub_summary_generator(title, body, cfg):
    return {
        "tldr": f"This is a TL;DR for {title}.",
        "key_points": ["point a", "point b", "point c"],
    }


# ============================ Phase 74: auto-summary ==================

def test_needs_summary_only_for_long_docs(fresh_db):
    short = _seed_doc(
        fresh_db, path="short.md", title="Short", body="Tiny note.",
    )
    long = _seed_doc(
        fresh_db, path="long.md", title="Long",
        body="A " * 5000,
    )
    assert synthesis.needs_summary(fresh_db, short) is False
    assert synthesis.needs_summary(fresh_db, long) is True


def test_materialize_summary_persists(fresh_db):
    fid = _seed_doc(
        fresh_db, path="long.md", title="Voyage rate limits",
        body="A long body about voyage rate limits. " * 200,
    )
    out = synthesis.materialize_summary(
        fresh_db, fid, generator=_stub_summary_generator,
    )
    assert out is not None
    assert "TL;DR" in out.tldr
    assert len(out.key_points) == 3
    # Persisted.
    fetched = synthesis.get_summary(fresh_db, fid)
    assert fetched is not None
    assert fetched.tldr == out.tldr


def test_materialize_summary_idempotent(fresh_db):
    """Re-running on a doc that already has a summary should be a no-op."""
    fid = _seed_doc(
        fresh_db, path="long.md", title="x", body="A " * 5000,
    )
    out1 = synthesis.materialize_summary(
        fresh_db, fid, generator=_stub_summary_generator,
    )
    out2 = synthesis.materialize_summary(
        fresh_db, fid, generator=_stub_summary_generator,
    )
    assert out1 is not None
    assert out2 is None  # already summarised


def test_materialize_summary_handles_generator_failure(fresh_db):
    fid = _seed_doc(
        fresh_db, path="long.md", title="x", body="A " * 5000,
    )
    def boom(t, b, c):
        raise RuntimeError("LLM down")
    out = synthesis.materialize_summary(
        fresh_db, fid, generator=boom,
    )
    assert out is None
    # No row persisted.
    assert synthesis.get_summary(fresh_db, fid) is None


def test_materialize_summary_skips_empty_response(fresh_db):
    fid = _seed_doc(
        fresh_db, path="long.md", title="x", body="A " * 5000,
    )
    out = synthesis.materialize_summary(
        fresh_db, fid, generator=lambda t, b, c: {},
    )
    assert out is None


def test_materialize_summary_skips_blank_tldr(fresh_db):
    """Generator returning empty TL;DR should not persist."""
    fid = _seed_doc(
        fresh_db, path="long.md", title="x", body="A " * 5000,
    )
    out = synthesis.materialize_summary(
        fresh_db, fid,
        generator=lambda t, b, c: {"tldr": "", "key_points": ["x"]},
    )
    assert out is None


def test_materialize_summaries_due_caps_per_run(fresh_db):
    for i in range(5):
        _seed_doc(
            fresh_db, path=f"long-{i}.md", title=f"L{i}",
            body="A " * 5000,
        )
    n = synthesis.materialize_summaries_due(
        fresh_db, cfg=None, max_per_run=3,
        generator=_stub_summary_generator,
    )
    assert n == 3


def test_get_summary_returns_none_for_missing(fresh_db):
    synthesis._ensure_schema(fresh_db)
    assert synthesis.get_summary(fresh_db, 9999) is None


# ============================ Phase 73: smart projects ================

def test_detect_project_clusters_returns_empty_with_no_data(fresh_db):
    assert synthesis.detect_project_clusters(fresh_db) == []


def test_detect_project_clusters_finds_linked_group(fresh_db):
    """Three recent docs that all link together → one cluster."""
    fid_a = _seed_doc(
        fresh_db, path="A.md", title="A", body="content a",
    )
    fid_b = _seed_doc(
        fresh_db, path="B.md", title="B", body="content b",
    )
    fid_c = _seed_doc(
        fresh_db, path="C.md", title="C", body="content c",
    )
    # Insert backlinks: A↔B, A↔C, B↔C with low (good) scores.
    n = time.time()
    for src, dst, score in [
        (fid_a, fid_b, 0.1),
        (fid_b, fid_a, 0.1),
        (fid_a, fid_c, 0.2),
        (fid_c, fid_a, 0.2),
        (fid_b, fid_c, 0.15),
        (fid_c, fid_b, 0.15),
    ]:
        fresh_db.execute(
            "INSERT INTO backlinks(src_file_id, dst_file_id, score, created_at) "
            "VALUES (?, ?, ?, ?)",
            (src, dst, score, n),
        )
    fresh_db.commit()
    clusters = synthesis.detect_project_clusters(fresh_db, min_size=2)
    assert len(clusters) >= 1
    # Member paths should include at least 2 of the docs.
    paths = clusters[0].member_paths
    assert sum(p in {"A.md", "B.md", "C.md"} for p in paths) >= 2


def test_detect_project_clusters_skips_single_orphan(fresh_db):
    """A doc with no backlinks shouldn't be returned as a cluster."""
    _seed_doc(fresh_db, path="alone.md", title="Alone", body="x")
    clusters = synthesis.detect_project_clusters(
        fresh_db, min_size=2,
    )
    assert clusters == []


def test_suggest_project_name_picks_common_words():
    titles = [
        "ML capstone discussion",
        "ML capstone planning",
        "ML capstone retrospective",
    ]
    name = synthesis._suggest_project_name(titles)
    # 'capstone' is the longest non-noise word in common.
    assert "capstone" in name.lower()


def test_suggest_project_name_falls_back_when_no_common():
    titles = ["alpha bravo", "charlie delta"]
    name = synthesis._suggest_project_name(titles)
    # Falls back to first title's words.
    assert name


def test_suggest_project_name_handles_empty():
    assert synthesis._suggest_project_name([]) == ""


# ============================ Phase 75: insights ======================

def test_is_topic_spike_requires_minimum_docs(fresh_db):
    """A token mentioned in just 1-2 recent docs should not spike."""
    # Seed two docs each mentioning 'voyage'.
    n = time.time()
    for i in range(2):
        fid = _seed_doc(
            fresh_db, path=f"recent-{i}.md", title=f"R{i}",
            body="discussion of voyage", indexed_at=n,
        )
        cur = fresh_db.execute(
            "SELECT id FROM chunks WHERE file_id = ?", (fid,),
        ).fetchone()
        fresh_db.execute(
            "INSERT INTO entities(chunk_id, text, text_lower, label) "
            "VALUES (?, 'Voyage', 'voyage', 'PRODUCT')",
            (cur["id"],),
        )
    fresh_db.commit()
    spikes = synthesis.detect_topic_spikes(fresh_db, min_docs=4)
    assert spikes == []


def test_topic_spike_fires_with_enough_recent_density(fresh_db):
    """4 mentions this week, 0 in prior period → spike."""
    n = time.time()
    for i in range(5):
        fid = _seed_doc(
            fresh_db, path=f"recent-{i}.md", title=f"R{i}",
            body="discussion", indexed_at=n,
        )
        cur = fresh_db.execute(
            "SELECT id FROM chunks WHERE file_id = ?", (fid,),
        ).fetchone()
        fresh_db.execute(
            "INSERT INTO entities(chunk_id, text, text_lower, label) "
            "VALUES (?, 'Voyage', 'voyage', 'PRODUCT')",
            (cur["id"],),
        )
    fresh_db.commit()
    spikes = synthesis.detect_topic_spikes(fresh_db, min_docs=4)
    assert any("voyage" in s.key for s in spikes)


def test_health_drift_returns_none_with_insufficient_data(fresh_db):
    """Need 10+ days of data to compute drift."""
    from secondbrain import health
    health.ingest_summaries(fresh_db, [])
    assert synthesis.detect_health_drift(fresh_db) is None


def test_health_drift_fires_when_below_threshold(fresh_db):
    """7-day avg 15%+ below 30-day → fires."""
    from datetime import datetime, timedelta

    from secondbrain import health
    from secondbrain.connectors.oura import DailySummary
    today = datetime.now().date()
    summaries = []
    # 21 days of avg 80, then 7 days of avg 60.
    for d in range(21):
        date = (today - timedelta(days=27 - d)).isoformat()
        summaries.append(DailySummary(date=date, sleep_score=80))
    for d in range(7):
        date = (today - timedelta(days=6 - d)).isoformat()
        summaries.append(DailySummary(date=date, sleep_score=60))
    health.ingest_summaries(fresh_db, summaries)
    insight = synthesis.detect_health_drift(
        fresh_db, "sleep_score", threshold_pct=15,
    )
    assert insight is not None
    assert "sleep" in insight.headline.lower()
    assert insight.payload["delta_pct"] < -15


def test_detect_insights_dedupes_recent_fires(fresh_db):
    """An insight that fired in the last 7 days shouldn't re-surface."""
    synthesis._ensure_schema(fresh_db)
    fresh_db.execute(
        "INSERT INTO synthesis_runs(kind, ran_at, insight_key) "
        "VALUES ('insight', ?, 'topic_spike:voyage')",
        (time.time(),),
    )
    fresh_db.commit()
    # Even if topic_spike would fire for voyage, dedup blocks it.
    n = time.time()
    for i in range(5):
        fid = _seed_doc(
            fresh_db, path=f"recent-{i}.md", title=f"R{i}",
            body="x", indexed_at=n,
        )
        cur = fresh_db.execute(
            "SELECT id FROM chunks WHERE file_id = ?", (fid,),
        ).fetchone()
        fresh_db.execute(
            "INSERT INTO entities(chunk_id, text, text_lower, label) "
            "VALUES (?, 'Voyage', 'voyage', 'PRODUCT')",
            (cur["id"],),
        )
    fresh_db.commit()
    insights = synthesis.detect_insights(fresh_db)
    assert all(i.key != "topic_spike:voyage" for i in insights)


def test_record_insight_fired_persists(fresh_db):
    synthesis._ensure_schema(fresh_db)
    insight = synthesis.Insight(
        key="x", kind="topic_spike", headline="h", detail="d",
    )
    synthesis.record_insight_fired(fresh_db, insight)
    row = fresh_db.execute(
        "SELECT insight_key FROM synthesis_runs WHERE kind = 'insight'",
    ).fetchone()
    assert row["insight_key"] == "x"


# ============================ Phase 72: weekly review =================

def test_assemble_weekly_review_no_data(fresh_db):
    review = synthesis.assemble_weekly_review(fresh_db)
    assert review.n_docs_indexed == 0
    assert review.tasks_done == []


def test_assemble_weekly_review_counts_docs(fresh_db):
    _seed_doc(
        fresh_db, path="x.md", title="x", body="recent doc",
    )
    review = synthesis.assemble_weekly_review(fresh_db)
    assert review.n_docs_indexed == 1


def test_assemble_weekly_review_picks_up_done_tasks(fresh_db):
    from secondbrain import tasks as tasks_mod
    tid = tasks_mod.add_manual(fresh_db, "Reply to recruiter")
    tasks_mod.mark_done(fresh_db, tid)
    review = synthesis.assemble_weekly_review(fresh_db)
    assert "Reply to recruiter" in review.tasks_done


def test_assemble_weekly_review_lingering_open_tasks(fresh_db):
    """Tasks open >7d should appear under lingering."""
    fresh_db.execute(
        "INSERT INTO tasks(text, text_lower, source_path, source_title, "
        " status, created_at) "
        "VALUES ('Old task', 'old task', 'manual', '(typed)', "
        " 'open', ?)",
        (time.time() - 14 * 86400,),
    )
    fresh_db.commit()
    review = synthesis.assemble_weekly_review(fresh_db)
    assert "Old task" in review.tasks_lingering


def test_format_weekly_review_md_includes_sections():
    review = synthesis.WeeklyReview(
        week_start="2026-04-20", week_end="2026-04-26",
        n_docs_indexed=10, n_meetings=3, n_lectures=4,
        n_jobs_applied=2,
        tasks_done=["a", "b"],
        tasks_lingering=["lingering"],
        top_topics=[("Sarah", 5)],
        health_summary="sleep 82",
    )
    md = synthesis.format_weekly_review_md(review)
    assert "Weekly review" in md
    assert "10 new doc" in md
    assert "3 meeting" in md
    assert "Done this week" in md
    assert "Lingering" in md
    assert "Top entities" in md
    assert "sleep 82" in md


def test_format_weekly_review_md_omits_empty_sections():
    review = synthesis.WeeklyReview(
        week_start="2026-04-20", week_end="2026-04-26",
        n_docs_indexed=0, n_meetings=0, n_lectures=0,
        n_jobs_applied=0, tasks_done=[], tasks_lingering=[],
        top_topics=[], health_summary="",
    )
    md = synthesis.format_weekly_review_md(review)
    assert "Done this week" not in md
    assert "Lingering" not in md


def test_format_weekly_review_truncates_long_done_list():
    review = synthesis.WeeklyReview(
        week_start="2026-04-20", week_end="2026-04-26",
        n_docs_indexed=0, n_meetings=0, n_lectures=0,
        n_jobs_applied=0,
        tasks_done=[f"task {i}" for i in range(20)],
        tasks_lingering=[],
        top_topics=[], health_summary="",
    )
    md = synthesis.format_weekly_review_md(review)
    assert "+10 more" in md


def test_has_recent_weekly_review(fresh_db):
    synthesis._ensure_schema(fresh_db)
    assert synthesis.has_recent_weekly_review(fresh_db) is False
    fresh_db.execute(
        "INSERT INTO synthesis_runs(kind, ran_at, virtual_path) "
        "VALUES ('weekly_review', ?, 'review://x')",
        (time.time(),),
    )
    fresh_db.commit()
    assert synthesis.has_recent_weekly_review(fresh_db) is True


def test_has_recent_weekly_review_old_does_not_count(fresh_db):
    synthesis._ensure_schema(fresh_db)
    fresh_db.execute(
        "INSERT INTO synthesis_runs(kind, ran_at, virtual_path) "
        "VALUES ('weekly_review', ?, 'review://x')",
        (time.time() - 30 * 86400,),
    )
    fresh_db.commit()
    assert synthesis.has_recent_weekly_review(fresh_db) is False
