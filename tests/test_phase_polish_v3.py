"""Tests for the second polish-pass features:

- ``--as-of`` CLI option + ``_parse_as_of`` (Phase 87 temporal queries)
- ``_filter_by_snapshot`` snapshot-bounded search (Phase 87)
- People backfill scheduler hook is registered (Phase 65)
- Smart-projects section in the daily brief (Phase 73)
- ``secondbrain setup`` wizard helpers (TOML key updater + quoter)

These cover the cross-cutting integration plumbing that doesn't fit
into a per-feature test file. The actual feature internals are tested
elsewhere (test_memory, test_people, test_synthesis); this file is
about wiring.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from unittest.mock import patch

from secondbrain.cli import (
    _RELATIVE_DATE_RE,
    _parse_as_of,
    _suggest_watched_folders,
    _toml_str,
    _toml_str_list,
    _update_toml_keys,
)

# ============================ _parse_as_of ============================

def test_parse_as_of_yesterday():
    now = time.time()
    ts = _parse_as_of("yesterday")
    assert ts is not None
    # Within 5 seconds of now-1d (just guards against clock drift in CI)
    assert abs(ts - (now - 86400)) < 5


def test_parse_as_of_last_week():
    now = time.time()
    ts = _parse_as_of("last week")
    assert ts is not None
    assert abs(ts - (now - 7 * 86400)) < 5


def test_parse_as_of_n_days_ago():
    now = time.time()
    ts = _parse_as_of("3 days ago")
    assert ts is not None
    assert abs(ts - (now - 3 * 86400)) < 5


def test_parse_as_of_n_months_ago():
    now = time.time()
    ts = _parse_as_of("6 months ago")
    assert ts is not None
    assert abs(ts - (now - 6 * 30 * 86400)) < 5


def test_parse_as_of_handles_singular_unit():
    """`1 day ago` (no plural s) should work — that's natural English."""
    now = time.time()
    ts = _parse_as_of("1 day ago")
    assert ts is not None
    assert abs(ts - (now - 86400)) < 5


def test_parse_as_of_iso_date():
    ts = _parse_as_of("2025-01-15")
    assert ts is not None
    # Just confirm it parsed to roughly that date — we don't care about TZ
    parsed = time.localtime(ts)
    assert parsed.tm_year == 2025
    assert parsed.tm_mon == 1
    assert parsed.tm_mday == 15


def test_parse_as_of_iso_datetime():
    ts = _parse_as_of("2025-01-15T10:30:00")
    assert ts is not None
    parsed = time.localtime(ts)
    assert parsed.tm_year == 2025
    assert parsed.tm_hour == 10


def test_parse_as_of_unparseable_returns_none():
    assert _parse_as_of("badger") is None
    assert _parse_as_of("") is None
    assert _parse_as_of("not a date") is None


def test_relative_date_re_case_insensitive():
    # The wizard / CLI treat input case-insensitively.
    assert _RELATIVE_DATE_RE.match("3 DAYS AGO") is not None
    assert _RELATIVE_DATE_RE.match(" 5 weeks ago ") is not None


# ============================ _filter_by_snapshot =====================

def _seed_file(conn, path: str) -> int:
    cur = conn.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at, content_hash) "
        "VALUES (?, ?, 1, 'document', ?, NULL)",
        (path, time.time(), time.time()),
    )
    conn.commit()
    return cur.lastrowid


def test_filter_by_snapshot_keeps_only_in_snapshot_files(fresh_db):
    """Phase 87: when a snapshot covers the requested time, results
    drop down to only the files that existed in that snapshot."""
    from secondbrain import memory
    from secondbrain.search import SearchResult, _filter_by_snapshot

    fid_old = _seed_file(fresh_db, "/old.md")
    _seed_file(fresh_db, "/new.md")  # exists in DB but not in old snapshot
    # Snapshot the world right now — both files in it.
    sid = memory.take_snapshot(fresh_db)
    # Hand-roll a snapshot row that only has fid_old (simulates "back
    # then, /new.md didn't exist yet")
    fresh_db.execute(
        "UPDATE index_snapshots SET file_ids_json = ?, taken_at = ? "
        "WHERE id = ?",
        (f"[{fid_old}]", time.time() - 10 * 86400, sid),
    )
    fresh_db.commit()

    results = [
        SearchResult(
            chunk_id=1, file_path="/old.md", chunk_index=0,
            text="hello", score=0.5, sources=("vector",),
        ),
        SearchResult(
            chunk_id=2, file_path="/new.md", chunk_index=0,
            text="world", score=0.5, sources=("vector",),
        ),
    ]
    out = _filter_by_snapshot(fresh_db, results, time.time() - 5 * 86400)
    paths = [r.file_path for r in out]
    assert "/old.md" in paths
    assert "/new.md" not in paths


def test_filter_by_snapshot_returns_input_when_no_snapshot(fresh_db):
    """No snapshots exist → don't lose results; pass through unchanged."""
    from secondbrain.search import SearchResult, _filter_by_snapshot

    _seed_file(fresh_db, "/x.md")
    results = [
        SearchResult(
            chunk_id=1, file_path="/x.md", chunk_index=0,
            text="hi", score=1.0, sources=("vector",),
        ),
    ]
    out = _filter_by_snapshot(fresh_db, results, time.time() - 365 * 86400)
    assert len(out) == 1


def test_filter_by_snapshot_empty_results_stays_empty(fresh_db):
    """Empty in, empty out — no SQL bind error from a zero-arg IN()."""
    from secondbrain import memory
    from secondbrain.search import _filter_by_snapshot

    _seed_file(fresh_db, "/x.md")
    memory.take_snapshot(fresh_db)
    out = _filter_by_snapshot(fresh_db, [], time.time())
    assert out == []


# ============================ daemon: people_backfill registered ======

def test_daemon_scheduler_includes_people_backfill(fresh_db, tmp_cfg):
    """The daemon's scheduler factory should register a job named
    ``people_backfill`` so newly-mentioned humans get profiles
    automatically."""
    from secondbrain.daemon import _build_daemon_scheduler

    sched = _build_daemon_scheduler(
        tmp_cfg, fresh_db, embedder=None, reranker=None,
    )
    job_names = set(sched.names())
    assert "people_backfill" in job_names, (
        f"Expected people_backfill in {job_names}"
    )


# ============================ daily brief: project_clusters ===========

def test_daily_brief_renders_project_clusters_section():
    """A populated project_clusters list should produce a rendered
    section the user can see."""
    from secondbrain.daily_brief import (
        DailyBrief,
        ProjectClusterLine,
        format_markdown,
    )

    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[],
        assignments_due_soon=[],
        open_action_items=[],
        queue_top=[],
        watchlist_highlights=[],
        project_clusters=[
            ProjectClusterLine(
                suggested_name="Voyage rate-limit investigation",
                seed_title="Voyage rate-limit notes",
                n_members=4,
                score=0.72,
            ),
        ],
    )
    md = format_markdown(brief, header_date="Today")
    assert "Possible projects forming" in md
    assert "Voyage rate-limit investigation" in md
    # Member count appears in the body so the user knows the cluster size
    assert "4 docs" in md


def test_daily_brief_clusters_count_as_actionable():
    """A brief with only project clusters shouldn't fall through to the
    'quiet day' branch — clusters are user-facing surfaces."""
    from secondbrain.daily_brief import (
        DailyBrief,
        ProjectClusterLine,
        _has_actionable_content,
    )

    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[],
        assignments_due_soon=[],
        open_action_items=[],
        queue_top=[],
        watchlist_highlights=[],
        project_clusters=[
            ProjectClusterLine(
                suggested_name="X", seed_title="X", n_members=3, score=0.5,
            ),
        ],
    )
    assert _has_actionable_content(brief) is True


def test_daily_brief_clusters_section_hidden_when_empty():
    """No clusters → no header (avoids dangling ## with nothing under it)."""
    from secondbrain.daily_brief import DailyBrief, format_markdown

    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[],
        assignments_due_soon=[],
        open_action_items=[],
        queue_top=[],
        watchlist_highlights=[],
        project_clusters=[],
    )
    md = format_markdown(brief, header_date="Today")
    assert "Possible projects forming" not in md


def test_project_clusters_section_handles_missing_synthesis(fresh_db):
    """If detect_project_clusters raises (eg. schema gap on a fresh
    brain), the section gracefully returns an empty list rather than
    aborting the whole brief."""
    from secondbrain import daily_brief

    @dataclass
    class _ExplodingCluster:  # noqa: D401 — test fake
        pass

    with patch(
        "secondbrain.synthesis.detect_project_clusters",
        side_effect=RuntimeError("boom"),
    ):
        out = daily_brief._project_clusters_section(fresh_db)
    assert out == []


# ============================ setup wizard helpers ====================

def test_update_toml_keys_replaces_existing():
    """An existing key gets its value rewritten in-place, preserving
    surrounding lines + indentation."""
    src = (
        "# header comment\n"
        "watched_folders = []\n"
        "chunk_size = 800\n"
        "# trailing comment\n"
    )
    out = _update_toml_keys(src, {"watched_folders": '["x"]'})
    assert 'watched_folders = ["x"]' in out
    # Comments preserved
    assert "# header comment" in out
    assert "# trailing comment" in out
    # Other keys untouched
    assert "chunk_size = 800" in out


def test_update_toml_keys_appends_missing():
    """A key not in the source gets appended at the end so the user's
    selection isn't silently dropped."""
    src = "watched_folders = []\n"
    out = _update_toml_keys(src, {"daily_brief_enabled": "true"})
    assert "daily_brief_enabled = true" in out


def test_update_toml_keys_multiple_at_once():
    """Multiple key updates apply in one pass."""
    src = "a = 1\nb = 2\n"
    out = _update_toml_keys(src, {"a": "10", "c": '"new"'})
    assert "a = 10" in out
    assert "b = 2" in out  # untouched
    assert 'c = "new"' in out


def test_update_toml_keys_preserves_other_assignments_with_same_prefix():
    """`watched_folders` and `watched_folders_extra` shouldn't collide
    on the regex — a prefix match must not eat the longer key."""
    src = "watched_folders = []\nwatched_folders_extra = [1]\n"
    out = _update_toml_keys(src, {"watched_folders": '["x"]'})
    assert 'watched_folders = ["x"]' in out
    assert "watched_folders_extra = [1]" in out


def test_toml_str_quotes_simple():
    assert _toml_str("hello") == '"hello"'


def test_toml_str_escapes_backslashes():
    """Windows paths shouldn't break the TOML they're written into."""
    assert _toml_str("C:\\Users\\me") == '"C:\\\\Users\\\\me"'


def test_toml_str_escapes_inner_quotes():
    assert _toml_str('he said "hi"') == '"he said \\"hi\\""'


def test_toml_str_list_empty():
    assert _toml_str_list([]) == "[]"


def test_toml_str_list_round_trips():
    out = _toml_str_list(["a", "b/c"])
    assert out == '["a", "b/c"]'


# ============================ _suggest_watched_folders ================

def test_suggest_watched_folders_only_returns_existing_dirs():
    """Whatever the suggester returns, each entry must be a real
    directory on this machine — we don't want the wizard offering to
    watch a path that doesn't exist."""
    suggestions = _suggest_watched_folders()
    for p in suggestions:
        assert p.is_dir(), f"{p} from suggestions must actually exist"
