"""Phase 47: tasks module tests.

Two layers covered:

  - **Extraction**: pure-string ``extract_candidates_from_text`` over
    the patterns we expect from Granola / Plaud / generic transcripts.
  - **Persistence + materialisation**: end-to-end against a real
    ``fresh_db``. UNIQUE on (text_lower, source_path) is the load-bearing
    invariant — re-running the extractor must not double-insert.
"""

from __future__ import annotations

import time

from secondbrain import tasks as tasks_mod

# ============================ extraction ==============================

def test_extract_action_items_section_plain_bullets():
    """Granola-style: `- item` bullets under `## Action items`."""
    text = (
        "# [meeting] Sprint planning\n\n"
        "## Summary\n"
        "We talked about the migration.\n\n"
        "## Action items\n"
        "- Email Sarah about the API contract\n"
        "- Draft the migration plan\n"
        "- Review the resume scorer PR\n\n"
        "## Transcript\n"
        "Alice: hello.\n"
    )
    out = list(tasks_mod.extract_candidates_from_text(text))
    assert out == [
        "Email Sarah about the API contract",
        "Draft the migration plan",
        "Review the resume scorer PR",
    ]


def test_extract_action_items_section_checkbox_bullets():
    """Manual-style: `- [ ] item` checkboxes under the same header."""
    text = (
        "# Meeting\n\n"
        "## Action Items\n"
        "- [ ] Ship the brief\n"
        "- [x] Already-done thing\n"
        "- [ ] Reply to recruiter\n"
    )
    out = list(tasks_mod.extract_candidates_from_text(text))
    # Closed checkbox filtered, open ones surface.
    assert "Ship the brief" in out
    assert "Reply to recruiter" in out
    assert "Already-done thing" not in out


def test_extract_falls_back_to_bare_checkboxes_outside_section():
    """No `## Action items` heading, but the doc has open checkboxes —
    still surface them."""
    text = (
        "# Notes from class\n\n"
        "Just regular notes here.\n\n"
        "- [ ] Buy textbook\n"
        "- [x] Already bought pencils\n"
        "More notes below.\n"
        "- [ ] Email TA\n"
    )
    out = list(tasks_mod.extract_candidates_from_text(text))
    assert "Buy textbook" in out
    assert "Email TA" in out
    assert "Already bought pencils" not in out


def test_extract_dedupes_within_one_chunk():
    """Action items section + bare checkbox of the same text — only
    surface once (de-dup is by lowercase text)."""
    text = (
        "## Action items\n"
        "- Send the deck\n\n"
        "Also, in the body:\n"
        "- [ ] Send the deck\n"
    )
    out = list(tasks_mod.extract_candidates_from_text(text))
    assert out.count("Send the deck") == 1


def test_extract_handles_no_action_items_at_all():
    text = "# Meeting\n\nJust a chat. No todos.\n"
    assert list(tasks_mod.extract_candidates_from_text(text)) == []


def test_extract_empty_text():
    assert list(tasks_mod.extract_candidates_from_text("")) == []


def test_extract_action_items_h3_heading():
    """Some tools use `### Action items` (h3) instead of `## Action items`."""
    text = (
        "## Notes\n\n"
        "### Action items\n"
        "- Update the doc\n"
    )
    out = list(tasks_mod.extract_candidates_from_text(text))
    assert out == ["Update the doc"]


def test_extract_strips_trailing_whitespace():
    text = "## Action items\n- Trim me   \n"
    out = list(tasks_mod.extract_candidates_from_text(text))
    assert out == ["Trim me"]


def test_extract_skips_blank_bullet():
    """A bullet with just whitespace shouldn't become an empty task."""
    text = "## Action items\n- \n- Real item\n"
    out = list(tasks_mod.extract_candidates_from_text(text))
    assert out == ["Real item"]


# ============================ persistence =============================

def test_add_manual_inserts_new_task(fresh_db):
    tid = tasks_mod.add_manual(fresh_db, "Buy milk")
    assert tid is not None
    t = tasks_mod.get(fresh_db, tid)
    assert t is not None
    assert t.text == "Buy milk"
    assert t.source_path == "manual"
    assert t.status == "open"


def test_add_manual_returns_existing_id_on_dupe(fresh_db):
    tid1 = tasks_mod.add_manual(fresh_db, "Buy milk")
    tid2 = tasks_mod.add_manual(fresh_db, "Buy milk")
    assert tid1 == tid2  # Same row, no double-insert.
    assert len(tasks_mod.list_open(fresh_db)) == 1


def test_add_manual_rejects_empty(fresh_db):
    assert tasks_mod.add_manual(fresh_db, "") is None
    assert tasks_mod.add_manual(fresh_db, "   ") is None
    assert tasks_mod.list_open(fresh_db) == []


def test_mark_done_changes_status_and_sets_completed_at(fresh_db):
    tid = tasks_mod.add_manual(fresh_db, "Reply to recruiter")
    assert tasks_mod.mark_done(fresh_db, tid)
    t = tasks_mod.get(fresh_db, tid)
    assert t.status == "done"
    assert t.completed_at is not None
    assert t.completed_at > 0


def test_mark_done_idempotent(fresh_db):
    tid = tasks_mod.add_manual(fresh_db, "X")
    assert tasks_mod.mark_done(fresh_db, tid) is True
    # Second call: no change.
    assert tasks_mod.mark_done(fresh_db, tid) is False


def test_mark_done_unknown_id_returns_false(fresh_db):
    assert tasks_mod.mark_done(fresh_db, 9999) is False


def test_mark_cancelled(fresh_db):
    tid = tasks_mod.add_manual(fresh_db, "Reorganise garage")
    assert tasks_mod.mark_cancelled(fresh_db, tid)
    t = tasks_mod.get(fresh_db, tid)
    assert t.status == "cancelled"


def test_delete_removes_row(fresh_db):
    tid = tasks_mod.add_manual(fresh_db, "Typo")
    assert tasks_mod.delete(fresh_db, tid)
    assert tasks_mod.get(fresh_db, tid) is None


def test_list_open_excludes_done_and_cancelled(fresh_db):
    t1 = tasks_mod.add_manual(fresh_db, "open")
    t2 = tasks_mod.add_manual(fresh_db, "done")
    t3 = tasks_mod.add_manual(fresh_db, "cancelled")
    tasks_mod.mark_done(fresh_db, t2)
    tasks_mod.mark_cancelled(fresh_db, t3)
    open_ids = {t.id for t in tasks_mod.list_open(fresh_db)}
    assert open_ids == {t1}


def test_list_recent_done_orders_by_completion(fresh_db):
    t1 = tasks_mod.add_manual(fresh_db, "first done")
    t2 = tasks_mod.add_manual(fresh_db, "second done")
    tasks_mod.mark_done(fresh_db, t1)
    # Tiny sleep so completed_at timestamps are distinguishable on Windows.
    time.sleep(0.02)
    tasks_mod.mark_done(fresh_db, t2)
    rows = tasks_mod.list_recent_done(fresh_db)
    # Most-recently-completed first.
    assert [r.id for r in rows] == [t2, t1]


# ===================== materialisation ================================

def _seed_transcript(conn, path: str, text: str, *, indexed_at=None):
    """Helper — insert a transcript-shaped file + one chunk."""
    n = time.time() if indexed_at is None else indexed_at
    cur = conn.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at, content_hash) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (path, n, 1, "url", n, None),
    )
    fid = cur.lastrowid
    conn.execute(
        "INSERT INTO chunks(file_id, chunk_index, text) VALUES (?, ?, ?)",
        (fid, 0, text),
    )
    conn.commit()
    return fid


def test_materialize_from_transcripts_pulls_action_items(fresh_db):
    _seed_transcript(
        fresh_db, "transcript://granola/abc",
        "# [meeting] Sprint planning\n\n"
        "## Action items\n"
        "- Email Sarah\n"
        "- Draft migration plan\n",
    )
    n = tasks_mod.materialize_from_transcripts(fresh_db)
    assert n == 2
    open_tasks = tasks_mod.list_open(fresh_db)
    texts = [t.text for t in open_tasks]
    assert "Email Sarah" in texts
    assert "Draft migration plan" in texts
    # Source path back-reference should match the doc.
    paths = {t.source_path for t in open_tasks}
    assert paths == {"transcript://granola/abc"}


def test_materialize_is_idempotent(fresh_db):
    """Re-running over the same content must NOT double-insert."""
    _seed_transcript(
        fresh_db, "transcript://granola/abc",
        "## Action items\n- Ship the brief\n",
    )
    first = tasks_mod.materialize_from_transcripts(fresh_db)
    second = tasks_mod.materialize_from_transcripts(fresh_db)
    assert first == 1
    assert second == 0
    assert len(tasks_mod.list_open(fresh_db)) == 1


def test_materialize_skips_old_transcripts(fresh_db):
    """Lookback window respected — old transcripts shouldn't materialise."""
    _seed_transcript(
        fresh_db, "transcript://granola/old",
        "## Action items\n- Stale promise\n",
        indexed_at=time.time() - 60 * 86400,
    )
    n = tasks_mod.materialize_from_transcripts(fresh_db, lookback_days=14)
    assert n == 0
    assert tasks_mod.list_open(fresh_db) == []


def test_materialize_skips_done_items(fresh_db):
    """If the user marked a task done, re-running the extractor must
    not re-create it — that would defeat the whole point of having
    persistent task state."""
    _seed_transcript(
        fresh_db, "transcript://granola/x",
        "## Action items\n- Send the deck\n",
    )
    tasks_mod.materialize_from_transcripts(fresh_db)
    [t] = tasks_mod.list_open(fresh_db)
    tasks_mod.mark_done(fresh_db, t.id)
    # Re-extract: should be 0 new tasks, the done one stays done.
    n = tasks_mod.materialize_from_transcripts(fresh_db)
    assert n == 0
    assert tasks_mod.list_open(fresh_db) == []
    [t2] = tasks_mod.list_recent_done(fresh_db)
    assert t2.id == t.id


def test_materialize_only_scans_transcript_paths(fresh_db):
    """A regular Markdown todos doc shouldn't get its checkboxes scraped
    — that's a separate workflow we haven't built yet."""
    _seed_transcript(
        fresh_db, "C:\\notes\\todos.md",
        "## Action items\n- [ ] Don't scrape me\n",
    )
    assert tasks_mod.materialize_from_transcripts(fresh_db) == 0


def test_materialize_picks_up_doc_title_from_h1(fresh_db):
    """source_title should be the H1 of the source doc, so the user
    can see which meeting an action item came from."""
    _seed_transcript(
        fresh_db, "transcript://granola/y",
        "# [meeting] Career chat with mentor\n"
        "## Action items\n- Update LinkedIn headline\n",
    )
    tasks_mod.materialize_from_transcripts(fresh_db)
    [t] = tasks_mod.list_open(fresh_db)
    assert t.source_title == "[meeting] Career chat with mentor"


def test_materialize_dedupes_across_chunks(fresh_db):
    """Same task text appearing in two chunks of the same doc must
    insert exactly one task row."""
    cur = fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at, content_hash) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("transcript://granola/dupe", time.time(), 1, "url",
         time.time(), None),
    )
    fid = cur.lastrowid
    # Two chunks — both have the same action item text.
    fresh_db.execute(
        "INSERT INTO chunks(file_id, chunk_index, text) VALUES (?, 0, ?)",
        (fid, "## Action items\n- Repeat me\n"),
    )
    fresh_db.execute(
        "INSERT INTO chunks(file_id, chunk_index, text) VALUES (?, 1, ?)",
        (fid, "## Action items\n- Repeat me\n"),
    )
    fresh_db.commit()
    n = tasks_mod.materialize_from_transcripts(fresh_db)
    assert n == 1
    assert len(tasks_mod.list_open(fresh_db)) == 1


# ============================ rendering ===============================

def test_format_task_line_includes_id_and_text(fresh_db):
    tid = tasks_mod.add_manual(fresh_db, "Buy milk")
    t = tasks_mod.get(fresh_db, tid)
    line = tasks_mod.format_task_line(t)
    assert f"#{tid}" in line
    assert "Buy milk" in line


def test_format_task_line_marks_done():
    from secondbrain.tasks import Task
    t = Task(
        id=1, text="x", source_path="manual", source_title="(typed)",
        status="done", created_at=0.0, completed_at=1.0, due_at=None,
    )
    assert "✓" in tasks_mod.format_task_line(t)


def test_format_task_line_marks_cancelled():
    from secondbrain.tasks import Task
    t = Task(
        id=1, text="x", source_path="manual", source_title="(typed)",
        status="cancelled", created_at=0.0, completed_at=1.0, due_at=None,
    )
    assert "✗" in tasks_mod.format_task_line(t)
