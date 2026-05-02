"""Phase 79 + 80 + 81: personal context tests.

Coverage:
  - habits: add/checkin idempotence, streak math, adherence
  - goals: add/progress, on_track, weekly aggregation
  - journal: upsert, mood correlation, recent
  - projects: create + tag + view, slug uniqueness
"""

from __future__ import annotations

import time
from datetime import date, timedelta

import pytest

from secondbrain import personal

# ============================ habits ==================================

def test_add_habit_creates_row(fresh_db):
    hid = personal.add_habit(fresh_db, "Workout")
    assert hid > 0
    habits = personal.list_habits(fresh_db)
    assert len(habits) == 1
    assert habits[0].name == "Workout"


def test_add_habit_idempotent_on_name(fresh_db):
    hid_a = personal.add_habit(fresh_db, "Workout")
    hid_b = personal.add_habit(fresh_db, "Workout")
    assert hid_a == hid_b
    assert len(personal.list_habits(fresh_db)) == 1


def test_add_habit_validates_cadence(fresh_db):
    with pytest.raises(ValueError):
        personal.add_habit(fresh_db, "x", cadence="annual")


def test_add_habit_n_per_week_requires_target(fresh_db):
    with pytest.raises(ValueError):
        personal.add_habit(fresh_db, "x", cadence="N_per_week")


def test_add_habit_rejects_blank(fresh_db):
    with pytest.raises(ValueError):
        personal.add_habit(fresh_db, "")


def test_archive_habit_hides_from_list(fresh_db):
    hid = personal.add_habit(fresh_db, "x")
    personal.archive_habit(fresh_db, hid)
    assert personal.list_habits(fresh_db) == []
    assert len(personal.list_habits(fresh_db, include_archived=True)) == 1


def test_checkin_idempotent_per_day(fresh_db):
    hid = personal.add_habit(fresh_db, "x")
    today = date.today()
    assert personal.checkin(fresh_db, hid, when=today) is True
    assert personal.checkin(fresh_db, hid, when=today) is False


def test_streak_math_consecutive_days(fresh_db):
    hid = personal.add_habit(fresh_db, "x")
    today = date.today()
    for i in range(5):
        personal.checkin(
            fresh_db, hid, when=today - timedelta(days=i),
        )
    s = personal.habit_status(fresh_db, hid)
    assert s.current_streak_days == 5
    assert s.longest_streak_days == 5


def test_streak_breaks_with_gap(fresh_db):
    hid = personal.add_habit(fresh_db, "x")
    today = date.today()
    # Yesterday + 5 days ago — gap means streak resets to 1.
    personal.checkin(fresh_db, hid, when=today - timedelta(days=1))
    personal.checkin(fresh_db, hid, when=today - timedelta(days=5))
    s = personal.habit_status(fresh_db, hid)
    assert s.current_streak_days == 1


def test_adherence_30d_count(fresh_db):
    hid = personal.add_habit(fresh_db, "x")
    today = date.today()
    for i in range(15):
        personal.checkin(
            fresh_db, hid, when=today - timedelta(days=i),
        )
    s = personal.habit_status(fresh_db, hid)
    assert s.checkins_last_30d == 15
    assert s.expected_30d == 30  # daily cadence


def test_habit_status_unknown_id(fresh_db):
    with pytest.raises(ValueError):
        personal.habit_status(fresh_db, 9999)


# ============================ goals ===================================

def test_add_goal(fresh_db):
    personal.add_goal(
        fresh_db, "Apply to jobs", target_per_week=5,
        description="Land an internship by May",
    )
    goals = personal.list_goals(fresh_db)
    assert len(goals) == 1
    assert goals[0].target_per_week == 5
    assert goals[0].description == "Land an internship by May"


def test_record_goal_progress_aggregates_this_week(fresh_db):
    gid = personal.add_goal(fresh_db, "x", target_per_week=5)
    today = date.today()
    week_start = today - timedelta(days=today.weekday())
    # Two events this week.
    personal.record_goal_progress(fresh_db, gid, count=1, when=week_start)
    personal.record_goal_progress(fresh_db, gid, count=2, when=today)
    s = personal.goal_status(fresh_db, gid)
    assert s.progress_this_week == 3
    assert s.on_track is False  # 3 < 5 target


def test_goal_on_track_when_at_target(fresh_db):
    gid = personal.add_goal(fresh_db, "x", target_per_week=2)
    personal.record_goal_progress(fresh_db, gid, count=2)
    assert personal.goal_status(fresh_db, gid).on_track is True


def test_goal_no_target_always_on_track(fresh_db):
    """A goal without a numeric target shouldn't be 'off track'."""
    gid = personal.add_goal(fresh_db, "Be kinder")
    s = personal.goal_status(fresh_db, gid)
    assert s.on_track is True


def test_list_goals_excludes_archived(fresh_db):
    personal.add_goal(fresh_db, "x")
    fresh_db.execute(
        "UPDATE goals SET archived_at = ? WHERE name = 'x'",
        (time.time(),),
    )
    fresh_db.commit()
    assert personal.list_goals(fresh_db) == []


# ============================ journal =================================

def test_journal_upsert_creates(fresh_db):
    eid = personal.upsert_journal(fresh_db, mood=4, text="Good day.")
    e = personal.get_journal(fresh_db)
    assert e.id == eid
    assert e.mood == 4
    assert e.text == "Good day."


def test_journal_upsert_updates_existing(fresh_db):
    eid = personal.upsert_journal(fresh_db, mood=3, text="Meh.")
    eid2 = personal.upsert_journal(fresh_db, mood=4, text="Better.")
    assert eid == eid2
    e = personal.get_journal(fresh_db)
    assert e.mood == 4
    assert e.text == "Better."


def test_journal_upsert_partial_update(fresh_db):
    """Updating just mood shouldn't clobber the text."""
    personal.upsert_journal(fresh_db, mood=3, text="long entry text")
    personal.upsert_journal(fresh_db, mood=5)
    e = personal.get_journal(fresh_db)
    assert e.mood == 5
    assert e.text == "long entry text"


def test_journal_clamps_mood(fresh_db):
    personal.upsert_journal(fresh_db, mood=99)
    e = personal.get_journal(fresh_db)
    assert e.mood == 5
    eid2 = personal.upsert_journal(
        fresh_db, mood=-1, when=date.today() - timedelta(days=1),
    )
    e2 = fresh_db.execute(
        "SELECT mood FROM journal_entries WHERE id = ?", (eid2,),
    ).fetchone()
    assert e2["mood"] == 1


def test_journal_recent_returns_window(fresh_db):
    today = date.today()
    for i in range(5):
        personal.upsert_journal(
            fresh_db, mood=3, text=f"day {i}",
            when=today - timedelta(days=i),
        )
    out = personal.recent_journal(fresh_db, days=3)
    assert len(out) <= 4  # window is days back from today


def test_mood_correlation_returns_empty_with_too_little_data(fresh_db):
    out = personal.mood_correlation_with_metric(fresh_db, "sleep_score")
    assert out == {}


def test_mood_correlation_computes_pearson(fresh_db):
    """Mood and sleep both rising in lockstep → positive r."""
    today = date.today()
    for i in range(10):
        d = (today - timedelta(days=i)).isoformat()
        mood = 5 - (i % 5)
        personal.upsert_journal(
            fresh_db, mood=mood, when=date.fromisoformat(d),
        )
        # Health metric mirrors mood values × 16 to give different scale.
        fresh_db.execute(
            "INSERT INTO health_metrics(date, metric, value, source, recorded_at) "
            "VALUES (?, 'sleep_score', ?, 'oura', ?)",
            (d, mood * 16.0, time.time()),
        )
    fresh_db.commit()
    out = personal.mood_correlation_with_metric(fresh_db, "sleep_score")
    assert out
    assert out["pearson_r"] > 0.5  # strong positive
    assert out["n"] >= 5


# ============================ projects ================================

def test_create_project_assigns_slug(fresh_db):
    pid = personal.create_project(fresh_db, "ML Capstone")
    p = personal.get_project_by_slug(fresh_db, "ml-capstone")
    assert p is not None
    assert p.id == pid
    assert p.name == "ML Capstone"


def test_create_project_idempotent_on_slug(fresh_db):
    pid_a = personal.create_project(fresh_db, "ML Capstone")
    pid_b = personal.create_project(fresh_db, "ml-capstone!!!")
    # Same slug, same id.
    assert pid_a == pid_b


def test_slug_handles_unicode_and_punctuation():
    assert personal.slug_for("Hello, World!!!") == "hello-world"
    assert personal.slug_for("---") == "untitled"
    assert personal.slug_for("Über") == "ber"  # ü stripped


def test_add_to_project_idempotent(fresh_db):
    pid = personal.create_project(fresh_db, "x")
    assert personal.add_to_project(
        fresh_db, pid, kind="file", ref_id=1,
    ) is True
    assert personal.add_to_project(
        fresh_db, pid, kind="file", ref_id=1,
    ) is False


def test_remove_from_project(fresh_db):
    pid = personal.create_project(fresh_db, "x")
    personal.add_to_project(fresh_db, pid, kind="task", ref_id=42)
    assert personal.remove_from_project(
        fresh_db, pid, kind="task", ref_id=42,
    ) is True
    assert personal.remove_from_project(
        fresh_db, pid, kind="task", ref_id=42,
    ) is False  # already gone


def test_add_to_project_validates_kind(fresh_db):
    pid = personal.create_project(fresh_db, "x")
    with pytest.raises(ValueError):
        personal.add_to_project(
            fresh_db, pid, kind="random", ref_id=1,
        )


def test_project_view_hydrates_files_and_tasks(fresh_db):
    """Verify project_view returns the full hydrated members."""
    # Seed a project + a file + a task.
    pid = personal.create_project(fresh_db, "x")
    cur = fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at) "
        "VALUES ('/notes/x.md', ?, 1, 'document', ?)",
        (time.time(), time.time()),
    )
    fid = cur.lastrowid
    cur = fresh_db.execute(
        "INSERT INTO tasks(text, text_lower, source_path, source_title, "
        " status, created_at) "
        "VALUES ('Do the thing', 'do the thing', 'manual', '(typed)', "
        " 'open', ?) RETURNING id",
        (time.time(),),
    )
    tid = int(cur.fetchone()["id"])
    fresh_db.commit()
    personal.add_to_project(fresh_db, pid, kind="file", ref_id=fid)
    personal.add_to_project(fresh_db, pid, kind="task", ref_id=tid)
    view = personal.project_view(fresh_db, pid)
    assert view is not None
    assert len(view.files) == 1
    assert view.files[0][1] == "/notes/x.md"
    assert len(view.tasks) == 1
    assert view.tasks[0][1] == "Do the thing"


def test_project_view_unknown_returns_none(fresh_db):
    assert personal.project_view(fresh_db, 9999) is None


def test_list_projects_excludes_archived(fresh_db):
    personal.create_project(fresh_db, "live")
    personal.create_project(fresh_db, "dead")
    fresh_db.execute(
        "UPDATE projects SET archived_at = ? WHERE slug = 'dead'",
        (time.time(),),
    )
    fresh_db.commit()
    slugs = [p.slug for p in personal.list_projects(fresh_db)]
    assert "live" in slugs
    assert "dead" not in slugs
