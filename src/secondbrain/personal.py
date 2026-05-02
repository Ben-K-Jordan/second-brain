"""Phase 79 + 80 + 81: personal context layers.

Three semi-related modules sharing nothing structural but all about
"the user as a tracked entity":

  - **Habits** — recurring intentions with daily/weekly check-ins.
    "You've journaled 23 days straight."
  - **Goals** — numeric weekly targets ("apply to 5 jobs/week"),
    progress events from anywhere (manual or hooked into Phase 47
    tasks for free-flowing tracking).
  - **Journal** — once-a-day prompt: 1-5 mood + sentence. Stored as
    structured rows AND as ``journal://YYYY-MM-DD`` docs so they
    flow through the same retrieval as everything else. Correlates
    with Oura (Phase 56).
  - **Projects** — explicit user-curated theme buckets. Tag a doc /
    task / person to a project; query "what's open in ML capstone?"

All four share one module because they're all "personal-context"
shaped: small, structured, mostly user-edited, and surface in the
daily brief / weekly review.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta

log = logging.getLogger(__name__)


# ---- Tunables ---------------------------------------------------------

# How many days of streak history to scan for status / surfacing.
# 365 covers the longest streaks the brief should ever celebrate.
_STREAK_LOOKBACK_DAYS = 365


# ============================ Phase 79: habits ========================

@dataclass
class Habit:
    id: int
    name: str
    cadence: str           # 'daily' | 'weekly' | 'N_per_week'
    target_per_week: int | None
    created_at: float
    archived_at: float | None


@dataclass
class HabitStatus:
    """Habit + computed signals: current streak, last 30d adherence."""
    habit: Habit
    current_streak_days: int = 0
    longest_streak_days: int = 0
    checkins_last_30d: int = 0
    expected_30d: int = 0           # how many we expected given the cadence
    last_checkin_date: str | None = None


def add_habit(
    conn: sqlite3.Connection, name: str,
    *, cadence: str = "daily", target_per_week: int | None = None,
) -> int:
    """Create a habit. Returns the new id, or the existing id if the
    name was already taken (idempotent)."""
    name = (name or "").strip()
    if not name:
        raise ValueError("name must be non-empty")
    if cadence not in ("daily", "weekly", "N_per_week"):
        raise ValueError(f"unknown cadence {cadence!r}")
    if cadence == "N_per_week" and (
        target_per_week is None or target_per_week < 1
    ):
        raise ValueError("N_per_week needs target_per_week >= 1")
    n = time.time()
    cur = conn.execute(
        "INSERT OR IGNORE INTO habits(name, cadence, target_per_week, created_at) "
        "VALUES (?, ?, ?, ?) RETURNING id",
        (name, cadence, target_per_week, n),
    )
    row = cur.fetchone()
    if row is not None:
        conn.commit()
        return int(row["id"])
    # Already existed.
    return int(conn.execute(
        "SELECT id FROM habits WHERE name = ?", (name,),
    ).fetchone()["id"])


def archive_habit(conn: sqlite3.Connection, habit_id: int) -> bool:
    cur = conn.execute(
        "UPDATE habits SET archived_at = ? WHERE id = ? AND archived_at IS NULL",
        (time.time(), habit_id),
    )
    conn.commit()
    return cur.rowcount > 0


def checkin(
    conn: sqlite3.Connection, habit_id: int,
    *, when: date | None = None, note: str | None = None,
) -> bool:
    """Record a check-in. Idempotent — same habit + same date is a
    no-op (not a duplicate). Returns True iff a new row landed."""
    d = (when or date.today()).isoformat()
    cur = conn.execute(
        "INSERT OR IGNORE INTO habit_checkins"
        "(habit_id, date, note, checked_at) "
        "VALUES (?, ?, ?, ?)",
        (habit_id, d, note, time.time()),
    )
    conn.commit()
    return cur.rowcount > 0


def list_habits(
    conn: sqlite3.Connection, *, include_archived: bool = False,
) -> list[Habit]:
    if include_archived:
        rows = conn.execute(
            "SELECT * FROM habits ORDER BY created_at ASC",
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM habits WHERE archived_at IS NULL "
            "ORDER BY created_at ASC",
        ).fetchall()
    return [_row_to_habit(r) for r in rows]


def habit_status(conn: sqlite3.Connection, habit_id: int) -> HabitStatus:
    """Compute streak + adherence for one habit."""
    row = conn.execute(
        "SELECT * FROM habits WHERE id = ?", (habit_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"habit #{habit_id} not found")
    habit = _row_to_habit(row)
    cutoff = (
        date.today() - timedelta(days=_STREAK_LOOKBACK_DAYS)
    ).isoformat()
    checkin_rows = conn.execute(
        "SELECT date FROM habit_checkins "
        "WHERE habit_id = ? AND date >= ? ORDER BY date ASC",
        (habit_id, cutoff),
    ).fetchall()
    dates = [r["date"] for r in checkin_rows]
    streak, longest = _compute_streaks(dates)
    last30_cutoff = (date.today() - timedelta(days=30)).isoformat()
    last30 = sum(1 for d in dates if d >= last30_cutoff)
    expected_30d = _expected_for_30d(habit)
    return HabitStatus(
        habit=habit,
        current_streak_days=streak,
        longest_streak_days=longest,
        checkins_last_30d=last30,
        expected_30d=expected_30d,
        last_checkin_date=dates[-1] if dates else None,
    )


def _compute_streaks(dates_asc: list[str]) -> tuple[int, int]:
    """Return (current, longest). dates_asc is ISO strings ascending.

    Current streak is the run of consecutive days ending today (or
    yesterday — gives the user grace for "haven't checked in yet
    today")."""
    if not dates_asc:
        return 0, 0
    parsed = [date.fromisoformat(d) for d in dates_asc]
    longest = 1
    run = 1
    for i in range(1, len(parsed)):
        if (parsed[i] - parsed[i - 1]).days == 1:
            run += 1
            longest = max(longest, run)
        else:
            run = 1
    today = date.today()
    last = parsed[-1]
    gap = (today - last).days
    if gap > 1:
        current = 0
    else:
        # Walk back from the end until non-consecutive.
        current = 1
        i = len(parsed) - 1
        while i > 0 and (parsed[i] - parsed[i - 1]).days == 1:
            current += 1
            i -= 1
    return current, longest


def _expected_for_30d(habit: Habit) -> int:
    if habit.cadence == "daily":
        return 30
    if habit.cadence == "weekly":
        return 4  # ~4.3 weeks/month, round down
    return int((habit.target_per_week or 1) * (30 / 7))


def _row_to_habit(row: sqlite3.Row) -> Habit:
    return Habit(
        id=int(row["id"]),
        name=row["name"],
        cadence=row["cadence"],
        target_per_week=row["target_per_week"],
        created_at=row["created_at"],
        archived_at=row["archived_at"],
    )


# ============================ Phase 79: goals =========================

@dataclass
class Goal:
    id: int
    name: str
    target_per_week: int | None
    description: str
    created_at: float
    achieved_at: float | None
    archived_at: float | None


@dataclass
class GoalStatus:
    goal: Goal
    progress_this_week: int = 0
    progress_last_30d: int = 0
    on_track: bool = True


def add_goal(
    conn: sqlite3.Connection, name: str,
    *, target_per_week: int | None = None, description: str = "",
) -> int:
    name = (name or "").strip()
    if not name:
        raise ValueError("name must be non-empty")
    n = time.time()
    cur = conn.execute(
        "INSERT OR IGNORE INTO goals"
        "(name, target_per_week, description, created_at) "
        "VALUES (?, ?, ?, ?) RETURNING id",
        (name, target_per_week, description, n),
    )
    row = cur.fetchone()
    if row:
        conn.commit()
        return int(row["id"])
    return int(conn.execute(
        "SELECT id FROM goals WHERE name = ?", (name,),
    ).fetchone()["id"])


def record_goal_progress(
    conn: sqlite3.Connection, goal_id: int,
    *, count: int = 1, note: str | None = None,
    when: date | None = None,
) -> int:
    """Append a progress event. Returns the new row id."""
    d = (when or date.today()).isoformat()
    cur = conn.execute(
        "INSERT INTO goal_progress(goal_id, date, count, note, recorded_at) "
        "VALUES (?, ?, ?, ?, ?) RETURNING id",
        (goal_id, d, count, note, time.time()),
    )
    rid = int(cur.fetchone()["id"])
    conn.commit()
    return rid


def list_goals(
    conn: sqlite3.Connection, *, include_archived: bool = False,
) -> list[Goal]:
    if include_archived:
        rows = conn.execute(
            "SELECT * FROM goals ORDER BY created_at ASC",
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM goals "
            "WHERE archived_at IS NULL AND achieved_at IS NULL "
            "ORDER BY created_at ASC",
        ).fetchall()
    return [_row_to_goal(r) for r in rows]


def goal_status(conn: sqlite3.Connection, goal_id: int) -> GoalStatus:
    row = conn.execute(
        "SELECT * FROM goals WHERE id = ?", (goal_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"goal #{goal_id} not found")
    goal = _row_to_goal(row)
    week_start = (
        date.today() - timedelta(days=date.today().weekday())
    ).isoformat()
    last30 = (date.today() - timedelta(days=30)).isoformat()
    week_n = conn.execute(
        "SELECT COALESCE(SUM(count), 0) AS n FROM goal_progress "
        "WHERE goal_id = ? AND date >= ?",
        (goal_id, week_start),
    ).fetchone()["n"] or 0
    last30_n = conn.execute(
        "SELECT COALESCE(SUM(count), 0) AS n FROM goal_progress "
        "WHERE goal_id = ? AND date >= ?",
        (goal_id, last30),
    ).fetchone()["n"] or 0
    on_track = (
        week_n >= (goal.target_per_week or 0)
        if goal.target_per_week else True
    )
    return GoalStatus(
        goal=goal,
        progress_this_week=int(week_n),
        progress_last_30d=int(last30_n),
        on_track=on_track,
    )


def _row_to_goal(row: sqlite3.Row) -> Goal:
    return Goal(
        id=int(row["id"]),
        name=row["name"],
        target_per_week=row["target_per_week"],
        description=row["description"] or "",
        created_at=row["created_at"],
        achieved_at=row["achieved_at"],
        archived_at=row["archived_at"],
    )


# ============================ Phase 80: journal =======================

@dataclass
class JournalEntry:
    id: int
    date: str
    mood: int | None
    text: str
    created_at: float
    updated_at: float


def upsert_journal(
    conn: sqlite3.Connection, *,
    when: date | None = None, mood: int | None = None,
    text: str = "",
) -> int:
    """Create-or-update today's entry. Mood is 1-5; clamp out-of-range
    silently. Returns the entry id."""
    d = (when or date.today()).isoformat()
    if mood is not None:
        mood = max(1, min(5, int(mood)))
    n = time.time()
    text = (text or "").strip()
    existing = conn.execute(
        "SELECT id FROM journal_entries WHERE date = ?", (d,),
    ).fetchone()
    if existing:
        # Update only the provided fields.
        if mood is not None and text:
            conn.execute(
                "UPDATE journal_entries SET mood = ?, text = ?, "
                "updated_at = ? WHERE id = ?",
                (mood, text, n, existing["id"]),
            )
        elif mood is not None:
            conn.execute(
                "UPDATE journal_entries SET mood = ?, updated_at = ? "
                "WHERE id = ?", (mood, n, existing["id"]),
            )
        elif text:
            conn.execute(
                "UPDATE journal_entries SET text = ?, updated_at = ? "
                "WHERE id = ?", (text, n, existing["id"]),
            )
        conn.commit()
        return int(existing["id"])
    cur = conn.execute(
        "INSERT INTO journal_entries(date, mood, text, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?) RETURNING id",
        (d, mood, text, n, n),
    )
    rid = int(cur.fetchone()["id"])
    conn.commit()
    return rid


def get_journal(
    conn: sqlite3.Connection, when: date | None = None,
) -> JournalEntry | None:
    d = (when or date.today()).isoformat()
    row = conn.execute(
        "SELECT * FROM journal_entries WHERE date = ?", (d,),
    ).fetchone()
    return _row_to_journal(row) if row else None


def recent_journal(
    conn: sqlite3.Connection, *, days: int = 14,
) -> list[JournalEntry]:
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        "SELECT * FROM journal_entries WHERE date >= ? "
        "ORDER BY date DESC",
        (cutoff,),
    ).fetchall()
    return [_row_to_journal(r) for r in rows]


def mood_correlation_with_metric(
    conn: sqlite3.Connection, metric: str = "sleep_score",
    *, days: int = 60,
) -> dict:
    """Cheap correlation between journal mood and an Oura metric.
    Returns ``{"n": int, "pearson_r": float, "mood_avg": float,
    "metric_avg": float}``. Returns empty dict if not enough data."""
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    rows = conn.execute(
        "SELECT j.date AS d, j.mood, h.value "
        "FROM journal_entries j "
        "JOIN health_metrics h ON h.date = j.date AND h.metric = ? "
        "WHERE j.mood IS NOT NULL AND j.date >= ?",
        (metric, cutoff),
    ).fetchall()
    if len(rows) < 5:
        return {}
    moods = [float(r["mood"]) for r in rows]
    metrics = [float(r["value"]) for r in rows]
    n = len(moods)
    mood_avg = sum(moods) / n
    metric_avg = sum(metrics) / n
    cov = sum(
        (moods[i] - mood_avg) * (metrics[i] - metric_avg)
        for i in range(n)
    ) / n
    var_m = sum((x - mood_avg) ** 2 for x in moods) / n
    var_h = sum((x - metric_avg) ** 2 for x in metrics) / n
    if var_m <= 0 or var_h <= 0:
        return {
            "n": n, "pearson_r": 0.0,
            "mood_avg": mood_avg, "metric_avg": metric_avg,
        }
    r = cov / (var_m ** 0.5 * var_h ** 0.5)
    return {
        "n": n, "pearson_r": r,
        "mood_avg": mood_avg, "metric_avg": metric_avg,
    }


def _row_to_journal(row: sqlite3.Row) -> JournalEntry:
    return JournalEntry(
        id=int(row["id"]),
        date=row["date"],
        mood=row["mood"],
        text=row["text"] or "",
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# ============================ Phase 81: projects ======================

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug_for(name: str) -> str:
    """Lowercase, hyphen-separated, ASCII-safe identifier."""
    return _SLUG_RE.sub("-", name.lower()).strip("-") or "untitled"


@dataclass
class Project:
    id: int
    slug: str
    name: str
    description: str
    status: str
    created_at: float
    archived_at: float | None


@dataclass
class ProjectView:
    """Project + member list ready for CLI / dashboard rendering."""
    project: Project
    files: list[tuple[int, str]]      # [(file_id, path), ...]
    tasks: list[tuple[int, str]]      # [(task_id, text), ...]
    people: list[tuple[int, str]]     # [(person_id, name), ...]


def create_project(
    conn: sqlite3.Connection, name: str,
    *, description: str = "",
) -> int:
    name = (name or "").strip()
    if not name:
        raise ValueError("name must be non-empty")
    slug = slug_for(name)
    n = time.time()
    cur = conn.execute(
        "INSERT OR IGNORE INTO projects(slug, name, description, created_at) "
        "VALUES (?, ?, ?, ?) RETURNING id",
        (slug, name, description, n),
    )
    row = cur.fetchone()
    if row is None:
        # Existing project — return its id.
        return int(conn.execute(
            "SELECT id FROM projects WHERE slug = ?", (slug,),
        ).fetchone()["id"])
    conn.commit()
    return int(row["id"])


def get_project_by_slug(
    conn: sqlite3.Connection, slug: str,
) -> Project | None:
    row = conn.execute(
        "SELECT * FROM projects WHERE slug = ?", (slug,),
    ).fetchone()
    return _row_to_project(row) if row else None


def list_projects(
    conn: sqlite3.Connection, *, include_archived: bool = False,
) -> list[Project]:
    if include_archived:
        rows = conn.execute(
            "SELECT * FROM projects ORDER BY created_at ASC",
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM projects WHERE archived_at IS NULL "
            "ORDER BY created_at ASC",
        ).fetchall()
    return [_row_to_project(r) for r in rows]


def add_to_project(
    conn: sqlite3.Connection, project_id: int,
    *, kind: str, ref_id: int, note: str | None = None,
) -> bool:
    """Tag a file/task/person to a project. Idempotent — same triple
    is a no-op."""
    if kind not in ("file", "task", "person"):
        raise ValueError(f"unknown kind {kind!r}")
    cur = conn.execute(
        "INSERT OR IGNORE INTO project_items"
        "(project_id, kind, ref_id, added_at, note) "
        "VALUES (?, ?, ?, ?, ?)",
        (project_id, kind, ref_id, time.time(), note),
    )
    conn.commit()
    return cur.rowcount > 0


def remove_from_project(
    conn: sqlite3.Connection, project_id: int,
    *, kind: str, ref_id: int,
) -> bool:
    cur = conn.execute(
        "DELETE FROM project_items "
        "WHERE project_id = ? AND kind = ? AND ref_id = ?",
        (project_id, kind, ref_id),
    )
    conn.commit()
    return cur.rowcount > 0


def project_view(
    conn: sqlite3.Connection, project_id: int,
) -> ProjectView | None:
    row = conn.execute(
        "SELECT * FROM projects WHERE id = ?", (project_id,),
    ).fetchone()
    if row is None:
        return None
    project = _row_to_project(row)
    items = conn.execute(
        "SELECT kind, ref_id FROM project_items "
        "WHERE project_id = ? ORDER BY added_at ASC",
        (project_id,),
    ).fetchall()
    files: list[tuple[int, str]] = []
    tasks: list[tuple[int, str]] = []
    people: list[tuple[int, str]] = []
    by_kind: dict[str, list[int]] = defaultdict(list)
    for it in items:
        by_kind[it["kind"]].append(int(it["ref_id"]))
    if by_kind.get("file"):
        f_rows = conn.execute(
            f"SELECT id, path FROM files "
            f"WHERE id IN ({','.join('?' * len(by_kind['file']))})",
            by_kind["file"],
        ).fetchall()
        files = [(int(r["id"]), r["path"]) for r in f_rows]
    if by_kind.get("task"):
        try:
            t_rows = conn.execute(
                f"SELECT id, text FROM tasks "
                f"WHERE id IN ({','.join('?' * len(by_kind['task']))})",
                by_kind["task"],
            ).fetchall()
            tasks = [(int(r["id"]), r["text"]) for r in t_rows]
        except sqlite3.OperationalError:
            pass
    if by_kind.get("person"):
        try:
            p_rows = conn.execute(
                f"SELECT id, display_name FROM people "
                f"WHERE id IN ({','.join('?' * len(by_kind['person']))})",
                by_kind["person"],
            ).fetchall()
            people = [(int(r["id"]), r["display_name"]) for r in p_rows]
        except sqlite3.OperationalError:
            pass
    return ProjectView(
        project=project, files=files, tasks=tasks, people=people,
    )


def _row_to_project(row: sqlite3.Row) -> Project:
    return Project(
        id=int(row["id"]),
        slug=row["slug"],
        name=row["name"],
        description=row["description"] or "",
        status=row["status"],
        created_at=row["created_at"],
        archived_at=row["archived_at"],
    )


# ============================ Phase 80 indexer hook ==================

def index_journal_entry(
    cfg, conn: sqlite3.Connection, embedder, entry: JournalEntry,
) -> str | None:
    """Persist a journal entry as a ``journal://YYYY-MM-DD`` doc so
    it flows through the normal retrieval pipeline. Idempotent —
    re-indexing the same date overwrites."""
    from .indexer import index_text
    body = (
        f"# Journal — {entry.date}\n\n"
        f"Mood: {entry.mood if entry.mood else '—'}/5\n\n"
        f"{entry.text}"
    )
    vp = f"journal://{entry.date}"
    try:
        index_text(
            conn, embedder, cfg,
            virtual_path=vp,
            title=f"Journal {entry.date}",
            content=body,
            mtime=entry.updated_at,
            kind="journal",
            source="journal",
        )
    except Exception as e:  # noqa: BLE001
        log.warning("personal: journal index failed: %s", e)
        return None
    return vp
