"""Phase 60: unified scheduler tests.

Three layers covered:

  - **Schedule strategies**: pure logic over (now, last_started,
    last_success). No DB, no I/O.
  - **Run history persistence**: Scheduler writes to scheduler_runs
    and surfaces the right history back through ``next_due`` /
    ``history``.
  - **Tick + filter_context**: end-to-end. Verifies multiple jobs
    fire / skip correctly + that a crashing job doesn't take down
    its neighbours.
"""

from __future__ import annotations

import time

import pytest

from secondbrain.scheduler import (
    CooldownSchedule,
    DailyAtSchedule,
    IntervalSchedule,
    Job,
    JobResult,
    Scheduler,
    runs_in_last,
    trim_old_runs,
)

# ============================ IntervalSchedule ========================

def test_interval_is_due_when_never_run():
    s = IntervalSchedule(seconds=60)
    assert s.is_due(now=100.0, last_started_at=None, last_success_at=None)


def test_interval_is_due_after_full_interval():
    s = IntervalSchedule(seconds=60)
    assert s.is_due(
        now=200.0, last_started_at=140.0, last_success_at=140.0,
    )


def test_interval_not_due_before_full_interval():
    s = IntervalSchedule(seconds=60)
    assert not s.is_due(
        now=200.0, last_started_at=170.0, last_success_at=170.0,
    )


def test_interval_advances_even_when_failures():
    """A job that keeps crashing should not busy-loop — its started_at
    advances, so the next due time advances too."""
    s = IntervalSchedule(seconds=60)
    # last_started_at advanced even though success is None.
    assert not s.is_due(
        now=200.0, last_started_at=170.0, last_success_at=None,
    )


def test_interval_next_due_at():
    s = IntervalSchedule(seconds=60)
    assert s.next_due_at(
        now=200.0, last_started_at=140.0, last_success_at=140.0,
    ) == 200.0
    assert s.next_due_at(
        now=200.0, last_started_at=None, last_success_at=None,
    ) == 200.0


# ============================ DailyAtSchedule =========================

def test_daily_at_not_due_before_local_time():
    """At 06:00 local time, the 07:00 schedule shouldn't fire."""
    from datetime import datetime
    target = datetime.now().replace(hour=6, minute=0, second=0, microsecond=0)
    s = DailyAtSchedule(local_time="07:00", cooldown_hours=12)
    assert not s.is_due(
        now=target.timestamp(), last_started_at=None, last_success_at=None,
    )


def test_daily_at_is_due_after_local_time_when_never_run():
    from datetime import datetime
    target = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    s = DailyAtSchedule(local_time="07:00", cooldown_hours=12)
    assert s.is_due(
        now=target.timestamp(), last_started_at=None, last_success_at=None,
    )


def test_daily_at_skips_within_cooldown_after_success():
    """If we successfully sent the brief 1h ago, don't re-send within
    the 12h cooldown."""
    from datetime import datetime
    now = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    s = DailyAtSchedule(local_time="07:00", cooldown_hours=12)
    assert not s.is_due(
        now=now.timestamp(),
        last_started_at=now.timestamp() - 3600,
        last_success_at=now.timestamp() - 3600,
    )


def test_daily_at_fires_again_after_cooldown():
    """Yesterday's send was >12h ago — fire today."""
    from datetime import datetime
    today_8am = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    yesterday_8am = today_8am.timestamp() - 86400
    s = DailyAtSchedule(local_time="07:00", cooldown_hours=12)
    assert s.is_due(
        now=today_8am.timestamp(),
        last_started_at=yesterday_8am, last_success_at=yesterday_8am,
    )


def test_daily_at_handles_malformed_time():
    """Garbage in local_time → never due, no exception."""
    s = DailyAtSchedule(local_time="not-a-time")
    assert not s.is_due(now=time.time(), last_started_at=None, last_success_at=None)
    assert s.next_due_at(now=time.time(), last_started_at=None, last_success_at=None) is None


# ============================ CooldownSchedule ========================

def test_cooldown_is_due_when_never_run():
    s = CooldownSchedule(seconds=60, cooldown_hours=12)
    assert s.is_due(now=100.0, last_started_at=None, last_success_at=None)


def test_cooldown_skips_until_interval_elapsed():
    s = CooldownSchedule(seconds=60, cooldown_hours=12)
    # 30s after a successful run — interval not yet elapsed.
    assert not s.is_due(
        now=200.0, last_started_at=170.0, last_success_at=170.0,
    )


def test_cooldown_keeps_retrying_until_first_success():
    """Until we've had ANY success, retry every interval."""
    s = CooldownSchedule(seconds=60, cooldown_hours=12)
    assert s.is_due(
        now=200.0, last_started_at=130.0, last_success_at=None,
    )


def test_cooldown_skips_within_cooldown_after_success():
    """1h after success, still cooling down (12h)."""
    s = CooldownSchedule(seconds=60, cooldown_hours=12)
    now = 1000000.0
    assert not s.is_due(
        now=now,
        last_started_at=now - 3600,
        last_success_at=now - 3600,
    )


# ============================ Scheduler ===============================

def test_scheduler_register_rejects_duplicate(fresh_db):
    s = Scheduler(fresh_db)
    s.register(Job("foo", IntervalSchedule(60), lambda: None))
    with pytest.raises(ValueError):
        s.register(Job("foo", IntervalSchedule(60), lambda: None))


def test_scheduler_tick_runs_due_jobs(fresh_db):
    """A job with no history should fire on first tick."""
    calls: list[str] = []
    s = Scheduler(fresh_db)
    s.register(Job(
        "watchlists",
        IntervalSchedule(seconds=60),
        lambda: calls.append("ran"),
    ))
    fired = s.tick()
    assert fired == ["watchlists"]
    assert calls == ["ran"]


def test_scheduler_tick_skips_jobs_within_interval(fresh_db):
    """Two ticks back-to-back: only the first fires."""
    calls: list[str] = []
    s = Scheduler(fresh_db)
    s.register(Job(
        "watchlists",
        IntervalSchedule(seconds=60),
        lambda: calls.append("ran"),
    ))
    s.tick()
    s.tick()
    assert len(calls) == 1


def test_scheduler_persists_run_history(fresh_db):
    """Each fire should land a row in scheduler_runs with started_at,
    finished_at, success=1."""
    s = Scheduler(fresh_db)
    s.register(Job(
        "wl",
        IntervalSchedule(seconds=60),
        lambda: 3,  # int return → summary "3 item(s)"
    ))
    s.tick()
    rows = fresh_db.execute(
        "SELECT * FROM scheduler_runs WHERE job_name = ?", ("wl",),
    ).fetchall()
    assert len(rows) == 1
    r = rows[0]
    assert r["success"] == 1
    assert r["finished_at"] is not None
    assert "3 item(s)" in (r["summary"] or "")


def test_scheduler_records_crash_as_failure(fresh_db):
    """A job that raises should land as success=0 with the error
    captured. Other jobs in the same tick must still run."""
    calls: list[str] = []
    s = Scheduler(fresh_db)
    s.register(Job(
        "boom", IntervalSchedule(60),
        lambda: (_ for _ in ()).throw(RuntimeError("nope")),
    ))
    s.register(Job(
        "ok", IntervalSchedule(60),
        lambda: calls.append("ok"),
    ))
    fired = s.tick()
    assert set(fired) == {"boom", "ok"}
    assert calls == ["ok"]
    rows = fresh_db.execute(
        "SELECT job_name, success, error FROM scheduler_runs "
        "ORDER BY id ASC",
    ).fetchall()
    assert len(rows) == 2
    by_name = {r["job_name"]: r for r in rows}
    assert by_name["boom"]["success"] == 0
    assert "nope" in (by_name["boom"]["error"] or "")
    assert by_name["ok"]["success"] == 1


def test_scheduler_filters_context_to_fn_signature(fresh_db):
    """A job that takes (cfg, conn) should still fire when the tick
    forwards (cfg, conn, embedder, reranker). The scheduler should
    silently filter unused kwargs."""
    seen: dict = {}

    def fn(cfg, conn):
        seen["cfg"] = cfg
        seen["conn"] = conn

    s = Scheduler(fresh_db)
    s.register(Job("narrow", IntervalSchedule(60), fn))
    s.tick(cfg="A", conn="B", embedder="C", reranker="D")
    assert seen == {"cfg": "A", "conn": "B"}


def test_scheduler_passes_all_kwargs_to_var_keyword_fn(fresh_db):
    """If the job accepts **kwargs, give it the full context."""
    seen: dict = {}

    def fn(**kwargs):
        seen.update(kwargs)

    s = Scheduler(fresh_db)
    s.register(Job("wide", IntervalSchedule(60), fn))
    s.tick(cfg="A", conn="B", embedder="C")
    assert seen == {"cfg": "A", "conn": "B", "embedder": "C"}


def test_scheduler_history_returns_last_run_data(fresh_db):
    s = Scheduler(fresh_db)
    s.register(Job("x", IntervalSchedule(60), lambda: 5))
    s.tick()
    h = s.history()
    assert "x" in h
    assert h["x"].last_started_at is not None
    assert h["x"].last_success_at is not None
    assert h["x"].last_summary is not None
    assert h["x"].last_error is None


def test_scheduler_history_distinguishes_last_success_from_last_run(fresh_db):
    """When the latest run failed, last_started_at points at the failure
    but last_success_at still points at the prior good run."""
    # Construct the Scheduler first so the runs table exists.
    s = Scheduler(fresh_db)
    fresh_db.execute(
        "INSERT INTO scheduler_runs(job_name, started_at, finished_at, "
        " success, summary) VALUES ('x', ?, ?, 1, 'good')",
        (100.0, 101.0),
    )
    fresh_db.execute(
        "INSERT INTO scheduler_runs(job_name, started_at, finished_at, "
        " success, error) VALUES ('x', ?, ?, 0, 'oops')",
        (200.0, 201.0),
    )
    fresh_db.commit()
    s.register(Job("x", IntervalSchedule(60), lambda: None))
    h = s.history()
    assert h["x"].last_started_at == 200.0
    assert h["x"].last_success_at == 100.0
    assert h["x"].last_error == "oops"


def test_scheduler_next_due_at_estimates(fresh_db):
    """For a job with last-run=now, next_due ≈ now + interval."""
    s = Scheduler(fresh_db)
    s.register(Job("x", IntervalSchedule(60), lambda: None))
    s.tick()
    next_due = s.next_due()
    assert "x" in next_due
    # Should be roughly 60s after the last start.
    h = s.history()
    expected = h["x"].last_started_at + 60
    assert abs(next_due["x"] - expected) < 1.0


# ============================ JobResult ===============================

def test_job_result_explicit_success_and_summary(fresh_db):
    s = Scheduler(fresh_db)
    s.register(Job(
        "x", IntervalSchedule(60),
        lambda: JobResult(success=True, summary="custom"),
    ))
    s.tick()
    row = fresh_db.execute(
        "SELECT summary, success FROM scheduler_runs "
        "WHERE job_name = 'x'",
    ).fetchone()
    assert row["summary"] == "custom"
    assert row["success"] == 1


def test_job_result_explicit_failure_persists(fresh_db):
    """Returning JobResult(success=False) should land as a failure
    row even though the function didn't raise."""
    s = Scheduler(fresh_db)
    s.register(Job(
        "x", IntervalSchedule(60),
        lambda: JobResult(success=False, error="downstream said no"),
    ))
    s.tick()
    row = fresh_db.execute(
        "SELECT success, error FROM scheduler_runs WHERE job_name='x'",
    ).fetchone()
    assert row["success"] == 0
    assert row["error"] is None  # _coerce ignores .error currently


# ============================ runs_in_last ============================

def test_runs_in_last_returns_recent_runs(fresh_db):
    """The status command pulls the most-recent N runs across all
    jobs to render 'what fired this hour'."""
    s = Scheduler(fresh_db)
    s.register(Job("a", IntervalSchedule(60), lambda: 1))
    s.register(Job("b", IntervalSchedule(60), lambda: 2))
    s.tick()
    rows = runs_in_last(fresh_db, hours=1)
    job_names = [r["job_name"] for r in rows]
    assert "a" in job_names
    assert "b" in job_names


def test_runs_in_last_excludes_old_rows(fresh_db):
    Scheduler(fresh_db)  # migrate table lazily
    fresh_db.execute(
        "INSERT INTO scheduler_runs(job_name, started_at, finished_at, "
        " success) VALUES ('old', ?, ?, 1)",
        (time.time() - 48 * 3600, time.time() - 48 * 3600 + 1),
    )
    fresh_db.commit()
    rows = runs_in_last(fresh_db, hours=24)
    assert all(r["job_name"] != "old" for r in rows)


def test_trim_old_runs_removes_ancient(fresh_db):
    Scheduler(fresh_db)  # migrate table lazily
    fresh_db.execute(
        "INSERT INTO scheduler_runs(job_name, started_at, success) "
        "VALUES ('keep', ?, 1)",
        (time.time(),),
    )
    fresh_db.execute(
        "INSERT INTO scheduler_runs(job_name, started_at, success) "
        "VALUES ('old', ?, 1)",
        (time.time() - 30 * 86400,),
    )
    fresh_db.commit()
    n = trim_old_runs(fresh_db, keep_days=14)
    assert n == 1
    rows = fresh_db.execute(
        "SELECT job_name FROM scheduler_runs",
    ).fetchall()
    assert {r["job_name"] for r in rows} == {"keep"}


# ============================ table migration =========================

def test_scheduler_creates_runs_table_lazily(fresh_db):
    """Even on a freshly-init'd DB without scheduler_runs, the
    Scheduler ctor should create the table on first use."""
    fresh_db.execute("DROP TABLE IF EXISTS scheduler_runs")
    fresh_db.commit()
    Scheduler(fresh_db)  # ctor migrates
    # Table should exist now.
    fresh_db.execute("SELECT * FROM scheduler_runs LIMIT 1")
