"""Phase 60: unified scheduler — one place that decides what runs when.

Before this module, the daemon's `while True` loop hand-rolled six
different "if now - last_X >= INTERVAL" checks for watchlists, digest,
brief, oura, event briefings, queue summariser. Each had its own
cooldown logic, its own catch-and-log error handling, its own
not-quite-the-same persistence pattern. Adding a seventh job meant
copying the boilerplate.

This module replaces that with a `Scheduler` that owns:

  - **Job registry**: each job has a name + a schedule + a callable.
    The schedule encapsulates "is it due?" so the daemon loop becomes
    one line: `scheduler.tick()`.
  - **Run history**: every fire persists to ``scheduler_runs`` with
    started/finished/success/error/result so the dashboard + CLI
    `status` can answer 'what fired in the last 24h, when, and how
    much did it cost?' without grovelling through log files.
  - **Introspection**: ``next_due_at(name)`` and ``last_run(name)``
    so a status command can render ETA + last-success without each
    feature reinventing the persistence shape.

Schedule shapes:

  - ``IntervalSchedule(seconds=N)`` — runs every N seconds. The default
    for fast pollers (watchlists, event briefings, queue summariser).
  - ``DailyAtSchedule(local_time="HH:MM", cooldown_hours=12)`` — at
    most once per day, after the local-time threshold. Cooldown
    prevents double-fire when a daemon restart crosses the boundary.
  - ``CooldownSchedule(seconds=N, cooldown_hours=12)`` — runs at most
    once per cooldown window; the interval is the *poll cadence*. Used
    for "sync this thing roughly daily" jobs (Oura).

All schedules are cooperative: nothing pre-empts a running job. SQLite
serialises writes anyway, so concurrent jobs would just contend; one
at a time keeps things deterministic.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

log = logging.getLogger(__name__)


# ---- Schedule strategies ---------------------------------------------

class Schedule(Protocol):
    """A schedule decides whether a job is due to run *now*, given
    when it last started + last succeeded. Implementations are pure
    functions of clock + history — no DB access, no side effects."""

    def is_due(
        self,
        now: float,
        last_started_at: float | None,
        last_success_at: float | None,
    ) -> bool: ...

    def next_due_at(
        self,
        now: float,
        last_started_at: float | None,
        last_success_at: float | None,
    ) -> float | None:
        """Best-effort estimate of the next epoch second this job will
        be due. Returns None when the schedule has no predictable next
        time (e.g. event-driven). Used by ``status`` for ETA display."""
        ...


@dataclass(frozen=True)
class IntervalSchedule:
    """Run every ``seconds`` seconds. The simplest schedule — fires
    at startup (when there's no last-run record), then every interval
    after.

    Job runs that fail are still recorded with their started_at, so
    the next due time advances regardless of success — i.e. a broken
    job doesn't busy-loop trying to recover."""
    seconds: float

    def is_due(self, now, last_started_at, last_success_at) -> bool:
        if last_started_at is None:
            return True
        return (now - last_started_at) >= self.seconds

    def next_due_at(self, now, last_started_at, last_success_at):
        if last_started_at is None:
            return now
        return last_started_at + self.seconds


@dataclass(frozen=True)
class DailyAtSchedule:
    """Run at most once per cooldown window, only after a local-time
    threshold has passed today. Used for the daily digest + daily
    brief — they're 'send once after morning'.

    The local_time field is HH:MM; we compare to ``datetime.now()``
    in the user's local zone (not UTC) so 'send at 7am' means 7am
    where the user is, not in UTC.
    """
    local_time: str = "07:00"
    cooldown_hours: float = 12.0

    def is_due(self, now, last_started_at, last_success_at) -> bool:
        target_today = self._target_today(now)
        if target_today is None:
            return False  # malformed local_time: never due
        if now < target_today:
            return False
        # Past today's threshold — but did we already fire recently?
        gate = last_success_at if last_success_at is not None else last_started_at
        return not (gate is not None
                    and (now - gate) < self.cooldown_hours * 3600)

    def next_due_at(self, now, last_started_at, last_success_at):
        target_today = self._target_today(now)
        if target_today is None:
            return None
        gate = last_success_at if last_success_at is not None else last_started_at
        if gate is not None:
            ready_after = gate + self.cooldown_hours * 3600
            if ready_after > target_today:
                # Cooldown extends past today's target — next fire is
                # tomorrow's target, or whichever is later.
                tomorrow_target = target_today + 86400
                return max(tomorrow_target, ready_after)
        if now < target_today:
            return target_today
        return now  # past target, off cooldown → ready now

    def _target_today(self, now: float) -> float | None:
        try:
            hh, mm = self.local_time.split(":")
            target = datetime.fromtimestamp(now).replace(
                hour=int(hh), minute=int(mm),
                second=0, microsecond=0,
            )
            return target.timestamp()
        except (ValueError, AttributeError):
            return None


@dataclass(frozen=True)
class CooldownSchedule:
    """Run at most once per cooldown window. Polls at ``seconds``
    interval but only fires when the cooldown since last success
    has elapsed. Used for 'roughly once a day, no specific time'
    jobs like Oura sync."""
    seconds: float
    cooldown_hours: float

    def is_due(self, now, last_started_at, last_success_at) -> bool:
        if last_started_at is None:
            return True
        if (now - last_started_at) < self.seconds:
            return False
        if last_success_at is None:
            return True  # never succeeded, keep retrying
        return (now - last_success_at) >= self.cooldown_hours * 3600

    def next_due_at(self, now, last_started_at, last_success_at):
        if last_started_at is None:
            return now
        ready_after_interval = last_started_at + self.seconds
        if last_success_at is None:
            return ready_after_interval
        ready_after_cooldown = last_success_at + self.cooldown_hours * 3600
        return max(ready_after_interval, ready_after_cooldown)


# ---- Job + result -----------------------------------------------------

@dataclass
class JobResult:
    """What a job hands back to the scheduler. Most callers just return
    None / int / bool and the scheduler coerces — this dataclass is
    for the cases where you want to surface a custom summary string."""
    success: bool = True
    summary: str | None = None
    error: str | None = None


@dataclass
class Job:
    name: str
    schedule: Schedule
    fn: Callable[..., Any]
    # When True, log INFO every successful run. When False, only log
    # when the job did meaningful work (fn returned truthy / non-zero).
    verbose: bool = False
    # Tags for cost-bucket attribution (Phase 63 hookup).
    tags: tuple[str, ...] = field(default_factory=tuple)


# ---- Scheduler --------------------------------------------------------

class Scheduler:
    """Owns the registry + tick loop + run history.

    Usage:

        sched = Scheduler(conn)
        sched.register(Job("watchlists", IntervalSchedule(60), run_due_watchlists))
        ...
        while not stopping:
            time.sleep(1)
            sched.tick(context={"cfg": cfg, "conn": conn, ...})
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._jobs: dict[str, Job] = {}
        _ensure_runs_table(conn)

    # ---- Registry ----

    def register(self, job: Job) -> None:
        if job.name in self._jobs:
            raise ValueError(f"Job {job.name!r} already registered")
        self._jobs[job.name] = job
        log.debug("scheduler: registered job %r (%s)",
                  job.name, type(job.schedule).__name__)

    def names(self) -> list[str]:
        return list(self._jobs.keys())

    # ---- Tick ----

    def tick(self, **context: Any) -> list[str]:
        """Run every job whose schedule says it's due. Returns the
        list of job names that fired this tick.

        Each job receives ``**context`` as keyword arguments. The
        scheduler only forwards kwargs the job's signature accepts,
        so jobs can be lean (``def watchlist(cfg, conn): ...``) without
        having to take ``**_kwargs``.

        Errors are caught + persisted to ``scheduler_runs`` — one job
        crashing never stops the others.
        """
        fired: list[str] = []
        now = time.time()
        for name, job in self._jobs.items():
            history = self._history_for(name)
            if not job.schedule.is_due(
                now, history.last_started_at, history.last_success_at,
            ):
                continue
            fired.append(name)
            self._run_one(job, now, context)
            now = time.time()  # let other jobs see the post-run clock
        return fired

    def _run_one(
        self, job: Job, started_at: float, context: dict[str, Any],
    ) -> None:
        kwargs = _filter_context(job.fn, context)
        run_id = _insert_run_started(self._conn, job.name, started_at)
        try:
            result = job.fn(**kwargs)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            log.warning("scheduler: job %r crashed: %s", job.name, err)
            _finish_run(
                self._conn, run_id,
                success=False, summary=None, error=err,
            )
            return
        success, summary = _coerce_result(result)
        _finish_run(
            self._conn, run_id,
            success=success, summary=summary, error=None,
        )
        if job.verbose or summary:
            log.info(
                "scheduler: job %r ran (success=%s, %s)",
                job.name, success, summary or "no summary",
            )

    # ---- Introspection ----

    def history(self) -> dict[str, JobHistory]:
        """Most-recent run snapshot for each registered job."""
        return {name: self._history_for(name) for name in self._jobs}

    def next_due(self) -> dict[str, float | None]:
        """Best-effort next-due epoch for each job (None when unknown)."""
        now = time.time()
        out: dict[str, float | None] = {}
        for name, job in self._jobs.items():
            h = self._history_for(name)
            out[name] = job.schedule.next_due_at(
                now, h.last_started_at, h.last_success_at,
            )
        return out

    def _history_for(self, name: str) -> JobHistory:
        row_started = self._conn.execute(
            "SELECT started_at, finished_at, success, summary, error "
            "FROM scheduler_runs WHERE job_name = ? "
            "ORDER BY started_at DESC LIMIT 1",
            (name,),
        ).fetchone()
        row_success = self._conn.execute(
            "SELECT started_at, finished_at "
            "FROM scheduler_runs WHERE job_name = ? AND success = 1 "
            "ORDER BY started_at DESC LIMIT 1",
            (name,),
        ).fetchone()
        return JobHistory(
            last_started_at=(
                row_started["started_at"] if row_started else None
            ),
            last_finished_at=(
                row_started["finished_at"] if row_started else None
            ),
            last_success_at=(
                row_success["started_at"] if row_success else None
            ),
            last_success_finished_at=(
                row_success["finished_at"] if row_success else None
            ),
            last_summary=(row_started["summary"] if row_started else None),
            last_error=(row_started["error"] if row_started else None),
        )


@dataclass
class JobHistory:
    """A frozen snapshot of a job's recent runs. Built from
    scheduler_runs; consumed by the scheduler itself + status command."""
    last_started_at: float | None = None
    last_finished_at: float | None = None
    last_success_at: float | None = None
    last_success_finished_at: float | None = None
    last_summary: str | None = None
    last_error: str | None = None


# ---- Persistence helpers ----------------------------------------------

def _ensure_runs_table(conn: sqlite3.Connection) -> None:
    """Lazy-migrate the runs table on first scheduler instantiation."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS scheduler_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_name TEXT NOT NULL,
            started_at REAL NOT NULL,
            finished_at REAL,
            success INTEGER,                 -- NULL while running, 0/1 after
            summary TEXT,
            error TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_sched_runs_job_started
            ON scheduler_runs(job_name, started_at DESC);
        CREATE INDEX IF NOT EXISTS idx_sched_runs_job_success_started
            ON scheduler_runs(job_name, success, started_at DESC);
    """)
    conn.commit()


def _insert_run_started(
    conn: sqlite3.Connection, job_name: str, started_at: float,
) -> int:
    cur = conn.execute(
        "INSERT INTO scheduler_runs(job_name, started_at) "
        "VALUES (?, ?) RETURNING id",
        (job_name, started_at),
    )
    rid = cur.fetchone()["id"]
    conn.commit()
    return int(rid)


def _finish_run(
    conn: sqlite3.Connection, run_id: int, *,
    success: bool, summary: str | None, error: str | None,
) -> None:
    conn.execute(
        "UPDATE scheduler_runs SET "
        "  finished_at = ?, success = ?, summary = ?, error = ? "
        "WHERE id = ?",
        (time.time(), 1 if success else 0, summary, error, run_id),
    )
    conn.commit()


def _coerce_result(result: Any) -> tuple[bool, str | None]:
    """Translate whatever a job returns into (success, summary).

    - JobResult → use its fields directly
    - bool → success-only result, no summary
    - int → success + summary like '3 items'
    - None → silent success
    - anything else → str(result) as summary
    """
    if isinstance(result, JobResult):
        return result.success, result.summary
    if result is None:
        return True, None
    if isinstance(result, bool):
        return result, None
    if isinstance(result, int):
        return True, f"{result} item(s)" if result else None
    return True, str(result)[:200]


def _filter_context(fn: Callable[..., Any], context: dict) -> dict:
    """Filter a context dict to just the kwargs ``fn`` actually accepts.

    The daemon passes a richer context (cfg, conn, embedder, reranker)
    than most jobs need. Without this filter every job would have to
    declare ``**_kwargs`` to be tolerant.
    """
    import inspect
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        # Builtins / C functions don't have introspectable sigs.
        return context
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return context
    accepted = {k for k in params}
    return {k: v for k, v in context.items() if k in accepted}


# ---- Reporting helpers (used by `secondbrain status`) ----------------

def runs_in_last(
    conn: sqlite3.Connection, hours: float = 24.0, limit: int = 200,
) -> list[dict]:
    """Recent runs across all jobs, newest first."""
    cutoff = time.time() - hours * 3600
    rows = conn.execute(
        "SELECT id, job_name, started_at, finished_at, success, "
        "       summary, error "
        "FROM scheduler_runs WHERE started_at >= ? "
        "ORDER BY started_at DESC LIMIT ?",
        (cutoff, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def trim_old_runs(conn: sqlite3.Connection, keep_days: int = 14) -> int:
    """Garbage-collect ancient runs so the table stays bounded. Called
    periodically by the daemon (registered as its own scheduled job).
    Returns count deleted."""
    cutoff = time.time() - keep_days * 86400
    cur = conn.execute(
        "DELETE FROM scheduler_runs WHERE started_at < ?", (cutoff,),
    )
    conn.commit()
    return cur.rowcount or 0
