"""Phase 56: health metrics — querying + ingestion side-table.

Connectors that produce numeric biometric data (Oura today; Apple
Health / Garmin / Whoop later) write structured values to
``health_metrics`` so the dashboard / CLI can plot trends without
re-parsing doc bodies.

This module owns:

- ``ingest_summaries(conn, summaries, source)``: bulk write daily
  summaries (from the Oura connector) into the metrics table. Called
  from ``cli.sync`` after the Oura connector runs.
- ``recent(conn, metric, days)``: time-series for one metric over a
  rolling window. Used by ``health show <metric>`` and the dashboard.
- ``average(conn, metric, days)`` / ``trend(conn, metric, days)``:
  cheap aggregates for the daily brief (Phase 44 + this phase).

Design principle: keep this module *thin*. The doc text in the index
is the source of truth for retrieval/chat. The metrics table is just
a fast structured cache for "give me my last 14 days of sleep scores."
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass

log = logging.getLogger(__name__)


# ---- Data shapes ------------------------------------------------------

@dataclass
class MetricPoint:
    """One row in ``health_metrics`` — date + value."""
    date: str
    value: float


@dataclass
class MetricSummary:
    """Aggregate stats for a metric over a window."""
    metric: str
    days: int
    n: int
    average: float | None
    minimum: float | None
    maximum: float | None
    latest: MetricPoint | None


# ---- Ingest -----------------------------------------------------------

def ingest_summaries(
    conn: sqlite3.Connection,
    summaries: Iterable,  # Iterable[oura.DailySummary]
    *,
    source: str = "oura",
) -> int:
    """Persist every numeric field on each ``DailySummary`` to
    ``health_metrics``. Returns the count of (date, metric) rows
    written/updated.

    Uses ``INSERT ... ON CONFLICT DO UPDATE`` so re-running the sync
    refreshes today's row when the day isn't over yet (Oura updates
    its scores throughout the day).
    """
    n_written = 0
    now = time.time()
    for ds in summaries:
        # Map of metric name → numeric value (or None to skip).
        fields = {
            "sleep_score": ds.sleep_score,
            "total_sleep_seconds": ds.total_sleep_seconds,
            "rem_seconds": ds.rem_seconds,
            "deep_seconds": ds.deep_seconds,
            "light_seconds": ds.light_seconds,
            "efficiency": ds.efficiency,
            "avg_hrv": ds.avg_hrv,
            "avg_resting_hr": ds.avg_resting_hr,
            "activity_score": ds.activity_score,
            "steps": ds.steps,
            "active_calories": ds.active_calories,
            "total_calories": ds.total_calories,
            "readiness_score": ds.readiness_score,
            "temperature_deviation": ds.temperature_deviation,
            "recovery_index": ds.recovery_index,
        }
        for metric, value in fields.items():
            if value is None:
                continue
            conn.execute(
                "INSERT INTO health_metrics"
                "(date, metric, value, source, recorded_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(date, metric, source) DO UPDATE SET "
                "  value = excluded.value, "
                "  recorded_at = excluded.recorded_at",
                (ds.date, metric, float(value), source, now),
            )
            n_written += 1
    conn.commit()
    return n_written


# ---- Querying ---------------------------------------------------------

def recent(
    conn: sqlite3.Connection,
    metric: str,
    *,
    days: int = 14,
    source: str = "oura",
) -> list[MetricPoint]:
    """Return the last ``days`` of values for ``metric``, oldest first.

    Useful shape for plotting + correlations. We rely on lexicographic
    ordering of YYYY-MM-DD strings for "last N days" — works because
    ISO dates sort the same as their actual chronology.
    """
    rows = conn.execute(
        "SELECT date, value FROM health_metrics "
        "WHERE metric = ? AND source = ? "
        "ORDER BY date DESC LIMIT ?",
        (metric, source, days),
    ).fetchall()
    # Reverse for chronological order — lets a plot read left-to-right
    # without the caller flipping it.
    return [MetricPoint(date=r["date"], value=r["value"])
            for r in reversed(rows)]


def summarise(
    conn: sqlite3.Connection,
    metric: str,
    *,
    days: int = 14,
    source: str = "oura",
) -> MetricSummary:
    """One-shot stats: mean / min / max / latest for a metric.

    Returns a ``MetricSummary`` with all-``None`` aggregates when no
    data exists in the window, so callers can render ``"—"`` without
    branching.
    """
    points = recent(conn, metric, days=days, source=source)
    if not points:
        return MetricSummary(
            metric=metric, days=days, n=0,
            average=None, minimum=None, maximum=None, latest=None,
        )
    vals = [p.value for p in points]
    return MetricSummary(
        metric=metric,
        days=days,
        n=len(points),
        average=sum(vals) / len(vals),
        minimum=min(vals),
        maximum=max(vals),
        latest=points[-1],
    )


def list_metrics(
    conn: sqlite3.Connection, source: str = "oura",
) -> list[str]:
    """Distinct metric names that the brain has data for. Used by
    ``health show`` when called without args."""
    rows = conn.execute(
        "SELECT DISTINCT metric FROM health_metrics WHERE source = ? "
        "ORDER BY metric",
        (source,),
    ).fetchall()
    return [r["metric"] for r in rows]


# ---- Rendering --------------------------------------------------------

def format_summary_line(summary: MetricSummary) -> str:
    """One-liner for CLI / brief output."""
    if summary.n == 0:
        return f"{summary.metric}: no data in last {summary.days} days"
    avg_s = (
        f"{summary.average:.1f}"
        if summary.average is not None and summary.average != int(summary.average)
        else f"{int(summary.average) if summary.average is not None else '—'}"
    )
    latest = summary.latest
    latest_s = (
        f"{int(latest.value) if latest.value == int(latest.value) else latest.value:.1f}"
    ) if latest else "—"
    return (
        f"{summary.metric}: avg {avg_s} over {summary.n} day(s) "
        f"(latest {latest_s} on {latest.date if latest else '?'})"
    )
