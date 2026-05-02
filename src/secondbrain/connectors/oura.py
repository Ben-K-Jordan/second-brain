"""Oura ring connector — sleep, activity, readiness, workouts, tags.

Pulls from Oura Cloud API v2 with a Personal Access Token. Each day
becomes one ``ConnectorDocument`` rolling up sleep + activity +
readiness scores plus any workouts and user-added tags. The chat agent
can answer "how was my sleep last week?" / "did I work out Tuesday?"
using the same hybrid retrieval as everything else.

A side-effect of ingestion: structured numeric values land in the
``health_metrics`` table (date, metric, value), so a CLI / dashboard
can plot trends or correlate scores with events without re-parsing
the doc bodies.

Setup:
  1. Visit https://cloud.ouraring.com/personal-access-tokens
  2. Create a new token (any name, no expiry).
  3. Either:

       secondbrain auth oura

     (saves to ~/.secondbrain/oura_credentials.json)

     OR set::

       OURA_TOKEN=<your token>

Env wins over the saved creds — same precedence rule as the Canvas
connector.

What's ingested per day:
  - Sleep score, total sleep, REM / deep / light, efficiency, HRV
  - Activity score, steps, calories
  - Readiness score, temperature deviation, recovery index
  - Workouts (any logged via Oura or auto-detected from HR)
  - User-added tags / journal entries from the Oura app

Network failures degrade gracefully — a single endpoint failing only
loses that section's data for the affected day(s).

API reference: https://cloud.ouraring.com/v2/docs
"""

from __future__ import annotations

import json as _json
import logging
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path as _Path

import requests

from ..config import Config
from . import USER_AGENT, ConnectorDocument

log = logging.getLogger(__name__)


# ---- Tunables ---------------------------------------------------------

_API_BASE = "https://api.ouraring.com/v2"
_TIMEOUT = 30
_DEFAULT_WINDOW_DAYS = 90


# ---- Auth -------------------------------------------------------------

def _credentials_path(cfg: Config) -> _Path:
    """Where `secondbrain auth oura` writes the saved token."""
    return cfg.data_dir / "oura_credentials.json"


def save_oura_credentials(cfg: Config, token: str) -> None:
    """Persist a token to a 0600 file in the data dir."""
    path = _credentials_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        _json.dump({"token": token.strip()}, f)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def load_oura_credentials(cfg: Config) -> str | None:
    """Read the saved token, or None if missing/corrupt."""
    path = _credentials_path(cfg)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = _json.load(f)
    except (OSError, _json.JSONDecodeError):
        return None
    tok = (data.get("token") or "").strip()
    return tok or None


def verify_oura_token(token: str) -> dict | None:
    """Probe `/usercollection/personal_info` to confirm the token works.

    Returns the user-info JSON on success, None on failure (bad token,
    network error). Cheap one-shot GET.
    """
    try:
        r = requests.get(
            f"{_API_BASE}/usercollection/personal_info",
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            },
            timeout=15,
        )
    except requests.RequestException as e:
        log.warning("Oura token verify failed: %s", type(e).__name__)
        return None
    if r.status_code != 200:
        log.warning("Oura token verify HTTP %s", r.status_code)
        return None
    try:
        return r.json()
    except ValueError:
        return None


def _resolve_token(cfg: Config | None = None) -> str | None:
    """Env wins over saved creds — same precedence as Canvas."""
    tok = (os.environ.get("OURA_TOKEN") or "").strip()
    if tok:
        return tok
    if cfg is not None:
        return load_oura_credentials(cfg)
    return None


# ---- Per-day shape ----------------------------------------------------

@dataclass
class DailySummary:
    """All Oura signals for one calendar day, merged across endpoints."""
    date: str                     # 'YYYY-MM-DD'
    sleep_score: int | None = None
    total_sleep_seconds: int | None = None
    rem_seconds: int | None = None
    deep_seconds: int | None = None
    light_seconds: int | None = None
    efficiency: int | None = None
    avg_hrv: int | None = None
    avg_resting_hr: int | None = None
    bedtime_start: str = ""
    bedtime_end: str = ""

    activity_score: int | None = None
    steps: int | None = None
    active_calories: int | None = None
    total_calories: int | None = None

    readiness_score: int | None = None
    temperature_deviation: float | None = None
    recovery_index: int | None = None

    workouts: list[dict] = field(default_factory=list)  # raw Oura workout dicts
    tags: list[dict] = field(default_factory=list)      # raw Oura tag dicts


# ---- API plumbing -----------------------------------------------------

def _fetch_paginated(
    s: requests.Session, url: str, params: dict,
) -> Iterator[dict]:
    """Yield ``data`` items across Oura's ``next_token`` pagination.

    Most days fit in a single page; the loop is here so longer windows
    work cleanly. Bails on non-200 responses with a logged warning so
    one bad endpoint doesn't tear down the whole sync.
    """
    next_token: str | None = None
    while True:
        p = dict(params)
        if next_token:
            p["next_token"] = next_token
        try:
            r = s.get(url, params=p, timeout=_TIMEOUT)
        except requests.RequestException as e:
            log.warning("oura: GET %s failed: %s", url, type(e).__name__)
            return
        if r.status_code == 401:
            log.warning("oura: 401 — token rejected for %s", url)
            return
        if r.status_code != 200:
            log.warning("oura: HTTP %s for %s", r.status_code, url)
            return
        try:
            payload = r.json()
        except ValueError:
            log.warning("oura: non-JSON response for %s", url)
            return
        yield from payload.get("data") or []
        next_token = payload.get("next_token")
        if not next_token:
            return


def _date_range(window_days: int) -> tuple[str, str]:
    """Inclusive [start, end] in YYYY-MM-DD covering the last N days
    (Oura's API uses calendar dates, not timestamps)."""
    today = datetime.now(tz=UTC).date()
    start = today - timedelta(days=max(1, int(window_days)))
    return start.isoformat(), today.isoformat()


# ---- Connector --------------------------------------------------------

class OuraConnector:
    name = "oura"

    def is_enabled(self, cfg: Config) -> bool:
        return _resolve_token(cfg) is not None

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]:
        token = _resolve_token(cfg)
        if not token:
            return
        window = int(getattr(cfg, "oura_window_days", _DEFAULT_WINDOW_DAYS))
        start, end = _date_range(window)
        s = requests.Session()
        s.headers.update({
            "Authorization": f"Bearer {token}",
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        })
        try:
            summaries = self._collect_summaries(s, start, end)
        finally:
            s.close()
        for date_str in sorted(summaries.keys()):
            doc = self._render_day(summaries[date_str])
            if doc is not None:
                yield doc

    # ---- collection ---------------------------------------------------

    def _collect_summaries(
        self, s: requests.Session, start: str, end: str,
    ) -> dict[str, DailySummary]:
        """Hit each endpoint, merge into a per-date map."""
        out: dict[str, DailySummary] = {}

        def _ds(date: str) -> DailySummary:
            if date not in out:
                out[date] = DailySummary(date=date)
            return out[date]

        # Sleep score + period stats. The /sleep endpoint returns one
        # record per sleep period (naps included); we merge by `day`.
        for item in _fetch_paginated(
            s, f"{_API_BASE}/usercollection/sleep",
            params={"start_date": start, "end_date": end},
        ):
            day = item.get("day")
            if not day:
                continue
            ds = _ds(day)
            # Use the longest period as the "main" sleep — Oura ranks
            # them, but length is a robust default.
            total = item.get("total_sleep_duration") or 0
            if (ds.total_sleep_seconds or 0) < total:
                ds.total_sleep_seconds = total
                ds.rem_seconds = item.get("rem_sleep_duration") or 0
                ds.deep_seconds = item.get("deep_sleep_duration") or 0
                ds.light_seconds = item.get("light_sleep_duration") or 0
                ds.efficiency = item.get("efficiency")
                ds.avg_hrv = item.get("average_hrv")
                ds.avg_resting_hr = item.get("average_heart_rate")
                ds.bedtime_start = item.get("bedtime_start") or ""
                ds.bedtime_end = item.get("bedtime_end") or ""

        # Daily sleep score (separate endpoint, just the score number).
        for item in _fetch_paginated(
            s, f"{_API_BASE}/usercollection/daily_sleep",
            params={"start_date": start, "end_date": end},
        ):
            day = item.get("day")
            if not day:
                continue
            _ds(day).sleep_score = item.get("score")

        # Daily activity.
        for item in _fetch_paginated(
            s, f"{_API_BASE}/usercollection/daily_activity",
            params={"start_date": start, "end_date": end},
        ):
            day = item.get("day")
            if not day:
                continue
            ds = _ds(day)
            ds.activity_score = item.get("score")
            ds.steps = item.get("steps")
            ds.active_calories = item.get("active_calories")
            ds.total_calories = item.get("total_calories")

        # Daily readiness.
        for item in _fetch_paginated(
            s, f"{_API_BASE}/usercollection/daily_readiness",
            params={"start_date": start, "end_date": end},
        ):
            day = item.get("day")
            if not day:
                continue
            ds = _ds(day)
            ds.readiness_score = item.get("score")
            ds.temperature_deviation = item.get("temperature_deviation")
            # recovery_index lives under contributors{} on the readiness
            # payload — pull it out if present.
            contributors = item.get("contributors") or {}
            if "recovery_index" in contributors:
                ds.recovery_index = contributors["recovery_index"]

        # Workouts (one record per logged session).
        for item in _fetch_paginated(
            s, f"{_API_BASE}/usercollection/workout",
            params={"start_date": start, "end_date": end},
        ):
            day = item.get("day")
            if not day:
                continue
            _ds(day).workouts.append(item)

        # Enhanced tags (user notes / mood / etc. from the Oura app).
        # The endpoint is gated; if it 404s for a user, we just skip.
        for item in _fetch_paginated(
            s, f"{_API_BASE}/usercollection/enhanced_tag",
            params={"start_date": start, "end_date": end},
        ):
            day = item.get("day")
            if not day:
                continue
            _ds(day).tags.append(item)

        return out

    # ---- rendering ----------------------------------------------------

    def _render_day(self, ds: DailySummary) -> ConnectorDocument | None:
        """Render a daily summary as a ConnectorDocument.

        Skips days that came back from the API but have no actual
        signal (rare, but happens when you wear the ring under 1h).
        Title front-loads the three big scores so search results are
        scannable: ``[health] 2026-04-15 sleep 87 / readiness 82 /
        activity 91``.
        """
        if (ds.sleep_score is None and ds.activity_score is None
                and ds.readiness_score is None and not ds.workouts
                and not ds.tags):
            return None

        title_bits: list[str] = []
        if ds.sleep_score is not None:
            title_bits.append(f"sleep {ds.sleep_score}")
        if ds.readiness_score is not None:
            title_bits.append(f"readiness {ds.readiness_score}")
        if ds.activity_score is not None:
            title_bits.append(f"activity {ds.activity_score}")
        title = (
            f"[health] {ds.date} " + " / ".join(title_bits)
            if title_bits else f"[health] {ds.date}"
        )

        lines: list[str] = [
            f"# {title}", "",
            f"Date: {ds.date}",
        ]
        if ds.sleep_score is not None or ds.total_sleep_seconds:
            lines.append("")
            lines.append("## Sleep")
            if ds.sleep_score is not None:
                lines.append(f"Score: {ds.sleep_score}")
            if ds.total_sleep_seconds:
                lines.append(
                    f"Total: {ds.total_sleep_seconds // 3600}h "
                    f"{(ds.total_sleep_seconds % 3600) // 60}m",
                )
            if ds.rem_seconds:
                lines.append(f"REM: {ds.rem_seconds // 60} min")
            if ds.deep_seconds:
                lines.append(f"Deep: {ds.deep_seconds // 60} min")
            if ds.light_seconds:
                lines.append(f"Light: {ds.light_seconds // 60} min")
            if ds.efficiency is not None:
                lines.append(f"Efficiency: {ds.efficiency}%")
            if ds.avg_hrv is not None:
                lines.append(f"Avg HRV: {ds.avg_hrv} ms")
            if ds.avg_resting_hr is not None:
                lines.append(f"Avg HR: {ds.avg_resting_hr} bpm")
            if ds.bedtime_start:
                lines.append(f"Bedtime: {ds.bedtime_start} → {ds.bedtime_end}")

        if ds.readiness_score is not None:
            lines.append("")
            lines.append("## Readiness")
            lines.append(f"Score: {ds.readiness_score}")
            if ds.temperature_deviation is not None:
                lines.append(
                    f"Temp deviation: {ds.temperature_deviation:+.2f}°C",
                )
            if ds.recovery_index is not None:
                lines.append(f"Recovery index: {ds.recovery_index}")

        if ds.activity_score is not None or ds.steps:
            lines.append("")
            lines.append("## Activity")
            if ds.activity_score is not None:
                lines.append(f"Score: {ds.activity_score}")
            if ds.steps is not None:
                lines.append(f"Steps: {ds.steps:,}")
            if ds.active_calories is not None:
                lines.append(f"Active cal: {ds.active_calories}")
            if ds.total_calories is not None:
                lines.append(f"Total cal: {ds.total_calories}")

        if ds.workouts:
            lines.append("")
            lines.append("## Workouts")
            for w in ds.workouts:
                act = w.get("activity") or w.get("classification") or "workout"
                start = w.get("start_datetime") or ""
                dur_s = int(w.get("duration") or 0)
                cal = w.get("calories")
                bits = [act]
                if dur_s:
                    bits.append(f"{dur_s // 60} min")
                if cal:
                    bits.append(f"{cal} cal")
                if start:
                    bits.append(f"@ {start[11:16]}")  # HH:MM
                lines.append(f"- {' · '.join(bits)}")

        if ds.tags:
            lines.append("")
            lines.append("## Tags / journal")
            for t in ds.tags:
                # Enhanced tags have a free-text 'comment' field plus
                # 'tag_type_code' (mood/diet/etc).
                comment = (t.get("comment") or "").strip()
                tag_type = t.get("tag_type_code") or t.get("custom_name") or ""
                if comment:
                    lines.append(f"- [{tag_type}] {comment}" if tag_type
                                 else f"- {comment}")
                elif tag_type:
                    lines.append(f"- {tag_type}")

        # mtime: 12:00 UTC of the day so time-decay surfaces yesterday
        # higher than last week. Using midnight would tie multiple days
        # at the same epoch when the system clock crosses TZ.
        try:
            d = datetime.fromisoformat(ds.date).replace(
                hour=12, tzinfo=UTC,
            )
            mtime = d.timestamp()
        except ValueError:
            mtime = time.time()

        return ConnectorDocument(
            source="oura",
            virtual_path=f"oura://daily/{ds.date}",
            title=title,
            content="\n".join(lines),
            mtime=mtime,
            kind="url",
            metadata={
                "kind": "health_daily",
                "date": ds.date,
                "sleep_score": ds.sleep_score,
                "readiness_score": ds.readiness_score,
                "activity_score": ds.activity_score,
                "steps": ds.steps,
                "total_sleep_seconds": ds.total_sleep_seconds,
                "avg_hrv": ds.avg_hrv,
                "avg_resting_hr": ds.avg_resting_hr,
                "temperature_deviation": ds.temperature_deviation,
                "workouts": len(ds.workouts),
                "tags": len(ds.tags),
            },
        )


# ---- Direct-export helpers (for tests + health.py) -------------------

def fetch_summaries(
    cfg: Config, *, window_days: int | None = None,
) -> list[DailySummary]:
    """Library entry point — call the connector and return raw daily
    summary dataclasses (rather than ConnectorDocuments). Used by
    ``health.py`` to populate the ``health_metrics`` table.
    """
    token = _resolve_token(cfg)
    if not token:
        return []
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {token}",
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    })
    w = (
        int(window_days) if window_days is not None
        else int(getattr(cfg, "oura_window_days", _DEFAULT_WINDOW_DAYS))
    )
    start, end = _date_range(w)
    try:
        summaries = OuraConnector()._collect_summaries(s, start, end)
    finally:
        s.close()
    return [summaries[d] for d in sorted(summaries.keys())]
