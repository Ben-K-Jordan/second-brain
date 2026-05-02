"""Phase 56: Oura connector + health metrics tests.

The connector hits a network API. Tests stub the HTTP layer with
``requests_mock``-style monkeypatching against a fake session so we
don't make real Oura calls (and so the test suite doesn't need a
token). The data shapes here mirror Oura API v2's actual responses.

Coverage:
  - Auth precedence: env > saved creds > unset
  - Token-credential save / load round-trip
  - Per-endpoint merge into ``DailySummary``
  - Doc rendering: title, score lines, workouts, tags
  - Skip-empty-day rule
  - ``health.ingest_summaries`` round-trip + idempotence
  - ``health.recent`` / ``health.summarise`` with empty-and-populated DBs
  - Sparkline edge cases
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from secondbrain import health
from secondbrain.connectors import oura

# ============================ auth ====================================

def test_auth_env_var_wins(monkeypatch, tmp_cfg):
    """OURA_TOKEN env var should override any saved credentials file."""
    oura.save_oura_credentials(tmp_cfg, "saved-token")
    monkeypatch.setenv("OURA_TOKEN", "env-token")
    assert oura._resolve_token(tmp_cfg) == "env-token"


def test_auth_saved_creds_used_when_no_env(tmp_cfg, monkeypatch):
    monkeypatch.delenv("OURA_TOKEN", raising=False)
    oura.save_oura_credentials(tmp_cfg, "saved-token")
    assert oura._resolve_token(tmp_cfg) == "saved-token"


def test_auth_returns_none_without_either(tmp_cfg, monkeypatch):
    monkeypatch.delenv("OURA_TOKEN", raising=False)
    assert oura._resolve_token(tmp_cfg) is None


def test_save_oura_credentials_writes_token(tmp_cfg):
    oura.save_oura_credentials(tmp_cfg, "abc123  ")
    path = oura._credentials_path(tmp_cfg)
    assert path.exists()
    with open(path) as f:
        data = json.load(f)
    assert data["token"] == "abc123"


def test_load_oura_credentials_returns_none_on_corrupt_file(tmp_cfg):
    path = oura._credentials_path(tmp_cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert oura.load_oura_credentials(tmp_cfg) is None


def test_load_oura_credentials_returns_none_when_missing(tmp_cfg):
    assert oura.load_oura_credentials(tmp_cfg) is None


def test_is_enabled_reflects_token_presence(tmp_cfg, monkeypatch):
    monkeypatch.delenv("OURA_TOKEN", raising=False)
    c = oura.OuraConnector()
    assert c.is_enabled(tmp_cfg) is False
    oura.save_oura_credentials(tmp_cfg, "x")
    assert c.is_enabled(tmp_cfg) is True


# ============================ rendering ===============================

def test_render_day_skips_when_no_signal():
    """Days with literally no scores or workouts shouldn't materialise.
    The Oura API can emit empty rows for days you didn't wear the ring."""
    ds = oura.DailySummary(date="2026-04-15")
    assert oura.OuraConnector()._render_day(ds) is None


def test_render_day_title_includes_three_scores():
    ds = oura.DailySummary(
        date="2026-04-15",
        sleep_score=87, readiness_score=82, activity_score=91,
    )
    doc = oura.OuraConnector()._render_day(ds)
    assert doc is not None
    assert "[health] 2026-04-15" in doc.title
    assert "sleep 87" in doc.title
    assert "readiness 82" in doc.title
    assert "activity 91" in doc.title


def test_render_day_handles_partial_data():
    """Activity-only days (e.g. ring not worn at night) should still render."""
    ds = oura.DailySummary(
        date="2026-04-15",
        activity_score=88, steps=12000,
    )
    doc = oura.OuraConnector()._render_day(ds)
    assert doc is not None
    assert "activity 88" in doc.title
    assert "12,000" in doc.content  # comma-formatted steps
    assert "## Sleep" not in doc.content
    assert "## Readiness" not in doc.content


def test_render_day_includes_sleep_breakdown():
    ds = oura.DailySummary(
        date="2026-04-15",
        sleep_score=87,
        total_sleep_seconds=8 * 3600,  # 8h
        rem_seconds=90 * 60,
        deep_seconds=75 * 60,
        light_seconds=4 * 3600,
        efficiency=92,
        avg_hrv=45,
        avg_resting_hr=58,
        bedtime_start="2026-04-14T23:30:00+00:00",
        bedtime_end="2026-04-15T07:30:00+00:00",
    )
    doc = oura.OuraConnector()._render_day(ds)
    assert "## Sleep" in doc.content
    assert "Total: 8h 0m" in doc.content
    assert "REM: 90 min" in doc.content
    assert "Avg HRV: 45 ms" in doc.content
    assert "Avg HR: 58 bpm" in doc.content
    assert "Bedtime: " in doc.content


def test_render_day_includes_workouts():
    ds = oura.DailySummary(
        date="2026-04-15",
        activity_score=88,
        workouts=[
            {"activity": "running", "duration": 45 * 60, "calories": 380,
             "start_datetime": "2026-04-15T06:30:00+00:00"},
            {"activity": "yoga", "duration": 30 * 60, "calories": 95,
             "start_datetime": "2026-04-15T18:00:00+00:00"},
        ],
    )
    doc = oura.OuraConnector()._render_day(ds)
    assert "## Workouts" in doc.content
    assert "running" in doc.content
    assert "45 min" in doc.content
    assert "380 cal" in doc.content
    assert "yoga" in doc.content
    # Time formatting (HH:MM) makes day-of-week patterns visible.
    assert "@ 06:30" in doc.content
    assert "@ 18:00" in doc.content


def test_render_day_includes_user_tags():
    ds = oura.DailySummary(
        date="2026-04-15",
        sleep_score=70,
        tags=[
            {"comment": "felt tired all day", "tag_type_code": "mood"},
            {"comment": "two espressos before bed", "tag_type_code": "diet"},
        ],
    )
    doc = oura.OuraConnector()._render_day(ds)
    assert "## Tags / journal" in doc.content
    assert "felt tired all day" in doc.content
    assert "two espressos" in doc.content


def test_render_day_metadata_includes_scores():
    """The dashboard / chat agent should be able to filter by score
    without parsing the body — store key numerics in metadata."""
    ds = oura.DailySummary(
        date="2026-04-15",
        sleep_score=87, readiness_score=82, activity_score=91,
        steps=12000, avg_hrv=45,
    )
    doc = oura.OuraConnector()._render_day(ds)
    assert doc.metadata["sleep_score"] == 87
    assert doc.metadata["readiness_score"] == 82
    assert doc.metadata["activity_score"] == 91
    assert doc.metadata["steps"] == 12000
    assert doc.metadata["avg_hrv"] == 45
    assert doc.metadata["date"] == "2026-04-15"


def test_render_day_uses_noon_utc_mtime():
    """mtime should be 12:00 UTC of the day so time-decay buckets
    cleanly into "yesterday vs last week" without TZ jitter."""
    from datetime import UTC, datetime
    ds = oura.DailySummary(date="2026-04-15", sleep_score=80)
    doc = oura.OuraConnector()._render_day(ds)
    expected = datetime(2026, 4, 15, 12, 0, 0, tzinfo=UTC).timestamp()
    assert abs(doc.mtime - expected) < 1.0


def test_render_day_title_no_scores_just_workouts():
    """Workout-only days (rare but possible) still render."""
    ds = oura.DailySummary(
        date="2026-04-15",
        workouts=[{"activity": "walk", "duration": 15 * 60}],
    )
    doc = oura.OuraConnector()._render_day(ds)
    assert doc is not None
    # Title degrades gracefully — bare date prefix.
    assert doc.title.startswith("[health] 2026-04-15")


# ============================ health.py ===============================

def test_ingest_summaries_writes_metrics(fresh_db):
    summaries = [
        oura.DailySummary(
            date="2026-04-15", sleep_score=87, readiness_score=82,
            activity_score=91, steps=12000, avg_hrv=45,
        ),
    ]
    n = health.ingest_summaries(fresh_db, summaries)
    assert n == 5  # five non-None numerics
    rows = fresh_db.execute(
        "SELECT metric, value FROM health_metrics WHERE date = ?",
        ("2026-04-15",),
    ).fetchall()
    metrics = {r["metric"]: r["value"] for r in rows}
    assert metrics["sleep_score"] == 87.0
    assert metrics["steps"] == 12000.0


def test_ingest_summaries_skips_none_fields(fresh_db):
    """Don't write rows for missing values — saves clutter and lets
    the dashboard show '—' for unknowns."""
    summaries = [oura.DailySummary(date="2026-04-15", sleep_score=87)]
    n = health.ingest_summaries(fresh_db, summaries)
    assert n == 1
    metrics = {
        r["metric"] for r in fresh_db.execute(
            "SELECT metric FROM health_metrics WHERE date = ?",
            ("2026-04-15",),
        )
    }
    assert metrics == {"sleep_score"}


def test_ingest_summaries_idempotent_updates_value(fresh_db):
    """Re-ingest should UPDATE, not duplicate. Oura updates today's
    score throughout the day so we want a refresh, not a 2nd row."""
    health.ingest_summaries(
        fresh_db,
        [oura.DailySummary(date="2026-04-15", sleep_score=70)],
    )
    health.ingest_summaries(
        fresh_db,
        [oura.DailySummary(date="2026-04-15", sleep_score=87)],
    )
    rows = fresh_db.execute(
        "SELECT value FROM health_metrics WHERE date = ? AND metric = 'sleep_score'",
        ("2026-04-15",),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["value"] == 87.0


def test_recent_returns_chronological_order(fresh_db):
    summaries = [
        oura.DailySummary(date="2026-04-13", sleep_score=70),
        oura.DailySummary(date="2026-04-14", sleep_score=80),
        oura.DailySummary(date="2026-04-15", sleep_score=85),
    ]
    health.ingest_summaries(fresh_db, summaries)
    points = health.recent(fresh_db, "sleep_score", days=7)
    # Oldest first — easy to plot left-to-right.
    assert [p.date for p in points] == ["2026-04-13", "2026-04-14", "2026-04-15"]
    assert [p.value for p in points] == [70.0, 80.0, 85.0]


def test_recent_respects_limit(fresh_db):
    summaries = [
        oura.DailySummary(date=f"2026-04-{d:02d}", sleep_score=70 + d)
        for d in range(1, 16)
    ]
    health.ingest_summaries(fresh_db, summaries)
    points = health.recent(fresh_db, "sleep_score", days=5)
    assert len(points) == 5


def test_summarise_handles_empty_db(fresh_db):
    s = health.summarise(fresh_db, "sleep_score", days=14)
    assert s.n == 0
    assert s.average is None
    assert s.latest is None


def test_summarise_computes_aggregates(fresh_db):
    summaries = [
        oura.DailySummary(date="2026-04-13", sleep_score=70),
        oura.DailySummary(date="2026-04-14", sleep_score=80),
        oura.DailySummary(date="2026-04-15", sleep_score=90),
    ]
    health.ingest_summaries(fresh_db, summaries)
    s = health.summarise(fresh_db, "sleep_score", days=14)
    assert s.n == 3
    assert s.average == 80.0
    assert s.minimum == 70.0
    assert s.maximum == 90.0
    assert s.latest.date == "2026-04-15"
    assert s.latest.value == 90.0


def test_list_metrics_returns_distinct(fresh_db):
    summaries = [
        oura.DailySummary(
            date="2026-04-15", sleep_score=87, activity_score=91,
        ),
        oura.DailySummary(
            date="2026-04-16", sleep_score=88, activity_score=89,
        ),
    ]
    health.ingest_summaries(fresh_db, summaries)
    metrics = set(health.list_metrics(fresh_db))
    assert metrics == {"sleep_score", "activity_score"}


# ============================ collection ==============================

class FakeSession:
    """Minimal stand-in for ``requests.Session`` exposing only the
    surface Oura's client uses: ``get(url, params, timeout)`` returning
    a ``FakeResponse`` with ``status_code`` + ``json()``.

    Routes are dict-keyed by URL substring.
    """

    def __init__(self, routes: dict[str, dict]):
        self.routes = routes
        self.headers: dict[str, str] = {}

    def get(self, url, params=None, timeout=None):
        for substr, payload in self.routes.items():
            if substr in url:
                return FakeResponse(payload)
        return FakeResponse({"data": []})

    def close(self):
        pass


@dataclass
class FakeResponse:
    payload: dict
    status_code: int = 200

    def json(self):
        return self.payload


def test_collect_summaries_merges_endpoints(monkeypatch):
    """Each Oura endpoint contributes a slice; the connector merges
    them by ``day``. Verify a single test day combines fields from
    sleep/daily_sleep/daily_activity/daily_readiness."""
    routes = {
        "/usercollection/sleep": {"data": [{
            "day": "2026-04-15",
            "total_sleep_duration": 28800,
            "rem_sleep_duration": 5400,
            "deep_sleep_duration": 4500,
            "light_sleep_duration": 14400,
            "efficiency": 92,
            "average_hrv": 45,
            "average_heart_rate": 58,
            "bedtime_start": "2026-04-14T23:30:00+00:00",
            "bedtime_end": "2026-04-15T07:30:00+00:00",
        }], "next_token": None},
        "/usercollection/daily_sleep": {"data": [{
            "day": "2026-04-15", "score": 87,
        }]},
        "/usercollection/daily_activity": {"data": [{
            "day": "2026-04-15", "score": 91, "steps": 12000,
            "active_calories": 450, "total_calories": 2400,
        }]},
        "/usercollection/daily_readiness": {"data": [{
            "day": "2026-04-15", "score": 82,
            "temperature_deviation": -0.1,
            "contributors": {"recovery_index": 88},
        }]},
        "/usercollection/workout": {"data": []},
        "/usercollection/enhanced_tag": {"data": []},
    }
    s = FakeSession(routes)
    out = oura.OuraConnector()._collect_summaries(
        s, "2026-04-15", "2026-04-15",
    )
    ds = out["2026-04-15"]
    assert ds.sleep_score == 87
    assert ds.total_sleep_seconds == 28800
    assert ds.activity_score == 91
    assert ds.steps == 12000
    assert ds.readiness_score == 82
    assert ds.recovery_index == 88
    assert abs(ds.temperature_deviation - (-0.1)) < 1e-9


def test_collect_summaries_keeps_longest_sleep_period():
    """When the user has a main sleep + a nap, we keep the longer
    period's stats — Oura's ``score`` already reflects the night."""
    routes = {
        "/usercollection/sleep": {"data": [
            {"day": "2026-04-15", "total_sleep_duration": 1800,  # nap
             "average_hrv": 30},
            {"day": "2026-04-15", "total_sleep_duration": 28800,  # main sleep
             "average_hrv": 50},
        ]},
        "/usercollection/daily_sleep": {"data": []},
        "/usercollection/daily_activity": {"data": []},
        "/usercollection/daily_readiness": {"data": []},
        "/usercollection/workout": {"data": []},
        "/usercollection/enhanced_tag": {"data": []},
    }
    s = FakeSession(routes)
    out = oura.OuraConnector()._collect_summaries(
        s, "2026-04-15", "2026-04-15",
    )
    ds = out["2026-04-15"]
    assert ds.total_sleep_seconds == 28800
    assert ds.avg_hrv == 50  # took stats from the longer period


def test_collect_summaries_swallows_endpoint_failure(monkeypatch):
    """A 401 on one endpoint shouldn't kill the rest of the sync."""
    class BadSession(FakeSession):
        def get(self, url, params=None, timeout=None):
            if "daily_readiness" in url:
                return FakeResponse({}, status_code=401)
            return super().get(url, params, timeout)

    routes = {
        "/usercollection/sleep": {"data": []},
        "/usercollection/daily_sleep": {"data": [{
            "day": "2026-04-15", "score": 87,
        }]},
        "/usercollection/daily_activity": {"data": []},
        "/usercollection/workout": {"data": []},
        "/usercollection/enhanced_tag": {"data": []},
    }
    s = BadSession(routes)
    out = oura.OuraConnector()._collect_summaries(
        s, "2026-04-15", "2026-04-15",
    )
    # daily_sleep still landed; readiness silently skipped.
    assert out["2026-04-15"].sleep_score == 87
    assert out["2026-04-15"].readiness_score is None


def test_collect_summaries_handles_empty_response():
    """No data → no summaries. No crash."""
    s = FakeSession({})
    out = oura.OuraConnector()._collect_summaries(
        s, "2026-04-15", "2026-04-15",
    )
    assert out == {}


# ============================ sparkline ===============================

def test_sparkline_handles_constant_series():
    """All-equal data should render as the middle level (or any level
    really) without dividing by zero."""
    from secondbrain.cli import _ascii_sparkline
    out = _ascii_sparkline([5.0, 5.0, 5.0, 5.0])
    assert len(out) == 4
    # Don't crash; render some non-empty character.
    assert all(ch != "" for ch in out)


def test_sparkline_orders_low_to_high():
    from secondbrain.cli import _ascii_sparkline
    out = _ascii_sparkline([0.0, 25.0, 50.0, 75.0, 100.0])
    # Strictly non-decreasing visually — convert glyphs back to indices.
    blocks = " ▁▂▃▄▅▆▇█"
    indices = [blocks.index(ch) for ch in out]
    assert indices == sorted(indices)


def test_sparkline_empty_returns_empty():
    from secondbrain.cli import _ascii_sparkline
    assert _ascii_sparkline([]) == ""
