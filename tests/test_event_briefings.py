"""Phase 38: pre-event briefings — schema, prompt builder, scheduler."""

from __future__ import annotations

import json
import time
from dataclasses import replace

import pytest

from secondbrain.db import (
    event_briefing_get,
    event_briefing_save,
    event_briefings_list,
    event_briefings_upcoming,
)
from secondbrain.event_briefing import (
    CalendarEvent,
    _gcal_start_ts,
    _human_when,
    _normalize_gcal_event,
    _serialize_event,
    build_prompt,
    manual_event,
)

# ============================ schema =================================

def test_event_briefing_save_and_get(fresh_db):
    bid = event_briefing_save(
        fresh_db,
        event_id="cal-1/ev-1", event_source="google_calendar",
        event_starts_at=time.time() + 600,
        event_title="Anthropic phone screen",
        event_url="https://calendar.google.com/...",
        event_payload_json='{"foo":"bar"}',
        briefing_text="Be ready for distributed-systems questions.",
        citations_json='[{"file_path":"https://x"}]',
        cents_spent=2.5,
    )
    assert bid > 0
    row = event_briefing_get(fresh_db, "cal-1/ev-1", "google_calendar")
    assert row is not None
    assert row["briefing_text"] == "Be ready for distributed-systems questions."
    assert row["cents_spent"] == 2.5


def test_event_briefing_save_replaces_on_conflict(fresh_db):
    """UNIQUE(event_id, event_source) → re-saving updates rather than
    accumulating duplicate rows."""
    starts = time.time() + 600
    event_briefing_save(
        fresh_db, event_id="ev1", event_source="ics",
        event_starts_at=starts, event_title="t", event_url="",
        event_payload_json="{}", briefing_text="first",
    )
    event_briefing_save(
        fresh_db, event_id="ev1", event_source="ics",
        event_starts_at=starts, event_title="t", event_url="",
        event_payload_json="{}", briefing_text="second",
    )
    rows = fresh_db.execute(
        "SELECT * FROM event_briefings WHERE event_id = ? AND event_source = ?",
        ("ev1", "ics"),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["briefing_text"] == "second"


def test_event_briefing_save_failure_path(fresh_db):
    """Errored runs persist a row with error populated."""
    event_briefing_save(
        fresh_db, event_id="ev2", event_source="manual",
        event_starts_at=time.time(), event_title="x",
        event_url=None, event_payload_json=None,
        briefing_text=None, error="rate limited",
    )
    row = event_briefing_get(fresh_db, "ev2", "manual")
    assert row["error"] == "rate limited"
    assert row["briefing_text"] is None


def test_event_briefings_list_orders_by_event_starts_desc(fresh_db):
    now = time.time()
    event_briefing_save(
        fresh_db, event_id="a", event_source="x",
        event_starts_at=now - 3600, event_title="old",
        event_url="", event_payload_json="{}",
        briefing_text="t",
    )
    event_briefing_save(
        fresh_db, event_id="b", event_source="x",
        event_starts_at=now + 3600, event_title="future",
        event_url="", event_payload_json="{}",
        briefing_text="t",
    )
    rows = event_briefings_list(fresh_db)
    # Most-recent (and future) event first.
    assert rows[0]["event_id"] == "b"


def test_event_briefings_upcoming_excludes_old(fresh_db):
    now = time.time()
    event_briefing_save(
        fresh_db, event_id="old", event_source="x",
        event_starts_at=now - 7200, event_title="old",
        event_url="", event_payload_json="{}", briefing_text="t",
    )
    event_briefing_save(
        fresh_db, event_id="soon", event_source="x",
        event_starts_at=now + 600, event_title="soon",
        event_url="", event_payload_json="{}", briefing_text="t",
    )
    upcoming = event_briefings_upcoming(fresh_db)
    ids = {r["event_id"] for r in upcoming}
    assert "soon" in ids
    assert "old" not in ids


# ===================== Google Calendar normalization ==================

def test_normalize_gcal_event_minimal():
    ev = {
        "id": "evid",
        "summary": "Anthropic phone screen",
        "start": {"dateTime": "2026-04-15T14:00:00Z"},
        "end":   {"dateTime": "2026-04-15T14:30:00Z"},
        "htmlLink": "https://calendar.google.com/event?eid=...",
        "organizer": {"email": "recruiter@anthropic.com"},
        "attendees": [
            {"email": "recruiter@anthropic.com"},
            {"email": "you@cornell.edu"},
        ],
    }
    out = _normalize_gcal_event(ev, "primary", "Personal")
    assert out is not None
    assert out.event_id == "primary/evid"
    assert out.title == "Anthropic phone screen"
    assert out.source == "google_calendar"
    assert out.calendar_name == "Personal"
    assert out.organizer_email == "recruiter@anthropic.com"
    # Organizer should be filtered out of the attendees list to avoid double-count.
    assert "recruiter@anthropic.com" not in out.attendees
    assert "you@cornell.edu" in out.attendees
    # Duration in seconds.
    assert out.duration_seconds == 30 * 60


def test_normalize_gcal_event_skips_cancelled():
    ev = {
        "id": "x", "summary": "y", "status": "cancelled",
        "start": {"dateTime": "2026-04-15T14:00:00Z"},
    }
    assert _normalize_gcal_event(ev, "p", "P") is None


def test_normalize_gcal_event_missing_id():
    ev = {"summary": "y", "start": {"dateTime": "2026-04-15T14:00:00Z"}}
    assert _normalize_gcal_event(ev, "p", "P") is None


def test_normalize_gcal_event_missing_start():
    ev = {"id": "x", "summary": "y"}
    assert _normalize_gcal_event(ev, "p", "P") is None


def test_gcal_start_ts_handles_all_day_event():
    ts = _gcal_start_ts({"date": "2026-04-15"})
    assert ts > 0


def test_gcal_start_ts_handles_garbage_returns_zero():
    assert _gcal_start_ts({}) == 0.0
    assert _gcal_start_ts({"dateTime": "garbage"}) == 0.0


# ============================ prompt builder ==========================

def test_build_prompt_includes_event_metadata():
    ev = CalendarEvent(
        event_id="x", source="google_calendar",
        starts_at=time.time() + 600,
        title="Anthropic phone screen",
        attendees=["recruiter@anthropic.com"],
        location="Zoom",
        description="Initial chat about the PM role.",
    )
    p = build_prompt(ev)
    assert "Anthropic phone screen" in p
    assert "recruiter@anthropic.com" in p
    assert "Zoom" in p
    assert "Initial chat" in p
    # Structure cues should be present so the model knows the format.
    assert "Quick context" in p
    assert "What you should know" in p
    assert "Suggested questions" in p
    assert "search_brain" in p
    assert "web_search" in p


def test_build_prompt_truncates_very_long_descriptions():
    ev = CalendarEvent(
        event_id="x", source="manual", starts_at=time.time(),
        title="t", description="A" * 5000,
    )
    p = build_prompt(ev)
    assert "[...truncated]" in p
    # Bounded - shouldn't include the entire 5000-char body.
    assert p.count("A") < 4000


def test_human_when_includes_relative_phrase():
    out = _human_when(time.time() + 600, 1800)
    assert "in 10 minutes" in out
    assert "30 min" in out  # duration


def test_human_when_handles_past_events():
    out = _human_when(time.time() - 600, 0)
    assert "started" in out  # past phrasing


# =========================== ad-hoc events ============================

def test_manual_event_parses_iso():
    ev = manual_event(
        "Coffee with Sarah",
        starts_at_iso="2026-04-20T14:00:00",
    )
    assert ev.title == "Coffee with Sarah"
    assert ev.source == "manual"
    assert ev.starts_at > 0


def test_manual_event_handles_z_suffix():
    ev = manual_event("t", starts_at_iso="2026-04-20T14:00:00Z")
    assert ev.starts_at > 0


def test_manual_event_rejects_garbage():
    with pytest.raises(ValueError, match="can't parse starts_at"):
        manual_event("t", starts_at_iso="not a date")


def test_manual_event_id_is_deterministic_for_same_inputs():
    """Same title + start time should produce the same id, so re-running
    `secondbrain brief now` for the same event upserts."""
    a = manual_event("Coffee", starts_at_iso="2026-04-20T14:00:00")
    b = manual_event("Coffee", starts_at_iso="2026-04-20T14:00:00")
    assert a.event_id == b.event_id


def test_serialize_event_round_trip():
    ev = CalendarEvent(
        event_id="x", source="manual",
        starts_at=1750000000.0,
        title="t", description="d",
        attendees=["a@b.com"], location="Zoom",
    )
    blob = _serialize_event(ev)
    parsed = json.loads(blob)
    assert parsed["title"] == "t"
    assert parsed["attendees"] == ["a@b.com"]
    assert parsed["starts_at"] == 1750000000.0


# ============================= scheduler ==============================

def test_scheduler_disabled_via_config(fresh_db, tmp_cfg, fake_embedder):
    """event_briefing_enabled=False short-circuits to 0 work."""
    from secondbrain.event_briefing import run_briefings_if_due

    cfg = replace(tmp_cfg, event_briefing_enabled=False)
    assert run_briefings_if_due(cfg, fresh_db, fake_embedder, None) == 0


def test_scheduler_skips_events_with_existing_briefing(
    fresh_db, tmp_cfg, fake_embedder, monkeypatch,
):
    """If a successful briefing already exists, don't regenerate."""
    from secondbrain import event_briefing as eb

    starts = time.time() + 300
    event = CalendarEvent(
        event_id="ev1", source="google_calendar",
        starts_at=starts, title="standup",
    )
    monkeypatch.setattr(eb, "_gather_due_events", lambda *a, **kw: [event])

    # Pre-populate a successful briefing.
    event_briefing_save(
        fresh_db, event_id="ev1", event_source="google_calendar",
        event_starts_at=starts, event_title="standup",
        event_url="", event_payload_json="{}",
        briefing_text="existing",
    )

    calls: list = []
    monkeypatch.setattr(eb, "generate_for_event",
                        lambda *a, **kw: calls.append(1) or {"ok": True})
    monkeypatch.setattr(eb, "notify", lambda *a, **kw: None)

    n = eb.run_briefings_if_due(tmp_cfg, fresh_db, fake_embedder, None)
    assert n == 0
    assert calls == [], "shouldn't regenerate when a successful briefing exists"


def test_scheduler_retries_errored_briefing(
    fresh_db, tmp_cfg, fake_embedder, monkeypatch,
):
    """Errored briefings should be retried (the failure was likely
    transient — network blip, budget that's since reset)."""
    from secondbrain import event_briefing as eb

    starts = time.time() + 300
    event = CalendarEvent(
        event_id="ev2", source="ics",
        starts_at=starts, title="meeting",
    )
    monkeypatch.setattr(eb, "_gather_due_events", lambda *a, **kw: [event])

    event_briefing_save(
        fresh_db, event_id="ev2", event_source="ics",
        event_starts_at=starts, event_title="meeting",
        event_url="", event_payload_json="{}",
        briefing_text=None, error="rate limited",
    )

    calls: list = []
    monkeypatch.setattr(eb, "generate_for_event",
                        lambda *a, **kw: calls.append(1) or {"ok": True})
    monkeypatch.setattr(eb, "notify", lambda *a, **kw: None)

    n = eb.run_briefings_if_due(tmp_cfg, fresh_db, fake_embedder, None)
    assert n == 1
    assert len(calls) == 1


def test_scheduler_respects_per_run_cap(
    fresh_db, tmp_cfg, fake_embedder, monkeypatch,
):
    """When 10 events are due but cap is 2, only 2 generate this tick."""
    from secondbrain import event_briefing as eb

    events = [
        CalendarEvent(
            event_id=f"ev{i}", source="google_calendar",
            starts_at=time.time() + 60 * i, title=f"e{i}",
        )
        for i in range(10)
    ]
    monkeypatch.setattr(eb, "_gather_due_events", lambda *a, **kw: events)

    calls: list = []
    monkeypatch.setattr(eb, "generate_for_event",
                        lambda *a, **kw: calls.append(1) or {"ok": True})
    monkeypatch.setattr(eb, "notify", lambda *a, **kw: None)

    cfg = replace(tmp_cfg, briefing_max_per_run=2)
    n = eb.run_briefings_if_due(cfg, fresh_db, fake_embedder, None)
    assert n == 2
    assert len(calls) == 2


def test_scheduler_notifies_on_success(
    fresh_db, tmp_cfg, fake_embedder, monkeypatch,
):
    """Successful generations fire a tray notification."""
    from secondbrain import event_briefing as eb

    event = CalendarEvent(
        event_id="ev3", source="manual",
        starts_at=time.time() + 600, title="standup",
    )
    monkeypatch.setattr(eb, "_gather_due_events", lambda *a, **kw: [event])

    monkeypatch.setattr(eb, "generate_for_event",
                        lambda *a, **kw: {"ok": True, "text": "ok", "cents": 0.5})
    fired: list = []
    monkeypatch.setattr(eb, "notify",
                        lambda title, msg, **kw: fired.append((title, msg)) or True)

    n = eb.run_briefings_if_due(tmp_cfg, fresh_db, fake_embedder, None)
    assert n == 1
    assert len(fired) == 1
    assert "standup" in fired[0][0]


# ========================= generate_for_event =========================

def test_generate_for_event_persists_success(
    fresh_db, tmp_cfg, fake_embedder, monkeypatch,
):
    from secondbrain import event_briefing as eb
    from secondbrain.chat import ChatResponse, Citation

    monkeypatch.setattr(
        eb, "ask_brain",
        lambda *a, **kw: ChatResponse(
            text="be prepared", citations=[
                Citation(chunk_id=1, file_path="https://x", chunk_index=0,
                         text="...", score=0.9, kind="web", url="https://x"),
            ],
        ),
    )
    event = CalendarEvent(
        event_id="ev4", source="manual",
        starts_at=time.time() + 600, title="t",
    )
    result = eb.generate_for_event(tmp_cfg, fresh_db, fake_embedder, None, event)
    assert result["ok"]
    row = event_briefing_get(fresh_db, "ev4", "manual")
    assert row is not None
    assert row["briefing_text"] == "be prepared"
    cites = json.loads(row["citations_json"])
    assert len(cites) == 1
    assert cites[0]["url"] == "https://x"


def test_generate_for_event_persists_budget_error(
    fresh_db, tmp_cfg, fake_embedder, monkeypatch,
):
    from secondbrain import event_briefing as eb
    from secondbrain.budget import BudgetExceededError

    def boom(*a, **kw):
        raise BudgetExceededError("anthropic", 1000.0, 500.0)

    monkeypatch.setattr(eb, "ask_brain", boom)
    event = CalendarEvent(
        event_id="ev5", source="manual",
        starts_at=time.time() + 600, title="t",
    )
    result = eb.generate_for_event(tmp_cfg, fresh_db, fake_embedder, None, event)
    assert result["ok"] is False
    row = event_briefing_get(fresh_db, "ev5", "manual")
    assert row is not None
    assert row["error"] is not None
    assert "budget" in row["error"].lower()


def test_generate_for_event_persists_unexpected_error(
    fresh_db, tmp_cfg, fake_embedder, monkeypatch,
):
    """Non-budget exceptions also land as a row with error populated, so
    the dashboard can surface what went wrong."""
    from secondbrain import event_briefing as eb

    def boom(*a, **kw):
        raise RuntimeError("some other failure")

    monkeypatch.setattr(eb, "ask_brain", boom)
    event = CalendarEvent(
        event_id="ev6", source="manual",
        starts_at=time.time() + 600, title="t",
    )
    result = eb.generate_for_event(tmp_cfg, fresh_db, fake_embedder, None, event)
    assert result["ok"] is False
    row = event_briefing_get(fresh_db, "ev6", "manual")
    assert "some other failure" in (row["error"] or "")
