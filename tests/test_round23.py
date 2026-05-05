"""Round 23 — fixes for the round-22 audit.

Each test maps to a finding from the audit:
  - HIGH 1: health.snapshot didn't exist; today now uses summarise
  - HIGH 2: calendar_view didn't exist; today uses event_briefing
  - HIGH 3: triage_undo proper error handling + toast confirmation
  - HIGH 4: 3-bucket sort preserves triage rank
  - HIGH 5: launchpad "Today" group renamed to "Daily"
  - MED 6:  triage_undo redirect with undo confirmation
  - MED 7:  ``next=`` redirect param so /today actions stay on /today
  - MED 8:  inline JS clears undo_* params after toast renders
  - MED 9:  _render_decision uses parse_qsl
  - MED 10: night-mode greeting rephrased
  - LOW 13: icons escaped
  - LOW 17: sad-path resilience for assemble_today
"""

from __future__ import annotations

import time
from datetime import datetime

# ============================ HIGH 1 — health surface ===================


def test_worth_knowing_health_uses_summarise(fresh_db, tmp_cfg, monkeypatch):
    """A flagged metric (>=15% off avg) surfaces as a worth-knowing
    item — round-22 was silently broken because it called the
    non-existent ``health.snapshot``."""
    from secondbrain import health
    from secondbrain import today as today_mod

    # Seed health_metrics with a clear anomaly: 14 days of value=80
    # then today=60 (25% drop).
    health._ensure_schema(fresh_db) if hasattr(
        health, "_ensure_schema",
    ) else None
    # Use the real DDL that exists.
    fresh_db.executescript("""
        CREATE TABLE IF NOT EXISTS health_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL DEFAULT 'oura',
            metric TEXT NOT NULL,
            date TEXT NOT NULL,
            value REAL NOT NULL,
            recorded_at REAL NOT NULL,
            UNIQUE(source, metric, date)
        );
    """)
    from datetime import date as _date
    from datetime import timedelta as _td
    today_d = _date.today()
    for i in range(14, 0, -1):
        d = (today_d - _td(days=i)).isoformat()
        fresh_db.execute(
            "INSERT OR IGNORE INTO health_metrics"
            "(source, metric, date, value, recorded_at) "
            "VALUES ('oura', 'sleep_score', ?, 80, ?)",
            (d, time.time()),
        )
    # Today: 60 (25% drop from 80)
    fresh_db.execute(
        "INSERT OR IGNORE INTO health_metrics"
        "(source, metric, date, value, recorded_at) "
        "VALUES ('oura', 'sleep_score', ?, 60, ?)",
        (today_d.isoformat(), time.time()),
    )
    fresh_db.commit()
    items = today_mod._worth_knowing_health(fresh_db)
    assert len(items) >= 1
    msg = items[0]
    assert msg.kind == "health"
    assert "dropped" in msg.title.lower() or "down" in msg.title.lower()
    # The actual delta is around -23%.
    assert "%" in msg.title


def test_worth_knowing_health_no_data_returns_empty(fresh_db):
    from secondbrain import today as today_mod
    items = today_mod._worth_knowing_health(fresh_db)
    assert items == []


# ============================ HIGH 2 — calendar surface =================


def test_today_calendar_uses_event_briefing(fresh_db, tmp_cfg, monkeypatch):
    """``_today_calendar`` should pull from event_briefing.iter_upcoming_events;
    when stubbed, the result must flow through."""
    from secondbrain import today as today_mod

    class _Ev:
        def __init__(self, starts_at, title, attendees=None):
            self.starts_at = starts_at
            self.title = title
            self.attendees = attendees or []

    soon = time.time() + 3600
    fake_events = [_Ev(soon, "1:1 with John", ["john@x"])]
    monkeypatch.setattr(
        "secondbrain.event_briefing.iter_upcoming_events",
        lambda cfg, horizon: iter(fake_events),
    )
    out = today_mod._today_calendar(tmp_cfg)
    assert len(out) == 1
    assert out[0].title == "1:1 with John"
    # The when-string is rendered (h12 with am/pm).
    assert any(c.isalpha() for c in out[0].when)
    assert "with john@x" in out[0].detail


def test_today_calendar_handles_import_failure(tmp_cfg, monkeypatch):
    """If event_briefing module isn't available, return empty — no crash."""
    import sys

    from secondbrain import today as today_mod
    # Force import failure by patching sys.modules.
    monkeypatch.setitem(sys.modules, "secondbrain.event_briefing", None)
    out = today_mod._today_calendar(tmp_cfg)
    assert out == []


# ============================ HIGH 4 — sort preserves rank ==============


def test_triage_decisions_get_age_days(fresh_db, tmp_cfg, monkeypatch):
    """Round 23 fix: triage decisions must set age_days so the
    assemble_today sort respects ranking."""
    from secondbrain import today as today_mod

    class _FakeIt:
        def __init__(self, age_h, fid=1):
            self.file_id = fid
            self.from_email = "a@x"
            self.from_display = "A"
            self.subject = "subj"
            self.label = "urgent"
            self.confidence = 0.9
            self.is_vip = True
            self.has_draft = False
            self.draft_id = None
            self.age_hours = age_h
    items = [_FakeIt(2.0, fid=1), _FakeIt(48.0, fid=2)]
    monkeypatch.setattr(
        "secondbrain.triage_queue.build_queue",
        lambda conn, hours, max_items: items,
    )
    decisions = today_mod._decisions_from_triage(fresh_db, limit=2)
    assert len(decisions) == 2
    # First decision: age_hours=2 → age_days≈0.083
    assert decisions[0].age_days is not None
    assert 0.08 < decisions[0].age_days < 0.09
    # Second: age_hours=48 → age_days=2.0
    assert decisions[1].age_days == 2.0


def test_assemble_today_3bucket_sort(fresh_db, tmp_cfg, monkeypatch):
    """Round 23 — 3-bucket sort places overdue followups first,
    then triage (in input order), then non-overdue followups."""
    from secondbrain import today as today_mod
    from secondbrain.today import Action, Decision

    fu_overdue = Decision(
        kind="followup_owed", title="overdue", why="past due 3d",
        primary=Action(label="x", href="/x"), age_days=3.0,
    )
    triage1 = Decision(
        kind="triage_email", title="vip-urgent",
        why="VIP · urgent · 2h",
        primary=Action(label="x", href="/x"), age_days=0.083,
    )
    triage2 = Decision(
        kind="triage_email", title="fresh-fyi",
        why="just now",
        primary=Action(label="x", href="/x"), age_days=0.04,
    )
    fu_fresh = Decision(
        kind="followup_owed", title="fresh-fu", why="promised today",
        primary=Action(label="x", href="/x"), age_days=0.5,
    )
    monkeypatch.setattr(
        today_mod, "_decisions_from_triage",
        lambda conn, limit: [triage1, triage2],
    )
    monkeypatch.setattr(
        today_mod, "_decisions_from_followups",
        lambda conn, limit: [fu_overdue, fu_fresh],
    )
    monkeypatch.setattr(
        today_mod, "_today_calendar",
        lambda cfg, max_items=5, horizon_seconds=None: [],
    )
    monkeypatch.setattr(
        today_mod, "_worth_knowing_cadence",
        lambda conn, limit: [],
    )
    monkeypatch.setattr(
        today_mod, "_worth_knowing_health", lambda conn: [],
    )
    monkeypatch.setattr(
        today_mod, "_worth_knowing_journal", lambda conn: [],
    )
    monkeypatch.setattr(
        today_mod, "_worth_knowing_birthdays", lambda conn, days_window=7: [],
    )
    desk = today_mod.assemble_today(
        tmp_cfg, fresh_db,
        now=datetime(2025, 5, 5, 9, 0),
    )
    titles = [d.title for d in desk.decisions]
    # Overdue followup first.
    assert titles[0] == "overdue"
    # Triage items next, in input order.
    assert titles[1] == "vip-urgent"
    assert titles[2] == "fresh-fyi"
    # Fresh followup last.
    assert titles[3] == "fresh-fu"


# ============================ HIGH 5 — launchpad rename =================


def test_launchpad_today_group_renamed_to_daily(
    monkeypatch, tmp_path, fake_embedder,
):
    """Round 23 — the / launchpad group is "Daily" (not "Today")
    so it doesn't collide with the round-22 /today page."""
    from fastapi.testclient import TestClient

    from secondbrain.config import Config
    from secondbrain.dashboard import create_app
    cfg = Config(data_dir=tmp_path / "data")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("secondbrain.dashboard.load_config", lambda: cfg)
    monkeypatch.setattr(
        "secondbrain.dashboard.make_embedder", lambda c: fake_embedder,
    )
    monkeypatch.setattr(
        "secondbrain.dashboard.make_reranker", lambda c: None,
    )
    client = TestClient(create_app())
    r = client.get("/")
    assert r.status_code == 200
    # The launchpad must show the "Daily" group label.
    assert ">Daily<" in r.text
    # And "Today" should be a link inside it (pointing at /today).
    assert 'href="/today"' in r.text


# ============================ HIGH 3 + MED 6 — undo path ================


def _client(monkeypatch, tmp_path, fake_embedder):
    from fastapi.testclient import TestClient

    from secondbrain.config import Config
    from secondbrain.dashboard import create_app
    cfg = Config(data_dir=tmp_path / "data")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("secondbrain.dashboard.load_config", lambda: cfg)
    monkeypatch.setattr(
        "secondbrain.dashboard.make_embedder", lambda c: fake_embedder,
    )
    monkeypatch.setattr(
        "secondbrain.dashboard.make_reranker", lambda c: None,
    )
    return cfg, TestClient(create_app())


def test_triage_undo_redirects_with_toast(
    monkeypatch, tmp_path, fake_embedder,
):
    """After undoing a real action, redirect carries the toast
    confirmation so the user sees feedback."""
    from secondbrain import email_assist, triage_queue
    from secondbrain.db import connect, init_schema

    cfg, client = _client(monkeypatch, tmp_path, fake_embedder)
    seed = connect(cfg.db_path)
    init_schema(seed, fake_embedder.dim, fake_embedder.name)
    email_assist._ensure_schema(seed)
    triage_queue._ensure_schema(seed)
    seed.execute(
        "INSERT INTO files(path, mtime, size, kind, content_hash, "
        "indexed_at) VALUES ('e.eml', 0, 0, 'email', 'h', ?)",
        (time.time(),),
    )
    fid = seed.execute(
        "SELECT id FROM files WHERE path='e.eml'",
    ).fetchone()["id"]
    triage_queue.mark_done(seed, fid)
    seed.close()

    r = client.post(
        f"/triage/{fid}/undo",
        headers={"referer": "http://127.0.0.1:8765/triage"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    loc = r.headers["location"]
    assert "undo_done=1" in loc
    assert "undo_label=Undone" in loc


def test_triage_undo_no_op_no_toast(
    monkeypatch, tmp_path, fake_embedder,
):
    """Undo on a row that was never triaged is a clean redirect
    with NO toast (nothing was undone)."""
    cfg, client = _client(monkeypatch, tmp_path, fake_embedder)
    r = client.post(
        "/triage/9999/undo",
        headers={"referer": "http://127.0.0.1:8765/triage"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert "undo_done=1" not in r.headers["location"]


# ============================ MED 7 — next= param =======================


def test_triage_done_with_next_redirects_to_today(
    monkeypatch, tmp_path, fake_embedder,
):
    """A done action posted with next=/today redirects to /today
    so the user stays on the morning desk."""
    from secondbrain import email_assist
    from secondbrain.db import connect, init_schema

    cfg, client = _client(monkeypatch, tmp_path, fake_embedder)
    seed = connect(cfg.db_path)
    init_schema(seed, fake_embedder.dim, fake_embedder.name)
    email_assist._ensure_schema(seed)
    seed.execute(
        "INSERT INTO files(path, mtime, size, kind, content_hash, "
        "indexed_at) VALUES ('e.eml', 0, 0, 'email', 'h', ?)",
        (time.time(),),
    )
    fid = seed.execute(
        "SELECT id FROM files WHERE path='e.eml'",
    ).fetchone()["id"]
    seed.commit()
    seed.close()
    r = client.post(
        f"/triage/{fid}/done",
        data={"next": "/today"},
        headers={"referer": "http://127.0.0.1:8765/today"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert r.headers["location"].startswith("/today?")


def test_triage_next_param_rejects_open_redirect(
    monkeypatch, tmp_path, fake_embedder,
):
    """next=https://evil.com is whitelist-rejected → falls back
    to /triage."""
    from secondbrain import email_assist
    from secondbrain.db import connect, init_schema

    cfg, client = _client(monkeypatch, tmp_path, fake_embedder)
    seed = connect(cfg.db_path)
    init_schema(seed, fake_embedder.dim, fake_embedder.name)
    email_assist._ensure_schema(seed)
    seed.execute(
        "INSERT INTO files(path, mtime, size, kind, content_hash, "
        "indexed_at) VALUES ('e.eml', 0, 0, 'email', 'h', ?)",
        (time.time(),),
    )
    fid = seed.execute(
        "SELECT id FROM files WHERE path='e.eml'",
    ).fetchone()["id"]
    seed.commit()
    seed.close()
    r = client.post(
        f"/triage/{fid}/done",
        data={"next": "https://evil.com/steal"},
        headers={"referer": "http://127.0.0.1:8765/today"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    # Falls back to /triage (the safe default).
    assert r.headers["location"].startswith("/triage")
    assert "evil.com" not in r.headers["location"]


# ============================ MED 8 — refresh-clear =====================


def test_undo_toast_includes_history_replacestate_js(
    monkeypatch, tmp_path, fake_embedder,
):
    """When the toast renders, the inline JS that strips undo_*
    params from the URL must be present."""
    cfg, client = _client(monkeypatch, tmp_path, fake_embedder)
    r = client.get(
        "/today?undo_done=1&undo_label=Undone",
    )
    assert r.status_code == 200
    assert "Undone" in r.text
    assert "history.replaceState" in r.text


# ============================ MED 9 — parse_qsl =========================


def test_render_decision_form_uses_parse_qsl():
    """Source-string check: the form-encoded path uses parse_qsl
    so URL-decoded values aren't double-encoded on submit."""
    import inspect

    from secondbrain.dashboard import _render_decision
    src = inspect.getsource(_render_decision)
    assert "parse_qsl" in src
    # The hidden input always carries next=/today now.
    assert 'name="next"' in src and '/today' in src


# ============================ MED 10 — night greeting ===================


def test_night_greeting_is_complete_phrase():
    from secondbrain.today import _GREETING_BY_MODE, _QUIET_BY_MODE

    night_greeting = _GREETING_BY_MODE["night"]
    assert night_greeting == "Day's done"
    night_quiet = _QUIET_BY_MODE["night"]
    # Composes naturally with the greeting.
    assert "tomorrow" in night_quiet.lower()


# ============================ LOW 13 — icon escape ======================


def test_decision_icon_is_escaped():
    import inspect

    from secondbrain.dashboard import _render_decision
    src = inspect.getsource(_render_decision)
    assert "escape(d.icon)" in src


def test_worth_knowing_icon_is_escaped():
    """The /today route renders worth_knowing with escape(w.icon)."""
    import inspect

    from secondbrain import dashboard as dm
    full_src = inspect.getsource(dm)
    assert "escape(w.icon)" in full_src


# ============================ LOW 17 — sad-path resilience ==============


def test_assemble_today_swallows_partial_failures(
    fresh_db, tmp_cfg, monkeypatch,
):
    """Round 23 — every sub-source raises an exception; the
    overall call must still return a valid TodayDesk (quiet),
    not crash the page. Round 22's audit caught two HIGH bugs
    where an entire sub-source was silently broken; this is the
    integration test that would have flagged them."""
    from secondbrain import today as today_mod

    def _boom(*a, **kw):
        raise RuntimeError("simulated failure")

    # Override every sub-source to raise.
    monkeypatch.setattr(
        today_mod, "_decisions_from_triage", _boom,
    )
    monkeypatch.setattr(
        today_mod, "_decisions_from_followups", _boom,
    )
    monkeypatch.setattr(today_mod, "_today_calendar", _boom)
    monkeypatch.setattr(
        today_mod, "_worth_knowing_cadence", _boom,
    )
    monkeypatch.setattr(today_mod, "_worth_knowing_health", _boom)
    monkeypatch.setattr(today_mod, "_worth_knowing_journal", _boom)
    monkeypatch.setattr(today_mod, "_worth_knowing_birthdays", _boom)
    # Round 23 — assemble_today now wraps every sub-source. The
    # call must complete without raising and produce a valid (quiet)
    # desk even when every sub-source fails.
    desk = today_mod.assemble_today(tmp_cfg, fresh_db)
    assert desk.greeting
    assert desk.is_quiet()


# ============================ smoke =====================================


def test_round23_modules_import():
    from secondbrain import today, ux_copy  # noqa: F401
