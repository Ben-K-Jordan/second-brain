"""Round 22 — EA-shaped UI tests.

  - today.py: assemble_today, time-of-day modes, greeting, render
  - ux_copy: empty states + adaptive variants + why-line phrases
  - dashboard: /today route, triage walkthrough, undo toast, nav
"""

from __future__ import annotations

import time
from datetime import datetime

# ============================ today module ===========================


def test_time_of_day_mode_buckets():
    from secondbrain.today import time_of_day_mode

    morning = datetime(2025, 5, 5, 7, 0)
    midday = datetime(2025, 5, 5, 12, 30)
    afternoon = datetime(2025, 5, 5, 15, 0)
    evening = datetime(2025, 5, 5, 19, 0)
    night = datetime(2025, 5, 5, 23, 0)
    assert time_of_day_mode(morning) == "morning"
    assert time_of_day_mode(midday) == "midday"
    assert time_of_day_mode(afternoon) == "afternoon"
    assert time_of_day_mode(evening) == "evening"
    assert time_of_day_mode(night) == "night"


def test_greeting_includes_name_and_date():
    from secondbrain.today import greeting_for

    now = datetime(2025, 5, 5, 8, 0)  # Mon
    g = greeting_for("Ben", "morning", now)
    assert "Good morning" in g
    assert "Ben" in g
    assert "Monday" in g
    assert "May" in g


def test_greeting_handles_empty_name():
    from secondbrain.today import greeting_for

    g = greeting_for("", "midday", datetime(2025, 5, 5, 12, 30))
    assert "Quick check-in" in g
    # No leading comma when name is empty.
    assert "Quick check-in," not in g


def test_assemble_today_quiet_state(fresh_db, tmp_cfg):
    from secondbrain import today as today_mod

    desk = today_mod.assemble_today(tmp_cfg, fresh_db)
    # Empty DB → no decisions, no upcoming, no worth_knowing.
    assert desk.is_quiet()
    assert desk.quiet_message is not None
    assert desk.greeting


def test_assemble_today_with_overdue_followup(fresh_db, tmp_cfg):
    """An overdue followup must show up as a decision with a
    'past due' why-line."""
    from secondbrain import followups
    from secondbrain import today as today_mod

    yesterday = time.time() - 86400
    followups.add_followup(
        fresh_db, direction="outgoing",
        topic="Send Q3 deck",
        description="Send Sarah the Q3 deck",
        person_name="Sarah",
        promised_at=time.time() - 5 * 86400,
        due_at=yesterday,
    )
    desk = today_mod.assemble_today(tmp_cfg, fresh_db)
    assert not desk.is_quiet()
    decisions = desk.decisions
    assert any(
        "Send Q3 deck" in d.title for d in decisions
    ), [d.title for d in decisions]
    overdue = next(
        d for d in decisions if "Send Q3 deck" in d.title
    )
    assert "past due" in overdue.why.lower() or "promised" in overdue.why.lower()


def test_assemble_today_evening_mode_quiet_message_differs():
    from secondbrain.today import _quiet_message_for

    morning = _quiet_message_for("morning")
    evening = _quiet_message_for("evening")
    assert morning != evening
    assert "Coffee" in morning
    assert "Day's wrapped" in evening or "wrap" in evening.lower()


def test_render_markdown_quiet_day():
    from secondbrain.today import (
        TodayDesk,
        render_markdown,
    )

    desk = TodayDesk(
        greeting="Good morning, Ben — Monday, May 5",
        mode="morning",
        decisions=[],
        upcoming=[],
        worth_knowing=[],
        quiet_message="Quiet morning. Coffee on me.",
        generated_at=time.time(),
    )
    md = render_markdown(desk)
    assert "Good morning, Ben" in md
    assert "Coffee on me" in md
    # No decisions section header.
    assert "decisions" not in md.lower()


def test_render_markdown_with_decisions():
    from secondbrain.today import (
        Action,
        Decision,
        TodayDesk,
        render_markdown,
    )

    desk = TodayDesk(
        greeting="Good morning",
        mode="morning",
        decisions=[Decision(
            kind="followup_owed",
            title="Send deck → Sarah",
            why="past due 2d",
            primary=Action(label="Mark done", href="/x", method="POST"),
        )],
        upcoming=[],
        worth_knowing=[],
        generated_at=time.time(),
    )
    md = render_markdown(desk)
    assert "1 decisions" in md
    assert "Send deck → Sarah" in md
    assert "past due 2d" in md


# ============================ ux_copy ================================


def test_empty_state_returns_known_copy():
    from secondbrain.ux_copy import empty_state

    assert "Inbox at zero" in empty_state("triage")
    assert "Take the win" in empty_state("followups_outgoing")
    # Unknown key falls back.
    assert empty_state("not_a_real_key") == "Nothing here."
    assert empty_state("nope", "custom fallback") == "custom fallback"


def test_adaptive_empty_swaps_by_hour():
    from secondbrain.ux_copy import adaptive_empty

    morning = datetime(2025, 5, 5, 8, 0)
    evening = datetime(2025, 5, 5, 19, 0)
    assert "Coffee" in adaptive_empty("triage", morning)
    # Evening fallback for notifications has its own copy.
    assert "Quiet evening" in adaptive_empty("notifications", evening)


def test_days_ago_phrase_humanises():
    from secondbrain.ux_copy import days_ago_phrase

    now = time.time()
    assert days_ago_phrase(now) == "today"
    assert days_ago_phrase(now - 86400 * 1.5) == "yesterday"
    assert "days" in days_ago_phrase(now - 86400 * 3)
    assert "wk" in days_ago_phrase(now - 86400 * 14)
    assert "mo" in days_ago_phrase(now - 86400 * 90)
    assert days_ago_phrase(None) == ""


def test_overdue_phrase():
    from secondbrain.ux_copy import overdue_phrase

    now = time.time()
    assert "past due" in overdue_phrase(now - 3 * 86400)
    assert overdue_phrase(now + 3600) == "due today"
    assert overdue_phrase(now + 86400 * 1.5) == "due tomorrow"
    assert "due in" in overdue_phrase(now + 86400 * 5)


# ============================ dashboard /today =======================


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


def test_today_page_renders_quiet_state(monkeypatch, tmp_path, fake_embedder):
    cfg, client = _client(monkeypatch, tmp_path, fake_embedder)
    r = client.get("/today")
    assert r.status_code == 200
    # The quiet-state class is always present when no decisions /
    # events / worth-knowing come back. Each time-of-day mode has
    # its own quiet message; assert structurally rather than per-
    # string so the test is TZ-of-runner-agnostic.
    assert 'today-quiet' in r.text
    assert 'today-greet' in r.text


def test_today_page_renders_decisions(
    monkeypatch, tmp_path, fake_embedder,
):
    """Seed an overdue followup and verify it shows on /today."""
    from secondbrain import followups
    from secondbrain.db import connect, init_schema

    cfg, client = _client(monkeypatch, tmp_path, fake_embedder)
    seed = connect(cfg.db_path)
    init_schema(seed, fake_embedder.dim, fake_embedder.name)
    yesterday = time.time() - 86400
    followups.add_followup(
        seed, direction="outgoing",
        topic="Send Q3 deck",
        description="Send Sarah the Q3 deck",
        person_name="Sarah",
        promised_at=time.time() - 5 * 86400,
        due_at=yesterday,
    )
    seed.close()
    r = client.get("/today")
    assert r.status_code == 200
    assert "Send Q3 deck" in r.text
    # Why-line present.
    assert "past due" in r.text.lower() or "promised" in r.text.lower()


def test_today_page_renders_undo_toast(
    monkeypatch, tmp_path, fake_embedder,
):
    cfg, client = _client(monkeypatch, tmp_path, fake_embedder)
    r = client.get("/today?undo_done=1&undo_label=Snoozed%207%20days")
    assert r.status_code == 200
    assert "Snoozed 7 days" in r.text
    assert "✓" in r.text or "toast" in r.text.lower()


def test_triage_walkthrough_shows_one_email(
    monkeypatch, tmp_path, fake_embedder,
):
    """Round 22 — /triage default renders walkthrough, not list."""
    from secondbrain import email_assist, triage_queue
    from secondbrain.db import connect, init_schema

    cfg, client = _client(monkeypatch, tmp_path, fake_embedder)
    seed = connect(cfg.db_path)
    init_schema(seed, fake_embedder.dim, fake_embedder.name)
    email_assist._ensure_schema(seed)
    triage_queue._ensure_schema(seed)
    now = time.time()
    # Seed two emails; walkthrough should show only the top-ranked.
    for i, sender in enumerate(["alice@x", "bob@y"]):
        seed.execute(
            "INSERT INTO files(path, mtime, size, kind, "
            "content_hash, indexed_at) "
            "VALUES (?, 0, 0, 'email', ?, ?)",
            (f"e{i}.eml", f"h{i}", now - 3600),
        )
        fid = seed.execute(
            "SELECT id FROM files WHERE path = ?", (f"e{i}.eml",),
        ).fetchone()["id"]
        seed.execute(
            "INSERT INTO chunks(file_id, chunk_index, text, "
            "start_offset) VALUES (?, 0, ?, 0)",
            (fid,
             f"From: {sender}\nSubject: Test {i}\n\nbody"),
        )
        seed.execute(
            "INSERT INTO email_classifications"
            "(file_id, label, confidence, classified_at) "
            "VALUES (?, 'urgent', 0.9, ?)", (fid, now),
        )
    seed.commit()
    seed.close()
    r = client.get("/triage")
    assert r.status_code == 200
    # Walkthrough renders the single-email card with action buttons.
    assert "Mark done" in r.text
    assert "Snooze" in r.text
    assert "Not for me" in r.text
    # Counter present.
    assert "left" in r.text.lower()
    # Show-all link.
    assert "show all" in r.text


def test_triage_show_all_renders_list(
    monkeypatch, tmp_path, fake_embedder,
):
    """?show=all flag falls back to the legacy list view."""
    from secondbrain import email_assist
    from secondbrain.db import connect, init_schema

    cfg, client = _client(monkeypatch, tmp_path, fake_embedder)
    seed = connect(cfg.db_path)
    init_schema(seed, fake_embedder.dim, fake_embedder.name)
    email_assist._ensure_schema(seed)
    now = time.time()
    seed.execute(
        "INSERT INTO files(path, mtime, size, kind, content_hash, "
        "indexed_at) VALUES ('e.eml', 0, 0, 'email', 'h', ?)",
        (now - 3600,),
    )
    fid = seed.execute(
        "SELECT id FROM files WHERE path='e.eml'",
    ).fetchone()["id"]
    seed.execute(
        "INSERT INTO chunks(file_id, chunk_index, text, "
        "start_offset) VALUES (?, 0, "
        "'From: alice@x\nSubject: Test', 0)", (fid,),
    )
    seed.execute(
        "INSERT INTO email_classifications"
        "(file_id, label, confidence, classified_at) "
        "VALUES (?, 'urgent', 0.9, ?)", (fid, now),
    )
    seed.commit()
    seed.close()
    r = client.get("/triage?show=all")
    assert r.status_code == 200
    assert "full list" in r.text or "ranked" in r.text.lower()


def test_triage_done_redirects_with_undo_params(
    monkeypatch, tmp_path, fake_embedder,
):
    """Round 22 — /triage/{id}/done redirects to /triage with the
    undo query string set."""
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
        headers={"referer": "http://127.0.0.1:8765/triage"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    assert "undo_done=1" in r.headers["location"]
    assert "undo_label=Marked+done" in r.headers["location"]
    assert "undo_kind=triage" in r.headers["location"]


def test_triage_undo_clears_state(
    monkeypatch, tmp_path, fake_embedder,
):
    """Round 22 — /triage/{id}/undo deletes the triage_state row,
    putting the email back in the queue."""
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
    # Verify state was set.
    row = seed.execute(
        "SELECT decision FROM triage_state WHERE file_id = ?", (fid,),
    ).fetchone()
    assert row["decision"] == "done"
    seed.close()

    r = client.post(
        f"/triage/{fid}/undo",
        headers={"referer": "http://127.0.0.1:8765/triage"},
        follow_redirects=False,
    )
    assert r.status_code == 303

    seed = connect(cfg.db_path)
    row = seed.execute(
        "SELECT * FROM triage_state WHERE file_id = ?", (fid,),
    ).fetchone()
    seed.close()
    assert row is None  # undo cleared it


def test_triage_undo_csrf_guard(monkeypatch, tmp_path, fake_embedder):
    cfg, client = _client(monkeypatch, tmp_path, fake_embedder)
    r = client.post(
        "/triage/1/undo",
        headers={"origin": "https://evil.com"},
    )
    assert r.status_code == 403


def test_today_in_primary_nav():
    from secondbrain.dashboard import _PRIMARY_NAV
    labels = {item[0] for item in _PRIMARY_NAV}
    assert "Today" in labels
    # Brief moved out of primary nav.
    assert "Brief" not in labels


def test_brief_moved_to_ea_group():
    from secondbrain.dashboard import _NAV_GROUPS
    ea_group = next(
        items for label, items in _NAV_GROUPS if label == "EA"
    )
    hrefs = {h for _, h in ea_group}
    assert "/brief" in hrefs


def test_round22_modules_import():
    from secondbrain import today, ux_copy  # noqa: F401
