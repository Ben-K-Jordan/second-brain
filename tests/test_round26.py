"""Round 26 — fixes for the round-25 follow-on audit.

Each test maps to a finding from the round-25 audit:

  - HIGH H1: /today action handlers honor next= form param so the
    redirect lands back on /today (not /followups, /drafts).
  - HIGH H2: today_page accepts undo_kind + undo_id query params so
    the round-22 undo toast Undo button actually works.
  - HIGH H3: tasks.materialize_promises_from_transcripts walks
    voice:// + capture:// paths (matched the round-25 H3 study fix).
  - HIGH H4: CLI ``thanks draft`` typer default is ``None`` so the
    runtime fallback to cfg.user_name (round-25 H2) actually triggers.
  - MED  M5: vault_export._classify routes journal://, slack://,
    linear://, github://, etc. to sensible folders (no longer ``misc``).
  - MED  M6: notification keys for followup_overdue, followup_stale,
    email_urgent are date/week-bucketed so they re-fire over time.
  - MED  M7: /agenda/note/{id}/discussed redirects back to the same
    person's agenda page (was the empty /agenda landing).
  - MED  M9: timeline.py /file links use file_id not unencoded path.
  - MED  M10: weekly_letter delegates to db.EMAIL_KIND_SQL /
    db.TRANSCRIPT_KIND_SQL (single source of truth).
"""

from __future__ import annotations

import inspect
import time

import pytest


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


# ============================ HIGH H1 — next= round-trip ================


def _h1_handler_signatures():
    from secondbrain import dashboard
    src = inspect.getsource(dashboard.create_app)
    return src


def test_followup_resolve_accepts_next():
    """``followup_resolve`` route must take a ``next`` Form param."""
    src = _h1_handler_signatures()
    # The handler reads next=Form("") AND uses _safe_followup_next.
    assert "def followup_resolve" in src
    # Locate the function body and check the next= + redirect.
    head = src.split("def followup_resolve", 1)[1].split("def ", 1)[0]
    assert 'next: str = Form("")' in head
    assert "_safe_followup_next(next)" in head


def test_followup_dismiss_accepts_next():
    src = _h1_handler_signatures()
    head = src.split("def followup_dismiss", 1)[1].split("def ", 1)[0]
    assert 'next: str = Form("")' in head
    assert "_safe_followup_next(next)" in head


def test_followup_snooze_accepts_next():
    src = _h1_handler_signatures()
    head = src.split("def followup_snooze", 1)[1].split("def ", 1)[0]
    assert 'next: str = Form("")' in head


def test_followup_unsnooze_accepts_next():
    src = _h1_handler_signatures()
    head = src.split("def followup_unsnooze", 1)[1].split("def ", 1)[0]
    assert 'next: str = Form("")' in head


def test_followup_edit_accepts_next():
    src = _h1_handler_signatures()
    head = src.split("def followup_edit", 1)[1].split("def ", 1)[0]
    assert 'next: str = Form("")' in head


def test_drafts_mark_sent_accepts_next():
    src = _h1_handler_signatures()
    head = src.split("def drafts_mark_sent", 1)[1].split("def ", 1)[0]
    assert 'next: str = Form("")' in head
    assert "_safe_drafts_next(next)" in head


def test_drafts_discard_accepts_next():
    src = _h1_handler_signatures()
    head = src.split("def drafts_discard", 1)[1].split("def ", 1)[0]
    assert 'next: str = Form("")' in head
    assert "_safe_drafts_next(next)" in head


def test_safe_next_factory_pattern_uses_broad_whitelist():
    """The round-26 refactor unified _safe_next / _safe_followup_next /
    _safe_drafts_next under a shared whitelist that includes /today,
    /triage, /followups, /drafts."""
    src = _h1_handler_signatures()
    # Whitelist constant present.
    assert "_SAFE_NEXT_PREFIXES" in src
    # All four targets are recognised.
    for prefix in ("/today", "/triage", "/followups", "/drafts"):
        assert f'"{prefix}"' in src, (
            f"prefix {prefix!r} missing from _SAFE_NEXT_PREFIXES"
        )


def test_safe_next_round_trips_today():
    """Submitting ?next=/today via a form must redirect to /today."""
    from secondbrain.dashboard import (
        _safe_drafts_next,
        _safe_followup_next,
        _safe_next,
    )
    assert _safe_next("/today") == "/today"
    assert _safe_followup_next("/today") == "/today"
    assert _safe_drafts_next("/today") == "/today"


def test_safe_next_rejects_open_redirect():
    """Off-site redirects must be blocked even if quoted carefully."""
    from secondbrain.dashboard import _safe_followup_next
    assert _safe_followup_next("https://evil.com") == "/followups"
    assert _safe_followup_next("//evil.com/path") == "/followups"
    assert _safe_followup_next("/etc/passwd") == "/followups"


def test_safe_next_allows_query_string_suffix():
    """``/today?foo=bar`` is fine since prefix /today matches."""
    from secondbrain.dashboard import _safe_drafts_next
    assert _safe_drafts_next("/today?undo_done=1") == "/today?undo_done=1"


# ============================ HIGH H2 — undo on /today ==================


def test_today_page_accepts_undo_params():
    """today_page must take undo_kind + undo_id so the toast's Undo
    button reaches a real handler."""
    from secondbrain import dashboard
    src = inspect.getsource(dashboard.create_app)
    head = src.split("def today_page", 1)[1].split("def ", 1)[0]
    assert "undo_kind: str" in head
    assert "undo_id: int" in head
    # And they get passed through to _undo_toast_html.
    assert "undo_kind" in head and "undo_id" in head


def test_undo_toast_html_uses_real_kind_and_id():
    from secondbrain.dashboard import _undo_toast_html
    # Round-22 toast renders only when undo_done=1. Triage kind is
    # the only kind that currently has a paired /undo route, so the
    # form-with-id only renders for that kind.
    html = _undo_toast_html(1, "Snoozed", "triage", 42)
    assert "Snoozed" in html
    # The toast embeds the id in the undo form's action.
    assert "42" in html
    # Empty when undo_done=0.
    assert _undo_toast_html(0, "Snoozed", "triage", 42) == ""


# ============================ HIGH H3 — tasks materializer ==============


def test_materialize_promises_walks_voice_and_capture():
    """The audit-found gap: round-25 fixed the analogous study
    function for canvas + voice. Tasks promise extraction was
    missed in the same sweep."""
    from secondbrain import tasks
    src = inspect.getsource(tasks.materialize_promises_from_transcripts)
    assert "voice://" in src
    assert "capture://" in src
    assert "transcript://" in src


# ============================ HIGH H4 — CLI thanks default ==============


def test_thanks_draft_cli_default_is_none(monkeypatch):
    """The typer Option default must be None so the round-25 fallback
    in meeting_thanks.generate_thanks_draft (cfg.user_name) actually
    triggers. Earlier the CLI clobbered with placeholder 'I'."""
    from secondbrain import cli
    sig = inspect.signature(cli.thanks_draft)
    user_name_param = sig.parameters["user_name"]
    # Typer wraps the default in an OptionInfo; pull .default off it.
    opt_info = user_name_param.default
    assert opt_info.default is None, (
        f"user_name typer default should be None; got {opt_info.default!r}"
    )


# ============================ MED M5 — vault classify ===================


def test_vault_classify_journal():
    from secondbrain.vault_export import _classify
    assert _classify("journal://2026-05-05", "document") == "journal"


def test_vault_classify_slack_to_captures():
    from secondbrain.vault_export import _classify
    assert _classify("slack://team/channel/msg-1", "document") == "captures"


def test_vault_classify_linear_to_captures():
    from secondbrain.vault_export import _classify
    assert _classify("linear://issue/ENG-42", "document") == "captures"


def test_vault_classify_github_to_captures():
    from secondbrain.vault_export import _classify
    assert _classify("github://repo/foo/issues/1", "document") == "captures"


def test_vault_classify_email_to_notes():
    """Email-shaped paths flow into ``notes`` (treated as documents)."""
    from secondbrain.vault_export import _classify
    assert _classify("imap://INBOX/42", "url") == "notes"
    assert _classify("gmail://thread/T1", "url") == "notes"


def test_vault_classify_unknown_still_misc():
    """Truly unknown prefixes still fall into ``misc``."""
    from secondbrain.vault_export import _classify
    assert _classify(
        "unknown-prefix://foo", "url",
    ) == "misc"


# ============================ MED M6 — bucketed notif keys ==============


def test_email_urgent_key_includes_date_bucket():
    """Round-26 fix: key has a ``:YYYY-MM-DD`` suffix so the
    notification re-fires once per day."""
    from secondbrain import notifications
    src = inspect.getsource(notifications._detect_email_urgent)
    # Either f-string interpolation or string concat with today.
    assert "email_urgent:file_id=" in src
    assert "today" in src or "date.today" in src


def test_followup_overdue_key_includes_date_bucket():
    from secondbrain import notifications
    src = inspect.getsource(notifications._detect_followup_overdue)
    assert "followup_overdue:" in src
    assert "today" in src or "date.today" in src


def test_followup_stale_key_includes_week_bucket():
    """Stale incoming followups re-surface weekly, not daily."""
    from secondbrain import notifications
    src = inspect.getsource(notifications._detect_followup_stale)
    assert "followup_stale:" in src
    assert "isocalendar" in src or "week" in src


def test_followup_overdue_re_fires_next_day(fresh_db):
    """End-to-end: after manually advancing the bucket, an overdue
    followup re-enqueues a fresh notification."""
    from secondbrain import followups, notifications
    notifications._ensure_schema(fresh_db)
    followups._ensure_schema(fresh_db)
    # Seed an overdue outgoing followup. The followups schema requires
    # description / source_kind / dedup_key so we satisfy NOT NULL.
    now = time.time()
    fresh_db.execute(
        "INSERT INTO followups"
        "(direction, person_id, person_name, topic, description, "
        " source_kind, source_excerpt, status, due_at, "
        " created_at, updated_at, dedup_key) "
        "VALUES ('outgoing', NULL, 'Sarah', ?, ?, "
        "        'manual', '', 'open', ?, ?, ?, 'rd26-key-1')",
        ("test topic", "follow up with Sarah",
         now - 86400, now - 86400, now - 86400),
    )
    fresh_db.commit()
    # First detection inserts row.
    n1 = notifications._detect_followup_overdue(fresh_db)
    assert n1 == 1
    # Second detection same day no-ops.
    n2 = notifications._detect_followup_overdue(fresh_db)
    assert n2 == 0
    # Now poke the row's key to simulate yesterday's bucket — the
    # equivalent of "a day passed". We can't time-travel cleanly so
    # we mutate the stored key.
    fresh_db.execute(
        "UPDATE notifications SET key = key || ':simulated-prior-day'",
    )
    fresh_db.commit()
    # New detection must insert a NEW notification with today's bucket.
    n3 = notifications._detect_followup_overdue(fresh_db)
    assert n3 == 1, (
        "expected re-fire once the date bucket rolls forward"
    )


# ============================ MED M7 — agenda discussed redirect ========


def test_agenda_discussed_redirects_to_person(
    monkeypatch, tmp_path, fake_embedder,
):
    """POST /agenda/note/{id}/discussed must redirect to
    /agenda?id=<person_id> so the user stays in the per-person
    review flow."""
    from secondbrain import agenda
    from secondbrain.db import connect, init_schema

    cfg, client = _client(monkeypatch, tmp_path, fake_embedder)
    seed = connect(cfg.db_path)
    init_schema(seed, fake_embedder.dim, fake_embedder.name)
    # Need a person row first.
    now = time.time()
    seed.execute(
        "INSERT INTO people"
        "(canonical_name, display_name, first_seen_at, last_seen_at) "
        "VALUES ('sarah', 'Sarah', ?, ?)", (now, now),
    )
    pid = seed.execute(
        "SELECT id FROM people WHERE canonical_name='sarah'",
    ).fetchone()["id"]
    note_id = agenda.add_note(seed, pid, "discuss the migration")
    seed.commit()
    seed.close()

    r = client.post(
        f"/agenda/note/{note_id}/discussed",
        headers={"origin": "http://127.0.0.1"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    location = r.headers.get("location") or ""
    assert location == f"/agenda?id={pid}", (
        f"Expected /agenda?id={pid}; got {location!r}"
    )


# ============================ MED M9 — timeline file_id =================


def test_timeline_file_events_use_file_id():
    """Files in the timeline link via ?file_id=N (round-25 H1 route)."""
    from secondbrain import timeline
    src = inspect.getsource(timeline._events_files)
    # No raw path interpolation.
    assert "?path={r['path']}" not in src
    assert "?path={r[\"path\"]}" not in src
    # file_id is used.
    assert "file_id=" in src


def test_timeline_email_triage_events_use_file_id():
    from secondbrain import timeline
    src = inspect.getsource(timeline._events_email_triage)
    assert "?path={r['path']}" not in src
    assert "file_id=" in src
    # And the SELECT now pulls file_id.
    assert "ec.file_id" in src


# ============================ MED M10 — weekly_letter SQL constants ====


def test_weekly_letter_uses_email_kind_sql():
    """weekly_letter._signal_files now delegates to the shared
    db.EMAIL_KIND_SQL / db.TRANSCRIPT_KIND_SQL constants instead of
    inlining a divergent filter."""
    from secondbrain import weekly_letter
    src = inspect.getsource(weekly_letter)
    # Either bare ref or _db.EMAIL_KIND_SQL — accept both.
    assert "EMAIL_KIND_SQL" in src
    assert "TRANSCRIPT_KIND_SQL" in src


def test_weekly_letter_emails_count_picks_up_kind_email(fresh_db):
    """The shared constant matches kind='email' rows that the round-24
    inline filter missed."""
    from secondbrain import weekly_letter

    # Seed a row with the legacy ``kind='email'`` shape.
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, content_hash, "
        "indexed_at) VALUES "
        "('legacy_email.eml', 0, 0, 'email', 'h', ?)",
        (time.time(),),
    )
    # And one with the imap:// path shape.
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, content_hash, "
        "indexed_at) VALUES "
        "('imap://INBOX/1', 0, 0, 'url', 'h2', ?)",
        (time.time(),),
    )
    fresh_db.commit()
    week_cutoff = time.time() - 7 * 86400
    out = weekly_letter._signal_counts(fresh_db, week_cutoff)
    assert out["emails"] == 2, (
        f"expected both legacy + imap rows; got {out['emails']}"
    )


# ============================ smoke =====================================


def test_round26_modules_import():
    """Sanity: every touched module still imports."""
    from secondbrain import (  # noqa: F401
        cli,
        dashboard,
        notifications,
        tasks,
        timeline,
        vault_export,
        weekly_letter,
    )


@pytest.mark.parametrize("path,expected", [
    ("transcript://m1", "transcripts"),
    ("voice://v1", "transcripts"),
    ("capture://c1", "captures"),
    ("canvas://x", "canvas"),
    ("oura://2026-05-05", "health"),
    ("review://2026W18", "reviews"),
    ("journal://2026-05-05", "journal"),
    ("imap://INBOX/1", "notes"),
    ("gmail://thread", "notes"),
    ("slack://msg", "captures"),
    ("linear://ENG-1", "captures"),
    ("github://repo/issue", "captures"),
    ("notion://page", "captures"),
    ("obsidian://vault/note", "captures"),
    ("pocket://item", "captures"),
    ("readwise://hl", "captures"),
])
def test_vault_classify_table(path, expected):
    """Parametric coverage of every prefix _classify now handles."""
    from secondbrain.vault_export import _classify
    assert _classify(path, "document") == expected
