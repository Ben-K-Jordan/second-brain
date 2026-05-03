"""Tests for polish-v3 round-4 wiring (the audit gap fixes):

- Local LLM fallback in synthesis / email_assist / chat
- Sensitive content redaction in daily_brief / digest / MCP
- New MCP tools (get_summary / list_insights / list_snapshots /
  as_of_search / local_llm_status)
- New daemon scheduler jobs (connector_sync / tasks_from_transcripts)
- New CLI commands (snapshot diff / projects promote / insights dismiss)
- New daily-brief sections (birthdays / annotations / nudges)

These tests focus on **wiring** — that the integration paths exist
and route correctly. Per-feature internals are tested in their
respective test files.
"""

from __future__ import annotations

import time
from unittest.mock import patch

# ============================ HIGH-1: Local LLM fallback ==============

def test_synthesis_falls_back_to_local_when_no_api_key(
    fresh_db, tmp_cfg, monkeypatch,
):
    """When ANTHROPIC_API_KEY is missing AND Ollama is available,
    the summary generator returns a TL;DR from the local model."""
    from secondbrain import local_llm, synthesis

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    # Stub the local LLM to return a fake completion.
    fake_out = local_llm.LocalCompletion(
        text="A one-sentence summary about the doc.",
        model="llama3.1",
        prompt_tokens=42,
        completion_tokens=10,
    )
    with patch.object(local_llm, "is_available", return_value=True), \
         patch.object(local_llm, "complete", return_value=fake_out):
        result = synthesis._default_summary_generator(
            "Test doc", "Body text " * 100, tmp_cfg,
        )
    assert result.get("tldr") == "A one-sentence summary about the doc."
    # local fallback can't reliably emit JSON, so key_points is empty
    assert result.get("key_points") == []


def test_synthesis_returns_empty_when_neither_path_works(
    fresh_db, tmp_cfg, monkeypatch,
):
    from secondbrain import local_llm, synthesis

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with patch.object(local_llm, "is_available", return_value=False):
        result = synthesis._default_summary_generator(
            "Test", "body", tmp_cfg,
        )
    assert result == {}


def test_email_classifier_falls_back_to_local(tmp_cfg, monkeypatch):
    """When Anthropic is unavailable, the classifier asks local for
    a single label string and parses it into the standard shape."""
    from secondbrain import email_assist, local_llm

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake_out = local_llm.LocalCompletion(
        text="urgent\n",  # leading whitespace tolerated
        model="llama3.1", prompt_tokens=20, completion_tokens=2,
    )
    with patch.object(local_llm, "is_available", return_value=True), \
         patch.object(local_llm, "complete", return_value=fake_out):
        out = email_assist._default_classifier(
            "boss@example.com", "URGENT: budget gap",
            "We need to discuss the Q3 numbers ASAP.",
            tmp_cfg,
        )
    assert out.get("label") == "urgent"
    assert out.get("rationale") == "local-llm"


def test_email_classifier_rejects_garbage_local_label(tmp_cfg, monkeypatch):
    """If local LLM returns something that isn't one of our 5 labels,
    we drop it on the floor rather than persisting nonsense."""
    from secondbrain import email_assist, local_llm

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake_out = local_llm.LocalCompletion(
        text="probably-spam-but-not-sure",
        model="llama3.1", prompt_tokens=20, completion_tokens=4,
    )
    with patch.object(local_llm, "is_available", return_value=True), \
         patch.object(local_llm, "complete", return_value=fake_out):
        out = email_assist._default_classifier(
            "x@y", "subj", "body", tmp_cfg,
        )
    assert out == {}


def test_email_drafter_falls_back_to_local(tmp_cfg, monkeypatch):
    """Round-6 update: the structured drafter expects JSON output. A
    capable local model that returns valid JSON gives us a real
    DraftOutput; less-capable ones return None and the caller skips."""
    import json as _json

    from secondbrain import email_assist, local_llm

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    payload = _json.dumps({
        "primary": "Tomorrow works for me — let's say 2pm.",
        "alternative": "Tomorrow at 2pm sounds great!",
        "reasoning": "Casual scheduling reply.",
        "confidence": 0.7,
        "open_questions": [],
    })
    fake_out = local_llm.LocalCompletion(
        text=payload,
        model="llama3.1", prompt_tokens=200, completion_tokens=80,
    )
    with patch.object(local_llm, "is_available", return_value=True), \
         patch.object(local_llm, "complete", return_value=fake_out):
        out = email_assist._default_drafter(
            from_="x@y", subject="hello", body="when do you have time?",
            style_samples="", user_name="Ben", cfg=tmp_cfg,
        )
    assert out is not None
    assert "Tomorrow" in out.primary
    assert "Tomorrow at 2pm" in out.alternative


def test_chat_local_fallback_helper_returns_response(
    fresh_db, tmp_cfg, fake_embedder, monkeypatch,
):
    """The standalone _ask_brain_local_fallback should produce a
    citation-bearing answer when local is up but Anthropic isn't."""
    from secondbrain import chat as chat_mod
    from secondbrain import local_llm

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake_out = local_llm.LocalCompletion(
        text="Based on your notes, the answer is X.",
        model="llama3.1", prompt_tokens=300, completion_tokens=12,
    )
    with patch.object(local_llm, "is_available", return_value=True), \
         patch.object(local_llm, "complete", return_value=fake_out):
        resp = chat_mod._ask_brain_local_fallback(
            tmp_cfg, fresh_db, fake_embedder,
            reranker=None, question="what was X?",
        )
    assert resp is not None
    assert "Based on your notes" in resp.text
    assert "answered locally" in resp.text


def test_chat_local_fallback_returns_none_when_local_down(
    fresh_db, tmp_cfg, fake_embedder,
):
    from secondbrain import chat as chat_mod
    from secondbrain import local_llm

    with patch.object(local_llm, "is_available", return_value=False):
        resp = chat_mod._ask_brain_local_fallback(
            tmp_cfg, fresh_db, fake_embedder,
            reranker=None, question="anything",
        )
    assert resp is None


# ============================ HIGH-2: Brief / digest redaction ========

def test_daily_brief_redacts_action_item_text():
    """Action items containing API-key-like patterns get masked
    before they hit the rendered markdown."""
    from secondbrain.daily_brief import (
        ActionItem,
        DailyBrief,
        format_markdown,
    )

    secret_text = "Update sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA in vault"
    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[],
        assignments_due_soon=[],
        open_action_items=[
            ActionItem(
                text=secret_text,
                source_path="t://x", source_title="meeting",
                task_id=1, age_days=0,
            ),
        ],
        queue_top=[], watchlist_highlights=[],
    )
    md = format_markdown(brief, header_date="Today")
    # The secret should NOT appear verbatim
    assert "sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" not in md
    # The non-secret context should still be there
    assert "Update" in md
    assert "in vault" in md


def test_daily_brief_redacts_queue_summary():
    from secondbrain.daily_brief import (
        DailyBrief,
        QueueItem,
        format_markdown,
    )

    brief = DailyBrief(
        generated_at=time.time(), today_events=[],
        assignments_due_soon=[], open_action_items=[],
        queue_top=[
            QueueItem(
                queue_id=1,
                url="https://x",
                title="An article",
                summary="Token AKIAIOSFODNN7EXAMPLE was committed.",
            ),
        ],
        watchlist_highlights=[],
    )
    md = format_markdown(brief, header_date="Today")
    assert "AKIAIOSFODNN7EXAMPLE" not in md


def test_digest_redacts_watchlist_answer(fresh_db, tmp_cfg):
    """The watchlist's Claude-generated answer gets sanitised
    before being rendered into the digest text body."""
    from secondbrain import digest

    rows = [{
        "watchlist": {
            "name": "test", "query": "q", "schedule_minutes": 60,
        },
        "all_new_paths": [],
        "latest_answer": (
            "Found AKIAIOSFODNN7EXAMPLE in the new doc — flag this."
        ),
        "latest_error": None,
        "run_count": 1,
    }]
    text = digest._render_text(rows, since_ts=time.time() - 3600)
    assert "AKIAIOSFODNN7EXAMPLE" not in text
    html = digest._render_html(rows, since_ts=time.time() - 3600)
    assert "AKIAIOSFODNN7EXAMPLE" not in html


# ============================ HIGH-3: MCP redaction ===================

def test_mcp_safe_helper_redacts():
    """The mcp_server's `_safe` shim wraps safety.redact_text."""
    from secondbrain import mcp_server

    secret = "AKIAIOSFODNN7EXAMPLE"
    out = mcp_server._safe(f"Token: {secret}")
    assert secret not in out


def test_mcp_safe_handles_none_and_empty():
    from secondbrain import mcp_server

    assert mcp_server._safe(None) == ""
    assert mcp_server._safe("") == ""


# ============================ MED-4: New MCP tools ====================

def test_mcp_tool_module_exposes_new_tools():
    """Every new MCP tool should be importable as a function on the
    module so the FastMCP decorator captured them at import time."""
    from secondbrain import mcp_server

    for name in (
        "get_summary", "list_insights", "list_snapshots",
        "as_of_search", "local_llm_status",
    ):
        assert hasattr(mcp_server, name), f"Missing MCP tool: {name}"
        assert callable(getattr(mcp_server, name))


# ============================ MED-6: Daemon scheduler =================

def test_daemon_scheduler_includes_connector_sync(fresh_db, tmp_cfg):
    from secondbrain.daemon import _build_daemon_scheduler

    sched = _build_daemon_scheduler(
        tmp_cfg, fresh_db, embedder=None, reranker=None,
    )
    assert "connector_sync" in set(sched.names())


def test_daemon_scheduler_includes_tasks_from_transcripts(
    fresh_db, tmp_cfg,
):
    from secondbrain.daemon import _build_daemon_scheduler

    sched = _build_daemon_scheduler(
        tmp_cfg, fresh_db, embedder=None, reranker=None,
    )
    assert "tasks_from_transcripts" in set(sched.names())


def test_run_sync_due_skips_when_no_connectors_configured(
    fresh_db, tmp_cfg,
):
    """With no env vars set + no config, every connector reports
    is_enabled=False; run_sync_due returns 0 and doesn't crash."""
    from secondbrain.sync import run_sync_due

    # No connectors configured → 0 indexed, no exceptions.
    result = run_sync_due(tmp_cfg, fresh_db, embedder=None)
    assert result == 0


# ============================ MED-7: CLI commands =====================

def test_cli_has_snapshot_diff_command():
    """The Typer subapp `snapshot` should expose a `diff` command."""
    from secondbrain.cli import snapshot_app

    cmd_names = {c.name for c in snapshot_app.registered_commands}
    assert "diff" in cmd_names


def test_cli_has_insights_dismiss_command():
    from secondbrain.cli import insights_app

    cmd_names = {c.name for c in insights_app.registered_commands}
    assert "dismiss" in cmd_names


def test_cli_has_projects_promote_command():
    from secondbrain.cli import projects_smart_app

    cmd_names = {c.name for c in projects_smart_app.registered_commands}
    assert "promote" in cmd_names


# ============================ MED-8: Birthdays in brief ==============

def test_parse_birthday_iso_with_year():
    from secondbrain.daily_brief import _parse_birthday

    m, d, year_known = _parse_birthday("1990-05-12")
    assert m == 5
    assert d == 12
    assert year_known is True


def test_parse_birthday_short_form():
    from secondbrain.daily_brief import _parse_birthday

    m, d, year_known = _parse_birthday("05-12")
    assert m == 5
    assert d == 12
    assert year_known is False


def test_parse_birthday_slash_separator():
    from secondbrain.daily_brief import _parse_birthday

    m, d, _ = _parse_birthday("12/25")
    assert m == 12
    assert d == 25


def test_parse_birthday_garbage_returns_none():
    from secondbrain.daily_brief import _parse_birthday

    assert _parse_birthday("not a date") == (None, None, False)
    assert _parse_birthday("") == (None, None, False)


def test_birthdays_section_surfaces_today(fresh_db):
    """A person whose birthday is today (any year) should appear
    with is_today=True."""
    from datetime import datetime

    from secondbrain import people as people_mod
    from secondbrain.daily_brief import _birthdays_section

    today = datetime.now().date()
    bday_str = f"{today.month:02d}-{today.day:02d}"
    pid = people_mod.upsert_person(fresh_db, display_name="Test Person")
    people_mod.set_field(fresh_db, pid, birthday=bday_str)
    out = _birthdays_section(fresh_db)
    matching = [b for b in out if b.name == "Test Person"]
    assert len(matching) == 1
    assert matching[0].is_today is True
    assert matching[0].days_until == 0


def test_birthdays_section_skips_far_future(fresh_db):
    """A birthday more than 7 days away shouldn't show in the brief."""
    from datetime import datetime, timedelta

    from secondbrain import people as people_mod
    from secondbrain.daily_brief import _birthdays_section

    far = datetime.now().date() + timedelta(days=30)
    bday_str = f"{far.month:02d}-{far.day:02d}"
    pid = people_mod.upsert_person(fresh_db, display_name="Far Future")
    people_mod.set_field(fresh_db, pid, birthday=bday_str)
    out = _birthdays_section(fresh_db)
    assert all(b.name != "Far Future" for b in out)


def test_brief_renders_birthday_today_with_cake():
    from secondbrain.daily_brief import (
        BirthdayLine,
        DailyBrief,
        format_markdown,
    )

    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[], assignments_due_soon=[], open_action_items=[],
        queue_top=[], watchlist_highlights=[],
        birthdays=[
            BirthdayLine(
                name="Sarah", days_until=0, age_turning=30, is_today=True,
            ),
        ],
    )
    md = format_markdown(brief, header_date="Today")
    assert "🎂" in md
    assert "Sarah" in md
    assert "turning 30" in md


# ============================ MED-9: Annotations in brief ============

def test_recent_annotations_section_returns_recent_only(fresh_db):
    """Only annotations created within the lookback window appear."""
    from secondbrain.daily_brief import _recent_annotations_section

    # Seed a file + 2 annotations: one fresh, one old.
    cur = fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at) "
        "VALUES (?, ?, 1, 'document', ?)",
        ("/test.pdf", time.time(), time.time()),
    )
    fid = cur.lastrowid
    fresh_db.execute(
        "INSERT INTO pdf_annotations"
        "(file_id, page, kind, anchor, note, color, created_at) "
        "VALUES (?, 3, 'highlight', 'fresh highlight text', NULL, NULL, ?)",
        (fid, time.time() - 3600),
    )
    fresh_db.execute(
        "INSERT INTO pdf_annotations"
        "(file_id, page, kind, anchor, note, color, created_at) "
        "VALUES (?, 5, 'highlight', 'old highlight', NULL, NULL, ?)",
        (fid, time.time() - 14 * 86400),
    )
    fresh_db.commit()
    out = _recent_annotations_section(fresh_db)
    anchors = [a.anchor for a in out]
    assert "fresh highlight text" in anchors
    assert "old highlight" not in anchors


def test_brief_renders_annotation_block():
    from secondbrain.daily_brief import (
        AnnotationLine,
        DailyBrief,
        format_markdown,
    )

    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[], assignments_due_soon=[], open_action_items=[],
        queue_top=[], watchlist_highlights=[],
        recent_annotations=[
            AnnotationLine(
                file_path="/p.pdf",
                page=7,
                kind="highlight",
                anchor="something I cared about",
                note="",
            ),
        ],
    )
    md = format_markdown(brief, header_date="Today")
    assert "Highlights from yesterday" in md
    assert "something I cared about" in md
    assert "p.7" in md  # page number rendered


# ============================ LOW-10/11: Nudges ======================

def test_nudges_render_when_flags_set():
    from secondbrain.daily_brief import DailyBrief, _render_nudges

    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[], assignments_due_soon=[], open_action_items=[],
        queue_top=[], watchlist_highlights=[],
        weekly_review_due=True,
        snapshot_due=True,
    )
    out = _render_nudges(brief)
    assert "Weekly review is overdue" in out
    assert "snapshot in the last 7 days" in out


def test_nudges_empty_when_nothing_due():
    from secondbrain.daily_brief import DailyBrief, _render_nudges

    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[], assignments_due_soon=[], open_action_items=[],
        queue_top=[], watchlist_highlights=[],
    )
    assert _render_nudges(brief) == ""


def test_snapshot_due_returns_true_on_fresh_brain(fresh_db):
    """A brain with no snapshots at all should report snapshot_due=True."""
    from secondbrain.daily_brief import _snapshot_due

    assert _snapshot_due(fresh_db) is True


def test_snapshot_due_returns_false_after_recent_snapshot(fresh_db):
    from secondbrain import memory as memory_mod
    from secondbrain.daily_brief import _snapshot_due

    memory_mod.take_snapshot(fresh_db)
    assert _snapshot_due(fresh_db) is False


# ============================ Cross-check: brief still renders =======

def test_full_brief_with_all_new_sections_renders_cleanly():
    """Smoke test — populate every new field and confirm the full
    brief markdown comes out with no Python errors."""
    from secondbrain.daily_brief import (
        AnnotationLine,
        BirthdayLine,
        DailyBrief,
        format_markdown,
    )

    brief = DailyBrief(
        generated_at=time.time(),
        today_events=[], assignments_due_soon=[], open_action_items=[],
        queue_top=[], watchlist_highlights=[],
        birthdays=[
            BirthdayLine(name="A", days_until=0, age_turning=None, is_today=True),
            BirthdayLine(name="B", days_until=3, age_turning=42, is_today=False),
        ],
        recent_annotations=[
            AnnotationLine(
                file_path="/x.pdf", page=1, kind="highlight",
                anchor="text", note="my note",
            ),
        ],
        weekly_review_due=True,
        snapshot_due=True,
    )
    md = format_markdown(brief, header_date="2026-05-02")
    assert "Birthdays this week" in md
    assert "Highlights from yesterday" in md
    assert "Nudges" in md
    # Quiet-day banner should NOT appear since birthdays count as actionable
    assert "Quiet day" not in md
