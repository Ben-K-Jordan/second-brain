"""Round 10 tests — fixes for the 10 audit downfalls.

Coverage:
  - Privacy: prompt-side redaction via _safe_for_prompt (#4)
  - Audit log: ai_audit table + wired through _llm_json_call (#6)
  - Cold-start voice: default profile + critique-skip (#5)
  - Health checks: per-integration ping + stale detection (#9)
  - Draft feedback: accept/reject + accept_rate stats (#2)
  - Per-draft cost: ai_audit.cost_for_window (#3)
  - Legacy drafter cleanup: string-returning stubs still work (#7)
"""

from __future__ import annotations

import json
import time
from unittest.mock import patch

from secondbrain import ai_audit, email_assist, health_checks

# ============================ #4 — prompt-side redaction =============

def test_safe_for_prompt_redacts_api_keys():
    """Phase 88 patterns get masked before the prompt is built."""
    out = email_assist._safe_for_prompt(
        "API key is sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        max_chars=5000,
    )
    assert "sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" not in out
    assert "API key is" in out


def test_safe_for_prompt_truncates_after_redact():
    """Order matters: redact first, then truncate. The truncation
    suffix '…' lands at the cap regardless."""
    long_text = "x" * 5000
    out = email_assist._safe_for_prompt(long_text, max_chars=100)
    assert len(out) == 101  # 100 chars + the … suffix
    assert out.endswith("…")


def test_safe_for_prompt_handles_empty_and_none():
    assert email_assist._safe_for_prompt("", max_chars=100) == ""
    assert email_assist._safe_for_prompt(None, max_chars=100) == ""


def test_safe_for_prompt_preserves_short_input():
    """Short clean input passes through unchanged."""
    assert email_assist._safe_for_prompt(
        "hello world", max_chars=100,
    ) == "hello world"


# ============================ #6 — AI audit log ======================

def test_ai_audit_schema_creates_table(fresh_db):
    ai_audit._ensure_schema(fresh_db)
    rows = fresh_db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='ai_actions'",
    ).fetchall()
    assert any(r["name"] == "ai_actions" for r in rows)


def test_ai_audit_record_action_persists(fresh_db):
    rid = ai_audit.record_action(
        fresh_db,
        kind="draft", feature="email_draft",
        model="claude-sonnet-4-6", status="success",
        prompt_chars=2400, response_chars=400, cents=2.5,
        summary="drafted reply to boss@x",
    )
    assert rid > 0
    actions = ai_audit.recent(fresh_db, limit=10)
    assert len(actions) == 1
    a = actions[0]
    assert a.kind == "draft"
    assert a.feature == "email_draft"
    assert a.cents == 2.5
    assert a.summary == "drafted reply to boss@x"


def test_ai_audit_record_action_swallows_errors(fresh_db):
    """Bad-shape input shouldn't crash the calling LLM pipeline.
    The audit logger is best-effort."""
    # Force an error by passing a non-serializable object as extra.
    rid = ai_audit.record_action(
        fresh_db,
        kind="draft", feature="email_draft",
        extra={"obj": object()},
    )
    # Returns 0 on failure but doesn't raise.
    assert rid == 0


def test_ai_audit_recent_filters_by_kind(fresh_db):
    for kind in ["draft", "draft", "analyze", "summary"]:
        ai_audit.record_action(
            fresh_db, kind=kind, feature=kind, model="m",
        )
    drafts = ai_audit.recent(fresh_db, kind="draft")
    assert len(drafts) == 2
    summaries = ai_audit.recent(fresh_db, kind="summary")
    assert len(summaries) == 1


def test_ai_audit_by_kind_aggregates(fresh_db):
    for kind in ["draft", "draft", "draft", "analyze"]:
        ai_audit.record_action(
            fresh_db, kind=kind, feature=kind, model="m",
        )
    counts = ai_audit.by_kind(fresh_db, days=30)
    assert counts["draft"] == 3
    assert counts["analyze"] == 1


def test_ai_audit_rollup_today(fresh_db):
    ai_audit.record_action(
        fresh_db, kind="draft", feature="email_draft", cents=1.5,
        prompt_chars=1000, response_chars=200,
    )
    ai_audit.record_action(
        fresh_db, kind="analyze", feature="email_analyze", cents=0.3,
        prompt_chars=500, response_chars=100,
    )
    rollup = ai_audit.rollup_today(fresh_db)
    assert "email_draft" in rollup
    assert rollup["email_draft"]["cents"] == 1.5
    assert rollup["email_analyze"]["n"] == 1


def test_ai_audit_trim_old_drops_ancient_rows(fresh_db):
    ai_audit._ensure_schema(fresh_db)
    fresh_db.execute(
        "INSERT INTO ai_actions(ts, kind, feature, status) "
        "VALUES (?, 'draft', 'email_draft', 'success')",
        (time.time() - 100 * 86400,),
    )
    fresh_db.execute(
        "INSERT INTO ai_actions(ts, kind, feature, status) "
        "VALUES (?, 'draft', 'email_draft', 'success')",
        (time.time() - 1 * 86400,),
    )
    fresh_db.commit()
    n = ai_audit.trim_old(fresh_db, keep_days=30)
    assert n == 1  # the 100-day-old row got dropped
    assert len(ai_audit.recent(fresh_db, limit=100)) == 1


def test_llm_json_call_records_audit_on_success(fresh_db, tmp_cfg, monkeypatch):
    """When conn is passed to _llm_json_call, every call writes
    one ai_actions row with the right status."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Force the local-LLM fallback path with a stub.
    from secondbrain import local_llm
    fake = local_llm.LocalCompletion(
        text=json.dumps({"k": "v"}), model="llama3.1",
        prompt_tokens=10, completion_tokens=5,
    )
    with patch.object(local_llm, "is_available", return_value=True), \
         patch.object(local_llm, "complete", return_value=fake):
        out = email_assist._llm_json_call(
            prompt="test prompt", cfg=tmp_cfg,
            model="claude-sonnet-4-6", max_tokens=100,
            feature="test_feature", note="t",
            conn=fresh_db,
            audit_kind="test",
            audit_summary="test call",
        )
    assert out == {"k": "v"}
    actions = ai_audit.recent(fresh_db, kind="test")
    assert len(actions) == 1
    assert actions[0].status == "fallback_local"
    assert actions[0].cents == 0.0  # no Anthropic call → no cost
    assert actions[0].prompt_chars == len("test prompt")


def test_llm_json_call_no_conn_skips_audit(fresh_db, tmp_cfg, monkeypatch):
    """Without conn, no audit row gets written (no-op gracefully)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from secondbrain import local_llm
    fake = local_llm.LocalCompletion(
        text=json.dumps({"k": "v"}), model="llama3.1",
        prompt_tokens=10, completion_tokens=5,
    )
    with patch.object(local_llm, "is_available", return_value=True), \
         patch.object(local_llm, "complete", return_value=fake):
        email_assist._llm_json_call(
            prompt="test", cfg=tmp_cfg,
            model="claude-sonnet-4-6", max_tokens=100,
            feature="test_feature", note="t",
            # no conn passed
        )
    # No actions should have landed.
    ai_audit._ensure_schema(fresh_db)
    assert ai_audit.recent(fresh_db) == []


def test_llm_json_call_records_parse_error(fresh_db, tmp_cfg, monkeypatch):
    """Garbage LLM output → status='parse_error' in audit log."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from secondbrain import local_llm
    fake = local_llm.LocalCompletion(
        text="not valid json",
        model="llama3.1", prompt_tokens=10, completion_tokens=5,
    )
    with patch.object(local_llm, "is_available", return_value=True), \
         patch.object(local_llm, "complete", return_value=fake):
        out = email_assist._llm_json_call(
            prompt="test", cfg=tmp_cfg,
            model="claude-sonnet-4-6", max_tokens=100,
            feature="test_feature", note="t",
            conn=fresh_db, audit_kind="test",
        )
    assert out is None
    actions = ai_audit.recent(fresh_db)
    assert any(a.status == "parse_error" for a in actions)


# ============================ #5 — cold-start voice fallback =========

def test_default_voice_profile_is_curated():
    """Bootstrap profile has the expected shape: real greeting/signoff
    patterns + the LLM-ism avoid list + n_samples=0 marker."""
    p = email_assist.default_voice_profile()
    assert p.n_samples == 0
    assert "hi {name}," in p.greetings
    assert "I hope this email finds you well" in p.avoided_phrases
    assert "Best regards," in p.avoided_phrases
    assert p.contraction_rate > 0.5  # leans casual


def test_get_voice_profile_or_default_returns_bootstrap_when_empty(fresh_db):
    """No saved profile → returns the curated default rather than None."""
    p = email_assist.get_voice_profile_or_default(fresh_db)
    assert p is not None
    assert p.n_samples == 0
    assert "I hope this email finds you well" in p.avoided_phrases


def test_get_voice_profile_or_default_returns_real_when_present(fresh_db):
    """When a real profile exists, it wins over the default."""
    real = email_assist.VoiceProfile(
        greetings=["yo {name}!"], sign_offs=["xo {name}"],
        avg_sentence_words=15, avg_reply_chars=300,
        contraction_rate=0.95, exclamation_rate=2, emoji_rate=1,
        common_openers=[], common_closers=[],
        avoided_phrases=[], register_notes="extremely casual",
        n_samples=42,
    )
    email_assist._save_voice_profile(fresh_db, real)
    p = email_assist.get_voice_profile_or_default(fresh_db)
    assert p.n_samples == 42
    assert "yo {name}!" in p.greetings


# ============================ #2 — feedback loop =====================

def _seed_draft(conn, fid=1, *, feedback=None):
    """Helper: insert a draft row directly. Bypasses generate_draft
    so we can pin feedback state."""
    email_assist._ensure_schema(conn)
    cur = conn.execute(
        "INSERT INTO email_drafts"
        "(file_id, draft_text, generated_at, feedback) "
        "VALUES (?, ?, ?, ?) RETURNING id",
        (fid, "draft text", time.time(), feedback),
    )
    return int(cur.fetchone()["id"])


def test_feedback_stats_empty_db_returns_zeros(fresh_db):
    stats = email_assist.feedback_stats(fresh_db)
    assert stats["accepted"] == 0
    assert stats["rejected"] == 0
    assert stats["accept_rate"] == 0.0


def test_feedback_stats_counts_by_label(fresh_db):
    # Need a real file row for the FK.
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at) "
        "VALUES ('x', ?, 1, 'document', ?)",
        (time.time(), time.time()),
    )
    fid = fresh_db.execute("SELECT id FROM files").fetchone()["id"]
    _seed_draft(fresh_db, fid, feedback="accepted")
    _seed_draft(fresh_db, fid, feedback="accepted")
    _seed_draft(fresh_db, fid, feedback="rejected")
    _seed_draft(fresh_db, fid, feedback=None)
    stats = email_assist.feedback_stats(fresh_db, days=30)
    assert stats["accepted"] == 2
    assert stats["rejected"] == 1
    assert stats["pending"] == 1
    assert abs(stats["accept_rate"] - 2 / 3) < 0.01


def test_mark_draft_sent_flips_feedback(fresh_db):
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at) "
        "VALUES ('x', ?, 1, 'document', ?)",
        (time.time(), time.time()),
    )
    fid = fresh_db.execute("SELECT id FROM files").fetchone()["id"]
    did = _seed_draft(fresh_db, fid)
    assert email_assist.mark_draft_sent(fresh_db, did) is True
    row = fresh_db.execute(
        "SELECT feedback, sent_at FROM email_drafts WHERE id = ?",
        (did,),
    ).fetchone()
    assert row["feedback"] == "accepted"
    assert row["sent_at"] is not None


def test_discard_draft_now_soft_deletes_with_feedback(fresh_db):
    """Round 10 (#2) — discard is now a soft-delete that flags
    feedback='rejected'. List-unsent excludes rejected rows."""
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at) "
        "VALUES ('x', ?, 1, 'document', ?)",
        (time.time(), time.time()),
    )
    fid = fresh_db.execute("SELECT id FROM files").fetchone()["id"]
    did = _seed_draft(fresh_db, fid)
    assert email_assist.discard_draft(
        fresh_db, did, reason="too formal",
    ) is True
    row = fresh_db.execute(
        "SELECT feedback, rejection_reason, sent_at "
        "FROM email_drafts WHERE id = ?",
        (did,),
    ).fetchone()
    # Soft-delete: row still exists.
    assert row is not None
    assert row["feedback"] == "rejected"
    assert row["rejection_reason"] == "too formal"
    assert row["sent_at"] is None
    # Still excluded from unsent list.
    assert email_assist.list_unsent_drafts(fresh_db) == []


# ============================ #3 — per-draft cost ====================

def test_cost_for_window_sums_calls_around_timestamp(fresh_db):
    target_ts = time.time() - 100
    # Three calls within window, two outside.
    for offset, cents in [(-5, 0.5), (0, 1.0), (5, 0.3)]:
        ai_audit.record_action(
            fresh_db, kind="draft", feature="email_draft",
            file_id=42, cents=cents,
        )
        # record_action stamps ts=now; rewrite to controlled ts.
        fresh_db.execute(
            "UPDATE ai_actions SET ts = ? WHERE id = (SELECT MAX(id) FROM ai_actions)",
            (target_ts + offset,),
        )
    for offset, cents in [(-200, 5.0), (200, 5.0)]:
        ai_audit.record_action(
            fresh_db, kind="draft", feature="email_draft",
            file_id=42, cents=cents,
        )
        fresh_db.execute(
            "UPDATE ai_actions SET ts = ? WHERE id = (SELECT MAX(id) FROM ai_actions)",
            (target_ts + offset,),
        )
    fresh_db.commit()
    cost = ai_audit.cost_for_window(
        fresh_db, file_id=42, around_ts=target_ts, window_seconds=90,
    )
    # Three within ±90s, total cents = 0.5 + 1.0 + 0.3 = 1.8
    assert cost["n"] == 3
    assert abs(cost["cents"] - 1.8) < 0.01


def test_cost_for_window_filters_by_file_id(fresh_db):
    """Only the requested file's actions count."""
    ai_audit.record_action(
        fresh_db, kind="draft", feature="email_draft",
        file_id=42, cents=1.0,
    )
    ai_audit.record_action(
        fresh_db, kind="draft", feature="email_draft",
        file_id=99, cents=10.0,
    )
    cost = ai_audit.cost_for_window(
        fresh_db, file_id=42, around_ts=time.time(),
        window_seconds=300,
    )
    assert cost["cents"] == 1.0  # not 11.0


# ============================ #9 — health checks =====================

def test_health_check_schema(fresh_db):
    health_checks._ensure_schema(fresh_db)
    rows = fresh_db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name='health_checks'",
    ).fetchall()
    assert any(r["name"] == "health_checks" for r in rows)


def test_check_anthropic_key_unset(tmp_cfg, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ok, err, _ = health_checks.check_anthropic_key(tmp_cfg)
    assert ok is False
    assert "not set" in err


def test_check_anthropic_key_wrong_shape(tmp_cfg, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "definitely-not-an-anthropic-key")
    ok, err, _ = health_checks.check_anthropic_key(tmp_cfg)
    assert ok is False
    assert "shape" in err


def test_check_anthropic_key_valid_shape(tmp_cfg, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fake-but-shape-OK")
    ok, _err, extra = health_checks.check_anthropic_key(tmp_cfg)
    assert ok is True
    assert extra["key_prefix"].startswith("sk-ant-api")


def test_check_voyage_key_shape(tmp_cfg, monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    tmp_cfg.voyage_api_key = ""
    ok, _err, _ = health_checks.check_voyage_key(tmp_cfg)
    assert ok is False


def test_check_watched_folders_missing(tmp_cfg, monkeypatch):
    from pathlib import Path
    tmp_cfg.watched_folders = [Path("/nonexistent/path/xyzzy")]
    ok, err, extra = health_checks.check_watched_folders(tmp_cfg)
    assert ok is False
    assert "missing" in err
    assert extra["missing"]


def test_check_watched_folders_empty_is_ok(tmp_cfg):
    """No folders configured isn't a failure — just unconfigured."""
    tmp_cfg.watched_folders = []
    ok, _err, extra = health_checks.check_watched_folders(tmp_cfg)
    assert ok is True
    assert extra["configured"] is False


def test_run_all_persists_results(fresh_db, tmp_cfg, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    tmp_cfg.voyage_api_key = ""
    statuses = health_checks.run_all(fresh_db, tmp_cfg)
    # All registered checks should have a status.
    assert "anthropic" in statuses
    assert "voyage" in statuses
    assert "watched_folders" in statuses
    # The two key-missing ones should be failing.
    assert statuses["anthropic"].ok is False
    assert statuses["voyage"].ok is False


def test_stale_failures_filters_recent_ok_runs(fresh_db, tmp_cfg, monkeypatch):
    """A check that's currently OK isn't 'stale', regardless of age."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test")
    health_checks.run_all(fresh_db, tmp_cfg)
    stale = health_checks.stale_failures(fresh_db)
    # Anthropic should be ok (key shape valid) — not stale.
    names = {s.name for s in stale}
    assert "anthropic" not in names


def test_health_status_days_since_ok(fresh_db, tmp_cfg, monkeypatch):
    """Once a check succeeds and then starts failing, days_since_ok
    counts up."""
    # First run: anthropic ok.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test")
    health_checks.run_all(fresh_db, tmp_cfg)
    # Second run: key revoked.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    health_checks.run_all(fresh_db, tmp_cfg)
    # Manually rewind the last_ok_at to simulate "5 days ago".
    fresh_db.execute(
        "UPDATE health_checks SET last_ok_at = ? WHERE name = 'anthropic'",
        (time.time() - 5 * 86400,),
    )
    fresh_db.commit()
    st = health_checks.get_status(fresh_db, "anthropic")
    assert st is not None
    assert st.ok is False
    assert st.days_since_ok == 5


# ============================ #7 — legacy drafter cleanup ============

def test_coerce_draft_output_from_string():
    out = email_assist._coerce_draft_output("hello world")
    assert out is not None
    assert out.primary == "hello world"
    assert out.alternative == ""
    assert out.confidence == 0.0


def test_coerce_draft_output_from_dict():
    raw = {
        "primary": "p", "alternative": "a",
        "reasoning": "r", "confidence": 0.7,
        "open_questions": ["q1"],
    }
    out = email_assist._coerce_draft_output(raw)
    assert out.primary == "p"
    assert out.alternative == "a"
    assert out.confidence == 0.7
    assert out.open_questions == ["q1"]


def test_coerce_draft_output_passes_through_DraftOutput():
    obj = email_assist.DraftOutput(
        primary="x", alternative="", reasoning="",
        confidence=0.5, open_questions=[],
    )
    assert email_assist._coerce_draft_output(obj) is obj


def test_coerce_draft_output_handles_empty():
    assert email_assist._coerce_draft_output(None) is None
    assert email_assist._coerce_draft_output("") is None
    assert email_assist._coerce_draft_output("   ") is None


def test_legacy_drafter_lambda_still_works(fresh_db, tmp_cfg):
    """Round 10 (#7) — the old drafter=lambda **kw: 'text' pattern
    should still produce a Draft via the unified path."""
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at) "
        "VALUES ('imap://msgid/x', ?, 1, 'url', ?)",
        (time.time(), time.time()),
    )
    fid = fresh_db.execute("SELECT id FROM files").fetchone()["id"]
    fresh_db.execute(
        "INSERT INTO chunks(file_id, chunk_index, text) "
        "VALUES (?, 0, '# subject\nFrom: x\nFolder: INBOX\n\nbody')",
        (fid,),
    )
    email_assist._ensure_schema(fresh_db)
    fresh_db.execute(
        "INSERT INTO email_classifications"
        "(file_id, label, confidence, classified_at) "
        "VALUES (?, 'urgent', 0.9, ?)",
        (fid, time.time()),
    )
    fresh_db.commit()
    d = email_assist.generate_draft(
        fresh_db, fid, cfg=tmp_cfg,
        drafter=lambda **kw: "legacy reply text",
    )
    assert d is not None
    assert d.draft_text == "legacy reply text"


# ============================ Daemon job registration ===============

def test_daemon_registers_round10_jobs(fresh_db, tmp_cfg):
    from secondbrain.daemon import _build_daemon_scheduler

    sched = _build_daemon_scheduler(
        tmp_cfg, fresh_db, embedder=None, reranker=None,
    )
    names = set(sched.names())
    assert "ai_audit_trim" in names
    assert "health_checks" in names
