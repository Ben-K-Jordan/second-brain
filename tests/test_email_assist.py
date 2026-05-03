"""Phase 82 + 83: email triage + auto-draft tests.

Coverage:
  - classify_one: stubbed classifier, label validation, idempotence
  - classify_due: bounded fan-out, picks unclassified IMAP docs
  - generate_draft: only for draftable labels, persists, doesn't auto-fire
  - mark_draft_sent / discard_draft
  - label_counts aggregation
"""

from __future__ import annotations

import time

from secondbrain import email_assist

# ============================ helpers =================================

def _seed_email(
    conn, *, path, from_, subject, body,
    indexed_at=None, folder="INBOX",
):
    n = indexed_at or time.time()
    cur = conn.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at, content_hash) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (path, n, len(body), "url", n, None),
    )
    fid = cur.lastrowid
    full = (
        f"# {subject}\n\n"
        f"From: {from_}\n"
        f"Folder: {folder}\n\n"
        f"{body}"
    )
    conn.execute(
        "INSERT INTO chunks(file_id, chunk_index, text) VALUES (?, 0, ?)",
        (fid, full),
    )
    conn.commit()
    return fid


# ============================ classify ================================

def test_needs_classification_true_for_unclassified(fresh_db):
    fid = _seed_email(
        fresh_db, path="imap://msgid/abc",
        from_="recruiter@example.com", subject="quick chat",
        body="Hi, are you free Tue?",
    )
    assert email_assist.needs_classification(fresh_db, fid) is True


def test_classify_one_persists(fresh_db):
    fid = _seed_email(
        fresh_db, path="imap://msgid/abc",
        from_="recruiter@example.com", subject="interview Tue",
        body="Want to schedule a chat?",
    )
    cls = email_assist.classify_one(
        fresh_db, fid,
        classifier=lambda f, s, b, c: {
            "label": "response", "confidence": 0.85,
            "rationale": "scheduling ask",
        },
    )
    assert cls.label == "response"
    assert cls.confidence == 0.85
    # Persisted.
    fetched = email_assist.get_classification(fresh_db, fid)
    assert fetched.label == "response"


def test_classify_one_falls_back_for_unknown_label(fresh_db):
    """A hallucinated label should NOT poison the table."""
    fid = _seed_email(
        fresh_db, path="imap://msgid/x", from_="x", subject="x", body="x",
    )
    cls = email_assist.classify_one(
        fresh_db, fid,
        classifier=lambda f, s, b, c: {"label": "INVENTED", "confidence": 1.0},
    )
    # Falls back to informational.
    assert cls.label == "informational"


def test_classify_one_idempotent(fresh_db):
    fid = _seed_email(
        fresh_db, path="imap://msgid/x", from_="x", subject="x", body="x",
    )
    cls_a = email_assist.classify_one(
        fresh_db, fid,
        classifier=lambda f, s, b, c: {"label": "urgent", "confidence": 0.9},
    )
    cls_b = email_assist.classify_one(
        fresh_db, fid,
        classifier=lambda f, s, b, c: {"label": "newsletter", "confidence": 0.1},
    )
    # Second call returns None (already classified) — first wins.
    assert cls_a is not None
    assert cls_b is None
    assert email_assist.get_classification(fresh_db, fid).label == "urgent"


def test_classify_one_handles_classifier_failure(fresh_db):
    fid = _seed_email(
        fresh_db, path="imap://msgid/x", from_="x", subject="x", body="x",
    )
    def boom(f, s, b, c):
        raise RuntimeError("classifier down")
    cls = email_assist.classify_one(
        fresh_db, fid, classifier=boom,
    )
    assert cls is None
    # Nothing persisted.
    assert email_assist.get_classification(fresh_db, fid) is None


def test_classify_one_handles_empty_response(fresh_db):
    fid = _seed_email(
        fresh_db, path="imap://msgid/x", from_="x", subject="x", body="x",
    )
    cls = email_assist.classify_one(
        fresh_db, fid, classifier=lambda f, s, b, c: {},
    )
    assert cls is None


def test_classify_due_caps_per_tick(fresh_db):
    """Fan-out bounded so a daemon catching up doesn't blow budget."""
    for i in range(5):
        _seed_email(
            fresh_db, path=f"imap://msgid/{i}",
            from_=f"x{i}@x.com", subject=f"e{i}", body=f"body {i}",
        )
    n = email_assist.classify_due(
        fresh_db, cfg=None, max_per_tick=3,
        classifier=lambda f, s, b, c: {"label": "informational"},
    )
    assert n == 3


def test_classify_due_skips_already_classified(fresh_db):
    fid = _seed_email(
        fresh_db, path="imap://msgid/x",
        from_="x", subject="x", body="x",
    )
    # Pre-classify.
    email_assist.classify_one(
        fresh_db, fid,
        classifier=lambda f, s, b, c: {"label": "urgent"},
    )
    n = email_assist.classify_due(
        fresh_db, cfg=None,
        classifier=lambda f, s, b, c: {"label": "newsletter"},
    )
    assert n == 0


def test_classify_due_skips_old_emails(fresh_db):
    """Emails outside the lookback window shouldn't trigger triage."""
    _seed_email(
        fresh_db, path="imap://msgid/old",
        from_="x", subject="x", body="x",
        indexed_at=time.time() - 30 * 86400,
    )
    n = email_assist.classify_due(
        fresh_db, cfg=None,
        classifier=lambda f, s, b, c: {"label": "urgent"},
    )
    assert n == 0


def test_classify_due_skips_non_imap_paths(fresh_db):
    """Only imap:// paths are candidates."""
    _seed_email(
        fresh_db, path="C:/notes/x.md",
        from_="x", subject="x", body="x",
    )
    n = email_assist.classify_due(
        fresh_db, cfg=None,
        classifier=lambda f, s, b, c: {"label": "urgent"},
    )
    assert n == 0


def test_label_counts_aggregates(fresh_db):
    for i, lab in enumerate(
        ["urgent", "urgent", "response", "newsletter"],
    ):
        fid = _seed_email(
            fresh_db, path=f"imap://msgid/{i}",
            from_="x", subject="x", body="x",
        )
        email_assist.classify_one(
            fresh_db, fid,
            classifier=lambda f, s, b, c, lab=lab: {"label": lab},
        )
    counts = email_assist.label_counts(fresh_db, days=30)
    assert counts.get("urgent") == 2
    assert counts.get("response") == 1
    assert counts.get("newsletter") == 1


def test_parse_email_header_extracts_from_and_subject():
    body = (
        "# Hello there\n\n"
        "From: sarah@example.com\n"
        "Folder: Inbox\n\n"
        "Body content"
    )
    from_, subject = email_assist._parse_email_header(body)
    assert from_ == "sarah@example.com"
    assert subject == "Hello there"


# ============================ drafts ==================================

def _classify(conn, fid, label):
    email_assist.classify_one(
        conn, fid,
        classifier=lambda f, s, b, c, lab=label: {"label": lab},
    )


def test_needs_draft_only_for_draftable_labels(fresh_db):
    fid_urgent = _seed_email(
        fresh_db, path="imap://msgid/u", from_="x", subject="x", body="x",
    )
    fid_news = _seed_email(
        fresh_db, path="imap://msgid/n", from_="x", subject="x", body="x",
    )
    _classify(fresh_db, fid_urgent, "urgent")
    _classify(fresh_db, fid_news, "newsletter")
    assert email_assist.needs_draft(fresh_db, fid_urgent) is True
    assert email_assist.needs_draft(fresh_db, fid_news) is False


def test_needs_draft_false_when_unclassified(fresh_db):
    fid = _seed_email(
        fresh_db, path="imap://msgid/x", from_="x", subject="x", body="x",
    )
    assert email_assist.needs_draft(fresh_db, fid) is False


def test_generate_draft_persists(fresh_db):
    fid = _seed_email(
        fresh_db, path="imap://msgid/u",
        from_="recruiter@x.com",
        subject="Quick chat",
        body="Free Tuesday for a 30 min call?",
    )
    _classify(fresh_db, fid, "urgent")
    draft = email_assist.generate_draft(
        fresh_db, fid,
        drafter=lambda **kw: (
            "Sounds good! Tuesday 2pm works. -- Ben"
        ),
    )
    assert draft is not None
    assert "Tuesday" in draft.draft_text
    # Persisted.
    drafts = email_assist.list_unsent_drafts(fresh_db)
    assert len(drafts) == 1
    assert drafts[0].id == draft.id


def test_generate_draft_skips_when_already_unsent(fresh_db):
    """Don't re-generate when there's already a draft awaiting review."""
    fid = _seed_email(
        fresh_db, path="imap://msgid/u", from_="x", subject="x", body="x",
    )
    _classify(fresh_db, fid, "response")
    email_assist.generate_draft(
        fresh_db, fid, drafter=lambda **kw: "draft 1",
    )
    second = email_assist.generate_draft(
        fresh_db, fid, drafter=lambda **kw: "draft 2",
    )
    assert second is None


def test_generate_draft_handles_drafter_failure(fresh_db):
    fid = _seed_email(
        fresh_db, path="imap://msgid/u", from_="x", subject="x", body="x",
    )
    _classify(fresh_db, fid, "urgent")
    def boom(**kw):
        raise RuntimeError("drafter down")
    assert email_assist.generate_draft(
        fresh_db, fid, drafter=boom,
    ) is None


def test_generate_draft_handles_empty_text(fresh_db):
    """Empty drafter output shouldn't persist."""
    fid = _seed_email(
        fresh_db, path="imap://msgid/u", from_="x", subject="x", body="x",
    )
    _classify(fresh_db, fid, "urgent")
    assert email_assist.generate_draft(
        fresh_db, fid, drafter=lambda **kw: "   ",
    ) is None


def test_mark_draft_sent(fresh_db):
    fid = _seed_email(
        fresh_db, path="imap://msgid/u", from_="x", subject="x", body="x",
    )
    _classify(fresh_db, fid, "urgent")
    draft = email_assist.generate_draft(
        fresh_db, fid, drafter=lambda **kw: "ok",
    )
    assert email_assist.mark_draft_sent(fresh_db, draft.id) is True
    # Idempotent — already sent.
    assert email_assist.mark_draft_sent(fresh_db, draft.id) is False
    # Now unsent_drafts list excludes it.
    assert email_assist.list_unsent_drafts(fresh_db) == []


def test_discard_draft(fresh_db):
    fid = _seed_email(
        fresh_db, path="imap://msgid/u", from_="x", subject="x", body="x",
    )
    _classify(fresh_db, fid, "urgent")
    draft = email_assist.generate_draft(
        fresh_db, fid, drafter=lambda **kw: "ok",
    )
    assert email_assist.discard_draft(fresh_db, draft.id) is True
    # Gone.
    assert email_assist.list_unsent_drafts(fresh_db) == []


def test_discard_draft_wont_remove_sent(fresh_db):
    """Sent drafts are an audit trail — discard shouldn't nuke them."""
    fid = _seed_email(
        fresh_db, path="imap://msgid/u", from_="x", subject="x", body="x",
    )
    _classify(fresh_db, fid, "urgent")
    draft = email_assist.generate_draft(
        fresh_db, fid, drafter=lambda **kw: "ok",
    )
    email_assist.mark_draft_sent(fresh_db, draft.id)
    # Discard refuses to delete a sent draft.
    assert email_assist.discard_draft(fresh_db, draft.id) is False


def test_generate_drafts_due_caps_fan_out(fresh_db):
    """Fan-out bounded so a daemon doesn't burn budget."""
    for i in range(5):
        fid = _seed_email(
            fresh_db, path=f"imap://msgid/{i}",
            from_="x", subject="x", body="x",
        )
        _classify(fresh_db, fid, "urgent")
    n = email_assist.generate_drafts_due(
        fresh_db, cfg=None, max_per_tick=2,
        drafter=lambda **kw: f"draft for {kw['from_']}",
    )
    assert n == 2


def test_generate_drafts_due_only_picks_draftable_labels(fresh_db):
    fid_news = _seed_email(
        fresh_db, path="imap://msgid/n", from_="x", subject="x", body="x",
    )
    _classify(fresh_db, fid_news, "newsletter")
    fid_urgent = _seed_email(
        fresh_db, path="imap://msgid/u", from_="x", subject="x", body="x",
    )
    _classify(fresh_db, fid_urgent, "urgent")
    n = email_assist.generate_drafts_due(
        fresh_db, cfg=None,
        drafter=lambda **kw: "ok",
    )
    # Only urgent gets a draft.
    assert n == 1
    drafts = email_assist.list_unsent_drafts(fresh_db)
    assert drafts[0].file_id == fid_urgent


# ============================ Round 6: structured drafter ============

def test_normalize_subject_strips_re_fwd_stack():
    assert email_assist._normalize_subject("Re: Fwd: hello") == "hello"
    assert email_assist._normalize_subject("RE: re: re: ping") == "ping"
    assert email_assist._normalize_subject("[ML-list] update") == "update"
    assert email_assist._normalize_subject("plain subject") == "plain subject"


def test_extract_email_address_handles_angle_brackets():
    assert email_assist._extract_email_address(
        "Sarah <sarah@example.com>",
    ) == "sarah@example.com"
    assert email_assist._extract_email_address(
        "bob@x.org",
    ) == "bob@x.org"
    assert email_assist._extract_email_address("(no email)") == ""
    assert email_assist._extract_email_address("") == ""


def test_gmail_thread_id_from_path():
    p = "gmail://thread/abc123/message/xyz"
    assert email_assist._gmail_thread_id_from_path(p) == "abc123"
    assert email_assist._gmail_thread_id_from_path("imap://x/1") == ""
    assert email_assist._gmail_thread_id_from_path("") == ""


def test_pull_thread_history_finds_gmail_siblings(fresh_db):
    """Two messages in the same Gmail thread should surface as
    history when we ask about one of them."""
    fid_a = _seed_email(
        fresh_db,
        path="gmail://thread/T1/message/m1",
        from_="boss@x", subject="Q3 review",
        body="What's the deck status?",
        indexed_at=time.time() - 86400,
    )
    fid_b = _seed_email(
        fresh_db,
        path="gmail://thread/T1/message/m2",
        from_="me@x", subject="Re: Q3 review",
        body="Almost done — sending tonight.",
        indexed_at=time.time() - 3600,
    )
    history = email_assist._pull_thread_history(
        fresh_db, file_id=fid_b, subject="Re: Q3 review",
    )
    paths = [p for p, _ in history]
    assert "gmail://thread/T1/message/m1" in paths
    assert "gmail://thread/T1/message/m2" not in paths  # excluded as source
    # Make sure fid_a's id was actually used; pylint guard.
    assert fid_a > 0


def test_pull_sender_history_finds_prior_messages(fresh_db):
    """Two emails from the same sender should both surface when we
    ask for sender history of one of them."""
    _seed_email(
        fresh_db,
        path="imap://msgid/p1",
        from_="<sarah@example.com>", subject="Q1 plan",
        body="Looping back on Q1 plan.",
        indexed_at=time.time() - 14 * 86400,
    )
    fid_b = _seed_email(
        fresh_db,
        path="imap://msgid/p2",
        from_="Sarah <sarah@example.com>", subject="lunch?",
        body="Free Friday for lunch?",
        indexed_at=time.time() - 3600,
    )
    history = email_assist._pull_sender_history(
        fresh_db, file_id=fid_b, sender_email="sarah@example.com",
    )
    paths = [p for p, _ in history]
    assert "imap://msgid/p1" in paths


def test_pull_sender_history_empty_when_no_address(fresh_db):
    """Empty sender_email is a no-op rather than a SQL error."""
    fid = _seed_email(
        fresh_db, path="imap://msgid/x",
        from_="(unknown)", subject="x", body="x",
    )
    out = email_assist._pull_sender_history(
        fresh_db, file_id=fid, sender_email="",
    )
    assert out == []


def test_format_analysis_block_renders_all_fields():
    a = email_assist.EmailAnalysis(
        intent="schedule",
        sender_relationship="recruiter",
        key_points=["Pick a time Tue/Wed", "Confirm role still open"],
        tone_signals=["formal", "warm"],
        length_target="short",
        open_questions=["Which time slot?"],
    )
    block = email_assist._format_analysis_block(a)
    assert "intent: schedule" in block
    assert "sender_relationship: recruiter" in block
    assert "Pick a time Tue/Wed" in block
    assert "Which time slot?" in block
    assert "formal, warm" in block


def test_default_drafter_returns_structured_output(fresh_db, tmp_cfg):
    """The new drafter returns a DraftOutput with primary +
    alternative + reasoning + open_questions. Verify the JSON
    parsing path produces all fields correctly."""
    from unittest.mock import patch

    fake_json = {
        "primary": "Tue at 2pm works for me. — Ben",
        "alternative": "Tuesday 2pm sounds great, see you then!",
        "reasoning": "Recruiter context, formal short reply matches their tone.",
        "confidence": 0.85,
        "open_questions": ["Which time zone?"],
    }
    with patch.object(email_assist, "_llm_json_call", return_value=fake_json):
        out = email_assist._default_drafter(
            from_="recruiter@x", subject="interview",
            body="Pick a time", style_samples="(none)",
            user_name="Ben", cfg=tmp_cfg,
        )
    assert out is not None
    assert out.primary.startswith("Tue at 2pm")
    assert "Tuesday 2pm" in out.alternative
    assert out.confidence == 0.85
    assert out.open_questions == ["Which time zone?"]


def test_default_drafter_returns_none_when_llm_fails(tmp_cfg):
    from unittest.mock import patch

    with patch.object(email_assist, "_llm_json_call", return_value=None):
        out = email_assist._default_drafter(
            from_="x", subject="y", body="z", style_samples="",
            user_name="Ben", cfg=tmp_cfg,
        )
    assert out is None


def test_default_drafter_clamps_confidence_to_unit_range(tmp_cfg):
    """LLMs sometimes return confidence > 1 or negative — we clamp."""
    from unittest.mock import patch

    fake = {
        "primary": "ok",
        "alternative": "",
        "reasoning": "",
        "confidence": 5.0,
        "open_questions": [],
    }
    with patch.object(email_assist, "_llm_json_call", return_value=fake):
        out = email_assist._default_drafter(
            from_="x", subject="y", body="z", style_samples="",
            user_name="Ben", cfg=tmp_cfg,
        )
    assert out.confidence == 1.0


def test_generate_draft_persists_metadata_json(fresh_db, tmp_cfg):
    """End-to-end: stub analyze + drafter, verify metadata_json is
    populated and list_unsent_drafts hydrates the new fields."""
    from unittest.mock import patch

    fid = _seed_email(
        fresh_db, path="imap://msgid/round6",
        from_="boss@x", subject="status",
        body="Where are we on Q3?",
    )
    _classify(fresh_db, fid, "urgent")
    fake_analysis = email_assist.EmailAnalysis(
        intent="question", sender_relationship="manager",
        key_points=["Q3 status update"],
        tone_signals=["formal"], length_target="short",
        open_questions=["Final number?"],
    )
    fake_output = email_assist.DraftOutput(
        primary="On track — final by Friday.",
        alternative="We're in good shape; expect numbers Friday.",
        reasoning="Manager + question intent → short factual reply.",
        confidence=0.9,
        open_questions=["Final number?"],
    )
    with patch.object(
        email_assist, "analyze_email", return_value=fake_analysis,
    ), patch.object(
        email_assist, "_default_drafter", return_value=fake_output,
    ):
        d = email_assist.generate_draft(
            fresh_db, fid, cfg=tmp_cfg, user_name="Ben",
        )
    assert d is not None
    assert d.draft_text == "On track — final by Friday."
    assert d.alternative_text and "Friday" in d.alternative_text
    assert d.analysis is not None
    assert d.analysis.intent == "question"
    assert d.analysis.sender_relationship == "manager"
    assert d.confidence == 0.9
    assert "Final number?" in d.open_questions

    # Round-trip via list_unsent_drafts to confirm metadata_json
    # rehydrates correctly.
    drafts = email_assist.list_unsent_drafts(fresh_db)
    assert len(drafts) == 1
    d2 = drafts[0]
    assert d2.analysis is not None
    assert d2.analysis.intent == "question"
    assert d2.alternative_text == fake_output.alternative


def test_legacy_drafter_path_still_works(fresh_db, tmp_cfg):
    """Stubbed drafter that returns plain text (the old test pattern)
    bypasses the new analysis pipeline cleanly."""
    fid = _seed_email(
        fresh_db, path="imap://msgid/legacy",
        from_="x", subject="y", body="z",
    )
    _classify(fresh_db, fid, "urgent")
    d = email_assist.generate_draft(
        fresh_db, fid, cfg=tmp_cfg,
        drafter=lambda **kw: "plain legacy reply",
    )
    assert d is not None
    assert d.draft_text == "plain legacy reply"
    # Legacy path → no analysis / alternative / etc.
    assert d.analysis is None
    assert d.alternative_text is None


def test_select_style_samples_smart_falls_back_to_general(fresh_db):
    """When no targeted samples exist, fall back to general Sent items
    rather than the bare 'no recent sent mail' string."""
    # Seed one general sent item not addressed to our target sender.
    _seed_email(
        fresh_db, path="imap://msgid/sent1",
        from_="me@x", subject="general",
        body="A reply about something else.",
        folder="Sent",
    )
    out = email_assist._select_style_samples_smart(
        fresh_db, sender_email="never-emailed@x.com",
        relationship="unknown",
    )
    # Should pick up the general sent item (it has 'Folder: Sent').
    assert "A reply about something else" in out


def test_select_style_samples_smart_empty_fallback_message(fresh_db):
    """No Sent items at all → friendly fallback string with the
    inferred relationship name baked in."""
    out = email_assist._select_style_samples_smart(
        fresh_db, sender_email="x@y", relationship="recruiter",
    )
    assert "no recent sent mail indexed" in out
    assert "inferred-recruiter" in out
