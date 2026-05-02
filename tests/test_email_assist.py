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
