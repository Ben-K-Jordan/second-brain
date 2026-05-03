"""Round 7 — voice fidelity for email drafts.

Three pillars under test:
  - Reply-pair indexer (Sent → parent linking via In-Reply-To /
    thread / subject heuristics).
  - Voice profile extractor (deterministic structural stats +
    LLM qualitative notes).
  - Few-shot retrieval + critique loop wired into the drafter.

These tests focus on the pure-Python deterministic helpers (no
network calls). The LLM-bound paths are covered by stubbing
``_llm_text_call`` / ``_llm_json_call``.
"""

from __future__ import annotations

import json
import time
from unittest.mock import patch

from secondbrain import email_assist

# ============================ helpers =================================

def _seed_email(
    conn, *, path, from_, subject, body, indexed_at=None,
    folder="INBOX", extra_headers="",
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
        f"Folder: {folder}\n"
        + (f"{extra_headers}\n" if extra_headers else "")
        + f"\n{body}"
    )
    conn.execute(
        "INSERT INTO chunks(file_id, chunk_index, text) VALUES (?, 0, ?)",
        (fid, full),
    )
    conn.commit()
    return fid


# ============================ schema ==================================

def test_round7_schema_creates_voice_tables(fresh_db):
    """The two new tables should exist after _ensure_schema runs."""
    email_assist._ensure_schema(fresh_db)
    rows = fresh_db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name IN ('email_style_profile', 'email_reply_pairs')",
    ).fetchall()
    names = {r["name"] for r in rows}
    assert "email_style_profile" in names
    assert "email_reply_pairs" in names


# ============================ Sent detection ===========================

def test_is_sent_item_recognises_imap_sent_folder():
    text = "# subject\nFrom: me@x\nFolder: Sent\n\nbody"
    assert email_assist._is_sent_item(text) is True


def test_is_sent_item_recognises_gmail_label():
    text = "# subject\nFrom: me@x\nLabels: Sent, Important\n\nbody"
    assert email_assist._is_sent_item(text) is True


def test_is_sent_item_negative():
    text = "# subject\nFrom: someone@x\nFolder: INBOX\n\nbody"
    assert email_assist._is_sent_item(text) is False


# ============================ Header parsing ===========================

def test_msgid_for_file(fresh_db):
    fid = _seed_email(
        fresh_db, path="imap://msgid/abc", from_="x", subject="y",
        body="hi",
        extra_headers="Message-ID: <abc.123@example.com>",
    )
    assert (
        email_assist._msgid_for_file(fresh_db, fid)
        == "abc.123@example.com"
    )


def test_inreplyto_for_file(fresh_db):
    fid = _seed_email(
        fresh_db, path="imap://msgid/x", from_="x", subject="y",
        body="hi",
        extra_headers="In-Reply-To: <parent.456@example.com>",
    )
    assert (
        email_assist._inreplyto_for_file(fresh_db, fid)
        == "parent.456@example.com"
    )


# ============================ Parent finding ===========================

def test_find_parent_via_in_reply_to(fresh_db):
    """When a Sent message has In-Reply-To matching another file's
    Message-ID, that file is the parent — strategy 1."""
    parent_id = _seed_email(
        fresh_db, path="imap://msgid/parent",
        from_="boss@x", subject="Q3 plan", body="Where's the deck?",
        extra_headers="Message-ID: <parent.123@x.com>",
    )
    reply_id = _seed_email(
        fresh_db, path="imap://msgid/reply",
        from_="me@x", subject="Re: Q3 plan", body="Almost done.",
        folder="Sent",
        extra_headers="In-Reply-To: <parent.123@x.com>",
    )
    fid, method = email_assist._find_parent_for_reply(
        fresh_db, reply_file_id=reply_id, subject="Re: Q3 plan",
    )
    assert fid == parent_id
    assert method == "in-reply-to"


def test_find_parent_via_gmail_thread(fresh_db):
    """Strategy 2: same gmail thread, most recent non-Sent."""
    parent_id = _seed_email(
        fresh_db, path="gmail://thread/T1/message/m1",
        from_="boss@x", subject="hello", body="ping",
    )
    reply_id = _seed_email(
        fresh_db, path="gmail://thread/T1/message/m2",
        from_="me@x", subject="Re: hello", body="hi back",
        folder="Sent",
    )
    fid, method = email_assist._find_parent_for_reply(
        fresh_db, reply_file_id=reply_id, subject="Re: hello",
    )
    assert fid == parent_id
    assert method == "thread"


def test_find_parent_via_subject_heuristic(fresh_db):
    """Strategy 3: subject normalises to the same root."""
    parent_id = _seed_email(
        fresh_db, path="imap://msgid/orig",
        from_="boss@x", subject="quarterly review",
        body="status please",
    )
    reply_id = _seed_email(
        fresh_db, path="imap://msgid/myreply",
        from_="me@x", subject="Re: quarterly review",
        body="on it", folder="Sent",
    )
    fid, method = email_assist._find_parent_for_reply(
        fresh_db, reply_file_id=reply_id,
        subject="Re: quarterly review",
    )
    assert fid == parent_id
    assert method == "subject"


def test_find_parent_returns_none_when_nothing_matches(fresh_db):
    reply_id = _seed_email(
        fresh_db, path="imap://msgid/orphan",
        from_="me@x", subject="totally unique subject",
        body="hi", folder="Sent",
    )
    fid, method = email_assist._find_parent_for_reply(
        fresh_db, reply_file_id=reply_id,
        subject="totally unique subject",
    )
    assert fid is None
    assert method == "none"


# ============================ index_reply_pairs =======================

def test_index_reply_pairs_persists_link(fresh_db):
    """End-to-end: a Sent message + its parent get linked in
    email_reply_pairs."""
    parent_id = _seed_email(
        fresh_db, path="gmail://thread/T2/message/p",
        from_="x@y", subject="ask", body="?",
    )
    reply_id = _seed_email(
        fresh_db, path="gmail://thread/T2/message/r",
        from_="me@x", subject="Re: ask", body="answer",
        folder="Sent",
    )
    n = email_assist.index_reply_pairs(fresh_db)
    assert n == 1
    row = fresh_db.execute(
        "SELECT incoming_file_id, link_method FROM email_reply_pairs "
        "WHERE reply_file_id = ?",
        (reply_id,),
    ).fetchone()
    assert row is not None
    assert row["incoming_file_id"] == parent_id
    assert row["link_method"] == "thread"


def test_index_reply_pairs_idempotent(fresh_db):
    """Running twice must not double-link the same reply."""
    _seed_email(
        fresh_db, path="gmail://thread/T3/message/p",
        from_="x@y", subject="ping", body="hi",
    )
    _seed_email(
        fresh_db, path="gmail://thread/T3/message/r",
        from_="me@x", subject="Re: ping", body="hi back",
        folder="Sent",
    )
    n1 = email_assist.index_reply_pairs(fresh_db)
    n2 = email_assist.index_reply_pairs(fresh_db)
    assert n1 == 1
    assert n2 == 0


def test_index_reply_pairs_skips_inbox_items(fresh_db):
    """Non-Sent items aren't candidate replies."""
    _seed_email(
        fresh_db, path="imap://msgid/inbox-only",
        from_="x", subject="y", body="z",
    )
    n = email_assist.index_reply_pairs(fresh_db)
    assert n == 0


# ============================ Voice profile extraction ================

def test_strip_email_headers_drops_prefix():
    text = (
        "# Re: hello\n"
        "From: me@x\n"
        "To: them@x\n"
        "Date: Mon\n"
        "Folder: Sent\n"
        "Message-ID: <a>\n"
        "In-Reply-To: <b>\n"
        "\n"
        "Hi Sarah,\n\nthanks for the note.\n"
    )
    out = email_assist._strip_email_headers(text)
    assert out.startswith("Hi Sarah,")
    assert "From:" not in out


def test_strip_email_headers_drops_quoted_reply():
    text = (
        "# Re: hello\n"
        "From: me@x\n"
        "Folder: Sent\n"
        "\n"
        "Quick thought below.\n"
        "\n"
        "On Mon, Jan 1, Sarah wrote:\n"
        "> the original message\n"
    )
    out = email_assist._strip_email_headers(text)
    assert "Quick thought" in out
    assert "Sarah wrote" not in out
    assert "original message" not in out


def test_extract_greeting_patterns_normalises_names():
    bodies = [
        "Hi Sarah,\nbody one",
        "Hi Bob,\nbody two",
        "Hi Alice,\nbody three",
    ]
    pats = email_assist._extract_greeting_patterns(bodies)
    # All three normalise to "hi {name},"
    assert "hi {name}," in pats


def test_extract_signoff_patterns_normalises_names():
    bodies = [
        "body one\n\nthanks,\nBen",
        "body two\n\nthanks,\nBen",
        "body three\n\nthanks,\nBen",
    ]
    pats = email_assist._extract_signoff_patterns(bodies)
    # Username collapses to {name}.
    assert any("{name}" in p for p in pats)


def test_avg_sentence_words_basic():
    bodies = ["This has four words. So does this one."]
    avg = email_assist._avg_sentence_words(bodies)
    assert 3.5 <= avg <= 4.5


def test_contraction_rate_all_contracted():
    bodies = ["I'd like to. I can't help. It's done."]
    rate = email_assist._contraction_rate(bodies)
    assert rate == 1.0


def test_contraction_rate_all_expanded():
    bodies = ["I would like to. I cannot help. It is done."]
    rate = email_assist._contraction_rate(bodies)
    assert rate == 0.0


def test_audit_llm_isms_flags_unused_phrases():
    bodies = [
        "Hi Sarah, just looping back. — Ben",
        "Quick update — done by Friday. Thanks, Ben",
    ]
    avoided = email_assist._audit_llm_isms(bodies)
    assert "I hope this email finds you well" in avoided
    assert "Best regards," in avoided


def test_audit_llm_isms_keeps_used_phrases():
    bodies = [
        "Hi! Best regards, Ben",
    ]
    avoided = email_assist._audit_llm_isms(bodies)
    assert "Best regards," not in avoided


def test_extract_voice_profile_persists(fresh_db):
    """End-to-end: seed Sent emails, run extraction with stubbed
    LLM, verify the profile lands in the DB."""
    for i in range(3):
        _seed_email(
            fresh_db, path=f"imap://msgid/sent{i}",
            from_="me@x", subject=f"reply {i}",
            body=(
                "Hi Sarah,\n\n"
                "Thanks for the ping — let's grab time Tuesday. "
                "I'll send a calendar hold.\n\n"
                "thanks,\nBen"
            ),
            folder="Sent",
        )
    with patch.object(
        email_assist, "_voice_register_notes",
        return_value="Warm, casual, short replies. Heavy contractions.",
    ):
        profile = email_assist.extract_voice_profile(
            fresh_db, cfg=object(),
        )
    assert profile is not None
    assert profile.n_samples == 3
    assert profile.contraction_rate > 0.5
    assert profile.register_notes.startswith("Warm")
    # Round-trip from the DB.
    loaded = email_assist.get_voice_profile(fresh_db)
    assert loaded is not None
    assert loaded.n_samples == 3
    assert loaded.register_notes == profile.register_notes


def test_extract_voice_profile_empty_corpus_returns_none(fresh_db):
    out = email_assist.extract_voice_profile(fresh_db, cfg=object())
    assert out is None


def test_needs_voice_profile_refresh_true_when_missing(fresh_db):
    assert email_assist.needs_voice_profile_refresh(fresh_db) is True


def test_needs_voice_profile_refresh_false_after_recent(fresh_db):
    p = email_assist.VoiceProfile(
        greetings=[], sign_offs=[], avg_sentence_words=10,
        avg_reply_chars=200, contraction_rate=0.5,
        exclamation_rate=0, emoji_rate=0,
        common_openers=[], common_closers=[],
        avoided_phrases=[], register_notes="", n_samples=10,
    )
    email_assist._save_voice_profile(fresh_db, p)
    assert email_assist.needs_voice_profile_refresh(fresh_db) is False


def test_format_voice_profile_block_renders_all_sections():
    p = email_assist.VoiceProfile(
        greetings=["hi {name},"], sign_offs=["thanks,\n{name}"],
        avg_sentence_words=8.5, avg_reply_chars=180,
        contraction_rate=0.85, exclamation_rate=0.3, emoji_rate=0.1,
        common_openers=["thanks for the ping"],
        common_closers=["let me know"],
        avoided_phrases=["Best regards,"],
        register_notes="Warm, casual.",
        n_samples=12,
    )
    block = email_assist._format_voice_profile_block(p)
    assert "12 sent emails" in block
    assert "hi {name}," in block
    assert "Best regards," in block
    assert "Warm, casual" in block
    # Numeric fields render as percentages / counts the model can use.
    assert "85%" in block
    assert "8.5 words" in block


# ============================ Few-shot retrieval ======================

def test_fewshot_reply_pairs_returns_pair_bodies(fresh_db):
    """Seed one paired email + reply, verify fewshot returns it."""
    parent_id = _seed_email(
        fresh_db, path="gmail://thread/F1/message/p",
        from_="x@y", subject="schedule",
        body="Can we meet Tuesday?",
    )
    reply_id = _seed_email(
        fresh_db, path="gmail://thread/F1/message/r",
        from_="me@x", subject="Re: schedule",
        body="Tuesday at 2 works for me. — Ben",
        folder="Sent",
    )
    email_assist.index_reply_pairs(fresh_db)
    pairs = email_assist.fewshot_reply_pairs(
        fresh_db, incoming_text="meeting request",
        embedder=None,  # fall back to recency order
    )
    assert len(pairs) == 1
    inc, rep = pairs[0]
    assert "Tuesday" in inc
    assert "Tuesday at 2" in rep
    # Reply body should have headers stripped.
    assert "Folder: Sent" not in rep
    assert parent_id and reply_id  # silences unused-var warnings


def test_fewshot_reply_pairs_empty_when_no_pairs(fresh_db):
    pairs = email_assist.fewshot_reply_pairs(
        fresh_db, incoming_text="hi", embedder=None,
    )
    assert pairs == []


def test_format_fewshot_block_handles_empty():
    out = email_assist._format_fewshot_block([])
    assert "no reply-pair examples" in out


def test_format_fewshot_block_renders_pairs():
    out = email_assist._format_fewshot_block([
        ("incoming1", "reply1"),
        ("incoming2", "reply2"),
    ])
    assert "EXAMPLE 1" in out
    assert "EXAMPLE 2" in out
    assert "reply1" in out
    assert "reply2" in out


# ============================ Critique ================================

def test_critique_returns_ok_when_llm_says_ok(tmp_cfg):
    p = email_assist.VoiceProfile(
        greetings=[], sign_offs=[], avg_sentence_words=10,
        avg_reply_chars=200, contraction_rate=0.5,
        exclamation_rate=0, emoji_rate=0,
        common_openers=[], common_closers=[],
        avoided_phrases=[], register_notes="", n_samples=10,
    )
    with patch.object(email_assist, "_llm_text_call", return_value="OK"):
        out = email_assist.critique_draft_against_voice(
            draft="hi", profile=p, cfg=tmp_cfg,
        )
    assert out == "OK"


def test_critique_returns_bullet_list_on_mismatches(tmp_cfg):
    p = email_assist.VoiceProfile(
        greetings=[], sign_offs=[], avg_sentence_words=10,
        avg_reply_chars=200, contraction_rate=0.5,
        exclamation_rate=0, emoji_rate=0,
        common_openers=[], common_closers=[],
        avoided_phrases=["Best regards,"], register_notes="",
        n_samples=10,
    )
    bullets = "- Uses banned phrase 'Best regards,'\n- Too long"
    with patch.object(
        email_assist, "_llm_text_call", return_value=bullets,
    ):
        out = email_assist.critique_draft_against_voice(
            draft="x", profile=p, cfg=tmp_cfg,
        )
    assert "Best regards" in out
    assert "Too long" in out


# ============================ End-to-end with voice ===================

def test_drafter_includes_voice_blocks_in_prompt(tmp_cfg):
    """Verify _default_drafter renders the voice profile + few-shot
    blocks into the prompt sent to the LLM."""
    p = email_assist.VoiceProfile(
        greetings=["hi {name},"], sign_offs=["—Ben"],
        avg_sentence_words=8, avg_reply_chars=150,
        contraction_rate=0.9, exclamation_rate=0.2, emoji_rate=0,
        common_openers=[], common_closers=[],
        avoided_phrases=["Best regards,"],
        register_notes="Casual.", n_samples=20,
    )
    captured: dict = {}

    def _capture(*, prompt, cfg, model, max_tokens, feature, note):
        captured["prompt"] = prompt
        return {
            "primary": "ok",
            "alternative": "",
            "reasoning": "",
            "confidence": 0.8,
            "open_questions": [],
        }
    with patch.object(email_assist, "_llm_json_call", side_effect=_capture):
        out = email_assist._default_drafter(
            from_="x@y", subject="hi", body="ping",
            style_samples="(none)", user_name="Ben", cfg=tmp_cfg,
            voice_profile=p,
            fewshot_pairs=[("incoming sample", "user reply sample")],
        )
    assert out is not None
    # Voice profile appears in prompt
    assert "voice profile" in captured["prompt"]
    assert "hi {name}," in captured["prompt"]
    assert "Best regards," in captured["prompt"]
    # Few-shot block appears
    assert "EXAMPLE 1" in captured["prompt"]
    assert "user reply sample" in captured["prompt"]


def test_generate_draft_runs_critique_when_profile_present(
    fresh_db, tmp_cfg,
):
    """End-to-end: when a profile is saved, generate_draft runs
    critique. When critique flags issues, regenerate_with_critique
    is invoked once."""
    # Seed an email + classification.
    fid = _seed_email(
        fresh_db, path="imap://msgid/round7",
        from_="boss@x", subject="status?",
        body="What's the Q3 status?",
    )
    email_assist._ensure_schema(fresh_db)
    fresh_db.execute(
        "INSERT INTO email_classifications"
        "(file_id, label, confidence, classified_at) "
        "VALUES (?, 'urgent', 0.9, ?)",
        (fid, time.time()),
    )
    fresh_db.commit()

    # Persist a voice profile so the critique branch fires.
    profile = email_assist.VoiceProfile(
        greetings=[], sign_offs=[], avg_sentence_words=10,
        avg_reply_chars=200, contraction_rate=0.5,
        exclamation_rate=0, emoji_rate=0,
        common_openers=[], common_closers=[],
        avoided_phrases=["Best regards,"], register_notes="",
        n_samples=5,
    )
    email_assist._save_voice_profile(fresh_db, profile)

    # First draft uses banned phrase → critique flags → regenerate.
    first_output = email_assist.DraftOutput(
        primary="On track. Best regards, Ben",
        alternative="alt",
        reasoning="r1", confidence=0.8, open_questions=[],
    )
    second_output = email_assist.DraftOutput(
        primary="On track — done by Friday. — Ben",
        alternative="alt2",
        reasoning="r2", confidence=0.85, open_questions=[],
    )

    drafter_calls = []

    def _drafter(**kw):
        drafter_calls.append(kw)
        return first_output

    regen_calls = []

    def _regen(**kw):
        regen_calls.append(kw)
        return second_output

    with patch.object(email_assist, "analyze_email", return_value=None), \
         patch.object(email_assist, "_default_drafter", side_effect=_drafter), \
         patch.object(
             email_assist, "critique_draft_against_voice",
             return_value="- uses banned phrase Best regards,",
         ), \
         patch.object(
             email_assist, "_regenerate_with_critique", side_effect=_regen,
         ):
        d = email_assist.generate_draft(
            fresh_db, fid, cfg=tmp_cfg, user_name="Ben",
        )
    assert d is not None
    # Final draft is the regenerated one.
    assert d.draft_text == second_output.primary
    assert len(regen_calls) == 1
    # Metadata should record the critique we got back.
    row = fresh_db.execute(
        "SELECT metadata_json FROM email_drafts WHERE id = ?",
        (d.id,),
    ).fetchone()
    meta = json.loads(row["metadata_json"])
    assert "voice_critique" in meta
    assert "Best regards" in meta["voice_critique"]
    assert meta["voice_profile_n_samples"] == 5


def test_generate_draft_skips_critique_when_no_profile(
    fresh_db, tmp_cfg,
):
    """No profile saved → critique branch is a no-op; first draft wins."""
    fid = _seed_email(
        fresh_db, path="imap://msgid/noprof",
        from_="boss@x", subject="status?", body="?",
    )
    email_assist._ensure_schema(fresh_db)
    fresh_db.execute(
        "INSERT INTO email_classifications"
        "(file_id, label, confidence, classified_at) "
        "VALUES (?, 'urgent', 0.9, ?)",
        (fid, time.time()),
    )
    fresh_db.commit()
    out = email_assist.DraftOutput(
        primary="ok", alternative="", reasoning="", confidence=0.5,
        open_questions=[],
    )
    with patch.object(email_assist, "analyze_email", return_value=None), \
         patch.object(email_assist, "_default_drafter", return_value=out), \
         patch.object(
             email_assist, "critique_draft_against_voice",
         ) as critique_mock:
        d = email_assist.generate_draft(
            fresh_db, fid, cfg=tmp_cfg, user_name="Ben",
        )
    assert d is not None
    critique_mock.assert_not_called()


# ============================ Daemon scheduler =======================

def test_daemon_registers_email_voice_jobs(fresh_db, tmp_cfg):
    from secondbrain.daemon import _build_daemon_scheduler
    sched = _build_daemon_scheduler(
        tmp_cfg, fresh_db, embedder=None, reranker=None,
    )
    names = set(sched.names())
    assert "email_reply_pairs_index" in names
    assert "email_voice_profile" in names
