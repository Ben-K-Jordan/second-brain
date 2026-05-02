"""Phase 42: transcript detection + course matching (Plaud / Otter / generic)."""

from __future__ import annotations

from secondbrain.transcripts import (
    Transcript,
    _extract_course_code,
    _looks_like_transcript,
    _strip_re,
    detect_transcript,
    match_canvas_course,
)

# ============================ detection ===============================

def test_detect_returns_none_for_empty_body():
    assert detect_transcript("noreply@plaud.app", "Anything", "") is None
    assert detect_transcript("noreply@plaud.app", "Anything", "   ") is None


def test_detect_plaud_by_sender():
    """Even a stub Plaud-like body should detect when the sender domain
    matches and the body has speaker-turn structure."""
    body = """\
Title: BME 410 lecture
Recorded on: 2026-04-15 14:30:00
Duration: 75 min

## Summary
Today we covered the diffusion equation.

## Transcript
Speaker 1 (00:00): Today we will derive Fick's law.
Speaker 1 (00:30): The flux is proportional to the gradient.
Speaker 2 (01:00): Wait, what's the assumption?
Speaker 1 (01:05): Steady state and isotropic medium.
"""
    t = detect_transcript("noreply@plaud.app", "BME 410 lecture", body)
    assert t is not None
    assert t.provider == "plaud"
    assert "Today we will derive Fick's law" in t.body
    # The transcript section should be extracted from "## Transcript" onward.
    assert "Today we covered" not in t.body
    # Summary captured separately.
    assert "diffusion equation" in t.summary
    assert t.duration_seconds == 75 * 60
    assert t.recorded_at > 0
    # Speakers detected, in order of first mention.
    assert t.speakers[:2] == ["Speaker 1", "Speaker 2"]


def test_detect_plaud_without_explicit_sections():
    """A Plaud email that doesn't have ## Summary / ## Transcript headings
    should still parse — we fall back to the whole body as the transcript."""
    body = "Speaker 1: hello\nSpeaker 2: hi back\nSpeaker 1: cool"
    t = detect_transcript("noreply@plaud.app", "Recording", body)
    assert t is not None
    assert t.provider == "plaud"
    assert "hello" in t.body


def test_detect_otter_via_sender():
    body = "Alice 0:00\nWelcome everyone\nBob 0:14\nThanks for having me\nAlice 0:30\nLet's start"
    t = detect_transcript("notes@otter.ai", "Sales call", body)
    assert t is not None
    assert t.provider == "otter"
    assert t.speakers[:2] == ["Alice", "Bob"]


def test_detect_otter_returns_none_when_format_doesnt_match():
    """An email from otter.ai that doesn't have the timestamp pattern
    shouldn't be claimed as an otter transcript."""
    body = "We've shipped a new feature for you to try."
    assert detect_transcript("news@otter.ai", "Otter newsletter", body) is None


def test_detect_generic_falls_back_for_speaker_heavy_body():
    """A body with multiple speaker-labeled lines but no known sender
    should still be recognised as a generic transcript."""
    body = """\
Sarah Smith: Welcome everyone.
John Doe: Thanks for having us.
Sarah Smith: Today's agenda is short.
John Doe: Great, let's begin.
"""
    t = detect_transcript("forwarded@friend.com", "1:1 with Sarah", body)
    assert t is not None
    assert t.provider == "generic"
    assert "Sarah Smith" in t.speakers


def test_detect_generic_skips_normal_email():
    """A regular email with one or two ': ' lines (e.g. URLs) shouldn't
    be misclassified as a transcript."""
    body = (
        "Hi Ben,\n\n"
        "Thanks for reaching out. The link is here: https://example.com\n\n"
        "Best,\nAlex"
    )
    assert detect_transcript("alex@example.com", "Re: chat", body) is None


def test_looks_like_transcript_threshold():
    """Three+ speaker-turn-style lines = transcript-shaped."""
    short = "A: hello\nB: hi"
    assert _looks_like_transcript(short) is False
    longer = "A: hello\nB: hi\nC: greetings\nD: cheers"
    assert _looks_like_transcript(longer) is True


# ============================ helpers ================================

def test_strip_re_removes_re_and_fwd():
    assert _strip_re("Re: chat") == "chat"
    assert _strip_re("Fwd: Re: BME 410") == "BME 410"
    assert _strip_re("FW: heads up") == "heads up"
    assert _strip_re("plain") == "plain"
    assert _strip_re("") == ""


def test_extract_course_code_basic():
    assert _extract_course_code("BME 410 lecture") == "BME 410"
    assert _extract_course_code("CS374 problem set") == "CS 374"
    assert _extract_course_code("BIOMG 1350 - week 4") == "BIOMG 1350"


def test_extract_course_code_no_match():
    assert _extract_course_code("Anthropic phone screen") == ""
    assert _extract_course_code("") == ""
    # Avoid false positives on plain numbers.
    assert _extract_course_code("Meeting at 3pm") == ""


def test_extract_course_code_picks_first():
    """When multiple codes appear (rare), take the first."""
    assert _extract_course_code(
        "BME 410 / CS 374 joint review",
    ) == "BME 410"


# ====================== match_canvas_course ==========================

def test_match_via_subject_regex():
    t = Transcript(
        provider="plaud", title="recording", body="x",
        raw_subject="BME 410 lecture - 2026-04-15",
    )
    assert match_canvas_course(t) == "BME 410"


def test_match_via_title_when_subject_lacks_code():
    """Plaud's email subject is sometimes just 'Recording'; the parsed
    Title field carries the course code."""
    t = Transcript(
        provider="plaud", title="BME 410 lecture", body="x",
        raw_subject="Recording 2026-04-15",
    )
    assert match_canvas_course(t) == "BME 410"


def test_match_via_canvas_course_list():
    """No code in subject/title; matches against canvas course names."""
    t = Transcript(
        provider="plaud", title="Lecture audio", body="x",
        raw_subject="Today's lecture in introduction to programming",
    )
    courses = [
        {"name": "Introduction to Programming", "course_code": "CS 100"},
        {"name": "Calculus 1", "course_code": "MATH 1110"},
    ]
    assert match_canvas_course(t, courses) == "CS 100"


def test_match_returns_empty_when_nothing_matches():
    t = Transcript(
        provider="plaud", title="Random", body="x",
        raw_subject="Audio note",
    )
    assert match_canvas_course(t) == ""
    assert match_canvas_course(t, []) == ""
    assert match_canvas_course(t, None) == ""


def test_match_subject_wins_over_courses_list():
    """A clear course code in the subject takes precedence over
    fuzzy-match against the canvas list — avoids false positives where
    course names appear in body text."""
    t = Transcript(
        provider="plaud", title="x", body="x",
        raw_subject="BME 410 review session",
    )
    courses = [
        {"name": "Calculus 1", "course_code": "MATH 1110"},
    ]
    assert match_canvas_course(t, courses) == "BME 410"


# ====================== full Plaud round-trip ========================

def test_plaud_email_round_trip_with_course_code_in_subject():
    """Realistic flow: Plaud emails a transcript with the course code
    in the subject (because the user named the recording)."""
    body = """\
Title: Today's lecture
Recorded on: 2026-04-15 14:30:00
Duration: 75 min

## Summary
Diffusion in biological tissues.

## Transcript
Professor Chen (00:00): Welcome back. Today we're covering diffusion.
Professor Chen (01:30): Fick's first law states that flux equals -D times grad C.
Student (15:00): How does this change in anisotropic media?
Professor Chen (15:30): Great question - the diffusivity becomes a tensor.
"""
    t = detect_transcript(
        "noreply@plaud.app", "BME 410 - lecture 2026-04-15", body,
    )
    assert t is not None
    assert t.provider == "plaud"
    assert match_canvas_course(t) == "BME 410"
    assert "Fick's first law" in t.body
    assert "diffusion in biological" in t.summary.lower()
    assert "Professor Chen" in t.speakers
    assert "Student" in t.speakers
    assert t.duration_seconds == 75 * 60


def test_plaud_email_no_course_code_anywhere():
    """When the user didn't put the course code anywhere, course_code
    stays empty — caller can decide whether to skip or store untagged."""
    body = (
        "Title: Random thoughts\n"
        "Recorded on: 2026-04-15 10:00:00\n\n"
        "Speaker 1: Just a quick voice memo\n"
        "Speaker 1: about something I read\n"
    )
    t = detect_transcript("noreply@plaud.app", "Recording", body)
    assert t is not None
    assert match_canvas_course(t) == ""


# ============= IMAP connector integration shape ======================

def test_imap_connector_builds_transcript_doc(monkeypatch):
    """The IMAP connector should produce a transcript:// virtual_path
    when the email looks like a Plaud transcript, with course tagged
    when detectable."""
    from secondbrain.connectors.imap_email import ImapEmailConnector

    body = """\
Title: BME 410 lecture
Recorded on: 2026-04-15 14:30:00
Duration: 60 min

## Transcript
Speaker 1 (00:00): Today's topic is diffusion.
Speaker 1 (00:30): Fick's law applies.
Speaker 2 (01:00): Question about boundary conditions?
"""
    c = ImapEmailConnector()
    doc = c._maybe_build_transcript_doc(
        from_="noreply@plaud.app",
        subject="BME 410 - 2026-04-15",
        body=body,
        date_hdr="Tue, 15 Apr 2026 15:00:00 +0000",
        mtime=0.0,
        folder="Plaud", msg_id="abc-123", uid=b"42",
    )
    assert doc is not None
    assert doc.virtual_path == "transcript://plaud/abc-123"
    assert doc.source == "transcript:plaud"
    assert "[BME 410]" in doc.title
    # The structured rendering surfaces course / duration / speakers.
    assert "Course: BME 410" in doc.content
    assert "Duration: 60 min" in doc.content
    assert "Speaker 1" in doc.content
    # Metadata for downstream queries / dashboard rendering.
    assert doc.metadata["provider"] == "plaud"
    assert doc.metadata["course_code"] == "BME 410"
    assert doc.metadata["duration_seconds"] == 3600


def test_imap_connector_falls_back_to_email_for_non_transcripts(monkeypatch):
    """Regular emails should NOT get the transcript treatment."""
    from secondbrain.connectors.imap_email import ImapEmailConnector

    c = ImapEmailConnector()
    doc = c._maybe_build_transcript_doc(
        from_="alex@example.com",
        subject="Re: catch up",
        body="Hi, thanks for reaching out. Let's grab coffee Friday.",
        date_hdr="Tue, 15 Apr 2026 15:00:00 +0000",
        mtime=0.0,
        folder="INBOX", msg_id="xyz", uid=b"1",
    )
    assert doc is None


# =============================== Granola =============================

GRANOLA_SAMPLE = """\
Date: 2026-04-15 14:30
Attendees: Ben Jordan, Sarah Smith, Alex Lee

## Summary
Quarterly planning sync. Decided to ship the new pricing experiment in
March, with Sarah owning the ramp.

## Key Points
- Pricing experiment ramps over 4 weeks
- Customer interviews continue through end of February
- Need to align with marketing on launch comms

## Action Items
- [ ] Sarah to draft the pricing one-pager by Friday
- [x] Ben to send out the customer interview slots
- [ ] Alex to sync with marketing on the launch plan

## Transcript
Sarah Smith: Let's lock in the experiment dates.
Ben Jordan: I think March 15 is realistic.
Alex Lee: I'll handle marketing alignment.
Sarah Smith: Great, we'll go with that.
"""


def test_detect_granola_via_sender():
    t = detect_transcript(
        "noreply@granola.so", "Notes from Quarterly planning",
        GRANOLA_SAMPLE,
    )
    assert t is not None
    assert t.provider == "granola"


def test_granola_extracts_summary():
    t = detect_transcript("noreply@granola.so", "Notes", GRANOLA_SAMPLE)
    assert t is not None
    assert "Quarterly planning sync" in t.summary
    assert "ship the new pricing" in t.summary


def test_granola_extracts_action_items():
    """Granola's checklist syntax (- [ ] / - [x]) should round-trip into
    the action_items list, with checkbox markers stripped."""
    t = detect_transcript("noreply@granola.so", "Notes", GRANOLA_SAMPLE)
    assert t is not None
    assert len(t.action_items) == 3
    assert any("pricing one-pager" in item for item in t.action_items)
    # Checked-off item still surfaces — the user might want to review
    # what was already shipped.
    assert any("customer interview slots" in item for item in t.action_items)
    # Markers stripped (no "- [ ]" / "- [x]" prefixes).
    for item in t.action_items:
        assert not item.startswith("- [")
        assert not item.startswith("[ ]")


def test_granola_extracts_attendees_from_header():
    t = detect_transcript("noreply@granola.so", "Notes", GRANOLA_SAMPLE)
    assert t is not None
    assert set(t.attendees) >= {"Ben Jordan", "Sarah Smith", "Alex Lee"}


def test_granola_extracts_recording_time():
    t = detect_transcript("noreply@granola.so", "Notes", GRANOLA_SAMPLE)
    assert t is not None
    assert t.recorded_at > 0


def test_granola_extracts_transcript_speakers():
    """Speakers come from the ## Transcript section, not the attendees
    line — different sections may include different people."""
    t = detect_transcript("noreply@granola.so", "Notes", GRANOLA_SAMPLE)
    assert t is not None
    assert "Sarah Smith" in t.speakers
    assert "Ben Jordan" in t.speakers


def test_granola_strips_subject_prefix():
    """'Notes from "Quarterly planning"' → title 'Quarterly planning'."""
    t = detect_transcript(
        "noreply@granola.so", 'Notes from "Quarterly planning"',
        GRANOLA_SAMPLE,
    )
    assert t is not None
    assert t.title == "Quarterly planning"


def test_granola_detected_by_body_shape_when_sender_unknown():
    """If a Granola note is forwarded from a personal email, body-shape
    detection should still pick it up via canonical headings."""
    t = detect_transcript(
        "ben@personal.com", "Fwd: Notes from sync", GRANOLA_SAMPLE,
    )
    assert t is not None
    assert t.provider == "granola"


def test_granola_without_transcript_section():
    """A Granola note that's notes-only (no transcript) should still
    parse — body becomes the structured notes themselves."""
    body = """\
## Summary
Quick alignment on the launch.

## Action Items
- [ ] Send the spec
- [ ] Schedule design review
"""
    t = detect_transcript("noreply@granola.so", "Notes from sync", body)
    assert t is not None
    assert t.provider == "granola"
    assert "Quick alignment" in t.summary
    assert len(t.action_items) == 2


def test_granola_imap_doc_uses_meeting_prefix_when_event_match(monkeypatch):
    """When a Granola transcript matches a calendar event, the IMAP-built
    doc should prefix the title with [meeting] and surface event metadata."""
    # Stub the calendar matcher to return a deterministic match.
    import secondbrain.transcripts as tx_mod
    from secondbrain.connectors.imap_email import ImapEmailConnector
    monkeypatch.setattr(
        tx_mod, "match_calendar_event",
        lambda cfg, t, **kw: ("primary/abc", "google_calendar",
                              "Quarterly planning"),
    )

    class FakeCfg:
        pass

    c = ImapEmailConnector()
    doc = c._maybe_build_transcript_doc(
        from_="noreply@granola.so",
        subject="Notes from Quarterly planning",
        body=GRANOLA_SAMPLE,
        date_hdr="Tue, 15 Apr 2026 15:00:00 +0000",
        mtime=0.0, folder="Granola", msg_id="m1", uid=b"1",
        cfg=FakeCfg(),
    )
    assert doc is not None
    assert "[meeting]" in doc.title
    assert doc.metadata["event_id"] == "primary/abc"
    assert doc.metadata["event_source"] == "google_calendar"
    assert doc.metadata["event_title"] == "Quarterly planning"
    # Action items rendered into the body so they're searchable.
    assert "## Action items" in doc.content


def test_granola_imap_doc_skips_calendar_match_without_cfg():
    """When cfg isn't passed (e.g. tests, ad-hoc ingest), event matching
    is skipped silently rather than crashing."""
    from secondbrain.connectors.imap_email import ImapEmailConnector

    c = ImapEmailConnector()
    doc = c._maybe_build_transcript_doc(
        from_="noreply@granola.so",
        subject="Notes from Quarterly planning",
        body=GRANOLA_SAMPLE,
        date_hdr="Tue, 15 Apr 2026 15:00:00 +0000",
        mtime=0.0, folder="Granola", msg_id="m1", uid=b"1",
        cfg=None,
    )
    assert doc is not None
    # Untagged transcript still ingests, just without [meeting] prefix.
    assert "[meeting]" not in doc.title
    assert doc.metadata["event_id"] == ""


def test_granola_imap_doc_tolerates_event_match_crash(monkeypatch):
    """A crash in the calendar-match path shouldn't take down the
    transcript ingest; we log + ingest untagged."""
    import secondbrain.transcripts as tx_mod
    from secondbrain.connectors.imap_email import ImapEmailConnector

    def boom(*a, **kw):
        raise RuntimeError("network unreachable")

    monkeypatch.setattr(tx_mod, "match_calendar_event", boom)

    class FakeCfg:
        pass

    c = ImapEmailConnector()
    doc = c._maybe_build_transcript_doc(
        from_="noreply@granola.so",
        subject="Notes from sync",
        body=GRANOLA_SAMPLE,
        date_hdr="", mtime=0.0,
        folder="Granola", msg_id="m1", uid=b"1",
        cfg=FakeCfg(),
    )
    assert doc is not None
    assert doc.metadata["event_id"] == ""


def test_imap_connector_uses_recording_timestamp_as_mtime():
    """When Plaud's body says recorded_at = X, that beats the email's
    Date header so time-decay surfaces lectures by when they happened,
    not when Plaud got around to emailing them."""
    from secondbrain.connectors.imap_email import ImapEmailConnector

    body = """\
Title: BME 410
Recorded on: 2026-04-15 14:30:00

Speaker 1: x
Speaker 2: y
Speaker 1: z
"""
    c = ImapEmailConnector()
    doc = c._maybe_build_transcript_doc(
        from_="noreply@plaud.app", subject="BME 410",
        body=body, date_hdr="", mtime=1700000000.0,
        folder="Plaud", msg_id="m", uid=b"1",
    )
    assert doc is not None
    # Recorded_at parses to ~ 1.776e9 (Apr 2026); not the email's
    # 1.7e9 (Nov 2023). So mtime should match the recording.
    assert doc.mtime > 1.77e9
