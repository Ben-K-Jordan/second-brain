"""Phase 35-37: Canvas connector parsers, application tracker CRUD,
resume-fit scoring."""

from __future__ import annotations

import time
from dataclasses import replace

import pytest

from secondbrain.connectors.canvas import (
    CanvasConnector,
    _format_when,
    _iso_to_ts,
    _next_link,
    _strip_html,
)
from secondbrain.db import (
    APPLICATION_STATUSES,
    application_create,
    application_delete,
    application_find_by_url,
    application_get,
    application_list,
    application_set_status,
    applied_role_urls,
)
from secondbrain.resume import (
    FIT_BANDS,
    cosine,
    fit_label,
    score_against_text,
)

# ============================== Canvas ================================

def test_canvas_disabled_without_env(tmp_cfg, monkeypatch):
    monkeypatch.delenv("CANVAS_BASE_URL", raising=False)
    monkeypatch.delenv("CANVAS_TOKEN", raising=False)
    assert CanvasConnector().is_enabled(tmp_cfg) is False


def test_canvas_disabled_with_only_url(tmp_cfg, monkeypatch):
    monkeypatch.setenv("CANVAS_BASE_URL", "https://canvas.illinois.edu")
    monkeypatch.delenv("CANVAS_TOKEN", raising=False)
    assert CanvasConnector().is_enabled(tmp_cfg) is False


def test_canvas_enabled_with_both(tmp_cfg, monkeypatch):
    monkeypatch.setenv("CANVAS_BASE_URL", "https://canvas.illinois.edu")
    monkeypatch.setenv("CANVAS_TOKEN", "test")
    assert CanvasConnector().is_enabled(tmp_cfg) is True


def test_canvas_strip_html_paragraphs():
    out = _strip_html("<p>Hello</p><p>World</p>")
    assert "Hello" in out and "World" in out
    assert "<p>" not in out


def test_canvas_strip_html_lists():
    """Lists need newlines so each item stays distinct in the embedded text."""
    out = _strip_html("<ul><li>A</li><li>B</li><li>C</li></ul>")
    # Item separation matters for embedding quality.
    assert "A" in out and "B" in out and "C" in out


def test_canvas_iso_to_ts_handles_z_suffix():
    ts = _iso_to_ts("2026-04-15T10:00:00Z")
    assert ts > 0


def test_canvas_iso_to_ts_handles_offset():
    ts = _iso_to_ts("2026-04-15T10:00:00+00:00")
    assert ts > 0


def test_canvas_iso_to_ts_garbage_returns_zero():
    """We use 0.0 (not now) as the sentinel for unparseable dates so
    the window-filter doesn't accidentally include garbage rows."""
    assert _iso_to_ts("not a date") == 0.0
    assert _iso_to_ts(None) == 0.0
    assert _iso_to_ts("") == 0.0


def test_canvas_format_when_zero_returns_empty():
    assert _format_when(0) == ""


def test_canvas_next_link_extracts_rel_next():
    header = (
        '<https://canvas.example.edu/api/v1/courses?page=2>; rel="next", '
        '<https://canvas.example.edu/api/v1/courses?page=10>; rel="last"'
    )
    assert _next_link(header) == "https://canvas.example.edu/api/v1/courses?page=2"


def test_canvas_next_link_no_next_returns_none():
    header = '<https://canvas.example.edu/api/v1/courses?page=10>; rel="last"'
    assert _next_link(header) is None


def test_canvas_next_link_empty_header():
    assert _next_link("") is None
    assert _next_link(None) is None


def test_canvas_render_assignment_with_due_date():
    """The render path stores due_at as mtime so time-decay surfaces
    upcoming assignments. Verified via the exposed render method."""
    c = CanvasConnector()
    asgn = {
        "id": 42,
        "name": "Project 1",
        "html_url": "https://canvas.example.edu/courses/1/assignments/42",
        "due_at": "2026-05-01T23:59:00Z",
        "points_possible": 100,
        "submission_types": ["online_upload"],
        "description": "<p>Build a thing.</p>",
        "submission": {
            "workflow_state": "submitted", "score": None, "grade": None,
            "submitted_at": "2026-04-30T20:00:00Z",
        },
    }
    doc = c._render_assignment(
        "https://canvas.example.edu", 1, "CS 374", "CS374", asgn,
    )
    assert doc is not None
    assert doc.virtual_path == "canvas://assignment/1/42"
    assert "[CS374]" in doc.title
    assert "Project 1" in doc.title
    assert "Build a thing" in doc.content
    assert "Due: 2026-05-01" in doc.content
    assert "Points: 100" in doc.content
    assert "submitted" in doc.content
    assert doc.metadata["kind"] == "assignment"
    assert doc.metadata["points"] == 100
    # mtime should be the due date (in seconds since epoch).
    assert doc.mtime == _iso_to_ts("2026-05-01T23:59:00Z")


def test_canvas_render_assignment_without_due_date_uses_now():
    """A project description without a due_at shouldn't sink to the bottom
    of search results - we use now() as the mtime."""
    c = CanvasConnector()
    asgn = {"id": 7, "name": "Project description", "due_at": None}
    doc = c._render_assignment("https://x", 1, "Course", "C", asgn)
    assert doc is not None
    assert abs(doc.mtime - time.time()) < 5


def test_canvas_render_announcement_minimal():
    c = CanvasConnector()
    ann = {
        "id": 9, "title": "Midterm details",
        "message": "<p>The midterm is next Friday.</p>",
        "posted_at": "2026-04-15T10:00:00Z",
        "author": {"display_name": "Prof Smith"},
    }
    doc = c._render_announcement(
        "https://canvas.example.edu", 1, "BME 410", "BME410", ann,
    )
    assert doc is not None
    assert doc.virtual_path == "canvas://announcement/1/9"
    assert "Midterm details" in doc.title
    assert "midterm is next Friday" in doc.content
    assert doc.metadata["author"] == "Prof Smith"


def test_canvas_render_syllabus_skipped_when_empty():
    c = CanvasConnector()
    course = {"name": "X", "syllabus_body": ""}
    assert c._render_syllabus("https://x", 1, "X", course) is None


def test_canvas_render_assignment_missing_id_returns_none():
    c = CanvasConnector()
    assert c._render_assignment("https://x", 1, "C", "C", {"name": "no id"}) is None


# ========================= Application tracker ========================

def test_application_create_minimal(fresh_db):
    aid = application_create(fresh_db, "Anthropic", "PM Intern")
    row = application_get(fresh_db, aid)
    assert row is not None
    assert row["company"] == "Anthropic"
    assert row["status"] == "applied"
    assert row["applied_at"] > 0


def test_application_create_full(fresh_db):
    aid = application_create(
        fresh_db, "Stripe", "TPM Intern",
        role_url="https://stripe.com/jobs/123",
        source="referral",
        notes="contacted via Sarah",
    )
    row = application_get(fresh_db, aid)
    assert row["role_url"] == "https://stripe.com/jobs/123"
    assert row["source"] == "referral"
    assert row["notes"] == "contacted via Sarah"


def test_application_create_invalid_status_raises(fresh_db):
    with pytest.raises(ValueError, match="unknown status"):
        application_create(fresh_db, "X", "Y", status="not-real")


def test_application_set_status_round_trip(fresh_db):
    aid = application_create(fresh_db, "X", "Y")
    application_set_status(fresh_db, aid, "interview")
    assert application_get(fresh_db, aid)["status"] == "interview"
    application_set_status(fresh_db, aid, "offer", notes="verbal")
    row = application_get(fresh_db, aid)
    assert row["status"] == "offer"
    assert row["notes"] == "verbal"


def test_application_set_status_invalid_raises(fresh_db):
    aid = application_create(fresh_db, "X", "Y")
    with pytest.raises(ValueError, match="unknown status"):
        application_set_status(fresh_db, aid, "bogus")


def test_application_list_filters(fresh_db):
    application_create(fresh_db, "A", "1", status="applied")
    application_create(fresh_db, "A", "2", status="interview")
    application_create(fresh_db, "B", "3", status="applied")
    rows = application_list(fresh_db, status="applied")
    assert len(rows) == 2
    rows = application_list(fresh_db, company="a")  # case-insensitive
    assert len(rows) == 2


def test_application_find_by_url(fresh_db):
    application_create(fresh_db, "X", "Y", role_url="https://x.com/job/1")
    found = application_find_by_url(fresh_db, "https://x.com/job/1")
    assert found is not None
    assert found["company"] == "X"
    assert application_find_by_url(fresh_db, "https://other.com") is None
    assert application_find_by_url(fresh_db, "") is None


def test_applied_role_urls(fresh_db):
    application_create(fresh_db, "A", "1", role_url="https://a.com/1")
    application_create(fresh_db, "B", "2", role_url="https://b.com/2")
    application_create(fresh_db, "C", "3")  # no URL — shouldn't appear
    urls = applied_role_urls(fresh_db)
    assert urls == {"https://a.com/1", "https://b.com/2"}


def test_application_delete(fresh_db):
    aid = application_create(fresh_db, "X", "Y")
    application_delete(fresh_db, aid)
    assert application_get(fresh_db, aid) is None


def test_all_statuses_in_constant():
    """Pin the closed taxonomy. Adding a new state should be a deliberate
    schema change, not a typo."""
    assert "applied" in APPLICATION_STATUSES
    assert "interview" in APPLICATION_STATUSES
    assert "offer" in APPLICATION_STATUSES
    assert "rejected" in APPLICATION_STATUSES


# ----- watchlist diff respects "already applied" --------------------

def test_compute_new_paths_skips_already_applied(fresh_db):
    """If the user records an application against a posting URL, the
    watchlist runner should not re-flag that posting as 'new' on the
    next run."""
    from secondbrain.db import (
        watchlist_create,
        watchlist_run_record_finish,
        watchlist_run_record_start,
    )
    from secondbrain.watchlist import _compute_new_paths

    wid = watchlist_create(fresh_db, "pm", "q")

    # Run 1: 2 results.
    rid1 = watchlist_run_record_start(fresh_db, wid)
    cites1 = [
        {"file_path": "https://anthropic.com/job/1"},
        {"file_path": "https://stripe.com/job/2"},
    ]
    np1, _ = _compute_new_paths(fresh_db, wid, rid1, cites1)
    watchlist_run_record_finish(
        fresh_db, rid1, citations_json='[{"file_path":"https://anthropic.com/job/1"},{"file_path":"https://stripe.com/job/2"}]',
        new_count=len(np1),
    )

    # User applies to the Stripe one.
    application_create(
        fresh_db, "Stripe", "TPM Intern", role_url="https://stripe.com/job/2",
    )

    # Run 2: same two results + a new figma posting. The Stripe one is
    # already applied; only the figma one should count as new.
    rid2 = watchlist_run_record_start(fresh_db, wid)
    cites2 = [
        {"file_path": "https://anthropic.com/job/1"},
        {"file_path": "https://stripe.com/job/2"},
        {"file_path": "https://figma.com/job/3"},
    ]
    new_paths, _ = _compute_new_paths(fresh_db, wid, rid2, cites2)
    assert "https://figma.com/job/3" in new_paths
    assert "https://stripe.com/job/2" not in new_paths, (
        "applied roles must not show up as 'new'"
    )


# ============================= Resume =================================

def test_cosine_perfect_match():
    a = [1.0, 2.0, 3.0]
    assert cosine(a, a) == pytest.approx(1.0)


def test_cosine_orthogonal():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_handles_empty():
    assert cosine([], [1.0, 2.0]) == 0.0
    assert cosine([1.0], []) == 0.0


def test_cosine_handles_size_mismatch():
    assert cosine([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


def test_fit_label_bands():
    assert fit_label(0.9) == "great fit"
    assert fit_label(0.65) == "great fit"  # exactly on the threshold
    assert fit_label(0.6) == "decent fit"
    assert fit_label(0.5) == "stretch"
    assert fit_label(0.3) == "weak"
    assert fit_label(0.0) == "weak"


def test_fit_bands_descend():
    """Bands should be sorted high-to-low so fit_label matches greedily."""
    thresholds = [t for t, _ in FIT_BANDS]
    assert thresholds == sorted(thresholds, reverse=True)


def test_score_against_text_no_resumes_returns_none(fake_embedder):
    assert score_against_text([], fake_embedder, "anything") is None


def test_score_against_text_picks_best_resume(fake_embedder):
    """When multiple resumes are configured, the highest-scoring one
    should win. The fake embedder is deterministic so we can verify the
    relative ordering."""
    from secondbrain.resume import ResumeProfile

    pm_emb = fake_embedder.embed_query("product manager pm internship strategy")
    eng_emb = fake_embedder.embed_query("software engineer python rust embedded")
    resumes = [
        ResumeProfile(name="pm", path="/x", text="...", embedding=pm_emb, indexed_at=0),
        ResumeProfile(name="eng", path="/y", text="...", embedding=eng_emb, indexed_at=0),
    ]
    # Score against text very similar to the PM resume.
    out = score_against_text(resumes, fake_embedder, "product manager pm internship strategy")
    assert out is not None
    score, name, label = out
    assert name == "pm"
    assert score > 0.9  # near-identical hash strings → near-1 cosine


def test_score_against_text_empty_text_returns_none(fake_embedder):
    from secondbrain.resume import ResumeProfile

    resumes = [ResumeProfile(name="x", path="/x", text="...",
                             embedding=fake_embedder.embed_query("x"),
                             indexed_at=0)]
    assert score_against_text(resumes, fake_embedder, "") is None


def test_resume_paths_resolves_env(tmp_path, tmp_cfg, monkeypatch):
    from secondbrain.resume import _resume_paths

    f = tmp_path / "resume.md"
    f.write_text("test resume content", encoding="utf-8")
    monkeypatch.setenv("RESUME_PATH", str(f))
    paths = _resume_paths(tmp_cfg)
    assert len(paths) == 1
    assert paths[0].name == "resume.md"


def test_resume_paths_dedups_config_and_env(tmp_path, tmp_cfg, monkeypatch):
    from secondbrain.resume import _resume_paths

    f = tmp_path / "resume.md"
    f.write_text("content", encoding="utf-8")
    monkeypatch.setenv("RESUME_PATH", str(f))
    cfg = replace(tmp_cfg, resume_paths=(str(f),))
    paths = _resume_paths(cfg)
    assert len(paths) == 1


def test_load_resumes_skips_missing_files(tmp_cfg, fake_embedder):
    from secondbrain.resume import load_resumes

    cfg = replace(tmp_cfg, resume_paths=("/nonexistent/path.md",))
    assert load_resumes(cfg, fake_embedder) == []


def test_load_resumes_returns_empty_when_unconfigured(tmp_cfg, fake_embedder):
    from secondbrain.resume import load_resumes

    assert load_resumes(tmp_cfg, fake_embedder) == []


def test_load_resumes_round_trip(tmp_path, tmp_cfg, fake_embedder):
    """End-to-end: write a resume, load it, get back a profile with an
    embedding of the right dim."""
    from secondbrain.resume import load_resumes

    f = tmp_path / "resume-pm.md"
    f.write_text(
        "# Resume\nProduct manager intern targeting summer 2026.",
        encoding="utf-8",
    )
    cfg = replace(tmp_cfg, resume_paths=(str(f),))
    profiles = load_resumes(cfg, fake_embedder)
    assert len(profiles) == 1
    assert profiles[0].name == "resume-pm"
    assert len(profiles[0].embedding) == fake_embedder.dim
    assert "summer 2026" in profiles[0].text
