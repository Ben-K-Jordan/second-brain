"""Smoke tests for the Phase 44 / 47 / 56 dashboard pages.

These verify the new pages register, render without exceptions, and
honour the `add` / `done` POST endpoints. Full HTML-content tests are
brittle and not what we want — we just check a few load-bearing
strings + 200 status codes.

The dashboard's `get_state()` reuses module-level singletons for the
DB connection. Tests run against the user's real index path, which
is fine because they're read-only / minimally-mutating; the side
effects (a test task showing up) are tagged with a 'pytest_marker'
text so the user can clean up if needed. (We also clean up after
ourselves below.)
"""

from __future__ import annotations

import pytest


def _patch_dashboard(monkeypatch, tmp_path, fake_embedder):
    """Hook the dashboard up to a temp DB + the deterministic fake
    embedder from conftest. Returns the cfg so seed-tests can grab
    a side connection."""
    from secondbrain.config import Config

    cfg = Config(data_dir=tmp_path / "data")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("secondbrain.dashboard.load_config", lambda: cfg)
    monkeypatch.setattr(
        "secondbrain.dashboard.make_embedder", lambda c: fake_embedder,
    )
    # Reranker is optional; stub to None so we don't try to load
    # cross-encoder weights in tests.
    monkeypatch.setattr("secondbrain.dashboard.make_reranker", lambda c: None)
    return cfg


@pytest.fixture
def client(monkeypatch, tmp_path, fake_embedder):
    """A test client backed by a fresh temp DB, so we don't pollute
    the user's real index. Uses the deterministic fake embedder
    so we don't need sentence-transformers / VOYAGE_API_KEY."""
    from fastapi.testclient import TestClient

    from secondbrain.dashboard import create_app

    _patch_dashboard(monkeypatch, tmp_path, fake_embedder)
    app = create_app()
    return TestClient(app)


# ============================ /brief ==================================

def test_brief_page_renders_empty(client):
    """No data → quiet-day fallback. Page should still 200 with
    the brief headline."""
    r = client.get("/brief")
    assert r.status_code == 200
    assert "Daily brief" in r.text


def test_brief_page_includes_secondary_text(client):
    """The page should also mention the email-it-now hint."""
    r = client.get("/brief")
    assert "secondbrain brief send" in r.text


# ============================ /tasks ==================================

def test_tasks_page_renders_empty(client):
    r = client.get("/tasks")
    assert r.status_code == 200
    assert "Inbox zero" in r.text


def test_tasks_add_creates_task(client):
    """POST /tasks/add → 303 redirect to /tasks; the new task should
    show up in the next GET."""
    r = client.post("/tasks/add", data={"text": "Smoke-test task"},
                    follow_redirects=False)
    assert r.status_code == 303
    r = client.get("/tasks")
    assert "Smoke-test task" in r.text
    # Inbox-zero string should be gone now.
    assert "Inbox zero" not in r.text


def test_tasks_done_marks_complete(client):
    """Add then complete via the dashboard. The completed task should
    move out of 'Open' into 'Recently done'."""
    client.post("/tasks/add", data={"text": "About to be done"})
    # Find the task id by hitting the page and parsing — simple grep
    # approach since we control the markup. Format: <code>#N</code>
    r = client.get("/tasks")
    import re
    m = re.search(r"<code>#(\d+)</code> About to be done", r.text)
    assert m, "Task should appear in Open"
    tid = int(m.group(1))
    r2 = client.post(f"/tasks/{tid}/done", follow_redirects=False)
    assert r2.status_code == 303
    # After completion, the task moves out of Open.
    r3 = client.get("/tasks")
    # 'Recently done' section should now contain the text.
    assert "About to be done" in r3.text
    # Open list should NOT have a clickable done button on it any more
    # — the task moved sections. We assert the body string still
    # appears (in done) but in a context without `done` button.
    open_section = r3.text.split("Recently done")[0]
    assert "About to be done" not in open_section


def test_tasks_add_empty_text_is_handled(client):
    """Posting empty text shouldn't 500 — the form's `required`
    attribute catches it client-side, but the server should also
    no-op gracefully if it sneaks through."""
    # FastAPI's Form(...) requires the field, so an empty value still
    # passes through. Verify no crash.
    r = client.post("/tasks/add", data={"text": "   "},
                    follow_redirects=False)
    assert r.status_code == 303


# ============================ /health =================================

def test_health_page_renders_empty_state(client):
    """No Oura data → friendly 'set me up' message instead of a crash."""
    r = client.get("/health")
    assert r.status_code == 200
    assert "No Oura data yet" in r.text


def test_healthz_endpoint_is_separate_from_health_ui(client):
    """Liveness probe must still exist + return JSON. The Phase 56
    UI page took /health, so the probe moved to /healthz."""
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_health_page_renders_data_when_metrics_present(
    monkeypatch, tmp_path, fake_embedder,
):
    """Seed health_metrics + verify the dashboard renders the cards.

    We pre-populate the DB before the dashboard binds to it, so when
    the first request comes in the page sees data."""
    import time

    from fastapi.testclient import TestClient

    from secondbrain.dashboard import create_app
    from secondbrain.db import connect, init_schema

    cfg = _patch_dashboard(monkeypatch, tmp_path, fake_embedder)
    # Pre-init the schema + seed metrics through a side connection.
    seed_conn = connect(cfg.db_path)
    init_schema(seed_conn, fake_embedder.dim, fake_embedder.name)
    for date, metric, value in [
        ("2026-04-13", "sleep_score", 80),
        ("2026-04-14", "sleep_score", 82),
        ("2026-04-15", "sleep_score", 90),
    ]:
        seed_conn.execute(
            "INSERT INTO health_metrics"
            "(date, metric, value, source, recorded_at) "
            "VALUES (?, ?, ?, 'oura', ?)",
            (date, metric, value, time.time()),
        )
    seed_conn.commit()
    seed_conn.close()

    app = create_app()
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert "Sleep" in r.text  # headline card label
    assert "<svg" in r.text  # sparkline rendered


# ============================ markdown helper =========================

def test_markdown_to_html_block_renders_basic_structure():
    """The brief view's tiny markdown subset needs to do H1-3, bullets,
    blockquotes, paragraphs, and escape <>&."""
    from secondbrain.dashboard import _markdown_to_html_block

    md = (
        "# Title\n\n"
        "## Section\n\n"
        "- one\n- two\n\n"
        "> quoted bit\n\n"
        "Regular paragraph with <html>.\n"
    )
    html = _markdown_to_html_block(md)
    assert "<h2>Title</h2>" in html
    assert "<h3>Section</h3>" in html
    assert "<li>one</li>" in html
    assert "<blockquote>quoted bit</blockquote>" in html
    assert "&lt;html&gt;" in html


# ============================ svg sparkline ==========================

def test_svg_sparkline_renders_polyline_with_multiple_points():
    from secondbrain.dashboard import _svg_sparkline
    svg = _svg_sparkline([1.0, 2.0, 3.0], width=100, height=20)
    assert "<svg" in svg
    assert "<polyline" in svg
    # Three points → three coordinate pairs in `points=`.
    assert svg.count(",") >= 3


def test_svg_sparkline_handles_single_point():
    from secondbrain.dashboard import _svg_sparkline
    svg = _svg_sparkline([5.0], width=100, height=20)
    # A single point should be a circle, not a polyline.
    assert "<circle" in svg
    assert "<polyline" not in svg


def test_svg_sparkline_handles_empty():
    from secondbrain.dashboard import _svg_sparkline
    assert _svg_sparkline([], width=100, height=20) == ""


def test_svg_sparkline_handles_constant_values():
    """Flat series → must not div-by-zero on (max - min)."""
    from secondbrain.dashboard import _svg_sparkline
    svg = _svg_sparkline([5.0, 5.0, 5.0], width=100, height=20)
    assert "<polyline" in svg


# ===================== file view + backlinks =========================

def test_file_view_shows_backlinks_when_present(
    monkeypatch, tmp_path, fake_embedder,
):
    """A file with neighbours in the backlinks table should render a
    'See also' card on its file view page."""
    import time

    from fastapi.testclient import TestClient

    from secondbrain import backlinks
    from secondbrain.dashboard import create_app
    from secondbrain.db import (
        connect,
        init_schema,
        replace_chunks,
        upsert_file,
    )

    cfg = _patch_dashboard(monkeypatch, tmp_path, fake_embedder)
    seed_conn = connect(cfg.db_path)
    init_schema(seed_conn, fake_embedder.dim, fake_embedder.name)

    # Seed two related docs (single chunk each — pass min_chunks=1).
    fid_a = upsert_file(
        seed_conn, path="A.md", mtime=time.time(), size=10,
        kind="document", content_hash=None,
    )
    replace_chunks(seed_conn, fid_a, [
        ("# Doc A\n\nAlpha content here.", fake_embedder.embed_query("alpha")),
    ])
    fid_b = upsert_file(
        seed_conn, path="B.md", mtime=time.time(), size=10,
        kind="document", content_hash=None,
    )
    replace_chunks(seed_conn, fid_b, [
        ("# Doc B\n\nAlpha-ish content.", fake_embedder.embed_query("alphaish")),
    ])
    backlinks.link_doc(seed_conn, fid_a, max_distance=10.0, min_chunks=1)
    seed_conn.commit()
    seed_conn.close()

    app = create_app()
    client = TestClient(app)
    r = client.get("/file?path=A.md")
    assert r.status_code == 200
    # The "See also" card should render with B as a link.
    assert "See also" in r.text
    assert "Doc B" in r.text or "B.md" in r.text


def test_file_view_skips_backlinks_when_none(client):
    """Empty 'See also' shouldn't render a card at all — no point
    showing a section header above zero rows."""
    # Empty DB; no files. File-view will 404-style render.
    r = client.get("/file?path=/missing.md")
    assert r.status_code == 200
    assert "See also" not in r.text


# ============== polish v3 dashboard pages ============================

def test_people_page_renders_empty(client):
    r = client.get("/people")
    assert r.status_code == 200
    assert "people backfill" in r.text


def test_habits_page_renders_empty(client):
    r = client.get("/habits")
    assert r.status_code == 200
    assert "habits add" in r.text


def test_journal_page_renders_with_form(client):
    """Journal page should always render the today-entry form."""
    r = client.get("/journal")
    assert r.status_code == 200
    # The form uses single-quoted attrs in the f-string template;
    # match either quoting style so we don't lock the renderer.
    assert "name='mood'" in r.text or 'name="mood"' in r.text
    assert "name='text'" in r.text or 'name="text"' in r.text


def test_journal_add_persists_entry(client):
    """POSTing the form should land an entry that the GET sees."""
    r = client.post(
        "/journal/add",
        data={"text": "Test entry", "mood": "4"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    r2 = client.get("/journal")
    assert "Test entry" in r2.text


def test_projects_page_renders_empty(client):
    r = client.get("/projects")
    assert r.status_code == 200
    assert "project new" in r.text


def test_drafts_page_renders_empty(client):
    r = client.get("/drafts")
    assert r.status_code == 200
    assert "No pending drafts" in r.text


def test_habits_checkin_creates_row(monkeypatch, tmp_path, fake_embedder):
    """POST /habits/<id>/checkin should land a check-in row."""
    from fastapi.testclient import TestClient

    from secondbrain import personal
    from secondbrain.dashboard import create_app
    from secondbrain.db import connect, init_schema

    cfg = _patch_dashboard(monkeypatch, tmp_path, fake_embedder)
    # Seed a habit through a side connection that shares the DB.
    seed_conn = connect(cfg.db_path)
    init_schema(seed_conn, fake_embedder.dim, fake_embedder.name)
    hid = personal.add_habit(seed_conn, "test-habit")
    seed_conn.close()

    app = create_app()
    client_inner = TestClient(app)
    r = client_inner.post(
        f"/habits/{hid}/checkin", follow_redirects=False,
    )
    assert r.status_code == 303
    # Verify the check-in landed via a fresh side connection.
    check_conn = connect(cfg.db_path)
    n = check_conn.execute(
        "SELECT COUNT(*) AS n FROM habit_checkins WHERE habit_id = ?",
        (hid,),
    ).fetchone()["n"]
    check_conn.close()
    assert n == 1


def test_file_view_renders_summary_when_present(
    monkeypatch, tmp_path, fake_embedder,
):
    """Phase 74 — TL;DR should render at the top of the file view
    when one exists."""
    import time

    from fastapi.testclient import TestClient

    from secondbrain.dashboard import create_app
    from secondbrain.db import connect, init_schema

    cfg = _patch_dashboard(monkeypatch, tmp_path, fake_embedder)
    seed_conn = connect(cfg.db_path)
    init_schema(seed_conn, fake_embedder.dim, fake_embedder.name)
    cur = seed_conn.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at) "
        "VALUES ('/notes/x.md', ?, 1, 'document', ?)",
        (time.time(), time.time()),
    )
    fid = cur.lastrowid
    seed_conn.execute(
        "INSERT INTO chunks(file_id, chunk_index, text) "
        "VALUES (?, 0, '# X\n\nbody')",
        (fid,),
    )
    # Inject a synthesis summary directly (bypass LLM).
    from secondbrain.synthesis import _ensure_schema
    _ensure_schema(seed_conn)
    import json
    seed_conn.execute(
        "INSERT INTO doc_summaries"
        "(file_id, tldr, key_points_json, generated_at, input_chars) "
        "VALUES (?, ?, ?, ?, ?)",
        (fid, "Tight TLDR.", json.dumps(["a", "b"]), time.time(), 100),
    )
    seed_conn.commit()
    seed_conn.close()

    app = create_app()
    client_inner = TestClient(app)
    r = client_inner.get("/file?path=/notes/x.md")
    assert r.status_code == 200
    assert "Summary" in r.text
    assert "Tight TLDR" in r.text


def test_file_view_redacts_sensitive_in_chunks(
    monkeypatch, tmp_path, fake_embedder,
):
    """Phase 88 polish — file view chunk previews go through redact.
    A doc with an SSN in its body shouldn't render the SSN."""
    import time

    from fastapi.testclient import TestClient

    from secondbrain.dashboard import create_app
    from secondbrain.db import connect, init_schema

    cfg = _patch_dashboard(monkeypatch, tmp_path, fake_embedder)
    seed_conn = connect(cfg.db_path)
    init_schema(seed_conn, fake_embedder.dim, fake_embedder.name)
    cur = seed_conn.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at) "
        "VALUES ('/notes/secret.md', ?, 1, 'document', ?)",
        (time.time(), time.time()),
    )
    fid = cur.lastrowid
    seed_conn.execute(
        "INSERT INTO chunks(file_id, chunk_index, text) "
        "VALUES (?, 0, ?)",
        (fid, "Personal info: 123-45-6789, never share."),
    )
    seed_conn.commit()
    seed_conn.close()

    app = create_app()
    client_inner = TestClient(app)
    r = client_inner.get("/file?path=/notes/secret.md")
    assert r.status_code == 200
    assert "[REDACTED:ssn]" in r.text
    assert "123-45-6789" not in r.text
