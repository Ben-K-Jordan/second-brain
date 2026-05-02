"""Phase 76 + 77 + 78: external tools tests.

Coverage:
  - Phase 76: tasks_sync push/pull with stubbed Todoist API
  - Phase 77: vault_export folder layout + frontmatter + wikilinks
  - Phase 78: ReadwiseConnector parsing of paginated responses
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from secondbrain import tasks as tasks_mod
from secondbrain import tasks_sync, vault_export

# ============================ Phase 76 tasks_sync =====================

class _FakeTodoistResponse:
    def __init__(self, status_code: int, json_data=None):
        self.status_code = status_code
        self._json = json_data
    def json(self):
        if self._json is None:
            raise ValueError("no body")
        return self._json


class _FakeTodoistSession:
    """Stub the requests.Session interface used by tasks_sync. Records
    every call so tests can assert on the right endpoints firing."""

    def __init__(self, *, posts=None, gets=None):
        self.posts: list[tuple[str, dict]] = []
        self.gets: list[str] = []
        self.headers: dict = {}
        self._post_responses = posts or []
        self._get_responses = gets or {}

    def post(self, url, json=None, timeout=None):
        self.posts.append((url, json or {}))
        if not self._post_responses:
            return _FakeTodoistResponse(500)
        return self._post_responses.pop(0)

    def get(self, url, timeout=None):
        self.gets.append(url)
        # Match by suffix to keep tests readable.
        for suffix, resp in self._get_responses.items():
            if url.endswith(suffix):
                return resp
        return _FakeTodoistResponse(404)

    def close(self):
        pass


def test_push_open_tasks_skips_when_no_token(fresh_db, monkeypatch):
    monkeypatch.delenv("TODOIST_TOKEN", raising=False)
    tasks_mod.add_manual(fresh_db, "Buy milk")
    pushed, errors = tasks_sync.push_open_tasks(fresh_db)
    assert pushed == 0
    assert errors == 0


def test_push_open_tasks_creates_remote(fresh_db, monkeypatch):
    monkeypatch.setenv("TODOIST_TOKEN", "test-token")
    tasks_mod.add_manual(fresh_db, "Buy milk")
    fake = _FakeTodoistSession(
        posts=[_FakeTodoistResponse(200, {"id": "12345"})],
    )
    monkeypatch.setattr(tasks_sync, "_session", lambda token: fake)
    pushed, errors = tasks_sync.push_open_tasks(fresh_db)
    assert pushed == 1
    assert errors == 0
    assert len(fake.posts) == 1
    # Local task should now have the remote id.
    row = fresh_db.execute(
        "SELECT external_id, external_provider FROM tasks "
        "WHERE text = 'Buy milk'",
    ).fetchone()
    assert row["external_id"] == "12345"
    assert row["external_provider"] == "todoist"


def test_push_open_tasks_skips_already_synced(fresh_db, monkeypatch):
    monkeypatch.setenv("TODOIST_TOKEN", "test-token")
    tid = tasks_mod.add_manual(fresh_db, "Already synced")
    fresh_db.execute(
        "UPDATE tasks SET external_id = '999', external_provider = 'todoist' "
        "WHERE id = ?",
        (tid,),
    )
    fresh_db.commit()
    fake = _FakeTodoistSession()
    monkeypatch.setattr(tasks_sync, "_session", lambda token: fake)
    pushed, _ = tasks_sync.push_open_tasks(fresh_db)
    assert pushed == 0
    assert fake.posts == []


def test_push_handles_api_error(fresh_db, monkeypatch):
    monkeypatch.setenv("TODOIST_TOKEN", "test-token")
    tasks_mod.add_manual(fresh_db, "Will fail")
    fake = _FakeTodoistSession(
        posts=[_FakeTodoistResponse(500)],
    )
    monkeypatch.setattr(tasks_sync, "_session", lambda token: fake)
    pushed, errors = tasks_sync.push_open_tasks(fresh_db)
    assert pushed == 0
    assert errors == 1


def test_pull_completions_marks_done_locally(fresh_db, monkeypatch):
    """If Todoist says is_completed: true, our task moves to done."""
    monkeypatch.setenv("TODOIST_TOKEN", "test-token")
    tid = tasks_mod.add_manual(fresh_db, "Reply to recruiter")
    fresh_db.execute(
        "UPDATE tasks SET external_id = '111', external_provider = 'todoist' "
        "WHERE id = ?",
        (tid,),
    )
    fresh_db.commit()
    fake = _FakeTodoistSession(
        gets={"/tasks/111": _FakeTodoistResponse(
            200, {"id": "111", "is_completed": True},
        )},
    )
    monkeypatch.setattr(tasks_sync, "_session", lambda token: fake)
    pulled, errors = tasks_sync.pull_remote_completions(fresh_db)
    assert pulled == 1
    assert errors == 0
    t = tasks_mod.get(fresh_db, tid)
    assert t.status == "done"


def test_pull_404_cancels_locally(fresh_db, monkeypatch):
    """Remote 404 = user deleted on Todoist → cancel locally."""
    monkeypatch.setenv("TODOIST_TOKEN", "test-token")
    tid = tasks_mod.add_manual(fresh_db, "Vanished task")
    fresh_db.execute(
        "UPDATE tasks SET external_id = 'gone', external_provider = 'todoist' "
        "WHERE id = ?",
        (tid,),
    )
    fresh_db.commit()
    fake = _FakeTodoistSession(
        gets={"/tasks/gone": _FakeTodoistResponse(404)},
    )
    monkeypatch.setattr(tasks_sync, "_session", lambda token: fake)
    tasks_sync.pull_remote_completions(fresh_db)
    t = tasks_mod.get(fresh_db, tid)
    assert t.status == "cancelled"


def test_pull_skips_open_remote(fresh_db, monkeypatch):
    """is_completed: false → leave alone."""
    monkeypatch.setenv("TODOIST_TOKEN", "test-token")
    tid = tasks_mod.add_manual(fresh_db, "Still open")
    fresh_db.execute(
        "UPDATE tasks SET external_id = '222', external_provider = 'todoist' "
        "WHERE id = ?",
        (tid,),
    )
    fresh_db.commit()
    fake = _FakeTodoistSession(
        gets={"/tasks/222": _FakeTodoistResponse(
            200, {"id": "222", "is_completed": False},
        )},
    )
    monkeypatch.setattr(tasks_sync, "_session", lambda token: fake)
    pulled, _ = tasks_sync.pull_remote_completions(fresh_db)
    assert pulled == 0
    t = tasks_mod.get(fresh_db, tid)
    assert t.status == "open"


def test_run_if_due_skips_when_no_token(fresh_db, tmp_cfg, monkeypatch):
    monkeypatch.delenv("TODOIST_TOKEN", raising=False)
    assert tasks_sync.run_if_due(tmp_cfg, fresh_db) is False


# ============================ Phase 77 vault_export ===================

def _seed_file(conn, *, path, kind, body, mtime=None, indexed_at=None):
    n = mtime or time.time()
    cur = conn.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at, content_hash) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (path, n, len(body), kind, indexed_at or n, None),
    )
    fid = cur.lastrowid
    conn.execute(
        "INSERT INTO chunks(file_id, chunk_index, text) VALUES (?, ?, ?)",
        (fid, 0, body),
    )
    conn.commit()
    return fid


def test_vault_export_writes_files_per_kind(fresh_db, tmp_path):
    _seed_file(fresh_db, path="/notes/a.md", kind="document",
               body="# Note A\n\nContent A.")
    _seed_file(fresh_db, path="transcript://granola/x", kind="url",
               body="# Sprint planning\n\nMeeting body.")
    _seed_file(fresh_db, path="capture://ios/123", kind="capture",
               body="# Quick thought\n\nCapture body.")
    out = tmp_path / "vault"
    result = vault_export.export_vault(fresh_db, out)
    assert result.files_written == 3
    assert (out / "notes").exists()
    assert (out / "transcripts").exists()
    assert (out / "captures").exists()


def test_vault_export_skips_empty_chunks(fresh_db, tmp_path):
    """Files with no usable body shouldn't write empty .md files."""
    cur = fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at) "
        "VALUES ('/empty.md', ?, 0, 'document', ?)",
        (time.time(), time.time()),
    )
    fid = cur.lastrowid
    fresh_db.execute(
        "INSERT INTO chunks(file_id, chunk_index, text) VALUES (?, 0, '')",
        (fid,),
    )
    fresh_db.commit()
    out = tmp_path / "vault"
    result = vault_export.export_vault(fresh_db, out)
    assert result.files_written == 0


def test_vault_export_includes_frontmatter(fresh_db, tmp_path):
    _seed_file(fresh_db, path="/notes/x.md", kind="document",
               body="# X\n\nbody")
    out = tmp_path / "vault"
    vault_export.export_vault(fresh_db, out)
    md_files = list(out.rglob("*.md"))
    assert len(md_files) == 1
    text = md_files[0].read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "path: /notes/x.md" in text
    assert "kind: document" in text
    assert "indexed_at:" in text


def test_vault_export_renders_backlinks(fresh_db, tmp_path):
    fid_a = _seed_file(
        fresh_db, path="/notes/a.md", kind="document",
        body="# Note A\n\nbody",
    )
    fid_b = _seed_file(
        fresh_db, path="/notes/b.md", kind="document",
        body="# Note B\n\nbody",
    )
    fresh_db.execute(
        "INSERT INTO backlinks(src_file_id, dst_file_id, score, created_at) "
        "VALUES (?, ?, 0.1, ?)",
        (fid_a, fid_b, time.time()),
    )
    fresh_db.commit()
    out = tmp_path / "vault"
    vault_export.export_vault(fresh_db, out)
    a_md = list((out / "notes").glob("a*.md"))[0]
    text = a_md.read_text(encoding="utf-8")
    assert "## Related" in text
    assert "[[" in text  # wikilink syntax


def test_vault_export_renders_people_mentions(fresh_db, tmp_path):
    fid = _seed_file(
        fresh_db, path="/notes/m.md", kind="document",
        body="# Meeting\n\nSarah was there.",
    )
    pid = fresh_db.execute(
        "INSERT INTO people(canonical_name, display_name, "
        " first_seen_at, last_seen_at) "
        "VALUES ('sarah', 'Sarah', ?, ?) RETURNING id",
        (time.time(), time.time()),
    ).fetchone()["id"]
    fresh_db.execute(
        "INSERT INTO person_mentions(person_id, chunk_id, file_id, mtime) "
        "SELECT ?, c.id, ?, ? FROM chunks c WHERE c.file_id = ? LIMIT 1",
        (pid, fid, time.time(), fid),
    )
    fresh_db.commit()
    out = tmp_path / "vault"
    vault_export.export_vault(fresh_db, out)
    md_files = list(out.rglob("*.md"))
    text = md_files[0].read_text(encoding="utf-8")
    assert "## People" in text
    assert "[[Sarah]]" in text


def test_vault_export_clean_wipes_existing(fresh_db, tmp_path):
    """clean=True should remove old .md files in our subdirs."""
    out = tmp_path / "vault"
    (out / "notes").mkdir(parents=True)
    stale = out / "notes" / "stale.md"
    stale.write_text("old", encoding="utf-8")
    _seed_file(fresh_db, path="/notes/new.md", kind="document",
               body="# N\n\nbody")
    vault_export.export_vault(fresh_db, out, clean=True)
    assert not stale.exists()


def test_vault_export_clean_preserves_unrelated_dirs(fresh_db, tmp_path):
    """If user has unrelated content in vault, clean shouldn't nuke it."""
    out = tmp_path / "vault"
    obsidian_config = out / ".obsidian"
    obsidian_config.mkdir(parents=True)
    obs_settings = obsidian_config / "config.json"
    obs_settings.write_text("{}", encoding="utf-8")
    vault_export.export_vault(fresh_db, out, clean=True)
    assert obs_settings.exists()


def test_vault_export_handles_filename_collisions(fresh_db, tmp_path):
    """Two files with the same slug should suffix to avoid overwriting."""
    _seed_file(
        fresh_db, path="/notes/a/foo.md", kind="document",
        body="# Foo\n\none",
    )
    _seed_file(
        fresh_db, path="/notes/b/foo.md", kind="document",
        body="# Foo\n\ntwo",
    )
    out = tmp_path / "vault"
    result = vault_export.export_vault(fresh_db, out)
    assert result.files_written == 2
    # Both must coexist on disk.
    assert len(list((out / "notes").glob("*.md"))) == 2


def test_vault_export_limit_caps_files(fresh_db, tmp_path):
    for i in range(5):
        _seed_file(
            fresh_db, path=f"/notes/{i}.md", kind="document",
            body=f"# {i}\n\nbody",
        )
    out = tmp_path / "vault"
    result = vault_export.export_vault(fresh_db, out, limit=2)
    assert result.files_written == 2


# ============================ Phase 78 Readwise =======================

@dataclass
class _RWResp:
    status_code: int
    payload: dict = field(default_factory=dict)
    def json(self):
        return self.payload


class _FakeRWSession:
    def __init__(self, route_payloads):
        self.headers = {}
        self.route_payloads = route_payloads
    def get(self, url, timeout=None):
        for substr, payload in self.route_payloads.items():
            if substr in url:
                return _RWResp(200, payload)
        return _RWResp(404)
    def close(self):
        pass


def test_readwise_disabled_without_token(monkeypatch, tmp_cfg):
    from secondbrain.connectors.readwise import ReadwiseConnector
    monkeypatch.delenv("READWISE_TOKEN", raising=False)
    assert ReadwiseConnector().is_enabled(tmp_cfg) is False


def test_readwise_renders_book_with_highlights(monkeypatch, tmp_cfg):
    from secondbrain.connectors import readwise as rw_mod
    from secondbrain.connectors.readwise import ReadwiseConnector
    monkeypatch.setenv("READWISE_TOKEN", "test")
    fake = _FakeRWSession({
        "/books/": {
            "results": [
                {
                    "id": 1, "title": "Sapiens",
                    "author": "Y. N. Harari",
                    "category": "books",
                    "source": "kindle",
                    "source_url": "https://example.com/sapiens",
                },
            ],
            "next": None,
        },
        "/highlights/": {
            "results": [
                {
                    "id": 100, "book_id": 1,
                    "text": "Cognitive revolution started 70k years ago.",
                    "note": "ties to language origin",
                    "location": 100,
                    "highlighted_at": "2026-04-15T10:00:00Z",
                    "tags": [{"name": "history"}],
                },
                {
                    "id": 101, "book_id": 1,
                    "text": "Wheat domesticated humans.",
                    "location": 200,
                    "highlighted_at": "2026-04-16T10:00:00Z",
                },
            ],
            "next": None,
        },
    })
    monkeypatch.setattr(rw_mod.requests, "Session", lambda: fake)
    docs = list(ReadwiseConnector().fetch(tmp_cfg))
    assert len(docs) == 1
    doc = docs[0]
    assert "Sapiens" in doc.title
    assert "Y. N. Harari" in doc.content
    assert "Cognitive revolution" in doc.content
    assert "Wheat domesticated" in doc.content
    assert doc.metadata["highlight_count"] == 2


def test_readwise_groups_highlights_by_book(monkeypatch, tmp_cfg):
    """Two books with one highlight each → two docs."""
    from secondbrain.connectors import readwise as rw_mod
    from secondbrain.connectors.readwise import ReadwiseConnector
    monkeypatch.setenv("READWISE_TOKEN", "test")
    fake = _FakeRWSession({
        "/books/": {
            "results": [
                {"id": 1, "title": "A", "author": "x", "category": "books"},
                {"id": 2, "title": "B", "author": "y", "category": "articles"},
            ],
            "next": None,
        },
        "/highlights/": {
            "results": [
                {"id": 1, "book_id": 1, "text": "from A"},
                {"id": 2, "book_id": 2, "text": "from B"},
            ],
            "next": None,
        },
    })
    monkeypatch.setattr(rw_mod.requests, "Session", lambda: fake)
    docs = list(ReadwiseConnector().fetch(tmp_cfg))
    titles = {d.title for d in docs}
    assert "[highlights] A" in titles
    assert "[highlights] B" in titles


def test_readwise_skips_book_without_highlights(monkeypatch, tmp_cfg):
    """Books with no highlights in the window shouldn't emit empty docs."""
    from secondbrain.connectors import readwise as rw_mod
    from secondbrain.connectors.readwise import ReadwiseConnector
    monkeypatch.setenv("READWISE_TOKEN", "test")
    fake = _FakeRWSession({
        "/books/": {
            "results": [
                {"id": 1, "title": "Empty", "author": "x", "category": "books"},
            ],
            "next": None,
        },
        "/highlights/": {"results": [], "next": None},
    })
    monkeypatch.setattr(rw_mod.requests, "Session", lambda: fake)
    docs = list(ReadwiseConnector().fetch(tmp_cfg))
    assert docs == []


def test_readwise_handles_http_error(monkeypatch, tmp_cfg):
    from secondbrain.connectors import readwise as rw_mod
    from secondbrain.connectors.readwise import ReadwiseConnector
    monkeypatch.setenv("READWISE_TOKEN", "test")

    class _BadSession:
        headers = {}
        def get(self, url, timeout=None):
            return _RWResp(500)
        def close(self):
            pass

    monkeypatch.setattr(rw_mod.requests, "Session", lambda: _BadSession())
    docs = list(ReadwiseConnector().fetch(tmp_cfg))
    assert docs == []
