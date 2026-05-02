"""Phase 69 + 70 + 71: mobile capture tests.

Coverage:
  - /api/capture endpoint: auth, content/url paths, kind tagging
  - IMAP capture detection: folder match + subject prefix match
  - photo_capture_folder folds into _resolve_folders
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

# ============================ /api/capture ============================

@pytest.fixture
def app_client(monkeypatch, tmp_path, fake_embedder):
    """Test client wired to a temp DB + valid bearer token. Each test
    gets its own client so state doesn't leak between cases.

    TestClient's client.host is 'testclient', which fails the
    dashboard's defense-in-depth loopback check. We spoof the ASGI
    scope's client tuple to ('127.0.0.1', 0) via a tiny middleware
    wrapper so the auth path matches production behaviour.
    """
    from fastapi.testclient import TestClient
    from starlette.types import ASGIApp, Receive, Scope, Send

    from secondbrain.config import Config
    from secondbrain.dashboard import (
        create_app,
        get_or_create_extension_token,
    )

    cfg = Config(data_dir=tmp_path / "data")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("secondbrain.dashboard.load_config", lambda: cfg)
    monkeypatch.setattr(
        "secondbrain.dashboard.make_embedder", lambda c: fake_embedder,
    )
    monkeypatch.setattr(
        "secondbrain.dashboard.make_reranker", lambda c: None,
    )
    token = get_or_create_extension_token(cfg)
    app = create_app()

    class _LoopbackSpoof:
        def __init__(self, app: ASGIApp):
            self.app = app
        async def __call__(
            self, scope: Scope, receive: Receive, send: Send,
        ) -> None:
            if scope.get("type") in ("http", "websocket"):
                scope = dict(scope)
                scope["client"] = ("127.0.0.1", 0)
            await self.app(scope, receive, send)

    client = TestClient(_LoopbackSpoof(app))
    return client, token


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_api_capture_rejects_unauthorized(app_client):
    client, _ = app_client
    r = client.post("/api/capture", json={"content": "hi"})
    assert r.status_code == 401


def test_api_capture_indexes_text(app_client):
    client, token = app_client
    r = client.post(
        "/api/capture",
        headers=_auth(token),
        json={
            "title": "Quick thought",
            "content": "I should write about voyage embeddings.",
            "source": "ios",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["virtual_path"].startswith("capture://ios/")
    assert body["chunks"] >= 1


def test_api_capture_rejects_empty_payload(app_client):
    """No content + no URL = nothing to do."""
    client, token = app_client
    r = client.post(
        "/api/capture", headers=_auth(token), json={"title": "x"},
    )
    assert r.status_code == 400


def test_api_capture_renders_url_into_body(app_client):
    """When both content AND url are provided, URL becomes a Source:
    line in the doc body. Verified by searching for the indexed content."""
    client, token = app_client
    r = client.post(
        "/api/capture",
        headers=_auth(token),
        json={
            "title": "Highlight",
            "content": "key claim from the article",
            "url": "https://example.com/article",
            "source": "ios",
        },
    )
    assert r.status_code == 200
    # Confirm the doc made it into the index by searching.
    r2 = client.get(
        "/api/extension/search",
        headers=_auth(token),
        params={"q": "key claim"},
    )
    assert r2.status_code == 200
    results = r2.json().get("results", [])
    # The 'Source: <url>' line + content should both be present in
    # at least one returned snippet.
    snippets = " ".join(r["snippet"] for r in results)
    assert "key claim" in snippets
    assert "example.com" in snippets


def test_api_capture_default_source(app_client):
    client, token = app_client
    r = client.post(
        "/api/capture", headers=_auth(token),
        json={"content": "no source given"},
    )
    assert r.status_code == 200
    assert r.json()["virtual_path"].startswith("capture://manual/")


def test_api_capture_truncates_long_source(app_client):
    """A weird client that sends a 2000-char source name shouldn't
    blow up the path."""
    client, token = app_client
    r = client.post(
        "/api/capture", headers=_auth(token),
        json={"content": "x", "source": "x" * 2000},
    )
    assert r.status_code == 200
    # source is truncated to 40 chars; path should be reasonable.
    assert len(r.json()["virtual_path"]) < 200


# ============================ IMAP capture detect =====================

def test_is_capture_email_matches_subject_prefix():
    """[capture] / [brain] prefixes mark a message as a forward-to-self."""
    from secondbrain.connectors.imap_email import _is_capture_email

    @dataclass
    class _Cfg:
        capture_imap_folders: tuple = ()

    cfg = _Cfg()
    assert _is_capture_email(cfg, "Inbox", "[capture] save this article")
    assert _is_capture_email(cfg, "Inbox", "[brain] note from the train")
    # Case-insensitive.
    assert _is_capture_email(cfg, "Inbox", "[CAPTURE] x")
    # Plain emails don't match.
    assert not _is_capture_email(cfg, "Inbox", "your weekly newsletter")


def test_is_capture_email_matches_folder():
    from secondbrain.connectors.imap_email import _is_capture_email

    @dataclass
    class _Cfg:
        capture_imap_folders: tuple = ("ToBrain", "Captures")

    cfg = _Cfg()
    assert _is_capture_email(cfg, "ToBrain", "anything")
    assert _is_capture_email(cfg, "captures", "anything")  # case-insensitive
    assert not _is_capture_email(cfg, "Inbox", "anything")


def test_is_capture_email_handles_missing_config():
    """A cfg without the capture_imap_folders attr at all (older
    config object) falls back to no folder match."""
    from secondbrain.connectors.imap_email import _is_capture_email

    class _Cfg:
        pass

    cfg = _Cfg()
    # No folder match → only subject can mark as capture.
    assert not _is_capture_email(cfg, "Inbox", "regular subject")
    assert _is_capture_email(cfg, "Inbox", "[capture] x")


# ============================ Phase 71 photo folder ===================

def test_resolve_folders_includes_photo_capture(tmp_path, monkeypatch):
    """The daemon's _resolve_folders should fold photo_capture_folder
    into the watch list alongside watched_folders."""
    from secondbrain.config import Config
    from secondbrain.daemon import _resolve_folders

    photos = tmp_path / "photos"
    photos.mkdir()
    notes = tmp_path / "notes"
    notes.mkdir()
    cfg = Config()
    cfg.watched_folders = [notes]
    cfg.photo_capture_folder = str(photos)
    out = _resolve_folders(cfg)
    paths = {p.name for p in out}
    assert "photos" in paths
    assert "notes" in paths


def test_resolve_folders_dedupes_overlap(tmp_path):
    """If photo_capture_folder == one of watched_folders (user
    misconfig), don't double-watch."""
    from secondbrain.config import Config
    from secondbrain.daemon import _resolve_folders

    shared = tmp_path / "shared"
    shared.mkdir()
    cfg = Config()
    cfg.watched_folders = [shared]
    cfg.photo_capture_folder = str(shared)
    out = _resolve_folders(cfg)
    assert len(out) == 1


def test_resolve_folders_ignores_missing_paths(tmp_path):
    """Non-existent paths shouldn't crash the daemon — they get
    dropped silently."""
    from secondbrain.config import Config
    from secondbrain.daemon import _resolve_folders

    cfg = Config()
    cfg.watched_folders = [tmp_path / "does-not-exist"]
    cfg.photo_capture_folder = "/also/not/here"
    out = _resolve_folders(cfg)
    assert out == []


def test_resolve_folders_empty_photo_config(tmp_path):
    """Empty photo_capture_folder string is the default — should
    leave watched_folders alone."""
    from secondbrain.config import Config
    from secondbrain.daemon import _resolve_folders

    notes = tmp_path / "notes"
    notes.mkdir()
    cfg = Config()
    cfg.watched_folders = [notes]
    cfg.photo_capture_folder = ""
    out = _resolve_folders(cfg)
    assert len(out) == 1
    assert out[0].name == "notes"
