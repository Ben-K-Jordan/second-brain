"""Shared pytest fixtures for the second-brain test suite.

Two flavors of fixture:

- ``tmp_cfg`` — a Config pointing at a fresh temp data_dir, with a tiny
  config.toml so loaders work. Use for unit tests that don't need a real DB.
- ``fresh_db`` — a fully-initialised SQLite connection on a temp path,
  with the schema migrations applied. Suitable for indexer/search/chat
  tests that need to round-trip rows without touching the user's real index.

Network-bound and slow tests are gated by the ``slow`` marker (see
pyproject.toml). Run only fast tests with::

    pytest -m "not slow"
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from secondbrain.config import Config


@pytest.fixture
def tmp_cfg(tmp_path: Path) -> Config:
    """Return a Config rooted at an isolated temp dir."""
    cfg = Config(data_dir=tmp_path / "data")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    return cfg


class _FakeEmbedder:
    """Deterministic fake embedder for tests.

    Hashes the input string into a stable [0, 1) float vector so the tests
    don't need a network call or a Voyage key. Same input → same vector.
    """

    name = "test-fake-embedder"
    dim = 16

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        import hashlib

        out: list[float] = []
        seed = text.encode("utf-8", errors="replace")
        for i in range(self.dim):
            h = hashlib.md5(seed + i.to_bytes(2, "little"), usedforsecurity=False)
            out.append(int.from_bytes(h.digest()[:4], "little") / 0xFFFFFFFF)
        # L2-normalise so vec0's distance metric behaves reasonably.
        norm = sum(x * x for x in out) ** 0.5 or 1.0
        return [x / norm for x in out]


@pytest.fixture
def fake_embedder() -> _FakeEmbedder:
    return _FakeEmbedder()


@pytest.fixture
def fresh_db(tmp_path: Path, fake_embedder: _FakeEmbedder) -> Iterator:
    """A clean SQLite connection with the schema migrated."""
    from secondbrain.db import connect, init_schema

    db_path = tmp_path / "test.db"
    conn = connect(db_path)
    init_schema(conn, fake_embedder.dim, fake_embedder.name)
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def _no_network_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip network-credential env vars from every test so fast tests can't
    accidentally call a paid API. ``slow`` tests can re-set these via their
    own monkeypatch if they really need them."""
    for var in (
        "VOYAGE_API_KEY", "ANTHROPIC_API_KEY",
        "GITHUB_TOKEN", "NOTION_TOKEN",
        "LINEAR_API_KEY", "SLACK_USER_TOKEN",
        "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET",
        "REDDIT_USERNAME", "REDDIT_PASSWORD",
        "POCKET_CONSUMER_KEY", "POCKET_ACCESS_TOKEN",
        "HN_USERNAME", "BLUESKY_HANDLE",
        "MASTODON_INSTANCE", "MASTODON_ACCESS_TOKEN",
        "X_ARCHIVE_PATH", "OBSIDIAN_VAULTS",
        "SUBSTACK_FEEDS", "CALENDAR_ICS_URL",
    ):
        monkeypatch.delenv(var, raising=False)
