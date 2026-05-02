"""Indexer chunking + dedup + URL fetch guards."""

from __future__ import annotations

from secondbrain.indexer import (
    _URL_FETCH_MAX_BYTES,
    _virtual_path_scheme,
    chunk_text,
    dedupe_existing,
    index_text,
)


def test_virtual_path_scheme():
    assert _virtual_path_scheme("reddit://post/abc") == "reddit"
    assert _virtual_path_scheme("github://owner/repo/issues/42") == "github"
    assert _virtual_path_scheme("plain string") == ""
    assert _virtual_path_scheme("") == ""


def test_chunk_text_paragraph_packing():
    """Three short paragraphs pack into one chunk under target_size."""
    text = "alpha\n\nbeta\n\ngamma"
    chunks = chunk_text(text, target_size=200, overlap=20)
    assert len(chunks) == 1
    body, offset = chunks[0]
    assert "alpha" in body and "gamma" in body
    assert offset == 0


def test_chunk_text_long_paragraph_split():
    """A paragraph larger than target_size splits with overlap."""
    long = "x" * 1000
    chunks = chunk_text(long, target_size=300, overlap=50)
    # step = 250, so we get ceil(1000/250) = 4 chunks
    assert len(chunks) >= 3
    # offsets should be monotonically non-decreasing
    offsets = [off for _, off in chunks]
    assert offsets == sorted(offsets)


def test_chunk_text_empty_returns_empty():
    assert chunk_text("") == []
    assert chunk_text("   \n\n   ") == []


def test_url_fetch_cap_is_sane():
    """Sanity floor: hostile servers shouldn't be able to fill the disk."""
    assert 50 * 1024 * 1024 <= _URL_FETCH_MAX_BYTES <= 1024 * 1024 * 1024


class _FakeEmbedderForIndex:
    """Embedder stub; index_text doesn't call back into Voyage when given this."""

    name = "test"
    dim = 8

    def embed_documents(self, texts):
        return [[0.1] * self.dim for _ in texts]

    def embed_query(self, q):
        return [0.1] * self.dim


def test_index_text_dedup_by_hash(fresh_db, tmp_cfg):
    """Two connector docs with identical text on the same source should
    collapse: the second becomes an alias of the first."""
    embedder = _FakeEmbedderForIndex()
    # init_schema in fresh_db used dim=16 but index_text only writes the
    # vec_chunks rows, which sqlite-vec validates per-row. Build a fresh
    # db with the matching dim instead.
    from secondbrain.db import connect, init_schema

    db = connect(tmp_cfg.data_dir / "fresh.db")
    init_schema(db, embedder.dim, embedder.name)

    body = "the quick brown fox jumps"
    r1 = index_text(
        db, embedder, tmp_cfg,
        virtual_path="reddit://post/aaa",
        title="first",
        content=body,
        mtime=0.0,
        source="reddit",
    )
    assert r1.status == "indexed"

    r2 = index_text(
        db, embedder, tmp_cfg,
        virtual_path="reddit://post/bbb",
        title="second",
        content=body,
        mtime=0.0,
        source="reddit",
    )
    assert r2.status == "alias", f"second should alias to first, got {r2.status}"
    assert "duplicate of reddit://post/aaa" in (r2.reason or "")
    db.close()


def test_index_text_cross_source_does_not_alias(fresh_db, tmp_cfg):
    """Same body from different sources stays distinct (M3)."""
    embedder = _FakeEmbedderForIndex()
    from secondbrain.db import connect, init_schema

    db = connect(tmp_cfg.data_dir / "cross.db")
    init_schema(db, embedder.dim, embedder.name)
    body = "identical body across two sources"
    r1 = index_text(
        db, embedder, tmp_cfg,
        virtual_path="reddit://post/x",
        title="reddit one", content=body, mtime=0.0, source="reddit",
    )
    r2 = index_text(
        db, embedder, tmp_cfg,
        virtual_path="hn://item/9", title="hn one",
        content=body, mtime=0.0, source="hacker_news",
    )
    assert r1.status == "indexed"
    assert r2.status == "indexed", "different sources should not alias even with identical text"
    db.close()


def test_dedupe_existing_dry_run_changes_nothing(fresh_db):
    """The dry-run flag must not mutate the database."""
    from secondbrain.db import upsert_file

    upsert_file(fresh_db, path="/a.md", mtime=0.0, size=10, kind="document", content_hash="dup")
    upsert_file(fresh_db, path="/b.md", mtime=0.0, size=10, kind="document", content_hash="dup")
    upsert_file(fresh_db, path="/c.md", mtime=0.0, size=10, kind="document", content_hash="other")
    fresh_db.commit()

    res = dedupe_existing(fresh_db, dry_run=True)
    assert res["groups_with_duplicates"] == 1
    assert res["duplicate_files_converted"] == 1
    assert isinstance(res["aliased"], list) and len(res["aliased"]) == 1

    # Real rows untouched
    rows = fresh_db.execute("SELECT path FROM files ORDER BY path").fetchall()
    assert {r["path"] for r in rows} == {"/a.md", "/b.md", "/c.md"}


def test_dedupe_existing_picks_oldest_indexed_at(fresh_db):
    """Canonical should be the oldest indexed row, not the lowest id (C11)."""
    # Insert b first (older indexed_at), then a (higher id but newer).
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at, content_hash) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("/b.md", 0.0, 10, "document", 100.0, "dup"),
    )
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at, content_hash) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("/a.md", 0.0, 10, "document", 200.0, "dup"),
    )
    fresh_db.commit()
    res = dedupe_existing(fresh_db)
    canonical_path, _ = res["aliased"][0]
    assert canonical_path == "/b.md", "canonical should be the oldest-indexed row"
