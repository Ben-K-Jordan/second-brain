"""Phase 52: auto-backlinks tests.

Two layers:

  - **compute_neighbors**: against a real fresh_db with seeded
    embeddings, verifies the source-file is excluded, distance ranking
    works, and the threshold is honoured.
  - **link_doc / get_backlinks**: end-to-end. Verifies bidirectional
    storage and that ``rebuild_all`` is idempotent + clean.
"""

from __future__ import annotations

import time

from secondbrain import backlinks

# Helpers ---------------------------------------------------------------

def _seed_file_with_embeddings(
    conn, *, path: str, chunk_texts: list[str],
    chunk_embeddings: list[list[float]],
) -> int:
    """Insert one file with N chunks + matching vec rows. Returns file_id.

    Uses ``replace_chunks`` so the SAVEPOINT path is exercised the same
    way the indexer does it.
    """
    from secondbrain.db import replace_chunks, upsert_file

    fid = upsert_file(
        conn, path=path, mtime=time.time(), size=len(path),
        kind="document", content_hash=None,
    )
    replace_chunks(
        conn, fid,
        list(zip(chunk_texts, chunk_embeddings, strict=True)),
    )
    conn.commit()
    return fid


def _vec(values: list[float], dim: int = 16) -> list[float]:
    """Pad/truncate a vector to ``dim`` and L2-normalise so distances
    stay in the unit sphere — matches conftest.py's _FakeEmbedder.dim."""
    v = list(values) + [0.0] * dim
    v = v[:dim]
    norm = sum(x * x for x in v) ** 0.5 or 1.0
    return [x / norm for x in v]


# ===================== compute_neighbors ==============================

def test_compute_neighbors_excludes_source_file(fresh_db):
    """The source file's own chunks should never appear in its
    neighbour list — distance 0 against themselves would dominate."""
    fid_a = _seed_file_with_embeddings(
        fresh_db, path="A",
        chunk_texts=["alpha"],
        chunk_embeddings=[_vec([1.0, 0.0])],
    )
    fid_b = _seed_file_with_embeddings(
        fresh_db, path="B",
        chunk_texts=["alpha-ish"],
        chunk_embeddings=[_vec([0.99, 0.01])],
    )
    out = backlinks.compute_neighbors(fresh_db, fid_a, k=5, max_distance=2.0)
    file_ids = [fid for fid, _ in out]
    assert fid_a not in file_ids
    assert fid_b in file_ids


def test_compute_neighbors_orders_by_distance(fresh_db):
    """Closer vectors should appear before farther ones."""
    fid_src = _seed_file_with_embeddings(
        fresh_db, path="src",
        chunk_texts=["query"],
        chunk_embeddings=[_vec([1.0, 0.0])],
    )
    fid_close = _seed_file_with_embeddings(
        fresh_db, path="close",
        chunk_texts=["very similar"],
        chunk_embeddings=[_vec([0.99, 0.01])],
    )
    fid_far = _seed_file_with_embeddings(
        fresh_db, path="far",
        chunk_texts=["different"],
        chunk_embeddings=[_vec([-1.0, 0.0])],
    )
    out = backlinks.compute_neighbors(
        fresh_db, fid_src, k=5, max_distance=10.0,
    )
    file_ids = [fid for fid, _ in out]
    assert file_ids.index(fid_close) < file_ids.index(fid_far)


def test_compute_neighbors_respects_max_distance(fresh_db):
    """Pairs above the threshold are dropped."""
    fid_src = _seed_file_with_embeddings(
        fresh_db, path="src",
        chunk_texts=["q"],
        chunk_embeddings=[_vec([1.0, 0.0])],
    )
    _seed_file_with_embeddings(
        fresh_db, path="far",
        chunk_texts=["unrelated"],
        chunk_embeddings=[_vec([-1.0, 0.0])],
    )
    out = backlinks.compute_neighbors(
        fresh_db, fid_src, k=5, max_distance=0.1,
    )
    # Anti-correlated pair is L2 ≈ 2.0, far above 0.1.
    assert out == []


def test_compute_neighbors_returns_top_k(fresh_db):
    """At most K neighbours, even when more pass the threshold."""
    fid_src = _seed_file_with_embeddings(
        fresh_db, path="src",
        chunk_texts=["q"],
        chunk_embeddings=[_vec([1.0, 0.0])],
    )
    # Six neighbours, all close.
    for i in range(6):
        _seed_file_with_embeddings(
            fresh_db, path=f"near-{i}",
            chunk_texts=[f"near {i}"],
            chunk_embeddings=[_vec([1.0, 0.001 * i])],
        )
    out = backlinks.compute_neighbors(
        fresh_db, fid_src, k=3, max_distance=10.0,
    )
    assert len(out) == 3


def test_compute_neighbors_keeps_best_distance_per_file(fresh_db):
    """When the source has many chunks and the candidate also has
    multiple matches across them, only the best (lowest) distance
    survives — not the average, not an extra row."""
    fid_src = _seed_file_with_embeddings(
        fresh_db, path="src",
        chunk_texts=["q1", "q2"],
        chunk_embeddings=[_vec([1.0, 0.0]), _vec([0.0, 1.0])],
    )
    fid_dst = _seed_file_with_embeddings(
        fresh_db, path="dst",
        chunk_texts=["close to first", "far from second"],
        chunk_embeddings=[_vec([0.99, 0.01]), _vec([-1.0, 0.0])],
    )
    out = backlinks.compute_neighbors(
        fresh_db, fid_src, k=5, max_distance=10.0,
    )
    matches = [d for fid, d in out if fid == fid_dst]
    assert len(matches) == 1
    # Must be the close pair's distance, not the mean.
    assert matches[0] < 0.2


def test_compute_neighbors_empty_for_chunkless_file(fresh_db):
    """A file with no chunks shouldn't crash; just return empty."""
    from secondbrain.db import upsert_file
    fid = upsert_file(
        fresh_db, path="empty", mtime=0.0, size=0,
        kind="document", content_hash=None,
    )
    fresh_db.commit()
    assert backlinks.compute_neighbors(fresh_db, fid) == []


# ===================== link_doc / persistence =========================

def test_link_doc_stores_both_directions(fresh_db):
    """Storage is bidirectional so both A → B and B → A queries work."""
    fid_a = _seed_file_with_embeddings(
        fresh_db, path="A",
        chunk_texts=["x"], chunk_embeddings=[_vec([1.0, 0.0])],
    )
    fid_b = _seed_file_with_embeddings(
        fresh_db, path="B",
        chunk_texts=["y"], chunk_embeddings=[_vec([0.99, 0.01])],
    )
    written = backlinks.link_doc(fresh_db, fid_a, max_distance=10.0, min_chunks=1)
    assert written == 2  # one neighbour × bidirectional rows
    # Both directions queryable.
    assert backlinks.get_backlinks(fresh_db, fid_a)[0].file_id == fid_b
    assert backlinks.get_backlinks(fresh_db, fid_b)[0].file_id == fid_a


def test_link_doc_idempotent_score_refreshes(fresh_db):
    """Re-running link_doc on the same source must not duplicate rows;
    it should refresh ``score`` (e.g. after re-embedding)."""
    fid_a = _seed_file_with_embeddings(
        fresh_db, path="A",
        chunk_texts=["x"], chunk_embeddings=[_vec([1.0, 0.0])],
    )
    _seed_file_with_embeddings(
        fresh_db, path="B",
        chunk_texts=["y"], chunk_embeddings=[_vec([0.99, 0.01])],
    )
    backlinks.link_doc(fresh_db, fid_a, max_distance=10.0, min_chunks=1)
    n_after_first = fresh_db.execute(
        "SELECT COUNT(*) AS n FROM backlinks",
    ).fetchone()["n"]
    backlinks.link_doc(fresh_db, fid_a, max_distance=10.0, min_chunks=1)
    n_after_second = fresh_db.execute(
        "SELECT COUNT(*) AS n FROM backlinks",
    ).fetchone()["n"]
    assert n_after_first == n_after_second == 2


def test_get_backlinks_orders_by_score_ascending(fresh_db):
    fid_src = _seed_file_with_embeddings(
        fresh_db, path="src",
        chunk_texts=["q"], chunk_embeddings=[_vec([1.0, 0.0])],
    )
    fid_close = _seed_file_with_embeddings(
        fresh_db, path="close",
        chunk_texts=["x"], chunk_embeddings=[_vec([0.99, 0.01])],
    )
    fid_mid = _seed_file_with_embeddings(
        fresh_db, path="mid",
        chunk_texts=["y"], chunk_embeddings=[_vec([0.5, 0.5])],
    )
    backlinks.link_doc(fresh_db, fid_src, max_distance=10.0, min_chunks=1)
    rows = backlinks.get_backlinks(fresh_db, fid_src)
    file_ids = [r.file_id for r in rows]
    assert file_ids.index(fid_close) < file_ids.index(fid_mid)


def test_backlinks_view_includes_path_and_title(fresh_db):
    fid_src = _seed_file_with_embeddings(
        fresh_db, path="src",
        chunk_texts=["q"], chunk_embeddings=[_vec([1.0, 0.0])],
    )
    fid_dst = _seed_file_with_embeddings(
        fresh_db, path="transcript://granola/abc",
        chunk_texts=["# Sprint planning\n\nNotes here."],
        chunk_embeddings=[_vec([0.99, 0.01])],
    )
    backlinks.link_doc(fresh_db, fid_src, max_distance=10.0, min_chunks=1)
    [view] = backlinks.get_backlinks(fresh_db, fid_src)
    assert view.file_id == fid_dst
    assert view.path == "transcript://granola/abc"
    assert view.title == "Sprint planning"
    assert 0.0 < view.similarity <= 1.0
    assert 0 <= view.percent <= 100


def test_get_backlinks_for_path(fresh_db):
    fid_src = _seed_file_with_embeddings(
        fresh_db, path="src",
        chunk_texts=["q"], chunk_embeddings=[_vec([1.0, 0.0])],
    )
    _seed_file_with_embeddings(
        fresh_db, path="dst",
        chunk_texts=["x"], chunk_embeddings=[_vec([0.99, 0.01])],
    )
    backlinks.link_doc(fresh_db, fid_src, max_distance=10.0, min_chunks=1)
    by_path = backlinks.get_backlinks_for_path(fresh_db, "src")
    by_id = backlinks.get_backlinks(fresh_db, fid_src)
    assert [v.file_id for v in by_path] == [v.file_id for v in by_id]


def test_get_backlinks_for_unknown_path_returns_empty(fresh_db):
    assert backlinks.get_backlinks_for_path(fresh_db, "no-such-path") == []


def test_clear_backlinks_for_removes_both_directions(fresh_db):
    fid_a = _seed_file_with_embeddings(
        fresh_db, path="A",
        chunk_texts=["x"], chunk_embeddings=[_vec([1.0, 0.0])],
    )
    fid_b = _seed_file_with_embeddings(
        fresh_db, path="B",
        chunk_texts=["y"], chunk_embeddings=[_vec([0.99, 0.01])],
    )
    backlinks.link_doc(fresh_db, fid_a, max_distance=10.0, min_chunks=1)
    n = backlinks.clear_backlinks_for(fresh_db, fid_a)
    assert n == 2
    assert backlinks.get_backlinks(fresh_db, fid_a) == []
    assert backlinks.get_backlinks(fresh_db, fid_b) == []


# ===================== rebuild_all ====================================

def test_rebuild_all_drops_then_recomputes(fresh_db):
    """rebuild_all should produce the same final state regardless of
    whether backlinks already existed. Plant a stale (high-distance)
    score on a real pair, then verify rebuild rewrites it with the
    correct value."""
    fid_a = _seed_file_with_embeddings(
        fresh_db, path="A",
        chunk_texts=["x"], chunk_embeddings=[_vec([1.0, 0.0])],
    )
    fid_b = _seed_file_with_embeddings(
        fresh_db, path="B",
        chunk_texts=["y"], chunk_embeddings=[_vec([0.99, 0.01])],
    )
    # Pre-seed with a wrong score (bigger than reality). Note: FK
    # constraint forces both ids to exist, so we're stress-testing the
    # "old data was wrong, redo it" path, not the "old data referenced
    # missing files" path (which CASCADE handles).
    fresh_db.execute(
        "INSERT INTO backlinks(src_file_id, dst_file_id, score, created_at) "
        "VALUES (?, ?, ?, ?)",
        (fid_a, fid_b, 99.0, time.time()),
    )
    fresh_db.commit()

    written = backlinks.rebuild_all(fresh_db, max_distance=10.0, min_chunks=1)
    assert written > 0
    # The (a, b) pair should now have the real (small) score.
    score = fresh_db.execute(
        "SELECT score FROM backlinks WHERE src_file_id = ? AND dst_file_id = ?",
        (fid_a, fid_b),
    ).fetchone()["score"]
    assert score < 1.0  # was 99.0 before rebuild


def test_rebuild_all_progress_callback_fires(fresh_db):
    _seed_file_with_embeddings(
        fresh_db, path="A",
        chunk_texts=["x"], chunk_embeddings=[_vec([1.0, 0.0])],
    )
    _seed_file_with_embeddings(
        fresh_db, path="B",
        chunk_texts=["y"], chunk_embeddings=[_vec([0.99, 0.01])],
    )
    seen: list[tuple[int, int]] = []
    backlinks.rebuild_all(
        fresh_db, max_distance=10.0, min_chunks=1,
        on_progress=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(1, 2), (2, 2)]


def test_rebuild_all_progress_callback_failure_doesnt_break_rebuild(fresh_db):
    """A misbehaving callback must not derail the rebuild — we want
    progress reporting to be safe."""
    _seed_file_with_embeddings(
        fresh_db, path="A",
        chunk_texts=["x"], chunk_embeddings=[_vec([1.0, 0.0])],
    )
    _seed_file_with_embeddings(
        fresh_db, path="B",
        chunk_texts=["y"], chunk_embeddings=[_vec([0.99, 0.01])],
    )
    def boom(done, total):
        raise RuntimeError("ui crashed")
    n = backlinks.rebuild_all(
        fresh_db, max_distance=10.0, min_chunks=1, on_progress=boom,
    )
    assert n > 0


# ===================== indexer integration ============================

def test_indexer_hook_links_after_replace_chunks(fresh_db):
    """When the indexer's _link_after_index helper runs against a
    populated db, downstream get_backlinks reflects the link.

    Note: we seed 2-chunk docs because the production hook respects
    the ``_MIN_CHUNKS_FOR_LINKING = 2`` threshold to avoid spammy
    matches off tiny stub docs. Tests that want to exercise the
    threshold separately use ``min_chunks=1``."""
    from secondbrain.indexer import _link_after_index

    fid_a = _seed_file_with_embeddings(
        fresh_db, path="A",
        chunk_texts=["alpha first", "alpha second"],
        chunk_embeddings=[_vec([1.0, 0.0]), _vec([0.99, 0.05])],
    )
    _seed_file_with_embeddings(
        fresh_db, path="B",
        chunk_texts=["beta first", "beta second"],
        chunk_embeddings=[_vec([0.99, 0.01]), _vec([0.98, 0.05])],
    )
    _link_after_index(fresh_db, fid_a)
    assert backlinks.get_backlinks(fresh_db, fid_a)


def test_indexer_hook_swallows_failures(fresh_db, monkeypatch):
    """If link_doc explodes, the indexer hook must not propagate."""
    import secondbrain.backlinks as bl_mod
    import secondbrain.indexer as idx_mod

    def boom(*a, **kw):
        raise RuntimeError("vec query failed")

    monkeypatch.setattr(bl_mod, "link_doc", boom)
    # Should not raise.
    idx_mod._link_after_index(fresh_db, 12345)


# ===================== min_chunks threshold ===========================

def test_link_doc_skips_single_chunk_docs_by_default(fresh_db):
    """Production behaviour: tiny stub docs (1 chunk) shouldn't enter
    the backlink graph — they make noisy matches based on stopword
    overlap rather than real semantic kinship."""
    fid_stub = _seed_file_with_embeddings(
        fresh_db, path="stub",
        chunk_texts=["one liner"],
        chunk_embeddings=[_vec([1.0, 0.0])],
    )
    _seed_file_with_embeddings(
        fresh_db, path="real",
        chunk_texts=["alpha", "beta"],
        chunk_embeddings=[_vec([0.99, 0.01]), _vec([0.95, 0.05])],
    )
    written = backlinks.link_doc(fresh_db, fid_stub, max_distance=10.0)
    assert written == 0  # default min_chunks=2 skipped it
    assert backlinks.get_backlinks(fresh_db, fid_stub) == []


def test_link_doc_min_chunks_override_lets_tests_exercise_logic(fresh_db):
    """Passing min_chunks=1 brings 1-chunk docs back into the graph
    — needed for the existing single-chunk fixture tests."""
    fid_a = _seed_file_with_embeddings(
        fresh_db, path="A",
        chunk_texts=["x"], chunk_embeddings=[_vec([1.0, 0.0])],
    )
    _seed_file_with_embeddings(
        fresh_db, path="B",
        chunk_texts=["y"], chunk_embeddings=[_vec([0.99, 0.01])],
    )
    written = backlinks.link_doc(
        fresh_db, fid_a, max_distance=10.0, min_chunks=1,
    )
    assert written > 0


def test_link_doc_skips_chunkless_files(fresh_db):
    """A file row with zero chunks (e.g. an image without OCR) must
    not crash the linker."""
    from secondbrain.db import upsert_file
    fid = upsert_file(
        fresh_db, path="empty.png", mtime=0.0, size=0,
        kind="image", content_hash=None,
    )
    fresh_db.commit()
    assert backlinks.link_doc(fresh_db, fid, max_distance=10.0) == 0
