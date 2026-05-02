"""Phase 84 + 85: PDF annotations + citation graph tests.

Coverage:
  - store/get/replace annotations idempotence
  - citation regex against author-year + numbered + reflist
  - resolve_citation to existing brain docs
  - process_after_index integration
"""

from __future__ import annotations

import time

from secondbrain import pdf_annotations as pa

# ============================ helpers =================================

def _seed_doc(
    conn, *, path, body, kind="document", indexed_at=None,
):
    n = indexed_at or time.time()
    cur = conn.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at, content_hash) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (path, n, len(body), kind, n, None),
    )
    fid = cur.lastrowid
    conn.execute(
        "INSERT INTO chunks(file_id, chunk_index, text) VALUES (?, 0, ?)",
        (fid, body),
    )
    conn.commit()
    return fid


# ============================ annotations =============================

def test_store_annotations_persists(fresh_db):
    fid = _seed_doc(fresh_db, path="paper.pdf", body="x")
    n = pa.store_annotations(fresh_db, fid, [
        {"page": 1, "kind": "highlight", "anchor": "key claim",
         "color": "#ffff00"},
        {"page": 2, "kind": "note", "anchor": "another bit",
         "note": "interesting"},
    ])
    assert n == 2
    out = pa.get_annotations(fresh_db, fid)
    assert len(out) == 2
    assert out[0].anchor == "key claim"


def test_store_annotations_replaces_existing(fresh_db):
    """Re-extracts should reflect the latest annotation state."""
    fid = _seed_doc(fresh_db, path="paper.pdf", body="x")
    pa.store_annotations(fresh_db, fid, [
        {"page": 1, "kind": "highlight", "anchor": "old"},
    ])
    pa.store_annotations(fresh_db, fid, [
        {"page": 1, "kind": "highlight", "anchor": "new"},
    ])
    out = pa.get_annotations(fresh_db, fid)
    assert len(out) == 1
    assert out[0].anchor == "new"


def test_store_annotations_empty_input_clears(fresh_db):
    """Passing [] removes existing annotations (re-extract found none)."""
    fid = _seed_doc(fresh_db, path="paper.pdf", body="x")
    pa.store_annotations(fresh_db, fid, [
        {"page": 1, "kind": "highlight", "anchor": "x"},
    ])
    pa.store_annotations(fresh_db, fid, [])
    assert pa.get_annotations(fresh_db, fid) == []


def test_store_annotations_truncates_long_anchor(fresh_db):
    """Anchor longer than the cap should be truncated, not rejected."""
    fid = _seed_doc(fresh_db, path="paper.pdf", body="x")
    big = "x" * 5000
    pa.store_annotations(fresh_db, fid, [
        {"page": 1, "kind": "highlight", "anchor": big},
    ])
    out = pa.get_annotations(fresh_db, fid)
    assert len(out) == 1
    assert len(out[0].anchor) <= 1001


def test_annotations_count(fresh_db):
    fid = _seed_doc(fresh_db, path="paper.pdf", body="x")
    pa.store_annotations(fresh_db, fid, [
        {"page": 1, "kind": "highlight", "anchor": "a"},
        {"page": 1, "kind": "underline", "anchor": "b"},
    ])
    assert pa.annotations_count(fresh_db, fid) == 2


def test_extract_annotations_handles_missing_pymupdf(monkeypatch):
    """When PyMuPDF isn't installed, return empty without crashing."""
    import sys
    # Block the import.
    monkeypatch.setitem(sys.modules, "fitz", None)
    out = pa.extract_annotations_from_pdf("/nonexistent.pdf")
    assert out == []


def test_normalize_annotation_kind_mapping():
    assert pa._normalize_annotation_kind("Highlight") == "highlight"
    assert pa._normalize_annotation_kind("Underline") == "underline"
    assert pa._normalize_annotation_kind("StrikeOut") == "strike"
    assert pa._normalize_annotation_kind("FreeText") == "note"
    assert pa._normalize_annotation_kind("Random") == ""


# ============================ citations ===============================

def test_extract_citations_inline_author_year():
    text = (
        "As shown by Smith et al., 2024, the model converges. "
        "Earlier work (Jones, 2021) corroborates."
    )
    out = pa.extract_citations_from_text(text)
    cited = [c["cited_text"] for c in out]
    assert any("Smith" in c and "2024" in c for c in cited)
    assert any("Jones" in c and "2021" in c for c in cited)


def test_extract_citations_handles_reflist():
    text = (
        "[12] Smith, J. (2024). The Title. Journal, 5, 123.\n"
        "[13] Jones, A. (2021). Other work. Other Journal."
    )
    out = pa.extract_citations_from_text(text)
    assert len(out) >= 2
    years = [c.get("year") for c in out]
    assert 2024 in years
    assert 2021 in years


def test_extract_citations_dedupes():
    """Same author-year mentioned twice → one row."""
    text = (
        "Smith, 2024 reports X. Later, Smith, 2024 also showed Y. "
        "Smith, 2024 is foundational."
    )
    out = pa.extract_citations_from_text(text)
    smith_count = sum(
        1 for c in out if "Smith" in c["cited_text"] and c["year"] == 2024
    )
    assert smith_count == 1


def test_extract_citations_filters_invalid_years():
    """Pattern matches '12345' as year; should reject."""
    text = "see Smith, 12345 for details"
    out = pa.extract_citations_from_text(text)
    assert all(
        c.get("year") is None or 1900 <= c["year"] <= 2100
        for c in out
    )


def test_extract_citations_caps_count():
    """Pathological input shouldn't bloat the citations table."""
    text = " ".join(
        f"Smith{i}, 202{i % 10}" for i in range(500)
    )
    out = pa.extract_citations_from_text(text)
    assert len(out) <= 200


def test_extract_citations_returns_empty_for_blank():
    assert pa.extract_citations_from_text("") == []
    assert pa.extract_citations_from_text(None) == []


def test_store_citations_persists(fresh_db):
    fid = _seed_doc(fresh_db, path="paper.pdf", body="x")
    n = pa.store_citations(fresh_db, fid, [
        {"cited_text": "Smith, 2024", "year": 2024},
        {"cited_text": "Jones, 2021", "year": 2021},
    ])
    assert n == 2
    cites = pa.get_citations_from(fresh_db, fid)
    assert len(cites) == 2


def test_store_citations_replaces_existing(fresh_db):
    fid = _seed_doc(fresh_db, path="paper.pdf", body="x")
    pa.store_citations(fresh_db, fid, [
        {"cited_text": "Smith, 2024", "year": 2024},
    ])
    pa.store_citations(fresh_db, fid, [
        {"cited_text": "Jones, 2021", "year": 2021},
    ])
    cites = pa.get_citations_from(fresh_db, fid)
    assert len(cites) == 1
    assert "Jones" in cites[0].cited_text


def test_resolve_citation_arxiv_id(fresh_db):
    """A citation with an arxiv id should resolve to the local file
    that has that id in its path."""
    arxiv_fid = _seed_doc(
        fresh_db, path="/papers/arxiv-2401.12345.pdf",
        body="# Some Paper\n\nbody",
    )
    src_fid = _seed_doc(
        fresh_db, path="essay.md",
        body="See arxiv-2401.12345 for the original paper.",
    )
    pa.store_citations(fresh_db, src_fid, [
        {"cited_text": "see arxiv-2401.12345 for details", "year": None},
    ])
    cites = pa.get_citations_from(fresh_db, src_fid)
    assert cites[0].cited_file_id == arxiv_fid


def test_resolve_citation_author_year(fresh_db):
    """When cited paper's first chunk has the cited author + year,
    resolve to its file_id."""
    smith_fid = _seed_doc(
        fresh_db, path="/papers/smith.pdf",
        body="# Smith et al. 2024 — some title\n\nbody",
    )
    src_fid = _seed_doc(
        fresh_db, path="essay.md",
        body="As shown by Smith et al., 2024, ...",
    )
    pa.store_citations(fresh_db, src_fid, [
        {"cited_text": "Smith et al., 2024", "year": 2024},
    ])
    cites = pa.get_citations_from(fresh_db, src_fid)
    assert cites[0].cited_file_id == smith_fid


def test_resolve_citation_unresolved_when_no_match(fresh_db):
    src_fid = _seed_doc(fresh_db, path="essay.md", body="x")
    pa.store_citations(fresh_db, src_fid, [
        {"cited_text": "Nobody, 1999", "year": 1999},
    ])
    cites = pa.get_citations_from(fresh_db, src_fid)
    assert cites[0].cited_file_id is None


def test_get_citations_to_returns_incoming(fresh_db):
    """A doc that's been cited should be queryable from the cited side."""
    target_fid = _seed_doc(
        fresh_db, path="/papers/foo.pdf",
        body="# Smith 2024 Paper",
    )
    src1_fid = _seed_doc(fresh_db, path="essay1.md", body="Smith, 2024 says")
    src2_fid = _seed_doc(fresh_db, path="essay2.md", body="see Smith, 2024")
    for sfid in (src1_fid, src2_fid):
        pa.store_citations(fresh_db, sfid, [
            {"cited_text": "Smith, 2024", "year": 2024},
        ])
    incoming = pa.get_citations_to(fresh_db, target_fid)
    assert len(incoming) == 2


# ============================ process_after_index =====================

def test_process_after_index_extracts_citations(fresh_db, monkeypatch):
    """For non-PDF files, citation parsing still runs on the chunk text."""
    fid = _seed_doc(
        fresh_db, path="essay.md",
        body="As Smith, 2024 showed, this works.",
    )
    n_annot, n_cite = pa.process_after_index(fresh_db, fid, "essay.md")
    assert n_annot == 0
    assert n_cite == 1


def test_process_after_index_skips_pdf_extraction_without_pymupdf(
    fresh_db, monkeypatch,
):
    """No PyMuPDF → annotation count stays 0; citation parse still runs."""
    fid = _seed_doc(
        fresh_db, path="paper.pdf",
        body="Smith, 2024 is cited here.",
    )
    import sys
    monkeypatch.setitem(sys.modules, "fitz", None)
    n_annot, n_cite = pa.process_after_index(fresh_db, fid, "paper.pdf")
    assert n_annot == 0
    assert n_cite == 1


def test_process_after_index_swallows_failures(fresh_db, monkeypatch):
    """Citation parse failure shouldn't propagate."""
    fid = _seed_doc(fresh_db, path="essay.md", body="x")
    monkeypatch.setattr(
        pa, "extract_citations_from_text",
        lambda t: (_ for _ in ()).throw(RuntimeError("bad")),
    )
    n_annot, n_cite = pa.process_after_index(fresh_db, fid, "essay.md")
    assert n_annot == 0
    assert n_cite == 0
