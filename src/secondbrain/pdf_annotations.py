"""Phase 84 + 85: PDF annotation extraction + citation graph.

Phase 84 — Annotation extraction
  When you highlight a paper in Preview / Acrobat / Drawboard, those
  markups land in the PDF's annotation layer. PyMuPDF (fitz) reads
  them out cleanly. We persist them so they're queryable + linkable
  back to the source doc.

  Why store separately from chunks: a highlight is metadata about
  a spot in the doc, not new content. The chunk pipeline already has
  the underlying text; what's interesting here is "the user CARED
  about THIS sentence" — that's high-signal for ranking and
  retrieval.

Phase 85 — Citation graph
  For academic PDFs, extract reference-style strings ("Smith et al.,
  2024", numbered "[15]" with reference list) into the citations
  table. When the cited paper is itself in the brain (matching by
  title / author / year), wire a hard edge cited_file_id → src_file_id.
  Otherwise the edge is a free-text reference for context.

  The graph powers "what cites this paper?" + "what does this paper
  cite?" queries that are invaluable for class research.

Both are best-effort: PyMuPDF missing → skip. Doc isn't a PDF →
skip. We never block ingest on annotation extraction.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)


# ---- Tunables ---------------------------------------------------------

# Bounded text length per annotation to avoid 50KB highlights blowing
# up the table. Most highlights are 1-3 sentences.
_MAX_ANCHOR_LEN = 1000
# Per-citation text bounded similarly.
_MAX_CITED_TEXT_LEN = 500
# Cap citations per doc to avoid pathological cases (a textbook with
# thousands of refs).
_MAX_CITATIONS_PER_DOC = 200


# ---- Data shapes ------------------------------------------------------

@dataclass
class PDFAnnotation:
    id: int
    file_id: int
    page: int
    kind: str       # 'highlight' | 'note' | 'underline' | 'strike'
    anchor: str
    note: str | None
    color: str | None
    created_at: float


@dataclass
class Citation:
    id: int
    src_file_id: int
    cited_file_id: int | None
    cited_text: str
    year: int | None
    created_at: float


# ============================ Phase 84: PyMuPDF =======================

def extract_annotations_from_pdf(path) -> list[dict]:
    """Read annotations out of a PDF file. Returns ``[{page, kind,
    anchor, note?, color?}, ...]``. Empty list when the file has no
    annotations or PyMuPDF is unavailable.

    Defensive: any exception in the parsing path becomes an empty
    list rather than crashing the indexer.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        log.debug("pdf_annotations: PyMuPDF not installed, skipping")
        return []
    try:
        doc = fitz.open(str(path))
    except Exception as e:  # noqa: BLE001
        log.warning(
            "pdf_annotations: failed to open %s: %s", path, type(e).__name__,
        )
        return []
    out: list[dict] = []
    try:
        for page_num, page in enumerate(doc, start=1):
            try:
                annots = page.annots()
            except Exception:  # noqa: BLE001
                continue
            if annots is None:
                continue
            for annot in annots:
                kind = _normalize_annotation_kind(
                    annot.type[1] if annot.type else "",
                )
                if not kind:
                    continue
                anchor = _annot_anchor_text(page, annot)
                if not anchor:
                    continue
                if len(anchor) > _MAX_ANCHOR_LEN:
                    anchor = anchor[:_MAX_ANCHOR_LEN] + "…"
                info = annot.info or {}
                content = (info.get("content") or "").strip()
                color = _color_to_hex(annot.colors)
                out.append({
                    "page": page_num,
                    "kind": kind,
                    "anchor": anchor,
                    "note": content or None,
                    "color": color,
                })
    finally:
        try:
            doc.close()
        except Exception:  # noqa: BLE001
            pass
    return out


_KIND_MAP = {
    "Highlight": "highlight",
    "Underline": "underline",
    "StrikeOut": "strike",
    "Squiggly": "underline",
    "Text": "note",
    "FreeText": "note",
    "Note": "note",
    "Caret": "note",
}


def _normalize_annotation_kind(raw: str) -> str:
    """Map PyMuPDF annotation type strings to our normalised set."""
    return _KIND_MAP.get(raw, "")


def _annot_anchor_text(page, annot) -> str:
    """Pull the text the annotation covers. PyMuPDF's get_textbox
    handles highlights/underlines; for plain notes we fall back to
    the annotation's own info content."""
    try:
        rect = annot.rect
    except Exception:  # noqa: BLE001
        return ""
    try:
        text = page.get_textbox(rect)
    except Exception:  # noqa: BLE001
        text = ""
    text = (text or "").strip()
    if not text:
        info = annot.info or {}
        text = (info.get("content") or "").strip()
    return " ".join(text.split())


def _color_to_hex(colors) -> str | None:
    """PyMuPDF returns colors as a dict with stroke/fill float tuples.
    Return a hex string for the most common (stroke) component."""
    if not colors:
        return None
    stroke = colors.get("stroke") or colors.get("fill")
    if not stroke:
        return None
    try:
        r, g, b = (int(round(c * 255)) for c in stroke[:3])
        return f"#{r:02x}{g:02x}{b:02x}"
    except (TypeError, ValueError):
        return None


# ============================ Phase 84: persistence ===================

def store_annotations(
    conn: sqlite3.Connection, file_id: int, annotations: list[dict],
) -> int:
    """Replace this file's annotations with the given list. Returns
    count newly inserted.

    Replace-style so re-indexing reflects the latest annotation state
    (the user might have added/removed marks between ingests).
    """
    conn.execute(
        "DELETE FROM pdf_annotations WHERE file_id = ?", (file_id,),
    )
    if not annotations:
        conn.commit()
        return 0
    n = time.time()
    inserted = 0
    for a in annotations:
        try:
            conn.execute(
                "INSERT OR IGNORE INTO pdf_annotations"
                "(file_id, page, kind, anchor, note, color, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    file_id, int(a.get("page") or 1),
                    a.get("kind") or "highlight",
                    (a.get("anchor") or "")[:_MAX_ANCHOR_LEN],
                    a.get("note"),
                    a.get("color"),
                    n,
                ),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            continue
    conn.commit()
    return inserted


def get_annotations(
    conn: sqlite3.Connection, file_id: int,
) -> list[PDFAnnotation]:
    rows = conn.execute(
        "SELECT * FROM pdf_annotations WHERE file_id = ? "
        "ORDER BY page ASC, id ASC",
        (file_id,),
    ).fetchall()
    return [
        PDFAnnotation(
            id=int(r["id"]),
            file_id=int(r["file_id"]),
            page=int(r["page"]),
            kind=r["kind"],
            anchor=r["anchor"],
            note=r["note"],
            color=r["color"],
            created_at=r["created_at"],
        )
        for r in rows
    ]


def annotations_count(
    conn: sqlite3.Connection, file_id: int,
) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM pdf_annotations WHERE file_id = ?",
        (file_id,),
    ).fetchone()
    return int(row["n"]) if row else 0


# ============================ Phase 85: citations =====================

# "Smith et al., 2024" / "Smith and Jones (2023)" / "(Smith, 2022)"
_INLINE_CITE_RE = re.compile(
    r"\b([A-Z][a-zA-Z\-]{1,30}(?:\s+et\s+al\.?|\s+(?:and|&)\s+"
    r"[A-Z][a-zA-Z\-]{1,30})?)\s*,?\s*\(?\s*(\d{4})\s*\)?",
)
# Numbered references: "[12]" or "[12, 13, 14]"
_NUMBERED_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# Reference list line: "[12] Smith, J. (2024). Title. Journal."
_REFLIST_RE = re.compile(
    r"^\s*\[(\d+)\]\s*(.+)$", re.MULTILINE,
)


def extract_citations_from_text(text: str) -> list[dict]:
    """Pull citation-shaped strings out of doc text. Returns
    ``[{cited_text, year}, ...]``. Lossy + heuristic — don't rely on
    100% recall; the citation graph is a navigation aid, not a
    bibliography tool.

    Three patterns:
      1. Inline author-year: 'Smith et al., 2024'
      2. Numbered inline: '[12]' (must have a matching ref list entry)
      3. Reference list: '[12] Smith J. ...' → captured as full text
    """
    if not text:
        return []
    out: list[dict] = []
    seen: set[str] = set()

    # Author-year inline.
    for m in _INLINE_CITE_RE.finditer(text):
        author = m.group(1).strip()
        year_s = m.group(2)
        try:
            year = int(year_s)
        except (ValueError, TypeError):
            year = None
        if year is None or not (1900 <= year <= 2100):
            continue
        cited = f"{author}, {year}"
        if cited in seen:
            continue
        seen.add(cited)
        out.append({"cited_text": cited, "year": year})

    # Reference-list entries.
    for m in _REFLIST_RE.finditer(text):
        ref_text = m.group(2).strip()
        if not ref_text:
            continue
        if ref_text in seen:
            continue
        seen.add(ref_text)
        # Try to pull a year out of the ref text.
        y_match = re.search(r"\b(19|20)\d{2}\b", ref_text)
        year = int(y_match.group(0)) if y_match else None
        cited = ref_text[:_MAX_CITED_TEXT_LEN]
        out.append({"cited_text": cited, "year": year})

    return out[:_MAX_CITATIONS_PER_DOC]


def store_citations(
    conn: sqlite3.Connection, src_file_id: int, citations: list[dict],
) -> int:
    """Replace src's citations with the given list. Tries to resolve
    each cited_text to an existing file_id by fuzzy title match;
    leaves cited_file_id NULL when no match.

    Returns count inserted.
    """
    conn.execute(
        "DELETE FROM citations WHERE src_file_id = ?", (src_file_id,),
    )
    if not citations:
        conn.commit()
        return 0
    n = time.time()
    inserted = 0
    for c in citations:
        text = (c.get("cited_text") or "").strip()
        if not text:
            continue
        text = text[:_MAX_CITED_TEXT_LEN]
        # Pass src_file_id so the resolver can exclude self-matches:
        # an essay that *cites* "Smith, 2024" also *contains* the
        # string "Smith, 2024", which without the exclusion would
        # cause the resolver to point the citation at itself.
        cited_fid = _resolve_citation(
            conn, text, c.get("year"), exclude_file_id=src_file_id,
        )
        try:
            conn.execute(
                "INSERT OR IGNORE INTO citations"
                "(src_file_id, cited_file_id, cited_text, year, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (src_file_id, cited_fid, text, c.get("year"), n),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            continue
    conn.commit()
    return inserted


def _resolve_citation(
    conn: sqlite3.Connection, cited_text: str, year: int | None,
    *, exclude_file_id: int | None = None,
) -> int | None:
    """Best-effort resolution of a citation string to a file_id in
    the brain. Two paths:

      1. Exact-author-and-year: a doc whose first chunk's H1 contains
         the cited_text's leading author surname AND the year.
      2. Substring on path/title: 'arxiv-2401.12345' style refs match
         file paths containing the same id.

    ``exclude_file_id`` (optional): never resolve to this file. The
    common case is passing the citing doc's own id so we don't
    accidentally make a paper cite itself when the citation string
    appears in its own body.

    Returns None when no confident match.
    """
    # Quick path: arxiv / DOI ids in the citation text.
    arxiv_match = re.search(r"\b(\d{4}\.\d{4,5})\b", cited_text)
    if arxiv_match:
        if exclude_file_id is not None:
            row = conn.execute(
                "SELECT id FROM files WHERE path LIKE ? AND id != ? "
                "LIMIT 1",
                (f"%{arxiv_match.group(1)}%", exclude_file_id),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT id FROM files WHERE path LIKE ? LIMIT 1",
                (f"%{arxiv_match.group(1)}%",),
            ).fetchone()
        if row:
            return int(row["id"])
    # Author-year heuristic: pull leading capitalised word + the year.
    words = cited_text.split()
    if not words or year is None:
        return None
    author_token = re.sub(r"[^A-Za-z\-]", "", words[0])
    if len(author_token) < 3:
        return None
    # Look for files whose first chunk's H1 line (not full body)
    # mentions both the author and the year. H1-only is a much
    # stricter heuristic: an essay that *cites* "Smith, 2024" in the
    # body shouldn't claim to BE that paper. Only docs whose title
    # genuinely names the author + year qualify.
    #
    # Bounded to 200-most-recent docs. Excludes the citing file.
    if exclude_file_id is not None:
        rows = conn.execute(
            "SELECT f.id, c.text "
            "FROM files f JOIN chunks c ON c.file_id = f.id "
            "WHERE c.chunk_index = 0 AND f.id != ? "
            "ORDER BY f.indexed_at DESC LIMIT 200",
            (exclude_file_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT f.id, c.text "
            "FROM files f JOIN chunks c ON c.file_id = f.id "
            "WHERE c.chunk_index = 0 "
            "ORDER BY f.indexed_at DESC LIMIT 200",
        ).fetchall()
    needle_year = str(year)
    author_lower = author_token.lower()
    for r in rows:
        h1 = _first_h1_line(r["text"] or "").lower()
        if not h1:
            continue
        if author_lower in h1 and needle_year in h1:
            return int(r["id"])
    return None


def _first_h1_line(text: str) -> str:
    """Return the first '# heading' line of a Markdown chunk, or an
    empty string when the doc doesn't lead with a heading."""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip()
        if s:
            return ""  # first non-empty line wasn't a heading
    return ""


def get_citations_from(
    conn: sqlite3.Connection, src_file_id: int,
) -> list[Citation]:
    """All citations made BY this doc (outgoing edges)."""
    rows = conn.execute(
        "SELECT * FROM citations WHERE src_file_id = ? "
        "ORDER BY year DESC, id ASC",
        (src_file_id,),
    ).fetchall()
    return [_row_to_citation(r) for r in rows]


def get_citations_to(
    conn: sqlite3.Connection, cited_file_id: int,
) -> list[Citation]:
    """All citations TO this doc (incoming edges) — only resolved
    ones (with cited_file_id), since unresolved free-text refs can't
    be queried from the target side."""
    rows = conn.execute(
        "SELECT * FROM citations WHERE cited_file_id = ? "
        "ORDER BY id DESC",
        (cited_file_id,),
    ).fetchall()
    return [_row_to_citation(r) for r in rows]


def _row_to_citation(row: sqlite3.Row) -> Citation:
    return Citation(
        id=int(row["id"]),
        src_file_id=int(row["src_file_id"]),
        cited_file_id=row["cited_file_id"],
        cited_text=row["cited_text"],
        year=row["year"],
        created_at=row["created_at"],
    )


# ============================ Indexer hook ============================

def process_after_index(
    conn: sqlite3.Connection, file_id: int, path: str,
) -> tuple[int, int]:
    """Indexer entry point. For PDF files, extract annotations + parse
    citation strings. For other docs, just citation parsing on the
    text body.

    Returns ``(n_annotations, n_citations)``. Best-effort — failures
    log + return zeros so the indexer never blocks.
    """
    n_annot = 0
    n_cite = 0
    is_pdf = str(path).lower().endswith(".pdf")
    if is_pdf:
        try:
            annots = extract_annotations_from_pdf(path)
            n_annot = store_annotations(conn, file_id, annots)
        except Exception as e:  # noqa: BLE001
            log.warning(
                "pdf_annotations: extract failed for %s: %s", path, e,
            )
    # Citations: parse from chunk text regardless of file kind, but
    # the heuristic-quality tradeoff means we mainly catch academic-
    # style content.
    try:
        body_rows = conn.execute(
            "SELECT text FROM chunks WHERE file_id = ? "
            "ORDER BY chunk_index ASC",
            (file_id,),
        ).fetchall()
        body = "\n".join(r["text"] or "" for r in body_rows)
        cites = extract_citations_from_text(body)
        if cites:
            n_cite = store_citations(conn, file_id, cites)
    except Exception as e:  # noqa: BLE001
        log.warning(
            "pdf_annotations: citation parse failed for %s: %s", path, e,
        )
    return n_annot, n_cite
