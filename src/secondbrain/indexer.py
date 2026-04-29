"""File extraction, chunking, and indexing pipeline."""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .config import (
    CODE_EXTENSIONS,
    Config,
    DOCUMENT_EXTENSIONS,
    classify_file,
    is_ignored,
)
from .db import delete_file, get_file_by_path, replace_chunks, upsert_file
from .embedder import Embedder

log = logging.getLogger(__name__)

# Markitdown is the workhorse for documents; it handles PDF/DOCX/PPTX/HTML/etc.
# We lazy-import it to keep startup fast for commands that don't index.
_markitdown_instance = None


def _get_markitdown():
    global _markitdown_instance
    if _markitdown_instance is None:
        from markitdown import MarkItDown

        _markitdown_instance = MarkItDown()
    return _markitdown_instance


@dataclass
class IndexResult:
    path: Path
    status: str  # "indexed" | "skipped" | "unchanged" | "deleted" | "error"
    chunks: int = 0
    reason: str | None = None


def file_hash(path: Path) -> str:
    """Stable content hash for change detection."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(64 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def extract_text(path: Path) -> str:
    """Pull text out of a file. Plain reads for text/code; markitdown for documents."""
    kind = classify_file(path)
    ext = path.suffix.lower()

    if ext in CODE_EXTENSIONS or ext in {".md", ".markdown", ".txt", ".rst", ".org"}:
        return path.read_text(encoding="utf-8", errors="replace")

    if kind == "document":
        md = _get_markitdown()
        result = md.convert(str(path))
        return result.text_content or ""

    if kind == "media":
        # Phase 1 will add Whisper transcription / OCR. For Phase 0 we skip.
        return ""

    # Last-ditch attempt for unknown types
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def chunk_text(text: str, target_size: int = 800, overlap: int = 150) -> list[str]:
    """Paragraph-aware chunking with character overlap.

    Splits on paragraph boundaries, packs into chunks near target_size, and falls back
    to character splits with overlap when a single paragraph is larger than target_size.
    """
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for p in paragraphs:
        if len(p) > target_size:
            flush()
            step = max(1, target_size - overlap)
            for i in range(0, len(p), step):
                chunks.append(p[i : i + target_size])
            continue
        if current and len(current) + len(p) + 2 > target_size:
            flush()
        current = f"{current}\n\n{p}" if current else p
    flush()
    return chunks


def should_index(path: Path, cfg: Config) -> tuple[bool, str | None]:
    """Decide whether a path is eligible for indexing."""
    if not path.exists() or not path.is_file():
        return False, "not a file"
    if is_ignored(path, cfg.ignore_globs):
        return False, "ignored by pattern"
    kind = classify_file(path)
    if kind == "other":
        return False, "unrecognized file type"
    if kind == "media":
        # Phase 0 skips media; will be handled in Phase 1 via transcription.
        return False, "media (transcription not yet implemented)"
    try:
        size = path.stat().st_size
    except OSError as e:
        return False, f"stat failed: {e}"
    if size > cfg.max_file_bytes:
        return False, f"file too large ({size} bytes)"
    if size == 0:
        return False, "empty file"
    if kind in {"document", "code"} and path.suffix.lower() not in (
        DOCUMENT_EXTENSIONS | CODE_EXTENSIONS
    ):
        return False, "extension not in allow list"
    return True, None


def index_file(
    conn: sqlite3.Connection,
    embedder: Embedder,
    cfg: Config,
    path: Path,
) -> IndexResult:
    """Index a single file (extract -> chunk -> embed -> store)."""
    ok, reason = should_index(path, cfg)
    if not ok:
        return IndexResult(path, "skipped", reason=reason)

    try:
        st = path.stat()
        chash = file_hash(path)
    except OSError as e:
        return IndexResult(path, "error", reason=f"stat/hash failed: {e}")

    existing = get_file_by_path(conn, str(path))
    if existing and existing["content_hash"] == chash:
        return IndexResult(path, "unchanged")

    try:
        text = extract_text(path)
    except Exception as e:
        log.warning("extraction failed for %s: %s", path, e)
        return IndexResult(path, "error", reason=f"extraction: {e}")

    if not text.strip():
        return IndexResult(path, "skipped", reason="no extractable text")

    chunks = chunk_text(text, target_size=cfg.chunk_size, overlap=cfg.chunk_overlap)
    if not chunks:
        return IndexResult(path, "skipped", reason="no chunks produced")

    try:
        embeddings = embedder.embed_documents(chunks)
    except Exception as e:
        log.warning("embedding failed for %s: %s", path, e)
        return IndexResult(path, "error", reason=f"embedding: {e}")

    kind = classify_file(path)
    file_id = upsert_file(
        conn,
        path=str(path),
        mtime=st.st_mtime,
        size=st.st_size,
        kind=kind,
        content_hash=chash,
    )
    replace_chunks(conn, file_id, list(zip(chunks, embeddings, strict=True)))
    conn.commit()
    return IndexResult(path, "indexed", chunks=len(chunks))


def remove_file(conn: sqlite3.Connection, path: Path) -> IndexResult:
    """Drop a file and its chunks from the index."""
    delete_file(conn, str(path))
    conn.commit()
    return IndexResult(path, "deleted")


def walk_folder(folder: Path, cfg: Config) -> Iterator[Path]:
    """Yield candidate files under folder, skipping ignored paths early."""
    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        if is_ignored(p, cfg.ignore_globs):
            continue
        yield p


def index_folder(
    conn: sqlite3.Connection,
    embedder: Embedder,
    cfg: Config,
    folder: Path,
    progress=None,
) -> dict[str, int]:
    """Walk a folder and index all eligible files. Returns counts."""
    counts = {"indexed": 0, "skipped": 0, "unchanged": 0, "error": 0}
    for p in walk_folder(folder, cfg):
        result = index_file(conn, embedder, cfg, p)
        counts[result.status] = counts.get(result.status, 0) + 1
        if progress:
            progress(result)
    return counts
