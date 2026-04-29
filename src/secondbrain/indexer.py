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
from .transcriber import Transcriber

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


def extract_text(path: Path, transcriber: Transcriber | None = None) -> str:
    """Pull text out of a file. Plain reads for text/code; markitdown for documents;
    Whisper transcription for audio/video when a transcriber is supplied."""
    kind = classify_file(path)
    ext = path.suffix.lower()

    if ext in CODE_EXTENSIONS or ext in {".md", ".markdown", ".txt", ".rst", ".org"}:
        return path.read_text(encoding="utf-8", errors="replace")

    if kind == "document":
        md = _get_markitdown()
        result = md.convert(str(path))
        return result.text_content or ""

    if kind == "media":
        # Audio/video: transcribe with Whisper. Images currently fall through
        # to the empty return; OCR/CLIP arrives in Phase 1.4.
        if transcriber is not None and ext in _AUDIO_VIDEO_EXTS:
            log.info("transcribing %s with %s", path.name, transcriber.name)
            return transcriber.transcribe(path)
        return ""

    # Last-ditch attempt for unknown types
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


_AUDIO_VIDEO_EXTS = frozenset({
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus",
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
})


def chunk_text(
    text: str, target_size: int = 800, overlap: int = 150
) -> list[tuple[str, int]]:
    """Paragraph-aware chunking with character overlap.

    Splits on paragraph boundaries, packs into chunks near target_size, and falls back
    to character splits with overlap when a single paragraph is larger than target_size.
    Returns ``[(chunk_text, start_offset), ...]`` where start_offset is the chunk's
    approximate location in the original text. Offsets are used to find the nearest
    preceding heading for contextual embedding.
    """
    text = text.strip()
    if not text:
        return []

    chunks: list[tuple[str, int]] = []
    current = ""
    current_start = 0
    cursor = 0

    def flush():
        nonlocal current, current_start
        if current.strip():
            chunks.append((current.strip(), current_start))
        current = ""

    paragraph_pattern = "\n\n"
    parts = text.split(paragraph_pattern)
    for p in parts:
        p_stripped = p.strip()
        p_offset = cursor
        cursor += len(p) + len(paragraph_pattern)
        if not p_stripped:
            continue
        if len(p_stripped) > target_size:
            flush()
            step = max(1, target_size - overlap)
            for i in range(0, len(p_stripped), step):
                chunks.append((p_stripped[i : i + target_size], p_offset + i))
            continue
        if current and len(current) + len(p_stripped) + 2 > target_size:
            flush()
        if not current:
            current_start = p_offset
        current = f"{current}\n\n{p_stripped}" if current else p_stripped
    flush()
    return chunks


_HEADING_PATTERNS = (
    "# ", "## ", "### ", "#### ",
    "Slide number:", "<!-- Slide number:",
)


def find_nearest_heading(full_text: str, offset: int, max_lookback: int = 4000) -> str | None:
    """Find the nearest heading-like line before ``offset``.

    Recognises Markdown ATX headings and Markitdown's ``<!-- Slide number: N -->``
    markers (followed by a slide title on the next non-empty line).
    """
    start = max(0, offset - max_lookback)
    window = full_text[start:offset]
    lines = window.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if not line:
            continue
        for prefix in ("####", "###", "##", "#"):
            if line.startswith(prefix + " "):
                return line.lstrip("# ").strip()
        if line.startswith("<!-- Slide number:") or line.startswith("Slide number:"):
            for j in range(i + 1, min(i + 5, len(lines))):
                title = lines[j].strip()
                if title and not title.startswith("<!--"):
                    return title
    return None


def build_context_prefix(path: Path, full_text: str, chunk_offset: int) -> str:
    """Build a short context preamble for an embedded chunk.

    Including the filename, parent folder, and nearest heading lets the embedder
    place each chunk in its document; recall on conceptual queries improves
    measurably vs. embedding bare chunks.
    """
    parts = [f"Document: {path.name}"]
    if path.parent.name:
        parts.append(f"Folder: {path.parent.name}")
    heading = find_nearest_heading(full_text, chunk_offset)
    if heading:
        parts.append(f"Section: {heading}")
    return "\n".join(parts) + "\n\n"


def should_index(path: Path, cfg: Config, transcriber: Transcriber | None = None) -> tuple[bool, str | None]:
    """Decide whether a path is eligible for indexing."""
    if not path.exists() or not path.is_file():
        return False, "not a file"
    if is_ignored(path, cfg.ignore_globs):
        return False, "ignored by pattern"
    kind = classify_file(path)
    if kind == "other":
        return False, "unrecognized file type"
    ext = path.suffix.lower()
    if kind == "media":
        if ext in _AUDIO_VIDEO_EXTS:
            if transcriber is None:
                return False, "audio/video skipped (transcriber disabled)"
        else:
            # Image: OCR/CLIP arrives in Phase 1.4.
            return False, "image (OCR/CLIP not yet implemented)"
    try:
        size = path.stat().st_size
    except OSError as e:
        return False, f"stat failed: {e}"
    if size > cfg.max_file_bytes:
        return False, f"file too large ({size} bytes)"
    if size == 0:
        return False, "empty file"
    if kind in {"document", "code"} and ext not in (
        DOCUMENT_EXTENSIONS | CODE_EXTENSIONS
    ):
        return False, "extension not in allow list"
    return True, None


def index_file(
    conn: sqlite3.Connection,
    embedder: Embedder,
    cfg: Config,
    path: Path,
    transcriber: Transcriber | None = None,
) -> IndexResult:
    """Index a single file (extract -> chunk -> embed -> store)."""
    ok, reason = should_index(path, cfg, transcriber=transcriber)
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
        text = extract_text(path, transcriber=transcriber)
    except Exception as e:
        log.warning("extraction failed for %s: %s", path, e)
        return IndexResult(path, "error", reason=f"extraction: {e}")

    if not text.strip():
        return IndexResult(path, "skipped", reason="no extractable text")

    chunked = chunk_text(text, target_size=cfg.chunk_size, overlap=cfg.chunk_overlap)
    if not chunked:
        return IndexResult(path, "skipped", reason="no chunks produced")

    chunk_texts = [c for c, _ in chunked]
    contextualized = [
        build_context_prefix(path, text, offset) + c for c, offset in chunked
    ]

    try:
        embeddings = embedder.embed_documents(contextualized)
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
    replace_chunks(conn, file_id, list(zip(chunk_texts, embeddings, strict=True)))
    conn.commit()
    return IndexResult(path, "indexed", chunks=len(chunk_texts))


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
    transcriber: Transcriber | None = None,
) -> dict[str, int]:
    """Walk a folder and index all eligible files. Returns counts."""
    counts = {"indexed": 0, "skipped": 0, "unchanged": 0, "error": 0}
    for p in walk_folder(folder, cfg):
        result = index_file(conn, embedder, cfg, p, transcriber=transcriber)
        counts[result.status] = counts.get(result.status, 0) + 1
        if progress:
            progress(result)
    return counts
