"""File extraction, chunking, and indexing pipeline."""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
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
from .db import delete_file, get_file_by_path, insert_entities, replace_chunks, upsert_file
from .embedder import Embedder
from .entities import EntityExtractor
from .imager import OCREngine
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


def extract_text(
    path: Path,
    transcriber: Transcriber | None = None,
    ocr_engine: OCREngine | None = None,
) -> str:
    """Pull text out of a file.

    - text/code/markdown: plain read
    - documents (PDF, DOCX, etc.): markitdown
    - audio/video: Whisper transcription (if transcriber supplied)
    - images: OCR (if ocr_engine supplied)
    """
    kind = classify_file(path)
    ext = path.suffix.lower()

    if ext in CODE_EXTENSIONS or ext in {".md", ".markdown", ".txt", ".rst", ".org"}:
        return path.read_text(encoding="utf-8", errors="replace")

    if kind == "document":
        md = _get_markitdown()
        result = md.convert(str(path))
        return result.text_content or ""

    if kind == "audio_video":
        if transcriber is not None:
            log.info("transcribing %s with %s", path.name, transcriber.name)
            return transcriber.transcribe(path)
        return ""

    if kind == "image":
        if ocr_engine is not None:
            log.info("OCR'ing %s with %s", path.name, ocr_engine.name)
            return ocr_engine.ocr(path)
        return ""

    # Last-ditch attempt for unknown types
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


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


def should_index(
    path: Path,
    cfg: Config,
    transcriber: Transcriber | None = None,
    ocr_engine: OCREngine | None = None,
) -> tuple[bool, str | None]:
    """Decide whether a path is eligible for indexing."""
    if not path.exists() or not path.is_file():
        return False, "not a file"
    if is_ignored(path, cfg.ignore_globs):
        return False, "ignored by pattern"
    kind = classify_file(path)
    if kind == "other":
        return False, "unrecognized file type"
    ext = path.suffix.lower()
    if kind == "audio_video" and transcriber is None:
        return False, "audio/video skipped (transcriber disabled)"
    if kind == "image" and ocr_engine is None:
        return False, "image skipped (OCR disabled)"
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
    ocr_engine: OCREngine | None = None,
    entity_extractor: EntityExtractor | None = None,
) -> IndexResult:
    """Index a single file (extract -> chunk -> embed -> entities -> store)."""
    ok, reason = should_index(path, cfg, transcriber=transcriber, ocr_engine=ocr_engine)
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
        text = extract_text(path, transcriber=transcriber, ocr_engine=ocr_engine)
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
    chunk_ids = replace_chunks(
        conn, file_id, list(zip(chunk_texts, embeddings, strict=True))
    )

    if entity_extractor is not None:
        try:
            for chunk_id, chunk_text_val in zip(chunk_ids, chunk_texts, strict=True):
                ents = entity_extractor.extract(chunk_text_val)
                if ents:
                    insert_entities(
                        conn,
                        chunk_id,
                        [(e.text, e.label) for e in ents],
                    )
        except Exception as e:
            # Entity failures are non-fatal: the chunk + embedding are already stored,
            # so the file is still searchable - we just won't have NER on it.
            log.warning("entity extraction failed for %s: %s", path, e)

    conn.commit()
    return IndexResult(path, "indexed", chunks=len(chunk_texts))


def remove_file(conn: sqlite3.Connection, path: Path) -> IndexResult:
    """Drop a file and its chunks from the index."""
    delete_file(conn, str(path))
    conn.commit()
    return IndexResult(path, "deleted")


_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 second-brain/0.0.1"
)


def _fetch_url_to_tempfile(url: str) -> tuple[str, str]:
    """Download a URL with a real user-agent so sites like Wikipedia don't 403 us.

    Returns (local_path, content_type). Caller must clean up the temp file.
    """
    import os
    import tempfile
    from urllib.parse import urlparse

    import requests

    resp = requests.get(
        url,
        headers={"User-Agent": _USER_AGENT, "Accept": "*/*"},
        timeout=60,
        allow_redirects=True,
    )
    resp.raise_for_status()
    ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
    ext_map = {
        "text/html": ".html",
        "application/xhtml+xml": ".html",
        "application/pdf": ".pdf",
        "text/plain": ".txt",
        "text/markdown": ".md",
        "application/json": ".json",
        "text/xml": ".xml",
        "application/xml": ".xml",
    }
    ext = ext_map.get(ctype)
    if not ext:
        path = urlparse(url).path
        _, suffix = os.path.splitext(path)
        ext = suffix or ".html"
    fd, tmp = tempfile.mkstemp(suffix=ext, prefix="sb-url-")
    try:
        os.write(fd, resp.content)
    finally:
        os.close(fd)
    return tmp, ctype


def index_url(
    conn: sqlite3.Connection,
    embedder: Embedder,
    cfg: Config,
    url: str,
    entity_extractor: EntityExtractor | None = None,
) -> IndexResult:
    """Fetch a URL, extract text via markitdown, and index it like a file.

    HTML pages are pre-fetched with a real User-Agent (Wikipedia / many sites
    reject the default markitdown UA), then converted from disk. YouTube URLs
    go straight through markitdown so its transcript path is used. PDFs are
    downloaded and parsed locally. The URL is stored as the "path" with
    kind='url'; content_hash on extracted text means re-ingesting an
    unchanged page is a no-op.
    """
    import os

    label_path = Path(url)
    md = _get_markitdown()
    is_youtube = "youtube.com/watch" in url or "youtu.be/" in url
    tmp_path: str | None = None
    try:
        if is_youtube:
            result = md.convert(url)
        else:
            tmp_path, _ctype = _fetch_url_to_tempfile(url)
            result = md.convert(tmp_path)
    except Exception as e:
        log.warning("URL fetch/convert failed for %s: %s", url, e)
        return IndexResult(label_path, "error", reason=f"fetch/convert: {e}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    text = (result.text_content or "").strip()
    if not text:
        return IndexResult(label_path, "skipped", reason="no extractable text")

    chash = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()
    existing = get_file_by_path(conn, url)
    if existing and existing["content_hash"] == chash:
        return IndexResult(label_path, "unchanged")

    chunked = chunk_text(text, target_size=cfg.chunk_size, overlap=cfg.chunk_overlap)
    if not chunked:
        return IndexResult(label_path, "skipped", reason="no chunks produced")

    chunk_texts = [c for c, _ in chunked]
    title = getattr(result, "title", None) or url
    contextualized = [
        f"Source: URL\nURL: {url}\nTitle: {title}\n\n{c}" for c in chunk_texts
    ]
    try:
        embeddings = embedder.embed_documents(contextualized)
    except Exception as e:
        log.warning("embedding failed for url %s: %s", url, e)
        return IndexResult(label_path, "error", reason=f"embedding: {e}")

    file_id = upsert_file(
        conn,
        path=url,
        mtime=time.time(),
        size=len(text.encode("utf-8", errors="replace")),
        kind="url",
        content_hash=chash,
    )
    chunk_ids = replace_chunks(
        conn, file_id, list(zip(chunk_texts, embeddings, strict=True))
    )

    if entity_extractor is not None:
        try:
            for chunk_id, chunk_text_val in zip(chunk_ids, chunk_texts, strict=True):
                ents = entity_extractor.extract(chunk_text_val)
                if ents:
                    insert_entities(
                        conn, chunk_id, [(e.text, e.label) for e in ents]
                    )
        except Exception as e:
            log.warning("entity extraction failed for url %s: %s", url, e)

    conn.commit()
    return IndexResult(label_path, "indexed", chunks=len(chunk_texts))


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
    ocr_engine: OCREngine | None = None,
    entity_extractor: EntityExtractor | None = None,
) -> dict[str, int]:
    """Walk a folder and index all eligible files. Returns counts."""
    counts = {"indexed": 0, "skipped": 0, "unchanged": 0, "error": 0}
    for p in walk_folder(folder, cfg):
        result = index_file(
            conn, embedder, cfg, p,
            transcriber=transcriber, ocr_engine=ocr_engine,
            entity_extractor=entity_extractor,
        )
        counts[result.status] = counts.get(result.status, 0) + 1
        if progress:
            progress(result)
    return counts
