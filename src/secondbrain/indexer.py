"""File extraction, chunking, and indexing pipeline."""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .config import (
    CODE_EXTENSIONS,
    DOCUMENT_EXTENSIONS,
    Config,
    classify_file,
    is_ignored,
)
from .db import (
    add_alias,
    delete_file,
    find_file_by_hash,
    get_file_by_path,
    insert_entities,
    replace_chunks,
    upsert_file,
    upsert_image_embedding,
)
from .embedder import Embedder
from .entities import EntityExtractor
from .image_embedder import ImageEmbedder
from .imager import OCREngine
from .transcriber import Transcriber

log = logging.getLogger(__name__)


def _link_after_index(conn, file_id: int) -> None:
    """Phase 52 + 66: compute backlinks AND person-mention links for
    the freshly-indexed file.

    Both calls are best-effort — a downstream failure must never take
    down an ingest. Logged at WARNING so transient hiccups are visible
    without blowing up the indexer pipeline.
    """
    try:
        from .backlinks import link_doc
        link_doc(conn, file_id)
    except Exception as e:  # noqa: BLE001
        log.warning("backlinks: link_doc failed for file_id=%s: %s",
                    file_id, e)
    try:
        from .people import link_after_index
        link_after_index(conn, file_id)
    except Exception as e:  # noqa: BLE001
        log.warning("people: link failed for file_id=%s: %s",
                    file_id, e)
    try:
        # Phase 84/85 hook needs the file's path. We have file_id;
        # pull path back so PDF annotation extraction can open the
        # underlying file.
        row = conn.execute(
            "SELECT path FROM files WHERE id = ?", (file_id,),
        ).fetchone()
        if row:
            from .pdf_annotations import process_after_index as _pa
            _pa(conn, file_id, row["path"])
    except Exception as e:  # noqa: BLE001
        log.warning(
            "pdf_annotations: hook failed for file_id=%s: %s",
            file_id, e,
        )


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


_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")


def strip_markdown_decorations(text: str) -> str:
    """Remove markdown link/image syntax for cleaner downstream NER.

    Markitdown emits e.g. '[Buzz Aldrin](/wiki/Buzz_Aldrin)' which spaCy then
    sees as one token bag and produces garbage entities like
    'Buzz Aldrin](/wiki/Buzz_Aldrin'. We keep the visible text, drop the URL.
    Embeddings still see the original links because the structure is
    informative there.
    """
    text = _MD_IMAGE.sub(r"\1", text)
    text = _MD_LINK.sub(r"\1", text)
    return text


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


def _virtual_path_scheme(virtual_path: str) -> str:
    """Extract the ``<scheme>`` from ``<scheme>://<rest>``; "" if none.

    Used as a fallback hash salt for connector docs that didn't pass an
    explicit ``source=``. Keeps Reddit/Linear/HN/etc. from ever pretending
    to be the same content as one another.
    """
    idx = virtual_path.find("://")
    return virtual_path[:idx] if idx > 0 else ""


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
    image_embedder: ImageEmbedder | None = None,
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
    if kind == "image" and ocr_engine is None and image_embedder is None:
        return False, "image skipped (OCR and multimodal disabled)"
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
    image_embedder: ImageEmbedder | None = None,
) -> IndexResult:
    """Index a single file (extract -> chunk -> embed -> entities -> store).

    For images, OCR text (if enabled) flows through the regular pipeline AND
    the image is also embedded via the multimodal model (if enabled) into a
    side table for semantic image search.
    """
    log.info("indexing %s", path)
    ok, reason = should_index(
        path, cfg,
        transcriber=transcriber, ocr_engine=ocr_engine, image_embedder=image_embedder,
    )
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

    # Hash-dedup across paths: if some other file with the same content hash
    # is already indexed, register this path as an alias and skip embedding.
    # First-seen wins as the canonical path; later copies become aliases.
    if existing is None:
        twin = find_file_by_hash(conn, chash)
        if twin is not None:
            add_alias(conn, twin["id"], str(path))
            conn.commit()
            return IndexResult(path, "alias", reason=f"duplicate of {twin['path']}")

    try:
        text = extract_text(path, transcriber=transcriber, ocr_engine=ocr_engine)
    except Exception as e:
        log.warning("extraction failed for %s: %s", path, e)
        return IndexResult(path, "error", reason=f"extraction: {e}")

    kind = classify_file(path)
    has_text = bool(text.strip())
    will_embed_image = kind == "image" and image_embedder is not None

    if not has_text and not will_embed_image:
        return IndexResult(path, "skipped", reason="no extractable text")

    chunk_texts: list[str] = []
    chunk_offsets: list[int] = []
    embeddings: list[list[float]] = []

    if has_text:
        chunked = chunk_text(text, target_size=cfg.chunk_size, overlap=cfg.chunk_overlap)
        if chunked:
            chunk_texts = [c for c, _ in chunked]
            chunk_offsets = [off for _, off in chunked]
            contextualized = [
                build_context_prefix(path, text, offset) + c for c, offset in chunked
            ]
            try:
                embeddings = embedder.embed_documents(contextualized)
            except Exception as e:
                log.warning("embedding failed for %s: %s", path, e)
                return IndexResult(path, "error", reason=f"embedding: {e}")

    file_id = upsert_file(
        conn,
        path=str(path),
        mtime=st.st_mtime,
        size=st.st_size,
        kind=kind,
        content_hash=chash,
    )
    chunk_ids: list[int] = []
    if chunk_texts:
        chunk_ids = replace_chunks(
            conn, file_id,
            list(zip(chunk_texts, embeddings, chunk_offsets, strict=True)),
        )

    if will_embed_image:
        try:
            img_emb = image_embedder.embed_image(path)
            upsert_image_embedding(
                conn, file_id, image_embedder.name, image_embedder.dim, img_emb
            )
        except Exception as e:
            # Don't fail the whole file - OCR (if any) is already stored.
            log.warning("image embedding failed for %s: %s", path, e)

    _run_entity_extraction(conn, chunk_ids, chunk_texts, entity_extractor, label=str(path))

    if chunk_ids:
        _link_after_index(conn, file_id)

    conn.commit()
    return IndexResult(path, "indexed", chunks=len(chunk_texts))


def remove_file(conn: sqlite3.Connection, path: Path) -> IndexResult:
    """Drop a file and its chunks from the index."""
    delete_file(conn, str(path))
    conn.commit()
    return IndexResult(path, "deleted")


def dedupe_existing(
    conn: sqlite3.Connection, dry_run: bool = False
) -> dict[str, int | list[tuple[str, str]]]:
    """Walk the existing index, find files sharing a content_hash, and convert
    duplicates to aliases.

    Keeps the oldest indexed file as canonical; the rest get their chunks /
    entities / image embeddings dropped (cascade) and their path moved into
    `file_aliases`. Returns counts plus an ``aliased`` list of
    ``(canonical_path, dup_path)`` tuples for reporting / verification.

    Idempotent. Safe to run on a populated index even after the indexer-side
    dedup is in place - it cleans up duplicates added before that fix shipped.

    Set ``dry_run=True`` to inspect what *would* be aliased without changing
    anything (useful for sanity-checking before committing).

    Acquires a write lock with ``BEGIN IMMEDIATE`` so a concurrent daemon
    write doesn't interleave with our delete-then-add-alias sequence. Without
    this, a busy_timeout retry storm could leave a duplicate's chunks
    half-deleted while the daemon was holding the writer.
    """
    # We need (id, indexed_at) per group to pick the *oldest indexed* row as
    # canonical. Earlier this used MIN(id) as a proxy, but ids only correlate
    # with insertion order, not indexed_at - after a reset / re-embed the
    # truly oldest ingest can have a higher id.
    if not dry_run:
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as e:
            log.warning(
                "dedupe could not acquire writer lock; another writer holds it: %s",
                e,
            )
            return {
                "groups_with_duplicates": 0,
                "duplicate_files_converted": 0,
                "chunks_freed": 0,
                "aliased": [],
            }
    rows = conn.execute(
        "SELECT content_hash, COUNT(*) AS n "
        "FROM files WHERE content_hash IS NOT NULL "
        "GROUP BY content_hash HAVING n > 1"
    ).fetchall()

    converted = 0
    chunks_freed = 0
    aliased: list[tuple[str, str]] = []
    for r in rows:
        chash = r["content_hash"]
        members = conn.execute(
            "SELECT id, path, indexed_at FROM files WHERE content_hash = ? "
            "ORDER BY indexed_at ASC, id ASC",
            (chash,),
        ).fetchall()
        if len(members) < 2:
            continue
        canonical = members[0]
        canonical_id = canonical["id"]
        for dup in members[1:]:
            dup_id = dup["id"]
            chunk_count = conn.execute(
                "SELECT COUNT(*) AS c FROM chunks WHERE file_id = ?", (dup_id,)
            ).fetchone()["c"]
            chunks_freed += chunk_count
            aliased.append((canonical["path"], dup["path"]))
            if dry_run:
                converted += 1
                continue
            # Cascade drops chunks, entities, image rows, and vec_ rows via
            # the explicit cleanup we already do in delete_file - reuse it.
            delete_file(conn, dup["path"])
            add_alias(conn, canonical_id, dup["path"])
            converted += 1
    if dry_run:
        # No commit / rollback needed - we never wrote anything.
        pass
    else:
        conn.commit()
    return {
        "groups_with_duplicates": len(rows),
        "duplicate_files_converted": converted,
        "chunks_freed": chunks_freed,
        "aliased": aliased,
    }


_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 second-brain/0.0.1"
)


# Hard ceiling on how much of a URL response we'll buffer to disk. Without
# this, a hostile or misconfigured server serving Content-Length: 100 GB
# blows out the disk. 200 MB is comfortably above any document we'd want
# in a personal knowledge base.
_URL_FETCH_MAX_BYTES = 200 * 1024 * 1024


class UnsafeURLError(ValueError):
    """Raised when a URL targets a private / loopback / link-local
    address that would let SSRF reach internal services. See
    ``_assert_url_is_public`` for the policy."""


def _assert_url_is_public(url: str) -> None:
    """Round 14 (audit-found gap H2) — SSRF guard for URL ingestion.

    The previous version called ``requests.get(url, allow_redirects=True)``
    on any user-supplied string. That meant anyone who could hand the
    user a link (or reach the localhost dashboard / MCP server)
    could pull:
      - ``http://169.254.169.254/...`` — cloud-metadata IMDS endpoints
      - ``http://127.0.0.1:<port>/...`` — other services on the host
      - ``http://10.x.x.x/...``, ``http://192.168.x.x/...`` — intranet
      - ``file://`` — local filesystem read

    This function rejects those and ``raise``s ``UnsafeURLError``.
    The scheme allowlist is ``http``/``https`` only; the host (or each
    DNS-resolved IP) must be public per the IPv4/IPv6 ``is_private``,
    ``is_loopback``, ``is_link_local``, ``is_reserved``, ``is_multicast``
    rules. Resolution is done up-front so we can re-validate on each
    redirect at the call site.
    """
    import ipaddress
    import socket
    from urllib.parse import urlparse

    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise UnsafeURLError(
            f"refusing scheme {scheme!r} (only http/https allowed)"
        )
    host = parsed.hostname
    if not host:
        raise UnsafeURLError("URL has no host")
    # Reject literal ``localhost`` / common loopback aliases up front;
    # they normally resolve to 127.0.0.1 but on quirky hosts file setups
    # could resolve elsewhere — we don't want to allow the *intent*.
    if host.lower() in ("localhost", "ip6-localhost", "ip6-loopback"):
        raise UnsafeURLError(f"refusing loopback host {host!r}")
    # If the host is already an IP literal, validate it directly.
    try:
        ip_obj = ipaddress.ip_address(host)
    except ValueError:
        ip_obj = None
    candidates: list[ipaddress._BaseAddress] = []
    if ip_obj is not None:
        candidates.append(ip_obj)
    else:
        # Resolve every A/AAAA record so a multi-record DNS poisoning
        # attempt (one public, one private) doesn't slip through.
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError as e:
            raise UnsafeURLError(f"could not resolve {host!r}: {e}") from e
        for info in infos:
            sockaddr = info[4]
            try:
                candidates.append(ipaddress.ip_address(sockaddr[0]))
            except (ValueError, IndexError):
                continue
        if not candidates:
            raise UnsafeURLError(f"no IPs resolved for {host!r}")
    for ip in candidates:
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise UnsafeURLError(
                f"refusing private/loopback IP {ip} for host {host!r}"
            )


def _fetch_url_to_tempfile(url: str) -> tuple[str, str]:
    """Download a URL with a real user-agent so sites like Wikipedia don't 403 us.

    Streams the response with a hard size cap. Returns (local_path,
    content_type). Caller must clean up the temp file.

    Round 14: SSRF-protected. Pre-flight ``_assert_url_is_public``
    rejects file://, loopback, RFC1918, link-local, and cloud-metadata
    addresses. Redirects are disabled at the requests layer so we
    can re-validate each hop ourselves up to ``_REDIRECT_MAX``.
    """
    import os
    import tempfile
    from urllib.parse import urljoin, urlparse

    import requests

    _assert_url_is_public(url)
    _REDIRECT_MAX = 5
    current = url
    resp = None
    for _ in range(_REDIRECT_MAX + 1):
        resp = requests.get(
            current,
            headers={"User-Agent": _USER_AGENT, "Accept": "*/*"},
            timeout=60,
            allow_redirects=False,
            stream=True,
        )
        # 3xx with a Location header → re-validate then loop.
        if 300 <= resp.status_code < 400:
            loc = resp.headers.get("location")
            resp.close()
            if not loc:
                raise UnsafeURLError(
                    f"{current} returned {resp.status_code} with no Location"
                )
            # Resolve relative redirects against the current URL.
            current = urljoin(current, loc)
            _assert_url_is_public(current)
            continue
        break
    else:
        if resp is not None:
            resp.close()
        raise UnsafeURLError(
            f"too many redirects (>{_REDIRECT_MAX}) starting at {url}"
        )
    assert resp is not None
    resp.raise_for_status()
    # Honor Content-Length up front - cheaper than streaming and bailing.
    declared = resp.headers.get("content-length")
    if declared:
        try:
            if int(declared) > _URL_FETCH_MAX_BYTES:
                resp.close()
                raise RuntimeError(
                    f"refusing to download {declared} bytes from {url} "
                    f"(cap {_URL_FETCH_MAX_BYTES})"
                )
        except ValueError:
            pass
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
    written = 0
    try:
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            if not chunk:
                continue
            written += len(chunk)
            if written > _URL_FETCH_MAX_BYTES:
                os.close(fd)
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                resp.close()
                raise RuntimeError(
                    f"response from {url} exceeded {_URL_FETCH_MAX_BYTES} bytes; aborted"
                )
            os.write(fd, chunk)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        resp.close()
    return tmp, ctype


def index_text(
    conn: sqlite3.Connection,
    embedder: Embedder,
    cfg: Config,
    virtual_path: str,
    title: str,
    content: str,
    mtime: float,
    kind: str = "url",
    source: str = "",
    entity_extractor: EntityExtractor | None = None,
) -> IndexResult:
    """Index a pre-extracted document from a connector or other non-filesystem source.

    Parallels ``index_url`` but takes the body in-memory instead of fetching it.
    Honors the same hash-dedup, contextual-prefix, and entity-extraction paths
    as filesystem ingest, so connector-sourced documents become first-class
    citizens of the index alongside files.
    """
    label_path = Path(virtual_path)
    text = (content or "").strip()
    if not text:
        return IndexResult(label_path, "skipped", reason="empty content")

    # Salt the hash with the source (e.g. "reddit", "linear") so a connector
    # doc never accidentally aliases to an unrelated local file that happens
    # to share text. Cross-source dedup is too coarse: a Reddit selftext and
    # a downloaded PDF with the same body are different *things* the user
    # might query for, even if their text is identical. Same-source dedup
    # (Reddit re-fetched, the same Linear issue at a new URL) still works.
    src_for_hash = source or _virtual_path_scheme(virtual_path)
    chash = hashlib.sha1(
        f"{src_for_hash}\n{text}".encode("utf-8", errors="replace")
    ).hexdigest()
    existing = get_file_by_path(conn, virtual_path)
    if existing and existing["content_hash"] == chash:
        return IndexResult(label_path, "unchanged")

    if existing is None:
        twin = find_file_by_hash(conn, chash)
        if twin is not None:
            add_alias(conn, twin["id"], virtual_path)
            conn.commit()
            return IndexResult(label_path, "alias", reason=f"duplicate of {twin['path']}")

    chunked = chunk_text(text, target_size=cfg.chunk_size, overlap=cfg.chunk_overlap)
    if not chunked:
        return IndexResult(label_path, "skipped", reason="no chunks produced")

    chunk_texts = [c for c, _ in chunked]
    chunk_offsets = [off for _, off in chunked]
    src_label = source or kind
    contextualized = [
        f"Source: {src_label}\nTitle: {title}\nPath: {virtual_path}\n\n{c}"
        for c in chunk_texts
    ]
    try:
        embeddings = embedder.embed_documents(contextualized)
    except Exception as e:
        log.warning("embedding failed for %s: %s", virtual_path, e)
        return IndexResult(label_path, "error", reason=f"embedding: {e}")

    file_id = upsert_file(
        conn,
        path=virtual_path,
        mtime=mtime,
        size=len(text.encode("utf-8", errors="replace")),
        kind=kind,
        content_hash=chash,
    )
    chunk_ids = replace_chunks(
        conn, file_id,
        list(zip(chunk_texts, embeddings, chunk_offsets, strict=True)),
    )

    _run_entity_extraction(
        conn, chunk_ids, chunk_texts, entity_extractor, label=virtual_path
    )

    if chunk_ids:
        _link_after_index(conn, file_id)

    conn.commit()
    return IndexResult(label_path, "indexed", chunks=len(chunk_texts))


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

    # Hash-dedup against existing primaries. The same article fetched via
    # mobile vs. desktop URL, or with vs. without ?utm_*, otherwise produces
    # two embedded copies. Match -> alias the new URL onto the existing row.
    if existing is None:
        twin = find_file_by_hash(conn, chash)
        if twin is not None:
            add_alias(conn, twin["id"], url)
            conn.commit()
            return IndexResult(
                label_path, "alias", reason=f"duplicate of {twin['path']}"
            )

    chunked = chunk_text(text, target_size=cfg.chunk_size, overlap=cfg.chunk_overlap)
    if not chunked:
        return IndexResult(label_path, "skipped", reason="no chunks produced")

    chunk_texts = [c for c, _ in chunked]
    chunk_offsets = [off for _, off in chunked]
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
        conn, file_id,
        list(zip(chunk_texts, embeddings, chunk_offsets, strict=True)),
    )

    _run_entity_extraction(conn, chunk_ids, chunk_texts, entity_extractor, label=url)

    if chunk_ids:
        _link_after_index(conn, file_id)

    conn.commit()
    return IndexResult(label_path, "indexed", chunks=len(chunk_texts))


def _run_entity_extraction(
    conn: sqlite3.Connection,
    chunk_ids: list[int],
    chunk_texts: list[str],
    entity_extractor: EntityExtractor | None,
    label: str,
) -> None:
    """Run NER over each chunk and insert results. Best-effort; failure logs
    but never aborts indexing - having chunks without entities is fine, but
    losing the chunk because NER threw is a much worse outcome.

    Extracted to a helper to dry up the three near-identical try/except
    blocks across index_text / index_url / index_file. Drift between those
    copies was a real risk.
    """
    if entity_extractor is None:
        return
    try:
        for chunk_id, chunk_text_val in zip(chunk_ids, chunk_texts, strict=True):
            ents = entity_extractor.extract(strip_markdown_decorations(chunk_text_val))
            if ents:
                insert_entities(
                    conn, chunk_id, [(e.text, e.label) for e in ents]
                )
    except Exception as e:
        log.warning("entity extraction failed for %s: %s", label, e)


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
    image_embedder: ImageEmbedder | None = None,
) -> dict[str, int]:
    """Walk a folder and index all eligible files. Returns counts."""
    counts = {"indexed": 0, "skipped": 0, "unchanged": 0, "error": 0, "alias": 0}
    for p in walk_folder(folder, cfg):
        result = index_file(
            conn, embedder, cfg, p,
            transcriber=transcriber, ocr_engine=ocr_engine,
            entity_extractor=entity_extractor,
            image_embedder=image_embedder,
        )
        counts[result.status] = counts.get(result.status, 0) + 1
        if progress:
            progress(result)
    return counts
