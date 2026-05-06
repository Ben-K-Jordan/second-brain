"""Phase 77: Obsidian / Markdown vault export.

Snapshots your brain to a portable Markdown vault. If the SQLite
brain ever fails or you outgrow this tool, you still have your
content in a format every Markdown editor reads natively.

Layout: ``<vault_root>/`` mirrors the brain by source kind:
    notes/        — files indexed from watched_folders
    transcripts/  — voice / Plaud / Granola transcripts
    captures/     — Phase 69-70 capture://* docs
    canvas/       — class assignments / announcements / syllabi
    health/       — Oura daily summaries
    reviews/      — Phase 72 weekly reviews
    misc/         — anything else

Each file:
  - YAML frontmatter with path, kind, source, mtime, indexed_at
  - The original chunk text concatenated
  - Backlinks (Phase 52) rendered as a `## Related` section with
    `[[wikilinks]]` to the other vault filename
  - Person mentions (Phase 65) rendered as `[[people/Sarah Chen]]`
    inline in a `## People` section

Idempotent: each file's path is derived from its virtual_path so
re-running overwrites cleanly. Use ``--clean`` to wipe the vault
before re-snapshotting.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)


# Path-character normalisation: filesystems disagree on what's legal.
# Strip these to underscore on the way out.
_BAD_PATH_CHARS = re.compile(r'[<>:"|?*\x00-\x1f\\]')


@dataclass
class ExportResult:
    files_written: int = 0
    bytes_written: int = 0
    errors: int = 0
    vault_root: str = ""


def _slug(text: str, *, max_len: int = 80) -> str:
    """Sanitise a string for use as a filesystem path component."""
    text = _BAD_PATH_CHARS.sub("_", text or "")
    text = text.replace("/", "_").strip().strip(".")
    if not text:
        text = "untitled"
    if len(text) > max_len:
        text = text[:max_len].rstrip()
    return text


def _classify(path: str, kind: str) -> str:
    """Map a virtual path / kind to one of our seven vault folders.

    Round 26 fix (audit-found gap M5) — added explicit handlers for
    every connector prefix the indexer emits so connector docs land in
    sensible folders instead of ``misc``. Email-shaped sources route
    to ``notes`` (treated as documents); journal entries get their own
    bucket; chat / project-tracker connectors flow into ``captures``
    so they're co-located with capture:// rapid-fire notes.
    """
    if path.startswith("transcript://") or path.startswith("voice://"):
        return "transcripts"
    if path.startswith("capture://"):
        return "captures"
    if path.startswith("canvas://"):
        return "canvas"
    if path.startswith("oura://"):
        return "health"
    if path.startswith("review://"):
        return "reviews"
    if path.startswith("journal://"):
        return "journal"
    if path.startswith(("imap://", "gmail://", "message://")):
        return "notes"
    if path.startswith((
        "slack://", "linear://", "github://",
        "notion://", "obsidian://", "pocket://", "readwise://",
    )):
        return "captures"
    if (path.startswith(("/", "C:\\", "C:/", "~"))
            or kind in ("document", "code", "image")):
        return "notes"
    return "misc"


def _vault_path(
    root: Path, file_path: str, title: str, kind: str,
) -> Path:
    """Pick a stable relative path for a brain doc inside the vault."""
    folder = _classify(file_path, kind)
    # For real filesystem paths, preserve a slug of the basename.
    # For virtual paths, slug the title.
    if file_path.startswith(("/", "C:\\", "C:/", "~")):
        base = Path(file_path).stem or "untitled"
    else:
        # virtual path → use the trailing component
        tail = file_path.split("/")[-1] if "/" in file_path else file_path
        base = title or tail or "untitled"
    return root / folder / f"{_slug(base)}.md"


def _hydrate_doc(
    conn: sqlite3.Connection, file_id: int,
) -> tuple[str, list[str]]:
    """Return (full_body_text, ordered_chunk_texts) for a file."""
    rows = conn.execute(
        "SELECT text FROM chunks WHERE file_id = ? "
        "ORDER BY chunk_index ASC",
        (file_id,),
    ).fetchall()
    chunks = [r["text"] or "" for r in rows]
    return "\n\n".join(chunks), chunks


def _doc_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip() or fallback
        if s:
            return fallback
    return fallback


def _backlinks_for(
    conn: sqlite3.Connection, file_id: int, *, limit: int = 8,
) -> list[tuple[str, str]]:
    """[(other_path, other_title), ...] sorted by similarity."""
    try:
        rows = conn.execute(
            "SELECT b.dst_file_id, f.path "
            "FROM backlinks b JOIN files f ON f.id = b.dst_file_id "
            "WHERE b.src_file_id = ? "
            "ORDER BY b.score ASC LIMIT ?",
            (file_id, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    out = []
    for r in rows:
        title_row = conn.execute(
            "SELECT text FROM chunks WHERE file_id = ? "
            "ORDER BY chunk_index ASC LIMIT 1",
            (int(r["dst_file_id"]),),
        ).fetchone()
        title = (
            _doc_title(title_row["text"] or "", r["path"])
            if title_row else r["path"]
        )
        out.append((r["path"], title))
    return out


def _people_for(
    conn: sqlite3.Connection, file_id: int,
) -> list[str]:
    """Display names of people mentioned in this file."""
    try:
        rows = conn.execute(
            "SELECT DISTINCT p.display_name FROM person_mentions m "
            "JOIN people p ON p.id = m.person_id "
            "WHERE m.file_id = ? "
            "ORDER BY p.display_name",
            (file_id,),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [r["display_name"] for r in rows]


def _wikilink(target_path: str, target_title: str) -> str:
    """Render [[wikilink]] using the vault filename, not the brain
    path. Obsidian resolves [[Title]] across folders, so the filename
    stem is the right anchor."""
    return f"[[{_slug(target_title or target_path)}]]"


def _frontmatter(
    *, path: str, kind: str, source: str | None,
    mtime: float, indexed_at: float,
) -> str:
    def _iso(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ",
        )

    lines = [
        "---",
        f"path: {path}",
        f"kind: {kind}",
    ]
    if source:
        lines.append(f"source: {source}")
    lines.append(f"mtime: {_iso(mtime)}")
    lines.append(f"indexed_at: {_iso(indexed_at)}")
    lines.append("---")
    return "\n".join(lines)


def export_vault(
    conn: sqlite3.Connection, vault_root: Path,
    *, clean: bool = False, limit: int | None = None,
) -> ExportResult:
    """Write every file in the brain to ``vault_root`` as Markdown
    with frontmatter + backlinks + people mentions.

    ``clean=True`` wipes the vault first (useful when paths shifted
    between exports). ``limit`` caps for testing.
    """
    vault_root.mkdir(parents=True, exist_ok=True)
    if clean:
        # Best-effort wipe — only delete files in our known subdirs
        # so we don't nuke an Obsidian vault that has unrelated stuff.
        for sub in (
            "notes", "transcripts", "captures", "canvas",
            "health", "reviews", "misc",
        ):
            sub_path = vault_root / sub
            if sub_path.exists():
                for p in sub_path.rglob("*.md"):
                    try:
                        p.unlink()
                    except OSError:
                        pass

    rows = conn.execute(
        "SELECT id, path, kind, mtime, indexed_at, content_hash "
        "FROM files ORDER BY id ASC "
        + (f"LIMIT {int(limit)}" if limit is not None else ""),
    ).fetchall()

    result = ExportResult(vault_root=str(vault_root))
    # Track filename collisions: same _slug for different paths means
    # we suffix the second one.
    used_paths: set[Path] = set()
    for r in rows:
        try:
            body, _chunks = _hydrate_doc(conn, int(r["id"]))
        except sqlite3.OperationalError as e:
            log.warning("vault export: hydrate failed for %s: %s",
                        r["path"], e)
            result.errors += 1
            continue
        if not body.strip():
            continue
        title = _doc_title(body, r["path"])
        target = _vault_path(vault_root, r["path"], title, r["kind"])
        # Collision suffix.
        original_target = target
        attempts = 1
        while target in used_paths:
            attempts += 1
            target = original_target.with_stem(
                f"{original_target.stem}-{attempts}",
            )
        used_paths.add(target)

        # Compose final markdown.
        parts = [
            _frontmatter(
                path=r["path"], kind=r["kind"], source=None,
                mtime=r["mtime"], indexed_at=r["indexed_at"],
            ),
            "",
            body.strip(),
        ]
        people = _people_for(conn, int(r["id"]))
        if people:
            parts.extend([
                "",
                "## People",
                "",
                ", ".join(_wikilink(f"people/{n}", n) for n in people),
            ])
        backs = _backlinks_for(conn, int(r["id"]))
        if backs:
            parts.extend(["", "## Related", ""])
            for path_, title_ in backs:
                parts.append(f"- {_wikilink(path_, title_)}")

        content = "\n".join(parts) + "\n"
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.write_text(content, encoding="utf-8")
        except OSError as e:
            log.warning("vault export: write failed for %s: %s",
                        target, e)
            result.errors += 1
            continue
        result.files_written += 1
        result.bytes_written += len(content.encode("utf-8"))
    return result
