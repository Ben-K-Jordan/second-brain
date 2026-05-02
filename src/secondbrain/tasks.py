"""Phase 47: tasks as first-class.

Action items in transcripts have been rendered into doc bodies since
Phase 43 (Granola action-items extraction). The daily brief grepped
`- [ ]` checkboxes off chunks. Both work — until you actually want to
*do* one of them. Then you need:

  - Persistent identity (so finishing one removes it from tomorrow's
    brief instead of getting re-extracted)
  - A way to mark complete (`tasks done <id>`)
  - A place to add ad-hoc tasks ("remind me to email Sarah") that
    didn't come from a transcript
  - Eventually, two-way sync with Apple Reminders / Todoist

This module owns the data model and the extraction. Sync stubs live
adjacent (`tasks_apple.py` / `tasks_todoist.py`) — kept as future work
unless the user asks. For now: local source of truth, with the
dataclass shape ready for later sync.

Extraction recognises two patterns in transcript-shaped docs:

  A. ``## Action items`` heading → bullet list (plain ``- text`` or
     checkbox ``- [ ] text``). Used by the IMAP transcript renderer
     since Phase 43.
  B. Bare ``- [ ] text`` checkboxes anywhere in the doc. Used by
     manual notes that include todos.

Closed checkboxes (``- [x]`` / ``- [X]``) are skipped — already done.

Materialisation is idempotent: ``materialize_from_transcripts`` reads
recent transcript chunks and INSERT-OR-IGNOREs into ``tasks``. The
UNIQUE on ``(text_lower, source_path)`` means re-running it is free.
The daily brief calls this before reading open tasks, so newly-ingested
transcripts surface their action items the next time you check in.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass

log = logging.getLogger(__name__)


# ---- Tunables ---------------------------------------------------------

# How far back to scan when materialising. Two weeks: keeps re-extract
# cost bounded once the table grows. Tasks are unique by text+path, so
# slipping the window doesn't double-insert.
_DEFAULT_LOOKBACK_DAYS = 14

# Action-items section pattern. Pulls everything from `## Action items`
# (or `### Action items`) up to the next H1/H2/H3 or end-of-string.
# Case-insensitive on the heading text since some sources use title-case
# ("Action Items"), some lower ("action items"). DOTALL so `.` spans
# newlines inside the captured block.
_ACTION_SECTION_RE = re.compile(
    r"^#{2,3}\s*Action\s*Items?\s*\n(.*?)(?=^#{1,3}\s|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)

# Bullet/checkbox line within a section: optional checkbox, mandatory
# text. Use ``[ \t]*`` instead of ``\s*`` between bullet and text so the
# regex can't bleed across newlines and grab the next line's content as
# the "text" of an otherwise-empty bullet. ``\s*$`` at the end *is*
# fine because ``$`` in MULTILINE pins to the same line.
_BULLET_RE = re.compile(
    r"^[ \t]*[-*][ \t]+(?:\[(?P<mark>[ xX])\][ \t]+)?(?P<text>.+?)[ \t]*$",
    re.MULTILINE,
)

# Loose bare-checkbox pattern for pattern (B): `- [ ] text` anywhere,
# regardless of section. Matches only OPEN ones (`[ ]``). Same
# horizontal-whitespace rule as above.
_OPEN_CHECKBOX_RE = re.compile(
    r"^[ \t]*[-*][ \t]+\[\s\][ \t]+(.+?)[ \t]*$",
    re.MULTILINE,
)


# ---- Data shape -------------------------------------------------------

@dataclass
class Task:
    """A single open/done task.

    ``source_path`` of ``'manual'`` means user-added (no back-reference);
    anything else is a virtual path that can be clicked back to the
    originating doc."""
    id: int
    text: str
    source_path: str
    source_title: str
    status: str            # 'open' | 'done' | 'cancelled'
    created_at: float
    completed_at: float | None
    due_at: float | None


# ============================ extraction ==============================

def extract_candidates_from_text(
    text: str,
) -> Iterator[str]:
    """Yield candidate task strings from one chunk of doc text.

    Tries pattern A (Action items section) first; falls back to
    pattern B (bare open checkboxes) for everything that didn't get
    captured. Yields each candidate text exactly once per chunk —
    de-duplication across chunks happens at insert time via the UNIQUE
    constraint.
    """
    seen: set[str] = set()

    # Pattern A: `## Action items` blocks. Iterate every match in the
    # text — long meetings sometimes have both a "Decisions" heading
    # and an "Action items" heading, and rarely repeat the section.
    for sec_m in _ACTION_SECTION_RE.finditer(text):
        block = sec_m.group(1)
        for line_m in _BULLET_RE.finditer(block):
            mark = line_m.group("mark")
            # Closed checkboxes ('x' / 'X') are already done — skip.
            if mark and mark.lower() == "x":
                continue
            t = (line_m.group("text") or "").strip()
            if not t:
                continue
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            yield t

    # Pattern B: bare `- [ ] text` outside a section. Don't re-yield
    # anything already captured by pattern A.
    for chk_m in _OPEN_CHECKBOX_RE.finditer(text):
        t = (chk_m.group(1) or "").strip()
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        yield t


# ============================ persistence =============================

def add_manual(
    conn: sqlite3.Connection, text: str, due_at: float | None = None,
) -> int | None:
    """Add a user-typed task. Returns the id (or None when the same
    text is already in the manual bucket — UNIQUE constraint).

    We use a synthetic source_path of 'manual' for ad-hoc tasks so the
    UNIQUE on (text_lower, source_path) still does what we want — same
    text from a meeting + same text typed manually are intentionally
    treated as separate tasks."""
    text = (text or "").strip()
    if not text:
        return None
    cur = conn.execute(
        "INSERT OR IGNORE INTO tasks"
        "(text, text_lower, source_path, source_title, status, "
        " created_at, due_at) "
        "VALUES (?, ?, 'manual', '(typed)', 'open', ?, ?)",
        (text, text.lower(), time.time(), due_at),
    )
    conn.commit()
    if cur.rowcount == 0:
        # Already exists. Look up + return the existing id so the
        # caller can still navigate to it.
        row = conn.execute(
            "SELECT id FROM tasks WHERE text_lower = ? AND source_path = 'manual'",
            (text.lower(),),
        ).fetchone()
        return int(row["id"]) if row else None
    return int(cur.lastrowid)


def insert_extracted(
    conn: sqlite3.Connection, *, text: str, source_path: str,
    source_title: str,
) -> int | None:
    """Insert a task discovered via extraction. INSERT-OR-IGNORE so
    re-running the extractor is idempotent. Returns the new id, or
    None if the task was already in the table."""
    text = (text or "").strip()
    if not text:
        return None
    cur = conn.execute(
        "INSERT OR IGNORE INTO tasks"
        "(text, text_lower, source_path, source_title, status, created_at) "
        "VALUES (?, ?, ?, ?, 'open', ?)",
        (text, text.lower(), source_path, source_title or source_path,
         time.time()),
    )
    if cur.rowcount == 0:
        return None
    conn.commit()
    return int(cur.lastrowid)


def mark_done(conn: sqlite3.Connection, task_id: int) -> bool:
    """Mark a task complete. Returns True if it changed; False if the
    id didn't exist or was already done."""
    cur = conn.execute(
        "UPDATE tasks SET status = 'done', completed_at = ? "
        "WHERE id = ? AND status != 'done'",
        (time.time(), task_id),
    )
    conn.commit()
    return cur.rowcount > 0


def mark_cancelled(conn: sqlite3.Connection, task_id: int) -> bool:
    """Mark a task cancelled (without finishing). Same shape as
    mark_done; useful for action items that turned out to be moot."""
    cur = conn.execute(
        "UPDATE tasks SET status = 'cancelled', completed_at = ? "
        "WHERE id = ? AND status != 'cancelled'",
        (time.time(), task_id),
    )
    conn.commit()
    return cur.rowcount > 0


def delete(conn: sqlite3.Connection, task_id: int) -> bool:
    """Hard-delete a task. Used by `tasks rm` for typos in manual adds."""
    cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    return cur.rowcount > 0


def list_open(
    conn: sqlite3.Connection, limit: int = 50,
) -> list[Task]:
    """Open tasks, newest first."""
    rows = conn.execute(
        "SELECT * FROM tasks WHERE status = 'open' "
        "ORDER BY created_at DESC, id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_row_to_task(r) for r in rows]


def list_recent_done(
    conn: sqlite3.Connection, limit: int = 20,
) -> list[Task]:
    """Recently-completed tasks. Useful for "what did I get done?"
    week-in-review surfaces."""
    rows = conn.execute(
        "SELECT * FROM tasks WHERE status = 'done' "
        "AND completed_at IS NOT NULL "
        "ORDER BY completed_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_row_to_task(r) for r in rows]


def get(conn: sqlite3.Connection, task_id: int) -> Task | None:
    row = conn.execute(
        "SELECT * FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    return _row_to_task(row) if row else None


def _row_to_task(row: sqlite3.Row) -> Task:
    return Task(
        id=int(row["id"]),
        text=row["text"],
        source_path=row["source_path"],
        source_title=row["source_title"] or row["source_path"],
        status=row["status"],
        created_at=row["created_at"],
        completed_at=row["completed_at"],
        due_at=row["due_at"],
    )


# ============================ materialisation =========================

def materialize_from_transcripts(
    conn: sqlite3.Connection,
    *,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
) -> int:
    """Scan recent transcript-shaped docs and insert any open action
    items into ``tasks``. Returns the count newly inserted.

    Idempotent — re-running over the same window is a no-op once
    everything's been captured. Cheap enough that the daily brief
    calls it on every render.
    """
    cutoff = time.time() - lookback_days * 86400
    rows = conn.execute(
        "SELECT f.id AS fid, f.path AS path, c.text AS text "
        "FROM chunks c JOIN files f ON f.id = c.file_id "
        "WHERE f.path LIKE 'transcript://%' "
        "  AND f.indexed_at >= ? "
        "ORDER BY f.indexed_at DESC, f.id DESC, c.chunk_index ASC",
        (cutoff,),
    ).fetchall()
    inserted = 0
    title_cache: dict[int, str] = {}
    for r in rows:
        fid = r["fid"]
        if fid not in title_cache:
            title_cache[fid] = _doc_title(conn, fid, r["path"])
        title = title_cache[fid]
        for cand in extract_candidates_from_text(r["text"] or ""):
            tid = insert_extracted(
                conn,
                text=cand,
                source_path=r["path"],
                source_title=title,
            )
            if tid is not None:
                inserted += 1
    if inserted:
        log.info("tasks: materialised %d new tasks from transcripts", inserted)
    return inserted


def _doc_title(
    conn: sqlite3.Connection, file_id: int, path: str,
) -> str:
    """First-chunk H1 → fall back to path. Same shape as daily_brief's
    helper but kept here so tasks doesn't depend on daily_brief."""
    row = conn.execute(
        "SELECT text FROM chunks WHERE file_id = ? "
        "ORDER BY chunk_index ASC LIMIT 1",
        (file_id,),
    ).fetchone()
    if row is None:
        return path
    for line in (row["text"] or "").splitlines():
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip() or path
    return path


# ============================ rendering ==============================

def format_task_line(task: Task) -> str:
    """One-liner for CLI listings."""
    head = f"#{task.id}"
    if task.status == "done":
        head = f"#{task.id} ✓"
    elif task.status == "cancelled":
        head = f"#{task.id} ✗"
    src = ""
    if task.source_path != "manual":
        src = f"  ({task.source_title})"
    return f"{head} {task.text}{src}"
