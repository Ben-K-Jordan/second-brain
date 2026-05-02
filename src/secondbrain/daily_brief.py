"""Phase 44: daily brief aggregator — your morning at a glance.

The other ``briefing.py`` modules synthesise *one thing* via Claude:

  - ``briefing.py`` → "what entered the brain" (free-form summary)
  - ``event_briefing.py`` → "what to know before this meeting" (per-event)

This module is different. It's a deterministic *aggregator* that pulls
together everything the rest of the system already knows about today
into one Markdown brief. No LLM call required for the assembly itself —
all the data already exists; we just collate.

Sections (any empty section is omitted):

  1. **Today's calendar** — events from now → end of day in your local TZ.
     Pulled live from Google Calendar + ICS via the same helpers
     ``event_briefing.py`` uses, so the brief reflects last-minute
     additions even if the connector hasn't sync'd yet.

  2. **Class assignments due soon** — Canvas assignments in the next 72h.
     Ranks by due date. Reads ``files`` rows by ``canvas://assignment/``
     path prefix; ``mtime`` is the assignment's due_at, so a simple
     range filter works.

  3. **Open action items** — Markdown ``- [ ]`` checkboxes from recent
     transcript ingestion (Granola / generic meeting notes — see
     ``transcripts.py``). Caps the list so a chatty meeting doesn't
     dominate the brief.

  4. **Reading queue** — top N unread items with their auto-summaries
     (Phase 40). Just lifts ``reading_queue_unread``.

  5. **Watchlist highlights** — finished runs in the last 24h that
     produced *new* items (``new_count > 0``). For each, names the
     watchlist + 1-2 sample new paths.

The output renders to Markdown (good for email + dashboard) and to a
Rich-tinted plaintext (CLI). When the daemon hook ships, the same
markdown render goes into an email send and a tray notification.

Cost: $0. Pure aggregation — Claude is only invoked if you opt into
the "polish this" pass via ``--polish`` (uses Haiku 4.5 for ~$0.001).
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from .config import Config

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


# ---- Tunables ---------------------------------------------------------

# How far ahead to surface Canvas assignments. 72h gives you the natural
# "this week" horizon without burying tomorrow's pop quiz under next
# Friday's project deadline.
_ASSIGNMENT_LOOKAHEAD_SECONDS = 72 * 3600
# Action items get pulled from transcripts ingested in the last N days.
# Two weeks: long enough to catch "I'll do X next week" promises that
# haven't been ticked, short enough that ancient unchecked items don't
# drown out fresh ones.
_ACTION_ITEM_LOOKBACK_DAYS = 14
# Cap items per section so the brief stays scannable.
_ACTION_ITEM_MAX = 10
_QUEUE_MAX = 5
_WATCHLIST_HIGHLIGHTS_MAX = 5
_WATCHLIST_NEW_PATHS_PER_RUN = 2

# Markdown checkbox pattern. Allowed leading bullets: -, *, plus optional
# whitespace. Matches both `- [ ]` (open) — open ones are what we surface.
# Closed ones (`- [x]`) are ignored as already done.
_CHECKBOX_RE = re.compile(r"^\s*[-*]\s*\[\s\]\s+(.+?)\s*$", re.MULTILINE)


# ---- Data shapes ------------------------------------------------------

@dataclass
class Assignment:
    """Canvas assignment surfaced in the brief."""
    title: str            # already includes [course-code] prefix
    due_at: float
    url: str
    path: str             # canvas://assignment/<cid>/<aid> for click-through


@dataclass
class ActionItem:
    """Open todo extracted from a transcript-shaped doc."""
    text: str
    source_path: str      # transcript:// virtual path so user can click back
    source_title: str


@dataclass
class QueueItem:
    """Slim view of a reading_queue row."""
    queue_id: int
    url: str
    title: str
    summary: str          # may be empty if the summariser hasn't run yet


@dataclass
class WatchlistHighlight:
    """A watchlist run that produced new items recently."""
    name: str
    new_count: int
    sample_paths: list[str] = field(default_factory=list)
    finished_at: float = 0.0


@dataclass
class DailyBrief:
    """The whole morning brief in one structured object."""
    generated_at: float
    today_events: list  # list[CalendarEvent] — typed loosely to avoid the import at module load
    assignments_due_soon: list[Assignment]
    open_action_items: list[ActionItem]
    queue_top: list[QueueItem]
    watchlist_highlights: list[WatchlistHighlight]


# ============================ assembly ================================

def assemble_brief(cfg: Config, conn: sqlite3.Connection) -> DailyBrief:
    """Pull every section's data and return a DailyBrief.

    Each section is best-effort — if a calendar source is unreachable or
    Canvas isn't configured, the affected section is empty rather than
    crashing the whole brief.
    """
    return DailyBrief(
        generated_at=time.time(),
        today_events=_today_events(cfg),
        assignments_due_soon=_assignments_due_soon(conn),
        open_action_items=_open_action_items(conn),
        queue_top=_queue_top(conn),
        watchlist_highlights=_watchlist_highlights(conn),
    )


# ---- section: today's calendar ---------------------------------------

def _seconds_until_end_of_local_day(now: float | None = None) -> int:
    """How many seconds remain in the user's local-time day?"""
    n = now if now is not None else time.time()
    local = datetime.fromtimestamp(n)
    end = local.replace(hour=23, minute=59, second=59, microsecond=0)
    return max(60, int(end.timestamp() - n))


def _today_events(cfg: Config) -> list:
    """Calendar events from now → end of local-time day. Best-effort —
    network/auth problems just yield no events."""
    try:
        from .event_briefing import iter_upcoming_events
    except ImportError:
        return []
    horizon = _seconds_until_end_of_local_day()
    try:
        events = list(iter_upcoming_events(cfg, horizon))
    except Exception as e:  # noqa: BLE001
        log.warning("daily brief: calendar fetch failed: %s", e)
        return []
    events.sort(key=lambda ev: ev.starts_at)
    return events


# ---- section: assignments due soon -----------------------------------

def _assignments_due_soon(conn: sqlite3.Connection) -> list[Assignment]:
    """Canvas assignments whose due-at falls in [now, now + 72h].

    The Canvas connector stores due_at as the file's mtime so existing
    indexes don't need a metadata table to surface time-sensitive items.
    """
    now = time.time()
    cutoff = now + _ASSIGNMENT_LOOKAHEAD_SECONDS
    rows = conn.execute(
        "SELECT id, path, mtime FROM files "
        "WHERE path LIKE 'canvas://assignment/%' "
        "  AND mtime >= ? AND mtime <= ? "
        "ORDER BY mtime ASC",
        (now, cutoff),
    ).fetchall()
    out: list[Assignment] = []
    for r in rows:
        title, url = _read_canvas_title_and_url(conn, r["id"], r["path"])
        out.append(Assignment(
            title=title, due_at=r["mtime"], url=url, path=r["path"],
        ))
    return out


_CANVAS_LINK_RE = re.compile(r"^Link:\s*(\S+)", re.MULTILINE)


def _read_canvas_title_and_url(
    conn: sqlite3.Connection, file_id: int, path: str,
) -> tuple[str, str]:
    """Pull the H1 + 'Link:' line from the assignment's first chunk so
    the brief can click through to Canvas. Falls back to the path if
    parsing fails."""
    row = conn.execute(
        "SELECT text FROM chunks WHERE file_id = ? "
        "ORDER BY chunk_index ASC LIMIT 1",
        (file_id,),
    ).fetchone()
    if row is None:
        return (path, "")
    text = row["text"] or ""
    title = ""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("# "):
            title = s[2:].strip()
            break
    url = ""
    m = _CANVAS_LINK_RE.search(text)
    if m:
        url = m.group(1).strip()
    return (title or path, url)


# ---- section: open action items --------------------------------------

def _open_action_items(conn: sqlite3.Connection) -> list[ActionItem]:
    """Pull Markdown ``- [ ]`` checkboxes from recent transcript-shaped
    docs. Caps the list so one long meeting doesn't dominate.

    We only look at ``transcript://`` paths — those are the ones the
    Granola / Plaud / generic transcript pipeline produces with
    structured action-items sections.
    """
    cutoff = time.time() - _ACTION_ITEM_LOOKBACK_DAYS * 86400
    rows = conn.execute(
        "SELECT f.id AS fid, f.path AS path, c.text AS text "
        "FROM chunks c JOIN files f ON f.id = c.file_id "
        "WHERE f.path LIKE 'transcript://%' "
        "  AND f.indexed_at >= ? "
        "  AND c.text LIKE '%[ ]%' "
        "ORDER BY f.indexed_at DESC, f.id DESC, c.chunk_index ASC",
        (cutoff,),
    ).fetchall()
    out: list[ActionItem] = []
    seen: set[tuple[str, str]] = set()  # dedupe by (path, text-lower)
    for r in rows:
        text = r["text"] or ""
        title = _read_doc_title(conn, r["fid"], r["path"])
        for m in _CHECKBOX_RE.finditer(text):
            item = m.group(1).strip()
            if not item:
                continue
            key = (r["path"], item.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(ActionItem(
                text=item, source_path=r["path"], source_title=title,
            ))
            if len(out) >= _ACTION_ITEM_MAX:
                return out
    return out


def _read_doc_title(
    conn: sqlite3.Connection, file_id: int, path: str,
) -> str:
    """Best-effort doc title from the first chunk's H1, falling back to
    the path. Cheap — single row lookup."""
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


# ---- section: reading queue ------------------------------------------

def _queue_top(conn: sqlite3.Connection) -> list[QueueItem]:
    from .db import reading_queue_unread

    rows = reading_queue_unread(conn, limit=_QUEUE_MAX)
    out: list[QueueItem] = []
    for r in rows:
        out.append(QueueItem(
            queue_id=r["id"],
            url=r["url"] or "",
            title=r["title"] or "",
            summary=(r["summary"] or "").strip(),
        ))
    return out


# ---- section: watchlist highlights -----------------------------------

def _watchlist_highlights(conn: sqlite3.Connection) -> list[WatchlistHighlight]:
    """Watchlist runs that finished in the last 24h with new items."""
    cutoff = time.time() - 24 * 3600
    rows = conn.execute(
        "SELECT wr.watchlist_id, wr.finished_at, wr.new_count, "
        "       wr.new_paths_json, w.name "
        "FROM watchlist_runs wr "
        "JOIN watchlists w ON w.id = wr.watchlist_id "
        "WHERE wr.finished_at IS NOT NULL "
        "  AND wr.error IS NULL "
        "  AND wr.finished_at >= ? "
        "  AND wr.new_count > 0 "
        "ORDER BY wr.finished_at DESC LIMIT ?",
        (cutoff, _WATCHLIST_HIGHLIGHTS_MAX),
    ).fetchall()
    out: list[WatchlistHighlight] = []
    for r in rows:
        sample = _parse_new_paths(r["new_paths_json"])
        out.append(WatchlistHighlight(
            name=r["name"] or f"watchlist {r['watchlist_id']}",
            new_count=int(r["new_count"] or 0),
            sample_paths=sample[:_WATCHLIST_NEW_PATHS_PER_RUN],
            finished_at=r["finished_at"] or 0.0,
        ))
    return out


def _parse_new_paths(raw: str | None) -> list[str]:
    """Watchlist runs persist new_paths_json as a JSON list of strings.
    Tolerate malformed/missing — returns an empty list."""
    if not raw:
        return []
    import json
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return [str(x) for x in data if x]


# ============================ rendering ===============================

def format_markdown(brief: DailyBrief, *, header_date: str | None = None) -> str:
    """Render the brief as Markdown — used by email send + dashboard.

    Empty sections are omitted entirely so the brief stays tight on
    sleepy days. ``header_date`` overrides the auto-generated date in
    the H1, useful for tests."""
    date_str = header_date or _format_date(brief.generated_at)
    lines: list[str] = [f"# Daily brief — {date_str}", ""]

    sections = list(_iter_section_blocks(brief))
    if not sections:
        lines.append("Nothing on the calendar, no due assignments, "
                     "no open action items. Quiet day. ☕")
        return "\n".join(lines).rstrip() + "\n"

    for block in sections:
        lines.append(block.rstrip())
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _iter_section_blocks(brief: DailyBrief) -> Iterator[str]:
    """Yield the markdown for each non-empty section, in display order."""
    if brief.today_events:
        yield _render_events(brief.today_events)
    if brief.assignments_due_soon:
        yield _render_assignments(brief.assignments_due_soon)
    if brief.open_action_items:
        yield _render_action_items(brief.open_action_items)
    if brief.queue_top:
        yield _render_queue(brief.queue_top)
    if brief.watchlist_highlights:
        yield _render_watchlist(brief.watchlist_highlights)


def _render_events(events: list) -> str:
    """Render today's calendar block. Each event is a one-liner."""
    out = ["## Today on the calendar", ""]
    for ev in events:
        when = time.strftime("%H:%M", time.localtime(ev.starts_at))
        bits = [f"- **{when}** {ev.title}"]
        if ev.location:
            bits.append(f"_{ev.location}_")
        if ev.attendees:
            atts = ", ".join(ev.attendees[:3])
            extra = "" if len(ev.attendees) <= 3 else f" +{len(ev.attendees) - 3}"
            bits.append(f"with {atts}{extra}")
        out.append(" — ".join(bits))
    return "\n".join(out)


def _render_assignments(items: list[Assignment]) -> str:
    out = ["## Class — due in the next 72h", ""]
    for a in items:
        when = time.strftime("%a %H:%M", time.localtime(a.due_at))
        link = f" — <{a.url}>" if a.url else ""
        out.append(f"- **{when}** {a.title}{link}")
    return "\n".join(out)


def _render_action_items(items: list[ActionItem]) -> str:
    out = ["## Open action items", ""]
    for it in items:
        # Re-render as an unticked checkbox so users can copy-paste
        # straight into Obsidian / a notes app and tick them off.
        out.append(f"- [ ] {it.text}  _(from: {it.source_title})_")
    return "\n".join(out)


def _render_queue(items: list[QueueItem]) -> str:
    out = [f"## Reading queue (top {len(items)})", ""]
    for q in items:
        title = q.title or q.url
        out.append(f"- [{title}]({q.url})")
        if q.summary:
            # Quote the summary so multi-line summaries don't break list
            # nesting in renderers that are strict about indent.
            for line in q.summary.splitlines():
                out.append(f"  > {line}")
    return "\n".join(out)


def _render_watchlist(items: list[WatchlistHighlight]) -> str:
    out = ["## Watchlists — new in the last 24h", ""]
    for h in items:
        out.append(f"- **{h.name}** — {h.new_count} new")
        for p in h.sample_paths:
            out.append(f"  - {p}")
    return "\n".join(out)


def _format_date(ts: float) -> str:
    return time.strftime("%A, %b %-d", time.localtime(ts)) if _supports_dash_d() \
        else time.strftime("%A, %b %d", time.localtime(ts))


def _supports_dash_d() -> bool:
    """Windows strftime doesn't support %-d (no zero pad). Cheap probe."""
    try:
        time.strftime("%-d", time.localtime(0))
        return True
    except (ValueError, TypeError):
        return False


# ============================ entrypoints =============================

def generate_brief_markdown(cfg: Config, conn: sqlite3.Connection) -> str:
    """One-shot helper: assemble + render. The CLI + daemon both call this."""
    return format_markdown(assemble_brief(cfg, conn))
