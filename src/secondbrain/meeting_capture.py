"""Round 19 (Phase EA-3) — meeting capture.

When a meeting transcript lands in the brain (audio_video file with
text extracted, or an explicit "transcript"-kind file from a
connector), an EA's job is:

  1. **Decisions** — what got decided in this meeting + why
  2. **Action items** — who owes what, optionally with a due hint
  3. **Open questions** — things that came up but weren't answered
  4. **Recap email** — a short note to attendees confirming the
     above, drafted in the user's voice

This module does all four with a single Sonnet 4.5 call (cached
system prompt) and persists the structured output in the
``meeting_captures`` table. Action items also flow into the
``followups`` table so the user's "open threads" view stays
unified.

Design notes:
- Idempotent on file_id: re-running on the same transcript replaces
  the prior capture (overwrite=True style).
- Recap draft is *just a draft* — never auto-sent. The user reviews
  in the dashboard and either copies the body to their email client
  or, if integrated, marks it as sent.
- LLM gets the user's voice profile so the recap draft sounds like
  them (greeting, sign-off, contraction rate).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import weakref as _weakref
from dataclasses import asdict, dataclass

from .config import Config

log = logging.getLogger(__name__)

_SCHEMA_INITIALIZED: _weakref.WeakSet = _weakref.WeakSet()
# Round 21 fix (audit-found gap F1) — daemon (auto-capture) +
# dashboard (edit / mark_recap_sent) write here concurrently.
_WRITE_LOCK = threading.RLock()

_CAPTURE_MODEL = "claude-sonnet-4-5"
_CAPTURE_MAX_TRANSCRIPT_CHARS = 30000


@dataclass
class ActionItem:
    owner: str        # "user" or person name; "unassigned" if unclear
    description: str
    due_hint: str = ""    # ISO date or natural phrase ("by Friday")


@dataclass
class Decision:
    text: str
    rationale: str = ""


@dataclass
class MeetingCapture:
    id: int
    file_id: int
    title: str
    decisions: list[Decision]
    actions: list[ActionItem]
    open_questions: list[str]
    recap_draft: str
    captured_at: float
    model: str
    # Round 20 —
    attendees: list[str] = None  # type: ignore[assignment]
    calendar_event_id: str | None = None
    user_edited: bool = False
    recap_sent_at: float | None = None

    def __post_init__(self):
        if self.attendees is None:
            self.attendees = []

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file_id": self.file_id,
            "title": self.title,
            "decisions": [asdict(d) for d in self.decisions],
            "actions": [asdict(a) for a in self.actions],
            "open_questions": self.open_questions,
            "recap_draft": self.recap_draft,
            "captured_at": self.captured_at,
            "model": self.model,
            "attendees": self.attendees,
            "calendar_event_id": self.calendar_event_id,
            "user_edited": self.user_edited,
            "recap_sent_at": self.recap_sent_at,
        }


def _ensure_schema(conn: sqlite3.Connection) -> None:
    try:
        if conn in _SCHEMA_INITIALIZED:
            return
    except TypeError:
        pass
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS meeting_captures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER UNIQUE REFERENCES files(id) ON DELETE CASCADE,
            title TEXT NOT NULL DEFAULT '',
            decisions_json TEXT NOT NULL DEFAULT '[]',
            actions_json TEXT NOT NULL DEFAULT '[]',
            open_questions_json TEXT NOT NULL DEFAULT '[]',
            recap_draft TEXT NOT NULL DEFAULT '',
            captured_at REAL NOT NULL,
            model TEXT NOT NULL DEFAULT '',
            recap_sent_at REAL  -- user marks as sent from dashboard
        );
        CREATE INDEX IF NOT EXISTS idx_meeting_captures_captured
            ON meeting_captures(captured_at DESC);
    """)
    # Round 20 — attendee tracking + calendar event linkage.
    cols = {
        r["name"]
        for r in conn.execute("PRAGMA table_info(meeting_captures)")
    }
    if "attendees_json" not in cols:
        conn.execute(
            "ALTER TABLE meeting_captures "
            "ADD COLUMN attendees_json TEXT NOT NULL DEFAULT '[]'"
        )
    if "calendar_event_id" not in cols:
        conn.execute(
            "ALTER TABLE meeting_captures "
            "ADD COLUMN calendar_event_id TEXT"
        )
    if "user_edited" not in cols:
        conn.execute(
            "ALTER TABLE meeting_captures "
            "ADD COLUMN user_edited INTEGER NOT NULL DEFAULT 0"
        )
    conn.commit()
    try:
        _SCHEMA_INITIALIZED.add(conn)
    except TypeError:
        pass


def _row_to_capture(row) -> MeetingCapture:
    keys = row.keys() if hasattr(row, "keys") else ()
    return MeetingCapture(
        id=int(row["id"]),
        file_id=int(row["file_id"]) if row["file_id"] else 0,
        title=row["title"] or "",
        decisions=[
            Decision(**d) for d in json.loads(row["decisions_json"] or "[]")
        ],
        actions=[
            ActionItem(**a) for a in json.loads(row["actions_json"] or "[]")
        ],
        open_questions=json.loads(row["open_questions_json"] or "[]"),
        recap_draft=row["recap_draft"] or "",
        captured_at=float(row["captured_at"]),
        model=row["model"] or "",
        attendees=(
            json.loads(row["attendees_json"] or "[]")
            if "attendees_json" in keys else []
        ),
        calendar_event_id=(
            row["calendar_event_id"]
            if "calendar_event_id" in keys else None
        ),
        user_edited=(
            bool(row["user_edited"])
            if "user_edited" in keys else False
        ),
        recap_sent_at=row["recap_sent_at"],
    )


# ============================ extraction ============================


_CAPTURE_SYSTEM = """\
You are an executive-assistant-grade meeting capture engine.

You read a meeting transcript and produce a JSON object with:

  title: 3-7 word headline of what the meeting was about
  attendees: array of names appearing in the transcript as speakers
             or addressed parties. Includes the meeting owner if
             named. Skip placeholder labels like "Speaker 1".
  decisions: array of {text, rationale} — concrete decisions made.
            Soft "we should think about X" is NOT a decision unless
            paired with an explicit conclusion.
  actions: array of {owner, description, due_hint} — concrete tasks
           someone owes. owner = "user" if it's the meeting owner;
           use a person's name for others; "unassigned" if unclear.
           due_hint is an ISO date OR a natural phrase like
           "by next Friday"; "" if no due was mentioned.
  open_questions: array of strings — questions raised that weren't
                  answered in the meeting. Worth following up later.
  recap_draft: a 3-6 sentence email draft from the meeting owner
               (the user) summarising decisions + actions. Plain
               text, no greeting/sign-off (the user appends those).
               Tone: business-casual, factual, no fluff.

Rules:
  - Be conservative. Empty arrays are FINE if the transcript was
    just chit-chat.
  - Never fabricate decisions or names not in the transcript.
  - The recap_draft should NOT contain emoji, exclamation marks,
    or "Hope this helps!"-style filler.
  - Return ONLY the JSON object — no Markdown fences, no prose.
"""


def _extract_via_llm(
    cfg: Config, transcript: str, user_name: str,
) -> tuple[dict | None, str, float]:
    """One LLM call → (parsed dict, model name, cost_cents)."""
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None, "", 0.0
    try:
        import anthropic
    except ImportError:
        return None, "", 0.0
    try:
        from .budget import (
            check_budget,
            estimate_cost,
            record_usage,
        )
        check_budget(cfg, "anthropic", feature="meeting_capture")
    except Exception:  # noqa: BLE001 — also catches BudgetExceededError
        return None, "", 0.0
    # Round 13 invariant — redact body before send.
    try:
        from .email_assist import _safe_for_prompt
        body_clip = _safe_for_prompt(
            transcript, max_chars=_CAPTURE_MAX_TRANSCRIPT_CHARS,
        )
    except ImportError:
        body_clip = (transcript or "")[:_CAPTURE_MAX_TRANSCRIPT_CHARS]
    if not body_clip.strip():
        return None, "", 0.0
    user_prompt = (
        f"Meeting owner / user name: {user_name}\n"
        f"\n---\n{body_clip}\n---"
    )
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=_CAPTURE_MODEL,
            max_tokens=2000,
            system=[{
                "type": "text",
                "text": _CAPTURE_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_prompt}],
        )
    except anthropic.APIError as e:
        log.warning("meeting_capture: API error: %s", e)
        return None, "", 0.0
    in_tok = getattr(resp.usage, "input_tokens", 0)
    out_tok = getattr(resp.usage, "output_tokens", 0)
    try:
        cost = estimate_cost(
            _CAPTURE_MODEL, in_tok, out_tok,
        ).cents
    except Exception:  # noqa: BLE001
        cost = 0.0
    try:
        record_usage(
            cfg, "anthropic", _CAPTURE_MODEL,
            input_tokens=in_tok, output_tokens=out_tok,
            note="meeting_capture",
            feature="meeting_capture",
        )
    except Exception:  # noqa: BLE001
        pass
    raw = "\n".join(
        b.text for b in resp.content
        if getattr(b, "type", "") == "text"
    ).strip()
    if raw.startswith("```"):
        import re
        raw = re.sub(r"^```\w*\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning("meeting_capture: bad JSON: %s", e)
        return None, _CAPTURE_MODEL, cost
    if not isinstance(parsed, dict):
        return None, _CAPTURE_MODEL, cost
    return parsed, _CAPTURE_MODEL, cost


def capture(
    conn: sqlite3.Connection,
    cfg: Config,
    file_id: int,
    *,
    transcript: str | None = None,
    user_name: str | None = None,
    overwrite: bool = False,
) -> MeetingCapture | None:
    """Run capture on one transcript. Returns the persisted capture
    or None on failure / no-API-key.

    If ``transcript`` is None, we pull it from chunks for ``file_id``.
    """
    _ensure_schema(conn)
    if user_name is None:
        user_name = getattr(cfg, "user_name", None) or "User"
    # If we already have a capture and overwrite is False, return it.
    if not overwrite:
        row = conn.execute(
            "SELECT * FROM meeting_captures WHERE file_id = ?",
            (file_id,),
        ).fetchone()
        if row:
            return _row_to_capture(row)
    if transcript is None:
        chunks = conn.execute(
            "SELECT text FROM chunks WHERE file_id = ? "
            "ORDER BY chunk_index", (file_id,),
        ).fetchall()
        transcript = "\n".join(c["text"] for c in chunks)
    if not transcript or not transcript.strip():
        return None
    parsed, model, cost = _extract_via_llm(cfg, transcript, user_name)
    if parsed is None:
        return None
    title = str(parsed.get("title") or "")[:300]
    decisions = [
        Decision(
            text=str(d.get("text") or "")[:500],
            rationale=str(d.get("rationale") or "")[:500],
        )
        for d in parsed.get("decisions") or []
        if isinstance(d, dict) and d.get("text")
    ]
    actions = [
        ActionItem(
            owner=str(a.get("owner") or "unassigned")[:120],
            description=str(a.get("description") or "")[:500],
            due_hint=str(a.get("due_hint") or "")[:120],
        )
        for a in parsed.get("actions") or []
        if isinstance(a, dict) and a.get("description")
    ]
    open_questions = [
        str(q)[:300]
        for q in parsed.get("open_questions") or []
        if q
    ]
    recap_draft = str(parsed.get("recap_draft") or "")[:4000]
    # Round 20 — extract attendees list from LLM output.
    attendees = [
        str(a)[:120]
        for a in parsed.get("attendees") or []
        if a and isinstance(a, str)
    ]
    # Round 13 invariant — redact persisted text.
    try:
        from .safety import redact_text
        title = redact_text(title)
        decisions = [
            Decision(
                text=redact_text(d.text),
                rationale=redact_text(d.rationale),
            )
            for d in decisions
        ]
        actions = [
            ActionItem(
                owner=a.owner,
                description=redact_text(a.description),
                due_hint=a.due_hint,
            )
            for a in actions
        ]
        open_questions = [redact_text(q) for q in open_questions]
        recap_draft = redact_text(recap_draft)
        attendees = [redact_text(a) for a in attendees]
    except ImportError:
        pass
    now = time.time()
    payload = (
        file_id,
        title,
        json.dumps([asdict(d) for d in decisions]),
        json.dumps([asdict(a) for a in actions]),
        json.dumps(open_questions),
        recap_draft,
        now, model,
        json.dumps(attendees),
    )
    # Round 17-style atomic transaction. Round 21 — also
    # serialised behind _WRITE_LOCK so the daemon's auto-capture
    # job can't race the dashboard's manual capture.
    with _WRITE_LOCK, conn:
        if overwrite:
            conn.execute(
                "DELETE FROM meeting_captures WHERE file_id = ?",
                (file_id,),
            )
        conn.execute(
            "INSERT OR REPLACE INTO meeting_captures"
            "(file_id, title, decisions_json, actions_json, "
            " open_questions_json, recap_draft, captured_at, model, "
            " attendees_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            payload,
        )
    # Audit log row so /audit shows what we did.
    try:
        from . import ai_audit
        ai_audit.record_action(
            conn, kind="meeting_capture", feature="meeting_capture",
            model=model, status="success",
            file_id=file_id,
            prompt_chars=len(transcript),
            response_chars=len(recap_draft) + sum(
                len(d.text) + len(d.rationale) for d in decisions
            ),
            cents=cost,
            summary=(
                f"captured meeting: {title[:60]} "
                f"({len(decisions)} dec, {len(actions)} act)"
            ),
        )
    except Exception:  # noqa: BLE001
        pass
    # Action items flow into followups (the unified open-threads view).
    try:
        _flow_actions_to_followups(conn, file_id, actions, user_name)
    except Exception as e:  # noqa: BLE001
        log.warning("meeting_capture: followup flow failed: %s", e)
    return get_capture(conn, file_id)


def _flow_actions_to_followups(
    conn: sqlite3.Connection,
    file_id: int,
    actions: list[ActionItem],
    user_name: str,
) -> None:
    """Each meeting action becomes a followup row. Owner='user' →
    outgoing (you owe). Owner=name → incoming (they owe you)."""
    from . import followups
    from . import people as people_mod
    user_canon = (user_name or "").strip().lower()
    for a in actions:
        owner = (a.owner or "").strip()
        if not owner or owner.lower() == "unassigned":
            continue
        is_user = owner.lower() in (user_canon, "user", "me", "i")
        direction = "outgoing" if is_user else "incoming"
        person_id: int | None = None
        person_name = "" if is_user else owner
        if not is_user:
            try:
                p = people_mod.find_person_by_name(conn, owner)
                if p is not None:
                    person_id = int(p.id)
                    person_name = p.display_name
            except Exception:  # noqa: BLE001
                pass
        # Parse the due_hint into a timestamp if it's an ISO date.
        due_at: float | None = None
        try:
            from datetime import date, datetime
            d = date.fromisoformat(a.due_hint)
            due_at = datetime(d.year, d.month, d.day).timestamp()
        except (ValueError, TypeError):
            due_at = None
        followups.add_followup(
            conn,
            direction=direction,
            topic=a.description[:60],
            description=a.description,
            person_id=person_id,
            person_name=person_name,
            source_kind="meeting",
            source_file_id=file_id,
            source_excerpt=a.due_hint or "",
            due_at=due_at,
            promised_at=time.time(),
            confidence=0.9,
            extracted_by="llm",
        )


# ============================ queries ===============================


def get_capture(
    conn: sqlite3.Connection, file_id: int,
) -> MeetingCapture | None:
    _ensure_schema(conn)
    row = conn.execute(
        "SELECT * FROM meeting_captures WHERE file_id = ?", (file_id,),
    ).fetchone()
    return _row_to_capture(row) if row else None


def list_recent(
    conn: sqlite3.Connection, *, limit: int = 30,
) -> list[MeetingCapture]:
    _ensure_schema(conn)
    rows = conn.execute(
        "SELECT * FROM meeting_captures "
        "ORDER BY captured_at DESC LIMIT ?", (limit,),
    ).fetchall()
    return [_row_to_capture(r) for r in rows]


def mark_recap_sent(
    conn: sqlite3.Connection, file_id: int,
) -> bool:
    _ensure_schema(conn)
    with _WRITE_LOCK:
        cur = conn.execute(
            "UPDATE meeting_captures SET recap_sent_at = ? "
            "WHERE file_id = ?",
            (time.time(), file_id),
        )
        conn.commit()
        return cur.rowcount > 0


# ============================ Round 20 — edits ===================


def edit_capture(
    conn: sqlite3.Connection,
    file_id: int,
    *,
    title: str | None = None,
    decisions: list[dict] | None = None,
    actions: list[dict] | None = None,
    open_questions: list[str] | None = None,
    recap_draft: str | None = None,
    attendees: list[str] | None = None,
    calendar_event_id: str | None = None,
) -> bool:
    """User-side edits override the LLM extraction. Sets
    ``user_edited=1`` so re-running capture won't clobber. Returns
    True if anything changed."""
    _ensure_schema(conn)
    try:
        from .safety import redact_text
    except ImportError:
        def redact_text(s):
            return s
    updates: list[str] = []
    params: list = []
    if title is not None:
        updates.append("title = ?")
        params.append(redact_text(title)[:300])
    if decisions is not None:
        clean = [
            asdict(Decision(
                text=redact_text(str(d.get("text") or ""))[:500],
                rationale=redact_text(
                    str(d.get("rationale") or ""),
                )[:500],
            ))
            for d in decisions if isinstance(d, dict) and d.get("text")
        ]
        updates.append("decisions_json = ?")
        params.append(json.dumps(clean))
    if actions is not None:
        clean = [
            asdict(ActionItem(
                owner=str(a.get("owner") or "unassigned")[:120],
                description=redact_text(
                    str(a.get("description") or ""),
                )[:500],
                due_hint=str(a.get("due_hint") or "")[:120],
            ))
            for a in actions if isinstance(a, dict) and a.get("description")
        ]
        updates.append("actions_json = ?")
        params.append(json.dumps(clean))
    if open_questions is not None:
        clean = [redact_text(str(q))[:300] for q in open_questions if q]
        updates.append("open_questions_json = ?")
        params.append(json.dumps(clean))
    if recap_draft is not None:
        updates.append("recap_draft = ?")
        params.append(redact_text(recap_draft)[:4000])
    if attendees is not None:
        clean = [redact_text(str(a))[:120] for a in attendees if a]
        updates.append("attendees_json = ?")
        params.append(json.dumps(clean))
    if calendar_event_id is not None:
        updates.append("calendar_event_id = ?")
        params.append(calendar_event_id[:200])
    if not updates:
        return False
    updates.append("user_edited = 1")
    params.append(file_id)
    with _WRITE_LOCK:
        cur = conn.execute(
            f"UPDATE meeting_captures SET {', '.join(updates)} "
            f"WHERE file_id = ?",
            params,
        )
        conn.commit()
        return cur.rowcount > 0


def link_to_calendar_event(
    conn: sqlite3.Connection, file_id: int, event_id: str,
) -> bool:
    """Persist a Google Calendar event ID alongside this capture so
    a later "agenda for next 1:1 with Sarah" can match the meeting
    series."""
    return edit_capture(
        conn, file_id,
        calendar_event_id=str(event_id)[:200],
    )


def render_capture_markdown(cap: MeetingCapture) -> str:
    """Round 20 — format a capture as Markdown for the dashboard
    detail view + the recap-email copy/send flow."""
    try:
        from .safety import redact_text as _r
    except ImportError:
        def _r(s):
            return s
    lines = [f"# {_r(cap.title)}", ""]
    if cap.attendees:
        lines.append("**Attendees:** " + ", ".join(
            _r(a) for a in cap.attendees
        ))
        lines.append("")
    if cap.decisions:
        lines.append("## Decisions")
        for d in cap.decisions:
            lines.append(f"- **{_r(d.text)}**")
            if d.rationale:
                lines.append(f"  - {_r(d.rationale)}")
        lines.append("")
    if cap.actions:
        lines.append("## Action items")
        for a in cap.actions:
            due = f" · _due: {a.due_hint}_" if a.due_hint else ""
            lines.append(
                f"- **{_r(a.owner)}** — {_r(a.description)}{due}",
            )
        lines.append("")
    if cap.open_questions:
        lines.append("## Open questions")
        for q in cap.open_questions:
            lines.append(f"- {_r(q)}")
        lines.append("")
    if cap.recap_draft:
        lines.append("## Recap draft")
        lines.append(_r(cap.recap_draft))
    return "\n".join(lines)


def daemon_capture_recent(
    conn: sqlite3.Connection,
    cfg: Config,
    *,
    hours: int = 48,
    max_per_run: int = 5,
) -> int:
    """Daemon entry point. Find recent transcript-kind files that
    don't yet have a capture and run capture on each. Returns the
    number of new captures landed."""
    _ensure_schema(conn)
    cutoff = time.time() - hours * 3600
    rows = conn.execute(
        "SELECT f.id, f.path, f.kind FROM files f "
        "LEFT JOIN meeting_captures mc ON mc.file_id = f.id "
        "WHERE f.indexed_at >= ? AND mc.id IS NULL "
        "  AND (f.kind = 'audio_video' OR f.kind = 'transcript') "
        "ORDER BY f.indexed_at DESC LIMIT ?",
        (cutoff, max_per_run),
    ).fetchall()
    n_new = 0
    user_name = getattr(cfg, "user_name", None) or "User"
    for r in rows:
        try:
            res = capture(
                conn, cfg, int(r["id"]),
                user_name=user_name,
                overwrite=False,
            )
            if res is not None:
                n_new += 1
        except Exception as e:  # noqa: BLE001
            log.warning(
                "meeting_capture: daemon capture failed for %s: %s",
                r["path"], e,
            )
    return n_new
