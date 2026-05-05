"""Round 20 (Phase EA-1+) — followups operations layer.

Extends the round-19 ``followups`` module with the lifecycle ops a
real EA would do daily:

  - **Auto-resolution**: scan recently-sent emails / journal mentions
    and mark matching open *outgoing* followups resolved. (Telltale:
    "I just sent the deck" → mark "Send Sarah the deck" resolved.)
  - **Snooze**: defer an open followup until a future date. Snoozed
    items stay status='open' but get hidden from the default view
    until ``snooze_until`` passes.
  - **Edit**: the user can override topic / description / due_at /
    person assignment after the fact.
  - **Bulk dismiss**: clear all open followups for one person, or all
    overdue, or all of one direction.
  - **Auto-nudge**: for stale *incoming* followups (where the other
    party still owes the user something), draft a polite "checking
    in on this" email in the user's voice profile.
  - **Per-person view**: list everything pending with one person.
  - **History**: closed/resolved followups + how long they took.

Design: this is purely an extensions module — keeps the round-19
``followups.py`` lean (extraction + storage) while putting all
operational logic here. The dashboard / MCP / daemon layer all call
through this module.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import weakref as _weakref
from dataclasses import dataclass

from . import followups as _followups
from .config import Config

log = logging.getLogger(__name__)

_SCHEMA_INITIALIZED: _weakref.WeakSet = _weakref.WeakSet()

# Round 21 fix (audit-found gap F1) — share the followups module's
# write lock so a single critical section covers add_followup()
# and the auto-resolve writes here. Without this, dashboard worker
# threads + daemon (extractor + auto-resolve) intermittently hit
# ``database is locked`` under contention.
_WRITE_LOCK = _followups._WRITE_LOCK

_AUTO_RESOLVE_MIN_CONFIDENCE = 0.7
_NUDGE_MODEL = "claude-haiku-4-5"
_NUDGE_MAX_INPUT_CHARS = 4000


def _ensure_extended_schema(conn: sqlite3.Connection) -> None:
    """Add the round-20 columns to the round-19 ``followups`` table.

    Idempotent. Safe to call repeatedly. The base table is created
    by ``followups._ensure_schema``; we extend it.
    """
    try:
        if conn in _SCHEMA_INITIALIZED:
            return
    except TypeError:
        pass
    # Make sure the base table exists first.
    from . import followups
    followups._ensure_schema(conn)
    # Now add our extensions if missing.
    cols = {
        r["name"]
        for r in conn.execute("PRAGMA table_info(followups)")
    }
    if "snooze_until" not in cols:
        conn.execute(
            "ALTER TABLE followups ADD COLUMN snooze_until REAL"
        )
    if "auto_resolve_evidence" not in cols:
        conn.execute(
            "ALTER TABLE followups ADD COLUMN "
            "auto_resolve_evidence TEXT"
        )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_followups_snooze "
        "ON followups(snooze_until) WHERE snooze_until IS NOT NULL"
    )
    # History view's bookkeeping table — what auto-resolved when.
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS followup_resolutions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            followup_id INTEGER NOT NULL,
            resolved_at REAL NOT NULL,
            resolution_kind TEXT NOT NULL,  -- 'manual' | 'auto' | 'dismissed'
            evidence TEXT NOT NULL DEFAULT '',
            evidence_file_id INTEGER REFERENCES files(id) ON DELETE SET NULL
        );
        CREATE INDEX IF NOT EXISTS idx_followup_resolutions_at
            ON followup_resolutions(resolved_at DESC);
    """)
    conn.commit()
    try:
        _SCHEMA_INITIALIZED.add(conn)
    except TypeError:
        pass


# ============================ snooze + edit ========================


def snooze(
    conn: sqlite3.Connection, followup_id: int, days: int,
) -> bool:
    """Defer this followup by ``days``. Stays open but hidden from
    the default view until ``snooze_until`` passes."""
    if days < 1:
        raise ValueError(f"days must be >= 1; got {days}")
    _ensure_extended_schema(conn)
    target = time.time() + days * 86400
    with _WRITE_LOCK:
        cur = conn.execute(
            "UPDATE followups SET snooze_until = ?, updated_at = ? "
            "WHERE id = ? AND status = 'open'",
            (target, time.time(), followup_id),
        )
        conn.commit()
        return cur.rowcount > 0


def unsnooze(conn: sqlite3.Connection, followup_id: int) -> bool:
    _ensure_extended_schema(conn)
    with _WRITE_LOCK:
        cur = conn.execute(
            "UPDATE followups SET snooze_until = NULL, updated_at = ? "
            "WHERE id = ?",
            (time.time(), followup_id),
        )
        conn.commit()
        return cur.rowcount > 0


def edit(
    conn: sqlite3.Connection, followup_id: int,
    *,
    topic: str | None = None,
    description: str | None = None,
    due_at: float | None = -1.0,   # sentinel; -1 = no change, None = clear
    person_id: int | None = -1,    # same sentinel
    person_name: str | None = None,
) -> bool:
    """Update one or more fields on an open followup. Returns True
    iff something changed."""
    _ensure_extended_schema(conn)
    updates: list[str] = []
    params: list = []
    # Round 13/14 invariant — re-redact persisted text if changed.
    try:
        from .safety import redact_text
    except ImportError:
        def redact_text(s):
            return s
    if topic is not None:
        updates.append("topic = ?")
        params.append(redact_text(topic))
    if description is not None:
        updates.append("description = ?")
        params.append(redact_text(description))
    if due_at != -1.0:
        updates.append("due_at = ?")
        params.append(due_at)
    if person_id != -1:
        updates.append("person_id = ?")
        params.append(person_id)
    if person_name is not None:
        updates.append("person_name = ?")
        params.append(redact_text(person_name))
    if not updates:
        return False
    updates.append("updated_at = ?")
    params.append(time.time())
    params.append(followup_id)
    with _WRITE_LOCK:
        cur = conn.execute(
            f"UPDATE followups SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        conn.commit()
        return cur.rowcount > 0


def bulk_dismiss(
    conn: sqlite3.Connection,
    *,
    person_id: int | None = None,
    overdue_only: bool = False,
    direction: str | None = None,
) -> int:
    """Bulk-dismiss matching open followups. Returns count flipped."""
    _ensure_extended_schema(conn)
    sql = (
        "UPDATE followups SET status = 'dismissed', "
        "resolved_at = ?, updated_at = ? "
        "WHERE status = 'open'"
    )
    params: list = [time.time(), time.time()]
    if person_id is not None:
        sql += " AND person_id = ?"
        params.append(person_id)
    if overdue_only:
        sql += " AND due_at IS NOT NULL AND due_at < ?"
        params.append(time.time())
    if direction:
        sql += " AND direction = ?"
        params.append(direction)
    with _WRITE_LOCK:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.rowcount


# ============================ queries ==============================


def list_visible_open(
    conn: sqlite3.Connection,
    *,
    direction: str | None = None,
    person_id: int | None = None,
    include_snoozed: bool = False,
    limit: int = 200,
):
    """Like ``followups.list_open`` but excludes snoozed by default.

    Snoozed items reappear once ``snooze_until`` passes.
    """
    _ensure_extended_schema(conn)
    from . import followups
    sql = "SELECT * FROM followups WHERE status = 'open'"
    params: list = []
    if direction:
        sql += " AND direction = ?"
        params.append(direction)
    if person_id is not None:
        sql += " AND person_id = ?"
        params.append(person_id)
    if not include_snoozed:
        sql += " AND (snooze_until IS NULL OR snooze_until <= ?)"
        params.append(time.time())
    sql += (
        " ORDER BY "
        "  CASE WHEN due_at IS NULL THEN 1 ELSE 0 END, "
        "  due_at ASC, "
        "  promised_at DESC, "
        "  created_at DESC "
        "LIMIT ?"
    )
    params.append(limit)
    rows = conn.execute(sql, tuple(params)).fetchall()
    return [followups._row_to_followup(r) for r in rows]


def list_for_person(
    conn: sqlite3.Connection, person_id: int,
    *,
    include_resolved: bool = False,
    limit: int = 100,
):
    """All followups (open + optional resolved) for one person."""
    _ensure_extended_schema(conn)
    from . import followups
    sql = (
        "SELECT * FROM followups WHERE person_id = ?"
    )
    if not include_resolved:
        sql += " AND status = 'open'"
    sql += " ORDER BY created_at DESC LIMIT ?"
    rows = conn.execute(sql, (person_id, limit)).fetchall()
    return [followups._row_to_followup(r) for r in rows]


def list_resolved_history(
    conn: sqlite3.Connection,
    *,
    days: int = 30,
    limit: int = 50,
):
    """Recently resolved/dismissed followups, with computed
    "time-to-resolve" if promised_at exists."""
    _ensure_extended_schema(conn)
    from . import followups
    cutoff = time.time() - days * 86400
    rows = conn.execute(
        "SELECT * FROM followups "
        "WHERE status IN ('resolved', 'dismissed') "
        "  AND resolved_at >= ? "
        "ORDER BY resolved_at DESC LIMIT ?",
        (cutoff, limit),
    ).fetchall()
    out = []
    for r in rows:
        f = followups._row_to_followup(r)
        elapsed = None
        if f.promised_at and f.resolved_at:
            elapsed = f.resolved_at - f.promised_at
        out.append((f, elapsed))
    return out


# ============================ auto-resolution =====================


def auto_resolve_from_sent_mail(
    conn: sqlite3.Connection,
    cfg: Config,
    *,
    hours: int = 12,
    user_name: str | None = None,
    max_resolutions_per_run: int = 20,
) -> int:
    """Scan recently-indexed sent emails / journal entries for
    evidence that an open *outgoing* followup is now done.

    Heuristic + LLM hybrid:
      1. Pull open outgoing followups (one per person at a time
         to keep prompts small).
      2. Pull recent emails the user sent in the same window.
      3. For each (followup, email) pair, do a cheap text overlap
         check; high overlap → ask Haiku "did this email fulfill
         this commitment?". Auto-resolve on a YES + high confidence.

    Returns the number of auto-resolved followups. Idempotent —
    once resolved a followup won't be re-checked.
    """
    _ensure_extended_schema(conn)
    if user_name is None:
        user_name = getattr(cfg, "user_name", None) or "User"
    cutoff = time.time() - hours * 3600
    # Pull open outgoing followups.
    open_rows = conn.execute(
        "SELECT * FROM followups "
        "WHERE status = 'open' AND direction = 'outgoing' "
        "  AND (snooze_until IS NULL OR snooze_until <= ?)"
        "ORDER BY promised_at ASC LIMIT 100",
        (time.time(),),
    ).fetchall()
    if not open_rows:
        return 0
    # Pull recent files that look like user-authored content. We
    # treat journal/voice entries as user-authored unconditionally
    # (they're literally the user typing/dictating). For email/
    # message files we MUST verify the user is the sender — the
    # round-21 audit found that without this filter, an incoming
    # email from Sarah ("Re: Q3 deck — looks great!") that mentions
    # the topic was being treated as evidence that the user fulfilled
    # the commitment.
    candidate_files = _candidate_user_authored_files(
        conn, cutoff=cutoff, cfg=cfg,
    )
    if not candidate_files:
        return 0
    n_resolved = 0
    from . import followups
    for r in open_rows:
        if n_resolved >= max_resolutions_per_run:
            break
        f = followups._row_to_followup(r)
        # Cheap pre-filter: at least some token overlap with the
        # followup's topic OR the person's name.
        candidates = []
        for cf in candidate_files:
            preview = (cf["preview"] or "").lower()
            if not preview:
                continue
            if _has_evidence_signal(preview, f, user_name):
                candidates.append(cf)
        if not candidates:
            continue
        # LLM check — was the followup actually fulfilled?
        verdict = _llm_check_resolution(
            cfg, followup=f, candidates=candidates,
            user_name=user_name,
        )
        if verdict and verdict.get("resolved") and (
            verdict.get("confidence") or 0
        ) >= _AUTO_RESOLVE_MIN_CONFIDENCE:
            evidence = verdict.get("evidence", "")[:500]
            evidence_fid = verdict.get("evidence_file_id")
            # Round 21 fix (audit-found gap A2) — validate the
            # LLM-returned file_id is in the candidate set we showed
            # it. The model can hallucinate fids; we coerce to None
            # rather than persist a bad reference.
            valid_fids = {int(c["fid"]) for c in candidates}
            if evidence_fid is not None:
                try:
                    evidence_fid = int(evidence_fid)
                    if evidence_fid not in valid_fids:
                        evidence_fid = None
                except (TypeError, ValueError):
                    evidence_fid = None
            _record_auto_resolution(
                conn, followup_id=f.id,
                evidence=evidence, evidence_file_id=evidence_fid,
            )
            n_resolved += 1
    return n_resolved


def _candidate_user_authored_files(
    conn: sqlite3.Connection, *, cutoff: float, cfg,
) -> list:
    """Round 21 fix (audit-found gap A1) — pull recent files that
    are user-authored.

    Two safe paths:
      1. ``kind in ('voice', 'journal')`` → always user-authored.
      2. ``kind in ('email', 'message')`` → must contain the user's
         email address in the From: header (case-insensitive). This
         filters out incoming mail that just happens to mention the
         topic.

    Without the email filter, an incoming reply with "Re: Q3 deck"
    matched the keyword scan and the LLM resolved the followup
    based on someone else's reply.
    """
    user_email = (
        getattr(cfg, "user_email", "") or ""
    ).strip().lower()
    user_name = (
        getattr(cfg, "user_name", "") or ""
    ).strip().lower()
    rows = conn.execute(
        "SELECT f.id AS fid, f.path, f.kind, f.indexed_at, "
        "       SUBSTR(c.text, 1, 1500) AS preview "
        "FROM files f "
        "JOIN chunks c ON c.file_id = f.id AND c.chunk_index = 0 "
        "WHERE f.indexed_at >= ? "
        "  AND (f.kind = 'email' OR f.kind = 'message' "
        "       OR f.kind = 'voice' OR f.kind = 'journal') "
        "ORDER BY f.indexed_at DESC LIMIT 200",
        (cutoff,),
    ).fetchall()
    out = []
    for r in rows:
        kind = r["kind"]
        if kind in ("voice", "journal"):
            out.append(r)
            continue
        # email / message — verify user is the sender.
        preview = (r["preview"] or "").lower()
        # Find the From: line.
        import re as _re
        m = _re.search(r"(?im)^from:\s*(.+?)$", preview)
        if not m:
            continue
        from_line = m.group(1).strip()
        # Match if user_email appears verbatim, OR if user_name
        # appears with an angle-brackets-style "Name <email>" form
        # AND user_email is empty (best-effort fallback).
        if user_email and user_email in from_line or user_name and not user_email and user_name in from_line:
            out.append(r)
        # Else skip — we can't verify it's user-authored.
        if len(out) >= 60:
            break
    return out


def _has_evidence_signal(
    preview: str, f, user_name: str,
) -> bool:
    """Cheap text overlap check before we spend an LLM call."""
    person = (f.person_name or "").lower().strip()
    topic_words = {
        w for w in f.topic.lower().split() if len(w) > 3
    }
    desc_words = {
        w for w in f.description.lower().split() if len(w) > 3
    }
    keywords = (topic_words | desc_words)
    if person and person not in preview:
        # Person isn't mentioned at all — unlikely to be evidence.
        # But allow if we have strong topic-keyword overlap.
        if not keywords:
            return False
        n_match = sum(1 for k in keywords if k in preview)
        return n_match >= 2
    n_match = sum(1 for k in keywords if k in preview)
    return n_match >= 1


_RESOLUTION_SYSTEM = """\
You are an executive assistant deciding whether a recent piece of
correspondence (sent email, journal note) fulfills a commitment the
user previously made.

You receive:
  - A specific commitment with topic, description, person, due date.
  - Up to 5 candidate documents (sent emails / journal entries) the
    user produced recently.

You return JSON:
  resolved: true if AT LEAST ONE of the candidates fulfills the
            commitment; false otherwise.
  confidence: 0.0-1.0 — your certainty.
  evidence_file_id: the candidate's file_id if resolved; null otherwise.
  evidence: a 1-sentence explanation citing the candidate's text.

Be conservative. False positives (resolving a commitment that wasn't
actually fulfilled) hurt more than false negatives. If the candidate
is "I'll send it tomorrow" rather than "here it is", do NOT resolve.

Return ONLY the JSON object.
"""


def _llm_check_resolution(
    cfg: Config, *, followup, candidates, user_name: str,
) -> dict | None:
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    try:
        from .budget import check_budget, record_usage
        check_budget(
            cfg, "anthropic", feature="followups",
        )
    except Exception:  # noqa: BLE001 — also catches BudgetExceededError
        return None
    try:
        from .email_assist import _safe_for_prompt
    except ImportError:
        def _safe_for_prompt(s, max_chars):
            return (s or "")[:max_chars]
    # Round 21 fix (audit-found gap C1) — redact candidate previews
    # before sending to Anthropic. ``_safe_for_prompt`` only does
    # truncation + prompt-injection guard; the round-13 invariant
    # is that user content also passes through ``redact_text``
    # before egress. Defense-in-depth on followup fields too —
    # they should already be redacted at write but let's be sure.
    try:
        from .safety import redact_text
    except ImportError:
        def redact_text(s):
            return s
    cands_block = []
    for c in candidates[:5]:
        body = _safe_for_prompt(
            redact_text(c["preview"] or ""), max_chars=1000,
        )
        cands_block.append(
            f"<file_id={c['fid']} path={c['path']!r}>\n{body}\n</file>"
        )
    prompt = (
        f"User name: {user_name}\n\n"
        f"COMMITMENT:\n"
        f"  topic: "
        f"{_safe_for_prompt(redact_text(followup.topic), max_chars=200)}\n"
        f"  description: "
        f"{_safe_for_prompt(redact_text(followup.description), max_chars=500)}\n"
        f"  person: {followup.person_name or 'unknown'}\n"
        f"  promised_at_ts: {followup.promised_at or 0}\n"
        f"  due_at_ts: {followup.due_at or 0}\n\n"
        f"CANDIDATES (recent user-produced content):\n"
        + "\n\n".join(cands_block)
    )
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=_NUDGE_MODEL,
            max_tokens=400,
            system=_RESOLUTION_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as e:
        log.warning("auto-resolve: API error: %s", e)
        return None
    in_tok = getattr(resp.usage, "input_tokens", 0)
    out_tok = getattr(resp.usage, "output_tokens", 0)
    try:
        record_usage(
            cfg, "anthropic", _NUDGE_MODEL,
            input_tokens=in_tok, output_tokens=out_tok,
            note="followups/auto-resolve",
            feature="followups",
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
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _record_auto_resolution(
    conn: sqlite3.Connection,
    *,
    followup_id: int,
    evidence: str,
    evidence_file_id: int | None,
) -> None:
    now = time.time()
    with _WRITE_LOCK:
        conn.execute(
            "UPDATE followups SET "
            "  status = 'resolved', resolved_at = ?, "
            "  auto_resolve_evidence = ?, updated_at = ? "
            "WHERE id = ? AND status = 'open'",
            (now, evidence[:500], now, followup_id),
        )
        conn.execute(
            "INSERT INTO followup_resolutions"
            "(followup_id, resolved_at, resolution_kind, "
            " evidence, evidence_file_id) "
            "VALUES (?, ?, 'auto', ?, ?)",
            (followup_id, now, evidence[:500], evidence_file_id),
        )
        conn.commit()


# ============================ auto-nudge ==========================


_NUDGE_SYSTEM = """\
You draft polite, brief "checking in" emails for the user. The user
made a request to someone weeks ago that hasn't been addressed yet,
and they want to nudge without being pushy.

You receive:
  - The original commitment topic + description.
  - Days since they asked.
  - Optional: the user's voice profile (greeting, sign-off, contraction
    rate, typical phrasing).

You return a JSON object:
  subject: 4-8 word subject line, optionally starting with "Re: "
           if it's a continuation of an earlier thread.
  body: 2-4 sentence email body in the user's voice. Plain text,
        no greeting/sign-off (the user appends those). Tone:
        warm-but-direct. Acknowledge that life happens. Make it easy
        to reply.

Rules:
  - No "I just wanted to follow up", no "sorry to bother you", no
    "I know you're busy" — these are anti-patterns the user dislikes.
  - Don't re-state the entire request; reference it briefly.
  - Don't include emoji or exclamation marks.
  - Return ONLY the JSON object.
"""


def draft_nudge(
    conn: sqlite3.Connection,
    cfg: Config,
    followup_id: int,
    *,
    user_name: str | None = None,
) -> dict | None:
    """Generate a nudge-email draft for one stale incoming followup.

    Returns {"subject": ..., "body": ...} or None on failure / no
    API key. Persists nothing — caller decides whether to push to
    email_drafts.
    """
    _ensure_extended_schema(conn)
    from . import followups
    f_row = conn.execute(
        "SELECT * FROM followups WHERE id = ?", (followup_id,),
    ).fetchone()
    if not f_row:
        return None
    f = followups._row_to_followup(f_row)
    if f.direction != "incoming" or f.status != "open":
        return None
    if user_name is None:
        user_name = getattr(cfg, "user_name", None) or "User"
    age_days = 0
    if f.promised_at:
        age_days = max(1, int((time.time() - f.promised_at) / 86400.0))

    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    try:
        from .budget import check_budget, record_usage
        check_budget(
            cfg, "anthropic", feature="followups",
        )
    except Exception:  # noqa: BLE001 — also catches BudgetExceededError
        return None
    # Pull voice profile if available — keeps nudge tone consistent.
    voice_block = ""
    try:
        from . import email_assist
        vp = email_assist.get_voice_profile_or_default(conn)
        if vp:
            voice_block = (
                "\nVoice profile:\n"
                + email_assist._format_voice_profile_block(vp)
                + "\n"
            )
    except Exception:  # noqa: BLE001
        pass

    try:
        from .email_assist import _safe_for_prompt
    except ImportError:
        def _safe_for_prompt(s, max_chars):
            return (s or "")[:max_chars]
    # Round 21 fix (audit-found gap C2) — redact before send.
    try:
        from .safety import redact_text
    except ImportError:
        def redact_text(s):
            return s
    prompt = (
        f"User name: {user_name}\n"
        f"Recipient: {redact_text(f.person_name or 'them')}\n"
        f"Topic: "
        f"{_safe_for_prompt(redact_text(f.topic), max_chars=200)}\n"
        f"Original ask: "
        f"{_safe_for_prompt(redact_text(f.description), max_chars=500)}\n"
        f"Days since ask: {age_days}\n"
        f"{voice_block}"
    )
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=_NUDGE_MODEL,
            max_tokens=400,
            system=_NUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as e:
        log.warning("nudge: API error: %s", e)
        return None
    in_tok = getattr(resp.usage, "input_tokens", 0)
    out_tok = getattr(resp.usage, "output_tokens", 0)
    try:
        record_usage(
            cfg, "anthropic", _NUDGE_MODEL,
            input_tokens=in_tok, output_tokens=out_tok,
            note="followups/nudge",
            feature="followups",
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
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    try:
        from .safety import redact_text
        parsed["subject"] = redact_text(
            str(parsed.get("subject") or ""),
        )[:200]
        parsed["body"] = redact_text(
            str(parsed.get("body") or ""),
        )[:2000]
    except ImportError:
        parsed["subject"] = str(parsed.get("subject") or "")[:200]
        parsed["body"] = str(parsed.get("body") or "")[:2000]
    return parsed


# ============================ stats ===============================


@dataclass
class FollowupStats:
    open_outgoing: int
    open_incoming: int
    overdue_count: int
    snoozed_count: int
    resolved_last_30d: int
    avg_resolve_days_30d: float | None
    auto_resolved_last_30d: int


def compute_stats(conn: sqlite3.Connection) -> FollowupStats:
    """Snapshot for the dashboard / daily-brief / weekly-letter."""
    _ensure_extended_schema(conn)
    now = time.time()
    cutoff = now - 30 * 86400
    open_out = conn.execute(
        "SELECT COUNT(*) AS n FROM followups "
        "WHERE status = 'open' AND direction = 'outgoing' "
        "  AND (snooze_until IS NULL OR snooze_until <= ?)",
        (now,),
    ).fetchone()["n"]
    open_in = conn.execute(
        "SELECT COUNT(*) AS n FROM followups "
        "WHERE status = 'open' AND direction = 'incoming' "
        "  AND (snooze_until IS NULL OR snooze_until <= ?)",
        (now,),
    ).fetchone()["n"]
    overdue = conn.execute(
        "SELECT COUNT(*) AS n FROM followups "
        "WHERE status = 'open' AND due_at IS NOT NULL "
        "  AND due_at < ?",
        (now,),
    ).fetchone()["n"]
    snoozed = conn.execute(
        "SELECT COUNT(*) AS n FROM followups "
        "WHERE status = 'open' AND snooze_until > ?",
        (now,),
    ).fetchone()["n"]
    resolved = conn.execute(
        "SELECT COUNT(*) AS n FROM followups "
        "WHERE status = 'resolved' AND resolved_at >= ?",
        (cutoff,),
    ).fetchone()["n"]
    auto_resolved = conn.execute(
        "SELECT COUNT(*) AS n FROM followup_resolutions "
        "WHERE resolution_kind = 'auto' AND resolved_at >= ?",
        (cutoff,),
    ).fetchone()["n"]
    avg_seconds = conn.execute(
        "SELECT AVG(resolved_at - promised_at) AS s FROM followups "
        "WHERE status = 'resolved' AND resolved_at >= ? "
        "  AND promised_at IS NOT NULL",
        (cutoff,),
    ).fetchone()["s"]
    avg_days = (
        avg_seconds / 86400.0 if avg_seconds is not None else None
    )
    return FollowupStats(
        open_outgoing=int(open_out),
        open_incoming=int(open_in),
        overdue_count=int(overdue),
        snoozed_count=int(snoozed),
        resolved_last_30d=int(resolved),
        avg_resolve_days_30d=avg_days,
        auto_resolved_last_30d=int(auto_resolved),
    )
