"""Polish v3 round 8 — auto-generate thank-you emails after coffee
chats / 1:1s / external meetings.

Pipeline mirrors the round-6/7 email drafter, but starts from a
calendar event instead of an inbox email:

  1. **Detect**: poll calendar for events that ended in the last
     ~48h with at least one external attendee. Each gets a row in
     ``meeting_thanks`` with status='pending_context'.
  2. **Match transcript**: look for transcript:// docs whose mtime
     is near the event end + whose title overlaps the event title.
     When found → status='ready' and the transcript becomes the
     draft's context.
  3. **Generate**: when ready (transcript found OR user supplied
     context via CLI/dashboard), call the thank-you drafter. Reuses
     the same voice profile + reply-pair few-shot from email_assist
     so the thank-you sounds like the user. Output lands in
     ``email_drafts`` so it shows up in the existing /drafts UI
     alongside reply drafts.
  4. **User reviews + sends** through the same /drafts surface.
     Marking sent flips the meeting_thanks row to 'sent' too.

Skip rules: pure-internal meetings (everyone @user-domain), known
recurring patterns ("standup", "1:1"-with-direct-report, "weekly
sync") get auto-marked skipped on detection. The user can override
via the CLI.

Cost: thank-you drafts share the email_assist budget bucket so
they show up in the same per-feature spend cap.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
import weakref as _weakref
from dataclasses import dataclass

log = logging.getLogger(__name__)


# Schema-init cache, mirrors email_assist.
_SCHEMA_INITIALIZED: _weakref.WeakSet = _weakref.WeakSet()


# ---- Tunables -------------------------------------------------------

# How far back to look for finished meetings on each daemon scan.
# 48h covers the natural "I had coffee yesterday and meant to send
# a thank-you today" lag; longer windows risk re-discovering meetings
# the user already decided not to thank for.
_THANKS_LOOKBACK_SECONDS = 48 * 3600
# Transcript-match window: a transcript with mtime within ±2h of the
# event end is plausibly the same meeting. Tight enough to avoid
# matching the standup transcript to the coffee chat 90min later.
_TRANSCRIPT_MATCH_WINDOW = 2 * 3600
# Skip very short meetings — those are usually quick logistics calls
# the user isn't going to thank-you anyone for.
_MIN_MEETING_SECONDS = 10 * 60
# Skip very long ones too — that's a workshop / class, different shape.
_MAX_MEETING_SECONDS = 4 * 3600
# Auto-draft cooldown: once we draft, wait this long before
# re-drafting the same meeting (e.g. on user-supplied new context).
_REDRAFT_COOLDOWN_SECONDS = 12 * 3600
# Daemon: at most this many drafts per tick so a sudden backlog
# (returning from vacation) doesn't blow the budget at once.
_THANKS_DRAFTS_PER_TICK = 3


_STATUS_PENDING = "pending_context"
_STATUS_READY = "ready"
_STATUS_DRAFTED = "drafted"
_STATUS_SENT = "sent"
_STATUS_SKIPPED = "skipped"

# Title patterns we never thank-you for. Case-insensitive substring
# match. "1:1" by itself is fine — it's recurring 1:1s with the
# same direct report that we'd want to skip, but a one-off 1:1 with
# someone you just met is exactly the case we want. So just the
# obvious internal meetings here.
_SKIP_TITLE_PATTERNS = (
    "standup", "stand-up", "stand up",
    "team sync", "team meeting", "weekly sync",
    "all hands", "all-hands", "town hall",
    "office hours", "open office",
    "lunch break", "no meetings",
    "block",  # "focus block", "no-meeting block"
    "ooo", "out of office",
)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    try:
        if conn in _SCHEMA_INITIALIZED:
            return
    except TypeError:
        pass
    # Ensure email_drafts exists first — meeting_thanks.draft_id FKs
    # against it. Without this, INSERT raises 'no such table' on
    # databases that haven't seen email_assist yet.
    try:
        from . import email_assist
        email_assist._ensure_schema(conn)
    except Exception as e:  # noqa: BLE001
        log.debug("meeting_thanks: email_assist schema init skipped: %s", e)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS meeting_thanks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL UNIQUE,    -- calendar event_id (gcal id or ics uid)
            event_title TEXT NOT NULL,
            starts_at REAL NOT NULL,
            ends_at REAL NOT NULL,
            attendees_json TEXT,              -- list of external attendee emails
            transcript_path TEXT,             -- transcript:// path when matched
            user_context TEXT,                -- user-supplied context when no transcript
            status TEXT NOT NULL,             -- one of _STATUS_*
            draft_id INTEGER REFERENCES email_drafts(id) ON DELETE SET NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_meeting_thanks_status
            ON meeting_thanks(status, starts_at DESC);
    """)
    conn.commit()
    try:
        _SCHEMA_INITIALIZED.add(conn)
    except TypeError:
        pass


# ---- Data shapes ----------------------------------------------------

@dataclass
class MeetingThanks:
    """One row in the meeting_thanks queue."""
    id: int
    event_id: str
    event_title: str
    starts_at: float
    ends_at: float
    attendees: list[str]
    transcript_path: str | None
    user_context: str | None
    status: str
    draft_id: int | None
    created_at: float
    updated_at: float

    @property
    def has_context(self) -> bool:
        """True iff there's enough context to draft from — either a
        matched transcript or user-supplied notes."""
        return bool(self.transcript_path or self.user_context)


# ---- Detection ------------------------------------------------------

def _own_email_domains(cfg) -> set[str]:
    """Domains that count as the user's "own" — used to filter out
    pure-internal meetings. Sources, in priority order:
      1. ``cfg.user_email`` (round 21 — the canonical user identity)
      2. ``cfg.imap_username`` if it looks like an email
      3. ``cfg.digest_smtp_user`` / ``cfg.digest_smtp_from``

    Returns a set of lowercased domain strings (empty when nothing
    is configured — in that case every meeting is treated as
    external, which is the safer default).

    Round 25 fix (audit-found gap H4) — added ``user_email``.
    Without it, Gmail-OAuth users (no IMAP password) with no
    digest configured ended up with ``own_domains == set()``,
    so the user was classified as an "external" attendee in
    their own meeting and got drafted thank-yous addressed back
    to themselves.
    """
    domains: set[str] = set()
    for attr in (
        "user_email",
        "imap_username", "digest_smtp_user", "digest_smtp_from",
    ):
        val = getattr(cfg, attr, "") or ""
        if "@" in val:
            domains.add(val.split("@", 1)[1].strip().lower())
    return domains


def _classify_attendees(
    attendees: list[str], own_domains: set[str], organizer: str,
) -> tuple[list[str], list[str]]:
    """Split attendees into (external, internal).

    External = doesn't match any of our own domains AND isn't the
    user themselves. Used both for "does this meeting warrant a
    thank-you?" gating and for deriving the To: list when drafting.
    """
    external: list[str] = []
    internal: list[str] = []
    organizer_lower = (organizer or "").strip().lower()
    for raw in attendees or []:
        addr = (raw or "").strip().lower()
        if not addr or "@" not in addr:
            continue
        domain = addr.split("@", 1)[1]
        is_self = (
            (own_domains and domain in own_domains
             and addr.split("@", 1)[0] in {"me", "self"})
            or addr == organizer_lower
        )
        if is_self:
            continue
        if own_domains and domain in own_domains:
            internal.append(addr)
        else:
            external.append(addr)
    return external, internal


def _looks_skippable(title: str, duration_seconds: int) -> bool:
    """Title / duration heuristics for "don't bother thanking-you
    for this." Recurring internal meetings + extremely short or long
    sessions get skipped automatically."""
    if duration_seconds and duration_seconds < _MIN_MEETING_SECONDS:
        return True
    if duration_seconds and duration_seconds > _MAX_MEETING_SECONDS:
        return True
    t = (title or "").lower()
    return any(p in t for p in _SKIP_TITLE_PATTERNS)


def register_pending_thanks(
    conn: sqlite3.Connection, cfg, *, lookback_seconds: int = _THANKS_LOOKBACK_SECONDS,
) -> int:
    """Round 8 daemon entrypoint — scan recent calendar events and
    register thank-you-eligible ones in ``meeting_thanks``.

    Idempotent — events already in the table get skipped via the
    UNIQUE event_id constraint.

    Returns the count of newly-registered meetings. Failures (no
    calendar configured, network) become zero rather than raising.
    """
    _ensure_schema(conn)
    try:
        from .event_briefing import iter_recent_events
    except ImportError:
        return 0
    try:
        events = list(iter_recent_events(cfg, lookback_seconds))
    except Exception as e:  # noqa: BLE001
        log.warning("meeting_thanks: calendar fetch failed: %s", e)
        return 0
    own = _own_email_domains(cfg)
    n_new = 0
    for ev in events:
        ends_at = ev.starts_at + (ev.duration_seconds or 0)
        external, _internal = _classify_attendees(
            ev.attendees, own, ev.organizer_email,
        )
        # Skip: no external attendees → pure internal meeting
        if not external:
            continue
        # Skip: title / duration patterns
        skippable = _looks_skippable(ev.title, ev.duration_seconds or 0)
        status = _STATUS_SKIPPED if skippable else _STATUS_PENDING
        # Try to match a transcript right away — if found, we can
        # promote pending_context → ready in one step.
        transcript_path = None
        if status != _STATUS_SKIPPED:
            transcript_path = _find_transcript_for_event(
                conn, title=ev.title, ends_at=ends_at,
            )
            if transcript_path:
                status = _STATUS_READY
        try:
            cur = conn.execute(
                "INSERT OR IGNORE INTO meeting_thanks"
                "(event_id, event_title, starts_at, ends_at, "
                " attendees_json, transcript_path, user_context, "
                " status, draft_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?, ?) "
                "RETURNING id",
                (
                    ev.event_id, ev.title, ev.starts_at, ends_at,
                    json.dumps(external), transcript_path,
                    status, time.time(), time.time(),
                ),
            )
            row = cur.fetchone()
            if row is not None:
                n_new += 1
        except sqlite3.IntegrityError:
            # Race / dup — UNIQUE on event_id handles it.
            pass
    if n_new:
        conn.commit()
        log.info("meeting_thanks: registered %d new meeting(s)", n_new)
    return n_new


def rematch_transcripts(conn: sqlite3.Connection) -> int:
    """Round 8 daemon entrypoint — for pending_context rows whose
    transcript wasn't available at registration time, retry the
    match. Useful when the IMAP transcript ingester runs *after*
    the calendar scan picks up the meeting.

    Promotes matched rows to status='ready'. Returns count promoted.
    """
    _ensure_schema(conn)
    rows = conn.execute(
        "SELECT id, event_title, ends_at FROM meeting_thanks "
        "WHERE status = ? AND transcript_path IS NULL",
        (_STATUS_PENDING,),
    ).fetchall()
    n = 0
    for r in rows:
        path = _find_transcript_for_event(
            conn, title=r["event_title"], ends_at=r["ends_at"],
        )
        if not path:
            continue
        conn.execute(
            "UPDATE meeting_thanks "
            "SET transcript_path = ?, status = ?, updated_at = ? "
            "WHERE id = ?",
            (path, _STATUS_READY, time.time(), r["id"]),
        )
        n += 1
    if n:
        conn.commit()
        log.info("meeting_thanks: promoted %d row(s) to 'ready'", n)
    return n


# ---- Transcript matching --------------------------------------------

# Strip prefixes like "[meeting]" / "[capture]" + Re:/Fwd: from
# transcript / event titles before comparing them.
_TITLE_NOISE_RE = re.compile(
    r"^\s*(?:\[[^\]]+\]\s*|(?:re|fwd?|fw)\s*[:\-]\s*)+",
    re.IGNORECASE,
)
# Tokenise a title into matchable words. Keeps alnum runs of length
# ≥ 3 so we don't match on stop words like "to" / "a" / "of".
_TITLE_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")


def _normalise_title(title: str) -> str:
    """Strip prefixes + lowercase. Keeps spaces so token splits work."""
    if not title:
        return ""
    s = _TITLE_NOISE_RE.sub("", title)
    return s.strip().lower()


def _title_overlap(a: str, b: str) -> float:
    """Jaccard overlap on the meaningful tokens of two titles. 0.0
    means nothing matches; 1.0 is identical token sets. Used as a
    cheap "is this the same meeting" heuristic between calendar
    title + transcript title."""
    ta = set(_TITLE_TOKEN_RE.findall(_normalise_title(a)))
    tb = set(_TITLE_TOKEN_RE.findall(_normalise_title(b)))
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    union = ta | tb
    return len(inter) / len(union) if union else 0.0


def _find_transcript_for_event(
    conn: sqlite3.Connection, *, title: str, ends_at: float,
) -> str | None:
    """Look up a transcript:// file that plausibly captures this
    meeting. Two-pronged: mtime within ±2h of event end (cheap SQL
    filter) AND title-token overlap ≥ 0.3 (cheap Python filter).
    Returns the path of the best match, or None."""
    if not ends_at:
        return None
    lo = ends_at - _TRANSCRIPT_MATCH_WINDOW
    hi = ends_at + _TRANSCRIPT_MATCH_WINDOW
    rows = conn.execute(
        "SELECT f.path, f.mtime, c.text "
        "FROM files f JOIN chunks c ON c.file_id = f.id "
        "WHERE f.path LIKE 'transcript://%' "
        "  AND f.mtime BETWEEN ? AND ? "
        "  AND c.chunk_index = 0 "
        "ORDER BY ABS(f.mtime - ?) ASC LIMIT 10",
        (lo, hi, ends_at),
    ).fetchall()
    if not rows:
        return None
    best_path: str | None = None
    best_score = 0.0
    for r in rows:
        # First H1 line is the transcript title.
        text = r["text"] or ""
        h1 = ""
        for ln in text.splitlines()[:5]:
            s = ln.strip()
            if s.startswith("# "):
                h1 = s[2:].strip()
                break
        score = _title_overlap(title, h1)
        if score > best_score:
            best_score = score
            best_path = r["path"]
    # 0.3 is intentionally permissive — calendar titles ("Coffee w/
    # Sarah") rarely match transcript titles ("Sarah <> Ben sync")
    # exactly, and a false positive only costs us a re-roll if the
    # user rejects the draft.
    return best_path if best_score >= 0.3 else None


# ---- Read / write helpers -------------------------------------------

def list_pending(
    conn: sqlite3.Connection, *, limit: int = 50,
) -> list[MeetingThanks]:
    """All meetings still waiting on context or a draft. Excludes
    skipped / sent / drafted (those have their own surfaces)."""
    _ensure_schema(conn)
    rows = conn.execute(
        "SELECT * FROM meeting_thanks "
        "WHERE status IN (?, ?) "
        "ORDER BY ends_at DESC LIMIT ?",
        (_STATUS_PENDING, _STATUS_READY, limit),
    ).fetchall()
    return [_row_to_thanks(r) for r in rows]


def list_all(
    conn: sqlite3.Connection, *, limit: int = 100,
) -> list[MeetingThanks]:
    _ensure_schema(conn)
    rows = conn.execute(
        "SELECT * FROM meeting_thanks "
        "ORDER BY ends_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_row_to_thanks(r) for r in rows]


def get(conn: sqlite3.Connection, mt_id: int) -> MeetingThanks | None:
    _ensure_schema(conn)
    row = conn.execute(
        "SELECT * FROM meeting_thanks WHERE id = ?", (mt_id,),
    ).fetchone()
    return _row_to_thanks(row) if row else None


def set_context(
    conn: sqlite3.Connection, mt_id: int, text: str,
) -> bool:
    """User-provided context for a meeting that didn't have a
    transcript. Promotes status to 'ready'."""
    _ensure_schema(conn)
    cleaned = (text or "").strip()
    if not cleaned:
        return False
    cur = conn.execute(
        "UPDATE meeting_thanks "
        "SET user_context = ?, status = ?, updated_at = ? "
        "WHERE id = ?",
        (cleaned, _STATUS_READY, time.time(), mt_id),
    )
    conn.commit()
    return cur.rowcount > 0


def mark_skipped(
    conn: sqlite3.Connection, mt_id: int,
) -> bool:
    _ensure_schema(conn)
    cur = conn.execute(
        "UPDATE meeting_thanks SET status = ?, updated_at = ? "
        "WHERE id = ? AND status NOT IN (?, ?)",
        (_STATUS_SKIPPED, time.time(), mt_id, _STATUS_SENT, _STATUS_DRAFTED),
    )
    conn.commit()
    return cur.rowcount > 0


def mark_sent_for_draft(
    conn: sqlite3.Connection, draft_id: int,
) -> bool:
    """Called when the user marks the linked email_drafts row as
    sent — flips the meeting_thanks status to 'sent' too. Best-
    effort hook; failure here doesn't block the email path."""
    _ensure_schema(conn)
    cur = conn.execute(
        "UPDATE meeting_thanks SET status = ?, updated_at = ? "
        "WHERE draft_id = ? AND status = ?",
        (_STATUS_SENT, time.time(), draft_id, _STATUS_DRAFTED),
    )
    conn.commit()
    return cur.rowcount > 0


def _row_to_thanks(row) -> MeetingThanks:
    try:
        atts = json.loads(row["attendees_json"] or "[]")
    except (TypeError, ValueError):
        atts = []
    return MeetingThanks(
        id=int(row["id"]),
        event_id=row["event_id"],
        event_title=row["event_title"],
        starts_at=float(row["starts_at"]),
        ends_at=float(row["ends_at"]),
        attendees=[str(x) for x in atts],
        transcript_path=row["transcript_path"],
        user_context=row["user_context"],
        status=row["status"],
        draft_id=int(row["draft_id"]) if row["draft_id"] is not None else None,
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )


# ---- Drafter --------------------------------------------------------

# Round-8 thank-you prompt — same JSON-shaped output as the round-6
# drafter so the existing /drafts UI hydrates without changes. Note
# the explicit "do NOT invent specifics" rule + the fallback to a
# generic-but-warm thank-you when context is thin.
_THANKS_PROMPT = """\
Write a thank-you email from {user_name} after a meeting. Output ONE
JSON object (no prose, no Markdown fences):

{{
  "primary": "<the thank-you body — no subject line, no quote block>",
  "alternative": "<a SECOND version with a different tone register (formal vs casual swap)>",
  "reasoning": "<1-3 sentences on the choices made>",
  "confidence": <float 0.0..1.0>,
  "open_questions": ["<things {user_name} must fill in before sending>"]
}}

THANK-YOU RULES — this is the entire reason this email exists
==============
- The PRIMARY job is to make the recipient feel seen. Reference 1-2
  SPECIFIC things from the meeting context — a topic discussed, a
  question they answered, an interest they shared. Generic
  "thanks for chatting" thank-yous are worse than not sending one.
- Keep it SHORT. 3-5 sentences for casual coffee chats; up to a
  paragraph for formal interviews. Anything longer feels obligated.
- Restate any commitments you made ("I'll send the link", "I'll
  intro you to X"). If you forgot what you committed to, use a
  ``<TODO: confirm what I said I'd do>`` placeholder.
- Voice fidelity is paramount — match {user_name}'s greeting +
  sign-off + sentence rhythm exactly as observed in the voice
  profile. NEVER use any phrase listed in "avoid these phrases".
- For unknowns (specific dates, deliverables, names from the
  conversation), use ``<TODO: brief description>`` placeholders
  inline. Do NOT invent specifics.
- The ALTERNATIVE flips the tone register — if the primary is
  warm-casual, the alternative is more formal-restrained, or
  vice-versa. Lets the user pick which voice fits.

VOICE PROFILE — match these patterns
==============
{voice_profile_block}

FEW-SHOT EXAMPLES — how {user_name} actually replied to similar emails
==============
{fewshot_block}

MEETING CONTEXT
==============
- Title: {meeting_title}
- When: {when_str}
- Attendee(s) you're thanking: {attendees_str}

{context_source_label}:
{meeting_context}
"""


def generate_thanks_draft(
    conn: sqlite3.Connection, cfg, mt_id: int,
    *,
    user_name: str | None = None,
    drafter=None,
) -> int | None:
    """Round 8 — generate a thank-you draft for a meeting.

    Pulls the meeting's transcript (or user_context), assembles a
    context block, calls the same voice-aware drafter machinery the
    inbox-reply pipeline uses, and persists the draft into
    ``email_drafts`` so it surfaces in the existing /drafts UI.

    Round 25 fix (audit-found gap H2): default ``user_name`` to
    ``cfg.user_name`` so the daemon path + dashboard /thanks/.../draft
    POST both pick up the round-21 config field.

    Returns the new draft_id on success, None when:
      - Meeting not found
      - Status isn't 'ready' (no context to draft from)
      - Drafter LLM failed
      - Already drafted within the redraft cooldown
    """
    if user_name is None:
        user_name = getattr(cfg, "user_name", None) or "I"
    _ensure_schema(conn)
    mt = get(conn, mt_id)
    if mt is None:
        return None
    if mt.status != _STATUS_READY:
        return None
    if not mt.has_context:
        return None
    if (
        mt.draft_id is not None
        and (time.time() - mt.updated_at) < _REDRAFT_COOLDOWN_SECONDS
    ):
        # Still within cooldown — caller should wait or set context
        # again to force a redraft.
        return None

    context_text, context_label = _build_meeting_context(conn, mt)
    when_str = time.strftime(
        "%a %b %d, %H:%M", time.localtime(mt.starts_at),
    )
    attendees_str = ", ".join(mt.attendees) or "(unknown)"

    # Pull voice profile + few-shot pairs from email_assist so the
    # thank-you draft uses the same voice-fidelity machinery as
    # the inbox replies.
    from . import email_assist

    # Round 11 — use the curated bootstrap profile when no real one
    # exists yet (mirrors the round-10 #5 fix in email_assist).
    # Otherwise thank-you drafts on a fresh install would drop to
    # generic-LLM voice — exactly what the round-7 work tried to kill.
    voice_profile = email_assist.get_voice_profile_or_default(conn)
    voice_block = email_assist._format_voice_profile_block(voice_profile)
    embedder = None
    try:
        from .embedder import make_embedder
        embedder = make_embedder(cfg)
    except Exception:  # noqa: BLE001
        embedder = None
    fewshot = email_assist.fewshot_reply_pairs(
        conn, incoming_text=context_text, embedder=embedder,
    )
    fewshot_block = email_assist._format_fewshot_block(fewshot)

    # Round 10 (#4) — redact every raw-content field before send.
    # Meeting transcripts are some of the highest-leak content in
    # the brain (people often share credentials / numbers verbally
    # and the notetaker captures them verbatim).
    safe_context = email_assist._safe_for_prompt(
        context_text, max_chars=4000,
    )
    safe_fewshot = email_assist._safe_for_prompt(
        fewshot_block, max_chars=8000,
    )

    prompt = _THANKS_PROMPT.format(
        user_name=user_name,
        meeting_title=mt.event_title,
        when_str=when_str,
        attendees_str=attendees_str,
        context_source_label=context_label,
        meeting_context=safe_context,
        voice_profile_block=voice_block,
        fewshot_block=safe_fewshot,
    )

    # Test injection: pass-in drafter takes the prompt directly. Production
    # uses the same _llm_json_call as the email_assist drafter so spend +
    # fallback behaviour are consistent.
    if drafter is not None:
        parsed = drafter(prompt=prompt, cfg=cfg)
    else:
        parsed = email_assist._llm_json_call(
            prompt=prompt, cfg=cfg,
            model="claude-sonnet-4-6", max_tokens=1000,
            feature="meeting_thanks",
            note=f"thanks/{mt.event_title[:30]}",
            conn=conn,
            audit_kind="thanks_draft",
            audit_summary=f"thanks draft for {mt.event_title[:60]!r}",
        )
    if parsed is None:
        return None
    primary = str(parsed.get("primary") or "").strip()
    alternative = str(parsed.get("alternative") or "").strip()
    if not (primary or alternative):
        return None
    try:
        confidence = float(parsed.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    open_qs = [
        str(q) for q in (parsed.get("open_questions") or [])
        if str(q).strip()
    ]
    reasoning = str(parsed.get("reasoning") or "")

    # Voice critique loop — only when a profile exists. Mirrors the
    # email_assist behaviour so thank-you drafts get the same
    # voice-fidelity guarantees.
    # Skip critique against the bootstrap profile (n_samples=0) —
    # nothing user-specific to grade against, same logic as
    # email_assist.generate_draft.
    critique_text = ""
    if (
        voice_profile is not None
        and voice_profile.n_samples > 0
        and primary
    ):
        critique_text = email_assist.critique_draft_against_voice(
            draft=primary, profile=voice_profile, cfg=cfg,
            conn=conn,
        )
        if critique_text and critique_text.strip().upper() != "OK":
            log.info("meeting_thanks: critique flagged; regenerating")
            # Round 10 (#4) — defensive redaction on the prior draft
            # echo even though it's the LLM's own output (could echo
            # a TODO placeholder the user filled with a real value).
            safe_primary = email_assist._safe_for_prompt(
                primary, max_chars=2500,
            )
            regen_prompt = (
                prompt
                + f"\n\nPRIOR DRAFT (REJECTED)\n==========\n{safe_primary}\n"
                + "\nVOICE CRITIQUE — fix every item below\n==========\n"
                + critique_text
            )
            regen = email_assist._llm_json_call(
                prompt=regen_prompt, cfg=cfg,
                model="claude-sonnet-4-6", max_tokens=1000,
                feature="meeting_thanks",
                note=f"thanks_regen/{mt.event_title[:30]}",
                conn=conn,
                audit_kind="thanks_regen",
                audit_summary=f"thanks regen for {mt.event_title[:60]!r}",
            )
            if regen is not None:
                primary = str(regen.get("primary") or primary).strip()
                alternative = (
                    str(regen.get("alternative") or alternative).strip()
                )
                reasoning = str(regen.get("reasoning") or reasoning)

    # Persist into email_drafts so the existing /drafts UI handles
    # it. file_id is NULL since this isn't a reply to an inbox
    # email — we use a sentinel via the meeting_thanks table for
    # the linkage.
    metadata_json = json.dumps({
        "kind": "meeting_thanks",
        "meeting_thanks_id": mt.id,
        "meeting_event_id": mt.event_id,
        "meeting_event_title": mt.event_title,
        "meeting_attendees": mt.attendees,
        "alternative_text": alternative,
        "reasoning": reasoning,
        "confidence": confidence,
        "open_questions": open_qs,
        "voice_critique": critique_text or "",
        "voice_profile_n_samples": (
            voice_profile.n_samples if voice_profile else 0
        ),
        "fewshot_pairs": len(fewshot),
        "schema_version": 4,
    })
    # Thank-you drafts have no incoming inbox file_id — we point at
    # the closest analogue (the matched transcript) when available,
    # or at the sentinel meeting_thanks row otherwise. This keeps
    # the existing email_drafts FK happy.
    parent_file_id = _resolve_parent_file_id(conn, mt)
    cur = conn.execute(
        "INSERT INTO email_drafts"
        "(file_id, draft_text, generated_at, metadata_json) "
        "VALUES (?, ?, ?, ?) RETURNING id",
        (parent_file_id, primary, time.time(), metadata_json),
    )
    draft_id = int(cur.fetchone()["id"])
    conn.execute(
        "UPDATE meeting_thanks "
        "SET status = ?, draft_id = ?, updated_at = ? "
        "WHERE id = ?",
        (_STATUS_DRAFTED, draft_id, time.time(), mt.id),
    )
    conn.commit()
    log.info(
        "meeting_thanks: drafted thank-you for %r (draft #%d)",
        mt.event_title, draft_id,
    )
    return draft_id


def _resolve_parent_file_id(
    conn: sqlite3.Connection, mt: MeetingThanks,
) -> int:
    """Find a file_id to anchor the draft to. Order:
      1. The matched transcript file_id (preferred — clicking through
         from /drafts surfaces the transcript context).
      2. The user_context as a fallback synthetic file (one-time
         insert if needed).
      3. Any indexed file as a last-ditch sentinel — the email_drafts
         FK is NOT NULL but ON DELETE CASCADE; we never want to
         accidentally orphan a draft."""
    if mt.transcript_path:
        row = conn.execute(
            "SELECT id FROM files WHERE path = ?", (mt.transcript_path,),
        ).fetchone()
        if row is not None:
            return int(row["id"])
    # Use a synthetic "thanks://event-id" file so subsequent calls
    # don't double-insert. This is cheap — just a row in `files` so
    # the FK has somewhere to point.
    synth_path = f"thanks://meeting/{mt.event_id}"
    row = conn.execute(
        "SELECT id FROM files WHERE path = ?", (synth_path,),
    ).fetchone()
    if row is not None:
        return int(row["id"])
    n = time.time()
    cur = conn.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at) "
        "VALUES (?, ?, 0, 'thanks', ?)",
        (synth_path, mt.ends_at, n),
    )
    return int(cur.lastrowid)


def _build_meeting_context(
    conn: sqlite3.Connection, mt: MeetingThanks,
) -> tuple[str, str]:
    """Return ``(context_text, label)`` — the context block fed to
    the drafter prompt, and the label describing where it came from.

    Preference: matched transcript wins (it's the actual record of
    what was said). User-supplied notes fill in when no transcript
    was found.
    """
    if mt.transcript_path:
        row = conn.execute(
            "SELECT id FROM files WHERE path = ?", (mt.transcript_path,),
        ).fetchone()
        if row is not None:
            from .email_assist import _read_file_body
            body = _read_file_body(conn, int(row["id"]))
            if body:
                # Cap so a long lecture transcript doesn't crowd the
                # prompt — the drafter only needs the highlights.
                clipped = body[:4000] + (" […]" if len(body) > 4000 else "")
                return clipped, "TRANSCRIPT"
    if mt.user_context:
        return mt.user_context.strip(), "USER NOTES"
    return "(no context available)", "NONE"


# ---- Daemon entrypoint ----------------------------------------------

def process_due_thanks(
    conn: sqlite3.Connection, cfg,
    *, max_per_tick: int = _THANKS_DRAFTS_PER_TICK,
) -> int:
    """Daemon hook: register new meetings, retry transcript matching,
    and auto-draft any rows in 'ready' state. Bounded by
    ``max_per_tick``.

    Returns count of newly-drafted thank-yous (not registrations).
    """
    register_pending_thanks(conn, cfg)
    rematch_transcripts(conn)
    rows = conn.execute(
        "SELECT id FROM meeting_thanks WHERE status = ? "
        "ORDER BY ends_at DESC LIMIT ?",
        (_STATUS_READY, max_per_tick),
    ).fetchall()
    n = 0
    for r in rows:
        if generate_thanks_draft(conn, cfg, int(r["id"])) is not None:
            n += 1
    if n:
        log.info("meeting_thanks: drafted %d thank-you(s) this tick", n)
    return n
