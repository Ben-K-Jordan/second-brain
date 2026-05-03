"""Phase 82 + 83: email triage + auto-draft replies.

Phase 82 — Triage
  Your inbox is recruiting + classes + newsletters. Not all of it
  deserves the same attention. The triage classifier reads each
  recently-ingested email (from the IMAP connector) and assigns one
  of these labels:

    - urgent       — needs reply / action this week
    - response     — wants a reply but not time-critical
    - informational — read-and-archive
    - newsletter   — passive
    - automated    — receipts, notifications, no action

  Stored in ``email_classifications`` so the morning brief can show
  "12 emails today: 2 urgent, 5 response, ...".

Phase 83 — Auto-draft replies
  For emails the triage classified ``urgent`` / ``response`` —
  generate a reply draft in your voice. Voice is captured via "style
  reference" mode: pull samples from your past sent mail (the IMAP
  connector already indexes Sent if configured) and pass to the LLM
  as few-shot.

  Drafts persist; you ALWAYS see them before send. We never auto-send.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
import weakref as _weakref
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


# ---- Tunables ---------------------------------------------------------

# Classification: one Haiku call per email is ~$0.0005. Bound the
# per-tick fan-out so a daemon catching up on 500 backlog emails
# doesn't blow the budget all at once.
_CLASSIFY_PER_TICK = 10
# Only triage emails ingested in the last N days. Older ones are
# probably already actioned + we'd be wasting a re-classification.
_CLASSIFY_LOOKBACK_DAYS = 7

# Triage labels we accept. Anything outside this set falls back to
# 'informational' so a hallucinated label doesn't poison the table.
_VALID_LABELS: tuple[str, ...] = (
    "urgent", "response", "informational", "newsletter", "automated",
)

# Drafts: only generate for these labels by default.
_DRAFTABLE_LABELS = {"urgent", "response"}
# Per-doc cap on cards; reuse phase 67's pattern of bounded fan-out.
_DRAFT_PER_TICK = 5

# Schema-init cache, mirrors synthesis.py.
_SCHEMA_INITIALIZED: _weakref.WeakSet = _weakref.WeakSet()


def _ensure_schema(conn: sqlite3.Connection) -> None:
    try:
        if conn in _SCHEMA_INITIALIZED:
            return
    except TypeError:
        pass
    conn.executescript("""
        -- Phase 82 — email triage labels.
        -- One row per file_id (an email doc). file_id CASCADEs so a
        -- deleted email auto-clears its label.
        CREATE TABLE IF NOT EXISTS email_classifications (
            file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
            label TEXT NOT NULL,            -- one of _VALID_LABELS
            confidence REAL,                -- 0..1 from the classifier
            classified_at REAL NOT NULL,
            note TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_email_class_label
            ON email_classifications(label);

        -- Phase 83 — generated reply drafts.
        -- One row per (file_id, generation). We don't auto-overwrite —
        -- the user might want to compare drafts. The active flag marks
        -- the latest unsent one for the dashboard's "show me my drafts"
        -- view.
        CREATE TABLE IF NOT EXISTS email_drafts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            draft_text TEXT NOT NULL,
            generated_at REAL NOT NULL,
            sent_at REAL,                   -- null until user marks sent
            cents_spent REAL,
            note TEXT,
            -- Polish v3 round 6: structured metadata JSON. Holds the
            -- analyzer output + alternative draft + reasoning + open
            -- questions. NULL on legacy rows from the single-call drafter.
            metadata_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_email_drafts_file
            ON email_drafts(file_id, generated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_email_drafts_unsent
            ON email_drafts(sent_at, generated_at DESC);
    """)
    # Backfill the metadata_json column on databases created before
    # the round-6 schema change. ALTER ADD COLUMN is idempotent-by-
    # try/except (SQLite doesn't support IF NOT EXISTS on columns
    # before 3.35; use the duplicate-column-error catch instead).
    try:
        conn.execute("ALTER TABLE email_drafts ADD COLUMN metadata_json TEXT")
        conn.commit()
    except sqlite3.OperationalError as e:
        # 'duplicate column name' = already migrated; anything else
        # leaves the schema untouched (fresh DBs hit the CREATE path).
        if "duplicate column" not in str(e).lower():
            log.debug("email_assist: ALTER add metadata_json skipped: %s", e)
    conn.commit()
    try:
        _SCHEMA_INITIALIZED.add(conn)
    except TypeError:
        pass


# ---- Data shapes -----------------------------------------------------

@dataclass
class Classification:
    file_id: int
    label: str
    confidence: float | None
    classified_at: float
    note: str | None = None


@dataclass
class EmailAnalysis:
    """Polish v3 round 6 — structured analysis of an incoming email,
    produced by ``analyze_email`` before the drafter runs.

    All fields are best-effort: a noisy / nonsense email might give
    intent='other' and an empty key_points list, but the drafter
    handles that gracefully.
    """
    intent: str               # 'question' | 'schedule' | 'action_request'
                              # | 'decline_invite' | 'share_info'
                              # | 'follow_up' | 'intro' | 'other'
    sender_relationship: str  # 'manager' | 'colleague' | 'recruiter'
                              # | 'vendor' | 'friend' | 'family'
                              # | 'unknown'
    key_points: list[str]     # bullet-list of asks the sender made
    tone_signals: list[str]   # eg ['formal', 'urgent'] / ['casual', 'warm']
    length_target: str        # 'short' | 'medium' | 'long'
    open_questions: list[str] # things the user must fill in (dates,
                              #  prices, names, decisions)


@dataclass
class DraftOutput:
    """Rich return shape from the new structured drafter — held here
    in memory only; the persisted Draft.metadata_json is JSON of this
    plus the originating EmailAnalysis."""
    primary: str              # the main reply
    alternative: str          # one variant with a different tone register
    reasoning: str            # 1-3 line explanation of choices
    confidence: float         # 0..1
    open_questions: list[str] # mirrors analysis.open_questions but
                              #  filtered to ones the draft actually
                              #  needs the user to answer


@dataclass
class Draft:
    id: int
    file_id: int
    draft_text: str
    generated_at: float
    sent_at: float | None = None
    # Round 6 — populated when the structured drafter ran. Older
    # rows leave these as None so existing callers stay compatible.
    analysis: EmailAnalysis | None = None
    alternative_text: str | None = None
    reasoning: str | None = None
    open_questions: list[str] = field(default_factory=list)
    confidence: float | None = None


# ============================ Phase 82 triage =========================

_TRIAGE_PROMPT = """\
Classify this email into ONE of these labels:

  urgent        — needs my reply or action within ~1 week (recruiter
                  follow-up with deadline, professor's question I owe
                  an answer to, time-sensitive logistics).
  response      — wants a reply but not time-critical (interview
                  scheduler, friendly check-in, optional questions).
  informational — read once and archive (announcements, updates,
                  status reports, things I should know but don't act on).
  newsletter    — periodic content (Substack, marketing, digest emails).
  automated     — receipts, notifications, "your password was changed",
                  GitHub mentions, 2FA codes.

Respond with a JSON object:
  {{"label": "<one of above>", "confidence": 0..1, "rationale": "≤10 words"}}

No prose, no Markdown fences. Just the JSON.

From: {from_}
Subject: {subject}
Body:
{body}
"""


def needs_classification(
    conn: sqlite3.Connection, file_id: int,
) -> bool:
    """True iff the email isn't yet classified."""
    _ensure_schema(conn)
    row = conn.execute(
        "SELECT 1 FROM email_classifications WHERE file_id = ?",
        (file_id,),
    ).fetchone()
    return row is None


def classify_one(
    conn: sqlite3.Connection,
    file_id: int,
    *,
    cfg=None,
    classifier=None,
) -> Classification | None:
    """Classify one email file. ``classifier`` callable ``(from_,
    subject, body, cfg) -> dict`` returns ``{label, confidence,
    rationale?}``. Default uses Claude Haiku."""
    _ensure_schema(conn)
    if not needs_classification(conn, file_id):
        return None
    rows = conn.execute(
        "SELECT chunk_index, text FROM chunks WHERE file_id = ? "
        "ORDER BY chunk_index ASC LIMIT 5",
        (file_id,),
    ).fetchall()
    if not rows:
        return None
    body = "\n\n".join(r["text"] or "" for r in rows)
    from_, subject = _parse_email_header(body)
    if classifier is None:
        classifier = _default_classifier
        if cfg is None:
            return None
    try:
        result = classifier(from_, subject, body, cfg)
    except Exception as e:  # noqa: BLE001
        log.warning("email_assist: classify crashed: %s", e)
        return None
    if not result:
        return None
    label = (result.get("label") or "").strip().lower()
    if label not in _VALID_LABELS:
        # Hallucinated label → fall back to informational rather than
        # storing garbage.
        label = "informational"
    confidence = result.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (ValueError, TypeError):
        confidence = None
    note = (result.get("rationale") or "").strip()[:200] or None
    n = time.time()
    conn.execute(
        "INSERT OR REPLACE INTO email_classifications"
        "(file_id, label, confidence, classified_at, note) "
        "VALUES (?, ?, ?, ?, ?)",
        (file_id, label, confidence, n, note),
    )
    conn.commit()
    return Classification(
        file_id=file_id, label=label, confidence=confidence,
        classified_at=n, note=note,
    )


def _parse_email_header(body: str) -> tuple[str, str]:
    """Extract `From:` and `Subject:` (== `# title`) from an indexed
    email body. The IMAP connector renders these on the first few
    lines."""
    from_ = ""
    subject = ""
    for line in body.splitlines()[:30]:
        s = line.strip()
        if s.startswith("# "):
            subject = s[2:].strip()
        elif s.lower().startswith("from:"):
            from_ = s[5:].strip()
        if from_ and subject:
            break
    return from_, subject


def _default_classifier(from_, subject, body, cfg) -> dict:
    """Real Haiku classifier. Bounded body length to keep cost tight.

    Phase 89 wiring: when Anthropic isn't usable, fall back to a
    local Ollama call asking for the single label string. Confidence
    drops because local models are noisier on classification, but
    the daemon stays running.
    """
    import os

    body_clip = body if len(body) <= 4000 else body[:4000] + "…"
    prompt = _TRIAGE_PROMPT.format(
        from_=from_ or "(unknown)",
        subject=subject or "(no subject)",
        body=body_clip,
    )

    # ---- Primary: Anthropic ----
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
        except ImportError:
            anthropic = None  # type: ignore[assignment]
        if anthropic is not None:
            from .budget import (
                BudgetExceededError,
                check_budget,
                record_usage,
            )
            try:
                check_budget(cfg, "anthropic", feature="email_triage")
                client = anthropic.Anthropic()
                resp = client.messages.create(
                    model="claude-haiku-4-5",
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}],
                )
                record_usage(
                    cfg, "anthropic", "claude-haiku-4-5",
                    input_tokens=resp.usage.input_tokens,
                    output_tokens=resp.usage.output_tokens,
                    feature="email_triage",
                    note=f"email_triage/{from_[:30]}",
                )
                text = "".join(
                    b.text for b in resp.content if b.type == "text"
                ).strip()
                if text.startswith("```"):
                    text = re.sub(r"^```\w*\s*", "", text)
                    text = re.sub(r"\s*```\s*$", "", text)
                try:
                    parsed = json.loads(text)
                    if parsed:
                        return parsed
                except json.JSONDecodeError:
                    pass
            except BudgetExceededError as e:
                log.info(
                    "email_assist: budget exhausted, trying local LLM: %s", e,
                )
            except anthropic.APIError as e:
                log.info(
                    "email_assist: API error, trying local LLM: %s", e,
                )

    # ---- Fallback: local Ollama ----
    try:
        from . import local_llm
    except ImportError:
        return {}
    if not local_llm.is_available(cfg):
        return {}
    local_prompt = (
        "Classify this email into exactly one label: urgent, response, "
        "informational, newsletter, automated.\n"
        "Reply with only the single word label.\n\n"
        f"From: {from_}\nSubject: {subject}\n\n{body_clip}"
    )
    out = local_llm.complete(local_prompt, cfg=cfg, max_tokens=20)
    if out is None:
        return {}
    label_raw = out.text.strip().lower().split()[0] if out.text.strip() else ""
    label = label_raw.strip(".,;:'\"")
    if label not in {
        "urgent", "response", "informational", "newsletter", "automated",
    }:
        return {}
    log.info(
        "email_assist: classified via local LLM (%s) → %s", out.model, label,
    )
    return {"label": label, "confidence": 0.6, "rationale": "local-llm"}


def classify_due(
    conn: sqlite3.Connection, cfg,
    *, max_per_tick: int = _CLASSIFY_PER_TICK,
    classifier=None,
) -> int:
    """Daemon entrypoint. Find recent unclassified email docs and
    classify a bounded batch."""
    _ensure_schema(conn)
    cutoff = time.time() - _CLASSIFY_LOOKBACK_DAYS * 86400
    # LEFT JOIN + IS NULL — same perf trick as synthesis.materialize_
    # summaries_due. On a backlog of 10k emails this is ~30× faster
    # than the NOT IN subquery.
    rows = conn.execute(
        "SELECT f.id FROM files f "
        "LEFT JOIN email_classifications ec ON ec.file_id = f.id "
        "WHERE f.path LIKE 'imap://%' "
        "  AND f.indexed_at >= ? "
        "  AND ec.file_id IS NULL "
        "ORDER BY f.indexed_at DESC LIMIT ?",
        (cutoff, max_per_tick),
    ).fetchall()
    n = 0
    for r in rows:
        if classify_one(
            conn, int(r["id"]), cfg=cfg, classifier=classifier,
        ) is not None:
            n += 1
    if n:
        log.info("email_assist: classified %d email(s)", n)
    return n


def get_classification(
    conn: sqlite3.Connection, file_id: int,
) -> Classification | None:
    _ensure_schema(conn)
    row = conn.execute(
        "SELECT * FROM email_classifications WHERE file_id = ?",
        (file_id,),
    ).fetchone()
    if row is None:
        return None
    return Classification(
        file_id=int(row["file_id"]),
        label=row["label"],
        confidence=row["confidence"],
        classified_at=row["classified_at"],
        note=row["note"],
    )


def label_counts(
    conn: sqlite3.Connection, *, days: int = 7,
) -> dict[str, int]:
    """{label: count} for the last N days. Used by the morning brief."""
    _ensure_schema(conn)
    cutoff = time.time() - days * 86400
    rows = conn.execute(
        "SELECT label, COUNT(*) AS n FROM email_classifications "
        "WHERE classified_at >= ? GROUP BY label",
        (cutoff,),
    ).fetchall()
    return {r["label"]: int(r["n"]) for r in rows}


# ============================ Phase 83 drafts =========================

_DRAFT_PROMPT = """\
You're drafting a reply to this email in {user_name}'s voice.

INCOMING EMAIL
==============
From: {from_}
Subject: {subject}

{body}

STYLE REFERENCE — recent replies {user_name} has actually sent
==============
{style_samples}

INSTRUCTIONS
==============
- Match the voice from the style reference (tone, length, sign-off).
- Be concrete; address the actual ask.
- If declining or postponing, do so politely without over-explaining.
- Do not invent commitments {user_name} can't keep.
- Output ONLY the reply body — no subject line, no quoted block.

Draft:
"""


# ============================ Round 6: structured drafter ============

_ANALYZE_PROMPT = """\
Analyze this incoming email so a downstream drafter can write a
high-quality reply.

Output ONE JSON object — no prose, no Markdown fences, no preamble.

Schema:
{{
  "intent": "<one of: question | schedule | action_request | decline_invite | share_info | follow_up | intro | other>",
  "sender_relationship": "<one of: manager | colleague | recruiter | vendor | friend | family | unknown>",
  "key_points": ["<each ask the sender made, one per element>", "..."],
  "tone_signals": ["<adjectives like formal, casual, urgent, warm, transactional>"],
  "length_target": "<short | medium | long — how long the REPLY should be>",
  "open_questions": ["<things the user MUST decide before replying — dates, prices, names, yes/no choices>"]
}}

Rules:
- ``key_points`` should be the SENDER's asks rephrased clearly, not
  a verbatim copy. If the email is just informational, leave empty.
- ``open_questions`` should ONLY include things the user has to
  decide that aren't determinable from the email itself or the
  user's context. Don't invent open questions for friendly emails.
- ``length_target``: short ≈ 1-2 sentences. medium ≈ a paragraph.
  long ≈ multi-paragraph. Match what the email warrants.
- Hint at relationship from the email address domain, signature,
  greeting, and tone — don't guess wildly when unclear, use 'unknown'.

INCOMING EMAIL
==============
From: {from_}
Subject: {subject}

{body}
"""


_DRAFT_PROMPT_V2 = """\
Draft a reply to this email in {user_name}'s voice. You have:
  · An analyzer's structured assessment of the email
  · The relevant prior thread + sender history
  · {user_name}'s real recent replies as voice reference
  · {user_name}'s knowledge-base context for topics mentioned

OUTPUT FORMAT
==============
ONE JSON object — no prose, no Markdown fences:

{{
  "primary": "<the main reply body — no subject line, no quote block>",
  "alternative": "<a second version with a DIFFERENT tone register (formal vs casual swap)>",
  "reasoning": "<1-3 sentences on the choices made>",
  "confidence": <float 0.0..1.0 — how sure you are this is good>,
  "open_questions": ["<filtered list of decisions the user must fill in>"]
}}

DRAFTING RULES
==============
- The PRIMARY draft uses the analyzer's ``length_target`` and
  ``tone_signals``. The ALTERNATIVE flips the tone register
  (formal ↔ casual) so the user can pick.
- For unknowns the user must fill in, use ``<TODO: brief description>``
  inline (e.g. ``Tuesday at <TODO: pick a time> works for me``).
- DO NOT invent commitments, dates, prices, or facts. When unsure,
  use a TODO placeholder.
- Match {user_name}'s sign-off pattern from the style samples —
  don't pick a generic "Best regards" if their samples show "thanks,".
- AVOID generic LLM opener phrases: "I hope this email finds you
  well", "Thank you for reaching out", "Please let me know if you
  have any further questions". If {user_name}'s style samples use
  similar openers, fine; otherwise skip them.
- If the email is purely informational and doesn't need a reply,
  set primary to "" and confidence to 0.0.
- Reference the prior thread + sender history when natural ("re your
  earlier point about X", "as we discussed last week") — but only
  when the prior actually supports it.

ANALYSIS (from upstream analyzer)
==============
{analysis_block}

INCOMING EMAIL
==============
From: {from_}
Subject: {subject}

{body}

THREAD HISTORY (oldest first; empty when no prior messages)
==============
{thread_block}

SENDER HISTORY ({user_name} ↔ this person, recent first)
==============
{sender_block}

STYLE REFERENCE — recent replies {user_name} has actually sent
==============
{style_samples}

BRAIN CONTEXT (relevant snippets from {user_name}'s knowledge base)
==============
{brain_block}
"""


def analyze_email(
    *, from_: str, subject: str, body: str, cfg,
) -> EmailAnalysis | None:
    """Round 6 — structured analysis of an incoming email.

    Single Haiku call returning JSON with intent / relationship /
    asks / tone / open_questions. Used by the new drafter pipeline
    to plan the reply before writing it.

    Returns None when neither Anthropic nor the local LLM produces
    parseable JSON — caller falls back to the legacy single-call
    drafter so we never hard-fail on a daemon tick.
    """

    body_clip = body if len(body) <= 4000 else body[:4000] + "…"
    prompt = _ANALYZE_PROMPT.format(
        from_=from_ or "(unknown)",
        subject=subject or "(no subject)",
        body=body_clip,
    )
    raw = _llm_json_call(
        prompt=prompt,
        cfg=cfg,
        model="claude-haiku-4-5",
        max_tokens=400,
        feature="email_analyze",
        note=f"analyze/{(from_ or '')[:30]}",
    )
    if not raw:
        return None
    try:
        return EmailAnalysis(
            intent=str(raw.get("intent") or "other"),
            sender_relationship=str(
                raw.get("sender_relationship") or "unknown",
            ),
            key_points=[str(x) for x in (raw.get("key_points") or [])],
            tone_signals=[str(x) for x in (raw.get("tone_signals") or [])],
            length_target=str(raw.get("length_target") or "medium"),
            open_questions=[
                str(x) for x in (raw.get("open_questions") or [])
            ],
        )
    except Exception as e:  # noqa: BLE001
        log.warning("email_assist: analyze parse failed: %s", e)
        return None


def _llm_json_call(
    *, prompt: str, cfg, model: str, max_tokens: int,
    feature: str, note: str,
) -> dict | None:
    """Shared helper: try Anthropic with budget guard, fall back to
    local Ollama, parse JSON. Returns the parsed dict or None.

    Centralises the try-Anthropic-then-local pattern so the analyzer
    + drafter share one code path. Strips Markdown fences before
    parsing because both Claude and small local models occasionally
    wrap output in ```json``` despite being told not to.
    """
    import os

    text = ""

    # ---- Primary: Anthropic ----
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
        except ImportError:
            anthropic = None  # type: ignore[assignment]
        if anthropic is not None:
            from .budget import (
                BudgetExceededError,
                check_budget,
                record_usage,
            )
            try:
                check_budget(cfg, "anthropic", feature=feature)
                client = anthropic.Anthropic()
                resp = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                record_usage(
                    cfg, "anthropic", model,
                    input_tokens=resp.usage.input_tokens,
                    output_tokens=resp.usage.output_tokens,
                    feature=feature, note=note,
                )
                text = "".join(
                    b.text for b in resp.content if b.type == "text"
                ).strip()
            except BudgetExceededError as e:
                log.info(
                    "email_assist: %s budget exhausted, trying local: %s",
                    feature, e,
                )
            except anthropic.APIError as e:
                log.info(
                    "email_assist: %s API error, trying local: %s",
                    feature, e,
                )

    # ---- Fallback: local Ollama ----
    if not text:
        try:
            from . import local_llm
        except ImportError:
            return None
        if not local_llm.is_available(cfg):
            return None
        out = local_llm.complete(prompt, cfg=cfg, max_tokens=max_tokens)
        if out is None:
            return None
        text = out.text.strip()
        log.info("email_assist: %s via local LLM (%s)", feature, out.model)

    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```\w*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        log.info("email_assist: %s JSON parse failed; raw=%r", feature, text[:200])
        return None


# ---- Round 6 retrieval helpers ---------------------------------------

# Strip Re:/Fwd:/[List] prefixes when matching emails to the same
# subject thread. Case-insensitive; handles "Re: Re:" stacking and
# bracketed list tags. Used as the IMAP-side thread heuristic when
# Gmail's threadId isn't in the path.
_SUBJ_PREFIX_RE = re.compile(
    r"^\s*(?:(?:re|fwd?|fw)\s*[:\-]\s*|\[[^\]]+\]\s*)+",
    re.IGNORECASE,
)
# Pull the angle-bracket-wrapped or bare email address out of a "From:"
# line so we can match the same correspondent across messages.
_EMAIL_ADDR_RE = re.compile(r"<([^>]+@[^>]+)>|([\w.+\-]+@[\w.\-]+)")
# How much prior context to pull. More is better quality, but tokens
# are real money. Three of each is the sweet spot — enough for the
# model to see "we discussed X last week" without ballooning prompts.
_HISTORY_MAX = 3
_HISTORY_BODY_CHARS = 1200
# Brain-context cap. Two short snippets is enough to ground a draft
# without crowding out the actual email content.
_BRAIN_CONTEXT_HITS = 2
_BRAIN_CONTEXT_CHARS = 600


def _normalize_subject(s: str) -> str:
    """Strip threading prefixes so Re/Fwd of the same subject collapse
    into one bucket. Case-insensitive output (lowercase) for direct
    string comparison."""
    if not s:
        return ""
    cur = s
    # Loop because "Re: Fwd: Re:" can stack.
    for _ in range(5):
        new = _SUBJ_PREFIX_RE.sub("", cur)
        if new == cur:
            break
        cur = new
    return cur.strip().lower()


def _extract_email_address(from_header: str) -> str:
    """Pull the bare 'name@host' out of a 'From:' line. Returns empty
    string when no match — caller should treat that as 'no sender
    history available'."""
    if not from_header:
        return ""
    m = _EMAIL_ADDR_RE.search(from_header)
    if not m:
        return ""
    return (m.group(1) or m.group(2) or "").strip().lower()


def _gmail_thread_id_from_path(path: str) -> str:
    """Gmail virtual paths look like 'gmail://thread/<tid>/message/<mid>'.
    Returns the thread id when present, empty string otherwise."""
    if not path or not path.startswith("gmail://thread/"):
        return ""
    rest = path[len("gmail://thread/"):]
    return rest.split("/", 1)[0] if rest else ""


def _pull_thread_history(
    conn: sqlite3.Connection,
    *,
    file_id: int,
    subject: str,
) -> list[tuple[str, str]]:
    """Round 6 — fetch other messages in the same thread. Returns
    ``[(path, body), ...]`` ordered oldest-first.

    Two paths:
      1. Gmail thread_id in the virtual_path → exact sibling lookup.
      2. IMAP / unknown source → subject-prefix match on the
         normalised subject across all email-shaped files.

    Excludes the originating file_id so the prompt doesn't echo the
    email back to itself.
    """
    row = conn.execute(
        "SELECT path FROM files WHERE id = ?", (file_id,),
    ).fetchone()
    if row is None:
        return []
    src_path = row["path"] or ""
    tid = _gmail_thread_id_from_path(src_path)
    if tid:
        rows = conn.execute(
            "SELECT f.id, f.path, f.mtime FROM files f "
            "WHERE f.path LIKE ? AND f.id != ? "
            "ORDER BY f.mtime ASC LIMIT ?",
            (f"gmail://thread/{tid}/%", file_id, _HISTORY_MAX),
        ).fetchall()
    else:
        norm = _normalize_subject(subject)
        if not norm:
            return []
        # Subject is rendered as the H1 in the indexed body. Use a
        # cheap LIKE on the normalised form — false positives don't
        # cost us much because it's only context for the LLM.
        rows = conn.execute(
            "SELECT DISTINCT f.id, f.path, f.mtime "
            "FROM files f JOIN chunks c ON c.file_id = f.id "
            "WHERE (f.path LIKE 'imap://%' OR f.path LIKE 'gmail://%') "
            "  AND f.id != ? "
            "  AND LOWER(c.text) LIKE ? "
            "ORDER BY f.mtime ASC LIMIT ?",
            (file_id, f"%# %{norm}%", _HISTORY_MAX),
        ).fetchall()
    out: list[tuple[str, str]] = []
    for r in rows:
        body = _read_file_body(conn, int(r["id"]))
        if body:
            out.append((r["path"], body))
    return out


def _pull_sender_history(
    conn: sqlite3.Connection,
    *,
    file_id: int,
    sender_email: str,
) -> list[tuple[str, str]]:
    """Round 6 — pull recent emails from the same correspondent (any
    thread). Returns ``[(path, body), ...]`` newest-first.

    Used to give the drafter context about the relationship — what's
    been discussed, what tone has been set, what the user's prior
    replies looked like to this person specifically.
    """
    if not sender_email:
        return []
    rows = conn.execute(
        "SELECT DISTINCT f.id, f.path, f.mtime "
        "FROM files f JOIN chunks c ON c.file_id = f.id "
        "WHERE (f.path LIKE 'imap://%' OR f.path LIKE 'gmail://%') "
        "  AND f.id != ? "
        "  AND LOWER(c.text) LIKE ? "
        "ORDER BY f.mtime DESC LIMIT ?",
        (file_id, f"%from:%{sender_email}%", _HISTORY_MAX),
    ).fetchall()
    out: list[tuple[str, str]] = []
    for r in rows:
        body = _read_file_body(conn, int(r["id"]))
        if body:
            out.append((r["path"], body))
    return out


def _read_file_body(conn: sqlite3.Connection, file_id: int) -> str:
    """Concatenate a file's chunks back into one string — capped to
    keep history blocks bounded in the prompt."""
    rows = conn.execute(
        "SELECT text FROM chunks WHERE file_id = ? "
        "ORDER BY chunk_index ASC",
        (file_id,),
    ).fetchall()
    body = "\n\n".join((r["text"] or "") for r in rows).strip()
    if len(body) > _HISTORY_BODY_CHARS:
        body = body[:_HISTORY_BODY_CHARS] + " […]"
    return body


def _select_style_samples_smart(
    conn: sqlite3.Connection,
    *,
    sender_email: str,
    relationship: str,
    n: int = 3,
) -> str:
    """Round 6 — pick style samples scoped to the inferred relationship.

    Prefers replies the user actually sent to *this* sender. Falls
    back to recent Sent items in general when no targeted samples
    are found. Without this, the random-sample picker would feed
    the recruiter-draft a casual reply to a friend.
    """
    rows: list = []
    if sender_email:
        rows = conn.execute(
            "SELECT c.text FROM chunks c JOIN files f ON f.id = c.file_id "
            "WHERE f.path LIKE 'imap://%' "
            "  AND LOWER(c.text) LIKE '%folder: sent%' "
            "  AND LOWER(c.text) LIKE ? "
            "ORDER BY f.indexed_at DESC LIMIT ?",
            (f"%to:%{sender_email}%", n),
        ).fetchall()
    if len(rows) < n:
        # Top up with general Sent items — anything is better than the
        # neutral-tone default the legacy picker fell back to.
        extra = conn.execute(
            "SELECT c.text FROM chunks c JOIN files f ON f.id = c.file_id "
            "WHERE f.path LIKE 'imap://%' "
            "  AND LOWER(c.text) LIKE '%folder: sent%' "
            "ORDER BY f.indexed_at DESC LIMIT ?",
            (n - len(rows),),
        ).fetchall()
        rows.extend(extra)
    if not rows:
        return (
            f"(no recent sent mail indexed; defaulting to a neutral tone "
            f"for an inferred-{relationship} sender)"
        )
    return "\n\n---\n\n".join(
        (r["text"] or "")[:1500] for r in rows[:n]
    )


def _brain_context_for_topics(
    conn: sqlite3.Connection,
    cfg,
    key_points: list[str],
) -> str:
    """Round 6 — pull 1-2 brain hits for the topics the analyzer
    identified, so the drafter can ground the reply in the user's
    actual knowledge instead of inventing context.

    Best-effort: requires an embedder + the search module. Returns
    an empty string when retrieval isn't available; the drafter
    handles that fine."""
    if not key_points:
        return ""
    try:
        from .embedder import make_embedder
        from .search import hybrid_search
    except ImportError:
        return ""
    try:
        embedder = make_embedder(cfg)
    except Exception as e:  # noqa: BLE001
        log.info("email_assist: brain context skipped (embedder): %s", e)
        return ""
    # Run one combined search across the top key points so we don't
    # spend N×$0.001 on N separate embeddings. Joining with "; " keeps
    # each ask distinct enough that the embedding lands in the right
    # neighbourhood.
    query = "; ".join(k for k in key_points[:3] if k)
    if not query.strip():
        return ""
    try:
        results = hybrid_search(
            conn, embedder, query, k=_BRAIN_CONTEXT_HITS,
        )
    except Exception as e:  # noqa: BLE001
        log.info("email_assist: brain context search failed: %s", e)
        return ""
    if not results:
        return ""
    blocks = []
    for r in results[:_BRAIN_CONTEXT_HITS]:
        snippet = r.text[:_BRAIN_CONTEXT_CHARS]
        if len(r.text) > _BRAIN_CONTEXT_CHARS:
            snippet += " […]"
        blocks.append(f"[{r.file_path}]\n{snippet}")
    return "\n\n".join(blocks)


def _format_history_block(items: list[tuple[str, str]]) -> str:
    """Render thread / sender history as a labelled block. Empty list
    becomes "(none)" so the drafter prompt always has a value."""
    if not items:
        return "(none)"
    parts = []
    for path, body in items:
        parts.append(f"--- {path} ---\n{body}")
    return "\n\n".join(parts)


def _format_analysis_block(a: EmailAnalysis) -> str:
    """Render an EmailAnalysis as a compact text block for the
    drafter prompt. Avoids re-serialising the JSON since the drafter
    doesn't need to parse it back."""
    points = "\n".join(f"  - {p}" for p in a.key_points) or "  (none)"
    todos = "\n".join(f"  - {q}" for q in a.open_questions) or "  (none)"
    tone = ", ".join(a.tone_signals) or "neutral"
    return (
        f"intent: {a.intent}\n"
        f"sender_relationship: {a.sender_relationship}\n"
        f"length_target: {a.length_target}\n"
        f"tone_signals: {tone}\n"
        f"key_points (sender's asks):\n{points}\n"
        f"open_questions (user must decide):\n{todos}"
    )


def _gather_style_samples(
    conn: sqlite3.Connection, n: int = 3,
) -> str:
    """Pull a few recent ``Sent`` items as voice reference. Returns
    one big string; empty when no Sent items have been ingested."""
    rows = conn.execute(
        "SELECT c.text FROM chunks c JOIN files f ON f.id = c.file_id "
        "WHERE f.path LIKE 'imap://%' "
        "  AND c.text LIKE '%Folder: Sent%' "
        "ORDER BY f.indexed_at DESC LIMIT ?",
        (n,),
    ).fetchall()
    if not rows:
        return "(no recent sent mail indexed — defaulting to a neutral tone)"
    return "\n\n---\n\n".join(
        (r["text"] or "")[:1500] for r in rows
    )


def needs_draft(conn: sqlite3.Connection, file_id: int) -> bool:
    """True iff the email is classified as draftable AND has no
    unsent draft yet."""
    _ensure_schema(conn)
    cls = get_classification(conn, file_id)
    if cls is None or cls.label not in _DRAFTABLE_LABELS:
        return False
    existing = conn.execute(
        "SELECT 1 FROM email_drafts WHERE file_id = ? AND sent_at IS NULL "
        "LIMIT 1",
        (file_id,),
    ).fetchone()
    return existing is None


def generate_draft(
    conn: sqlite3.Connection,
    file_id: int,
    *,
    cfg=None,
    user_name: str = "I",
    drafter=None,
) -> Draft | None:
    """Generate a reply draft using the round-6 structured pipeline:

      1. ``analyze_email`` — Haiku JSON: intent / relationship /
         key_points / tone / open_questions.
      2. Targeted retrieval — thread history, sender history,
         relationship-scoped style samples, brain context for the
         topics the analyzer identified.
      3. ``_default_drafter`` — Sonnet, fed all of the above; outputs
         primary + alternative + reasoning + open_questions in JSON.

    Persists with sent_at=NULL so the user can review. The structured
    metadata lives in ``metadata_json``; legacy callers that read
    ``draft_text`` still work unchanged.

    The ``drafter`` argument is preserved for tests that want to
    inject a stub. Test stubs that return a plain string still
    work — we treat that as the legacy single-version path.
    """
    _ensure_schema(conn)
    if not needs_draft(conn, file_id):
        return None
    rows = conn.execute(
        "SELECT chunk_index, text FROM chunks WHERE file_id = ? "
        "ORDER BY chunk_index ASC",
        (file_id,),
    ).fetchall()
    if not rows:
        return None
    body = "\n\n".join(r["text"] or "" for r in rows)
    from_, subject = _parse_email_header(body)

    # Test-injection path: a custom drafter that takes the legacy
    # kwargs returns just text. We don't run analysis or retrieval
    # in that case — keeps existing tests fast + deterministic.
    if drafter is not None:
        return _persist_legacy_draft(
            conn, file_id, drafter, from_, subject, body, user_name, cfg,
        )
    if cfg is None:
        return None

    # ---- Stage 1: analyze ----
    analysis = analyze_email(
        from_=from_, subject=subject, body=body, cfg=cfg,
    )
    # ---- Stage 2: targeted retrieval ----
    sender_email = _extract_email_address(from_)
    relationship = analysis.sender_relationship if analysis else "unknown"
    key_points = analysis.key_points if analysis else []
    thread_hist = _pull_thread_history(
        conn, file_id=file_id, subject=subject,
    )
    sender_hist = _pull_sender_history(
        conn, file_id=file_id, sender_email=sender_email,
    )
    style_samples = _select_style_samples_smart(
        conn, sender_email=sender_email, relationship=relationship,
    )
    brain_block = _brain_context_for_topics(conn, cfg, key_points)

    # ---- Stage 3: draft ----
    output = _default_drafter(
        from_=from_, subject=subject, body=body,
        style_samples=style_samples,
        user_name=user_name, cfg=cfg,
        analysis=analysis,
        thread_history=thread_hist,
        sender_history=sender_hist,
        brain_context=brain_block,
    )
    if output is None or not (output.primary or output.alternative).strip():
        return None

    metadata_json = json.dumps({
        "analysis": (
            {
                "intent": analysis.intent,
                "sender_relationship": analysis.sender_relationship,
                "key_points": analysis.key_points,
                "tone_signals": analysis.tone_signals,
                "length_target": analysis.length_target,
                "open_questions": analysis.open_questions,
            } if analysis else None
        ),
        "alternative_text": output.alternative,
        "reasoning": output.reasoning,
        "confidence": output.confidence,
        "open_questions": output.open_questions,
        "sender_email": sender_email,
        "thread_messages": len(thread_hist),
        "sender_messages": len(sender_hist),
        "schema_version": 2,
    })
    cur = conn.execute(
        "INSERT INTO email_drafts"
        "(file_id, draft_text, generated_at, metadata_json) "
        "VALUES (?, ?, ?, ?) RETURNING id",
        (file_id, output.primary, time.time(), metadata_json),
    )
    rid = int(cur.fetchone()["id"])
    conn.commit()
    return Draft(
        id=rid, file_id=file_id, draft_text=output.primary,
        generated_at=time.time(),
        analysis=analysis,
        alternative_text=output.alternative,
        reasoning=output.reasoning,
        open_questions=output.open_questions,
        confidence=output.confidence,
    )


def _persist_legacy_draft(
    conn, file_id, drafter, from_, subject, body, user_name, cfg,
) -> Draft | None:
    """Test-only path: a stubbed drafter (callable returning plain
    text) bypasses analysis + retrieval. Keeps the existing tests
    fast + deterministic; production code never hits this branch."""
    style_samples = _gather_style_samples(conn)
    try:
        text = drafter(
            from_=from_, subject=subject, body=body,
            style_samples=style_samples,
            user_name=user_name, cfg=cfg,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("email_assist: legacy drafter crashed: %s", e)
        return None
    text = (text or "").strip()
    if not text:
        return None
    cur = conn.execute(
        "INSERT INTO email_drafts(file_id, draft_text, generated_at) "
        "VALUES (?, ?, ?) RETURNING id",
        (file_id, text, time.time()),
    )
    rid = int(cur.fetchone()["id"])
    conn.commit()
    return Draft(
        id=rid, file_id=file_id, draft_text=text,
        generated_at=time.time(),
    )


def _default_drafter(
    *,
    from_, subject, body, style_samples, user_name, cfg,
    analysis: EmailAnalysis | None = None,
    thread_history: list[tuple[str, str]] | None = None,
    sender_history: list[tuple[str, str]] | None = None,
    brain_context: str = "",
) -> DraftOutput | None:
    """Round 6 — structured drafter via Claude Sonnet.

    Takes the analyzer's plan + retrieval bundle and asks Sonnet for
    a JSON-shaped output: primary draft + alternative-tone version
    + reasoning + confidence + filtered open_questions.

    Falls back to local Ollama for the same prompt when Anthropic
    is unavailable. Local models are mediocre at structured-JSON
    output, but a degraded primary draft (with empty alternative)
    still beats nothing.

    Returns None when both paths fail or the JSON is unparseable —
    caller handles by skipping the persist.
    """
    body_clip = body if len(body) <= 6000 else body[:6000] + "…"
    analysis_block = (
        _format_analysis_block(analysis) if analysis is not None
        else "(analyzer unavailable; infer from email body)"
    )
    thread_block = _format_history_block(thread_history or [])
    sender_block = _format_history_block(sender_history or [])
    brain_block = brain_context or "(no relevant brain context found)"

    prompt = _DRAFT_PROMPT_V2.format(
        user_name=user_name,
        from_=from_ or "(unknown)",
        subject=subject or "(no subject)",
        body=body_clip,
        analysis_block=analysis_block,
        thread_block=thread_block,
        sender_block=sender_block,
        style_samples=style_samples,
        brain_block=brain_block,
    )

    parsed = _llm_json_call(
        prompt=prompt,
        cfg=cfg,
        model="claude-sonnet-4-6",
        max_tokens=1200,
        feature="email_draft",
        note=f"email_draft/{(from_ or '')[:30]}",
    )
    if parsed is None:
        return None

    primary = str(parsed.get("primary") or "").strip()
    alternative = str(parsed.get("alternative") or "").strip()
    reasoning = str(parsed.get("reasoning") or "").strip()
    try:
        confidence = float(parsed.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    open_questions = [
        str(q) for q in (parsed.get("open_questions") or [])
        if str(q).strip()
    ]
    if not (primary or alternative):
        # Nothing usable came back — let the caller drop it on the floor.
        return None
    return DraftOutput(
        primary=primary,
        alternative=alternative,
        reasoning=reasoning,
        confidence=confidence,
        open_questions=open_questions,
    )


def list_unsent_drafts(
    conn: sqlite3.Connection, *, limit: int = 50,
) -> list[Draft]:
    _ensure_schema(conn)
    rows = conn.execute(
        "SELECT * FROM email_drafts WHERE sent_at IS NULL "
        "ORDER BY generated_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_row_to_draft(r) for r in rows]


def _row_to_draft(r) -> Draft:
    """Hydrate a Draft dataclass, parsing the optional metadata_json
    column for round-6 structured drafts. Legacy rows (NULL metadata)
    just fill in draft_text + sent_at and leave the new fields None."""
    raw_meta = None
    try:
        raw_meta = r["metadata_json"]
    except (IndexError, KeyError):
        # ALTER ADD COLUMN not yet applied (very old db) — silently skip.
        pass
    analysis = None
    alt_text = None
    reasoning = None
    open_qs: list[str] = []
    confidence = None
    if raw_meta:
        try:
            meta = json.loads(raw_meta)
        except (TypeError, ValueError):
            meta = {}
        if isinstance(meta, dict):
            a = meta.get("analysis")
            if isinstance(a, dict):
                try:
                    analysis = EmailAnalysis(
                        intent=str(a.get("intent") or "other"),
                        sender_relationship=str(
                            a.get("sender_relationship") or "unknown",
                        ),
                        key_points=[
                            str(x) for x in (a.get("key_points") or [])
                        ],
                        tone_signals=[
                            str(x) for x in (a.get("tone_signals") or [])
                        ],
                        length_target=str(
                            a.get("length_target") or "medium",
                        ),
                        open_questions=[
                            str(x) for x in (a.get("open_questions") or [])
                        ],
                    )
                except Exception:  # noqa: BLE001
                    analysis = None
            alt_text = meta.get("alternative_text") or None
            reasoning = meta.get("reasoning") or None
            open_qs = [str(q) for q in (meta.get("open_questions") or [])]
            try:
                if meta.get("confidence") is not None:
                    confidence = float(meta["confidence"])
            except (TypeError, ValueError):
                confidence = None
    return Draft(
        id=int(r["id"]), file_id=int(r["file_id"]),
        draft_text=r["draft_text"],
        generated_at=r["generated_at"],
        sent_at=r["sent_at"],
        analysis=analysis,
        alternative_text=alt_text,
        reasoning=reasoning,
        open_questions=open_qs,
        confidence=confidence,
    )


def mark_draft_sent(conn: sqlite3.Connection, draft_id: int) -> bool:
    _ensure_schema(conn)
    cur = conn.execute(
        "UPDATE email_drafts SET sent_at = ? "
        "WHERE id = ? AND sent_at IS NULL",
        (time.time(), draft_id),
    )
    conn.commit()
    return cur.rowcount > 0


def discard_draft(conn: sqlite3.Connection, draft_id: int) -> bool:
    """Hard-delete an unsent draft. Used when the user decided to
    write the reply themselves."""
    _ensure_schema(conn)
    cur = conn.execute(
        "DELETE FROM email_drafts WHERE id = ? AND sent_at IS NULL",
        (draft_id,),
    )
    conn.commit()
    return cur.rowcount > 0


def generate_drafts_due(
    conn: sqlite3.Connection, cfg,
    *, max_per_tick: int = _DRAFT_PER_TICK,
    drafter=None,
) -> int:
    """Daemon entrypoint. Find draftable emails without a draft +
    generate."""
    _ensure_schema(conn)
    rows = conn.execute(
        "SELECT ec.file_id FROM email_classifications ec "
        "LEFT JOIN email_drafts ed "
        "  ON ed.file_id = ec.file_id AND ed.sent_at IS NULL "
        "WHERE ec.label IN ('urgent', 'response') "
        "  AND ed.file_id IS NULL "
        "ORDER BY ec.classified_at DESC LIMIT ?",
        (max_per_tick,),
    ).fetchall()
    n = 0
    for r in rows:
        if generate_draft(
            conn, int(r["file_id"]), cfg=cfg, drafter=drafter,
        ) is not None:
            n += 1
    if n:
        log.info("email_assist: generated %d draft(s)", n)
    return n
