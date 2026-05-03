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
import os
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


def _safe_for_prompt(text: str | None, *, max_chars: int) -> str:
    """Round 10 (#4) — redact-then-truncate before user content
    enters an LLM prompt that ships to Anthropic.

    Phase 88's ``redact_text`` previously fired only at *render* time;
    this is the prompt-assembly companion. Order matters: redact first
    so we don't truncate mid-redaction-marker, then trim. Idempotent
    and cheap (regex-only).
    """
    if not text:
        return ""
    try:
        from .safety import redact_text
        clean = redact_text(text)
    except ImportError:
        clean = text
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars] + "…"


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
    # Round 10 (#2) — draft feedback columns. ``feedback`` is NULL
    # when the user hasn't acted yet; one of 'accepted' / 'rejected'
    # / 'edited' once they do. ``rejection_reason`` lets the user
    # leave a one-line note ('too formal', 'wrong tone') that the
    # weekly stats surface as patterns to fix in the drafter.
    for col, ddl in (
        ("feedback", "ALTER TABLE email_drafts ADD COLUMN feedback TEXT"),
        ("rejection_reason",
         "ALTER TABLE email_drafts ADD COLUMN rejection_reason TEXT"),
    ):
        try:
            conn.execute(ddl)
            conn.commit()
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e).lower():
                log.debug("email_assist: ALTER add %s skipped: %s", col, e)
    # Round 7 — voice profile + reply pairs. The profile is a single
    # JSON blob refreshed weekly; reply pairs link a Sent message to
    # the email it replied to so few-shot retrieval can surface real
    # (incoming, user_reply) examples to the drafter.
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS email_style_profile (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            profile_json TEXT NOT NULL,
            sent_count INTEGER NOT NULL,    -- how many sent items fed the profile
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS email_reply_pairs (
            -- One row per (incoming_email, user_reply) pair detected in
            -- the index. UNIQUE on the reply so we don't double-link a
            -- single Sent message.
            reply_file_id INTEGER PRIMARY KEY
                REFERENCES files(id) ON DELETE CASCADE,
            incoming_file_id INTEGER NOT NULL
                REFERENCES files(id) ON DELETE CASCADE,
            link_method TEXT NOT NULL,      -- 'in-reply-to' | 'thread' | 'subject'
            indexed_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_reply_pairs_incoming
            ON email_reply_pairs(incoming_file_id);
    """)
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


def _default_classifier(
    from_, subject, body, cfg,
    *,
    conn: sqlite3.Connection | None = None,
    file_id: int | None = None,
) -> dict:
    """Real Haiku classifier. Bounded body length to keep cost tight.

    Phase 89 wiring: when Anthropic isn't usable, fall back to a
    local Ollama call asking for the single label string. Confidence
    drops because local models are noisier on classification, but
    the daemon stays running.
    """
    import os

    # Round 10 (#4 fix) — redact raw email body before sending to
    # Anthropic. Phase 88 patterns (API keys, JWTs, SSNs) get masked
    # at prompt-assembly time, not just at render time. The triage
    # classifier doesn't need to see secrets — it works on subject /
    # sender / general tone.
    body_clip = _safe_for_prompt(body, max_chars=4000)
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

DRAFTING RULES — VOICE FIDELITY IS THE PRIMARY GOAL
==============
- {user_name}'s voice profile (computed from their actual sent
  emails) is the source of truth. Match it concretely:
    · Open with one of the observed greeting patterns.
    · Close with one of the observed sign-off patterns.
    · Use contractions at the observed rate.
    · Match observed sentence length within +/- 30%.
    · NEVER use any phrase listed in "avoid these phrases" — those
      were checked against the user's corpus and they don't use them.
- The FEW-SHOT EXAMPLES below are real (incoming, user_reply) pairs.
  Mimic the structural pattern of the closest example: how {user_name}
  opened, how long they replied, where they put commitments, how they
  signed off. Don't copy facts — just structure + voice.
- The PRIMARY draft uses the analyzer's ``length_target`` and
  ``tone_signals``. The ALTERNATIVE flips the tone register
  (formal ↔ casual) so the user can pick.
- For unknowns the user must fill in, use ``<TODO: brief description>``
  inline (e.g. ``Tuesday at <TODO: pick a time> works for me``).
- DO NOT invent commitments, dates, prices, or facts. When unsure,
  use a TODO placeholder.
- If the email is purely informational and doesn't need a reply,
  set primary to "" and confidence to 0.0.
- Reference the prior thread + sender history when natural ("re your
  earlier point about X", "as we discussed last week") — but only
  when the prior actually supports it.

VOICE PROFILE — match these patterns
==============
{voice_profile_block}

FEW-SHOT EXAMPLES — how {user_name} actually replied to similar emails
==============
{fewshot_block}

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

STYLE REFERENCE — extra raw sent-message context
==============
{style_samples}

BRAIN CONTEXT (relevant snippets from {user_name}'s knowledge base)
==============
{brain_block}
"""


def analyze_email(
    *, from_: str, subject: str, body: str, cfg,
    conn: sqlite3.Connection | None = None,
    file_id: int | None = None,
) -> EmailAnalysis | None:
    """Round 6 — structured analysis of an incoming email.

    Single Haiku call returning JSON with intent / relationship /
    asks / tone / open_questions. Used by the new drafter pipeline
    to plan the reply before writing it.

    Returns None when neither Anthropic nor the local LLM produces
    parseable JSON — caller falls back to the legacy single-call
    drafter so we never hard-fail on a daemon tick.
    """

    # Round 10 (#4) — redact before send, same rationale as classifier.
    body_clip = _safe_for_prompt(body, max_chars=4000)
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
        conn=conn,
        audit_kind="analyze",
        audit_summary=f"analyzed email from {from_[:60]!r}",
        audit_file_id=file_id,
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
    conn: sqlite3.Connection | None = None,
    audit_kind: str = "",
    audit_summary: str = "",
    audit_file_id: int | None = None,
    audit_person_id: int | None = None,
) -> dict | None:
    """Shared helper: try Anthropic with budget guard, fall back to
    local Ollama, parse JSON. Returns the parsed dict or None.

    Round 10 (#6) — when ``conn`` is given, every call records one
    row in ``ai_actions`` with status (success / fallback_local /
    budget_exceeded / api_error / no_provider / parse_error) + cost
    + chars + model. Audit fields are optional kwargs so existing
    callers keep working unchanged; daemon-owned + dashboard call
    sites pass them.
    """
    import os
    text = ""
    used_model = model
    status = "success"
    err_msg = ""
    cents_spent = 0.0
    response_text = ""
    final_kind = audit_kind or feature

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
                # Approx cost in cents using the budget module's table.
                try:
                    from .budget import estimate_cost
                    cents_spent = estimate_cost(
                        model,
                        input_tokens=resp.usage.input_tokens,
                        output_tokens=resp.usage.output_tokens,
                    ).cents
                except Exception:  # noqa: BLE001
                    cents_spent = 0.0
                text = "".join(
                    b.text for b in resp.content if b.type == "text"
                ).strip()
            except BudgetExceededError as e:
                log.info(
                    "email_assist: %s budget exhausted, trying local: %s",
                    feature, e,
                )
                status = "budget_exceeded"
                err_msg = str(e)
            except anthropic.APIError as e:
                log.info(
                    "email_assist: %s API error, trying local: %s",
                    feature, e,
                )
                status = "api_error"
                err_msg = str(e)
        else:
            status = "no_provider"
    else:
        status = "no_provider"

    # ---- Fallback: local Ollama ----
    if not text:
        try:
            from . import local_llm
        except ImportError:
            _maybe_audit(
                conn, kind=final_kind, feature=feature, model=used_model,
                status="no_provider", prompt_chars=len(prompt),
                response_chars=0, cents=cents_spent,
                summary=audit_summary, error="local_llm import failed",
                file_id=audit_file_id, person_id=audit_person_id,
            )
            return None
        if not local_llm.is_available(cfg):
            _maybe_audit(
                conn, kind=final_kind, feature=feature, model=used_model,
                status=status if status != "success" else "no_provider",
                prompt_chars=len(prompt), response_chars=0,
                cents=cents_spent, summary=audit_summary,
                error=err_msg or "no LLM available",
                file_id=audit_file_id, person_id=audit_person_id,
            )
            return None
        out = local_llm.complete(prompt, cfg=cfg, max_tokens=max_tokens)
        if out is None:
            _maybe_audit(
                conn, kind=final_kind, feature=feature, model=used_model,
                status=status if status != "success" else "api_error",
                prompt_chars=len(prompt), response_chars=0,
                cents=cents_spent, summary=audit_summary,
                error=err_msg or "local llm returned None",
                file_id=audit_file_id, person_id=audit_person_id,
            )
            return None
        text = out.text.strip()
        used_model = out.model
        status = "fallback_local"
        log.info("email_assist: %s via local LLM (%s)", feature, out.model)

    response_text = text
    if not text:
        _maybe_audit(
            conn, kind=final_kind, feature=feature, model=used_model,
            status="parse_error", prompt_chars=len(prompt),
            response_chars=0, cents=cents_spent, summary=audit_summary,
            error="empty response",
            file_id=audit_file_id, person_id=audit_person_id,
        )
        return None
    if text.startswith("```"):
        text = re.sub(r"^```\w*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        log.info("email_assist: %s JSON parse failed; raw=%r", feature, text[:200])
        _maybe_audit(
            conn, kind=final_kind, feature=feature, model=used_model,
            status="parse_error", prompt_chars=len(prompt),
            response_chars=len(response_text), cents=cents_spent,
            summary=audit_summary, error="JSON parse failed",
            file_id=audit_file_id, person_id=audit_person_id,
        )
        return None
    if not isinstance(parsed, dict):
        _maybe_audit(
            conn, kind=final_kind, feature=feature, model=used_model,
            status="parse_error", prompt_chars=len(prompt),
            response_chars=len(response_text), cents=cents_spent,
            summary=audit_summary, error="not a JSON object",
            file_id=audit_file_id, person_id=audit_person_id,
        )
        return None
    _maybe_audit(
        conn, kind=final_kind, feature=feature, model=used_model,
        status=status, prompt_chars=len(prompt),
        response_chars=len(response_text), cents=cents_spent,
        summary=audit_summary, error="",
        file_id=audit_file_id, person_id=audit_person_id,
    )
    return parsed


def _maybe_audit(conn, **kwargs) -> None:
    """Log to ai_actions when a conn is available; silent no-op
    otherwise. Keeps audit logging optional + crash-safe."""
    if conn is None:
        return
    try:
        from . import ai_audit
        ai_audit.record_action(conn, **kwargs)
    except Exception:  # noqa: BLE001
        pass


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
    """Generate a reply draft using the structured pipeline:

      1. ``analyze_email`` — Haiku JSON: intent / relationship /
         key_points / tone / open_questions.
      2. Targeted retrieval — thread history, sender history,
         relationship-scoped style samples, brain context for the
         topics the analyzer identified.
      3. ``_default_drafter`` — Sonnet, fed all of the above; outputs
         primary + alternative + reasoning + open_questions in JSON.

    Round 10 (#7) — the ``drafter`` parameter is now a stub that
    REPLACES ``_default_drafter`` in step 3 (not a separate code
    path). Same kwargs, same return contract. For backwards-compat
    with older tests, a stub that returns a plain string gets
    auto-wrapped into ``DraftOutput(primary=str)``.
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

    # ---- Stage 1: analyze ----
    # Skip analysis when cfg is missing (test path) — _default_drafter
    # / stub still gets called with analysis=None.
    analysis = None
    if cfg is not None and drafter is None:
        analysis = analyze_email(
            from_=from_, subject=subject, body=body, cfg=cfg,
            conn=conn, file_id=file_id,
        )
    # ---- Stage 2: targeted retrieval (round 6 + 7) ----
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
    brain_block = ""
    if cfg is not None and drafter is None:
        brain_block = _brain_context_for_topics(conn, cfg, key_points)
    # Round 7 — voice fidelity inputs. Round 10 (#5): use the
    # curated bootstrap profile when no real one exists yet so cold-
    # start drafts don't drop to generic-LLM voice.
    voice_profile = get_voice_profile_or_default(conn)
    embedder = None
    if cfg is not None and drafter is None:
        try:
            from .embedder import make_embedder
            embedder = make_embedder(cfg)
        except Exception:  # noqa: BLE001
            embedder = None
    fewshot = fewshot_reply_pairs(
        conn, incoming_text=body, embedder=embedder,
    )

    # ---- Stage 3: draft (production OR injected stub) ----
    drafter_fn = drafter if drafter is not None else _default_drafter
    try:
        raw_output = drafter_fn(
            from_=from_, subject=subject, body=body,
            style_samples=style_samples,
            user_name=user_name, cfg=cfg,
            analysis=analysis,
            thread_history=thread_hist,
            sender_history=sender_hist,
            brain_context=brain_block,
            voice_profile=voice_profile,
            fewshot_pairs=fewshot,
            conn=conn,
            file_id=file_id,
        )
    except TypeError:
        # Backwards-compat: older test stubs use ``**kw`` and don't
        # accept the new kwargs. Call with a minimal shape.
        raw_output = drafter_fn(
            from_=from_, subject=subject, body=body,
            style_samples=style_samples,
            user_name=user_name, cfg=cfg,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("email_assist: drafter crashed: %s", e)
        return None
    output = _coerce_draft_output(raw_output)
    if output is None or not (output.primary or output.alternative).strip():
        return None

    # ---- Stage 4: voice critique + at-most-one regenerate ----
    # Only runs when we have a profile to critique against. The
    # regenerate path passes the critique back as a do-not-do list
    # so the second draft addresses the specific mismatches.
    # Round 10 (#5) — only critique against a REAL profile (n_samples
    # > 0). The bootstrap default has nothing user-specific to grade
    # the draft against, so a critique pass is wasted spend.
    critique_text = ""
    if (
        voice_profile is not None
        and voice_profile.n_samples > 0
        and output.primary.strip()
    ):
        critique_text = critique_draft_against_voice(
            draft=output.primary, profile=voice_profile, cfg=cfg,
            conn=conn, file_id=file_id,
        )
        if critique_text and critique_text.strip().upper() != "OK":
            log.info(
                "email_assist: voice critique flagged mismatches; "
                "regenerating once",
            )
            output2 = _regenerate_with_critique(
                from_=from_, subject=subject, body=body,
                style_samples=style_samples,
                user_name=user_name, cfg=cfg,
                analysis=analysis,
                thread_history=thread_hist,
                sender_history=sender_hist,
                brain_context=brain_block,
                voice_profile=voice_profile,
                fewshot_pairs=fewshot,
                prior_draft=output.primary,
                critique=critique_text,
            )
            if output2 is not None and output2.primary.strip():
                output = output2

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
        # Round 7 — voice-fidelity fields.
        "voice_profile_n_samples": (
            voice_profile.n_samples if voice_profile else 0
        ),
        "fewshot_pairs": len(fewshot),
        "voice_critique": critique_text or "",
        "schema_version": 3,
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


def _coerce_draft_output(raw) -> DraftOutput | None:
    """Round 10 (#7) — adapter for backwards-compatible test stubs.

    The new contract is that drafters return ``DraftOutput | None``,
    but older tests pass stubs that return plain strings (the legacy
    single-call drafter shape). Wrap those into the new shape so
    we don't have to touch every test simultaneously.
    """
    if raw is None:
        return None
    if isinstance(raw, DraftOutput):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        return DraftOutput(
            primary=text, alternative="", reasoning="",
            confidence=0.0, open_questions=[],
        )
    if isinstance(raw, dict):
        # Some test stubs return the raw JSON shape directly. Hydrate
        # via the same path the production drafter uses.
        primary = str(raw.get("primary") or "").strip()
        if not primary and not raw.get("alternative"):
            return None
        try:
            confidence = float(raw.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        return DraftOutput(
            primary=primary,
            alternative=str(raw.get("alternative") or "").strip(),
            reasoning=str(raw.get("reasoning") or ""),
            confidence=max(0.0, min(1.0, confidence)),
            open_questions=[
                str(q) for q in (raw.get("open_questions") or [])
                if str(q).strip()
            ],
        )
    return None


def _default_drafter(
    *,
    from_, subject, body, style_samples, user_name, cfg,
    analysis: EmailAnalysis | None = None,
    thread_history: list[tuple[str, str]] | None = None,
    sender_history: list[tuple[str, str]] | None = None,
    brain_context: str = "",
    voice_profile: VoiceProfile | None = None,
    fewshot_pairs: list[tuple[str, str]] | None = None,
    conn: sqlite3.Connection | None = None,
    file_id: int | None = None,
) -> DraftOutput | None:
    """Round 6 + 7 — structured drafter via Claude Sonnet.

    Takes the analyzer's plan + retrieval bundle + the user's
    extracted voice profile + few-shot reply pairs, and asks Sonnet
    for JSON-shaped output: primary draft + alternative-tone version
    + reasoning + confidence + filtered open_questions.

    Voice profile + few-shot pairs are the round-7 additions that
    make drafts actually sound like the user. Without them the
    drafter falls back to default voice rules.

    Falls back to local Ollama for the same prompt when Anthropic
    is unavailable. Local models are mediocre at structured-JSON
    output, but a degraded primary draft (with empty alternative)
    still beats nothing.

    Returns None when both paths fail or the JSON is unparseable —
    caller handles by skipping the persist.
    """
    # Round 10 (#4) — redact every raw-content field before send.
    # The drafter prompt is the most data-heavy LLM call in the
    # codebase (incoming email + thread history + sender history +
    # user's sent-mail style samples + brain search hits) so this
    # is where prompt-side redaction matters most.
    body_clip = _safe_for_prompt(body, max_chars=6000)
    analysis_block = (
        _format_analysis_block(analysis) if analysis is not None
        else "(analyzer unavailable; infer from email body)"
    )
    thread_block = _safe_for_prompt(
        _format_history_block(thread_history or []), max_chars=8000,
    )
    sender_block = _safe_for_prompt(
        _format_history_block(sender_history or []), max_chars=8000,
    )
    brain_block = _safe_for_prompt(
        brain_context or "(no relevant brain context found)",
        max_chars=4000,
    )
    voice_block = (
        _format_voice_profile_block(voice_profile)
        if voice_profile is not None
        else "(no voice profile yet — fall back to general rules: be "
        "concrete, match the inferred tone, avoid generic LLM phrases)"
    )
    # Style samples + few-shot pairs are user-authored sent mail,
    # which is the highest-signal but also highest-risk content.
    # Redact while preserving voice patterns (Phase 88 only masks
    # secret-shaped substrings — sentence rhythm + sign-offs survive).
    style_samples_clean = _safe_for_prompt(style_samples, max_chars=6000)
    fewshot_block = _safe_for_prompt(
        _format_fewshot_block(fewshot_pairs or []),
        max_chars=8000,
    )

    prompt = _DRAFT_PROMPT_V2.format(
        user_name=user_name,
        from_=from_ or "(unknown)",
        subject=subject or "(no subject)",
        body=body_clip,
        analysis_block=analysis_block,
        thread_block=thread_block,
        sender_block=sender_block,
        style_samples=style_samples_clean,
        brain_block=brain_block,
        voice_profile_block=voice_block,
        fewshot_block=fewshot_block,
    )

    parsed = _llm_json_call(
        prompt=prompt,
        cfg=cfg,
        model="claude-sonnet-4-6",
        max_tokens=1200,
        feature="email_draft",
        note=f"email_draft/{(from_ or '')[:30]}",
        conn=conn,
        audit_kind="draft",
        audit_summary=f"drafted reply to {from_[:60]!r}",
        audit_file_id=file_id,
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
    """Round 10 (#2) — filter out rejected drafts (they're soft-
    deleted now to preserve the feedback signal, but the user
    doesn't want to see them again)."""
    _ensure_schema(conn)
    rows = conn.execute(
        "SELECT * FROM email_drafts "
        "WHERE sent_at IS NULL "
        "  AND (feedback IS NULL OR feedback != 'rejected') "
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
    """Round 10 (#2) — also flag feedback='accepted' so the rolling
    accept-rate stat reflects this draft."""
    _ensure_schema(conn)
    cur = conn.execute(
        "UPDATE email_drafts "
        "SET sent_at = ?, feedback = COALESCE(feedback, 'accepted') "
        "WHERE id = ? AND sent_at IS NULL",
        (time.time(), draft_id),
    )
    conn.commit()
    return cur.rowcount > 0


def discard_draft(
    conn: sqlite3.Connection, draft_id: int,
    *, reason: str = "",
) -> bool:
    """Round 10 (#2) — soft-deletion: mark feedback='rejected' so the
    weekly stats can show 'you rejected X% of drafts'.

    Previously discarded drafts were hard-deleted, which lost the
    signal we need to evaluate the drafter. They still vanish from
    ``list_unsent_drafts`` because that filters on sent_at IS NULL —
    we now also exclude rows with feedback='rejected'.
    """
    _ensure_schema(conn)
    cur = conn.execute(
        "UPDATE email_drafts "
        "SET feedback = 'rejected', rejection_reason = ? "
        "WHERE id = ? AND sent_at IS NULL",
        (reason[:500] if reason else None, draft_id),
    )
    conn.commit()
    return cur.rowcount > 0


def feedback_stats(
    conn: sqlite3.Connection, *, days: int = 7,
) -> dict[str, int | float]:
    """Round 10 (#2) — accept / reject / pending counts over the
    last N days. Powers the /drafts header stat + the morning brief.

    Returns dict with keys: accepted, rejected, pending, total,
    accept_rate (0..1, computed from accepted/(accepted+rejected),
    not against pending). 0.0 when no acted-on drafts."""
    _ensure_schema(conn)
    cutoff = time.time() - days * 86400
    rows = conn.execute(
        "SELECT feedback, COUNT(*) AS n FROM email_drafts "
        "WHERE generated_at >= ? "
        "GROUP BY feedback",
        (cutoff,),
    ).fetchall()
    counts = {r["feedback"] or "pending": int(r["n"]) for r in rows}
    accepted = counts.get("accepted", 0)
    rejected = counts.get("rejected", 0)
    pending = counts.get("pending", 0) + counts.get("edited", 0)
    total = accepted + rejected + pending
    acted = accepted + rejected
    accept_rate = accepted / acted if acted else 0.0
    return {
        "accepted": accepted,
        "rejected": rejected,
        "pending": pending,
        "total": total,
        "accept_rate": accept_rate,
    }


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


# ============================================================
# Round 7 — voice fidelity. Three pieces:
#   1. Reply-pair indexer: links each Sent email to the email it
#      replied to, persisted in email_reply_pairs.
#   2. Voice-profile extractor: scans Sent items, extracts structural
#      patterns (greeting / sign-off / sentence length / opener &
#      closer phrases / tone register) into one JSON blob.
#   3. Few-shot retrieval at draft time: semantic search over linked
#      parent emails to pull the (incoming, user_reply) pairs most
#      similar to the current incoming email — so the drafter sees
#      real examples of how the user replies to similar messages.
# ============================================================

# How many Sent items to scan when extracting the voice profile.
# More gives better stats, but the qualitative LLM pass costs more
# tokens. 50 is a sweet spot for typical inboxes.
_VOICE_PROFILE_MAX_SAMPLES = 50
# Refresh cadence — daemon hook re-runs at most once a week.
_VOICE_PROFILE_REFRESH_DAYS = 7
# Few-shot pairs to feed the drafter. Three is enough variation
# without ballooning the prompt; each pair clips at ~1200 chars.
_FEWSHOT_PAIR_K = 3
_FEWSHOT_PAIR_CHARS = 1200
# In-Reply-To / References parsing — emails embed these as headers
# the IMAP connector renders into the message body. We grep for them
# rather than re-parse the raw email so this works on whatever the
# connector materialised.
_INREPLYTO_RE = re.compile(
    r"^In-Reply-To:\s*<?([^>\s]+)>?", re.MULTILINE | re.IGNORECASE,
)
_MSGID_LINE_RE = re.compile(
    r"^Message-ID:\s*<?([^>\s]+)>?", re.MULTILINE | re.IGNORECASE,
)


def _is_sent_item(text: str) -> bool:
    """Sent items have 'Folder: Sent' or '/Sent' in their first chunk —
    that's how the IMAP / Gmail connectors mark outbound mail."""
    if not text:
        return False
    head = text[:2000].lower()
    return ("folder: sent" in head
            or "labels: sent" in head
            or "/sent" in head)


def _msgid_for_file(conn: sqlite3.Connection, file_id: int) -> str:
    """Pull the Message-ID rendered into the email body. Returns ""
    when the email pre-dates the connector that emits the header
    (legacy IMAP rows) — that just means we fall back to subject-
    based linking."""
    row = conn.execute(
        "SELECT text FROM chunks WHERE file_id = ? "
        "ORDER BY chunk_index ASC LIMIT 1",
        (file_id,),
    ).fetchone()
    if row is None:
        return ""
    text = row["text"] or ""
    m = _MSGID_LINE_RE.search(text)
    return (m.group(1) if m else "").strip()


def _inreplyto_for_file(conn: sqlite3.Connection, file_id: int) -> str:
    """Pull the In-Reply-To header value if rendered in the body."""
    row = conn.execute(
        "SELECT text FROM chunks WHERE file_id = ? "
        "ORDER BY chunk_index ASC LIMIT 1",
        (file_id,),
    ).fetchone()
    if row is None:
        return ""
    text = row["text"] or ""
    m = _INREPLYTO_RE.search(text)
    return (m.group(1) if m else "").strip()


def _find_parent_for_reply(
    conn: sqlite3.Connection, *, reply_file_id: int, subject: str,
) -> tuple[int | None, str]:
    """Locate the email a Sent message is replying to.

    Tries three strategies in order — first match wins:
      1. ``In-Reply-To`` header in body → exact lookup by Message-ID
         line in the candidate parent.
      2. Gmail thread_id sibling: the most recent non-Sent message
         in the same gmail://thread/<tid>/.
      3. Subject heuristic: most recent non-Sent message whose H1
         normalises to the same root subject.

    Returns ``(parent_file_id_or_None, link_method)``.
    """
    # ---- Strategy 1: In-Reply-To ----
    irt = _inreplyto_for_file(conn, reply_file_id)
    if irt:
        # Look for any file whose body has Message-ID: <irt>.
        cand = conn.execute(
            "SELECT f.id FROM files f JOIN chunks c ON c.file_id = f.id "
            "WHERE (f.path LIKE 'imap://%' OR f.path LIKE 'gmail://%') "
            "  AND c.chunk_index = 0 "
            "  AND c.text LIKE ? "
            "  AND f.id != ? "
            "LIMIT 1",
            (f"%Message-ID:%{irt}%", reply_file_id),
        ).fetchone()
        if cand:
            return int(cand["id"]), "in-reply-to"
    # ---- Strategy 2: Gmail thread sibling ----
    src_path_row = conn.execute(
        "SELECT path FROM files WHERE id = ?", (reply_file_id,),
    ).fetchone()
    src_path = src_path_row["path"] if src_path_row else ""
    tid = _gmail_thread_id_from_path(src_path or "")
    if tid:
        cand = conn.execute(
            "SELECT f.id, c.text FROM files f "
            "JOIN chunks c ON c.file_id = f.id "
            "WHERE f.path LIKE ? AND f.id != ? AND c.chunk_index = 0 "
            "ORDER BY f.mtime DESC LIMIT 5",
            (f"gmail://thread/{tid}/%", reply_file_id),
        ).fetchall()
        # Skip other Sent items in the thread — we want incoming.
        for r in cand:
            if not _is_sent_item(r["text"] or ""):
                return int(r["id"]), "thread"
    # ---- Strategy 3: subject heuristic ----
    norm = _normalize_subject(subject)
    if not norm:
        return None, "none"
    cand = conn.execute(
        "SELECT f.id, c.text FROM files f "
        "JOIN chunks c ON c.file_id = f.id "
        "WHERE (f.path LIKE 'imap://%' OR f.path LIKE 'gmail://%') "
        "  AND c.chunk_index = 0 "
        "  AND f.id != ? "
        "  AND LOWER(c.text) LIKE ? "
        "ORDER BY f.mtime DESC LIMIT 5",
        (reply_file_id, f"%# %{norm}%"),
    ).fetchall()
    for r in cand:
        if not _is_sent_item(r["text"] or ""):
            return int(r["id"]), "subject"
    return None, "none"


def index_reply_pairs(
    conn: sqlite3.Connection, *, max_per_run: int = 200,
) -> int:
    """Round 7 — link each new Sent email to the email it replied to.

    Walks unlinked Sent items (newest first) and writes one row per
    pair into ``email_reply_pairs``. Bounded by ``max_per_run`` so a
    huge initial backlog doesn't block a daemon tick. Returns the
    number of new pairs created.
    """
    _ensure_schema(conn)
    rows = conn.execute(
        "SELECT f.id, c.text FROM files f "
        "JOIN chunks c ON c.file_id = f.id "
        "LEFT JOIN email_reply_pairs p ON p.reply_file_id = f.id "
        "WHERE (f.path LIKE 'imap://%' OR f.path LIKE 'gmail://%') "
        "  AND c.chunk_index = 0 "
        "  AND p.reply_file_id IS NULL "
        "ORDER BY f.mtime DESC LIMIT ?",
        (max_per_run * 3,),  # over-fetch since many won't be Sent
    ).fetchall()
    n_new = 0
    for r in rows:
        if not _is_sent_item(r["text"] or ""):
            continue
        fid = int(r["id"])
        # Pull subject from the chunk text's H1.
        subject = ""
        for ln in (r["text"] or "").splitlines()[:5]:
            s = ln.strip()
            if s.startswith("# "):
                subject = s[2:].strip()
                break
        parent_id, method = _find_parent_for_reply(
            conn, reply_file_id=fid, subject=subject,
        )
        if parent_id is None:
            continue
        try:
            conn.execute(
                "INSERT OR IGNORE INTO email_reply_pairs"
                "(reply_file_id, incoming_file_id, link_method, indexed_at) "
                "VALUES (?, ?, ?, ?)",
                (fid, parent_id, method, time.time()),
            )
            n_new += 1
        except sqlite3.IntegrityError:
            # Race / dup — the UNIQUE on reply_file_id covers it.
            pass
        if n_new >= max_per_run:
            break
    if n_new:
        conn.commit()
        log.info("email_assist: indexed %d reply pair(s)", n_new)
    return n_new


# ---- Voice profile extraction ----------------------------------------

@dataclass
class VoiceProfile:
    """Round 7 — structured snapshot of how the user actually writes
    email replies. Stored as a single JSON blob, refreshed weekly.

    The fields are deliberately concrete + checkable. The drafter
    prompt renders these as bullet points and the critique pass
    flags drafts that violate them."""
    greetings: list[str]          # eg ['hi {name},', '{name},', '(no greeting)']
    sign_offs: list[str]          # eg ['—Ben', 'thanks,\nBen', 'Ben']
    avg_sentence_words: float     # mean across all sentences
    avg_reply_chars: int          # mean reply length
    contraction_rate: float       # fraction of "I'd" vs "I would" etc
    exclamation_rate: float       # exclamations per reply
    emoji_rate: float             # emojis per reply
    common_openers: list[str]     # frequent first sentences / phrases
    common_closers: list[str]     # frequent final sentences
    avoided_phrases: list[str]    # generic LLM-isms the user *never* uses
    register_notes: str           # qualitative LLM summary of voice
    n_samples: int                # how many sent items fed this profile


# Generic LLM-isms that get auto-banned unless we observe them in
# the user's actual sent mail. The voice extractor checks each one
# against the corpus and only adds it to ``avoided_phrases`` when
# the user genuinely doesn't use it.
_LLM_ISMS_TO_AUDIT = (
    "I hope this email finds you well",
    "I hope this finds you well",
    "Thank you for reaching out",
    "Please don't hesitate to",
    "Please let me know if you have any further questions",
    "I look forward to hearing from you",
    "Best regards,",
    "Warm regards,",
    "I wanted to reach out",
    "I am writing to",
)


def extract_voice_profile(
    conn: sqlite3.Connection, cfg=None,
    *, max_samples: int = _VOICE_PROFILE_MAX_SAMPLES,
) -> VoiceProfile | None:
    """Round 7 — scan the user's recent Sent items and extract a
    structured voice profile.

    Hybrid: deterministic stats (greeting / sign-off / sentence
    length / contractions) + one Haiku call for the qualitative
    register notes. Returns None when there are no Sent items to
    learn from — the drafter falls back to its default voice rules.
    """
    _ensure_schema(conn)
    rows = conn.execute(
        "SELECT c.text FROM chunks c JOIN files f ON f.id = c.file_id "
        "WHERE c.chunk_index = 0 "
        "  AND (f.path LIKE 'imap://%' OR f.path LIKE 'gmail://%') "
        "  AND LOWER(c.text) LIKE '%folder: sent%' "
        "ORDER BY f.indexed_at DESC LIMIT ?",
        (max_samples,),
    ).fetchall()
    bodies = [_strip_email_headers(r["text"] or "") for r in rows]
    bodies = [b for b in bodies if len(b) >= 30]  # drop near-empty
    if not bodies:
        return None
    greetings = _extract_greeting_patterns(bodies)
    sign_offs = _extract_signoff_patterns(bodies)
    avg_words = _avg_sentence_words(bodies)
    avg_chars = int(sum(len(b) for b in bodies) / len(bodies))
    contraction_rate = _contraction_rate(bodies)
    excl_rate = sum(b.count("!") for b in bodies) / len(bodies)
    emoji_rate = _emoji_rate(bodies)
    openers = _common_first_phrases(bodies)
    closers = _common_last_phrases(bodies)
    avoided = _audit_llm_isms(bodies)

    # Qualitative LLM pass — short Haiku call summarising voice.
    register_notes = _voice_register_notes(bodies, cfg) if cfg else ""

    profile = VoiceProfile(
        greetings=greetings,
        sign_offs=sign_offs,
        avg_sentence_words=round(avg_words, 1),
        avg_reply_chars=avg_chars,
        contraction_rate=round(contraction_rate, 2),
        exclamation_rate=round(excl_rate, 2),
        emoji_rate=round(emoji_rate, 2),
        common_openers=openers,
        common_closers=closers,
        avoided_phrases=avoided,
        register_notes=register_notes,
        n_samples=len(bodies),
    )
    _save_voice_profile(conn, profile)
    return profile


def _strip_email_headers(text: str) -> str:
    """Remove the 'From:/To:/Subject:/Folder:' header lines the
    connectors prepend, leaving just the actual reply body."""
    lines = text.splitlines()
    body_start = 0
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            body_start = i + 1
            continue
        # H1 + From/To/Date/Folder lines all live in the prefix.
        if (s.startswith("# ") or s.lower().startswith(
            ("from:", "to:", "cc:", "bcc:", "date:", "folder:",
             "labels:", "message-id:", "in-reply-to:", "references:"),
        )):
            body_start = i + 1
            continue
        break
    body = "\n".join(lines[body_start:]).strip()
    # Quoted-reply blocks ("On ... wrote:" / "> ...") inflate stats
    # and aren't the user's voice; drop them.
    body = _strip_quoted_reply(body)
    return body


_QUOTED_REPLY_RE = re.compile(
    r"\n\s*(On .+ wrote:|From:.+|-----Original Message-----).*",
    re.DOTALL,
)


def _strip_quoted_reply(body: str) -> str:
    """Trim quoted-reply blocks from a sent-message body."""
    body = _QUOTED_REPLY_RE.sub("", body)
    # Bare ">" quote lines — keep the first occurrence's preceding
    # text only.
    out_lines = []
    for ln in body.splitlines():
        if ln.lstrip().startswith(">"):
            break
        out_lines.append(ln)
    return "\n".join(out_lines).strip()


def _extract_greeting_patterns(bodies: list[str]) -> list[str]:
    """Pull the first non-empty line of each body, normalise
    addressed-name to ``{name}``, return the most-common patterns
    (≥ 2 occurrences)."""
    from collections import Counter

    pats: list[str] = []
    for b in bodies:
        first = ""
        for ln in b.splitlines():
            s = ln.strip()
            if s:
                first = s
                break
        if not first:
            continue
        # Cap so a long opener doesn't masquerade as a greeting.
        if len(first) > 60:
            continue
        # Normalise common name patterns: "Hi Sarah," → "hi {name},"
        # IGNORECASE so leading "Hi"/"Hey"/"Hello"/"Dear" match.
        norm = re.sub(
            r"\b(hi|hey|hello|dear)\s+[A-Z][a-z]+",
            lambda m: m.group(1).lower() + " {name}",
            first, count=1, flags=re.IGNORECASE,
        )
        # "Sarah," at start → "{name},"
        norm = re.sub(
            r"^[A-Z][a-z]+(,)", r"{name}\1", norm,
        )
        norm = norm.strip().lower()
        if norm:
            pats.append(norm)
    return [p for p, _n in Counter(pats).most_common(5) if _n >= 2]


def _extract_signoff_patterns(bodies: list[str]) -> list[str]:
    """Pull the last 1-2 non-empty lines of each body and surface
    the most-common sign-off shapes."""
    from collections import Counter

    pats: list[str] = []
    for b in bodies:
        lines = [ln for ln in b.splitlines() if ln.strip()]
        if not lines:
            continue
        tail = lines[-2:] if len(lines) >= 2 else lines[-1:]
        joined = "\n".join(s.strip() for s in tail)
        if len(joined) > 80:
            joined = lines[-1].strip()
        # Replace user's name (any single capitalised word at the
        # end) with {name} so different signers fold together if
        # this corpus mixes accounts.
        norm = re.sub(r"\b[A-Z][a-z]+$", "{name}", joined)
        norm = norm.lower().strip()
        if norm:
            pats.append(norm)
    return [p for p, _n in Counter(pats).most_common(5) if _n >= 2]


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _avg_sentence_words(bodies: list[str]) -> float:
    """Mean sentence length in words across all bodies. Cheap proxy
    for "do they write short or long sentences."""
    total_w = 0
    total_s = 0
    for b in bodies:
        for s in _SENT_SPLIT_RE.split(b):
            words = s.split()
            if not words:
                continue
            total_w += len(words)
            total_s += 1
    return total_w / total_s if total_s else 0.0


_CONTRACTION_PAIRS = (
    (r"\bI(?:'|’)d\b", r"\bI would\b"),
    (r"\bI(?:'|’)ll\b", r"\bI will\b"),
    (r"\bI(?:'|’)m\b", r"\bI am\b"),
    (r"\bdon(?:'|’)t\b", r"\bdo not\b"),
    (r"\bcan(?:'|’)t\b", r"\bcannot\b"),
    (r"\bit(?:'|’)s\b", r"\bit is\b"),
    (r"\bthat(?:'|’)s\b", r"\bthat is\b"),
    (r"\bwon(?:'|’)t\b", r"\bwill not\b"),
)


def _contraction_rate(bodies: list[str]) -> float:
    """Fraction of (contracted | uncontracted) pairs that the user
    actually contracted. 1.0 = always contracts, 0.0 = always formal."""
    contracted = 0
    expanded = 0
    text = " ".join(bodies)
    for c, e in _CONTRACTION_PAIRS:
        contracted += len(re.findall(c, text, re.IGNORECASE))
        expanded += len(re.findall(e, text, re.IGNORECASE))
    total = contracted + expanded
    return contracted / total if total else 0.5


_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F6FF"   # emoji blocks
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]",
)


def _emoji_rate(bodies: list[str]) -> float:
    return sum(len(_EMOJI_RE.findall(b)) for b in bodies) / len(bodies)


def _common_first_phrases(bodies: list[str]) -> list[str]:
    """First non-greeting sentence across replies; surface frequent
    opener phrases (≥ 2 occurrences). Useful so the drafter knows
    'this user always opens with X' patterns."""
    from collections import Counter

    phrases: list[str] = []
    for b in bodies:
        sentences = _SENT_SPLIT_RE.split(b)
        # Skip the first sentence if it's just a greeting ("Hi Sarah,")
        for s in sentences:
            cleaned = s.strip()
            if not cleaned:
                continue
            if cleaned.lower().startswith(("hi ", "hey ", "hello ", "dear ")):
                continue
            # Take just the first 8 words as a phrase signature.
            words = cleaned.split()[:8]
            phrases.append(" ".join(words).lower())
            break
    return [p for p, _n in Counter(phrases).most_common(5) if _n >= 2]


def _common_last_phrases(bodies: list[str]) -> list[str]:
    """Last non-signoff sentence — captures recurring closer phrases
    like 'Let me know what you think' or 'Talk soon'."""
    from collections import Counter

    phrases: list[str] = []
    for b in bodies:
        sentences = [s.strip() for s in _SENT_SPLIT_RE.split(b) if s.strip()]
        if len(sentences) < 2:
            continue
        # Sentence before the sign-off line (avoids "—Ben" matching).
        candidate = sentences[-1] if "," in sentences[-1] else sentences[-2]
        words = candidate.split()[:8]
        phrases.append(" ".join(words).lower())
    return [p for p, _n in Counter(phrases).most_common(5) if _n >= 2]


def _audit_llm_isms(bodies: list[str]) -> list[str]:
    """Check each generic-LLM phrase against the corpus. Add it to
    the avoid-list when the user genuinely doesn't use it. Means the
    drafter knows 'this user never says "Best regards," — don't put
    it in the draft.'"""
    text = " ".join(bodies).lower()
    avoided: list[str] = []
    for phrase in _LLM_ISMS_TO_AUDIT:
        if phrase.lower() not in text:
            avoided.append(phrase)
    return avoided


_VOICE_NOTES_PROMPT = """\
Below are recent reply emails written by ONE person. Describe their
voice in 3-5 short sentences focused on what makes their writing
distinctive vs generic professional email. Cover: warmth, formality,
typical sentence length, idiosyncratic vocabulary, opener/closer
habits.

Output ONLY the prose summary — no headings, no bullets, no preamble.

EMAILS
======
{samples}
"""


def _voice_register_notes(bodies: list[str], cfg) -> str:
    """One Haiku call — qualitative voice summary. Failure returns
    empty string; the drafter falls back to the structural patterns.

    Round 10 (#4) — sent-mail samples get redacted before they hit
    the prompt. Voice patterns (sentence rhythm, sign-offs) survive
    Phase 88 masking because it only catches secret-shaped substrings.
    """
    sample_block = "\n\n---\n\n".join(
        _safe_for_prompt(b, max_chars=800) for b in bodies[:10]
    )
    if not sample_block.strip():
        return ""
    parsed = _llm_text_call(
        prompt=_VOICE_NOTES_PROMPT.format(samples=sample_block),
        cfg=cfg,
        model="claude-haiku-4-5",
        max_tokens=300,
        feature="voice_profile",
        note="voice_register",
    )
    return (parsed or "").strip()


def _llm_text_call(
    *, prompt: str, cfg, model: str, max_tokens: int,
    feature: str, note: str,
    conn: sqlite3.Connection | None = None,
    audit_kind: str = "",
    audit_summary: str = "",
    audit_file_id: int | None = None,
    audit_person_id: int | None = None,
) -> str:
    """Like _llm_json_call but for plain-text output (voice notes,
    critique). Same try-Anthropic-then-local pattern. Same round-10
    audit hooks (optional ``conn`` enables ai_actions logging)."""
    text = ""
    used_model = model
    status = "success"
    err_msg = ""
    cents_spent = 0.0
    final_kind = audit_kind or feature

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic
        except ImportError:
            anthropic = None  # type: ignore[assignment]
        if anthropic is not None:
            from .budget import (
                BudgetExceededError,
                check_budget,
                estimate_cost,
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
                try:
                    cents_spent = estimate_cost(
                        model,
                        input_tokens=resp.usage.input_tokens,
                        output_tokens=resp.usage.output_tokens,
                    ).cents
                except Exception:  # noqa: BLE001
                    cents_spent = 0.0
                text = "".join(
                    b.text for b in resp.content if b.type == "text"
                ).strip()
            except BudgetExceededError as e:
                log.info("email_assist: %s budget exhausted: %s", feature, e)
                status = "budget_exceeded"
                err_msg = str(e)
            except anthropic.APIError as e:
                log.info("email_assist: %s API error: %s", feature, e)
                status = "api_error"
                err_msg = str(e)
        else:
            status = "no_provider"
    else:
        status = "no_provider"
    if not text:
        try:
            from . import local_llm
        except ImportError:
            _maybe_audit(
                conn, kind=final_kind, feature=feature, model=used_model,
                status=status, prompt_chars=len(prompt),
                response_chars=0, cents=cents_spent,
                summary=audit_summary, error=err_msg or "no LLM available",
                file_id=audit_file_id, person_id=audit_person_id,
            )
            return ""
        if not local_llm.is_available(cfg):
            _maybe_audit(
                conn, kind=final_kind, feature=feature, model=used_model,
                status=status, prompt_chars=len(prompt),
                response_chars=0, cents=cents_spent,
                summary=audit_summary, error=err_msg or "no LLM available",
                file_id=audit_file_id, person_id=audit_person_id,
            )
            return ""
        out = local_llm.complete(prompt, cfg=cfg, max_tokens=max_tokens)
        if out is None:
            _maybe_audit(
                conn, kind=final_kind, feature=feature, model=used_model,
                status="api_error", prompt_chars=len(prompt),
                response_chars=0, cents=cents_spent,
                summary=audit_summary,
                error=err_msg or "local llm returned None",
                file_id=audit_file_id, person_id=audit_person_id,
            )
            return ""
        text = out.text.strip()
        used_model = out.model
        status = "fallback_local"
    _maybe_audit(
        conn, kind=final_kind, feature=feature, model=used_model,
        status=status, prompt_chars=len(prompt),
        response_chars=len(text), cents=cents_spent,
        summary=audit_summary, error="",
        file_id=audit_file_id, person_id=audit_person_id,
    )
    return text


def _save_voice_profile(
    conn: sqlite3.Connection, profile: VoiceProfile,
) -> None:
    _ensure_schema(conn)
    payload = {
        "greetings": profile.greetings,
        "sign_offs": profile.sign_offs,
        "avg_sentence_words": profile.avg_sentence_words,
        "avg_reply_chars": profile.avg_reply_chars,
        "contraction_rate": profile.contraction_rate,
        "exclamation_rate": profile.exclamation_rate,
        "emoji_rate": profile.emoji_rate,
        "common_openers": profile.common_openers,
        "common_closers": profile.common_closers,
        "avoided_phrases": profile.avoided_phrases,
        "register_notes": profile.register_notes,
    }
    conn.execute(
        "INSERT OR REPLACE INTO email_style_profile"
        "(id, profile_json, sent_count, updated_at) "
        "VALUES (1, ?, ?, ?)",
        (json.dumps(payload), profile.n_samples, time.time()),
    )
    conn.commit()


def default_voice_profile() -> VoiceProfile:
    """Round 10 (#5) — curated bootstrap profile for new users who
    haven't indexed their Sent folder yet.

    Without this, the drafter would fall through to a generic-LLM
    voice — which is exactly the cliché we tried to kill in round 7.
    The default leans casual-warm-concise (the most common modern
    professional voice) and explicitly lists banned phrases so the
    drafter doesn't open with 'I hope this email finds you well'
    on a brand-new install.

    The profile gets replaced as soon as the user indexes any Sent
    mail — the round-7 extractor runs weekly via the daemon.
    """
    return VoiceProfile(
        greetings=["hi {name},", "{name},", "hey {name},"],
        sign_offs=["thanks,\n{name}", "—{name}", "{name}"],
        avg_sentence_words=12.0,
        avg_reply_chars=240,
        contraction_rate=0.85,
        exclamation_rate=0.1,
        emoji_rate=0.0,
        common_openers=[],
        common_closers=[],
        avoided_phrases=[
            "I hope this email finds you well",
            "I hope this finds you well",
            "Thank you for reaching out",
            "Please don't hesitate to",
            "Please let me know if you have any further questions",
            "I look forward to hearing from you",
            "Best regards,",
            "Warm regards,",
            "I wanted to reach out",
            "I am writing to",
        ],
        register_notes=(
            "Casual but professional. Uses contractions. Short "
            "sentences, direct asks. Skips boilerplate openers / "
            "closers. (Default — refines as the user's Sent folder "
            "gets indexed.)"
        ),
        n_samples=0,  # marker that this is the bootstrap profile
    )


def get_voice_profile_or_default(
    conn: sqlite3.Connection,
) -> VoiceProfile:
    """Round 10 (#5) — same as ``get_voice_profile`` but returns the
    curated default when no profile exists yet. The drafter prefers
    this over None so cold-start drafts don't degrade to generic
    LLM voice."""
    profile = get_voice_profile(conn)
    if profile is None:
        return default_voice_profile()
    return profile


def get_voice_profile(
    conn: sqlite3.Connection,
) -> VoiceProfile | None:
    """Read the persisted profile. Returns None when no profile has
    been computed yet — drafter falls back to the default rules."""
    _ensure_schema(conn)
    row = conn.execute(
        "SELECT profile_json, sent_count, updated_at "
        "FROM email_style_profile WHERE id = 1",
    ).fetchone()
    if row is None:
        return None
    try:
        p = json.loads(row["profile_json"])
    except (TypeError, ValueError):
        return None
    if not isinstance(p, dict):
        return None
    return VoiceProfile(
        greetings=list(p.get("greetings") or []),
        sign_offs=list(p.get("sign_offs") or []),
        avg_sentence_words=float(p.get("avg_sentence_words") or 0.0),
        avg_reply_chars=int(p.get("avg_reply_chars") or 0),
        contraction_rate=float(p.get("contraction_rate") or 0.5),
        exclamation_rate=float(p.get("exclamation_rate") or 0.0),
        emoji_rate=float(p.get("emoji_rate") or 0.0),
        common_openers=list(p.get("common_openers") or []),
        common_closers=list(p.get("common_closers") or []),
        avoided_phrases=list(p.get("avoided_phrases") or []),
        register_notes=str(p.get("register_notes") or ""),
        n_samples=int(row["sent_count"] or 0),
    )


def needs_voice_profile_refresh(
    conn: sqlite3.Connection,
    *, days: int = _VOICE_PROFILE_REFRESH_DAYS,
) -> bool:
    """True iff the profile is missing or older than ``days``."""
    _ensure_schema(conn)
    row = conn.execute(
        "SELECT updated_at FROM email_style_profile WHERE id = 1",
    ).fetchone()
    if row is None:
        return True
    return (time.time() - float(row["updated_at"])) > days * 86400


def refresh_voice_profile_if_due(
    conn: sqlite3.Connection, cfg,
) -> bool:
    """Daemon entrypoint: re-extract weekly. Returns True iff a
    refresh actually ran."""
    if not needs_voice_profile_refresh(conn):
        return False
    return extract_voice_profile(conn, cfg) is not None


def _format_voice_profile_block(p: VoiceProfile) -> str:
    """Render the profile as a compact text block for the drafter
    prompt. Concrete patterns the model can copy verbatim."""
    def _fmt_list(items: list[str], empty: str = "(none observed)") -> str:
        if not items:
            return empty
        return "  - " + "\n  - ".join(items)
    return (
        f"voice profile (extracted from {p.n_samples} sent emails):\n"
        f"  greetings to copy: \n{_fmt_list(p.greetings)}\n"
        f"  sign-offs to copy: \n{_fmt_list(p.sign_offs)}\n"
        f"  avg sentence length: {p.avg_sentence_words:.1f} words\n"
        f"  avg reply length: {p.avg_reply_chars} chars\n"
        f"  contraction usage: {p.contraction_rate:.0%} (1.0 = always uses I'd / can't / it's)\n"
        f"  exclamation marks per reply: {p.exclamation_rate:.1f}\n"
        f"  emojis per reply: {p.emoji_rate:.1f}\n"
        f"  common opener phrases: \n{_fmt_list(p.common_openers)}\n"
        f"  common closer phrases: \n{_fmt_list(p.common_closers)}\n"
        f"  avoid these phrases (user never uses them): \n{_fmt_list(p.avoided_phrases)}\n"
        f"  voice notes: {p.register_notes or '(none)'}"
    )


# ---- Few-shot reply-pair retrieval -----------------------------------

def fewshot_reply_pairs(
    conn: sqlite3.Connection,
    *,
    incoming_text: str,
    embedder=None,
    k: int = _FEWSHOT_PAIR_K,
) -> list[tuple[str, str]]:
    """Round 7 — find the (incoming, user_reply) pairs whose incoming
    is most similar to the current incoming email. Returns
    ``[(incoming_body, reply_body), ...]``.

    Uses hybrid_search over the parent (incoming) emails so we land
    on truly-similar conversations, not just same-sender ones. Falls
    back to recency-ordered pairs when the embedder isn't available.
    """
    _ensure_schema(conn)
    pair_rows = conn.execute(
        "SELECT incoming_file_id, reply_file_id FROM email_reply_pairs "
        "ORDER BY indexed_at DESC LIMIT 500",
    ).fetchall()
    if not pair_rows:
        return []
    incoming_ids = [int(r["incoming_file_id"]) for r in pair_rows]
    reply_by_incoming = {
        int(r["incoming_file_id"]): int(r["reply_file_id"])
        for r in pair_rows
    }

    selected_ids: list[int] = []
    if embedder is not None:
        try:
            from .search import hybrid_search
            results = hybrid_search(
                conn, embedder, incoming_text, k=k * 4,
            )
            # Resolve result file_paths back to file_ids and intersect
            # with our paired-incoming set.
            paths = [r.file_path for r in results]
            if paths:
                placeholders = ",".join("?" * len(paths))
                rows = conn.execute(
                    f"SELECT id FROM files WHERE path IN ({placeholders})",
                    paths,
                ).fetchall()
                hit_ids = {int(r["id"]) for r in rows}
                paired_set = set(incoming_ids)
                # Preserve search-relevance order.
                seen: set[int] = set()
                for r in results:
                    row = conn.execute(
                        "SELECT id FROM files WHERE path = ?",
                        (r.file_path,),
                    ).fetchone()
                    if row is None:
                        continue
                    fid = int(row["id"])
                    if fid in hit_ids and fid in paired_set and fid not in seen:
                        selected_ids.append(fid)
                        seen.add(fid)
                        if len(selected_ids) >= k:
                            break
        except Exception as e:  # noqa: BLE001
            log.info("email_assist: fewshot semantic search failed: %s", e)

    # Fallback / top-up — fill remaining slots from most-recent pairs.
    if len(selected_ids) < k:
        for fid in incoming_ids:
            if fid in selected_ids:
                continue
            selected_ids.append(fid)
            if len(selected_ids) >= k:
                break

    # Hydrate pair bodies.
    pairs: list[tuple[str, str]] = []
    for inc_id in selected_ids:
        rep_id = reply_by_incoming.get(inc_id)
        if rep_id is None:
            continue
        inc_body = _read_file_body(conn, inc_id)
        rep_body = _read_file_body(conn, rep_id)
        if not (inc_body and rep_body):
            continue
        pairs.append((
            inc_body[:_FEWSHOT_PAIR_CHARS],
            _strip_email_headers(rep_body)[:_FEWSHOT_PAIR_CHARS],
        ))
    return pairs


def _format_fewshot_block(pairs: list[tuple[str, str]]) -> str:
    """Render reply pairs as a labelled few-shot block. The drafter
    sees these as 'here's how the user actually replied to similar
    emails' examples to mimic structurally."""
    if not pairs:
        return "(no reply-pair examples available yet)"
    parts = []
    for i, (inc, rep) in enumerate(pairs, 1):
        parts.append(
            f"--- EXAMPLE {i} ---\n"
            f"INCOMING:\n{inc}\n\n"
            f"USER REPLIED:\n{rep}",
        )
    return "\n\n".join(parts)


# ---- Voice critique pass ---------------------------------------------

_VOICE_CRITIQUE_PROMPT = """\
A draft reply was written for the user. Critique it ONLY against the
voice profile below. List any specific mismatches (banned phrases
used, greeting/signoff that doesn't match observed patterns, sentence
length way off, formality drift). If the draft genuinely matches the
profile, respond with the single token OK and nothing else.

Output format: either OK, or a bulleted list of mismatches (one per
line, prefixed with "- "). No prose around it.

VOICE PROFILE
=============
{profile_block}

DRAFT
=====
{draft}
"""


def critique_draft_against_voice(
    *, draft: str, profile: VoiceProfile, cfg,
    conn: sqlite3.Connection | None = None,
    file_id: int | None = None,
) -> str:
    """Round 7 — Haiku call comparing the draft to the voice profile.
    Returns "OK" when the draft passes, otherwise a bullet list of
    mismatches. Caller decides whether to regenerate.

    Round 10 (#4) — defensive redaction on the draft. The draft is
    the LLM's own output so secrets shouldn't be there, but if the
    user typed a real value into a ``<TODO: ...>`` placeholder the
    drafter may have echoed it; mask before re-sending."""
    if not draft.strip():
        return "OK"
    prompt = _VOICE_CRITIQUE_PROMPT.format(
        profile_block=_format_voice_profile_block(profile),
        draft=_safe_for_prompt(draft.strip(), max_chars=4000),
    )
    txt = _llm_text_call(
        prompt=prompt, cfg=cfg,
        model="claude-haiku-4-5", max_tokens=200,
        feature="voice_critique", note="critique",
        conn=conn, audit_kind="voice_critique",
        audit_summary="critiqued draft against voice profile",
        audit_file_id=file_id,
    )
    return (txt or "").strip() or "OK"


# Critique-augmented prompt — appended to _DRAFT_PROMPT_V2 when we
# regenerate after a failed voice critique. Tells Sonnet exactly
# what was wrong with the prior attempt so it can fix the specific
# mismatches instead of drifting in a new direction.
_DRAFT_REGENERATE_SUFFIX = """\

PRIOR DRAFT (REJECTED — do NOT repeat the same mistakes)
==============
{prior_draft}

VOICE CRITIQUE — fix every item below in the new draft
==============
{critique}
"""


def _regenerate_with_critique(
    *,
    prior_draft: str, critique: str,
    **drafter_kwargs,
) -> DraftOutput | None:
    """Round 7 — second-attempt drafter, fed the prior draft + the
    voice critique. Reuses _default_drafter machinery via a one-shot
    prompt augmentation: same retrieval, same analysis, but the
    drafter sees what the prior attempt got wrong.

    Capped at one regeneration so a stubborn voice mismatch can't
    burn the budget in a loop.
    """
    body_clip = (
        drafter_kwargs["body"]
        if len(drafter_kwargs["body"]) <= 6000
        else drafter_kwargs["body"][:6000] + "…"
    )
    analysis = drafter_kwargs.get("analysis")
    voice_profile = drafter_kwargs.get("voice_profile")
    analysis_block = (
        _format_analysis_block(analysis) if analysis is not None
        else "(analyzer unavailable)"
    )
    voice_block = (
        _format_voice_profile_block(voice_profile)
        if voice_profile is not None else "(no profile)"
    )
    fewshot_block = _format_fewshot_block(
        drafter_kwargs.get("fewshot_pairs") or [],
    )
    prompt = (
        _DRAFT_PROMPT_V2.format(
            user_name=drafter_kwargs["user_name"],
            from_=drafter_kwargs["from_"] or "(unknown)",
            subject=drafter_kwargs["subject"] or "(no subject)",
            body=body_clip,
            analysis_block=analysis_block,
            thread_block=_format_history_block(
                drafter_kwargs.get("thread_history") or [],
            ),
            sender_block=_format_history_block(
                drafter_kwargs.get("sender_history") or [],
            ),
            style_samples=drafter_kwargs.get("style_samples") or "",
            brain_block=drafter_kwargs.get("brain_context")
            or "(no relevant brain context found)",
            voice_profile_block=voice_block,
            fewshot_block=fewshot_block,
        )
        + _DRAFT_REGENERATE_SUFFIX.format(
            prior_draft=prior_draft.strip(),
            critique=critique.strip(),
        )
    )
    parsed = _llm_json_call(
        prompt=prompt,
        cfg=drafter_kwargs["cfg"],
        model="claude-sonnet-4-6",
        max_tokens=1200,
        feature="email_draft",
        note=(
            f"email_draft_regen/"
            f"{(drafter_kwargs.get('from_') or '')[:30]}"
        ),
    )
    if parsed is None:
        return None
    primary = str(parsed.get("primary") or "").strip()
    alternative = str(parsed.get("alternative") or "").strip()
    if not primary and not alternative:
        return None
    try:
        confidence = float(parsed.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return DraftOutput(
        primary=primary,
        alternative=alternative,
        reasoning=str(parsed.get("reasoning") or ""),
        confidence=max(0.0, min(1.0, confidence)),
        open_questions=[
            str(q) for q in (parsed.get("open_questions") or [])
            if str(q).strip()
        ],
    )
