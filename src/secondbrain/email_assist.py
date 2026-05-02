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
from dataclasses import dataclass

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
            note TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_email_drafts_file
            ON email_drafts(file_id, generated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_email_drafts_unsent
            ON email_drafts(sent_at, generated_at DESC);
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
class Draft:
    id: int
    file_id: int
    draft_text: str
    generated_at: float
    sent_at: float | None = None


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
  {"label": "<one of above>", "confidence": 0..1, "rationale": "≤10 words"}

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
    """Real Haiku classifier. Bounded body length to keep cost tight."""
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {}
    try:
        import anthropic
    except ImportError:
        return {}
    from .budget import (
        BudgetExceededError,
        check_budget,
        record_usage,
    )
    try:
        check_budget(cfg, "anthropic", feature="email_triage")
    except BudgetExceededError as e:
        log.warning("email_assist: budget exceeded: %s", e)
        return {}
    body_clip = body if len(body) <= 4000 else body[:4000] + "…"
    prompt = _TRIAGE_PROMPT.format(
        from_=from_ or "(unknown)",
        subject=subject or "(no subject)",
        body=body_clip,
    )
    client = anthropic.Anthropic()
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as e:
        log.warning("email_assist: classify API error: %s", e)
        return {}
    record_usage(
        cfg, "anthropic", "claude-haiku-4-5",
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        feature="email_triage",
        note=f"email_triage/{from_[:30]}",
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def classify_due(
    conn: sqlite3.Connection, cfg,
    *, max_per_tick: int = _CLASSIFY_PER_TICK,
    classifier=None,
) -> int:
    """Daemon entrypoint. Find recent unclassified email docs and
    classify a bounded batch."""
    _ensure_schema(conn)
    cutoff = time.time() - _CLASSIFY_LOOKBACK_DAYS * 86400
    rows = conn.execute(
        "SELECT f.id FROM files f "
        "WHERE f.path LIKE 'imap://%' "
        "  AND f.indexed_at >= ? "
        "  AND f.id NOT IN (SELECT file_id FROM email_classifications) "
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
    """Generate a reply draft. Persists with sent_at=NULL so the user
    can review before marking sent."""
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
    style_samples = _gather_style_samples(conn)
    if drafter is None:
        drafter = _default_drafter
        if cfg is None:
            return None
    try:
        text = drafter(
            from_=from_, subject=subject, body=body,
            style_samples=style_samples,
            user_name=user_name, cfg=cfg,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("email_assist: drafter crashed: %s", e)
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
    *, from_, subject, body, style_samples, user_name, cfg,
) -> str:
    """Real drafter via Claude Sonnet (better voice mimicry than Haiku)."""
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return ""
    try:
        import anthropic
    except ImportError:
        return ""
    from .budget import (
        BudgetExceededError,
        check_budget,
        record_usage,
    )
    try:
        check_budget(cfg, "anthropic", feature="email_draft")
    except BudgetExceededError as e:
        log.warning("email_assist: budget exceeded: %s", e)
        return ""
    body_clip = body if len(body) <= 6000 else body[:6000] + "…"
    prompt = _DRAFT_PROMPT.format(
        user_name=user_name,
        from_=from_ or "(unknown)",
        subject=subject or "(no subject)",
        body=body_clip,
        style_samples=style_samples,
    )
    client = anthropic.Anthropic()
    try:
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as e:
        log.warning("email_assist: draft API error: %s", e)
        return ""
    record_usage(
        cfg, "anthropic", "claude-sonnet-4-6",
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        feature="email_draft",
        note=f"email_draft/{from_[:30]}",
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    return text


def list_unsent_drafts(
    conn: sqlite3.Connection, *, limit: int = 50,
) -> list[Draft]:
    _ensure_schema(conn)
    rows = conn.execute(
        "SELECT * FROM email_drafts WHERE sent_at IS NULL "
        "ORDER BY generated_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [
        Draft(
            id=int(r["id"]), file_id=int(r["file_id"]),
            draft_text=r["draft_text"],
            generated_at=r["generated_at"],
            sent_at=r["sent_at"],
        )
        for r in rows
    ]


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
        "WHERE ec.label IN ('urgent', 'response') "
        "  AND ec.file_id NOT IN ("
        "    SELECT file_id FROM email_drafts WHERE sent_at IS NULL"
        "  ) "
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
