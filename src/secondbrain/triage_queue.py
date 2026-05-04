"""Round 19 (Phase EA-6) — morning email triage queue.

The existing ``email_assist`` pipeline classifies emails (urgent /
follow_up / fyi / spam / review) and drafts replies. What it
doesn't do is *order* them for the user's morning attention.

A real EA hands you the inbox already sorted: "5 emails need a
decision today, in this order." This module provides that
ordering by combining:

  - VIP tier of the sender (round 19 — VIPs jump the queue)
  - Email classification urgency (urgent > follow_up > review > fyi)
  - Open-followup-with-this-sender freshness (older = higher rank)
  - Days since last reply to this sender

Output: a ranked list of ``TriageItem`` rows the dashboard's
``/triage`` view (or a chat tool) can walk through one-by-one.
Each item links to its existing draft (if email_assist already
ran) so the user just hits approve/edit/skip.

Design notes:
- Read-only over the email + drafts tables. No new schema.
- Caps at ``max_items`` so a 200-email morning isn't overwhelming;
  the user can always re-run for round 2.
- Doesn't call an LLM — pure ranking from existing classifications.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class TriageItem:
    file_id: int
    path: str
    from_email: str
    from_display: str
    subject: str
    label: str               # 'urgent' | 'follow_up' | 'review' | 'fyi'
    confidence: float
    is_vip: bool
    has_draft: bool
    draft_id: int | None
    body_preview: str
    received_at: float
    age_hours: float
    rank_score: float


def _redact(text: str) -> str:
    try:
        from .safety import redact_text
        return redact_text(text or "")
    except ImportError:
        return text or ""


_LABEL_BASE = {
    "urgent": 100.0,
    "follow_up": 60.0,
    "review": 40.0,
    "fyi": 10.0,
    "spam": -1.0,
    None: 30.0,
}


def _rank_score(
    *, label: str | None, confidence: float, age_hours: float,
    is_vip: bool, has_draft: bool,
) -> float:
    """Higher = more urgent for the user. Combination of:
      - Label base score (urgent=100, fyi=10, etc.)
      - VIP bonus (+50)
      - Age decay (urgent emails compound; fyi emails fade)
      - "Has draft ready" small bonus so we surface the easy wins
    """
    base = _LABEL_BASE.get(label, _LABEL_BASE[None])
    if is_vip:
        base += 50.0
    # Age effect: urgent emails get MORE urgent over time; fyi
    # emails get LESS interesting. Tune by label.
    if label == "urgent":
        base += min(40.0, age_hours * 1.5)
    elif label == "follow_up":
        base += min(20.0, age_hours * 0.5)
    elif label == "fyi":
        base -= min(8.0, age_hours * 0.1)
    if has_draft:
        base += 5.0
    base *= max(0.5, confidence)
    return base


def build_queue(
    conn: sqlite3.Connection,
    *,
    hours: int = 48,
    max_items: int = 12,
    min_score: float = 25.0,
) -> list[TriageItem]:
    """Build the morning triage queue. Looks at the last ``hours``
    of indexed email-kind files, joins to email_classifications +
    email_drafts, ranks, returns the top N."""
    cutoff = time.time() - hours * 3600
    try:
        # email_drafts uses ``sent_at IS NULL`` to mean "still pending"
        # (no explicit status column). LEFT JOIN to the latest unsent
        # draft per file via a subquery so a file with multiple drafts
        # surfaces the freshest pending one.
        rows = conn.execute(
            "SELECT f.id AS file_id, f.path, f.indexed_at, "
            "       SUBSTR(c.text, 1, 400) AS preview, "
            "       ec.label, ec.confidence, "
            "       ed.id AS draft_id "
            "FROM files f "
            "LEFT JOIN chunks c ON c.file_id = f.id AND c.chunk_index = 0 "
            "LEFT JOIN email_classifications ec ON ec.file_id = f.id "
            "LEFT JOIN email_drafts ed ON ed.file_id = f.id "
            "    AND ed.sent_at IS NULL "
            "WHERE f.indexed_at >= ? "
            "  AND (f.kind = 'email' OR f.kind = 'message') "
            "ORDER BY f.indexed_at DESC LIMIT 300",
            (cutoff,),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    if not rows:
        return []
    # Lazy person lookup for VIP check.
    try:
        from . import people as people_mod
    except ImportError:
        people_mod = None  # type: ignore[assignment]
    out: list[TriageItem] = []
    for r in rows:
        preview = _redact((r["preview"] or "")[:300])
        # Parse "From: ..." and "Subject: ..." out of the preview.
        from_email, from_display, subject = _parse_email_headers(
            preview,
        )
        is_vip = False
        if people_mod is not None and from_email:
            try:
                is_vip = people_mod.is_vip_email(conn, from_email)
            except Exception:  # noqa: BLE001
                is_vip = False
        age_hours = max(0.0, (time.time() - float(r["indexed_at"])) / 3600.0)
        score = _rank_score(
            label=r["label"], confidence=float(r["confidence"] or 0.5),
            age_hours=age_hours, is_vip=is_vip,
            has_draft=r["draft_id"] is not None,
        )
        if score < min_score and not is_vip:
            continue
        out.append(TriageItem(
            file_id=int(r["file_id"]),
            path=r["path"] or "",
            from_email=from_email,
            from_display=from_display,
            subject=subject,
            label=r["label"] or "fyi",
            confidence=float(r["confidence"] or 0.5),
            is_vip=is_vip,
            has_draft=r["draft_id"] is not None,
            draft_id=(int(r["draft_id"]) if r["draft_id"] else None),
            body_preview=preview,
            received_at=float(r["indexed_at"]),
            age_hours=age_hours,
            rank_score=score,
        ))
    out.sort(key=lambda i: i.rank_score, reverse=True)
    return out[:max_items]


def _parse_email_headers(preview: str) -> tuple[str, str, str]:
    """Extract (email_addr, display_name, subject) from a raw email
    preview. Best-effort — handles common shapes:

        From: Sarah <s@x.com>
        Subject: Re: Q3 numbers

    Returns ('s@x.com', 'Sarah', 'Re: Q3 numbers') or ('','','')
    if nothing parseable.
    """
    import re
    if not preview:
        return "", "", ""
    from_match = re.search(
        r"(?im)^From:\s*(.+?)$", preview,
    )
    subject_match = re.search(
        r"(?im)^Subject:\s*(.+?)$", preview,
    )
    raw_from = (from_match.group(1) if from_match else "").strip()
    subject = (subject_match.group(1) if subject_match else "").strip()
    # Parse 'Name <email@x>' or just 'email@x'.
    addr_match = re.search(r"<([^>]+)>", raw_from)
    if addr_match:
        email_addr = addr_match.group(1).strip()
        display = raw_from.split("<")[0].strip().strip('"')
    else:
        email_addr = raw_from
        display = raw_from.split("@")[0] if "@" in raw_from else raw_from
    return email_addr, display, subject
