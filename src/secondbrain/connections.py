"""Round 9-B — stale-connection detector.

The brain knows everyone you've corresponded with (people table) and
when you last saw them (last_seen_at, kept fresh by the indexer's
mention-linking). This module surfaces folks worth reaching back out
to: high prior relationship strength + long silence.

Heuristic:
  score = mention_count * decay(days_since_seen)

Where ``decay`` ramps up linearly past ``min_age_days`` (default 60)
and saturates at the ``max_age_days`` plateau (default 365). So a
person you've never gone silent on (last_seen 2 weeks ago) scores 0;
someone you saw 90 days ago with 50 prior mentions scores high; and
someone who's been gone 5 years (probably a moved-on relationship)
scores about the same as 1 year — we don't keep escalating forever.

No new schema. Pure read against people + we don't persist scores;
they're cheap to recompute.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)


# ---- Tunables -------------------------------------------------------

# Don't suggest reconnecting until this many days of silence have
# passed. Two months covers the "we used to talk weekly, now we
# haven't in a while" case without flagging recent contacts.
_MIN_AGE_DAYS = 60
# Cap the silence-bonus past this. Five-year-gone contacts probably
# don't want a cold "checking in" email; the user can still surface
# them via the explicit /people page if they want to reach out.
_MAX_AGE_DAYS = 365
# Need at least this many prior mentions before someone counts as a
# real connection vs. a one-off cc.
_MIN_MENTIONS = 3
# Default cap on how many to surface in the brief / CLI output.
_DEFAULT_LIMIT = 5


@dataclass
class StaleConnection:
    """One stale connection candidate, ready for rendering."""
    person_id: int
    name: str
    email: str
    company: str
    role: str
    days_since_seen: int
    mention_count: int
    score: float

    @property
    def months_since_seen(self) -> int:
        return self.days_since_seen // 30


def find_stale_connections(
    conn: sqlite3.Connection,
    *,
    min_age_days: int = _MIN_AGE_DAYS,
    max_age_days: int = _MAX_AGE_DAYS,
    min_mentions: int = _MIN_MENTIONS,
    limit: int = _DEFAULT_LIMIT,
) -> list[StaleConnection]:
    """Round 9-B — return the top stale connections worth reaching out
    to.

    Best-effort: missing schema (fresh brain with no people table)
    yields an empty list rather than raising.
    """
    try:
        rows = conn.execute(
            "SELECT id, canonical_name, display_name, email, "
            "  company, role, last_seen_at, mention_count "
            "FROM people "
            "WHERE mention_count >= ? "
            "ORDER BY mention_count DESC LIMIT 500",
            (min_mentions,),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    now = time.time()
    candidates: list[StaleConnection] = []
    for r in rows:
        last_seen = r["last_seen_at"] or 0.0
        if not last_seen:
            continue
        days_since = max(0, int((now - last_seen) // 86400))
        if days_since < min_age_days:
            continue
        # Linear ramp from 0 at min_age → 1 at max_age, clamped.
        ramp = max(
            0.0,
            min(
                1.0,
                (days_since - min_age_days)
                / max(1, max_age_days - min_age_days),
            ),
        )
        score = float(r["mention_count"] or 0) * (0.5 + 0.5 * ramp)
        candidates.append(StaleConnection(
            person_id=int(r["id"]),
            name=r["display_name"] or r["canonical_name"],
            email=r["email"] or "",
            company=r["company"] or "",
            role=r["role"] or "",
            days_since_seen=days_since,
            mention_count=int(r["mention_count"] or 0),
            score=score,
        ))
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:limit]


def render_stale_block(items: list[StaleConnection]) -> str:
    """Markdown for the daily brief. Empty list → empty string so
    the brief skips the section entirely on quiet days."""
    if not items:
        return ""
    lines = ["## Worth reaching back out", ""]
    for c in items:
        months = c.months_since_seen
        when = (
            f"{months} month{'s' if months != 1 else ''}"
            if months >= 1 else f"{c.days_since_seen}d"
        )
        meta_bits: list[str] = []
        if c.role:
            meta_bits.append(c.role)
        if c.company:
            meta_bits.append(f"@ {c.company}")
        meta = f" _{'  '.join(meta_bits)}_" if meta_bits else ""
        lines.append(
            f"- **{c.name}** — last seen {when} ago "
            f"({c.mention_count} prior mentions){meta}",
        )
    return "\n".join(lines)
