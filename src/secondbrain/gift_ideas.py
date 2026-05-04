"""Round 19 (Phase EA-7) — birthday gift ideation.

When a person's birthday is within ``HORIZON_DAYS``, generate 3
gift ideas tailored to what we know about them — profile notes,
recent conversation topics, recurring entities (hobbies / books /
brands they care about). Persisted in ``gift_ideas`` so we don't
re-generate on every page load; users can also dismiss / regenerate.

Privacy: every input that goes into the LLM prompt passes through
``_safe_for_prompt`` (round 13 invariant); the persisted ideas are
re-redacted before write.

Cost: one Haiku call per person per occasion. With ~10 birthdays/
year in the horizon, this is sub-$0.01 of Anthropic spend annually.
Gated by per-feature budget bucket ``gift_ideas``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import weakref as _weakref
from dataclasses import asdict, dataclass

from .config import Config

log = logging.getLogger(__name__)

_SCHEMA_INITIALIZED: _weakref.WeakSet = _weakref.WeakSet()

_GIFT_MODEL = "claude-haiku-4-5"
HORIZON_DAYS = 14


@dataclass
class GiftIdea:
    title: str
    description: str
    why: str            # 1 sentence connecting to what we know
    price_range: str = ""  # "$20-50", optional


@dataclass
class GiftIdeas:
    person_id: int
    occasion: str       # 'birthday'
    ideas: list[GiftIdea]
    generated_at: float
    model: str

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "occasion": self.occasion,
            "ideas": [asdict(i) for i in self.ideas],
            "generated_at": self.generated_at,
            "model": self.model,
        }


def _ensure_schema(conn: sqlite3.Connection) -> None:
    try:
        if conn in _SCHEMA_INITIALIZED:
            return
    except TypeError:
        pass
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS gift_ideas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL REFERENCES people(id)
                ON DELETE CASCADE,
            occasion TEXT NOT NULL DEFAULT 'birthday',
            ideas_json TEXT NOT NULL DEFAULT '[]',
            generated_at REAL NOT NULL,
            model TEXT NOT NULL DEFAULT '',
            UNIQUE(person_id, occasion)
        );
        CREATE INDEX IF NOT EXISTS idx_gift_ideas_generated
            ON gift_ideas(generated_at DESC);
    """)
    conn.commit()
    try:
        _SCHEMA_INITIALIZED.add(conn)
    except TypeError:
        pass


_GIFT_SYSTEM = """\
You are an executive assistant suggesting birthday gifts for a person
the user knows. The user gives you what they know about the person
(profile notes, recent topics, hobbies). You return EXACTLY 3 gift
ideas as a JSON array of objects with:

  title: 4-8 word gift name
  description: 1-2 sentence concrete description (specific brand/item
               OK, but if vague is fine too)
  why: 1 sentence linking to what the user knows about the person
  price_range: rough dollar range like "$20-50" or "$$$"; "" if N/A

Rules:
  - Be concrete and varied across the 3. Don't suggest 3 books.
  - Don't suggest anything generic ("a nice candle"). Tie to context.
  - Don't suggest anything inappropriate (no alcohol if context
    suggests sobriety; no diet products; etc.).
  - Return ONLY the JSON array.
"""


def _gather_context(
    conn: sqlite3.Connection, person_id: int,
) -> str:
    """Pull profile + recent topics + journal mentions to feed the
    LLM. All passed through redaction."""
    try:
        from . import people as people_mod
        p = people_mod.get_person(conn, person_id)
    except Exception:  # noqa: BLE001
        return ""
    if p is None:
        return ""
    parts: list[str] = [f"Display name: {p.display_name}"]
    if p.role:
        parts.append(f"Role: {p.role}")
    if p.company:
        parts.append(f"Company: {p.company}")
    if p.notes:
        parts.append(f"User's notes about them: {p.notes}")
    # Recent shared topics (entities) — last 90 days.
    cutoff = time.time() - 90 * 86400
    try:
        rows = conn.execute(
            "SELECT MIN(e.text) AS text, e.label, COUNT(*) AS n "
            "FROM person_mentions pm "
            "JOIN chunks c ON c.file_id = pm.file_id "
            "JOIN entities e ON e.chunk_id = c.id "
            "JOIN files f ON f.id = pm.file_id "
            "WHERE pm.person_id = ? AND f.indexed_at >= ? "
            "  AND e.label IN ('PRODUCT','WORK_OF_ART','EVENT','ORG') "
            "GROUP BY e.text_lower, e.label "
            "ORDER BY n DESC LIMIT 12",
            (person_id, cutoff),
        ).fetchall()
        if rows:
            topics = ", ".join(
                f"{r['text']} ({r['label']})" for r in rows
            )
            parts.append(f"Recent shared topics: {topics}")
    except sqlite3.OperationalError:
        pass
    # Journal entries mentioning them.
    try:
        jrows = conn.execute(
            "SELECT text FROM journal_entries "
            "WHERE LOWER(text) LIKE ? "
            "ORDER BY date DESC LIMIT 5",
            (f"%{p.display_name.lower()}%",),
        ).fetchall()
        if jrows:
            joined = " | ".join(
                (j["text"] or "")[:200] for j in jrows
            )
            parts.append(f"Journal mentions: {joined}")
    except sqlite3.OperationalError:
        pass
    raw = "\n".join(parts)
    try:
        from .email_assist import _safe_for_prompt
        return _safe_for_prompt(raw, max_chars=3000)
    except ImportError:
        return raw[:3000]


def generate_for_person(
    conn: sqlite3.Connection,
    cfg: Config,
    person_id: int,
    *,
    occasion: str = "birthday",
    overwrite: bool = False,
) -> GiftIdeas | None:
    """Generate (and persist) 3 gift ideas for one person. Returns
    None if no API key or budget exceeded.

    Idempotent: if a row exists and overwrite=False, returns the
    cached row.
    """
    _ensure_schema(conn)
    if not overwrite:
        row = conn.execute(
            "SELECT * FROM gift_ideas WHERE person_id = ? "
            "AND occasion = ?",
            (person_id, occasion),
        ).fetchone()
        if row:
            return _row_to_ideas(row)

    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    try:
        from .budget import (
            check_budget,
            record_usage,
        )
        check_budget(cfg, "anthropic", feature="gift_ideas")
    except Exception:  # noqa: BLE001 — also catches BudgetExceededError
        return None

    context = _gather_context(conn, person_id)
    if not context:
        return None
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=_GIFT_MODEL,
            max_tokens=800,
            system=_GIFT_SYSTEM,
            messages=[{
                "role": "user",
                "content": (
                    f"Occasion: {occasion}\n\n"
                    f"What I know about them:\n{context}"
                ),
            }],
        )
    except anthropic.APIError as e:
        log.warning("gift_ideas: API error: %s", e)
        return None
    in_tok = getattr(resp.usage, "input_tokens", 0)
    out_tok = getattr(resp.usage, "output_tokens", 0)
    try:
        record_usage(
            cfg, "anthropic", _GIFT_MODEL,
            input_tokens=in_tok, output_tokens=out_tok,
            note=f"gift_ideas/{person_id}",
            feature="gift_ideas",
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
        items = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning("gift_ideas: bad JSON: %s", e)
        return None
    if not isinstance(items, list):
        return None
    ideas: list[GiftIdea] = []
    try:
        from .safety import redact_text
    except ImportError:
        def redact_text(s):
            return s
    for it in items[:3]:
        if not isinstance(it, dict):
            continue
        ideas.append(GiftIdea(
            title=redact_text(str(it.get("title") or ""))[:120],
            description=redact_text(
                str(it.get("description") or "")
            )[:400],
            why=redact_text(str(it.get("why") or ""))[:300],
            price_range=str(it.get("price_range") or "")[:30],
        ))
    if not ideas:
        return None
    now = time.time()
    conn.execute(
        "INSERT OR REPLACE INTO gift_ideas"
        "(person_id, occasion, ideas_json, generated_at, model) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            person_id, occasion,
            json.dumps([asdict(i) for i in ideas]),
            now, _GIFT_MODEL,
        ),
    )
    conn.commit()
    return GiftIdeas(
        person_id=person_id, occasion=occasion,
        ideas=ideas, generated_at=now, model=_GIFT_MODEL,
    )


def _row_to_ideas(row) -> GiftIdeas:
    return GiftIdeas(
        person_id=int(row["person_id"]),
        occasion=row["occasion"] or "birthday",
        ideas=[
            GiftIdea(**i)
            for i in json.loads(row["ideas_json"] or "[]")
            if isinstance(i, dict)
        ],
        generated_at=float(row["generated_at"]),
        model=row["model"] or "",
    )


def get_for_person(
    conn: sqlite3.Connection, person_id: int,
    occasion: str = "birthday",
) -> GiftIdeas | None:
    _ensure_schema(conn)
    row = conn.execute(
        "SELECT * FROM gift_ideas "
        "WHERE person_id = ? AND occasion = ?",
        (person_id, occasion),
    ).fetchone()
    return _row_to_ideas(row) if row else None


def list_for_upcoming_birthdays(
    conn: sqlite3.Connection,
    cfg: Config,
    *,
    horizon_days: int = HORIZON_DAYS,
    auto_generate: bool = True,
    max_generations: int = 5,
) -> list[tuple]:
    """For each person with a birthday in the next ``horizon_days``,
    return (person, days_until, gift_ideas). If ``auto_generate``,
    call the LLM for the first ``max_generations`` people who don't
    yet have cached ideas.

    Returns: [(Person, days_until: int, GiftIdeas|None), ...]
    """
    from datetime import date

    from . import people as people_mod
    rows = people_mod.list_people(conn, order="name", limit=1000)
    today = date.today()
    horizon = today + _td(days=horizon_days)
    out: list[tuple] = []
    n_generated = 0
    for p in rows:
        bday = (p.birthday or "").strip()
        if not bday:
            continue
        try:
            parts = bday.split("-")
            if len(parts) == 3:
                mm, dd = int(parts[1]), int(parts[2])
            elif len(parts) == 2:
                mm, dd = int(parts[0]), int(parts[1])
            else:
                continue
        except ValueError:
            continue
        try:
            from .notifications import _safe_date_in_year
            this_year = _safe_date_in_year(today.year, mm, dd)
            if this_year is None:
                continue
            if this_year < today:
                this_year = _safe_date_in_year(today.year + 1, mm, dd)
                if this_year is None:
                    continue
        except ImportError:
            try:
                this_year = date(today.year, mm, dd)
                if this_year < today:
                    this_year = date(today.year + 1, mm, dd)
            except ValueError:
                continue
        if this_year > horizon:
            continue
        days_until = (this_year - today).days
        existing = get_for_person(conn, p.id)
        if existing is None and auto_generate and n_generated < max_generations:
            existing = generate_for_person(conn, cfg, p.id)
            n_generated += 1
        out.append((p, days_until, existing))
    out.sort(key=lambda t: t[1])
    return out


def _td(*, days: int):
    """Local helper to keep the import surface small."""
    from datetime import timedelta
    return timedelta(days=days)
