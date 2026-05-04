"""Phase 65 + 66: people as first-class.

The ``entities`` table is per-chunk and noisy — every spaCy PERSON
mention lands there with whatever surface form appeared in the text.
``people`` is the curated layer above: one row per resolved human,
with aliases that fold all the surface forms back to the canonical
identity.

What a person carries:

  - **Profile**: display name, primary email, company, role, notes,
    birthday. User-editable via ``secondbrain people edit``.
  - **Mention history**: a backlink-style index from chunks to people.
    Lets retrieval answer 'every doc that mentions Sarah' in O(log n)
    without re-running entity extraction.
  - **Activity**: first/last seen timestamps + mention count for
    `secondbrain people` listing sorted by recency or volume.

Two materialisation paths:

  1. ``materialize_from_entities()``: bulk-resolve PERSON entities
    into people. Run once on schema migration + on demand. Uses
    canonical-name dedup so 'Sarah Chen' / 'sarah chen' / 'Sarah
    chen' merge.
  2. ``link_chunk_mentions(chunk_id, text)``: called by the indexer
    after each chunk lands. Looks up known aliases in the chunk text
    and inserts person_mentions rows. This is the Phase 66 hook that
    makes 'when "Sarah" appears in a new doc, link to her profile'
    automatic.

Future hooks (not built yet, schema-supports):

  - Per-person calendar event affinity (mentions in events)
  - Communication recency: 'last contacted 17d ago' from email
  - Auto-fill email + company from email connector signals
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


# ---- Tunables ---------------------------------------------------------

# When auto-linking per chunk, cap how many distinct aliases we scan
# in one pass. With ~thousands of aliases the regex compile would
# dominate; we bucket the scan into batches of this size.
_ALIAS_BATCH_SIZE = 200

# Aliases shorter than this are too ambiguous to auto-link safely
# ('Al', 'Bo', 'Pi') — they'd produce a flood of false positives.
# Users who genuinely have a 2-letter alias for someone can add it
# manually + opt out via people.unlink.
_MIN_ALIAS_LEN = 3


# ---- Data shapes ------------------------------------------------------

@dataclass
class Person:
    id: int
    canonical_name: str
    display_name: str
    email: str = ""
    company: str = ""
    role: str = ""
    notes: str = ""
    birthday: str = ""
    first_seen_at: float = 0.0
    last_seen_at: float = 0.0
    mention_count: int = 0


@dataclass
class PersonProfile:
    """A profile view fed to the dashboard / CLI / chat agent. Combines
    the static fields + recent mentions + computed signals."""
    person: Person
    aliases: list[str] = field(default_factory=list)
    recent_mentions: list[MentionRow] = field(default_factory=list)
    days_since_seen: int = 0
    days_since_first_seen: int = 0


@dataclass
class MentionRow:
    """One row in a person's mention timeline."""
    chunk_id: int
    file_id: int
    file_path: str
    chunk_text_preview: str  # first ~200 chars
    mtime: float


# ============================ resolution ==============================

def canonicalize(name: str) -> str:
    """Lowercase + collapse whitespace. The de-dup key for ``people``.

    Round 18 fix (audit-found gap M7) — NFC-normalize Unicode before
    casefolding. Without this, the same name typed via macOS keyboard
    (which produces NFD: ``o`` + combining acute) and a connector
    that produces NFC (single precomposed ``ó``) compared unequal,
    creating duplicate ``people`` rows that no merge ever caught.
    ``casefold`` is the Unicode-aware lowering that handles cases
    like German ß → ss correctly.
    """
    import unicodedata
    return " ".join(unicodedata.normalize("NFC", name).casefold().split())


def upsert_person(
    conn: sqlite3.Connection,
    *,
    display_name: str,
    email: str = "",
    company: str = "",
    role: str = "",
    when: float | None = None,
) -> int:
    """Find-or-create a person row by canonical_name.

    Touches ``last_seen_at`` + bumps ``mention_count`` so callers get
    "found in another doc" semantics for free. Returns the person id.
    """
    canonical = canonicalize(display_name)
    if not canonical:
        raise ValueError("display_name must be non-empty")
    n = when if when is not None else time.time()
    row = conn.execute(
        "SELECT id FROM people WHERE canonical_name = ?", (canonical,),
    ).fetchone()
    if row is not None:
        pid = int(row["id"])
        # Bump activity. Keep the user's edits to display_name / email /
        # company / role intact — only fill them when previously empty
        # so connector sync doesn't clobber manual data.
        conn.execute(
            "UPDATE people SET "
            "  last_seen_at = MAX(last_seen_at, ?), "
            "  mention_count = mention_count + 1, "
            "  email = CASE WHEN email = '' OR email IS NULL THEN ? ELSE email END, "
            "  company = CASE WHEN company = '' OR company IS NULL THEN ? ELSE company END, "
            "  role = CASE WHEN role = '' OR role IS NULL THEN ? ELSE role END "
            "WHERE id = ?",
            (n, email or "", company or "", role or "", pid),
        )
        conn.commit()
        return pid
    cur = conn.execute(
        "INSERT INTO people"
        "(canonical_name, display_name, email, company, role, "
        " first_seen_at, last_seen_at, mention_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 1) RETURNING id",
        (canonical, display_name.strip(),
         email or "", company or "", role or "", n, n),
    )
    pid = int(cur.fetchone()["id"])
    # Seed an alias for the canonical name itself so link_chunk_mentions
    # picks it up without an explicit add_alias call.
    add_alias(conn, pid, display_name.strip())
    conn.commit()
    return pid


def add_alias(conn: sqlite3.Connection, person_id: int, alias: str) -> bool:
    """Register an alias. Idempotent — re-adds are no-ops. Returns True
    if a new row landed.

    Round 18 fix (audit-found gap M7) — uses ``canonicalize`` for
    ``alias_lower`` so NFC vs. NFD spellings of the same alias dedup
    cleanly across reads/writes.
    """
    a = alias.strip()
    if not a:
        return False
    cur = conn.execute(
        "INSERT OR IGNORE INTO person_aliases"
        "(person_id, alias, alias_lower, created_at) "
        "VALUES (?, ?, ?, ?)",
        (person_id, a, canonicalize(a), time.time()),
    )
    conn.commit()
    return cur.rowcount > 0


def remove_alias(conn: sqlite3.Connection, alias: str) -> bool:
    """Drop an alias by its surface text. Returns True iff a row was
    deleted. Used by ``people unlink`` when the auto-linker fires on
    a false positive."""
    cur = conn.execute(
        "DELETE FROM person_aliases WHERE alias_lower = ?",
        (canonicalize(alias.strip()),),
    )
    conn.commit()
    return cur.rowcount > 0


def merge_people(
    conn: sqlite3.Connection, into_id: int, from_id: int,
) -> None:
    """Merge ``from_id`` into ``into_id``. Aliases + mentions move
    over; the source row is deleted. Used by ``people merge`` when
    the auto-resolver split one human into two profiles."""
    if into_id == from_id:
        return
    conn.execute(
        "UPDATE OR IGNORE person_aliases SET person_id = ? "
        "WHERE person_id = ?",
        (into_id, from_id),
    )
    # Any aliases that conflicted (UPDATE OR IGNORE skipped) get
    # deleted with their source row's CASCADE.
    conn.execute(
        "UPDATE OR IGNORE person_mentions SET person_id = ? "
        "WHERE person_id = ?",
        (into_id, from_id),
    )
    # Roll-up mention_count + extend window.
    conn.execute(
        "UPDATE people SET "
        "  mention_count = mention_count + (SELECT mention_count FROM people WHERE id = ?), "
        "  first_seen_at = MIN(first_seen_at, "
        "    (SELECT first_seen_at FROM people WHERE id = ?)), "
        "  last_seen_at = MAX(last_seen_at, "
        "    (SELECT last_seen_at FROM people WHERE id = ?)) "
        "WHERE id = ?",
        (from_id, from_id, from_id, into_id),
    )
    conn.execute("DELETE FROM people WHERE id = ?", (from_id,))
    conn.commit()


# ============================ retrieval ===============================

def get_person(
    conn: sqlite3.Connection, person_id: int,
) -> Person | None:
    row = conn.execute(
        "SELECT * FROM people WHERE id = ?", (person_id,),
    ).fetchone()
    return _row_to_person(row) if row else None


def find_by_alias(
    conn: sqlite3.Connection, alias: str,
) -> Person | None:
    """Resolve any alias (or canonical name) to a person."""
    row = conn.execute(
        "SELECT p.* FROM people p "
        "JOIN person_aliases a ON a.person_id = p.id "
        "WHERE a.alias_lower = ? LIMIT 1",
        (alias.strip().lower(),),
    ).fetchone()
    return _row_to_person(row) if row else None


def list_people(
    conn: sqlite3.Connection, *, order: str = "recent", limit: int = 100,
) -> list[Person]:
    """List people, ordered by recency (default) or mention count.

    'recent' → ``last_seen_at DESC``: who you've encountered lately.
    'mentions' → ``mention_count DESC``: who shows up most often.
    'name' → ``display_name ASC``: alphabetical.
    """
    order_clause = {
        "recent": "last_seen_at DESC",
        "mentions": "mention_count DESC, last_seen_at DESC",
        "name": "display_name ASC",
    }.get(order, "last_seen_at DESC")
    rows = conn.execute(
        f"SELECT * FROM people ORDER BY {order_clause} LIMIT ?",
        (limit,),
    ).fetchall()
    return [_row_to_person(r) for r in rows]


def search_people(
    conn: sqlite3.Connection, query: str, limit: int = 20,
) -> list[Person]:
    """Substring search across name + email. Case-insensitive."""
    q = query.strip().lower()
    if not q:
        return []
    like = f"%{q}%"
    rows = conn.execute(
        "SELECT DISTINCT p.* FROM people p "
        "LEFT JOIN person_aliases a ON a.person_id = p.id "
        "WHERE p.canonical_name LIKE ? "
        "   OR LOWER(p.email) LIKE ? "
        "   OR a.alias_lower LIKE ? "
        "ORDER BY p.last_seen_at DESC LIMIT ?",
        (like, like, like, limit),
    ).fetchall()
    return [_row_to_person(r) for r in rows]


def get_aliases(conn: sqlite3.Connection, person_id: int) -> list[str]:
    rows = conn.execute(
        "SELECT alias FROM person_aliases WHERE person_id = ? "
        "ORDER BY alias",
        (person_id,),
    ).fetchall()
    return [r["alias"] for r in rows]


def recent_mentions(
    conn: sqlite3.Connection, person_id: int, limit: int = 20,
) -> list[MentionRow]:
    """Most-recent docs that mention this person. Used by the profile
    view + chat agent's "what did Sarah and I talk about?" answer.

    Returns one row per (chunk, mtime); the dashboard groups by file.
    """
    rows = conn.execute(
        "SELECT m.chunk_id, m.file_id, m.mtime, "
        "       f.path AS file_path, c.text AS chunk_text "
        "FROM person_mentions m "
        "JOIN files f ON f.id = m.file_id "
        "JOIN chunks c ON c.id = m.chunk_id "
        "WHERE m.person_id = ? "
        "ORDER BY m.mtime DESC LIMIT ?",
        (person_id, limit),
    ).fetchall()
    out: list[MentionRow] = []
    for r in rows:
        text = (r["chunk_text"] or "").strip()
        if len(text) > 200:
            text = text[:200] + "…"
        out.append(MentionRow(
            chunk_id=int(r["chunk_id"]),
            file_id=int(r["file_id"]),
            file_path=r["file_path"],
            chunk_text_preview=text,
            mtime=r["mtime"],
        ))
    return out


def profile_for(
    conn: sqlite3.Connection, person_id: int,
) -> PersonProfile | None:
    p = get_person(conn, person_id)
    if p is None:
        return None
    now = time.time()
    return PersonProfile(
        person=p,
        aliases=get_aliases(conn, person_id),
        recent_mentions=recent_mentions(conn, person_id, limit=20),
        days_since_seen=max(
            0, int((now - p.last_seen_at) // 86400),
        ),
        days_since_first_seen=max(
            0, int((now - p.first_seen_at) // 86400),
        ),
    )


# ---- Full context (round 9 shared helper) ----------------------------

@dataclass
class PersonContext:
    """Everything the brain knows about a person, in one shape.

    Powers meeting prep (round 9-A), stale-connection detection
    (round 9-B), and recipient-matching for the structured task
    extractor (round 9-C). Each field is best-effort: missing
    schemas (fresh brain, no email indexed yet) just yield empty
    lists, never raise.
    """
    person: Person
    aliases: list[str]
    days_since_seen: int
    days_since_first_seen: int
    # Recent docs that mention them (limit ~10).
    recent_mentions: list[MentionRow]
    # Email files where they appear as From: (limit ~5). Path strings
    # so callers can hydrate via the existing search infra.
    prior_emails: list[tuple[str, float]]   # [(path, mtime), ...]
    # Open tasks whose text or source mentions this person (limit ~10).
    open_tasks: list[tuple[int, str]]       # [(task_id, text), ...]
    # Top entities co-occurring in their mention chunks — proxy for
    # "topics this person cares about / is associated with".
    co_topics: list[tuple[str, int]]        # [(entity_text, count), ...]


def gather_full_context(
    conn: sqlite3.Connection, person_id: int,
    *,
    n_mentions: int = 10,
    n_emails: int = 5,
    n_tasks: int = 10,
    n_topics: int = 8,
) -> PersonContext | None:
    """Round 9 — pull everything we know about a person into one
    structured shape. Each query is independently best-effort so a
    missing schema doesn't cascade.

    The output is intentionally retrieval-only (no LLM calls). Callers
    that want a polished prose render layer it on top with their own
    prompt.
    """
    p = get_person(conn, person_id)
    if p is None:
        return None
    now = time.time()
    aliases = get_aliases(conn, person_id)

    # Recent mentions — already a public helper, just reuse.
    mentions = recent_mentions(conn, person_id, limit=n_mentions)

    # Prior emails: any email file whose first chunk has 'From: <name>'
    # or 'From: <email>'. We match aliases case-insensitively.
    prior_emails: list[tuple[str, float]] = []
    if aliases:
        like_clauses = " OR ".join(
            "LOWER(c.text) LIKE ?" for _ in aliases
        )
        params: list = []
        for alias in aliases:
            params.append(f"%from:%{alias.lower()}%")
        try:
            rows = conn.execute(
                "SELECT DISTINCT f.path, f.mtime FROM files f "
                "JOIN chunks c ON c.file_id = f.id "
                f"WHERE c.chunk_index = 0 AND ({like_clauses}) "
                "  AND (f.path LIKE 'imap://%' OR f.path LIKE 'gmail://%') "
                "ORDER BY f.mtime DESC LIMIT ?",
                [*params, n_emails],
            ).fetchall()
            prior_emails = [(r["path"], float(r["mtime"])) for r in rows]
        except sqlite3.OperationalError:
            # Email tables not present yet on a fresh brain — fine.
            prior_emails = []

    # Open tasks mentioning the person. We match against task text +
    # source_title. No FK by person_id (the round-9-C extractor adds
    # that), so substring-match on aliases is the heuristic.
    open_tasks: list[tuple[int, str]] = []
    if aliases:
        try:
            like_clauses = " OR ".join(
                "LOWER(text) LIKE ? OR LOWER(source_title) LIKE ?"
                for _ in aliases
            )
            params = []
            for alias in aliases:
                params.extend([f"%{alias.lower()}%", f"%{alias.lower()}%"])
            rows = conn.execute(
                "SELECT id, text FROM tasks "
                "WHERE status = 'open' "
                f"  AND ({like_clauses}) "
                "ORDER BY created_at DESC LIMIT ?",
                [*params, n_tasks],
            ).fetchall()
            open_tasks = [(int(r["id"]), r["text"]) for r in rows]
        except sqlite3.OperationalError:
            open_tasks = []

    # Co-occurring topics: entities that show up in chunks where
    # this person is mentioned. Cheap signal for "what do we usually
    # talk about with them."
    co_topics: list[tuple[str, int]] = []
    try:
        rows = conn.execute(
            "SELECT e.text, COUNT(*) AS n FROM entities e "
            "JOIN person_mentions pm ON pm.chunk_id = e.chunk_id "
            "WHERE pm.person_id = ? "
            "  AND e.label IN ('ORG', 'PRODUCT', 'WORK_OF_ART', 'EVENT') "
            "GROUP BY e.text_lower ORDER BY n DESC LIMIT ?",
            (person_id, n_topics),
        ).fetchall()
        co_topics = [(r["text"], int(r["n"])) for r in rows]
    except sqlite3.OperationalError:
        co_topics = []

    return PersonContext(
        person=p,
        aliases=aliases,
        days_since_seen=max(0, int((now - p.last_seen_at) // 86400)),
        days_since_first_seen=max(
            0, int((now - p.first_seen_at) // 86400),
        ),
        recent_mentions=mentions,
        prior_emails=prior_emails,
        open_tasks=open_tasks,
        co_topics=co_topics,
    )


def gather_full_context_by_alias(
    conn: sqlite3.Connection, name_or_email: str,
) -> PersonContext | None:
    """Convenience wrapper — accepts a name / alias / email and
    resolves to a person_id before pulling context. Returns None
    when the alias doesn't map to anyone."""
    p = find_by_alias(conn, name_or_email)
    if p is None:
        return None
    return gather_full_context(conn, p.id)


# ============================ profile edits ===========================

def set_field(
    conn: sqlite3.Connection, person_id: int,
    *, email: str | None = None, company: str | None = None,
    role: str | None = None, notes: str | None = None,
    birthday: str | None = None,
) -> bool:
    """Update one or more profile fields. Empty string clears; None
    leaves unchanged. Returns True iff at least one value changed."""
    updates: list[str] = []
    params: list = []
    for col, value in [
        ("email", email), ("company", company), ("role", role),
        ("notes", notes), ("birthday", birthday),
    ]:
        if value is not None:
            updates.append(f"{col} = ?")
            params.append(value)
    if not updates:
        return False
    params.append(person_id)
    cur = conn.execute(
        f"UPDATE people SET {', '.join(updates)} WHERE id = ?", params,
    )
    conn.commit()
    return cur.rowcount > 0


# ============================ mention linking ==========================

# Cache for compiled alias regexes — cleared by clear_alias_cache()
# on alias edits so we don't serve stale matchers.
_ALIAS_REGEX_CACHE: tuple[re.Pattern[str], dict[str, int]] | None = None


def clear_alias_cache() -> None:
    """Drop the compiled alias regex. Call after add/remove_alias when
    a long-running daemon would otherwise serve stale matchers."""
    global _ALIAS_REGEX_CACHE
    _ALIAS_REGEX_CACHE = None


def _build_alias_matcher(
    conn: sqlite3.Connection,
) -> tuple[re.Pattern[str] | None, dict[str, int]]:
    """Compile every alias into one big regex with word boundaries.
    Returns (pattern, alias_lower → person_id map). Empty alias table
    → (None, {})."""
    global _ALIAS_REGEX_CACHE
    if _ALIAS_REGEX_CACHE is not None:
        return _ALIAS_REGEX_CACHE
    rows = conn.execute(
        "SELECT alias, alias_lower, person_id FROM person_aliases",
    ).fetchall()
    if not rows:
        return None, {}
    # Filter out too-short aliases (per _MIN_ALIAS_LEN). Sort by length
    # descending so 'Sarah Chen' wins before 'Sarah' would.
    parts: list[str] = []
    mapping: dict[str, int] = {}
    for r in sorted(rows, key=lambda x: len(x["alias_lower"]), reverse=True):
        alias_lower = r["alias_lower"]
        if len(alias_lower) < _MIN_ALIAS_LEN:
            continue
        mapping[alias_lower] = int(r["person_id"])
        parts.append(re.escape(r["alias"]))
    if not parts:
        return None, {}
    # Word boundaries so 'Sarah' doesn't match inside 'Sarahsplaining'.
    pattern = re.compile(
        r"\b(?:" + "|".join(parts) + r")\b",
        re.IGNORECASE,
    )
    _ALIAS_REGEX_CACHE = (pattern, mapping)
    return pattern, mapping


def link_chunk_mentions(
    conn: sqlite3.Connection,
    chunk_id: int, file_id: int, text: str, mtime: float,
) -> int:
    """Scan ``text`` for any known alias, insert person_mentions rows
    for matches. Idempotent — UNIQUE on (person_id, chunk_id) keeps
    re-runs from duplicating.

    Returns the number of distinct people newly linked. Called by
    the indexer hook after each chunk lands.
    """
    pattern, mapping = _build_alias_matcher(conn)
    if pattern is None:
        return 0
    seen_persons: set[int] = set()
    for m in pattern.finditer(text or ""):
        alias_lower = m.group(0).lower()
        pid = mapping.get(alias_lower)
        if pid is None:
            continue
        if pid in seen_persons:
            continue
        seen_persons.add(pid)
    if not seen_persons:
        return 0
    inserted = 0
    for pid in seen_persons:
        cur = conn.execute(
            "INSERT OR IGNORE INTO person_mentions"
            "(person_id, chunk_id, file_id, mtime) "
            "VALUES (?, ?, ?, ?)",
            (pid, chunk_id, file_id, mtime),
        )
        if cur.rowcount > 0:
            inserted += 1
            # Bump person activity timestamps so the listing reflects
            # the latest mention without a separate update pass.
            conn.execute(
                "UPDATE people SET "
                "  last_seen_at = MAX(last_seen_at, ?) "
                "WHERE id = ?",
                (mtime, pid),
            )
    if inserted:
        conn.commit()
    return inserted


def link_file_mentions(
    conn: sqlite3.Connection, file_id: int,
) -> int:
    """Re-run mention linking across every chunk of a file. Used by
    the indexer's per-file hook + by ``people relink`` for backfills.
    Returns total mentions inserted."""
    rows = conn.execute(
        "SELECT id, text FROM chunks WHERE file_id = ?", (file_id,),
    ).fetchall()
    file_row = conn.execute(
        "SELECT mtime FROM files WHERE id = ?", (file_id,),
    ).fetchone()
    if file_row is None:
        return 0
    mtime = file_row["mtime"]
    total = 0
    for r in rows:
        total += link_chunk_mentions(
            conn, int(r["id"]), file_id, r["text"] or "", mtime,
        )
    return total


# ============================ bulk materialisation ====================

# Minimum mention count before an entity becomes a person. Single-shot
# spaCy false positives ('John' inside a code variable name) shouldn't
# create profiles. Two distinct chunks is a reasonable bar.
_MIN_ENTITY_MENTIONS_TO_PROMOTE = 2


def materialize_from_entities(
    conn: sqlite3.Connection,
    *,
    min_mentions: int = _MIN_ENTITY_MENTIONS_TO_PROMOTE,
) -> int:
    """Promote PERSON entities into ``people`` rows. Idempotent.

    Bulk pass for backfilling on a brain that pre-dates the people
    module. Skips entities that appear only once — too noisy.
    Returns the count of new people created.
    """
    rows = conn.execute(
        "SELECT text, COUNT(DISTINCT chunk_id) AS n "
        "FROM entities WHERE label = 'PERSON' "
        "GROUP BY text_lower "
        "HAVING n >= ? "
        "ORDER BY n DESC",
        (min_mentions,),
    ).fetchall()
    created = 0
    for r in rows:
        # upsert_person dedups by canonical_name so re-running is safe.
        existing = find_by_alias(conn, r["text"])
        if existing is not None:
            continue
        upsert_person(conn, display_name=r["text"])
        created += 1
    if created:
        # Invalidate alias cache so subsequent link_chunk_mentions
        # picks up the new aliases.
        clear_alias_cache()
    return created


# ============================ helpers =================================

def _row_to_person(row: sqlite3.Row) -> Person:
    return Person(
        id=int(row["id"]),
        canonical_name=row["canonical_name"],
        display_name=row["display_name"],
        email=row["email"] or "",
        company=row["company"] or "",
        role=row["role"] or "",
        notes=row["notes"] or "",
        birthday=row["birthday"] or "",
        first_seen_at=row["first_seen_at"],
        last_seen_at=row["last_seen_at"],
        mention_count=int(row["mention_count"]),
    )


# ---- Indexer hook ----------------------------------------------------

def link_after_index(conn: sqlite3.Connection, file_id: int) -> None:
    """Indexer-side wrapper. Best-effort — failures log but don't
    propagate. Mirrors the Phase 52 backlinks hook style.

    Called by ``indexer._link_people_after_index`` after each file
    finishes indexing. Idempotent thanks to the UNIQUE constraint
    on person_mentions.
    """
    try:
        link_file_mentions(conn, file_id)
    except Exception as e:  # noqa: BLE001
        log.warning("people: link_after_index for %s failed: %s", file_id, e)
