"""Phase 67 + 68: spaced repetition + knowledge gap detection.

You're already capturing every class lecture via Plaud → transcript
ingest → docs with `[course]` titles. Right now that's a search
corpus. This module turns it into a study system.

Two halves:

  1. **Card generation** — for each `[course]` doc, generate a small
     batch of conceptual Q&A flashcards via Claude Haiku. Cards persist
     to ``study_cards`` so re-runs don't regenerate. Idempotent via
     UNIQUE(file_id, question).

  2. **Spaced repetition (SM-2)** — each card carries ease + interval +
     next_due_at. The CLI quiz session asks due cards, the user grades
     0-5, and the schedule updates per the SM-2 algorithm. Weak cards
     bubble up to next session; well-known cards retire to weekly /
     monthly review.

Phase 68 piggybacks: when ``ask_brain`` returns weak results (top
retrieval score below threshold OR fewer than N hits), the question
lands in ``knowledge_gaps``. The weekly review (Phase 72) surfaces
top gaps as study targets.

Cost guard: card generation is bounded per-doc (default 5 cards) and
per-tick (default 3 docs). One full course's worth of lectures
materialises over a week of background runs, not all at once.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)


# ---- Tunables ---------------------------------------------------------

# Cards generated per doc when materialising. Lectures are dense; 5
# concepts at a time keeps generation cheap (~$0.01 per doc on Haiku
# 4.5) and the user's session friction low.
_CARDS_PER_DOC = 5
# Per-tick cap on docs to materialise. With 5 cards × 3 docs the
# daemon spends ~$0.03 per pass.
_DEFAULT_DOCS_PER_TICK = 3
# Initial SM-2 ease factor.
_INITIAL_EASE = 2.5
# Minimum ease — anything below this gets clamped (SM-2 standard).
_MIN_EASE = 1.3
# Map a 0-5 grade to (correct?, ease delta, interval rule). 0-2 = wrong,
# 3-5 = correct with varying confidence.
_GRADE_RULES = {
    0: (False, -0.20, "reset"),
    1: (False, -0.15, "reset"),
    2: (False, -0.10, "reset"),
    3: (True, -0.05, "advance"),
    4: (True, 0.0, "advance"),
    5: (True, 0.10, "advance"),
}

# Knowledge-gap threshold. ask_brain results with the top hit's
# retrieval score below this (lower is better in our cosine-distance
# regime) get treated as "weak" and logged.
_WEAK_RESULT_MIN_SCORE = 0.35
# OR if fewer than this many hits returned, also log.
_WEAK_RESULT_MIN_HITS = 2


# ---- Data shapes -----------------------------------------------------

@dataclass
class StudyCard:
    id: int
    file_id: int
    course_code: str
    concept: str
    question: str
    answer: str
    chunk_id: int | None
    ease: float
    interval_days: float
    next_due_at: float
    last_reviewed_at: float | None
    review_count: int
    correct_count: int

    @property
    def accuracy(self) -> float:
        return self.correct_count / self.review_count if self.review_count else 0.0


@dataclass
class KnowledgeGap:
    id: int
    question: str
    asked_at: float
    top_score: float | None
    n_results: int
    resolved_at: float | None
    note: str | None


# ============================ card generation =========================

# Detect [course-code] prefix in titles like "[BME 410] Lecture 3".
# We capture the alphanumeric prefix (collapsed: "BME410") for grouping.
_COURSE_PREFIX_RE = re.compile(r"^\s*\[([A-Z]{2,5}[\s\-]?\d{3,4})\]")


def extract_course_code(title: str) -> str:
    """Pull the course code out of a `[course] title` doc title.

    Empty string when the title doesn't match the prefix shape (so
    the caller can decide whether to skip)."""
    if not title:
        return ""
    m = _COURSE_PREFIX_RE.match(title)
    if not m:
        return ""
    # Normalize 'BME 410' / 'BME-410' → 'BME410' for stable grouping.
    return re.sub(r"[\s\-]+", "", m.group(1)).upper()


def needs_cards(
    conn: sqlite3.Connection, file_id: int, target: int = _CARDS_PER_DOC,
) -> bool:
    """True iff this doc has fewer than ``target`` cards. Used by the
    daemon hook to decide whether to generate."""
    n = conn.execute(
        "SELECT COUNT(*) AS n FROM study_cards WHERE file_id = ?",
        (file_id,),
    ).fetchone()["n"]
    return (n or 0) < target


def docs_pending_cards(
    conn: sqlite3.Connection, *, limit: int = _DEFAULT_DOCS_PER_TICK,
) -> list[tuple[int, str, str]]:
    """Return [(file_id, title, course_code), ...] for course docs
    that don't have cards yet. Newest first so just-recorded lectures
    materialise quickly.

    Uses the chunks' first-line H1 to find titles since `files.title`
    isn't a column (the title lives in the chunk body)."""
    # LEFT JOIN over NOT IN: at scale (lots of cards) the subquery
    # was rebuilding the full set per outer row. Anti-join via NULL
    # check is index-friendly.
    rows = conn.execute(
        "SELECT f.id, c.text "
        "FROM files f "
        "JOIN chunks c ON c.file_id = f.id "
        "LEFT JOIN study_cards sc ON sc.file_id = f.id "
        "WHERE c.chunk_index = 0 "
        "  AND (f.path LIKE 'transcript://%' OR f.path LIKE 'imap://%') "
        "  AND sc.file_id IS NULL "
        "ORDER BY f.indexed_at DESC LIMIT ?",
        (limit * 4,),  # over-fetch then filter to course-coded ones
    ).fetchall()
    out: list[tuple[int, str, str]] = []
    for r in rows:
        title = _first_line_h1(r["text"] or "")
        course = extract_course_code(title)
        if course:
            out.append((int(r["id"]), title, course))
            if len(out) >= limit:
                break
    return out


def _first_line_h1(text: str) -> str:
    """Pull the H1 out of a Markdown chunk body. Returns empty string
    when the first line isn't a heading."""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip()
        if s:
            return ""  # first non-empty line wasn't an H1
    return ""


def materialize_cards(
    conn: sqlite3.Connection,
    file_id: int,
    *,
    cfg=None,
    cards_per_doc: int = _CARDS_PER_DOC,
    generator=None,
) -> int:
    """Generate cards for one doc. Returns count newly inserted.

    ``generator`` is a callable ``(title, body, n) -> list[dict]``
    that returns ``[{concept, question, answer, chunk_id?}, ...]``.
    Default: uses Claude Haiku via ``_default_generator``. Tests
    substitute a deterministic stub.
    """
    if not needs_cards(conn, file_id, cards_per_doc):
        return 0
    body_rows = conn.execute(
        "SELECT id, chunk_index, text FROM chunks WHERE file_id = ? "
        "ORDER BY chunk_index ASC",
        (file_id,),
    ).fetchall()
    if not body_rows:
        return 0
    title = _first_line_h1(body_rows[0]["text"] or "")
    course = extract_course_code(title)
    body = "\n\n".join(r["text"] or "" for r in body_rows)
    # Heuristic chunk_id mapping: pick the first chunk containing the
    # answer text (rough but better than null). Falls back to chunk 0.
    chunk_id_by_text: dict[str, int] = {
        r["text"]: int(r["id"]) for r in body_rows if r["text"]
    }
    if generator is None:
        generator = _default_generator
        if cfg is None:
            log.warning("study: cfg required for default generator")
            return 0
    try:
        cards = generator(title, body, cards_per_doc, cfg)
    except Exception as e:  # noqa: BLE001
        log.warning("study: generator crashed for file_id=%s: %s",
                    file_id, e)
        return 0
    if not cards:
        return 0
    n = time.time()
    inserted = 0
    for c in cards:
        question = (c.get("question") or "").strip()
        answer = (c.get("answer") or "").strip()
        if not question or not answer:
            continue
        concept = (c.get("concept") or "").strip()[:60] or "general"
        # Best-effort chunk attribution.
        chunk_id = c.get("chunk_id")
        if chunk_id is None:
            for chunk_text, cid in chunk_id_by_text.items():
                if answer.lower()[:40] in chunk_text.lower():
                    chunk_id = cid
                    break
        cur = conn.execute(
            "INSERT OR IGNORE INTO study_cards"
            "(file_id, course_code, concept, question, answer, "
            " chunk_id, ease, interval_days, next_due_at, "
            " review_count, correct_count, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, 0, 0, ?)",
            (file_id, course, concept, question, answer,
             chunk_id, _INITIAL_EASE, n, n),
        )
        if cur.rowcount > 0:
            inserted += 1
    conn.commit()
    return inserted


_GENERATOR_PROMPT = """\
You're generating short, exam-style flashcards from a class lecture.

Title: {title}

Lecture body (excerpted):
{body}

Generate exactly {n} cards as a JSON array. Each card object has:
  - concept: a 2-5 word topic tag (lowercase, e.g. "tonotopic map")
  - question: a single conceptual question a final exam might ask
  - answer: a 1-3 sentence answer grounded in the lecture content

Rules:
  - Conceptual, not factoid (avoid "what year did X happen").
  - Cover diverse parts of the lecture — don't make all 5 about the same paragraph.
  - Question should stand alone (no "according to the lecture" or "based on the body").
  - Answer must be derivable from the lecture body above.

Respond with ONLY the JSON array. No prose, no Markdown fences."""


def _default_generator(
    title: str, body: str, n: int, cfg,
) -> list[dict]:
    """Real generator backed by Claude Haiku 4.5. Bounded budget
    via the per-feature 'study' bucket (Phase 63)."""
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return []
    try:
        import anthropic
    except ImportError:
        return []
    from .budget import (
        BudgetExceededError,
        check_budget,
        record_usage,
    )

    try:
        check_budget(cfg, "anthropic", feature="study")
    except BudgetExceededError as e:
        log.warning("study: budget exceeded: %s", e)
        return []
    # Round 14 (audit-found gap H1) — redact PII / secret-shaped
    # substrings before the lecture body leaves for Anthropic. Every
    # other LLM call site in the app applies _safe_for_prompt; this
    # was the missed one. Cap to 12k chars (covers 99%-ile lectures
    # and skips closing chitchat). Title goes through the same path
    # in case the user names a lecture after a person / project.
    from .email_assist import _safe_for_prompt
    body_clip = _safe_for_prompt(body, max_chars=12000)
    title_clean = _safe_for_prompt(title, max_chars=200)
    prompt = _GENERATOR_PROMPT.format(title=title_clean, body=body_clip, n=n)
    client = anthropic.Anthropic()
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as e:
        log.warning("study: API error: %s", e)
        return []
    record_usage(
        cfg, "anthropic", "claude-haiku-4-5",
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        feature="study",
        note=f"study/cards/{title[:40]}",
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    # Strip optional ```json fences.
    if text.startswith("```"):
        text = re.sub(r"^```\w*\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        cards = json.loads(text)
    except json.JSONDecodeError as e:
        log.warning("study: failed to parse generator output: %s", e)
        return []
    if not isinstance(cards, list):
        return []
    return cards


# ============================ scheduling (SM-2) =======================

def grade_card(
    conn: sqlite3.Connection, card_id: int, grade: int,
) -> StudyCard | None:
    """Apply a 0-5 grade, update SM-2 state, return the new card.
    Returns None if card_id is unknown."""
    row = conn.execute(
        "SELECT * FROM study_cards WHERE id = ?", (card_id,),
    ).fetchone()
    if row is None:
        return None
    grade = max(0, min(5, int(grade)))
    correct, ease_delta, interval_rule = _GRADE_RULES[grade]
    new_ease = max(_MIN_EASE, row["ease"] + ease_delta)
    if interval_rule == "reset":
        new_interval = 0.5  # see again in ~12h
    else:
        # SM-2: interval(n) = max(1, interval(n-1) * ease).
        prev = row["interval_days"] or 0
        new_interval = (
            1.0 if prev <= 0
            else 6.0 if prev < 1.5
            else prev * new_ease
        )
    n = time.time()
    conn.execute(
        "UPDATE study_cards SET "
        "  ease = ?, interval_days = ?, "
        "  next_due_at = ?, last_reviewed_at = ?, "
        "  review_count = review_count + 1, "
        "  correct_count = correct_count + ? "
        "WHERE id = ?",
        (new_ease, new_interval, n + new_interval * 86400, n,
         1 if correct else 0, card_id),
    )
    conn.commit()
    return get_card(conn, card_id)


def get_card(
    conn: sqlite3.Connection, card_id: int,
) -> StudyCard | None:
    row = conn.execute(
        "SELECT * FROM study_cards WHERE id = ?", (card_id,),
    ).fetchone()
    return _row_to_card(row) if row else None


def due_cards(
    conn: sqlite3.Connection, *,
    course_code: str | None = None, limit: int = 20,
) -> list[StudyCard]:
    """Cards whose next_due_at has passed. Filtered by course_code
    when given (case-sensitive on the canonical form)."""
    n = time.time()
    if course_code:
        rows = conn.execute(
            "SELECT * FROM study_cards "
            "WHERE next_due_at <= ? AND course_code = ? "
            "ORDER BY next_due_at ASC LIMIT ?",
            (n, course_code, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM study_cards WHERE next_due_at <= ? "
            "ORDER BY next_due_at ASC LIMIT ?",
            (n, limit),
        ).fetchall()
    return [_row_to_card(r) for r in rows]


def cards_for_course(
    conn: sqlite3.Connection, course_code: str,
) -> list[StudyCard]:
    rows = conn.execute(
        "SELECT * FROM study_cards WHERE course_code = ? "
        "ORDER BY concept ASC, id ASC",
        (course_code,),
    ).fetchall()
    return [_row_to_card(r) for r in rows]


def weak_concepts(
    conn: sqlite3.Connection, *,
    course_code: str | None = None, limit: int = 10,
) -> list[tuple[str, float, int]]:
    """Top 'weak' concepts by accuracy. Returns [(concept, accuracy,
    review_count), ...]. Filters out concepts with fewer than 3
    reviews — too few to be meaningful."""
    if course_code:
        rows = conn.execute(
            "SELECT concept, "
            "  CAST(SUM(correct_count) AS REAL) / NULLIF(SUM(review_count), 0) AS acc, "
            "  SUM(review_count) AS n "
            "FROM study_cards WHERE course_code = ? "
            "GROUP BY concept HAVING n >= 3 "
            "ORDER BY acc ASC LIMIT ?",
            (course_code, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT concept, "
            "  CAST(SUM(correct_count) AS REAL) / NULLIF(SUM(review_count), 0) AS acc, "
            "  SUM(review_count) AS n "
            "FROM study_cards "
            "GROUP BY concept HAVING n >= 3 "
            "ORDER BY acc ASC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        (r["concept"], float(r["acc"] or 0.0), int(r["n"]))
        for r in rows
    ]


def _row_to_card(row: sqlite3.Row) -> StudyCard:
    return StudyCard(
        id=int(row["id"]),
        file_id=int(row["file_id"]),
        course_code=row["course_code"],
        concept=row["concept"],
        question=row["question"],
        answer=row["answer"],
        chunk_id=row["chunk_id"],
        ease=float(row["ease"]),
        interval_days=float(row["interval_days"]),
        next_due_at=float(row["next_due_at"]),
        last_reviewed_at=row["last_reviewed_at"],
        review_count=int(row["review_count"]),
        correct_count=int(row["correct_count"]),
    )


# ============================ knowledge gaps (68) =====================

def is_weak_result(
    n_results: int, top_score: float | None,
) -> bool:
    """Decision rule for 'this question got weak retrieval'."""
    if n_results < _WEAK_RESULT_MIN_HITS:
        return True
    if top_score is None:
        return True
    # Lower distance = better. Threshold: above this, the brain didn't
    # have a strong match. Different score scales (FTS / vec) → use
    # the worst-case interpretation: a top score >= threshold is weak.
    return top_score >= _WEAK_RESULT_MIN_SCORE


def log_gap(
    conn: sqlite3.Connection,
    question: str,
    *,
    n_results: int,
    top_score: float | None,
) -> int | None:
    """Persist a knowledge gap. Returns the new id, or None when the
    question is empty / not actually weak.

    Called by ``ask_brain`` after retrieval; safe to call regardless
    of whether the result was weak — this function gates internally."""
    q = (question or "").strip()
    if not q:
        return None
    if not is_weak_result(n_results, top_score):
        return None
    cur = conn.execute(
        "INSERT INTO knowledge_gaps"
        "(question, asked_at, top_score, n_results) "
        "VALUES (?, ?, ?, ?) RETURNING id",
        (q, time.time(), top_score, n_results),
    )
    rid = int(cur.fetchone()["id"])
    conn.commit()
    return rid


def list_gaps(
    conn: sqlite3.Connection, *,
    include_resolved: bool = False, limit: int = 20,
) -> list[KnowledgeGap]:
    if include_resolved:
        rows = conn.execute(
            "SELECT * FROM knowledge_gaps "
            "ORDER BY asked_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM knowledge_gaps WHERE resolved_at IS NULL "
            "ORDER BY asked_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_gap(r) for r in rows]


def resolve_gap(
    conn: sqlite3.Connection, gap_id: int, note: str | None = None,
) -> bool:
    cur = conn.execute(
        "UPDATE knowledge_gaps SET resolved_at = ?, note = ? "
        "WHERE id = ? AND resolved_at IS NULL",
        (time.time(), note, gap_id),
    )
    conn.commit()
    return cur.rowcount > 0


def _row_to_gap(row: sqlite3.Row) -> KnowledgeGap:
    return KnowledgeGap(
        id=int(row["id"]),
        question=row["question"],
        asked_at=row["asked_at"],
        top_score=row["top_score"],
        n_results=int(row["n_results"]),
        resolved_at=row["resolved_at"],
        note=row["note"],
    )


# ============================ daemon hook =============================

def materialize_due_cards(
    conn: sqlite3.Connection, cfg,
    *, docs_per_tick: int = _DEFAULT_DOCS_PER_TICK,
) -> int:
    """Daemon entrypoint — generate cards for a few course docs that
    don't have them yet. Returns count of cards created.

    Idempotent + bounded: at most ``docs_per_tick`` docs per call,
    each producing up to ``_CARDS_PER_DOC`` cards. The Scheduler
    drives this on a slow cadence (every 30 min by default)."""
    pending = docs_pending_cards(conn, limit=docs_per_tick)
    if not pending:
        return 0
    total = 0
    for file_id, _title, _course in pending:
        total += materialize_cards(conn, file_id, cfg=cfg)
    if total:
        log.info("study: materialised %d new card(s) across %d doc(s)",
                 total, len(pending))
    return total
