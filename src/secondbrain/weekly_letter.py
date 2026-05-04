"""Round 16 (Phase B) — proactive weekly review as a personal letter.

The pre-existing ``synthesis.py`` weekly review is a stats dump (counts,
top entities, lingering tasks) rendered as bullet lists. This module
upgrades that into an actual *letter* — a short, personal, narrative
synthesis written by Claude Sonnet that:

  - References specific events from the week (not generic affirmations).
  - Notices cross-source patterns (journal mood vs sleep, tasks vs
    health, recurring themes across emails / meetings / files).
  - Surfaces open threads — things you mentioned but didn't resolve.
  - Calls back to the prior letter for continuity.
  - Ends with one specific, actionable suggestion (not a long list).

Design notes:

  - Sonnet (not Haiku) — quality matters. ~$0.05 / week is fine.
  - Single LLM call. Multi-step would over-engineer for marginal gain.
  - Heavy prompt engineering: anti-pattern instructions explicit.
  - Stats-only fallback when Anthropic is unavailable so the daemon
    never produces nothing on a Sunday.
  - Per-week idempotency via UNIQUE(week_end). Re-running on the same
    Sunday no-ops; ``--regenerate`` flips it.
  - Prior letter is included for context so successive weeks build on
    each other (the model can say "you mentioned wanting X last week —
    you started it Tuesday").

Privacy:

  - Journal text is the highest-signal input but also the most personal.
    We pass it through ``redact_text`` (masks API keys / SSNs /
    credit cards) but NOT through truncation-style ``_safe_for_prompt``
    truncation — losing personal context would crater the letter
    quality. Sending personal thoughts to Anthropic is the whole
    premise of the system; the user opted in.
  - Email subjects/snippets, meeting transcripts, and tasks all go
    through ``redact_text``.
  - The full letter is stored in the local DB; the signals_json
    snapshot is also stored so we can audit what was sent.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import weakref as _weakref
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from .config import Config

log = logging.getLogger(__name__)

_LETTER_MODEL = "claude-sonnet-4-5"
"""Sonnet 4.5 — the most expensive call in the system, deliberately.
Weekly cadence keeps the cost trivial (~$0.03-0.08 per letter)."""

_LETTER_MAX_TOKENS = 1500
"""Letters are short by design — ~700 words. 1500 tokens is plenty
of headroom for the model to write naturally without truncation."""

_DEDUP_DAYS = 5
"""If a letter exists within this many days for the same week, skip
re-generation. Lets a Sunday daemon restart not re-fire."""

_SCHEMA_INITIALIZED: _weakref.WeakSet = _weakref.WeakSet()


# ============================ schema ===================================


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """One row per generated letter, indexed by week_end (Sunday).

    ``signals_json`` is the structured snapshot of inputs we sent to
    the LLM — useful for debugging "why did the letter say X?" months
    later. ``letter_md`` is the rendered output. Cost tracking is for
    the dashboard's spend breakdown.
    """
    try:
        if conn in _SCHEMA_INITIALIZED:
            return
    except TypeError:
        pass
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS weekly_letters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            week_start TEXT NOT NULL,
            week_end TEXT NOT NULL,
            signals_json TEXT NOT NULL,
            letter_md TEXT NOT NULL,
            model TEXT NOT NULL,
            cost_cents REAL NOT NULL DEFAULT 0,
            generated_at REAL NOT NULL,
            UNIQUE(week_end)
        );
        CREATE INDEX IF NOT EXISTS idx_weekly_letters_generated
            ON weekly_letters(generated_at DESC);
    """)
    conn.commit()
    try:
        _SCHEMA_INITIALIZED.add(conn)
    except TypeError:
        pass


# ============================ data classes =============================


@dataclass
class WeeklyLetter:
    id: int
    week_start: str
    week_end: str
    signals_json: str
    letter_md: str
    model: str
    cost_cents: float
    generated_at: float

    @property
    def signals(self) -> dict:
        try:
            return json.loads(self.signals_json)
        except (json.JSONDecodeError, ValueError):
            return {}


# ============================ signal extraction ========================


@dataclass
class _Signals:
    """Internal staging dict for the LLM prompt. Serialised to
    signals_json. Each section is independently best-effort — if a
    table doesn't exist (legacy DB) we return empty + log."""
    week_start: str
    week_end: str
    counts: dict = field(default_factory=dict)
    tasks: dict = field(default_factory=dict)
    journal: list = field(default_factory=list)
    journal_mood_avg: float | None = None
    health: dict = field(default_factory=dict)
    habits: list = field(default_factory=list)
    goals: list = field(default_factory=list)
    top_entities: list = field(default_factory=list)
    insights: list = field(default_factory=list)
    email_volume: dict = field(default_factory=dict)
    meetings: list = field(default_factory=list)
    drafts_pending: int = 0
    knowledge_gaps: list = field(default_factory=list)
    notable_files: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "week_start": self.week_start,
            "week_end": self.week_end,
            "counts": self.counts,
            "tasks": self.tasks,
            "journal": self.journal,
            "journal_mood_avg": self.journal_mood_avg,
            "health": self.health,
            "habits": self.habits,
            "goals": self.goals,
            "top_entities": self.top_entities,
            "insights": self.insights,
            "email_volume": self.email_volume,
            "meetings": self.meetings,
            "drafts_pending": self.drafts_pending,
            "knowledge_gaps": self.knowledge_gaps,
            "notable_files": self.notable_files,
        }


def _redact(text: str | None) -> str:
    """Redact secret-shaped substrings (API keys, SSNs, credit cards)
    only. We deliberately do NOT truncate or paraphrase — the LLM
    needs full context for a good letter."""
    if not text:
        return ""
    try:
        from .safety import redact_text
        return redact_text(text)
    except ImportError:
        return text


def _signal_counts(conn: sqlite3.Connection, week_cutoff: float) -> dict:
    """How much of each kind landed this week."""
    out: dict[str, int] = {}
    try:
        out["docs_indexed"] = int(conn.execute(
            "SELECT COUNT(*) AS n FROM files WHERE indexed_at >= ?",
            (week_cutoff,),
        ).fetchone()["n"] or 0)
    except sqlite3.OperationalError:
        out["docs_indexed"] = 0
    for key, where in [
        ("emails", "kind = 'email' OR path LIKE 'email://%'"),
        ("meetings", "path LIKE 'transcript://%'"),
        ("urls_ingested", "kind = 'url'"),
        ("voice_notes", "source = 'voice' OR path LIKE 'voice://%'"),
    ]:
        try:
            out[key] = int(conn.execute(
                f"SELECT COUNT(*) AS n FROM files "
                f"WHERE indexed_at >= ? AND ({where})",
                (week_cutoff,),
            ).fetchone()["n"] or 0)
        except sqlite3.OperationalError:
            out[key] = 0
    return out


def _signal_tasks(conn: sqlite3.Connection, week_cutoff: float) -> dict:
    """Tasks completed this week + lingering > 7 days."""
    out: dict[str, Any] = {
        "completed": [],
        "completed_count": 0,
        "added_count": 0,
        "lingering": [],
    }
    try:
        done = conn.execute(
            "SELECT text, completed_at FROM tasks "
            "WHERE status = 'done' AND completed_at >= ? "
            "ORDER BY completed_at DESC LIMIT 30",
            (week_cutoff,),
        ).fetchall()
        out["completed"] = [_redact(r["text"]) for r in done]
        out["completed_count"] = len(done)
        out["added_count"] = int(conn.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE created_at >= ?",
            (week_cutoff,),
        ).fetchone()["n"] or 0)
        lingering_cutoff = time.time() - 7 * 86400
        ling = conn.execute(
            "SELECT text FROM tasks WHERE status = 'open' "
            "AND created_at <= ? "
            "ORDER BY created_at ASC LIMIT 10",
            (lingering_cutoff,),
        ).fetchall()
        out["lingering"] = [_redact(r["text"]) for r in ling]
    except sqlite3.OperationalError:
        pass
    return out


def _signal_journal(
    conn: sqlite3.Connection, end_dt: datetime,
) -> tuple[list[dict], float | None]:
    """Pull last 7 days of journal entries. Returns (entries, mood_avg)."""
    cutoff = (end_dt.date() - timedelta(days=7)).isoformat()
    try:
        rows = conn.execute(
            "SELECT date, mood, text FROM journal_entries "
            "WHERE date >= ? ORDER BY date ASC",
            (cutoff,),
        ).fetchall()
    except sqlite3.OperationalError:
        return [], None
    entries = []
    moods = []
    for r in rows:
        text = _redact(r["text"] or "")
        if r["mood"] is not None:
            moods.append(int(r["mood"]))
        entries.append({
            "date": r["date"],
            "mood": r["mood"],
            "text": text,
        })
    mood_avg = sum(moods) / len(moods) if moods else None
    return entries, mood_avg


def _signal_health(conn: sqlite3.Connection) -> dict:
    """7-day vs prior-7d trend for each tracked metric."""
    out: dict[str, dict] = {}
    try:
        from . import health as health_mod
    except ImportError:
        return out
    for metric in ("sleep_score", "readiness_score", "activity_score",
                   "hrv", "resting_hr"):
        try:
            week = health_mod.recent(conn, metric, days=7)
            prior = health_mod.recent(conn, metric, days=14)
        except Exception:  # noqa: BLE001
            continue
        if not week:
            continue
        prior_only = [p for p in prior if p not in week]
        week_avg = sum(p.value for p in week) / len(week)
        delta_pct = None
        if prior_only:
            prior_avg = sum(p.value for p in prior_only) / len(prior_only)
            if prior_avg > 0:
                delta_pct = (week_avg - prior_avg) / prior_avg * 100.0
        out[metric] = {
            "week_avg": round(week_avg, 1),
            "delta_pct_vs_prior_week": (
                round(delta_pct, 1) if delta_pct is not None else None
            ),
            "n_days": len(week),
        }
    return out


def _signal_habits(conn: sqlite3.Connection) -> list[dict]:
    """Per-habit weekly check-in count + current streak."""
    try:
        from . import personal
    except ImportError:
        return []
    try:
        habits = personal.list_habits(conn)
    except sqlite3.OperationalError:
        return []
    out = []
    for h in habits:
        try:
            status = personal.habit_status(conn, h.id)
        except Exception:  # noqa: BLE001
            continue
        # Count check-ins in the last 7 days (status only gives 30d).
        try:
            cutoff = (date.today() - timedelta(days=7)).isoformat()
            n_week = conn.execute(
                "SELECT COUNT(*) AS n FROM habit_checkins "
                "WHERE habit_id = ? AND date >= ?",
                (h.id, cutoff),
            ).fetchone()["n"] or 0
        except sqlite3.OperationalError:
            n_week = 0
        out.append({
            "name": h.name,
            "cadence": h.cadence,
            "checkins_this_week": int(n_week),
            "current_streak_days": status.current_streak_days,
        })
    return out


def _signal_goals(conn: sqlite3.Connection) -> list[dict]:
    """Active goals + linked task progress."""
    try:
        from . import personal
    except ImportError:
        return []
    try:
        goals = personal.list_goals(conn)
    except (sqlite3.OperationalError, AttributeError):
        return []
    out = []
    for g in goals:
        try:
            # GoalStatus is in personal but the API surface varies by
            # version; be defensive.
            n_done = int(conn.execute(
                "SELECT COUNT(*) AS n FROM tasks "
                "WHERE goal_id = ? AND status = 'done'",
                (g.id,),
            ).fetchone()["n"] or 0)
            n_open = int(conn.execute(
                "SELECT COUNT(*) AS n FROM tasks "
                "WHERE goal_id = ? AND status = 'open'",
                (g.id,),
            ).fetchone()["n"] or 0)
        except sqlite3.OperationalError:
            n_done = n_open = 0
        out.append({
            "name": g.name,
            "tasks_done_total": n_done,
            "tasks_open": n_open,
        })
    return out


def _signal_top_entities(
    conn: sqlite3.Connection, week_cutoff: float,
) -> list[dict]:
    """People / orgs / projects mentioned across the week's docs.

    Round 17 fix (audit-found gap H4) — proper GROUP BY. The earlier
    version selected ``e.text, e.label`` while grouping only by
    ``e.text_lower``; SQLite tolerates this but returns *some*
    arbitrary row's text/label. So if the same person showed up as
    "Sarah" and "sarah" in two different docs, the surfaced casing
    was non-deterministic; if an entity had two labels (PERSON in
    one doc, ORG in another) we surfaced an arbitrary one.

    Fix: aggregate text via MIN() (deterministic) and group by both
    text_lower AND label so PERSON-vs-ORG of the same surface form
    don't collapse into one row with a wrong label.
    """
    try:
        rows = conn.execute(
            "SELECT MIN(e.text) AS text, e.label, "
            "       COUNT(DISTINCT c.file_id) AS n "
            "FROM entities e JOIN chunks c ON c.id = e.chunk_id "
            "JOIN files f ON f.id = c.file_id "
            "WHERE f.indexed_at >= ? "
            "  AND e.label IN "
            "      ('PERSON','ORG','PRODUCT','WORK_OF_ART','EVENT','GPE') "
            "GROUP BY e.text_lower, e.label ORDER BY n DESC LIMIT 12",
            (week_cutoff,),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        {"text": _redact(r["text"]), "label": r["label"], "n_docs": int(r["n"])}
        for r in rows
    ]


def _signal_insights(conn: sqlite3.Connection) -> list[dict]:
    """Active insights (topic spikes / health drift) from synthesis."""
    try:
        from . import synthesis
        ins = synthesis.detect_insights(conn)
    except Exception:  # noqa: BLE001
        return []
    return [
        {"kind": i.kind, "headline": _redact(i.headline),
         "detail": _redact(i.detail)}
        for i in ins[:6]
    ]


def _signal_email_volume(
    conn: sqlite3.Connection, week_cutoff: float,
) -> dict:
    """Phase 82's email_classifications table tracks triage labels.
    Roll up by label so we can say 'you got 14 emails labeled urgent'."""
    out: dict[str, Any] = {"by_label": {}, "total": 0}
    try:
        rows = conn.execute(
            "SELECT ec.label, COUNT(*) AS n "
            "FROM email_classifications ec JOIN files f ON f.id = ec.file_id "
            "WHERE f.indexed_at >= ? "
            "GROUP BY ec.label ORDER BY n DESC",
            (week_cutoff,),
        ).fetchall()
        out["by_label"] = {r["label"]: int(r["n"]) for r in rows}
        out["total"] = sum(out["by_label"].values())
    except sqlite3.OperationalError:
        pass
    return out


def _signal_meetings(
    conn: sqlite3.Connection, week_cutoff: float,
) -> list[dict]:
    """Recent meetings — title + first 200 chars of transcript."""
    try:
        rows = conn.execute(
            "SELECT f.id, f.path, c.text "
            "FROM files f JOIN chunks c ON c.file_id = f.id "
            "WHERE f.indexed_at >= ? AND f.path LIKE 'transcript://%' "
            "  AND c.chunk_index = 0 "
            "ORDER BY f.indexed_at DESC LIMIT 8",
            (week_cutoff,),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    out = []
    seen: set[int] = set()
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        snippet = _redact((r["text"] or "")[:300])
        title = (r["path"] or "").split("/")[-1] or "(untitled)"
        out.append({"title": title, "snippet": snippet})
    return out


def _signal_drafts_pending(conn: sqlite3.Connection) -> int:
    try:
        return int(conn.execute(
            "SELECT COUNT(*) AS n FROM email_drafts "
            "WHERE status = 'pending'",
        ).fetchone()["n"] or 0)
    except sqlite3.OperationalError:
        return 0


def _signal_knowledge_gaps(
    conn: sqlite3.Connection, week_cutoff: float,
) -> list[dict]:
    """Active knowledge gaps the user hasn't followed up on."""
    try:
        rows = conn.execute(
            "SELECT topic, mentions FROM knowledge_gaps "
            "WHERE last_seen >= ? AND status = 'open' "
            "ORDER BY mentions DESC LIMIT 6",
            (week_cutoff,),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        {"topic": _redact(r["topic"]), "mentions": int(r["mentions"])}
        for r in rows
    ]


def _signal_notable_files(
    conn: sqlite3.Connection, week_cutoff: float,
) -> list[dict]:
    """Files indexed this week with a summary, ordered by recency.
    Gives the LLM concrete examples of what was added to the brain."""
    try:
        rows = conn.execute(
            "SELECT f.path, c.text "
            "FROM files f JOIN chunks c ON c.file_id = f.id "
            "WHERE f.indexed_at >= ? AND c.chunk_index = 0 "
            "  AND f.kind != 'email' "
            "ORDER BY f.indexed_at DESC LIMIT 10",
            (week_cutoff,),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    out = []
    seen: set[str] = set()
    for r in rows:
        if r["path"] in seen:
            continue
        seen.add(r["path"])
        title = (r["path"] or "").split("/")[-1] or "(unnamed)"
        snippet = _redact((r["text"] or "")[:200])
        out.append({"title": title, "snippet": snippet})
    return out


def assemble_signals(
    conn: sqlite3.Connection, end_dt: datetime | None = None,
) -> _Signals:
    """Pull every signal into one structured object. Pure data — no
    LLM call here. End is the cutoff (defaults to now); week is the
    7-day window ending at end."""
    end_dt = end_dt or datetime.now()
    week_cutoff = end_dt.timestamp() - 7 * 86400
    start_dt = end_dt - timedelta(days=7)

    journal_entries, mood_avg = _signal_journal(conn, end_dt)
    return _Signals(
        week_start=start_dt.strftime("%Y-%m-%d"),
        week_end=end_dt.strftime("%Y-%m-%d"),
        counts=_signal_counts(conn, week_cutoff),
        tasks=_signal_tasks(conn, week_cutoff),
        journal=journal_entries,
        journal_mood_avg=mood_avg,
        health=_signal_health(conn),
        habits=_signal_habits(conn),
        goals=_signal_goals(conn),
        top_entities=_signal_top_entities(conn, week_cutoff),
        insights=_signal_insights(conn),
        email_volume=_signal_email_volume(conn, week_cutoff),
        meetings=_signal_meetings(conn, week_cutoff),
        drafts_pending=_signal_drafts_pending(conn),
        knowledge_gaps=_signal_knowledge_gaps(conn, week_cutoff),
        notable_files=_signal_notable_files(conn, week_cutoff),
    )


# ============================ LLM call ================================


_SYSTEM_PROMPT = """You are writing a weekly personal letter to the user. Your voice is warm, observant, and honest — like a smart friend who has been paying attention. Not corporate, not therapy-speak, not motivational-poster.

CRITICAL CONSTRAINTS:
- Total length under 700 words.
- Be specific. Cite actual numbers, names (first-name only), and events from the data.
- Notice patterns ACROSS signals (e.g., "your journal mentions 'tired' on the same days your sleep score dropped"). If there are no real patterns, say so plainly — "nothing stood out across signals this week" — instead of inventing one.
- Do NOT compliment without evidence. "Great work this week!" without specifics is hollow.
- Do NOT recommend without specific data. "Try journaling more" is generic; "you skipped journal Thu/Fri — was something off those days?" is specific.
- ONE concrete suggestion at the end, not a list.
- Avoid these words and phrases entirely: "amazing", "journey", "incredible", "powerful", "transformative", "unleash", "elevate", "synergy", "leverage" (as a verb), "deep dive", "level up", emoji.
- Never start a sentence with "I noticed that you" — too clinical. Just say what you noticed.
- If the data is sparse (a quiet week), write a short letter (2-3 paragraphs) acknowledging that. Don't pad.

STRUCTURE:
1. **One opening line** that captures the shape of the week. No heading.
2. **## Looking back** — 2-3 paragraphs synthesizing what happened. Pull from the data. Cite specifics. If meetings or files came up that connect to a goal or theme, surface that.
3. **## Patterns I noticed** — 1-2 paragraphs of cross-signal observations. Or "Nothing stood out across signals this week."
4. **## Open threads** — bullet list (3-5 items) of things mentioned but not resolved. Pull from lingering tasks, unanswered journal questions, knowledge gaps, pending drafts.
5. **## One thing for next week** — a single specific suggestion with the rationale for why this and not something else.

If a prior letter is provided, reference it ONCE if relevant (e.g., "last week you mentioned X — looks like it Y'd"). Don't force it.
"""


_USER_PROMPT_TEMPLATE = """Here are this week's signals (week of {week_start} → {week_end}). Write the letter now.

PRIOR LETTER (last week, for continuity reference only):
---
{prior_letter}
---

THIS WEEK'S SIGNALS (JSON):
```json
{signals_json}
```
"""


def _format_prompt(
    signals: _Signals, prior_letter: str | None,
) -> str:
    return _USER_PROMPT_TEMPLATE.format(
        week_start=signals.week_start,
        week_end=signals.week_end,
        prior_letter=(prior_letter or "(no prior letter)").strip(),
        signals_json=json.dumps(signals.to_dict(), indent=2, default=str),
    )


def generate_letter(
    cfg: Config,
    conn: sqlite3.Connection,
    signals: _Signals,
    prior_letter: str | None = None,
) -> tuple[str, str, float]:
    """Call Sonnet to produce the letter markdown.

    Returns ``(letter_md, model, cost_cents)``. Falls back to
    stats-only formatting when Anthropic is unavailable or the
    budget is exhausted, so the daemon never produces nothing.
    """
    import os

    fallback = _format_stats_only(signals)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.info("weekly_letter: no Anthropic key, using stats-only fallback")
        return fallback, "fallback:stats", 0.0
    try:
        import anthropic
    except ImportError:
        log.info("weekly_letter: anthropic SDK missing, fallback")
        return fallback, "fallback:stats", 0.0
    try:
        from .budget import (
            BudgetExceededError,
            check_budget,
            estimate_cost,
            record_usage,
        )
        check_budget(cfg, "anthropic", feature="weekly_review")
    except BudgetExceededError as e:
        log.warning("weekly_letter: budget exceeded, fallback: %s", e)
        return fallback, "fallback:stats", 0.0
    except ImportError:
        return fallback, "fallback:stats", 0.0

    prompt = _format_prompt(signals, prior_letter)
    client = anthropic.Anthropic()
    try:
        resp = client.messages.create(
            model=_LETTER_MODEL,
            max_tokens=_LETTER_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.APIError as e:
        log.warning("weekly_letter: API error, fallback: %s", e)
        return fallback, "fallback:stats", 0.0

    try:
        record_usage(
            cfg, "anthropic", _LETTER_MODEL,
            input_tokens=getattr(resp.usage, "input_tokens", 0),
            output_tokens=getattr(resp.usage, "output_tokens", 0),
            note="weekly_review",
            feature="weekly_review",
        )
        cost = estimate_cost(
            _LETTER_MODEL,
            input_tokens=getattr(resp.usage, "input_tokens", 0),
            output_tokens=getattr(resp.usage, "output_tokens", 0),
        ).cents
    except Exception:  # noqa: BLE001
        cost = 0.0

    text = "\n".join(
        b.text for b in resp.content if getattr(b, "type", "") == "text"
    ).strip()
    if not text:
        return fallback, "fallback:stats", cost
    # Audit log so the user can see what model + cost / time.
    # Round 17 (audit-found gap C2) — explicitly enumerate the
    # personal-data signals that went outbound. The `summary` is what
    # shows up in /audit; the user grepping "what did I send to
    # Anthropic last Sunday?" sees journal-entry count + flag at a
    # glance, not just a raw `prompt_chars=X` number.
    n_journal = len(signals.journal)
    has_health = bool(signals.health)
    has_meetings = bool(signals.meetings)
    summary_bits = [f"weekly letter for {signals.week_end}"]
    if n_journal:
        summary_bits.append(f"sent {n_journal} journal day(s)")
    if has_health:
        summary_bits.append("incl. health metrics")
    if has_meetings:
        summary_bits.append("incl. meeting snippets")
    try:
        from . import ai_audit
        ai_audit.record_action(
            conn, kind="weekly_review", feature="weekly_review",
            model=_LETTER_MODEL, status="success",
            prompt_chars=len(prompt),
            response_chars=len(text),
            cents=cost,
            summary=" — ".join(summary_bits),
            extra={
                "n_journal_entries": n_journal,
                "has_health_signals": has_health,
                "has_meeting_snippets": has_meetings,
                "n_top_entities": len(signals.top_entities),
            },
        )
    except Exception:  # noqa: BLE001
        pass
    return text, _LETTER_MODEL, cost


def _format_stats_only(signals: _Signals) -> str:
    """Fallback rendering when LLM unavailable. Plain stats, no flourish."""
    lines = [
        f"# Weekly review — {signals.week_start} → {signals.week_end}",
        "",
        "_(Generated without an LLM — Anthropic unavailable. Falls back "
        "to a stats summary.)_",
        "",
        "## Looking back",
        f"- Indexed {signals.counts.get('docs_indexed', 0)} new doc(s)",
    ]
    for k, label in [
        ("emails", "email(s)"),
        ("meetings", "meeting(s)"),
        ("urls_ingested", "URL(s) ingested"),
        ("voice_notes", "voice note(s)"),
    ]:
        n = signals.counts.get(k, 0)
        if n:
            lines.append(f"- {n} {label}")
    if signals.tasks.get("completed_count"):
        lines.append(
            f"- Completed {signals.tasks['completed_count']} task(s); "
            f"added {signals.tasks.get('added_count', 0)}"
        )
    if signals.journal_mood_avg is not None:
        lines.append(
            f"- Journaled {len(signals.journal)} day(s); "
            f"mood avg {signals.journal_mood_avg:.1f}/5"
        )
    if signals.health:
        for metric, h in signals.health.items():
            delta = h.get("delta_pct_vs_prior_week")
            if delta is None:
                lines.append(f"- {metric.replace('_', ' ')}: {h['week_avg']}")
            else:
                arrow = "↑" if delta >= 0 else "↓"
                lines.append(
                    f"- {metric.replace('_', ' ')}: {h['week_avg']} "
                    f"({arrow}{abs(delta):.0f}% vs prior week)"
                )
    if signals.tasks.get("completed"):
        lines.extend(["", "## Done this week"])
        for t in signals.tasks["completed"][:10]:
            lines.append(f"- {t}")
    if signals.tasks.get("lingering"):
        lines.extend(["", "## Open threads (lingering > 7d)"])
        for t in signals.tasks["lingering"]:
            lines.append(f"- [ ] {t}")
    if signals.top_entities:
        lines.extend(["", "## Top topics"])
        for e in signals.top_entities[:8]:
            lines.append(f"- **{e['text']}** ({e['n_docs']} doc(s))")
    if signals.insights:
        lines.extend(["", "## Worth your attention"])
        for ins in signals.insights:
            lines.append(f"- **{ins['headline']}** — {ins['detail']}")
    return "\n".join(lines).rstrip() + "\n"


# ============================ persistence ==============================


def _row_to_letter(row) -> WeeklyLetter:
    return WeeklyLetter(
        id=int(row["id"]),
        week_start=row["week_start"],
        week_end=row["week_end"],
        signals_json=row["signals_json"],
        letter_md=row["letter_md"],
        model=row["model"],
        cost_cents=float(row["cost_cents"] or 0),
        generated_at=float(row["generated_at"]),
    )


def latest_letter(conn: sqlite3.Connection) -> WeeklyLetter | None:
    _ensure_schema(conn)
    row = conn.execute(
        "SELECT * FROM weekly_letters "
        "ORDER BY generated_at DESC LIMIT 1",
    ).fetchone()
    return _row_to_letter(row) if row else None


def list_letters(
    conn: sqlite3.Connection, limit: int = 12,
) -> list[WeeklyLetter]:
    _ensure_schema(conn)
    rows = conn.execute(
        "SELECT * FROM weekly_letters "
        "ORDER BY generated_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [_row_to_letter(r) for r in rows]


def get_letter(
    conn: sqlite3.Connection, week_end: str,
) -> WeeklyLetter | None:
    _ensure_schema(conn)
    row = conn.execute(
        "SELECT * FROM weekly_letters WHERE week_end = ?",
        (week_end,),
    ).fetchone()
    return _row_to_letter(row) if row else None


def has_recent_letter(conn: sqlite3.Connection, *, days: int = 5) -> bool:
    """Did we already generate a letter in the last N days?"""
    _ensure_schema(conn)
    cutoff = time.time() - days * 86400
    row = conn.execute(
        "SELECT 1 FROM weekly_letters WHERE generated_at >= ? LIMIT 1",
        (cutoff,),
    ).fetchone()
    return row is not None


def _save_letter(
    conn: sqlite3.Connection,
    *,
    signals: _Signals,
    letter_md: str,
    model: str,
    cost_cents: float,
    overwrite: bool = False,
) -> WeeklyLetter:
    """Persist (or replace) a letter for the given week_end.

    Round 17 fix (audit-found gap E2) — wrap the DELETE + INSERT in
    a single transaction. If a crash hit between the two on
    overwrite=True, the row was gone with no replacement.
    """
    _ensure_schema(conn)
    payload = (
        signals.week_start, signals.week_end,
        json.dumps(signals.to_dict(), default=str),
        letter_md, model, cost_cents, time.time(),
    )
    # ``with conn`` opens an implicit transaction; commits on success
    # and rolls back on exception. Round-15 used the same pattern in
    # indexer.py for the analogous file+chunks atomic-write fix.
    with conn:
        if overwrite:
            conn.execute(
                "DELETE FROM weekly_letters WHERE week_end = ?",
                (signals.week_end,),
            )
        conn.execute(
            "INSERT INTO weekly_letters"
            "(week_start, week_end, signals_json, letter_md, model, "
            " cost_cents, generated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            payload,
        )
    return latest_letter(conn) or _row_to_letter({
        "id": 0,
        **dict(zip(
            ("week_start", "week_end", "signals_json", "letter_md",
             "model", "cost_cents", "generated_at"), payload, strict=True,
        )),
    })


# ============================ public entry points ======================


def generate_and_save(
    cfg: Config,
    conn: sqlite3.Connection,
    *,
    end_dt: datetime | None = None,
    overwrite: bool = False,
) -> WeeklyLetter:
    """Synchronous: assemble + LLM + save in one call. Use this from
    the CLI / dashboard "Generate now" button.

    If a letter already exists for the same week_end and ``overwrite``
    is False, returns the existing letter (idempotent).
    """
    end_dt = end_dt or datetime.now()
    signals = assemble_signals(conn, end_dt)
    existing = get_letter(conn, signals.week_end)
    if existing is not None and not overwrite:
        log.info("weekly_letter: already exists for %s, returning",
                 signals.week_end)
        return existing
    prior = None
    rows = list_letters(conn, limit=2)
    if rows:
        # If we're regenerating, the latest IS the one we're about to
        # replace; pick the second-most-recent for "prior".
        if existing is not None and len(rows) >= 2:
            prior = rows[1].letter_md
        elif existing is None:
            prior = rows[0].letter_md
    letter_md, model, cost = generate_letter(cfg, conn, signals, prior)
    return _save_letter(
        conn,
        signals=signals,
        letter_md=letter_md,
        model=model,
        cost_cents=cost,
        overwrite=overwrite,
    )


def run_weekly_letter_if_due(
    cfg: Config, conn: sqlite3.Connection,
) -> WeeklyLetter | None:
    """Daemon entry point. Sundays only, once per week.

    Replaces the older ``synthesis.run_weekly_review_if_due`` for the
    daemon path. The old stats-only review is still available via
    ``secondbrain review --stats-only`` if a user wants it.
    """
    if datetime.now().weekday() != 6:  # 6 = Sunday
        return None
    if has_recent_letter(conn, days=_DEDUP_DAYS):
        return None
    try:
        return generate_and_save(cfg, conn)
    except Exception:  # noqa: BLE001
        log.exception("weekly_letter: generate_and_save crashed")
        return None
