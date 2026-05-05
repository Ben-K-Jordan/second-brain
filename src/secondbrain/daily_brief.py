"""Phase 44: daily brief aggregator — your morning at a glance.

The other ``briefing.py`` modules synthesise *one thing* via Claude:

  - ``briefing.py`` → "what entered the brain" (free-form summary)
  - ``event_briefing.py`` → "what to know before this meeting" (per-event)

This module is different. It's a deterministic *aggregator* that pulls
together everything the rest of the system already knows about today
into one Markdown brief. No LLM call required for the assembly itself —
all the data already exists; we just collate.

Sections (any empty section is omitted):

  1. **Today's calendar** — events from now → end of day in your local TZ.
     Pulled live from Google Calendar + ICS via the same helpers
     ``event_briefing.py`` uses, so the brief reflects last-minute
     additions even if the connector hasn't sync'd yet.

  2. **Class assignments due soon** — Canvas assignments in the next 72h.
     Ranks by due date. Reads ``files`` rows by ``canvas://assignment/``
     path prefix; ``mtime`` is the assignment's due_at, so a simple
     range filter works.

  3. **Open action items** — Markdown ``- [ ]`` checkboxes from recent
     transcript ingestion (Granola / generic meeting notes — see
     ``transcripts.py``). Caps the list so a chatty meeting doesn't
     dominate the brief.

  4. **Reading queue** — top N unread items with their auto-summaries
     (Phase 40). Just lifts ``reading_queue_unread``.

  5. **Watchlist highlights** — finished runs in the last 24h that
     produced *new* items (``new_count > 0``). For each, names the
     watchlist + 1-2 sample new paths.

The output renders to Markdown (good for email + dashboard) and to a
Rich-tinted plaintext (CLI). When the daemon hook ships, the same
markdown render goes into an email send and a tray notification.

Cost: $0. Pure aggregation — Claude is only invoked if you opt into
the "polish this" pass via ``--polish`` (uses Haiku 4.5 for ~$0.001).
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from .config import Config

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


def _safe(s: str | None) -> str:
    """Phase 88 — apply sensitive-content redaction before any string
    leaves this module via the SMTP relay or tray notification.

    The brief renders chunk text (action items, queue summaries,
    transcripts) and LLM output (insights, summaries) — both routes
    can carry secrets the user pasted into a notes app once and
    forgot about. Cheap (regex-only) and idempotent.
    """
    if not s:
        return ""
    try:
        from .safety import redact_text
    except ImportError:
        return s
    return redact_text(s)


# ---- Tunables ---------------------------------------------------------

# How far ahead to surface Canvas assignments. 72h gives you the natural
# "this week" horizon without burying tomorrow's pop quiz under next
# Friday's project deadline.
_ASSIGNMENT_LOOKAHEAD_SECONDS = 72 * 3600
# Action items get pulled from transcripts ingested in the last N days.
# Two weeks: long enough to catch "I'll do X next week" promises that
# haven't been ticked, short enough that ancient unchecked items don't
# drown out fresh ones.
_ACTION_ITEM_LOOKBACK_DAYS = 14
# Cap items per section so the brief stays scannable.
_ACTION_ITEM_MAX = 10
_QUEUE_MAX = 5
_WATCHLIST_HIGHLIGHTS_MAX = 5
_WATCHLIST_NEW_PATHS_PER_RUN = 2

# ---- Data shapes ------------------------------------------------------

@dataclass
class Assignment:
    """Canvas assignment surfaced in the brief."""
    title: str            # already includes [course-code] prefix
    due_at: float
    url: str
    path: str             # canvas://assignment/<cid>/<aid> for click-through


@dataclass
class ActionItem:
    """Open todo extracted from a transcript-shaped doc."""
    text: str
    source_path: str      # transcript:// virtual path so user can click back
    source_title: str
    task_id: int = 0      # Phase 47 task id — lets the user run `tasks done X`
    age_days: int = 0     # how long it's been open (created → now)


@dataclass
class QueueItem:
    """Slim view of a reading_queue row."""
    queue_id: int
    url: str
    title: str
    summary: str          # may be empty if the summariser hasn't run yet


@dataclass
class WatchlistHighlight:
    """A watchlist run that produced new items recently."""
    name: str
    new_count: int
    sample_paths: list[str] = field(default_factory=list)
    finished_at: float = 0.0


@dataclass
class HealthSnapshot:
    """At-a-glance numerics from the Oura connector (Phase 56). Each
    metric carries today's value plus the trailing-window average so
    the brief can render `sleep 76 (-7% vs 14d avg)`."""
    metrics: list[HealthMetricLine] = field(default_factory=list)


@dataclass
class HealthMetricLine:
    metric: str           # 'sleep_score' | 'readiness_score' | 'activity_score' etc.
    label: str            # human-readable: 'Sleep' / 'Readiness' / 'Activity'
    latest: float         # last value
    latest_date: str      # 'YYYY-MM-DD'
    average: float        # rolling-window average
    delta_pct: float      # (latest - avg) / avg * 100 — signed


@dataclass
class CompletedTask:
    """A task ticked off in the recent past — fed to "Yesterday's wins"."""
    text: str
    completed_at: float


@dataclass
class HabitLine:
    """One row in the morning brief's habits section."""
    name: str
    streak_days: int
    expected_30d: int
    actual_30d: int


@dataclass
class GoalLine:
    """One row in the morning brief's goals section."""
    name: str
    target_per_week: int | None
    progress_this_week: int
    on_track: bool


@dataclass
class InsightLine:
    """One Phase-75 'I noticed X' surface."""
    headline: str
    detail: str


@dataclass
class EmailTriageLine:
    """Phase 82 — counts of unread email by triage label, last 7d."""
    urgent: int = 0
    response: int = 0
    other: int = 0   # informational + newsletter + automated, rolled up


@dataclass
class GapLine:
    """Phase 68 — top open knowledge gaps surfaced in the brief."""
    gap_id: int
    question: str


@dataclass
class ProjectClusterLine:
    """Phase 73 — auto-detected cluster the brief surfaces."""
    suggested_name: str
    seed_title: str
    n_members: int
    score: float


@dataclass
class BirthdayLine:
    """Phase 65 hookup — a person whose birthday lands within the
    brief's lookahead window."""
    name: str
    days_until: int          # 0 = today, 1 = tomorrow, etc.
    age_turning: int | None  # None when birth year unknown
    is_today: bool


@dataclass
class AnnotationLine:
    """Phase 84 hookup — a PDF annotation extracted recently. We
    surface the highlighted text + the file path so the user can
    jump back to the source PDF."""
    file_path: str
    page: int
    kind: str          # 'highlight' | 'note' | 'underline' | 'strike'
    anchor: str        # the highlighted text (may be empty for pure notes)
    note: str          # the user's note (may be empty for pure highlights)


@dataclass
class DailyBrief:
    """The whole morning brief in one structured object."""
    generated_at: float
    today_events: list  # list[CalendarEvent] — typed loosely to avoid the import at module load
    assignments_due_soon: list[Assignment]
    open_action_items: list[ActionItem]
    queue_top: list[QueueItem]
    watchlist_highlights: list[WatchlistHighlight]
    # Phase-cross polish below — added in the second pass.
    health: HealthSnapshot | None = None       # Phase 56 hookup
    yesterday_done: list[CompletedTask] = field(default_factory=list)
    revisit_suggestions: list[RevisitSuggestion] = field(default_factory=list)
    # Polish v3 (post-Phase 89): personal context + active surfacing.
    habits: list[HabitLine] = field(default_factory=list)        # Phase 79
    goals: list[GoalLine] = field(default_factory=list)          # Phase 79
    insights: list[InsightLine] = field(default_factory=list)    # Phase 75
    email: EmailTriageLine | None = None                         # Phase 82
    knowledge_gaps: list[GapLine] = field(default_factory=list)  # Phase 68
    pending_email_drafts: int = 0                                # Phase 83
    project_clusters: list[ProjectClusterLine] = field(          # Phase 73
        default_factory=list,
    )
    birthdays: list[BirthdayLine] = field(default_factory=list)  # Phase 65
    recent_annotations: list[AnnotationLine] = field(            # Phase 84
        default_factory=list,
    )
    # Lightweight "nudge" flags — surfaced as one-line reminders.
    weekly_review_due: bool = False  # Phase 72
    snapshot_due: bool = False       # Phase 87
    # Round 10 (#9) — fragile-integration breakage warnings.
    # list[HealthStatus]; rendered by health_checks helper.
    stale_health: list = field(default_factory=list)
    # Round 9-B — stale connections worth reaching back out to.
    # Free-form list[StaleConnection]; rendered by connections module.
    stale_connections: list = field(default_factory=list)
    # Round 9-A — upcoming external meetings within the brief window
    # that have brain-grounded prep available. Free-form so we don't
    # need to import meeting_prep types here.
    upcoming_preps: list = field(default_factory=list)
    # Round 20 — EA-shaped sections: open follow-ups (both
    # directions, capped) and morning triage queue summary. Both
    # are populated by ``_followups_section`` / ``_triage_section``
    # (see helpers below) and are stored as opaque list[dict] /
    # dict so this dataclass doesn't pull the modules at import.
    followups_open: list[dict] = field(default_factory=list)
    triage_today: dict = field(default_factory=dict)


@dataclass
class RevisitSuggestion:
    """An older doc to surface on quiet days — picked from the brain's
    indexed corpus weighted by recency × backlink-density."""
    path: str
    title: str
    aged_days: int


# ============================ assembly ================================

def assemble_brief(cfg: Config, conn: sqlite3.Connection) -> DailyBrief:
    """Pull every section's data and return a DailyBrief.

    Each section is best-effort — if a calendar source is unreachable or
    Canvas isn't configured, the affected section is empty rather than
    crashing the whole brief.
    """
    brief = DailyBrief(
        generated_at=time.time(),
        today_events=_today_events(cfg),
        assignments_due_soon=_assignments_due_soon(conn),
        open_action_items=_open_action_items(conn),
        queue_top=_queue_top(conn),
        watchlist_highlights=_watchlist_highlights(conn),
        health=_health_snapshot(conn),
        yesterday_done=_yesterday_done(conn),
        habits=_habits_section(conn),
        goals=_goals_section(conn),
        insights=_insights_section(conn),
        email=_email_section(conn),
        knowledge_gaps=_gaps_section(conn),
        pending_email_drafts=_pending_drafts_count(conn),
        project_clusters=_project_clusters_section(conn),
        birthdays=_birthdays_section(conn),
        recent_annotations=_recent_annotations_section(conn),
        weekly_review_due=_weekly_review_due(conn),
        snapshot_due=_snapshot_due(conn),
        stale_connections=_stale_connections_section(conn),
        upcoming_preps=_upcoming_preps_section(cfg, conn),
        stale_health=_stale_health_section(conn),
        # Round 20 EA additions —
        followups_open=_followups_section(conn),
        triage_today=_triage_section(conn),
    )
    # Revisit suggestions only fire on quiet days — otherwise we'd
    # bury the time-sensitive content.
    if not _has_actionable_content(brief):
        brief.revisit_suggestions = _revisit_suggestions(conn)
    return brief


def _habits_section(conn: sqlite3.Connection) -> list[HabitLine]:
    """Pull active habits + their streak data. Phase 79 hookup —
    surfaces 'you've journaled 23 days straight'-style nudges."""
    try:
        from . import personal as personal_mod
    except ImportError:
        return []
    try:
        habits = personal_mod.list_habits(conn)
    except Exception as e:  # noqa: BLE001
        log.warning("daily brief: habits query failed: %s", e)
        return []
    out: list[HabitLine] = []
    for h in habits[:8]:
        try:
            status = personal_mod.habit_status(conn, h.id)
        except Exception:  # noqa: BLE001
            continue
        out.append(HabitLine(
            name=h.name,
            streak_days=status.current_streak_days,
            expected_30d=status.expected_30d,
            actual_30d=status.checkins_last_30d,
        ))
    return out


def _goals_section(conn: sqlite3.Connection) -> list[GoalLine]:
    """Pull active goals + this-week progress. Phase 79 hookup."""
    try:
        from . import personal as personal_mod
    except ImportError:
        return []
    try:
        goals = personal_mod.list_goals(conn)
    except Exception as e:  # noqa: BLE001
        log.warning("daily brief: goals query failed: %s", e)
        return []
    out: list[GoalLine] = []
    for g in goals[:8]:
        try:
            status = personal_mod.goal_status(conn, g.id)
        except Exception:  # noqa: BLE001
            continue
        out.append(GoalLine(
            name=g.name,
            target_per_week=g.target_per_week,
            progress_this_week=status.progress_this_week,
            on_track=status.on_track,
        ))
    return out


def _insights_section(conn: sqlite3.Connection) -> list[InsightLine]:
    """Pull active insights from Phase 75. The dedup window in
    detect_insights ensures we don't re-surface the same insight
    daily."""
    try:
        from . import synthesis
    except ImportError:
        return []
    try:
        raw = synthesis.detect_insights(conn)
    except Exception as e:  # noqa: BLE001
        log.warning("daily brief: insights query failed: %s", e)
        return []
    return [
        InsightLine(headline=i.headline, detail=i.detail)
        for i in raw[:5]
    ]


def _email_section(conn: sqlite3.Connection) -> EmailTriageLine | None:
    """Phase 82 hookup. Returns None when no triage data exists yet."""
    try:
        from . import email_assist
    except ImportError:
        return None
    try:
        counts = email_assist.label_counts(conn, days=7)
    except Exception as e:  # noqa: BLE001
        log.warning("daily brief: email counts failed: %s", e)
        return None
    if not counts:
        return None
    return EmailTriageLine(
        urgent=int(counts.get("urgent", 0)),
        response=int(counts.get("response", 0)),
        other=int(
            counts.get("informational", 0)
            + counts.get("newsletter", 0)
            + counts.get("automated", 0),
        ),
    )


def _gaps_section(conn: sqlite3.Connection) -> list[GapLine]:
    """Phase 68 hookup — top open knowledge gaps."""
    try:
        from . import study
    except ImportError:
        return []
    try:
        gaps = study.list_gaps(conn, limit=5)
    except Exception as e:  # noqa: BLE001
        log.warning("daily brief: gaps query failed: %s", e)
        return []
    return [GapLine(gap_id=g.id, question=g.question) for g in gaps]


def _project_clusters_section(
    conn: sqlite3.Connection,
) -> list[ProjectClusterLine]:
    """Phase 73 hookup — top auto-detected project clusters from
    the backlinks graph. Capped at 3 to keep the brief tight; the
    detector itself is bounded by min_cluster_size = 3 so we don't
    surface noise."""
    try:
        from . import synthesis
    except ImportError:
        return []
    try:
        clusters = synthesis.detect_project_clusters(
            conn, max_clusters=3, min_size=3,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("daily brief: project clusters failed: %s", e)
        return []
    return [
        ProjectClusterLine(
            suggested_name=c.suggested_name,
            seed_title=c.seed_title,
            n_members=len(c.member_paths),
            score=c.score,
        )
        for c in clusters
    ]


# Lookahead window for birthday surfacing. 7d covers "this week" so
# the user has time to schedule a card / message; the section drops
# off after the day passes (no "5 days late" guilt-trips).
_BIRTHDAY_LOOKAHEAD_DAYS = 7


def _birthdays_section(
    conn: sqlite3.Connection,
) -> list[BirthdayLine]:
    """Phase 65 hookup — find people whose birthday lands in the
    next 7 days. Tolerant about format: accepts ``MM-DD`` or
    ``YYYY-MM-DD``; anything unparseable is silently skipped (the
    field is user-edited, so noise is expected).
    """
    try:
        from . import people as people_mod
    except ImportError:
        return []
    try:
        rows = people_mod.list_people(conn, limit=10_000)
    except Exception as e:  # noqa: BLE001
        log.warning("daily brief: people fetch failed: %s", e)
        return []
    today = datetime.now().date()
    out: list[BirthdayLine] = []
    for p in rows:
        bday = (p.birthday or "").strip()
        if not bday:
            continue
        m, d, year_known = _parse_birthday(bday)
        if m is None or d is None:
            continue
        # Find the next anniversary on or after today.
        try:
            anniv = today.replace(month=m, day=d)
        except ValueError:
            # E.g. Feb 29 in a non-leap year → fall back to Feb 28.
            try:
                anniv = today.replace(month=m, day=28 if m == 2 else d)
            except ValueError:
                continue
        if anniv < today:
            try:
                anniv = anniv.replace(year=today.year + 1)
            except ValueError:
                continue
        delta = (anniv - today).days
        if delta > _BIRTHDAY_LOOKAHEAD_DAYS:
            continue
        age_turning = None
        if year_known:
            try:
                birth_year = int(bday.split("-", 1)[0])
                age_turning = anniv.year - birth_year
            except (ValueError, IndexError):
                pass
        out.append(BirthdayLine(
            name=p.display_name or p.canonical_name or "(unnamed)",
            days_until=delta,
            age_turning=age_turning,
            is_today=(delta == 0),
        ))
    out.sort(key=lambda b: b.days_until)
    return out


def _parse_birthday(raw: str) -> tuple[int | None, int | None, bool]:
    """Return (month, day, year_known). Accepts 'YYYY-MM-DD',
    'MM-DD', 'MM/DD'. Returns (None, None, False) on parse failure.
    """
    s = raw.strip()
    if not s:
        return (None, None, False)
    sep = "-" if "-" in s else ("/" if "/" in s else None)
    if sep is None:
        return (None, None, False)
    parts = s.split(sep)
    try:
        if len(parts) == 3:
            # YYYY-MM-DD
            return (int(parts[1]), int(parts[2]), True)
        if len(parts) == 2:
            # MM-DD
            return (int(parts[0]), int(parts[1]), False)
    except ValueError:
        pass
    return (None, None, False)


# Lookback window for annotations. 36h matches the yesterday-done
# window — same "what did you actually do recently?" framing.
_ANNOTATION_LOOKBACK_SECONDS = 36 * 3600
# Cap so a long highlighting session doesn't dominate the brief.
_ANNOTATIONS_MAX = 8


def _recent_annotations_section(
    conn: sqlite3.Connection,
) -> list[AnnotationLine]:
    """Phase 84 hookup — surface PDF highlights / notes the user
    made in the last 36h. Joins to ``files`` for the path so the
    rendered brief includes a click-back target."""
    cutoff = time.time() - _ANNOTATION_LOOKBACK_SECONDS
    try:
        rows = conn.execute(
            "SELECT a.page, a.kind, a.anchor, a.note, f.path "
            "FROM pdf_annotations a "
            "JOIN files f ON f.id = a.file_id "
            "WHERE a.created_at >= ? "
            "ORDER BY a.created_at DESC LIMIT ?",
            (cutoff, _ANNOTATIONS_MAX),
        ).fetchall()
    except sqlite3.OperationalError:
        # Brain pre-dates Phase 84 schema. Quietly skip.
        return []
    return [
        AnnotationLine(
            file_path=r["path"],
            page=int(r["page"]),
            kind=r["kind"] or "highlight",
            anchor=(r["anchor"] or "").strip(),
            note=(r["note"] or "").strip(),
        )
        for r in rows
    ]


def _stale_connections_section(conn: sqlite3.Connection) -> list:
    """Round 9-B hookup — surface a few stale connections worth
    reaching back out to. Empty list when no candidates clear the
    threshold."""
    try:
        from . import connections
        return connections.find_stale_connections(conn, limit=4)
    except Exception as e:  # noqa: BLE001
        log.warning("daily brief: stale connections failed: %s", e)
        return []


def _upcoming_preps_section(cfg, conn: sqlite3.Connection) -> list:
    """Round 9-A hookup — pull prep for today's upcoming external
    meetings. Cached by meeting_prep so this is cheap."""
    try:
        from . import meeting_prep
        return meeting_prep.upcoming_preps(
            conn, cfg, lookahead_seconds=24 * 3600,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("daily brief: upcoming preps failed: %s", e)
        return []


def _stale_health_section(conn: sqlite3.Connection) -> list:
    """Round 10 (#9) hookup — surface health checks that have been
    failing > 24h. Calendar disconnections, IMAP auth failures,
    missing API keys, etc."""
    try:
        from . import health_checks
        return health_checks.stale_failures(conn)
    except Exception as e:  # noqa: BLE001
        log.warning("daily brief: stale health check failed: %s", e)
        return []


def _followups_section(conn: sqlite3.Connection) -> list[dict]:
    """Round 20 — open follow-ups (both directions, with overdue
    surfaced first). Returned as opaque dicts so DailyBrief stays
    decoupled from the followups module's dataclass shape."""
    try:
        from . import followups, followups_ops
        # Round 21 fix (audit-found gap A9) — pull more per side
        # then sort+truncate to a single hard cap. Earlier 6+6=12
        # → trim to 10 silently dropped 2 every time the user had
        # heavy load. Now 8+8=16 with explicit cap=10 makes the
        # truncation visible (and gives the sort more to work with).
        out_rows = followups_ops.list_visible_open(
            conn, direction="outgoing", limit=8,
        )
        in_rows = followups_ops.list_visible_open(
            conn, direction="incoming", limit=8,
        )
        # Surface overdue first.
        rows = sorted(
            out_rows + in_rows,
            key=lambda f: (
                0 if (f.due_at and f.due_at < time.time()) else 1,
                f.due_at or float("inf"),
                -(f.promised_at or 0),
            ),
        )[:10]
        return followups.serialise_for_brief(rows)
    except Exception as e:  # noqa: BLE001
        log.warning("daily brief: followups section failed: %s", e)
        return []


def _triage_section(conn: sqlite3.Connection) -> dict:
    """Round 20 — count + top-3 senders for the morning triage
    queue. Saves the user a click on quiet days; on busy days hints
    at the magnitude. Format: {"count": int, "top_senders": [str]}."""
    try:
        from . import triage_queue
        queue = triage_queue.build_queue(
            conn, hours=48, max_items=12,
        )
        if not queue:
            return {"count": 0, "top_senders": []}
        senders: list[str] = []
        seen: set[str] = set()
        for it in queue:
            who = it.from_display or it.from_email or "(unknown)"
            if who in seen:
                continue
            seen.add(who)
            senders.append(who)
            if len(senders) >= 3:
                break
        return {"count": len(queue), "top_senders": senders}
    except Exception as e:  # noqa: BLE001
        log.warning("daily brief: triage section failed: %s", e)
        return {"count": 0, "top_senders": []}


def _weekly_review_due(conn: sqlite3.Connection) -> bool:
    """Phase 72 — true on a Sunday when no weekly review has been
    generated in the last 5 days. (5d window so a daemon restart on
    Sunday morning doesn't clear the nudge.) Returns False on any
    other day so the brief stays focused mid-week."""
    try:
        from datetime import datetime as _dt
        if _dt.now().weekday() != 6:  # 6 = Sunday
            return False
        from . import synthesis
        return not synthesis.has_recent_weekly_review(conn, days=5)
    except Exception as e:  # noqa: BLE001
        log.warning("daily brief: weekly-review-due check failed: %s", e)
        return False


def _snapshot_due(conn: sqlite3.Connection) -> bool:
    """Phase 87 — true when no index snapshot has been taken in the
    last 7 days (matching the daemon's ``take_snapshot_if_due`` cadence).
    The brief surfaces this as a soft nudge so the user can run
    ``secondbrain snapshot take`` themselves on machines where the
    daemon doesn't run continuously."""
    try:
        from . import memory as memory_mod
        return memory_mod.needs_snapshot(conn)
    except Exception as e:  # noqa: BLE001
        log.warning("daily brief: snapshot-due check failed: %s", e)
        return False


def _pending_drafts_count(conn: sqlite3.Connection) -> int:
    """Phase 83 — count of email drafts awaiting your review."""
    try:
        from . import email_assist
    except ImportError:
        return 0
    try:
        return len(email_assist.list_unsent_drafts(conn, limit=200))
    except Exception:  # noqa: BLE001
        return 0


def _has_actionable_content(brief: DailyBrief) -> bool:
    """A 'live' brief has at least one section with stuff to do today.

    Extended in the polish pass to include Phase 75 insights (you
    should clock these), Phase 82 urgent emails, Phase 83 pending
    drafts, and Phase 68 knowledge gaps — all of which carry weight
    against the 'quiet day' classification.
    """
    return bool(
        brief.today_events
        or brief.assignments_due_soon
        or brief.open_action_items
        or brief.queue_top
        or brief.insights
        or (brief.email and brief.email.urgent)
        or brief.pending_email_drafts
        or brief.knowledge_gaps
        or brief.project_clusters
        or brief.birthdays,
    )


# ---- section: today's calendar ---------------------------------------

def _seconds_until_end_of_local_day(now: float | None = None) -> int:
    """How many seconds remain in the user's local-time day?"""
    n = now if now is not None else time.time()
    local = datetime.fromtimestamp(n)
    end = local.replace(hour=23, minute=59, second=59, microsecond=0)
    return max(60, int(end.timestamp() - n))


def _today_events(cfg: Config) -> list:
    """Calendar events from now → end of local-time day. Best-effort —
    network/auth problems just yield no events."""
    try:
        from .event_briefing import iter_upcoming_events
    except ImportError:
        return []
    horizon = _seconds_until_end_of_local_day()
    try:
        events = list(iter_upcoming_events(cfg, horizon))
    except Exception as e:  # noqa: BLE001
        log.warning("daily brief: calendar fetch failed: %s", e)
        return []
    events.sort(key=lambda ev: ev.starts_at)
    return events


# ---- section: assignments due soon -----------------------------------

def _assignments_due_soon(conn: sqlite3.Connection) -> list[Assignment]:
    """Canvas assignments whose due-at falls in [now, now + 72h].

    The Canvas connector stores due_at as the file's mtime so existing
    indexes don't need a metadata table to surface time-sensitive items.
    """
    now = time.time()
    cutoff = now + _ASSIGNMENT_LOOKAHEAD_SECONDS
    rows = conn.execute(
        "SELECT id, path, mtime FROM files "
        "WHERE path LIKE 'canvas://assignment/%' "
        "  AND mtime >= ? AND mtime <= ? "
        "ORDER BY mtime ASC",
        (now, cutoff),
    ).fetchall()
    out: list[Assignment] = []
    for r in rows:
        title, url = _read_canvas_title_and_url(conn, r["id"], r["path"])
        out.append(Assignment(
            title=title, due_at=r["mtime"], url=url, path=r["path"],
        ))
    return out


_CANVAS_LINK_RE = re.compile(r"^Link:\s*(\S+)", re.MULTILINE)


def _read_canvas_title_and_url(
    conn: sqlite3.Connection, file_id: int, path: str,
) -> tuple[str, str]:
    """Pull the H1 + 'Link:' line from the assignment's first chunk so
    the brief can click through to Canvas. Falls back to the path if
    parsing fails."""
    row = conn.execute(
        "SELECT text FROM chunks WHERE file_id = ? "
        "ORDER BY chunk_index ASC LIMIT 1",
        (file_id,),
    ).fetchone()
    if row is None:
        return (path, "")
    text = row["text"] or ""
    title = ""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("# "):
            title = s[2:].strip()
            break
    url = ""
    m = _CANVAS_LINK_RE.search(text)
    if m:
        url = m.group(1).strip()
    return (title or path, url)


# ---- section: open action items --------------------------------------

def _open_action_items(conn: sqlite3.Connection) -> list[ActionItem]:
    """Open tasks, sourced from the ``tasks`` table (Phase 47).

    Materialises any new action items from recently-ingested
    transcripts on the fly so the brief always reflects the latest
    meeting — without re-extracting items the user has already ticked
    off (those stay ``done`` in the table).

    Each item carries its task id + age in days so the rendered brief
    lets the user run ``tasks done <id>`` directly, and surfaces the
    "this has been open for 3 weeks" signal that nudges follow-through.

    Round 13 fix (audit-found gap) — wrap the import + list_open call
    in try/except. Every other section in this module degrades to an
    empty list on failure; this one used to crash the entire brief
    if e.g. the tasks schema migration raced.
    """
    try:
        from . import tasks as tasks_mod
    except ImportError as e:
        log.warning("daily brief: tasks module unavailable: %s", e)
        return []

    # Idempotent — INSERT-OR-IGNORE per item, so this is safe to call
    # on every brief render.
    try:
        tasks_mod.materialize_from_transcripts(
            conn, lookback_days=_ACTION_ITEM_LOOKBACK_DAYS,
        )
    except Exception as e:  # noqa: BLE001
        # Materialisation is best-effort — a malformed chunk shouldn't
        # take down the whole brief.
        log.warning("daily brief: task materialisation failed: %s", e)
    try:
        rows = tasks_mod.list_open(conn, limit=_ACTION_ITEM_MAX)
    except Exception as e:  # noqa: BLE001
        log.warning("daily brief: tasks list_open failed: %s", e)
        return []
    now = time.time()
    out: list[ActionItem] = []
    for t in rows:
        age_days = max(0, int((now - t.created_at) // 86400))
        out.append(ActionItem(
            text=t.text,
            source_path=t.source_path,
            source_title=t.source_title,
            task_id=t.id,
            age_days=age_days,
        ))
    return out


# ---- section: health snapshot (Phase 56 cross-link) ------------------

# Metrics surfaced in the brief, in display order. Subset of what
# Oura emits — we don't want to drown the brief in temperature
# deviations and step counts; just the three Oura headline scores.
_BRIEF_HEALTH_METRICS: tuple[tuple[str, str], ...] = (
    ("sleep_score", "Sleep"),
    ("readiness_score", "Readiness"),
    ("activity_score", "Activity"),
)
# Window for the rolling average. 14 days = ~2 weeks, long enough to
# smooth weekend / weekday variation without rolling in seasonal drift.
_HEALTH_AVG_WINDOW_DAYS = 14


def _health_snapshot(conn: sqlite3.Connection) -> HealthSnapshot | None:
    """Pull today's-or-latest Oura values + 14d averages. Returns None
    if there's literally no health data — saves a section header in
    the rendered brief."""
    try:
        from . import health as health_mod
    except ImportError:
        return None
    lines: list[HealthMetricLine] = []
    for metric, label in _BRIEF_HEALTH_METRICS:
        summary = health_mod.summarise(
            conn, metric, days=_HEALTH_AVG_WINDOW_DAYS,
        )
        if summary.n == 0 or summary.latest is None or summary.average is None:
            continue
        delta = (
            (summary.latest.value - summary.average) / summary.average * 100.0
            if summary.average != 0 else 0.0
        )
        lines.append(HealthMetricLine(
            metric=metric,
            label=label,
            latest=summary.latest.value,
            latest_date=summary.latest.date,
            average=summary.average,
            delta_pct=delta,
        ))
    if not lines:
        return None
    return HealthSnapshot(metrics=lines)


# ---- section: yesterday's wins (completed tasks) ---------------------

# Window for "yesterday's wins". 36h covers same-day morning briefs
# (since you completed at-most-yesterday tasks) without mixing in
# stuff finished a week ago.
_YESTERDAY_WINDOW_SECONDS = 36 * 3600
_YESTERDAY_DONE_MAX = 5


def _yesterday_done(conn: sqlite3.Connection) -> list[CompletedTask]:
    """Pull tasks completed in the last 36h. The micro-feedback loop
    of "here's what you got done" matters as much as "here's what's
    next" — and on quiet days, it's nice to see your week's progress
    even without an explicit weekly review."""
    cutoff = time.time() - _YESTERDAY_WINDOW_SECONDS
    rows = conn.execute(
        "SELECT text, completed_at FROM tasks "
        "WHERE status = 'done' AND completed_at IS NOT NULL "
        "  AND completed_at >= ? "
        "ORDER BY completed_at DESC LIMIT ?",
        (cutoff, _YESTERDAY_DONE_MAX),
    ).fetchall()
    return [
        CompletedTask(text=r["text"], completed_at=r["completed_at"])
        for r in rows
    ]


# ---- section: revisit suggestions (quiet-day fallback) ---------------

# Cap on revisit suggestions — three is enough to feel exploratory
# without dominating an empty brief.
_REVISIT_MAX = 3
# Don't surface anything younger than this; the point is to revisit
# stuff you've forgotten, not stuff you indexed yesterday.
_REVISIT_MIN_AGE_DAYS = 30
# Skip docs younger than the brain itself — surfacing your *only* doc
# isn't useful. Pre-flight: count files first.
_REVISIT_MIN_BRAIN_FILES = 50


def _revisit_suggestions(conn: sqlite3.Connection) -> list[RevisitSuggestion]:
    """Pick a few older docs to surface on quiet days. Heuristic: at
    least 30d old, prefers docs with backlinks (signals semantic
    density), randomised so morning briefs don't keep showing the
    same five docs.

    If the brain is too small (< 50 files) this returns empty — there's
    no point recommending stuff when the user has only indexed the
    setup notes. Quiet day stays quiet.
    """
    n_files = conn.execute(
        "SELECT COUNT(*) AS n FROM files",
    ).fetchone()["n"]
    if (n_files or 0) < _REVISIT_MIN_BRAIN_FILES:
        return []
    cutoff = time.time() - _REVISIT_MIN_AGE_DAYS * 86400
    rows = conn.execute(
        "SELECT f.id, f.path, f.indexed_at, "
        "  COALESCE(("
        "    SELECT COUNT(*) FROM backlinks b WHERE b.src_file_id = f.id"
        "  ), 0) AS link_density "
        "FROM files f "
        "WHERE f.indexed_at <= ? "
        "ORDER BY link_density DESC, RANDOM() "
        "LIMIT ?",
        (cutoff, _REVISIT_MAX),
    ).fetchall()
    now = time.time()
    out: list[RevisitSuggestion] = []
    for r in rows:
        title = _read_doc_title(conn, r["id"], r["path"])
        aged = max(0, int((now - (r["indexed_at"] or now)) // 86400))
        out.append(RevisitSuggestion(
            path=r["path"], title=title, aged_days=aged,
        ))
    return out


def _read_doc_title(
    conn: sqlite3.Connection, file_id: int, path: str,
) -> str:
    """Best-effort doc title from the first chunk's H1, falling back to
    the path. Cheap — single row lookup."""
    row = conn.execute(
        "SELECT text FROM chunks WHERE file_id = ? "
        "ORDER BY chunk_index ASC LIMIT 1",
        (file_id,),
    ).fetchone()
    if row is None:
        return path
    for line in (row["text"] or "").splitlines():
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip() or path
    return path


# ---- section: reading queue ------------------------------------------

def _queue_top(conn: sqlite3.Connection) -> list[QueueItem]:
    from .db import reading_queue_unread

    rows = reading_queue_unread(conn, limit=_QUEUE_MAX)
    out: list[QueueItem] = []
    for r in rows:
        out.append(QueueItem(
            queue_id=r["id"],
            url=r["url"] or "",
            title=r["title"] or "",
            summary=(r["summary"] or "").strip(),
        ))
    return out


# ---- section: watchlist highlights -----------------------------------

def _watchlist_highlights(conn: sqlite3.Connection) -> list[WatchlistHighlight]:
    """Watchlist runs that finished in the last 24h with new items."""
    cutoff = time.time() - 24 * 3600
    rows = conn.execute(
        "SELECT wr.watchlist_id, wr.finished_at, wr.new_count, "
        "       wr.new_paths_json, w.name "
        "FROM watchlist_runs wr "
        "JOIN watchlists w ON w.id = wr.watchlist_id "
        "WHERE wr.finished_at IS NOT NULL "
        "  AND wr.error IS NULL "
        "  AND wr.finished_at >= ? "
        "  AND wr.new_count > 0 "
        "ORDER BY wr.finished_at DESC LIMIT ?",
        (cutoff, _WATCHLIST_HIGHLIGHTS_MAX),
    ).fetchall()
    out: list[WatchlistHighlight] = []
    for r in rows:
        sample = _parse_new_paths(r["new_paths_json"])
        out.append(WatchlistHighlight(
            name=r["name"] or f"watchlist {r['watchlist_id']}",
            new_count=int(r["new_count"] or 0),
            sample_paths=sample[:_WATCHLIST_NEW_PATHS_PER_RUN],
            finished_at=r["finished_at"] or 0.0,
        ))
    return out


def _parse_new_paths(raw: str | None) -> list[str]:
    """Watchlist runs persist new_paths_json as a JSON list of strings.
    Tolerate malformed/missing — returns an empty list."""
    if not raw:
        return []
    import json
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return [str(x) for x in data if x]


# ============================ rendering ===============================

def format_markdown(brief: DailyBrief, *, header_date: str | None = None) -> str:
    """Render the brief as Markdown — used by email send + dashboard.

    Empty sections are omitted entirely so the brief stays tight on
    sleepy days. ``header_date`` overrides the auto-generated date in
    the H1, useful for tests."""
    date_str = header_date or _format_date(brief.generated_at)
    lines: list[str] = [f"# Daily brief — {date_str}", ""]

    sections = list(_iter_section_blocks(brief))
    if not sections:
        lines.append("Nothing on the calendar, no due assignments, "
                     "no open action items. Quiet day. ☕")
        return "\n".join(lines).rstrip() + "\n"

    # If the only sections present are passive ones (health / done /
    # revisit), prepend a one-line "quiet day" banner so the user
    # immediately knows there's no action expected of them.
    if not _has_actionable_content(brief):
        lines.append("_Quiet day — no events, deadlines, or open items._")
        lines.append("")

    for block in sections:
        lines.append(block.rstrip())
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _iter_section_blocks(brief: DailyBrief) -> Iterator[str]:
    """Yield the markdown for each non-empty section, in display order.

    Order is intentional: time-sensitive content (calendar, due
    assignments) front-loads, then proactive insights the user
    should clock immediately, then action items they can knock off,
    then physical context (health), then slower-burn surfaces
    (email, queue, watchlists, gaps), then personal-tracking
    surfaces (habits, goals, yesterday's wins), then quiet-day
    fallbacks.
    """
    if brief.stale_health:
        # Round 10 (#9) — health failures front-loaded so the user
        # notices before reading the rest of the brief (which may
        # itself be incomplete due to the broken integration).
        yield _render_stale_health(brief.stale_health)
    if brief.insights:
        yield _render_insights(brief.insights)
    if brief.birthdays:
        # Birthdays are time-sensitive — front-load alongside the
        # calendar so the user clocks "today's the day" before
        # reading anything else.
        yield _render_birthdays(brief.birthdays)
    if brief.today_events:
        yield _render_events(brief.today_events)
    if brief.upcoming_preps:
        # Round 9-A — meeting prep block right after the calendar.
        block = _render_upcoming_preps(brief.upcoming_preps)
        if block:
            yield block
    if brief.assignments_due_soon:
        yield _render_assignments(brief.assignments_due_soon)
    if brief.open_action_items:
        yield _render_action_items(brief.open_action_items)
    # Round 20 — open follow-ups (both directions) right after
    # action items so the user sees their full "open threads"
    # surface in one place.
    if brief.followups_open:
        yield _render_followups(brief.followups_open)
    if brief.health is not None and brief.health.metrics:
        yield _render_health(brief.health)
    if brief.email is not None and (
        brief.email.urgent or brief.email.response or brief.email.other
    ):
        yield _render_email(brief.email, brief.pending_email_drafts)
    # Round 20 — morning triage queue summary right after the
    # email section so the user sees "10 emails for triage today"
    # alongside the urgent-vs-response counts.
    if brief.triage_today and brief.triage_today.get("count"):
        yield _render_triage(brief.triage_today)
    if brief.queue_top:
        yield _render_queue(brief.queue_top)
    if brief.watchlist_highlights:
        yield _render_watchlist(brief.watchlist_highlights)
    if brief.project_clusters:
        yield _render_project_clusters(brief.project_clusters)
    if brief.knowledge_gaps:
        yield _render_gaps(brief.knowledge_gaps)
    if brief.habits:
        yield _render_habits(brief.habits)
    if brief.goals:
        yield _render_goals(brief.goals)
    if brief.yesterday_done:
        yield _render_yesterday_done(brief.yesterday_done)
    if brief.recent_annotations:
        yield _render_annotations(brief.recent_annotations)
    if brief.stale_connections:
        # Round 9-B — quiet-day-ish surface; no urgency, just a nudge.
        try:
            from . import connections as _conn_mod
            block = _conn_mod.render_stale_block(brief.stale_connections)
            if block:
                yield block
        except Exception as e:  # noqa: BLE001
            log.warning("daily brief: stale render failed: %s", e)
    nudge_block = _render_nudges(brief)
    if nudge_block:
        yield nudge_block
    if brief.revisit_suggestions:
        yield _render_revisit(brief.revisit_suggestions)


def _render_events(events: list) -> str:
    """Render today's calendar block. Each event is a one-liner."""
    out = ["## Today on the calendar", ""]
    for ev in events:
        when = time.strftime("%H:%M", time.localtime(ev.starts_at))
        bits = [f"- **{when}** {_safe(ev.title)}"]
        if ev.location:
            bits.append(f"_{_safe(ev.location)}_")
        if ev.attendees:
            atts = ", ".join(_safe(a) for a in ev.attendees[:3])
            extra = "" if len(ev.attendees) <= 3 else f" +{len(ev.attendees) - 3}"
            bits.append(f"with {atts}{extra}")
        out.append(" — ".join(bits))
    return "\n".join(out)


def _render_assignments(items: list[Assignment]) -> str:
    out = ["## Class — due in the next 72h", ""]
    for a in items:
        when = time.strftime("%a %H:%M", time.localtime(a.due_at))
        link = f" — <{a.url}>" if a.url else ""
        out.append(f"- **{when}** {_safe(a.title)}{link}")
    return "\n".join(out)


def _render_action_items(items: list[ActionItem]) -> str:
    """Render open tasks with their id (so user can `tasks done <id>`)
    and age in days when meaningful (≥ 1). Text + source title both
    go through ``_safe`` — transcript-extracted action items are the
    most-common path for stray secrets to land in the brief."""
    out = ["## Open action items", ""]
    for it in items:
        prefix = f"`#{it.task_id}`" if it.task_id else ""
        age = ""
        if it.age_days >= 1:
            age = f" _({it.age_days}d)_"
        src = _safe(it.source_title)
        suffix = f"  _(from: {src})_" if src else ""
        bits = ["- [ ]"]
        if prefix:
            bits.append(prefix)
        bits.append(_safe(it.text))
        line = " ".join(bits) + age + suffix
        out.append(line)
    out.append("")
    out.append("_Run `secondbrain tasks done <id>` to close._")
    return "\n".join(out)


def _render_followups(items: list[dict]) -> str:
    """Round 20 — open follow-ups in both directions, with overdue
    flagged inline. Each item is the dict produced by
    ``followups.serialise_for_brief``."""
    if not items:
        return ""
    out_outgoing = [r for r in items if r["direction"] == "outgoing"]
    out_incoming = [r for r in items if r["direction"] == "incoming"]
    out = ["## Open follow-ups", ""]
    if out_outgoing:
        out.append(f"**You owe ({len(out_outgoing)})**")
        for r in out_outgoing:
            person = r.get("person") or "(unknown)"
            age = r.get("age_days")
            age_str = (
                f" · _{int(age)}d_" if age is not None and age >= 1
                else ""
            )
            due_str = ""
            if r.get("due_at") and r["due_at"] < time.time():
                due_str = " · **overdue**"
            elif r.get("due_at"):
                from datetime import date as _date
                due_iso = _date.fromtimestamp(r["due_at"]).isoformat()
                due_str = f" · due {due_iso}"
            out.append(
                f"- {_safe(r['topic'])} → {_safe(person)}"
                f"{age_str}{due_str}"
            )
    if out_incoming:
        if out_outgoing:
            out.append("")
        out.append(f"**Owed to you ({len(out_incoming)})**")
        for r in out_incoming:
            person = r.get("person") or "(unknown)"
            age = r.get("age_days")
            age_str = (
                f" · _{int(age)}d_" if age is not None and age >= 1
                else ""
            )
            out.append(
                f"- {_safe(r['topic'])} ← {_safe(person)}{age_str}"
            )
    out.append("")
    out.append("_See [/followups](/followups)._")
    return "\n".join(out)


def _render_triage(triage: dict) -> str:
    """Round 20 — morning email triage queue summary. ``triage`` is
    a dict like {count: 7, top_senders: [...]}."""
    n = triage.get("count") or 0
    if not n:
        return ""
    senders = triage.get("top_senders") or []
    bits = [f"## Morning triage — {n} email(s) need a decision"]
    if senders:
        sender_list = ", ".join(_safe(s) for s in senders)
        bits.append(f"_Top: {sender_list}_")
    bits.append("")
    bits.append("_See [/triage](/triage) to walk through._")
    return "\n".join(bits)


def _render_health(snap: HealthSnapshot) -> str:
    """Render the Oura at-a-glance block. Each line: latest + delta-vs-avg.

    A negative delta past the threshold gets a `↓` emoji; positive
    `↑`; in-band stays plain. Visual nudge on out-of-norm days
    without forcing the user to read percentages."""
    out = ["## Health (Oura)", ""]
    for m in snap.metrics:
        arrow = ""
        if abs(m.delta_pct) >= 5:
            arrow = " ↓" if m.delta_pct < 0 else " ↑"
        latest_str = (
            f"{int(m.latest)}" if abs(m.latest - int(m.latest)) < 1e-6
            else f"{m.latest:.1f}"
        )
        avg_str = (
            f"{int(m.average)}" if abs(m.average - int(m.average)) < 1e-6
            else f"{m.average:.1f}"
        )
        out.append(
            f"- **{m.label}**: {latest_str} "
            f"_(avg {avg_str}, {m.delta_pct:+.0f}%{arrow})_  "
            f"[{m.latest_date}]"
        )
    return "\n".join(out)


def _render_yesterday_done(items: list[CompletedTask]) -> str:
    out = ["## Recently done", ""]
    for t in items:
        when = time.strftime("%a %H:%M", time.localtime(t.completed_at))
        out.append(f"- ✓ {_safe(t.text)}  _({when})_")
    return "\n".join(out)


def _render_revisit(items: list[RevisitSuggestion]) -> str:
    """Quiet-day fallback: 'remember this?' suggestions from the index."""
    out = ["## Worth revisiting", "", "_Quiet day — picked from your archive:_", ""]
    for r in items:
        out.append(f"- **{_safe(r.title)}** _({r.aged_days}d ago)_  ")
        out.append(f"  `{_safe(r.path)}`")
    return "\n".join(out)


def _render_queue(items: list[QueueItem]) -> str:
    out = [f"## Reading queue (top {len(items)})", ""]
    for q in items:
        title = _safe(q.title or q.url)
        out.append(f"- [{title}]({q.url})")
        if q.summary:
            # Quote the summary so multi-line summaries don't break list
            # nesting in renderers that are strict about indent.
            for line in _safe(q.summary).splitlines():
                out.append(f"  > {line}")
    return "\n".join(out)


def _render_watchlist(items: list[WatchlistHighlight]) -> str:
    out = ["## Watchlists — new in the last 24h", ""]
    for h in items:
        out.append(f"- **{_safe(h.name)}** — {h.new_count} new")
        for p in h.sample_paths:
            out.append(f"  - {_safe(p)}")
    return "\n".join(out)


def _render_insights(items: list[InsightLine]) -> str:
    """Phase 75 — proactive 'I noticed X' surfacing. Front of brief."""
    out = ["## Worth noticing", ""]
    for ins in items:
        out.append(f"- **{_safe(ins.headline)}**")
        if ins.detail:
            out.append(f"  {_safe(ins.detail)}")
    return "\n".join(out)


def _render_email(email: EmailTriageLine, n_drafts: int) -> str:
    """Phase 82 + 83 — email triage counts + pending drafts to review."""
    out = ["## Email", ""]
    bits = []
    if email.urgent:
        bits.append(f"**{email.urgent} urgent**")
    if email.response:
        bits.append(f"{email.response} need response")
    if email.other:
        bits.append(f"{email.other} other")
    if bits:
        out.append(f"- {' · '.join(bits)} _(last 7 days)_")
    if n_drafts > 0:
        out.append(
            f"- ✉️ **{n_drafts} draft(s) awaiting review** "
            f"_(`secondbrain drafts list`)_",
        )
    return "\n".join(out)


def _render_stale_health(items: list) -> str:
    """Round 10 (#9) — top-of-brief warning when third-party
    integrations have been failing for >24h. Each row tells the user
    what's broken and how long it's been."""
    if not items:
        return ""
    out = ["## ⚠ Integrations need attention", ""]
    for s in items:
        days = s.days_since_ok or 0
        when_str = (
            f"{days}d" if days > 0 else "since last check"
        )
        out.append(
            f"- **{_safe(s.name)}** broken for {when_str}: "
            f"_{_safe(s.error)}_",
        )
    out.append("")
    out.append("_Run `secondbrain doctor` for details._")
    return "\n".join(out)


def _render_upcoming_preps(items: list) -> str:
    """Round 9-A — compact meeting-prep block for the morning brief.

    One line per upcoming meeting with a count of attendees + the
    most-stale relationship (so the user sees 'haven't seen Sarah in
    4 months' as a glance-able cue). Full prep lives at /prep."""
    if not items:
        return ""
    out = ["## Upcoming meetings — prep ready", ""]
    for p in items[:5]:
        bits = [p.when_str]
        if p.attendees:
            bits.append(f"{len(p.attendees)} ext")
            # Surface the most-stale attendee as the headline.
            stalest = max(p.attendees, key=lambda a: a.days_since_seen)
            if stalest.days_since_seen >= 30:
                months = stalest.days_since_seen // 30
                if months >= 1:
                    bits.append(f"{stalest.name} {months}mo")
        out.append(
            f"- **{_safe(p.title)}** _({' · '.join(bits)})_  ",
        )
    out.append("")
    out.append("_Full prep at `/prep`._")
    return "\n".join(out)


def _render_nudges(brief: DailyBrief) -> str:
    """Phase 72/87 — soft reminders the user can act on. Empty
    string means 'nothing to nudge today'; the iter_section_blocks
    caller skips empty yields."""
    bits: list[str] = []
    if brief.weekly_review_due:
        bits.append(
            "- Weekly review is overdue — run "
            "`secondbrain review` (or wait for the daemon).",
        )
    if brief.snapshot_due:
        bits.append(
            "- No snapshot in the last 7 days — "
            "`secondbrain snapshot take` to checkpoint.",
        )
    if not bits:
        return ""
    return "\n".join(["## Nudges", "", *bits])


def _render_annotations(items: list[AnnotationLine]) -> str:
    """Phase 84 — PDF highlights / notes from the last 36h. Quote
    each anchor so the user can re-read what they cared about
    yesterday without opening the file. Falls back to ``note`` text
    when the highlight has no anchor (pure note annotation)."""
    out = ["## Highlights from yesterday", ""]
    for a in items:
        body = _safe(a.anchor) or _safe(a.note) or "(empty)"
        if len(body) > 240:
            body = body[:240].rstrip() + "…"
        out.append(
            f"- _p.{a.page}_ — {body}  `{_safe(a.file_path)}`",
        )
        if a.note and a.anchor:
            out.append(f"  > note: {_safe(a.note)}")
    return "\n".join(out)


def _render_birthdays(items: list[BirthdayLine]) -> str:
    """Phase 65 — heads-up on birthdays in the next week. Today's
    birthday gets the cake emoji + bold; future ones get a date
    annotation."""
    out = ["## Birthdays this week", ""]
    for b in items:
        name = _safe(b.name)
        age_bit = (
            f" (turning {b.age_turning})" if b.age_turning else ""
        )
        if b.is_today:
            out.append(f"- 🎂 **TODAY** — {name}{age_bit}")
        elif b.days_until == 1:
            out.append(f"- tomorrow — {name}{age_bit}")
        else:
            out.append(
                f"- in {b.days_until}d — {name}{age_bit}",
            )
    return "\n".join(out)


def _render_project_clusters(items: list[ProjectClusterLine]) -> str:
    """Phase 73 — auto-detected project clusters from the backlinks
    graph. We surface these as candidate working sets the user might
    want to formalise (give a real name, pin to dashboard, etc.)."""
    out = ["## Possible projects forming", "",
           "_Recently-clustered docs that look like a working set:_",
           ""]
    for c in items:
        name = _safe(c.suggested_name or c.seed_title)
        seed = _safe(c.seed_title)
        out.append(
            f"- **{name}** — {c.n_members} docs around _{seed}_",
        )
    return "\n".join(out)


def _render_gaps(items: list[GapLine]) -> str:
    """Phase 68 — questions ask_brain couldn't answer well."""
    out = ["## Knowledge gaps", "",
           "_Questions you asked that returned weak results — study targets:_",
           ""]
    for g in items:
        out.append(f"- _#{g.gap_id}_ {_safe(g.question)[:120]}")
    return "\n".join(out)


def _render_habits(items: list[HabitLine]) -> str:
    """Phase 79 — habit streaks + 30d adherence."""
    out = ["## Habits", ""]
    for h in items:
        # Streak emoji at 7+ days, fire at 30+, mountain at 100+.
        marker = ""
        if h.streak_days >= 100:
            marker = "🏔 "
        elif h.streak_days >= 30:
            marker = "🔥 "
        elif h.streak_days >= 7:
            marker = "✨ "
        adh = (
            f" _({h.actual_30d}/{h.expected_30d} this month)_"
            if h.expected_30d else ""
        )
        out.append(
            f"- {marker}**{h.name}** — {h.streak_days}d streak{adh}",
        )
    return "\n".join(out)


def _render_goals(items: list[GoalLine]) -> str:
    """Phase 79 — goal progress this week."""
    out = ["## Goals (this week)", ""]
    for g in items:
        if g.target_per_week:
            track = "✓" if g.on_track else "·"
            out.append(
                f"- [{track}] **{g.name}** — "
                f"{g.progress_this_week}/{g.target_per_week}",
            )
        else:
            out.append(
                f"- **{g.name}** — {g.progress_this_week} this week",
            )
    return "\n".join(out)


def _format_date(ts: float) -> str:
    return time.strftime("%A, %b %-d", time.localtime(ts)) if _supports_dash_d() \
        else time.strftime("%A, %b %d", time.localtime(ts))


def _supports_dash_d() -> bool:
    """Windows strftime doesn't support %-d (no zero pad). Cheap probe."""
    try:
        time.strftime("%-d", time.localtime(0))
        return True
    except (ValueError, TypeError):
        return False


# ============================ entrypoints =============================

def generate_brief_markdown(cfg: Config, conn: sqlite3.Connection) -> str:
    """One-shot helper: assemble + render. The CLI + daemon both call this."""
    return format_markdown(assemble_brief(cfg, conn))


# ============================ scheduler ===============================

# How long after a successful send before we'd send again. 12h means
# you can rerun on the same day if you flip enabled off + back on, but
# the auto-fire path won't re-send within the same day.
_BRIEF_RESEND_COOLDOWN_SECONDS = 12 * 3600


def _ensure_brief_runs_table(conn: sqlite3.Connection) -> None:
    """One-row-per-send log so the daemon can answer 'when did we last
    send the brief?'. Cheap to migrate in lazily."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_brief_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sent_at REAL NOT NULL,
            success INTEGER NOT NULL,
            error TEXT,
            recipients TEXT
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_brief_runs_sent_at "
        "ON daily_brief_runs(sent_at DESC)"
    )
    conn.commit()


def last_brief_sent_at(conn: sqlite3.Connection) -> float | None:
    """Most-recent successful send timestamp; None if never sent."""
    _ensure_brief_runs_table(conn)
    row = conn.execute(
        "SELECT sent_at FROM daily_brief_runs WHERE success = 1 "
        "ORDER BY sent_at DESC LIMIT 1",
    ).fetchone()
    return row["sent_at"] if row else None


def _record_brief_run(
    conn: sqlite3.Connection, *, success: bool, error: str | None,
    recipients: str | None,
) -> None:
    _ensure_brief_runs_table(conn)
    conn.execute(
        "INSERT INTO daily_brief_runs(sent_at, success, error, recipients) "
        "VALUES (?, ?, ?, ?)",
        (time.time(), 1 if success else 0, error, recipients),
    )
    conn.commit()


def send_brief(cfg: Config, conn: sqlite3.Connection) -> tuple[bool, str]:
    """Build today's brief and email it via the digest SMTP config.

    Reuses the digest's SMTP plumbing — same host/user/password.
    Returns ``(success, message)``. Failures log + persist; success
    persists too so the daemon can suppress re-sends.
    """
    import os
    import smtplib
    from email.message import EmailMessage

    if not getattr(cfg, "daily_brief_enabled", False):
        return False, "daily_brief_enabled is false in config"
    to = (cfg.digest_to or "").strip()
    if not to:
        return False, "digest_to is empty (the brief reuses digest SMTP config)"
    password = os.environ.get("SECONDBRAIN_SMTP_PASSWORD", "")
    if not password:
        return False, "SECONDBRAIN_SMTP_PASSWORD env var not set"

    md = generate_brief_markdown(cfg, conn)
    msg = EmailMessage()
    msg["From"] = cfg.digest_smtp_from or cfg.digest_smtp_user
    msg["To"] = to
    msg["Subject"] = "second-brain daily brief"
    msg.set_content(md)
    # HTML rendering: a minimal Markdown-to-HTML pass so the email
    # client renders headings + bullets nicely. We keep this naive
    # rather than dragging in a full Markdown lib for one path —
    # users can read either part.
    msg.add_alternative(_minimal_md_to_html(md), subtype="html")

    try:
        with smtplib.SMTP(
            cfg.digest_smtp_host, cfg.digest_smtp_port, timeout=30,
        ) as s:
            s.ehlo()
            s.starttls()
            s.ehlo()
            s.login(cfg.digest_smtp_user, password)
            s.send_message(msg)
    except (smtplib.SMTPException, OSError) as e:
        err = f"{type(e).__name__}: {e}"
        log.warning("daily brief send failed: %s", err)
        _record_brief_run(conn, success=False, error=err, recipients=to)
        return False, err
    log.info("daily brief sent to %s", to)
    _record_brief_run(conn, success=True, error=None, recipients=to)
    return True, f"sent to {to}"


def run_brief_if_due(cfg: Config, conn: sqlite3.Connection) -> bool:
    """Daemon hook — send the brief once per local-time day, after
    the configured ``daily_brief_send_time``.

    Mirrors ``digest.run_digest_if_due``: passes a 12h cooldown after
    successful sends so a daemon restart doesn't double-fire."""
    if not getattr(cfg, "daily_brief_enabled", False):
        return False
    if not (cfg.digest_to or "").strip():
        return False
    raw = getattr(cfg, "daily_brief_send_time", "07:00")
    try:
        hh, mm = raw.split(":")
        target_h, target_m = int(hh), int(mm)
    except (ValueError, AttributeError):
        log.warning(
            "daily_brief_send_time %r isn't HH:MM; skipping", raw,
        )
        return False
    now = datetime.now()
    target = now.replace(
        hour=target_h, minute=target_m, second=0, microsecond=0,
    )
    if now < target:
        return False
    last = last_brief_sent_at(conn)
    if last is not None and (time.time() - last) < _BRIEF_RESEND_COOLDOWN_SECONDS:
        return False
    success, info = send_brief(cfg, conn)
    log.info("daily brief auto-fire: success=%s msg=%s", success, info)
    return True


_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$")
_BULLET_RE = re.compile(r"^(\s*)[-*]\s+(.*)$")
_BLOCKQUOTE_RE = re.compile(r"^(\s*)>\s*(.*)$")


def _minimal_md_to_html(md: str) -> str:
    """Tiny Markdown subset → HTML for the email body. Handles H1-3,
    bullets (incl. checkboxes), blockquotes, blank lines. Anything
    fancier (links, code, tables) renders as plain text in HTML —
    the text/plain alternative covers high-fidelity reading."""
    out = ["<html><body style='font-family:system-ui;max-width:680px;margin:auto;'>"]
    in_list = False
    for raw_line in md.splitlines():
        line = raw_line.rstrip()
        if not line:
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append("<br/>")
            continue
        h = _HEADING_RE.match(line)
        if h:
            if in_list:
                out.append("</ul>")
                in_list = False
            level = len(h.group(1))
            out.append(f"<h{level}>{_html_escape(h.group(2))}</h{level}>")
            continue
        b = _BULLET_RE.match(line)
        if b:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{_html_escape(b.group(2))}</li>")
            continue
        q = _BLOCKQUOTE_RE.match(line)
        if q:
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(
                "<blockquote style='color:#555;border-left:3px solid #ccc;"
                f"padding-left:8px;margin:2px 0;'>{_html_escape(q.group(2))}"
                "</blockquote>",
            )
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        out.append(f"<p>{_html_escape(line)}</p>")
    if in_list:
        out.append("</ul>")
    out.append("</body></html>")
    return "\n".join(out)


def _html_escape(s: str) -> str:
    """Just enough to keep `<` / `>` / `&` from breaking the email."""
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )
