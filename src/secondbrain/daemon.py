"""Headless and tray-icon runners that keep the watcher alive in the background.

`run_daemon` is for terminal/Task-Scheduler use; `run_tray` adds a system tray
icon with status and quit. Both consume the same `watched_folders` from config
so the user just edits config.toml once.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from .config import Config
from .connectors.oura import run_oura_sync_if_due
from .daily_brief import run_brief_if_due
from .db import checkpoint_wal, connect, init_schema
from .digest import run_digest_if_due
from .embedder import make_embedder
from .entities import make_entity_extractor
from .event_briefing import run_briefings_if_due
from .image_embedder import make_image_embedder
from .imager import make_ocr_engine
from .indexer import IndexResult, index_folder
from .reading_queue import run_summariser_if_due
from .reranker import make_reranker
from .scheduler import (
    CooldownSchedule,
    DailyAtSchedule,
    IntervalSchedule,
    Job,
    Scheduler,
    trim_old_runs,
)
from .transcriber import make_transcriber
from .watcher import Watcher
from .watchlist import run_due_watchlists

log = logging.getLogger(__name__)


# How often the daemon should checkpoint the WAL (and how often the tray's
# bootstrap loop should as well). 10 minutes is comfortable - a busy reader
# can hold the snapshot for that long without the WAL being a problem.
_WAL_CHECKPOINT_INTERVAL_SEC = 600

# How often the daemon should poll for due watchlists. Watchlist schedules
# are in minutes, so 60s polling is enough to keep them within ~1 minute
# of their requested cadence without busy-looping.
_WATCHLIST_POLL_INTERVAL_SEC = 60


def _make_log_handlers(log_path: Path) -> list[logging.Handler]:
    """Rotate the daemon log so a long-running install doesn't accumulate
    hundreds of MB of indexing chatter. 10MB x 3 backups = ~30MB ceiling.
    """
    rotating = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    return [rotating]


def _setup_state(cfg: Config):
    embedder = make_embedder(cfg)
    conn = connect(cfg.db_path)
    init_schema(conn, embedder.dim, embedder.name)

    transcriber = None
    if cfg.transcribe_enabled:
        try:
            transcriber = make_transcriber(cfg)
        except ImportError as e:
            log.warning("transcription disabled: %s", e)

    ocr_engine = None
    if cfg.ocr_enabled:
        try:
            ocr_engine = make_ocr_engine(cfg)
        except ImportError as e:
            log.warning("OCR disabled: %s", e)

    entity_extractor = None
    if cfg.entities_enabled:
        try:
            entity_extractor = make_entity_extractor(cfg)
        except (ImportError, RuntimeError) as e:
            log.warning("entity extraction disabled: %s", e)

    image_embedder = None
    if cfg.image_embed_enabled:
        try:
            image_embedder = make_image_embedder(cfg)
        except Exception as e:
            log.warning("multimodal image embedder disabled: %s", e)

    return conn, embedder, transcriber, ocr_engine, entity_extractor, image_embedder


class _Stats:
    """Thread-safe rolling stats for the running daemon, surfaced in the tray menu."""

    def __init__(self):
        self._lock = threading.Lock()
        self.indexed = 0
        self.errors = 0
        self.last_path: Path | None = None
        self.last_status: str = ""
        self.started_at = time.time()

    def record(self, r: IndexResult) -> None:
        with self._lock:
            self.last_path = r.path
            self.last_status = r.status
            if r.status == "indexed":
                self.indexed += 1
            elif r.status == "error":
                self.errors += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "indexed": self.indexed,
                "errors": self.errors,
                "last_path": str(self.last_path) if self.last_path else None,
                "last_status": self.last_status,
                "uptime": time.time() - self.started_at,
            }


def _resolve_folders(cfg: Config) -> list[Path]:
    """Combine watched_folders + photo_capture_folder (Phase 71) into
    one watch list. The photo folder gets the same treatment as any
    other watched dir — OCR + multimodal embed pipeline already runs
    on images via the existing indexer."""
    raw_paths: list[str] = list(cfg.watched_folders)
    if getattr(cfg, "photo_capture_folder", ""):
        raw_paths.append(cfg.photo_capture_folder)
    folders = [Path(p).expanduser() for p in raw_paths]
    # Dedupe while preserving order (resolve absolute path) so a user
    # who lists the same folder twice doesn't get double-indexing.
    seen: set[Path] = set()
    out: list[Path] = []
    for f in folders:
        try:
            resolved = f.resolve()
        except OSError:
            resolved = f
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            out.append(resolved)
    return out


def _bootstrap(cfg: Config, conn, embedder, transcriber, ocr_engine, entity_extractor, image_embedder, folders, stats: _Stats):
    """Index each folder once on startup so changes since last run are caught."""
    for folder in folders:
        log.info("bootstrap indexing %s", folder)
        index_folder(
            conn, embedder, cfg, folder,
            progress=stats.record,
            transcriber=transcriber,
            ocr_engine=ocr_engine,
            entity_extractor=entity_extractor,
            image_embedder=image_embedder,
        )


def run_daemon(cfg: Config, log_path: Path | None = None) -> None:
    """Headless runner. Blocks until SIGINT/SIGTERM."""
    if log_path is None:
        log_path = cfg.data_dir / "daemon.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[*_make_log_handlers(log_path), logging.StreamHandler()],
    )

    folders = _resolve_folders(cfg)
    if not folders:
        log.error(
            "no watched_folders configured. Edit %s and add paths under "
            "watched_folders, then restart.",
            cfg.config_path,
        )
        return

    log.info("starting daemon, watching: %s", [str(f) for f in folders])
    conn, embedder, transcriber, ocr_engine, entity_extractor, image_embedder = _setup_state(cfg)
    stats = _Stats()

    _bootstrap(
        cfg, conn, embedder, transcriber, ocr_engine, entity_extractor,
        image_embedder, folders, stats,
    )
    log.info("bootstrap done (indexed=%d errors=%d)", stats.indexed, stats.errors)

    watcher = Watcher(
        cfg, conn, embedder,
        on_event=stats.record,
        transcriber=transcriber, ocr_engine=ocr_engine,
        entity_extractor=entity_extractor,
        image_embedder=image_embedder,
    )
    watcher.start(folders)
    # Build a reranker once for watchlist runs so we don't re-instantiate
    # per-poll. Cheap; just loads model metadata.
    try:
        reranker = make_reranker(cfg)
    except Exception as e:  # noqa: BLE001
        log.warning("reranker init failed; watchlists will run without rerank: %s", e)
        reranker = None

    sched = _build_daemon_scheduler(cfg, conn, embedder, reranker)
    log.info("watcher running. Ctrl-C to stop.")
    log.info("scheduler: %d jobs registered: %s",
             len(sched.names()), ", ".join(sched.names()))
    try:
        while True:
            time.sleep(1)
            sched.tick(
                cfg=cfg, conn=conn, embedder=embedder, reranker=reranker,
            )
    except KeyboardInterrupt:
        log.info("stopping...")
    finally:
        watcher.stop()
        # One final checkpoint on shutdown so the next start finds a clean WAL.
        try:
            checkpoint_wal(conn)
        except Exception:  # noqa: BLE001
            pass
        conn.close()


def _build_daemon_scheduler(
    cfg: Config, conn, embedder, reranker,
) -> Scheduler:
    """Wire every periodic job into one scheduler. Each job's name is
    visible in `secondbrain status`; its schedule encapsulates the
    'is it due?' decision; its callable does the actual work and
    returns ``int`` (count) / ``bool`` (did-something) / ``None`` for
    silent success.

    Schedules:

      - ``IntervalSchedule`` for poll-cadence jobs (watchlists, event
        briefings, queue summariser) that decide internally what's due.
      - ``DailyAtSchedule`` for "send at 8am" jobs (digest, daily brief)
        that fire at most once per cooldown window.
      - ``CooldownSchedule`` for "roughly every N hours" jobs (Oura
        sync, WAL checkpoint, runs trim).
    """
    sched = Scheduler(conn)

    # WAL checkpoint — keeps the WAL bounded so a long-running daemon
    # with an active reader (the dashboard) doesn't let it grow.
    sched.register(Job(
        name="wal_checkpoint",
        schedule=IntervalSchedule(seconds=_WAL_CHECKPOINT_INTERVAL_SEC),
        fn=lambda conn: checkpoint_wal(conn),
    ))

    # Runs-table garbage collection — keep 14 days of history.
    sched.register(Job(
        name="trim_scheduler_runs",
        schedule=CooldownSchedule(
            seconds=3600, cooldown_hours=24,
        ),
        fn=lambda conn: trim_old_runs(conn, keep_days=14),
    ))

    # Watchlists — each due watchlist may take several seconds
    # (Claude + tool use). Serial within the scheduler tick.
    sched.register(Job(
        name="watchlists",
        schedule=IntervalSchedule(
            seconds=_WATCHLIST_POLL_INTERVAL_SEC,
        ),
        fn=lambda cfg, conn, embedder, reranker: run_due_watchlists(
            cfg, conn, embedder, reranker,
        ),
    ))

    # Pre-event briefings — checks calendars for soon-starting events.
    sched.register(Job(
        name="event_briefings",
        schedule=IntervalSchedule(
            seconds=_WATCHLIST_POLL_INTERVAL_SEC,
        ),
        fn=lambda cfg, conn, embedder, reranker: run_briefings_if_due(
            cfg, conn, embedder, reranker,
        ),
    ))

    # Reading-queue summariser — cheap relative to watchlist runs.
    sched.register(Job(
        name="read_queue_summariser",
        schedule=IntervalSchedule(
            seconds=_WATCHLIST_POLL_INTERVAL_SEC,
        ),
        fn=lambda cfg, conn, embedder, reranker: run_summariser_if_due(
            cfg, conn, embedder, reranker,
        ),
    ))

    # Daily digest — once per local-time day after digest_send_time.
    sched.register(Job(
        name="daily_digest",
        schedule=DailyAtSchedule(
            local_time=cfg.digest_send_time, cooldown_hours=12,
        ),
        fn=lambda cfg, conn: run_digest_if_due(cfg, conn),
    ))

    # Daily brief — once per local-time day after daily_brief_send_time.
    # Same shape as digest but separate cadence config.
    sched.register(Job(
        name="daily_brief",
        schedule=DailyAtSchedule(
            local_time=getattr(cfg, "daily_brief_send_time", "07:00"),
            cooldown_hours=12,
        ),
        fn=lambda cfg, conn: run_brief_if_due(cfg, conn),
    ))

    # Oura sync — roughly daily; cooldown gates re-sync within window.
    sched.register(Job(
        name="oura_sync",
        schedule=CooldownSchedule(
            seconds=_WATCHLIST_POLL_INTERVAL_SEC, cooldown_hours=12,
        ),
        fn=lambda cfg, conn, embedder: run_oura_sync_if_due(
            cfg, conn, embedder,
        ),
    ))

    # Study card materialiser (Phase 67) — generates flashcards for
    # `[course]` docs that don't have any yet. Slow cadence (30 min)
    # + per-tick cap so we trickle through the backlog instead of
    # spending the budget all at once.
    from .study import materialize_due_cards
    sched.register(Job(
        name="study_card_materialiser",
        schedule=IntervalSchedule(seconds=30 * 60),
        fn=lambda cfg, conn: materialize_due_cards(conn, cfg),
    ))

    # Phase 74: auto-summary materialiser. Picks long unsummarised
    # docs from the backlog at a slow cadence so a weekend ingest
    # push doesn't blow the budget all at once.
    from .synthesis import (
        materialize_summaries_due,
        run_weekly_review_if_due,
    )
    sched.register(Job(
        name="auto_summariser",
        schedule=IntervalSchedule(seconds=30 * 60),
        fn=lambda cfg, conn: materialize_summaries_due(
            conn, cfg, max_per_run=3,
        ),
    ))

    # Phase 72: weekly review — Sundays only, at most once per week.
    sched.register(Job(
        name="weekly_review",
        schedule=IntervalSchedule(seconds=60 * 60),  # check hourly
        fn=lambda cfg, conn, embedder: run_weekly_review_if_due(
            cfg, conn, embedder,
        ),
    ))

    # Phase 76: Todoist sync — push new tasks + pull remote completions.
    # 30-min cadence is responsive enough for "did I just finish that
    # in Todoist?" without burning through the rate limit.
    from .tasks_sync import run_if_due as _todoist_if_due
    sched.register(Job(
        name="todoist_sync",
        schedule=IntervalSchedule(seconds=30 * 60),
        fn=lambda cfg, conn: _todoist_if_due(cfg, conn),
    ))

    # Phase 82: email triage — classify recent IMAP docs.
    from .email_assist import classify_due as _email_classify
    sched.register(Job(
        name="email_triage",
        schedule=IntervalSchedule(seconds=15 * 60),
        fn=lambda cfg, conn: _email_classify(conn, cfg),
    ))

    # Phase 83: draft generation. Slower cadence — Sonnet calls cost
    # more than Haiku triage; spread out so cumulative spend stays
    # well below the 'email_draft' bucket cap.
    from .email_assist import generate_drafts_due as _email_drafts
    sched.register(Job(
        name="email_drafts",
        schedule=IntervalSchedule(seconds=60 * 60),
        fn=lambda cfg, conn: _email_drafts(conn, cfg),
    ))

    # Phase 87: weekly index snapshots for temporal queries. Runs
    # hourly but only fires when the last snapshot is > 7d old.
    from .memory import take_snapshot_if_due
    sched.register(Job(
        name="index_snapshot",
        schedule=IntervalSchedule(seconds=60 * 60),
        fn=lambda cfg, conn: take_snapshot_if_due(cfg, conn),
    ))

    # Phase 65: people backfill — promote PERSON entities into the
    # people table when they cross the 2-mention threshold. Runs
    # every 6h so newly-mentioned humans get profiles within a day
    # without us re-scanning the entities table on every brief.
    from .people import (
        clear_alias_cache as _people_clear_cache,
    )
    from .people import (
        materialize_from_entities as _people_backfill,
    )

    def _people_backfill_job(conn):
        n = _people_backfill(conn)
        # Invalidate the alias-matcher cache when new aliases land
        # so the next link_chunk_mentions sees them.
        if n:
            _people_clear_cache()
        return n

    sched.register(Job(
        name="people_backfill",
        schedule=CooldownSchedule(seconds=60 * 60, cooldown_hours=6),
        fn=_people_backfill_job,
    ))

    # Polish v3 round 4: schedule connector sync. Without this, the
    # daemon never pulls Gmail / GitHub / Notion / etc. — `secondbrain
    # sync` was manual-only. Cooldown 60min + interval 60min gives
    # roughly hourly syncs without busy-spinning when the daemon
    # restarts mid-cooldown.
    from .sync import run_sync_due as _run_sync_due
    sched.register(Job(
        name="connector_sync",
        schedule=CooldownSchedule(seconds=60 * 60, cooldown_hours=1),
        fn=lambda cfg, conn, embedder: _run_sync_due(cfg, conn, embedder),
    ))

    # Polish v3 round 4: tasks.materialize_from_transcripts on the
    # daemon scheduler. Previously ran only on-demand from CLI / brief
    # / dashboard — meaning a user who never opens those got no auto-
    # extracted tasks. 30min interval is responsive enough for
    # "I just had a meeting, where's my action item?" without churning.
    from .tasks import materialize_from_transcripts as _task_materialize
    sched.register(Job(
        name="tasks_from_transcripts",
        schedule=IntervalSchedule(seconds=30 * 60),
        fn=lambda conn: _task_materialize(conn),
    ))

    # Round 7: voice fidelity for email drafts.
    #   - email_reply_pairs_index: hourly link new Sent items to their
    #     incoming parents so few-shot retrieval has fresh data.
    #   - voice_profile_refresh: weekly extract of the user's voice
    #     profile from their last ~50 sent emails. Cooldown 7d.
    from .email_assist import (
        index_reply_pairs as _email_index_pairs,
    )
    from .email_assist import (
        refresh_voice_profile_if_due as _email_voice_refresh,
    )
    sched.register(Job(
        name="email_reply_pairs_index",
        schedule=IntervalSchedule(seconds=60 * 60),
        fn=lambda conn: _email_index_pairs(conn),
    ))
    sched.register(Job(
        name="email_voice_profile",
        schedule=CooldownSchedule(seconds=60 * 60, cooldown_hours=24 * 7),
        fn=lambda cfg, conn: _email_voice_refresh(conn, cfg),
    ))

    # Round 8 — meeting thanks. Hourly scan to register newly-finished
    # meetings, retry transcript matches, and auto-draft thank-yous
    # for rows where context is available. Bounded by max_per_tick
    # inside the helper so a vacation backlog doesn't blow the budget.
    from .meeting_thanks import process_due_thanks as _process_thanks
    sched.register(Job(
        name="meeting_thanks",
        schedule=IntervalSchedule(seconds=60 * 60),
        fn=lambda cfg, conn: _process_thanks(conn, cfg),
    ))

    # Round 9-A — pre-fetch meeting prep for the next 24h so the
    # CLI / dashboard / brief reads are instant. 30min interval
    # matches the prep cache TTL.
    from .meeting_prep import prefetch_upcoming as _prefetch_prep
    sched.register(Job(
        name="meeting_prep_prefetch",
        schedule=IntervalSchedule(seconds=30 * 60),
        fn=lambda cfg, conn: _prefetch_prep(conn, cfg),
    ))

    # Round 9-C — structured promise extractor. Catches natural-
    # language commitments ("I'll send Sarah the design doc by
    # Friday") that the regex-based materializer misses. Capped at
    # 5 transcripts per tick because each is one Haiku call.
    from .tasks import (
        materialize_promises_from_transcripts as _extract_promises,
    )
    sched.register(Job(
        name="task_promises",
        schedule=IntervalSchedule(seconds=60 * 60),
        fn=lambda cfg, conn: _extract_promises(conn, cfg),
    ))

    # Round 10 (#6) — nightly trim of the AI audit log (30d retention).
    from .ai_audit import trim_old as _ai_trim
    sched.register(Job(
        name="ai_audit_trim",
        schedule=CooldownSchedule(seconds=60 * 60, cooldown_hours=24),
        fn=lambda conn: _ai_trim(conn),
    ))

    # Round 10 (#9) — health checks for OAuth tokens, API keys, IMAP,
    # local LLM, watched folders. Fires every 6h (handled by run_if_due
    # internally); cached results power the dashboard banner + the
    # brief's "your calendar's been disconnected for 3 days" nudge.
    from .health_checks import run_if_due as _health_check
    sched.register(Job(
        name="health_checks",
        schedule=IntervalSchedule(seconds=60 * 60),
        fn=lambda cfg, conn: _health_check(conn, cfg),
    ))

    return sched


def _make_tray_image():
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((4, 4, 60, 60), fill=(33, 150, 243, 255))
    # Inner brain-ish shape
    draw.ellipse((16, 16, 32, 32), fill=(255, 255, 255, 255))
    draw.ellipse((32, 16, 48, 32), fill=(255, 255, 255, 255))
    draw.ellipse((24, 32, 40, 48), fill=(255, 255, 255, 255))
    return img


def run_tray(cfg: Config) -> None:
    """System tray runner. Provides status menu and a clean quit path.

    Click 'Quit' to stop the watcher and release the DB.
    Requires the [tray] extra: pip install -e .[tray]
    """
    try:
        import pystray
    except ImportError as e:
        raise ImportError(
            "Tray icon requires the [tray] extra. Install with: pip install -e .[tray]"
        ) from e

    folders = _resolve_folders(cfg)
    if not folders:
        print(
            f"No watched_folders configured. Edit {cfg.config_path} and add "
            "paths under watched_folders, then restart.",
            file=sys.stderr,
        )
        sys.exit(1)

    log_path = cfg.data_dir / "daemon.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=_make_log_handlers(log_path),
    )
    log.info("starting tray, watching: %s", [str(f) for f in folders])

    conn, embedder, transcriber, ocr_engine, entity_extractor, image_embedder = _setup_state(cfg)
    stats = _Stats()
    watcher = Watcher(
        cfg, conn, embedder,
        on_event=stats.record,
        transcriber=transcriber, ocr_engine=ocr_engine,
        entity_extractor=entity_extractor,
        image_embedder=image_embedder,
    )

    def bootstrap_async():
        _bootstrap(
            cfg, conn, embedder, transcriber, ocr_engine, entity_extractor,
            image_embedder, folders, stats,
        )
        log.info("bootstrap done")

    threading.Thread(target=bootstrap_async, name="sb-bootstrap", daemon=True).start()
    watcher.start(folders)

    icon: pystray.Icon

    def show_status(_):
        s = stats.snapshot()
        msg = (
            f"Indexed: {s['indexed']}\n"
            f"Errors: {s['errors']}\n"
            f"Uptime: {int(s['uptime'])}s\n"
            f"Last: {s['last_status']} {s['last_path'] or ''}"
        )
        icon.notify(msg, "second-brain status")

    def open_data_dir(_):
        if sys.platform == "win32":
            os.startfile(str(cfg.data_dir))  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(cfg.data_dir)])
        else:
            subprocess.Popen(["xdg-open", str(cfg.data_dir)])

    def open_logs(_):
        if sys.platform == "win32":
            os.startfile(str(log_path))  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(log_path)])
        else:
            subprocess.Popen(["xdg-open", str(log_path)])

    def quit_app(_):
        log.info("tray quit requested")
        icon.stop()

    icon = pystray.Icon(
        "second-brain",
        _make_tray_image(),
        "second-brain (watching)",
        menu=pystray.Menu(
            pystray.MenuItem("Status", show_status, default=True),
            pystray.MenuItem("Open data dir", open_data_dir),
            pystray.MenuItem("Open log", open_logs),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", quit_app),
        ),
    )

    try:
        icon.run()
    finally:
        log.info("stopping watcher...")
        watcher.stop()
        try:
            checkpoint_wal(conn)
        except Exception:  # noqa: BLE001
            pass
        conn.close()
        log.info("stopped")
