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
