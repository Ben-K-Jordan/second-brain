"""Headless and tray-icon runners that keep the watcher alive in the background.

`run_daemon` is for terminal/Task-Scheduler use; `run_tray` adds a system tray
icon with status and quit. Both consume the same `watched_folders` from config
so the user just edits config.toml once.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from .config import Config
from .db import connect, init_schema
from .embedder import make_embedder
from .entities import make_entity_extractor
from .image_embedder import make_image_embedder
from .imager import make_ocr_engine
from .indexer import IndexResult, index_folder
from .transcriber import make_transcriber
from .watcher import Watcher

log = logging.getLogger(__name__)


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
    folders = [Path(p).expanduser() for p in cfg.watched_folders]
    return [f for f in folders if f.exists()]


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
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
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
    log.info("watcher running. Ctrl-C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("stopping...")
    finally:
        watcher.stop()
        conn.close()


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
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
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

    icon: "pystray.Icon"

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
        conn.close()
        log.info("stopped")
