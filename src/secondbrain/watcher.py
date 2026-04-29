"""Filesystem watcher with debounced event processing."""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .config import Config, is_ignored
from .embedder import Embedder
from .entities import EntityExtractor
from .imager import OCREngine
from .indexer import IndexResult, index_file, remove_file
from .transcriber import Transcriber

log = logging.getLogger(__name__)

DEBOUNCE_SECONDS = 1.5


class _DebouncedHandler(FileSystemEventHandler):
    """Coalesces rapid events per-path. Save dialogs typically emit 3-5 events."""

    def __init__(self, cfg: Config):
        super().__init__()
        self._cfg = cfg
        self._pending: dict[Path, tuple[float, str]] = {}
        self._lock = threading.Lock()

    def _enqueue(self, path: Path, action: str) -> None:
        if is_ignored(path, self._cfg.ignore_globs):
            return
        with self._lock:
            self._pending[path] = (time.time(), action)

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._enqueue(Path(event.src_path), "upsert")

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._enqueue(Path(event.src_path), "upsert")

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._enqueue(Path(event.src_path), "delete")
            self._enqueue(Path(event.dest_path), "upsert")

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._enqueue(Path(event.src_path), "delete")

    def drain_ready(self) -> list[tuple[Path, str]]:
        """Return events that have been quiet for at least DEBOUNCE_SECONDS."""
        now = time.time()
        ready: list[tuple[Path, str]] = []
        with self._lock:
            for path, (ts, action) in list(self._pending.items()):
                if now - ts >= DEBOUNCE_SECONDS:
                    ready.append((path, action))
                    del self._pending[path]
        return ready


class Watcher:
    """Watches one or more folders and indexes changes incrementally."""

    def __init__(
        self,
        cfg: Config,
        conn: sqlite3.Connection,
        embedder: Embedder,
        on_event: Callable[[IndexResult], None] | None = None,
        transcriber: Transcriber | None = None,
        ocr_engine: OCREngine | None = None,
        entity_extractor: EntityExtractor | None = None,
    ):
        self._cfg = cfg
        self._conn = conn
        self._embedder = embedder
        self._transcriber = transcriber
        self._ocr_engine = ocr_engine
        self._entity_extractor = entity_extractor
        self._on_event = on_event or (lambda r: None)
        self._handler = _DebouncedHandler(cfg)
        self._observer = Observer()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None

    def _process_pending(self) -> None:
        for path, action in self._handler.drain_ready():
            try:
                if action == "delete":
                    result = remove_file(self._conn, path)
                else:
                    result = index_file(
                        self._conn, self._embedder, self._cfg, path,
                        transcriber=self._transcriber,
                        ocr_engine=self._ocr_engine,
                        entity_extractor=self._entity_extractor,
                    )
                self._on_event(result)
            except Exception as e:
                log.exception("watcher failed on %s: %s", path, e)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._process_pending()
            self._stop.wait(0.5)

    def start(self, folders: list[Path]) -> None:
        for folder in folders:
            if not folder.exists():
                log.warning("watch folder does not exist: %s", folder)
                continue
            self._observer.schedule(self._handler, str(folder), recursive=True)
        self._observer.start()
        self._worker = threading.Thread(target=self._run, name="sb-watcher", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        self._observer.stop()
        self._observer.join(timeout=5)
        if self._worker:
            self._worker.join(timeout=5)
