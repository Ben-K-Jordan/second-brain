"""Phase 62: parallel connector sync — fan out fetches, serialise writes.

Before this module, ``secondbrain sync all`` ran connectors strictly
sequentially: GitHub → Notion → Gmail → Drive → ... Each connector
spent most of its wall-clock time blocked on the network. With N
connectors averaging ~5s of latency each, 10 connectors took ~50s
even though they don't share any state.

This module runs the network-bound part (``connector.fetch()``) in a
thread pool with bounded queueing. A single consumer thread pulls
``(connector, doc)`` items off the queue and runs the indexer write
path — that part stays serial because:

1. The Voyage embedder has its own rate limits + batching that
   benefits from a single caller's view of the work.
2. SQLite serialises writers anyway; concurrent indexer threads
   would just contend for the writer lock + foreign-key constraints
   on the chunks table.
3. The entity extractor (spaCy) holds GIL-bound CPU; running N of
   those in parallel doesn't help on a Python build with GIL.

The win is purely on the *fetch* leg: 10 connectors that each spend
~5s waiting for their respective APIs now overlap instead of
serialising. Bound the worker count so we don't fork-bomb the user's
machine — defaults to 8 which is enough for the usual 10-15
configured connectors.

Callers (CLI ``sync`` + MCP ``sync_source``) get the same per-connector
counts dict they had before, so the calling code didn't need a
restructure — only the orchestrator did.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any

log = logging.getLogger(__name__)


# ---- Tunables ---------------------------------------------------------

# Default worker count. 8 covers the typical connector fan-out without
# straining the user's machine; configurable per-call for tests.
_DEFAULT_MAX_WORKERS = 8
# Queue depth: how many fetched-but-not-yet-indexed docs can buffer
# before back-pressure kicks in. 200 is enough to keep the consumer
# fed without ballooning RSS on a big drive sync.
_QUEUE_DEPTH = 200
# Sentinel value to signal a connector worker has finished. Using a
# unique object so a connector emitting None as a doc (it shouldn't,
# but defensive) doesn't accidentally close its lane.
_SENTINEL = object()


# ---- Shapes ----------------------------------------------------------

@dataclass
class ConnectorCounts:
    """Per-connector outcome breakdown — same keys the CLI's old loop
    accumulated, kept stable so callers don't need a refactor."""
    name: str
    indexed: int = 0
    skipped: int = 0
    unchanged: int = 0
    alias: int = 0
    error: int = 0
    error_msg: str | None = None  # populated when fetch() itself crashed

    def total(self) -> int:
        return self.indexed + self.skipped + self.unchanged + self.alias + self.error


@dataclass
class SyncReport:
    """Aggregate result of a parallel sync run. Preserves per-connector
    counts so callers can render the same status table."""
    counts: dict[str, ConnectorCounts] = field(default_factory=dict)
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.finished_at - self.started_at)

    def grand_total(self) -> ConnectorCounts:
        """Roll-up across connectors. Useful for the CLI's bottom line."""
        out = ConnectorCounts(name="(total)")
        for c in self.counts.values():
            out.indexed += c.indexed
            out.skipped += c.skipped
            out.unchanged += c.unchanged
            out.alias += c.alias
            out.error += c.error
        return out


# ---- Orchestration ----------------------------------------------------

def parallel_sync(
    connectors: Iterable,                 # Iterable[Connector instance]
    cfg,                                  # Config
    *,
    index_doc: Callable[[Any, str], str],  # (doc, source_name) -> status
    on_progress: Callable[[ConnectorCounts], None] | None = None,
    max_workers: int = _DEFAULT_MAX_WORKERS,
    queue_depth: int = _QUEUE_DEPTH,
) -> SyncReport:
    """Run every connector's ``fetch(cfg)`` in parallel; index docs
    serially through ``index_doc``.

    The caller supplies ``index_doc`` so this module stays
    indexer-agnostic — the CLI gets to wire its full embedder /
    entity-extractor / image-embedder context, while tests can hand
    in a stub.

    ``on_progress`` (when provided) fires once per *completed*
    connector with its final counts. The CLI uses this for the
    "[sync] gmail done — indexed=12" line; daemon usage skips it.

    Returns a ``SyncReport`` with per-connector counts + duration.
    """
    connectors = list(connectors)
    started = time.time()
    report = SyncReport(started_at=started)
    if not connectors:
        report.finished_at = time.time()
        return report

    counts = {c.name: ConnectorCounts(name=c.name) for c in connectors}
    report.counts = counts
    queue: Queue = Queue(maxsize=queue_depth)
    workers_done = threading.Event()
    pending = {"n": len(connectors)}
    pending_lock = threading.Lock()
    # Track completed connector counts so on_progress fires only once
    # per connector — using a separate flag keeps the consumer thread
    # logic simple.
    completed: set[str] = set()

    def producer(connector) -> None:
        """One thread per connector — pulls docs from .fetch() into
        the queue, then enqueues the sentinel to mark completion."""
        try:
            for doc in connector.fetch(cfg):
                queue.put((connector.name, doc))
        except Exception as e:  # noqa: BLE001
            log.warning(
                "sync: connector %r fetch crashed: %s",
                connector.name, type(e).__name__,
            )
            counts[connector.name].error_msg = (
                f"{type(e).__name__}: {e}"
            )[:500]
        finally:
            queue.put((connector.name, _SENTINEL))

    # Start producers.
    pool = ThreadPoolExecutor(
        max_workers=max(1, min(max_workers, len(connectors))),
        thread_name_prefix="sb-sync",
    )
    for c in connectors:
        pool.submit(producer, c)

    # Consume serially. Use a short queue.get timeout so shutdown
    # latency stays low — once workers_done is set, the consumer
    # notices on the next poll (within ~50ms) instead of holding
    # for the full default timeout.
    try:
        while True:
            try:
                name, item = queue.get(timeout=0.05)
            except Empty:
                if workers_done.is_set():
                    break
                continue
            if item is _SENTINEL:
                with pending_lock:
                    pending["n"] -= 1
                    if pending["n"] == 0:
                        workers_done.set()
                if name not in completed:
                    completed.add(name)
                    if on_progress is not None:
                        try:
                            on_progress(counts[name])
                        except Exception as e:  # noqa: BLE001
                            log.warning(
                                "sync: on_progress callback crashed: %s", e,
                            )
                continue
            try:
                status = index_doc(item, name)
            except Exception as e:  # noqa: BLE001
                log.warning(
                    "sync: indexing %s doc crashed: %s",
                    name, type(e).__name__,
                )
                counts[name].error += 1
                continue
            bucket = counts[name]
            if status == "indexed":
                bucket.indexed += 1
            elif status == "unchanged":
                bucket.unchanged += 1
            elif status == "alias":
                bucket.alias += 1
            elif status == "skipped":
                bucket.skipped += 1
            elif status == "error":
                bucket.error += 1
            # Unknown status → silently count nothing (forward
            # compatibility with new IndexResult states).
    finally:
        pool.shutdown(wait=True)

    report.finished_at = time.time()
    return report
