"""Phase 62: parallel connector sync tests.

Tests cover:

  - **Fairness**: every connector's docs reach the indexer, regardless
    of how slow each fetch is.
  - **Isolation**: a connector that crashes mid-fetch doesn't kill its
    neighbours; the rest still complete.
  - **Counts**: per-connector and grand-total counts are correct.
  - **on_progress**: fires once per connector with its final counts.
  - **Empty**: empty connector list completes quickly.

Tests use fake connectors that emit deterministic docs so we don't
hit any network. Threads are bounded so the test stays fast.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from secondbrain.sync import ConnectorCounts, SyncReport, parallel_sync


@dataclass
class FakeDoc:
    """Stand-in for ConnectorDocument — only the fields parallel_sync
    needs to forward to the index_doc closure."""
    virtual_path: str


@dataclass
class FakeConnector:
    name: str
    docs: list[FakeDoc]
    delay_per_doc: float = 0.0  # simulate network latency for parallelism tests
    raise_after: int = -1       # -1 = never; otherwise raise after N docs

    def fetch(self, cfg) -> Any:
        for i, d in enumerate(self.docs):
            if self.raise_after >= 0 and i >= self.raise_after:
                raise RuntimeError(f"{self.name} crashed at doc {i}")
            if self.delay_per_doc > 0:
                time.sleep(self.delay_per_doc)
            yield d


# ============================ basic fan-out ===========================

def test_parallel_sync_handles_empty_connector_list():
    """Edge case — no connectors. Should return an empty report fast."""
    indexed: list = []
    report = parallel_sync(
        [], cfg=None,
        index_doc=lambda doc, name: indexed.append((name, doc)),
    )
    assert isinstance(report, SyncReport)
    assert report.counts == {}


def test_parallel_sync_indexes_every_doc_from_every_connector():
    """Two connectors, three docs each — all six reach the indexer."""
    a = FakeConnector(
        name="a", docs=[FakeDoc(f"a/{i}") for i in range(3)],
    )
    b = FakeConnector(
        name="b", docs=[FakeDoc(f"b/{i}") for i in range(3)],
    )

    indexed: list[tuple[str, str]] = []

    def _index(doc, name):
        indexed.append((name, doc.virtual_path))
        return "indexed"

    report = parallel_sync([a, b], cfg=None, index_doc=_index)
    assert report.counts["a"].indexed == 3
    assert report.counts["b"].indexed == 3
    assert len(indexed) == 6
    paths_a = {p for n, p in indexed if n == "a"}
    paths_b = {p for n, p in indexed if n == "b"}
    assert paths_a == {"a/0", "a/1", "a/2"}
    assert paths_b == {"b/0", "b/1", "b/2"}


def test_parallel_sync_grand_total_aggregates():
    a = FakeConnector(name="a", docs=[FakeDoc("a/0"), FakeDoc("a/1")])
    b = FakeConnector(name="b", docs=[FakeDoc("b/0")])

    def _index(doc, name):
        return "indexed"

    report = parallel_sync([a, b], cfg=None, index_doc=_index)
    total = report.grand_total()
    assert total.indexed == 3
    assert total.error == 0


def test_parallel_sync_counts_status_buckets_correctly():
    """Different status returns from index_doc → different counters."""
    statuses = ["indexed", "unchanged", "alias", "skipped", "error"]
    a = FakeConnector(
        name="a", docs=[FakeDoc(f"a/{s}") for s in statuses],
    )

    # Map paths to the status they should report.
    status_map = dict(zip([f"a/{s}" for s in statuses], statuses, strict=True))

    def _index(doc, name):
        return status_map[doc.virtual_path]

    report = parallel_sync([a], cfg=None, index_doc=_index)
    c = report.counts["a"]
    assert c.indexed == 1
    assert c.unchanged == 1
    assert c.alias == 1
    assert c.skipped == 1
    assert c.error == 1


def test_parallel_sync_records_unknown_status_silently():
    """Forward-compat: a new IndexResult status shouldn't break the
    counter — just go uncounted."""
    a = FakeConnector(name="a", docs=[FakeDoc("a/0")])

    def _index(doc, name):
        return "future_status_we_dont_know_about"

    report = parallel_sync([a], cfg=None, index_doc=_index)
    c = report.counts["a"]
    # No counter incremented for unknown.
    assert c.total() == 0


# ===================== isolation against crashes ======================

def test_parallel_sync_isolates_crashing_connector():
    """When one connector raises during fetch, others should complete
    normally and the crashing one's counts get an error_msg."""
    good = FakeConnector(
        name="good", docs=[FakeDoc(f"g/{i}") for i in range(3)],
    )
    bad = FakeConnector(
        name="bad", docs=[FakeDoc(f"b/{i}") for i in range(3)],
        raise_after=1,  # one doc through, then explode
    )

    def _index(doc, name):
        return "indexed"

    report = parallel_sync([good, bad], cfg=None, index_doc=_index)
    assert report.counts["good"].indexed == 3
    # bad got 1 doc through before crashing.
    assert report.counts["bad"].indexed == 1
    assert "crashed" in (report.counts["bad"].error_msg or "")


def test_parallel_sync_isolates_indexer_crashes():
    """If the indexer raises mid-doc, the others (in same connector + in
    parallel connectors) still process. The crashed doc bumps `error`."""
    a = FakeConnector(
        name="a", docs=[FakeDoc(f"a/{i}") for i in range(5)],
    )

    def _index(doc, name):
        if doc.virtual_path == "a/2":
            raise RuntimeError("bad doc")
        return "indexed"

    report = parallel_sync([a], cfg=None, index_doc=_index)
    c = report.counts["a"]
    assert c.indexed == 4  # the four good docs
    assert c.error == 1


def test_parallel_sync_stops_after_all_sentinels():
    """The consumer loop should terminate once every connector has
    sent its sentinel — even if no docs were emitted."""
    empty = FakeConnector(name="empty", docs=[])
    report = parallel_sync(
        [empty], cfg=None,
        index_doc=lambda doc, name: "indexed",
    )
    assert report.counts["empty"].total() == 0
    # Test asserts merely by reaching this line — the parallel_sync
    # call would hang forever if sentinels weren't being processed.


# ============================ progress callback =======================

def test_parallel_sync_fires_on_progress_once_per_connector():
    """on_progress should fire exactly once per connector when its
    sentinel arrives, with the final counts."""
    a = FakeConnector(
        name="a", docs=[FakeDoc(f"a/{i}") for i in range(2)],
    )
    b = FakeConnector(
        name="b", docs=[FakeDoc(f"b/{i}") for i in range(3)],
    )
    seen: list[ConnectorCounts] = []

    def _on_progress(counts):
        seen.append(counts)

    parallel_sync(
        [a, b], cfg=None,
        index_doc=lambda doc, name: "indexed",
        on_progress=_on_progress,
    )
    names = sorted(c.name for c in seen)
    assert names == ["a", "b"]
    by_name = {c.name: c for c in seen}
    assert by_name["a"].indexed == 2
    assert by_name["b"].indexed == 3


def test_parallel_sync_swallows_progress_callback_failure():
    """A misbehaving on_progress callback shouldn't derail the sync."""
    a = FakeConnector(name="a", docs=[FakeDoc("a/0")])

    def boom(c):
        raise RuntimeError("ui crashed")

    report = parallel_sync(
        [a], cfg=None,
        index_doc=lambda doc, name: "indexed",
        on_progress=boom,
    )
    assert report.counts["a"].indexed == 1


# ============================ parallelism win =========================

def test_parallel_sync_actually_overlaps_fetches():
    """Two connectors that each take 200ms to fetch should complete
    in <300ms when run in parallel — proving network calls overlap.

    This is the smoking-gun test for Phase 62; without it, we'd have
    no signal that the threading isn't accidentally serialised."""
    a = FakeConnector(
        name="a", docs=[FakeDoc("a/0")], delay_per_doc=0.20,
    )
    b = FakeConnector(
        name="b", docs=[FakeDoc("b/0")], delay_per_doc=0.20,
    )
    started = time.time()
    parallel_sync(
        [a, b], cfg=None,
        index_doc=lambda doc, name: "indexed",
        max_workers=2,
    )
    elapsed = time.time() - started
    # Sequential would be ~0.40s; parallel ~0.20s + indexer overhead.
    # Allow some slack on slow CI.
    assert elapsed < 0.35, (
        f"sync took {elapsed:.3f}s — fetches did not overlap"
    )


def test_parallel_sync_serializes_index_calls():
    """The single-consumer design means index_doc is never called
    concurrently — even with 4 producer threads."""
    n_concurrent = {"current": 0, "max": 0}
    lock = threading.Lock()

    def _index(doc, name):
        with lock:
            n_concurrent["current"] += 1
            n_concurrent["max"] = max(
                n_concurrent["max"], n_concurrent["current"],
            )
        # Simulate small indexer work.
        time.sleep(0.005)
        with lock:
            n_concurrent["current"] -= 1
        return "indexed"

    connectors = [
        FakeConnector(name=f"c{i}", docs=[FakeDoc(f"c{i}/{j}") for j in range(5)])
        for i in range(4)
    ]
    parallel_sync(
        connectors, cfg=None, index_doc=_index, max_workers=4,
    )
    assert n_concurrent["max"] == 1, (
        f"index_doc ran concurrently (max={n_concurrent['max']}) — "
        f"the single-consumer invariant is broken"
    )


# ============================ duration ================================

def test_sync_report_duration_reflects_clock_time():
    a = FakeConnector(name="a", docs=[FakeDoc("a/0")], delay_per_doc=0.05)
    report = parallel_sync(
        [a], cfg=None,
        index_doc=lambda doc, name: "indexed",
    )
    assert 0.04 < report.duration_seconds < 1.0


# ============================ smoke ===================================

def test_connector_counts_total_sums_buckets():
    c = ConnectorCounts(name="x", indexed=3, skipped=1, error=2)
    assert c.total() == 6


def test_grand_total_excludes_error_msg():
    """grand_total rolls counts but doesn't carry per-connector
    error_msg strings — those stay attached to specific connectors."""
    a = FakeConnector(
        name="a", docs=[FakeDoc(f"a/{i}") for i in range(2)],
        raise_after=0,
    )
    report = parallel_sync(
        [a], cfg=None, index_doc=lambda doc, name: "indexed",
    )
    total = report.grand_total()
    # ConnectorCounts has no error_msg attr exposed in totals.
    assert total.error_msg is None or total.error_msg == ""
