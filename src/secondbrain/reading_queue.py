"""Reading queue: high-value items from watchlists + news, with a
60-second pre-summary so you can scan on the go.

Trigger paths:
- Watchlist runs auto-enqueue items where ``fit_label == "great fit"``,
  or every new item if the watchlist's preset is news/ai/markets/research
  (where the user wants every result, not just job-fit ones).
- ``secondbrain read add <url>`` for one-off captures.
- News-connector ingestion (Phase 29) doesn't auto-enqueue; users opt in
  via watchlists scoped to news domains.

Summarisation pass:
- Polled by the daemon (same cadence as watchlists). For each pending
  queue item, ``ask_brain`` is invoked with a fetch-and-summarise prompt
  scoped to the item URL. The model uses ``web_search`` (when enabled)
  to pull the article + writes a tight 60-second precis with key points.
- Errors persist as ``summary_error`` so they don't loop.

Cost: ~$0.01-0.05 per summary. Capped per-tick by
``cfg.read_queue_summarise_per_run`` (default 5).

Read / skip lifecycle:
- ``read_at`` set when user marks read.
- ``skipped_at`` set when user marks skipped.
- Either terminates the row from the unread view; both are visible in
  history so you can find something you read three weeks ago.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

from .budget import BudgetExceededError, daily_spend_cents
from .chat import ask_brain
from .config import Config
from .db import (
    reading_queue_enqueue,
    reading_queue_pending_summary,
    reading_queue_set_summary,
    reading_queue_unread_count,
)
from .embedder import Embedder
from .notify import notify
from .reranker import Reranker

log = logging.getLogger(__name__)


# Watchlist presets where "every new item" goes into the queue. For job
# presets the queue is opt-in via fit_label="great fit" only — surfacing
# every middling-fit posting would defeat the point of the queue.
_BULK_ENQUEUE_PRESETS = frozenset({
    "news", "ai", "markets", "research", "dev",
})


def enqueue_from_watchlist_run(
    conn,
    watchlist_id: int,
    watchlist_preset: str | None,
    new_items: list[dict],
) -> int:
    """Called by the watchlist runner after a successful run. Enqueues
    items that meet the threshold:

    - For job-flavoured watchlists: only items tagged ``fit_label ==
      "great fit"``. Avoids the queue filling with stretch-fit roles.
    - For news / ai / markets / research / dev presets: everything new.

    Returns the count actually enqueued (deduped by URL).
    """
    if not new_items:
        return 0
    enqueue_all = (watchlist_preset or "") in _BULK_ENQUEUE_PRESETS
    enqueued = 0
    source = f"watchlist:{watchlist_id}"
    for item in new_items:
        url = (item.get("file_path") or item.get("url") or "").strip()
        if not url or not _is_http_url(url):
            continue
        fit_label = item.get("fit_label")
        fit_score = item.get("fit_score")
        # Filter for job-flavoured: keep only "great fit"; skip rest.
        if not enqueue_all and fit_label != "great fit":
            continue
        rid = reading_queue_enqueue(
            conn, url=url, title=item.get("page_title") or item.get("title") or "",
            source=source, fit_label=fit_label, fit_score=fit_score,
        )
        if rid is not None:
            enqueued += 1
    return enqueued


def _is_http_url(s: str) -> bool:
    """We only ever queue actual web URLs — chunks from the brain go in
    by their virtual_path which isn't useful for "click + read."""
    try:
        p = urlparse(s)
    except ValueError:
        return False
    return p.scheme in ("http", "https") and bool(p.netloc)


# --------------------------- summarisation -----------------------------

_SUMMARY_PROMPT_TMPL = """\
The user has queued this URL for a quick scan-summary:

  {url}
  Title (best guess): {title}

Fetch the page via web_search and produce a 60-second pre-read in
3-5 lines:

1. **One-sentence headline** — what's actually here.
2. **Why it might matter** — 1-2 lines of relevance.
3. **Key facts / numbers / quotes** — bullets, max 3.
4. **Verdict** — should the user spend more time on it? (yes/skim/skip)

Be tight, factual, and grounded in what's on the page. Don't pad. If
web_search can't reach the URL or returns nothing useful, say so plainly
in one sentence — don't make things up.
"""


def summarise_pending(
    cfg: Config, conn, embedder: Embedder, reranker: Reranker | None,
) -> int:
    """Generate summaries for the next batch of unsummarised items.

    Bounded by ``cfg.read_queue_summarise_per_run``. Returns count of
    summaries successfully generated this call. Failures land as
    ``summary_error`` so they stop showing up in the pending list.
    """
    if not getattr(cfg, "read_queue_enabled", True):
        return 0
    cap = int(getattr(cfg, "read_queue_summarise_per_run", 5) or 5)
    rows = reading_queue_pending_summary(conn, limit=cap)
    if not rows:
        return 0

    generated = 0
    for row in rows:
        try:
            summary = _summarise_one(
                cfg, conn, embedder, reranker,
                url=row["url"], title=row["title"],
            )
        except BudgetExceededError as e:
            reading_queue_set_summary(
                conn, row["id"], summary=None,
                error=f"Anthropic budget exceeded: {e}",
            )
            continue
        except Exception as e:  # noqa: BLE001
            log.warning("read queue: summary failed for %s: %s", row["url"], e)
            reading_queue_set_summary(
                conn, row["id"], summary=None,
                error=str(e)[:500],
            )
            continue
        if summary is None:
            reading_queue_set_summary(
                conn, row["id"], summary=None,
                error="empty response",
            )
            continue
        reading_queue_set_summary(conn, row["id"], summary=summary)
        generated += 1
    return generated


def _summarise_one(
    cfg: Config, conn, embedder: Embedder, reranker: Reranker | None,
    url: str, title: str,
) -> str | None:
    prompt = _SUMMARY_PROMPT_TMPL.format(url=url, title=title or "(unknown)")
    spend_before = 0.0
    try:
        spend_before = daily_spend_cents(cfg, "anthropic")
    except Exception:  # noqa: BLE001
        pass
    response = ask_brain(cfg, conn, embedder, reranker, prompt)
    text = (response.text or "").strip()
    if not text:
        return None
    try:
        cents = max(0.0, daily_spend_cents(cfg, "anthropic") - spend_before)
        log.info("read queue: summary for %s (~%.2f cents)", url, cents)
    except Exception:  # noqa: BLE001
        pass
    return text


# --------------------------- daemon hook -------------------------------

def run_summariser_if_due(
    cfg: Config, conn, embedder: Embedder, reranker: Reranker | None,
) -> int:
    """Daemon entrypoint. Returns count generated this tick. Notifies the
    user when the unread-count crosses configurable threshold."""
    n = summarise_pending(cfg, conn, embedder, reranker)
    if n > 0:
        try:
            unread = reading_queue_unread_count(conn)
            threshold = int(getattr(cfg, "read_queue_notify_threshold", 5) or 5)
            if unread >= threshold:
                notify(
                    f"second-brain: {unread} items in your reading queue",
                    "Open the dashboard /queue to scan summaries.",
                )
        except Exception as e:  # noqa: BLE001
            log.warning("read queue notify failed: %s", e)
    return n


# --- helpers for last-run preset lookup --------------------------------

def watchlist_preset_for(conn, watchlist_id: int) -> str | None:
    """Try to infer the preset name a watchlist uses by matching its
    saved allowed_domains against the named presets. Cheap; fallible.

    The watchlist runner uses this to decide whether ALL new items go
    into the reading queue (news/research/etc.) or only "great fit"
    job ones.
    """
    from .db import watchlist_get_domains
    from .presets import PRESETS

    domains = watchlist_get_domains(conn, watchlist_id)
    if not domains:
        return None
    domain_set = set(domains)
    for name, plist in PRESETS.items():
        if domain_set == set(plist):
            return name
    return None
