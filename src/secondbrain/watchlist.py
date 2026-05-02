"""Recurring saved queries — the watchlist runner.

A watchlist is a query you want answered on a schedule. The daemon polls
``watchlist_due`` every minute or so; for each due watchlist it runs
``run_watchlist`` which:

  1. Calls ``ask_brain`` with a wrapper prompt that asks Claude "what's
     new about <query> since <last_run_at>?". The model is free to use
     ``search_brain`` (catch up on what you've already seen) and
     ``web_search`` (find what's new).
  2. Records the synthesized answer + citations + cost into
     ``watchlist_runs``.
  3. Bumps ``watchlists.last_run_at`` so the schedule advances.

Failures are caught and stored in ``watchlist_runs.error`` so the
dashboard can surface them.

Designed to run from the daemon thread; opens its own DB connection
because watchdog handlers run in worker threads.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime

from .budget import BudgetExceededError, daily_spend_cents
from .chat import ChatResponse, ask_brain
from .config import Config
from .db import (
    applied_role_urls,
    connect,
    init_schema,
    watchlist_due,
    watchlist_get_domains,
    watchlist_latest_run,
    watchlist_previous_run,
    watchlist_run_record_finish,
    watchlist_run_record_start,
)
from .embedder import make_embedder
from .notify import notify
from .reranker import make_reranker
from .resume import load_resumes, score_against_text

log = logging.getLogger(__name__)


def _build_prompt(query: str, last_run_at: float | None) -> str:
    """Wrap the user's query with a "what's new since X" framing."""
    when = "ever" if not last_run_at else (
        datetime.fromtimestamp(last_run_at, tz=UTC)
        .strftime("%Y-%m-%d %H:%M UTC")
    )
    return (
        f"Watchlist query: {query}\n\n"
        f"Find what's new or relevant since {when}. Use web_search for "
        f"fresh content; use search_brain to check what I've already seen "
        f"(so you can highlight what's actually NEW, not just repeat). "
        f"Answer as a tight bulleted list of items with one-line summaries "
        f"and links. If nothing meaningful has changed, just say so."
    )


def run_watchlist(
    cfg: Config, conn, embedder, reranker,
    watchlist_id: int, query: str, last_run_at: float | None,
) -> ChatResponse | None:
    """Run a single watchlist and persist the result. Returns the
    ChatResponse on success, or None on a budget / API error."""
    run_id = watchlist_run_record_start(conn, watchlist_id)
    spend_before = 0.0
    try:
        spend_before = daily_spend_cents(cfg, "anthropic")
    except Exception:  # noqa: BLE001
        pass

    prompt = _build_prompt(query, last_run_at)
    # Per-watchlist domain scoping overrides cfg.web_search_allowed_domains
    # for this run only. Lets one user have a "jobs" watchlist scoped to
    # LinkedIn/Indeed and a "news" watchlist scoped to news outlets without
    # touching global config.
    domains = watchlist_get_domains(conn, watchlist_id)
    try:
        response = ask_brain(
            cfg, conn, embedder, reranker, prompt,
            web_search_allowed_domains=domains,
        )
    except BudgetExceededError as e:
        watchlist_run_record_finish(
            conn, run_id, error=f"Anthropic budget exceeded: {e}",
        )
        return None
    except Exception as e:  # noqa: BLE001
        log.warning("watchlist %s failed: %s", watchlist_id, e)
        watchlist_run_record_finish(conn, run_id, error=str(e)[:500])
        return None

    cents_spent: float | None = None
    try:
        cents_spent = max(0.0, daily_spend_cents(cfg, "anthropic") - spend_before)
    except Exception:  # noqa: BLE001
        pass

    # Resume-fit scoring (Phase 37): for each citation, pick the best
    # resume (when multiple are configured) and tag the citation with its
    # cosine score + label. Cheap — one embed per citation, no LLM call.
    # Skipped automatically when no resume_paths are configured.
    resumes = load_resumes(cfg, embedder)

    cites_payload: list[dict] = []
    for c in response.citations:
        d: dict = {
            "kind": c.kind,
            "file_path": c.file_path,
            "url": c.url,
            "page_title": c.page_title,
            "chunk_index": c.chunk_index,
            "score": round(c.score, 4),
            "text": c.text if len(c.text) <= 600 else c.text[:600] + "…",
        }
        if resumes and c.text:
            try:
                fit = score_against_text(resumes, embedder, c.text)
                if fit is not None:
                    d["fit_score"] = round(fit[0], 4)
                    d["fit_resume"] = fit[1]
                    d["fit_label"] = fit[2]
            except Exception as e:  # noqa: BLE001
                log.warning("fit scoring failed for %s: %s", c.file_path, e)
        cites_payload.append(d)

    # Diff against the previous successful run: which citation paths
    # weren't in the prior run? Used by the dashboard's "what's new"
    # callout, tray notifications, and email digest.
    new_paths, new_count = _compute_new_paths(conn, watchlist_id, run_id, cites_payload)

    watchlist_run_record_finish(
        conn, run_id,
        answer=response.text,
        citations_json=json.dumps(cites_payload),
        cents_spent=cents_spent,
        new_paths_json=json.dumps(new_paths) if new_paths else None,
        new_count=new_count,
    )
    log.info(
        "watchlist %s ran in %.2f cents; %d citations; %d new since last run",
        watchlist_id,
        cents_spent or 0.0,
        len(response.citations),
        new_count,
    )

    # Enqueue eligible new items into the reading queue. Job-flavoured
    # watchlists only enqueue "great fit" matches; news/ai/markets/research
    # presets enqueue everything new (one summary per item).
    if new_count > 0:
        try:
            from .reading_queue import enqueue_from_watchlist_run, watchlist_preset_for
            preset = watchlist_preset_for(conn, watchlist_id)
            new_items = [c for c in cites_payload if c.get("file_path") in set(new_paths)]
            queued = enqueue_from_watchlist_run(
                conn, watchlist_id=watchlist_id,
                watchlist_preset=preset,
                new_items=new_items,
            )
            if queued:
                log.info("watchlist %s queued %d item(s) for read-summary", watchlist_id, queued)
        except Exception as e:  # noqa: BLE001
            log.warning("read-queue enqueue failed for watchlist %s: %s", watchlist_id, e)

    # Tray / desktop notification when there's something genuinely new.
    # The first run has no prior to compare against; we treat it as
    # "everything new" by setting new_count to len(citations), but skip
    # the notification because the user just created the watchlist and
    # is presumably looking at the dashboard already.
    if new_count > 0 and _has_prior_run(conn, watchlist_id, run_id):
        try:
            wl_name = _name(conn, watchlist_id) or f"watchlist {watchlist_id}"
            notify(
                f"second-brain: {new_count} new on '{wl_name}'",
                response.text[:240] if response.text else
                "Open the dashboard to see what's new.",
            )
        except Exception as e:  # noqa: BLE001
            log.warning("notify after watchlist run failed: %s", e)

    return response


def _name(conn, watchlist_id: int) -> str | None:
    row = conn.execute(
        "SELECT name FROM watchlists WHERE id = ?", (watchlist_id,),
    ).fetchone()
    return row["name"] if row else None


def _has_prior_run(conn, watchlist_id: int, before_run_id: int) -> bool:
    return watchlist_previous_run(conn, watchlist_id, before_run_id) is not None


def _compute_new_paths(
    conn, watchlist_id: int, this_run_id: int,
    cites_payload: list[dict],
) -> tuple[list[str], int]:
    """Return (new_paths, new_count): paths in this run not in the previous,
    and not already in the user's application tracker.

    Identity = file_path (which is the URL for web citations and the
    virtual_path for brain citations). On first run with no prior, we
    treat every citation as new since there's nothing to diff against -
    the caller decides whether to notify.

    Application-aware filtering: if the user has saved a job application
    with role_url matching one of this run's citations, that citation is
    NOT counted as new. So a watchlist that surfaces "PM internships" won't
    keep flagging roles you've already applied to.
    """
    this_paths = [c.get("file_path") for c in cites_payload if c.get("file_path")]
    if not this_paths:
        return [], 0

    # Pull "already applied" set once. Empty when the table is fresh.
    try:
        applied = applied_role_urls(conn)
    except Exception:  # noqa: BLE001
        applied = set()

    prev = watchlist_previous_run(conn, watchlist_id, this_run_id)
    if prev is None or not prev["citations_json"]:
        # Nothing to diff against - everything not already-applied is "new".
        new_paths = [p for p in this_paths if p not in applied]
        return new_paths, len(new_paths)
    try:
        prev_cites = json.loads(prev["citations_json"]) or []
    except json.JSONDecodeError:
        prev_cites = []
    prev_paths = {c.get("file_path") for c in prev_cites if c.get("file_path")}
    new_paths = [p for p in this_paths if p not in prev_paths and p not in applied]
    return new_paths, len(new_paths)


def run_due_watchlists(cfg: Config, conn=None, embedder=None, reranker=None) -> int:
    """Run every watchlist whose schedule has elapsed. Returns count.

    Pass an existing ``conn`` / ``embedder`` / ``reranker`` to reuse the
    daemon's singletons; otherwise we open our own.
    """
    own_conn = conn is None
    if own_conn:
        embedder = make_embedder(cfg)
        conn = connect(cfg.db_path)
        init_schema(conn, embedder.dim, embedder.name)
        reranker = make_reranker(cfg)
    try:
        due = watchlist_due(conn)
        if not due:
            return 0
        for row in due:
            log.info("watchlist due: %s (%s)", row["name"], row["query"][:60])
            try:
                run_watchlist(
                    cfg, conn, embedder, reranker,
                    row["id"], row["query"], row["last_run_at"],
                )
            except Exception as e:  # noqa: BLE001
                # Belt & braces: the function itself catches and stores
                # its errors, but if anything escapes that path, we want
                # one bad watchlist not to take down the rest.
                log.warning("watchlist %s crashed: %s", row["id"], e)
        return len(due)
    finally:
        if own_conn:
            conn.close()


def latest_summary(conn, watchlist_id: int) -> dict | None:
    """Return the most recent finished run for the dashboard's "what's
    new" panel: text + parsed citations + finished_at + error + diff."""
    row = watchlist_latest_run(conn, watchlist_id)
    if row is None:
        return None
    cites: list = []
    if row["citations_json"]:
        try:
            cites = json.loads(row["citations_json"])
        except json.JSONDecodeError:
            cites = []
    new_paths: list = []
    # Older rows (pre-Phase 30) don't have the new_paths_json column.
    try:
        npj = row["new_paths_json"]
    except (IndexError, KeyError):
        npj = None
    if npj:
        try:
            new_paths = json.loads(npj)
        except json.JSONDecodeError:
            new_paths = []
    try:
        new_count = int(row["new_count"] or 0)
    except (IndexError, KeyError, TypeError, ValueError):
        new_count = 0
    return {
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "answer": row["answer"] or "",
        "citations": cites,
        "error": row["error"],
        "cents_spent": row["cents_spent"],
        "new_paths": new_paths,
        "new_count": new_count,
    }


# Re-export for "from secondbrain.watchlist import time"
_ = time
