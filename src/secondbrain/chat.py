"""Chat-with-your-brain: conversational Q&A grounded in the index.

Architecture:
- Claude (Sonnet 4.6 by default) drives the conversation.
- We expose ONE tool to it: ``search_brain(query, k)``. The tool runs a
  hybrid retrieval against the local index and returns chunk text + path +
  chunk_id back to the model.
- The model decides when to call the tool, when to refine, and when to
  answer. We cap tool-use rounds at ``cfg.chat_max_tool_iterations`` so a
  pathological loop can't burn budget.
- Every API call goes through ``check_budget`` + ``record_usage`` so the
  spend ledger reflects chat traffic.
- Each answer surfaces the chunk_ids the model actually retrieved as
  citations — the UI links back to the source files.

Two surfaces:
- Streaming generator (``stream_chat``) for the dashboard / CLI.
- One-shot blocking helper (``ask_brain``) for MCP and quick-answer scripts.

History is just a list of Anthropic message dicts; callers own persistence.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .budget import BudgetExceededError, check_budget, record_usage
from .config import Config
from .embedder import Embedder
from .reranker import Reranker
from .search import SearchResult, hybrid_search

log = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You are the user's second brain — a friendly, concise assistant that answers \
questions grounded in their personal knowledge base. The base contains \
their files, notes, browser history, calendar events, emails (Gmail), Drive \
docs, GitHub issues, Linear tickets, Slack messages, Reddit/HN/Pocket items, \
X archive content, and past chat conversations with you.

You have two tools:
- ``search_brain(query, k)``: semantic + keyword search over the user's \
  indexed content. Use this for anything personal, anything they've already \
  written/saved/visited.
- ``web_search`` (when available): live results from the open web. Use this \
  for time-sensitive questions ("what PM internships came out today", \
  current news, fresh information) and for topics the user hasn't yet \
  indexed. Prefer this when the question references "today", "this week", \
  "latest", or asks about something happening now.

Rules:
1. Always search before answering anything you can't be certain of. Don't \
   guess. Pick the right tool: brain for personal/historical, web for \
   fresh/external.
2. If both tools are relevant (e.g. "have I been keeping up with X?"), \
   use both — search the brain for what they already know, then web for \
   what's new since then.
3. If a search returns nothing relevant, say so plainly. Don't fabricate.
4. Cite specific sources. For brain results use the path \
   (e.g. "github://owner/repo/issues/42"); for web results cite the URL. \
   The UI renders both as links.
5. Keep answers tight. Bullet points and short paragraphs over essays.
6. When refining: if a search came back with related-but-not-quite-right \
   context, refine the query and search again. You can search up to a few \
   times per turn.
7. When the user asks a follow-up question, prefer reusing context from \
   earlier in the conversation; only re-search if the new question is on \
   a different topic.
"""


_SEARCH_TOOL: dict[str, Any] = {
    "name": "search_brain",
    "description": (
        "Search the user's personal knowledge base. Returns up to k matched "
        "chunks with their source path, chunk index, and text. Use for any "
        "factual question about the user's files, notes, emails, calendar, "
        "GitHub/Linear/Slack/Reddit/HN content, etc."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language search query.",
            },
            "k": {
                "type": "integer",
                "description": "Max number of chunks to return (default 6, max 12).",
                "minimum": 1,
                "maximum": 12,
            },
        },
        "required": ["query"],
    },
}


@dataclass
class Citation:
    """One source the model retrieved while answering.

    Brain citations come from ``search_brain`` and have a chunk_id +
    chunk_index pointing into the local index. Web citations come from
    Anthropic's server-side web_search and have ``kind="web"`` with the
    page URL stored in ``file_path`` (so the existing dashboard rendering
    "just works") plus ``url`` and ``page_title`` for richer display.
    """
    chunk_id: int
    file_path: str
    chunk_index: int
    text: str
    score: float
    kind: str = "brain"
    url: str = ""
    page_title: str = ""


@dataclass
class ChatTurnEvent:
    """One streamed event from a chat turn.

    ``kind`` is one of:
      - "text":    a token / text-delta chunk (data is the delta string)
      - "search":  the model called search_brain (data is {"query": ..., "k": ...})
      - "results": search returned (data is a list[Citation] dict)
      - "done":    final answer assembled (data is {"text": ..., "citations": [...]})
      - "error":   something went wrong (data is the error message)
    """
    kind: str
    data: Any = None


@dataclass
class ChatResponse:
    """One-shot result of ``ask_brain``."""
    text: str
    citations: list[Citation] = field(default_factory=list)
    iterations: int = 0  # how many search rounds happened


def _tool_search(
    cfg: Config,
    conn,
    embedder: Embedder,
    reranker: Reranker | None,
    query: str,
    k: int,
) -> list[SearchResult]:
    """Run a hybrid search exactly like the dashboard's /search would."""
    return hybrid_search(
        conn, embedder, query,
        k=max(1, min(k or cfg.chat_search_k, 12)),
        alpha=cfg.hybrid_alpha,
        reranker=reranker,
        rerank_overfetch=cfg.rerank_overfetch,
        use_adaptive_alpha=cfg.adaptive_alpha,
        time_decay_weight=cfg.time_decay_weight if cfg.time_decay_enabled else 0.0,
        time_decay_half_life_days=cfg.time_decay_half_life_days,
        use_hyde=cfg.hyde_enabled,
        hyde_model=cfg.hyde_model,
        personal_prefixes=cfg.personal_path_prefixes,
        personal_boost=cfg.personal_path_boost,
        download_prefixes=cfg.download_path_prefixes,
        download_demote=cfg.download_path_demote,
        click_boost_max=cfg.click_boost_max if cfg.click_boost_enabled else 1.0,
        click_boost_half_life_days=cfg.click_boost_half_life_days,
        cfg=cfg,
    )


def _format_search_result(results: list[SearchResult]) -> str:
    """Render search results as a tool-result text block for the model."""
    if not results:
        return "No matches found in the brain."
    lines = [f"Found {len(results)} chunk(s):"]
    for i, r in enumerate(results, 1):
        # Cap each chunk so a 50-chunk megafile doesn't blow the context window.
        snippet = r.text if len(r.text) <= 1500 else r.text[:1500] + "\n[…truncated]"
        lines.append("")
        lines.append(
            f"--- ({i}) chunk_id={r.chunk_id} path={r.file_path} "
            f"chunk_index={r.chunk_index} score={r.score:.3f} ---"
        )
        lines.append(snippet)
    return "\n".join(lines)


def _normalize_history(history: list[dict] | None) -> list[dict]:
    """Defensive copy of the history we'll mutate during the turn."""
    if not history:
        return []
    return [dict(m) for m in history]


def _extract_web_search_rows(block: Any) -> list[dict[str, str]]:
    """Pull URL / title / snippet from a web_search_tool_result block.

    The Anthropic SDK returns these as Pydantic objects with a ``content``
    field that's a list of items. Each item has ``url`` and ``title``;
    ``encrypted_content`` is opaque and we ignore it. We're tolerant of
    both the nested-object shape and a stringified-JSON fallback.
    """
    content = getattr(block, "content", None)
    if content is None:
        return []
    if isinstance(content, str):
        # Error case: content is a string error message, not a result list.
        return []
    rows: list[dict[str, str]] = []
    for item in content:
        # Items are usually web_search_result objects with .url / .title
        url = getattr(item, "url", None) or (
            item.get("url") if isinstance(item, dict) else None
        ) or ""
        title = getattr(item, "title", None) or (
            item.get("title") if isinstance(item, dict) else None
        ) or ""
        if not url:
            continue
        rows.append({"url": url, "title": title or url})
    return rows


def stream_chat(
    cfg: Config,
    conn,
    embedder: Embedder,
    reranker: Reranker | None,
    user_message: str,
    history: list[dict] | None = None,
    system_prompt: str | None = None,
    web_search_allowed_domains: list[str] | None = None,
) -> Iterator[ChatTurnEvent]:
    """Run one chat turn, streaming events to the caller.

    ``history`` is a list of Anthropic message dicts (role/content pairs)
    representing the conversation so far. The caller is responsible for
    persisting it across turns. We append the user message, run the model
    (with possible tool-call rounds), and yield a final ``done`` event whose
    data includes the assistant's reply and updated history. The caller
    should replace its history with that updated list.

    ``system_prompt`` overrides the default search-grounded persona. Passing
    e.g. "You are my code reviewer" lets the user keep one conversation
    pinned to a different role without burning through chat history. None
    means "use the built-in default".

    ``web_search_allowed_domains`` overrides ``cfg.web_search_allowed_domains``
    for this single call. Watchlists set this from their per-watchlist
    domain list so a "jobs" watchlist scopes web search to job sites and
    a "news" watchlist hits news outlets.
    """
    if not user_message.strip():
        yield ChatTurnEvent("error", "empty user message")
        return

    try:
        check_budget(cfg, "anthropic")
    except BudgetExceededError as e:
        yield ChatTurnEvent("error", f"Daily Anthropic budget exceeded: {e}")
        return

    try:
        import anthropic
    except ImportError:
        yield ChatTurnEvent("error", "anthropic SDK not installed; pip install anthropic")
        return

    client = anthropic.Anthropic()
    messages = _normalize_history(history)
    messages.append({"role": "user", "content": user_message})

    # Use the conversation-specific persona when provided. Default falls
    # back to the search-grounded brain prompt.
    active_system_prompt = (system_prompt or "").strip() or _SYSTEM_PROMPT

    # Build the tool list. search_brain is always available; web_search is
    # opt-in (cfg.web_search_enabled). The web_search tool is server-side -
    # Anthropic executes it and returns results inline; we don't run anything
    # locally, but we DO record the cost.
    base_tools: list[dict[str, Any]] = [_SEARCH_TOOL]
    if cfg.web_search_enabled:
        web_tool: dict[str, Any] = {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": max(1, cfg.web_search_max_uses_per_turn),
        }
        # Per-call override (e.g. from a watchlist's preset) wins over the
        # global config. ``[]`` and ``None`` both mean "no restriction".
        effective_domains = (
            list(web_search_allowed_domains)
            if web_search_allowed_domains is not None
            else list(cfg.web_search_allowed_domains)
        )
        if effective_domains:
            web_tool["allowed_domains"] = effective_domains
        base_tools.append(web_tool)

    citations_by_id: dict[int, Citation] = {}
    final_text_parts: list[str] = []

    for iteration in range(cfg.chat_max_tool_iterations + 1):
        # Force a no-tools final answer on the last allowed iteration so the
        # model can't loop us forever even if its instruction-following slips.
        force_answer = iteration == cfg.chat_max_tool_iterations
        # True SDK streaming: emit text-delta events as they arrive instead
        # of waiting for the full response. The dashboard's SSE consumer
        # picks up each delta and paints it into the bubble immediately,
        # so long answers feel instant rather than batchy.
        try:
            with client.messages.stream(
                model=cfg.chat_model,
                max_tokens=cfg.chat_max_tokens,
                system=active_system_prompt,
                tools=[] if force_answer else base_tools,
                messages=messages,
            ) as stream:
                this_iter_text: list[str] = []
                # text_stream yields plain str deltas - one per token-ish
                # chunk. We only emit non-empty deltas to keep the wire quiet.
                for delta in stream.text_stream:
                    if not delta:
                        continue
                    this_iter_text.append(delta)
                    yield ChatTurnEvent("text", delta)
                # After exit the stream has fully resolved the final Message.
                response = stream.get_final_message()
        except anthropic.APIError as e:
            log.warning("chat API call failed: %s", e)
            yield ChatTurnEvent("error", f"Anthropic API error: {e}")
            return

        try:
            record_usage(
                cfg, "anthropic", cfg.chat_model,
                input_tokens=getattr(response.usage, "input_tokens", 0),
                output_tokens=getattr(response.usage, "output_tokens", 0),
                note=f"chat:iter{iteration}",
            )
        except Exception as e:  # noqa: BLE001
            log.warning("chat usage recording failed: %s", e)

        # Server-side web_search billing: each request to the tool costs
        # ~$0.01 regardless of result count. Anthropic exposes the count
        # via response.usage.server_tool_use.web_search_requests.
        web_search_requests = 0
        try:
            stu = getattr(response.usage, "server_tool_use", None)
            if stu is not None:
                web_search_requests = int(getattr(stu, "web_search_requests", 0) or 0)
        except Exception:  # noqa: BLE001
            web_search_requests = 0
        if web_search_requests > 0:
            try:
                record_usage(
                    cfg, "anthropic", "anthropic-web-search",
                    input_tokens=web_search_requests,
                    note=f"web_search:iter{iteration}",
                )
            except Exception as e:  # noqa: BLE001
                log.warning("web_search usage recording failed: %s", e)

        # Pull text into the running tally, surface client-side tool calls
        # (search_brain), and harvest server-side web_search citations as
        # they live in the resolved Message's content blocks.
        tool_calls: list[Any] = []
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                if block.text:
                    final_text_parts.append(block.text)
            elif btype == "tool_use":
                tool_calls.append(block)
            elif btype == "web_search_tool_result":
                # Server tool result: a list of source rows. Stream a
                # ``results`` event for the dashboard's progress trail and
                # capture each result as a Citation so it shows up in the
                # final sources list. Don't loop back to the model - the
                # web tool is server-side; Claude already has the content.
                rows = _extract_web_search_rows(block)
                if rows:
                    yield ChatTurnEvent("results", [
                        {
                            "file_path": r["url"],
                            "chunk_index": None,
                            "score": 1.0,
                            "kind": "web",
                            "page_title": r.get("title", ""),
                        }
                        for r in rows
                    ])
                    for r in rows:
                        url = r.get("url") or ""
                        if not url or url in citations_by_id:
                            continue
                        # Use a stable surrogate "chunk_id" derived from the
                        # URL so de-dup works across iterations and matches
                        # the citations_by_id contract (int keys). Negative
                        # to avoid colliding with real chunk ids.
                        sid = -(abs(hash(url)) % (1 << 31))
                        citations_by_id[sid] = Citation(
                            chunk_id=sid,
                            file_path=url,
                            chunk_index=0,
                            text=r.get("snippet") or r.get("title") or "",
                            score=1.0,
                            kind="web",
                            url=url,
                            page_title=r.get("title", ""),
                        )

        if response.stop_reason == "tool_use" and tool_calls and not force_answer:
            # Append the assistant turn (text + tool_use) to history, then
            # the tool_result(s), and loop again.
            messages.append({"role": "assistant", "content": response.content})
            tool_results: list[dict] = []
            for tc in tool_calls:
                if tc.name != "search_brain":
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": f"Unknown tool {tc.name!r}",
                        "is_error": True,
                    })
                    continue
                args = tc.input or {}
                q = (args.get("query") or "").strip()
                k = int(args.get("k") or cfg.chat_search_k)
                yield ChatTurnEvent("search", {"query": q, "k": k})
                try:
                    results = _tool_search(cfg, conn, embedder, reranker, q, k)
                except BudgetExceededError as e:
                    yield ChatTurnEvent("error", f"Voyage budget exceeded: {e}")
                    return
                except Exception as e:  # noqa: BLE001
                    log.warning("chat search failed: %s", e)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": f"Search failed: {e}",
                        "is_error": True,
                    })
                    continue
                for r in results:
                    citations_by_id.setdefault(r.chunk_id, Citation(
                        chunk_id=r.chunk_id,
                        file_path=r.file_path,
                        chunk_index=r.chunk_index,
                        text=r.text,
                        score=r.score,
                    ))
                yield ChatTurnEvent("results", [
                    {
                        "chunk_id": r.chunk_id,
                        "file_path": r.file_path,
                        "chunk_index": r.chunk_index,
                        "score": round(r.score, 4),
                    }
                    for r in results
                ])
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": _format_search_result(results),
                })
            messages.append({"role": "user", "content": tool_results})
            continue

        # No more tool calls (or we're forcing answer) — assemble the reply.
        # Even if we forced no-tools, the model still emitted text blocks above.
        # Append the assistant message to history for the caller.
        messages.append({"role": "assistant", "content": response.content})
        break

    final_text = "".join(final_text_parts).strip() or "(no response)"
    # Sort: brain citations first (sorted by score), then web citations
    # (preserve discovery order). Mixed UIs render the brain stuff with
    # rich snippets and the web stuff as link cards.
    brain_cites = sorted(
        (c for c in citations_by_id.values() if c.kind == "brain"),
        key=lambda c: -c.score,
    )
    web_cites = [c for c in citations_by_id.values() if c.kind == "web"]
    citations = list(brain_cites) + list(web_cites)
    yield ChatTurnEvent("done", {
        "text": final_text,
        "citations": [
            {
                "chunk_id": c.chunk_id,
                "file_path": c.file_path,
                "chunk_index": c.chunk_index,
                "score": round(c.score, 4),
                # Truncate the text we send to the UI; the full text lives
                # in the index and the user can click through.
                "text": c.text if len(c.text) <= 800 else c.text[:800] + "…",
                "kind": c.kind,
                "url": c.url,
                "page_title": c.page_title,
            }
            for c in citations
        ],
        "history": messages,
    })


def ask_brain(
    cfg: Config,
    conn,
    embedder: Embedder,
    reranker: Reranker | None,
    question: str,
    history: list[dict] | None = None,
    system_prompt: str | None = None,
    web_search_allowed_domains: list[str] | None = None,
) -> ChatResponse:
    """One-shot blocking version of ``stream_chat`` for MCP / scripts.

    Drains the streaming events, accumulating text and citations, and
    returns a ``ChatResponse``. Useful when the caller doesn't care about
    intermediate progress.
    """
    text_parts: list[str] = []
    citations: list[Citation] = []
    iterations = 0
    error: str | None = None

    for event in stream_chat(
        cfg, conn, embedder, reranker, question, history,
        system_prompt=system_prompt,
        web_search_allowed_domains=web_search_allowed_domains,
    ):
        if event.kind == "text":
            text_parts.append(event.data)
        elif event.kind == "search":
            iterations += 1
        elif event.kind == "done":
            for c in event.data.get("citations", []) or []:
                citations.append(Citation(
                    chunk_id=c["chunk_id"],
                    # chunk_index comes through as None for web citations;
                    # default to 0 so the dataclass typing stays clean.
                    chunk_index=c.get("chunk_index") or 0,
                    file_path=c["file_path"],
                    text=c.get("text", ""),
                    score=c["score"],
                    kind=c.get("kind", "brain"),
                    url=c.get("url", ""),
                    page_title=c.get("page_title", ""),
                ))
            text_parts = [event.data.get("text", "") or "".join(text_parts)]
        elif event.kind == "error":
            error = event.data
            break

    if error:
        return ChatResponse(text=f"[error] {error}", citations=[], iterations=iterations)
    return ChatResponse(
        text="".join(text_parts).strip(),
        citations=citations,
        iterations=iterations,
    )


# `json` is referenced indirectly through the JSONResponse path; ensure it's
# kept on the dependency graph in case someone strips unused imports.
_ = json
