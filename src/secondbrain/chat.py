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
and X archive content.

You have one tool: ``search_brain(query, k)``. Use it eagerly for any \
factual question — your training data does NOT contain the user's personal \
content. When in doubt, search.

Rules:
1. Always search before answering anything that would require knowing what's \
   in the user's brain. Don't guess.
2. If the search returns nothing relevant, say "I couldn't find that in your \
   brain" — don't fabricate.
3. Cite specific sources by referring to them by their path (e.g. \
   "in github://owner/repo/issues/42"). The UI renders those as links.
4. Keep answers tight. Bullet points and short paragraphs over essays.
5. If a search came back with related-but-not-quite-right context, refine \
   the query and search again. You can search up to a few times per turn.
6. When the user asks a follow-up question, prefer reusing context from \
   earlier in the conversation; only re-search if the new question is on a \
   different topic.
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
    """One source the model retrieved while answering."""
    chunk_id: int
    file_path: str
    chunk_index: int
    text: str
    score: float


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


def stream_chat(
    cfg: Config,
    conn,
    embedder: Embedder,
    reranker: Reranker | None,
    user_message: str,
    history: list[dict] | None = None,
) -> Iterator[ChatTurnEvent]:
    """Run one chat turn, streaming events to the caller.

    ``history`` is a list of Anthropic message dicts (role/content pairs)
    representing the conversation so far. The caller is responsible for
    persisting it across turns. We append the user message, run the model
    (with possible tool-call rounds), and yield a final ``done`` event whose
    data includes the assistant's reply and updated history. The caller
    should replace its history with that updated list.
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

    citations_by_id: dict[int, Citation] = {}
    final_text_parts: list[str] = []

    for iteration in range(cfg.chat_max_tool_iterations + 1):
        # Force a no-tools final answer on the last allowed iteration so the
        # model can't loop us forever even if its instruction-following slips.
        force_answer = iteration == cfg.chat_max_tool_iterations
        try:
            response = client.messages.create(
                model=cfg.chat_model,
                max_tokens=cfg.chat_max_tokens,
                system=_SYSTEM_PROMPT,
                tools=[] if force_answer else [_SEARCH_TOOL],
                messages=messages,
            )
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

        # Stream text blocks first, defer tool_use blocks for after.
        tool_calls: list[Any] = []
        for block in response.content:
            if block.type == "text":
                if block.text:
                    final_text_parts.append(block.text)
                    yield ChatTurnEvent("text", block.text)
            elif block.type == "tool_use":
                tool_calls.append(block)

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
    citations = sorted(citations_by_id.values(), key=lambda c: -c.score)
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

    for event in stream_chat(cfg, conn, embedder, reranker, question, history):
        if event.kind == "text":
            text_parts.append(event.data)
        elif event.kind == "search":
            iterations += 1
        elif event.kind == "done":
            for c in event.data.get("citations", []) or []:
                citations.append(Citation(
                    chunk_id=c["chunk_id"],
                    file_path=c["file_path"],
                    chunk_index=c["chunk_index"],
                    text=c.get("text", ""),
                    score=c["score"],
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
