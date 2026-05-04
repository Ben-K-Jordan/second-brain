"""MCP server exposing the second-brain to AI assistants."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .config import Config, load_config
from .db import (
    connect,
    connect_readonly,
    init_schema,
    search_images,
    stats,
)
from .embedder import make_embedder
from .image_embedder import make_image_embedder
from .reranker import make_reranker
from .search import hybrid_search, keyword_only, vector_only

log = logging.getLogger(__name__)

mcp = FastMCP("second-brain")

# Module-level singletons; lazily initialised so importing this file is cheap.
_cfg: Config | None = None
_conn = None         # writer — used by tools that mutate (add_task,
                      # complete_task, sync_source, ingest_url).
_read_conn = None    # read-only — used by every query / search /
                      # listing tool. Won't contend with the daemon's
                      # writer transactions; physically prevented from
                      # writing as a defense-in-depth measure.
_embedder = None
_reranker = None


def _get_state():
    """Return (cfg, writer_conn, embedder, reranker). Used by tools
    that need to write (add a task, complete a task, run sync, etc.)."""
    global _cfg, _conn, _embedder, _reranker
    if _conn is None:
        _cfg = load_config()
        _embedder = make_embedder(_cfg)
        _reranker = make_reranker(_cfg)
        _conn = connect(_cfg.db_path)
        init_schema(_conn, _embedder.dim, _embedder.name)
    return _cfg, _conn, _embedder, _reranker


def _get_read_state():
    """Return (cfg, read_conn, embedder, reranker). Read-only path —
    used by every search / listing tool so they don't contend with
    the daemon's write lock or accidentally mutate state.

    The read_conn is opened lazily and cached for the process lifetime;
    sqlite-vec needs ``load_extension`` per connection, so creating
    fresh per-call would be wasteful for a long-running stdio server.
    """
    global _cfg, _read_conn, _embedder, _reranker
    if _read_conn is None:
        # Make sure the writer has been opened first — it runs schema
        # migrations and verifies embedder compatibility. The read-only
        # conn just attaches to the existing DB file.
        _get_state()
        _read_conn = connect_readonly(_cfg.db_path)
    return _cfg, _read_conn, _embedder, _reranker


def _query_log_path(cfg) -> Path:
    return cfg.data_dir / "queries.jsonl"


def _log_query(cfg, query: str, tool: str, results: list) -> None:
    """Append a record of an AI-driven search to the query log so the user
    can audit what's been retrieved on their behalf. Best-effort - logging
    failure must not break the search.
    """
    path = _query_log_path(cfg)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Capture what was returned, not just that a query happened.
        top_paths = [r.file_path for r in results[:10]]
        row = {
            "ts": time.time(),
            "tool": tool,
            "query": query,
            "k": len(results),
            "top_paths": top_paths,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except OSError as e:
        log.warning("query log write failed: %s", e)


def _matching_entity_keys(conn, query: str, fuzzy: bool) -> list[str]:
    """Return entity text_lower values that match the query.

    With ``fuzzy=False``, returns only the exact lowercase match. With
    ``fuzzy=True``, also matches whole-word substrings in either direction
    so 'Rowling' finds 'J.K. Rowling' and vice versa. This is the lightweight
    end of canonicalisation - it doesn't merge entities in the DB, just
    treats them as siblings at query time.
    """
    q_lower = " ".join(query.lower().split()).strip()
    if not q_lower:
        return []
    if not fuzzy:
        return [q_lower]
    pattern = re.compile(r"\b" + re.escape(q_lower) + r"\b")
    matches: set[str] = {q_lower}
    rows = conn.execute("SELECT DISTINCT text_lower FROM entities").fetchall()
    for r in rows:
        t = r["text_lower"]
        if t == q_lower:
            continue
        if pattern.search(t) or re.search(r"\b" + re.escape(t) + r"\b", q_lower):
            matches.add(t)
    return list(matches)


def _safe(s: str | None) -> str:
    """Phase 88 — redact sensitive content (API keys, secrets, etc.)
    before surfacing chunk / file / memory text to the MCP client.
    Indexes still hold the verbatim text; we mask only at the render
    boundary so search recall isn't crippled."""
    if not s:
        return ""
    try:
        from .safety import redact_text
    except ImportError:
        return s
    return redact_text(s)


def _format_results(results, header: str) -> str:
    if not results:
        return f"{header}\n\n(no matches)"
    lines = [header, ""]
    for i, r in enumerate(results, 1):
        sources = "+".join(r.sources)
        tag = "reranked" if r.reranked else sources
        lines.append(f"### {i}. {r.file_path} (chunk {r.chunk_index}, via {tag}, score={r.score:.4f})")
        snippet = r.text if len(r.text) <= 1200 else r.text[:1200] + "..."
        lines.append(_safe(snippet))
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def search_brain(
    query: str,
    k: int = 10,
    folder: str | None = None,
    kind: str | None = None,
    since_days: int | None = None,
) -> str:
    """Hybrid search across your indexed files (vector + keyword, fused, then reranked).

    Applies query-adaptive alpha (push toward BM25 for proper-noun queries,
    toward vector for prose) and a gentle recency boost. Returns matched
    text chunks with file paths so you can cite or open them.

    Optional filters scope the search:
      - ``folder``: path-prefix match. Pair with `list_folders` to discover prefixes.
      - ``kind``: 'document' / 'code' / 'audio_video' / 'image' / 'url'.
      - ``since_days``: only files modified within the last N days.
    """
    # Read-only conn — won't contend with the daemon's writer.
    cfg, conn, embedder, reranker = _get_read_state()
    results = hybrid_search(
        conn, embedder, query, k=k, alpha=cfg.hybrid_alpha,
        reranker=reranker, rerank_overfetch=cfg.rerank_overfetch,
        use_adaptive_alpha=cfg.adaptive_alpha,
        time_decay_weight=cfg.time_decay_weight if cfg.time_decay_enabled else 0.0,
        time_decay_half_life_days=cfg.time_decay_half_life_days,
        path_prefix=folder,
        kind=kind,
        since_days=since_days,
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
    _log_query(cfg, query, "search_brain", results)
    header = f"# Hybrid search: {query!r}"
    if folder or kind or since_days is not None:
        bits = []
        if folder:
            bits.append(f"folder={folder}")
        if kind:
            bits.append(f"kind={kind}")
        if since_days is not None:
            bits.append(f"since={since_days}d")
        header += "  [filters: " + ", ".join(bits) + "]"
    return _format_results(results, header)


@mcp.tool()
def ask_brain(question: str) -> str:
    """One-shot grounded Q&A: ask Claude (Sonnet 4.6 by default) to answer a
    question by searching your indexed knowledge.

    Use this instead of ``search_brain`` when you want a synthesized answer
    rather than raw chunks. The model decides how many searches to run, then
    writes a concise answer with explicit citations to the chunks it used.
    Costs a few cents per call against your Anthropic budget.
    """
    from .chat import ask_brain as _ask

    cfg, conn, embedder, reranker = _get_state()
    response = _ask(cfg, conn, embedder, reranker, question)
    if not response.citations:
        return response.text
    cite_lines = ["", "## Sources"]
    for i, c in enumerate(response.citations, 1):
        cite_lines.append(f"({i}) {c.file_path} · chunk {c.chunk_index} · score {c.score:.3f}")
    return response.text + "\n" + "\n".join(cite_lines)


@mcp.tool()
def vector_search(query: str, k: int = 10) -> str:
    """Pure semantic (vector) search. Best for conceptual questions where exact wording differs."""
    cfg, conn, embedder, reranker = _get_read_state()
    results = vector_only(
        conn, embedder, query, k=k,
        reranker=reranker, rerank_overfetch=cfg.rerank_overfetch,
    )
    return _format_results(results, f"# Vector search: {query!r}")


@mcp.tool()
def keyword_search(query: str, k: int = 10) -> str:
    """Pure BM25 keyword search. Best for proper nouns, IDs, exact strings."""
    cfg, conn, embedder, reranker = _get_read_state()
    results = keyword_only(
        conn, embedder, query, k=k,
        reranker=reranker, rerank_overfetch=cfg.rerank_overfetch,
    )
    return _format_results(results, f"# Keyword search: {query!r}")


@mcp.tool()
def find_by_tag(tag: str, k: int = 20) -> str:
    """Find chunks with a given LLM-assigned topic tag.

    Tags come from `secondbrain tag` (opt-in CLI command). Use this when
    you remember the topic of something but not the exact wording — e.g.
    "everything tagged 'capital budgeting'".
    """
    _, conn, _, _ = _get_state()
    rows = conn.execute(
        "SELECT c.text, c.chunk_index, f.path, f.mtime "
        "FROM chunk_tags t "
        "JOIN chunks c ON c.id = t.chunk_id "
        "JOIN files f ON f.id = c.file_id "
        "WHERE LOWER(t.tag) = ? "
        "ORDER BY f.mtime DESC LIMIT ?",
        (tag.strip().lower(), k),
    ).fetchall()
    if not rows:
        return f"(no chunks tagged {tag!r}; run `secondbrain tag` to populate tags)"
    lines = [f"# Chunks tagged {tag!r}", ""]
    for r in rows:
        snippet = r["text"] if len(r["text"]) <= 600 else r["text"][:600] + "..."
        lines.append(f"### {r['path']} (chunk {r['chunk_index']})")
        lines.append(snippet)
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def list_tags(top_n: int = 50) -> str:
    """List the most-used topic tags with counts. Pair with `find_by_tag`."""
    _, conn, _, _ = _get_state()
    rows = conn.execute(
        "SELECT tag, COUNT(DISTINCT chunk_id) AS n "
        "FROM chunk_tags GROUP BY tag ORDER BY n DESC LIMIT ?",
        (top_n,),
    ).fetchall()
    if not rows:
        return "(no tags yet — run `secondbrain tag` to populate)"
    return "\n".join(f"  {r['n']:5d}  {r['tag']}" for r in rows)


@mcp.tool()
def daily_briefing(hours: int = 24) -> str:
    """Generate a Claude-written briefing of what's entered the brain recently.

    Reads files indexed in the last N hours, pulls top entities and sample
    chunks, and asks Claude Opus 4.7 to write a short, useful summary covering
    what's new, recurring threads, and anomalies worth attention. Requires
    ANTHROPIC_API_KEY in the environment.
    """
    from .briefing import generate_briefing

    cfg, conn, _, _ = _get_state()
    return generate_briefing(conn, cfg, hours=hours)


@mcp.tool()
def list_chat_conversations(limit: int = 20) -> str:
    """Round 16 (Phase F): list dashboard chat conversations.

    The dashboard's ``/chat`` page persists every conversation in the
    same SQLite as the brain. This tool surfaces them to MCP clients
    (Claude Desktop, etc.) so you can pick up a conversation across
    surfaces — start it on the dashboard at lunch, continue it from
    Claude Desktop after dinner.

    Returns a markdown table: id, last-updated, title, message count.
    """
    from datetime import datetime

    from .db import chat_list_conversations

    _, conn, _, _ = _get_state()
    rows = chat_list_conversations(conn, limit=limit)
    if not rows:
        return "_No chat conversations yet._"
    lines = ["| id | last updated | title | msgs |",
             "|----|--------------|-------|------|"]
    for r in rows:
        when = datetime.fromtimestamp(r["updated_at"]).strftime("%Y-%m-%d %H:%M")
        title = (r["title"] or "(untitled)")[:60].replace("|", "\\|")
        lines.append(f"| {r['id']} | {when} | {title} | {r['n_messages']} |")
    return "\n".join(lines)


@mcp.tool()
def get_chat_conversation(conversation_id: int, max_messages: int = 50) -> str:
    """Fetch the message history for a dashboard chat conversation.

    Use after ``list_chat_conversations`` to pick up where you left
    off on the dashboard. Returns the messages as Markdown so you can
    quote them back conversationally.

    Round 17 fix (audit-found gap H2): every message body, the
    title, and the system prompt pass through ``_safe`` (which
    applies ``redact_text``) on their way out. Secrets pasted into
    a previous turn don't get exfiltrated to the LLM consumer.
    """
    import json

    from .db import chat_get_conversation, chat_get_messages

    _, conn, _, _ = _get_state()
    conv = chat_get_conversation(conn, conversation_id)
    if conv is None:
        return f"_Conversation #{conversation_id} not found._"
    rows = chat_get_messages(conn, conversation_id)
    if max_messages and len(rows) > max_messages:
        rows = rows[-max_messages:]
        truncated = True
    else:
        truncated = False
    lines = [
        f"# {_safe(conv['title']) or 'Conversation'}",
        f"_Conversation #{conversation_id}, "
        f"{len(rows)} message(s) shown_",
        "",
    ]
    if truncated:
        lines.append("_(showing only the most recent " +
                     str(max_messages) + " messages)_\n")
    if conv["system_prompt"]:
        lines.append(
            "**System prompt:** " + _safe(conv["system_prompt"][:300]),
        )
        lines.append("")
    for row in rows:
        role = row["role"]
        try:
            content = json.loads(row["content_json"])
        except (json.JSONDecodeError, TypeError):
            content = row["content_json"]
        if isinstance(content, list):
            # Tool blocks — just stringify text parts.
            text = " ".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in content
            )
        else:
            text = str(content)
        lines.append(f"### {role}")
        lines.append(_safe(text))
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def append_chat_message(
    conversation_id: int, role: str, text: str,
) -> str:
    """Append a message to a dashboard chat conversation.

    This lets an MCP-side conversation feed back into the dashboard
    history. Use it when you want to log "asked Claude about X via
    Claude Desktop" so the dashboard's ``/chat/{id}`` view shows
    the full cross-surface conversation. Role must be 'user' or
    'assistant'.

    Round 17 fix (audit-found gap H2): the persisted text passes
    through ``_safe`` (redact_text) so secrets pasted into Claude
    Desktop don't land in the dashboard chat history unmasked, where
    they'd be visible to anyone reading the page or fetched back via
    ``get_chat_conversation``.

    Returns the new message id.
    """
    import json

    from .db import chat_append_message, chat_get_conversation

    _, conn, _, _ = _get_state()
    if role not in ("user", "assistant"):
        return f"_Bad role {role!r}; must be 'user' or 'assistant'._"
    if chat_get_conversation(conn, conversation_id) is None:
        return f"_Conversation #{conversation_id} not found._"
    redacted = _safe(text)
    mid = chat_append_message(
        conn, conversation_id, role, json.dumps(redacted),
    )
    return f"OK: appended message #{mid} to conversation #{conversation_id}"


@mcp.tool()
def create_chat_conversation(title: str) -> str:
    """Start a new dashboard chat conversation. Returns the new id.

    Pair with ``append_chat_message`` to seed the conversation with
    the user/assistant turns from an MCP-side exchange.
    """
    from .db import chat_create_conversation

    _, conn, _, _ = _get_state()
    cid = chat_create_conversation(conn, title.strip() or "(untitled)")
    return f"OK: created conversation #{cid} ({title.strip()})"


@mcp.tool()
def weekly_review(regenerate: bool = False) -> str:
    """Round 16 (Phase B): Show this week's personal letter — a Sonnet-written
    synthesis of what happened across email, journal, tasks, habits, health,
    meetings, and insights.

    Default: returns the existing letter for this week (cached, idempotent).
    Pass ``regenerate=True`` to force a fresh LLM call (replaces the existing
    letter for this week).

    The letter is also generated automatically by the daemon every Sunday.
    Use this tool from a chat to ask "what did I do this week?" — Claude will
    pull the letter and reference it conversationally.
    """
    from . import weekly_letter

    cfg, conn, _, _ = _get_state()
    if regenerate:
        letter = weekly_letter.generate_and_save(cfg, conn, overwrite=True)
        prefix = "**(Just regenerated)**\n\n"
    else:
        letter = weekly_letter.latest_letter(conn)
        if letter is None:
            # No letter yet — generate one now.
            letter = weekly_letter.generate_and_save(cfg, conn)
            prefix = "**(Generated on first request)**\n\n"
        else:
            prefix = ""
    return prefix + letter.letter_md


@mcp.tool()
def sync_source(source: str = "all") -> str:
    """Pull recent documents from a connector (github / notion / browser /
    calendar) into the index. Use 'all' to run every configured connector.

    Connectors read credentials from env vars (GITHUB_TOKEN, NOTION_TOKEN,
    CALENDAR_ICS_URL); browser reads local Chrome/Edge SQLite history. If a
    connector isn't configured, it's silently skipped.
    """
    from .connectors import all_connectors, get_connector
    from .entities import make_entity_extractor
    from .indexer import index_text

    cfg, conn, embedder, _ = _get_state()
    entity_extractor = None
    if cfg.entities_enabled:
        try:
            entity_extractor = make_entity_extractor(cfg)
        except (ImportError, RuntimeError):
            entity_extractor = None

    if source == "all":
        connector_classes = all_connectors()
    else:
        cls = get_connector(source)
        if cls is None:
            return f"Unknown connector: {source!r}"
        connector_classes = [cls]

    lines = []
    for cls in connector_classes:
        c = cls()
        if not c.is_enabled(cfg):
            lines.append(f"skip   {c.name}: not configured")
            continue
        counts = {"indexed": 0, "skipped": 0, "unchanged": 0, "alias": 0, "error": 0}
        try:
            for doc in c.fetch(cfg):
                result = index_text(
                    conn, embedder, cfg,
                    virtual_path=doc.virtual_path,
                    title=doc.title,
                    content=doc.content,
                    mtime=doc.mtime,
                    kind=doc.kind,
                    source=doc.source,
                    entity_extractor=entity_extractor,
                )
                counts[result.status] = counts.get(result.status, 0) + 1
        except Exception as e:
            lines.append(f"error  {c.name}: {e}")
            continue
        lines.append(
            f"done   {c.name}: indexed={counts['indexed']} unchanged={counts['unchanged']} "
            f"alias={counts['alias']} errors={counts['error']}"
        )
    return "\n".join(lines) if lines else "(no connectors ran)"


@mcp.tool()
def recent_queries(n: int = 30) -> str:
    """Show the most recent queries the AI has run against your brain.

    Useful for auditing what context has been retrieved on your behalf.
    Reads from ~/.secondbrain/queries.jsonl.
    """
    cfg, _, _, _ = _get_state()
    path = _query_log_path(cfg)
    if not path.exists():
        return "(no queries logged yet)"
    rows: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError as e:
        return f"(log read failed: {e})"

    rows = rows[-n:]
    rows.reverse()
    if not rows:
        return "(no queries logged yet)"
    lines = []
    for r in rows:
        ts = r.get("ts", 0)
        age_min = (time.time() - ts) / 60 if ts else 0
        lines.append(
            f"[{age_min:6.1f}m ago] {r.get('tool', '?'):14s} k={r.get('k', 0):2d}  {r.get('query', '')!r}"
        )
        for p in r.get("top_paths", [])[:3]:
            lines.append(f"             -> {p}")
    return "\n".join(lines)


@mcp.tool()
def ingest_url(url: str) -> str:
    """Fetch a URL and add its contents to the brain (article, PDF, YouTube, ...).

    Returns a one-line status. Re-ingesting an unchanged URL is a no-op.
    """
    from .entities import make_entity_extractor
    from .indexer import index_url

    cfg, conn, embedder, _ = _get_state()
    entity_extractor = None
    if cfg.entities_enabled:
        try:
            entity_extractor = make_entity_extractor(cfg)
        except (ImportError, RuntimeError):
            entity_extractor = None
    result = index_url(conn, embedder, cfg, url, entity_extractor=entity_extractor)
    if result.status == "indexed":
        return f"Indexed {url} ({result.chunks} chunks)"
    if result.status == "unchanged":
        return f"Unchanged: {url}"
    return f"{result.status}: {url} ({result.reason})"


@mcp.tool()
def image_search(query: str, k: int = 10) -> str:
    """Semantic search over your indexed images via voyage-multimodal-3.

    Embeds the query as text and finds visually similar images. Works for
    "the diagram of system architecture", "screenshot with a chart",
    "photo of my whiteboard", etc. Pair with `get_file` to view the
    matching image's path.
    """
    cfg, conn, _, _ = _get_state()
    if not cfg.image_embed_enabled:
        return "(image embedding disabled in config)"
    img_embedder = make_image_embedder(cfg)
    if img_embedder is None:
        return "(no multimodal embedder available - set VOYAGE_API_KEY)"
    q_emb = img_embedder.embed_text_query(query)
    rows = search_images(conn, q_emb, k=k)
    if not rows:
        return "(no images indexed yet)"
    lines = [f"# Image search: {query!r}", ""]
    for _img_id, path, mtime, distance in rows:
        age_days = (time.time() - mtime) / 86400 if mtime else 0
        lines.append(f"  distance={distance:.4f}  ({age_days:.1f}d ago)  {path}")
    return "\n".join(lines)


@mcp.tool()
def get_file(path: str) -> str:
    """Return the full text contents of a file by path. Sensitive
    content (API keys, secrets) is redacted at the render boundary
    via Phase 88 patterns — the underlying file is untouched."""
    p = Path(path)
    if not p.exists():
        return f"File not found: {path}"
    if not p.is_file():
        return f"Not a file: {path}"
    try:
        return _safe(p.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        return f"Error reading file: {e}"


@mcp.tool()
def get_recent(n: int = 20, days: int | None = None) -> str:
    """List the most recently modified files in the index.

    If ``days`` is given, restrict to files modified within that window.
    Useful for "what was I working on this week" queries.
    """
    _, conn, _, _ = _get_state()
    if days is not None:
        cutoff = time.time() - days * 86400
        rows = conn.execute(
            "SELECT path, mtime, kind FROM files WHERE mtime >= ? ORDER BY mtime DESC LIMIT ?",
            (cutoff, n),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT path, mtime, kind FROM files ORDER BY mtime DESC LIMIT ?", (n,)
        ).fetchall()
    if not rows:
        return "(no matching files)"
    lines = []
    for r in rows:
        age_days = (time.time() - r["mtime"]) / 86400
        lines.append(f"[{r['kind']:12s}] {r['path']}  ({age_days:.1f}d ago)")
    return "\n".join(lines)


@mcp.tool()
def list_folders(top_n: int = 30) -> str:
    """List the distinct parent folders represented in the index, with file counts.

    Useful for an assistant to discover the structure of your brain before
    drilling in with `search_brain` or `files_in_folder`.
    """
    _, conn, _, _ = _get_state()
    # Sqlite has no dirname; fall back to splitting in Python so it works
    # uniformly across forward/back slashes.
    rows = conn.execute("SELECT path FROM files").fetchall()
    counts: dict[str, int] = {}
    for r in rows:
        p = Path(r["path"]).parent.as_posix()
        counts[p] = counts.get(p, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])[:top_n]
    if not ordered:
        return "(index is empty)"
    return "\n".join(f"{n:5d}  {folder}" for folder, n in ordered)


@mcp.tool()
def list_file_types() -> str:
    """Counts of indexed files grouped by kind (document/code/audio_video/image)."""
    _, conn, _, _ = _get_state()
    rows = conn.execute(
        "SELECT kind, COUNT(*) AS n FROM files GROUP BY kind ORDER BY n DESC"
    ).fetchall()
    if not rows:
        return "(index is empty)"
    return "\n".join(f"{r['n']:5d}  {r['kind']}" for r in rows)


@mcp.tool()
def files_in_folder(folder_prefix: str, limit: int = 50) -> str:
    """List indexed files whose path starts with ``folder_prefix``.

    Path-prefix match (case-insensitive on Windows). Pair with `list_folders`
    to discover viable prefixes.
    """
    _, conn, _, _ = _get_state()
    pattern = folder_prefix.replace("\\", "/").rstrip("/") + "%"
    rows = conn.execute(
        "SELECT path, mtime, kind FROM files "
        "WHERE REPLACE(path, '\\', '/') LIKE ? "
        "ORDER BY mtime DESC LIMIT ?",
        (pattern, limit),
    ).fetchall()
    if not rows:
        return f"(no files matching prefix {folder_prefix!r})"
    return "\n".join(f"[{r['kind']:12s}] {r['path']}" for r in rows)


@mcp.tool()
def index_status() -> str:
    """Report what's in the index: file count, chunk count, embedder, last update."""
    cfg, conn, _, reranker = _get_state()
    s = stats(conn)
    last = s["last_indexed_at"]
    last_str = "never" if last is None else f"{last:.0f} (epoch)"
    rerank = f"{reranker.name}" if reranker else "disabled"
    return (
        f"Files: {s['files']}\n"
        f"Chunks: {s['chunks']}\n"
        f"Entities: {s.get('entities', 0)}\n"
        f"Embedder: {s['embedder']} (dim={s['embedding_dim']})\n"
        f"Reranker: {rerank}\n"
        f"Last indexed: {last_str}"
    )


@mcp.tool()
def list_entities(label: str | None = None, top_n: int = 30) -> str:
    """Most-mentioned entities in the brain, with chunk counts.

    Optional ``label`` filter (PERSON, ORG, GPE, LOC, FAC, PRODUCT, EVENT,
    WORK_OF_ART, LAW, DATE, MONEY, LANGUAGE, NORP). Useful for an assistant
    to find recurring people, organizations, projects, etc.
    """
    _, conn, _, _ = _get_state()
    if label:
        rows = conn.execute(
            "SELECT text, label, COUNT(DISTINCT chunk_id) AS n "
            "FROM entities WHERE label = ? "
            "GROUP BY text_lower, label ORDER BY n DESC LIMIT ?",
            (label.upper(), top_n),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT text, label, COUNT(DISTINCT chunk_id) AS n "
            "FROM entities GROUP BY text_lower, label ORDER BY n DESC LIMIT ?",
            (top_n,),
        ).fetchall()
    if not rows:
        return "(no entities indexed; install [ner] extra and re-index)"
    lines = [f"{r['n']:5d}  [{r['label']:10s}]  {r['text']}" for r in rows]
    return "\n".join(lines)


@mcp.tool()
def find_mentions(entity: str, k: int = 10, fuzzy: bool = True) -> str:
    """Find chunks that mention an entity. Fuzzy by default.

    With ``fuzzy=True`` (default), 'Rowling' matches 'J.K. Rowling' and vice
    versa via whole-word substring matching in both directions. Pass
    ``fuzzy=False`` to require an exact match.
    """
    _, conn, _, _ = _get_state()
    keys = _matching_entity_keys(conn, entity, fuzzy)
    if not keys:
        return f"(no mentions of {entity!r})"
    placeholders = ",".join("?" * len(keys))
    rows = conn.execute(
        f"SELECT DISTINCT c.id AS chunk_id, c.text, c.chunk_index, f.path, f.mtime "
        f"FROM entities e "
        f"JOIN chunks c ON c.id = e.chunk_id "
        f"JOIN files f ON f.id = c.file_id "
        f"WHERE e.text_lower IN ({placeholders}) "
        f"ORDER BY f.mtime DESC LIMIT ?",
        [*keys, k],
    ).fetchall()
    if not rows:
        return f"(no mentions of {entity!r})"
    title = f"Mentions of {entity!r}"
    if fuzzy and len(keys) > 1:
        title += f"  (fuzzy matched {len(keys)} aliases)"
    lines = [f"# {title}", ""]
    for r in rows:
        snippet = r["text"] if len(r["text"]) <= 600 else r["text"][:600] + "..."
        lines.append(f"### {r['path']} (chunk {r['chunk_index']})")
        lines.append(_safe(snippet))
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def entity_timeline(entity: str, limit: int = 30, fuzzy: bool = True) -> str:
    """Files that mention an entity, sorted by file mtime (newest first).

    A timeline view: when does this person/org/thing show up in your brain?
    Fuzzy by default - 'Rowling' covers 'J.K. Rowling'.
    """
    _, conn, _, _ = _get_state()
    keys = _matching_entity_keys(conn, entity, fuzzy)
    if not keys:
        return f"(no mentions of {entity!r})"
    placeholders = ",".join("?" * len(keys))
    rows = conn.execute(
        f"SELECT DISTINCT f.path, f.mtime, f.kind "
        f"FROM entities e "
        f"JOIN chunks c ON c.id = e.chunk_id "
        f"JOIN files f ON f.id = c.file_id "
        f"WHERE e.text_lower IN ({placeholders}) "
        f"ORDER BY f.mtime DESC LIMIT ?",
        [*keys, limit],
    ).fetchall()
    if not rows:
        return f"(no mentions of {entity!r})"
    lines = [f"# Timeline of {entity!r}", ""]
    for r in rows:
        age_days = (time.time() - r["mtime"]) / 86400
        lines.append(f"  {age_days:6.1f}d ago  [{r['kind']:12s}]  {r['path']}")
    return "\n".join(lines)


@mcp.tool()
def entity_neighbors(entity: str, top_n: int = 20, fuzzy: bool = True) -> str:
    """Entities that most often co-occur with the given entity in the same chunk.

    This is the implicit knowledge graph in your brain: who shows up together,
    which projects connect to which people, what topics cluster. Self-join on
    chunk_id, ranked by distinct co-occurring chunks. Fuzzy default treats
    'Rowling' / 'J.K. Rowling' as the same when collecting co-occurrences.
    """
    _, conn, _, _ = _get_state()
    keys = _matching_entity_keys(conn, entity, fuzzy)
    if not keys:
        return f"(no entity matching {entity!r})"
    placeholders = ",".join("?" * len(keys))
    rows = conn.execute(
        f"SELECT b.text AS text, b.label AS label, "
        f"       COUNT(DISTINCT b.chunk_id) AS n "
        f"FROM entities a "
        f"JOIN entities b ON a.chunk_id = b.chunk_id "
        f"WHERE a.text_lower IN ({placeholders}) "
        f"  AND b.text_lower NOT IN ({placeholders}) "
        f"GROUP BY b.text_lower, b.label "
        f"ORDER BY n DESC LIMIT ?",
        [*keys, *keys, top_n],
    ).fetchall()
    if not rows:
        return f"(no co-occurrences for {entity!r})"
    title = f"Neighbors of {entity!r}"
    if fuzzy and len(keys) > 1:
        title += f"  (fuzzy matched {len(keys)} aliases)"
    lines = [f"# {title}", ""]
    for r in rows:
        lines.append(f"  {r['n']:4d}  [{r['label']:12s}]  {r['text']}")
    return "\n".join(lines)


# =============================================================
# Phase 44 / 47 / 52 / 56 — chat-agent-facing tools.
#
# These let an AI assistant interact with the tasks / health / links
# / brief surfaces without shelling out to the CLI. Each tool returns
# a concise plaintext payload the model can quote back to the user.
# =============================================================

@mcp.tool()
def morning_brief() -> str:
    """Generate today's daily brief — calendar + assignments due
    soon + open action items + reading queue + watchlist hits +
    Oura health snapshot. Pure aggregation; no LLM call to assemble.

    Use this when the user asks 'what's on my plate?' / 'how's my
    day looking?' / 'morning brief'."""
    from .daily_brief import generate_brief_markdown

    cfg, conn, _, _ = _get_state()
    return generate_brief_markdown(cfg, conn)


@mcp.tool()
def list_open_tasks(limit: int = 20) -> str:
    """List the user's currently-open action items. Each line includes
    the task id (so you can complete them via ``complete_task``), age
    in days, and the source doc when extracted from a transcript.

    Materialises new items from recent transcripts before returning,
    so the list is always fresh."""
    from . import tasks as tasks_mod

    cfg, conn, _, _ = _get_state()
    try:
        tasks_mod.materialize_from_transcripts(conn)
    except Exception:  # noqa: BLE001
        pass
    rows = tasks_mod.list_open(conn, limit=limit)
    if not rows:
        return "No open tasks. Inbox zero."
    now = time.time()
    lines = [f"Open tasks ({len(rows)}):"]
    for t in rows:
        age = max(0, int((now - t.created_at) // 86400))
        suffix = (
            f"  (from: {_safe(t.source_title)})"
            if t.source_path != "manual" else ""
        )
        age_label = f" [{age}d]" if age > 0 else ""
        lines.append(f"  #{t.id}{age_label}  {_safe(t.text)}{suffix}")
    return "\n".join(lines)


@mcp.tool()
def add_task(text: str) -> str:
    """Add a manual task to the user's task list. Returns the new id
    so subsequent calls can reference it.

    Use this when the user says 'remind me to ...' / 'add a task ...'
    in conversation."""
    from . import tasks as tasks_mod

    cfg, conn, _, _ = _get_state()
    tid = tasks_mod.add_manual(conn, text)
    if tid is None:
        return "(empty task text — nothing added)"
    return f"Added task #{tid}: {text}"


@mcp.tool()
def complete_task(task_id: int) -> str:
    """Mark one task complete by id. Returns confirmation, or 'not
    found' / 'already done' when the call was a no-op.

    Use this when the user says 'I did X' / 'mark X done' — find the
    matching id via ``list_open_tasks`` first if you don't know it."""
    from . import tasks as tasks_mod

    cfg, conn, _, _ = _get_state()
    t = tasks_mod.get(conn, task_id)
    if t is None:
        return f"Task #{task_id} not found."
    if not tasks_mod.mark_done(conn, task_id):
        return f"Task #{task_id} was already done."
    return f"✓ #{task_id}: {t.text}"


@mcp.tool()
def search_tasks(query: str, include_done: bool = False) -> str:
    """Find tasks by substring match (case-insensitive). When
    ``include_done`` is true, also searches completed/cancelled
    history."""
    from . import tasks as tasks_mod

    cfg, conn, _, _ = _get_state()
    rows = tasks_mod.search(conn, query, include_done=include_done)
    if not rows:
        return f"No tasks match {query!r}."
    lines = [f"Tasks matching {query!r} ({len(rows)}):"]
    for t in rows:
        marker = (
            "✓" if t.status == "done" else
            "✗" if t.status == "cancelled" else " "
        )
        lines.append(f"  {marker} #{t.id}  {_safe(t.text)}")
    return "\n".join(lines)


@mcp.tool()
def find_related(path: str, limit: int = 5) -> str:
    """Find docs semantically similar to the given path (Phase 52
    backlinks). Returns the top related docs with similarity scores.

    Use this for 'what else have I written about X?' / 'find related
    notes' / 'see also' style questions. ``path`` can be a virtual
    path (transcript://, voice://, canvas://, oura://, etc.) or a
    filesystem path; substring match is used when not exact."""
    from .backlinks import get_backlinks_for_path

    cfg, conn, _, _ = _get_state()
    rows = get_backlinks_for_path(conn, path, limit=limit)
    if not rows:
        # Substring lookup — paths are long.
        candidate = conn.execute(
            "SELECT path FROM files WHERE path LIKE ? "
            "ORDER BY indexed_at DESC LIMIT 1",
            (f"%{path}%",),
        ).fetchone()
        if candidate is not None:
            rows = get_backlinks_for_path(
                conn, candidate["path"], limit=limit,
            )
    if not rows:
        return f"No backlinks found for {path!r}. Run `secondbrain links rebuild` if this is a fresh brain."
    lines = [f"Related to {path!r}:"]
    for r in rows:
        lines.append(f"  [{r.percent}%] {r.title}  ({r.path})")
    return "\n".join(lines)


@mcp.tool()
def health_summary(days: int = 14) -> str:
    """Trailing summary of every health metric the brain has data for
    (Phase 56 — Oura ring). Rolling N-day average with min/max + the
    latest value.

    Use this for 'how's my sleep been?' / 'health check' / 'am I
    sleeping enough this week?' style questions."""
    from . import health as health_mod

    cfg, conn, _, _ = _get_state()
    metrics = health_mod.list_metrics(conn)
    if not metrics:
        return "No Oura data yet. Run `secondbrain sync oura` first."
    lines = [f"Health summary — last {days} days:"]
    for m in metrics:
        s = health_mod.summarise(conn, m, days=days)
        if s.n == 0:
            continue
        lines.append("  " + health_mod.format_summary_line(s))
    return "\n".join(lines)


@mcp.tool()
def health_metric(metric: str, days: int = 14) -> str:
    """Day-by-day values for a single metric (sleep_score, activity_score,
    readiness_score, steps, avg_hrv, etc.). Useful when the user asks
    about a specific trend or wants raw numbers."""
    from . import health as health_mod

    cfg, conn, _, _ = _get_state()
    points = health_mod.recent(conn, metric, days=days)
    if not points:
        return f"No data for {metric!r} in the last {days} days."
    lines = [f"{metric} — last {len(points)} day(s):"]
    for p in points:
        lines.append(f"  {p.date}: {p.value:g}")
    return "\n".join(lines)


@mcp.tool()
def find_person(query: str, limit: int = 5) -> str:
    """Phase 65 — find a person by name or email substring. Returns
    profile snapshots including last-seen, mention count, recent
    docs, and aliases. Use this when the user asks 'who is Sarah?',
    'when did I last meet with Prof. Garcia?', or 'show me everyone
    I've met from Anthropic'."""
    from . import people as people_mod

    cfg, conn, _, _ = _get_read_state()
    rows = people_mod.search_people(conn, query, limit=limit)
    if not rows:
        return f"No people match {query!r}."
    lines = [f"Matches for {query!r}:"]
    for p in rows:
        days = max(0, int((time.time() - p.last_seen_at) // 86400))
        bits = [f"#{p.id} {p.display_name}"]
        if p.email:
            bits.append(p.email)
        if p.role:
            bits.append(p.role)
        if p.company:
            bits.append(p.company)
        bits.append(f"{p.mention_count} mentions, last seen {days}d ago")
        lines.append("  " + " · ".join(bits))
    return "\n".join(lines)


@mcp.tool()
def person_profile(name: str) -> str:
    """Phase 65 — full profile + recent mention timeline for one person.
    Pass the canonical name, an alias, or any name that resolves
    uniquely; ambiguous queries return a list to disambiguate."""
    from . import people as people_mod

    cfg, conn, _, _ = _get_read_state()
    p = people_mod.find_by_alias(conn, name)
    if p is None:
        rows = people_mod.search_people(conn, name, limit=5)
        if not rows:
            return f"No person matches {name!r}."
        if len(rows) > 1:
            return (
                "Multiple matches; specify more precisely:\n"
                + "\n".join(f"  #{r.id} {r.display_name}" for r in rows)
            )
        p = rows[0]
    profile = people_mod.profile_for(conn, p.id)
    lines = [f"# {profile.person.display_name}"]
    if profile.person.email:
        lines.append(f"Email: {profile.person.email}")
    if profile.person.role:
        lines.append(f"Role: {profile.person.role}")
    if profile.person.company:
        lines.append(f"Company: {profile.person.company}")
    lines.append(
        f"Mentions: {profile.person.mention_count} "
        f"(first seen {profile.days_since_first_seen}d ago, "
        f"last {profile.days_since_seen}d ago)",
    )
    if profile.aliases:
        lines.append(f"Aliases: {', '.join(profile.aliases)}")
    if profile.person.notes:
        lines.append("")
        lines.append(f"Notes: {profile.person.notes}")
    if profile.recent_mentions:
        lines.append("")
        lines.append("## Recent mentions")
        for m in profile.recent_mentions[:10]:
            when = time.strftime(
                "%Y-%m-%d", time.localtime(m.mtime),
            )
            lines.append(f"  [{when}] {m.file_path}")
            lines.append(f"    {m.chunk_text_preview[:160]}")
    return "\n".join(lines)


@mcp.tool()
def study_status(course: str = "") -> str:
    """Phase 67 — study card status: total cards, due now, weak
    concepts. Pass a course code (e.g. 'BME410') to scope, or omit
    for an overall view."""
    from . import study as study_mod

    cfg, conn, _, _ = _get_read_state()
    code = course.upper().replace(" ", "").replace("-", "") if course else None
    if code:
        cards = study_mod.cards_for_course(conn, code)
    else:
        rows = conn.execute("SELECT * FROM study_cards").fetchall()
        cards = [study_mod._row_to_card(r) for r in rows]
    if not cards:
        return (
            "No study cards yet. Run `secondbrain study generate` to "
            "materialise from your class transcripts."
        )
    n_due = len(study_mod.due_cards(conn, course_code=code, limit=1000))
    n_reviewed = sum(1 for c in cards if c.review_count > 0)
    lines = [
        f"Cards: {len(cards)} total, {n_due} due now, "
        f"{n_reviewed} reviewed at least once",
    ]
    weak = study_mod.weak_concepts(conn, course_code=code, limit=5)
    if weak:
        lines.append("")
        lines.append("Weak concepts (<3 reviews → not shown):")
        for concept, acc, n in weak:
            lines.append(f"  {acc:.0%}  {concept} ({n} reviews)")
    return "\n".join(lines)


@mcp.tool()
def list_knowledge_gaps(limit: int = 10) -> str:
    """Phase 68 — questions you asked the brain that came back
    weak. These are good study targets / things to learn next."""
    from . import study as study_mod

    cfg, conn, _, _ = _get_read_state()
    rows = study_mod.list_gaps(conn, limit=limit)
    if not rows:
        return "No knowledge gaps logged."
    lines = [f"Open knowledge gaps ({len(rows)}):"]
    for g in rows:
        when = time.strftime("%Y-%m-%d", time.localtime(g.asked_at))
        lines.append(f"  [{when}] #{g.id} {g.question[:100]}")
    return "\n".join(lines)


@mcp.tool()
def remember_fact(key: str, content: str, kind: str = "fact") -> str:
    """Phase 86 — persist a fact / preference / context across chat
    conversations. ``key`` is a short topic anchor (e.g. 'voyage-key'
    or 'meeting-style'). ``kind`` is 'fact' / 'preference' / 'context'.

    Use this when the user says 'remember that ...', 'always do X for
    me', or shares a stable preference you should carry to future
    conversations. The next session's chat will see the fact in the
    system prompt automatically."""
    from .memory import remember

    cfg, conn, _, _ = _get_state()
    try:
        mid = remember(
            conn, key=key, content=content, kind=kind, confidence=0.9,
        )
    except ValueError as e:
        return f"Couldn't remember: {e}"
    return f"Remembered #{mid}: '{key}' = {content}"


@mcp.tool()
def recall_memories(query: str, limit: int = 5) -> str:
    """Phase 86 — search across persisted cross-conversation memories.
    Returns the top-K most relevant by token overlap. Memory content
    goes through Phase 88 redaction in case the user's older chats
    pasted secrets into the recall buffer."""
    from .memory import most_relevant_memories

    cfg, conn, _, _ = _get_read_state()
    rows = most_relevant_memories(conn, query, k=limit)
    if not rows:
        return f"No memories match {query!r}."
    lines = [f"Memories matching {query!r}:"]
    for m in rows:
        lines.append(f"  · [{m.kind}] {_safe(m.key)}: {_safe(m.content)}")
    return "\n".join(lines)


@mcp.tool()
def list_habits() -> str:
    """Phase 79 — list active habits with current streak + 30-day
    adherence. Use to answer 'how am I doing on my habits?' style
    questions."""
    from . import personal

    cfg, conn, _, _ = _get_read_state()
    habits = personal.list_habits(conn)
    if not habits:
        return "No habits configured."
    lines = ["Habits:"]
    for h in habits:
        s = personal.habit_status(conn, h.id)
        adh = (
            f"{s.checkins_last_30d}/{s.expected_30d}"
            if s.expected_30d else f"{s.checkins_last_30d}"
        )
        lines.append(
            f"  · #{h.id} {h.name} ({h.cadence}) — "
            f"{s.current_streak_days}d streak, {adh} this month",
        )
    return "\n".join(lines)


@mcp.tool()
def list_goals() -> str:
    """Phase 79 — active goals with this-week progress + on-track flag."""
    from . import personal

    cfg, conn, _, _ = _get_read_state()
    goals = personal.list_goals(conn)
    if not goals:
        return "No goals configured."
    lines = ["Goals (this week):"]
    for g in goals:
        s = personal.goal_status(conn, g.id)
        if g.target_per_week:
            track = "✓" if s.on_track else "·"
            lines.append(
                f"  [{track}] #{g.id} {g.name} — "
                f"{s.progress_this_week}/{g.target_per_week}",
            )
        else:
            lines.append(
                f"  · #{g.id} {g.name} — {s.progress_this_week} "
                "(no weekly target)",
            )
    return "\n".join(lines)


@mcp.tool()
def add_journal(text: str = "", mood: int = 0) -> str:
    """Phase 80 — add or update today's journal entry. ``mood`` 1-5
    (0 = leave unchanged), ``text`` free-form. Use when the user
    says 'log mood 4', 'journal: had a great day', or similar."""
    from . import personal

    cfg, conn, _, _ = _get_state()
    eid = personal.upsert_journal(
        conn, mood=mood if mood > 0 else None, text=text,
    )
    return f"Journal entry #{eid} saved."


@mcp.tool()
def list_projects() -> str:
    """Phase 81 — explicit project tracker. List active projects."""
    from . import personal

    cfg, conn, _, _ = _get_read_state()
    projects = personal.list_projects(conn)
    if not projects:
        return "No projects configured."
    lines = ["Projects:"]
    for p in projects:
        lines.append(f"  · #{p.id} {p.slug} ({p.status}) — {p.name}")
    return "\n".join(lines)


@mcp.tool()
def project_overview(slug: str) -> str:
    """Phase 81 — full view of a project: tagged files, tasks, people."""
    from . import personal

    cfg, conn, _, _ = _get_read_state()
    p = personal.get_project_by_slug(conn, slug)
    if p is None:
        return f"No project '{slug}'."
    view = personal.project_view(conn, p.id)
    lines = [f"# {view.project.name} ({view.project.slug})"]
    if view.project.description:
        lines.append(view.project.description)
    if view.files:
        lines.append(f"\nFiles ({len(view.files)}):")
        for _fid, path in view.files[:20]:
            lines.append(f"  · {path}")
    if view.tasks:
        lines.append(f"\nTasks ({len(view.tasks)}):")
        for tid, text in view.tasks[:20]:
            lines.append(f"  · #{tid} {text}")
    if view.people:
        lines.append(f"\nPeople ({len(view.people)}):")
        for _pid, name in view.people:
            lines.append(f"  · {name}")
    return "\n".join(lines)


@mcp.tool()
def list_email_drafts() -> str:
    """Phase 83 — pending email drafts awaiting your review."""
    from . import email_assist

    cfg, conn, _, _ = _get_read_state()
    drafts = email_assist.list_unsent_drafts(conn)
    if not drafts:
        return "No pending drafts."
    lines = [f"Pending drafts ({len(drafts)}):"]
    for d in drafts:
        when = time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(d.generated_at),
        )
        preview = _safe(d.draft_text[:160].replace("\n", " "))
        lines.append(f"  · #{d.id} [{when}] {preview}…")
    return "\n".join(lines)


@mcp.tool()
def find_related_via_citations(path: str, direction: str = "outgoing") -> str:
    """Phase 85 — citation graph navigation. ``direction`` is
    'outgoing' (what THIS doc cites) or 'incoming' (what cites this).

    Use to answer 'what did this paper cite?' or 'who cites this
    paper?' research questions."""
    from . import pdf_annotations as pa

    cfg, conn, _, _ = _get_read_state()
    row = conn.execute(
        "SELECT id FROM files WHERE path = ?", (path,),
    ).fetchone()
    if row is None:
        return f"Doc not in index: {path}"
    fid = int(row["id"])
    if direction == "incoming":
        cites = pa.get_citations_to(conn, fid)
        if not cites:
            return f"No incoming citations to {path}."
        lines = [f"Citations TO {path}:"]
        for c in cites:
            year = f" ({c.year})" if c.year else ""
            lines.append(f"  · {c.cited_text}{year}")
    else:
        cites = pa.get_citations_from(conn, fid)
        if not cites:
            return f"No outgoing citations from {path}."
        lines = [f"Citations FROM {path}:"]
        for c in cites:
            resolved = " → " if c.cited_file_id else " (unresolved)"
            year = f" ({c.year})" if c.year else ""
            lines.append(f"  · {c.cited_text}{year}{resolved}")
    return "\n".join(lines)


# =============================================================
# Phase 73 / 74 / 75 / 87 / 89 — chat-agent-facing tools that
# surface the recent synthesis features to MCP. Without these,
# Claude couldn't reach summaries, insights, snapshots, or the
# local-LLM health indicator from a conversation.
# =============================================================

@mcp.tool()
def get_summary(path: str) -> str:
    """Phase 74 — TL;DR + key points for a file. Returns the auto-
    generated summary, falling back to '(no summary yet)' if the
    summariser hasn't reached this doc.

    Use when the user asks 'what's in this file?' / 'summarise
    <path>' before you decide whether to call ``get_file`` for the
    full text."""
    from . import synthesis

    cfg, conn, _, _ = _get_read_state()
    row = conn.execute(
        "SELECT id FROM files WHERE path = ?", (path,),
    ).fetchone()
    if row is None:
        return f"Doc not in index: {path}"
    summary = synthesis.get_summary(conn, int(row["id"]))
    if summary is None:
        return f"(no summary yet for {path})"
    lines = [f"# Summary of {path}", "", _safe(summary.tldr), ""]
    if summary.key_points:
        lines.append("## Key points")
        for kp in summary.key_points:
            lines.append(f"- {_safe(kp)}")
    return "\n".join(lines)


@mcp.tool()
def list_insights(limit: int = 10) -> str:
    """Phase 75 — proactive 'I noticed X' insights from your data
    (topic spikes, health drift). Already-shown insights are
    deduped for 7 days so this won't re-surface yesterday's.

    Use when the user asks 'what's new?' / 'anything I should
    know?' / 'what have you noticed lately?'."""
    from . import synthesis

    cfg, conn, _, _ = _get_read_state()
    insights = synthesis.detect_insights(conn)
    if not insights:
        return "No fresh insights right now."
    lines = [f"Insights ({len(insights[:limit])}):"]
    for i in insights[:limit]:
        lines.append(f"  · **{_safe(i.headline)}**")
        if i.detail:
            lines.append(f"    {_safe(i.detail)}")
    return "\n".join(lines)


@mcp.tool()
def list_snapshots(limit: int = 10) -> str:
    """Phase 87 — list weekly index snapshots so the agent can pick
    one to use with ``as_of_search``. Each snapshot captures the
    set of files that existed at a specific point in time.

    Use when the user asks 'what did I know two weeks ago?' / 'as
    of <date>' / 'compare to last month' style queries."""
    from . import memory as memory_mod

    cfg, conn, _, _ = _get_read_state()
    snaps = memory_mod.list_snapshots(conn, limit=limit)
    if not snaps:
        return (
            "No snapshots taken yet. The daemon takes one weekly "
            "automatically; run `secondbrain snapshot take` to "
            "force one now."
        )
    lines = [f"Snapshots ({len(snaps)}):"]
    for s in snaps:
        when = time.strftime("%Y-%m-%d", time.localtime(s.taken_at))
        label = f" [{s.label}]" if s.label else ""
        lines.append(
            f"  · #{s.id}  {when}{label}  ({s.n_files} files)",
        )
    return "\n".join(lines)


@mcp.tool()
def as_of_search(query: str, when: str, k: int = 8) -> str:
    """Phase 87 — search the brain as it existed at a point in time.
    ``when`` accepts 'YYYY-MM-DD' or relative phrases ('yesterday',
    'last week', 'N {days,weeks,months,years} ago'). Filters results
    to files in the closest preceding weekly snapshot.

    Use when the user asks 'what did I know about X back then?'."""
    from .cli import _parse_as_of
    from .search import hybrid_search

    cfg, conn, embedder, reranker = _get_state()
    ts = _parse_as_of(when)
    if ts is None:
        return f"Couldn't parse 'when' = {when!r}. Try 'YYYY-MM-DD' or 'N days ago'."
    results = hybrid_search(
        conn, embedder, query, k=k, reranker=reranker, as_of_ts=ts,
    )
    when_str = time.strftime("%Y-%m-%d", time.localtime(ts))
    return _format_results(
        results, f"Search for {query!r} as of {when_str}",
    )


@mcp.tool()
def local_llm_status() -> str:
    """Phase 89 — report whether the local Ollama fallback is
    reachable and which models are pulled. Useful diagnostic when
    the user asks 'is local LLM working?' or you've just fallen
    back to it and want to confirm.
    """
    from . import local_llm

    cfg, _, _, _ = _get_read_state()
    if not local_llm.is_available(cfg):
        host = getattr(cfg, "local_llm_host", "http://localhost:11434")
        return (
            f"Local LLM (Ollama) is NOT reachable at {host}. "
            "Install + start Ollama, then `ollama pull llama3.1`."
        )
    models = local_llm.list_models(cfg)
    if not models:
        return (
            "Ollama is reachable but no models are pulled. Run "
            "`ollama pull llama3.1` to enable the fallback."
        )
    default = getattr(cfg, "local_llm_model", "llama3.1")
    return (
        f"Local LLM ready. Default model: {default}. "
        f"Available: {', '.join(models)}"
    )


@mcp.tool()
def list_ai_actions(
    limit: int = 20, kind: str = "",
) -> str:
    """Round 10 (#6) — list the most-recent AI actions logged in
    ``ai_actions``. Use to answer 'what has the assistant done
    recently?' / 'why was that draft generated?' style questions.

    ``kind`` filters by action type ('draft', 'analyze',
    'thanks_draft', 'voice_critique', 'extract_promise', etc).
    Empty string returns all kinds."""
    from . import ai_audit

    cfg, conn, _, _ = _get_read_state()
    rows = ai_audit.recent(
        conn, limit=limit,
        kind=(kind or None) if kind else None,
    )
    if not rows:
        return "No AI actions logged yet."
    lines = [f"Recent AI actions ({len(rows)}):"]
    for r in rows:
        when = time.strftime("%m-%d %H:%M", time.localtime(r.ts))
        cost = (
            f" ${r.cents / 100:.4f}" if r.cents > 0 else ""
        )
        err = f" ERROR: {r.error}" if r.error else ""
        lines.append(
            f"  · {when} [{r.kind}] {r.status} ({r.model}){cost}"
            f" — {_safe(r.summary)}{err}",
        )
    return "\n".join(lines)


# ============================ Round 19 — EA MCP tools ==============


@mcp.tool()
def list_followups(direction: str = "", limit: int = 30) -> str:
    """Round 19 (Phase EA-1) — list open follow-ups.

    direction: "outgoing" (you owe), "incoming" (others owe you),
               or "" for both.
    Returns a Markdown summary, redacted for safety.
    """
    from . import followups
    _, conn, _, _ = _get_state()
    rows = followups.list_open(
        conn,
        direction=direction or None,
        limit=max(1, min(int(limit or 30), 200)),
    )
    if not rows:
        return "_No open follow-ups._"
    lines = ["# Open follow-ups", ""]
    out_rows = [r for r in rows if r.direction == "outgoing"]
    in_rows = [r for r in rows if r.direction == "incoming"]
    if out_rows and (not direction or direction == "outgoing"):
        lines.append(f"## You owe ({len(out_rows)})")
        for r in out_rows:
            age_days = (
                int((time.time() - r.promised_at) / 86400.0)
                if r.promised_at else None
            )
            age = f" · {age_days}d" if age_days is not None else ""
            who = f" → {_safe(r.person_name)}" if r.person_name else ""
            lines.append(
                f"- **{_safe(r.topic)}**{who}{age}"
            )
        lines.append("")
    if in_rows and (not direction or direction == "incoming"):
        lines.append(f"## Owed to you ({len(in_rows)})")
        for r in in_rows:
            age_days = (
                int((time.time() - r.promised_at) / 86400.0)
                if r.promised_at else None
            )
            age = f" · {age_days}d" if age_days is not None else ""
            who = f" ← {_safe(r.person_name)}" if r.person_name else ""
            lines.append(f"- **{_safe(r.topic)}**{who}{age}")
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def add_followup(
    direction: str,
    topic: str,
    description: str,
    person_name: str = "",
    due_iso_date: str = "",
) -> str:
    """Manually add a follow-up. direction must be 'outgoing' or
    'incoming'. due_iso_date is "YYYY-MM-DD" or "" for none."""
    from . import followups
    from . import people as people_mod
    _, conn, _, _ = _get_state()
    person_id: int | None = None
    if person_name:
        try:
            p = people_mod.find_person_by_name(conn, person_name)
            if p:
                person_id = int(p.id)
        except Exception:
            pass
    due_at: float | None = None
    if due_iso_date:
        try:
            from datetime import date as _date
            from datetime import datetime as _dt
            d = _date.fromisoformat(due_iso_date)
            due_at = _dt(d.year, d.month, d.day).timestamp()
        except (ValueError, TypeError):
            return f"_Bad due_iso_date: {due_iso_date!r}_"
    new_id = followups.add_followup(
        conn,
        direction=direction,
        topic=topic,
        description=description,
        person_id=person_id,
        person_name=person_name,
        source_kind="manual",
        due_at=due_at,
        promised_at=time.time(),
        confidence=1.0,
        extracted_by="manual",
    )
    return (
        f"OK: added follow-up #{new_id}"
        if new_id else "_Failed to add follow-up._"
    )


@mcp.tool()
def resolve_followup(followup_id: int) -> str:
    """Mark a follow-up as resolved (no longer pending)."""
    from . import followups
    _, conn, _, _ = _get_state()
    if followups.mark_resolved(conn, followup_id):
        return f"OK: follow-up #{followup_id} resolved."
    return f"_Follow-up #{followup_id} not found or already resolved._"


@mcp.tool()
def build_agenda(person_id: int = 0, person_name: str = "") -> str:
    """Round 19 (Phase EA-2) — build a 1:1 agenda.

    Pass either person_id (preferred) or person_name (resolved via
    alias / canonical_name / display_name LIKE). Returns a Markdown
    pre-meeting card with last meeting, open follow-ups (both
    directions), recent email threads, journal mentions, and shared
    topics.
    """
    from . import agenda
    from . import people as people_mod
    _, conn, _, _ = _get_state()
    if not person_id and person_name:
        p = people_mod.find_person_by_name(conn, person_name)
        if p is None:
            return f"_No person matched {person_name!r}._"
        person_id = int(p.id)
    if not person_id:
        return "_Need person_id or person_name._"
    result = agenda.build_agenda(conn, person_id)
    if result is None:
        return f"_Person #{person_id} not found._"
    return agenda.render_markdown(result)


@mcp.tool()
def capture_meeting(file_id: int, overwrite: bool = False) -> str:
    """Round 19 (Phase EA-3) — extract decisions / action items /
    open questions / recap draft from a meeting transcript.

    The transcript must already be indexed in the brain (file_id
    points at a kind='audio_video' or kind='transcript' file).
    Action items flow into the followups tracker. Returns the
    capture as Markdown, including the recap draft you can copy/edit.
    """
    from . import meeting_capture
    cfg, conn, _, _ = _get_state()
    cap = meeting_capture.capture(
        conn, cfg, file_id, overwrite=overwrite,
    )
    if cap is None:
        return f"_Capture failed for file #{file_id}._"
    lines = [f"# Meeting capture: {_safe(cap.title)}", ""]
    if cap.decisions:
        lines.append("## Decisions")
        for d in cap.decisions:
            lines.append(f"- **{_safe(d.text)}**")
            if d.rationale:
                lines.append(f"  - {_safe(d.rationale)}")
        lines.append("")
    if cap.actions:
        lines.append("## Action items")
        for a in cap.actions:
            due = f" · _due: {a.due_hint}_" if a.due_hint else ""
            lines.append(
                f"- **{_safe(a.owner)}** — {_safe(a.description)}{due}",
            )
        lines.append("")
    if cap.open_questions:
        lines.append("## Open questions")
        for q in cap.open_questions:
            lines.append(f"- {_safe(q)}")
        lines.append("")
    if cap.recap_draft:
        lines.append("## Recap draft")
        lines.append(_safe(cap.recap_draft))
    return "\n".join(lines)


@mcp.tool()
def list_meeting_captures(limit: int = 10) -> str:
    """List recent meeting captures."""
    from . import meeting_capture
    _, conn, _, _ = _get_state()
    rows = meeting_capture.list_recent(
        conn, limit=max(1, min(int(limit), 50)),
    )
    if not rows:
        return "_No captures yet._"
    lines = ["| When | Title | Decisions | Actions |",
             "|------|-------|-----------|---------|"]
    for r in rows:
        when = time.strftime("%m-%d %H:%M", time.localtime(r.captured_at))
        lines.append(
            f"| {when} | {_safe(r.title)[:40]} | "
            f"{len(r.decisions)} | {len(r.actions)} |"
        )
    return "\n".join(lines)


@mcp.tool()
def list_overdue_contacts(tier: str = "vip", limit: int = 10) -> str:
    """Round 19 (Phase EA-5) — list people whose cadence target has
    passed (you should reach out)."""
    from . import people as people_mod
    _, conn, _, _ = _get_state()
    tier_filter = (
        [tier] if tier in ("vip", "regular", "casual") else None
    )
    rows = people_mod.list_overdue_contacts(
        conn,
        limit=max(1, min(int(limit), 50)),
        tier_filter=tier_filter,
    )
    if not rows:
        return "_Everyone is in good standing._"
    lines = ["# Overdue contacts", ""]
    for o in rows:
        lines.append(
            f"- **{_safe(o.person.display_name)}** "
            f"({o.person.tier}) — "
            f"{o.days_since_contact}d since contact "
            f"({o.days_overdue}d past target)"
        )
    return "\n".join(lines)


@mcp.tool()
def set_person_tier(
    person_id: int = 0,
    person_name: str = "",
    tier: str = "regular",
    cadence_days: int = 0,
) -> str:
    """Round 19 (Phase EA-5) — set a person's tier (vip|regular|casual)
    and optional cadence target (days). cadence_days=0 clears.

    Pass person_id (preferred) or person_name."""
    from . import people as people_mod
    _, conn, _, _ = _get_state()
    if not person_id and person_name:
        p = people_mod.find_person_by_name(conn, person_name)
        if p is None:
            return f"_No person matched {person_name!r}._"
        person_id = int(p.id)
    if not person_id:
        return "_Need person_id or person_name._"
    try:
        people_mod.set_field(
            conn, person_id, tier=tier,
            cadence_days=int(cadence_days) if cadence_days else 0,
        )
    except ValueError as e:
        return f"_Invalid: {e}_"
    return f"OK: person #{person_id} → tier={tier} cadence={cadence_days}d"


@mcp.tool()
def gift_ideas_for(
    person_id: int = 0, person_name: str = "",
) -> str:
    """Round 19 (Phase EA-7) — generate (or fetch) 3 gift ideas
    for a person. Cached after first call."""
    from . import gift_ideas
    from . import people as people_mod
    cfg, conn, _, _ = _get_state()
    if not person_id and person_name:
        p = people_mod.find_person_by_name(conn, person_name)
        if p is None:
            return f"_No person matched {person_name!r}._"
        person_id = int(p.id)
    if not person_id:
        return "_Need person_id or person_name._"
    ideas = gift_ideas.generate_for_person(conn, cfg, person_id)
    if ideas is None or not ideas.ideas:
        return "_Could not generate ideas (no API key, no profile data, or budget exceeded)._"
    p = people_mod.get_person(conn, person_id)
    name = p.display_name if p else f"#{person_id}"
    lines = [f"# Gift ideas for {_safe(name)}", ""]
    for i in ideas.ideas:
        price = f" ({i.price_range})" if i.price_range else ""
        lines.append(f"## {_safe(i.title)}{price}")
        lines.append(_safe(i.description))
        if i.why:
            lines.append(f"\n_Why: {_safe(i.why)}_")
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def morning_triage(hours: int = 48, limit: int = 10) -> str:
    """Round 19 (Phase EA-6) — ranked list of emails that need your
    decision today. Order: VIP × urgency × age."""
    from . import triage_queue
    _, conn, _, _ = _get_state()
    queue = triage_queue.build_queue(
        conn, hours=hours, max_items=max(1, min(int(limit), 30)),
    )
    if not queue:
        return "_Inbox zero — nothing in window needs your decision._"
    lines = ["# Morning triage", ""]
    for it in queue:
        vip = " · VIP" if it.is_vip else ""
        draft = (
            f" · _draft #{it.draft_id} ready_"
            if it.has_draft else ""
        )
        lines.append(
            f"- **[{it.label}]** {_safe(it.from_display)}: "
            f"{_safe(it.subject) or '(no subject)'} "
            f"({int(it.age_hours)}h{vip}{draft})"
        )
    return "\n".join(lines)


@mcp.tool()
def end_of_day() -> str:
    """Round 19 (Phase EA-9) — end-of-day wrap-up: today's metrics,
    items slipping past a week, tomorrow's calendar."""
    from . import eod_wrapup
    _, conn, _, _ = _get_state()
    return eod_wrapup.render_markdown(eod_wrapup.build_wrapup(conn))


@mcp.tool()
def remind_if(
    description: str,
    condition_kind: str,
    fire_after_iso: str = "",
    email_from: str = "",
    followup_id: int = 0,
) -> str:
    """Round 19 (Phase EA-10) — create a conditional reminder.

    condition_kind:
      - "date_passed": fires at fire_after_iso (YYYY-MM-DD or
        YYYY-MM-DDTHH:MM:SS).
      - "no_reply_from": fires at fire_after_iso if no email from
        email_from has arrived since the reminder was created.
      - "followup_unresolved": fires at fire_after_iso if followup_id
        is still status='open'.
    """
    from . import conditional_reminders
    _, conn, _, _ = _get_state()
    fire_after: float | None = None
    if fire_after_iso:
        try:
            from datetime import datetime as _dt
            fire_after = _dt.fromisoformat(fire_after_iso).timestamp()
        except ValueError:
            return f"_Bad fire_after_iso: {fire_after_iso!r}_"
    cond: dict = {}
    if condition_kind == "no_reply_from":
        if not email_from:
            return "_no_reply_from needs email_from._"
        cond = {"email": email_from, "since_ts": time.time()}
    elif condition_kind == "followup_unresolved":
        if not followup_id:
            return "_followup_unresolved needs followup_id._"
        cond = {"followup_id": int(followup_id)}
    elif condition_kind == "date_passed":
        if fire_after is None:
            return "_date_passed needs fire_after_iso._"
    else:
        return (
            f"_Unsupported condition_kind {condition_kind!r}; "
            f"try date_passed, no_reply_from, followup_unresolved._"
        )
    try:
        rid = conditional_reminders.add_reminder(
            conn,
            description=description,
            condition_kind=condition_kind,
            condition=cond,
            fire_after=fire_after,
        )
    except ValueError as e:
        return f"_{e}_"
    return f"OK: reminder #{rid} scheduled."


@mcp.tool()
def find_open_time_slots(
    days_ahead: int = 7,
    duration_minutes: int = 30,
    earliest_hour: int = 9,
    latest_hour: int = 17,
    busy_events_json: str = "[]",
) -> str:
    """Round 19 (Phase EA-4) — find candidate meeting slots.

    The CALLER (typically Claude itself via the Google Calendar MCP)
    fetches the busy events and passes them as JSON via
    ``busy_events_json``. Each event must have ``start`` and ``end``
    with ``dateTime`` (ISO) keys, matching the Google Calendar
    list_events shape. Returns ranked candidate slots in human-
    readable Markdown.
    """
    import json as _json
    from datetime import date as _date
    from datetime import timedelta as _td

    from . import scheduling
    try:
        events = _json.loads(busy_events_json or "[]")
    except _json.JSONDecodeError as e:
        return f"_Bad busy_events_json: {e}_"
    busy = scheduling.parse_busy_blocks(events)
    today = _date.today()
    slots = scheduling.find_open_slots(
        busy,
        window_start=today,
        window_end=today + _td(days=max(1, int(days_ahead))),
        prefs=scheduling.SchedulingPrefs(
            duration_minutes=int(duration_minutes),
            earliest_hour=int(earliest_hour),
            latest_hour=int(latest_hour),
        ),
    )
    if not slots:
        return "_No open slots in window._"
    lines = ["# Open slots", ""]
    for s in slots:
        dow = s.start.strftime("%A")
        date_str = s.start.strftime("%b %d").replace(" 0", " ")
        time_range = (
            f"{scheduling._fmt_time(s.start)}–"
            f"{scheduling._fmt_time(s.end)}"
        )
        lines.append(f"- {dow} {date_str} · {time_range}")
    return "\n".join(lines)


def run() -> None:
    """Run the MCP server over stdio. Used by `secondbrain serve`."""
    _get_state()  # warm caches before we start serving
    mcp.run()


if __name__ == "__main__":
    run()
