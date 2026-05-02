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


def _format_results(results, header: str) -> str:
    if not results:
        return f"{header}\n\n(no matches)"
    lines = [header, ""]
    for i, r in enumerate(results, 1):
        sources = "+".join(r.sources)
        tag = "reranked" if r.reranked else sources
        lines.append(f"### {i}. {r.file_path} (chunk {r.chunk_index}, via {tag}, score={r.score:.4f})")
        snippet = r.text if len(r.text) <= 1200 else r.text[:1200] + "..."
        lines.append(snippet)
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
    """Return the full text contents of a file by path."""
    p = Path(path)
    if not p.exists():
        return f"File not found: {path}"
    if not p.is_file():
        return f"Not a file: {path}"
    try:
        return p.read_text(encoding="utf-8", errors="replace")
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
        lines.append(snippet)
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
            f"  (from: {t.source_title})"
            if t.source_path != "manual" else ""
        )
        age_label = f" [{age}d]" if age > 0 else ""
        lines.append(f"  #{t.id}{age_label}  {t.text}{suffix}")
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
        lines.append(f"  {marker} #{t.id}  {t.text}")
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


def run() -> None:
    """Run the MCP server over stdio. Used by `secondbrain serve`."""
    _get_state()  # warm caches before we start serving
    mcp.run()


if __name__ == "__main__":
    run()
