"""MCP server exposing the second-brain to AI assistants."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .config import Config, load_config
from .db import connect, init_schema, search_images, stats
from .embedder import make_embedder
from .image_embedder import make_image_embedder
from .reranker import make_reranker
from .search import hybrid_search, keyword_only, vector_only

log = logging.getLogger(__name__)

mcp = FastMCP("second-brain")

# Module-level singletons; lazily initialised so importing this file is cheap.
_cfg: Config | None = None
_conn = None
_embedder = None
_reranker = None


def _get_state():
    global _cfg, _conn, _embedder, _reranker
    if _conn is None:
        _cfg = load_config()
        _embedder = make_embedder(_cfg)
        _reranker = make_reranker(_cfg)
        _conn = connect(_cfg.db_path)
        init_schema(_conn, _embedder.dim, _embedder.name)
    return _cfg, _conn, _embedder, _reranker


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
        if pattern.search(t):
            matches.add(t)
        elif re.search(r"\b" + re.escape(t) + r"\b", q_lower):
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
    cfg, conn, embedder, reranker = _get_state()
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
    cfg, conn, embedder, reranker = _get_state()
    results = vector_only(
        conn, embedder, query, k=k,
        reranker=reranker, rerank_overfetch=cfg.rerank_overfetch,
    )
    return _format_results(results, f"# Vector search: {query!r}")


@mcp.tool()
def keyword_search(query: str, k: int = 10) -> str:
    """Pure BM25 keyword search. Best for proper nouns, IDs, exact strings."""
    cfg, conn, embedder, reranker = _get_state()
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


def run() -> None:
    """Run the MCP server over stdio. Used by `secondbrain serve`."""
    _get_state()  # warm caches before we start serving
    mcp.run()


if __name__ == "__main__":
    run()
