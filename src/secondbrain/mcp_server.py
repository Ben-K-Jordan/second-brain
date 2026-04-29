"""MCP server exposing the second-brain to AI assistants."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .config import Config, load_config
from .db import connect, init_schema, stats
from .embedder import make_embedder
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
    )
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
def find_mentions(entity: str, k: int = 10) -> str:
    """Find chunks that mention an entity (case-insensitive exact-text match).

    Returns up to k chunks with file paths so you can read context. Pair with
    `list_entities` to discover what's worth searching.
    """
    _, conn, _, _ = _get_state()
    rows = conn.execute(
        "SELECT c.text, c.chunk_index, f.path, f.mtime, e.label "
        "FROM entities e "
        "JOIN chunks c ON c.id = e.chunk_id "
        "JOIN files f ON f.id = c.file_id "
        "WHERE e.text_lower = ? "
        "ORDER BY f.mtime DESC LIMIT ?",
        (entity.lower(), k),
    ).fetchall()
    if not rows:
        return f"(no mentions of {entity!r})"
    lines = [f"# Mentions of {entity!r}", ""]
    for r in rows:
        snippet = r["text"] if len(r["text"]) <= 600 else r["text"][:600] + "..."
        lines.append(
            f"### [{r['label']}] {r['path']} (chunk {r['chunk_index']})"
        )
        lines.append(snippet)
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def entity_timeline(entity: str, limit: int = 30) -> str:
    """Files that mention an entity, sorted by file mtime (newest first).

    A timeline view: when does this person/org/thing show up in your brain?
    """
    _, conn, _, _ = _get_state()
    rows = conn.execute(
        "SELECT DISTINCT f.path, f.mtime, f.kind "
        "FROM entities e "
        "JOIN chunks c ON c.id = e.chunk_id "
        "JOIN files f ON f.id = c.file_id "
        "WHERE e.text_lower = ? "
        "ORDER BY f.mtime DESC LIMIT ?",
        (entity.lower(), limit),
    ).fetchall()
    if not rows:
        return f"(no mentions of {entity!r})"
    lines = [f"# Timeline of {entity!r}", ""]
    for r in rows:
        age_days = (time.time() - r["mtime"]) / 86400
        lines.append(f"  {age_days:6.1f}d ago  [{r['kind']:12s}]  {r['path']}")
    return "\n".join(lines)


def run() -> None:
    """Run the MCP server over stdio. Used by `secondbrain serve`."""
    _get_state()  # warm caches before we start serving
    mcp.run()


if __name__ == "__main__":
    run()
