"""MCP server exposing the second-brain to AI assistants."""

from __future__ import annotations

import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .config import Config, load_config
from .db import connect, init_schema, stats
from .embedder import make_embedder
from .search import hybrid_search, keyword_only, vector_only

log = logging.getLogger(__name__)

mcp = FastMCP("second-brain")

# Module-level singletons; lazily initialised so importing this file is cheap.
_cfg: Config | None = None
_conn = None
_embedder = None


def _get_state():
    global _cfg, _conn, _embedder
    if _conn is None:
        _cfg = load_config()
        _embedder = make_embedder(_cfg)
        _conn = connect(_cfg.db_path)
        init_schema(_conn, _embedder.dim, _embedder.name)
    return _cfg, _conn, _embedder


def _format_results(results, header: str) -> str:
    if not results:
        return f"{header}\n\n(no matches)"
    lines = [header, ""]
    for i, r in enumerate(results, 1):
        sources = "+".join(r.sources)
        lines.append(f"### {i}. {r.file_path} (chunk {r.chunk_index}, via {sources}, score={r.score:.4f})")
        snippet = r.text if len(r.text) <= 1200 else r.text[:1200] + "..."
        lines.append(snippet)
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
def search_brain(query: str, k: int = 10) -> str:
    """Hybrid search across your indexed files (vector + keyword, fused).

    Returns matched text chunks with file paths so you can cite or open them.
    Best for most questions.
    """
    cfg, conn, embedder = _get_state()
    results = hybrid_search(conn, embedder, query, k=k, alpha=cfg.hybrid_alpha)
    return _format_results(results, f"# Hybrid search: {query!r}")


@mcp.tool()
def vector_search(query: str, k: int = 10) -> str:
    """Pure semantic (vector) search. Best for conceptual questions where exact wording differs."""
    _, conn, embedder = _get_state()
    results = vector_only(conn, embedder, query, k=k)
    return _format_results(results, f"# Vector search: {query!r}")


@mcp.tool()
def keyword_search(query: str, k: int = 10) -> str:
    """Pure BM25 keyword search. Best for proper nouns, IDs, exact strings."""
    _, conn, embedder = _get_state()
    results = keyword_only(conn, embedder, query, k=k)
    return _format_results(results, f"# Keyword search: {query!r}")


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
def get_recent(n: int = 20) -> str:
    """List the most recently modified files in the index."""
    _, conn, _ = _get_state()
    rows = conn.execute(
        "SELECT path, mtime, kind FROM files ORDER BY mtime DESC LIMIT ?", (n,)
    ).fetchall()
    if not rows:
        return "(index is empty)"
    return "\n".join(f"[{r['kind']}] {r['path']}" for r in rows)


@mcp.tool()
def index_status() -> str:
    """Report what's in the index: file count, chunk count, embedder, last update."""
    _, conn, _ = _get_state()
    s = stats(conn)
    last = s["last_indexed_at"]
    last_str = "never" if last is None else f"{last:.0f} (epoch)"
    return (
        f"Files: {s['files']}\n"
        f"Chunks: {s['chunks']}\n"
        f"Embedder: {s['embedder']} (dim={s['embedding_dim']})\n"
        f"Last indexed: {last_str}"
    )


def run() -> None:
    """Run the MCP server over stdio. Used by `secondbrain serve`."""
    _get_state()  # warm caches before we start serving
    mcp.run()


if __name__ == "__main__":
    run()
