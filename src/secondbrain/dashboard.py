"""Local web dashboard for browsing, searching, and managing the second-brain.

Runs at http://127.0.0.1:8765 by default. FastAPI + HTMX + Pico.css. No build
step, no JS framework, no external dependencies beyond the ones MCP already
pulled in (starlette / uvicorn) plus jinja2 for templates.

Pages:
- /            overview: stats, recent files, top entities, recent URLs
- /search      hybrid search with filters
- /entities    full entity browser, filterable by label
- /entity/<n>  per-entity detail: mentions + neighbors + timeline
- /folders     folder tree with file counts
- /file        read-only file viewer
- /ingest      URL ingest form
"""

from __future__ import annotations

import logging
import threading
import time
import urllib.parse
import webbrowser
from html import escape
from pathlib import Path

from .briefing import generate_briefing
from .budget import spend_summary
from .config import load_config
from .db import connect, init_schema, stats
from .embedder import make_embedder
from .entities import make_entity_extractor
from .indexer import index_url
from .reranker import make_reranker
from .search import hybrid_search

log = logging.getLogger(__name__)


# --- HTML rendering helpers (no Jinja dep — embedded templates as f-strings) ---

CSS = """
:root {
    --bg: #0e0e10;
    --surface: #16161a;
    --surface-2: #1e1e24;
    --border: #2a2a32;
    --text: #e6e6ea;
    --text-dim: #9090a0;
    --accent: #4d9fff;
    --accent-soft: #2c5fa0;
    --good: #6ed46e;
    --warn: #e6b86d;
    --bad: #e66d6d;
    --mono: 'SF Mono', 'Cascadia Mono', 'JetBrains Mono', Consolas, monospace;
    --sans: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
* { box-sizing: border-box; }
html, body {
    margin: 0; padding: 0; background: var(--bg); color: var(--text);
    font-family: var(--sans); font-size: 14.5px; line-height: 1.5;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
header {
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    display: flex; align-items: center; gap: 28px;
    position: sticky; top: 0; background: var(--bg); z-index: 10;
}
header .brand { font-weight: 700; letter-spacing: -0.01em; font-size: 16px; }
header nav a { color: var(--text-dim); margin-right: 16px; }
header nav a.active, header nav a:hover { color: var(--text); }
main { padding: 24px; max-width: 1200px; margin: 0 auto; }
h1, h2, h3 { font-weight: 600; letter-spacing: -0.01em; }
h1 { font-size: 24px; margin: 0 0 16px; }
h2 { font-size: 18px; margin: 24px 0 12px; }
h3 { font-size: 15px; margin: 12px 0 6px; }
.grid { display: grid; gap: 24px; grid-template-columns: 1fr 1fr; }
@media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
.card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px;
}
.card h2 { margin-top: 0; }
.stat { display: flex; justify-content: space-between; padding: 4px 0; }
.stat .k { color: var(--text-dim); }
.stat .v { font-family: var(--mono); }
.muted { color: var(--text-dim); }
.path { font-family: var(--mono); font-size: 12.5px; color: var(--text-dim); word-break: break-all; }
.label { display: inline-block; padding: 1px 6px; border-radius: 3px;
    font-size: 11px; background: var(--accent-soft); color: var(--text);
    font-family: var(--mono); margin-right: 4px; }
.label.PERSON { background: #4d6dff; }
.label.ORG    { background: #ff8c4d; }
.label.GPE, .label.LOC, .label.FAC { background: #4dff8c; color: #062b15; }
.label.PRODUCT, .label.WORK_OF_ART { background: #c44dff; }
.label.DATE { background: #4dffe6; color: #06292c; }
.label.MONEY { background: #ffe14d; color: #2c2806; }
.label.LAW, .label.NORP, .label.LANGUAGE, .label.EVENT { background: #ff4d6d; }
.search-box input, .ingest-box input {
    width: 100%; padding: 10px 12px; font-size: 15px;
    background: var(--surface-2); color: var(--text);
    border: 1px solid var(--border); border-radius: 6px;
    font-family: var(--sans);
}
.search-box input:focus, .ingest-box input:focus {
    outline: none; border-color: var(--accent);
}
.filters { display: flex; gap: 8px; margin: 8px 0 16px; flex-wrap: wrap; }
.filters input, .filters select {
    padding: 6px 8px; font-size: 13px;
    background: var(--surface-2); color: var(--text);
    border: 1px solid var(--border); border-radius: 4px;
}
button {
    padding: 8px 14px; background: var(--accent); color: white;
    border: none; border-radius: 6px; font-size: 14px; cursor: pointer;
    font-family: var(--sans);
}
button:hover { background: #3d8fef; }
button.ghost { background: transparent; color: var(--text-dim); border: 1px solid var(--border); }
button.ghost:hover { background: var(--surface-2); color: var(--text); }
.result {
    border-top: 1px solid var(--border); padding: 12px 0; margin: 0;
}
.result h3 { margin: 0 0 4px; font-weight: 500; font-size: 13.5px; }
.result .snippet { white-space: pre-wrap; font-family: var(--mono); font-size: 12.5px;
    color: var(--text); background: var(--surface-2); padding: 10px;
    border-radius: 4px; max-height: 280px; overflow: auto; }
.result .meta { color: var(--text-dim); font-size: 12px; margin: 4px 0; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid var(--border); }
th { color: var(--text-dim); font-weight: 500; font-size: 12px; text-transform: uppercase; }
td.num { font-family: var(--mono); text-align: right; color: var(--text-dim); }
.empty { padding: 40px 20px; text-align: center; color: var(--text-dim); }
.warn { color: var(--warn); }
#cy { width: 100%; height: 80vh; background: var(--surface); border-radius: 8px;
    border: 1px solid var(--border); }
"""


def _layout(title: str, body: str, active: str = "") -> str:
    nav_items = [
        ("Overview", "/"),
        ("Search", "/search"),
        ("Graph", "/graph"),
        ("Entities", "/entities"),
        ("Folders", "/folders"),
        ("Briefing", "/briefing"),
        ("Ingest", "/ingest"),
    ]
    nav = " ".join(
        f'<a href="{href}" class="{"active" if href.split("/")[1] == active else ""}">{escape(name)}</a>'
        for name, href in nav_items
    )
    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{escape(title)} — second-brain</title>
    <script src="https://unpkg.com/htmx.org@1.9.12" crossorigin="anonymous"></script>
    <style>{CSS}</style>
</head>
<body>
    <header>
        <div class="brand">second-brain</div>
        <nav>{nav}</nav>
    </header>
    <main>{body}</main>
</body>
</html>
"""


def _result_block(r) -> str:
    sources = "+".join(r.sources) if r.sources else "rerank"
    age = ""
    if r.mtime is not None:
        days = (time.time() - r.mtime) / 86400
        age = f" · {days:.1f}d ago"
    snippet = escape(r.text if len(r.text) <= 1500 else r.text[:1500] + "…")
    file_link = (
        f'<a href="/file?path={urllib.parse.quote_plus(r.file_path)}">'
        f"{escape(r.file_path)}</a>"
    )
    return f"""
<article class="result">
    <h3>{file_link}</h3>
    <div class="meta">chunk {r.chunk_index} · {sources}{age} · score {r.score:.4f}</div>
    <div class="snippet">{snippet}</div>
</article>"""


def _entity_link(text: str, label: str = "") -> str:
    href = "/entity?name=" + urllib.parse.quote_plus(text)
    label_html = f'<span class="label {label}">{label}</span>' if label else ""
    return f'{label_html}<a href="{href}">{escape(text)}</a>'


# --- App factory ---

def create_app():
    """Build the FastAPI app. Heavy imports happen here so importing dashboard
    is cheap when the user isn't using the dashboard."""
    from fastapi import FastAPI, Form, Query, Request
    from fastapi.responses import HTMLResponse, RedirectResponse

    app = FastAPI(title="second-brain", docs_url=None, redoc_url=None)

    state = {"cfg": None, "conn": None, "embedder": None, "reranker": None}

    def get_state():
        if state["conn"] is None:
            cfg = load_config()
            embedder = make_embedder(cfg)
            conn = connect(cfg.db_path)
            init_schema(conn, embedder.dim, embedder.name)
            state["cfg"] = cfg
            state["embedder"] = embedder
            state["conn"] = conn
            state["reranker"] = make_reranker(cfg)
        return state["cfg"], state["conn"], state["embedder"], state["reranker"]

    # --- Routes ---

    @app.get("/", response_class=HTMLResponse)
    def index():
        cfg, conn, embedder, reranker = get_state()
        s = stats(conn)
        recent_rows = conn.execute(
            "SELECT path, mtime, kind FROM files ORDER BY mtime DESC LIMIT 10"
        ).fetchall()
        ent_rows = conn.execute(
            "SELECT text, label, COUNT(DISTINCT chunk_id) AS n "
            "FROM entities GROUP BY text_lower, label "
            "ORDER BY n DESC LIMIT 12"
        ).fetchall()

        recent_html = "".join(
            f'<div class="stat"><span class="path">'
            f'<a href="/file?path={urllib.parse.quote_plus(r["path"])}">{escape(r["path"])}</a></span>'
            f'<span class="v"><span class="label">{r["kind"]}</span>'
            f' {(time.time() - r["mtime"]) / 86400:.1f}d</span></div>'
            for r in recent_rows
        ) or '<div class="muted">(no files indexed yet)</div>'

        ents_html = "".join(
            f'<div class="stat">'
            f'<span>{_entity_link(r["text"], r["label"])}</span>'
            f'<span class="v">{r["n"]}</span></div>'
            for r in ent_rows
        ) or '<div class="muted">(no entities yet — install [ner] extra and reindex)</div>'

        rerank = reranker.name if reranker else "disabled"
        spend = spend_summary(cfg)
        spend_lines = []
        for provider, bucket in spend.items():
            if provider not in ("voyage", "anthropic"):
                continue
            cap_cents = (
                cfg.daily_budget_cents_voyage if provider == "voyage"
                else cfg.daily_budget_cents_anthropic
            )
            cap_str = "no cap" if cap_cents == 0 else f"of ${cap_cents / 100:.2f}"
            warn = "warn" if cap_cents > 0 and bucket["cents"] >= cap_cents * 0.8 else ""
            spend_lines.append(
                f'<div class="stat"><span class="k">{provider}</span>'
                f'<span class="v {warn}">${bucket["cents"] / 100:.4f} {cap_str} '
                f'<span class="muted">· {bucket["calls"]} calls · '
                f'{bucket["tokens"]:,} tok</span></span></div>'
            )
        spend_html = "".join(spend_lines) or '<div class="muted">(no spend recorded yet)</div>'

        body = f"""
<h1>Overview</h1>
<div class="grid">
    <div class="card">
        <h2>Index</h2>
        <div class="stat"><span class="k">Files</span><span class="v">{s['files']}</span></div>
        <div class="stat"><span class="k">Aliases</span><span class="v">{s.get('aliases', 0)} <span class="muted">(dup paths)</span></span></div>
        <div class="stat"><span class="k">Chunks</span><span class="v">{s['chunks']}</span></div>
        <div class="stat"><span class="k">Entities</span><span class="v">{s.get('entities', 0)}</span></div>
        <div class="stat"><span class="k">Embedder</span><span class="v">{escape(str(s['embedder']))} (dim {s['embedding_dim']})</span></div>
        <div class="stat"><span class="k">Reranker</span><span class="v">{escape(rerank)}</span></div>
    </div>
    <div class="card">
        <h2>Today's spend (last 24h)</h2>
        {spend_html}
        <div class="muted" style="margin-top:8px;font-size:12px;">
            Caps refuse new paid calls once hit. Edit `daily_budget_cents_*` in config.toml.
        </div>
    </div>
    <div class="card">
        <h2>Quick search</h2>
        <form action="/search" method="get" class="search-box">
            <input type="text" name="q" placeholder="Search your brain…" autofocus>
        </form>
        <h3 style="margin-top: 18px;">Quick ingest</h3>
        <form action="/ingest" method="post" class="ingest-box">
            <input type="text" name="url" placeholder="https://...">
        </form>
    </div>
    <div class="card">
        <h2>Recent files</h2>
        {recent_html}
    </div>
    <div class="card">
        <h2>Top entities</h2>
        {ents_html}
    </div>
</div>"""
        return HTMLResponse(_layout("Overview", body, "/"))

    @app.get("/search", response_class=HTMLResponse)
    def search_page(
        q: str = Query("", alias="q"),
        folder: str = "",
        kind: str = "",
        since_days: int | None = None,
        k: int = 10,
    ):
        cfg, conn, embedder, reranker = get_state()
        kinds = [r["kind"] for r in conn.execute(
            "SELECT DISTINCT kind FROM files ORDER BY kind"
        ).fetchall()]
        kind_options = '<option value="">any kind</option>' + "".join(
            f'<option value="{escape(kk)}"{" selected" if kk == kind else ""}>{escape(kk)}</option>'
            for kk in kinds
        )

        results_html = ""
        if q:
            results = hybrid_search(
                conn, embedder, q, k=k, alpha=cfg.hybrid_alpha,
                reranker=reranker, rerank_overfetch=cfg.rerank_overfetch,
                use_adaptive_alpha=cfg.adaptive_alpha,
                time_decay_weight=cfg.time_decay_weight if cfg.time_decay_enabled else 0.0,
                time_decay_half_life_days=cfg.time_decay_half_life_days,
                path_prefix=folder or None,
                kind=kind or None,
                since_days=since_days,
            )
            if results:
                results_html = "".join(_result_block(r) for r in results)
            else:
                results_html = '<div class="empty">No matches.</div>'

        body = f"""
<h1>Search</h1>
<form method="get" action="/search">
    <div class="search-box">
        <input type="text" name="q" value="{escape(q)}" placeholder="Search your brain…" autofocus>
    </div>
    <div class="filters">
        <input type="text" name="folder" value="{escape(folder)}" placeholder="folder prefix (optional)">
        <select name="kind">{kind_options}</select>
        <input type="number" name="since_days" value="{since_days if since_days is not None else ''}" placeholder="since days" min="1">
        <input type="number" name="k" value="{k}" min="1" max="50" style="width: 70px;">
        <button type="submit">Search</button>
    </div>
</form>
<div>{results_html}</div>"""
        return HTMLResponse(_layout("Search", body, "search"))

    @app.get("/entities", response_class=HTMLResponse)
    def entities_page(label: str = "", limit: int = 100):
        cfg, conn, embedder, reranker = get_state()
        labels = [r["label"] for r in conn.execute(
            "SELECT DISTINCT label FROM entities ORDER BY label"
        ).fetchall()]
        label_options = '<option value="">all labels</option>' + "".join(
            f'<option value="{escape(ll)}"{" selected" if ll == label else ""}>{escape(ll)}</option>'
            for ll in labels
        )

        if label:
            rows = conn.execute(
                "SELECT text, label, COUNT(DISTINCT chunk_id) AS n "
                "FROM entities WHERE label = ? GROUP BY text_lower, label "
                "ORDER BY n DESC LIMIT ?",
                (label, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT text, label, COUNT(DISTINCT chunk_id) AS n "
                "FROM entities GROUP BY text_lower, label "
                "ORDER BY n DESC LIMIT ?",
                (limit,),
            ).fetchall()

        rows_html = "".join(
            f"<tr><td class='num'>{r['n']}</td>"
            f"<td><span class='label {r['label']}'>{r['label']}</span></td>"
            f"<td><a href='/entity?name={urllib.parse.quote_plus(r['text'])}'>{escape(r['text'])}</a></td></tr>"
            for r in rows
        ) or '<tr><td colspan="3" class="empty">No entities. Install the [ner] extra and reindex.</td></tr>'

        body = f"""
<h1>Entities</h1>
<form method="get" action="/entities" class="filters">
    <select name="label">{label_options}</select>
    <input type="number" name="limit" value="{limit}" min="10" max="500" style="width: 80px;">
    <button type="submit">Filter</button>
</form>
<div class="card">
<table>
    <thead><tr><th>chunks</th><th>label</th><th>entity</th></tr></thead>
    <tbody>{rows_html}</tbody>
</table>
</div>"""
        return HTMLResponse(_layout("Entities", body, "entities"))

    @app.get("/entity", response_class=HTMLResponse)
    def entity_detail(name: str):
        from .mcp_server import _matching_entity_keys

        cfg, conn, embedder, reranker = get_state()
        keys = _matching_entity_keys(conn, name, fuzzy=True)
        if not keys:
            body = f'<h1>{escape(name)}</h1><div class="empty">No mentions found.</div>'
            return HTMLResponse(_layout(name, body, "entities"))

        placeholders = ",".join("?" * len(keys))
        # Mentions
        ment_rows = conn.execute(
            f"SELECT DISTINCT c.text, c.chunk_index, f.path, f.mtime "
            f"FROM entities e JOIN chunks c ON c.id = e.chunk_id "
            f"JOIN files f ON f.id = c.file_id "
            f"WHERE e.text_lower IN ({placeholders}) "
            f"ORDER BY f.mtime DESC LIMIT 12",
            keys,
        ).fetchall()
        # Neighbors
        neigh_rows = conn.execute(
            f"SELECT b.text, b.label, COUNT(DISTINCT b.chunk_id) AS n "
            f"FROM entities a JOIN entities b ON a.chunk_id = b.chunk_id "
            f"WHERE a.text_lower IN ({placeholders}) "
            f"  AND b.text_lower NOT IN ({placeholders}) "
            f"GROUP BY b.text_lower, b.label "
            f"ORDER BY n DESC LIMIT 18",
            [*keys, *keys],
        ).fetchall()
        # Files
        files_rows = conn.execute(
            f"SELECT DISTINCT f.path, f.mtime, f.kind "
            f"FROM entities e JOIN chunks c ON c.id = e.chunk_id "
            f"JOIN files f ON f.id = c.file_id "
            f"WHERE e.text_lower IN ({placeholders}) "
            f"ORDER BY f.mtime DESC LIMIT 30",
            keys,
        ).fetchall()

        ments_html = "".join(
            f'<article class="result">'
            f'<h3><a href="/file?path={urllib.parse.quote_plus(r["path"])}">{escape(r["path"])}</a></h3>'
            f'<div class="meta">chunk {r["chunk_index"]}</div>'
            f'<div class="snippet">{escape(r["text"][:1200])}</div>'
            f'</article>'
            for r in ment_rows
        ) or '<div class="muted">no mentions</div>'

        neigh_html = "".join(
            f'<div class="stat"><span>{_entity_link(r["text"], r["label"])}</span>'
            f'<span class="v">{r["n"]}</span></div>'
            for r in neigh_rows
        ) or '<div class="muted">no co-occurrences</div>'

        files_html = "".join(
            f'<div class="stat"><span class="path">'
            f'<a href="/file?path={urllib.parse.quote_plus(r["path"])}">{escape(r["path"])}</a></span>'
            f'<span class="v"><span class="label">{r["kind"]}</span> '
            f'{(time.time() - r["mtime"]) / 86400:.1f}d</span></div>'
            for r in files_rows
        ) or '<div class="muted">no files</div>'

        alias_note = (
            f' <span class="muted">(fuzzy matched {len(keys)} aliases)</span>'
            if len(keys) > 1 else ""
        )
        body = f"""
<h1>{escape(name)}{alias_note}</h1>
<div class="grid">
    <div class="card"><h2>Co-occurring entities</h2>{neigh_html}</div>
    <div class="card"><h2>Files</h2>{files_html}</div>
</div>
<h2>Recent mentions</h2>
{ments_html}"""
        return HTMLResponse(_layout(name, body, "entities"))

    @app.get("/folders", response_class=HTMLResponse)
    def folders_page():
        cfg, conn, embedder, reranker = get_state()
        rows = conn.execute("SELECT path FROM files").fetchall()
        counts: dict[str, int] = {}
        for r in rows:
            p = Path(r["path"]).parent.as_posix()
            counts[p] = counts.get(p, 0) + 1
        ordered = sorted(counts.items(), key=lambda kv: -kv[1])

        rows_html = "".join(
            f'<tr><td class="num">{n}</td>'
            f'<td><a href="/search?folder={urllib.parse.quote_plus(folder)}">'
            f'<span class="path">{escape(folder)}</span></a></td></tr>'
            for folder, n in ordered
        ) or '<tr><td colspan="2" class="empty">No folders indexed yet.</td></tr>'

        body = f"""
<h1>Folders</h1>
<div class="card"><table>
    <thead><tr><th>files</th><th>folder</th></tr></thead>
    <tbody>{rows_html}</tbody>
</table></div>"""
        return HTMLResponse(_layout("Folders", body, "folders"))

    @app.get("/graph", response_class=HTMLResponse)
    def graph_page(top_n: int = 100, min_cooccur: int = 2):
        body = f"""
<h1>Knowledge graph</h1>
<form method="get" action="/graph" class="filters">
    <label class="muted">Top entities:</label>
    <input type="number" name="top_n" value="{top_n}" min="20" max="500" style="width: 90px;">
    <label class="muted">Min co-occurrences:</label>
    <input type="number" name="min_cooccur" value="{min_cooccur}" min="1" max="20" style="width: 70px;">
    <button type="submit">Reload</button>
    <span class="muted">Click any node to drill into its entity page. Drag nodes to reposition. Scroll to zoom.</span>
</form>
<div id="cy"></div>
<script src="https://cdn.jsdelivr.net/npm/cytoscape@3.30.4/dist/cytoscape.min.js"></script>
<script>
fetch('/graph/data?top_n={top_n}&min_cooccur={min_cooccur}').then(r => r.json()).then(data => {{
    const cy = cytoscape({{
        container: document.getElementById('cy'),
        elements: [...data.nodes, ...data.edges],
        style: [
            {{ selector: 'node', style: {{
                'background-color': 'data(color)',
                'label': 'data(label)',
                'color': '#e6e6ea',
                'font-size': 11,
                'font-family': '-apple-system, "Segoe UI", sans-serif',
                'width': 'data(size)',
                'height': 'data(size)',
                'text-outline-color': '#0e0e10',
                'text-outline-width': 2,
                'text-margin-y': -4,
            }} }},
            {{ selector: 'edge', style: {{
                'width': 'data(weight)',
                'line-color': '#3a3a45',
                'curve-style': 'haystack',
                'opacity': 0.55,
            }} }},
            {{ selector: 'node:selected', style: {{
                'border-width': 3,
                'border-color': '#4d9fff',
            }} }},
        ],
        layout: {{
            name: 'cose',
            idealEdgeLength: 90,
            nodeOverlap: 20,
            refresh: 20,
            fit: true,
            padding: 30,
            randomize: true,
            componentSpacing: 100,
            nodeRepulsion: 400000,
            edgeElasticity: 100,
            nestingFactor: 5,
            gravity: 80,
            numIter: 1000,
            initialTemp: 200,
            coolingFactor: 0.95,
            minTemp: 1.0,
        }},
        wheelSensitivity: 0.2,
    }});
    cy.on('tap', 'node', (evt) => {{
        const name = evt.target.data('raw_text');
        window.location.href = '/entity?name=' + encodeURIComponent(name);
    }});
}});
</script>"""
        return HTMLResponse(_layout("Graph", body, "graph"))

    @app.get("/graph/data")
    def graph_data(top_n: int = 100, min_cooccur: int = 2):
        cfg, conn, _, _ = get_state()
        # Step 1: pick top_n entities by chunk count.
        ent_rows = conn.execute(
            "SELECT text, label, text_lower, COUNT(DISTINCT chunk_id) AS n "
            "FROM entities GROUP BY text_lower, label "
            "ORDER BY n DESC LIMIT ?",
            (top_n,),
        ).fetchall()
        # Step 2: pull co-occurrences, filtered to that entity set, above the
        # threshold. Self-join on chunk_id; use casefold-lower as the canonical
        # identity so 'Apollo 11' and 'apollo 11' merge.
        keys = {r["text_lower"] for r in ent_rows}
        if not keys or len(keys) < 2:
            return {"nodes": [], "edges": []}
        placeholders = ",".join("?" * len(keys))
        edge_rows = conn.execute(
            f"SELECT a.text_lower AS a, b.text_lower AS b, "
            f"       COUNT(DISTINCT a.chunk_id) AS w "
            f"FROM entities a "
            f"JOIN entities b ON a.chunk_id = b.chunk_id AND a.text_lower < b.text_lower "
            f"WHERE a.text_lower IN ({placeholders}) "
            f"  AND b.text_lower IN ({placeholders}) "
            f"GROUP BY a.text_lower, b.text_lower "
            f"HAVING w >= ? "
            f"ORDER BY w DESC",
            [*keys, *keys, min_cooccur],
        ).fetchall()

        label_color = {
            "PERSON": "#4d6dff",
            "ORG": "#ff8c4d",
            "GPE": "#4dff8c",
            "LOC": "#4dff8c",
            "FAC": "#4dff8c",
            "PRODUCT": "#c44dff",
            "WORK_OF_ART": "#c44dff",
            "DATE": "#4dffe6",
            "MONEY": "#ffe14d",
            "EVENT": "#ff4d6d",
            "LAW": "#ff4d6d",
            "NORP": "#ff4d6d",
            "LANGUAGE": "#ff4d6d",
        }
        max_n = max((r["n"] for r in ent_rows), default=1)
        # Drop nodes with no edges - they show up as orphans and clutter the layout.
        connected: set[str] = set()
        for e in edge_rows:
            connected.add(e["a"])
            connected.add(e["b"])
        nodes = []
        for r in ent_rows:
            if r["text_lower"] not in connected:
                continue
            size = 16 + 38 * (r["n"] / max_n)
            nodes.append({
                "data": {
                    "id": r["text_lower"],
                    "label": r["text"],
                    "raw_text": r["text"],
                    "color": label_color.get(r["label"], "#888"),
                    "size": round(size),
                    "n": r["n"],
                    "ent_label": r["label"],
                }
            })
        max_w = max((r["w"] for r in edge_rows), default=1)
        edges = []
        for r in edge_rows:
            edges.append({
                "data": {
                    "id": f"{r['a']}__{r['b']}",
                    "source": r["a"],
                    "target": r["b"],
                    "weight": round(1 + 6 * (r["w"] / max_w), 1),
                    "raw_w": r["w"],
                }
            })
        return {"nodes": nodes, "edges": edges}

    @app.get("/file", response_class=HTMLResponse)
    def file_view(path: str):
        cfg, conn, embedder, reranker = get_state()
        row = conn.execute(
            "SELECT path, kind, mtime, size FROM files WHERE path = ?", (path,)
        ).fetchone()
        if not row:
            return HTMLResponse(_layout(path, f'<h1>{escape(path)}</h1><div class="empty">Not in index.</div>'))

        chunks = conn.execute(
            "SELECT chunk_index, text FROM chunks WHERE file_id = ("
            "  SELECT id FROM files WHERE path = ?) ORDER BY chunk_index",
            (path,),
        ).fetchall()
        body_chunks = "".join(
            f'<article class="result"><h3>chunk {r["chunk_index"]}</h3>'
            f'<div class="snippet">{escape(r["text"])}</div></article>'
            for r in chunks
        )
        kind = row["kind"]
        age = (time.time() - row["mtime"]) / 86400
        is_url = path.startswith("http://") or path.startswith("https://")
        open_link = f'<a href="{escape(path)}" target="_blank">Open ↗</a>' if is_url else ""
        body = f"""
<h1>File <span class="muted">[{kind}]</span></h1>
<p class="path">{escape(path)} {open_link}</p>
<p class="muted">{len(chunks)} chunks · {row["size"] / 1024:.1f} KB · {age:.1f}d ago</p>
{body_chunks}"""
        return HTMLResponse(_layout(Path(path).name or path, body))

    @app.get("/briefing", response_class=HTMLResponse)
    def briefing_page(hours: int = 24):
        body = f"""
<h1>Daily briefing</h1>
<form method="get" action="/briefing" class="filters">
    <label class="muted">Look-back window:</label>
    <input type="number" name="hours" value="{hours}" min="1" max="168" style="width: 90px;">
    <button type="submit">Generate</button>
    <span class="muted">Uses Claude Opus 4.7 + your ANTHROPIC_API_KEY. ~5–10s, fractions of a cent per call.</span>
</form>
<div id="briefing-result" hx-get="/briefing/run?hours={hours}" hx-trigger="load" hx-swap="innerHTML">
    <div class="empty">Generating briefing — this can take 5–15 seconds…</div>
</div>"""
        return HTMLResponse(_layout("Briefing", body, "briefing"))

    @app.get("/briefing/run", response_class=HTMLResponse)
    def briefing_run(hours: int = 24):
        cfg, conn, _, _ = get_state()
        text = generate_briefing(conn, cfg, hours=hours)
        # Render as a card with preserved newlines. Markdown is rendered as
        # plain pre text — keeps the LLM output verbatim and avoids HTML escaping
        # surprises if the model emits angle brackets.
        return HTMLResponse(
            f'<div class="card"><div class="snippet" style="max-height: none; '
            f'background: var(--surface); padding: 0; font-family: var(--sans); '
            f'font-size: 14.5px; white-space: pre-wrap;">{escape(text)}</div></div>'
        )

    @app.get("/ingest", response_class=HTMLResponse)
    def ingest_page():
        body = """
<h1>Ingest URL</h1>
<form method="post" action="/ingest" class="card" style="max-width: 720px;">
    <p class="muted">Fetches a URL and indexes it. Supports HTML articles, PDFs at URLs, and YouTube transcripts.</p>
    <div class="ingest-box">
        <input type="text" name="url" placeholder="https://en.wikipedia.org/wiki/..." autofocus>
    </div>
    <div style="margin-top: 12px;">
        <button type="submit">Ingest</button>
    </div>
</form>"""
        return HTMLResponse(_layout("Ingest", body, "ingest"))

    @app.post("/ingest", response_class=HTMLResponse)
    def ingest_action(url: str = Form(...)):
        cfg, conn, embedder, _ = get_state()
        url = url.strip()
        if not url:
            return RedirectResponse(url="/ingest", status_code=303)
        entity_extractor = None
        if cfg.entities_enabled:
            try:
                entity_extractor = make_entity_extractor(cfg)
            except (ImportError, RuntimeError):
                pass
        result = index_url(conn, embedder, cfg, url, entity_extractor=entity_extractor)
        status_class = {
            "indexed": "good", "unchanged": "muted", "skipped": "warn", "error": "bad",
        }.get(result.status, "muted")
        body = f"""
<h1>Ingest result</h1>
<div class="card">
    <p><strong style="color: var(--{status_class})">{result.status}</strong>: <span class="path">{escape(url)}</span></p>
    {f'<p>{result.chunks} chunks indexed.</p>' if result.status == "indexed" else ""}
    {f'<p class="muted">{escape(result.reason or "")}</p>' if result.reason else ""}
    <p>
        <a href="/file?path={urllib.parse.quote_plus(url)}">View in index</a>
        ·
        <a href="/ingest">Ingest another</a>
        ·
        <a href="/">Home</a>
    </p>
</div>"""
        return HTMLResponse(_layout("Ingest result", body, "ingest"))

    @app.get("/health")
    def health():
        return {"ok": True}

    return app


def run_dashboard(
    host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True
) -> None:
    """Launch the dashboard. Requires the [dashboard] extra.

    Bind defaults to 127.0.0.1 - this is a single-user tool; do not expose
    publicly without thinking through auth.
    """
    try:
        import uvicorn
    except ImportError as e:
        raise ImportError(
            "Dashboard requires the [dashboard] extra. "
            "Install with: pip install -e .[dashboard]"
        ) from e
    try:
        from fastapi import FastAPI  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Dashboard requires the [dashboard] extra. "
            "Install with: pip install -e .[dashboard]"
        ) from e

    app = create_app()
    url = f"http://{host}:{port}"
    print(f"Dashboard: {url}")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
