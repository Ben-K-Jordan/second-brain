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

import json
import logging
import secrets
import threading
import time
import urllib.parse
import webbrowser
from html import escape
from pathlib import Path
from typing import Any

# Module-level FastAPI Request import — needed because dashboard routes
# are defined inside create_app() (a closure), and `from __future__
# import annotations` makes type hints strings. FastAPI resolves those
# via typing.get_type_hints() with the function's module __globals__,
# NOT the closure scope, so a `request: Request` annotation can't see
# a Request imported only inside create_app(). Lifting it here fixes
# that without restructuring the closure.
from fastapi import Request  # noqa: F401  (resolved as forward-ref)

from .briefing import generate_briefing
from .budget import spend_summary
from .chat import stream_chat
from .config import Config, load_config
from .db import connect, init_schema, stats
from .embedder import make_embedder
from .entities import make_entity_extractor
from .indexer import index_url
from .mcp_server import _log_query
from .reranker import make_reranker
from .search import hybrid_search

log = logging.getLogger(__name__)


# Round 19 — module-level redact helper used by EA-feature routes.
# Wraps safety.redact_text but tolerates the import being unavailable
# (import-time dep cycle prevention; same pattern as elsewhere).
def _safe(text):  # noqa: ANN001
    try:
        from .safety import redact_text
        return redact_text(text or "")
    except ImportError:
        return text or ""


def _extension_token_path(cfg: Config) -> Path:
    return cfg.data_dir / "extension_token.txt"


def get_or_create_extension_token(cfg: Config) -> str:
    """Per-install random secret the browser extension must present as
    ``Authorization: Bearer ...`` on /api/extension/* calls.

    Stored at ``<data_dir>/extension_token.txt`` with mode 0600 where the
    OS supports it. Without this, the dashboard's CORS allow-list lets any
    JavaScript on chatgpt.com / x.com / etc. exfiltrate the entire index.
    """
    path = _extension_token_path(cfg)
    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    path.write_text(token, encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        # Windows ignores the bits anyway; fine.
        pass
    return token


# --- HTML rendering helpers (no Jinja dep — embedded templates as f-strings) ---

CSS = """
/*  ─── second-brain · 80s hacker theme ─────────────────────────────  */
:root {
    --bg:           #0a0a0a;     /* near-black with phosphor warmth */
    --bg-elevated:  #111111;
    --bg-card:      #0e0e0e;
    --bg-hover:     #161616;
    --bg-input:     #050505;     /* deeper than bg for "inset terminal" */
    --border:       #1f1f1f;
    --border-strong:#2c2c2c;
    --grid:         rgba(127,255,127,0.04);

    --text:         #d4d4c8;     /* warm phosphor white */
    --text-2:       #8a8a7a;
    --text-3:       #555548;
    --text-4:       #2e2e2a;

    --green:        #7fff7f;     /* phosphor primary */
    --green-dim:    #4abe4a;
    --green-glow:   rgba(127,255,127,0.45);
    --green-soft:   rgba(127,255,127,0.10);

    --amber:        #ffb700;     /* warning / secondary accent */
    --amber-glow:   rgba(255,183,0,0.40);

    --cyan:         #5af0ff;
    --magenta:      #ff5af0;
    --red:          #ff4d4d;

    --accent:       var(--green);
    --accent-glow:  var(--green-glow);
    --accent-soft:  var(--green-soft);

    --good: #7fff7f;
    --warn: #ffb700;
    --bad:  #ff4d4d;

    --mono: 'JetBrains Mono', 'Berkeley Mono', 'IBM Plex Mono', 'SF Mono',
            'Cascadia Mono', Consolas, 'Courier New', monospace;
    --sans: var(--mono);   /* committing to mono everywhere */

    --s-1: 4px; --s-2: 8px; --s-3: 12px; --s-4: 16px;
    --s-5: 24px; --s-6: 32px; --s-7: 48px; --s-8: 64px;

    --r:   2px;   /* sharp, terminal edges */
    --r-md: 3px;
    --r-lg: 4px;

    --shadow-glow: 0 0 24px rgba(127,255,127,0.10);
    --shadow-pop:  0 0 0 1px var(--green-glow), 0 0 60px rgba(127,255,127,0.18), 0 14px 50px rgba(0,0,0,0.85);

    --transition: 120ms steps(8, end);   /* steppy retro feel on color shifts */
    --ease: 160ms cubic-bezier(.2,.8,.2,1);
}

* { box-sizing: border-box; }
*::selection { background: var(--green); color: #000; }

html, body {
    margin: 0; padding: 0;
    background: var(--bg); color: var(--text);
    font-family: var(--mono); font-size: 13.5px; line-height: 1.55;
    font-feature-settings: 'liga' 0, 'calt' 0;  /* disable code ligatures */
    -webkit-font-smoothing: antialiased;
}

/* Subtle dot-grid background — feels like terminal graph paper */
body::before {
    content: ""; position: fixed; inset: 0; z-index: -2;
    background-image: radial-gradient(var(--grid) 1px, transparent 1px);
    background-size: 24px 24px;
    background-position: 0 0;
    pointer-events: none;
    opacity: 0.6;
}
/* Soft phosphor vignette */
body::after {
    content: ""; position: fixed; inset: 0; z-index: -1; pointer-events: none;
    background: radial-gradient(70% 50% at 50% -10%, rgba(127,255,127,0.05), transparent 60%),
                radial-gradient(50% 40% at 100% 100%, rgba(255,183,0,0.025), transparent 70%);
}

a { color: var(--green); text-decoration: none; transition: color var(--ease); }
a:hover { color: #b8ffb8; text-shadow: 0 0 8px var(--green-glow); }

/* Sticky header */
header {
    border-bottom: 1px solid var(--border-strong);
    padding: var(--s-3) var(--s-5);
    display: flex; align-items: center; gap: var(--s-5);
    position: sticky; top: 0; z-index: 10;
    background: rgba(10,10,10,0.92);
    backdrop-filter: blur(10px);
}
header .brand {
    font-weight: 700; font-size: 14px;
    color: var(--green); text-shadow: 0 0 12px var(--green-glow);
    letter-spacing: 0.02em;
}
header .brand::before { content: "▮ "; color: var(--green); animation: blink 1.4s steps(2) infinite; }
header nav { display: flex; gap: 2px; }
header nav a {
    color: var(--text-2); padding: 6px 10px; font-size: 12.5px;
    transition: all var(--ease); border: 1px solid transparent;
    border-radius: var(--r);
}
header nav a:hover {
    color: var(--green); background: var(--green-soft);
    border-color: var(--border-strong);
    text-shadow: 0 0 6px var(--green-glow);
}
header nav a.active {
    color: var(--green); background: var(--green-soft);
    border-color: var(--green-dim);
    text-shadow: 0 0 6px var(--green-glow);
}
header .spacer { flex: 1; }
header .kbd-hint {
    display: inline-flex; align-items: center; gap: var(--s-2);
    padding: 5px 10px; border-radius: var(--r);
    background: var(--bg-input); border: 1px solid var(--border-strong);
    color: var(--text-2); font-size: 11.5px;
    cursor: pointer; transition: all var(--ease); font-family: var(--mono);
}
header .kbd-hint:hover {
    color: var(--green); border-color: var(--green-dim);
    box-shadow: inset 0 0 0 1px var(--green-soft), 0 0 12px var(--green-soft);
}
/* Nav badges — count chips that show pending state. JS populates
   `[data-badge]` after page load via /api/nav-counts; the empty
   state collapses to display:none so non-pending items stay clean. */
.nav-badge {
    display: none;
    margin-left: 6px;
    padding: 0 5px; min-width: 16px; height: 14px; line-height: 14px;
    border-radius: 7px; background: var(--green-soft);
    border: 1px solid var(--green-dim); color: var(--green);
    font-size: 10px; text-align: center; font-family: var(--mono);
    text-shadow: 0 0 4px var(--green-glow);
}
.nav-badge.has-count { display: inline-block; }
.nav-badge.urgent {
    background: rgba(255,77,77,0.10); border-color: var(--red);
    color: var(--red); text-shadow: 0 0 4px rgba(255,77,77,0.45);
}
/* "More ▾" dropdown — overflow menu for the long tail of pages.
   Uses native <details>/<summary> so no JS toggle needed. */
.nav-more { position: relative; }
.nav-more summary {
    list-style: none; cursor: pointer;
    color: var(--text-2); padding: 6px 10px; font-size: 12.5px;
    border: 1px solid transparent; border-radius: var(--r);
    transition: all var(--ease);
}
.nav-more summary::-webkit-details-marker { display: none; }
.nav-more summary:hover {
    color: var(--green); background: var(--green-soft);
    border-color: var(--border-strong);
}
.nav-more[open] summary {
    color: var(--green); background: var(--green-soft);
    border-color: var(--green-dim);
}
.nav-more-pop {
    position: absolute; top: calc(100% + 6px); right: 0;
    min-width: 520px; max-width: calc(100vw - 32px);
    padding: var(--s-4);
    background: var(--bg-elevated);
    border: 1px solid var(--border-strong); border-radius: var(--r);
    box-shadow: var(--shadow-pop);
    display: grid; grid-template-columns: repeat(3, 1fr); gap: var(--s-4);
    z-index: 20;
}
/* Round 11 — responsive: collapse to 2 columns on tablets, 1 on
   phones. Without this the dropdown overflows + the urgent health
   badge gets clipped on Mobile Safari (~390px viewport). */
@media (max-width: 720px) {
    .nav-more-pop {
        min-width: 280px;
        grid-template-columns: repeat(2, 1fr);
    }
}
@media (max-width: 800px) {
    .nav-more-pop {
        min-width: 220px; max-width: calc(100vw - 16px);
        grid-template-columns: 1fr;
    }
    /* Header itself wraps so primary-nav buttons + brand stay visible.
       Round 16 raised primary nav to 8 items (added Review + Inbox);
       round-17 audit flagged that this would wrap on 1024px. Bumping
       the wrap threshold from 480px to 800px so narrow laptops get
       a clean two-row nav instead of a single-row overflow. */
    header { flex-wrap: wrap; }
    header nav { flex-wrap: wrap; }
}
.nav-more-group h4 {
    margin: 0 0 var(--s-2);
    font-size: 10.5px; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--text-3);
}
.nav-more-group h4::before { content: "// "; color: var(--text-4); }
.nav-more-group a {
    display: block; padding: 4px 6px; font-size: 12px;
    color: var(--text-2); border-radius: var(--r);
    transition: all var(--ease);
}
.nav-more-group a:hover {
    color: var(--green); background: var(--green-soft);
    text-shadow: 0 0 6px var(--green-glow);
}
/* Launchpad — Overview-page grid of grouped page links. */
.launchpad {
    background: var(--bg-card);
    border: 1px solid var(--border-strong);
    border-radius: var(--r); padding: var(--s-5);
    margin-bottom: var(--s-5);
}
.launchpad-grid {
    display: grid; gap: var(--s-5);
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
}
.launchpad-group h3 {
    margin: 0 0 var(--s-2);
    font-size: 10.5px; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--green-dim);
}
.launchpad-group h3::before { content: "[ "; color: var(--green-dim); }
.launchpad-group h3::after  { content: " ]"; color: var(--green-dim); }
.launchpad-group a {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 4px 6px; font-size: 12.5px;
    color: var(--text); border-radius: var(--r);
    transition: all var(--ease);
}
.launchpad-group a:hover {
    color: var(--green); background: var(--green-soft);
    text-shadow: 0 0 6px var(--green-glow);
}
.launchpad-group a .meta {
    color: var(--text-3); font-size: 11px;
}
.kbd {
    display: inline-block; padding: 1px 5px; border-radius: 2px;
    background: #000; border: 1px solid var(--border-strong);
    font-family: var(--mono); font-size: 10.5px; color: var(--green);
    box-shadow: inset 0 -2px 0 #000, 0 0 4px var(--green-soft);
}

main { padding: var(--s-6) var(--s-5); max-width: 1280px; margin: 0 auto; }

h1, h2, h3, h4 { font-weight: 700; color: var(--text); letter-spacing: 0; }
h1 {
    font-size: 22px; margin: 0 0 var(--s-5); color: var(--green);
    text-shadow: 0 0 14px var(--green-glow);
}
h1::before { content: "$ "; color: var(--green-dim); }
h2 {
    font-size: 14px; margin: var(--s-6) 0 var(--s-3);
    color: var(--text); text-transform: uppercase; letter-spacing: 0.04em;
}
h2::before { content: "// "; color: var(--text-3); }
h3 { font-size: 13px; margin: var(--s-3) 0 var(--s-2); color: var(--green-dim); }

.grid {
    display: grid; gap: var(--s-4);
    grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
}

/* Cards as terminal "boxes" */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border-strong);
    border-radius: var(--r); padding: var(--s-5);
    transition: border-color var(--ease), box-shadow var(--ease);
    position: relative;
}
.card:hover {
    border-color: var(--green-dim);
    box-shadow: inset 0 0 0 1px var(--green-soft), 0 0 20px rgba(127,255,127,0.06);
}
.card h2 { margin-top: 0; }
.card h2:first-child::before { content: "[ "; color: var(--green-dim); }
.card h2:first-child::after  { content: " ]"; color: var(--green-dim); }

.stat {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: var(--s-2) 0; border-bottom: 1px dashed var(--border-strong);
    gap: var(--s-3); min-height: 30px;
}
.stat:last-child { border-bottom: none; }
.stat .k {
    color: var(--text-2); font-size: 12px;
    font-family: var(--mono);
}
.stat .k::before { content: "› "; color: var(--text-3); }
.stat .v {
    font-family: var(--mono); font-size: 12.5px;
    color: var(--green); text-align: right;
}
.muted { color: var(--text-3); }
.path {
    font-family: var(--mono); font-size: 11.5px;
    color: var(--text-2); word-break: break-all;
}

/* Entity labels — phosphor variants per type */
.label {
    display: inline-block; padding: 2px 6px; border-radius: 2px;
    font-size: 10px; font-weight: 700; letter-spacing: 0.06em;
    font-family: var(--mono); margin-right: var(--s-1);
    text-transform: uppercase;
    border: 1px solid;
}
.label.PERSON      { color: var(--green);   border-color: var(--green-dim);  background: rgba(127,255,127,0.06); }
.label.ORG         { color: var(--amber);   border-color: rgba(255,183,0,0.4); background: rgba(255,183,0,0.06); }
.label.GPE,
.label.LOC,
.label.FAC         { color: var(--cyan);    border-color: rgba(90,240,255,0.4); background: rgba(90,240,255,0.06); }
.label.PRODUCT,
.label.WORK_OF_ART { color: var(--magenta); border-color: rgba(255,90,240,0.4); background: rgba(255,90,240,0.06); }
.label.DATE        { color: #b8ffb8; border-color: rgba(184,255,184,0.4); background: rgba(184,255,184,0.05); }
.label.MONEY       { color: var(--amber); border-color: rgba(255,183,0,0.6); background: rgba(255,183,0,0.10); }
.label.LAW,
.label.NORP,
.label.LANGUAGE,
.label.EVENT       { color: var(--red); border-color: rgba(255,77,77,0.4); background: rgba(255,77,77,0.06); }

/* Inputs — terminal prompt style */
.search-box { position: relative; }
.search-box::before {
    content: ">"; position: absolute; left: 14px; top: 50%;
    transform: translateY(-50%); color: var(--green); font-family: var(--mono);
    font-weight: 700; pointer-events: none;
}
.search-box input, .ingest-box input {
    width: 100%; padding: 12px 14px 12px 32px; font-size: 14px;
    background: var(--bg-input); color: var(--green);
    border: 1px solid var(--border-strong); border-radius: var(--r);
    font-family: var(--mono); caret-color: var(--green);
    transition: all var(--ease);
}
.ingest-box input { padding-left: 14px; }
.search-box input:focus, .ingest-box input:focus,
.filters input:focus, .filters select:focus {
    outline: none; border-color: var(--green);
    box-shadow: 0 0 0 1px var(--green-glow), 0 0 14px var(--green-soft);
    background: #000;
}
.filters {
    display: flex; gap: var(--s-2); margin: var(--s-2) 0 var(--s-5);
    flex-wrap: wrap; align-items: center;
}
.filters input, .filters select {
    padding: 6px 10px; font-size: 12px;
    background: var(--bg-input); color: var(--green);
    border: 1px solid var(--border-strong); border-radius: var(--r);
    font-family: var(--mono); transition: all var(--ease);
}
.filters label { color: var(--text-3); font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }

/* Buttons — terminal call-to-action */
button {
    padding: 7px 14px; background: var(--bg-card); color: var(--green);
    border: 1px solid var(--green-dim); border-radius: var(--r);
    font-size: 12px; font-weight: 700; letter-spacing: 0.04em;
    cursor: pointer; font-family: var(--mono);
    text-transform: uppercase;
    transition: all var(--ease);
}
button:hover {
    background: var(--green-soft); color: var(--green);
    box-shadow: 0 0 14px var(--green-soft), inset 0 0 0 1px var(--green-glow);
    text-shadow: 0 0 6px var(--green-glow);
}
button:active { transform: translateY(1px); }
button.ghost {
    background: transparent; color: var(--text-2);
    border-color: var(--border-strong);
}
button.ghost:hover {
    background: var(--bg-hover); color: var(--text); border-color: var(--green-dim);
    box-shadow: none; text-shadow: none;
}

/* Search results — output blocks */
.result {
    border: 1px solid var(--border-strong); border-radius: var(--r);
    padding: var(--s-4); margin-bottom: var(--s-3);
    background: var(--bg-card);
    transition: border-color var(--ease);
    position: relative;
}
.result::before {
    content: ""; position: absolute; left: 0; top: 0; bottom: 0;
    width: 2px; background: var(--green-dim);
    transition: background var(--ease);
}
.result:hover {
    border-color: var(--green-dim);
}
.result:hover::before { background: var(--green); }
.result h3 {
    margin: 0 0 var(--s-1); font-weight: 600; font-size: 12.5px;
    color: var(--text); text-transform: none;
}
.result h3::before { content: ""; }
.result h3 a { color: var(--green-dim); }
.result h3 a:hover { color: var(--green); text-shadow: 0 0 6px var(--green-glow); }
.result .snippet {
    white-space: pre-wrap; font-family: var(--mono); font-size: 11.5px;
    color: var(--text); background: #000;
    padding: var(--s-3); border-radius: var(--r);
    border: 1px solid var(--border);
    max-height: 280px; overflow: auto;
    line-height: 1.65;
}
.result .meta {
    color: var(--amber); font-size: 11px; margin: var(--s-1) 0 var(--s-2);
    font-family: var(--mono); letter-spacing: 0.02em;
}

/* Tables */
table { width: 100%; border-collapse: collapse; font-family: var(--mono); }
th, td {
    padding: var(--s-2) var(--s-3); text-align: left;
    border-bottom: 1px dashed var(--border-strong);
    font-size: 12px;
}
th {
    color: var(--green-dim); font-weight: 700; font-size: 11px;
    text-transform: uppercase; letter-spacing: 0.06em;
    border-bottom-style: solid;
}
tr:hover td { background: var(--green-soft); color: var(--green); }
td.num { font-family: var(--mono); text-align: right; color: var(--amber); }

.empty {
    padding: var(--s-7) var(--s-5); text-align: center;
    color: var(--text-3); font-size: 13px; font-family: var(--mono);
}
.empty::before { content: "// "; color: var(--text-4); }
.warn { color: var(--warn); }
.good { color: var(--good); }
.bad  { color: var(--bad); }

/* Graph canvas */
#cy {
    width: 100%; height: 78vh;
    background: #050505;
    border-radius: var(--r);
    border: 1px solid var(--border-strong);
    box-shadow: inset 0 0 60px rgba(127,255,127,0.04);
}
.graph-overlay {
    position: absolute; top: var(--s-3); right: var(--s-3);
    background: rgba(10,10,10,0.94); backdrop-filter: blur(8px);
    border: 1px solid var(--green-dim); border-radius: var(--r);
    padding: var(--s-3); font-size: 11.5px; max-width: 220px;
    box-shadow: var(--shadow-glow);
    font-family: var(--mono);
}
.graph-overlay h4 {
    margin: 0 0 var(--s-2); font-size: 10.5px; color: var(--green);
    text-transform: uppercase; letter-spacing: 0.06em;
}
.graph-overlay h4::before { content: "▸ "; }
.graph-overlay .legend-row {
    display: flex; align-items: center; gap: var(--s-2);
    padding: 3px 0; cursor: pointer; user-select: none;
    transition: opacity var(--ease);
    color: var(--text-2);
}
.graph-overlay .legend-row:hover { color: var(--green); }
.graph-overlay .legend-row.disabled { opacity: 0.30; }
.graph-overlay .swatch {
    width: 10px; height: 10px; border-radius: 1px;
    box-shadow: 0 0 6px currentColor;
}
.graph-wrap { position: relative; }

/* Command palette */
#palette-backdrop {
    position: fixed; inset: 0; z-index: 100;
    background: rgba(0,0,0,0.7);
    backdrop-filter: blur(6px);
    display: none; align-items: flex-start; justify-content: center;
    padding-top: 12vh;
    animation: fadeIn 120ms ease-out;
}
#palette-backdrop.open { display: flex; }
#palette {
    width: min(640px, 92vw); max-height: 70vh;
    background: var(--bg-elevated);
    border: 1px solid var(--green-dim); border-radius: var(--r);
    box-shadow: var(--shadow-pop);
    display: flex; flex-direction: column; overflow: hidden;
    animation: popIn 180ms cubic-bezier(.2,.8,.2,1);
    font-family: var(--mono);
}
#palette-input {
    width: 100%; padding: var(--s-4) var(--s-5);
    border: none; border-bottom: 1px solid var(--border-strong);
    background: transparent; color: var(--green);
    font-family: var(--mono); font-size: 15px;
    caret-color: var(--green);
}
#palette-input::placeholder { color: var(--text-3); }
#palette-input:focus { outline: none; }
#palette-results { overflow-y: auto; padding: var(--s-2) 0; }
#palette-results .group-title {
    padding: var(--s-2) var(--s-5) var(--s-1);
    color: var(--green-dim); font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
}
#palette-results .group-title::before { content: "── "; }
#palette-results .item {
    display: flex; align-items: center; gap: var(--s-3);
    padding: var(--s-2) var(--s-5); cursor: pointer;
    color: var(--text); font-size: 13px;
    border-left: 2px solid transparent;
    transition: background var(--ease), color var(--ease);
}
#palette-results .item.selected {
    background: var(--green-soft); color: var(--green);
    border-left-color: var(--green);
    text-shadow: 0 0 6px var(--green-glow);
}
#palette-results .item .icon {
    color: var(--green-dim); font-family: var(--mono); font-size: 12px;
    width: 24px; text-align: center;
}
#palette-results .item.selected .icon { color: var(--green); }
#palette-results .item .item-meta {
    color: var(--amber); font-size: 10.5px; margin-left: auto;
    font-family: var(--mono);
}
#palette-footer {
    padding: var(--s-2) var(--s-5); border-top: 1px solid var(--border-strong);
    color: var(--text-3); font-size: 10.5px;
    display: flex; gap: var(--s-4); align-items: center; font-family: var(--mono);
}

@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes popIn {
    from { opacity: 0; transform: translateY(-6px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes blink { 50% { opacity: 0; } }

/* Briefing renders nicer */
.briefing-body h1 { font-size: 20px; }
.briefing-body h2 {
    font-size: 13px; margin-top: var(--s-5);
    color: var(--green); font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.04em;
}
.briefing-body p { color: var(--text); }
.briefing-body ul { padding-left: var(--s-5); }
.briefing-body li { color: var(--text); margin: var(--s-1) 0; }
"""


NAV_BADGES_JS = r"""
(function () {
    // Populate count chips on the primary nav (Tasks / Drafts /
    // Insights). One fetch per page load against /api/nav-counts.
    // Failure is silent — the badges just stay hidden.
    fetch('/api/nav-counts', {credentials: 'same-origin'})
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (data) {
            if (!data) return;
            document.querySelectorAll('[data-badge]').forEach(function (el) {
                const key = el.getAttribute('data-badge');
                const n = data[key];
                if (typeof n !== 'number' || n <= 0) return;
                el.textContent = n > 99 ? '99+' : String(n);
                el.classList.add('has-count');
                // Drafts + urgent insights get the red urgent style;
                // tasks stay green even when they pile up.
                if (data.urgent && data.urgent[key]) {
                    el.classList.add('urgent');
                }
            });
        })
        .catch(function () { /* best-effort, badges stay hidden */ });
})();
"""


CLICK_BEACON_JS = r"""
(function () {
    // Listens for clicks on any [data-sb-click] link and fires a tiny POST
    // to /api/click before the navigation happens. Uses sendBeacon when
    // available so the request reliably ships even though we're leaving
    // the page. Falls back to a fire-and-forget fetch otherwise.
    document.addEventListener('click', function (ev) {
        const a = ev.target && ev.target.closest && ev.target.closest('[data-sb-click]');
        if (!a) return;
        const path = a.getAttribute('data-sb-path') || '';
        const chunkId = a.getAttribute('data-sb-chunk') || '';
        const source = a.getAttribute('data-sb-source') || 'unknown';
        if (!path) return;
        const payload = JSON.stringify({ path, chunk_id: chunkId, source });
        try {
            if (navigator.sendBeacon) {
                navigator.sendBeacon(
                    '/api/click',
                    new Blob([payload], { type: 'application/json' })
                );
            } else {
                fetch('/api/click', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: payload,
                    keepalive: true,
                }).catch(() => { /* best-effort */ });
            }
        } catch (_) { /* swallow */ }
    }, true);
})();
"""


CHAT_JS = r"""
(function () {
    const log = document.getElementById('chat-log');
    const form = document.getElementById('chat-form');
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send');
    if (!log || !form || !input) return;

    function escapeHtml(s) {
        return (s || "")
            .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
    }

    function autoscroll() {
        // Stay pinned to the bottom while the answer streams in.
        log.scrollTop = log.scrollHeight;
    }

    function renderUser(text) {
        const wrap = document.createElement('div');
        wrap.className = 'chat-msg chat-user';
        const bub = document.createElement('div');
        bub.className = 'chat-bubble';
        bub.textContent = text;
        wrap.appendChild(bub);
        log.appendChild(wrap);
        autoscroll();
    }

    function newAssistantTurn() {
        const wrap = document.createElement('div');
        wrap.className = 'chat-msg chat-assistant';
        const bub = document.createElement('div');
        bub.className = 'chat-bubble chat-typing';
        wrap.appendChild(bub);
        const events = document.createElement('div');
        events.className = 'chat-events';
        wrap.appendChild(events);
        const cites = document.createElement('div');
        cites.className = 'chat-citations';
        wrap.appendChild(cites);
        log.appendChild(wrap);
        autoscroll();
        return { bub, events, cites };
    }

    function renderEvent(evEl, ev) {
        if (ev.kind === 'search') {
            const line = document.createElement('div');
            line.className = 'chat-event-search';
            line.textContent = `searching: ${ev.data.query} (k=${ev.data.k})`;
            evEl.appendChild(line);
        } else if (ev.kind === 'results') {
            const line = document.createElement('div');
            line.className = 'chat-event-result';
            line.textContent = `→ ${ev.data.length} chunk(s)`;
            evEl.appendChild(line);
        }
        autoscroll();
    }

    function renderCitations(citEl, citations) {
        citEl.innerHTML = '';
        if (!citations || citations.length === 0) return;
        const header = document.createElement('div');
        header.className = 'muted';
        header.style.cssText = 'font-size:11px;letter-spacing:0.06em;text-transform:uppercase;';
        header.textContent = `Sources (${citations.length})`;
        citEl.appendChild(header);
        citations.forEach((c) => {
            const card = document.createElement('div');
            card.className = 'chat-citation' + (c.kind === 'web' ? ' chat-citation-web' : '');
            const link = document.createElement('a');
            if (c.kind === 'web') {
                // External link to the source page; open in new tab so the
                // chat doesn't navigate away.
                link.href = c.url || c.file_path;
                link.target = '_blank';
                link.rel = 'noopener noreferrer';
                link.textContent = c.page_title || c.url || c.file_path;
            } else {
                link.href = `/file?path=${encodeURIComponent(c.file_path)}`;
                link.textContent = c.file_path + (c.chunk_index !== undefined && c.chunk_index !== null ? ` · chunk ${c.chunk_index}` : '');
            }
            card.appendChild(link);
            if (c.kind === 'web' && (c.url || c.file_path)) {
                const sub = document.createElement('div');
                sub.className = 'chat-citation-suburl';
                sub.textContent = c.url || c.file_path;
                card.appendChild(sub);
            }
            if (c.text) {
                const sn = document.createElement('div');
                sn.className = 'chat-citation-snippet';
                sn.textContent = c.text;
                card.appendChild(sn);
            }
            citEl.appendChild(card);
        });
    }

    async function send(message) {
        renderUser(message);
        const { bub, events, cites } = newAssistantTurn();
        const fd = new FormData();
        fd.append('message', message);
        let resp;
        try {
            resp = await fetch('/api/chat/message', { method: 'POST', body: fd });
        } catch (e) {
            bub.classList.remove('chat-typing');
            bub.textContent = `[network error] ${e}`;
            return;
        }
        if (!resp.ok || !resp.body) {
            bub.classList.remove('chat-typing');
            bub.textContent = `[error] HTTP ${resp.status}`;
            return;
        }
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';
        let assembled = '';
        let doneSeen = false;
        while (!doneSeen) {
            const { value, done } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });
            const lines = buf.split('\n');
            buf = lines.pop() || '';
            for (const ln of lines) {
                if (!ln.startsWith('data:')) continue;
                const payload = ln.slice(5).trim();
                if (payload === '[DONE]') { doneSeen = true; break; }
                let ev;
                try { ev = JSON.parse(payload); } catch (_) { continue; }
                if (ev.kind === 'text') {
                    assembled += ev.data;
                    bub.textContent = assembled;
                    autoscroll();
                } else if (ev.kind === 'search' || ev.kind === 'results') {
                    renderEvent(events, ev);
                } else if (ev.kind === 'done') {
                    if (ev.data && ev.data.text) {
                        assembled = ev.data.text;
                        bub.textContent = assembled;
                    }
                    renderCitations(cites, ev.data && ev.data.citations);
                } else if (ev.kind === 'meta') {
                    // First message of a fresh conversation: server tells us
                    // the new conversation id. Update the URL (and the page
                    // title bar) without forcing a full reload, so the user
                    // can deep-link or refresh and stay on the same chat.
                    if (ev.data && ev.data.created_now && ev.data.cid) {
                        try {
                            window.history.replaceState({}, '', `/chat/${ev.data.cid}`);
                        } catch (_) { /* old browser; harmless */ }
                    }
                } else if (ev.kind === 'error') {
                    assembled += `\n[error] ${ev.data}`;
                    bub.textContent = assembled;
                }
            }
        }
        bub.classList.remove('chat-typing');
    }

    form.addEventListener('submit', (e) => {
        e.preventDefault();
        const text = (input.value || '').trim();
        if (!text) return;
        input.value = '';
        sendBtn.disabled = true;
        send(text).finally(() => { sendBtn.disabled = false; input.focus(); });
    });
    // Cmd/Ctrl-Enter to send.
    input.addEventListener('keydown', (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
            e.preventDefault(); form.requestSubmit();
        }
    });
    // System-prompt editor toggle.
    const editLink = document.getElementById('chat-edit-prompt');
    const spForm = document.getElementById('chat-sp-form');
    if (editLink && spForm) {
        editLink.addEventListener('click', (e) => {
            e.preventDefault();
            spForm.style.display = spForm.style.display === 'none' ? 'flex' : 'none';
        });
    }
    autoscroll();
})();
"""


PALETTE_JS = r"""
(function() {
    const backdrop = document.getElementById('palette-backdrop');
    const input    = document.getElementById('palette-input');
    const results  = document.getElementById('palette-results');
    const opener   = document.getElementById('open-palette');

    let items = [];        // flat list of selectable items (in order)
    let selectedIdx = 0;
    let abortCtrl = null;

    // ⌘K can now jump to any page in the app — not just the small
    // hand-picked subset the original palette had. Order roughly by
    // most-likely-to-want; fuzzy match takes care of the rest.
    const STATIC_PAGES = [
        {kind: 'page', icon: '⌂', label: 'Overview',   href: '/',              meta: ''},
        {kind: 'page', icon: '☼', label: 'Brief',      href: '/brief',         meta: 'morning'},
        {kind: 'page', icon: '⌕', label: 'Search',     href: '/search',        meta: ''},
        {kind: 'page', icon: '⌨', label: 'Chat',       href: '/chat',          meta: 'ask the brain'},
        {kind: 'page', icon: '✓', label: 'Tasks',      href: '/tasks',         meta: ''},
        {kind: 'page', icon: '✉', label: 'Drafts',     href: '/drafts',        meta: 'email'},
        {kind: 'page', icon: '🤝', label: 'Thanks',     href: '/thanks',        meta: 'meeting follow-up'},
        {kind: 'page', icon: '📋', label: 'Prep',       href: '/prep',          meta: 'upcoming meeting prep'},
        {kind: 'page', icon: '!', label: 'Insights',   href: '/insights',      meta: ''},
        {kind: 'page', icon: '◇', label: 'Habits',     href: '/habits',        meta: ''},
        {kind: 'page', icon: '✎', label: 'Journal',    href: '/journal',       meta: ''},
        {kind: 'page', icon: '♥', label: 'Health',     href: '/health',        meta: 'oura'},
        {kind: 'page', icon: '☺', label: 'People',     href: '/people',        meta: ''},
        {kind: 'page', icon: '⌘', label: 'Memory',     href: '/memory',        meta: 'chat recall'},
        {kind: 'page', icon: '⌘', label: 'Projects',   href: '/projects',      meta: ''},
        {kind: 'page', icon: '?', label: 'Study',      href: '/study/review',  meta: 'flashcards'},
        {kind: 'page', icon: '⏱', label: 'Snapshots',  href: '/snapshots',     meta: 'temporal'},
        {kind: 'page', icon: '◊', label: 'Graph',      href: '/graph',         meta: ''},
        {kind: 'page', icon: '※', label: 'Entities',   href: '/entities',      meta: ''},
        {kind: 'page', icon: '⊟', label: 'Folders',    href: '/folders',       meta: ''},
        {kind: 'page', icon: '◉', label: 'Watch',      href: '/watch',         meta: 'watchlists'},
        {kind: 'page', icon: '☐', label: 'Queue',      href: '/queue',         meta: 'reading'},
        {kind: 'page', icon: '✦', label: 'Apps',       href: '/applications',  meta: 'job apps'},
        {kind: 'page', icon: '☼', label: 'Briefings',  href: '/briefings',     meta: 'pre-meeting'},
        {kind: 'page', icon: '☼', label: 'Daily',      href: '/briefing',      meta: 'briefing'},
        {kind: 'page', icon: '#', label: 'Queries',    href: '/queries',       meta: 'history'},
        {kind: 'page', icon: '⊕', label: 'Audit',      href: '/audit',         meta: 'AI actions log'},
        {kind: 'page', icon: '⚙', label: 'Diagnostics', href: '/health/system', meta: 'integration health'},
        {kind: 'page', icon: '+', label: 'Ingest URL', href: '/ingest',        meta: ''},
    ];

    function open() {
        backdrop.classList.add('open');
        input.value = '';
        input.focus();
        renderResults('');
    }
    function close() {
        backdrop.classList.remove('open');
    }
    function isOpen() { return backdrop.classList.contains('open'); }

    function escapeHtml(s) {
        return String(s).replace(/[&<>"']/g, c => (
            {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]
        ));
    }

    function fuzzy(q, s) {
        // tiny fuzzy: lowercased substring match. Good enough for nav + small lists.
        if (!q) return true;
        return s.toLowerCase().includes(q.toLowerCase());
    }

    function renderResults(query) {
        const q = query.trim();
        const groups = [];
        items = [];

        // Pages first (always shown, filtered by query)
        const pages = STATIC_PAGES.filter(p => fuzzy(q, p.label));
        if (pages.length) {
            groups.push({title: 'Pages', items: pages});
            for (const p of pages) items.push({...p, action: () => location.href = p.href});
        }

        // Search action — always present if there's a query
        if (q.length >= 2) {
            const sa = {kind: 'search', icon: '⌕',
                        label: 'Search the brain for "' + q + '"',
                        href: '/search?q=' + encodeURIComponent(q),
                        meta: 'enter'};
            groups.push({title: 'Action', items: [sa]});
            items.push({...sa, action: () => location.href = sa.href});
        }

        renderGroups(groups);

        // Server-side: entities + files (only when query is non-trivial)
        if (q.length >= 2) {
            if (abortCtrl) abortCtrl.abort();
            abortCtrl = new AbortController();
            fetch('/api/palette?q=' + encodeURIComponent(q), {signal: abortCtrl.signal})
                .then(r => r.json())
                .then(data => {
                    if (q !== input.value.trim()) return; // stale response
                    if (data.entities && data.entities.length) {
                        groups.push({title: 'Entities', items: data.entities});
                        for (const e of data.entities) items.push({...e, action: () => location.href = e.href});
                    }
                    if (data.files && data.files.length) {
                        groups.push({title: 'Files', items: data.files});
                        for (const f of data.files) items.push({...f, action: () => location.href = f.href});
                    }
                    renderGroups(groups);
                })
                .catch(() => { /* aborted */ });
        }
    }

    function renderGroups(groups) {
        let html = '';
        let idx = 0;
        for (const g of groups) {
            html += '<div class="group-title">' + escapeHtml(g.title) + '</div>';
            for (const it of g.items) {
                const cls = idx === selectedIdx ? 'item selected' : 'item';
                html += '<div class="' + cls + '" data-idx="' + idx + '">'
                     + '<span class="icon">' + (it.icon || '·') + '</span>'
                     + '<span>' + escapeHtml(it.label) + '</span>'
                     + (it.meta ? '<span class="item-meta">' + escapeHtml(it.meta) + '</span>' : '')
                     + '</div>';
                idx++;
            }
        }
        if (idx === 0) html = '<div class="empty">No results.</div>';
        results.innerHTML = html;

        results.querySelectorAll('.item').forEach(el => {
            el.addEventListener('click', () => {
                const i = parseInt(el.dataset.idx, 10);
                if (items[i] && items[i].action) items[i].action();
            });
            el.addEventListener('mouseenter', () => {
                selectedIdx = parseInt(el.dataset.idx, 10);
                results.querySelectorAll('.item').forEach((e2, j) => {
                    e2.classList.toggle('selected', j === selectedIdx);
                });
            });
        });
    }

    document.addEventListener('keydown', (e) => {
        const cmdK = (e.key === 'k' || e.key === 'K') && (e.metaKey || e.ctrlKey);
        if (cmdK) { e.preventDefault(); open(); return; }

        if (!isOpen()) {
            // Page-level shortcut: '/' focuses search if not in an input
            if (e.key === '/' && !['INPUT','TEXTAREA'].includes(document.activeElement.tagName)) {
                e.preventDefault(); open();
            }
            return;
        }

        if (e.key === 'Escape') { close(); return; }
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            selectedIdx = Math.min(items.length - 1, selectedIdx + 1);
            updateSelection();
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            selectedIdx = Math.max(0, selectedIdx - 1);
            updateSelection();
        } else if (e.key === 'Enter') {
            e.preventDefault();
            if (items[selectedIdx] && items[selectedIdx].action) items[selectedIdx].action();
        }
    });

    function updateSelection() {
        results.querySelectorAll('.item').forEach((el, i) => {
            el.classList.toggle('selected', i === selectedIdx);
            if (i === selectedIdx) el.scrollIntoView({block: 'nearest'});
        });
    }

    input.addEventListener('input', () => {
        selectedIdx = 0;
        renderResults(input.value);
    });

    backdrop.addEventListener('click', (e) => {
        if (e.target === backdrop) close();
    });

    if (opener) opener.addEventListener('click', open);
})();
"""


# Primary nav — kept tight (6 items) so it doesn't wrap. Items with a
# pending-state count get a [data-badge] span that JS populates from
# /api/nav-counts on page load. Anything not on this short list lives
# in the "More ▾" dropdown below or is reachable via ⌘K.
_PRIMARY_NAV = [
    # (label, href, badge_key) — badge_key=None means no count chip
    # Round 22 (Phase EA-UI) — Today is the new EA-style landing.
    # Replaces "Brief" as the first nav item; Brief stays in the
    # More dropdown for the comprehensive emailable digest.
    ("Today",    "/today",    None),
    ("Review",   "/review",   None),  # Round 16 (Phase B) — weekly letter
    ("Chat",     "/chat",     None),
    ("Tasks",    "/tasks",    "tasks"),
    ("Search",   "/search",   None),
    ("Drafts",   "/drafts",   "drafts"),
    ("Thanks",   "/thanks",   "thanks"),
    # Round 16 (Phase C) — proactive notifications.
    ("Inbox",    "/notifications", "notifications"),
]

# Overflow nav — grouped by purpose so users can scan to the right
# section instead of reading 25 alphabetical labels. Each tuple:
# (group_title, [(label, href), ...]).
_NAV_GROUPS = [
    # Round 21 fix (audit-found gap I1) — surface the EA features
    # in the More dropdown. Round 19+20 added 6 pages
    # (/followups, /agenda, /triage, /capture, /scheduling, /eod)
    # but none were in nav, making them effectively undiscoverable.
    ("EA", [
        ("Follow-ups", "/followups"),
        ("Agenda",     "/agenda"),
        ("Triage",     "/triage"),
        ("Capture",    "/capture"),
        ("Scheduling", "/scheduling"),
        ("EOD",        "/eod"),
        # Round 22 — moved from primary nav. Today is the new EA
        # front-door; Brief stays here for the full emailable digest.
        ("Brief",      "/brief"),
    ]),
    ("Personal", [
        ("Habits",   "/habits"),
        ("Journal",  "/journal"),
        ("Health",   "/health"),
        ("People",   "/people"),
        ("Memory",   "/memory"),
        ("Timeline", "/timeline"),  # Round 16 (Phase G)
    ]),
    ("Knowledge", [
        ("Insights",  "/insights"),
        ("Projects",  "/projects"),
        ("Study",     "/study/review"),
        ("Snapshots", "/snapshots"),
        ("Graph",     "/graph"),
        ("Entities",  "/entities"),
    ]),
    ("Sources", [
        ("Prep",       "/prep"),
        ("Watch",      "/watch"),
        ("Queue",      "/queue"),
        ("Apps",       "/applications"),
        ("Briefings",  "/briefings"),
        ("Daily",      "/briefing"),
    ]),
    ("System", [
        ("Overview",     "/"),
        ("Diagnostics",  "/health/system"),
        ("Audit",        "/audit"),
        ("Folders",      "/folders"),
        ("Queries",      "/queries"),
        ("Ingest",       "/ingest"),
    ]),
]


# Round 13 — module-level callback set by create_app() so _layout can
# fetch initial badges (most importantly: stale health failures) at
# render time without each route having to compute + pass them. The
# JS fetch remains the live source; this just bridges the cold-load
# gap so a broken integration is visible at first paint.
_INITIAL_BADGES_FN: object | None = None


def _is_same_origin_request(request: Request) -> bool:
    """Round 14 (audit-found gap M1) — same-origin guard helper.

    Lifted from the inlined Origin/Referer check that round 13
    sprinkled across ~12 mutation routes (and the audit then found
    11 *more* mutation routes that hadn't been guarded at all).
    Centralising it here means future routes get the protection by
    default and the policy lives in exactly one place.

    Policy: at least one of Origin / Referer must start with
    ``http://127.0.0.1`` or ``http://localhost``. The dashboard binds
    127.0.0.1 by default (see ``run_dashboard``), so cross-origin
    pages can submit forms but won't have a same-origin Referer or
    Origin — both of which are set by the browser itself and aren't
    forge-able from JS in cross-origin contexts.
    """
    origin = request.headers.get("origin", "")
    referer = request.headers.get("referer", "")
    same_origin_prefixes = ("http://127.0.0.1", "http://localhost")
    return (
        any(origin.startswith(p) for p in same_origin_prefixes)
        or any(referer.startswith(p) for p in same_origin_prefixes)
    )


def _layout(
    title: str, body: str, active: str = "",
    *, initial_badges: dict | None = None,
) -> str:
    """Render the page chrome.

    Round 13 fix (audit-found cold-load gap) — ``initial_badges`` lets
    routes pre-render badge state at first paint so a stale health
    failure shows the urgent indicator immediately rather than waiting
    on the /api/nav-counts JS fetch. Without this, JS-disabled or
    slow-network users never saw the badge at all.

    Routes that pass ``initial_badges=None`` (the default) get badges
    computed via the module-level ``_INITIAL_BADGES_FN`` registered
    by ``create_app``. Tests that import _layout standalone get
    empty badges — same as the pre-round-13 behavior.

    The dict shape mirrors /api/nav-counts.
    """
    if initial_badges is None and _INITIAL_BADGES_FN is not None:
        try:
            initial_badges = _INITIAL_BADGES_FN()  # type: ignore[operator]
        except Exception:  # noqa: BLE001
            initial_badges = None
    badges = initial_badges or {}
    urgent = badges.get("urgent", {}) if isinstance(badges, dict) else {}

    def _badge_attrs(key: str) -> str:
        n = badges.get(key, 0) if isinstance(badges, dict) else 0
        if not isinstance(n, int) or n <= 0:
            return ""
        cls = "nav-badge has-count"
        if urgent.get(key):
            cls += " urgent"
        return (
            f' class="{cls}" data-badge="{key}" data-initial-count="{n}"'
        )

    # Primary nav items — first slug after "/" is the active marker.
    def _primary_badge(badge: str | None) -> str:
        if not badge:
            return ""
        attrs = _badge_attrs(badge)
        if attrs:
            n = badges.get(badge, 0)
            label = "99+" if n > 99 else str(n)
            return f'<span{attrs}>{label}</span>'
        return f'<span class="nav-badge" data-badge="{badge}"></span>'

    primary_html = "".join(
        f'<a href="{href}" '
        f'class="{"active" if href.split("/")[1] == active else ""}">'
        f'{escape(name)}{_primary_badge(badge)}</a>'
        for name, href, badge in _PRIMARY_NAV
    )
    # Health badge in the More dropdown — server-rendered when stale.
    health_attrs = _badge_attrs("health")
    if health_attrs:
        n = badges.get("health", 0)
        more_health_badge = (
            f'<span{health_attrs}>{"99+" if n > 99 else n}</span>'
        )
    else:
        more_health_badge = (
            '<span class="nav-badge urgent" data-badge="health"></span>'
        )
    # Overflow dropdown — grouped pages reached via "More ▾".
    more_html = "".join(
        '<div class="nav-more-group">'
        f'<h4>{escape(title_g)}</h4>'
        + "".join(
            f'<a href="{href}">{escape(name)}</a>'
            for name, href in items
        )
        + '</div>'
        for title_g, items in _NAV_GROUPS
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
        <div class="brand"><a href="/" style="color:inherit;">second-brain</a></div>
        <nav>
            {primary_html}
            <details class="nav-more">
                <summary>More ▾{more_health_badge}</summary>
                <div class="nav-more-pop">{more_html}</div>
            </details>
        </nav>
        <div class="spacer"></div>
        <button class="kbd-hint" id="open-palette" title="Open command palette">
            <span>Search</span> <span class="kbd">⌘K</span>
        </button>
    </header>
    <main>{body}</main>

    <!-- Command palette -->
    <div id="palette-backdrop">
        <div id="palette" role="dialog" aria-modal="true">
            <input id="palette-input" placeholder="Search entities, files, pages…" autocomplete="off" spellcheck="false">
            <div id="palette-results"></div>
            <div id="palette-footer">
                <span><span class="kbd">↑↓</span> navigate</span>
                <span><span class="kbd">↵</span> open</span>
                <span><span class="kbd">esc</span> close</span>
            </div>
        </div>
    </div>
    <script>{PALETTE_JS}</script>
    <script>{NAV_BADGES_JS}</script>
    <script>{CLICK_BEACON_JS}</script>
</body>
</html>
"""


def _result_block(
    r, source_label: str = "search", *, tldr: str | None = None,
) -> str:
    sources = "+".join(r.sources) if r.sources else "rerank"
    age = ""
    if r.mtime is not None:
        days = (time.time() - r.mtime) / 86400
        age = f" · {days:.1f}d ago"
    # Phase 88 polish: snippet preview goes through redact so SSNs /
    # API keys / JWTs don't render to the dashboard.
    from .safety import redact_text as _redact
    snippet = escape(_redact(
        r.text if len(r.text) <= 1500 else r.text[:1500] + "…",
    ))
    # data-* attributes drive the click-feedback beacon. The JS in CLICK_JS
    # listens for clicks anywhere inside .result and POSTs /api/click with
    # the path/chunk_id; subsequent searches lift recently-opened paths.
    file_link = (
        f'<a href="/file?path={urllib.parse.quote_plus(r.file_path)}" '
        f'data-sb-click="1" data-sb-path="{escape(r.file_path)}" '
        f'data-sb-chunk="{r.chunk_id}" data-sb-source="{escape(source_label)}">'
        f"{escape(r.file_path)}</a>"
    )
    # Phase 74: TL;DR rendered as a one-line italic above the snippet
    # when present. Conditional so we don't waste vertical space on
    # docs without summaries.
    tldr_html = (
        f'<div class="tldr"><em>TL;DR:</em> {escape(tldr)}</div>'
        if tldr else ""
    )
    return f"""
<article class="result">
    <h3>{file_link}</h3>
    <div class="meta">chunk {r.chunk_index} · {sources}{age} · score {r.score:.4f}</div>
    {tldr_html}
    <div class="snippet">{snippet}</div>
</article>"""


def _render_decision(d) -> str:
    """Round 22 — render one Decision card on /today.

    ``d`` is a ``today.Decision``. Renders a card with an icon,
    title, why-line, and inline action buttons. Primary action is
    a styled button; secondary actions are plain links/forms.
    """
    def _btn(action, *, primary: bool = False) -> str:
        css = "today-btn"
        if primary or action.style == "primary":
            css += " today-btn-primary"
        elif action.style == "subtle":
            css += " today-btn-subtle"
        if action.method.upper() == "POST":
            # Round 14 same-origin guard requires Referer; this is
            # a same-origin form, so it'll match.
            # Round 23 fix (audit-found gap M9) — use parse_qsl so
            # a query value that's already URL-encoded (e.g.
            # ``label=Snoozed%207%20days``) gets decoded ONCE here
            # and the browser re-encodes ONCE on submit. Without
            # this, ``%20`` round-trips as ``%2520``.
            from urllib.parse import parse_qsl
            href, _sep, query = action.href.partition("?")
            hidden = ""
            if query:
                for k, v in parse_qsl(query, keep_blank_values=True):
                    hidden += (
                        f'<input type="hidden" '
                        f'name="{escape(k)}" '
                        f'value="{escape(v)}">'
                    )
            # Round 23 — also pass next= so /today's actions stay
            # on /today after the round-trip.
            hidden += (
                '<input type="hidden" name="next" value="/today">'
            )
            return (
                f'<form method="post" action="{escape(href)}" '
                f'class="today-action-form">'
                f'  {hidden}'
                f'  <button type="submit" class="{css}">'
                f'    {escape(action.label)}'
                f'  </button>'
                f'</form>'
            )
        return (
            f'<a class="{css}" href="{escape(action.href)}">'
            f'{escape(action.label)}</a>'
        )

    actions_html = _btn(d.primary, primary=True)
    for s in d.secondary:
        actions_html += _btn(s)
    return (
        f'<div class="today-decision">'
        f'  <div class="today-decision-icon">{escape(d.icon)}</div>'
        f'  <div class="today-decision-body">'
        f'    <div class="today-decision-title">'
        f'      {escape(_safe(d.title))}'
        f'    </div>'
        f'    <div class="today-decision-why muted">'
        f'      {escape(_safe(d.why))}'
        f'    </div>'
        f'    <div class="today-decision-actions">{actions_html}</div>'
        f'  </div>'
        f'</div>'
    )


def _undo_toast_html(
    undo_done: int, undo_label: str,
    undo_kind: str, undo_id: int,
) -> str:
    """Round 22 — server-rendered undo toast strip.

    Renders an inline "✓ <label> [undo]" banner at the top of the
    page when the previous action redirected here with the undo
    query params. The undo button POSTs back to a paired route
    (e.g. /triage/{id}/undo) which reverses the action.

    Round 23 fix (audit-found gap M8) — inline JS calls
    ``history.replaceState`` to strip the undo_* query params from
    the URL bar AFTER the toast renders. Without this, refreshing
    the page 5 minutes later re-rendered the same "Marked done"
    toast misleadingly.
    """
    from html import escape as _esc
    if not undo_done or not undo_label:
        return ""
    undo_form = ""
    if undo_kind == "triage" and undo_id:
        undo_form = (
            f'<form method="post" '
            f'action="/triage/{int(undo_id)}/undo" '
            f'class="toast-undo-form">'
            f'  <button type="submit" class="toast-undo-btn">'
            f'    undo</button></form>'
        )
    # Inline script strips the undo_* params after render so a
    # refresh doesn't re-show the toast. Falls back gracefully if
    # history.replaceState is unavailable (very old browsers).
    cleanup_js = """
<script>
(function() {
    try {
        if (history && history.replaceState) {
            var u = new URL(window.location.href);
            ['undo_done','undo_label','undo_kind','undo_id'].forEach(
                function(k){ u.searchParams.delete(k); }
            );
            history.replaceState({}, '', u.toString());
        }
    } catch (_) { /* swallow */ }
})();
</script>"""
    return (
        f'<div class="toast">'
        f'  <span class="toast-check">✓</span>'
        f'  <span>{_esc(_safe(undo_label))}</span>'
        f'  {undo_form}'
        f'</div>'
        f'{cleanup_js}'
    )


def _render_triage_walkthrough(
    *, queue, done_today: int,
    undo_done: int = 0, undo_label: str = "",
    undo_kind: str = "", undo_id: int = 0,
):
    """Round 22 — single-email decision walkthrough for /triage.

    Shows ONE email at a time with action buttons inline. After
    any action POSTs, the queue rebuilds (the acted-on email drops
    out via triage_state) and the next item is the new top.

    Pattern: like an EA walking you through your inbox one decision
    at a time, not a database list view.
    """
    from html import escape as _esc

    from fastapi.responses import HTMLResponse
    toast_html = _undo_toast_html(
        undo_done, undo_label, undo_kind, undo_id,
    )
    it = queue[0]
    remaining = len(queue)
    vip_badge = (
        '<span class="tag urgent">VIP</span>'
        if it.is_vip else ""
    )
    label_class = {
        "urgent": "urgent", "follow_up": "warn",
        "review": "muted", "fyi": "muted",
    }.get(it.label, "muted")
    age_str = (
        f"{int(it.age_hours)}h ago"
        if it.age_hours and it.age_hours < 24
        else f"{int(it.age_hours / 24)}d ago"
    )
    sender = it.from_display or it.from_email or "(unknown)"
    subject = it.subject or "(no subject)"
    preview = (it.body_preview or "")[:400]
    # The actions row.
    actions = []
    if it.draft_id:
        actions.append(
            f'<form method="post" '
            f'action="/drafts/{it.draft_id}/sent" '
            f'class="walk-action-form">'
            f'  <button class="walk-btn walk-btn-primary">'
            f'    Send draft</button></form>'
        )
        actions.append(
            f'<a class="walk-btn" href="/drafts#d{it.draft_id}">'
            f'Edit draft</a>'
        )
    else:
        actions.append(
            f'<a class="walk-btn walk-btn-primary" '
            f'href="/file?file_id={it.file_id}">Open</a>'
        )
    actions.append(
        f'<form method="post" action="/triage/{it.file_id}/done" '
        f'class="walk-action-form">'
        f'  <button class="walk-btn">Mark done</button></form>'
    )
    actions.append(
        f'<form method="post" action="/triage/{it.file_id}/snooze" '
        f'class="walk-action-form">'
        f'  <input type="hidden" name="hours" value="24">'
        f'  <button class="walk-btn walk-btn-subtle">'
        f'  Snooze 1d</button></form>'
    )
    actions.append(
        f'<form method="post" action="/triage/{it.file_id}/skip" '
        f'class="walk-action-form">'
        f'  <button class="walk-btn walk-btn-subtle">'
        f'  Not for me</button></form>'
    )

    progress_chip = ""
    if done_today:
        progress_chip = (
            f'<span class="walk-progress-chip">'
            f'  ✓ {done_today} cleared today'
            f'</span>'
        )

    body = f"""
{toast_html}
<div class="walk-header">
  <h1 class="walk-h1">Triage</h1>
  <span class="walk-counter">{remaining} left</span>
  {progress_chip}
  <a class="walk-show-all muted" href="/triage?show=all">show all</a>
</div>

<div class="walk-card">
  <div class="walk-meta">
    <span class="tag {label_class}">{_esc(it.label or "?")}</span>
    {vip_badge}
    <span class="muted">{_esc(_safe(sender))}</span>
    <span class="muted">·</span>
    <span class="muted">{age_str}</span>
  </div>
  <h2 class="walk-subject">{_esc(_safe(subject))}</h2>
  <p class="walk-preview muted">{_esc(_safe(preview))}</p>
  <div class="walk-actions">{"".join(actions)}</div>
</div>

<style>
.walk-header {{
    display: flex; align-items: center; gap: var(--s-3);
    margin-bottom: var(--s-4);
}}
.walk-h1 {{ margin: 0; }}
.walk-counter {{
    font-size: 11px; color: var(--text-3);
    background: var(--bg-card); padding: 2px 8px;
    border-radius: var(--r); border: 1px solid var(--border);
}}
.walk-progress-chip {{
    font-size: 11px; color: var(--green);
    background: var(--green-soft); padding: 2px 8px;
    border-radius: var(--r); border: 1px solid var(--green-dim);
}}
.walk-show-all {{ margin-left: auto; font-size: 11px; }}
.walk-card {{
    background: var(--bg-card); border: 1px solid var(--border-strong);
    border-radius: var(--r); padding: var(--s-5);
    box-shadow: var(--shadow-glow);
}}
.walk-meta {{ font-size: 11.5px; margin-bottom: var(--s-2); }}
.walk-subject {{
    margin: var(--s-2) 0 var(--s-3); font-size: 18px;
    font-weight: 500; color: var(--text);
}}
.walk-preview {{
    font-size: 13px; line-height: 1.6; max-height: 200px;
    overflow: auto; white-space: pre-wrap;
    background: var(--bg-input); padding: var(--s-3);
    border-radius: var(--r); margin: var(--s-3) 0;
    border: 1px solid var(--border);
}}
.walk-actions {{
    display: flex; gap: var(--s-2); flex-wrap: wrap;
    margin-top: var(--s-4);
}}
.walk-action-form {{ display: inline; margin: 0; }}
.walk-btn {{
    display: inline-block; padding: 6px 14px; font-size: 12.5px;
    background: var(--bg-elevated); color: var(--text-2);
    border: 1px solid var(--border-strong); border-radius: var(--r);
    cursor: pointer; font-family: var(--mono);
    transition: all var(--ease);
}}
.walk-btn:hover {{
    background: var(--bg-hover); color: var(--text);
    border-color: var(--text-3);
}}
.walk-btn-primary {{
    background: var(--green-soft); color: var(--green);
    border-color: var(--green-dim);
}}
.walk-btn-primary:hover {{
    background: var(--green-dim); color: #000;
}}
.walk-btn-subtle {{ color: var(--text-3); border-color: var(--border); }}
.tag {{
    font-size: 10px; padding: 1px 6px;
    border-radius: var(--r); margin-right: 4px;
}}
.tag.urgent {{ background: rgba(255,77,77,0.10); color: var(--red); }}
.tag.warn {{ background: rgba(255,183,0,0.10); color: var(--amber); }}
</style>
"""
    return HTMLResponse(_layout("Triage", body, "triage"))


def _today_styles() -> str:
    """CSS scoped to the /today page. Visual goal: warm, narrative,
    less dense than the database-grid pages elsewhere."""
    return """
<style>
.today-greet {
    font-size: 22px; font-weight: 500; margin: var(--s-3) 0 var(--s-5);
    color: var(--text);
}
.today-quiet {
    font-size: 16px; color: var(--text-2);
    margin-top: var(--s-7); text-align: center;
}
.today-section { margin: var(--s-5) 0; }
.today-h2 {
    font-size: 12px; font-weight: 600; color: var(--text-3);
    text-transform: uppercase; letter-spacing: 0.05em;
    margin: 0 0 var(--s-3); padding-bottom: var(--s-1);
    border-bottom: 1px solid var(--border);
}
.today-num {
    color: var(--green); font-weight: 700; font-size: 14px;
    margin-right: var(--s-1);
}
.today-decisions { display: flex; flex-direction: column; gap: var(--s-3); }
.today-decision {
    display: flex; gap: var(--s-3); padding: var(--s-3);
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: var(--r);
    transition: border-color var(--ease);
}
.today-decision:hover { border-color: var(--border-strong); }
.today-decision-icon {
    font-size: 18px; line-height: 1.2;
    flex-shrink: 0; width: 24px; text-align: center;
}
.today-decision-body { flex: 1; min-width: 0; }
.today-decision-title {
    font-weight: 500; font-size: 14px; margin-bottom: 3px;
    color: var(--text);
}
.today-decision-why {
    font-size: 12px; color: var(--text-2);
    font-style: italic; margin-bottom: var(--s-2);
}
.today-decision-actions {
    display: flex; gap: var(--s-2); flex-wrap: wrap;
    margin-top: var(--s-2);
}
.today-action-form { display: inline; margin: 0; }
.today-btn {
    display: inline-block; padding: 4px 10px; font-size: 11.5px;
    background: var(--bg-elevated); color: var(--text-2);
    border: 1px solid var(--border-strong); border-radius: var(--r);
    cursor: pointer; transition: all var(--ease);
    font-family: var(--mono);
}
.today-btn:hover {
    background: var(--bg-hover); color: var(--text);
    border-color: var(--text-3);
}
.today-btn-primary {
    background: var(--green-soft); color: var(--green);
    border-color: var(--green-dim);
}
.today-btn-primary:hover {
    background: var(--green-dim); color: #000;
    text-shadow: 0 0 6px var(--green-glow);
}
.today-btn-subtle {
    background: transparent; color: var(--text-3);
    border-color: var(--border);
}
.today-cal {
    list-style: none; padding: 0; margin: 0;
}
.today-cal li {
    padding: var(--s-2) 0;
    border-bottom: 1px dashed var(--border);
    font-size: 13px;
}
.today-cal li:last-child { border-bottom: none; }
.today-when {
    font-family: var(--mono); color: var(--amber);
    font-weight: 600; margin-right: var(--s-2);
    display: inline-block; min-width: 60px;
}
.today-prep {
    margin-left: var(--s-2); font-size: 11px;
    color: var(--green); text-decoration: underline;
}
.today-wk {
    list-style: none; padding: 0; margin: 0;
    display: flex; flex-direction: column; gap: var(--s-3);
}
.today-wk li {
    display: flex; align-items: flex-start; gap: var(--s-2);
    padding: var(--s-2) 0;
}
.today-wk-icon { font-size: 16px; flex-shrink: 0; }
.today-wk-body { flex: 1; min-width: 0; font-size: 13px; }
.today-why { font-size: 11.5px; }
.today-wk-action {
    font-size: 11px; padding: 2px 8px;
    background: transparent; color: var(--text-3);
    border: 1px solid var(--border); border-radius: var(--r);
    flex-shrink: 0;
}
.today-wk-action:hover { color: var(--text); border-color: var(--text-3); }
.today-end {
    margin-top: var(--s-7); font-size: 12px; text-align: center;
}
.toast {
    display: inline-flex; align-items: center; gap: var(--s-2);
    padding: var(--s-2) var(--s-3);
    background: var(--green-soft); color: var(--green);
    border: 1px solid var(--green-dim); border-radius: var(--r);
    font-size: 12px; margin-bottom: var(--s-3);
}
.toast-check { font-weight: 700; }
.toast-undo-form { display: inline; margin: 0 0 0 var(--s-2); }
.toast-undo-btn {
    background: transparent; border: 1px solid var(--green-dim);
    color: var(--green); padding: 1px 8px; border-radius: var(--r);
    font-size: 11px; cursor: pointer;
    font-family: var(--mono);
}
.toast-undo-btn:hover { background: var(--green-dim); color: #000; }
@media (max-width: 600px) {
    .today-decision { flex-direction: column; }
    .today-when { min-width: auto; }
}
</style>
"""


def _markdown_to_html_block(md: str) -> str:
    """Tiny markdown subset → HTML for the dashboard's brief view.
    Mirrors the email rendering in ``daily_brief._minimal_md_to_html``
    but kept module-local so the dashboard doesn't import that
    helper (would create a daily_brief → dashboard import loop in
    some refactors)."""
    out: list[str] = []
    in_list = False
    for raw_line in md.splitlines():
        line = raw_line.rstrip()
        if not line:
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append("")
            continue
        # Heading?
        if line.startswith("# "):
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<h2>{escape(line[2:])}</h2>")
            continue
        if line.startswith("## "):
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<h3>{escape(line[3:])}</h3>")
            continue
        if line.startswith("### "):
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<h4>{escape(line[4:])}</h4>")
            continue
        # Bullet?
        stripped = line.lstrip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{escape(stripped[2:])}</li>")
            continue
        # Blockquote?
        if stripped.startswith("> "):
            if in_list:
                out.append("</ul>")
                in_list = False
            out.append(f"<blockquote>{escape(stripped[2:])}</blockquote>")
            continue
        # Plain paragraph.
        if in_list:
            out.append("</ul>")
            in_list = False
        out.append(f"<p>{escape(line)}</p>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


def _render_health_card(metric: str, summary, points) -> str:
    """One card per Oura headline metric — score, 14d avg, sparkline."""
    label_map = {
        "sleep_score": "Sleep",
        "readiness_score": "Readiness",
        "activity_score": "Activity",
    }
    label = label_map.get(metric, metric.replace("_", " ").title())
    latest_v = (
        f"{int(summary.latest.value)}"
        if summary.latest and abs(
            summary.latest.value - int(summary.latest.value)
        ) < 1e-6
        else (f"{summary.latest.value:.1f}" if summary.latest else "—")
    )
    avg_v = (
        f"{summary.average:.0f}" if summary.average is not None else "—"
    )
    delta_html = ""
    if summary.latest and summary.average:
        delta_pct = (summary.latest.value - summary.average) / summary.average * 100.0
        arrow = "↑" if delta_pct >= 5 else "↓" if delta_pct <= -5 else ""
        klass = (
            "good" if delta_pct >= 5 else "warn" if delta_pct <= -5 else ""
        )
        delta_html = (
            f'<span class="{klass}">{arrow} {delta_pct:+.0f}%</span>'
        )
    # Inline SVG sparkline for the trend.
    spark_html = _svg_sparkline([p.value for p in points], width=200, height=40)
    return f"""
<div class="card">
  <h2>{escape(label)}</h2>
  <div class="stat" style="font-size:32px;font-weight:600;">
    <span>{latest_v}</span>
    <span class="v" style="font-size:14px;">{delta_html}</span>
  </div>
  <div class="stat"><span class="muted">14-day avg</span>
    <span class="v">{avg_v}</span></div>
  <div class="stat"><span class="muted">days with data</span>
    <span class="v">{summary.n}</span></div>
  <div style="margin-top:12px;">{spark_html}</div>
</div>"""


def _render_summary_block(conn, file_id: int) -> str:
    """Phase 74 — TL;DR + key points card at the top of file view.
    Empty string when no summary exists yet."""
    try:
        from .synthesis import get_summary
    except ImportError:
        return ""
    try:
        s = get_summary(conn, file_id)
    except Exception:  # noqa: BLE001
        return ""
    if s is None:
        return ""
    points_html = ""
    if s.key_points:
        points_html = (
            "<ul style='margin:8px 0 0;'>"
            + "".join(f"<li>{escape(p)}</li>" for p in s.key_points[:5])
            + "</ul>"
        )
    return (
        "<div class='card' style='margin-bottom:16px;'>"
        "<h2>Summary</h2>"
        f"<p><strong>{escape(s.tldr)}</strong></p>"
        f"{points_html}"
        "</div>"
    )


def _render_annotations_block(conn, file_id: int) -> str:
    """Phase 84 — highlights / notes the user made on a PDF."""
    try:
        from .pdf_annotations import get_annotations
    except ImportError:
        return ""
    try:
        annots = get_annotations(conn, file_id)
    except Exception:  # noqa: BLE001
        return ""
    if not annots:
        return ""
    items: list[str] = []
    for a in annots[:30]:
        marker = {
            "highlight": "▌", "underline": "_",
            "strike": "—", "note": "✎",
        }.get(a.kind, "·")
        note_html = (
            f"<div class='muted' style='font-style:italic;margin-left:24px;'>"
            f"{escape(a.note)}</div>"
            if a.note else ""
        )
        items.append(
            f'<div class="stat">'
            f'<span><span class="muted">p{a.page}</span> '
            f'{marker} {escape(a.anchor)}</span></div>'
            f'{note_html}',
        )
    extra = (
        f'<p class="muted">+{len(annots) - 30} more</p>'
        if len(annots) > 30 else ""
    )
    return (
        f"<div class='card' style='margin-bottom:16px;'>"
        f"<h2>Annotations ({len(annots)})</h2>"
        + "".join(items)
        + extra
        + "</div>"
    )


def _render_citations_block(conn, file_id: int) -> str:
    """Phase 85 — outgoing + incoming citation graph."""
    try:
        from .pdf_annotations import get_citations_from, get_citations_to
    except ImportError:
        return ""
    try:
        outgoing = get_citations_from(conn, file_id)
        incoming = get_citations_to(conn, file_id)
    except Exception:  # noqa: BLE001
        return ""
    if not outgoing and not incoming:
        return ""
    sections: list[str] = []
    if outgoing:
        out_items = []
        for c in outgoing[:20]:
            year = f" ({c.year})" if c.year else ""
            link = ""
            if c.cited_file_id:
                p = conn.execute(
                    "SELECT path FROM files WHERE id = ?",
                    (c.cited_file_id,),
                ).fetchone()
                if p:
                    link = (
                        f' → <a href="/file?path='
                        f'{urllib.parse.quote_plus(p["path"])}">in brain</a>'
                    )
            out_items.append(
                f'<div class="stat">'
                f'<span>{escape(c.cited_text)}{year}{link}</span></div>',
            )
        sections.append(
            f"<div class='card'><h2>Cites ({len(outgoing)})</h2>"
            + "".join(out_items)
            + (f"<p class='muted'>+{len(outgoing) - 20} more</p>"
               if len(outgoing) > 20 else "")
            + "</div>",
        )
    if incoming:
        in_items = []
        for c in incoming[:20]:
            p = conn.execute(
                "SELECT path FROM files WHERE id = ?",
                (c.src_file_id,),
            ).fetchone()
            src_path = p["path"] if p else "?"
            in_items.append(
                f'<div class="stat">'
                f'<span><a href="/file?path='
                f'{urllib.parse.quote_plus(src_path)}">'
                f'{escape(src_path)}</a></span></div>',
            )
        sections.append(
            f"<div class='card'><h2>Cited by ({len(incoming)})</h2>"
            + "".join(in_items) + "</div>",
        )
    return "".join(sections)


def _render_backlinks_block(conn, path: str) -> str:
    """Render the 'See also' panel for a file view. Pulls from
    Phase 52's backlinks table — empty when the doc has no neighbours
    yet (e.g. just-ingested or under the min_chunks threshold)."""
    try:
        from .backlinks import get_backlinks_for_path
    except ImportError:
        return ""
    try:
        rows = get_backlinks_for_path(conn, path, limit=8)
    except Exception:  # noqa: BLE001
        return ""
    if not rows:
        return ""
    items: list[str] = []
    for r in rows:
        items.append(
            f'<div class="stat">'
            f'<span><a href="/file?path={urllib.parse.quote_plus(r.path)}">'
            f'{escape(r.title)}</a> '
            f'<span class="muted">{escape(r.path)}</span></span>'
            f'<span class="v">{r.percent}%</span>'
            f'</div>',
        )
    return (
        '<div class="card" style="margin-top:24px;">'
        '<h2>See also</h2>'
        + "".join(items)
        + '</div>'
    )


def _svg_sparkline(values: list[float], *, width: int, height: int) -> str:
    """Inline SVG line chart. No external deps — keeps the dashboard
    self-contained. Renders a flat line at the middle for constant
    series."""
    if not values:
        return ""
    if len(values) == 1:
        # Single point — just a dot.
        return (
            f'<svg viewBox="0 0 {width} {height}" '
            f'style="width:100%;height:{height}px;">'
            f'<circle cx="{width / 2}" cy="{height / 2}" r="3" '
            f'fill="currentColor"/></svg>'
        )
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0
    pad = 4
    inner_h = height - 2 * pad
    inner_w = width - 2 * pad
    pts = []
    for i, v in enumerate(values):
        x = pad + (i / (len(values) - 1)) * inner_w
        # Y axis flipped: SVG y grows downward, so subtract.
        y = pad + (1 - (v - lo) / span) * inner_h
        pts.append(f"{x:.1f},{y:.1f}")
    points_str = " ".join(pts)
    return (
        f'<svg viewBox="0 0 {width} {height}" '
        f'style="width:100%;height:{height}px;">'
        f'<polyline points="{points_str}" fill="none" '
        f'stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round"/></svg>'
    )


def _entity_link(text: str, label: str = "") -> str:
    href = "/entity?name=" + urllib.parse.quote_plus(text)
    label_html = f'<span class="label {label}">{label}</span>' if label else ""
    return f'{label_html}<a href="{href}">{escape(text)}</a>'


# --- App factory ---

def create_app():
    """Build the FastAPI app. Heavy imports happen here so importing dashboard
    is cheap when the user isn't using the dashboard."""
    from fastapi import FastAPI, Form, Query
    from fastapi.responses import HTMLResponse, RedirectResponse

    app = FastAPI(title="second-brain", docs_url=None, redoc_url=None)

    state = {
        "cfg": None,
        "conn": None,           # writer — chat history, ingest, click log
        "read_conn": None,      # read-only — search, file view, listings
        "embedder": None,
        "reranker": None,
    }

    def get_state():
        """Return the writer connection. Used by routes that mutate
        (chat send, ingest, click feedback, task add/done, settings)."""
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

    def get_read_state():
        """Return a read-only connection. Used by query / listing /
        search routes — won't contend with the daemon's writer
        transactions, and physically prevents accidental writes (an
        UPDATE/INSERT through this conn raises).

        Lazily opens the read-only conn on first use; caches it for
        the process lifetime since sqlite-vec needs ``load_extension``
        per connection."""
        if state["read_conn"] is None:
            from .db import connect_readonly
            # Make sure the writer / schema migrations have run first.
            get_state()
            state["read_conn"] = connect_readonly(state["cfg"].db_path)
        return (
            state["cfg"], state["read_conn"],
            state["embedder"], state["reranker"],
        )

    # Round 13 — register a server-render callback for nav badges so
    # _layout can show the urgent health indicator at first paint
    # (rather than waiting on the /api/nav-counts JS fetch). Cheap:
    # one COUNT query per page render, gracefully degrades to empty
    # on any error.
    def _compute_initial_badges() -> dict:
        out: dict = {
            "tasks": 0, "drafts": 0, "insights": 0,
            "thanks": 0, "health": 0,
            "urgent": {"drafts": False, "thanks": False, "health": False},
        }
        try:
            _, conn, _, _ = get_state()
        except Exception:  # noqa: BLE001
            return out
        # Health stale count is the most important — it's why we
        # added this. Count only; full state comes from the JS fetch.
        try:
            from . import health_checks
            stale = health_checks.stale_failures(conn)
            if stale:
                out["health"] = len(stale)
                out["urgent"]["health"] = True
        except Exception:  # noqa: BLE001
            pass
        return out

    global _INITIAL_BADGES_FN
    _INITIAL_BADGES_FN = _compute_initial_badges

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

        # Launchpad: every page in the app, grouped by purpose. Lets
        # the user reach anywhere from Overview without scanning a
        # 25-item top nav. Counts (tasks/drafts/insights) get JS-
        # populated badges via /api/nav-counts.
        launchpad_groups = [
            # Round 23 fix (audit-found gap H5) — renamed from
            # "Today" to "Daily" so it doesn't collide with the
            # round-22 ``/today`` page (which IS in primary nav).
            # Two surfaces called "Today" was a discoverability bug.
            ("Daily", [
                ("Today",     "/today",         None),
                ("Brief",     "/brief",         None),
                ("Chat",      "/chat",          None),
                ("Tasks",     "/tasks",         "tasks"),
                ("Drafts",    "/drafts",        "drafts"),
                ("Thanks",    "/thanks",        "thanks"),
                ("Insights",  "/insights",      "insights"),
                ("Search",    "/search",        None),
            ]),
            ("Personal", [
                ("Habits",    "/habits",        None),
                ("Journal",   "/journal",       None),
                ("Health",    "/health",        None),
                ("People",    "/people",        None),
                ("Memory",    "/memory",        None),
            ]),
            ("Knowledge", [
                ("Projects",  "/projects",      None),
                ("Study",     "/study/review",  None),
                ("Snapshots", "/snapshots",     None),
                ("Graph",     "/graph",         None),
                ("Entities",  "/entities",      None),
                ("Folders",   "/folders",       None),
            ]),
            ("Sources & system", [
                ("Prep",         "/prep",          None),
                ("Watch",        "/watch",         None),
                ("Queue",        "/queue",         None),
                ("Apps",         "/applications",  None),
                ("Briefings",    "/briefings",     None),
                ("Daily",        "/briefing",      None),
                ("Diagnostics",  "/health/system", None),
                ("Audit",        "/audit",         None),
                ("Queries",      "/queries",       None),
                ("Ingest",       "/ingest",        None),
            ]),
        ]
        launchpad_html = "".join(
            '<div class="launchpad-group">'
            f'<h3>{escape(title_g)}</h3>'
            + "".join(
                f'<a href="{href}">'
                f'<span>{escape(name)}</span>'
                + (f'<span class="meta nav-badge" data-badge="{badge}"></span>'
                   if badge else "")
                + '</a>'
                for name, href, badge in items
            )
            + '</div>'
            for title_g, items in launchpad_groups
        )

        body = f"""
<h1>Overview</h1>
<section class="launchpad">
    <div class="launchpad-grid">{launchpad_html}</div>
</section>
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
        # Read-only conn — search is the heaviest read path and runs
        # on every keystroke through the palette; using a read-only
        # snapshot lets it coexist with the daemon's write lock.
        cfg, conn, embedder, reranker = get_read_state()
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
            if results:
                # Phase 74: lazy-fetch TL;DRs for the result paths so
                # each block can render its summary preview without a
                # separate per-block query.
                summary_by_path: dict[str, str] = {}
                try:
                    paths = list({r.file_path for r in results})
                    placeholders = ",".join("?" * len(paths))
                    sum_rows = conn.execute(
                        f"SELECT f.path, s.tldr FROM doc_summaries s "
                        f"JOIN files f ON f.id = s.file_id "
                        f"WHERE f.path IN ({placeholders})",
                        paths,
                    ).fetchall()
                    summary_by_path = {
                        r["path"]: r["tldr"] for r in sum_rows
                    }
                except Exception:  # noqa: BLE001
                    pass
                results_html = "".join(
                    _result_block(r, tldr=summary_by_path.get(r.file_path))
                    for r in results
                )
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
    <label>Top entities</label>
    <input type="number" name="top_n" value="{top_n}" min="20" max="500" style="width: 80px;">
    <label>Min co-occurrences</label>
    <input type="number" name="min_cooccur" value="{min_cooccur}" min="1" max="20" style="width: 60px;">
    <button type="submit">Reload</button>
    <span class="muted">Hover to highlight neighbors · Click to open entity · Drag · Scroll to zoom</span>
</form>
<div class="graph-wrap">
    <div id="cy"></div>
    <div class="graph-overlay" id="legend">
        <h4>Filter by label</h4>
        <div id="legend-rows"></div>
    </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/cytoscape@3.30.4/dist/cytoscape.min.js"></script>
<script>
const LABEL_COLORS = {{
    PERSON: '#7fff7f', ORG: '#ffb700',
    GPE: '#5af0ff', LOC: '#5af0ff', FAC: '#5af0ff',
    PRODUCT: '#ff5af0', WORK_OF_ART: '#ff5af0',
    DATE: '#b8ffb8', MONEY: '#ffd566',
    EVENT: '#ff4d4d', LAW: '#ff4d4d', NORP: '#ff4d4d', LANGUAGE: '#ff4d4d',
}};

fetch('/graph/data?top_n={top_n}&min_cooccur={min_cooccur}').then(r => r.json()).then(data => {{
    const cy = cytoscape({{
        container: document.getElementById('cy'),
        elements: [...data.nodes, ...data.edges],
        style: [
            {{ selector: 'node', style: {{
                'background-color': 'data(color)',
                'label': 'data(label)',
                'color': '#d4d4c8',
                'font-size': 11,
                'font-weight': 600,
                'font-family': "'JetBrains Mono', 'SF Mono', 'Consolas', monospace",
                'width': 'data(size)',
                'height': 'data(size)',
                'text-outline-color': '#050505',
                'text-outline-width': 3,
                'text-margin-y': -3,
                'border-width': 0,
                'transition-property': 'opacity, border-width, background-color',
                'transition-duration': '180ms',
                'transition-timing-function': 'ease',
            }} }},
            {{ selector: 'edge', style: {{
                'width': 'data(weight)',
                'line-color': '#2a2a2a',
                'curve-style': 'haystack',
                'opacity': 0.55,
                'transition-property': 'opacity, line-color, width',
                'transition-duration': '180ms',
                'transition-timing-function': 'ease',
            }} }},
            {{ selector: '.dim',  style: {{ 'opacity': 0.06 }} }},
            {{ selector: '.neighbor', style: {{
                'opacity': 1.0,
                'border-width': 2.5,
                'border-color': '#7fff7f',
            }} }},
            {{ selector: 'edge.neighbor', style: {{
                'opacity': 0.95,
                'line-color': '#7fff7f',
            }} }},
            {{ selector: '.focus', style: {{
                'border-width': 3.5,
                'border-color': '#b8ffb8',
            }} }},
        ],
        layout: {{
            name: 'cose',
            idealEdgeLength: 110,
            nodeOverlap: 25,
            refresh: 20,
            fit: true,
            padding: 50,
            randomize: true,
            componentSpacing: 120,
            nodeRepulsion: 500000,
            edgeElasticity: 90,
            gravity: 70,
            numIter: 1200,
            initialTemp: 220,
            coolingFactor: 0.96,
            minTemp: 1.0,
            animate: false,
        }},
        wheelSensitivity: 0.18,
        minZoom: 0.15, maxZoom: 4,
    }});

    // Hover-highlight: dim non-neighbors, color neighbor edges
    cy.on('mouseover', 'node', (evt) => {{
        const node = evt.target;
        const neighbors = node.closedNeighborhood();
        cy.elements().addClass('dim');
        neighbors.removeClass('dim').addClass('neighbor');
        node.removeClass('neighbor').addClass('focus');
    }});
    cy.on('mouseout', 'node', () => {{
        cy.elements().removeClass('dim neighbor focus');
    }});

    cy.on('tap', 'node', (evt) => {{
        const name = evt.target.data('raw_text');
        window.location.href = '/entity?name=' + encodeURIComponent(name);
    }});

    // Build legend with toggleable label filters
    const presentLabels = new Set();
    cy.nodes().forEach(n => presentLabels.add(n.data('ent_label')));
    const legend = document.getElementById('legend-rows');
    const labelOrder = ['PERSON','ORG','GPE','LOC','FAC','PRODUCT','WORK_OF_ART','DATE','MONEY','EVENT','LAW','NORP','LANGUAGE'];
    const sorted = labelOrder.filter(l => presentLabels.has(l));
    const hidden = new Set();
    function rerender() {{
        cy.batch(() => {{
            cy.nodes().forEach(n => {{
                const lab = n.data('ent_label');
                n.style('display', hidden.has(lab) ? 'none' : 'element');
            }});
        }});
    }}
    legend.innerHTML = sorted.map(lab =>
        `<div class="legend-row" data-label="${{lab}}">
            <div class="swatch" style="background:${{LABEL_COLORS[lab] || '#888'}}"></div>
            <span>${{lab}}</span>
        </div>`
    ).join('');
    legend.querySelectorAll('.legend-row').forEach(row => {{
        row.addEventListener('click', () => {{
            const lab = row.dataset.label;
            if (hidden.has(lab)) {{ hidden.delete(lab); row.classList.remove('disabled'); }}
            else                 {{ hidden.add(lab); row.classList.add('disabled'); }}
            rerender();
        }});
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
            "PERSON": "#7fff7f",
            "ORG": "#ffb700",
            "GPE": "#5af0ff",
            "LOC": "#5af0ff",
            "FAC": "#5af0ff",
            "PRODUCT": "#ff5af0",
            "WORK_OF_ART": "#ff5af0",
            "DATE": "#b8ffb8",
            "MONEY": "#ffd566",
            "EVENT": "#ff4d4d",
            "LAW": "#ff4d4d",
            "NORP": "#ff4d4d",
            "LANGUAGE": "#ff4d4d",
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
    def file_view(
        path: str | None = None, file_id: int | None = None,
    ):
        """Round 25 fix (audit-found gap H1) — accept ``file_id``
        as well as ``path``. Round 22's EA UI generates
        ``/file?file_id=N`` links from 6 places (today, agenda,
        triage walkthrough, command palette, followups, capture);
        without this signature the route returned 422 on every
        click, silently breaking the EA UI's primary navigation."""
        # Read-only — pure rendering, no mutations.
        cfg, conn, embedder, reranker = get_read_state()
        if file_id is not None and not path:
            row_path = conn.execute(
                "SELECT path FROM files WHERE id = ?", (int(file_id),),
            ).fetchone()
            if row_path is None:
                return HTMLResponse(_layout(
                    f"file_id={file_id}",
                    f'<h1>file_id {int(file_id)}</h1>'
                    f'<div class="empty">Not in index.</div>',
                ))
            path = row_path["path"]
        if not path:
            return HTMLResponse(_layout(
                "File", '<h1>File</h1>'
                '<div class="empty">Need either ?path=... or '
                '?file_id=N.</div>',
            ), status_code=400)
        row = conn.execute(
            "SELECT id, path, kind, mtime, size FROM files WHERE path = ?", (path,)
        ).fetchone()
        if not row:
            return HTMLResponse(_layout(path, f'<h1>{escape(path)}</h1><div class="empty">Not in index.</div>'))

        # Phase 88: redact sensitive content in chunk previews so
        # API keys / SSNs / tokens stored in the index don't render
        # to the dashboard.
        from .safety import redact_text as _redact

        chunks = conn.execute(
            "SELECT chunk_index, text FROM chunks WHERE file_id = ("
            "  SELECT id FROM files WHERE path = ?) ORDER BY chunk_index",
            (path,),
        ).fetchall()
        body_chunks = "".join(
            f'<article class="result"><h3>chunk {r["chunk_index"]}</h3>'
            f'<div class="snippet">{escape(_redact(r["text"]))}</div></article>'
            for r in chunks
        )
        kind = row["kind"]
        age = (time.time() - row["mtime"]) / 86400
        is_url = path.startswith("http://") or path.startswith("https://")
        open_link = f'<a href="{escape(path)}" target="_blank">Open ↗</a>' if is_url else ""

        # Phase 74: auto-summary card at the top so users + the chat
        # agent see the TL;DR before scrolling chunks. Only renders
        # when a summary actually exists.
        summary_html = _render_summary_block(conn, int(row["id"]))

        # Phase 84: PDF annotation card — surfaces highlights /
        # notes the user made in their PDF reader.
        annotations_html = _render_annotations_block(conn, int(row["id"]))

        # Phase 85: citation graph card.
        citations_html = _render_citations_block(conn, int(row["id"]))

        # Phase 52: surface "see also" backlinks at the bottom so the
        # file view becomes a wayfinding hub, not just a content dump.
        backlinks_html = _render_backlinks_block(conn, path)

        body = f"""
<h1>File <span class="muted">[{kind}]</span></h1>
<p class="path">{escape(path)} {open_link}</p>
<p class="muted">{len(chunks)} chunks · {row["size"] / 1024:.1f} KB · {age:.1f}d ago</p>
{summary_html}
{annotations_html}
{body_chunks}
{citations_html}
{backlinks_html}"""
        return HTMLResponse(_layout(Path(path).name or path, body))

    # --- Chat with your brain ----------------------------------------
    # Conversations persist to the index DB (chat_conversations + chat_messages),
    # so a `secondbrain reset` clears them with everything else. The browser
    # session cookie tracks which conversation is "current" for that browser.
    # Switching to /chat/N picks up that conversation; /chat/new starts fresh.

    def _chat_json_default(o: Any) -> Any:
        """Make Anthropic SDK content-block objects JSON-serialisable."""
        if hasattr(o, "model_dump"):
            return o.model_dump(exclude_none=True)
        if hasattr(o, "__dict__"):
            return o.__dict__
        return str(o)

    def _serialize_chat_history(history: list[dict]) -> list[dict]:
        """Convert a chat history (mix of strings + Anthropic content blocks)
        into JSON-safe dicts so it survives a DB round-trip."""
        out: list[dict] = []
        for msg in history:
            content = msg.get("content")
            if isinstance(content, str):
                out.append({"role": msg["role"], "content": content})
                continue
            if isinstance(content, list):
                blocks: list[Any] = []
                for b in content:
                    if hasattr(b, "model_dump"):
                        blocks.append(b.model_dump(exclude_none=True))
                    elif isinstance(b, dict):
                        blocks.append(b)
                    else:
                        blocks.append({"type": "text", "text": str(b)})
                out.append({"role": msg["role"], "content": blocks})
        return out

    def _load_history_from_db(conn, conversation_id: int) -> list[dict]:
        """Replay chat_messages rows into Anthropic-format history."""
        from .db import chat_get_messages

        history: list[dict] = []
        for row in chat_get_messages(conn, conversation_id):
            try:
                content = json.loads(row["content_json"])
            except (json.JSONDecodeError, TypeError):
                continue
            history.append({"role": row["role"], "content": content})
        return history

    def _render_history_html(conn, conversation_id: int) -> str:
        """Render past turns of a saved conversation as HTML for /chat/N."""
        from .db import chat_get_messages

        chunks: list[str] = []
        for row in chat_get_messages(conn, conversation_id):
            role = row["role"]
            try:
                content = json.loads(row["content_json"])
            except (json.JSONDecodeError, TypeError):
                continue
            if role == "user":
                txt = content if isinstance(content, str) else ""
                if not txt:
                    continue
                chunks.append(
                    f'<div class="chat-msg chat-user"><div class="chat-bubble">'
                    f'{escape(txt)}</div></div>'
                )
            elif role == "assistant":
                if not isinstance(content, list):
                    continue
                txt = "".join(
                    b.get("text", "") if isinstance(b, dict) and b.get("type") == "text" else ""
                    for b in content
                )
                if not txt.strip():
                    continue
                cite_html = ""
                if row["citations_json"]:
                    try:
                        cites = json.loads(row["citations_json"])
                    except json.JSONDecodeError:
                        cites = []
                    if cites:
                        cite_html = (
                            '<div class="chat-citations">'
                            '<div class="muted" style="font-size:11px;letter-spacing:0.06em;'
                            f'text-transform:uppercase;">Sources ({len(cites)})</div>'
                        )
                        for c in cites:
                            href = "/file?path=" + urllib.parse.quote_plus(c.get("file_path", ""))
                            label = escape(c.get("file_path", "")) + (
                                f' · chunk {c.get("chunk_index")}'
                                if c.get("chunk_index") is not None else ""
                            )
                            snip = escape(c.get("text", ""))
                            cite_html += (
                                f'<div class="chat-citation"><a href="{href}">{label}</a>'
                                f'<div class="chat-citation-snippet">{snip}</div></div>'
                            )
                        cite_html += "</div>"
                chunks.append(
                    f'<div class="chat-msg chat-assistant"><div class="chat-bubble">'
                    f'{escape(txt)}</div>{cite_html}</div>'
                )
        return "".join(chunks)

    def _resolve_active_cid(request: Request, conn) -> int | None:
        """Return the conversation_id this browser session is currently on,
        or None if the user is on the 'new' chat page."""
        from .db import chat_get_conversation

        raw = request.cookies.get("sb_chat_cid", "")
        if not raw:
            return None
        try:
            cid = int(raw)
        except ValueError:
            return None
        if chat_get_conversation(conn, cid) is None:
            return None
        return cid

    def _chat_page_body(cfg: Config, history_html: str, cid: int | None,
                       title: str | None, system_prompt: str | None = None) -> str:
        """Shared HTML for /chat (new) and /chat/N (existing)."""
        title_bar = ""
        if cid is not None:
            sp_indicator = (
                ' <span class="chat-sp-indicator" title="Custom system prompt active">●</span>'
                if system_prompt else ""
            )
            title_bar = (
                f'<div class="chat-titlebar">'
                f'  <span class="chat-title">{escape(title or "(untitled)")}{sp_indicator}</span>'
                f'  <a href="#" class="chat-titlebar-link" id="chat-edit-prompt">edit prompt</a>'
                f'  <a href="/chat/list" class="chat-titlebar-link">all chats</a>'
                f'  <a href="/chat" class="chat-titlebar-link">+ new</a>'
                f'</div>'
                # Hidden by default; toggled open by chat-edit-prompt link.
                f'<form id="chat-sp-form" method="post" action="/chat/{cid}/system_prompt" '
                f'      class="chat-sp-form" style="display:none;">'
                f'  <label class="muted">Custom system prompt for this conversation only:</label>'
                f'  <textarea name="system_prompt" rows="4" '
                f'            placeholder="e.g. You are my code reviewer. Always cite specific lines.">'
                f'{escape(system_prompt or "")}</textarea>'
                f'  <div class="chat-form-row">'
                f'    <button type="submit">Save</button>'
                f'    <button type="submit" name="clear" value="1" '
                f'            class="chat-sp-clear">Clear (use default)</button>'
                f'    <span class="muted" style="font-size:11px;">'
                f'      Empty = use the built-in search-grounded brain prompt.'
                f'    </span>'
                f'  </div>'
                f'</form>'
            )
        else:
            title_bar = (
                '<div class="chat-titlebar">'
                '  <span class="chat-title">New conversation</span>'
                '  <a href="/chat/list" class="chat-titlebar-link">all chats</a>'
                '</div>'
            )
        empty_state = (
            '<div class="empty">Ask anything that lives in your brain.</div>'
        )
        return f"""
<h1>Chat with your brain</h1>
<p class="muted" style="margin-top:-8px;">
    Conversational Q&amp;A grounded in everything you've indexed. Uses
    <code>{escape(cfg.chat_model)}</code> with <code>search_brain</code> as a
    tool — answers cite their sources. Conversations persist; revisit them
    on the <a href="/chat/list">all chats</a> page.
</p>
{title_bar}
<div id="chat-log" class="chat-log">{history_html or empty_state}</div>

<form id="chat-form" class="chat-form">
    <textarea id="chat-input" rows="2" placeholder="What was that thing about Voyage rate limits?" autofocus></textarea>
    <div class="chat-form-row">
        <button type="submit" id="chat-send">Send</button>
        <span style="flex:1"></span>
        {"<a href='/chat' class='chat-reset'>start new conversation</a>" if cid else ""}
    </div>
</form>

<style>
.chat-titlebar {{
    display: flex; align-items: center; gap: 14px;
    padding: 8px 12px; margin-bottom: 8px;
    border: 1px solid var(--border); border-radius: 4px;
    background: #0c0c0c; font-size: 12.5px;
}}
.chat-title {{ flex: 1; color: var(--green); font-family: var(--mono); }}
.chat-titlebar-link {{ color: #888; font-size: 12px; cursor: pointer; }}
.chat-titlebar-link:hover {{ color: var(--green); }}
.chat-sp-indicator {{ color: var(--green); margin-left: 4px; }}
.chat-sp-form {{
    display: flex; flex-direction: column; gap: 8px;
    margin: 0 0 14px 0; padding: 12px;
    border: 1px solid var(--green-dim); border-radius: 4px;
    background: #0a140a;
}}
.chat-sp-form textarea {{
    width: 100%; box-sizing: border-box; resize: vertical;
    background: #0e0e0e; color: var(--fg);
    border: 1px solid var(--border); border-radius: 2px;
    padding: 8px 10px; font: 12.5px var(--mono);
}}
.chat-sp-clear {{ background: transparent; color: #888; }}
.chat-log {{
    display: flex; flex-direction: column; gap: 14px;
    padding: 16px; min-height: 280px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 4px;
}}
.chat-msg {{ display: flex; flex-direction: column; }}
.chat-user {{ align-items: flex-end; }}
.chat-bubble {{
    max-width: 80%; padding: 10px 14px; border-radius: 4px;
    border: 1px solid var(--border); white-space: pre-wrap; line-height: 1.55;
}}
.chat-user .chat-bubble {{ background: #16201a; border-color: var(--green-dim); }}
.chat-assistant .chat-bubble {{ background: #111; }}
.chat-events {{
    margin-top: 6px; font-size: 11.5px; color: #888;
    border-left: 2px solid var(--border); padding-left: 10px;
}}
.chat-event-search::before {{ content: "⌕ "; color: var(--green); }}
.chat-event-result {{ font-family: var(--mono); margin-left: 10px; opacity: 0.85; }}
.chat-citations {{ margin-top: 12px; display: flex; flex-direction: column; gap: 8px; }}
.chat-citation {{
    background: #0f140f; border: 1px solid var(--border); border-left: 3px solid var(--green-dim);
    padding: 8px 12px; font-size: 12.5px;
}}
.chat-citation a {{ color: var(--green); }}
.chat-citation .chat-citation-snippet {{
    margin-top: 6px; color: #aaa; white-space: pre-wrap; font-family: var(--mono);
    font-size: 11.5px; max-height: 80px; overflow: hidden;
}}
.chat-citation-web {{ border-left-color: #5af0ff; }}
.chat-citation-web a::before {{ content: "↗ "; color: #5af0ff; }}
.chat-citation-suburl {{
    margin-top: 4px; color: #5af0ff; font-family: var(--mono); font-size: 11px;
    opacity: 0.8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}}
.chat-form {{
    margin-top: 14px; display: flex; flex-direction: column; gap: 8px;
}}
.chat-form textarea {{
    width: 100%; box-sizing: border-box; resize: vertical;
    background: #0e0e0e; color: var(--fg);
    border: 1px solid var(--border); border-radius: 2px;
    padding: 10px 12px; font: 14px var(--mono);
}}
.chat-form-row {{ display: flex; align-items: center; gap: 14px; }}
.chat-reset {{ color: #888; font-size: 12px; }}
.chat-typing::after {{
    content: "▮"; color: var(--green);
    animation: chat-blink 1s steps(1) infinite;
}}
@keyframes chat-blink {{ 50% {{ opacity: 0; }} }}
</style>

<script>{CHAT_JS}</script>
"""

    @app.get("/chat", response_class=HTMLResponse)
    def chat_page(request: Request):
        """New conversation. Doesn't create a row until the user sends."""
        cfg, conn, _, _ = get_state()
        # If the user just clicked "new", clear their cookie so the next
        # message starts a fresh row.
        body = _chat_page_body(cfg, "", None, None)
        resp = HTMLResponse(_layout("Chat", body, "chat"))
        resp.delete_cookie("sb_chat_cid", path="/")
        return resp

    @app.get("/chat/list", response_class=HTMLResponse)
    def chat_list_page():
        """All saved conversations, most recent first."""
        from .db import chat_list_conversations

        cfg, conn, _, _ = get_state()
        rows = chat_list_conversations(conn, limit=200)
        if not rows:
            body = (
                "<h1>Past conversations</h1>"
                "<div class='empty'>No saved chats yet. "
                "<a href='/chat'>Start one →</a></div>"
            )
            return HTMLResponse(_layout("Chat history", body, "chat"))
        items: list[str] = []
        for r in rows:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["updated_at"]))
            items.append(
                f'<a class="chat-listrow" href="/chat/{r["id"]}">'
                f'  <div class="chat-listrow-title">{escape(r["title"])}</div>'
                f'  <div class="chat-listrow-meta">'
                f'    <span>{when}</span>'
                f'    <span>{r["n_messages"]} message{"" if r["n_messages"] == 1 else "s"}</span>'
                f'  </div>'
                f'</a>'
            )
        body = f"""
<h1>Past conversations</h1>
<p class="muted" style="margin-top:-8px;">
    All your previous chats. Click to revisit; new turns continue the
    same conversation. <a href="/chat">Start a new one →</a>
</p>
<div class="chat-list">{"".join(items)}</div>
<style>
.chat-list {{ display: flex; flex-direction: column; gap: 8px; }}
.chat-listrow {{
    display: flex; flex-direction: column; gap: 4px;
    padding: 12px 14px; background: #0e0e0e;
    border: 1px solid var(--border); border-left: 3px solid var(--green-dim);
    color: var(--fg); text-decoration: none;
}}
.chat-listrow:hover {{ background: #131c14; }}
.chat-listrow-title {{ font-family: var(--mono); color: var(--green); }}
.chat-listrow-meta {{ display: flex; gap: 16px; color: #888; font-size: 11.5px; }}
</style>"""
        return HTMLResponse(_layout("Chat history", body, "chat"))

    @app.get("/chat/{cid:int}", response_class=HTMLResponse)
    def chat_view(cid: int, request: Request):
        from .db import chat_get_conversation, chat_get_system_prompt

        cfg, conn, _, _ = get_state()
        row = chat_get_conversation(conn, cid)
        if row is None:
            return HTMLResponse(
                _layout("Chat", "<h1>Not found</h1>", "chat"), status_code=404,
            )
        history_html = _render_history_html(conn, cid)
        sp = chat_get_system_prompt(conn, cid)
        body = _chat_page_body(cfg, history_html, cid, row["title"], system_prompt=sp)
        resp = HTMLResponse(_layout("Chat", body, "chat"))
        # Stick the user to this conversation for subsequent message posts.
        resp.set_cookie(
            "sb_chat_cid", str(cid),
            httponly=True, samesite="strict", path="/",
        )
        return resp

    @app.post("/chat/{cid:int}/system_prompt")
    async def chat_set_prompt(cid: int, request: Request):
        """Save (or clear) the per-conversation system prompt."""
        from .db import chat_get_conversation, chat_set_system_prompt

        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        _, conn, _, _ = get_state()
        if chat_get_conversation(conn, cid) is None:
            return HTMLResponse("<h1>Not found</h1>", status_code=404)
        form = await request.form()
        if form.get("clear"):
            chat_set_system_prompt(conn, cid, None)
        else:
            sp = (form.get("system_prompt") or "").strip()
            chat_set_system_prompt(conn, cid, sp or None)
        return RedirectResponse(url=f"/chat/{cid}", status_code=303)

    @app.post("/chat/{cid:int}/delete")
    def chat_delete(cid: int, request: Request):
        from .db import chat_delete_conversation

        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        _, conn, _, _ = get_state()
        chat_delete_conversation(conn, cid)
        return RedirectResponse(url="/chat/list", status_code=303)

    @app.post("/api/chat/message")
    async def chat_message(request: Request):
        """SSE-style streaming: emits one JSON line per event, then [DONE].

        On the first message of a fresh chat, lazily creates the conversation
        row and returns its id in the ``done`` event so the client can update
        its URL. Subsequent messages append to the same conversation.

        Body: form field ``message``. Cookie ``sb_chat_cid`` selects which
        conversation to append to.
        """
        from fastapi.responses import StreamingResponse

        from .db import (
            chat_append_message,
            chat_create_conversation,
            chat_get_system_prompt,
        )

        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        cfg, conn, embedder, reranker = get_state()
        form = await request.form()
        user_msg = (form.get("message") or "").strip()
        # Resolve current conversation, creating one lazily on first send.
        active_cid = _resolve_active_cid(request, conn)
        created_now = False
        if active_cid is None and user_msg:
            # Title with a truncated version of the first message; the user
            # can rename later if we add the UI.
            title = user_msg[:60] + ("…" if len(user_msg) > 60 else "")
            active_cid = chat_create_conversation(conn, title)
            created_now = True

        # Replay history from DB so multi-turn context works across reloads.
        history = _load_history_from_db(conn, active_cid) if active_cid else []
        system_prompt = (
            chat_get_system_prompt(conn, active_cid) if active_cid else None
        )

        def gen():
            if not user_msg:
                yield 'data: {"kind":"error","data":"empty message"}\n\n'
                yield 'data: [DONE]\n\n'
                return
            # Persist the user turn before streaming, so a crash mid-stream
            # still leaves a coherent record (assistant turn just won't exist).
            chat_append_message(
                conn, active_cid, "user", json.dumps(user_msg),
            )
            updated_history: list[dict] | None = None
            citations_for_assistant: list[dict] = []
            for event in stream_chat(
                cfg, conn, embedder, reranker, user_msg, history,
                system_prompt=system_prompt,
            ):
                payload = json.dumps(
                    {"kind": event.kind, "data": event.data},
                    default=_chat_json_default,
                )
                yield f"data: {payload}\n\n"
                if event.kind == "done" and isinstance(event.data, dict):
                    updated_history = event.data.get("history")
                    citations_for_assistant = event.data.get("citations") or []
            if updated_history is not None:
                serialized = _serialize_chat_history(updated_history)
                # Find the assistant turn we just generated (last role==assistant)
                # and persist its content + citations.
                for msg in reversed(serialized):
                    if msg.get("role") == "assistant":
                        chat_append_message(
                            conn, active_cid, "assistant",
                            json.dumps(msg["content"]),
                            citations_json=json.dumps(citations_for_assistant),
                        )
                        break
            # Tell the client which conversation this belonged to so it can
            # update its URL without a full page navigation.
            yield (
                f'data: {{"kind":"meta","data":{{"cid":{active_cid},'
                f'"created_now":{str(created_now).lower()}}}}}\n\n'
            )
            yield 'data: [DONE]\n\n'

        resp = StreamingResponse(gen(), media_type="text/event-stream")
        if created_now and active_cid is not None:
            resp.set_cookie(
                "sb_chat_cid", str(active_cid),
                httponly=True, samesite="strict", path="/",
            )
        return resp

    # --- Watchlists --------------------------------------------------

    @app.get("/watch", response_class=HTMLResponse)
    def watch_page():
        from .db import watchlist_get_domains, watchlist_list
        from .presets import PRESETS
        from .presets import names as preset_names
        from .watchlist import latest_summary

        cfg, conn, _, _ = get_state()
        rows = watchlist_list(conn)
        items_html: list[str] = []
        for r in rows:
            sched = r["schedule_minutes"]
            if sched >= 1440 and sched % 1440 == 0:
                every = f"{sched // 1440}d"
            elif sched >= 60 and sched % 60 == 0:
                every = f"{sched // 60}h"
            else:
                every = f"{sched}m"
            last = "(never)" if not r["last_run_at"] else time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(r["last_run_at"])
            )
            domains = watchlist_get_domains(conn, r["id"])
            scope_chip = ""
            if domains:
                # Try to label as a preset name when the saved set matches.
                matched_preset = None
                domain_set = set(domains)
                for pn, plist in PRESETS.items():
                    if domain_set == set(plist):
                        matched_preset = pn
                        break
                if matched_preset:
                    label = f"preset: {matched_preset}"
                else:
                    label = f"{len(domains)} domain{'s' if len(domains) != 1 else ''}"
                tip = ", ".join(domains)
                scope_chip = (
                    f' <span class="watch-scope" title="{escape(tip)}">'
                    f'⛓ {escape(label)}</span>'
                )
            s = latest_summary(conn, r["id"])
            answer_html = ""
            cite_html = ""
            new_html = ""
            if s and s.get("new_count"):
                # "what's new since last run" badge - the headline thing
                # the user wants to see at a glance.
                count = s["new_count"]
                np = s.get("new_paths") or []
                tip = "\n".join(np[:25])
                if len(np) > 25:
                    tip += f"\n... and {len(np) - 25} more"
                new_html = (
                    f'<div class="watch-new-badge" title="{escape(tip)}">'
                    f'⚡ {count} new since last run</div>'
                )
            if s and s.get("answer"):
                answer_html = (
                    '<div class="watch-answer">'
                    f'{escape(s["answer"])}'
                    '</div>'
                )
                cites = s.get("citations") or []
                # Mark which citations are new in this run so they get a
                # visual highlight in the list.
                new_set = set(s.get("new_paths") or []) if s else set()
                if cites:
                    cite_pieces: list[str] = []
                    for c in cites:
                        kind = c.get("kind", "brain")
                        is_new = c.get("file_path") in new_set
                        new_chip = (
                            ' <span class="watch-cite-new">NEW</span>'
                            if is_new else ""
                        )
                        # Fit chip — only when a resume is configured and
                        # the watchlist runner attached fit_label to this
                        # citation.
                        fit_chip = ""
                        if c.get("fit_label"):
                            klass = (
                                "watch-cite-fit "
                                f"watch-cite-fit-{c['fit_label'].split()[0]}"
                            )
                            fit_chip = (
                                f' <span class="{klass}" '
                                f'title="resume: {escape(c.get("fit_resume", ""))} '
                                f'· cosine {c.get("fit_score", 0):.2f}">'
                                f'{escape(c["fit_label"])}</span>'
                            )
                        if kind == "web":
                            url = c.get("url") or c.get("file_path", "")
                            label = c.get("page_title") or url
                            cite_pieces.append(
                                f'<a href="{escape(url)}" target="_blank" '
                                f'rel="noopener noreferrer" class="watch-cite watch-cite-web">'
                                f'↗ {escape(label)}{new_chip}{fit_chip}</a>'
                            )
                        else:
                            fp = c.get("file_path", "")
                            cite_pieces.append(
                                f'<a href="/file?path={urllib.parse.quote_plus(fp)}" '
                                f'class="watch-cite">{escape(fp)}{new_chip}{fit_chip}</a>'
                            )
                    cite_html = (
                        '<div class="watch-cites">' + "".join(cite_pieces) + "</div>"
                    )
            elif s and s.get("error"):
                answer_html = (
                    f'<div class="watch-answer watch-error">'
                    f'last run errored: {escape(s["error"])}</div>'
                )

            on_off_btn = (
                '<button name="action" value="disable">disable</button>'
                if r["enabled"]
                else '<button name="action" value="enable">enable</button>'
            )
            items_html.append(f"""
<article class="watch-card">
    <header class="watch-head">
        <span class="watch-name">{escape(r['name'])}{scope_chip}</span>
        <span class="watch-sched">every {every} · last: {last}</span>
    </header>
    <div class="watch-q">"{escape(r['query'])}"</div>
    {new_html}
    {answer_html}
    {cite_html}
    <form method="post" action="/watch/{r['id']}/action" class="watch-actions">
        <button name="action" value="run">run now</button>
        {on_off_btn}
        <button name="action" value="delete" class="watch-danger"
                onclick="return confirm('Delete this watchlist?');">delete</button>
    </form>
</article>""")

        items = "".join(items_html) or (
            '<div class="empty">No watchlists yet. Add one below.</div>'
        )
        preset_options = '<option value="">(none — generic web search)</option>' + "".join(
            f'<option value="{escape(p)}">{escape(p)} '
            f'({len(PRESETS[p])} hosts)</option>'
            for p in preset_names()
        )
        body = f"""
<h1>Watchlists</h1>
<p class="muted" style="margin-top:-8px;">
    Recurring saved queries. The daemon runs each one on its schedule and
    captures a fresh "what's new since last run" summary using
    <code>{escape(cfg.chat_model)}</code> with web search and your brain
    as tools.
    {"<strong style='color:var(--green);'>Web search is enabled.</strong>"
     if cfg.web_search_enabled else
     "<strong style='color:#ff5c5c;'>Web search is OFF</strong> "
     "in config; watchlists will only see your indexed brain. "
     "Set <code>web_search_enabled = true</code> in config.toml to turn on."}
</p>

<form method="post" action="/watch/new" class="watch-new card">
    <div class="watch-new-row">
        <input type="text" name="name" placeholder="name (e.g. pm-internships)" required>
        <select name="every">
            <option value="15m">every 15m</option>
            <option value="1h">every hour</option>
            <option value="6h">every 6h</option>
            <option value="1d" selected>every day</option>
            <option value="3d">every 3 days</option>
            <option value="7d">every week</option>
        </select>
        <select name="preset" title="Scope web search to a curated domain list.">
            {preset_options}
        </select>
    </div>
    <textarea name="query" rows="2" required
        placeholder="What product manager internships came out today at top tech companies?"></textarea>
    <input type="text" name="extra_domains" class="watch-extra"
        placeholder="extra domains, comma-separated (e.g. anthropic.com, openai.com)">
    <div class="watch-form-row">
        <button type="submit">+ create watchlist</button>
        <span class="muted" style="font-size:11px;">
            ~$0.02-0.10 per run · pick a preset to keep results focused.
        </span>
    </div>
</form>

<div class="watch-list">{items}</div>

<style>
.watch-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-left: 3px solid var(--green-dim);
    padding: 14px 16px; margin: 12px 0;
}}
.watch-head {{ display: flex; justify-content: space-between; gap: 12px; }}
.watch-name {{ font-weight: 600; color: var(--green); font-family: var(--mono); }}
.watch-sched {{ font-size: 11.5px; color: #888; }}
.watch-q {{ margin: 6px 0; color: #ccc; font-style: italic; font-size: 13px; }}
.watch-answer {{
    margin: 10px 0; padding: 10px 12px;
    background: #0a0a0a; border-left: 2px solid var(--border);
    white-space: pre-wrap; font-size: 13px; line-height: 1.55;
}}
.watch-error {{ color: #ff5c5c; }}
.watch-cites {{ display: flex; flex-direction: column; gap: 4px; margin: 8px 0; }}
.watch-cite {{
    color: var(--green); font-size: 11.5px; font-family: var(--mono);
    text-decoration: none; padding: 2px 0;
}}
.watch-cite-web {{ color: #5af0ff; }}
.watch-cite:hover {{ text-decoration: underline; }}
.watch-new-badge {{
    display: inline-block; margin: 6px 0;
    padding: 4px 10px;
    background: #1c2814; color: #b8ffb8;
    border: 1px solid #4abe4a;
    border-radius: 2px;
    font-size: 12px; font-family: var(--mono);
    letter-spacing: 0.04em;
    box-shadow: 0 0 12px rgba(127,255,127,0.2);
    cursor: help;
}}
.watch-cite-new {{
    display: inline-block; margin-left: 6px;
    padding: 0 4px;
    background: #1c2814; color: #b8ffb8;
    border: 1px solid #4abe4a;
    font-size: 9.5px; letter-spacing: 0.08em;
    border-radius: 2px;
    vertical-align: middle;
}}
.watch-cite-fit {{
    display: inline-block; margin-left: 6px;
    padding: 0 5px; font-size: 9.5px; letter-spacing: 0.04em;
    border-radius: 2px; vertical-align: middle; cursor: help;
    border: 1px solid var(--border);
}}
.watch-cite-fit-great {{ background: #16201a; color: #b8ffb8; border-color: #4abe4a; }}
.watch-cite-fit-decent {{ background: #1c1c1c; color: #ffd566; border-color: #5a4a14; }}
.watch-cite-fit-stretch {{ background: #1a1414; color: #ffaa66; border-color: #5a3a14; }}
.watch-cite-fit-weak {{ background: #1a1414; color: #888; }}
.watch-actions {{
    display: flex; gap: 10px; margin-top: 10px;
}}
.watch-actions button {{ font-size: 11px; }}
.watch-danger {{ background: transparent; color: #ff5c5c; border-color: #4a1c1c; }}
.watch-new {{
    margin-top: 12px; display: flex; flex-direction: column; gap: 8px;
}}
.watch-new-row {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.watch-new-row input {{ flex: 1 1 200px; min-width: 200px; }}
.watch-new-row select {{ flex: 0 0 auto; }}
.watch-extra {{ width: 100%; }}
.watch-scope {{
    display: inline-block; margin-left: 8px;
    padding: 1px 6px; font-size: 10.5px;
    background: #0f1c1f; color: #5af0ff;
    border: 1px solid #1d3a44; border-radius: 2px;
    letter-spacing: 0.04em;
}}
.watch-new textarea, .watch-new input, .watch-new select {{
    background: #0e0e0e; color: var(--fg);
    border: 1px solid var(--border); border-radius: 2px;
    padding: 8px 10px; font: 13px var(--mono);
}}
.watch-form-row {{ display: flex; gap: 12px; align-items: center; }}
</style>"""
        return HTMLResponse(_layout("Watchlists", body, "watch"))

    @app.post("/watch/new")
    async def watch_new(request: Request):
        from .db import watchlist_create
        from .presets import resolve as resolve_preset

        _, conn, _, _ = get_state()
        # Round 14 — uses the centralised _is_same_origin_request helper.
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        form = await request.form()
        name = (form.get("name") or "").strip()
        query = (form.get("query") or "").strip()
        every = (form.get("every") or "1d").strip()
        preset = (form.get("preset") or "").strip() or None
        extras_raw = (form.get("extra_domains") or "").strip()
        extras = [e.strip() for e in extras_raw.split(",") if e.strip()]
        if not name or not query:
            return RedirectResponse(url="/watch", status_code=303)
        # Reuse the CLI's parser so we accept the same notation.
        from .cli import _parse_every  # noqa: PLC0415
        try:
            minutes = _parse_every(every)
        except Exception:  # noqa: BLE001
            minutes = 1440
        try:
            allowed = resolve_preset(preset, extras)
        except ValueError:
            # Invalid preset name; fall back to no scoping rather than 500.
            allowed = None
        watchlist_create(
            conn, name, query, schedule_minutes=minutes,
            allowed_domains=allowed,
        )
        return RedirectResponse(url="/watch", status_code=303)

    @app.post("/watch/{watchlist_id:int}/action")
    async def watch_action(watchlist_id: int, request: Request):
        from .db import (
            watchlist_delete,
            watchlist_get,
            watchlist_set_enabled,
        )
        from .watchlist import run_watchlist

        cfg, conn, embedder, reranker = get_state()
        # Round 14 — uses the centralised _is_same_origin_request helper.
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        form = await request.form()
        action = (form.get("action") or "").strip()
        row = watchlist_get(conn, watchlist_id)
        if row is None:
            return RedirectResponse(url="/watch", status_code=303)
        if action == "delete":
            watchlist_delete(conn, watchlist_id)
        elif action == "disable":
            watchlist_set_enabled(conn, watchlist_id, False)
        elif action == "enable":
            watchlist_set_enabled(conn, watchlist_id, True)
        elif action == "run":
            run_watchlist(
                cfg, conn, embedder, reranker,
                row["id"], row["query"], row["last_run_at"],
            )
        return RedirectResponse(url="/watch", status_code=303)

    # --- Application tracker -----------------------------------------

    @app.get("/applications", response_class=HTMLResponse)
    def applications_page(status: str | None = None):
        from .db import APPLICATION_STATUSES, application_list

        _, conn, _, _ = get_state()
        rows = application_list(conn, status=status)
        # Status counts for the filter chips at the top.
        all_rows = application_list(conn)
        counts: dict[str, int] = dict.fromkeys(APPLICATION_STATUSES, 0)
        for r in all_rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1

        chips = ['<a href="/applications" '
                 f'class="apps-chip{" apps-chip-active" if not status else ""}">'
                 f'all ({len(all_rows)})</a>']
        for sname in APPLICATION_STATUSES:
            n = counts.get(sname, 0)
            klass = "apps-chip" + (" apps-chip-active" if status == sname else "")
            chips.append(
                f'<a href="/applications?status={sname}" class="{klass}">'
                f'{escape(sname)} ({n})</a>'
            )

        rows_html: list[str] = []
        for r in rows:
            when = time.strftime("%Y-%m-%d", time.localtime(r["applied_at"]))
            url_html = (
                f'<a href="{escape(r["role_url"])}" target="_blank" '
                f'rel="noopener noreferrer" class="apps-url">link ↗</a>'
                if r["role_url"] else ""
            )
            notes_html = (
                f'<div class="apps-notes">{escape(r["notes"])}</div>'
                if r["notes"] else ""
            )
            # Inline status-update form: each option is a button.
            buttons = "".join(
                f'<button name="new_status" value="{s}" '
                f'class="apps-status-btn apps-status-{s}'
                f'{" apps-status-on" if s == r["status"] else ""}">{s}</button>'
                for s in APPLICATION_STATUSES
            )
            rows_html.append(f"""
<tr class="apps-row">
    <td class="apps-co">{escape(r['company'])}</td>
    <td class="apps-role">{escape(r['role_title'])}</td>
    <td class="apps-meta">{escape(r['source'] or '')} · {when}</td>
    <td class="apps-actions">{url_html}</td>
    <td>
        <form method="post" action="/applications/{r['id']}/status" class="apps-status-form">
            {buttons}
        </form>
    </td>
    <td>
        <form method="post" action="/applications/{r['id']}/delete"
              class="apps-delete-form">
            <button type="submit" class="apps-delete-btn"
                    onclick="return confirm('Delete this application?');">×</button>
        </form>
    </td>
</tr>
{('<tr><td colspan="6" class="apps-notes-row">' + notes_html + '</td></tr>') if notes_html else ''}
""")

        rows_body = (
            "".join(rows_html)
            or '<tr><td colspan="6" class="empty">No applications yet.</td></tr>'
        )

        body = f"""
<h1>Applications</h1>
<p class="muted" style="margin-top:-8px;">
    Track jobs you've applied to. Watchlists skip already-applied roles
    when surfacing "new" items, and the chat agent can answer "have I
    applied to X?" against this list.
</p>

<div class="apps-chips">{"".join(chips)}</div>

<form method="post" action="/applications/new" class="apps-new card">
    <div class="apps-new-row">
        <input type="text" name="company" placeholder="Company" required>
        <input type="text" name="role_title" placeholder="Role title" required>
    </div>
    <div class="apps-new-row">
        <input type="text" name="role_url" placeholder="Posting URL (optional but recommended)">
        <input type="text" name="source" placeholder="Source: linkedin / referral / handshake / ..."
            style="max-width:280px;">
    </div>
    <textarea name="notes" rows="2" placeholder="Notes (recruiter contact, deadline, etc.)"></textarea>
    <button type="submit">+ record application</button>
</form>

<table class="apps-table">{rows_body}</table>

<style>
.apps-chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 14px 0; }}
.apps-chip {{
    padding: 4px 10px; font-size: 12px; font-family: var(--mono);
    background: #0e0e0e; color: #888; border: 1px solid var(--border);
    border-radius: 2px; text-decoration: none;
}}
.apps-chip:hover {{ color: var(--green); border-color: var(--green-dim); }}
.apps-chip-active {{ color: var(--green); border-color: var(--green-dim); background: #131c14; }}
.apps-new {{
    display: flex; flex-direction: column; gap: 8px; margin: 8px 0 18px 0;
}}
.apps-new-row {{ display: flex; gap: 8px; }}
.apps-new input, .apps-new textarea {{
    flex: 1; box-sizing: border-box;
    background: #0e0e0e; color: var(--fg);
    border: 1px solid var(--border); border-radius: 2px;
    padding: 8px 10px; font: 13px var(--mono);
}}
.apps-new textarea {{ resize: vertical; }}
.apps-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.apps-row td {{
    padding: 8px 10px; border-bottom: 1px dashed var(--border);
    vertical-align: middle;
}}
.apps-co {{ font-family: var(--mono); color: var(--green); white-space: nowrap; }}
.apps-meta {{ color: #888; font-size: 11.5px; white-space: nowrap; }}
.apps-url {{ color: var(--green); font-family: var(--mono); font-size: 11.5px; }}
.apps-status-form {{ display: flex; gap: 2px; flex-wrap: wrap; }}
.apps-status-btn {{
    background: transparent; color: #888; border: 1px solid var(--border);
    border-radius: 2px; padding: 2px 6px; font-size: 10.5px;
    font-family: var(--mono); cursor: pointer; letter-spacing: 0.04em;
}}
.apps-status-btn:hover {{ color: var(--green); border-color: var(--green-dim); }}
.apps-status-on {{ background: #131c14; color: var(--green); border-color: var(--green-dim); }}
.apps-status-offer.apps-status-on {{ background: #16201a; color: #b8ffb8; }}
.apps-status-rejected.apps-status-on {{ background: #2a1414; color: #ff8c8c; border-color: #5a2828; }}
.apps-delete-form {{ display: inline; }}
.apps-delete-btn {{
    background: transparent; color: #555; border: none; cursor: pointer;
    font-size: 18px; padding: 0 4px;
}}
.apps-delete-btn:hover {{ color: #ff5c5c; }}
.apps-notes-row {{ padding: 0 !important; }}
.apps-notes {{
    padding: 8px 14px; background: #0a0a0a; color: #aaa; font-size: 12.5px;
    border-bottom: 1px solid var(--border);
}}
</style>"""
        return HTMLResponse(_layout("Applications", body, "applications"))

    @app.post("/applications/new")
    async def applications_new(request: Request):
        from .db import application_create

        _, conn, _, _ = get_state()
        # Round 14 — uses the centralised _is_same_origin_request helper.
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        form = await request.form()
        company = (form.get("company") or "").strip()
        role = (form.get("role_title") or "").strip()
        if not company or not role:
            return RedirectResponse(url="/applications", status_code=303)
        application_create(
            conn, company=company, role_title=role,
            role_url=(form.get("role_url") or "").strip() or None,
            source=(form.get("source") or "").strip() or None,
            notes=(form.get("notes") or "").strip() or None,
        )
        return RedirectResponse(url="/applications", status_code=303)

    @app.post("/applications/{aid:int}/status")
    async def applications_status(aid: int, request: Request):
        from .db import APPLICATION_STATUSES, application_get, application_set_status

        _, conn, _, _ = get_state()
        # Round 14 — uses the centralised _is_same_origin_request helper.
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        form = await request.form()
        new_status = (form.get("new_status") or "").strip()
        if (
            application_get(conn, aid) is None
            or new_status not in APPLICATION_STATUSES
        ):
            return RedirectResponse(url="/applications", status_code=303)
        application_set_status(conn, aid, new_status)
        return RedirectResponse(url="/applications", status_code=303)

    @app.post("/applications/{aid:int}/delete")
    async def applications_delete(aid: int, request: Request):
        from .db import application_delete

        _, conn, _, _ = get_state()
        # Round 14 — uses the centralised _is_same_origin_request helper.
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        application_delete(conn, aid)
        return RedirectResponse(url="/applications", status_code=303)

    # --- Pre-event briefings ---------------------------------------

    @app.get("/briefings", response_class=HTMLResponse)
    def briefings_page():
        from .db import event_briefing_get
        from .event_briefing import iter_upcoming_events

        cfg, conn, _, _ = get_state()
        # Look 12h out so the page shows the rest of today's meetings.
        upcoming = sorted(
            iter_upcoming_events(cfg, 12 * 3600),
            key=lambda e: e.starts_at,
        )

        upcoming_html: list[str] = []
        for ev in upcoming:
            existing = event_briefing_get(conn, ev.event_id, ev.source)
            when = time.strftime(
                "%a %H:%M", time.localtime(ev.starts_at),
            )
            mins_until = max(0, int((ev.starts_at - time.time()) // 60))
            mins_str = (
                f"in {mins_until} min" if mins_until < 60
                else f"in {mins_until // 60}h {mins_until % 60}m"
            )
            attendees_str = ", ".join(ev.attendees[:5]) + (
                f" + {len(ev.attendees) - 5} more"
                if len(ev.attendees) > 5 else ""
            )
            if existing is None:
                status = (
                    '<span class="brief-status brief-pending">briefing pending…</span>'
                )
                briefing_block = ""
            elif existing["error"]:
                status = (
                    f'<span class="brief-status brief-error">'
                    f'errored: {escape(existing["error"][:80])}</span>'
                )
                briefing_block = ""
            else:
                status = (
                    '<span class="brief-status brief-ready">briefing ready</span>'
                )
                briefing_block = (
                    '<div class="brief-text">'
                    + escape(existing["briefing_text"] or "").replace("\n", "<br>")
                    + '</div>'
                )

            cal_link = (
                f' <a href="{escape(ev.url)}" target="_blank" '
                f'rel="noopener noreferrer" class="brief-cal-link">'
                f'open in calendar ↗</a>'
                if ev.url else ""
            )
            regen_form = (
                f'<form method="post" action="/briefings/regenerate" '
                f'class="brief-regen-form">'
                f'<input type="hidden" name="event_id" value="{escape(ev.event_id)}">'
                f'<input type="hidden" name="event_source" value="{escape(ev.source)}">'
                f'<button type="submit">'
                + ("regenerate" if existing else "generate now")
                + '</button></form>'
            )
            upcoming_html.append(f"""
<article class="brief-card">
  <header class="brief-head">
    <span class="brief-when">{when} <span class="brief-rel">({mins_str})</span></span>
    <span class="brief-title">{escape(ev.title)}{cal_link}</span>
  </header>
  <div class="brief-meta">
    {f'<span>📍 {escape(ev.location)}</span>' if ev.location else ''}
    {f'<span>👥 {escape(attendees_str)}</span>' if attendees_str else ''}
    {f'<span class="brief-cal">{escape(ev.calendar_name)}</span>' if ev.calendar_name else ''}
  </div>
  <div class="brief-row">
    {status}
    {regen_form}
  </div>
  {briefing_block}
</article>""")

        upcoming_block = (
            "".join(upcoming_html)
            or '<div class="empty">No events in the next 12 hours.</div>'
        )

        body = f"""
<h1>Pre-event briefings</h1>
<p class="muted" style="margin-top:-8px;">
    Before each event on your calendar(s), the daemon generates a "what
    you should know" brief — pulling from your indexed brain plus
    targeted web search for unfamiliar attendees / companies. Lookahead:
    <code>{cfg.briefing_lookahead_minutes} min</code>.
    {"<strong style='color:#ff5c5c;'>Web search is OFF</strong>; "
     "set <code>web_search_enabled = true</code> in config.toml so "
     "briefings can research attendees."
     if not cfg.web_search_enabled else ""}
</p>

<h2>Upcoming · next 12h</h2>
{upcoming_block}

<details class="brief-adhoc">
  <summary>+ ad-hoc briefing (event not on a calendar yet)</summary>
  <form method="post" action="/briefings/adhoc" class="brief-adhoc-form">
    <input type="text" name="title" placeholder="Event title" required>
    <input type="text" name="starts_at"
        placeholder="Start time, ISO-8601 (e.g. 2026-04-20T14:00)" required>
    <input type="text" name="location" placeholder="Location (optional)">
    <input type="text" name="attendees"
        placeholder="Attendees, comma-separated (optional)">
    <textarea name="description" rows="3" placeholder="Description (optional)"></textarea>
    <button type="submit">generate ad-hoc briefing</button>
  </form>
</details>

<style>
.brief-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-left: 3px solid var(--green-dim);
    padding: 14px 16px; margin: 12px 0;
}}
.brief-head {{ display: flex; gap: 14px; align-items: baseline; flex-wrap: wrap; }}
.brief-when {{ color: var(--green); font-family: var(--mono); font-weight: 600; }}
.brief-rel {{ color: #888; font-weight: 400; font-size: 11.5px; }}
.brief-title {{ flex: 1; }}
.brief-cal-link {{ font-size: 11.5px; color: #5af0ff; margin-left: 6px; font-family: var(--mono); }}
.brief-meta {{ display: flex; gap: 14px; margin: 6px 0; color: #888; font-size: 12px; flex-wrap: wrap; }}
.brief-cal {{ font-family: var(--mono); }}
.brief-row {{ display: flex; align-items: center; gap: 12px; margin: 8px 0; }}
.brief-status {{ font-size: 11.5px; font-family: var(--mono); letter-spacing: 0.04em; }}
.brief-pending {{ color: #888; }}
.brief-ready {{ color: var(--green); }}
.brief-error {{ color: #ff5c5c; }}
.brief-regen-form button {{
    font-size: 11px; padding: 4px 10px;
    background: transparent; color: #888; border: 1px solid var(--border);
    cursor: pointer; font-family: var(--mono);
}}
.brief-regen-form button:hover {{ color: var(--green); border-color: var(--green-dim); }}
.brief-text {{
    margin-top: 12px; padding: 12px 14px;
    background: #0a0a0a; border-left: 2px solid var(--border);
    line-height: 1.55; font-size: 13.5px;
}}
.brief-adhoc {{ margin-top: 24px; }}
.brief-adhoc summary {{
    cursor: pointer; padding: 8px 12px; background: #0e0e0e;
    border: 1px solid var(--border); border-radius: 2px; color: #888;
    font-size: 12.5px; font-family: var(--mono);
}}
.brief-adhoc summary:hover {{ color: var(--green); }}
.brief-adhoc-form {{
    display: flex; flex-direction: column; gap: 8px;
    padding: 14px; margin-top: 8px;
    background: var(--surface); border: 1px solid var(--border); border-radius: 4px;
}}
.brief-adhoc-form input, .brief-adhoc-form textarea {{
    background: #0e0e0e; color: var(--fg);
    border: 1px solid var(--border); border-radius: 2px;
    padding: 8px 10px; font: 13px var(--mono);
}}
</style>"""
        return HTMLResponse(_layout("Briefings", body, "briefings"))

    @app.post("/briefings/regenerate")
    async def briefings_regenerate(request: Request):
        from .event_briefing import generate_for_event, iter_upcoming_events

        cfg, conn, embedder, reranker = get_state()
        # CSRF guard.
        # Round 14 — uses the centralised _is_same_origin_request helper.
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        form = await request.form()
        eid = (form.get("event_id") or "").strip()
        src = (form.get("event_source") or "").strip()
        # Find the event in the upcoming window so we have current details.
        for ev in iter_upcoming_events(cfg, 24 * 3600):
            if ev.event_id == eid and ev.source == src:
                generate_for_event(cfg, conn, embedder, reranker, ev)
                break
        return RedirectResponse(url="/briefings", status_code=303)

    @app.post("/briefings/adhoc")
    async def briefings_adhoc(request: Request):
        from .event_briefing import generate_for_event, manual_event

        cfg, conn, embedder, reranker = get_state()
        # Round 14 — uses the centralised _is_same_origin_request helper.
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        form = await request.form()
        title = (form.get("title") or "").strip()
        starts_at = (form.get("starts_at") or "").strip()
        if not title or not starts_at:
            return RedirectResponse(url="/briefings", status_code=303)
        attendees_raw = (form.get("attendees") or "").strip()
        attendees = [
            a.strip() for a in attendees_raw.split(",") if a.strip()
        ]
        try:
            ev = manual_event(
                title=title, starts_at_iso=starts_at,
                description=(form.get("description") or "").strip(),
                attendees=attendees,
                location=(form.get("location") or "").strip(),
            )
        except ValueError:
            return RedirectResponse(url="/briefings", status_code=303)
        generate_for_event(cfg, conn, embedder, reranker, ev)
        return RedirectResponse(url="/briefings", status_code=303)

    # --- Reading queue ----------------------------------------------

    @app.get("/queue", response_class=HTMLResponse)
    def queue_page(history: int = 0):
        from .db import (
            reading_queue_history,
            reading_queue_unread,
            reading_queue_unread_count,
        )

        _, conn, _, _ = get_state()
        rows = (
            reading_queue_history(conn, limit=200) if history
            else reading_queue_unread(conn, limit=200)
        )
        unread_count = reading_queue_unread_count(conn)

        items_html: list[str] = []
        for r in rows:
            added = time.strftime("%Y-%m-%d", time.localtime(r["added_at"]))
            title = r["title"] or r["url"]
            host = ""
            try:
                from urllib.parse import urlparse as _urlparse
                host = _urlparse(r["url"]).netloc
            except Exception:  # noqa: BLE001
                pass
            fit_chip = ""
            if r["fit_label"]:
                cls = f"queue-fit-{r['fit_label'].split()[0]}"
                fit_chip = (
                    f'<span class="queue-fit {cls}">{escape(r["fit_label"])}</span>'
                )
            if r["summary_error"]:
                summary_block = (
                    f'<div class="queue-error">summary error: '
                    f'{escape(r["summary_error"][:200])}</div>'
                )
            elif r["summary"]:
                summary_block = (
                    f'<div class="queue-summary">'
                    f'{escape(r["summary"]).replace(chr(10), "<br>")}'
                    f'</div>'
                )
            else:
                summary_block = (
                    '<div class="queue-pending">summary pending…</div>'
                )

            actions_html = ""
            if not history:
                actions_html = f"""
<form method="post" action="/queue/{r['id']}/action" class="queue-actions">
    <button name="action" value="read" class="queue-btn queue-btn-read">read</button>
    <button name="action" value="skipped" class="queue-btn queue-btn-skip">skip</button>
</form>"""
            else:
                if r["read_at"]:
                    actions_html = '<span class="queue-status-read">✓ read</span>'
                elif r["skipped_at"]:
                    actions_html = '<span class="queue-status-skip">skipped</span>'

            items_html.append(f"""
<article class="queue-card">
  <header class="queue-head">
    <a href="{escape(r['url'])}" target="_blank" rel="noopener noreferrer"
       class="queue-title">{escape(title)} ↗</a>
    {fit_chip}
  </header>
  <div class="queue-meta">
    <span>{escape(host)}</span>
    <span>{escape(r['source'])}</span>
    <span>{added}</span>
  </div>
  {summary_block}
  {actions_html}
</article>""")

        items = (
            "".join(items_html)
            or '<div class="empty">'
            + ('No history yet.' if history else 'Nothing in your queue.')
            + '</div>'
        )

        toggle = (
            '<a href="/queue" class="queue-toggle">unread (' + str(unread_count) + ')</a> · '
            '<a href="/queue?history=1" class="queue-toggle queue-toggle-on">history</a>'
            if history else
            '<a href="/queue" class="queue-toggle queue-toggle-on">unread (' + str(unread_count) + ')</a> · '
            '<a href="/queue?history=1" class="queue-toggle">history</a>'
        )

        body = f"""
<h1>Reading queue</h1>
<p class="muted" style="margin-top:-8px;">
    Your watchlists auto-enqueue high-fit jobs and every news/research
    hit. The daemon writes a 60-second pre-read summary so you can scan
    instead of opening every tab. Mark <strong>read</strong> when you've
    finished one; <strong>skip</strong> when you don't care.
</p>

<div class="queue-toggles">{toggle}</div>

<form method="post" action="/queue/add" class="queue-add card">
    <input type="text" name="url" placeholder="https://... (manual add)" required>
    <input type="text" name="title" placeholder="Optional title">
    <button type="submit">+ queue this URL</button>
</form>

<div class="queue-list">{items}</div>

<style>
.queue-toggles {{ margin: 12px 0; font-family: var(--mono); font-size: 12px; }}
.queue-toggle {{ color: #888; text-decoration: none; padding: 0 4px; }}
.queue-toggle:hover {{ color: var(--green); }}
.queue-toggle-on {{ color: var(--green); }}
.queue-add {{ display: flex; gap: 8px; margin: 12px 0; }}
.queue-add input {{
    flex: 1; box-sizing: border-box;
    background: #0e0e0e; color: var(--fg);
    border: 1px solid var(--border); border-radius: 2px;
    padding: 8px 10px; font: 13px var(--mono);
}}
.queue-card {{
    background: var(--surface); border: 1px solid var(--border);
    border-left: 3px solid var(--green-dim);
    padding: 14px 16px; margin: 10px 0;
}}
.queue-head {{ display: flex; gap: 10px; align-items: baseline; flex-wrap: wrap; }}
.queue-title {{ color: var(--green); font-weight: 600; flex: 1; word-break: break-word; }}
.queue-meta {{
    display: flex; gap: 14px; margin: 6px 0;
    color: #888; font-size: 11.5px; font-family: var(--mono);
}}
.queue-summary {{
    margin: 10px 0 6px 0; padding: 10px 14px;
    background: #0a0a0a; border-left: 2px solid var(--border);
    line-height: 1.6; font-size: 13px;
}}
.queue-pending {{ color: #888; font-size: 11.5px; font-style: italic; padding: 8px 0; }}
.queue-error {{ color: #ff5c5c; font-size: 12px; padding: 6px 0; }}
.queue-fit {{
    display: inline-block; padding: 1px 6px; font-size: 10.5px;
    border-radius: 2px; border: 1px solid var(--border);
    letter-spacing: 0.04em;
}}
.queue-fit-great {{ background: #16201a; color: #b8ffb8; border-color: #4abe4a; }}
.queue-fit-decent {{ background: #1c1c1c; color: #ffd566; border-color: #5a4a14; }}
.queue-fit-stretch {{ background: #1a1414; color: #ffaa66; border-color: #5a3a14; }}
.queue-actions {{ display: flex; gap: 6px; margin-top: 6px; }}
.queue-btn {{
    background: transparent; border: 1px solid var(--border);
    color: #888; padding: 4px 10px; cursor: pointer;
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.04em;
    border-radius: 2px;
}}
.queue-btn-read:hover {{ color: var(--green); border-color: var(--green-dim); }}
.queue-btn-skip:hover {{ color: #ff5c5c; border-color: #5a2828; }}
.queue-status-read {{ color: var(--green); font-size: 11.5px; font-family: var(--mono); }}
.queue-status-skip {{ color: #888; font-size: 11.5px; font-family: var(--mono); }}
</style>"""
        return HTMLResponse(_layout("Queue", body, "queue"))

    @app.post("/queue/add")
    async def queue_add(request: Request):
        from .db import reading_queue_enqueue

        _, conn, _, _ = get_state()
        # Round 14 — uses the centralised _is_same_origin_request helper.
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        form = await request.form()
        url = (form.get("url") or "").strip()
        title = (form.get("title") or "").strip()
        if url:
            reading_queue_enqueue(conn, url=url, title=title, source="manual")
        return RedirectResponse(url="/queue", status_code=303)

    @app.post("/queue/{qid:int}/action")
    async def queue_action(qid: int, request: Request):
        from .db import (
            reading_queue_get,
            reading_queue_mark_read,
            reading_queue_mark_skipped,
        )

        _, conn, _, _ = get_state()
        # Round 14 — uses the centralised _is_same_origin_request helper.
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        form = await request.form()
        action = (form.get("action") or "").strip()
        if reading_queue_get(conn, qid) is None:
            return RedirectResponse(url="/queue", status_code=303)
        if action == "read":
            reading_queue_mark_read(conn, qid)
        elif action == "skipped":
            reading_queue_mark_skipped(conn, qid)
        return RedirectResponse(url="/queue", status_code=303)

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

    @app.get("/queries", response_class=HTMLResponse)
    def queries_page(limit: int = 100):
        """Audit panel: shows recent AI-driven searches against your brain."""
        import json as _json

        cfg, _, _, _ = get_state()
        log_path = cfg.data_dir / "queries.jsonl"
        rows: list[dict] = []
        if log_path.exists():
            try:
                with open(log_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rows.append(_json.loads(line))
                        except _json.JSONDecodeError:
                            continue
            except OSError:
                pass
        rows = rows[-limit:]
        rows.reverse()

        if not rows:
            body = """
<h1>Query log</h1>
<div class="empty">
    No queries logged yet. AI assistants that call <code>search_brain</code>
    via MCP will show up here so you can see what's being retrieved.
</div>"""
            return HTMLResponse(_layout("Queries", body, "queries"))

        # Aggregate stats
        total = len(rows)
        by_tool: dict[str, int] = {}
        for r in rows:
            tool = r.get("tool", "?")
            by_tool[tool] = by_tool.get(tool, 0) + 1
        tool_summary = ", ".join(f"{k}={v}" for k, v in sorted(by_tool.items()))

        # Render rows
        items = []
        for r in rows:
            ts = r.get("ts", 0)
            age_min = (time.time() - ts) / 60 if ts else 0
            age_str = (
                f"{age_min:.1f}m ago" if age_min < 60
                else f"{age_min/60:.1f}h ago" if age_min < 1440
                else f"{age_min/1440:.1f}d ago"
            )
            paths_html = "".join(
                f'<div class="path"><a href="/file?path={urllib.parse.quote_plus(p)}">{escape(p)}</a></div>'
                for p in r.get("top_paths", [])[:5]
            ) or '<div class="muted">(no results)</div>'
            items.append(f"""
<article class="result">
    <h3>{escape(r.get('query', ''))!s}</h3>
    <div class="meta">{escape(r.get('tool', '?'))} · k={r.get('k', 0)} · {age_str}</div>
    {paths_html}
</article>""")

        body = f"""
<h1>Query log <span class="muted" style="font-size:13px;">({total} recent · {escape(tool_summary)})</span></h1>
<p class="muted" style="margin-bottom:24px;">Every <code>search_brain</code> call from any MCP-connected AI gets logged here. The query, the tool, and the file paths returned — so you know what's leaving the brain.</p>
{''.join(items)}"""
        return HTMLResponse(_layout("Queries", body, "queries"))

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
    def ingest_action(request: Request, url: str = Form(...)):
        cfg, conn, embedder, _ = get_state()
        # CSRF guard: form-POSTs aren't gated by CORS, so any page the
        # user visits could otherwise force-ingest an arbitrary URL by
        # submitting a hidden form to http://127.0.0.1:8765/ingest.
        # Round 14 — uses centralised _is_same_origin_request helper.
        if not _is_same_origin_request(request):
            return HTMLResponse(
                "<h1>Forbidden</h1>"
                "<p>Cross-origin POSTs to /ingest are blocked.</p>",
                status_code=403,
            )
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

    @app.get("/healthz")
    def healthz():
        """Liveness probe — `{ok: true}` if the FastAPI process is up.
        Renamed from `/health` so the user-facing health page (Phase 56)
        can own that route."""
        return {"ok": True}

    # --- Phase 44: morning brief view ---------------------------------

    @app.get("/brief", response_class=HTMLResponse)
    def brief_page():
        """Render today's morning brief from the live aggregator.
        Re-renders on every load — cheap (pure SQL aggregation), so
        the page always reflects fresh data."""
        from .daily_brief import generate_brief_markdown

        cfg, conn, _, _ = get_state()
        md = generate_brief_markdown(cfg, conn)
        body = (
            "<h1>Daily brief</h1>"
            "<div class='brief-md'>"
            + _markdown_to_html_block(md)
            + "</div>"
            "<p class='muted' style='margin-top:24px;'>"
            "Reloads on each visit. Run "
            "<code>secondbrain brief send</code> to email it now."
            "</p>"
        )
        return HTMLResponse(_layout("Daily brief", body, "brief"))

    # --- Round 16 (Phase B): weekly review letter --------------------

    @app.get("/review", response_class=HTMLResponse)
    def review_page(week_end: str | None = None):
        """Show the weekly letter. Default: latest. Pass
        ``?week_end=YYYY-MM-DD`` to view a historical letter.

        The letter is auto-generated by the daemon every Sunday;
        users can also force a regen via the form below."""
        from . import weekly_letter as wl

        _, conn, _, _ = get_state()
        if week_end:
            letter = wl.get_letter(conn, week_end)
        else:
            letter = wl.latest_letter(conn)
        history = wl.list_letters(conn, limit=20)
        # Sidebar: history list with active-week highlight.
        active_week = letter.week_end if letter else None
        history_items = []
        for h in history:
            cls = "active" if h.week_end == active_week else ""
            history_items.append(
                f'<li><a href="/review?week_end={h.week_end}" '
                f'class="{cls}">{escape(h.week_end)} '
                f'<span class="muted" style="font-size:11px;">'
                f'· {h.model.split(":")[0]}</span></a></li>'
            )
        history_html = (
            f'<aside class="card">'
            f'<h2 style="margin-top:0;">History</h2>'
            f'<ul class="review-history">{"".join(history_items)}</ul>'
            f'<form method="post" action="/review/regenerate" '
            f'style="margin-top:16px;">'
            f'<button type="submit" class="link-btn">Regenerate this week</button>'
            f'</form>'
            f'</aside>'
            if history else
            '<aside class="card empty">'
            '<h2>No letters yet</h2>'
            '<p class="muted">Click below to generate this week\'s letter, '
            'or wait — the daemon writes one every Sunday morning.</p>'
            '<form method="post" action="/review/regenerate">'
            '<button type="submit" class="link-btn">Generate now</button>'
            '</form>'
            '</aside>'
        )
        if letter is None:
            main = (
                '<div class="card empty">'
                '<h2>Nothing to show</h2>'
                '<p>No letter generated yet. Use the button on the right.</p>'
                '</div>'
            )
        else:
            from datetime import datetime as _dt
            gen_str = _dt.fromtimestamp(letter.generated_at).strftime(
                "%a %b %d, %Y at %H:%M",
            )
            main = (
                '<div class="card review-letter">'
                + _markdown_to_html_block(letter.letter_md)
                + f'<p class="muted" style="margin-top:24px;'
                  f'font-size:11px;border-top:1px solid var(--border);'
                  f'padding-top:12px;">'
                  f'Generated {escape(gen_str)} · model '
                  f'<code>{escape(letter.model)}</code> · '
                  f'{letter.cost_cents:.2f}¢ · '
                  f'covers {escape(letter.week_start)} → '
                  f'{escape(letter.week_end)}'
                  f'</p>'
                + '</div>'
            )
        body = (
            f'<h1>Weekly letter</h1>'
            f'<div class="review-grid">'
            f'<div class="review-main">{main}</div>'
            f'{history_html}'
            f'</div>'
            # Bit of inline CSS scoped to this page.
            '<style>'
            '.review-grid { display: grid; '
            '  grid-template-columns: 1fr 240px; gap: 24px; '
            '  align-items: start; }'
            '@media (max-width: 900px) { '
            '  .review-grid { grid-template-columns: 1fr; } }'
            '.review-letter { font-size: 14px; line-height: 1.7; '
            '  max-width: 720px; }'
            '.review-letter h1, .review-letter h2 { '
            '  border-bottom: 1px solid var(--border); '
            '  padding-bottom: 4px; }'
            '.review-history { list-style: none; padding: 0; margin: 0; }'
            '.review-history li { margin: 4px 0; }'
            '.review-history a { display: block; padding: 6px 10px; '
            '  border-radius: var(--r-md); text-decoration: none; '
            '  color: var(--text); }'
            '.review-history a:hover { background: var(--bg-hover); }'
            '.review-history a.active { background: var(--accent-soft); '
            '  color: var(--accent); }'
            '</style>'
        )
        return HTMLResponse(_layout("Weekly review", body, "review"))

    @app.post("/review/regenerate")
    def review_regenerate(request: Request):
        """Force a fresh LLM call for this week's letter."""
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        from . import weekly_letter as wl
        cfg, conn, _, _ = get_state()
        try:
            letter = wl.generate_and_save(cfg, conn, overwrite=True)
            return RedirectResponse(
                url=f"/review?week_end={letter.week_end}", status_code=303,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("review regen failed")
            return HTMLResponse(
                f'<h1>Regeneration failed</h1><pre>{escape(str(e))}</pre>'
                f'<p><a href="/review">Back</a></p>',
                status_code=500,
            )

    # --- Round 16 (Phase C): notifications -------------------------

    @app.get("/notifications", response_class=HTMLResponse)
    def notifications_page():
        from . import notifications as notif_mod
        _, conn, _, _ = get_state()
        # Round 17 fix (audit-found perf gap) — call the throttled
        # variant so a tab-refresh storm doesn't trigger 7 detectors
        # × full-table scans per load. Hourly scheduler is the source
        # of truth; this is just a "freshen on visit" hook.
        try:
            notif_mod.detect_all_throttled(conn)
        except Exception:  # noqa: BLE001
            pass
        pending = notif_mod.list_pending(conn, limit=50)
        recent_dismissed = [
            n for n in notif_mod.list_recent(conn, limit=80)
            if n.status != "pending"
        ][:30]

        urgency_class = {"high": "urgent", "med": "med", "low": "low"}

        def _row(n):
            ago = max(0, int((time.time() - n.created_at) // 60))
            ago_str = (
                f"{ago}m ago" if ago < 60 else
                f"{ago // 60}h ago" if ago < 1440 else
                f"{ago // 1440}d ago"
            )
            href_html = (
                f'<a href="{escape(n.href)}" class="notif-link">'
                f'{escape(n.title)}</a>' if n.href else escape(n.title)
            )
            cls = urgency_class.get(n.urgency, "med")
            actions = ""
            if n.status == "pending":
                actions = (
                    f'<form method="post" '
                    f'action="/notifications/{n.id}/dismiss" '
                    f'style="display:inline">'
                    f'<button class="link-btn" type="submit">'
                    f'dismiss</button></form>'
                )
            body_html = (
                f'<p class="notif-body">{escape(n.body)}</p>'
                if n.body else ""
            )
            return (
                f'<div class="notif-row notif-{cls}">'
                f'  <div class="notif-meta">'
                f'    <span class="notif-kind">{escape(n.kind)}</span>'
                f'    <span class="muted">· {escape(ago_str)}</span>'
                f'    <span class="notif-actions">{actions}</span>'
                f'  </div>'
                f'  <div class="notif-title">{href_html}</div>'
                f'  {body_html}'
                f'</div>'
            )

        if pending:
            pending_html = "".join(_row(n) for n in pending)
        else:
            pending_html = (
                '<div class="empty">'
                '<p>Nothing pending. The brain will surface things '
                'here when there\'s something worth your attention — '
                'urgent emails, birthdays in 3 days, missed habits, '
                'broken integrations, that kind of thing.</p>'
                '</div>'
            )
        history_html = (
            "".join(_row(n) for n in recent_dismissed)
            if recent_dismissed else
            '<p class="muted">Nothing yet.</p>'
        )
        body = (
            '<h1>Inbox</h1>'
            f'<div class="card">'
            f'<h2 style="margin-top:0;">Pending '
            f'<span class="muted" style="font-size:13px;">'
            f'({len(pending)})</span></h2>'
            f'<div class="notif-list">{pending_html}</div>'
            f'<form method="post" action="/notifications/dismiss-all" '
            f'style="margin-top:16px;">'
            f'<button class="link-btn" type="submit">'
            f'Dismiss all</button></form>'
            f'</div>'
            f'<details class="card" style="margin-top:24px;">'
            f'<summary><h2 style="display:inline;margin:0;">'
            f'Recent</h2></summary>'
            f'<div class="notif-list" style="margin-top:12px;">'
            f'{history_html}</div>'
            f'</details>'
            '<style>'
            '.notif-row { padding: 12px; border-left: 3px solid var(--border); '
            '  margin-bottom: 8px; background: var(--bg-elevated); }'
            '.notif-urgent { border-left-color: var(--red); }'
            '.notif-med { border-left-color: var(--amber); }'
            '.notif-low { border-left-color: var(--text-3); }'
            '.notif-meta { font-size: 11px; color: var(--text-2); '
            '  display: flex; gap: 8px; align-items: center; }'
            '.notif-kind { background: var(--bg); padding: 1px 6px; '
            '  border-radius: 2px; }'
            '.notif-actions { margin-left: auto; }'
            '.notif-title { margin-top: 4px; font-weight: 500; }'
            '.notif-body { margin-top: 4px; color: var(--text-2); '
            '  font-size: 13px; }'
            '.notif-link { color: var(--accent); }'
            '</style>'
        )
        return HTMLResponse(_layout(
            f"Inbox ({len(pending)})", body, "notifications",
        ))

    @app.post("/notifications/{nid:int}/dismiss")
    def notifications_dismiss(nid: int, request: Request):
        from . import notifications as notif_mod
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        _, conn, _, _ = get_state()
        notif_mod.mark_dismissed(conn, nid)
        return RedirectResponse(url="/notifications", status_code=303)

    @app.post("/notifications/dismiss-all")
    def notifications_dismiss_all(request: Request):
        from . import notifications as notif_mod
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        _, conn, _, _ = get_state()
        notif_mod.dismiss_all(conn)
        return RedirectResponse(url="/notifications", status_code=303)

    # --- Round 16 (Phase G): unified timeline -----------------------

    @app.get("/timeline", response_class=HTMLResponse)
    def timeline_page(
        # Round 18 fix (audit-found gap H2) — the route's own links
        # generate ``?date=...`` but the parameter was named
        # ``date_str``, so FastAPI never bound it and every nav link
        # snapped back to today. ``Query(alias="date")`` makes the
        # parameter accept ``?date=...`` without shadowing Python's
        # built-in ``date``.
        date_str: str | None = Query(None, alias="date"),
        days: int = 1,
        kinds: str | None = None,
    ):
        """Unified chronological view across files / tasks / journal /
        habits / health / email / notifications / weekly letters.

        Query: ``?date=YYYY-MM-DD&days=7&kinds=task,journal``.
        """
        from datetime import date as _date
        from datetime import timedelta as _td

        from . import timeline

        _, conn, _, _ = get_state()
        # Round 18 fix (audit-found gap H1) — closed-vocabulary
        # filter for ``kinds`` BEFORE reflecting it back into HTML.
        # The earlier code interpolated raw ``kinds=`` into ``href``
        # attributes; an attacker-supplied query string like
        # ``?kinds="><script>alert(1)</script>`` broke out of the
        # attribute and ran JS in the dashboard origin (loopback
        # access to every authenticated route). Whitelist + escape.
        all_kinds = [
            "file", "task", "journal", "habit", "health",
            "email", "notif", "weekly",
        ]
        _all_kinds_set = set(all_kinds)
        kind_set: set[str] | None = None
        if kinds:
            kind_set = {
                k.strip() for k in kinds.split(",")
                if k.strip() in _all_kinds_set
            } or None
        # Re-serialise ONLY the validated tokens so the query-string
        # we reflect back is provably safe (no quotes, angle brackets,
        # etc. — only [a-z]+ tokens from ``all_kinds``).
        safe_kinds_str = ",".join(sorted(kind_set)) if kind_set else ""

        # Cap the window so a 365-day query doesn't stall the page.
        days = max(1, min(int(days or 1), 30))
        since_ts, until_ts = timeline.parse_window(date_str, days)
        events = timeline.assemble(
            conn, since_ts, until_ts, kinds=kind_set,
        )
        grouped = timeline.group_by_date(events)

        # Header: prev/next-day nav + filter chips.
        try:
            anchor = _date.fromisoformat(date_str) if date_str else _date.today()
        except ValueError:
            anchor = _date.today()
        prev_d = (anchor - _td(days=days)).isoformat()
        next_d = (anchor + _td(days=days)).isoformat()
        days_options = [1, 3, 7, 14, 30]
        days_chip_html = " ".join(
            f'<a class="chip {"active" if d == days else ""}" '
            f'href="/timeline?date={anchor.isoformat()}&days={d}'
            + (f'&kinds={safe_kinds_str}' if safe_kinds_str else '')
            + '">'
            f'{d}d</a>'
            for d in days_options
        )
        active_kinds = kind_set or set(all_kinds)
        kind_chip_html = " ".join(
            f'<a class="chip {"active" if k in active_kinds else ""}" '
            f'href="/timeline?date={anchor.isoformat()}&days={days}'
            f'&kinds={k}">{k}</a>'
            for k in all_kinds
        ) + (
            f' <a class="chip" href="/timeline?date={anchor.isoformat()}'
            f'&days={days}">all</a>'
        )

        if not events:
            grouped_html = (
                '<div class="empty card">'
                '<p>Nothing recorded in this window.</p></div>'
            )
        else:
            day_blocks = []
            for d in sorted(grouped.keys(), reverse=True):
                rows_html = "".join(
                    f'<div class="tl-row">'
                    f'  <span class="tl-time">{e.time_str}</span>'
                    f'  <span class="tl-icon">{e.icon or "·"}</span>'
                    f'  <span class="tl-kind">{escape(e.kind)}</span>'
                    f'  <span class="tl-title">'
                    + (
                        f'<a href="{escape(e.href)}">{escape(e.title)}</a>'
                        if e.href else escape(e.title)
                    )
                    + '</span>'
                    + (
                        f'<span class="tl-detail muted">'
                        f' — {escape(e.detail)}</span>'
                        if e.detail else ''
                    )
                    + '</div>'
                    for e in grouped[d]
                )
                day_blocks.append(
                    f'<section class="tl-day">'
                    f'<h2 class="tl-date">{escape(d)}'
                    f'<span class="muted" style="font-size:12px;'
                    f'font-weight:normal;"> · {len(grouped[d])} '
                    f'event(s)</span></h2>'
                    f'<div class="tl-rows">{rows_html}</div>'
                    f'</section>'
                )
            grouped_html = "\n".join(day_blocks)

        body = (
            f'<h1>Timeline</h1>'
            f'<div class="tl-controls card">'
            f'  <div class="tl-nav">'
            f'    <a href="/timeline?date={prev_d}&days={days}'
            + (f'&kinds={safe_kinds_str}' if safe_kinds_str else '')
            + f'" class="link-btn">← {days}d earlier</a>'
            f'    <span class="tl-anchor"><strong>'
            f'{escape(anchor.isoformat())}</strong>'
            f' <span class="muted">({days} day{"s" if days != 1 else ""})'
            f'</span></span>'
            f'    <a href="/timeline?date={next_d}&days={days}'
            + (f'&kinds={safe_kinds_str}' if safe_kinds_str else '')
            + f'" class="link-btn">{days}d later →</a>'
            f'  </div>'
            f'  <div class="tl-chips">'
            f'    <span class="muted" style="font-size:11px;">window:</span> '
            f'{days_chip_html}'
            f'  </div>'
            f'  <div class="tl-chips">'
            f'    <span class="muted" style="font-size:11px;">kind:</span> '
            f'{kind_chip_html}'
            f'  </div>'
            f'</div>'
            f'{grouped_html}'
            '<style>'
            '.tl-controls { display: flex; flex-direction: column; '
            '  gap: 12px; }'
            '.tl-nav { display: flex; align-items: center; '
            '  justify-content: space-between; }'
            '.tl-anchor { font-size: 14px; }'
            '.tl-chips { display: flex; gap: 4px; flex-wrap: wrap; '
            '  align-items: center; }'
            '.chip { display: inline-block; padding: 2px 10px; '
            '  border: 1px solid var(--border); border-radius: 12px; '
            '  font-size: 11px; text-decoration: none; color: var(--text-2); }'
            '.chip:hover { background: var(--bg-hover); '
            '  color: var(--text); }'
            '.chip.active { background: var(--accent-soft); '
            '  border-color: var(--accent); color: var(--accent); }'
            '.tl-day { margin-top: 24px; }'
            '.tl-date { border-bottom: 1px solid var(--border); '
            '  padding-bottom: 4px; margin-bottom: 8px; }'
            '.tl-rows { font-family: var(--mono); font-size: 13px; }'
            '.tl-row { display: grid; '
            '  grid-template-columns: 56px 24px 110px 1fr; '
            '  gap: 8px; padding: 4px 6px; '
            '  border-bottom: 1px dotted var(--border); '
            '  align-items: baseline; }'
            '.tl-row:hover { background: var(--bg-hover); }'
            '.tl-time { color: var(--text-3); }'
            '.tl-icon { text-align: center; }'
            '.tl-kind { color: var(--text-2); font-size: 11px; }'
            '.tl-title { white-space: nowrap; overflow: hidden; '
            '  text-overflow: ellipsis; }'
            '.tl-detail { font-size: 12px; }'
            '@media (max-width: 700px) { '
            '  .tl-row { grid-template-columns: 44px 18px 1fr; }'
            '  .tl-kind, .tl-detail { display: none; } }'
            '</style>'
        )
        return HTMLResponse(_layout("Timeline", body, "timeline"))

    # ============================ Round 19 — EA features ==============

    @app.get("/followups", response_class=HTMLResponse)
    def followups_page(
        person_id: int = 0,
        show: str = "active",  # 'active' (default), 'snoozed', 'history'
    ):
        """Round 19+20 — bidirectional commitment tracker.

        Round 20 additions:
          - ``?person_id=N`` filters to one person.
          - ``?show=snoozed`` shows snoozed items.
          - ``?show=history`` shows recently resolved/dismissed.
          - Snooze / edit / nudge buttons inline per row.
          - Header stats: open/overdue/avg-resolve-days.
        """
        from . import followups as fu_mod
        from . import followups_ops

        _, conn, _, _ = get_state()
        if show == "history":
            history = followups_ops.list_resolved_history(conn, days=30)
            rows_html = []
            for f, elapsed in history:
                elapsed_str = (
                    f" · resolved in {int(elapsed/86400.0)}d"
                    if elapsed else ""
                )
                kind = (
                    "→" if f.direction == "outgoing" else "←"
                )
                rows_html.append(
                    f'<div class="fu-row">'
                    f'<div class="fu-meta">'
                    f'<span class="muted">{escape(f.status)}</span>'
                    f'{elapsed_str}</div>'
                    f'<div>{kind} <strong>'
                    f'{escape(_safe(f.topic))}</strong></div>'
                    f'<div class="muted">'
                    f'{escape(_safe(f.person_name)) or "—"}</div>'
                    f'</div>'
                )
            body = (
                f'<h1>Follow-up history</h1>'
                f'<p class="muted">{len(history)} resolved/dismissed in '
                f'the last 30 days.</p>'
                f'<p><a href="/followups">← back to active</a></p>'
                f'{"".join(rows_html)}'
            )
            return HTMLResponse(_layout("Follow-ups", body, "followups"))

        include_snoozed = (show == "snoozed")
        out_rows = followups_ops.list_visible_open(
            conn, direction="outgoing",
            person_id=(person_id or None),
            include_snoozed=include_snoozed,
            limit=100,
        )
        in_rows = followups_ops.list_visible_open(
            conn, direction="incoming",
            person_id=(person_id or None),
            include_snoozed=include_snoozed,
            limit=100,
        )
        overdue = fu_mod.list_overdue(conn)
        stats = followups_ops.compute_stats(conn)

        def _row_html(f) -> str:
            age_days = (
                int((time.time() - f.promised_at) / 86400.0)
                if f.promised_at else None
            )
            age_str = f"{age_days}d" if age_days is not None else "—"
            due_str = ""
            overdue_class = ""
            if f.due_at:
                from datetime import date as _date
                due_d = _date.fromtimestamp(f.due_at).isoformat()
                if f.due_at < time.time():
                    overdue_class = "fu-overdue"
                    due_str = (
                        f' <span class="muted">overdue '
                        f'(was {escape(due_d)})</span>'
                    )
                else:
                    due_str = (
                        f' <span class="muted">due {escape(due_d)}'
                        f'</span>'
                    )
            person_link = ""
            if f.person_id:
                person_link = (
                    f'<a class="muted" '
                    f'href="/followups?person_id={f.person_id}">'
                    f'{escape(_safe(f.person_name)) or "—"}</a>'
                )
            elif f.person_name:
                person_link = (
                    f'<span class="muted">'
                    f'{escape(_safe(f.person_name))}</span>'
                )
            else:
                person_link = '<span class="muted">—</span>'
            source_link = ""
            if f.source_file_id:
                source_link = (
                    f' <a class="muted" '
                    f'href="/file?file_id={f.source_file_id}">'
                    f'(source)</a>'
                )
            # Snooze badge.
            snooze_html = ""
            try:
                row = conn.execute(
                    "SELECT snooze_until FROM followups WHERE id = ?",
                    (f.id,),
                ).fetchone()
                if row and row["snooze_until"] and (
                    row["snooze_until"] > time.time()
                ):
                    from datetime import date as _date
                    until_d = _date.fromtimestamp(
                        row["snooze_until"],
                    ).isoformat()
                    snooze_html = (
                        f' <span class="tag warn">snoozed → '
                        f'{escape(until_d)}</span>'
                    )
            except Exception:  # noqa: BLE001
                pass
            # Inline action buttons.
            nudge_btn = (
                f'  <form method="post" '
                f'    action="/followups/{f.id}/nudge" '
                f'    style="display:inline;">'
                f'    <button type="submit" title="Draft nudge email">'
                f'nudge</button>'
                f'  </form>'
            ) if f.direction == "incoming" else ""
            unsnooze_btn = (
                f'  <form method="post" '
                f'    action="/followups/{f.id}/unsnooze" '
                f'    style="display:inline;">'
                f'    <button type="submit">unsnooze</button>'
                f'  </form>'
            ) if snooze_html else ""
            return (
                f'<div class="fu-row {overdue_class}" id="fu{f.id}">'
                f'<div class="fu-meta">'
                f'  {person_link} <span class="muted">·</span>'
                f'  <span class="muted">{age_str} old</span>'
                f'  {due_str}{source_link}{snooze_html}'
                f'</div>'
                f'<div class="fu-topic">{escape(_safe(f.topic))}</div>'
                f'<div class="fu-desc muted">'
                f'{escape(_safe(f.description))}</div>'
                f'<div class="fu-actions">'
                f'  <form method="post" '
                f'    action="/followups/{f.id}/resolve" '
                f'    style="display:inline;">'
                f'    <button type="submit">resolve</button>'
                f'  </form>'
                f'  <form method="post" '
                f'    action="/followups/{f.id}/snooze" '
                f'    style="display:inline;">'
                f'    <input type="hidden" name="days" value="7">'
                f'    <button type="submit" '
                f'      title="Snooze 7d">snooze 7d</button>'
                f'  </form>'
                f'  {nudge_btn}'
                f'  {unsnooze_btn}'
                f'  <form method="post" '
                f'    action="/followups/{f.id}/dismiss" '
                f'    style="display:inline;">'
                f'    <button type="submit" '
                f'      style="background:#222;">dismiss</button>'
                f'  </form>'
                f'</div>'
                f'</div>'
            )

        out_html = "".join(_row_html(f) for f in out_rows) or (
            '<div class="muted">_(nothing pending — clean slate)_</div>'
        )
        in_html = "".join(_row_html(f) for f in in_rows) or (
            '<div class="muted">_(nothing pending — clean slate)_</div>'
        )

        person_filter_html = ""
        if person_id:
            from . import people as people_mod
            p = people_mod.get_person(conn, person_id)
            if p:
                person_filter_html = (
                    f'<div class="card" style="background:'
                    f'var(--green-soft);">'
                    f'Filtering: <strong>{escape(_safe(p.display_name))}'
                    f'</strong> '
                    f'<a href="/followups">(clear)</a></div>'
                )
        view_toggle_html = (
            f'<p class="muted">View: '
            f'<a href="/followups">active</a> · '
            f'<a href="/followups?show=snoozed">'
            f'snoozed ({stats.snoozed_count})</a> · '
            f'<a href="/followups?show=history">history</a></p>'
        )

        avg_str = (
            f"{stats.avg_resolve_days_30d:.1f}d"
            if stats.avg_resolve_days_30d is not None else "—"
        )
        stats_html = (
            f'<div class="fu-stats">'
            f'  <span><strong>{stats.open_outgoing}</strong> '
            f'  you owe</span>'
            f'  <span><strong>{stats.open_incoming}</strong> '
            f'  owed to you</span>'
            f'  <span><strong>{stats.overdue_count}</strong> '
            f'  overdue</span>'
            f'  <span class="muted">avg resolve: {avg_str}</span>'
            f'  <span class="muted">resolved last 30d: '
            f'  {stats.resolved_last_30d} '
            f'  ({stats.auto_resolved_last_30d} auto)</span>'
            f'</div>'
        )

        overdue_strip = ""
        if overdue:
            overdue_strip = (
                f'<div class="card fu-overdue-strip">'
                f'<strong>{len(overdue)} overdue</strong>'
                f' — items past their due date'
                f'</div>'
            )
        body = (
            f'<h1>Follow-ups</h1>'
            f'<p class="muted">'
            f'  Bidirectional commitment tracker. Outgoing = you owe '
            f'  others; Incoming = others owe you.'
            f'</p>'
            f'{stats_html}'
            f'{view_toggle_html}'
            f'{person_filter_html}'
            f'{overdue_strip}'
            f'<div class="fu-cols">'
            f'  <section class="card">'
            f'    <h2>You owe ({len(out_rows)})</h2>'
            f'    {out_html}'
            f'  </section>'
            f'  <section class="card">'
            f'    <h2>Owed to you ({len(in_rows)})</h2>'
            f'    {in_html}'
            f'  </section>'
            f'</div>'
            '<style>'
            '.fu-cols { display: grid; grid-template-columns: 1fr 1fr; '
            '  gap: var(--s-3); margin-top: var(--s-3); }'
            '@media (max-width: 800px) { .fu-cols { '
            '  grid-template-columns: 1fr; } }'
            '.fu-row { padding: var(--s-2) 0; '
            '  border-bottom: 1px dashed var(--border); }'
            '.fu-row:last-child { border-bottom: none; }'
            '.fu-row.fu-overdue { border-left: 2px solid var(--red); '
            '  padding-left: var(--s-2); }'
            '.fu-meta { font-size: 11px; }'
            '.fu-topic { font-weight: bold; margin-top: 2px; }'
            '.fu-desc { font-size: 12.5px; margin-top: 2px; }'
            '.fu-actions { margin-top: var(--s-1); }'
            '.fu-actions button { font-size: 11px; '
            '  padding: 2px 8px; margin-right: 4px; }'
            '.fu-overdue-strip { background: rgba(255,77,77,0.10); '
            '  border-color: var(--red); margin-top: var(--s-3); }'
            '.fu-stats { display: flex; gap: var(--s-3); '
            '  flex-wrap: wrap; padding: var(--s-2) var(--s-3); '
            '  background: var(--bg-card); '
            '  border: 1px solid var(--border); '
            '  border-radius: var(--r); margin: var(--s-2) 0; }'
            '.fu-stats span { font-size: 12px; }'
            '.tag { font-size: 10px; padding: 1px 6px; '
            '  border-radius: var(--r); margin-left: 4px; }'
            '.tag.warn { background: rgba(255,183,0,0.10); '
            '  color: var(--amber); }'
            '</style>'
        )
        return HTMLResponse(_layout("Follow-ups", body, "followups"))

    @app.post("/followups/{followup_id:int}/resolve")
    def followup_resolve(followup_id: int, request: Request):
        from . import followups as fu_mod
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        _, conn, _, _ = get_state()
        fu_mod.mark_resolved(conn, followup_id)
        return RedirectResponse(url="/followups", status_code=303)

    @app.post("/followups/{followup_id:int}/dismiss")
    def followup_dismiss(followup_id: int, request: Request):
        from . import followups as fu_mod
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        _, conn, _, _ = get_state()
        fu_mod.mark_dismissed(conn, followup_id)
        return RedirectResponse(url="/followups", status_code=303)

    @app.get("/agenda", response_class=HTMLResponse)
    def agenda_page(id: int = 0):  # noqa: A002
        """Round 19 (Phase EA-2) — agenda index OR per-person card.

        Without ``?id=`` shows a list of people to pick from.
        With ``?id=`` renders the 1:1 agenda for that person.
        """
        from . import agenda as agenda_mod
        from . import people as people_mod

        _, conn, _, _ = get_state()
        if id:
            result = agenda_mod.build_agenda(conn, id)
            if result is None:
                return HTMLResponse(
                    "<h1>Person not found</h1>", status_code=404,
                )
            md = agenda_mod.render_markdown(result)
            # Round 21 fix (audit-found gap I2) — render an "Add
            # note" form + per-note "discussed" buttons inline so
            # the user-curated notes layer is actually usable from
            # the dashboard. Previously the POST endpoints existed
            # but had no UI affordances.
            user_notes = agenda_mod.list_notes(conn, id)
            notes_html_rows = "".join(
                f'<li>'
                f'  <span>{escape(_safe(n.text))}</span> '
                f'  <form method="post" '
                f'    action="/agenda/note/{n.id}/discussed" '
                f'    style="display:inline;">'
                f'    <button type="submit">discussed</button>'
                f'  </form>'
                f'</li>'
                for n in user_notes
            )
            notes_section = (
                f'<section class="card">'
                f'  <h2>Things to bring up</h2>'
                f'  <form method="post" action="/agenda/{id}/note">'
                f'    <input type="text" name="text" '
                f'      placeholder="Add a topic..." '
                f'      style="width:60%;">'
                f'    <button type="submit">Add</button>'
                f'  </form>'
                + (
                    f'<ul class="agenda-note-list">{notes_html_rows}</ul>'
                    if notes_html_rows else
                    '<p class="muted">_(no pending notes)_</p>'
                )
                + '</section>'
            )
            body = (
                f'<h1>1:1 with {escape(_safe(result.person_name))}</h1>'
                f'<p class="muted">'
                f'  {result.total_items} item(s) on the table.'
                f'</p>'
                f'{notes_section}'
                f'<div class="briefing-body">'
                f'{_markdown_to_html_block(md)}'
                f'</div>'
                f'<details style="margin-top:var(--s-4);">'
                f'<summary class="muted">Markdown source</summary>'
                f'<pre style="white-space:pre-wrap;">{escape(md)}</pre>'
                f'</details>'
                f'<p style="margin-top:var(--s-4);">'
                f'<a class="link-btn" href="/agenda">'
                f'← all people</a></p>'
                '<style>'
                '.agenda-note-list { padding-left: var(--s-4); }'
                '.agenda-note-list li { margin: var(--s-1) 0; }'
                '.agenda-note-list button { font-size: 11px; '
                '  padding: 2px 8px; background: #222; }'
                '</style>'
            )
            return HTMLResponse(_layout(
                f"Agenda · {result.person_name}", body, "agenda",
            ))
        # Index: list recent people for picking.
        recent = people_mod.list_people(conn, order="recent", limit=30)
        rows_html = "".join(
            f'<li><a href="/agenda?id={p.id}">'
            f'{escape(_safe(p.display_name))}</a>'
            + (
                ' <span class="tag">VIP</span>'
                if p.tier == "vip" else ""
            )
            + (
                f' <span class="muted">({escape(_safe(p.role))})</span>'
                if p.role else ""
            )
            + '</li>'
            for p in recent
        )
        body = (
            f'<h1>Agenda</h1>'
            f'<p class="muted">'
            f'  Pick a person to build a 1:1 agenda. Pulls open '
            f'  follow-ups, recent emails, journal mentions, and '
            f'  shared topics into a single pre-meeting card.'
            f'</p>'
            f'<ul class="agenda-people">{rows_html}</ul>'
            '<style>'
            '.agenda-people { list-style: none; padding: 0; }'
            '.agenda-people li { padding: var(--s-1) 0; }'
            '.tag { font-size: 10px; padding: 1px 6px; '
            '  background: var(--green-soft); color: var(--green); '
            '  border-radius: var(--r); margin-left: 4px; }'
            '</style>'
        )
        return HTMLResponse(_layout("Agenda", body, "agenda"))

    # ============================ Round 22 — /today ====================

    @app.get("/today", response_class=HTMLResponse)
    def today_page(undo_done: int = 0, undo_label: str = ""):
        """Round 22 (Phase EA-UI) — the morning desk.

        EA-shaped front door. Replaces the "browse my data" landing
        with a single narrative surface: greeting, top decisions,
        today's calendar, "worth knowing", and a quiet-day fallback.

        Time-of-day mode (morning/midday/afternoon/evening/night)
        shifts the framing AND the data slice (evening pulls
        tomorrow's calendar instead of today's).

        ``?undo_done=1&undo_label=Snoozed%207%20days`` renders the
        round-22 undo toast at the top of the page after a
        round-tripped action.
        """
        from . import today as today_mod
        cfg, conn, _, _ = get_state()
        desk = today_mod.assemble_today(cfg, conn)

        # Round 22 — undo toast strip if a paired action just
        # round-tripped through here. Round 23 — share the
        # ``_undo_toast_html`` helper so the inline JS that strips
        # undo_* params on render is consistent across /today and
        # /triage.
        toast_html = _undo_toast_html(
            undo_done, undo_label, "today", 0,
        )

        if desk.is_quiet():
            body = (
                toast_html
                + f'<h1 class="today-greet">{escape(desk.greeting)}</h1>'
                + '<p class="today-quiet">'
                + f'  {escape(desk.quiet_message or "All clear.")}'
                + '</p>'
                + _today_styles()
            )
            return HTMLResponse(_layout("Today", body, "today"))

        # Decisions
        dec_html_rows = []
        for d in desk.decisions:
            dec_html_rows.append(_render_decision(d))
        dec_section = ""
        if desk.decisions:
            verb = {
                "morning":   "decisions for you this morning",
                "midday":    "things waiting on you",
                "afternoon": "still open",
                "evening":   "loose ends from today",
                "night":     "open from yesterday",
            }.get(desk.mode, "open")
            dec_section = (
                f'<section class="today-section">'
                f'  <h2 class="today-h2">'
                f'    <span class="today-num">{len(desk.decisions)}</span> '
                f'    {verb}'
                f'  </h2>'
                f'  <div class="today-decisions">'
                f'    {"".join(dec_html_rows)}'
                f'  </div>'
                f'</section>'
            )

        # Calendar
        cal_html = ""
        if desk.upcoming:
            header = (
                "Coming up today"
                if desk.mode in ("morning", "midday", "afternoon")
                else "Tomorrow"
            )
            cal_rows = []
            for ev in desk.upcoming:
                prep_link = ""
                if ev.prep_href:
                    prep_link = (
                        f' <a class="today-prep" href="{escape(ev.prep_href)}">'
                        f'prep ready</a>'
                    )
                detail = (
                    f' <span class="muted">— {escape(_safe(ev.detail))}</span>'
                    if ev.detail else ''
                )
                cal_rows.append(
                    f'<li>'
                    f'  <span class="today-when">{escape(ev.when)}</span> '
                    f'  <span>{escape(_safe(ev.title))}</span>'
                    f'  {detail}{prep_link}'
                    f'</li>'
                )
            cal_html = (
                f'<section class="today-section">'
                f'  <h2 class="today-h2">{header}</h2>'
                f'  <ul class="today-cal">{"".join(cal_rows)}</ul>'
                f'</section>'
            )

        # Worth knowing
        wk_html = ""
        if desk.worth_knowing:
            wk_rows = []
            for w in desk.worth_knowing:
                action_html = ""
                if w.action:
                    action_html = (
                        f'  <a class="today-wk-action" '
                        f'href="{escape(w.action.href)}">'
                        f'{escape(w.action.label)}</a>'
                    )
                wk_rows.append(
                    f'<li>'
                    f'  <span class="today-wk-icon">{escape(w.icon)}</span>'
                    f'  <div class="today-wk-body">'
                    f'    <div>{escape(_safe(w.title))}</div>'
                    f'    <div class="muted today-why">_{escape(_safe(w.why))}_</div>'
                    f'  </div>'
                    f'  {action_html}'
                    f'</li>'
                )
            wk_html = (
                f'<section class="today-section">'
                f'  <h2 class="today-h2">Worth knowing</h2>'
                f'  <ul class="today-wk">{"".join(wk_rows)}</ul>'
                f'</section>'
            )

        body = (
            toast_html
            + f'<h1 class="today-greet">{escape(desk.greeting)}</h1>'
            + dec_section + cal_html + wk_html
            + '<p class="today-end muted">'
            + '  _That\'s it for now._'
            + '</p>'
            + _today_styles()
        )
        return HTMLResponse(_layout("Today", body, "today"))

    # / route — soft-launch period. Existing / at line ~1765 stays
    # for power users; nav now points at /today as the front door.
    # A future round may flip / to redirect.

    # ============================ /triage (walkthrough or list) ====

    @app.get("/triage", response_class=HTMLResponse)
    def triage_page(
        show: str = "",
        undo_done: int = 0,
        undo_label: str = "",
        undo_kind: str = "",
        undo_id: int = 0,
    ):
        """Round 19 (Phase EA-6) — morning email triage queue.

        Round 22 (Phase EA-UI) — default rendering is now the
        single-email walkthrough. ``?show=all`` keeps the legacy
        list-view for power users.

        ``?undo_done=1&undo_label=...&undo_kind=triage&undo_id=N``
        renders the round-22 undo toast at the top of the page.
        """
        from . import triage_queue

        _, conn, _, _ = get_state()
        queue = triage_queue.build_queue(conn)
        # Round 22 — count of done-today for the progress chip.
        try:
            done_today = triage_queue.done_count_today(conn)
        except Exception:  # noqa: BLE001
            done_today = 0
        if not queue:
            from .ux_copy import adaptive_empty
            done_chip = (
                f'<p class="muted" style="text-align:center;">'
                f'You cleared {done_today} today. Nice work.</p>'
                if done_today else ""
            )
            toast_html = _undo_toast_html(
                undo_done, undo_label, undo_kind, undo_id,
            )
            body = (
                f'{toast_html}'
                '<h1>Triage</h1>'
                '<div class="empty card">'
                f'  <p>{adaptive_empty("triage")}</p></div>'
                f'{done_chip}'
            )
            return HTMLResponse(_layout("Triage", body, "triage"))

        # Round 22 walkthrough: focused single-email view with the
        # next-up item front and centre. The full ranked list lives
        # at ?show=all for power users.
        if show != "all":
            return _render_triage_walkthrough(
                queue=queue, done_today=done_today,
                undo_done=undo_done, undo_label=undo_label,
                undo_kind=undo_kind, undo_id=undo_id,
            )
        items_html = []
        for it in queue:
            vip_badge = (
                '<span class="tag urgent">VIP</span>'
                if it.is_vip else ""
            )
            label_class = {
                "urgent": "urgent", "follow_up": "warn",
                "review": "muted", "fyi": "muted",
            }.get(it.label, "muted")
            draft_link = (
                f' · <a href="/drafts">draft #{it.draft_id} ready</a>'
                if it.has_draft else ""
            )
            items_html.append(
                f'<div class="triage-row">'
                f'  <div class="triage-meta">'
                f'    <span class="tag {label_class}">{it.label}</span>'
                f'    {vip_badge}'
                f'    <span class="muted">'
                f'    {escape(_safe(it.from_display))}</span>'
                f'    <span class="muted">·</span>'
                f'    <span class="muted">{int(it.age_hours)}h</span>'
                f'  </div>'
                f'  <div class="triage-subject">'
                f'    <a href="/file?file_id={it.file_id}">'
                f'    {escape(_safe(it.subject)) or "(no subject)"}</a>'
                f'    {draft_link}'
                f'  </div>'
                f'</div>'
            )
        body = (
            f'<h1>Triage queue (full list)</h1>'
            f'<p class="muted">'
            f'  {len(queue)} email(s) need your decision today, '
            f'  ranked by VIP × urgency × age. '
            f'  <a href="/triage">walk through one at a time →</a>'
            f'</p>'
            + "".join(items_html)
            + '<style>'
            '.triage-row { padding: var(--s-2); '
            '  border-bottom: 1px dashed var(--border); }'
            '.triage-meta { font-size: 11px; }'
            '.triage-subject { margin-top: 2px; }'
            '.tag { font-size: 10px; padding: 1px 6px; '
            '  border-radius: var(--r); margin-right: 4px; }'
            '.tag.urgent { background: rgba(255,77,77,0.10); '
            '  color: var(--red); }'
            '.tag.warn { background: rgba(255,183,0,0.10); '
            '  color: var(--amber); }'
            '</style>'
        )
        return HTMLResponse(_layout("Triage", body, "triage"))

    @app.get("/gifts", response_class=HTMLResponse)
    def gifts_page():
        """Round 19 (Phase EA-7) — birthday gift ideas."""
        from . import gift_ideas

        cfg, conn, _, _ = get_state()
        try:
            entries = gift_ideas.list_for_upcoming_birthdays(
                conn, cfg, auto_generate=False,
            )
        except Exception as e:  # noqa: BLE001
            entries = []
            log.warning("gifts page: %s", e)
        if not entries:
            body = (
                '<h1>Gift ideas</h1>'
                '<div class="empty card">'
                '  <p>No birthdays in the next 14 days.</p></div>'
            )
            return HTMLResponse(_layout("Gifts", body, "gifts"))
        rows_html = []
        for p, days_until, ideas in entries:
            ideas_html = ""
            if ideas and ideas.ideas:
                ideas_html = "<ul class='gift-ideas'>" + "".join(
                    f"<li><strong>{escape(_safe(i.title))}</strong>"
                    + (
                        f" <span class='muted'>{escape(i.price_range)}</span>"
                        if i.price_range else ""
                    )
                    + f"<br><span>{escape(_safe(i.description))}</span>"
                    + (
                        f"<br><em class='muted'>{escape(_safe(i.why))}</em>"
                        if i.why else ""
                    )
                    + "</li>"
                    for i in ideas.ideas
                ) + "</ul>"
            else:
                ideas_html = (
                    f'<form method="post" '
                    f'action="/gifts/{p.id}/generate" '
                    f'style="display:inline;">'
                    f'<button type="submit">Generate ideas</button>'
                    f'</form>'
                )
            rows_html.append(
                f'<div class="card gift-card">'
                f'<h3>{escape(_safe(p.display_name))}'
                f'  <span class="muted">'
                f'  · birthday in {days_until}d</span></h3>'
                f'{ideas_html}'
                f'</div>'
            )
        body = (
            '<h1>Gift ideas</h1>'
            '<p class="muted">Birthdays in the next 14 days. '
            'Click "generate" to get 3 AI-suggested gifts.</p>'
            + "".join(rows_html)
            + '<style>'
            '.gift-card { margin-bottom: var(--s-3); }'
            '.gift-ideas { padding-left: var(--s-4); }'
            '.gift-ideas li { margin-bottom: var(--s-2); }'
            '</style>'
        )
        return HTMLResponse(_layout("Gifts", body, "gifts"))

    @app.post("/gifts/{person_id:int}/generate")
    def gifts_generate(person_id: int, request: Request):
        from . import gift_ideas
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        cfg, conn, _, _ = get_state()
        try:
            gift_ideas.generate_for_person(
                conn, cfg, person_id, overwrite=True,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("gifts/generate failed: %s", e)
        return RedirectResponse(url="/gifts", status_code=303)

    @app.get("/threads", response_class=HTMLResponse)
    def threads_page():
        """Round 19 (Phase EA-8) — standing-thread tracker."""
        from . import standing_threads

        _, conn, _, _ = get_state()
        rows = standing_threads.list_threads(conn, limit=30)
        if not rows:
            body = (
                '<h1>Standing threads</h1>'
                '<div class="empty card">'
                '  <p>No long-running threads detected yet. '
                '  The detector runs hourly and needs ≥5 emails over '
                '  ≥14 days to flag a thread.</p></div>'
            )
            return HTMLResponse(_layout("Threads", body, "threads"))
        rows_html = []
        for t in rows:
            from datetime import date as _date
            first = _date.fromtimestamp(t.first_message_at).isoformat()
            last = _date.fromtimestamp(t.last_message_at).isoformat()
            summary_html = (
                f'<p>{escape(_safe(t.summary_md))}</p>'
                if t.summary_md else
                f'<form method="post" '
                f'action="/threads/{t.id}/summarize" '
                f'style="display:inline;">'
                f'<button type="submit">Summarize</button>'
                f'</form>'
            )
            rows_html.append(
                f'<div class="card">'
                f'<h3>{escape(_safe(t.topic))}</h3>'
                f'<p class="muted">'
                f'{t.n_messages} message(s) · {first} → {last}'
                f'</p>'
                f'{summary_html}'
                f'</div>'
            )
        body = (
            '<h1>Standing threads</h1>'
            '<p class="muted">Long-running conversations across '
            'people / topics. Click summarize for a Sonnet-generated '
            'recap of decisions + open questions.</p>'
            + "".join(rows_html)
        )
        return HTMLResponse(_layout("Threads", body, "threads"))

    @app.post("/threads/{thread_id:int}/summarize")
    def threads_summarize(thread_id: int, request: Request):
        from . import standing_threads
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        cfg, conn, _, _ = get_state()
        try:
            standing_threads.summarize(
                conn, cfg, thread_id, force=True,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("threads/summarize failed: %s", e)
        return RedirectResponse(url="/threads", status_code=303)

    @app.get("/eod", response_class=HTMLResponse)
    def eod_page():
        """Round 19 (Phase EA-9) — end-of-day wrap-up."""
        from . import eod_wrapup

        _, conn, _, _ = get_state()
        w = eod_wrapup.build_wrapup(conn)
        md = eod_wrapup.render_markdown(w)
        body = (
            f'<h1>End of day</h1>'
            f'<div class="briefing-body">'
            f'{_markdown_to_html_block(md)}'
            f'</div>'
        )
        return HTMLResponse(_layout("EOD", body, "eod"))

    # ============================ Round 20 — EA expansions ==============

    # --- Followups ops (snooze, edit, nudge, bulk dismiss) ------------

    @app.post("/followups/{followup_id:int}/snooze")
    def followup_snooze(
        followup_id: int, request: Request,
        days: int = Form(7),
    ):
        from . import followups_ops
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        _, conn, _, _ = get_state()
        try:
            followups_ops.snooze(conn, followup_id, days=days)
        except ValueError as e:
            return HTMLResponse(str(e), status_code=400)
        return RedirectResponse(url="/followups", status_code=303)

    @app.post("/followups/{followup_id:int}/unsnooze")
    def followup_unsnooze(followup_id: int, request: Request):
        from . import followups_ops
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        _, conn, _, _ = get_state()
        followups_ops.unsnooze(conn, followup_id)
        return RedirectResponse(url="/followups", status_code=303)

    @app.post("/followups/{followup_id:int}/edit")
    def followup_edit(
        followup_id: int, request: Request,
        topic: str = Form(""),
        description: str = Form(""),
        due_iso: str = Form(""),
    ):
        from . import followups_ops
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        _, conn, _, _ = get_state()
        due_at: float | None = -1.0
        if due_iso:
            try:
                from datetime import date as _date
                from datetime import datetime as _dt
                d = _date.fromisoformat(due_iso)
                due_at = _dt(d.year, d.month, d.day).timestamp()
            except (ValueError, TypeError):
                due_at = -1.0
        followups_ops.edit(
            conn, followup_id,
            topic=(topic or None) if topic else None,
            description=(description or None) if description else None,
            due_at=due_at,
        )
        return RedirectResponse(url="/followups", status_code=303)

    @app.post("/followups/{followup_id:int}/nudge")
    def followup_nudge(followup_id: int, request: Request):
        """Generate a nudge-email draft + redirect to a preview page
        (rendered inline as a separate route below)."""
        from . import followups_ops
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        cfg, conn, _, _ = get_state()
        try:
            draft = followups_ops.draft_nudge(conn, cfg, followup_id)
        except Exception as e:  # noqa: BLE001
            log.warning("nudge draft failed: %s", e)
            draft = None
        if draft is None:
            return HTMLResponse(
                "<h1>Could not draft nudge</h1>"
                "<p>No API key, budget exceeded, or LLM error. "
                "Try again later.</p><p><a href='/followups'>← Back</a></p>",
                status_code=200,
            )
        body = (
            f'<h1>Nudge draft</h1>'
            f'<p class="muted">Copy/paste into your email client. '
            f'No data sent automatically.</p>'
            f'<div class="card"><h2>Subject</h2>'
            f'<pre>{escape(_safe(draft.get("subject", "")))}</pre>'
            f'<h2>Body</h2>'
            f'<pre style="white-space:pre-wrap;">'
            f'{escape(_safe(draft.get("body", "")))}</pre></div>'
            f'<p><a class="link-btn" href="/followups">'
            f'← Back to follow-ups</a></p>'
        )
        return HTMLResponse(_layout("Nudge draft", body, "followups"))

    @app.post("/followups/bulk_dismiss")
    def followup_bulk_dismiss(
        request: Request,
        person_id: int = Form(0),
        overdue_only: int = Form(0),
        direction: str = Form(""),
    ):
        from . import followups_ops
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        _, conn, _, _ = get_state()
        followups_ops.bulk_dismiss(
            conn,
            person_id=person_id or None,
            overdue_only=bool(overdue_only),
            direction=direction or None,
        )
        return RedirectResponse(url="/followups", status_code=303)

    # --- Capture detail page + edit + recap-sent flag ------------------

    @app.get("/capture", response_class=HTMLResponse)
    def capture_detail(file_id: int = 0):
        """Per-meeting capture detail view — decisions / actions /
        questions / recap draft + edit links."""
        from . import meeting_capture
        _, conn, _, _ = get_state()
        if not file_id:
            captures = meeting_capture.list_recent(conn, limit=30)
            if not captures:
                body = (
                    '<h1>Meeting captures</h1>'
                    '<div class="empty card"><p>No captures yet. '
                    'Index a meeting transcript and the daemon will '
                    'capture it within an hour.</p></div>'
                )
                return HTMLResponse(_layout("Captures", body, "capture"))
            rows_html = []
            for cap in captures:
                when = time.strftime(
                    "%Y-%m-%d %H:%M",
                    time.localtime(cap.captured_at),
                )
                rows_html.append(
                    f'<li><a href="/capture?file_id={cap.file_id}">'
                    f'{escape(_safe(cap.title))}</a> '
                    f'<span class="muted">· {when} · '
                    f'{len(cap.decisions)} dec · '
                    f'{len(cap.actions)} act</span></li>'
                )
            body = (
                '<h1>Meeting captures</h1>'
                f'<ul>{"".join(rows_html)}</ul>'
            )
            return HTMLResponse(_layout("Captures", body, "capture"))
        cap = meeting_capture.get_capture(conn, file_id)
        if cap is None:
            return HTMLResponse(
                "<h1>Capture not found</h1>", status_code=404,
            )
        md = meeting_capture.render_capture_markdown(cap)
        sent_str = ""
        if cap.recap_sent_at:
            sent_str = (
                f'<p class="muted">Recap sent '
                f'{time.strftime("%Y-%m-%d %H:%M", time.localtime(cap.recap_sent_at))}.</p>'
            )
        else:
            sent_str = (
                f'<form method="post" action="/capture/{cap.file_id}/recap_sent" '
                f'style="display:inline;">'
                f'<button type="submit">Mark recap as sent</button></form>'
            )
        edited_badge = (
            ' <span class="tag">edited</span>'
            if cap.user_edited else ""
        )
        body = (
            f'<h1>Meeting capture{edited_badge}</h1>'
            f'<p class="muted">file #{cap.file_id} · '
            f'captured via {escape(cap.model)} · '
            f'{time.strftime("%Y-%m-%d %H:%M", time.localtime(cap.captured_at))}</p>'
            f'<div class="briefing-body">'
            f'{_markdown_to_html_block(md)}'
            f'</div>'
            f'{sent_str}'
            '<style>.tag { font-size: 10px; padding: 1px 6px; '
            ' background: var(--amber); color: #000; '
            ' border-radius: var(--r); margin-left: 4px; }</style>'
        )
        return HTMLResponse(_layout(
            f"Capture · {cap.title[:40]}", body, "capture",
        ))

    @app.post("/capture/{file_id:int}/recap_sent")
    def capture_mark_recap_sent(file_id: int, request: Request):
        from . import meeting_capture
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        _, conn, _, _ = get_state()
        meeting_capture.mark_recap_sent(conn, file_id)
        return RedirectResponse(
            url=f"/capture?file_id={file_id}", status_code=303,
        )

    # --- Per-person agenda notes + tier UI ----------------------------

    @app.post("/agenda/{person_id:int}/note")
    def agenda_add_note(
        person_id: int, request: Request,
        text: str = Form(""),
    ):
        from . import agenda
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        _, conn, _, _ = get_state()
        if text.strip():
            agenda.add_note(conn, person_id, text.strip())
        return RedirectResponse(
            url=f"/agenda?id={person_id}", status_code=303,
        )

    @app.post("/agenda/note/{note_id:int}/discussed")
    def agenda_note_discussed(note_id: int, request: Request):
        from . import agenda
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        _, conn, _, _ = get_state()
        agenda.mark_discussed(conn, note_id)
        return RedirectResponse(url="/agenda", status_code=303)

    @app.post("/person/{person_id:int}/tier")
    def person_set_tier(
        person_id: int, request: Request,
        tier: str = Form("regular"),
        cadence_days: int = Form(0),
    ):
        from . import people as people_mod
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        _, conn, _, _ = get_state()
        try:
            people_mod.set_field(
                conn, person_id, tier=tier, cadence_days=cadence_days,
            )
        except ValueError as e:
            return HTMLResponse(str(e), status_code=400)
        return RedirectResponse(
            url=f"/person?id={person_id}", status_code=303,
        )

    # --- Triage queue actions -----------------------------------------

    def _safe_next(next_param: str) -> str:
        """Round 23 fix (audit-found gap M7) — ``next=`` redirect
        param so an action POSTed from /today returns to /today
        instead of dumping the user into the triage walkthrough.

        Whitelist the allowed targets to avoid open-redirect: only
        same-origin paths starting with ``/today`` or ``/triage``."""
        if not next_param:
            return "/triage"
        candidate = str(next_param).strip()
        if candidate in ("/today", "/triage"):
            return candidate
        if candidate.startswith("/today?") or candidate.startswith("/triage?"):
            return candidate
        return "/triage"

    @app.post("/triage/{file_id:int}/done")
    def triage_done(
        file_id: int, request: Request, next: str = Form(""),
    ):
        from urllib.parse import urlencode

        from . import triage_queue
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        _, conn, _, _ = get_state()
        triage_queue.mark_done(conn, file_id)
        target = _safe_next(next)
        sep = "&" if "?" in target else "?"
        qs = urlencode({
            "undo_done": "1",
            "undo_label": "Marked done",
            "undo_kind": "triage",
            "undo_id": str(file_id),
        })
        return RedirectResponse(
            url=f"{target}{sep}{qs}", status_code=303,
        )

    @app.post("/triage/{file_id:int}/skip")
    def triage_skip(
        file_id: int, request: Request, next: str = Form(""),
    ):
        from urllib.parse import urlencode

        from . import triage_queue
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        _, conn, _, _ = get_state()
        triage_queue.mark_skipped(conn, file_id)
        target = _safe_next(next)
        sep = "&" if "?" in target else "?"
        qs = urlencode({
            "undo_done": "1",
            "undo_label": "Skipped",
            "undo_kind": "triage",
            "undo_id": str(file_id),
        })
        return RedirectResponse(
            url=f"{target}{sep}{qs}", status_code=303,
        )

    @app.post("/triage/{file_id:int}/snooze")
    def triage_snooze(
        file_id: int, request: Request,
        hours: int = Form(24), next: str = Form(""),
    ):
        from urllib.parse import urlencode

        from . import triage_queue
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        _, conn, _, _ = get_state()
        try:
            triage_queue.snooze(conn, file_id, hours=hours)
        except ValueError as e:
            return HTMLResponse(str(e), status_code=400)
        target = _safe_next(next)
        sep = "&" if "?" in target else "?"
        qs = urlencode({
            "undo_done": "1",
            "undo_label": f"Snoozed {hours}h",
            "undo_kind": "triage",
            "undo_id": str(file_id),
        })
        return RedirectResponse(
            url=f"{target}{sep}{qs}", status_code=303,
        )

    # Round 22 — undo endpoint. Reverses a recent triage decision
    # by clearing the row in triage_state. CSRF-guarded; safe-by-
    # default since it's the user's own action being walked back.
    #
    # Round 23 fix (audit-found gap H3 + M6) — narrow the except
    # to the expected sqlite errors so a real bug doesn't silently
    # look successful. Redirect with paired undo-toast so the user
    # sees confirmation rather than a silent reload.
    @app.post("/triage/{file_id:int}/undo")
    def triage_undo(file_id: int, request: Request):
        import sqlite3 as _sq3
        from urllib.parse import urlencode

        from . import triage_queue
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        _, conn, _, _ = get_state()
        try:
            triage_queue._ensure_schema(conn)
            with triage_queue._WRITE_LOCK:
                cur = conn.execute(
                    "DELETE FROM triage_state WHERE file_id = ?",
                    (file_id,),
                )
                conn.commit()
            n_cleared = cur.rowcount
        except _sq3.OperationalError as e:
            log.warning("triage_undo: DB error: %s", e)
            return HTMLResponse(
                f"Database error: {e}", status_code=500,
            )
        if n_cleared == 0:
            return RedirectResponse(url="/triage", status_code=303)
        qs = urlencode({
            "undo_done": "1",
            "undo_label": "Undone",
        })
        return RedirectResponse(
            url=f"/triage?{qs}", status_code=303,
        )

    # --- Scheduling page ---------------------------------------------

    @app.get("/scheduling", response_class=HTMLResponse)
    def scheduling_page():
        """Round 20 — find-time UI + recent proposals + per-person prefs."""
        from . import scheduling as sched_mod
        _, conn, _, _ = get_state()
        proposals = sched_mod.list_recent_proposals(conn, limit=20)
        prop_html = ""
        if proposals:
            rows = []
            for p in proposals:
                when = time.strftime(
                    "%Y-%m-%d %H:%M",
                    time.localtime(p["proposed_at"]),
                )
                outcome_class = {
                    "scheduled": "good",
                    "declined": "bad",
                    "expired": "muted",
                    "pending": "warn",
                }.get(p["outcome"], "muted")
                rows.append(
                    f'<li><span class="tag {outcome_class}">'
                    f'{p["outcome"]}</span> '
                    f'{escape(_safe(p["person_name"]))} '
                    f'<span class="muted">· {when}</span></li>'
                )
            prop_html = (
                f'<h2>Recent proposals</h2><ul>{"".join(rows)}</ul>'
            )
        body = (
            '<h1>Scheduling</h1>'
            '<div class="card">'
            '<p class="muted">'
            '  This page logs scheduling proposals. To find time for '
            '  a meeting, use the chat ("find me 30 min with Sarah '
            '  next week") or the <code>find_open_time_slots</code> '
            '  MCP tool with your calendar busy events.'
            '</p>'
            '</div>'
            f'{prop_html}'
            '<style>'
            '.tag { font-size: 10px; padding: 1px 6px; '
            '  border-radius: var(--r); margin-right: 4px; }'
            '.tag.good { background: var(--green-soft); '
            '  color: var(--green); }'
            '.tag.bad { background: rgba(255,77,77,0.10); '
            '  color: var(--red); }'
            '.tag.warn { background: rgba(255,183,0,0.10); '
            '  color: var(--amber); }'
            '</style>'
        )
        return HTMLResponse(_layout("Scheduling", body, "scheduling"))

    # --- Phase 47: tasks view -----------------------------------------

    @app.get("/tasks", response_class=HTMLResponse)
    def tasks_page():
        """Open + recently-done tasks. The list re-materialises from
        recent transcripts on every load (idempotent — safe)."""
        from . import tasks as tasks_mod

        cfg, conn, _, _ = get_state()
        try:
            tasks_mod.materialize_from_transcripts(conn)
        except Exception:  # noqa: BLE001
            pass
        open_rows = tasks_mod.list_open(conn, limit=200)
        done_rows = tasks_mod.list_recent_done(conn, limit=20)
        now = time.time()

        # Round 11 — pre-resolve recipient + due hint for tasks the
        # round-9-C structured extractor populated. Single bulk
        # query for any open task with a recipient, instead of N+1.
        person_names: dict[int, str] = {}
        recipient_ids = {t.recipient_person_id for t in open_rows
                         if t.recipient_person_id}
        if recipient_ids:
            placeholders = ",".join("?" * len(recipient_ids))
            for r in conn.execute(
                f"SELECT id, display_name FROM people "
                f"WHERE id IN ({placeholders})",
                list(recipient_ids),
            ).fetchall():
                person_names[int(r["id"])] = r["display_name"]

        def _row(t, *, mark_done: bool) -> str:
            age_d = max(0, int((now - t.created_at) // 86400))
            age_html = (
                f'<span class="muted">{age_d}d</span>' if age_d else ""
            )
            src = (
                f'<span class="muted"> · from '
                f'<a href="/file?path={urllib.parse.quote_plus(t.source_path)}">'
                f'{escape(t.source_title)}</a></span>'
                if t.source_path != "manual" else ""
            )
            # Round 11 — surface recipient + due hint when known.
            extras = []
            if t.recipient_person_id and t.recipient_person_id in person_names:
                name = person_names[t.recipient_person_id]
                extras.append(
                    f'→ <a href="/person?id={t.recipient_person_id}">'
                    f'{escape(name)}</a>'
                )
            if t.due_hint:
                extras.append(f'due {escape(t.due_hint)}')
            extras_html = (
                f' <span class="muted" style="font-size:11px;">'
                f'· {" · ".join(extras)}</span>'
                if extras else ""
            )
            done_btn = (
                f'<form method="post" action="/tasks/{t.id}/done" '
                f'style="display:inline">'
                f'<button class="link-btn" type="submit">✓ done</button>'
                f'</form>'
                if mark_done else ""
            )
            return (
                f'<div class="stat">'
                f'<span><code>#{t.id}</code> {escape(t.text)}'
                f'{extras_html}{src}</span>'
                f'<span class="v">{age_html} {done_btn}</span>'
                f'</div>'
            )

        open_html = (
            "".join(_row(t, mark_done=True) for t in open_rows)
            or '<div class="muted">Inbox zero. Nothing to do.</div>'
        )
        done_html = (
            "".join(
                f'<div class="stat">'
                f'<span><code>#{t.id}</code> {escape(t.text)}</span>'
                f'<span class="v muted">'
                f'{time.strftime("%a %H:%M", time.localtime(t.completed_at))}</span>'
                f'</div>'
                for t in done_rows
            )
            or '<div class="muted">(no completed tasks yet)</div>'
        )

        body = f"""
<h1>Tasks</h1>
<div class="grid">
  <div class="card">
    <h2>Open ({len(open_rows)})</h2>
    {open_html}
    <form method="post" action="/tasks/add"
          style="margin-top:16px;display:flex;gap:8px;">
      <input type="text" name="text" placeholder="Add a task…"
             style="flex:1;" required>
      <button type="submit">Add</button>
    </form>
  </div>
  <div class="card">
    <h2>Recently done</h2>
    {done_html}
  </div>
</div>"""
        return HTMLResponse(_layout("Tasks", body, "tasks"))

    @app.post("/tasks/add")
    def tasks_add(request: Request, text: str = Form(...)):
        from fastapi.responses import RedirectResponse

        from . import tasks as tasks_mod
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        cfg, conn, _, _ = get_state()
        tasks_mod.add_manual(conn, text)
        return RedirectResponse(url="/tasks", status_code=303)

    @app.post("/tasks/{task_id:int}/done")
    def tasks_mark_done(task_id: int, request: Request):
        from fastapi.responses import RedirectResponse

        from . import tasks as tasks_mod
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        cfg, conn, _, _ = get_state()
        tasks_mod.mark_done(conn, task_id)
        return RedirectResponse(url="/tasks", status_code=303)

    # --- Phase 56: health view ----------------------------------------

    @app.get("/health", response_class=HTMLResponse)
    def health_view():
        """Oura ring vitals + per-metric trend cards."""
        from . import health as health_mod

        cfg, conn, _, _ = get_state()
        metrics = health_mod.list_metrics(conn)
        if not metrics:
            body = (
                "<h1>Health</h1>"
                "<div class='card'><p class='muted'>"
                "No Oura data yet. Run "
                "<code>secondbrain auth oura</code> then "
                "<code>secondbrain sync oura</code> to backfill."
                "</p></div>"
            )
            return HTMLResponse(_layout("Health", body, "health"))

        # Headline cards: the three big scores get prominent display.
        headline_metrics = ("sleep_score", "readiness_score", "activity_score")
        cards: list[str] = []
        for m in headline_metrics:
            s = health_mod.summarise(conn, m, days=14)
            if s.n == 0:
                continue
            points = health_mod.recent(conn, m, days=14)
            cards.append(_render_health_card(m, s, points))

        # Secondary metrics — render as a compact table.
        rest = [m for m in metrics if m not in headline_metrics]
        rest_html = ""
        if rest:
            rows_html = []
            for m in rest:
                s = health_mod.summarise(conn, m, days=14)
                if s.n == 0:
                    continue
                latest_v = (
                    f"{s.latest.value:g}" if s.latest else "—"
                )
                avg_v = (
                    f"{s.average:.1f}" if s.average is not None else "—"
                )
                rows_html.append(
                    f"<tr><td>{escape(m)}</td><td>{latest_v}</td>"
                    f"<td>{avg_v}</td>"
                    f"<td class='muted'>{s.n}d</td></tr>"
                )
            if rows_html:
                rest_html = (
                    "<div class='card'><h2>Other metrics</h2>"
                    "<table><thead><tr><th>metric</th><th>latest</th>"
                    "<th>14d avg</th><th>n</th></tr></thead>"
                    "<tbody>" + "".join(rows_html) + "</tbody></table></div>"
                )

        body = (
            "<h1>Health</h1>"
            "<div class='grid'>" + "".join(cards) + "</div>"
            + rest_html
        )
        return HTMLResponse(_layout("Health", body, "health"))

    # --- Phase 65: people page -----------------------------------------

    @app.get("/people", response_class=HTMLResponse)
    def people_view():
        from . import connections
        from . import people as people_mod

        cfg, conn, _, _ = get_read_state()
        rows = people_mod.list_people(conn, order="recent", limit=100)
        if not rows:
            body = (
                "<h1>People</h1>"
                "<div class='card'><p class='muted'>"
                "No people yet. Run "
                "<code>secondbrain people backfill</code> to seed from "
                "PERSON entities the spaCy NER has already extracted."
                "</p></div>"
            )
            return HTMLResponse(_layout("People", body, "people"))
        now = time.time()

        # Round 9-B — stale connections at the top of the page.
        stale = connections.find_stale_connections(conn, limit=8)
        stale_html = ""
        if stale:
            stale_items = "".join(
                f'<div class="stat">'
                f'<span><a href="/person?id={c.person_id}">'
                f'{escape(c.name)}</a> '
                f'<span class="muted">{escape(c.email or "")}</span></span>'
                f'<span class="v muted">'
                f'{c.mention_count} mentions · '
                f'{c.months_since_seen}mo since seen'
                f'</span></div>'
                for c in stale
            )
            stale_html = (
                "<div class='card' style='margin-bottom:16px;"
                "border-color:var(--amber);'>"
                "<h2>Worth reaching back out</h2>"
                f"{stale_items}</div>"
            )

        items_html = "".join(
            f'<div class="stat">'
            f'<span><a href="/person?id={p.id}">{escape(p.display_name)}</a> '
            f'<span class="muted">{escape(p.email or "")}</span></span>'
            f'<span class="v muted">'
            f'{p.mention_count} mentions · '
            f'{max(0, int((now - p.last_seen_at) // 86400))}d'
            f'</span></div>'
            for p in rows
        )
        body = (
            f"<h1>People</h1>{stale_html}<div class='card'>"
            f"<h2>Recent ({len(rows)})</h2>{items_html}</div>"
        )
        return HTMLResponse(_layout("People", body, "people"))

    @app.get("/person", response_class=HTMLResponse)
    def person_detail(id: int):  # noqa: A002
        from . import people as people_mod

        cfg, conn, _, _ = get_read_state()
        profile = people_mod.profile_for(conn, id)
        if profile is None:
            return HTMLResponse(_layout(
                f"Person #{id}",
                f"<h1>Person #{id} not found</h1>", "people",
            ))
        meta_lines = [
            f"<div class='stat'><span class='muted'>Mentions</span>"
            f"<span class='v'>{profile.person.mention_count}</span></div>",
            f"<div class='stat'><span class='muted'>First seen</span>"
            f"<span class='v'>{profile.days_since_first_seen}d ago</span></div>",
            f"<div class='stat'><span class='muted'>Last seen</span>"
            f"<span class='v'>{profile.days_since_seen}d ago</span></div>",
        ]
        if profile.person.email:
            meta_lines.append(
                f"<div class='stat'><span class='muted'>Email</span>"
                f"<span class='v'>{escape(profile.person.email)}</span></div>",
            )
        if profile.person.role:
            meta_lines.append(
                f"<div class='stat'><span class='muted'>Role</span>"
                f"<span class='v'>{escape(profile.person.role)}</span></div>",
            )
        if profile.person.company:
            meta_lines.append(
                f"<div class='stat'><span class='muted'>Company</span>"
                f"<span class='v'>{escape(profile.person.company)}</span></div>",
            )
        mentions_html = ""
        if profile.recent_mentions:
            items = []
            for m in profile.recent_mentions:
                when = time.strftime(
                    "%Y-%m-%d", time.localtime(m.mtime),
                )
                items.append(
                    f'<article class="result">'
                    f'<h3><a href="/file?path={urllib.parse.quote_plus(m.file_path)}">'
                    f'{escape(m.file_path)}</a></h3>'
                    f'<div class="meta">{when}</div>'
                    f'<div class="snippet">{escape(m.chunk_text_preview)}</div>'
                    f'</article>',
                )
            mentions_html = (
                f"<div class='card'><h2>Recent mentions "
                f"({len(profile.recent_mentions)})</h2>"
                + "".join(items) + "</div>"
            )
        aliases_html = (
            f"<p class='muted'>Aliases: {escape(', '.join(profile.aliases))}</p>"
            if profile.aliases else ""
        )
        # Round 10 (#10) — inline edit form for the profile fields
        # most likely to need correction (email / role / company /
        # birthday / notes). All optional; submitting partially-filled
        # form only updates non-empty fields.
        edit_html = (
            f"<details class='card' style='margin-top:12px;'>"
            f"<summary><strong>Edit profile</strong></summary>"
            f"<form method='post' action='/person/{id}/edit' "
            f"style='margin-top:12px;'>"
            f"  <label>Email <input name='email' type='email' "
            f"   value='{escape(profile.person.email)}' "
            f"   style='width:100%;'></label><br><br>"
            f"  <label>Role <input name='role' "
            f"   value='{escape(profile.person.role)}' "
            f"   style='width:100%;'></label><br><br>"
            f"  <label>Company <input name='company' "
            f"   value='{escape(profile.person.company)}' "
            f"   style='width:100%;'></label><br><br>"
            f"  <label>Birthday <input name='birthday' "
            f"   placeholder='YYYY-MM-DD or MM-DD' "
            f"   value='{escape(profile.person.birthday)}' "
            f"   style='width:100%;'></label><br><br>"
            f"  <label>Notes <textarea name='notes' rows='3' "
            f"   style='width:100%;'>{escape(profile.person.notes)}"
            f"</textarea></label><br><br>"
            f"  <button type='submit'>Save</button>"
            f"</form></details>"
        )
        body = (
            f"<h1>{escape(profile.person.display_name)}</h1>"
            f"{aliases_html}"
            f"<div class='card'><h2>Profile</h2>"
            + "".join(meta_lines) + "</div>"
            + (f"<div class='card'><h2>Notes</h2><p>{escape(profile.person.notes)}</p></div>"
               if profile.person.notes else "")
            + edit_html
            + mentions_html
        )
        return HTMLResponse(_layout(
            profile.person.display_name, body, "people",
        ))

    @app.post("/person/{person_id:int}/edit")
    def person_edit(
        request: Request,
        person_id: int,
        email: str = Form(""),
        role: str = Form(""),
        company: str = Form(""),
        birthday: str = Form(""),
        notes: str = Form(""),
    ):
        """Round 11 fix (audit-found bug) — POST handler for the
        inline person-edit form.

        ``set_field`` semantics: ``None`` preserves the existing
        value, empty string clears it. The previous version of this
        handler always passed the form value (defaulting to ``""``
        from ``Form("")``), which meant a programmatic / partial POST
        would wipe every field the caller didn't include.

        Fix: pass each value to ``set_field`` ONLY when it differs
        from the currently-stored value. That way:
          - User edits one field on the form → other fields re-submit
            their existing value → no-ops, no clearing.
          - User intentionally clears a field on the form → submitted
            ``""`` differs from prior value → cleared.
          - User leaves a field blank that was already blank → no-op.
          - Programmatic POST missing a field → defaults to ``""`` →
            still no-op when the stored value was already empty.
        """
        from fastapi.responses import RedirectResponse

        from . import people as people_mod

        # Round 13 fix (audit-found bug) — same-origin guard. Every
        # other state-mutating POST in dashboard.py has this; the
        # round-10 #10 person_edit handler shipped without one. The
        # browser-extension API allow-list (CORS) plus this missing
        # guard left a CSRF hole reachable from a compromised
        # extension page.
        # Round 14 — uses the centralised _is_same_origin_request helper.
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        cfg, conn, _, _ = get_state()
        existing = people_mod.get_person(conn, person_id)
        if existing is None:
            return RedirectResponse(url="/people", status_code=303)
        # Build kwargs only for fields that actually changed.
        updates: dict[str, str] = {}
        for field_name, submitted in (
            ("email", email),
            ("role", role),
            ("company", company),
            ("birthday", birthday),
            ("notes", notes),
        ):
            current = getattr(existing, field_name, "") or ""
            if (submitted or "") != current:
                updates[field_name] = submitted
        if updates:
            people_mod.set_field(conn, person_id, **updates)
        return RedirectResponse(
            url=f"/person?id={person_id}", status_code=303,
        )

    # --- Phase 79: habits page -----------------------------------------

    @app.get("/habits", response_class=HTMLResponse)
    def habits_view():
        from . import personal

        cfg, conn, _, _ = get_state()
        habits = personal.list_habits(conn)
        if not habits:
            body = (
                "<h1>Habits</h1><div class='card'><p class='muted'>"
                "No habits yet. Run "
                "<code>secondbrain habits add &lt;name&gt;</code> "
                "to start tracking.</p></div>"
            )
            return HTMLResponse(_layout("Habits", body, "habits"))
        items = []
        for h in habits:
            s = personal.habit_status(conn, h.id)
            adh = (
                f"{s.checkins_last_30d}/{s.expected_30d}"
                if s.expected_30d else f"{s.checkins_last_30d}"
            )
            marker = (
                "🏔" if s.current_streak_days >= 100
                else "🔥" if s.current_streak_days >= 30
                else "✨" if s.current_streak_days >= 7 else ""
            )
            items.append(
                f'<div class="stat">'
                f'<span>{marker} <strong>{escape(h.name)}</strong> '
                f'<span class="muted">({h.cadence})</span></span>'
                f'<span class="v">{s.current_streak_days}d streak '
                f'<span class="muted">· {adh}/30d</span> '
                f'<form method="post" action="/habits/{h.id}/checkin" '
                f'style="display:inline">'
                f'<button class="link-btn" type="submit">✓ today</button>'
                f'</form></span>'
                f'</div>',
            )
        goals = personal.list_goals(conn)
        goals_html = ""
        if goals:
            g_items = []
            for g in goals:
                gs = personal.goal_status(conn, g.id)
                target = (
                    f"/{g.target_per_week}" if g.target_per_week else ""
                )
                track = "good" if gs.on_track else "warn"
                g_items.append(
                    f'<div class="stat">'
                    f'<span><strong>{escape(g.name)}</strong></span>'
                    f'<span class="v {track}">{gs.progress_this_week}{target} '
                    f'<span class="muted">this wk</span></span>'
                    f'</div>',
                )
            goals_html = (
                "<div class='card'><h2>Goals</h2>"
                + "".join(g_items) + "</div>"
            )
        body = (
            "<h1>Habits</h1><div class='card'><h2>Habits</h2>"
            + "".join(items) + "</div>"
            + goals_html
        )
        return HTMLResponse(_layout("Habits", body, "habits"))

    @app.post("/habits/{habit_id:int}/checkin")
    def habits_checkin_post(habit_id: int, request: Request):
        from fastapi.responses import RedirectResponse

        from . import personal
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        cfg, conn, _, _ = get_state()
        personal.checkin(conn, habit_id)
        return RedirectResponse(url="/habits", status_code=303)

    # --- Phase 80: journal page ----------------------------------------

    @app.get("/journal", response_class=HTMLResponse)
    def journal_view():
        from . import personal

        cfg, conn, _, _ = get_state()
        entries = personal.recent_journal(conn, days=30)
        items = []
        for e in entries:
            mood_str = "·" * (e.mood or 0) if e.mood else "—"
            items.append(
                f'<div class="card" style="margin-bottom:8px;">'
                f'<div class="stat"><strong>{e.date}</strong>'
                f'<span class="v muted">{mood_str} ({e.mood or "—"}/5)</span></div>'
                f'<p>{escape(e.text or "(no text)")}</p>'
                f'</div>',
            )
        # Today's entry form.
        today_entry = personal.get_journal(conn)
        today_text = today_entry.text if today_entry else ""
        today_mood = today_entry.mood if today_entry and today_entry.mood else 3
        body = (
            f"<h1>Journal</h1>"
            f"<div class='card'><h2>Today</h2>"
            f"<form method='post' action='/journal/add'>"
            f"  <label>Mood (1-5): "
            f"  <input type='number' name='mood' min='1' max='5' "
            f"   value='{today_mood}' style='width:60px;'></label>"
            f"  <textarea name='text' rows='3' style='width:100%;margin-top:8px;'>"
            f"{escape(today_text)}</textarea>"
            f"  <button type='submit' style='margin-top:8px;'>Save</button>"
            f"</form></div>"
            + (f"<h2 style='margin-top:24px;'>Recent ({len(entries)})</h2>"
               + "".join(items) if items else "")
        )
        return HTMLResponse(_layout("Journal", body, "journal"))

    @app.post("/journal/add")
    def journal_add_post(
        request: Request,
        text: str = Form(""), mood: int = Form(0),
    ):
        from fastapi.responses import RedirectResponse

        from . import personal
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        cfg, conn, embedder, _ = get_state()
        personal.upsert_journal(
            conn, mood=mood if mood > 0 else None, text=text,
        )
        entry = personal.get_journal(conn)
        if entry:
            personal.index_journal_entry(cfg, conn, embedder, entry)
        return RedirectResponse(url="/journal", status_code=303)

    # --- Phase 81: projects page ---------------------------------------

    @app.get("/projects", response_class=HTMLResponse)
    def projects_view():
        from . import personal

        cfg, conn, _, _ = get_read_state()
        projects = personal.list_projects(conn)
        if not projects:
            body = (
                "<h1>Projects</h1><div class='card'><p class='muted'>"
                "No projects yet. Run "
                "<code>secondbrain project new &lt;name&gt;</code>."
                "</p></div>"
            )
            return HTMLResponse(_layout("Projects", body, "projects"))
        items = "".join(
            f'<div class="stat">'
            f'<span><a href="/project?slug={p.slug}">'
            f'<strong>{escape(p.name)}</strong></a> '
            f'<span class="muted">({p.slug})</span></span>'
            f'<span class="v muted">{p.status}</span></div>'
            for p in projects
        )
        body = (
            f"<h1>Projects</h1><div class='card'><h2>Active</h2>{items}</div>"
        )
        return HTMLResponse(_layout("Projects", body, "projects"))

    @app.get("/project", response_class=HTMLResponse)
    def project_detail(slug: str):
        from . import personal

        cfg, conn, _, _ = get_read_state()
        p = personal.get_project_by_slug(conn, slug)
        if p is None:
            return HTMLResponse(_layout(
                slug, f"<h1>Project '{escape(slug)}' not found</h1>",
                "projects",
            ))
        view = personal.project_view(conn, p.id)
        sections: list[str] = []
        if view.files:
            files_html = "".join(
                f'<div class="stat">'
                f'<a href="/file?path={urllib.parse.quote_plus(path)}">'
                f'{escape(path)}</a></div>'
                for _fid, path in view.files
            )
            sections.append(
                f"<div class='card'><h2>Files ({len(view.files)})</h2>"
                + files_html + "</div>",
            )
        if view.tasks:
            tasks_html = "".join(
                f'<div class="stat">'
                f'<span><code>#{tid}</code> {escape(text)}</span></div>'
                for tid, text in view.tasks
            )
            sections.append(
                f"<div class='card'><h2>Tasks ({len(view.tasks)})</h2>"
                + tasks_html + "</div>",
            )
        if view.people:
            ppl_html = "".join(
                f'<div class="stat"><a href="/person?id={pid}">'
                f'{escape(name)}</a></div>'
                for pid, name in view.people
            )
            sections.append(
                f"<div class='card'><h2>People ({len(view.people)})</h2>"
                + ppl_html + "</div>",
            )
        body = (
            f"<h1>{escape(view.project.name)}</h1>"
            + (f"<p>{escape(view.project.description)}</p>"
               if view.project.description else "")
            + "".join(sections)
        )
        return HTMLResponse(_layout(view.project.name, body, "projects"))

    # --- Phase 83: drafts page -----------------------------------------

    @app.get("/drafts", response_class=HTMLResponse)
    def drafts_view():
        from . import ai_audit, email_assist
        from .safety import redact_text as _redact

        cfg, conn, _, _ = get_state()
        drafts = email_assist.list_unsent_drafts(conn)
        if not drafts:
            from .ux_copy import empty_state
            body = (
                "<h1>Email drafts</h1>"
                "<div class='card'><p class='muted'>"
                f"{empty_state('drafts')} "
                "<small>The daemon drafts replies to urgent / "
                "response-worthy emails on its next tick.</small>"
                "</p></div>"
            )
            return HTMLResponse(_layout("Drafts", body, "drafts"))
        items = []
        for d in drafts:
            when = time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(d.generated_at),
            )

            # Round 6: rich metadata block when the structured drafter
            # ran. Old single-version drafts skip the meta + alt sections
            # cleanly because the fields are None / empty.
            meta_html = ""
            if d.analysis is not None:
                a = d.analysis
                tone = ", ".join(a.tone_signals) or "neutral"
                points_html = "".join(
                    f"<li>{escape(_redact(p))}</li>"
                    for p in a.key_points
                ) or "<li class='muted'>(none)</li>"
                meta_html = (
                    f'<div class="meta-row" style="display:flex;gap:18px;'
                    f'flex-wrap:wrap;font-size:12px;color:var(--text-2);'
                    f'margin:4px 0 12px;">'
                    f'<span><span class="muted">intent:</span> {escape(a.intent)}</span>'
                    f'<span><span class="muted">to:</span> {escape(a.sender_relationship)}</span>'
                    f'<span><span class="muted">tone:</span> {escape(tone)}</span>'
                    f'<span><span class="muted">length:</span> {escape(a.length_target)}</span>'
                    + (f'<span><span class="muted">conf:</span> '
                       f'{d.confidence:.0%}</span>' if d.confidence else '')
                    + f'</div>'
                    f'<details style="margin:6px 0;font-size:12px;">'
                    f'<summary class="muted">sender asked for…</summary>'
                    f'<ul style="margin:6px 0 0 0;">{points_html}</ul>'
                    f'</details>'
                )

            # Reasoning bubble — short LLM explanation of choices.
            reasoning_html = ""
            if d.reasoning:
                reasoning_html = (
                    f'<details style="margin:8px 0;font-size:12px;">'
                    f'<summary class="muted">why this draft</summary>'
                    f'<p style="margin:6px 0 0 0;color:var(--text-2);">'
                    f'{escape(_redact(d.reasoning))}</p>'
                    f'</details>'
                )

            # Open questions checklist — explicit TODOs the user has to
            # decide before sending. Rendered as a checkbox list so the
            # user can mentally tick them off as they fill placeholders.
            todos_html = ""
            if d.open_questions:
                todos_html = (
                    '<div class="card" style="background:rgba(255,183,0,0.05);'
                    'border-color:var(--amber);margin:8px 0;padding:8px 12px;">'
                    '<div class="muted" style="font-size:11px;'
                    'text-transform:uppercase;letter-spacing:0.05em;'
                    'margin-bottom:4px;">Decide before sending</div>'
                    + '<ul style="margin:0;padding-left:18px;">'
                    + "".join(
                        f'<li>{escape(_redact(q))}</li>'
                        for q in d.open_questions
                    )
                    + '</ul></div>'
                )

            # Primary draft block + (optional) alternative-tone version
            # in a side-by-side fold-out. Use <details> so users with
            # the right tone-match on the primary don't have to look
            # at the alternative.
            primary_html = (
                f'<div class="muted" style="font-size:11px;'
                f'text-transform:uppercase;letter-spacing:0.05em;'
                f'margin:12px 0 4px;">Primary</div>'
                f'<pre style="white-space:pre-wrap;font-family:inherit;'
                f'margin:0;background:var(--bg-input);padding:10px;'
                f'border-radius:var(--r);border:1px solid var(--border-strong);">'
                f'{escape(_redact(d.draft_text))}</pre>'
            )
            alt_html = ""
            if d.alternative_text and d.alternative_text != d.draft_text:
                alt_html = (
                    f'<details style="margin-top:8px;">'
                    f'<summary class="muted" style="font-size:11px;'
                    f'text-transform:uppercase;letter-spacing:0.05em;'
                    f'cursor:pointer;">Alternative tone</summary>'
                    f'<pre style="white-space:pre-wrap;font-family:inherit;'
                    f'margin:6px 0 0 0;background:var(--bg-input);padding:10px;'
                    f'border-radius:var(--r);border:1px solid var(--border-strong);">'
                    f'{escape(_redact(d.alternative_text))}</pre>'
                    f'</details>'
                )

            # Round 10 (#3) — per-draft cost: sum LLM calls within
            # 90s of the generated_at timestamp keyed to this file.
            cost_info = ai_audit.cost_for_window(
                conn, file_id=d.file_id, around_ts=d.generated_at,
            )
            cost_html = ""
            if cost_info["cents"] > 0:
                cost_html = (
                    f'<span class="muted" style="font-size:11px;">'
                    f' · ${cost_info["cents"] / 100:.4f} '
                    f'({cost_info["n"]} call{"s" if cost_info["n"] != 1 else ""})'
                    f'</span>'
                )

            items.append(
                f'<div class="card" style="margin-bottom:12px;">'
                f'<div class="stat" style="border-bottom:none;padding-bottom:0;">'
                f'<strong>Draft #{d.id}</strong>'
                f'<span class="v muted">{when}{cost_html}</span>'
                f'</div>'
                f'{meta_html}'
                f'{todos_html}'
                f'{primary_html}'
                f'{alt_html}'
                f'{reasoning_html}'
                f'<div style="margin-top:12px;">'
                f'<form method="post" action="/drafts/{d.id}/sent" '
                f'style="display:inline;margin-right:8px;">'
                f'<button type="submit">Mark sent</button></form>'
                f'<form method="post" action="/drafts/{d.id}/discard" '
                f'style="display:inline;">'
                f'<button type="submit" class="link-btn">Discard</button>'
                f'</form></div>'
                f'</div>',
            )
        # Round 10 (#2) — accept-rate stat at top of /drafts. Lets
        # the user see "I rejected 60% of drafts last week" without
        # having to count.
        stats = email_assist.feedback_stats(conn, days=14)
        stats_html = ""
        if stats["accepted"] + stats["rejected"] > 0:
            rate_pct = int(stats["accept_rate"] * 100)
            stats_html = (
                f"<div class='card' style='margin-bottom:16px;"
                f"font-size:13px;'>"
                f"<strong>Last 14 days:</strong> "
                f"{stats['accepted']} accepted, "
                f"{stats['rejected']} rejected — "
                f"<strong>{rate_pct}%</strong> accept rate"
                f"</div>"
            )
        body = (
            f"<h1>Email drafts ({len(drafts)})</h1>"
            + stats_html
            + "".join(items)
        )
        return HTMLResponse(_layout("Drafts", body, "drafts"))

    @app.post("/drafts/{draft_id:int}/sent")
    def drafts_mark_sent(draft_id: int, request: Request):
        from fastapi.responses import RedirectResponse

        from . import email_assist, meeting_thanks
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        cfg, conn, _, _ = get_state()
        email_assist.mark_draft_sent(conn, draft_id)
        # Round 8 — also flip any linked meeting_thanks row to 'sent'
        # so the /thanks page reflects reality.
        try:
            meeting_thanks.mark_sent_for_draft(conn, draft_id)
        except Exception:  # noqa: BLE001
            pass
        return RedirectResponse(url="/drafts", status_code=303)

    @app.post("/drafts/{draft_id:int}/discard")
    def drafts_discard(draft_id: int, request: Request):
        from fastapi.responses import RedirectResponse

        from . import email_assist
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        cfg, conn, _, _ = get_state()
        email_assist.discard_draft(conn, draft_id)
        return RedirectResponse(url="/drafts", status_code=303)

    # --- Round 8: meeting thanks page ---------------------------------

    @app.get("/thanks", response_class=HTMLResponse)
    def thanks_view():
        """List meetings waiting on context or a draft. Form per row
        for "I had this conversation about X" notes; transcript-
        matched rows show the source path. Drafted rows link back to
        /drafts so the user has one place to actually send from."""
        from . import meeting_thanks
        from .safety import redact_text as _redact

        cfg, conn, _, _ = get_state()
        rows = meeting_thanks.list_pending(conn)
        drafted = [
            r for r in meeting_thanks.list_all(conn, limit=20)
            if r.status in ("drafted", "sent")
        ]
        if not rows and not drafted:
            body = (
                "<h1>Meeting thanks</h1><div class='card'><p class='muted'>"
                "No recent meetings to thank-you yet. The daemon scans "
                "calendar events hourly; you can also force a scan via "
                "<code>secondbrain thanks scan</code>."
                "</p></div>"
            )
            return HTMLResponse(_layout("Thanks", body, "thanks"))

        items: list[str] = []
        for mt in rows:
            when = time.strftime(
                "%a %b %d, %H:%M", time.localtime(mt.starts_at),
            )
            attendees = escape(", ".join(mt.attendees) or "(unknown)")
            ctx_html = ""
            if mt.transcript_path:
                ctx_html = (
                    f"<div class='stat' style='border-bottom:none;'>"
                    f"<span class='muted'>Transcript matched:</span>"
                    f"<span class='v'><code>{escape(mt.transcript_path)}</code></span>"
                    f"</div>"
                    f"<form method='post' action='/thanks/{mt.id}/draft' "
                    f"style='display:inline;margin-right:8px;'>"
                    f"<button type='submit'>Draft thank-you now</button>"
                    f"</form>"
                )
            else:
                # Free-text context form. Once submitted, the row flips
                # to 'ready' and the daemon picks it up next tick.
                ctx_html = (
                    f"<form method='post' action='/thanks/{mt.id}/context' "
                    f"style='margin-top:8px;'>"
                    f"<label class='muted' style='font-size:11px;"
                    f"text-transform:uppercase;letter-spacing:0.05em;'>"
                    f"What did you talk about?</label>"
                    f"<textarea name='text' rows='4' "
                    f"style='width:100%;margin-top:4px;' "
                    f"placeholder='Topics discussed, anything you "
                    f"committed to follow up on, personal details "
                    f"worth referencing…'></textarea>"
                    f"<div style='margin-top:8px;'>"
                    f"<button type='submit'>Save context + queue draft</button>"
                    f"</div></form>"
                )
            skip_html = (
                f"<form method='post' action='/thanks/{mt.id}/skip' "
                f"style='display:inline;'>"
                f"<button type='submit' class='link-btn'>Skip</button>"
                f"</form>"
            )
            items.append(
                f"<div class='card' style='margin-bottom:12px;'>"
                f"<div class='stat'>"
                f"<strong>{escape(_redact(mt.event_title))}</strong>"
                f"<span class='v muted'>{when}</span>"
                f"</div>"
                f"<div class='stat' style='border-bottom:none;padding-bottom:0;'>"
                f"<span class='muted'>To:</span>"
                f"<span class='v'>{attendees}</span>"
                f"</div>"
                f"{ctx_html}"
                f"<div style='margin-top:8px;'>{skip_html}</div>"
                f"</div>",
            )
        drafted_html = ""
        if drafted:
            sub_items = []
            for mt in drafted[:8]:
                when = time.strftime(
                    "%a %b %d", time.localtime(mt.starts_at),
                )
                link = (
                    f"<a href='/drafts'>draft #{mt.draft_id}</a>"
                    if mt.draft_id else "(draft missing)"
                )
                marker = "✓" if mt.status == "sent" else "✎"
                sub_items.append(
                    f"<div class='stat'>"
                    f"<span>{marker} {escape(mt.event_title)}</span>"
                    f"<span class='v muted'>{when} · {link}</span>"
                    f"</div>",
                )
            drafted_html = (
                "<h2>Already drafted / sent</h2>"
                "<div class='card'>" + "".join(sub_items) + "</div>"
            )

        body = (
            f"<h1>Meeting thanks ({len(rows)} pending)</h1>"
            + "".join(items)
            + drafted_html
        )
        return HTMLResponse(_layout("Thanks", body, "thanks"))

    @app.post("/thanks/{mt_id:int}/context")
    def thanks_set_context(
        mt_id: int, request: Request, text: str = Form(""),
    ):
        from fastapi.responses import RedirectResponse

        from . import meeting_thanks
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        cfg, conn, _, _ = get_state()
        meeting_thanks.set_context(conn, mt_id, text)
        return RedirectResponse(url="/thanks", status_code=303)

    @app.post("/thanks/{mt_id:int}/skip")
    def thanks_skip_post(mt_id: int, request: Request):
        from fastapi.responses import RedirectResponse

        from . import meeting_thanks
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        cfg, conn, _, _ = get_state()
        meeting_thanks.mark_skipped(conn, mt_id)
        return RedirectResponse(url="/thanks", status_code=303)

    @app.post("/thanks/{mt_id:int}/draft")
    def thanks_draft_post(mt_id: int, request: Request):
        """Synchronously draft + redirect to /drafts so the user
        sees the result immediately. The daemon would do the same
        on its next tick — this is just the "do it now" button."""
        from fastapi.responses import RedirectResponse

        from . import meeting_thanks
        if not _is_same_origin_request(request):
            return HTMLResponse("Forbidden", status_code=403)
        cfg, conn, _, _ = get_state()
        meeting_thanks.generate_thanks_draft(conn, cfg, mt_id)
        return RedirectResponse(url="/drafts", status_code=303)

    # --- Round 10 (#6): AI audit log ----------------------------------

    # Round 10 (#9) — health page (separate from /audit so the user
    # can drill into integration failures without the AI noise).

    @app.get("/health/system", response_class=HTMLResponse)
    def system_health_view():
        from . import health_checks

        cfg, conn, _, _ = get_state()
        statuses = health_checks.list_status(conn)
        if not statuses:
            # Trigger an immediate run when the page is hit cold.
            try:
                health_checks.run_all(conn, cfg)
                statuses = health_checks.list_status(conn)
            except Exception:  # noqa: BLE001
                statuses = []
        if not statuses:
            body = (
                "<h1>Diagnostics</h1>"
                "<div class='card'><p class='muted'>"
                "No health checks have run yet. Run "
                "<code>secondbrain doctor</code> in a terminal."
                "</p></div>"
            )
            # Round 11 — pass an active slug that matches NO nav item
            # so we don't highlight the Personal/Health (Oura) entry
            # by accident. The page title in the H1 tells the user
            # where they are.
            return HTMLResponse(_layout("Diagnostics", body, "diagnostics"))
        items = []
        for s in statuses:
            ok_marker = "✓" if s.ok else "✗"
            ok_class = "good" if s.ok else "bad"
            when = time.strftime(
                "%a %b %d %H:%M", time.localtime(s.last_checked_at),
            )
            err_html = ""
            if s.error and not s.ok:
                err_html = (
                    f"<div class='muted' style='color:var(--bad);"
                    f"font-size:12px;margin-top:4px;'>{escape(s.error)}</div>"
                )
            items.append(
                f"<div class='stat' style='align-items:flex-start;'>"
                f"<span style='flex:1;'>"
                f"<strong>{escape(s.name)}</strong> "
                f"<span class='muted' style='font-size:11px;'>"
                f"checked {when}</span>"
                f"{err_html}"
                f"</span>"
                f"<span class='v {ok_class}'>{ok_marker}</span>"
                f"</div>",
            )
        body = (
            "<h1>Diagnostics</h1>"
            "<div class='card'><h2>Integrations</h2>"
            + "".join(items)
            + "</div>"
        )
        return HTMLResponse(_layout("Diagnostics", body, "diagnostics"))

    @app.get("/audit", response_class=HTMLResponse)
    def audit_view():
        """Round 10 (#6) — list every AI action with cost / status /
        summary. Lets the user answer "why did the AI do X?" without
        reading source code.

        Round 11 fix (audit-found bug) — uses the writer conn, not
        the read-only conn. ``ai_audit.recent`` calls ``_ensure_schema``
        which runs ``CREATE TABLE IF NOT EXISTS`` + ``COMMIT`` — that
        raises through a read-only conn on a brain old enough to lack
        the ai_actions table. Same risk for ``health_checks.list_status``."""
        from . import ai_audit

        cfg, conn, _, _ = get_state()
        rows = ai_audit.recent(conn, limit=200)
        if not rows:
            body = (
                "<h1>AI audit log</h1>"
                "<div class='card'><p class='muted'>"
                "No AI actions logged yet. Generate a draft, run "
                "summarisation, or trigger a thank-you and the "
                "events will land here."
                "</p></div>"
            )
            return HTMLResponse(_layout("Audit", body, "audit"))
        kind_counts = ai_audit.by_kind(conn, days=7)
        rollup = ai_audit.rollup_today(conn)
        total_cents = sum(s["cents"] for s in rollup.values())
        total_calls = sum(s["n"] for s in rollup.values())

        # Top: 24h summary card with per-feature breakdown.
        rollup_rows = "".join(
            f"<div class='stat'>"
            f"<span><strong>{escape(feat)}</strong> "
            f"<span class='muted'>{stats['n']} call(s)</span></span>"
            f"<span class='v'>${stats['cents'] / 100:.4f}</span>"
            f"</div>"
            for feat, stats in rollup.items()
        )
        rollup_html = (
            "<div class='card' style='margin-bottom:16px;'>"
            f"<h2>Last 24h — ${total_cents / 100:.4f} across "
            f"{total_calls} call(s)</h2>"
            f"{rollup_rows or '<p class=muted>(no spend recorded)</p>'}"
            "</div>"
        )

        # Kind chips for the last 7 days.
        chips_html = ""
        if kind_counts:
            chip_items = " ".join(
                f"<span class='kbd' style='margin:2px;'>"
                f"{escape(k)}: {n}</span>"
                for k, n in kind_counts.items()
            )
            chips_html = (
                f"<div class='card' style='margin-bottom:16px;'>"
                f"<h2>Last 7 days by kind</h2>{chip_items}</div>"
            )

        # Action table.
        items = []
        for r in rows:
            when = time.strftime(
                "%a %b %d %H:%M:%S", time.localtime(r.ts),
            )
            cost = (
                f"${r.cents / 100:.4f}"
                if r.cents > 0 else "·"
            )
            status_class = (
                "good" if r.status == "success"
                else "warn" if r.status == "fallback_local"
                else "bad"
            )
            err_html = (
                f"<div class='muted' style='font-size:11px;"
                f"color:var(--bad);'>{escape(r.error)}</div>"
                if r.error else ""
            )
            # Round 11 — clickable links to the underlying surface.
            # Draft / thanks-draft → /drafts. File-scoped actions →
            # /file?path=… (we look up the path on demand).
            link_html = ""
            if r.kind in ("draft", "thanks_draft", "thanks_regen"):
                link_html = (
                    " <a href='/drafts' class='muted' "
                    "style='font-size:11px;'>→ drafts</a>"
                )
            elif r.file_id:
                # Resolve file_id to path on the fly (cheap).
                frow = conn.execute(
                    "SELECT path FROM files WHERE id = ?",
                    (r.file_id,),
                ).fetchone()
                if frow:
                    fp = urllib.parse.quote_plus(frow["path"])
                    link_html = (
                        f" <a href='/file?path={fp}' class='muted' "
                        f"style='font-size:11px;'>→ source</a>"
                    )
            items.append(
                f"<div class='stat' style='align-items:flex-start;'>"
                f"<span style='flex:1;'>"
                f"<span class='muted' style='font-size:11px;'>{when}</span> "
                f"<strong>{escape(r.kind)}</strong> "
                f"<span class='muted'>{escape(r.model[:24])}</span>"
                f"{link_html}"
                f"<div style='margin-top:2px;'>"
                f"{escape(r.summary)}</div>"
                f"{err_html}"
                f"</span>"
                f"<span class='v {status_class}'>"
                f"<div>{escape(r.status)}</div>"
                f"<div class='muted' style='font-size:11px;'>{cost}</div>"
                f"</span></div>"
            )
        body = (
            f"<h1>AI audit log</h1>"
            f"{rollup_html}"
            f"{chips_html}"
            f"<div class='card'><h2>Recent actions ({len(rows)})</h2>"
            + "".join(items)
            + "</div>"
        )
        return HTMLResponse(_layout("Audit", body, "audit"))

    # --- Round 9-A: meeting prep page ---------------------------------

    @app.get("/prep", response_class=HTMLResponse)
    def prep_view():
        """List upcoming external meetings with brain-grounded prep.

        Each meeting card shows attendee names + days-since-seen
        + open tasks involving them + topics that come up. Click
        through for the full markdown prep doc."""
        from . import meeting_prep
        from .safety import redact_text as _redact

        cfg, conn, _, _ = get_state()
        try:
            preps = meeting_prep.upcoming_preps(conn, cfg)
        except Exception:  # noqa: BLE001
            # Calendar fetch / OAuth issues become an empty list; the
            # page renders the friendly "no upcoming" fallback below.
            preps = []
        if not preps:
            body = (
                "<h1>Upcoming meetings — prep</h1>"
                "<div class='card'><p class='muted'>"
                "No upcoming external meetings in the next 24h. "
                "(Or your calendar isn't connected — see the "
                "<code>secondbrain auth google</code> docs.)"
                "</p></div>"
            )
            return HTMLResponse(_layout("Prep", body, "prep"))

        items: list[str] = []
        for p in preps:
            attendee_blocks = []
            for a in p.attendees:
                meta_bits: list[str] = []
                if a.days_since_seen:
                    meta_bits.append(f"{a.days_since_seen}d since seen")
                if a.n_prior_emails:
                    meta_bits.append(f"{a.n_prior_emails} prior email(s)")
                if a.n_open_tasks:
                    meta_bits.append(f"{a.n_open_tasks} open task(s)")
                meta_html = ""
                if meta_bits:
                    meta_html = (
                        f"<div class='muted' style='font-size:11px;"
                        f"margin-top:2px;'>"
                        f"{escape(' · '.join(meta_bits))}</div>"
                    )
                tasks_html = ""
                if a.open_task_lines:
                    items_html = "".join(
                        f"<li>{escape(_redact(t))}</li>"
                        for t in a.open_task_lines[:5]
                    )
                    tasks_html = (
                        "<details style='margin-top:6px;font-size:12px;'>"
                        "<summary class='muted'>Open with this person</summary>"
                        f"<ul style='margin:4px 0 0 0;'>{items_html}</ul>"
                        "</details>"
                    )
                topics_html = ""
                if a.co_topics:
                    topics_html = (
                        f"<div class='muted' style='font-size:12px;"
                        f"margin-top:6px;'>topics: "
                        f"{escape(', '.join(a.co_topics[:6]))}</div>"
                    )
                first_time = (
                    a.days_since_seen == 0
                    and a.n_prior_emails == 0
                    and a.n_open_tasks == 0
                )
                first_html = ""
                if first_time:
                    first_html = (
                        "<div class='muted' style='font-size:12px;"
                        "margin-top:4px;'>First time you're seeing "
                        "this person in your brain.</div>"
                    )
                attendee_blocks.append(
                    f"<div style='margin-top:10px;'>"
                    f"<strong>{escape(_redact(a.name))}</strong> "
                    f"<span class='muted'>&lt;{escape(a.email)}&gt;</span>"
                    f"{meta_html}"
                    f"{first_html}"
                    f"{tasks_html}"
                    f"{topics_html}"
                    f"</div>",
                )
            duration = (
                f"{p.duration_minutes} min" if p.duration_minutes else ""
            )
            location = f" · {escape(p.location)}" if p.location else ""
            items.append(
                f"<div class='card' style='margin-bottom:12px;'>"
                f"<div class='stat' style='border-bottom:none;padding-bottom:0;'>"
                f"<strong>{escape(_redact(p.title))}</strong>"
                f"<span class='v muted'>{escape(p.when_str)} · "
                f"{escape(duration)}{location}</span>"
                f"</div>"
                + "".join(attendee_blocks)
                + "</div>",
            )
        body = (
            f"<h1>Upcoming meetings — prep ({len(preps)})</h1>"
            + "".join(items)
        )
        return HTMLResponse(_layout("Prep", body, "prep"))

    # --- Browser extension API ----------------------------------------
    # These endpoints are consumed by the multi-AI bridge browser
    # extension. CORS is allowed for the AI hosts so a content script
    # running on chat.openai.com / gemini.google.com / etc. can fetch
    # context from the local dashboard. The dashboard binds to 127.0.0.1
    # only, so these endpoints are not reachable from elsewhere on the
    # network.

    _EXTENSION_ALLOWED_ORIGINS = {
        "https://chat.openai.com",
        "https://chatgpt.com",
        "https://gemini.google.com",
        "https://www.perplexity.ai",
        "https://perplexity.ai",
        "https://x.com",
        "https://grok.com",
        "https://chat.deepseek.com",
    }

    def _extension_cors(origin: str | None) -> dict[str, str]:
        if origin and origin in _EXTENSION_ALLOWED_ORIGINS:
            return {
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                # Authorization must be in the allowlist or the browser strips
                # it from cross-origin preflight-required requests.
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
                "Vary": "Origin",
            }
        return {}

    def _extension_authorized(request: Request) -> bool:
        """Bearer-token + 127.0.0.1-only check.

        The CORS allow-list is *not* authentication - any JS running on one
        of those origins (a content script in another extension, an XSS) can
        otherwise read the entire index. Extension surfaces the per-install
        token from ``secondbrain auth extension`` and presents it here.
        """
        # We only ever bind to 127.0.0.1, but defense-in-depth: reject if
        # the request didn't come from loopback.
        client_host = (request.client.host if request.client else "") or ""
        if client_host not in {"127.0.0.1", "::1", "localhost"}:
            return False
        cfg, _, _, _ = get_state()
        expected = get_or_create_extension_token(cfg)
        header = request.headers.get("authorization", "")
        if not header.lower().startswith("bearer "):
            return False
        presented = header.split(" ", 1)[1].strip()
        return secrets.compare_digest(presented, expected)

    @app.options("/api/extension/{path:path}")
    def extension_preflight(request: Request, path: str):  # noqa: ARG001
        from fastapi.responses import Response as _Resp

        return _Resp(headers=_extension_cors(request.headers.get("origin")))

    @app.get("/api/extension/health")
    def extension_health(request: Request):
        from fastapi.responses import JSONResponse as _JSON

        cors = _extension_cors(request.headers.get("origin"))
        if not _extension_authorized(request):
            return _JSON({"ok": False, "error": "unauthorized"}, status_code=401, headers=cors)
        return _JSON({"ok": True, "name": "second-brain"}, headers=cors)

    @app.get("/api/extension/search")
    def extension_search(request: Request, q: str = "", k: int = 5):
        """Return compact context blocks for the browser extension to inject
        before a user's prompt to ChatGPT / Gemini / etc."""
        from fastapi.responses import JSONResponse as _JSON

        cors = _extension_cors(request.headers.get("origin"))
        if not _extension_authorized(request):
            return _JSON({"error": "unauthorized"}, status_code=401, headers=cors)
        if not q.strip():
            return _JSON({"results": []}, headers=cors)
        cfg, conn, embedder, reranker = get_state()
        results = hybrid_search(
            conn, embedder, q, k=min(max(k, 1), 10),
            alpha=cfg.hybrid_alpha,
            reranker=reranker, rerank_overfetch=cfg.rerank_overfetch,
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
        # Log this query in the same audit trail as MCP-driven queries.
        try:
            _log_query(cfg, q, "extension", results)
        except Exception:
            pass
        out = []
        for r in results:
            out.append({
                "path": r.file_path,
                "chunk_index": r.chunk_index,
                "snippet": r.text if len(r.text) <= 1500 else r.text[:1500] + "...",
                "score": round(r.score, 4),
            })
        return _JSON({"results": out, "query": q}, headers=cors)

    @app.post("/api/capture")
    async def api_capture(request: Request):
        """Phase 69: capture endpoint for iOS Shortcuts / browser
        bookmarklets / curl one-liners. Auth via the same bearer token
        as ``/api/extension/*``.

        Request body (JSON):
            {
              "title":   "headline of what's being saved",
              "content": "main text — selection / note / URL",
              "url":     "optional canonical URL",
              "source":  "ios" | "shortcut" | "bookmarklet" | "curl"
            }

        The doc lands at ``capture://<source>/<timestamp>`` and gets
        indexed inline so it shows up in search immediately. Idempotent
        only insofar as duplicate URLs land in the alias table — same
        text content gets re-indexed.

        Returns ``{ok, virtual_path, chunks}`` on success.
        """
        from fastapi.responses import JSONResponse as _JSON

        cors = _extension_cors(request.headers.get("origin"))
        if not _extension_authorized(request):
            return _JSON(
                {"error": "unauthorized"}, status_code=401, headers=cors,
            )
        try:
            body = await request.json()
        except Exception:
            return _JSON(
                {"error": "JSON body required"},
                status_code=400, headers=cors,
            )
        title = (body.get("title") or "").strip()
        content = (body.get("content") or "").strip()
        url = (body.get("url") or "").strip()
        source = (body.get("source") or "manual").strip()[:40]
        if not content and not url:
            return _JSON(
                {"error": "need content or url"},
                status_code=400, headers=cors,
            )

        cfg, conn, embedder, _ = get_state()
        # If only a URL was provided, route through the URL ingestion
        # path (fetches + parses) rather than indexing the URL string.
        if url and not content:
            from .indexer import index_url
            try:
                result = index_url(
                    conn, embedder, cfg, url=url,
                    entity_extractor=None,
                )
            except Exception as e:  # noqa: BLE001
                return _JSON(
                    {"error": f"ingest failed: {e}"},
                    status_code=500, headers=cors,
                )
            return _JSON(
                {
                    "ok": True, "virtual_path": url,
                    "status": result.status,
                    "chunks": result.chunks or 0,
                }, headers=cors,
            )

        # Plain-text capture.
        from .indexer import index_text
        ts = int(time.time())
        virtual_path = f"capture://{source}/{ts}"
        # Header for the rendered content gives the title + URL
        # context if present.
        rendered = f"# {title or '(captured note)'}"
        if url:
            rendered += f"\n\nSource: {url}"
        rendered += f"\n\n{content}"
        try:
            result = index_text(
                conn, embedder, cfg,
                virtual_path=virtual_path,
                title=title or content[:60],
                content=rendered,
                mtime=time.time(),
                kind="capture",
                source=source,
            )
        except Exception as e:  # noqa: BLE001
            return _JSON(
                {"error": f"index failed: {e}"},
                status_code=500, headers=cors,
            )
        return _JSON(
            {
                "ok": True, "virtual_path": virtual_path,
                "status": result.status,
                "chunks": result.chunks or 0,
            }, headers=cors,
        )

    @app.post("/api/click")
    async def api_click(request: Request):
        """Lightweight click beacon. The dashboard front-end POSTs here
        whenever the user opens a search/chat/palette result. We log the
        path so subsequent searches lift it via the click-recency boost.

        Round 18 fix (audit-found gap H3) — same-origin guard. The
        round-14 CSRF sweep covered every other state-mutating POST
        but missed this one. ``log_click`` writes to the ``clicks``
        table which biases search ranking via the click-recency
        boost; without a guard, a malicious page the user visits in
        another tab can poll-flood arbitrary paths into the log to
        skew search results. Browsers always send Referer on
        same-origin sendBeacon, so the existing helper still
        accepts the dashboard's own JS.
        """
        from fastapi.responses import JSONResponse as _JSON

        from .db import log_click

        if not _is_same_origin_request(request):
            return _JSON({"ok": False, "error": "forbidden"}, status_code=403)
        try:
            body = await request.json()
        except Exception:
            try:
                form = await request.form()
                body = dict(form)
            except Exception:
                return {"ok": False, "error": "could not parse body"}
        path = (body.get("path") or "").strip()
        if not path:
            return {"ok": False, "error": "missing path"}
        source = (body.get("source") or "unknown").strip()[:32]
        chunk_id_raw = body.get("chunk_id")
        try:
            chunk_id = int(chunk_id_raw) if chunk_id_raw is not None else None
        except (TypeError, ValueError):
            chunk_id = None
        _, conn, _, _ = get_state()
        log_click(conn, path, source, chunk_id=chunk_id)
        return {"ok": True}

    @app.get("/api/nav-counts")
    def api_nav_counts():
        """Pending-state counts for the primary nav badges. Cheap —
        small reads against the writer conn (round 11 fix: was
        read-only, but several of the count helpers — ai_audit,
        health_checks, email_drafts — call ``_ensure_schema`` which
        writes). Failures inside any one count fall back to 0 so a
        missing schema (fresh brain) doesn't break the nav."""
        cfg, conn, _, _ = get_state()
        out = {
            "tasks": 0, "drafts": 0, "insights": 0, "thanks": 0,
            "urgent": {"drafts": False, "thanks": False},
        }
        # Open tasks
        try:
            from . import tasks as tasks_mod
            out["tasks"] = len(tasks_mod.list_open(conn, limit=200))
        except Exception:  # noqa: BLE001
            pass
        # Pending drafts
        try:
            from . import email_assist
            n_drafts = len(email_assist.list_unsent_drafts(conn, limit=200))
            out["drafts"] = n_drafts
            if n_drafts >= 5:
                out["urgent"]["drafts"] = True
        except Exception:  # noqa: BLE001
            pass
        # Active (un-deduped) insights — kept for "More" surfacing.
        try:
            from . import synthesis
            insights = synthesis.detect_insights(conn)
            out["insights"] = len(insights)
        except Exception:  # noqa: BLE001
            pass
        # Round 8 — pending meeting thanks. Counts both 'pending_context'
        # (need user input) AND 'ready' (waiting for daemon to draft).
        try:
            from . import meeting_thanks
            n_thanks = len(meeting_thanks.list_pending(conn, limit=200))
            out["thanks"] = n_thanks
            # Anything older than 24h with no draft is getting stale —
            # surface as urgent so the user is nudged to act.
            if n_thanks >= 3:
                out["urgent"]["thanks"] = True
        except Exception:  # noqa: BLE001
            pass
        # Round 11 — surface stale health failures as the badge on
        # the (out-of-primary-nav) Audit slot. Visible from every
        # page so a broken integration doesn't hide for days.
        try:
            from . import health_checks
            stale = health_checks.stale_failures(conn)
            out["health"] = len(stale)
            if stale:
                out["urgent"]["health"] = True
        except Exception:  # noqa: BLE001
            pass
        # Round 16 (Phase C) — pending notifications.
        try:
            from . import notifications as notif_mod
            n_notif = notif_mod.count_pending(conn)
            out["notifications"] = n_notif
            # Any high-urgency unread → mark urgent.
            if n_notif > 0:
                pending = notif_mod.list_pending(conn, limit=50)
                if any(n.urgency == "high" for n in pending):
                    out["urgent"]["notifications"] = True
        except Exception:  # noqa: BLE001
            pass
        return out

    @app.get("/api/palette")
    def api_palette(q: str = ""):
        """Mixed search for the command palette: entities + files matching q.
        High-traffic (fires on every keystroke) → read-only conn."""
        cfg, conn, _, _ = get_read_state()
        q = q.strip()
        if len(q) < 2:
            return {"entities": [], "files": []}

        # Entities — case-insensitive substring on text_lower; rank by chunk count.
        ent_rows = conn.execute(
            "SELECT text, label, COUNT(DISTINCT chunk_id) AS n "
            "FROM entities WHERE text_lower LIKE ? "
            "GROUP BY text_lower, label ORDER BY n DESC LIMIT 8",
            (f"%{q.lower()}%",),
        ).fetchall()
        entities = [
            {
                "icon": "※",
                "label": r["text"],
                "href": f"/entity?name={urllib.parse.quote_plus(r['text'])}",
                "meta": f"{r['label']} · {r['n']}",
            }
            for r in ent_rows
        ]

        # Files — match by filename (last path segment) for now.
        file_rows = conn.execute(
            "SELECT path, kind FROM files "
            "WHERE LOWER(path) LIKE ? ORDER BY mtime DESC LIMIT 8",
            (f"%{q.lower()}%",),
        ).fetchall()
        files = [
            {
                "icon": "▣" if r["kind"] == "url" else "□",
                "label": Path(r["path"]).name or r["path"],
                "href": f"/file?path={urllib.parse.quote_plus(r['path'])}",
                "meta": r["kind"],
            }
            for r in file_rows
        ]

        return {"entities": entities, "files": files}

    # ============================================================
    # Phase 73 / 75 / 86 / 87 — pages added in the polish-v3 audit:
    # surface flagship recent features that previously had only a
    # CLI / MCP entry point.
    # ============================================================

    @app.get("/snapshots", response_class=HTMLResponse)
    def snapshots_view():
        """Phase 87 — list weekly index snapshots so the user can
        eyeball the timeline of their brain. Each row links to a
        scoped search that filters to that snapshot's file set."""
        from . import memory as memory_mod

        cfg, conn, _, _ = get_read_state()
        snaps = memory_mod.list_snapshots(conn, limit=50)
        if not snaps:
            body = (
                "<h1>Snapshots</h1>"
                "<div class='card'><p class='muted'>"
                "No snapshots yet. The daemon takes one weekly; run "
                "<code>secondbrain snapshot take</code> to capture "
                "the current state immediately."
                "</p></div>"
            )
            return HTMLResponse(_layout("Snapshots", body, "snapshots"))
        rows_html = []
        now = time.time()
        for s in snaps:
            when = time.strftime("%Y-%m-%d", time.localtime(s.taken_at))
            age = max(0, int((now - s.taken_at) // 86400))
            label = f" <span class='muted'>· {escape(s.label)}</span>" if s.label else ""
            rows_html.append(
                f"<div class='stat'>"
                f"<span><strong>#{s.id}</strong> {when}{label}</span>"
                f"<span class='v muted'>{s.n_files} files · {age}d ago</span>"
                f"</div>",
            )
        body = (
            f"<h1>Snapshots ({len(snaps)})</h1>"
            f"<div class='card'><p class='muted'>"
            f"Snapshots support temporal queries — "
            f"<code>secondbrain search 'X' --as-of '2 weeks ago'</code> "
            f"filters results to the closest preceding snapshot."
            f"</p>{''.join(rows_html)}</div>"
        )
        return HTMLResponse(_layout("Snapshots", body, "snapshots"))

    @app.get("/insights", response_class=HTMLResponse)
    def insights_view():
        """Phase 75 — proactive 'I noticed X' surfacing. Same data the
        daily brief uses, surfaced standalone for ad-hoc viewing."""
        from . import synthesis

        cfg, conn, _, _ = get_read_state()
        insights = synthesis.detect_insights(conn)
        if not insights:
            body = (
                "<h1>Insights</h1>"
                "<div class='card'><p class='muted'>"
                "Nothing new to flag right now. Insights surface "
                "topic spikes (entities trending in your recent docs) "
                "and health drift (Oura metrics out of band). "
                "Already-shown insights are deduped for 7 days."
                "</p></div>"
            )
            return HTMLResponse(_layout("Insights", body, "insights"))
        from .safety import redact_text as _redact
        items_html = "".join(
            f"<div class='card' style='margin-bottom:12px;'>"
            f"<h3 style='margin:0 0 6px 0;'>{escape(_redact(i.headline))}</h3>"
            f"<div class='muted' style='font-size:0.85em;'>{escape(i.kind)}</div>"
            f"<p style='margin:8px 0 0 0;'>{escape(_redact(i.detail))}</p>"
            f"</div>"
            for i in insights
        )
        body = f"<h1>Insights ({len(insights)})</h1>{items_html}"
        return HTMLResponse(_layout("Insights", body, "insights"))

    @app.get("/study/review", response_class=HTMLResponse)
    def study_review_view():
        """Phase 67 — flashcard review session. Lists due cards;
        each card has 'show answer' + 4 SM-2 grade buttons. Stays
        server-rendered (one card per page) to keep the dashboard
        dependency-free."""
        from . import study

        cfg, conn, _, _ = get_state()
        due = study.due_cards(conn, limit=20)
        if not due:
            body = (
                "<h1>Flashcard review</h1>"
                "<div class='card'><p class='muted'>"
                "No cards due. Run "
                "<code>secondbrain study quiz</code> to see all your "
                "decks, or wait for the daemon to materialise more "
                "cards from your <code>[course]</code> docs."
                "</p></div>"
            )
            return HTMLResponse(_layout("Study", body, "study"))
        from .safety import redact_text as _redact
        items_html = []
        for c in due:
            items_html.append(
                f"<div class='card' style='margin-bottom:12px;'>"
                f"<div class='stat'>"
                f"<span><strong>Card #{c.id}</strong> "
                f"<span class='muted'>{escape(c.concept)}</span></span>"
                f"<span class='v muted'>"
                f"{escape(c.course_code)} · ease {c.ease:.2f}"
                f"</span></div>"
                f"<details style='margin-top:8px;'>"
                f"<summary><strong>Q.</strong> {escape(_redact(c.question))}</summary>"
                f"<p style='margin:8px 0 0 0;'>"
                f"<strong>A.</strong> {escape(_redact(c.answer))}"
                f"</p></details>"
                f"<div class='muted' style='font-size:0.85em;margin-top:8px;'>"
                f"Grade via "
                f"<code>secondbrain study grade {c.id} 0|3|4|5</code>"
                f"</div></div>",
            )
        body = (
            f"<h1>Flashcard review ({len(due)} due)</h1>"
            + "".join(items_html)
        )
        return HTMLResponse(_layout("Study", body, "study"))

    @app.get("/memory", response_class=HTMLResponse)
    def memory_view():
        """Phase 86 — list cross-conversation memories so the user
        can audit what the chat agent has stashed about them. Each
        row shows kind / key / content / last referenced."""
        from . import memory as memory_mod

        cfg, conn, _, _ = get_read_state()
        mems = memory_mod.list_memories(conn, limit=100)
        if not mems:
            body = (
                "<h1>Chat memories</h1>"
                "<div class='card'><p class='muted'>"
                "No memories yet. The chat agent extracts persistent "
                "facts (preferences, recurring projects, family info) "
                "from your conversations as they happen."
                "</p></div>"
            )
            return HTMLResponse(_layout("Memory", body, "memory"))
        from .safety import redact_text as _redact
        items_html = []
        for m in mems:
            items_html.append(
                f"<div class='stat' style='align-items:flex-start;'>"
                f"<span style='flex:1;'>"
                f"<span class='muted'>[{escape(m.kind)}]</span> "
                f"<strong>{escape(_redact(m.key))}</strong>: "
                f"{escape(_redact(m.content))}"
                f"</span>"
                f"<span class='v muted'>"
                f"{m.reference_count} refs · "
                f"conf {m.confidence:.2f}"
                f"</span></div>",
            )
        body = (
            f"<h1>Chat memories ({len(mems)})</h1>"
            f"<div class='card'>{''.join(items_html)}</div>"
        )
        return HTMLResponse(_layout("Memory", body, "memory"))

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
