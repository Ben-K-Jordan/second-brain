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

    const STATIC_PAGES = [
        {kind: 'page', icon: '⌂', label: 'Overview',  href: '/',          meta: ''},
        {kind: 'page', icon: '⌕', label: 'Search',    href: '/search',    meta: ''},
        {kind: 'page', icon: '◊', label: 'Graph',     href: '/graph',     meta: ''},
        {kind: 'page', icon: '※', label: 'Entities',  href: '/entities',  meta: ''},
        {kind: 'page', icon: '⊟', label: 'Folders',   href: '/folders',   meta: ''},
        {kind: 'page', icon: '☼', label: 'Briefing',  href: '/briefing',  meta: ''},
        {kind: 'page', icon: '+',  label: 'Ingest URL',href: '/ingest',   meta: ''},
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


def _layout(title: str, body: str, active: str = "") -> str:
    nav_items = [
        ("Overview", "/"),
        ("Brief", "/brief"),
        ("Chat", "/chat"),
        ("Tasks", "/tasks"),
        ("Health", "/health"),
        ("Briefings", "/briefings"),
        ("Queue", "/queue"),
        ("Search", "/search"),
        ("Watch", "/watch"),
        ("Apps", "/applications"),
        ("Graph", "/graph"),
        ("Entities", "/entities"),
        ("Folders", "/folders"),
        ("Daily", "/briefing"),
        ("Queries", "/queries"),
        ("Ingest", "/ingest"),
    ]
    nav = "".join(
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
    <script>{CLICK_BEACON_JS}</script>
</body>
</html>
"""


def _result_block(r, source_label: str = "search") -> str:
    sources = "+".join(r.sources) if r.sources else "rerank"
    age = ""
    if r.mtime is not None:
        days = (time.time() - r.mtime) / 86400
        age = f" · {days:.1f}d ago"
    snippet = escape(r.text if len(r.text) <= 1500 else r.text[:1500] + "…")
    # data-* attributes drive the click-feedback beacon. The JS in CLICK_JS
    # listens for clicks anywhere inside .result and POSTs /api/click with
    # the path/chunk_id; subsequent searches lift recently-opened paths.
    file_link = (
        f'<a href="/file?path={urllib.parse.quote_plus(r.file_path)}" '
        f'data-sb-click="1" data-sb-path="{escape(r.file_path)}" '
        f'data-sb-chunk="{r.chunk_id}" data-sb-source="{escape(source_label)}">'
        f"{escape(r.file_path)}</a>"
    )
    return f"""
<article class="result">
    <h3>{file_link}</h3>
    <div class="meta">chunk {r.chunk_index} · {sources}{age} · score {r.score:.4f}</div>
    <div class="snippet">{snippet}</div>
</article>"""


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

        # Phase 52: surface "see also" backlinks at the bottom so the
        # file view becomes a wayfinding hub, not just a content dump.
        backlinks_html = _render_backlinks_block(conn, path)

        body = f"""
<h1>File <span class="muted">[{kind}]</span></h1>
<p class="path">{escape(path)} {open_link}</p>
<p class="muted">{len(chunks)} chunks · {row["size"] / 1024:.1f} KB · {age:.1f}d ago</p>
{body_chunks}
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
    def chat_delete(cid: int):
        from .db import chat_delete_conversation

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
        # Same-origin guard mirroring /ingest.
        origin = request.headers.get("origin", "")
        referer = request.headers.get("referer", "")
        same_origin_prefixes = ("http://127.0.0.1", "http://localhost")
        if not (
            any(origin.startswith(p) for p in same_origin_prefixes)
            or any(referer.startswith(p) for p in same_origin_prefixes)
        ):
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
        # Same-origin guard.
        origin = request.headers.get("origin", "")
        referer = request.headers.get("referer", "")
        same_origin_prefixes = ("http://127.0.0.1", "http://localhost")
        if not (
            any(origin.startswith(p) for p in same_origin_prefixes)
            or any(referer.startswith(p) for p in same_origin_prefixes)
        ):
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
        # Same-origin guard.
        origin = request.headers.get("origin", "")
        referer = request.headers.get("referer", "")
        same_origin_prefixes = ("http://127.0.0.1", "http://localhost")
        if not (
            any(origin.startswith(p) for p in same_origin_prefixes)
            or any(referer.startswith(p) for p in same_origin_prefixes)
        ):
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
        origin = request.headers.get("origin", "")
        referer = request.headers.get("referer", "")
        same_origin_prefixes = ("http://127.0.0.1", "http://localhost")
        if not (
            any(origin.startswith(p) for p in same_origin_prefixes)
            or any(referer.startswith(p) for p in same_origin_prefixes)
        ):
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
        origin = request.headers.get("origin", "")
        referer = request.headers.get("referer", "")
        same_origin_prefixes = ("http://127.0.0.1", "http://localhost")
        if not (
            any(origin.startswith(p) for p in same_origin_prefixes)
            or any(referer.startswith(p) for p in same_origin_prefixes)
        ):
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
        origin = request.headers.get("origin", "")
        referer = request.headers.get("referer", "")
        same_origin_prefixes = ("http://127.0.0.1", "http://localhost")
        if not (
            any(origin.startswith(p) for p in same_origin_prefixes)
            or any(referer.startswith(p) for p in same_origin_prefixes)
        ):
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
        origin = request.headers.get("origin", "")
        referer = request.headers.get("referer", "")
        same_origin_prefixes = ("http://127.0.0.1", "http://localhost")
        if not (
            any(origin.startswith(p) for p in same_origin_prefixes)
            or any(referer.startswith(p) for p in same_origin_prefixes)
        ):
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
        origin = request.headers.get("origin", "")
        referer = request.headers.get("referer", "")
        same_origin_prefixes = ("http://127.0.0.1", "http://localhost")
        if not (
            any(origin.startswith(p) for p in same_origin_prefixes)
            or any(referer.startswith(p) for p in same_origin_prefixes)
        ):
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
        origin = request.headers.get("origin", "")
        referer = request.headers.get("referer", "")
        same_origin_prefixes = ("http://127.0.0.1", "http://localhost")
        if not (
            any(origin.startswith(p) for p in same_origin_prefixes)
            or any(referer.startswith(p) for p in same_origin_prefixes)
        ):
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
        # CSRF guard: form-POSTs aren't gated by CORS, so any page the user
        # visits could otherwise force-ingest an arbitrary URL by submitting
        # a hidden form to http://127.0.0.1:8765/ingest. Require the Origin
        # (or Referer, when Origin isn't sent) to be the dashboard itself.
        origin = request.headers.get("origin", "")
        referer = request.headers.get("referer", "")
        same_origin_prefixes = ("http://127.0.0.1", "http://localhost")
        same_origin = any(origin.startswith(p) for p in same_origin_prefixes) or any(
            referer.startswith(p) for p in same_origin_prefixes
        )
        if not same_origin:
            return HTMLResponse(
                "<h1>Forbidden</h1><p>Cross-origin POSTs to /ingest are blocked.</p>",
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
            done_btn = (
                f'<form method="post" action="/tasks/{t.id}/done" '
                f'style="display:inline">'
                f'<button class="link-btn" type="submit">✓ done</button>'
                f'</form>'
                if mark_done else ""
            )
            return (
                f'<div class="stat">'
                f'<span><code>#{t.id}</code> {escape(t.text)}{src}</span>'
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
    def tasks_add(text: str = Form(...)):
        from fastapi.responses import RedirectResponse

        from . import tasks as tasks_mod
        cfg, conn, _, _ = get_state()
        tasks_mod.add_manual(conn, text)
        return RedirectResponse(url="/tasks", status_code=303)

    @app.post("/tasks/{task_id:int}/done")
    def tasks_mark_done(task_id: int):
        from fastapi.responses import RedirectResponse

        from . import tasks as tasks_mod
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

    @app.post("/api/click")
    async def api_click(request: Request):
        """Lightweight click beacon. The dashboard front-end POSTs here
        whenever the user opens a search/chat/palette result. We log the
        path so subsequent searches lift it via the click-recency boost.
        """
        from .db import log_click

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

    @app.get("/api/palette")
    def api_palette(q: str = ""):
        """Mixed search for the command palette: entities + files matching q."""
        cfg, conn, _, _ = get_state()
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
