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
        ("Search", "/search"),
        ("Graph", "/graph"),
        ("Entities", "/entities"),
        ("Folders", "/folders"),
        ("Briefing", "/briefing"),
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
