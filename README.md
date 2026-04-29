# second-brain

A personal knowledge base that auto-ingests files from your computer and exposes them to AI assistants over MCP.

## What it does

1. **Watches folders** you point it at (e.g. `~/Documents`, `~/Downloads`) and reacts to file changes.
2. **Indexes** each file — extracts text, chunks it, embeds it, stores it in a local SQLite database.
3. **Exposes an MCP server** so any MCP-capable AI client (Claude Desktop, Claude Code, Cursor, etc.) can search your knowledge base.

Original files stay where they are. The index lives in `~/.secondbrain/`.

## Status

Early scaffolding. Nothing works yet.

## Planned stack

- Python 3.11+
- [`watchdog`](https://pypi.org/project/watchdog/) — filesystem watching
- [`markitdown`](https://github.com/microsoft/markitdown) — text extraction from PDF / DOCX / PPTX / HTML / images
- [`sentence-transformers`](https://www.sbert.net/) — local embeddings (no API cost, works offline)
- [`sqlite-vec`](https://github.com/asg017/sqlite-vec) — vector search inside plain SQLite
- [`mcp`](https://github.com/modelcontextprotocol/python-sdk) — Anthropic's MCP Python SDK
- [`typer`](https://typer.tiangolo.com/) — CLI

## Roadmap

- [ ] MVP: watch one folder, index `.md` / `.txt` / `.pdf`, expose `search_brain(query)` MCP tool
- [ ] Broader file-type coverage (DOCX, PPTX, HTML, code files)
- [ ] Incremental re-indexing on file changes (not full rebuild)
- [ ] Configurable embedder (local ↔ API)
- [ ] Multi-folder watch
- [ ] Web UI for browsing / search
- [ ] Multi-device sync
