# second-brain

A personal, local-first knowledge base that auto-ingests files from your computer and exposes them to **any** AI assistant over MCP.

> Not another note-taking app. The point is to make everything you already have — documents, downloads, screenshots, code, transcripts — searchable by you and by AI assistants you use, without uploading any of it to a vendor.

## Why this exists

Existing "second brain" tools each pick one axis and ignore the others:

|                                | Obsidian | Logseq | mem.ai | Reflect | Notion AI | second-brain |
|--------------------------------|:-:|:-:|:-:|:-:|:-:|:-:|
| Ambient filesystem ingest      | — | — | — | — | — | ✓ |
| AI-agnostic (any LLM)          | partial | partial | — | — | — | ✓ |
| Hybrid retrieval (vec + BM25)  | — | — | ? | ? | ? | ✓ |
| Local-first + optional sync    | ✓ | ✓ | — | — | — | ✓ |
| Plugin architecture            | ✓ | ✓ | — | — | — | (planned) |

## What it does today (Phase 0)

- **Watches folders** you point it at and re-indexes files whose contents change.
- **Extracts text** from PDF, DOCX, PPTX, XLSX, HTML, EPUB, Markdown, code, JSON/YAML/TOML, and more (via `markitdown`).
- **Embeds** chunks with [Voyage AI](https://www.voyageai.com/) by default (best-in-class retrieval quality, ~$0.18/1M tokens) or local `sentence-transformers` if you'd rather not use an API.
- **Hybrid search** (vector + BM25, fused with Reciprocal Rank Fusion) — recall that pure-vector search misses.
- **Skips secrets and binaries** by default (`.env`, `*.key`, `*.pem`, executables, archives, `node_modules/`, `.git/`, etc.).
- **MCP server** with five tools: `search_brain`, `vector_search`, `keyword_search`, `get_file`, `get_recent`, `index_status`. Plug into Claude Desktop, Claude Code, Cursor, Cline, or anything else that speaks MCP.

## Roadmap

- **Phase 0 (current)** — spine: ingest → chunk → embed → hybrid search → MCP.
- **Phase 1** — Whisper transcription for audio/video, OCR for images and screenshots, browser history, clipboard daemon, Slack/Discord/Notion importers.
- **Phase 2** — cross-encoder reranking (Voyage rerank-2), HyDE query rewriting, time-decay scoring.
- **Phase 3** — entity extraction, knowledge graph, graph queries exposed as MCP tools.
- **Phase 4** — Tauri desktop UI: search, browse, daily timeline, graph view.
- **Phase 5** — encrypted multi-device sync.
- **Phase 6** — plugin SDK and reference plugins.

## Install

```bash
git clone https://github.com/benj9/second-brain.git
cd second-brain
python -m venv .venv && source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -e .

# Optional: local embedder fallback (~2 GB of torch + models)
pip install -e .[local]
```

Requires Python 3.11+.

## Quick start

```bash
# 1. Initialize the data directory and config.
secondbrain init

# 2. Set your Voyage key (or skip and use local).
export VOYAGE_API_KEY=...           # macOS/Linux
$env:VOYAGE_API_KEY = "..."         # Windows PowerShell

# 3. Index a folder.
secondbrain index ~/Downloads

# 4. Search from the CLI to confirm it works.
secondbrain search "that pdf about taxes"

# 5. Or run the MCP server and connect from Claude Desktop / Code / Cursor.
secondbrain serve
```

### Watch mode (continuous indexing)

```bash
secondbrain watch ~/Downloads
```

Files added, modified, moved, or deleted are reflected in the index after a short debounce.

### Connect to Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "second-brain": {
      "command": "secondbrain",
      "args": ["serve"]
    }
  }
}
```

Now Claude can call `search_brain`, `get_file`, etc. against your indexed files.

## Configuration

Config lives at the OS-appropriate user data dir (`~/.local/share/secondbrain/config.toml` on Linux, `%LOCALAPPDATA%\secondbrain\config.toml` on Windows). Run `secondbrain init` to create one with sane defaults.

Highlights:
- `embedder_provider` — `"auto"`, `"voyage"`, or `"local"`.
- `chunk_size` / `chunk_overlap` — chunking knobs.
- `extra_ignore_globs` — additional patterns on top of the built-in safety list.
- `hybrid_alpha` — `0.0` for keyword only, `1.0` for vector only, `0.5` balanced.

## Privacy

- Original files **never leave your machine**. Only chunked text snippets are sent to the embedder API (when using Voyage). Switch to the local embedder for fully offline operation.
- The default ignore list excludes `.env` files, SSH keys, certificates, password files, executables, archives, and version-control / dependency directories.
- The index lives at the user data dir as plain SQLite. Delete with `secondbrain reset`.

## License

MIT — see [LICENSE](./LICENSE).
