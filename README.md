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

## What it does today

- **Watches folders** you point it at and re-indexes files whose contents change. Run `secondbrain daemon` (headless) or `secondbrain tray` (system tray) for ambient ingestion.
- **Extracts text** from PDF, DOCX, PPTX, XLSX, HTML, EPUB, Markdown, code, JSON/YAML/TOML, and more (via `markitdown`).
- **Transcribes audio and video** (.mp3, .mp4, .m4a, .mov, etc.) locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — voice memos and recordings become searchable text.
- **OCRs images** (.png, .jpg, screenshots, photos of whiteboards) via Tesseract for in-image text.
- **Semantic image search** via Voyage `voyage-multimodal-3` — find images by what they depict ("the diagram of system architecture", "screenshot showing a chart"), not just by OCR'd text.
- **Ingests URLs** with `secondbrain ingest <url>` — handles HTML articles (with proper User-Agent so Wikipedia / news sites don't 403), PDF URLs, and YouTube transcripts.
- **Embeds** chunks with Voyage by default (`voyage-3`, ~$0.18/1M tokens) or local `sentence-transformers` for offline operation.
- **Contextual chunking** — each chunk is embedded with a `Document: …\nSection: …` preamble so the embedder sees document context, not bare fragments.
- **Hybrid search** (vector + BM25, fused with Reciprocal Rank Fusion) plus **cross-encoder reranking** (`voyage-rerank-2-lite`), **query-adaptive alpha** (proper-noun queries lean BM25, prose leans vector), and a **gentle recency boost** (configurable half-life).
- **Search filters**: `search_brain(query, folder=, kind=, since_days=)` for scoped queries.
- **Knowledge graph layer**: spaCy NER pulls people / orgs / places / dates / money per chunk, queryable via `list_entities`, `find_mentions`, `entity_timeline`, and `entity_neighbors` (co-occurrence). Fuzzy matching: "Rowling" finds "J.K. Rowling" and vice versa.
- **Local web dashboard** at http://localhost:8765 (`secondbrain dashboard`) — overview, search, entity browser, folder tree, file viewer, URL ingest form.
- **Skips secrets and binaries** by default (`.env`, `*.key`, `*.pem`, executables, archives, `node_modules/`, `.git/`, etc.).
- **MCP server** with 14 tools for AI assistants: `search_brain`, `vector_search`, `keyword_search`, `image_search`, `get_file`, `get_recent`, `list_folders`, `list_file_types`, `files_in_folder`, `index_status`, `list_entities`, `find_mentions`, `entity_timeline`, `entity_neighbors`, `ingest_url`. Plug into Claude Desktop, Claude Code, Cursor, Cline, or anything else that speaks MCP.

## Roadmap

- **Phase 0** — spine: ingest → chunk → embed → hybrid search → MCP. ✅
- **Phase 1.1** — cross-encoder reranking. ✅
- **Phase 1.2** — contextual chunking. ✅
- **Phase 1.3** — Whisper transcription for audio/video. ✅
- **Phase 1.4** — image OCR via Tesseract. ✅
- **Phase 1.5** — background daemon + system tray icon. ✅
- **Phase 1.6** — smart URL ingest (HTML / PDF / YouTube). ✅
- **Phase 1.7** — semantic image search via voyage-multimodal-3. ✅
- **Phase 2.1** — time-decay scoring. ✅
- **Phase 2.2** — query-adaptive hybrid alpha. ✅
- **Phase 2.3** — discovery MCP tools (browse the brain). ✅
- **Phase 2.4** — search filters (folder / kind / since_days). ✅
- **Phase 3.1** — entity extraction. ✅
- **Phase 3.2** — fuzzy entity matching. ✅
- **Phase 3.3** — entity co-occurrence graph queries. ✅
- **Phase 4** — local web dashboard. ✅
- **Phase 5** — encrypted multi-device sync.
- **Phase 6** — plugin SDK and reference plugins (Notion / Slack / Anki).
- **Phase 7** — HyDE query rewriting; daily briefing (needs LLM key); tests + CI.

## Install

```bash
git clone https://github.com/Ben-K-Jordan/second-brain.git
cd second-brain
python -m venv .venv && source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -e .

# Optional extras (any combination):
pip install -e .[local]      # local embedder fallback (sentence-transformers + torch, ~2 GB)
pip install -e .[whisper]    # audio/video transcription via faster-whisper
pip install -e .[ocr]        # OCR for images via pytesseract (also requires Tesseract binary)
pip install -e .[ner]        # entity extraction via spaCy (also: python -m spacy download en_core_web_sm)
pip install -e .[dashboard]  # local web dashboard (FastAPI + uvicorn)
pip install -e .[tray]       # system tray app (pystray)
```

For higher-quality NER on prose, swap the spaCy model after install:

```bash
python -m spacy download en_core_web_lg     # ~750 MB, noticeably better
# then in ~/.secondbrain/config.toml:
# spacy_model = "en_core_web_lg"
```

Requires Python 3.11+.

### Tesseract install (for `[ocr]`)

The `pytesseract` Python package shells out to a Tesseract binary, which must be installed separately:

```bash
winget install UB-Mannheim.TesseractOCR    # Windows
brew install tesseract                      # macOS
sudo apt install tesseract-ocr              # Linux (Debian/Ubuntu)
```

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

Foreground (one folder, blocks the terminal):

```bash
secondbrain watch ~/Downloads
```

Headless background (recommended — uses `watched_folders` from config):

```bash
secondbrain daemon
```

System tray app (background with status menu):

```bash
pip install -e .[tray]
secondbrain tray
```

Edit `~/.secondbrain/config.toml` (or the OS-appropriate path printed by `secondbrain init`) and add the folders you want watched:

```toml
watched_folders = ["C:/Users/me/Downloads", "C:/Users/me/Documents/notes"]
```

### Autostart on login

Once `secondbrain tray` works manually, autostart it via your OS:

- **Windows:** Win+R → `shell:startup` → drop a shortcut to `secondbrain tray` in the folder. Or use Task Scheduler with the trigger "At log on" pointing at `pythonw.exe -m secondbrain.cli tray`.
- **macOS:** create a LaunchAgent plist under `~/Library/LaunchAgents/`.
- **Linux:** systemd user service or your DE's "Startup Applications".

Files added, modified, moved, or deleted are reflected in the index after a short debounce. New files in your watched folders are searchable from any AI assistant within seconds, no terminal required.

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
