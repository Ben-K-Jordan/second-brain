# second-brain

A personal, local-first knowledge base that auto-ingests files from your computer, watches your email + calendar + browser history, drafts replies in your voice, writes you a weekly letter every Sunday, and exposes everything to **any** AI assistant over MCP.

> Not another note-taking app. The point is to make everything you already have — documents, downloads, screenshots, code, transcripts, emails, journal entries, health metrics — searchable by you and by AI assistants you use, **without uploading any of it to a vendor**.

---

## What's in here

The codebase is one Python package (`secondbrain/`) that ships with:

- **A file watcher daemon** (`secondbrain daemon`) that ambient-indexes everything in your watched folders.
- **A web dashboard** (`secondbrain dashboard` → `http://127.0.0.1:8765`) with 30+ pages: timeline, weekly letter, tasks, drafts, journal, habits, health, people, projects, search, chat.
- **A local LLM chat surface** that runs Anthropic Claude (Sonnet 4.5 by default) grounded in your brain. Multi-turn, persisted, with the same conversation visible across the dashboard, the CLI, and Claude Desktop.
- **An MCP server** (`secondbrain serve`) exposing 50+ tools so Claude Desktop / Claude Code / Cursor can search, summarize, draft, and reason over your knowledge base.
- **A typer-based CLI** with 30+ commands. Lists below.
- **27 data connectors**: Gmail, Google Calendar, Google Drive, IMAP, GitHub, Linear, Notion, Slack, Reddit, Hacker News, Pocket, Substack, RSS, Bluesky, Mastodon, Obsidian vaults, browser history, X/Twitter archive, Oura, Readwise, Canvas LMS, jobs (Greenhouse/Lever/Ashby), Apple iMessage chat.db, and more.
- **Email assistance**: triage incoming mail by urgency, draft replies in your voice (extracted from your sent-mail patterns), generate thank-you notes after meetings.
- **Daily brief + weekly letter**: morning email digest (`secondbrain brief send`); Sunday personal letter from Claude Sonnet that synthesizes the week (`secondbrain review`).
- **Smart notifications**: tray pop-ups for time-sensitive things only — urgent emails, birthdays in 3 days, broken integrations, missed habits.
- **Encrypted backups**: `secondbrain backup --encrypt out.age` (AES-256-GCM, PBKDF2 key derivation).

Local-first by design. The only outbound calls are to Anthropic (chat / drafting / summarization) and Voyage (embeddings), both rate-limited via per-feature daily caps. Nothing else leaves your machine.

---

## Install

Requires **Python 3.11+**.

```bash
git clone https://github.com/Ben-K-Jordan/second-brain.git
cd second-brain
pip install -e .
```

Optional extras (install only what you need):

```bash
pip install -e .[local]      # offline embedder (sentence-transformers)
pip install -e .[whisper]    # audio/video transcription (faster-whisper)
pip install -e .[voice]      # voice capture + transcription
pip install -e .[ocr]        # screenshot/image OCR (pytesseract; install Tesseract separately)
pip install -e .[ner]        # spaCy NER for entity extraction
pip install -e .[dashboard]  # FastAPI + uvicorn for the web dashboard
pip install -e .[tray]       # system tray icon (pystray)
pip install -e .[dev]        # pytest + ruff
```

For OCR, also install Tesseract on your OS:

- **macOS**: `brew install tesseract`
- **Ubuntu/Debian**: `sudo apt install tesseract-ocr`
- **Windows**: download from [UB-Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki).

For tray on Linux you'll also need `apt install libxss1 libappindicator3-1` or equivalent.

---

## Quick start

```bash
# 1. Initialise the data dir + config.
secondbrain init

# 2. Set keys (in your shell rc).
export ANTHROPIC_API_KEY=sk-ant-api03-...
export VOYAGE_API_KEY=pa-...

# 3. Edit ~/.secondbrain/config.toml, set watched_folders to the
#    directories you want indexed. Then:
secondbrain index ~/Documents

# 4. Search.
secondbrain search "what was that paper on attention?"

# 5. Open the dashboard.
secondbrain dashboard       # opens http://127.0.0.1:8765

# 6. Start the daemon (or tray) for ambient ingestion.
secondbrain daemon          # headless
# OR
secondbrain tray            # system tray

# 7. Connect to Claude Desktop / Claude Code / Cursor.
secondbrain serve           # MCP over stdio
```

---

## CLI reference

Run any command with `--help` for full options. Subcommand groups (`tasks`, `habits`, etc.) have their own subcommands; `secondbrain tasks --help` lists them.

### Daily use

| Command | What it does |
|---|---|
| `secondbrain dashboard` | Open the web UI at `http://127.0.0.1:8765`. |
| `secondbrain chat "<msg>"` | One-shot grounded chat (uses Sonnet by default). |
| `secondbrain search "<q>"` | Hybrid search; print top results. |
| `secondbrain ingest <url>` | Pull a URL into the index (HTML / PDF / YouTube transcript). |
| `secondbrain capture` | Voice-capture a note (records, transcribes, indexes). |
| `secondbrain brief` | Show today's morning brief. `secondbrain brief send` emails it. |
| `secondbrain review` | Show this week's weekly letter (Sonnet-written synthesis). `--regenerate`, `--history`, `--stats-only`. |

### Watching + indexing

| Command | What it does |
|---|---|
| `secondbrain daemon` | Headless: file watcher + scheduler (drafts, briefs, syncs). |
| `secondbrain tray` | System tray version with pop-up notifications. |
| `secondbrain watch` | One-shot: file watcher only, no scheduler. |
| `secondbrain index <path>` | One-shot full index of a folder. |
| `secondbrain dedupe` | Find + alias content-hash duplicates. |
| `secondbrain reset` | Wipe the index + start fresh. |

### Personal data

| Command | What it does |
|---|---|
| `secondbrain tasks` | Open / done / overdue tasks; `tasks add`, `tasks done`. |
| `secondbrain habits` | Habits + streaks; `habits add`, `habits checkin`. |
| `secondbrain goals` | Goals + linked progress; `goals add`. |
| `secondbrain journal` | Journal entries; `journal add`, `journal recent`. |
| `secondbrain health` | Oura-fed metrics + trends. |
| `secondbrain people` | Contacts + birthdays + history. |
| `secondbrain memory` | Cross-conversation memory facts (`remember`, `recall`). |

### Email + drafting

| Command | What it does |
|---|---|
| `secondbrain drafts` | List pending email drafts; `drafts sent`, `drafts discard`. |
| `secondbrain thanks` | Pending thank-you notes from meetings. |
| `secondbrain digest send` | Email digest of recent activity. |

### Knowledge synthesis

| Command | What it does |
|---|---|
| `secondbrain insights` | Active "I noticed X" patterns from your data. |
| `secondbrain projects` | Auto-clustered projects from your activity. |
| `secondbrain study` | Spaced-repetition study cards from lectures. |
| `secondbrain prep` | Pre-meeting prep packets from your calendar. |
| `secondbrain gaps` | Knowledge gaps you've encountered but not followed up. |
| `secondbrain snapshot` | Manual conversation memory snapshots. |

### Sources

| Command | What it does |
|---|---|
| `secondbrain sync <source>` | Sync a connector (gmail / notion / etc.). `sync all` for everything. |
| `secondbrain auth` | OAuth flow setup for Google services. |
| `secondbrain watch ...` | Watchlist subcommands (saved searches that re-run). |
| `secondbrain apply` | Job-application tracker subcommands. |

### Operations

| Command | What it does |
|---|---|
| `secondbrain doctor` | Health check: API keys, integrations, watched folders. Exits non-zero on failure. |
| `secondbrain backup <path>` | WAL-safe SQLite backup. Add `--encrypt` for AES-256-GCM. |
| `secondbrain restore <path>` | Restore from backup (auto-detects encrypted). |
| `secondbrain status` | What the daemon's been doing; `--jobs`, `--spend`. |
| `secondbrain spend` | LLM cost rollup (per-provider, per-feature, per-day). |
| `secondbrain audit` | View AI action audit log (every LLM call). |
| `secondbrain serve` | Start the MCP server over stdio. |
| `secondbrain version` | Print version. |

---

## Connect to Claude Desktop

Add this to `~/Library/Application Support/Claude/claude_desktop_config.json` (or the Windows / Linux equivalent):

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

Restart Claude Desktop. You'll see ~50 tools in the MCP panel — `search_brain`, `ask_brain`, `morning_brief`, `weekly_review`, `add_task`, `complete_task`, `list_email_drafts`, `recall_memories`, `daily_briefing`, `find_person`, `entity_timeline`, `project_overview`, `health_summary`, etc. The full list is documented at the top of `src/secondbrain/mcp_server.py`.

---

## The weekly letter

Every Sunday morning, the daemon assembles a structured snapshot of your week — counts, tasks, journal, habits, health, top entities, insights, meetings, drafts, knowledge gaps — and asks Claude Sonnet 4.5 to write you a personal letter. Not a stats dump: an actual short narrative that:

- References specific events (cites numbers, names, dates).
- Notices patterns ACROSS signals ("your journal mentions 'tired' on the same days your sleep dropped").
- Surfaces open threads (lingering tasks, unanswered journal questions).
- Calls back to the prior week's letter for continuity.
- Ends with ONE specific actionable suggestion.

Read it on the dashboard at `/review`, in your terminal with `secondbrain review`, or have your AI assistant pull it via the `weekly_review` MCP tool. ~$0.05/week against your Anthropic budget.

If Anthropic is unavailable (key removed, budget exhausted), the daemon falls back to a stats-only review so you always have something Sunday morning.

---

## Smart notifications

The daemon detects time-sensitive things hourly and queues notifications. The tray pops up the high/medium ones; the dashboard `/notifications` page shows everything pending.

Categories:
- **email_urgent** — a triaged-as-urgent email arrived
- **birthday** — a contact's birthday is in the next 3 days
- **journal_nudge** — you haven't journaled in 3+ days
- **stale_health** — a health check has been failing 24h+
- **review_ready** — a new weekly letter is in
- **draft_pending** — 5+ unreviewed email drafts piling up
- **task_overdue** — a task with a due date passed

Notifications are idempotent (UNIQUE on key) so they won't re-spam.

---

## Encrypted backups

```bash
# Export the passphrase to skip the prompt:
export SECONDBRAIN_BACKUP_PASSPHRASE='correct horse battery staple'

# Encrypted backup:
secondbrain backup ~/Backups/sb-2025-04.db.age --encrypt

# Restore (auto-detects encrypted):
secondbrain restore ~/Backups/sb-2025-04.db.age
```

Format: AES-256-GCM with PBKDF2-HMAC-SHA256 key derivation (600,000 iterations). Tampering or wrong passphrase fails MAC verification cleanly. SQLite `.backup()` under the hood so it's WAL-aware (the daemon can keep running during the backup).

---

## Configuration

Config lives at `~/.secondbrain/config.toml` (or the platform equivalent — `secondbrain init` will tell you). Key settings:

```toml
# Where to look.
watched_folders = ["/Users/you/Documents", "/Users/you/Notes"]
extra_ignore_globs = ["*.bak", "*/backup_*"]

# Embedder.
embedder_provider = "voyage"   # or "local" for sentence-transformers
embedder_model = "voyage-3"

# Chat.
chat_model = "claude-sonnet-4-5"
chat_max_tool_iterations = 5
web_search_enabled = true

# Daily budgets (cents per day, per provider).
daily_budget_cents_voyage = 500
daily_budget_cents_anthropic = 500

# Per-feature caps within the provider budget.
[feature_budget_cents]
chat = 200
briefing = 50
weekly_review = 20
embed = 200

# Brief / digest email.
daily_brief_enabled = true
daily_brief_send_time = "07:30"
digest_smtp_host = "smtp.gmail.com"
digest_smtp_port = 587
digest_smtp_user = "you@example.com"
digest_to = "you@example.com"
# Set $SECONDBRAIN_SMTP_PASSWORD in your shell env.

# iMessage (macOS path; on Windows copy chat.db to a known location).
imessage_db_path = "~/Library/Messages/chat.db"

# Obsidian vaults to ingest.
obsidian_vaults = ["/Users/you/Vault"]
```

---

## Privacy

By default:

- **Dashboard binds 127.0.0.1** only.
- **MCP server is stdio-only** (not network-bound).
- **Drafts never auto-send.** Only `digest send` and `daily_brief` send mail, gated on explicit env var + config.
- **Secret-shaped substrings** (API keys, SSNs, credit cards) are masked from every prompt that goes outbound. See `safety.py` and `_safe_for_prompt`.
- **API keys** stay in env — never written to DB or logs.
- **No backup/sync code** — nothing uploads to S3/Dropbox/GitHub. The encrypted `backup` command writes to a local path.
- **AI audit log** stores only char counts + 80-char redacted summaries — never full prompts or responses.
- **CSRF guards** on every state-mutating dashboard POST (same-origin Origin/Referer check).
- **SSRF guards** on URL ingestion: rejects file://, RFC1918, loopback, link-local, cloud-metadata IPs, and re-validates redirect hops.
- **Every LLM call passes through a budget gate.** Per-provider cap + per-feature cap. Fails closed.

---

## Architecture

```
                      ┌──────────────────────────┐
                      │  ~/.secondbrain/         │
                      │     index.db (SQLite)    │
                      │     config.toml          │
                      │     daemon.log           │
                      │     spend.jsonl          │
                      └──────────┬───────────────┘
                                 │
        ┌────────────────────────┼─────────────────────────┐
        │                        │                         │
   ┌────▼─────┐          ┌───────▼──────┐         ┌────────▼────────┐
   │   CLI    │          │   Daemon     │         │  Dashboard      │
   │ (typer)  │          │ (watchdog +  │         │  (FastAPI on    │
   │          │          │  scheduler)  │         │   127.0.0.1)    │
   └────┬─────┘          └───────┬──────┘         └────────┬────────┘
        │                        │                         │
        │                        ▼                         │
        │             ┌─────────────────────┐              │
        │             │     Indexer         │              │
        │             │  • text extraction  │              │
        │             │  • OCR / Whisper    │              │
        │             │  • chunking         │              │
        │             │  • embeddings       │              │
        │             │  • entity NER       │              │
        │             │  • link detection   │              │
        │             └──────────┬──────────┘              │
        │                        │                         │
        ▼                        ▼                         ▼
   ┌──────────┐         ┌──────────────┐         ┌─────────────────┐
   │  MCP     │         │  Connectors  │         │  Search +       │
   │  Server  │         │ Gmail, Cal,  │         │  Chat (Claude)  │
   │ (stdio)  │         │ Notion, ...  │         │                 │
   └──────────┘         └──────────────┘         └─────────────────┘
```

SQLite (with `sqlite-vec` for vector search and FTS5 for keyword search) is the only persistence. WAL mode for concurrency. The whole DB is a single file you can back up.

---

## Roadmap

This was originally Phase 0 → 7. The codebase is at Phase 80+ now and that numbering became unwieldy; it's evolved into "round-based audit-driven hardening". The most recent rounds:

- **Round 10-12**: feature work on email assistance, meeting thanks, voice profiles, audit log.
- **Round 13-14**: security hardening (CSRF sweep, SSRF guard, API-key fingerprinting, prompt-side redaction across all LLM call sites).
- **Round 15**: reliability + cost (per-feature budget caps, atomic indexer transactions, daemon error recovery, backup command).
- **Round 16** (this one): proactive weekly letter, smart notifications, iMessage ingester, encrypted backups, MCP chat-conversation continuity, unified timeline view.

What's next is open. If you have ideas, file an issue.

---

## Contributing / development

```bash
pip install -e .[dev,dashboard,local,whisper,ocr]
pytest -q                    # ~1,200 tests, ~30s
ruff check src tests         # lint
```

The codebase is heavily documented with inline rationale comments. Look for `# Round NN ...` markers — they trace the audit-driven evolution.

---

## License

MIT. See `LICENSE`.
