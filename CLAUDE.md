# second-brain — Claude Code architecture notes

> **Stop and read me first.** This file orients Claude Code (or any
> agent) onto the codebase before making changes. The README is for
> humans installing it; FEATURES.md is the surface index; this file
> is the architectural map.

## What this is

A personal-knowledge-base CLI + dashboard + MCP server. Single user
(the owner), local SQLite + sqlite-vec. The user runs it on their
laptop with watchpaths pointed at Documents / Downloads / notes
folders + connector accounts (Gmail, Calendar, Oura, Canvas, etc).

## Source layout

```
src/secondbrain/
  __init__.py
  cli.py                    # Typer CLI — every command lives here
  dashboard.py              # FastAPI dashboard — every page + form
  daemon.py                 # Scheduler + watcher orchestration
  mcp_server.py             # MCP tools exposed to Claude / Cursor
  config.py                 # Config dataclass + load + defaults
  db.py                     # Schema + low-level SQLite helpers

  # Core indexing pipeline
  indexer.py                # Folder/URL → chunks → embeddings
  embedder.py               # Voyage + local fallback
  chunker.py                # contextual chunking
  search.py                 # hybrid_search + retrievers
  reranker.py               # cross-encoder
  entities.py               # spaCy NER

  # Connectors
  connectors/
    google_calendar.py
    google_drive.py
    gmail.py
    imap_email.py           # transcript-aware IMAP
    canvas.py
    oura.py
    obsidian.py
    notion.py
    github.py
    ...

  # Knowledge surfaces
  people.py                 # people table + alias matcher + gather_full_context
  connections.py            # round 9-B stale-connection detector
  meeting_prep.py           # round 9-A pre-meeting prep
  meeting_thanks.py         # round 8 post-meeting thank-yous
  email_assist.py           # rounds 6/7/10 — analyze + draft + voice + critique
  tasks.py                  # phase 47 + round 9-C structured promise extractor
  study.py                  # phase 67 flashcards + phase 68 knowledge gaps
  daily_brief.py            # phase 44 morning aggregator (~17 sections now)
  briefing.py               # phase 41 "what's new in the brain" digest
  digest.py                 # phase 7 daily watchlist digest email
  synthesis.py              # phase 73/74/75/72 — clusters + summaries + insights + weekly review
  personal.py               # phase 79/80/81 — habits + goals + journal + projects
  health.py                 # phase 56 — Oura metrics + summarise
  pdf_annotations.py        # phase 84 + 85 — PDF highlights + citation graph
  memory.py                 # phase 86 chat memory + phase 87 snapshots
  reading_queue.py          # phase 40 read-it-later
  watchlist.py              # phase 6/7 watchlist runner
  watcher.py                # filesystem watcher
  scheduler.py              # job registry + tick + run history
  budget.py                 # API spend caps + usage ledger
  safety.py                 # phase 88 redaction (API keys / SSNs / JWTs)
  local_llm.py              # phase 89 Ollama fallback
  transcripts.py            # AI-notetaker doc detection (Granola/Otter/Plaud)
  event_briefing.py         # google calendar / ICS adapters
  sync.py                   # parallel connector sync runner
  setup.py                  # interactive setup wizard

  # Round 10 additions (privacy + observability)
  ai_audit.py               # AI action audit log (every LLM call)
  health_checks.py          # Cross-integration ping + dashboard banner
```

## Where features live

| Feature                          | Module                  |
|----------------------------------|-------------------------|
| Inbox-reply drafts               | `email_assist.py`       |
| Post-meeting thank-yous          | `meeting_thanks.py`     |
| Pre-meeting prep                 | `meeting_prep.py`       |
| Stale-connection suggestions     | `connections.py`        |
| Voice profile + reply pairs      | `email_assist.py` (round 7) |
| AI action audit log              | `ai_audit.py`           |
| Integration health checks        | `health_checks.py`      |
| Privacy redaction (render+prompt) | `safety.py` + `_safe_for_prompt` in `email_assist.py` |
| Local LLM fallback               | `local_llm.py` (used everywhere via `_llm_json_call`) |
| Per-feature budget               | `budget.py`             |

## Key conventions

### LLM calls go through `_llm_json_call` / `_llm_text_call`

Both live in `email_assist.py` and handle:
- Anthropic primary path with `check_budget` + `record_usage`
- Local Ollama fallback when Anthropic isn't usable
- JSON parsing (with markdown-fence stripping)
- Round 10 (#6) audit logging when `conn` is passed

When adding a new LLM-using feature, route it through these helpers
unless you have a specific reason not to. Pass:
- `conn` → enables audit logging
- `audit_kind`, `audit_summary` → human-readable record
- `audit_file_id` / `audit_person_id` → linkage for later queries

### Privacy redaction at TWO boundaries

1. **Prompt-side** (round 10 #4): every raw user content field that
   goes INTO an LLM prompt gets redacted via
   `email_assist._safe_for_prompt(text, max_chars=N)`. This covers
   email bodies, transcripts, sent-mail style samples, brain context.
2. **Render-side** (Phase 88): every chunk/text rendered to
   dashboard / MCP / digest output goes through `safety.redact_text`.

Both apply the same patterns (API keys, JWTs, SSNs, etc). A new
LLM call that sends user data MUST route through `_safe_for_prompt`.

### Schema migrations: lazy `ALTER ADD COLUMN`

The codebase uses `try: ALTER ADD COLUMN ...; except: if 'duplicate
column' in str(e): pass`. Each module's `_ensure_schema()` is called
at every public entrypoint and short-circuits via a WeakSet cache.

When adding a column, follow the round-6 example in
`email_assist._ensure_schema` (search for `metadata_json`). When
adding a table that FKs another module's table, call that module's
`_ensure_schema` first (see `meeting_thanks._ensure_schema`).

### Daemon job registration

`daemon._build_daemon_scheduler` is the canonical job registry —
every periodic task gets a `Scheduler.register(Job(...))` call. As
of round 10 there are ~25 jobs. To add one:

```python
sched.register(Job(
    name="my_feature",
    schedule=IntervalSchedule(seconds=N) | CooldownSchedule(...) | DailyAtSchedule(...),
    fn=lambda cfg, conn: my_feature_run(conn, cfg),
))
```

The lambda's signature is filtered against the context the
scheduler passes (`cfg`, `conn`, `embedder`, `reranker`); only the
kwargs your fn needs get forwarded.

### Dashboard nav

Primary nav is **6 items**, locked in `dashboard._PRIMARY_NAV`.
Adding new pages: put them in `_NAV_GROUPS` (under Personal /
Knowledge / Sources / System) AND in `STATIC_PAGES` in
`PALETTE_JS` so ⌘K can reach them. The launchpad on `/` is updated
inline in `index()`.

Badge counts come from `/api/nav-counts` — JS populates them on
page load. To add a new badge, add the count to that endpoint and
add `<span class="nav-badge" data-badge="key">` in the nav item.

### Tests

`tests/` mirrors `src/secondbrain/`. Key fixtures in `conftest.py`:
- `fresh_db` — temp SQLite with full schema
- `tmp_cfg` — `Config(data_dir=tmp_path)`
- `fake_embedder` — deterministic, no network

LLM-bound tests stub `_llm_json_call` / `_llm_text_call` directly
via `unittest.mock.patch.object`. The legacy `drafter=lambda` test
pattern still works (round 10 #7 routes through the same
`_default_drafter` slot, with a string-to-DraftOutput adapter).

Run: `.venv/Scripts/python.exe -m pytest -x -q -m "not slow"`
should pass with all 1,037 tests in ~17s.

## Recent rounds (chronological)

- **Round 6** — structured email drafter (analyze → retrieve → draft, JSON output)
- **Round 7** — voice fidelity (extracted profile + reply-pair few-shot + critique loop)
- **Round 8** — auto thank-you emails after meetings (`meeting_thanks.py`)
- **Round 9** — meeting prep (A) + stale connections (B) + structured promise extractor (C)
- **Round 10** — fix the 10 audit downfalls:
  - #1 `secondbrain doctor` cross-integration health check
  - #2 draft accept/reject feedback + accept-rate stats
  - #3 per-feature spend transparency (per-draft cost on /drafts)
  - #4 prompt-side privacy redaction
  - #5 cold-start voice fallback (curated default profile)
  - #6 AI audit log (`ai_audit.py`)
  - #7 removed parallel legacy drafter path
  - #8 docs (this file + FEATURES.md)
  - #9 health checks for OAuth / API keys / IMAP / Ollama / watched folders
  - #10 inline edit forms on dashboard (people first)

## When the user asks "what's broken?"

In rough order of likelihood:

1. **Calendar OAuth refresh** — Google revokes tokens for unverified
   personal apps weekly. Run `secondbrain doctor` or check `/health/system`.
2. **`SECONDBRAIN_IMAP_PASSWORD` not in subprocess env** — when a
   service launches `secondbrain serve` (Claude Desktop MCP) it
   doesn't inherit user env vars. Add `env: {...}` to the MCP config.
3. **Daemon not running** — most features only fire on the daemon
   schedule. CLI commands work but stale data accumulates.
4. **Voyage / Anthropic key not in env** — same subprocess issue.

## Shipping changes

The user runs `.venv/Scripts/python.exe -m pytest -x -q -m "not slow"`
+ `.venv/Scripts/python.exe -m ruff check src/ tests/` before every
commit. Both must pass. Keep tests + lint clean as a hard rule.
