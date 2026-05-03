# Feature index

Complete catalog of what `secondbrain` does, organized by surface. The
README has the elevator pitch + install; this file is the reference.

For each feature: the CLI command(s), the dashboard page(s), and the
MCP tool(s) that expose it. "—" means not yet surfaced via that path.

---

## Daily-use surfaces

### Brief — your morning at a glance
`secondbrain brief` · dashboard `/brief` · MCP `morning_brief`

Aggregator that pulls everything you need to read once per morning:

- **Health-check warnings** front-loaded (round 10 #9): when a fragile
  integration like Google Calendar OAuth has been failing > 24h, this
  shows up before anything else.
- **Insights** — proactive "I noticed X" patterns (Phase 75).
- **Birthdays this week** (round 9-A: Phase 65 hookup).
- **Today's calendar** events.
- **Upcoming meeting prep** for external meetings (round 9-A) — one-line
  per meeting with the most-stale attendee surfaced.
- **Class assignments due in next 72h** (Canvas).
- **Open action items** with task ids you can `tasks done <id>`.
- **Health snapshot** — Oura sleep / readiness / activity vs 14-day average.
- **Email** — urgent + response counts + pending draft count.
- **Reading queue** top-N with TL;DRs.
- **Watchlist** new-in-last-24h.
- **Possible projects forming** — auto-detected backlink clusters (Phase 73).
- **Knowledge gaps** — questions ask_brain answered weakly (Phase 68).
- **Habits** with streaks + 30d adherence.
- **Goals** with this-week progress.
- **Recently done** — yesterday's wins (last 36h).
- **Highlights from yesterday** — PDF annotations made in last 36h.
- **Worth reaching back out** — stale connections (round 9-B).
- **Nudges** — weekly review due, snapshot due.
- **Worth revisiting** — quiet-day fallback to older archive items.

Markdown render goes to email when `daily_brief_enabled = true`.

### Chat — talk to your brain
`secondbrain ask "..."` · dashboard `/chat` · MCP `ask_brain`

Sonnet 4.6 grounded in your index. Tools: `search_brain`, `web_search`
(opt-in). Phase 86 cross-conversation memory recall — relevant facts
from prior chats prepend to the system prompt automatically. Phase 89
local-LLM fallback when Anthropic budget is exhausted.

### Tasks
`secondbrain tasks (add/list/done/show/search/rm)` · dashboard `/tasks`
· MCP `add_task` · `complete_task` · `list_open_tasks` · `search_tasks`

Phase 47 first-class tasks. Materialized from transcript checkboxes
AND, since round 9-C, from natural-language promises ("I'll send Sarah
the design doc by Friday") via a structured Haiku extractor that also
matches the recipient against the people table + records the due hint.

### Drafts (email)
`secondbrain drafts list` · dashboard `/drafts` · MCP via `list_email_drafts`

Phase 82/83 + round 6/7 voice-aware drafter. For each "urgent" or
"response" classified email, the daemon runs the structured pipeline:

1. **Analyze** (Haiku JSON) — intent / relationship / asks / tone /
   open_questions.
2. **Targeted retrieval** — thread history (Gmail thread_id or IMAP
   subject family), sender history (last 3 emails between you and them),
   relationship-scoped style samples, brain context for mentioned topics.
3. **Voice profile** (round 7) — extracted weekly from your last 50
   sent emails: greetings, sign-offs, sentence length, contraction rate,
   common opener / closer phrases, audited list of generic-LLM
   phrases you don't use.
4. **Few-shot reply pairs** (round 7) — semantic search over indexed
   parent emails finds the 3 most similar incomings; their (incoming,
   user_reply) pairs feed the drafter as concrete examples.
5. **Draft** (Sonnet JSON) — primary + alternative-tone version +
   reasoning + confidence + `<TODO: ...>` placeholders for unknowns.
6. **Voice critique** (Haiku) — compares draft to profile; if it
   flags mismatches (banned phrases, wrong sign-off), regenerate
   once with the critique attached.

Round 10 (#5) — cold-start uses a curated default voice profile so
brand-new users don't get generic-LLM drafts before any Sent mail
is indexed.

Round 10 (#2) — accept/reject feedback tracked. Mark sent → flagged
accepted; discard → flagged rejected. /drafts header shows
"Last 14 days: X accepted, Y rejected — Z% accept rate".

Round 10 (#3) — per-draft cost shown next to the timestamp
("$0.0234 (4 calls)").

### Thanks (post-meeting)
`secondbrain thanks (list/scan/context/skip/draft)` · dashboard `/thanks`

Round 8. Auto-generates thank-you emails after coffee chats /
1:1s / external meetings:

1. Hourly daemon scan picks up calendar events that ended in
   the last 48h with at least one external attendee.
2. Recurring patterns ("standup", "all hands") + extreme durations
   (<10 min, >4h) auto-skipped.
3. Transcript matched by mtime + title-token jaccard ≥ 0.3.
4. With transcript → status `ready` → daemon auto-drafts.
5. Without transcript → status `pending_context`; user provides
   context via `secondbrain thanks context <id> "..."` or the
   /thanks dashboard form.
6. Drafts use the same voice profile + reply-pair few-shot as
   round 7. Thanks-specific prompt emphasizes referencing 1-2
   specific things from the meeting.
7. Drafts land in `email_drafts` so the existing /drafts UI
   handles them. Marking sent flips the meeting_thanks row too.

### Prep (pre-meeting)
`secondbrain prep [show <event-id>]` · dashboard `/prep`

Round 9-A. For each upcoming external meeting (next 24h), builds
brain-grounded prep per attendee:
- People profile (mention count, days since seen, role/company)
- Open tasks involving them
- Topics that come up around them (co-occurring entities)
- Recent docs that mention them

Daemon prefetches every 30 min so reads are instant. Brief
surfaces a one-line digest with the most-stale attendee.

### Insights
`secondbrain insights (list/dismiss <key>)` · dashboard `/insights`
· MCP `list_insights`

Phase 75. Topic spikes in recent docs + Oura health drift. Deduped
for 7 days; `dismiss <key>` extends the dedup window.

### Search
`secondbrain search "..."` · dashboard `/search` · MCP `search_brain`

Hybrid (vector + BM25 + Reciprocal Rank Fusion) + cross-encoder
rerank + adaptive alpha + time decay + path multiplier + click
recency boost. Filters: `--folder`, `--kind`, `--since-days`,
`--as-of "yesterday"|"6 months ago"|"YYYY-MM-DD"`. Auto-summary
TL;DRs render above each hit.

---

## Personal tracking

| Surface  | CLI                                     | Dashboard | MCP                  |
|----------|-----------------------------------------|-----------|----------------------|
| Habits   | `secondbrain habits ...`                | `/habits` | `list_habits`        |
| Goals    | `secondbrain goals ...`                 | —         | `list_goals`         |
| Journal  | `secondbrain journal add "..."`         | `/journal`| `add_journal`        |
| Health   | `secondbrain health ...`                | `/health` | `health_metric` · `health_summary` |
| People   | `secondbrain people (list/show/edit/alias/stale)` | `/people` · `/person?id=` | `find_person` · `person_profile` |
| Memory   | `secondbrain memory (list/recall/forget)` | `/memory` | `recall_memories` · `remember_fact` |

People: round 10 (#10) added inline edit form (email / role /
company / birthday / notes) on `/person?id=`.

---

## Knowledge surfaces

| Surface     | CLI                              | Dashboard          | MCP                   |
|-------------|----------------------------------|--------------------|------------------------|
| Projects    | `secondbrain projects (list/promote)` | `/projects` · `/project?id=` | `list_projects` · `project_overview` |
| Study cards | `secondbrain study (quiz/grade)` | `/study/review`    | `study_status`         |
| Snapshots   | `secondbrain snapshot (take/list/diff)` | `/snapshots` | `list_snapshots` · `as_of_search` |
| Graph       | —                                | `/graph`           | `entity_neighbors` · `entity_timeline` · `find_mentions` |
| Entities    | `secondbrain entities`           | `/entities` · `/entity?name=` | `list_entities` · `find_by_tag` |

---

## Sources & ingestion

| Surface     | CLI                              | Dashboard       |
|-------------|----------------------------------|-----------------|
| Watch       | `secondbrain watch (new/run/list)` | `/watch`      |
| Queue       | `secondbrain read (...)`         | `/queue`        |
| Apps (jobs) | `secondbrain apps (...)`         | `/applications` |
| Briefings   | —                                | `/briefings`    |
| Daily       | `secondbrain briefing`           | `/briefing`     |
| Ingest URL  | `secondbrain ingest <url>`       | `/ingest` · MCP `ingest_url` |
| Sync        | `secondbrain sync (all/<source>)` | —             |

---

## System surfaces

| Surface  | CLI                              | Dashboard          | MCP                  |
|----------|----------------------------------|--------------------|------------------------|
| Overview | `secondbrain status`             | `/`                | `index_status`         |
| Health   | `secondbrain doctor`             | `/health/system`   | —                      |
| Audit    | `secondbrain audit [rollup]`     | `/audit`           | `list_ai_actions`      |
| Folders  | —                                | `/folders`         | `list_folders` · `files_in_folder` |
| Queries  | —                                | `/queries`         | `recent_queries`       |
| Setup    | `secondbrain setup`              | —                  | —                      |

Round 10 (#1) — `secondbrain doctor`: cross-integration health check.
Round 10 (#6) — `/audit`: every AI action logged with cost / status / summary.

---

## Daemon scheduler

`secondbrain daemon` runs all of the below at the listed cadence:

| Job                       | Cadence                  | What it does                                                  |
|---------------------------|--------------------------|---------------------------------------------------------------|
| `wal_checkpoint`          | every 10 min             | Bound the SQLite WAL                                          |
| `trim_scheduler_runs`     | 24h cooldown             | GC scheduler-runs table                                       |
| `watchlists`              | every 1 min              | Run due watchlists (Sonnet + tool use)                        |
| `event_briefings`         | every 1 min              | Pre-meeting "what to know" briefings                          |
| `read_queue_summariser`   | every 1 min              | Summarise reading queue items                                 |
| `daily_digest`            | daily at `digest_send_time` | Watchlist digest email                                     |
| `daily_brief`             | daily at `daily_brief_send_time` | Morning brief email                                   |
| `oura_sync`               | 12h cooldown             | Oura ring sync                                                |
| `study_card_materialiser` | every 30 min             | Generate flashcards for `[course]` docs                       |
| `auto_summariser`         | every 30 min             | Backfill TL;DRs                                               |
| `weekly_review`           | every 1 hour             | Sunday weekly review (Phase 72)                               |
| `todoist_sync`            | every 30 min             | Phase 76 Todoist push/pull                                    |
| `email_triage`            | every 15 min             | Classify recent emails                                        |
| `email_drafts`            | every 1 hour             | Generate reply drafts                                         |
| `index_snapshot`          | every 1 hour             | Weekly index snapshot (Phase 87)                              |
| `people_backfill`         | 6h cooldown              | Promote PERSON entities → people                              |
| `connector_sync`          | 1h cooldown              | Pull from every configured connector                          |
| `tasks_from_transcripts`  | every 30 min             | Regex-extract `- [ ]` tasks from transcripts                  |
| `email_reply_pairs_index` | every 1 hour             | Link Sent → parent for few-shot retrieval (round 7)           |
| `email_voice_profile`     | weekly cooldown          | Re-extract voice profile from sent mail (round 7)             |
| `meeting_thanks`          | every 1 hour             | Detect + draft thank-yous (round 8)                           |
| `meeting_prep_prefetch`   | every 30 min             | Build prep for upcoming meetings (round 9-A)                  |
| `task_promises`           | every 1 hour             | Structured promise extractor (round 9-C)                      |
| `ai_audit_trim`           | 24h cooldown             | Keep 30d of audit log (round 10 #6)                           |
| `health_checks`           | every 1 hour             | Ping all integrations (round 10 #9)                           |

---

## Configuration

Config lives at `~/.secondbrain/config.toml` (path varies by OS via
`platformdirs`). Key sections:

- **Indexing**: `watched_folders`, `chunk_size`, `chunk_overlap`,
  `max_file_bytes`, `extra_ignore_globs`.
- **Embedder**: `embedder_provider` ('auto' / 'voyage' / 'local'),
  `voyage_model`, `local_model`.
- **Search**: `hybrid_alpha`, `adaptive_alpha`, `rerank_enabled`,
  `time_decay_*`, `personal_path_*`, `download_path_*`,
  `click_boost_*`.
- **AI features**: `chat_model`, `briefing_model`, `tag_model`,
  `web_search_enabled`, `hyde_enabled`.
- **Connectors**: `obsidian_vaults`, `substack_feeds`,
  `news_topics`, `imap_*`, `oura_*`, `canvas_*`, `readwise_*`,
  `jobs_*`, etc.
- **Email + brief**: `digest_*`, `daily_brief_*`.
- **Budget**: `daily_budget_cents_voyage`, `daily_budget_cents_anthropic`,
  `feature_budget_cents` (per-feature caps).
- **Local LLM**: `local_llm_host`, `local_llm_model`.

API keys live in env vars (never config), and the round-10
privacy audit redacts every prompt-side data send through
`safety.redact_text` before it leaves the local machine.
