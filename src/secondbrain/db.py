"""SQLite storage with sqlite-vec for vectors and FTS5 for keywords."""

from __future__ import annotations

import sqlite3
import struct
import time
from collections.abc import Iterable
from pathlib import Path

import sqlite_vec


def serialize_f32(vec: Iterable[float]) -> bytes:
    """Pack a vector of floats as little-endian float32 bytes (sqlite-vec format)."""
    vec = list(vec)
    return struct.pack(f"<{len(vec)}f", *vec)


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a read/write connection with sqlite-vec loaded and sensible pragmas.

    ``check_same_thread=False`` is required because the daemon hands one
    connection to a watchdog worker thread, which then calls into the indexer
    on file events. Caller is responsible for not running concurrent writers
    *within* a process; busy_timeout already serializes across processes.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    # Wait up to 30s for a lock before failing. The indexer's per-file write
    # transactions on big spreadsheets (1500+ chunks + entities + vectors)
    # can run for several seconds end-to-end; 5s left status queries failing.
    conn.execute("PRAGMA busy_timeout = 30000")
    # Auto-checkpoint after every ~1000 pages of WAL (4MB at default page size).
    # Without this, a long-running daemon with an active reader connection can
    # let the WAL grow unbounded - we previously hit a 893MB -wal file. This
    # is a passive checkpoint; it doesn't block writers.
    conn.execute("PRAGMA wal_autocheckpoint = 1000")
    return conn


def checkpoint_wal(conn: sqlite3.Connection) -> None:
    """Force a TRUNCATE checkpoint, shrinking the -wal file to zero.

    Called periodically by the daemon and on clean shutdown. Safe to call from
    any thread that holds the connection. If a reader holds an old snapshot,
    the checkpoint may only partially complete - that's fine, it'll catch up.
    """
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.OperationalError:
        # Another writer holds the lock; we'll get the next opportunity.
        pass


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Open a read-only connection. Does not contend for the write lock - used
    by `status`, `spend`, and search paths so they coexist cleanly with a busy
    daemon. Skips init_schema; the caller is asserting the DB is already set up.
    """
    if not db_path.exists():
        raise FileNotFoundError(
            f"No index at {db_path}. Run `secondbrain index <folder>` first."
        )
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA query_only = 1")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def init_schema(conn: sqlite3.Connection, embedding_dim: int, embedder_name: str) -> None:
    """Create tables if they don't exist. Verifies embedder compatibility on re-init."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE NOT NULL,
            mtime REAL NOT NULL,
            size INTEGER NOT NULL,
            kind TEXT NOT NULL,
            indexed_at REAL NOT NULL,
            content_hash TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_files_mtime ON files(mtime DESC);

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            start_offset INTEGER,
            UNIQUE(file_id, chunk_index)
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);

        -- Hash-based dedup: when the same content lives at multiple paths
        -- (e.g. Downloads + OneDrive copies), only one row in `files` carries
        -- the embeddings; the other paths land here as aliases. Saves
        -- embedding cost and keeps search results de-duplicated.
        CREATE TABLE IF NOT EXISTS file_aliases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            path TEXT UNIQUE NOT NULL,
            discovered_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_file_aliases_file_id ON file_aliases(file_id);

        CREATE INDEX IF NOT EXISTS idx_files_content_hash ON files(content_hash);

        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
            text TEXT NOT NULL,
            text_lower TEXT NOT NULL,
            label TEXT NOT NULL,
            UNIQUE(chunk_id, text_lower, label)
        );
        CREATE INDEX IF NOT EXISTS idx_entities_chunk_id ON entities(chunk_id);
        CREATE INDEX IF NOT EXISTS idx_entities_text_lower ON entities(text_lower);
        CREATE INDEX IF NOT EXISTS idx_entities_label ON entities(label);

        -- LLM-generated topic tags. Populated by `secondbrain tag` (opt-in).
        -- Tags are stored lowercased; queries match case-insensitively.
        CREATE TABLE IF NOT EXISTS chunk_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
            tag TEXT NOT NULL,
            UNIQUE(chunk_id, tag)
        );
        CREATE INDEX IF NOT EXISTS idx_chunk_tags_tag ON chunk_tags(tag);
        CREATE INDEX IF NOT EXISTS idx_chunk_tags_chunk_id ON chunk_tags(chunk_id);

        CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
            text,
            content='chunks',
            content_rowid='id',
            tokenize='porter unicode61'
        );

        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO fts_chunks(rowid, text) VALUES (new.id, new.text);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO fts_chunks(fts_chunks, rowid, text) VALUES ('delete', old.id, old.text);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO fts_chunks(fts_chunks, rowid, text) VALUES ('delete', old.id, old.text);
            INSERT INTO fts_chunks(rowid, text) VALUES (new.id, new.text);
        END;
    """)

    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
        f"chunk_id INTEGER PRIMARY KEY, embedding FLOAT[{embedding_dim}])"
    )

    # Image embedding side-table. Independent dimension because the multimodal
    # model is separate from the text embedder; default voyage-multimodal-3
    # is 1024-dim. The dim is captured per-row at insert time via the vec0
    # virtual-table syntax we already use; we just need a stable schema here.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS images ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,"
        "  embedder TEXT NOT NULL,"
        "  embedding_dim INTEGER NOT NULL,"
        "  indexed_at REAL NOT NULL,"
        "  UNIQUE(file_id, embedder)"
        ")"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_images_file_id ON images(file_id)")
    # Default to 1024 (voyage-multimodal-3). If a user later switches to a
    # different-dim multimodal model, we'll need a `secondbrain reset --images`
    # equivalent; for now this matches all known multimodal options.
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_images USING vec0("
        "image_id INTEGER PRIMARY KEY, embedding FLOAT[1024])"
    )

    # Migration: add chunks.start_offset on databases created before that
    # column existed. SQLite's CREATE TABLE IF NOT EXISTS won't add it to a
    # pre-existing table; we add it via ALTER if absent.
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(chunks)")}
    if "start_offset" not in cols:
        conn.execute("ALTER TABLE chunks ADD COLUMN start_offset INTEGER")

    # Migration: chat_conversations.system_prompt was added in Phase 24.
    # Older DBs created with Phase 18 still need the column.
    chat_cols = {row["name"] for row in conn.execute("PRAGMA table_info(chat_conversations)")}
    if chat_cols and "system_prompt" not in chat_cols:
        conn.execute("ALTER TABLE chat_conversations ADD COLUMN system_prompt TEXT")

    # Migration: watchlists.allowed_domains_json was added in Phase 27.
    wl_cols = {row["name"] for row in conn.execute("PRAGMA table_info(watchlists)")}
    if wl_cols and "allowed_domains_json" not in wl_cols:
        conn.execute("ALTER TABLE watchlists ADD COLUMN allowed_domains_json TEXT")

    # Migration: watchlist_runs.new_paths_json + new_count added in Phase 30.
    run_cols = {row["name"] for row in conn.execute("PRAGMA table_info(watchlist_runs)")}
    if run_cols and "new_paths_json" not in run_cols:
        conn.execute("ALTER TABLE watchlist_runs ADD COLUMN new_paths_json TEXT")
    if run_cols and "new_count" not in run_cols:
        conn.execute(
            "ALTER TABLE watchlist_runs ADD COLUMN new_count INTEGER NOT NULL DEFAULT 0"
        )

    # Chat conversations: each conversation has many turns, each turn has a
    # role (user / assistant) and a content blob. Citations are stored per
    # assistant turn so we can re-render past chats with their sources.
    # An optional system_prompt overrides the default chat persona for this
    # one conversation ("you are my code reviewer", etc.).
    # Kept in the same DB so a `secondbrain reset` clears chat history too.
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chat_conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            system_prompt TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_chat_conv_updated
            ON chat_conversations(updated_at DESC);

        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL
                REFERENCES chat_conversations(id) ON DELETE CASCADE,
            seq INTEGER NOT NULL,
            role TEXT NOT NULL,            -- 'user' | 'assistant'
            content_json TEXT NOT NULL,    -- JSON string ('text' for user; blocks list for assistant)
            citations_json TEXT,           -- JSON list of citation dicts (assistant only)
            created_at REAL NOT NULL,
            UNIQUE(conversation_id, seq)
        );
        CREATE INDEX IF NOT EXISTS idx_chat_msgs_conv
            ON chat_messages(conversation_id, seq);

        -- Watchlists: saved recurring queries. The daemon runs each
        -- watchlist on its schedule, captures the synthesized answer +
        -- citations, and stores the run in watchlist_runs. The dashboard's
        -- /watch page surfaces what's new since the last run.
        CREATE TABLE IF NOT EXISTS watchlists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            query TEXT NOT NULL,
            schedule_minutes INTEGER NOT NULL DEFAULT 1440,  -- daily
            last_run_at REAL,
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at REAL NOT NULL,
            -- JSON list of host strings; when set, this watchlist's web_search
            -- is restricted to those domains. Overrides cfg.web_search_allowed_domains.
            allowed_domains_json TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_watchlists_enabled ON watchlists(enabled);

        CREATE TABLE IF NOT EXISTS watchlist_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            watchlist_id INTEGER NOT NULL
                REFERENCES watchlists(id) ON DELETE CASCADE,
            started_at REAL NOT NULL,
            finished_at REAL,
            answer TEXT,           -- synthesized text from ask_brain
            citations_json TEXT,   -- JSON list of citation dicts
            error TEXT,            -- non-null when the run failed
            cents_spent REAL,      -- estimated cost of this run
            -- Diff bookkeeping (Phase 30): URLs/paths in this run that
            -- weren't in the previous run, and the count of same. Used by
            -- the dashboard's "what's new" highlight + tray notifications +
            -- email digests.
            new_paths_json TEXT,
            new_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_wlruns_wl_started
            ON watchlist_runs(watchlist_id, started_at DESC);

        -- Reading queue: high-value items pulled from watchlist runs and
        -- (optionally) news syncs, with a 60-second pre-summary so you
        -- can decide on the train whether to read the full thing.
        -- UNIQUE on url so the same posting/article doesn't queue twice.
        CREATE TABLE IF NOT EXISTS reading_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL UNIQUE,
            title TEXT,
            source TEXT NOT NULL,           -- 'watchlist:<id>' | 'news' | 'manual'
            added_at REAL NOT NULL,
            summary TEXT,                   -- null until summariser runs
            summary_generated_at REAL,
            summary_error TEXT,
            read_at REAL,
            skipped_at REAL,
            -- Optional context inherited from the source — for watchlist
            -- items we copy fit_label / fit_score so the queue UI can
            -- show "great fit" without re-running the embedder.
            fit_label TEXT,
            fit_score REAL
        );
        CREATE INDEX IF NOT EXISTS idx_queue_unread
            ON reading_queue(read_at, skipped_at, added_at DESC);

        -- Pre-event briefings: one row per event we've ever briefed for.
        -- Triggered by the daemon when an event is starting within the
        -- configured lookahead window. UNIQUE on (event_id, event_source)
        -- so we don't re-generate for the same event unless the user
        -- explicitly asks (the dashboard has a regenerate button).
        CREATE TABLE IF NOT EXISTS event_briefings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL,
            event_source TEXT NOT NULL,    -- 'google_calendar' | 'ics' | 'manual'
            event_starts_at REAL NOT NULL,
            event_title TEXT,
            event_url TEXT,                -- click-through to the actual calendar event
            event_payload_json TEXT,       -- raw event for re-rendering / debugging
            generated_at REAL NOT NULL,
            briefing_text TEXT,
            citations_json TEXT,
            error TEXT,
            cents_spent REAL,
            UNIQUE(event_id, event_source)
        );
        CREATE INDEX IF NOT EXISTS idx_briefings_starts
            ON event_briefings(event_starts_at);

        -- Application tracker: jobs you've actually applied to. Lets the
        -- chat agent answer "have I already applied to X?" and the
        -- watchlist agent skip duplicates. role_url is the canonical
        -- identity (matches a posting's url / virtual_path so the
        -- existing brain can hop from posting → application status).
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT NOT NULL,
            role_title TEXT NOT NULL,
            role_url TEXT,                  -- canonical posting URL when known
            applied_at REAL NOT NULL,       -- epoch seconds
            status TEXT NOT NULL DEFAULT 'applied',  -- 'applied'|'screen'|'interview'|'offer'|'rejected'|'withdrawn'|'ghosted'
            source TEXT,                    -- e.g. 'linkedin', 'greenhouse:anthropic', 'referral'
            notes TEXT,
            updated_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_apps_role_url ON applications(role_url);
        CREATE INDEX IF NOT EXISTS idx_apps_status ON applications(status);
        CREATE INDEX IF NOT EXISTS idx_apps_applied_at ON applications(applied_at DESC);

        -- Click-feedback: records which result paths the user actually
        -- opened from /search, /chat, /entity, etc. Used as a passive
        -- recency-weighted boost in ranking.
        CREATE TABLE IF NOT EXISTS click_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL,
            chunk_id INTEGER,              -- nullable: clicks from non-search paths
            source TEXT NOT NULL,          -- 'search' | 'chat' | 'palette' | 'entity'
            ts REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_click_path_ts ON click_log(path, ts DESC);
        CREATE INDEX IF NOT EXISTS idx_click_ts ON click_log(ts DESC);

        -- People (Phase 65): de-duped, profile-shaped view of the
        -- entities table. ``entities`` is per-chunk noisy (every spaCy
        -- mention); this is one row per *resolved* person with their
        -- mention history, contact info, and relationship metadata.
        --
        -- The canonical_name is the lowercase form that anchors
        -- de-duplication: 'sarah chen' / 'Sarah Chen' / 'S Chen' all
        -- map here. Aliases live in person_aliases below.
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_name TEXT NOT NULL,           -- lowercase, dedup key
            display_name TEXT NOT NULL,             -- preferred capitalisation
            email TEXT,                             -- primary email when known
            company TEXT,
            role TEXT,                              -- "PM", "Professor", etc.
            notes TEXT,                             -- user-edited free text
            birthday TEXT,                          -- 'MM-DD' or 'YYYY-MM-DD'
            first_seen_at REAL NOT NULL,
            last_seen_at REAL NOT NULL,
            mention_count INTEGER NOT NULL DEFAULT 0,
            UNIQUE(canonical_name)
        );
        CREATE INDEX IF NOT EXISTS idx_people_email ON people(email);
        CREATE INDEX IF NOT EXISTS idx_people_last_seen
            ON people(last_seen_at DESC);

        -- Aliases for a person — same human, multiple text forms.
        -- Resolves "S. Chen" to person id N when "S. Chen" appears in
        -- a doc. UNIQUE on alias_lower so "Sarah" can only map to one
        -- person at a time (last-writer-wins through user re-tagging).
        CREATE TABLE IF NOT EXISTS person_aliases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
            alias TEXT NOT NULL,                    -- preserve casing
            alias_lower TEXT NOT NULL UNIQUE,
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_person_aliases_person
            ON person_aliases(person_id);

        -- Per-mention link from a chunk to a person (Phase 66 — auto-
        -- link entity mentions). One row per (chunk, person) pair so
        -- we can render a doc with people highlighted + answer "what
        -- docs mention Sarah?" in O(log n).
        CREATE TABLE IF NOT EXISTS person_mentions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
            chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
            file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            mtime REAL NOT NULL,
            UNIQUE(person_id, chunk_id)
        );
        CREATE INDEX IF NOT EXISTS idx_person_mentions_person_mtime
            ON person_mentions(person_id, mtime DESC);
        CREATE INDEX IF NOT EXISTS idx_person_mentions_file
            ON person_mentions(file_id);

        -- Study cards (Phase 67): flashcards generated from class
        -- transcripts. One row per (course_doc, question). Cards are
        -- materialised lazily by the LLM at first study or by the
        -- daemon's background generator.
        --
        -- Tracks SM-2-style spaced repetition state: ease, interval,
        -- next_due. The CLI quiz updates this in-place. Cards from a
        -- doc that no longer exists CASCADE on file delete.
        CREATE TABLE IF NOT EXISTS study_cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            course_code TEXT NOT NULL,                 -- 'BME410' / '' for non-class
            concept TEXT NOT NULL,                     -- short topic tag
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
            -- SM-2 state.
            ease REAL NOT NULL DEFAULT 2.5,
            interval_days REAL NOT NULL DEFAULT 0,
            next_due_at REAL NOT NULL,                 -- when to surface next
            last_reviewed_at REAL,
            review_count INTEGER NOT NULL DEFAULT 0,
            correct_count INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            UNIQUE(file_id, question)
        );
        CREATE INDEX IF NOT EXISTS idx_study_due
            ON study_cards(course_code, next_due_at);
        CREATE INDEX IF NOT EXISTS idx_study_file
            ON study_cards(file_id);
        CREATE INDEX IF NOT EXISTS idx_study_concept
            ON study_cards(course_code, concept);

        -- Knowledge-gap log (Phase 68): when ask_brain returns weak
        -- results (low retrieval score / 'I don't know enough'), log
        -- the question. Weekly review surfaces top-N gaps as study
        -- targets.
        CREATE TABLE IF NOT EXISTS knowledge_gaps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            asked_at REAL NOT NULL,
            top_score REAL,                            -- best retrieval score
            n_results INTEGER NOT NULL DEFAULT 0,
            resolved_at REAL,                          -- user marked it done
            note TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_gaps_unresolved
            ON knowledge_gaps(resolved_at, asked_at DESC);

        -- Health metrics (Phase 56): structured numeric values from
        -- the Oura connector (and future Apple Health / Garmin etc).
        -- Stored separately from doc bodies so trend / correlation
        -- queries don't need to re-parse Markdown.
        --
        -- UNIQUE on (date, metric, source) so re-syncing the connector
        -- updates rather than duplicates rows.
        CREATE TABLE IF NOT EXISTS health_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,           -- 'YYYY-MM-DD'
            metric TEXT NOT NULL,         -- 'sleep_score' | 'steps' | etc.
            value REAL NOT NULL,
            source TEXT NOT NULL,         -- 'oura' | future: 'apple_health'
            recorded_at REAL NOT NULL,
            UNIQUE(date, metric, source)
        );
        CREATE INDEX IF NOT EXISTS idx_health_date_metric
            ON health_metrics(date DESC, metric);
        CREATE INDEX IF NOT EXISTS idx_health_metric_date
            ON health_metrics(metric, date DESC);

        -- Backlinks (Phase 52): pairs of files that are semantically
        -- similar enough to surface as "see also" context. Computed on
        -- ingest by averaging the new doc's chunk embeddings, querying
        -- vec_chunks for nearest neighbours, and aggregating to file
        -- granularity.
        --
        -- Storage policy: bidirectional pairs (one row per direction).
        -- Means double the rows but trivial query (WHERE src = ?). Bounded
        -- by O(K × num_files) — at K=5 and 10k files that's 50k rows, fine.
        -- ``score`` is sqlite-vec's distance (LOWER is more similar) so we
        -- can ORDER BY score ASC without conversion.
        CREATE TABLE IF NOT EXISTS backlinks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            src_file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            dst_file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
            score REAL NOT NULL,
            created_at REAL NOT NULL,
            UNIQUE(src_file_id, dst_file_id)
        );
        CREATE INDEX IF NOT EXISTS idx_backlinks_src
            ON backlinks(src_file_id, score ASC);

        -- Tasks (Phase 47): first-class action items. Materialised lazily
        -- from transcript-shaped docs (Granola action items, generic
        -- meeting `- [ ]` checkboxes) and from manual `tasks add` calls.
        --
        -- UNIQUE on (text_lower, source_path) so the same item can't
        -- double-extract on every brief render. ``source_path`` is the
        -- transcript:// virtual path (or 'manual' for ad-hoc tasks).
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            text_lower TEXT NOT NULL,
            source_path TEXT NOT NULL,        -- transcript:// path or 'manual'
            source_title TEXT,                -- doc title for back-reference UX
            status TEXT NOT NULL DEFAULT 'open',  -- 'open' | 'done' | 'cancelled'
            created_at REAL NOT NULL,
            completed_at REAL,
            due_at REAL,                      -- nullable; future use
            -- External-sync hooks (Phase 47.x). Empty string when not
            -- synced; provider names: 'apple_reminders' | 'todoist' | ''.
            external_id TEXT NOT NULL DEFAULT '',
            external_provider TEXT NOT NULL DEFAULT '',
            UNIQUE(text_lower, source_path)
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_status_created
            ON tasks(status, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_tasks_source_path ON tasks(source_path);
    """)

    existing_dim = get_meta(conn, "embedding_dim")
    existing_embedder = get_meta(conn, "embedder_name")
    if existing_dim and int(existing_dim) != embedding_dim:
        raise RuntimeError(
            f"Index was built with embedding dim {existing_dim} (embedder "
            f"'{existing_embedder}'), but configured embedder '{embedder_name}' "
            f"uses dim {embedding_dim}. Rebuild with: secondbrain reset"
        )
    set_meta(conn, "embedding_dim", str(embedding_dim))
    set_meta(conn, "embedder_name", embedder_name)
    conn.commit()


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def upsert_file(
    conn: sqlite3.Connection,
    path: str,
    mtime: float,
    size: int,
    kind: str,
    content_hash: str | None = None,
) -> int:
    """Insert or update a file row, returning its id."""
    cur = conn.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at, content_hash) "
        "VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(path) DO UPDATE SET "
        "  mtime = excluded.mtime, "
        "  size = excluded.size, "
        "  kind = excluded.kind, "
        "  indexed_at = excluded.indexed_at, "
        "  content_hash = excluded.content_hash "
        "RETURNING id",
        (path, mtime, size, kind, time.time(), content_hash),
    )
    row = cur.fetchone()
    return row["id"]


def get_file_by_path(conn: sqlite3.Connection, path: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM files WHERE path = ?", (path,)).fetchone()


def find_file_by_hash(conn: sqlite3.Connection, content_hash: str) -> sqlite3.Row | None:
    """Find an existing primary file with the given content hash.

    Used to detect cross-path duplicates so we register them as aliases instead
    of embedding the same content twice. Returns the canonical file row.
    """
    return conn.execute(
        "SELECT * FROM files WHERE content_hash = ? LIMIT 1",
        (content_hash,),
    ).fetchone()


def add_alias(conn: sqlite3.Connection, file_id: int, path: str) -> None:
    """Record an alternate path for a file. No-op if the alias already exists."""
    conn.execute(
        "INSERT OR IGNORE INTO file_aliases(file_id, path, discovered_at) "
        "VALUES (?, ?, ?)",
        (file_id, path, time.time()),
    )


def get_aliases(conn: sqlite3.Connection, file_id: int) -> list[str]:
    rows = conn.execute(
        "SELECT path FROM file_aliases WHERE file_id = ? ORDER BY discovered_at",
        (file_id,),
    ).fetchall()
    return [r["path"] for r in rows]


def aliased_paths_set(conn: sqlite3.Connection) -> set[str]:
    """Return all paths currently registered as aliases (any file_id)."""
    rows = conn.execute("SELECT path FROM file_aliases").fetchall()
    return {r["path"] for r in rows}


def delete_file(conn: sqlite3.Connection, path: str) -> None:
    """Remove a file and all its chunks (cascade) and vector rows.

    Cleans both vec_chunks and vec_images explicitly - they're sqlite-vec
    virtual tables without foreign keys, so cascades from `files` don't reach
    them. Without this, a delete leaks vector rows.
    """
    row = get_file_by_path(conn, path)
    if not row:
        return
    file_id = row["id"]
    chunk_ids = [
        r["id"] for r in conn.execute("SELECT id FROM chunks WHERE file_id = ?", (file_id,))
    ]
    for cid in chunk_ids:
        conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (cid,))
    image_ids = [
        r["id"] for r in conn.execute("SELECT id FROM images WHERE file_id = ?", (file_id,))
    ]
    for iid in image_ids:
        conn.execute("DELETE FROM vec_images WHERE image_id = ?", (iid,))
    conn.execute("DELETE FROM files WHERE id = ?", (file_id,))


def replace_chunks(
    conn: sqlite3.Connection,
    file_id: int,
    chunks: list[tuple[str, list[float]]] | list[tuple[str, list[float], int | None]],
) -> list[int]:
    """Atomically replace all chunks (and their vectors) for a file.

    Accepts ``(text, embedding)`` or ``(text, embedding, start_offset)`` tuples
    — the latter is used by callers that track byte offsets back into the
    original file (citation provenance). Returns the new chunk IDs in order.

    Wrapped in a SAVEPOINT so a crash mid-replace doesn't leave the file row
    with content_hash set but no chunks. The caller still has to commit the
    outer transaction, but a thrown exception here unwinds atomically.
    """
    conn.execute("SAVEPOINT replace_chunks")
    try:
        old_ids = [
            r["id"] for r in conn.execute(
                "SELECT id FROM chunks WHERE file_id = ?", (file_id,)
            )
        ]
        for cid in old_ids:
            conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (cid,))
            conn.execute("DELETE FROM entities WHERE chunk_id = ?", (cid,))
        conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))

        new_ids: list[int] = []
        for idx, item in enumerate(chunks):
            if len(item) == 3:
                text, embedding, start_offset = item
            else:
                text, embedding = item
                start_offset = None
            cur = conn.execute(
                "INSERT INTO chunks(file_id, chunk_index, text, start_offset) "
                "VALUES (?, ?, ?, ?) RETURNING id",
                (file_id, idx, text, start_offset),
            )
            chunk_id = cur.fetchone()["id"]
            new_ids.append(chunk_id)
            conn.execute(
                "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, serialize_f32(embedding)),
            )
        conn.execute("RELEASE SAVEPOINT replace_chunks")
        return new_ids
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT replace_chunks")
        conn.execute("RELEASE SAVEPOINT replace_chunks")
        raise


def upsert_image_embedding(
    conn: sqlite3.Connection,
    file_id: int,
    embedder_name: str,
    embedding_dim: int,
    embedding: list[float],
) -> int:
    """Replace any existing image embedding for this file+embedder, return image_id."""
    old = conn.execute(
        "SELECT id FROM images WHERE file_id = ? AND embedder = ?",
        (file_id, embedder_name),
    ).fetchone()
    if old:
        conn.execute("DELETE FROM vec_images WHERE image_id = ?", (old["id"],))
        conn.execute("DELETE FROM images WHERE id = ?", (old["id"],))
    cur = conn.execute(
        "INSERT INTO images(file_id, embedder, embedding_dim, indexed_at) "
        "VALUES (?, ?, ?, ?) RETURNING id",
        (file_id, embedder_name, embedding_dim, time.time()),
    )
    image_id = cur.fetchone()["id"]
    conn.execute(
        "INSERT INTO vec_images(image_id, embedding) VALUES (?, ?)",
        (image_id, serialize_f32(embedding)),
    )
    return image_id


def search_images(
    conn: sqlite3.Connection, query_embedding: list[float], k: int
) -> list[tuple[int, str, float, float]]:
    """Return [(image_id, file_path, mtime, distance)] for the k nearest images."""
    rows = conn.execute(
        "SELECT v.image_id, v.distance, f.path, f.mtime "
        "FROM vec_images v "
        "JOIN images i ON i.id = v.image_id "
        "JOIN files f ON f.id = i.file_id "
        "WHERE v.embedding MATCH ? AND v.k = ? "
        "ORDER BY v.distance",
        (serialize_f32(query_embedding), k),
    ).fetchall()
    return [(r["image_id"], r["path"], r["mtime"], r["distance"]) for r in rows]


def insert_entities(
    conn: sqlite3.Connection,
    chunk_id: int,
    entities: list[tuple[str, str]],
) -> None:
    """Insert (text, label) entities for a chunk. Dedupes by (chunk, text_lower, label)."""
    for text, label in entities:
        if not text:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO entities(chunk_id, text, text_lower, label) "
            "VALUES (?, ?, ?, ?)",
            (chunk_id, text, text.lower(), label),
        )


def chat_create_conversation(conn: sqlite3.Connection, title: str) -> int:
    """Make a new chat conversation; return its id."""
    now = time.time()
    cur = conn.execute(
        "INSERT INTO chat_conversations(title, created_at, updated_at) "
        "VALUES (?, ?, ?) RETURNING id",
        (title, now, now),
    )
    cid = cur.fetchone()["id"]
    conn.commit()
    return cid


def chat_append_message(
    conn: sqlite3.Connection,
    conversation_id: int,
    role: str,
    content_json: str,
    citations_json: str | None = None,
) -> int:
    """Append a message to a conversation. Returns the new message id."""
    now = time.time()
    seq_row = conn.execute(
        "SELECT COALESCE(MAX(seq), -1) + 1 AS next_seq "
        "FROM chat_messages WHERE conversation_id = ?",
        (conversation_id,),
    ).fetchone()
    seq = seq_row["next_seq"]
    cur = conn.execute(
        "INSERT INTO chat_messages"
        "(conversation_id, seq, role, content_json, citations_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?) RETURNING id",
        (conversation_id, seq, role, content_json, citations_json, now),
    )
    mid = cur.fetchone()["id"]
    conn.execute(
        "UPDATE chat_conversations SET updated_at = ? WHERE id = ?",
        (now, conversation_id),
    )
    conn.commit()
    return mid


def chat_list_conversations(
    conn: sqlite3.Connection, limit: int = 50
) -> list[sqlite3.Row]:
    """Most-recently-updated conversations first, with message counts."""
    return conn.execute(
        "SELECT c.id, c.title, c.created_at, c.updated_at, "
        "       (SELECT COUNT(*) FROM chat_messages m WHERE m.conversation_id = c.id) AS n_messages "
        "FROM chat_conversations c "
        "ORDER BY c.updated_at DESC "
        "LIMIT ?",
        (limit,),
    ).fetchall()


def chat_get_conversation(
    conn: sqlite3.Connection, conversation_id: int
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM chat_conversations WHERE id = ?",
        (conversation_id,),
    ).fetchone()


def chat_get_messages(
    conn: sqlite3.Connection, conversation_id: int
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM chat_messages WHERE conversation_id = ? ORDER BY seq ASC",
        (conversation_id,),
    ).fetchall()


def chat_delete_conversation(conn: sqlite3.Connection, conversation_id: int) -> None:
    conn.execute(
        "DELETE FROM chat_conversations WHERE id = ?", (conversation_id,)
    )
    conn.commit()


def chat_rename_conversation(
    conn: sqlite3.Connection, conversation_id: int, title: str
) -> None:
    conn.execute(
        "UPDATE chat_conversations SET title = ? WHERE id = ?",
        (title, conversation_id),
    )
    conn.commit()


def chat_set_system_prompt(
    conn: sqlite3.Connection, conversation_id: int, system_prompt: str | None,
) -> None:
    """Override the default chat persona for a single conversation.

    Pass ``None`` (or empty string) to clear and revert to the default
    system prompt baked into ``chat.py``.
    """
    if system_prompt is not None and not system_prompt.strip():
        system_prompt = None
    conn.execute(
        "UPDATE chat_conversations SET system_prompt = ? WHERE id = ?",
        (system_prompt, conversation_id),
    )
    conn.commit()


def chat_get_system_prompt(
    conn: sqlite3.Connection, conversation_id: int,
) -> str | None:
    row = conn.execute(
        "SELECT system_prompt FROM chat_conversations WHERE id = ?",
        (conversation_id,),
    ).fetchone()
    if row is None:
        return None
    val = row["system_prompt"]
    return val if val and val.strip() else None


# --- Watchlists --------------------------------------------------------

def watchlist_create(
    conn: sqlite3.Connection, name: str, query: str, schedule_minutes: int = 1440,
    allowed_domains: list[str] | None = None,
) -> int:
    """Create a recurring saved query. ``schedule_minutes`` defaults to 1440
    (once a day). When ``allowed_domains`` is non-empty, the watchlist's
    web search is restricted to those hosts (overrides
    cfg.web_search_allowed_domains). Returns the new id."""
    import json as _json

    domains_json = _json.dumps(allowed_domains) if allowed_domains else None
    cur = conn.execute(
        "INSERT INTO watchlists"
        "(name, query, schedule_minutes, enabled, created_at, allowed_domains_json) "
        "VALUES (?, ?, ?, 1, ?, ?) RETURNING id",
        (name, query, max(5, int(schedule_minutes)), time.time(), domains_json),
    )
    wid = cur.fetchone()["id"]
    conn.commit()
    return wid


def watchlist_set_domains(
    conn: sqlite3.Connection, watchlist_id: int, allowed_domains: list[str] | None,
) -> None:
    """Replace the watchlist's allowed_domains list. Pass ``None`` or an
    empty list to fall back to cfg.web_search_allowed_domains."""
    import json as _json

    payload = _json.dumps(allowed_domains) if allowed_domains else None
    conn.execute(
        "UPDATE watchlists SET allowed_domains_json = ? WHERE id = ?",
        (payload, watchlist_id),
    )
    conn.commit()


def watchlist_get_domains(
    conn: sqlite3.Connection, watchlist_id: int,
) -> list[str] | None:
    import json as _json

    row = conn.execute(
        "SELECT allowed_domains_json FROM watchlists WHERE id = ?",
        (watchlist_id,),
    ).fetchone()
    if not row or not row["allowed_domains_json"]:
        return None
    try:
        return list(_json.loads(row["allowed_domains_json"]))
    except (_json.JSONDecodeError, TypeError):
        return None


def watchlist_list(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """All watchlists, with last-run timestamp included for the dashboard."""
    return conn.execute(
        "SELECT id, name, query, schedule_minutes, last_run_at, enabled, created_at "
        "FROM watchlists ORDER BY created_at DESC"
    ).fetchall()


def watchlist_get(conn: sqlite3.Connection, watchlist_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM watchlists WHERE id = ?", (watchlist_id,),
    ).fetchone()


def watchlist_delete(conn: sqlite3.Connection, watchlist_id: int) -> None:
    conn.execute("DELETE FROM watchlists WHERE id = ?", (watchlist_id,))
    conn.commit()


def watchlist_set_enabled(
    conn: sqlite3.Connection, watchlist_id: int, enabled: bool,
) -> None:
    conn.execute(
        "UPDATE watchlists SET enabled = ? WHERE id = ?",
        (1 if enabled else 0, watchlist_id),
    )
    conn.commit()


def watchlist_due(conn: sqlite3.Connection, now: float | None = None) -> list[sqlite3.Row]:
    """Return enabled watchlists whose next-run time has passed.

    A watchlist is due when ``last_run_at`` is null (never run) or when
    ``now - last_run_at >= schedule_minutes * 60``. The daemon's scheduler
    polls this and runs each due watchlist via ``run_watchlist``.
    """
    n = now if now is not None else time.time()
    return conn.execute(
        "SELECT id, name, query, schedule_minutes, last_run_at, enabled "
        "FROM watchlists WHERE enabled = 1 AND ("
        "  last_run_at IS NULL OR (? - last_run_at) >= (schedule_minutes * 60)"
        ")",
        (n,),
    ).fetchall()


def watchlist_run_record_start(conn: sqlite3.Connection, watchlist_id: int) -> int:
    """Insert a watchlist_runs row marking the start of a run; return id."""
    cur = conn.execute(
        "INSERT INTO watchlist_runs(watchlist_id, started_at) "
        "VALUES (?, ?) RETURNING id",
        (watchlist_id, time.time()),
    )
    rid = cur.fetchone()["id"]
    conn.commit()
    return rid


def watchlist_run_record_finish(
    conn: sqlite3.Connection,
    run_id: int,
    answer: str | None = None,
    citations_json: str | None = None,
    error: str | None = None,
    cents_spent: float | None = None,
    new_paths_json: str | None = None,
    new_count: int = 0,
) -> None:
    """Mark a watchlist run finished and persist the synthesized answer."""
    conn.execute(
        "UPDATE watchlist_runs SET finished_at = ?, answer = ?, "
        "citations_json = ?, error = ?, cents_spent = ?, "
        "new_paths_json = ?, new_count = ? "
        "WHERE id = ?",
        (
            time.time(), answer, citations_json, error, cents_spent,
            new_paths_json, int(new_count), run_id,
        ),
    )
    # Also bump last_run_at on the parent so watchlist_due moves on.
    conn.execute(
        "UPDATE watchlists SET last_run_at = ("
        "  SELECT started_at FROM watchlist_runs WHERE id = ?"
        ") WHERE id = ("
        "  SELECT watchlist_id FROM watchlist_runs WHERE id = ?"
        ")",
        (run_id, run_id),
    )
    conn.commit()


def watchlist_previous_run(
    conn: sqlite3.Connection, watchlist_id: int, before_run_id: int,
) -> sqlite3.Row | None:
    """Find the most recent successful run before ``before_run_id``.

    "Successful" = finished_at IS NOT NULL AND error IS NULL. Used by
    run_watchlist to compute the diff against the last good run.
    """
    return conn.execute(
        "SELECT * FROM watchlist_runs WHERE watchlist_id = ? "
        "AND id < ? AND finished_at IS NOT NULL AND error IS NULL "
        "ORDER BY started_at DESC, id DESC LIMIT 1",
        (watchlist_id, before_run_id),
    ).fetchone()


def watchlist_runs(
    conn: sqlite3.Connection, watchlist_id: int, limit: int = 50,
) -> list[sqlite3.Row]:
    # Tiebreak by id DESC because time.time() on Windows has ~16ms
    # resolution; without it, two runs in quick succession can sort
    # non-deterministically.
    return conn.execute(
        "SELECT * FROM watchlist_runs WHERE watchlist_id = ? "
        "ORDER BY started_at DESC, id DESC LIMIT ?",
        (watchlist_id, limit),
    ).fetchall()


def watchlist_latest_run(
    conn: sqlite3.Connection, watchlist_id: int,
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM watchlist_runs WHERE watchlist_id = ? "
        "ORDER BY started_at DESC, id DESC LIMIT 1",
        (watchlist_id,),
    ).fetchone()


# --- Reading queue ----------------------------------------------------

def reading_queue_enqueue(
    conn: sqlite3.Connection,
    url: str,
    title: str | None,
    source: str,
    fit_label: str | None = None,
    fit_score: float | None = None,
) -> int | None:
    """Add a URL to the reading queue. Returns the id (or None if the
    URL was already queued — UNIQUE constraint).

    Idempotent: if the same URL is already in the queue, the existing row
    is left alone. This is important because watchlist runs surface the
    same items multiple times (it's the diff that's new, not always the
    URL); we don't want to keep re-summarising and re-notifying.
    """
    if not url:
        return None
    try:
        cur = conn.execute(
            "INSERT INTO reading_queue(url, title, source, added_at, "
            "fit_label, fit_score) VALUES (?, ?, ?, ?, ?, ?) RETURNING id",
            (url, title or "", source, time.time(), fit_label, fit_score),
        )
        rid = cur.fetchone()["id"]
        conn.commit()
        return rid
    except sqlite3.IntegrityError:
        # Already queued — leave the existing row in place.
        return None


def reading_queue_pending_summary(
    conn: sqlite3.Connection, limit: int = 10,
) -> list[sqlite3.Row]:
    """Items whose summary hasn't been generated yet (and haven't errored).

    The daemon picks these up after each watchlist tick and generates
    summaries one at a time. Errored ones aren't auto-retried — the
    user can click "regenerate" from the queue UI.
    """
    return conn.execute(
        "SELECT * FROM reading_queue "
        "WHERE summary IS NULL AND summary_error IS NULL "
        "AND read_at IS NULL AND skipped_at IS NULL "
        "ORDER BY added_at ASC LIMIT ?",
        (limit,),
    ).fetchall()


def reading_queue_set_summary(
    conn: sqlite3.Connection,
    queue_id: int,
    summary: str | None,
    error: str | None = None,
) -> None:
    conn.execute(
        "UPDATE reading_queue SET summary = ?, summary_error = ?, "
        "summary_generated_at = ? WHERE id = ?",
        (summary, error, time.time(), queue_id),
    )
    conn.commit()


def reading_queue_unread(
    conn: sqlite3.Connection, limit: int = 100,
) -> list[sqlite3.Row]:
    """Everything you haven't read or skipped, newest first."""
    return conn.execute(
        "SELECT * FROM reading_queue "
        "WHERE read_at IS NULL AND skipped_at IS NULL "
        "ORDER BY added_at DESC LIMIT ?",
        (limit,),
    ).fetchall()


def reading_queue_history(
    conn: sqlite3.Connection, limit: int = 100,
) -> list[sqlite3.Row]:
    """Read + skipped rows for the dashboard's archive view."""
    return conn.execute(
        "SELECT * FROM reading_queue "
        "WHERE read_at IS NOT NULL OR skipped_at IS NOT NULL "
        "ORDER BY COALESCE(read_at, skipped_at) DESC LIMIT ?",
        (limit,),
    ).fetchall()


def reading_queue_get(
    conn: sqlite3.Connection, queue_id: int,
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM reading_queue WHERE id = ?", (queue_id,),
    ).fetchone()


def reading_queue_mark_read(conn: sqlite3.Connection, queue_id: int) -> None:
    conn.execute(
        "UPDATE reading_queue SET read_at = ? WHERE id = ?",
        (time.time(), queue_id),
    )
    conn.commit()


def reading_queue_mark_skipped(conn: sqlite3.Connection, queue_id: int) -> None:
    conn.execute(
        "UPDATE reading_queue SET skipped_at = ? WHERE id = ?",
        (time.time(), queue_id),
    )
    conn.commit()


def reading_queue_unread_count(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM reading_queue "
        "WHERE read_at IS NULL AND skipped_at IS NULL",
    ).fetchone()
    return int(row["n"]) if row else 0


# --- Pre-event briefings ----------------------------------------------

def event_briefing_get(
    conn: sqlite3.Connection, event_id: str, event_source: str,
) -> sqlite3.Row | None:
    """Look up a previously-generated briefing for an event."""
    return conn.execute(
        "SELECT * FROM event_briefings WHERE event_id = ? AND event_source = ?",
        (event_id, event_source),
    ).fetchone()


def event_briefing_save(
    conn: sqlite3.Connection,
    event_id: str,
    event_source: str,
    event_starts_at: float,
    event_title: str,
    event_url: str | None,
    event_payload_json: str | None,
    briefing_text: str | None,
    citations_json: str | None = None,
    error: str | None = None,
    cents_spent: float | None = None,
) -> int:
    """Insert (or replace via UNIQUE conflict) a briefing row.

    Returns the id of the resulting row. Replacing is intentional: when
    the user clicks "regenerate" we want to overwrite the old briefing,
    not accumulate stale ones.
    """
    cur = conn.execute(
        "INSERT INTO event_briefings"
        "(event_id, event_source, event_starts_at, event_title, event_url, "
        " event_payload_json, generated_at, briefing_text, citations_json, "
        " error, cents_spent) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(event_id, event_source) DO UPDATE SET "
        "  event_starts_at = excluded.event_starts_at, "
        "  event_title = excluded.event_title, "
        "  event_url = excluded.event_url, "
        "  event_payload_json = excluded.event_payload_json, "
        "  generated_at = excluded.generated_at, "
        "  briefing_text = excluded.briefing_text, "
        "  citations_json = excluded.citations_json, "
        "  error = excluded.error, "
        "  cents_spent = excluded.cents_spent "
        "RETURNING id",
        (
            event_id, event_source, event_starts_at, event_title,
            event_url, event_payload_json, time.time(),
            briefing_text, citations_json, error, cents_spent,
        ),
    )
    bid = cur.fetchone()["id"]
    conn.commit()
    return bid


def event_briefings_list(
    conn: sqlite3.Connection, limit: int = 50,
) -> list[sqlite3.Row]:
    """Most-recent briefings first (by generated_at). For the dashboard."""
    return conn.execute(
        "SELECT * FROM event_briefings "
        "ORDER BY event_starts_at DESC, id DESC LIMIT ?",
        (limit,),
    ).fetchall()


def event_briefings_upcoming(
    conn: sqlite3.Connection, since: float | None = None,
) -> list[sqlite3.Row]:
    """Briefings for events starting in the future (or recent past)."""
    cutoff = since if since is not None else time.time() - 3600
    return conn.execute(
        "SELECT * FROM event_briefings WHERE event_starts_at >= ? "
        "ORDER BY event_starts_at ASC",
        (cutoff,),
    ).fetchall()


# --- Application tracker ----------------------------------------------

# Single source of truth for valid statuses; CLI + dashboard use this.
APPLICATION_STATUSES = (
    "applied", "screen", "interview", "offer", "rejected", "withdrawn", "ghosted",
)


def application_create(
    conn: sqlite3.Connection,
    company: str,
    role_title: str,
    role_url: str | None = None,
    applied_at: float | None = None,
    status: str = "applied",
    source: str | None = None,
    notes: str | None = None,
) -> int:
    """Record a job application. Returns the new id.

    ``role_url`` is the canonical posting URL when known (matches a
    JobsConnector virtual_path or a careers.* page). The watchlist agent
    uses it to dedupe "you've already applied to this" so it doesn't
    keep surfacing applied roles as "new".
    """
    if status not in APPLICATION_STATUSES:
        raise ValueError(
            f"unknown status {status!r}; valid: {', '.join(APPLICATION_STATUSES)}"
        )
    now = time.time()
    cur = conn.execute(
        "INSERT INTO applications"
        "(company, role_title, role_url, applied_at, status, source, notes, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?) RETURNING id",
        (
            company.strip(), role_title.strip(),
            (role_url or "").strip() or None,
            applied_at if applied_at is not None else now,
            status, (source or None), (notes or None), now,
        ),
    )
    aid = cur.fetchone()["id"]
    conn.commit()
    return aid


def application_list(
    conn: sqlite3.Connection,
    status: str | None = None,
    company: str | None = None,
) -> list[sqlite3.Row]:
    """List applications, optionally filtered. Most recently applied first."""
    where: list[str] = []
    params: list = []
    if status:
        where.append("status = ?")
        params.append(status)
    if company:
        where.append("LOWER(company) = LOWER(?)")
        params.append(company)
    sql = "SELECT * FROM applications"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY applied_at DESC"
    return conn.execute(sql, params).fetchall()


def application_get(
    conn: sqlite3.Connection, application_id: int,
) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM applications WHERE id = ?", (application_id,),
    ).fetchone()


def application_find_by_url(
    conn: sqlite3.Connection, url: str,
) -> sqlite3.Row | None:
    """Find an application by canonical posting URL. Used by the watchlist
    agent to detect "I already applied to this" before surfacing it as
    a "new" item."""
    if not url:
        return None
    return conn.execute(
        "SELECT * FROM applications WHERE role_url = ? ORDER BY applied_at DESC LIMIT 1",
        (url,),
    ).fetchone()


def application_set_status(
    conn: sqlite3.Connection, application_id: int, status: str,
    notes: str | None = None,
) -> None:
    if status not in APPLICATION_STATUSES:
        raise ValueError(
            f"unknown status {status!r}; valid: {', '.join(APPLICATION_STATUSES)}"
        )
    now = time.time()
    if notes is None:
        conn.execute(
            "UPDATE applications SET status = ?, updated_at = ? WHERE id = ?",
            (status, now, application_id),
        )
    else:
        conn.execute(
            "UPDATE applications SET status = ?, notes = ?, updated_at = ? "
            "WHERE id = ?",
            (status, notes, now, application_id),
        )
    conn.commit()


def application_delete(conn: sqlite3.Connection, application_id: int) -> None:
    conn.execute("DELETE FROM applications WHERE id = ?", (application_id,))
    conn.commit()


def applied_role_urls(conn: sqlite3.Connection) -> set[str]:
    """All canonical role_urls the user has on file. Convenience for the
    watchlist agent to filter "already applied" out of new-item lists.
    """
    rows = conn.execute(
        "SELECT DISTINCT role_url FROM applications WHERE role_url IS NOT NULL "
        "AND role_url != ''",
    ).fetchall()
    return {r["role_url"] for r in rows}


# --- Click-feedback ---------------------------------------------------

def log_click(
    conn: sqlite3.Connection,
    path: str,
    source: str,
    chunk_id: int | None = None,
) -> None:
    """Record that the user opened ``path`` from a result list. Recent
    clicks bump the path's ranking via ``recent_click_boost``."""
    conn.execute(
        "INSERT INTO click_log(path, chunk_id, source, ts) VALUES (?, ?, ?, ?)",
        (path, chunk_id, source, time.time()),
    )
    conn.commit()


def recent_clicks_by_path(
    conn: sqlite3.Connection, since_seconds: float = 30 * 86400
) -> dict[str, float]:
    """Return ``{path: most_recent_click_ts}`` for clicks newer than the cutoff.

    Used by the search ranker to compute a small recency boost on paths the
    user has actually opened recently. Default window is 30 days.
    """
    cutoff = time.time() - since_seconds
    rows = conn.execute(
        "SELECT path, MAX(ts) AS last_ts FROM click_log "
        "WHERE ts >= ? GROUP BY path",
        (cutoff,),
    ).fetchall()
    return {r["path"]: r["last_ts"] for r in rows}


def stats(conn: sqlite3.Connection) -> dict[str, int | str | None]:
    files = conn.execute("SELECT COUNT(*) AS c FROM files").fetchone()["c"]
    chunks = conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
    entities = conn.execute("SELECT COUNT(*) AS c FROM entities").fetchone()["c"]
    aliases = conn.execute("SELECT COUNT(*) AS c FROM file_aliases").fetchone()["c"]
    last = conn.execute("SELECT MAX(indexed_at) AS t FROM files").fetchone()["t"]
    return {
        "files": files,
        "chunks": chunks,
        "entities": entities,
        "aliases": aliases,
        "last_indexed_at": last,
        "embedder": get_meta(conn, "embedder_name"),
        "embedding_dim": get_meta(conn, "embedding_dim"),
    }
