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
    """Open a read/write connection with sqlite-vec loaded and sensible pragmas."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
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
    return conn


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
    import time as _time

    conn.execute(
        "INSERT OR IGNORE INTO file_aliases(file_id, path, discovered_at) "
        "VALUES (?, ?, ?)",
        (file_id, path, _time.time()),
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
    """
    old_ids = [
        r["id"] for r in conn.execute("SELECT id FROM chunks WHERE file_id = ?", (file_id,))
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
    return new_ids


def upsert_image_embedding(
    conn: sqlite3.Connection,
    file_id: int,
    embedder_name: str,
    embedding_dim: int,
    embedding: list[float],
) -> int:
    """Replace any existing image embedding for this file+embedder, return image_id."""
    import time as _time

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
        (file_id, embedder_name, embedding_dim, _time.time()),
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
