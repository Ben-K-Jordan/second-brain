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
    """Open a connection with sqlite-vec loaded and sensible pragmas."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
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
            UNIQUE(file_id, chunk_index)
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);

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


def delete_file(conn: sqlite3.Connection, path: str) -> None:
    """Remove a file and all its chunks (cascade) and vector rows."""
    row = get_file_by_path(conn, path)
    if not row:
        return
    file_id = row["id"]
    chunk_ids = [
        r["id"] for r in conn.execute("SELECT id FROM chunks WHERE file_id = ?", (file_id,))
    ]
    for cid in chunk_ids:
        conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (cid,))
    conn.execute("DELETE FROM files WHERE id = ?", (file_id,))


def replace_chunks(
    conn: sqlite3.Connection,
    file_id: int,
    chunks: list[tuple[str, list[float]]],
) -> None:
    """Atomically replace all chunks (and their vectors) for a file."""
    old_ids = [
        r["id"] for r in conn.execute("SELECT id FROM chunks WHERE file_id = ?", (file_id,))
    ]
    for cid in old_ids:
        conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (cid,))
    conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))

    for idx, (text, embedding) in enumerate(chunks):
        cur = conn.execute(
            "INSERT INTO chunks(file_id, chunk_index, text) VALUES (?, ?, ?) RETURNING id",
            (file_id, idx, text),
        )
        chunk_id = cur.fetchone()["id"]
        conn.execute(
            "INSERT INTO vec_chunks(chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, serialize_f32(embedding)),
        )


def stats(conn: sqlite3.Connection) -> dict[str, int | str | None]:
    files = conn.execute("SELECT COUNT(*) AS c FROM files").fetchone()["c"]
    chunks = conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
    last = conn.execute("SELECT MAX(indexed_at) AS t FROM files").fetchone()["t"]
    return {
        "files": files,
        "chunks": chunks,
        "last_indexed_at": last,
        "embedder": get_meta(conn, "embedder_name"),
        "embedding_dim": get_meta(conn, "embedding_dim"),
    }
