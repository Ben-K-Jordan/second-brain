"""Schema migrations + chat persistence + click-feedback storage."""

from __future__ import annotations

import time

from secondbrain.db import (
    aliased_paths_set,
    chat_append_message,
    chat_create_conversation,
    chat_delete_conversation,
    chat_get_conversation,
    chat_get_messages,
    chat_list_conversations,
    chat_rename_conversation,
    log_click,
    recent_clicks_by_path,
)


def test_chat_lifecycle(fresh_db):
    cid = chat_create_conversation(fresh_db, "first chat")
    chat_append_message(fresh_db, cid, "user", '"hello"')
    chat_append_message(fresh_db, cid, "assistant", '[{"type":"text","text":"hi"}]')
    chat_append_message(fresh_db, cid, "user", '"follow-up"')

    msgs = chat_get_messages(fresh_db, cid)
    assert [m["seq"] for m in msgs] == [0, 1, 2], "seq should be monotonic from 0"
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]

    rows = chat_list_conversations(fresh_db)
    assert len(rows) == 1
    assert rows[0]["n_messages"] == 3
    assert rows[0]["title"] == "first chat"

    chat_rename_conversation(fresh_db, cid, "renamed")
    assert chat_get_conversation(fresh_db, cid)["title"] == "renamed"

    chat_delete_conversation(fresh_db, cid)
    assert chat_get_conversation(fresh_db, cid) is None
    # cascade should drop messages
    assert chat_get_messages(fresh_db, cid) == []


def test_chat_seq_isolated_per_conversation(fresh_db):
    a = chat_create_conversation(fresh_db, "a")
    b = chat_create_conversation(fresh_db, "b")
    chat_append_message(fresh_db, a, "user", '"a1"')
    chat_append_message(fresh_db, b, "user", '"b1"')
    chat_append_message(fresh_db, a, "user", '"a2"')
    a_seqs = [m["seq"] for m in chat_get_messages(fresh_db, a)]
    b_seqs = [m["seq"] for m in chat_get_messages(fresh_db, b)]
    assert a_seqs == [0, 1]
    assert b_seqs == [0]


def test_click_log_aggregates_by_path(fresh_db):
    """recent_clicks_by_path returns the most-recent timestamp per
    path. We log b.md FIRST, then a.md (twice) — so a.md should be
    the more recent of the two paths.

    Sleeps are 25ms each: Windows time.time() has ~16ms resolution, so
    the previous 10ms gaps could collapse and make the comparison
    non-deterministic.
    """
    log_click(fresh_db, "/notes/b.md", "chat")
    time.sleep(0.025)
    log_click(fresh_db, "/notes/a.md", "search", chunk_id=1)
    time.sleep(0.025)
    log_click(fresh_db, "/notes/a.md", "search", chunk_id=2)

    by_path = recent_clicks_by_path(fresh_db)
    assert set(by_path) == {"/notes/a.md", "/notes/b.md"}
    # a.md's second click is the most recent of all three → a.md > b.md.
    assert by_path["/notes/a.md"] > by_path["/notes/b.md"]


def test_click_log_window_filter(fresh_db):
    """Clicks older than the window are excluded."""
    fresh_db.execute(
        "INSERT INTO click_log(path, chunk_id, source, ts) VALUES (?, ?, ?, ?)",
        ("/old.md", None, "search", time.time() - 100 * 86400),
    )
    fresh_db.execute(
        "INSERT INTO click_log(path, chunk_id, source, ts) VALUES (?, ?, ?, ?)",
        ("/new.md", None, "search", time.time() - 1),
    )
    fresh_db.commit()
    recent = recent_clicks_by_path(fresh_db, since_seconds=30 * 86400)
    assert "/new.md" in recent
    assert "/old.md" not in recent


def test_alias_helpers(fresh_db):
    """The file_aliases table round-trips paths via add_alias / aliased_paths_set."""
    from secondbrain.db import add_alias, upsert_file

    file_id = upsert_file(
        fresh_db, path="/canonical.md", mtime=0.0, size=10,
        kind="document", content_hash="abc",
    )
    add_alias(fresh_db, file_id, "/Downloads/copy.md")
    add_alias(fresh_db, file_id, "/OneDrive/copy.md")
    fresh_db.commit()
    assert "/Downloads/copy.md" in aliased_paths_set(fresh_db)
    assert "/OneDrive/copy.md" in aliased_paths_set(fresh_db)
