"""Phase 86 + 87: cross-conversation memory + temporal queries tests."""

from __future__ import annotations

import time

import pytest

from secondbrain import memory

# ============================ Phase 86 memories =======================

def test_remember_persists(fresh_db):
    mid = memory.remember(
        fresh_db,
        key="user-name",
        content="The user is Ben.",
    )
    m = memory.get_memory(fresh_db, "user-name")
    assert m.id == mid
    assert m.content == "The user is Ben."
    assert m.kind == "fact"


def test_remember_idempotent_on_key(fresh_db):
    """Same key updates content + bumps confidence (max-of)."""
    mid_a = memory.remember(
        fresh_db, key="x", content="first", confidence=0.5,
    )
    mid_b = memory.remember(
        fresh_db, key="x", content="second", confidence=0.9,
    )
    assert mid_a == mid_b
    m = memory.get_memory(fresh_db, "x")
    assert m.content == "second"
    assert m.confidence == 0.9


def test_remember_keeps_higher_confidence(fresh_db):
    """If a re-write provides LOWER confidence, keep the higher value
    so manual high-confidence facts aren't downgraded by auto-extracts."""
    memory.remember(fresh_db, key="x", content="x", confidence=0.9)
    memory.remember(fresh_db, key="x", content="x", confidence=0.5)
    assert memory.get_memory(fresh_db, "x").confidence == 0.9


def test_remember_validates_inputs(fresh_db):
    with pytest.raises(ValueError):
        memory.remember(fresh_db, key="", content="x")
    with pytest.raises(ValueError):
        memory.remember(fresh_db, key="x", content="")
    with pytest.raises(ValueError):
        memory.remember(fresh_db, key="x", content="y", kind="bogus")


def test_remember_lowercases_key(fresh_db):
    memory.remember(fresh_db, key="USER-NAME", content="Ben")
    assert memory.get_memory(fresh_db, "user-name").content == "Ben"
    assert memory.get_memory(fresh_db, "USER-NAME").content == "Ben"


def test_forget(fresh_db):
    memory.remember(fresh_db, key="x", content="y")
    assert memory.forget(fresh_db, "x") is True
    assert memory.forget(fresh_db, "x") is False  # already gone
    assert memory.get_memory(fresh_db, "x") is None


def test_list_memories_filters_by_kind(fresh_db):
    memory.remember(fresh_db, key="a", content="x", kind="fact")
    memory.remember(fresh_db, key="b", content="x", kind="preference")
    facts = memory.list_memories(fresh_db, kind="fact")
    assert len(facts) == 1
    assert facts[0].key == "a"


def test_most_relevant_memories_token_overlap(fresh_db):
    memory.remember(
        fresh_db, key="voyage",
        content="The user uses Voyage embeddings.",
    )
    memory.remember(
        fresh_db, key="claude",
        content="The user prefers Claude Sonnet.",
    )
    memory.remember(
        fresh_db, key="random",
        content="Unrelated trivia about cats.",
    )
    out = memory.most_relevant_memories(
        fresh_db, "what embeddings do I use?", k=5,
    )
    assert len(out) >= 1
    assert out[0].key == "voyage"


def test_most_relevant_returns_empty_for_blank_query(fresh_db):
    memory.remember(fresh_db, key="x", content="y")
    assert memory.most_relevant_memories(fresh_db, "") == []


def test_most_relevant_skips_low_confidence(fresh_db):
    """A memory tagged 0.1 confidence shouldn't surface even on
    perfect token overlap."""
    memory.remember(
        fresh_db, key="weak", content="The user uses ferrets.",
        confidence=0.1,
    )
    out = memory.most_relevant_memories(fresh_db, "ferrets")
    assert all(m.key != "weak" for m in out)


def test_mark_referenced_bumps_count(fresh_db):
    mid = memory.remember(fresh_db, key="x", content="y")
    n = memory.mark_referenced(fresh_db, [mid])
    assert n == 1
    m = memory.get_memory(fresh_db, "x")
    assert m.reference_count == 1
    assert m.last_referenced_at is not None


def test_mark_referenced_handles_empty_list(fresh_db):
    assert memory.mark_referenced(fresh_db, []) == 0


def test_render_memories_for_prompt():
    from secondbrain.memory import Memory
    mems = [
        Memory(
            id=1, key="x", content="The user uses Voyage.",
            kind="fact", source_conversation_id=None,
            created_at=0, last_referenced_at=None,
            reference_count=0, confidence=0.9,
        ),
        Memory(
            id=2, key="y", content="Prefers concise answers.",
            kind="preference", source_conversation_id=None,
            created_at=0, last_referenced_at=None,
            reference_count=0, confidence=0.9,
        ),
    ]
    out = memory.render_memories_for_prompt(mems)
    assert "Voyage" in out
    assert "concise" in out
    assert "·" in out  # fact bullet
    assert "★" in out  # preference bullet


def test_render_empty_memories_returns_empty():
    assert memory.render_memories_for_prompt([]) == ""


def test_render_memories_caps_at_max_in_prompt():
    from secondbrain.memory import Memory
    big = [
        Memory(
            id=i, key=f"k{i}", content=f"fact {i}",
            kind="fact", source_conversation_id=None,
            created_at=0, last_referenced_at=None,
            reference_count=0, confidence=0.9,
        )
        for i in range(50)
    ]
    out = memory.render_memories_for_prompt(big)
    # _MAX_MEMORIES_IN_PROMPT = 20
    assert out.count("·") == 20


# ============================ Phase 87 snapshots ======================

_seed_counter = {"n": 0}


def _seed_files(conn, n: int) -> list[int]:
    """Counter-based path so successive calls within one test don't
    collide on the unique-path constraint (Windows time.time() = ~16ms
    resolution would otherwise re-use the same path for back-to-back
    seeds in the same test)."""
    ids = []
    for _ in range(n):
        _seed_counter["n"] += 1
        cur = conn.execute(
            "INSERT INTO files(path, mtime, size, kind, indexed_at) "
            "VALUES (?, ?, 1, 'document', ?)",
            (f"/notes/{_seed_counter['n']}.md",
             time.time(), time.time()),
        )
        ids.append(int(cur.lastrowid))
    conn.commit()
    return ids


def test_take_snapshot_captures_all_file_ids(fresh_db):
    ids = _seed_files(fresh_db, 5)
    sid = memory.take_snapshot(fresh_db)
    snaps = memory.list_snapshots(fresh_db)
    assert len(snaps) == 1
    assert snaps[0].id == sid
    assert snaps[0].file_ids == set(ids)
    assert snaps[0].n_files == 5


def test_needs_snapshot_first_time(fresh_db):
    assert memory.needs_snapshot(fresh_db) is True


def test_needs_snapshot_false_after_recent(fresh_db):
    memory.take_snapshot(fresh_db)
    assert memory.needs_snapshot(fresh_db) is False


def test_needs_snapshot_true_after_old_snapshot(fresh_db):
    """Snapshot from 30d ago — overdue."""
    fresh_db.execute(
        "INSERT INTO index_snapshots(taken_at, file_ids_json, n_files) "
        "VALUES (?, '[]', 0)",
        (time.time() - 30 * 86400,),
    )
    fresh_db.commit()
    assert memory.needs_snapshot(fresh_db) is True


def test_snapshot_at_returns_closest_preceding(fresh_db):
    """Three snapshots at different times → snapshot_at returns the
    most recent one not after the requested time."""
    now = time.time()
    for offset in (30, 14, 7):
        fresh_db.execute(
            "INSERT INTO index_snapshots(taken_at, file_ids_json, n_files) "
            "VALUES (?, '[]', 0)",
            (now - offset * 86400,),
        )
    fresh_db.commit()
    # Asking for "10 days ago" should return the 14d snapshot (the
    # 7d one is in the *future* relative to that target).
    snap = memory.snapshot_at(fresh_db, now - 10 * 86400)
    assert snap is not None
    assert abs(snap.taken_at - (now - 14 * 86400)) < 1


def test_snapshot_at_returns_none_when_no_snapshot_before(fresh_db):
    """Asking for a time before any snapshot exists."""
    fresh_db.execute(
        "INSERT INTO index_snapshots(taken_at, file_ids_json, n_files) "
        "VALUES (?, '[]', 0)",
        (time.time(),),
    )
    fresh_db.commit()
    assert memory.snapshot_at(fresh_db, 1.0) is None


def test_filter_to_snapshot_excludes_files_not_in_snapshot(fresh_db):
    """A file added AFTER the snapshot should be filtered out."""
    ids = _seed_files(fresh_db, 3)
    memory.take_snapshot(fresh_db)
    # Add another file post-snapshot.
    new_id = _seed_files(fresh_db, 1)[0]
    snap = memory.snapshot_at(fresh_db, time.time())
    # Filter the full set — new_id should be removed.
    snap_taken_at = snap.taken_at
    filtered = memory.filter_to_snapshot(
        fresh_db, ids + [new_id], when=snap_taken_at,
    )
    assert new_id not in filtered
    for original in ids:
        assert original in filtered


def test_filter_to_snapshot_no_snapshot_returns_input(fresh_db):
    """When no snapshot covers the requested time, return input
    unchanged — graceful degradation, not an error."""
    out = memory.filter_to_snapshot(
        fresh_db, [1, 2, 3], when=time.time(),
    )
    assert out == [1, 2, 3]


def test_take_snapshot_if_due_skips_when_recent(fresh_db, tmp_cfg):
    memory.take_snapshot(fresh_db)
    assert memory.take_snapshot_if_due(tmp_cfg, fresh_db) is False


def test_take_snapshot_if_due_fires_when_overdue(fresh_db, tmp_cfg):
    assert memory.take_snapshot_if_due(tmp_cfg, fresh_db) is True
    snaps = memory.list_snapshots(fresh_db)
    assert len(snaps) == 1


# ============== chat-memory-recall plumbing (polish) =================

def test_chat_recall_pipeline_surfaces_memories_to_prompt(
    fresh_db, monkeypatch,
):
    """Polish-pass test: stream_chat should recall relevant memories
    and prepend them to the system prompt. The bug we're guarding
    against: ask_brain bumped reference_count on a never-set
    `_last_memory_ids` attribute, so memory was wired in name only.
    """
    from secondbrain import chat as chat_mod

    # Seed a memory the recall should find.
    memory.remember(
        fresh_db, key="voyage",
        content="The user uses Voyage embeddings exclusively.",
    )
    captured: dict = {"system": None}

    # Stub stream_chat to capture the assembled system prompt without
    # actually calling Anthropic.
    def _fake_stream(
        cfg, conn, embedder, reranker, user_message,
        history, system_prompt=None, web_search_allowed_domains=None,
    ):
        captured["system_was_provided"] = system_prompt is not None
        # Mirror the system-prompt assembly that stream_chat does.
        from secondbrain.chat import _SYSTEM_PROMPT
        active = (system_prompt or "").strip() or _SYSTEM_PROMPT
        from secondbrain.memory import (
            most_relevant_memories,
            render_memories_for_prompt,
        )
        relevant = most_relevant_memories(conn, user_message, k=8)
        if relevant:
            chat_mod.stream_chat._last_memory_ids = [m.id for m in relevant]
            block = render_memories_for_prompt(relevant)
            if block:
                active = f"{active}\n\n{block}"
        captured["system"] = active
        # Minimal event stream — done immediately.
        from secondbrain.chat import ChatTurnEvent
        yield ChatTurnEvent(kind="done",
                            data={"text": "ok", "citations": []})

    monkeypatch.setattr(chat_mod, "stream_chat", _fake_stream)

    response = chat_mod.ask_brain(
        cfg=None, conn=fresh_db, embedder=None, reranker=None,
        question="What embeddings do I use?",
    )
    assert response is not None
    # The prompt assembled inside the stub should include our memory.
    assert "Voyage" in captured["system"]
    # And mark_referenced should have bumped the count.
    m = memory.get_memory(fresh_db, "voyage")
    assert m.reference_count == 1
    assert m.last_referenced_at is not None


def test_chat_recall_skips_when_no_relevant_memory(
    fresh_db, monkeypatch,
):
    """Empty recall should not break the prompt (no memory block,
    no reference bump)."""
    from secondbrain import chat as chat_mod

    captured: dict = {"system": ""}

    def _fake_stream(
        cfg, conn, embedder, reranker, user_message,
        history, system_prompt=None, web_search_allowed_domains=None,
    ):
        from secondbrain.chat import _SYSTEM_PROMPT
        from secondbrain.memory import most_relevant_memories
        active = (system_prompt or "").strip() or _SYSTEM_PROMPT
        relevant = most_relevant_memories(conn, user_message, k=8)
        chat_mod.stream_chat._last_memory_ids = [
            m.id for m in relevant
        ]
        captured["n_memories"] = len(relevant)
        captured["system"] = active
        from secondbrain.chat import ChatTurnEvent
        yield ChatTurnEvent(kind="done",
                            data={"text": "ok", "citations": []})

    monkeypatch.setattr(chat_mod, "stream_chat", _fake_stream)
    chat_mod.ask_brain(
        cfg=None, conn=fresh_db, embedder=None, reranker=None,
        question="random unrelated question",
    )
    assert captured["n_memories"] == 0
    # The system prompt should NOT contain any memory block header.
    assert "Things to remember" not in captured["system"]
