"""Phase 65 + 66: people module tests.

Coverage:
  - canonicalize + dedup invariants
  - upsert_person create-or-find
  - alias add / remove / merge_people
  - search + listing orders
  - link_chunk_mentions auto-linker (Phase 66)
  - bulk materialize_from_entities
  - profile_for + recent_mentions
"""

from __future__ import annotations

import time

import pytest

from secondbrain import people as people_mod

# ============================ canonicalize ============================

def test_canonicalize_lowercases_and_trims():
    assert people_mod.canonicalize("Sarah Chen") == "sarah chen"
    assert people_mod.canonicalize("  SARAH   CHEN  ") == "sarah chen"


def test_canonicalize_collapses_internal_whitespace():
    assert people_mod.canonicalize("S.\tChen") == "s. chen"


# ============================ upsert / dedup ==========================

def test_upsert_creates_new_person(fresh_db):
    pid = people_mod.upsert_person(
        fresh_db, display_name="Sarah Chen", email="sarah@example.com",
        role="PM",
    )
    assert pid > 0
    p = people_mod.get_person(fresh_db, pid)
    assert p.display_name == "Sarah Chen"
    assert p.email == "sarah@example.com"
    assert p.role == "PM"
    assert p.mention_count == 1


def test_upsert_dedupes_by_canonical_name(fresh_db):
    """Same person with different casing should map to one row."""
    pid_a = people_mod.upsert_person(fresh_db, display_name="Sarah Chen")
    pid_b = people_mod.upsert_person(fresh_db, display_name="sarah chen")
    pid_c = people_mod.upsert_person(fresh_db, display_name="SARAH CHEN")
    assert pid_a == pid_b == pid_c
    rows = fresh_db.execute("SELECT * FROM people").fetchall()
    assert len(rows) == 1


def test_upsert_bumps_mention_count_on_repeat(fresh_db):
    pid = people_mod.upsert_person(fresh_db, display_name="X")
    people_mod.upsert_person(fresh_db, display_name="x")
    people_mod.upsert_person(fresh_db, display_name="X")
    p = people_mod.get_person(fresh_db, pid)
    assert p.mention_count == 3


def test_upsert_preserves_user_edits_on_repeat(fresh_db):
    """If the user manually set email/company/role, a later auto-import
    that re-upserts the same canonical name shouldn't overwrite."""
    pid = people_mod.upsert_person(
        fresh_db, display_name="Sarah", email="sarah@manual.com",
    )
    # Simulate a connector re-importing without email.
    people_mod.upsert_person(
        fresh_db, display_name="Sarah", email="auto-import@ignored.com",
    )
    p = people_mod.get_person(fresh_db, pid)
    # User-edited email persists.
    assert p.email == "sarah@manual.com"


def test_upsert_fills_empty_fields_on_repeat(fresh_db):
    """Conversely, when a field WAS empty, a connector providing it
    should populate. 'Don't clobber user input' isn't 'never improve'."""
    pid = people_mod.upsert_person(fresh_db, display_name="Sarah")
    people_mod.upsert_person(
        fresh_db, display_name="Sarah", email="auto@example.com",
    )
    p = people_mod.get_person(fresh_db, pid)
    assert p.email == "auto@example.com"


def test_upsert_seeds_canonical_alias(fresh_db):
    """A new person should be auto-aliased so the linker can find them."""
    pid = people_mod.upsert_person(fresh_db, display_name="Sarah Chen")
    aliases = people_mod.get_aliases(fresh_db, pid)
    assert "Sarah Chen" in aliases


def test_upsert_rejects_blank_name(fresh_db):
    with pytest.raises(ValueError):
        people_mod.upsert_person(fresh_db, display_name="")
    with pytest.raises(ValueError):
        people_mod.upsert_person(fresh_db, display_name="   ")


# ============================ aliases ================================

def test_add_alias_idempotent(fresh_db):
    pid = people_mod.upsert_person(fresh_db, display_name="Sarah Chen")
    assert people_mod.add_alias(fresh_db, pid, "Sarah") is True
    assert people_mod.add_alias(fresh_db, pid, "Sarah") is False  # dup
    aliases = people_mod.get_aliases(fresh_db, pid)
    assert "Sarah" in aliases


def test_find_by_alias_resolves_either_alias_or_canonical(fresh_db):
    pid = people_mod.upsert_person(fresh_db, display_name="Sarah Chen")
    people_mod.add_alias(fresh_db, pid, "S Chen")
    p_via_alias = people_mod.find_by_alias(fresh_db, "S Chen")
    p_via_canonical = people_mod.find_by_alias(fresh_db, "Sarah Chen")
    p_via_lowercase = people_mod.find_by_alias(fresh_db, "sarah chen")
    assert p_via_alias.id == pid
    assert p_via_canonical.id == pid
    assert p_via_lowercase.id == pid


def test_find_by_alias_returns_none_for_unknown(fresh_db):
    assert people_mod.find_by_alias(fresh_db, "nobody") is None


def test_remove_alias(fresh_db):
    pid = people_mod.upsert_person(fresh_db, display_name="Sarah")
    people_mod.add_alias(fresh_db, pid, "S.C.")
    assert people_mod.remove_alias(fresh_db, "S.C.") is True
    assert people_mod.remove_alias(fresh_db, "S.C.") is False  # gone


# ============================ merge ==================================

def test_merge_people_consolidates(fresh_db):
    """When auto-resolution split one human into two, merge_people
    folds aliases + mentions into the survivor."""
    pid_a = people_mod.upsert_person(fresh_db, display_name="Sarah Chen")
    pid_b = people_mod.upsert_person(fresh_db, display_name="S Chen")
    people_mod.add_alias(fresh_db, pid_b, "Sarah C")
    people_mod.merge_people(fresh_db, into_id=pid_a, from_id=pid_b)
    # Source row gone.
    assert people_mod.get_person(fresh_db, pid_b) is None
    # Aliases moved.
    aliases = people_mod.get_aliases(fresh_db, pid_a)
    assert "Sarah C" in aliases
    assert "S Chen" in aliases


def test_merge_into_self_is_noop(fresh_db):
    pid = people_mod.upsert_person(fresh_db, display_name="X")
    people_mod.merge_people(fresh_db, into_id=pid, from_id=pid)
    assert people_mod.get_person(fresh_db, pid) is not None


# ============================ listing / search ========================

def test_list_people_orders_recent(fresh_db):
    pid_a = people_mod.upsert_person(fresh_db, display_name="A",
                                     when=time.time() - 100)
    pid_b = people_mod.upsert_person(fresh_db, display_name="B",
                                     when=time.time())
    rows = people_mod.list_people(fresh_db, order="recent")
    assert [r.id for r in rows] == [pid_b, pid_a]


def test_list_people_orders_by_mentions(fresh_db):
    pid_a = people_mod.upsert_person(fresh_db, display_name="A")
    pid_b = people_mod.upsert_person(fresh_db, display_name="B")
    # Bump A's mention count higher.
    for _ in range(5):
        people_mod.upsert_person(fresh_db, display_name="A")
    rows = people_mod.list_people(fresh_db, order="mentions")
    assert rows[0].id == pid_a
    assert rows[1].id == pid_b


def test_list_people_orders_by_name(fresh_db):
    people_mod.upsert_person(fresh_db, display_name="Charlie")
    people_mod.upsert_person(fresh_db, display_name="Alice")
    people_mod.upsert_person(fresh_db, display_name="Bob")
    rows = people_mod.list_people(fresh_db, order="name")
    assert [r.display_name for r in rows] == ["Alice", "Bob", "Charlie"]


def test_search_people_substring(fresh_db):
    people_mod.upsert_person(fresh_db, display_name="Sarah Chen",
                             email="sarah@example.com")
    people_mod.upsert_person(fresh_db, display_name="Bob Smith")
    # By name:
    rows = people_mod.search_people(fresh_db, "sarah")
    assert len(rows) == 1
    assert rows[0].display_name == "Sarah Chen"
    # By email:
    rows = people_mod.search_people(fresh_db, "example.com")
    assert len(rows) == 1


# ============================ profile fields =========================

def test_set_field_updates(fresh_db):
    pid = people_mod.upsert_person(fresh_db, display_name="X")
    assert people_mod.set_field(
        fresh_db, pid, email="new@example.com", role="Engineer",
    )
    p = people_mod.get_person(fresh_db, pid)
    assert p.email == "new@example.com"
    assert p.role == "Engineer"


def test_set_field_no_args_returns_false(fresh_db):
    pid = people_mod.upsert_person(fresh_db, display_name="X")
    assert people_mod.set_field(fresh_db, pid) is False


def test_set_field_clears_with_empty_string(fresh_db):
    """Empty string explicitly clears (vs None which leaves alone)."""
    pid = people_mod.upsert_person(
        fresh_db, display_name="X", role="PM",
    )
    people_mod.set_field(fresh_db, pid, role="")
    p = people_mod.get_person(fresh_db, pid)
    assert p.role == ""


# ===================== Phase 66: mention linking =====================

def _seed_chunk(conn, *, file_path: str, text: str) -> tuple[int, int]:
    """Helper — insert one file with one chunk. Returns (file_id, chunk_id)."""
    n = time.time()
    cur = conn.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at, content_hash) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (file_path, n, len(text), "document", n, None),
    )
    fid = cur.lastrowid
    cur = conn.execute(
        "INSERT INTO chunks(file_id, chunk_index, text) VALUES (?, ?, ?)",
        (fid, 0, text),
    )
    cid = cur.lastrowid
    conn.commit()
    return fid, cid


def test_link_chunk_mentions_finds_canonical_name(fresh_db):
    pid = people_mod.upsert_person(fresh_db, display_name="Sarah Chen")
    people_mod.clear_alias_cache()
    fid, cid = _seed_chunk(
        fresh_db, file_path="meeting.md",
        text="Met with Sarah Chen today to discuss the migration.",
    )
    n = people_mod.link_chunk_mentions(
        fresh_db, cid, fid, "Met with Sarah Chen today to discuss the migration.",
        time.time(),
    )
    assert n == 1
    rows = fresh_db.execute(
        "SELECT * FROM person_mentions WHERE chunk_id = ?", (cid,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["person_id"] == pid


def test_link_chunk_mentions_finds_alias(fresh_db):
    pid = people_mod.upsert_person(fresh_db, display_name="Sarah Chen")
    people_mod.add_alias(fresh_db, pid, "Sarah")
    people_mod.clear_alias_cache()
    fid, cid = _seed_chunk(
        fresh_db, file_path="x.md", text="Sarah said the deck is ready.",
    )
    n = people_mod.link_chunk_mentions(
        fresh_db, cid, fid, "Sarah said the deck is ready.", time.time(),
    )
    assert n == 1


def test_link_chunk_mentions_uses_word_boundaries(fresh_db):
    """'Sarah' should not match inside 'Sarahsplaining'."""
    people_mod.upsert_person(fresh_db, display_name="Sarah")
    people_mod.clear_alias_cache()
    fid, cid = _seed_chunk(
        fresh_db, file_path="x.md", text="Stop Sarahsplaining to us.",
    )
    n = people_mod.link_chunk_mentions(
        fresh_db, cid, fid, "Stop Sarahsplaining to us.", time.time(),
    )
    assert n == 0


def test_link_chunk_mentions_skips_short_aliases(fresh_db):
    """Two-letter aliases ('Al', 'Bo') are too noisy to auto-link."""
    pid = people_mod.upsert_person(fresh_db, display_name="Albert")
    people_mod.add_alias(fresh_db, pid, "Al")
    people_mod.clear_alias_cache()
    fid, cid = _seed_chunk(
        fresh_db, file_path="x.md", text="Al and Bo went to the meeting.",
    )
    n = people_mod.link_chunk_mentions(
        fresh_db, cid, fid, "Al and Bo went to the meeting.", time.time(),
    )
    assert n == 0  # 'Al' < min length; canonical 'Albert' not in text


def test_link_chunk_mentions_idempotent(fresh_db):
    people_mod.upsert_person(fresh_db, display_name="Sarah Chen")
    people_mod.clear_alias_cache()
    fid, cid = _seed_chunk(
        fresh_db, file_path="x.md", text="Sarah Chen joined.",
    )
    n1 = people_mod.link_chunk_mentions(
        fresh_db, cid, fid, "Sarah Chen joined.", time.time(),
    )
    n2 = people_mod.link_chunk_mentions(
        fresh_db, cid, fid, "Sarah Chen joined.", time.time(),
    )
    assert n1 == 1
    assert n2 == 0  # UNIQUE on (person_id, chunk_id) blocks dup


def test_link_chunk_mentions_dedupes_within_chunk(fresh_db):
    """If 'Sarah Chen' appears 3 times in one chunk, only one row."""
    people_mod.upsert_person(fresh_db, display_name="Sarah Chen")
    people_mod.clear_alias_cache()
    fid, cid = _seed_chunk(
        fresh_db, file_path="x.md",
        text="Sarah Chen said. Sarah Chen replied. Sarah Chen agreed.",
    )
    n = people_mod.link_chunk_mentions(
        fresh_db, cid, fid,
        "Sarah Chen said. Sarah Chen replied. Sarah Chen agreed.",
        time.time(),
    )
    assert n == 1


def test_link_chunk_mentions_handles_multiple_people(fresh_db):
    pid_a = people_mod.upsert_person(fresh_db, display_name="Sarah Chen")
    pid_b = people_mod.upsert_person(fresh_db, display_name="Bob Smith")
    people_mod.clear_alias_cache()
    fid, cid = _seed_chunk(
        fresh_db, file_path="x.md", text="Sarah Chen and Bob Smith met.",
    )
    n = people_mod.link_chunk_mentions(
        fresh_db, cid, fid,
        "Sarah Chen and Bob Smith met.", time.time(),
    )
    assert n == 2
    rows = fresh_db.execute(
        "SELECT person_id FROM person_mentions WHERE chunk_id = ?", (cid,),
    ).fetchall()
    assert {r["person_id"] for r in rows} == {pid_a, pid_b}


def test_link_chunk_mentions_no_aliases_returns_zero(fresh_db):
    """Empty aliases table → no work, no crash."""
    fid, cid = _seed_chunk(
        fresh_db, file_path="x.md", text="some text",
    )
    n = people_mod.link_chunk_mentions(
        fresh_db, cid, fid, "some text", time.time(),
    )
    assert n == 0


def test_link_chunk_mentions_updates_last_seen(fresh_db):
    """Linking a new mention should bump the person's last_seen_at."""
    pid = people_mod.upsert_person(
        fresh_db, display_name="Sarah", when=100.0,
    )
    people_mod.clear_alias_cache()
    fid, cid = _seed_chunk(
        fresh_db, file_path="x.md", text="Sarah was there.",
    )
    later = 200.0
    people_mod.link_chunk_mentions(
        fresh_db, cid, fid, "Sarah was there.", later,
    )
    p = people_mod.get_person(fresh_db, pid)
    assert p.last_seen_at == later


# ============================ profile view ===========================

def test_profile_for_includes_mentions(fresh_db):
    pid = people_mod.upsert_person(fresh_db, display_name="Sarah Chen")
    people_mod.clear_alias_cache()
    fid, cid = _seed_chunk(
        fresh_db, file_path="meeting.md",
        text="Sarah Chen said the API is shipping next week.",
    )
    people_mod.link_chunk_mentions(
        fresh_db, cid, fid,
        "Sarah Chen said the API is shipping next week.",
        time.time(),
    )
    profile = people_mod.profile_for(fresh_db, pid)
    assert profile.person.id == pid
    assert "Sarah Chen" in profile.aliases
    assert len(profile.recent_mentions) == 1
    m = profile.recent_mentions[0]
    assert m.file_path == "meeting.md"
    assert "Sarah Chen" in m.chunk_text_preview


def test_profile_for_returns_none_for_unknown(fresh_db):
    assert people_mod.profile_for(fresh_db, 9999) is None


def test_profile_truncates_long_chunk_preview(fresh_db):
    """Aliases under _MIN_ALIAS_LEN (3) wouldn't match — use a real
    name that survives the auto-linker's safety filter."""
    pid = people_mod.upsert_person(fresh_db, display_name="Sarah")
    people_mod.clear_alias_cache()
    long_text = "Sarah " + "blah " * 200
    fid, cid = _seed_chunk(
        fresh_db, file_path="x.md", text=long_text,
    )
    people_mod.link_chunk_mentions(
        fresh_db, cid, fid, long_text, time.time(),
    )
    profile = people_mod.profile_for(fresh_db, pid)
    assert len(profile.recent_mentions[0].chunk_text_preview) <= 201
    assert profile.recent_mentions[0].chunk_text_preview.endswith("…")


# ===================== bulk materialisation ==========================

def test_materialize_promotes_frequent_entities(fresh_db):
    """An entity mentioned in N+ chunks gets promoted to a person."""
    # Seed: 3 chunks with the same PERSON entity.
    fid, _ = _seed_chunk(fresh_db, file_path="a.md", text="A")
    for i in range(3):
        cur = fresh_db.execute(
            "INSERT INTO chunks(file_id, chunk_index, text) "
            "VALUES (?, ?, ?)",
            (fid, i + 1, f"Sarah Chen at meeting {i}"),
        )
        cid = cur.lastrowid
        fresh_db.execute(
            "INSERT INTO entities(chunk_id, text, text_lower, label) "
            "VALUES (?, ?, ?, 'PERSON')",
            (cid, "Sarah Chen", "sarah chen"),
        )
    fresh_db.commit()
    n = people_mod.materialize_from_entities(fresh_db, min_mentions=2)
    assert n == 1
    p = people_mod.find_by_alias(fresh_db, "Sarah Chen")
    assert p is not None


def test_materialize_skips_one_off_entities(fresh_db):
    """Single-mention entities are too noisy to promote."""
    fid, cid = _seed_chunk(
        fresh_db, file_path="a.md", text="John said hi",
    )
    fresh_db.execute(
        "INSERT INTO entities(chunk_id, text, text_lower, label) "
        "VALUES (?, ?, ?, 'PERSON')",
        (cid, "John", "john"),
    )
    fresh_db.commit()
    n = people_mod.materialize_from_entities(fresh_db, min_mentions=2)
    assert n == 0


def test_materialize_idempotent(fresh_db):
    """Re-running over the same entities shouldn't double-create people."""
    fid, _ = _seed_chunk(fresh_db, file_path="a.md", text="A")
    for i in range(3):
        cur = fresh_db.execute(
            "INSERT INTO chunks(file_id, chunk_index, text) "
            "VALUES (?, ?, ?)",
            (fid, i + 1, f"S {i}"),
        )
        cid = cur.lastrowid
        fresh_db.execute(
            "INSERT INTO entities(chunk_id, text, text_lower, label) "
            "VALUES (?, ?, ?, 'PERSON')",
            (cid, "Sarah", "sarah"),
        )
    fresh_db.commit()
    n1 = people_mod.materialize_from_entities(fresh_db, min_mentions=2)
    n2 = people_mod.materialize_from_entities(fresh_db, min_mentions=2)
    assert n1 == 1
    assert n2 == 0


# ============================ link_after_index =======================

def test_link_after_index_bulk_links_all_chunks(fresh_db):
    """The indexer's per-file hook should re-scan every chunk so new
    aliases pick up old docs without a manual relink."""
    pid = people_mod.upsert_person(fresh_db, display_name="Sarah")
    people_mod.clear_alias_cache()
    cur = fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at, content_hash) "
        "VALUES ('x.md', ?, 100, 'document', ?, NULL)",
        (time.time(), time.time()),
    )
    fid = cur.lastrowid
    for i in range(3):
        fresh_db.execute(
            "INSERT INTO chunks(file_id, chunk_index, text) VALUES (?, ?, ?)",
            (fid, i, f"chunk {i}: Sarah said x"),
        )
    fresh_db.commit()
    people_mod.link_after_index(fresh_db, fid)
    rows = fresh_db.execute(
        "SELECT * FROM person_mentions WHERE person_id = ?", (pid,),
    ).fetchall()
    assert len(rows) == 3


def test_link_after_index_swallows_failure(fresh_db, monkeypatch):
    """A broken alias matcher shouldn't take down the indexer."""
    def boom(*a, **kw):
        raise RuntimeError("matcher crashed")
    monkeypatch.setattr(people_mod, "link_file_mentions", boom)
    # Should not raise.
    people_mod.link_after_index(fresh_db, 1)
