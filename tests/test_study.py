"""Phase 67 + 68: study mode + knowledge gap tests.

Coverage:
  - extract_course_code parser
  - materialize_cards uses stub generator (no LLM call in tests)
  - SM-2 grade math: ease bumps, intervals, reset on wrong
  - due_cards / weak_concepts queries
  - log_gap / is_weak_result rules + resolve_gap
"""

from __future__ import annotations

import time

from secondbrain import study

# ============================ course code =============================

def test_extract_course_code_canonical():
    assert study.extract_course_code("[BME 410] Lecture 3") == "BME410"
    assert study.extract_course_code("[CS-374] Notes") == "CS374"
    assert study.extract_course_code("[BIOMG1350] foo") == "BIOMG1350"


def test_extract_course_code_returns_empty_when_no_prefix():
    assert study.extract_course_code("Lecture 3") == ""
    assert study.extract_course_code("[meeting] standup") == ""
    assert study.extract_course_code("") == ""


# ============================ docs_pending_cards ======================

def _seed_doc(conn, *, path: str, title: str, body_extra: str = ""):
    n = time.time()
    cur = conn.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at, content_hash) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (path, n, 100, "document", n, None),
    )
    fid = cur.lastrowid
    body = f"# {title}\n\n{body_extra or 'Lecture content here.'}"
    conn.execute(
        "INSERT INTO chunks(file_id, chunk_index, text) VALUES (?, ?, ?)",
        (fid, 0, body),
    )
    conn.commit()
    return fid


def test_docs_pending_cards_finds_course_transcripts(fresh_db):
    fid_a = _seed_doc(
        fresh_db, path="transcript://plaud/abc",
        title="[BME 410] Tonotopic maps",
    )
    # Non-course doc should be excluded.
    _seed_doc(
        fresh_db, path="transcript://granola/x",
        title="[meeting] standup",
    )
    # Non-transcript doc should be excluded.
    _seed_doc(
        fresh_db, path="C:/notes/random.md",
        title="[BME 410] random",
    )
    out = study.docs_pending_cards(fresh_db)
    file_ids = [t[0] for t in out]
    assert fid_a in file_ids
    assert len(out) == 1


def test_docs_pending_cards_skips_docs_with_cards(fresh_db):
    fid = _seed_doc(
        fresh_db, path="transcript://plaud/x",
        title="[CS 374] Trees",
    )
    fresh_db.execute(
        "INSERT INTO study_cards"
        "(file_id, course_code, concept, question, answer, "
        " ease, interval_days, next_due_at, created_at) "
        "VALUES (?, 'CS374', 'tree', 'Q', 'A', 2.5, 0, ?, ?)",
        (fid, time.time(), time.time()),
    )
    fresh_db.commit()
    out = study.docs_pending_cards(fresh_db)
    assert all(t[0] != fid for t in out)


# ============================ materialize_cards =======================

def _stub_generator(title, body, n, cfg=None):
    """Return n deterministic cards. Used in tests instead of the
    real LLM-backed generator."""
    return [
        {
            "concept": f"concept-{i}",
            "question": f"What is concept {i} from {title}?",
            "answer": f"Answer {i} grounded in the body.",
        }
        for i in range(n)
    ]


def test_materialize_cards_inserts_n_cards(fresh_db):
    fid = _seed_doc(
        fresh_db, path="transcript://plaud/abc",
        title="[BME 410] Lecture 1",
    )
    n = study.materialize_cards(
        fresh_db, fid, cards_per_doc=5, generator=_stub_generator,
    )
    assert n == 5
    rows = fresh_db.execute(
        "SELECT * FROM study_cards WHERE file_id = ?", (fid,),
    ).fetchall()
    assert len(rows) == 5
    assert all(r["course_code"] == "BME410" for r in rows)


def test_materialize_cards_idempotent(fresh_db):
    """Re-running over the same doc must NOT double-insert."""
    fid = _seed_doc(
        fresh_db, path="transcript://plaud/abc",
        title="[BME 410] L1",
    )
    n1 = study.materialize_cards(
        fresh_db, fid, cards_per_doc=3, generator=_stub_generator,
    )
    n2 = study.materialize_cards(
        fresh_db, fid, cards_per_doc=3, generator=_stub_generator,
    )
    assert n1 == 3
    assert n2 == 0  # already at target


def test_materialize_cards_skips_chunkless_files(fresh_db):
    """A file row with no chunks shouldn't crash."""
    cur = fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, indexed_at, content_hash) "
        "VALUES ('x', ?, 0, 'document', ?, NULL)",
        (time.time(), time.time()),
    )
    fid = cur.lastrowid
    fresh_db.commit()
    assert study.materialize_cards(
        fresh_db, fid, generator=_stub_generator,
    ) == 0


def test_materialize_cards_handles_generator_failure(fresh_db):
    """A generator that raises shouldn't take down the daemon."""
    fid = _seed_doc(
        fresh_db, path="transcript://plaud/x",
        title="[BME 410] L1",
    )
    def boom(title, body, n, cfg=None):
        raise RuntimeError("LLM down")
    n = study.materialize_cards(fresh_db, fid, generator=boom)
    assert n == 0


def test_materialize_cards_skips_blank_questions(fresh_db):
    """Generator output with empty Q or A is filtered."""
    fid = _seed_doc(
        fresh_db, path="transcript://plaud/x",
        title="[BME 410] L1",
    )
    def half_blank(title, body, n, cfg=None):
        return [
            {"concept": "x", "question": "Real Q", "answer": "Real A"},
            {"concept": "y", "question": "", "answer": "no Q"},
            {"concept": "z", "question": "no A", "answer": ""},
        ]
    n = study.materialize_cards(
        fresh_db, fid, cards_per_doc=10, generator=half_blank,
    )
    assert n == 1


# ============================ SM-2 grading ============================

_seed_counter = {"n": 0}


def _seed_card(conn, *, ease=2.5, interval_days=0, course_code="BME410"):
    """Seed a doc + a card. Counter-based path so successive calls
    within one test (Windows time.time() = ~16ms resolution) don't
    collide on the unique-path constraint."""
    _seed_counter["n"] += 1
    fid = _seed_doc(
        conn, path=f"transcript://plaud/seed-{_seed_counter['n']}",
        title="[BME 410] L",
    )
    n = time.time()
    cur = conn.execute(
        "INSERT INTO study_cards"
        "(file_id, course_code, concept, question, answer, "
        " ease, interval_days, next_due_at, created_at) "
        "VALUES (?, ?, 'c', 'Q', 'A', ?, ?, ?, ?) RETURNING id",
        (fid, course_code, ease, interval_days,
         n - 1, n),  # due in the past
    )
    return int(cur.fetchone()["id"])


def test_grade_5_advances_interval(fresh_db):
    cid = _seed_card(fresh_db)
    updated = study.grade_card(fresh_db, cid, 5)
    assert updated.review_count == 1
    assert updated.correct_count == 1
    # First-time correct → 1-day interval.
    assert updated.interval_days >= 1.0
    assert updated.ease > 2.5  # bumped up


def test_grade_3_bare_pass_lowers_ease(fresh_db):
    cid = _seed_card(fresh_db)
    updated = study.grade_card(fresh_db, cid, 3)
    assert updated.review_count == 1
    assert updated.correct_count == 1
    assert updated.ease < 2.5  # bumped down slightly


def test_grade_0_resets_interval(fresh_db):
    cid = _seed_card(fresh_db, interval_days=14)
    updated = study.grade_card(fresh_db, cid, 0)
    assert updated.correct_count == 0
    assert updated.review_count == 1
    # Reset → next due in ~12h, interval stored as 0.5.
    assert updated.interval_days == 0.5
    assert updated.ease < 2.5


def test_grade_clamps_ease_at_minimum(fresh_db):
    """SM-2's ease floor is 1.3 — never go below."""
    cid = _seed_card(fresh_db, ease=1.4)
    # Several wrong grades.
    for _ in range(5):
        study.grade_card(fresh_db, cid, 0)
    card = study.get_card(fresh_db, cid)
    assert card.ease >= 1.3


def test_grade_unknown_card_returns_none(fresh_db):
    assert study.grade_card(fresh_db, 9999, 5) is None


def test_grade_clamps_grade_to_valid_range(fresh_db):
    """Out-of-range grades get clamped — 7 → 5, -2 → 0."""
    cid = _seed_card(fresh_db)
    out = study.grade_card(fresh_db, cid, 7)
    assert out is not None
    assert out.correct_count == 1  # treated as 5


def test_grade_subsequent_correct_uses_new_ease_on_interval(fresh_db):
    """After two consecutive 5s, the second interval should equal
    the first × the new ease (SM-2)."""
    cid = _seed_card(fresh_db)
    after_first = study.grade_card(fresh_db, cid, 5)
    # First correct → interval = 1.0 by our table.
    assert after_first.interval_days == 1.0
    after_second = study.grade_card(fresh_db, cid, 5)
    # Second: since prev (1.0) < 1.5, the table sets new = 6.0.
    assert after_second.interval_days == 6.0


# ============================ due_cards ===============================

def test_due_cards_returns_only_past_due(fresh_db):
    cid_due = _seed_card(fresh_db)
    # Force this card to be due IN THE FUTURE.
    fresh_db.execute(
        "UPDATE study_cards SET next_due_at = ? WHERE id = ?",
        (time.time() + 86400, cid_due),
    )
    fresh_db.commit()
    # Insert a second card that's due now.
    cid_now = _seed_card(fresh_db)
    rows = study.due_cards(fresh_db)
    ids = [c.id for c in rows]
    assert cid_now in ids
    assert cid_due not in ids


def test_due_cards_filters_by_course(fresh_db):
    _seed_card(fresh_db, course_code="BME410")
    _seed_card(fresh_db, course_code="CS374")
    rows = study.due_cards(fresh_db, course_code="CS374")
    assert all(c.course_code == "CS374" for c in rows)


# ============================ weak_concepts ===========================

def test_weak_concepts_orders_by_accuracy_ascending(fresh_db):
    cid_a = _seed_card(fresh_db)
    cid_b = _seed_card(fresh_db)
    fresh_db.execute(
        "UPDATE study_cards SET concept = 'A' WHERE id = ?", (cid_a,),
    )
    fresh_db.execute(
        "UPDATE study_cards SET concept = 'B' WHERE id = ?", (cid_b,),
    )
    fresh_db.commit()
    # A: 1/3 correct. B: 3/3 correct.
    fresh_db.execute(
        "UPDATE study_cards SET review_count = 3, correct_count = 1 "
        "WHERE id = ?", (cid_a,),
    )
    fresh_db.execute(
        "UPDATE study_cards SET review_count = 3, correct_count = 3 "
        "WHERE id = ?", (cid_b,),
    )
    fresh_db.commit()
    weak = study.weak_concepts(fresh_db)
    assert weak[0][0] == "A"  # weakest first
    assert weak[0][1] < weak[1][1]


def test_weak_concepts_filters_low_review_count(fresh_db):
    """Concepts with <3 reviews should be excluded — too few to be
    meaningful."""
    cid = _seed_card(fresh_db)
    fresh_db.execute(
        "UPDATE study_cards SET concept = 'A', "
        "review_count = 1, correct_count = 0 WHERE id = ?", (cid,),
    )
    fresh_db.commit()
    weak = study.weak_concepts(fresh_db)
    assert all(n >= 3 for _, _, n in weak)


# ============================ knowledge gaps ==========================

def test_is_weak_result_low_score_is_weak():
    assert study.is_weak_result(n_results=5, top_score=0.6)
    assert not study.is_weak_result(n_results=5, top_score=0.1)


def test_is_weak_result_few_results_is_weak():
    assert study.is_weak_result(n_results=0, top_score=0.05)
    assert study.is_weak_result(n_results=1, top_score=0.05)


def test_is_weak_result_none_score_is_weak():
    assert study.is_weak_result(n_results=5, top_score=None)


def test_log_gap_persists_weak_question(fresh_db):
    gid = study.log_gap(
        fresh_db, "what is X?", n_results=0, top_score=None,
    )
    assert gid is not None
    rows = fresh_db.execute(
        "SELECT * FROM knowledge_gaps WHERE id = ?", (gid,),
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["question"] == "what is X?"


def test_log_gap_skips_strong_results(fresh_db):
    """Don't pollute the gaps table when retrieval was actually fine."""
    gid = study.log_gap(
        fresh_db, "what is X?", n_results=10, top_score=0.05,
    )
    assert gid is None


def test_log_gap_skips_empty_question(fresh_db):
    assert study.log_gap(
        fresh_db, "", n_results=0, top_score=None,
    ) is None
    assert study.log_gap(
        fresh_db, "   ", n_results=0, top_score=None,
    ) is None


def test_list_gaps_excludes_resolved_by_default(fresh_db):
    gid = study.log_gap(fresh_db, "X", n_results=0, top_score=None)
    study.resolve_gap(fresh_db, gid)
    rows = study.list_gaps(fresh_db)
    assert all(g.id != gid for g in rows)


def test_list_gaps_includes_resolved_with_flag(fresh_db):
    gid = study.log_gap(fresh_db, "X", n_results=0, top_score=None)
    study.resolve_gap(fresh_db, gid)
    rows = study.list_gaps(fresh_db, include_resolved=True)
    assert any(g.id == gid for g in rows)


def test_resolve_gap_idempotent(fresh_db):
    gid = study.log_gap(fresh_db, "X", n_results=0, top_score=None)
    assert study.resolve_gap(fresh_db, gid) is True
    assert study.resolve_gap(fresh_db, gid) is False  # already resolved


def test_resolve_gap_with_note(fresh_db):
    gid = study.log_gap(fresh_db, "X", n_results=0, top_score=None)
    study.resolve_gap(fresh_db, gid, note="found in chapter 7")
    row = fresh_db.execute(
        "SELECT note FROM knowledge_gaps WHERE id = ?", (gid,),
    ).fetchone()
    assert row["note"] == "found in chapter 7"


# ============================ daemon hook =============================

def test_materialize_due_cards_caps_per_tick(fresh_db, monkeypatch):
    """The daemon hook must respect docs_per_tick so a large backlog
    materialises gradually."""
    # Seed 5 course docs.
    for i in range(5):
        _seed_doc(
            fresh_db, path=f"transcript://plaud/cap-{i}",
            title=f"[BME 410] Lecture {i}",
        )
    # Replace materialize_cards with a stub that always inserts 5
    # records — bypasses the generator + LLM path.
    original = study.materialize_cards
    def stub_mat(conn, fid, *, cfg=None, cards_per_doc=5, generator=None):
        return original(
            conn, fid, cfg=cfg, cards_per_doc=cards_per_doc,
            generator=_stub_generator,
        )
    monkeypatch.setattr(study, "materialize_cards", stub_mat)
    n = study.materialize_due_cards(fresh_db, cfg=None, docs_per_tick=2)
    # 2 docs × 5 cards = 10 cards.
    assert n == 10


def test_materialize_due_cards_stops_when_caught_up(fresh_db, monkeypatch):
    """When all course docs already have cards, return 0 immediately."""
    monkeypatch.setattr(study, "_default_generator", _stub_generator)
    n = study.materialize_due_cards(fresh_db, cfg=None)
    assert n == 0
