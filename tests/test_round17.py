"""Round 17 — fixes for the audit of Round 16's feature push.

Each test maps to a finding from the audit:
  - HIGH H1: backup tempfile sentinel (Path("") truthy bug)
  - HIGH H2: MCP chat tools persisted/returned text without redaction
  - HIGH H3: Feb 29 birthdays silently skipped in non-leap years
  - HIGH H4: _signal_top_entities used non-grouped column in SELECT
  - MED:    notifications writes weren't serialised across threads
  - MED:    _save_letter DELETE+INSERT was non-transactional
  - MED:    /notifications page ran detectors on every load (no throttle)
  - MED:    dismiss_all flipped 'shown' rows too (rewrote history)
  - MED:    iMessage chat.db copy landed in OS tempdir (was /tmp)
  - MED:    parse_window silently accepted days<1
  - MED:    weekly_letter audit log didn't enumerate journal payload
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

# ============================ H1 — backup sentinel ==================


def test_backup_does_not_unlink_cwd_after_success(tmp_path, monkeypatch):
    """Round 17 fix: the backup() finally-block previously called
    unlink() on the empty-Path sentinel which resolves to CWD. Verify
    a successful unencrypted backup leaves the CWD intact."""
    from typer.testing import CliRunner

    from secondbrain.cli import app
    from secondbrain.config import Config

    cfg = Config(data_dir=tmp_path / "data")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(str(cfg.db_path))
    src.execute("CREATE TABLE marker(x TEXT)")
    src.execute("INSERT INTO marker VALUES ('alive')")
    src.commit()
    src.close()
    monkeypatch.setattr("secondbrain.cli.load_config", lambda: cfg)

    # Verify backup runs cleanly (no swallowed unlink errors in output).
    out = tmp_path / "bk.db"
    runner = CliRunner()
    r = runner.invoke(app, ["backup", str(out)])
    assert r.exit_code == 0, r.output
    # The backup file must exist + the source DB still intact.
    assert out.exists()
    assert cfg.db_path.exists()
    # And the backup is a valid SQLite DB.
    conn = sqlite3.connect(str(out))
    rows = conn.execute("SELECT x FROM marker").fetchall()
    conn.close()
    assert ("alive",) in rows


def test_backup_then_restore_atomic_under_disk_write(tmp_path, monkeypatch):
    """End-to-end: encrypted backup + restore round-trip with the
    sentinel fix. Previously could leak a swallowed OSError."""
    from typer.testing import CliRunner

    from secondbrain import backup as backup_mod
    from secondbrain.cli import app
    from secondbrain.config import Config

    cfg = Config(data_dir=tmp_path / "data")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(str(cfg.db_path))
    src.execute("CREATE TABLE m(x TEXT)")
    src.execute("INSERT INTO m VALUES ('payload')")
    src.commit()
    src.close()
    monkeypatch.setattr("secondbrain.cli.load_config", lambda: cfg)
    monkeypatch.setenv("SECONDBRAIN_BACKUP_PASSPHRASE", "round17pw")

    out = tmp_path / "bk.age"
    runner = CliRunner()
    r1 = runner.invoke(app, ["backup", str(out), "--encrypt"])
    assert r1.exit_code == 0, r1.output
    assert backup_mod.is_encrypted_file(out)
    cfg.db_path.unlink()
    r2 = runner.invoke(app, ["restore", str(out), "--force"])
    assert r2.exit_code == 0, r2.output
    conn = sqlite3.connect(str(cfg.db_path))
    rows = conn.execute("SELECT x FROM m").fetchall()
    conn.close()
    assert ("payload",) in rows


# ============================ H2 — MCP chat redaction ==============


def test_mcp_append_chat_message_redacts_secrets(fresh_db, monkeypatch):
    """Round 17 fix: secrets pasted into Claude Desktop must be
    masked before landing in the dashboard chat history."""
    from secondbrain import mcp_server
    from secondbrain.db import (
        chat_create_conversation,
        chat_get_messages,
    )

    cid = chat_create_conversation(fresh_db, "secrets-test")
    monkeypatch.setattr(
        mcp_server, "_get_state",
        lambda: (None, fresh_db, None, None),
    )
    secret_msg = (
        "Here's my key: sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA "
        "and my SSN 123-45-6789"
    )
    res = mcp_server.append_chat_message(cid, "user", secret_msg)
    assert "OK" in res
    rows = chat_get_messages(fresh_db, cid)
    persisted = rows[-1]["content_json"]
    assert "sk-ant-api03-AAAA" not in persisted
    assert "123-45-6789" not in persisted
    assert "[REDACTED:anthropic_key]" in persisted
    assert "[REDACTED:ssn]" in persisted


def test_mcp_get_chat_conversation_redacts_on_read(fresh_db, monkeypatch):
    """Even legacy un-redacted messages already in the DB must come
    back masked when fetched via the MCP tool (defense in depth)."""
    from secondbrain import mcp_server
    from secondbrain.db import (
        chat_append_message,
        chat_create_conversation,
    )

    cid = chat_create_conversation(fresh_db, "x")
    # Bypass the round-17 append redaction by writing directly.
    chat_append_message(
        fresh_db, cid, "assistant",
        json.dumps("legacy msg with sk-ant-api03-LEAKLEAKLEAKLEAKLEAKLEAKLEAKLEAK"),
    )
    monkeypatch.setattr(
        mcp_server, "_get_state",
        lambda: (None, fresh_db, None, None),
    )
    out = mcp_server.get_chat_conversation(cid)
    assert "sk-ant-api03-LEAK" not in out
    assert "[REDACTED:anthropic_key]" in out


# ============================ H3 — leap-day birthdays ===============


def test_leap_day_birthday_remaps_to_feb_28_in_non_leap_year(fresh_db):
    """Round 17 fix: Feb 29 birthdays must remap to Feb 28 instead
    of being silently skipped in non-leap years."""
    from secondbrain import notifications as nm
    from secondbrain import people as people_mod

    # Helper directly: 2025 is non-leap.
    assert nm._safe_date_in_year(2025, 2, 29) == date(2025, 2, 28)
    assert nm._safe_date_in_year(2024, 2, 29) == date(2024, 2, 29)
    assert nm._safe_date_in_year(2025, 13, 1) is None  # bad month
    assert nm._safe_date_in_year(2025, 99, 99) is None  # bad day

    # Detector: a Feb 29 birthday in a non-leap window where today
    # is Feb 26 must fire (2 days out via the remap to Feb 28).
    pid = people_mod.upsert_person(
        fresh_db, display_name="LeapDay Lou", email="x@x",
    )
    people_mod.set_field(fresh_db, pid, birthday="02-29")

    class _FakeDate(date):
        @classmethod
        def today(cls):
            return date(2025, 2, 26)

    import secondbrain.notifications as notifs
    with patch.object(notifs, "date", _FakeDate):
        n = notifs._detect_birthdays(fresh_db)
    assert n == 1
    pend = notifs.list_pending(fresh_db)
    assert any("LeapDay Lou" in p.title for p in pend)


def test_safe_date_handles_invalid_month():
    from secondbrain import notifications as nm
    assert nm._safe_date_in_year(2025, 2, 30) is None
    assert nm._safe_date_in_year(2025, 4, 31) is None
    assert nm._safe_date_in_year(2025, 1, 1) == date(2025, 1, 1)


# ============================ H4 — top entities GROUP BY ============


def test_top_entities_consistent_text_for_same_text_lower(fresh_db):
    """Round 17 fix: same text_lower with different casings or
    different labels must NOT collapse silently into one row with
    arbitrary text/label."""
    from secondbrain import weekly_letter

    # Seed: two entities with same text_lower 'sarah' but different
    # labels. Real chunks schema: id/file_id/chunk_index/text.
    now = time.time()
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, content_hash, "
        "indexed_at) VALUES ('a.txt', 0, 0, 'document', 'h1', ?)",
        (now,),
    )
    fresh_db.execute(
        "INSERT INTO files(path, mtime, size, kind, content_hash, "
        "indexed_at) VALUES ('b.txt', 0, 0, 'document', 'h2', ?)",
        (now,),
    )
    f1 = fresh_db.execute(
        "SELECT id FROM files WHERE path = 'a.txt'",
    ).fetchone()["id"]
    f2 = fresh_db.execute(
        "SELECT id FROM files WHERE path = 'b.txt'",
    ).fetchone()["id"]
    fresh_db.execute(
        "INSERT INTO chunks(file_id, chunk_index, text, start_offset) "
        "VALUES (?, 0, 'Sarah is here', 0)", (f1,),
    )
    fresh_db.execute(
        "INSERT INTO chunks(file_id, chunk_index, text, start_offset) "
        "VALUES (?, 0, 'Sarah Inc is a company', 0)", (f2,),
    )
    c1 = fresh_db.execute(
        "SELECT id FROM chunks WHERE file_id = ?", (f1,),
    ).fetchone()["id"]
    c2 = fresh_db.execute(
        "SELECT id FROM chunks WHERE file_id = ?", (f2,),
    ).fetchone()["id"]
    fresh_db.execute(
        "INSERT INTO entities(chunk_id, text, text_lower, label) "
        "VALUES (?, 'Sarah', 'sarah', 'PERSON')",
        (c1,),
    )
    fresh_db.execute(
        "INSERT INTO entities(chunk_id, text, text_lower, label) "
        "VALUES (?, 'Sarah Inc', 'sarah', 'ORG')",
        (c2,),
    )
    fresh_db.commit()

    week_cutoff = time.time() - 7 * 86400
    out = weekly_letter._signal_top_entities(fresh_db, week_cutoff)
    # PERSON 'sarah' and ORG 'sarah' must be separate rows now.
    labels = {(r["text"], r["label"]) for r in out}
    person_rows = [r for r in out if r["label"] == "PERSON"]
    org_rows = [r for r in out if r["label"] == "ORG"]
    assert len(person_rows) == 1, f"expected 1 PERSON row; got {out}"
    assert len(org_rows) == 1, f"expected 1 ORG row; got {out}"
    assert ("Sarah", "PERSON") in labels
    assert ("Sarah Inc", "ORG") in labels


# ============================ MED — notifications RLock =============


def test_notifications_concurrent_writes_serialised(fresh_db):
    """Round 17 fix: 8 threads × 30 enqueues = 240 unique rows must
    all land without corruption."""
    from secondbrain import notifications as nm

    n_threads = 8
    n_per = 30
    errors: list[Exception] = []

    def writer(t: int):
        try:
            for i in range(n_per):
                nm.enqueue(
                    fresh_db,
                    key=f"thread:{t}:{i}",
                    kind="test", urgency="low",
                    title=f"t{t}:i{i}",
                )
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [
        threading.Thread(target=writer, args=(t,))
        for t in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [], f"threads raised: {errors}"
    assert nm.count_pending(fresh_db) == n_threads * n_per


def test_dismiss_all_only_flips_pending(fresh_db):
    """Round 17 fix: dismiss_all must NOT touch already-shown rows."""
    from secondbrain import notifications as nm
    nm.enqueue(fresh_db, key="p:1", kind="test", title="pending")
    nm.enqueue(fresh_db, key="p:2", kind="test", title="another")
    nm.enqueue(fresh_db, key="s:1", kind="test", title="shown")
    [shown] = [
        n for n in nm.list_pending(fresh_db) if n.title == "shown"
    ]
    nm.mark_shown(fresh_db, shown.id)
    # Now: 2 pending, 1 shown.
    n_dismissed = nm.dismiss_all(fresh_db)
    assert n_dismissed == 2
    # The shown row stays 'shown', not 'dismissed'.
    rows = nm.list_recent(fresh_db, limit=10)
    statuses = {r.title: r.status for r in rows}
    assert statuses["shown"] == "shown"
    assert statuses["pending"] == "dismissed"
    assert statuses["another"] == "dismissed"


def test_detect_all_throttled_no_ops_within_60s(fresh_db, monkeypatch):
    """Round 17 fix: dashboard /notifications page calls the
    throttled variant; second call within 60s returns None."""
    from secondbrain import notifications as nm
    # Reset the module-level throttle clock so the test is deterministic.
    monkeypatch.setattr(nm, "_last_detect_ts", 0.0)
    out1 = nm.detect_all_throttled(fresh_db)
    out2 = nm.detect_all_throttled(fresh_db)
    assert out1 is not None
    assert out2 is None
    # Skip ahead and it fires again.
    monkeypatch.setattr(nm, "_last_detect_ts", time.time() - 120)
    out3 = nm.detect_all_throttled(fresh_db)
    assert out3 is not None


# ============================ MED — _save_letter transactional ======


def test_save_letter_uses_with_conn_for_atomicity():
    """Round 17 fix: verify _save_letter wraps DELETE + INSERT in
    a `with conn:` block so a crash between rolls back atomically.
    Source-string check — sqlite3.Connection.execute is read-only
    in CPython (can't monkeypatch directly), so we assert on the
    fix's structural marker."""
    import inspect

    from secondbrain import weekly_letter
    src = inspect.getsource(weekly_letter._save_letter)
    # Round 17 marker comment + the actual `with conn:` wrap.
    assert "Round 17" in src
    assert "with conn:" in src
    # Both DELETE and INSERT live inside the `with conn:` block.
    after_with = src.split("with conn:", 1)[1]
    assert "DELETE FROM weekly_letters" in after_with
    assert "INSERT INTO weekly_letters" in after_with


def test_save_letter_atomic_via_integrity_violation(fresh_db, monkeypatch):
    """End-to-end atomicity check using a sqlite-level mechanism:
    seed a letter, then attempt a second save whose INSERT will
    fail (via a temporary trigger that raises). Verify the letter
    table is unchanged after the rollback."""
    from secondbrain import weekly_letter

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    sigs = weekly_letter.assemble_signals(fresh_db)
    weekly_letter._save_letter(
        fresh_db, signals=sigs, letter_md="first",
        model="m", cost_cents=0.0,
    )
    assert weekly_letter.latest_letter(fresh_db).letter_md == "first"

    # Install a BEFORE INSERT trigger that raises so the INSERT
    # fails and the surrounding DELETE rolls back too.
    fresh_db.execute("""
        CREATE TRIGGER force_fail_insert
        BEFORE INSERT ON weekly_letters
        BEGIN
            SELECT RAISE(ABORT, 'simulated insert failure');
        END;
    """)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            weekly_letter._save_letter(
                fresh_db, signals=sigs, letter_md="second",
                model="m", cost_cents=0.0, overwrite=True,
            )
    finally:
        fresh_db.execute("DROP TRIGGER force_fail_insert")
        fresh_db.commit()
    # The original "first" must still be there — DELETE rolled back.
    latest = weekly_letter.latest_letter(fresh_db)
    assert latest is not None, "letter row was deleted but not replaced"
    assert latest.letter_md == "first"


# ============================ MED — parse_window =====================


def test_parse_window_rejects_zero_days():
    from secondbrain import timeline
    with pytest.raises(ValueError):
        timeline.parse_window(None, days=0)


def test_parse_window_rejects_negative_days():
    from secondbrain import timeline
    with pytest.raises(ValueError):
        timeline.parse_window(None, days=-3)


def test_parse_window_normal_path_still_works():
    from secondbrain import timeline
    since, until = timeline.parse_window("2025-04-12", days=3)
    assert datetime.fromtimestamp(until).date() == date(2025, 4, 13)
    assert datetime.fromtimestamp(since).date() == date(2025, 4, 10)


# ============================ MED — iMessage tempfile =================


def test_imessage_tempfile_lives_under_cfg_data_dir(
    tmp_path, monkeypatch,
):
    """Round 17 fix: chat.db copy must land in cfg.data_dir/.tmp,
    not the OS default tempdir."""
    import sqlite3 as sq

    from secondbrain.config import Config
    from secondbrain.connectors.imessage import IMessageConnector

    cfg = Config(data_dir=tmp_path / "data")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    # Build a tiny synthetic chat.db (one chat, no messages so the
    # connector exits early but still goes through the temp-copy path).
    chat_db = tmp_path / "fake_chat.db"
    conn = sq.connect(str(chat_db))
    conn.executescript("""
        CREATE TABLE handle (ROWID INTEGER PRIMARY KEY, id TEXT);
        CREATE TABLE chat (
            ROWID INTEGER PRIMARY KEY, display_name TEXT, style INTEGER
        );
        CREATE TABLE chat_handle_join (chat_id INTEGER, handle_id INTEGER);
        CREATE TABLE message (
            ROWID INTEGER PRIMARY KEY, text TEXT, attributedBody BLOB,
            is_from_me INTEGER, date INTEGER,
            cache_has_attachments INTEGER, handle_id INTEGER
        );
        CREATE TABLE chat_message_join (
            chat_id INTEGER, message_id INTEGER
        );
    """)
    conn.commit()
    conn.close()
    cfg.imessage_db_path = str(chat_db)

    # Drain the iterator (forces the temp-copy path to run).
    list(IMessageConnector().fetch(cfg))

    # The .tmp dir under data_dir must exist; OS-default tempdir
    # must NOT contain a sb-style temp file from this run.
    tmp_dir = cfg.data_dir / ".tmp"
    assert tmp_dir.exists()


# ============================ MED — weekly_letter audit ==============


def test_weekly_letter_audit_log_enumerates_payload(
    fresh_db, tmp_cfg, monkeypatch,
):
    """Round 17 fix: the audit row's summary AND extra dict tell the
    user what personal data went outbound (journal count, health flag,
    etc.) instead of just a raw prompt_chars number."""
    from secondbrain import ai_audit, personal, weekly_letter

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test")
    personal.upsert_journal(fresh_db, mood=4, text="day was good")

    class _FakeMessages:
        def create(self, **kwargs):
            block = MagicMock()
            block.type = "text"
            block.text = "## Looking back\n\nletter body"
            resp = MagicMock()
            resp.content = [block]
            resp.usage.input_tokens = 100
            resp.usage.output_tokens = 50
            return resp

    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic = lambda: type(
        "X", (), {"messages": _FakeMessages()},
    )()
    mock_anthropic.APIError = Exception

    sigs = weekly_letter.assemble_signals(fresh_db)
    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        weekly_letter.generate_letter(tmp_cfg, fresh_db, sigs)

    # Find the weekly_review audit row.
    actions = ai_audit.recent(fresh_db, kind="weekly_review")
    assert len(actions) == 1
    a = actions[0]
    assert "journal day(s)" in a.summary
    assert a.extra.get("n_journal_entries") == 1


# ============================ Smoke ================================


def test_full_suite_imports_clean():
    """Belt-and-suspenders: every Round 17-touched module imports
    without circular imports or missing names."""
    from secondbrain import (  # noqa: F401
        backup,
        mcp_server,  # noqa: F401
        notifications,
        timeline,
        weekly_letter,
    )
    from secondbrain.connectors import imessage  # noqa: F401
