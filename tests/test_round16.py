"""Round 16 — feature tests (weekly letter, encrypted backup,
iMessage ingester, notifications, MCP chat tools, timeline view).

Each section is independently runnable; they share fixtures via
``conftest.py`` (fresh_db, tmp_cfg, fake_embedder).
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ============================ Phase B — weekly letter ===============


def _seed_basic_signals(conn):
    """Drop in a few rows via the real APIs so we hit the real schema."""
    from secondbrain import personal, tasks

    # One done-this-week + one lingering open task.
    now = time.time()
    tasks.add_manual(conn, "write the weekly recap")
    tid = conn.execute(
        "SELECT id FROM tasks WHERE text = 'write the weekly recap'",
    ).fetchone()["id"]
    # Mark done a day ago.
    conn.execute(
        "UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ?",
        (now - 86400, tid),
    )
    tasks.add_manual(conn, "call dentist")
    # Backdate so it counts as lingering.
    conn.execute(
        "UPDATE tasks SET created_at = ? WHERE text = 'call dentist'",
        (now - 10 * 86400,),
    )
    conn.commit()

    # Journal entry today.
    personal.upsert_journal(
        conn, mood=4, text="long week but felt productive",
    )


def test_assemble_signals_returns_structured_data(fresh_db):
    """Round 16 fix: signals must include tasks, journal, counts."""
    from secondbrain import weekly_letter

    _seed_basic_signals(fresh_db)
    sigs = weekly_letter.assemble_signals(fresh_db)
    d = sigs.to_dict()
    assert "week_start" in d and "week_end" in d
    assert d["counts"]["docs_indexed"] == 0  # nothing indexed
    assert d["tasks"]["completed_count"] >= 1
    assert any("dentist" in t for t in d["tasks"]["lingering"])
    assert d["journal"]
    assert d["journal_mood_avg"] == 4.0


def test_signals_redact_secrets_from_journal(fresh_db):
    """Round 16: journal text passes through redact_text — API keys
    in personal notes shouldn't reach Anthropic."""
    from secondbrain import personal, weekly_letter

    personal.upsert_journal(
        fresh_db, mood=3,
        text=(
            "wrote down sk-ant-api03-"
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        ),
    )
    sigs = weekly_letter.assemble_signals(fresh_db)
    journal_text = sigs.journal[0]["text"]
    assert "sk-ant-api03-AAAA" not in journal_text
    assert "[REDACTED:anthropic_key]" in journal_text


def test_generate_letter_falls_back_when_no_anthropic_key(
    fresh_db, tmp_cfg, monkeypatch,
):
    """No ANTHROPIC_API_KEY → stats-only fallback. Letter still gets
    a value — we never produce nothing."""
    from secondbrain import weekly_letter

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _seed_basic_signals(fresh_db)
    sigs = weekly_letter.assemble_signals(fresh_db)
    letter_md, model, cost = weekly_letter.generate_letter(
        tmp_cfg, fresh_db, sigs,
    )
    assert "Weekly review" in letter_md
    assert model.startswith("fallback")
    assert cost == 0.0


def test_generate_letter_calls_sonnet_when_key_present(
    fresh_db, tmp_cfg, monkeypatch,
):
    """With a key + stubbed SDK, the letter comes back as the
    fake-Sonnet response and audit is logged."""
    from secondbrain import weekly_letter

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test")
    _seed_basic_signals(fresh_db)

    captured: dict = {}

    class _FakeMessages:
        def create(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            captured["system"] = kwargs.get("system", "")
            captured["model"] = kwargs["model"]
            block = MagicMock()
            block.type = "text"
            block.text = "## Looking back\n\nReal letter from fake Sonnet."
            resp = MagicMock()
            resp.content = [block]
            resp.usage.input_tokens = 200
            resp.usage.output_tokens = 60
            return resp

    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic = lambda: type(
        "X", (), {"messages": _FakeMessages()},
    )()
    mock_anthropic.APIError = Exception

    sigs = weekly_letter.assemble_signals(fresh_db)
    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        letter_md, model, cost = weekly_letter.generate_letter(
            tmp_cfg, fresh_db, sigs,
        )
    assert "Real letter from fake Sonnet" in letter_md
    assert model == "claude-sonnet-4-5"
    # Cost may be 0.0 if the model isn't in the price catalog yet —
    # the important assertion is "we hit the SDK and got the response".
    assert cost >= 0
    assert "personal letter" in captured["system"].lower()


def test_generate_and_save_is_idempotent_per_week(
    fresh_db, tmp_cfg, monkeypatch,
):
    """Calling generate_and_save twice for the same week returns the
    SAME letter row (no duplicate INSERT, no double LLM spend).
    overwrite=True forces a fresh call."""
    from secondbrain import weekly_letter

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _seed_basic_signals(fresh_db)
    a = weekly_letter.generate_and_save(tmp_cfg, fresh_db)
    b = weekly_letter.generate_and_save(tmp_cfg, fresh_db)
    assert a.id == b.id
    c = weekly_letter.generate_and_save(
        tmp_cfg, fresh_db, overwrite=True,
    )
    assert c.id != a.id  # new row replaced the old
    rows = fresh_db.execute(
        "SELECT COUNT(*) AS n FROM weekly_letters",
    ).fetchone()["n"]
    assert rows == 1  # overwrite deletes then inserts


def test_run_weekly_letter_only_fires_on_sunday(
    fresh_db, tmp_cfg, monkeypatch,
):
    """run_weekly_letter_if_due returns None on non-Sundays.
    Mocks datetime.now to verify."""
    from secondbrain import weekly_letter

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _seed_basic_signals(fresh_db)

    class _FakeDT:
        @staticmethod
        def now():
            # Wednesday
            return datetime(2025, 1, 1, 9, 0)

    monkeypatch.setattr(weekly_letter, "datetime", _FakeDT)
    out = weekly_letter.run_weekly_letter_if_due(tmp_cfg, fresh_db)
    assert out is None

    class _FakeDTSunday:
        @staticmethod
        def now():
            return datetime(2025, 1, 5, 9, 0)  # Sunday

        @staticmethod
        def fromtimestamp(ts):
            return datetime.fromtimestamp(ts)

    monkeypatch.setattr(weekly_letter, "datetime", _FakeDTSunday)
    out = weekly_letter.run_weekly_letter_if_due(tmp_cfg, fresh_db)
    assert out is not None


# ============================ Phase E — encrypted backup ============


def test_backup_encrypt_decrypt_roundtrip(tmp_path):
    """Round 16: AES-GCM round-trip preserves bytes exactly."""
    from secondbrain import backup

    src = tmp_path / "src.bin"
    enc = tmp_path / "enc.age"
    dec = tmp_path / "dec.bin"
    payload = b"".join(bytes([i % 256]) for i in range(50_000))
    src.write_bytes(payload)
    backup.encrypt_file(src, enc, "passphrase-1")
    assert backup.is_encrypted_file(enc)
    backup.decrypt_file(enc, dec, "passphrase-1")
    assert dec.read_bytes() == payload


def test_backup_decrypt_wrong_passphrase_raises(tmp_path):
    from secondbrain import backup

    src = tmp_path / "src.bin"
    enc = tmp_path / "enc.age"
    dec = tmp_path / "dec.bin"
    src.write_bytes(b"hello world" * 100)
    backup.encrypt_file(src, enc, "right")
    with pytest.raises(backup.BadPassphraseError):
        backup.decrypt_file(enc, dec, "wrong")


def test_backup_decrypt_tampered_ciphertext_raises(tmp_path):
    """Flipping a single byte in ciphertext must fail MAC verify."""
    from secondbrain import backup

    src = tmp_path / "src.bin"
    enc = tmp_path / "enc.age"
    dec = tmp_path / "dec.bin"
    src.write_bytes(b"important payload")
    backup.encrypt_file(src, enc, "p")
    blob = bytearray(enc.read_bytes())
    # Flip a byte well inside the ciphertext (after header).
    blob[-50] ^= 0xFF
    enc.write_bytes(bytes(blob))
    with pytest.raises(backup.BadPassphraseError):
        backup.decrypt_file(enc, dec, "p")


def test_is_encrypted_file_distinguishes_sqlite(tmp_path):
    """A bare SQLite file is NOT mistaken for an encrypted backup."""
    from secondbrain import backup

    db = tmp_path / "x.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t (x)")
    conn.commit()
    conn.close()
    assert not backup.is_encrypted_file(db)


def test_backup_cli_roundtrip_unencrypted(tmp_path, monkeypatch):
    """`secondbrain backup` + `secondbrain restore` works end-to-end
    on a real (small) DB without encryption."""
    from typer.testing import CliRunner

    from secondbrain.cli import app
    from secondbrain.config import Config

    cfg = Config(data_dir=tmp_path / "data")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(str(cfg.db_path))
    src.execute("PRAGMA journal_mode = WAL")
    src.execute("CREATE TABLE marker(x TEXT)")
    src.execute("INSERT INTO marker VALUES ('alive')")
    src.commit()
    src.close()
    monkeypatch.setattr("secondbrain.cli.load_config", lambda: cfg)
    out = tmp_path / "bk.db"
    runner = CliRunner()
    r1 = runner.invoke(app, ["backup", str(out)])
    assert r1.exit_code == 0, r1.output
    assert out.exists()
    # Trash the original then restore.
    cfg.db_path.unlink()
    r2 = runner.invoke(app, ["restore", str(out), "--force"])
    assert r2.exit_code == 0, r2.output
    # Verify content survived.
    conn = sqlite3.connect(str(cfg.db_path))
    rows = conn.execute("SELECT x FROM marker").fetchall()
    conn.close()
    assert ("alive",) in rows


def test_backup_cli_encrypted_roundtrip(tmp_path, monkeypatch):
    """Encrypted backup + restore via the CLI, with passphrase from env."""
    from typer.testing import CliRunner

    from secondbrain.cli import app
    from secondbrain.config import Config

    cfg = Config(data_dir=tmp_path / "data")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(str(cfg.db_path))
    src.execute("CREATE TABLE secret(s TEXT)")
    src.execute("INSERT INTO secret VALUES ('classified')")
    src.commit()
    src.close()
    monkeypatch.setattr("secondbrain.cli.load_config", lambda: cfg)
    monkeypatch.setenv("SECONDBRAIN_BACKUP_PASSPHRASE", "test-pw-9000")
    out = tmp_path / "bk.age"
    runner = CliRunner()
    r1 = runner.invoke(app, ["backup", str(out), "--encrypt"])
    assert r1.exit_code == 0, r1.output
    from secondbrain import backup as backup_mod
    assert backup_mod.is_encrypted_file(out)
    cfg.db_path.unlink()
    r2 = runner.invoke(app, ["restore", str(out), "--force"])
    assert r2.exit_code == 0, r2.output
    conn = sqlite3.connect(str(cfg.db_path))
    rows = conn.execute("SELECT s FROM secret").fetchall()
    conn.close()
    assert ("classified",) in rows


# ============================ Phase D — iMessage ====================


def _build_fake_chat_db(path: Path) -> None:
    """Construct a minimal Apple chat.db schema with two threads."""
    conn = sqlite3.connect(str(path))
    conn.executescript("""
        CREATE TABLE handle (
            ROWID INTEGER PRIMARY KEY AUTOINCREMENT,
            id TEXT
        );
        CREATE TABLE chat (
            ROWID INTEGER PRIMARY KEY AUTOINCREMENT,
            display_name TEXT,
            style INTEGER
        );
        CREATE TABLE chat_handle_join (
            chat_id INTEGER,
            handle_id INTEGER
        );
        CREATE TABLE message (
            ROWID INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            attributedBody BLOB,
            is_from_me INTEGER,
            date INTEGER,
            cache_has_attachments INTEGER,
            handle_id INTEGER
        );
        CREATE TABLE chat_message_join (
            chat_id INTEGER,
            message_id INTEGER
        );
    """)
    # Two handles, one chat (1:1 with sarah, 5 messages).
    conn.execute("INSERT INTO handle(id) VALUES ('+15555550100')")
    conn.execute("INSERT INTO chat(display_name, style) VALUES ('Sarah', 45)")
    conn.execute("INSERT INTO chat_handle_join VALUES (1, 1)")
    # Apple stores nanoseconds since 2001. Seed 5 messages spanning 2 days.
    base_ns = (
        int((datetime.now() - timedelta(days=2)).timestamp() - 978307200)
        * 1_000_000_000
    )
    for i, (text, from_me, sender) in enumerate([
        ("hey can you grab milk", 0, 1),
        ("yep on it", 1, None),
        ("call you in 10", 0, 1),
        ("ok", 1, None),
        ("dinner at 7?", 0, 1),
    ]):
        ts = base_ns + i * 60_000_000_000
        conn.execute(
            "INSERT INTO message(text, is_from_me, date, "
            "cache_has_attachments, handle_id) VALUES (?, ?, ?, 0, ?)",
            (text, from_me, ts, sender),
        )
        conn.execute(
            "INSERT INTO chat_message_join(chat_id, message_id) VALUES (1, ?)",
            (i + 1,),
        )
    conn.commit()
    conn.close()


def test_imessage_connector_disabled_when_no_db(tmp_cfg):
    """Without a configured chat.db path, connector reports disabled."""
    from secondbrain.connectors.imessage import IMessageConnector
    assert IMessageConnector().is_enabled(tmp_cfg) is False


def test_imessage_connector_reads_threads(tmp_cfg, tmp_path, monkeypatch):
    """With a fake chat.db, the connector yields one ConnectorDocument
    per chat with messages rendered as Markdown."""
    from secondbrain.connectors.imessage import IMessageConnector

    chat_db = tmp_path / "chat.db"
    _build_fake_chat_db(chat_db)
    tmp_cfg.imessage_db_path = str(chat_db)

    docs = list(IMessageConnector().fetch(tmp_cfg))
    assert len(docs) == 1
    doc = docs[0]
    assert doc.source == "imessage"
    assert doc.virtual_path.startswith("imessage://chat-")
    assert doc.title == "Sarah"
    # Body must contain the conversation.
    assert "milk" in doc.content
    assert "**Me**" in doc.content
    assert doc.metadata["n_messages"] == 5


def test_imessage_apple_ts_conversion():
    """Sanity-check the Apple ns → Unix conversion."""
    from secondbrain.connectors.imessage import _apple_ts_to_unix
    # Jan 1, 2001 in Apple seconds = 0 → Unix 978307200.
    assert _apple_ts_to_unix(0) == 0.0  # null-ish guard
    assert abs(_apple_ts_to_unix(1) - 978307201) < 0.1
    # nanoseconds path
    apple_ns = 700_000_000_000_000_000  # 22 years × 1e9 ≈ 2023-ish
    out = _apple_ts_to_unix(apple_ns)
    assert out > 1_600_000_000  # post-2020


# ============================ Phase C — notifications ===============


def test_notifications_enqueue_is_idempotent(fresh_db):
    from secondbrain import notifications as nm
    n1 = nm.enqueue(
        fresh_db, key="x:1", kind="test",
        title="hello", body="b",
    )
    n2 = nm.enqueue(
        fresh_db, key="x:1", kind="test",
        title="hello", body="b",
    )
    assert n1 is True
    assert n2 is False
    assert nm.count_pending(fresh_db) == 1


def test_notifications_dismiss_marks_status(fresh_db):
    from secondbrain import notifications as nm
    nm.enqueue(fresh_db, key="x:1", kind="test", title="t")
    [n] = nm.list_pending(fresh_db)
    nm.mark_dismissed(fresh_db, n.id)
    assert nm.count_pending(fresh_db) == 0
    recent = nm.list_recent(fresh_db)
    assert recent[0].status == "dismissed"


def test_notifications_birthday_detector_fires_on_horizon(fresh_db):
    """A person with a birthday in the next 3 days should fire one
    'birthday' notification."""
    from secondbrain import notifications as nm
    from secondbrain import people as people_mod

    today = date.today()
    upcoming = (today + timedelta(days=2)).strftime("%m-%d")
    pid = people_mod.upsert_person(fresh_db, display_name="Alex")
    people_mod.set_field(fresh_db, pid, birthday=upcoming)
    n = nm._detect_birthdays(fresh_db)
    assert n == 1
    pend = nm.list_pending(fresh_db)
    assert any("Alex" in p.title for p in pend)


def test_notifications_journal_nudge_after_3_days(fresh_db, monkeypatch):
    """No journal entries → nudge fires (low urgency)."""
    from secondbrain import notifications as nm

    # journal_entries table is created lazily via personal._ensure_schema.
    # Force-create now by importing personal (touches the same module)
    # and calling a no-op against it.
    from secondbrain import personal
    personal.recent_journal(fresh_db, days=1)  # creates table if absent

    n = nm._detect_journal_nudge(fresh_db)
    assert n == 1
    [pend] = nm.list_pending(fresh_db)
    assert "journal" in pend.title.lower()
    assert pend.urgency == "low"
    # Re-running same week should not duplicate.
    again = nm._detect_journal_nudge(fresh_db)
    assert again == 0


def test_notifications_pop_to_tray_marks_shown(fresh_db):
    """Tray pop must call icon.notify and flip status to 'shown'."""
    from secondbrain import notifications as nm
    nm.enqueue(
        fresh_db, key="urg:1", kind="email_urgent", urgency="high",
        title="Urgent thing", body="open it",
    )
    nm.enqueue(
        fresh_db, key="low:1", kind="journal_nudge", urgency="low",
        title="hey", body="..",
    )
    fake_icon = MagicMock()
    n = nm.pop_to_tray(fresh_db, fake_icon)
    assert n == 1  # only the high-urgency one
    fake_icon.notify.assert_called_once()
    # The low-urgency stays pending; the high gets shown.
    pending = nm.list_pending(fresh_db)
    assert len(pending) == 1
    assert pending[0].urgency == "low"


# ============================ Phase F — MCP chat tools ==============


def test_mcp_list_chat_conversations(fresh_db, monkeypatch):
    """The MCP tool returns a markdown table of conversations from the DB."""
    from secondbrain import mcp_server
    from secondbrain.db import (
        chat_append_message,
        chat_create_conversation,
    )

    cid = chat_create_conversation(fresh_db, "Test conversation")
    chat_append_message(fresh_db, cid, "user", json.dumps("hello"))

    # Stub _get_state so the tool finds our DB.
    monkeypatch.setattr(
        mcp_server, "_get_state",
        lambda: (None, fresh_db, None, None),
    )
    out = mcp_server.list_chat_conversations(limit=5)
    assert "Test conversation" in out
    assert f"| {cid} |" in out


def test_mcp_get_chat_conversation_renders_messages(fresh_db, monkeypatch):
    from secondbrain import mcp_server
    from secondbrain.db import (
        chat_append_message,
        chat_create_conversation,
    )

    cid = chat_create_conversation(fresh_db, "x")
    chat_append_message(fresh_db, cid, "user", json.dumps("first"))
    chat_append_message(fresh_db, cid, "assistant", json.dumps("reply"))

    monkeypatch.setattr(
        mcp_server, "_get_state",
        lambda: (None, fresh_db, None, None),
    )
    out = mcp_server.get_chat_conversation(cid)
    assert "first" in out and "reply" in out
    assert "user" in out and "assistant" in out


def test_mcp_append_chat_message_persists(fresh_db, monkeypatch):
    from secondbrain import mcp_server
    from secondbrain.db import (
        chat_create_conversation,
        chat_get_messages,
    )

    cid = chat_create_conversation(fresh_db, "y")
    monkeypatch.setattr(
        mcp_server, "_get_state",
        lambda: (None, fresh_db, None, None),
    )
    res = mcp_server.append_chat_message(cid, "user", "from MCP")
    assert "OK" in res
    rows = chat_get_messages(fresh_db, cid)
    assert any("from MCP" in (r["content_json"] or "") for r in rows)


def test_mcp_append_rejects_bad_role(fresh_db, monkeypatch):
    from secondbrain import mcp_server
    from secondbrain.db import chat_create_conversation
    cid = chat_create_conversation(fresh_db, "z")
    monkeypatch.setattr(
        mcp_server, "_get_state",
        lambda: (None, fresh_db, None, None),
    )
    res = mcp_server.append_chat_message(cid, "system", "nope")
    assert "Bad role" in res


# ============================ Phase G — timeline ====================


def test_timeline_assembles_events_from_multiple_sources(fresh_db):
    from secondbrain import personal, tasks, timeline

    tasks.add_manual(fresh_db, "test task")
    personal.upsert_journal(fresh_db, mood=4, text="a thought")

    since, until = timeline.parse_window(None, days=1)
    events = timeline.assemble(fresh_db, since, until)
    kinds = {e.kind for e in events}
    assert "task_created" in kinds
    assert "journal" in kinds
    # Sorted descending by ts.
    for a, b in zip(events, events[1:], strict=False):
        assert a.ts >= b.ts


def test_timeline_filter_by_kind(fresh_db):
    from secondbrain import personal, tasks, timeline

    tasks.add_manual(fresh_db, "task")
    personal.upsert_journal(fresh_db, mood=3, text="j")
    since, until = timeline.parse_window(None, 1)
    only_tasks = timeline.assemble(
        fresh_db, since, until, kinds={"task"},
    )
    assert all(e.kind.startswith("task") for e in only_tasks)


def test_timeline_parse_window_default():
    from secondbrain import timeline
    since, until = timeline.parse_window(None, 1)
    assert until > since
    assert until - since <= 86400 + 1


def test_timeline_parse_window_explicit_date():
    from secondbrain import timeline
    since, until = timeline.parse_window("2025-04-12", 3)
    assert datetime.fromtimestamp(until).date() == date(2025, 4, 13)
    assert datetime.fromtimestamp(since).date() == date(2025, 4, 10)


# ============================ Smoke: dashboard routes ===============


def test_review_page_renders_when_no_letters(
    monkeypatch, tmp_path, fake_embedder,
):
    from fastapi.testclient import TestClient

    from secondbrain.config import Config
    from secondbrain.dashboard import create_app

    cfg = Config(data_dir=tmp_path / "data")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("secondbrain.dashboard.load_config", lambda: cfg)
    monkeypatch.setattr(
        "secondbrain.dashboard.make_embedder", lambda c: fake_embedder,
    )
    monkeypatch.setattr(
        "secondbrain.dashboard.make_reranker", lambda c: None,
    )
    client = TestClient(create_app())
    r = client.get("/review")
    assert r.status_code == 200
    assert "No letters yet" in r.text


def test_notifications_page_renders_empty(
    monkeypatch, tmp_path, fake_embedder,
):
    from fastapi.testclient import TestClient

    from secondbrain.config import Config
    from secondbrain.dashboard import create_app

    cfg = Config(data_dir=tmp_path / "data")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("secondbrain.dashboard.load_config", lambda: cfg)
    monkeypatch.setattr(
        "secondbrain.dashboard.make_embedder", lambda c: fake_embedder,
    )
    monkeypatch.setattr(
        "secondbrain.dashboard.make_reranker", lambda c: None,
    )
    client = TestClient(create_app())
    r = client.get("/notifications")
    assert r.status_code == 200
    assert "Inbox" in r.text


def test_timeline_page_renders_empty(
    monkeypatch, tmp_path, fake_embedder,
):
    from fastapi.testclient import TestClient

    from secondbrain.config import Config
    from secondbrain.dashboard import create_app

    cfg = Config(data_dir=tmp_path / "data")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("secondbrain.dashboard.load_config", lambda: cfg)
    monkeypatch.setattr(
        "secondbrain.dashboard.make_embedder", lambda c: fake_embedder,
    )
    monkeypatch.setattr(
        "secondbrain.dashboard.make_reranker", lambda c: None,
    )
    client = TestClient(create_app())
    r = client.get("/timeline")
    assert r.status_code == 200
    assert "Timeline" in r.text


def test_review_regenerate_csrf_blocked(
    monkeypatch, tmp_path, fake_embedder,
):
    """The /review/regenerate POST is same-origin guarded."""
    from fastapi.testclient import TestClient

    from secondbrain.config import Config
    from secondbrain.dashboard import create_app

    cfg = Config(data_dir=tmp_path / "data")
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("secondbrain.dashboard.load_config", lambda: cfg)
    monkeypatch.setattr(
        "secondbrain.dashboard.make_embedder", lambda c: fake_embedder,
    )
    monkeypatch.setattr(
        "secondbrain.dashboard.make_reranker", lambda c: None,
    )
    client = TestClient(create_app())
    r = client.post(
        "/review/regenerate",
        headers={"origin": "https://attacker.example"},
        follow_redirects=False,
    )
    assert r.status_code == 403


# ============================ MCP tool surfaces =====================


def test_mcp_weekly_review_tool_returns_letter(
    fresh_db, tmp_cfg, monkeypatch,
):
    """The weekly_review MCP tool returns the latest letter (or
    generates if none)."""
    from secondbrain import mcp_server, weekly_letter
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _seed_basic_signals(fresh_db)
    weekly_letter.generate_and_save(tmp_cfg, fresh_db)
    monkeypatch.setattr(
        mcp_server, "_get_state",
        lambda: (tmp_cfg, fresh_db, None, None),
    )
    out = mcp_server.weekly_review()
    assert "Weekly review" in out or "Looking back" in out


# ============================ pyproject deps ========================


def test_pyproject_declares_cryptography():
    """Round 16 (Phase E) — encrypted backup needs the cryptography lib."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    txt = pyproject.read_text(encoding="utf-8")
    start = txt.find("\ndependencies = [")
    closing = txt.find("\n]\n", start)
    deps = txt[start:closing + 3]
    assert '"cryptography' in deps


def test_imessage_connector_registered():
    """IMessageConnector is in the all_connectors registry so
    `secondbrain sync imessage` works."""
    from secondbrain.connectors import all_connectors
    names = [c().name for c in all_connectors()]
    assert "imessage" in names
