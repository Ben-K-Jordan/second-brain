"""Smoke tests for the connector registry + parser internals.

We deliberately don't hit any network here. The protocol checks are pure
shape verification; the parser tests use canned input fixtures that mirror
the real upstream payloads.
"""

from __future__ import annotations

from secondbrain.connectors import (
    USER_AGENT,
    Connector,
    ConnectorDocument,
    all_connectors,
    get_connector,
    respect_retry_after,
)


def test_registry_is_nonempty_and_unique():
    classes = all_connectors()
    assert len(classes) >= 13
    names = [cls().name for cls in classes]
    assert len(names) == len(set(names)), f"duplicate connector names: {names}"


def test_each_connector_implements_protocol(tmp_cfg):
    """Every registered connector should be runtime-Protocol compatible
    AND report disabled when the env is empty (with a few documented
    exceptions: browser, which scans for installed browser profiles, and
    chat_history, which reads our own DB and is always enabled)."""
    always_enabled = {"browser", "chat_history"}
    for cls in all_connectors():
        instance = cls()
        assert isinstance(instance, Connector), f"{cls.__name__} doesn't satisfy Connector"
        assert isinstance(instance.name, str) and instance.name
        if instance.name in always_enabled:
            continue
        assert instance.is_enabled(tmp_cfg) is False, (
            f"{instance.name} should be disabled when its env / config is empty"
        )


def test_get_connector_resolves_by_name():
    cls = get_connector("github")
    assert cls is not None and cls().name == "github"
    assert get_connector("does-not-exist") is None


def test_connector_document_required_fields():
    """Required fields without defaults; defaults wired correctly."""
    doc = ConnectorDocument(
        source="test", virtual_path="test://1", title="t",
        content="body", mtime=0.0,
    )
    assert doc.kind == "url"
    assert doc.metadata == {}


def test_user_agent_includes_version():
    from secondbrain import __version__

    assert __version__ in USER_AGENT


# --- respect_retry_after -------------------------------------------------

class _FakeResp:
    def __init__(self, status_code: int, headers: dict | None = None):
        self.status_code = status_code
        self.headers = headers or {}


def test_respect_retry_after_non_429_returns_false(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert respect_retry_after(_FakeResp(200)) is False
    assert respect_retry_after(_FakeResp(500)) is False


def test_respect_retry_after_honours_header(monkeypatch):
    """Retry-After value should be parsed and clamped."""
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    assert respect_retry_after(_FakeResp(429, {"Retry-After": "3"})) is True
    assert sleeps == [3.0]


def test_respect_retry_after_clamps_to_max(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    respect_retry_after(_FakeResp(429, {"Retry-After": "9999"}), max_wait=10.0)
    assert sleeps == [10.0]


def test_respect_retry_after_default_when_header_missing(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    respect_retry_after(_FakeResp(429))
    assert sleeps and sleeps[0] >= 0.5
