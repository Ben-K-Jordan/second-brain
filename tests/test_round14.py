"""Round 14 — fixes for the third external security audit.

Each test maps to a finding from the audit:
  - HIGH H1: study.py sent lecture body to Anthropic unredacted
  - HIGH H2: indexer URL fetch had no SSRF guard
  - MEDIUM M1: 11 dashboard POST routes had no CSRF same-origin guard
  - MEDIUM M2: health-check stored 12 chars of API key (now SHA fingerprint)
  - LOW L1: email_assist triage/analyze/draft prompts sent from_/subject raw
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from secondbrain import health_checks, indexer
from secondbrain.indexer import UnsafeURLError, _assert_url_is_public

# ============================ H1 — study.py redaction ===============

def test_study_default_generator_redacts_body(
    fresh_db, tmp_cfg, monkeypatch,
):
    """Round 14 fix: lecture body must go through _safe_for_prompt
    before leaving for Anthropic. Previously sent raw."""
    from secondbrain import study
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test")

    captured: dict = {}

    class _FakeMessages:
        def create(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            resp = MagicMock()
            resp.usage.input_tokens = 100
            resp.usage.output_tokens = 50
            block = MagicMock()
            block.type = "text"
            block.text = "[]"
            resp.content = [block]
            return resp

    class _FakeClient:
        def __init__(self):
            self.messages = _FakeMessages()

    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic = lambda: _FakeClient()
    mock_anthropic.APIError = Exception

    body_with_secret = (
        "Today's lecture covered RNA polymerase. "
        "My API key is sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA "
        "and my SSN is 123-45-6789. "
        "The transcription factor binds the promoter."
    )
    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        study._default_generator(
            "Bio Lecture 4", body_with_secret, n=3, cfg=tmp_cfg,
        )
    sent = captured["messages"][0]["content"]
    # Redaction tokens should appear in place of secrets.
    assert "[REDACTED:anthropic_key]" in sent
    assert "[REDACTED:ssn]" in sent
    # Raw secret bytes must not have left.
    assert "sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" not in sent
    assert "123-45-6789" not in sent


# ============================ H2 — SSRF guard =======================

def test_ssrf_rejects_loopback_literal():
    with pytest.raises(UnsafeURLError):
        _assert_url_is_public("http://127.0.0.1/admin")


def test_ssrf_rejects_loopback_hostname():
    with pytest.raises(UnsafeURLError):
        _assert_url_is_public("http://localhost/x")


def test_ssrf_rejects_rfc1918():
    for host in ("10.0.0.1", "192.168.1.1", "172.16.0.1"):
        with pytest.raises(UnsafeURLError):
            _assert_url_is_public(f"http://{host}/")


def test_ssrf_rejects_link_local_imds():
    """169.254.169.254 is the AWS / GCP / Azure cloud-metadata
    address — the canonical SSRF target."""
    with pytest.raises(UnsafeURLError):
        _assert_url_is_public("http://169.254.169.254/latest/meta-data/")


def test_ssrf_rejects_file_scheme():
    with pytest.raises(UnsafeURLError):
        _assert_url_is_public("file:///etc/passwd")


def test_ssrf_rejects_ftp_scheme():
    with pytest.raises(UnsafeURLError):
        _assert_url_is_public("ftp://example.com/")


def test_ssrf_rejects_no_host():
    with pytest.raises(UnsafeURLError):
        _assert_url_is_public("http:///no-host")


def test_ssrf_accepts_real_public_host(monkeypatch):
    """Public DNS resolution should pass. Stub getaddrinfo to a
    public IP so the test is deterministic + offline."""
    import socket
    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda host, port: [(0, 0, 0, "", ("93.184.216.34", 0))],
    )
    # Should not raise.
    _assert_url_is_public("https://example.com/path")


def test_ssrf_blocks_dns_to_private_ip(monkeypatch):
    """If a public-looking hostname resolves to a private IP, we
    must reject — DNS rebinding / poisoning style attack."""
    import socket
    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda host, port: [(0, 0, 0, "", ("10.0.0.5", 0))],
    )
    with pytest.raises(UnsafeURLError):
        _assert_url_is_public("https://innocent-looking.com/")


def test_ssrf_rejects_redirect_to_private_ip(monkeypatch):
    """A 302 → http://127.0.0.1/ must be blocked even when the
    initial URL was public."""
    import socket

    import requests
    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda host, port: [(0, 0, 0, "", ("93.184.216.34", 0))],
    )

    call_count = {"n": 0}

    def fake_get(url, **kw):
        call_count["n"] += 1
        resp = MagicMock()
        resp.headers = {"location": "http://127.0.0.1:8080/admin"}
        resp.status_code = 302
        resp.close = MagicMock()
        return resp

    # `requests` is imported inside _fetch_url_to_tempfile, so patch
    # at the package level — that's what the function-local import
    # binds to.
    monkeypatch.setattr(requests, "get", fake_get)
    with pytest.raises(UnsafeURLError):
        indexer._fetch_url_to_tempfile("https://example.com/")
    # The request actually hit the server (redirect), but the second
    # hop's pre-flight rejected it.
    assert call_count["n"] == 1


# ============================ M1 — CSRF guards ======================

def _client_with_dashboard(monkeypatch, tmp_path, fake_embedder):
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
    return cfg, TestClient(create_app())


# 12 routes * (cross-origin → 403, no header → 403) = quick sweep.
@pytest.mark.parametrize(
    "method,url,data",
    [
        ("post", "/chat/1/system_prompt", {"system_prompt": "x"}),
        ("post", "/chat/1/delete", {}),
        ("post", "/api/chat/message", {"message": "hi"}),
        ("post", "/tasks/add", {"text": "x"}),
        ("post", "/tasks/1/done", {}),
        ("post", "/habits/1/checkin", {}),
        ("post", "/journal/add", {"text": "x", "mood": 3}),
        ("post", "/drafts/1/sent", {}),
        ("post", "/drafts/1/discard", {}),
        ("post", "/thanks/1/context", {"text": "x"}),
        ("post", "/thanks/1/skip", {}),
        ("post", "/thanks/1/draft", {}),
    ],
)
def test_csrf_guard_blocks_cross_origin(
    monkeypatch, tmp_path, fake_embedder, method, url, data,
):
    """Each round-14-guarded POST returns 403 when the request
    carries a cross-origin Origin header (no localhost Referer)."""
    _, client = _client_with_dashboard(monkeypatch, tmp_path, fake_embedder)
    r = client.request(
        method.upper(), url, data=data,
        headers={"origin": "https://evil.com"},
        follow_redirects=False,
    )
    assert r.status_code == 403, (
        f"{method.upper()} {url} should be CSRF-blocked; got {r.status_code}"
    )


def test_csrf_helper_recognises_localhost(
    monkeypatch, tmp_path, fake_embedder,
):
    """Sanity: a localhost Referer is accepted (so the guard isn't
    just blanket-blocking)."""
    from secondbrain.dashboard import _is_same_origin_request

    class _Req:
        def __init__(self, headers):
            self.headers = headers

    assert _is_same_origin_request(
        _Req({"referer": "http://127.0.0.1:8765/foo"}),
    )
    assert _is_same_origin_request(
        _Req({"origin": "http://localhost:8765"}),
    )
    assert not _is_same_origin_request(_Req({}))
    assert not _is_same_origin_request(
        _Req({"origin": "https://attacker.example"}),
    )


# ============================ M2 — key fingerprint ==================

def test_anthropic_key_extras_uses_fingerprint(tmp_cfg, monkeypatch):
    """Round 14: the extras dict must NOT contain raw key prefix."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-SECRET-SUFFIX-XYZ")
    ok, _err, extra = health_checks.check_anthropic_key(tmp_cfg)
    assert ok is True
    assert "key_prefix" not in extra
    fingerprint = extra["key_fingerprint"]
    # 8-hex-char SHA truncation, no key bytes.
    assert len(fingerprint) == 8
    assert all(c in "0123456789abcdef" for c in fingerprint)
    assert "sk-ant" not in fingerprint
    assert "SECRET" not in fingerprint


def test_voyage_key_extras_uses_fingerprint(tmp_cfg, monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-SECRET-VALUE-12345")
    tmp_cfg.voyage_api_key = ""
    ok, _err, extra = health_checks.check_voyage_key(tmp_cfg)
    assert ok is True
    assert "key_prefix" not in extra
    fingerprint = extra["key_fingerprint"]
    assert len(fingerprint) == 8
    assert "pa-" not in fingerprint
    assert "SECRET" not in fingerprint


def test_key_fingerprint_is_deterministic_and_unique(tmp_cfg, monkeypatch):
    """Same key → same fingerprint; different keys → different ones."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-AAA")
    _, _, extra1 = health_checks.check_anthropic_key(tmp_cfg)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-AAA")
    _, _, extra2 = health_checks.check_anthropic_key(tmp_cfg)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-BBB")
    _, _, extra3 = health_checks.check_anthropic_key(tmp_cfg)
    assert extra1["key_fingerprint"] == extra2["key_fingerprint"]
    assert extra1["key_fingerprint"] != extra3["key_fingerprint"]


def test_anthropic_key_wrong_shape_no_longer_leaks_prefix(
    tmp_cfg, monkeypatch,
):
    """Round 14: the wrong-shape error message used to embed key[:8].
    It now omits any key bytes."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "wrong-prefix-but-some-bytes-XYZ")
    ok, err, _ = health_checks.check_anthropic_key(tmp_cfg)
    assert ok is False
    assert "wrong-pre" not in err
    assert "XYZ" not in err


# ============================ L1 — header redaction =================

def test_classifier_prompt_redacts_from_and_subject(monkeypatch, fresh_db):
    """Round 14 fix: from_/subject go through _safe_for_prompt.
    The classifier triages PII-laden display names + signed subjects."""
    from secondbrain import email_assist
    from secondbrain.config import Config

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-test")
    cfg = Config()

    captured: dict = {}

    class _FakeMessages:
        def create(self, **kwargs):
            captured["messages"] = kwargs["messages"]
            resp = MagicMock()
            resp.usage.input_tokens = 50
            resp.usage.output_tokens = 5
            block = MagicMock()
            block.type = "text"
            block.text = '{"label": "fyi", "confidence": 0.9}'
            resp.content = [block]
            resp.stop_reason = "end_turn"
            return resp

    class _FakeClient:
        def __init__(self, *a, **kw):
            self.messages = _FakeMessages()

    mock_anthropic = MagicMock()
    mock_anthropic.Anthropic = _FakeClient
    mock_anthropic.APIError = Exception

    with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
        email_assist._default_classifier(
            from_="John Doe <john@example.com> SSN 123-45-6789",
            subject=(
                "Re: confirmation token "
                "sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
            ),
            body="hi",
            cfg=cfg,
            conn=fresh_db,
        )
    sent = captured["messages"][0]["content"]
    assert "[REDACTED:ssn]" in sent
    assert "[REDACTED:anthropic_key]" in sent
    assert "123-45-6789" not in sent
    assert "sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" not in sent
