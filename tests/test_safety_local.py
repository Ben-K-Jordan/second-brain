"""Phase 88 + 89: sensitive-content masking + local LLM fallback tests."""

from __future__ import annotations

from secondbrain import local_llm, safety

# ============================ Phase 88 redaction ======================

def test_redact_handles_empty():
    out = safety.redact("")
    assert out.text == ""
    assert out.total == 0


def test_redact_ssn():
    out = safety.redact("My SSN is 123-45-6789, careful.")
    assert "[REDACTED:ssn]" in out.text
    assert "123-45-6789" not in out.text
    assert out.counts.get("ssn") == 1


def test_redact_anthropic_key():
    raw = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz123"
    out = safety.redact(f"key is {raw}")
    assert "[REDACTED:anthropic_key]" in out.text
    assert raw not in out.text


def test_redact_openai_key():
    raw = "sk-proj-abcdefghijklmnopqrstuvwxyz123"
    out = safety.redact(f"key: {raw}")
    assert "[REDACTED:openai_key]" in out.text


def test_redact_github_token():
    raw = "ghp_abcdefghijklmnopqrstuvwxyz123"
    out = safety.redact(f"token={raw}")
    assert "[REDACTED:github_token]" in out.text


def test_redact_aws_key():
    raw = "AKIA1234567890ABCDEF"
    out = safety.redact(f"export AWS_KEY={raw}")
    assert "[REDACTED:aws_key]" in out.text


def test_redact_jwt():
    raw = (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
        "eyJzdWIiOiIxMjM0NTYifQ.SflKxwRJSMeKKF2QT4f"
    )
    out = safety.redact(f"Authorization: {raw}")
    assert "[REDACTED:jwt]" in out.text


def test_redact_bearer_header():
    """Bearer header tokens get caught even if shape is unfamiliar."""
    out = safety.redact("Authorization: Bearer abc123def456ghi789")
    assert "[REDACTED:bearer]" in out.text


def test_redact_credit_card_luhn_valid():
    """Visa test number 4111-1111-1111-1111 is Luhn-valid → redact."""
    out = safety.redact("Card: 4111-1111-1111-1111 saved.")
    assert "[REDACTED:credit_card]" in out.text


def test_redact_credit_card_luhn_invalid_left_alone():
    """A 16-digit ID that doesn't pass Luhn (e.g. an order number)
    should NOT be redacted — keeps false-positive rate low on
    business-y content."""
    out = safety.redact("Order ID 1234-5678-9012-3456 confirmed.")
    # Luhn check fails for that string → not redacted as credit_card.
    assert "1234-5678-9012-3456" in out.text


def test_redact_idempotent():
    """Re-running redact on already-redacted text should be a no-op."""
    s = "SSN 123-45-6789"
    once = safety.redact(s).text
    twice = safety.redact(once).text
    assert once == twice


def test_redact_counts_each_match():
    out = safety.redact(
        "First key sk-ant-api01-aaaaaaaaaaaaaaaaaaaa "
        "and second sk-ant-api02-bbbbbbbbbbbbbbbbbbbb."
    )
    assert out.counts.get("anthropic_key") == 2


def test_redact_multiple_kinds_in_one_text():
    out = safety.redact(
        "SSN 111-22-3333; key sk-ant-api03-abcdefghijklmnopqrstuvwxyz; "
        "card 4111111111111111."
    )
    assert "[REDACTED:ssn]" in out.text
    assert "[REDACTED:anthropic_key]" in out.text
    assert "[REDACTED:credit_card]" in out.text
    assert out.total >= 3


def test_has_sensitive_true_when_match_present():
    assert safety.has_sensitive("SSN 123-45-6789") is True


def test_has_sensitive_false_for_clean_text():
    assert safety.has_sensitive("Just normal text here.") is False


def test_redact_text_convenience():
    out = safety.redact_text("SSN 123-45-6789")
    assert isinstance(out, str)
    assert "[REDACTED:ssn]" in out


def test_luhn_valid_known_test_numbers():
    """Industry-standard test card numbers should pass."""
    assert safety._luhn_valid("4111111111111111") is True   # Visa
    assert safety._luhn_valid("5555555555554444") is True   # MC
    assert safety._luhn_valid("378282246310005") is True    # Amex


def test_luhn_invalid_random():
    assert safety._luhn_valid("1234567890123456") is False
    assert safety._luhn_valid("1") is False  # too short


# ============================ Phase 89 local LLM ======================

def test_is_available_returns_false_when_unreachable(monkeypatch):
    """When Ollama isn't running, is_available should return False
    quickly (not raise)."""
    import requests

    def boom(url, timeout=None):
        raise requests.ConnectionError("not running")

    monkeypatch.setattr(local_llm.requests, "get", boom)
    assert local_llm.is_available() is False


def test_is_available_returns_true_on_200(monkeypatch):
    class _R:
        status_code = 200
        def json(self):
            return {"models": []}
    monkeypatch.setattr(local_llm.requests, "get", lambda *a, **kw: _R())
    assert local_llm.is_available() is True


def test_list_models_returns_names(monkeypatch):
    class _R:
        status_code = 200
        def json(self):
            return {"models": [
                {"name": "llama3.1"},
                {"name": "mistral"},
            ]}
    monkeypatch.setattr(local_llm.requests, "get", lambda *a, **kw: _R())
    assert local_llm.list_models() == ["llama3.1", "mistral"]


def test_list_models_empty_when_unreachable(monkeypatch):
    import requests
    monkeypatch.setattr(
        local_llm.requests, "get",
        lambda *a, **kw: (_ for _ in ()).throw(requests.ConnectionError()),
    )
    assert local_llm.list_models() == []


def test_complete_returns_none_when_unavailable(monkeypatch):
    monkeypatch.setattr(local_llm, "is_available", lambda cfg=None: False)
    assert local_llm.complete("hi") is None


def test_complete_returns_completion_on_success(monkeypatch):
    monkeypatch.setattr(local_llm, "is_available", lambda cfg=None: True)
    class _R:
        status_code = 200
        def json(self):
            return {
                "response": "  Hello back.  ",
                "prompt_eval_count": 5, "eval_count": 12,
            }
    monkeypatch.setattr(local_llm.requests, "post", lambda *a, **kw: _R())
    out = local_llm.complete("hi", model="llama3.1")
    assert out is not None
    assert out.text == "Hello back."
    assert out.model == "llama3.1"
    assert out.prompt_tokens == 5
    assert out.completion_tokens == 12


def test_complete_returns_none_for_empty_response(monkeypatch):
    monkeypatch.setattr(local_llm, "is_available", lambda cfg=None: True)
    class _R:
        status_code = 200
        def json(self):
            return {"response": "  "}
    monkeypatch.setattr(local_llm.requests, "post", lambda *a, **kw: _R())
    assert local_llm.complete("hi") is None


def test_complete_returns_none_on_http_error(monkeypatch):
    monkeypatch.setattr(local_llm, "is_available", lambda cfg=None: True)
    class _R:
        status_code = 500
        def json(self): return {}
    monkeypatch.setattr(local_llm.requests, "post", lambda *a, **kw: _R())
    assert local_llm.complete("hi") is None


def test_complete_returns_none_on_request_exception(monkeypatch):
    """Network blip → graceful None, not propagated exception."""
    import requests
    monkeypatch.setattr(local_llm, "is_available", lambda cfg=None: True)
    monkeypatch.setattr(
        local_llm.requests, "post",
        lambda *a, **kw: (_ for _ in ()).throw(
            requests.ConnectionError("dropped"),
        ),
    )
    assert local_llm.complete("hi") is None


def test_complete_with_fallback_uses_primary_when_it_works():
    text, source = local_llm.complete_with_fallback(
        "test prompt",
        primary_fn=lambda prompt, system, max_tokens: "primary answer",
    )
    assert text == "primary answer"
    assert source == "primary"


def test_complete_with_fallback_uses_local_on_budget_exceeded(monkeypatch):
    """When primary raises BudgetExceededError, fall back to local."""
    from secondbrain.budget import BudgetExceededError

    monkeypatch.setattr(local_llm, "is_available", lambda cfg=None: True)
    class _R:
        status_code = 200
        def json(self):
            return {"response": "local answer"}
    monkeypatch.setattr(local_llm.requests, "post", lambda *a, **kw: _R())

    def primary(prompt, system, max_tokens):
        raise BudgetExceededError("anthropic", 600, 500)

    text, source = local_llm.complete_with_fallback(
        "x", primary_fn=primary,
    )
    assert text == "local answer"
    assert source == "local"


def test_complete_with_fallback_uses_local_on_generic_failure(monkeypatch):
    """Any primary exception (not just budget) triggers fallback."""
    monkeypatch.setattr(local_llm, "is_available", lambda cfg=None: True)
    class _R:
        status_code = 200
        def json(self): return {"response": "fallback"}
    monkeypatch.setattr(local_llm.requests, "post", lambda *a, **kw: _R())

    def primary(prompt, system, max_tokens):
        raise RuntimeError("API down")

    text, source = local_llm.complete_with_fallback(
        "x", primary_fn=primary,
    )
    assert text == "fallback"
    assert source == "local"


def test_complete_with_fallback_returns_failed_when_local_unavailable(
    monkeypatch,
):
    """Both primary AND local down → ('', 'failed') — let caller decide."""
    monkeypatch.setattr(local_llm, "is_available", lambda cfg=None: False)

    def primary(prompt, system, max_tokens):
        raise RuntimeError("API down")

    text, source = local_llm.complete_with_fallback(
        "x", primary_fn=primary,
    )
    assert text == ""
    assert source == "failed"


def test_resolve_host_uses_cfg_when_set():
    class _Cfg:
        local_llm_host = "http://homelab:11434"
    assert local_llm._resolve_host(_Cfg()) == "http://homelab:11434"


def test_resolve_host_default_when_no_cfg():
    assert local_llm._resolve_host(None) == "http://localhost:11434"


def test_resolve_model_uses_cfg_when_set():
    class _Cfg:
        local_llm_model = "mistral"
    assert local_llm._resolve_model(_Cfg()) == "mistral"


# ============== chat-citation redaction (polish v3) ==================

def test_format_search_result_redacts_chunk_text():
    """Phase 88 polish — sensitive content in chunks shouldn't leak
    into the model's tool-result feedback. Plugs the upstream hole."""
    from dataclasses import dataclass

    from secondbrain.chat import _format_search_result

    @dataclass
    class _R:
        chunk_id: int
        file_path: str
        chunk_index: int
        score: float
        text: str

    rs = [_R(1, "/x.md", 0, 0.5, "SSN 111-22-3333 in here.")]
    out = _format_search_result(rs)
    assert "[REDACTED:ssn]" in out
    assert "111-22-3333" not in out


def test_format_search_result_handles_clean_text():
    """No sensitive content → output unchanged (no false positives
    on regular prose)."""
    from dataclasses import dataclass

    from secondbrain.chat import _format_search_result

    @dataclass
    class _R:
        chunk_id: int
        file_path: str
        chunk_index: int
        score: float
        text: str

    rs = [_R(1, "/x.md", 0, 0.5, "This is normal prose.")]
    out = _format_search_result(rs)
    assert "REDACTED" not in out
    assert "normal prose" in out
