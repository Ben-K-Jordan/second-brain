"""Phase 89: local LLM fallback (Ollama).

When the user's offline OR the daily Anthropic budget is exhausted,
fall back to a local model so the brain keeps working — slower,
lower quality, but functional.

Implementation: thin Ollama HTTP client. No SDK dependency; requests
+ JSON is enough for the chat endpoint. Ollama runs at
http://localhost:11434 by default; the user installs it separately
and pulls a model (``ollama pull llama3.1``).

Two surfaces:

  1. ``is_available()``: pings Ollama. Cheap (50ms timeout).
  2. ``complete(prompt, model)``: synchronous one-shot completion.
     Used as a fallback when Anthropic raises BudgetExceededError.

Cost: zero (local). Quality: depends on the model — llama3.1 8B is
roughly equivalent to a small cloud model. The user trades latency
+ quality for never having a "budget exceeded" hard stop.

Not used as a primary path — the brain's other features still
prefer Anthropic when the budget allows. This is purely a circuit
breaker.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

log = logging.getLogger(__name__)


# ---- Tunables ---------------------------------------------------------

_DEFAULT_HOST = "http://localhost:11434"
_DEFAULT_MODEL = "llama3.1"
_PING_TIMEOUT = 0.5  # second — fast fail when Ollama isn't running
_GENERATE_TIMEOUT = 120  # local generation can be slow on CPU


@dataclass
class LocalCompletion:
    """Mirror of an Anthropic completion just enough for callers
    that want a uniform interface."""
    text: str
    model: str
    prompt_tokens: int  # ollama doesn't always report; best-effort
    completion_tokens: int


def _resolve_host(cfg=None) -> str:
    """Where to reach Ollama. Reads ``cfg.local_llm_host`` when
    available, falls back to the default. Lets users point at a
    remote Ollama instance on a homelab box if they prefer."""
    if cfg is not None:
        h = getattr(cfg, "local_llm_host", "")
        if h:
            return h
    return _DEFAULT_HOST


def _resolve_model(cfg=None) -> str:
    if cfg is not None:
        m = getattr(cfg, "local_llm_model", "")
        if m:
            return m
    return _DEFAULT_MODEL


def is_available(cfg=None) -> bool:
    """Cheap ping to confirm Ollama is reachable + healthy."""
    host = _resolve_host(cfg)
    try:
        r = requests.get(f"{host}/api/tags", timeout=_PING_TIMEOUT)
    except (requests.RequestException, OSError):
        return False
    return r.status_code == 200


def list_models(cfg=None) -> list[str]:
    """Return the names of locally-pulled Ollama models. Empty list
    when Ollama isn't reachable."""
    host = _resolve_host(cfg)
    try:
        r = requests.get(f"{host}/api/tags", timeout=_PING_TIMEOUT)
    except (requests.RequestException, OSError):
        return []
    if r.status_code != 200:
        return []
    try:
        models = r.json().get("models") or []
    except ValueError:
        return []
    return [m.get("name", "") for m in models if m.get("name")]


def complete(
    prompt: str,
    *,
    cfg=None,
    model: str | None = None,
    system: str | None = None,
    max_tokens: int = 1024,
) -> LocalCompletion | None:
    """Synchronous one-shot completion. Returns None when Ollama is
    unavailable or the call fails — caller decides how to surface.

    Args:
        prompt: user message text.
        model: override cfg.local_llm_model. None uses config / default.
        system: optional system prompt.
        max_tokens: cap on generation length.
    """
    if not is_available(cfg):
        return None
    host = _resolve_host(cfg)
    model_name = model or _resolve_model(cfg)
    payload: dict = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    if system:
        payload["system"] = system
    try:
        r = requests.post(
            f"{host}/api/generate", json=payload,
            timeout=_GENERATE_TIMEOUT,
        )
    except requests.RequestException as e:
        log.warning("local_llm: request failed: %s", type(e).__name__)
        return None
    if r.status_code != 200:
        log.warning("local_llm: HTTP %s from %s", r.status_code, host)
        return None
    try:
        body = r.json()
    except ValueError:
        return None
    text = (body.get("response") or "").strip()
    if not text:
        return None
    return LocalCompletion(
        text=text,
        model=model_name,
        prompt_tokens=int(body.get("prompt_eval_count") or 0),
        completion_tokens=int(body.get("eval_count") or 0),
    )


def complete_with_fallback(
    prompt: str,
    *,
    primary_fn,
    cfg=None,
    system: str | None = None,
    max_tokens: int = 1024,
) -> tuple[str, str]:
    """Run ``primary_fn(prompt, system, max_tokens)`` first; on
    BudgetExceededError or transient failure, fall back to local.

    Returns ``(text, source)`` where source is 'primary' / 'local' /
    'failed'. Used by chat / synthesis paths that want graceful
    degradation rather than hard failure.
    """
    from .budget import BudgetExceededError

    try:
        text = primary_fn(prompt=prompt, system=system, max_tokens=max_tokens)
        if text:
            return text, "primary"
    except BudgetExceededError as e:
        log.info(
            "local_llm: primary budget-exceeded, falling back local: %s", e,
        )
    except Exception as e:  # noqa: BLE001
        log.warning(
            "local_llm: primary failed, trying local: %s", type(e).__name__,
        )
    out = complete(
        prompt, cfg=cfg, system=system, max_tokens=max_tokens,
    )
    if out is None:
        return "", "failed"
    return out.text, "local"
