"""API spend tracker and daily budget cap.

Three pieces:

1. **Ledger**: every paid API call appends a row to `~/.secondbrain/spend.jsonl`
   with timestamp, provider, model, tokens, and estimated USD cost.
2. **Cap**: before a paid call, we sum the last 24 hours from the ledger and
   refuse if it exceeds the configured daily limit. Default $5/day per provider.
3. **Pricing**: a small static table converts (model, input_tokens, output_tokens)
   into a USD cost estimate. Prices change occasionally; tune in config or here.

Defense in depth - this isn't a substitute for provider-side billing limits
(set those too). It catches runaway loops fast and gives users visibility.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Config

log = logging.getLogger(__name__)


# Cross-process exclusive file lock using stdlib only. The daemon, dashboard,
# and MCP server can all be writing the spend ledger from different processes;
# threading.Lock is not enough.
if sys.platform == "win32":
    import msvcrt

    @contextlib.contextmanager
    def _flock(f):  # type: ignore[no-untyped-def]
        # Lock 1 byte at offset 0; LK_LOCK blocks until acquired.
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            try:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
else:
    import fcntl

    @contextlib.contextmanager
    def _flock(f):  # type: ignore[no-untyped-def]
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# USD per 1M tokens. Most embedders only have an input price; chat/rerank
# split into input/output. Prices verified 2026-04 against published rates;
# bump as needed.
_PRICES_USD_PER_M: dict[str, dict[str, float]] = {
    # Voyage embeddings
    "voyage-3":            {"input": 0.18},
    "voyage-3-large":      {"input": 0.18},
    "voyage-3-lite":       {"input": 0.06},
    "voyage-code-3":       {"input": 0.18},
    # Voyage multimodal
    "voyage-multimodal-3": {"input": 0.12},
    # Voyage rerank
    "rerank-2":            {"input": 0.10},
    "rerank-2-lite":       {"input": 0.05},
    # Anthropic chat (input vs output)
    "claude-opus-4-7":     {"input": 5.00, "output": 25.00},
    "claude-opus-4-6":     {"input": 5.00, "output": 25.00},
    "claude-sonnet-4-6":   {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5":    {"input": 1.00, "output": 5.00},
    # Anthropic web-search server tool: billed at $10 per 1k searches.
    # We bill it as a faux "model" with input_tokens = number of searches
    # so it shows up alongside chat traffic in the spend ledger.
    "anthropic-web-search": {"input": 10_000.0},  # $10 per 1k = $10_000 per 1M
}


class BudgetExceededError(RuntimeError):
    """Raised when a paid call would push today's spend over the configured cap."""

    def __init__(self, provider: str, current_cents: float, cap_cents: float):
        super().__init__(
            f"Daily {provider} spend cap exceeded: "
            f"${current_cents / 100:.4f} >= ${cap_cents / 100:.2f}. "
            f"Raise the cap in config.toml or wait 24 hours."
        )
        self.provider = provider
        self.current_cents = current_cents
        self.cap_cents = cap_cents


@dataclass
class CostEstimate:
    model: str
    input_tokens: int
    output_tokens: int
    cents: float


def estimate_cost(model: str, input_tokens: int, output_tokens: int = 0) -> CostEstimate:
    """Best-effort USD-cents estimate for a single API call."""
    prices = _PRICES_USD_PER_M.get(model, {})
    in_per_m = prices.get("input", 0.0)
    out_per_m = prices.get("output", 0.0)
    cents = (in_per_m * input_tokens + out_per_m * output_tokens) / 1_000_000 * 100
    return CostEstimate(model, input_tokens, output_tokens, cents)


def _ledger_path(cfg: Config) -> Path:
    return cfg.data_dir / "spend.jsonl"


_LEDGER_LOCK = threading.Lock()


def record_usage(
    cfg: Config,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int = 0,
    note: str = "",
) -> CostEstimate:
    """Append a row to the spend ledger. Returns the cost estimate."""
    estimate = estimate_cost(model, input_tokens, output_tokens)
    row = {
        "ts": time.time(),
        "provider": provider,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cents": round(estimate.cents, 6),
        "note": note,
    }
    path = _ledger_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(row) + "\n"
    # In-process lock for thread safety, file lock for cross-process safety
    # (daemon + dashboard + MCP can all write concurrently). fsync ensures
    # the row survives a crash - otherwise the cap under-reports, the next
    # call passes, and the loop blows the budget for real.
    with _LEDGER_LOCK, open(path, "a", encoding="utf-8") as f, _flock(f):
        f.write(payload)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            # Some filesystems / WSL on certain mounts can refuse
            # fsync on append-only writes. The flush already made it
            # to OS buffers; that's the best we can do.
            pass
    return estimate


def daily_spend_cents(cfg: Config, provider: str | None = None, hours: float = 24.0) -> float:
    """Sum spend in the last ``hours`` from the ledger.

    If ``provider`` is None, sums across all providers; otherwise filters.
    Missing or unreadable ledger files return 0 (no spend recorded yet).
    """
    path = _ledger_path(cfg)
    if not path.exists():
        return 0.0
    cutoff = time.time() - hours * 3600
    total = 0.0
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("ts", 0) < cutoff:
                    continue
                if provider and row.get("provider") != provider:
                    continue
                total += float(row.get("cents", 0))
    except OSError as e:
        log.warning("could not read spend ledger %s: %s", path, e)
        return 0.0
    return total


def check_budget(cfg: Config, provider: str) -> None:
    """Refuse the call if the provider's 24h spend has hit its cap.

    Caps live in config: ``daily_budget_cents_voyage`` and
    ``daily_budget_cents_anthropic`` (cents because integer cents avoid float
    parsing surprises in TOML). Set to 0 to **explicitly disable** the cap.

    Fails closed: if the config field is missing or None, we refuse the call
    rather than silently allowing unlimited spend. The user previously hit a
    case where None bypassed the cap and a runaway loop spent $6.47 against
    a $5 cap. Better to halt and ask "why is the cap None?" than to leak.
    """
    cap = (
        cfg.daily_budget_cents_voyage if provider == "voyage"
        else cfg.daily_budget_cents_anthropic
    )
    if cap is None:
        raise BudgetExceededError(provider, 0.0, 0.0)
    if cap <= 0:
        return  # explicit "disabled" sentinel
    spent = daily_spend_cents(cfg, provider=provider)
    if spent >= cap:
        raise BudgetExceededError(provider, spent, cap)


def spend_summary(cfg: Config) -> dict:
    """Snapshot for dashboards - totals + counts per provider for the last 24h."""
    out: dict[str, dict[str, float]] = {
        "voyage":    {"cents": 0.0, "calls": 0, "tokens": 0},
        "anthropic": {"cents": 0.0, "calls": 0, "tokens": 0},
    }
    path = _ledger_path(cfg)
    if not path.exists():
        return out
    cutoff = time.time() - 24 * 3600
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("ts", 0) < cutoff:
                    continue
                provider = row.get("provider", "unknown")
                bucket = out.setdefault(provider, {"cents": 0.0, "calls": 0, "tokens": 0})
                bucket["cents"] += float(row.get("cents", 0))
                bucket["calls"] += 1
                bucket["tokens"] += int(row.get("input_tokens", 0)) + int(row.get("output_tokens", 0))
    except OSError:
        pass
    return out
