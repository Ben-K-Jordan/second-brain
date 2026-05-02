"""Budget cap fail-closed behaviour + ledger fsync round-trip."""

from __future__ import annotations

import json

import pytest

from secondbrain.budget import (
    BudgetExceededError,
    check_budget,
    daily_spend_cents,
    estimate_cost,
    record_usage,
)


def test_estimate_cost_known_model():
    est = estimate_cost("voyage-3", input_tokens=1_000_000)
    assert 0 < est.cents < 100  # cheap, but > 0


def test_estimate_cost_unknown_model_returns_zero():
    est = estimate_cost("unknown-model-xyz", input_tokens=1_000_000)
    assert est.cents == 0.0


def test_record_and_sum_round_trip(tmp_cfg):
    record_usage(tmp_cfg, "voyage", "voyage-3", input_tokens=1_000_000)
    record_usage(tmp_cfg, "voyage", "voyage-3", input_tokens=2_000_000)
    total = daily_spend_cents(tmp_cfg, "voyage")
    expected = estimate_cost("voyage-3", 3_000_000).cents
    assert abs(total - expected) < 1e-6


def test_check_budget_allows_below_cap(tmp_cfg):
    tmp_cfg.daily_budget_cents_voyage = 100
    record_usage(tmp_cfg, "voyage", "voyage-3", input_tokens=1_000_000)
    # Well under 100c, should pass.
    check_budget(tmp_cfg, "voyage")


def test_check_budget_blocks_above_cap(tmp_cfg):
    tmp_cfg.daily_budget_cents_voyage = 1
    # Spend ~$0.18 for 1M voyage-3 tokens — well over 1c cap.
    record_usage(tmp_cfg, "voyage", "voyage-3", input_tokens=1_000_000)
    with pytest.raises(BudgetExceededError):
        check_budget(tmp_cfg, "voyage")


def test_check_budget_fails_closed_when_cap_is_none(tmp_cfg):
    """C6: a None cap must NOT silently allow unlimited spend."""
    tmp_cfg.daily_budget_cents_voyage = None  # type: ignore[assignment]
    with pytest.raises(BudgetExceededError):
        check_budget(tmp_cfg, "voyage")


def test_check_budget_disabled_when_cap_is_zero(tmp_cfg):
    """Zero is the explicit 'disabled' sentinel."""
    tmp_cfg.daily_budget_cents_voyage = 0
    record_usage(tmp_cfg, "voyage", "voyage-3", input_tokens=10_000_000)
    # No raise expected.
    check_budget(tmp_cfg, "voyage")


def test_record_usage_writes_jsonl(tmp_cfg):
    record_usage(tmp_cfg, "anthropic", "claude-haiku-4-5",
                 input_tokens=100, output_tokens=50, note="test")
    path = tmp_cfg.data_dir / "spend.jsonl"
    assert path.exists()
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["provider"] == "anthropic"
    assert rows[0]["note"] == "test"
    assert rows[0]["input_tokens"] == 100
