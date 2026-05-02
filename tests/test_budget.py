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


# ===================== Phase 63: per-feature budgets =================

def test_record_usage_persists_explicit_feature(tmp_cfg):
    """When the caller passes feature= explicitly, it lands on the row."""
    record_usage(
        tmp_cfg, "anthropic", "claude-haiku-4-5",
        input_tokens=100, output_tokens=50, feature="watchlist",
    )
    path = tmp_cfg.data_dir / "spend.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["feature"] == "watchlist"


def test_record_usage_infers_feature_from_note_prefix(tmp_cfg):
    """Existing callers pass note="watchlist/N items" without
    feature=. The inference recovers the feature so they still
    flow into the per-feature buckets cleanly."""
    record_usage(
        tmp_cfg, "anthropic", "claude-haiku-4-5",
        input_tokens=100, note="watchlist/3 items",
    )
    record_usage(
        tmp_cfg, "anthropic", "claude-haiku-4-5",
        input_tokens=100, note="briefing/event-x",
    )
    path = tmp_cfg.data_dir / "spend.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["feature"] == "watchlist"
    assert rows[1]["feature"] == "briefing"


def test_record_usage_infers_voyage_as_embed_feature(tmp_cfg):
    """Voyage usage with no note defaults to 'embed' since that's
    overwhelmingly what voyage is used for."""
    record_usage(tmp_cfg, "voyage", "voyage-3", input_tokens=1000)
    path = tmp_cfg.data_dir / "spend.jsonl"
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert row["feature"] == "embed"


def test_daily_spend_cents_filters_by_feature(tmp_cfg):
    record_usage(
        tmp_cfg, "anthropic", "claude-haiku-4-5",
        input_tokens=1_000_000, feature="watchlist",
    )
    record_usage(
        tmp_cfg, "anthropic", "claude-haiku-4-5",
        input_tokens=1_000_000, feature="chat",
    )
    watch_total = daily_spend_cents(
        tmp_cfg, "anthropic", feature="watchlist",
    )
    chat_total = daily_spend_cents(
        tmp_cfg, "anthropic", feature="chat",
    )
    overall = daily_spend_cents(tmp_cfg, "anthropic")
    assert watch_total > 0
    assert chat_total > 0
    assert abs(watch_total + chat_total - overall) < 1e-6


def test_check_budget_per_feature_caps_block_runaway(tmp_cfg):
    """If watchlist's per-feature cap is hit, watchlist calls fail
    even when the global cap has headroom — preventing one feature
    from starving the others."""
    tmp_cfg.daily_budget_cents_anthropic = 1000  # huge global cap
    tmp_cfg.feature_budget_cents = {"watchlist": 1}  # 1c cap on watchlist

    record_usage(
        tmp_cfg, "anthropic", "claude-haiku-4-5",
        input_tokens=1_000_000, feature="watchlist",
    )
    # Watchlist is over its own cap.
    with pytest.raises(BudgetExceededError):
        check_budget(tmp_cfg, "anthropic", feature="watchlist")
    # But chat — same provider, different feature — is unaffected.
    check_budget(tmp_cfg, "anthropic", feature="chat")


def test_check_budget_no_feature_cap_uses_global_only(tmp_cfg):
    """When a feature has no entry in feature_budget_cents, only the
    global cap applies — backward compat."""
    tmp_cfg.daily_budget_cents_anthropic = 100  # 100c global
    tmp_cfg.feature_budget_cents = {}  # no per-feature caps

    record_usage(
        tmp_cfg, "anthropic", "claude-haiku-4-5",
        input_tokens=100, feature="chat",
    )
    # Well under global, no cap.
    check_budget(tmp_cfg, "anthropic", feature="chat")


def test_check_budget_disabled_provider_still_enforces_feature(tmp_cfg):
    """An explicit cap=0 on the provider still leaves room for a
    per-feature cap. Useful when the user wants 'anything goes overall
    but chat capped at $2/day' style config."""
    tmp_cfg.daily_budget_cents_anthropic = 0  # disabled global
    tmp_cfg.feature_budget_cents = {"chat": 1}

    record_usage(
        tmp_cfg, "anthropic", "claude-haiku-4-5",
        input_tokens=1_000_000, feature="chat",
    )
    with pytest.raises(BudgetExceededError):
        check_budget(tmp_cfg, "anthropic", feature="chat")


def test_check_budget_zero_feature_cap_means_disabled(tmp_cfg):
    """Mirroring provider-level: 0 on the feature cap = disabled,
    not 'block all spend'. Lets users tag without enforcing."""
    tmp_cfg.daily_budget_cents_anthropic = 0
    tmp_cfg.feature_budget_cents = {"chat": 0}

    record_usage(
        tmp_cfg, "anthropic", "claude-haiku-4-5",
        input_tokens=1_000_000, feature="chat",
    )
    check_budget(tmp_cfg, "anthropic", feature="chat")  # no raise


def test_check_budget_no_feature_arg_is_legacy_compatible(tmp_cfg):
    """Existing callers that don't pass feature= still work — they
    get the provider-level check only."""
    tmp_cfg.daily_budget_cents_anthropic = 1
    tmp_cfg.feature_budget_cents = {"chat": 100}

    record_usage(
        tmp_cfg, "anthropic", "claude-haiku-4-5",
        input_tokens=1_000_000, feature="chat",
    )
    # Provider cap is 1c; we spent way more.
    with pytest.raises(BudgetExceededError):
        check_budget(tmp_cfg, "anthropic")  # no feature arg


def test_spend_summary_includes_by_feature_breakdown(tmp_cfg):
    """The dashboard's spend card needs to render '$0.42 chat / $0.18
    watchlist' — verify the by_feature sub-dict is populated."""
    from secondbrain.budget import spend_summary

    record_usage(
        tmp_cfg, "anthropic", "claude-haiku-4-5",
        input_tokens=1_000_000, feature="chat",
    )
    record_usage(
        tmp_cfg, "anthropic", "claude-haiku-4-5",
        input_tokens=2_000_000, feature="watchlist",
    )
    summary = spend_summary(tmp_cfg)
    by_feat = summary["anthropic"]["by_feature"]
    assert "chat" in by_feat
    assert "watchlist" in by_feat
    assert by_feat["watchlist"]["cents"] > by_feat["chat"]["cents"]
    assert by_feat["chat"]["calls"] == 1
    assert by_feat["watchlist"]["calls"] == 1


def test_legacy_rows_without_feature_field_default_to_other(tmp_cfg):
    """Spend ledgers from before Phase 63 don't have a 'feature'
    field. spend_summary should bucket those into 'other' so the
    dashboard doesn't render `None: $X`."""
    from secondbrain.budget import _ledger_path, spend_summary

    # Manually write a legacy row.
    path = _ledger_path(tmp_cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    import time as _t
    legacy = {
        "ts": _t.time(),
        "provider": "anthropic",
        "model": "claude-haiku-4-5",
        "input_tokens": 100, "output_tokens": 50,
        "cents": 0.5,
        "note": "test",
        # no 'feature' key
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(legacy) + "\n")

    summary = spend_summary(tmp_cfg)
    by_feat = summary["anthropic"]["by_feature"]
    assert "other" in by_feat
    assert by_feat["other"]["calls"] == 1
