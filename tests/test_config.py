"""Config loader, default-toml, and validation."""

from __future__ import annotations

import pytest

from secondbrain.config import (
    Config,
    _validate_config,
    default_config_toml,
    load_config,
)


def test_default_toml_parses(tmp_path, monkeypatch):
    """The shipped default config should load cleanly with no overrides."""
    monkeypatch.setenv("SB_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "config.toml").write_text(default_config_toml(), encoding="utf-8")
    cfg = load_config()
    assert cfg.chunk_size > 0
    assert cfg.chunk_overlap < cfg.chunk_size
    assert cfg.daily_budget_cents_voyage >= 0


def test_validate_chunk_overlap_must_be_smaller():
    cfg = Config()
    cfg.chunk_size = 100
    cfg.chunk_overlap = 200
    with pytest.raises(ValueError, match="chunk_overlap"):
        _validate_config(cfg)


def test_validate_negative_chunk_size_rejected():
    cfg = Config()
    cfg.chunk_size = 0
    with pytest.raises(ValueError, match="chunk_size"):
        _validate_config(cfg)


def test_validate_short_personal_prefix_warns(caplog):
    """Very short prefixes match too broadly; we warn but don't reject."""
    cfg = Config()
    cfg.personal_path_prefixes = ("/",)
    with caplog.at_level("WARNING"):
        _validate_config(cfg)
    assert any("personal_path_prefixes" in r.message for r in caplog.records)
