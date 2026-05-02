"""Search ranker math: time decay, click-recency boost, path multiplier."""

from __future__ import annotations

import time

from secondbrain.search import (
    _click_recency_multiplier,
    _path_score_multiplier,
    _time_decay_factor,
    adaptive_alpha,
    should_use_hyde,
)


def test_click_recency_just_now_is_max():
    assert _click_recency_multiplier(time.time(), 1.25, 14) > 1.24


def test_click_recency_at_half_life_is_midpoint():
    half = time.time() - 14 * 86400
    val = _click_recency_multiplier(half, 1.25, 14)
    assert 1.124 < val < 1.126  # midpoint between 1.0 and 1.25


def test_click_recency_disabled_returns_one():
    assert _click_recency_multiplier(time.time(), 1.0, 14) == 1.0


def test_click_recency_no_click_returns_one():
    assert _click_recency_multiplier(None, 1.25, 14) == 1.0


def test_time_decay_now_is_one():
    assert _time_decay_factor(time.time(), 365) > 0.999


def test_time_decay_at_half_life_is_half():
    val = _time_decay_factor(time.time() - 365 * 86400, 365)
    assert 0.49 < val < 0.51


def test_path_multiplier_personal_path():
    m = _path_score_multiplier(
        "/Users/me/Documents/notes.md",
        personal_prefixes=("/Documents/",), personal_boost=1.5,
        download_prefixes=("/Downloads/",), download_demote=0.5,
    )
    assert m == 1.5


def test_path_multiplier_download_path():
    m = _path_score_multiplier(
        "/Users/me/Downloads/random.pdf",
        personal_prefixes=("/Documents/",), personal_boost=1.5,
        download_prefixes=("/Downloads/",), download_demote=0.5,
    )
    assert m == 0.5


def test_path_multiplier_neither_returns_one():
    m = _path_score_multiplier(
        "/tmp/random.txt",
        personal_prefixes=("/Documents/",), personal_boost=1.5,
        download_prefixes=("/Downloads/",), download_demote=0.5,
    )
    assert m == 1.0


def test_adaptive_alpha_long_prose_leans_vector():
    # 8+ tokens of prose should push toward vector (alpha >= default)
    a = adaptive_alpha("what was the discussion about voyage rate limits last week", default=0.5)
    assert a >= 0.5


def test_adaptive_alpha_id_query_leans_keyword():
    # An ID-bearing short query should pull toward BM25
    a = adaptive_alpha("ABC-1234", default=0.5)
    assert a < 0.5


def test_should_use_hyde_question_form():
    assert should_use_hyde("what was that thing about voyage rate limits") is True
    assert should_use_hyde("rate limits?") is False  # too short
    assert should_use_hyde("voyage") is False        # too short
