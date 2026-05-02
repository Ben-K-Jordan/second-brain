"""Phase 27-29: domain presets, watchlist scoping, jobs + news connectors."""

from __future__ import annotations

from dataclasses import replace

import pytest

from secondbrain.connectors.jobs import (
    _config_companies,
    _render_ashby_job,
    _render_greenhouse_job,
    _render_lever_job,
)
from secondbrain.connectors.news import NewsConnector
from secondbrain.db import (
    watchlist_create,
    watchlist_get_domains,
    watchlist_set_domains,
)
from secondbrain.presets import PRESETS, names, resolve

# ----------------------------- Presets --------------------------------

def test_preset_names_complete():
    expected = {"jobs", "news", "markets", "research", "ai", "dev"}
    assert expected.issubset(set(names()))


def test_jobs_preset_includes_major_boards():
    """The jobs preset is the 'recruiting use case' deliverable; pin its
    contents so a future regression in PRESETS surfaces immediately."""
    j = set(PRESETS["jobs"])
    assert {"linkedin.com", "indeed.com", "joinhandshake.com",
            "lever.co", "greenhouse.io", "ashbyhq.com",
            "wellfound.com", "ycombinator.com"} <= j


def test_resolve_no_preset_no_extras_returns_none():
    assert resolve(None, None) is None
    assert resolve(None, []) is None


def test_resolve_preset_only():
    out = resolve("jobs", None)
    assert out is not None and "linkedin.com" in out


def test_resolve_extras_only_lowercases_and_strips():
    out = resolve(None, ["FOO.com  ", "https://Bar.com/", "  "])
    assert out == ["foo.com", "bar.com"]


def test_resolve_combines_preset_with_extras():
    out = resolve("jobs", ["mycompany.com"])
    assert "mycompany.com" in out
    assert "linkedin.com" in out


def test_resolve_dedups_extras_against_preset():
    out = resolve("jobs", ["linkedin.com"])
    assert out.count("linkedin.com") == 1


def test_resolve_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown preset"):
        resolve("nonexistent", None)


# -------------------- Watchlist domain persistence --------------------

def test_watchlist_create_with_domains_round_trip(fresh_db):
    wid = watchlist_create(
        fresh_db, "pm-int", "q",
        allowed_domains=["linkedin.com", "indeed.com"],
    )
    assert watchlist_get_domains(fresh_db, wid) == ["linkedin.com", "indeed.com"]


def test_watchlist_create_no_domains_returns_none(fresh_db):
    wid = watchlist_create(fresh_db, "open", "q")
    assert watchlist_get_domains(fresh_db, wid) is None


def test_watchlist_set_domains_update(fresh_db):
    wid = watchlist_create(fresh_db, "x", "q")
    watchlist_set_domains(fresh_db, wid, ["a.com", "b.com"])
    assert watchlist_get_domains(fresh_db, wid) == ["a.com", "b.com"]


def test_watchlist_set_domains_clear(fresh_db):
    wid = watchlist_create(fresh_db, "x", "q", allowed_domains=["a.com"])
    watchlist_set_domains(fresh_db, wid, None)
    assert watchlist_get_domains(fresh_db, wid) is None
    watchlist_set_domains(fresh_db, wid, ["c.com"])
    watchlist_set_domains(fresh_db, wid, [])  # empty list also clears
    assert watchlist_get_domains(fresh_db, wid) is None


def test_watchlist_get_domains_handles_corrupt_json(fresh_db):
    """If somebody manually edits the DB and breaks the JSON, we shouldn't
    crash the whole watchlist run."""
    wid = watchlist_create(fresh_db, "x", "q")
    fresh_db.execute(
        "UPDATE watchlists SET allowed_domains_json = ? WHERE id = ?",
        ("not valid json {{{", wid),
    )
    fresh_db.commit()
    assert watchlist_get_domains(fresh_db, wid) is None


# -------------------- Jobs connector renderers ------------------------

def test_render_greenhouse_minimal():
    j = {
        "id": 12345, "title": "Software Engineer Intern",
        "absolute_url": "https://boards.greenhouse.io/anthropic/jobs/12345",
        "updated_at": "2026-04-01T12:00:00Z",
        "location": {"name": "San Francisco"},
        "departments": [{"name": "Engineering"}],
        "offices": [{"name": "SF Office"}],
        "content": "<p>We&#39;re hiring.</p><p>Join us!</p>",
    }
    doc = _render_greenhouse_job("anthropic", j)
    assert doc is not None
    assert doc.virtual_path == "jobs://greenhouse/anthropic/12345"
    assert "Software Engineer Intern" in doc.title
    assert "[anthropic]" in doc.title
    assert "boards.greenhouse.io" in doc.content
    assert "San Francisco" in doc.content
    # HTML should be stripped + entities decoded
    assert "<p>" not in doc.content
    assert "We're hiring" in doc.content
    assert doc.metadata["provider"] == "greenhouse"
    assert doc.metadata["company"] == "anthropic"


def test_render_greenhouse_missing_id_returns_none():
    assert _render_greenhouse_job("x", {"title": "no id"}) is None


def test_render_lever_minimal():
    j = {
        "id": "abc-123", "text": "PM Intern",
        "applyUrl": "https://jobs.lever.co/co/abc-123/apply",
        "createdAt": 1700000000000,  # epoch milliseconds
        "categories": {"location": "NYC", "team": "Product",
                       "commitment": "Internship"},
        "descriptionPlain": "Join our PM team.",
        "lists": [
            {"text": "What you'll do", "content": "<p>Build things</p>"},
        ],
    }
    doc = _render_lever_job("acme", j)
    assert doc is not None
    assert doc.virtual_path == "jobs://lever/acme/abc-123"
    assert "PM Intern" in doc.title
    assert "Build things" in doc.content  # bullet list HTML stripped
    assert doc.metadata["provider"] == "lever"
    # createdAt parsed from ms
    assert 1.69e9 < doc.mtime < 1.71e9


def test_render_lever_missing_id_returns_none():
    assert _render_lever_job("x", {"text": "no id"}) is None


def test_render_ashby_minimal():
    j = {
        "id": "ash-1", "title": "Engineer", "location": "Remote",
        "department": "Eng", "employmentType": "FullTime",
        "jobUrl": "https://jobs.ashbyhq.com/modal/ash-1",
        "publishedAt": "2026-04-15T10:00:00Z",
        "descriptionHtml": "<h2>About</h2><p>Best place ever.</p>",
        "compensation": {"compensationTierSummary": "$150K - $200K"},
    }
    doc = _render_ashby_job("modal", j)
    assert doc is not None
    assert doc.virtual_path == "jobs://ashby/modal/ash-1"
    assert "$150K" in doc.content
    assert "Best place ever" in doc.content
    assert doc.metadata["provider"] == "ashby"


def test_jobs_connector_disabled_when_no_companies(tmp_cfg):
    from secondbrain.connectors.jobs import JobsConnector

    assert JobsConnector().is_enabled(tmp_cfg) is False


def test_jobs_connector_enabled_when_any_provider_set(tmp_cfg):
    from secondbrain.connectors.jobs import JobsConnector

    cfg = replace(tmp_cfg, jobs_greenhouse=("anthropic",))
    assert JobsConnector().is_enabled(cfg) is True


def test_config_companies_dedups_and_lowercases():
    """The connector lowercases + dedups + strips entries so the user can
    paste sloppy lists in config.toml without surprises."""
    class _Stub:
        jobs_greenhouse = ("Anthropic", "anthropic", " STRIPE ")
    out = _config_companies(_Stub(), "greenhouse")
    assert out == ["anthropic", "stripe"]


# -------------------------- News connector ----------------------------

def test_news_connector_disabled_without_key(tmp_cfg, monkeypatch):
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)
    cfg = replace(tmp_cfg, news_topics=("ai",))
    assert NewsConnector().is_enabled(cfg) is False


def test_news_connector_disabled_without_topics(tmp_cfg, monkeypatch):
    monkeypatch.setenv("NEWSAPI_KEY", "test")
    assert NewsConnector().is_enabled(tmp_cfg) is False


def test_news_connector_enabled_with_both(tmp_cfg, monkeypatch):
    monkeypatch.setenv("NEWSAPI_KEY", "test")
    cfg = replace(tmp_cfg, news_topics=("artificial intelligence",))
    assert NewsConnector().is_enabled(cfg) is True


def test_news_render_skips_removed_articles():
    """NewsAPI returns "[Removed]" placeholder articles for retracted
    pieces. We shouldn't index those."""
    nc = NewsConnector()
    art = {"title": "[Removed]", "url": "https://example.com/x",
           "publishedAt": "2026-04-15T10:00:00Z"}
    assert nc._render_article(art, "ai") is None


def test_news_render_skips_missing_url():
    nc = NewsConnector()
    art = {"title": "Real Title", "url": "",
           "publishedAt": "2026-04-15T10:00:00Z"}
    assert nc._render_article(art, "ai") is None


def test_news_render_full_article():
    nc = NewsConnector()
    art = {
        "title": "Anthropic launches new model",
        "url": "https://techcrunch.com/2026/04/15/anthropic-new",
        "description": "Big announcement.",
        "content": "Big announcement extended.",
        "author": "Jane Doe",
        "source": {"name": "TechCrunch"},
        "publishedAt": "2026-04-15T10:00:00Z",
    }
    doc = nc._render_article(art, "anthropic")
    assert doc is not None
    assert doc.title == "Anthropic launches new model"
    assert doc.virtual_path == "https://techcrunch.com/2026/04/15/anthropic-new"
    assert "TechCrunch" in doc.content
    assert "Jane Doe" in doc.content
    assert doc.metadata["topic"] == "anthropic"


# ---------- Watchlist runner picks up per-watchlist domains -----------

def test_run_watchlist_passes_domain_override(fresh_db, tmp_cfg, monkeypatch):
    """run_watchlist should call ask_brain with web_search_allowed_domains
    set to whatever the watchlist has saved."""
    from secondbrain import watchlist as wl_mod

    captured: dict = {}

    def fake_ask(cfg, conn, embedder, reranker, prompt, **kwargs):
        captured["domains"] = kwargs.get("web_search_allowed_domains")
        from secondbrain.chat import ChatResponse
        return ChatResponse(text="ok", citations=[], iterations=0)

    monkeypatch.setattr(wl_mod, "ask_brain", fake_ask)

    wid = watchlist_create(
        fresh_db, "scoped", "q",
        allowed_domains=["linkedin.com", "indeed.com"],
    )
    wl_mod.run_watchlist(tmp_cfg, fresh_db, None, None, wid, "q", None)
    assert captured["domains"] == ["linkedin.com", "indeed.com"]


def test_run_watchlist_passes_none_when_no_domains(fresh_db, tmp_cfg, monkeypatch):
    """When no domains are set, ask_brain should get None - which means
    'use cfg.web_search_allowed_domains'."""
    from secondbrain import watchlist as wl_mod

    captured: dict = {}

    def fake_ask(cfg, conn, embedder, reranker, prompt, **kwargs):
        captured["domains"] = kwargs.get("web_search_allowed_domains")
        from secondbrain.chat import ChatResponse
        return ChatResponse(text="ok", citations=[], iterations=0)

    monkeypatch.setattr(wl_mod, "ask_brain", fake_ask)

    wid = watchlist_create(fresh_db, "open", "q")
    wl_mod.run_watchlist(tmp_cfg, fresh_db, None, None, wid, "q", None)
    assert captured["domains"] is None
