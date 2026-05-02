"""Phase 33+34: RSS + IMAP connectors, SmartRecruiters + Recruitee parsers."""

from __future__ import annotations

from dataclasses import replace

from secondbrain.connectors.imap_email import (
    ImapEmailConnector,
    _extract_body,
    _strip_html,
)
from secondbrain.connectors.imap_email import (
    _config as imap_config,
)
from secondbrain.connectors.jobs import (
    JobsConnector,
    _render_recruitee_offer,
    _render_smartrecruiters_summary,
)
from secondbrain.connectors.rss import RSSConnector, _feeds, _parse_date

# ----------------------------- RSS -----------------------------------

def test_rss_disabled_when_no_feeds(tmp_cfg):
    assert RSSConnector().is_enabled(tmp_cfg) is False


def test_rss_enabled_via_config(tmp_cfg):
    cfg = replace(tmp_cfg, rss_feeds=("https://example.com/feed",))
    assert RSSConnector().is_enabled(cfg) is True


def test_rss_enabled_via_env(tmp_cfg, monkeypatch):
    monkeypatch.setenv("RSS_FEEDS", "https://x.com/feed")
    assert RSSConnector().is_enabled(tmp_cfg) is True


def test_rss_feeds_dedup_across_config_and_env(tmp_cfg, monkeypatch):
    monkeypatch.setenv("RSS_FEEDS", "https://a.com/feed, https://b.com/feed")
    cfg = replace(tmp_cfg, rss_feeds=("https://a.com/feed",))
    feeds = _feeds(cfg)
    assert feeds == ["https://a.com/feed", "https://b.com/feed"]


def test_rss_parse_date_rfc822():
    ts = _parse_date("Wed, 15 Apr 2026 12:00:00 +0000")
    assert ts > 0


def test_rss_parse_date_iso():
    ts = _parse_date("2026-04-15T12:00:00Z")
    assert ts > 0


def test_rss_parse_date_garbage_returns_now():
    ts = _parse_date("not a date")
    import time
    assert abs(ts - time.time()) < 5


# ---------------------------- IMAP ------------------------------------

def test_imap_strip_html():
    out = _strip_html("<p>hello <b>world</b></p><br>line2")
    assert "hello" in out and "world" in out
    assert "<p>" not in out and "<b>" not in out


def test_imap_disabled_without_password(tmp_cfg, monkeypatch):
    monkeypatch.delenv("SECONDBRAIN_IMAP_PASSWORD", raising=False)
    cfg = replace(
        tmp_cfg,
        imap_host="imap.gmail.com", imap_username="me@x.com",
        imap_folders=("INBOX",),
    )
    assert ImapEmailConnector().is_enabled(cfg) is False


def test_imap_disabled_without_host(tmp_cfg, monkeypatch):
    monkeypatch.setenv("SECONDBRAIN_IMAP_PASSWORD", "x")
    cfg = replace(tmp_cfg, imap_username="me@x.com",
                  imap_folders=("INBOX",))
    assert ImapEmailConnector().is_enabled(cfg) is False


def test_imap_disabled_without_folders(tmp_cfg, monkeypatch):
    monkeypatch.setenv("SECONDBRAIN_IMAP_PASSWORD", "x")
    cfg = replace(tmp_cfg, imap_host="imap.gmail.com", imap_username="me@x.com")
    assert ImapEmailConnector().is_enabled(cfg) is False


def test_imap_enabled_when_all_set(tmp_cfg, monkeypatch):
    monkeypatch.setenv("SECONDBRAIN_IMAP_PASSWORD", "x")
    cfg = replace(
        tmp_cfg,
        imap_host="imap.gmail.com", imap_username="me@x.com",
        imap_folders=("LinkedIn",),
    )
    assert ImapEmailConnector().is_enabled(cfg) is True


def test_imap_config_returns_dict(tmp_cfg):
    cfg = replace(
        tmp_cfg,
        imap_host="imap.gmail.com", imap_username="me@x.com",
        imap_folders=("INBOX", "LinkedIn"), imap_window_days=30,
    )
    c = imap_config(cfg)
    assert c is not None
    assert c["host"] == "imap.gmail.com"
    assert c["folders"] == ["INBOX", "LinkedIn"]
    assert c["window_days"] == 30


def test_imap_extract_body_plain():
    """Single-part text/plain message round-trips through _extract_body."""
    import email
    msg = email.message_from_string(
        "Subject: t\nContent-Type: text/plain; charset=utf-8\n\nhello world"
    )
    assert _extract_body(msg) == "hello world"


def test_imap_extract_body_multipart_prefers_plain():
    """When both text/plain and text/html exist, prefer plain."""
    import email
    msg = email.message_from_string(
        "MIME-Version: 1.0\n"
        "Content-Type: multipart/alternative; boundary=BB\n\n"
        "--BB\nContent-Type: text/plain\n\nplain version\n\n"
        "--BB\nContent-Type: text/html\n\n<p>html version</p>\n--BB--\n"
    )
    out = _extract_body(msg)
    assert "plain version" in out
    assert "<p>" not in out


def test_imap_extract_body_html_only_strips_tags():
    import email
    msg = email.message_from_string(
        "Content-Type: text/html\n\n<p>hello <b>world</b></p>"
    )
    assert "hello" in _extract_body(msg)
    assert "<p>" not in _extract_body(msg)


# ------------------------ SmartRecruiters -----------------------------

def test_render_smartrecruiters_minimal():
    j = {
        "id": "ABC-1234",
        "name": "Senior Software Engineer",
        "location": {"city": "Berlin", "region": "Berlin", "country": "DE"},
        "department": {"label": "Engineering"},
        "typeOfEmployment": {"label": "Permanent"},
        "industry": {"label": "Tech"},
        "company": {"name": "Acme GmbH"},
        "applyUrl": "https://jobs.example.com/apply/ABC-1234",
        "releasedDate": "2026-04-15T10:00:00Z",
        "jobAd": {
            "sections": {
                "jobDescription": {
                    "title": "About the role",
                    "text": "<p>Build great things.</p>",
                },
                "qualifications": {
                    "title": "Requirements",
                    "text": "<p>5+ years exp.</p>",
                },
            },
        },
    }
    doc = _render_smartrecruiters_summary("acme", j)
    assert doc is not None
    assert doc.virtual_path == "jobs://smartrecruiters/acme/ABC-1234"
    assert "Senior Software Engineer" in doc.title
    assert "[Acme GmbH]" in doc.title
    assert "Berlin, Berlin, DE" in doc.content
    assert "Build great things" in doc.content
    assert "5+ years exp" in doc.content
    assert doc.metadata["provider"] == "smartrecruiters"


def test_render_smartrecruiters_missing_id_returns_none():
    assert _render_smartrecruiters_summary("acme", {"name": "no id"}) is None


def test_render_smartrecruiters_falls_back_to_canonical_url():
    """When applyUrl isn't set, we synthesize the canonical jobs page URL."""
    j = {"id": "X-1", "name": "Job"}
    doc = _render_smartrecruiters_summary("acme", j)
    assert "jobs.smartrecruiters.com/acme/X-1" in doc.metadata["url"]


# ----------------------------- Recruitee ------------------------------

def test_render_recruitee_minimal():
    j = {
        "id": 4242, "title": "Frontend Engineer",
        "location": "Amsterdam", "department": "Product",
        "employment_type": "permanent",
        "description": "<p>Make UIs.</p>",
        "requirements": "<p>React experience.</p>",
        "careers_url": "https://someco.recruitee.com/o/frontend-engineer",
        "created_at": "2026-04-15T10:00:00Z",
    }
    doc = _render_recruitee_offer("someco", j)
    assert doc is not None
    assert doc.virtual_path == "jobs://recruitee/someco/4242"
    assert "Frontend Engineer" in doc.title
    assert "Amsterdam" in doc.content
    assert "Make UIs" in doc.content
    assert "React experience" in doc.content
    assert doc.metadata["provider"] == "recruitee"


def test_render_recruitee_missing_id_returns_none():
    assert _render_recruitee_offer("co", {"title": "no id"}) is None


# --------------------- JobsConnector enabled paths --------------------

def test_jobs_connector_enabled_for_smartrecruiters(tmp_cfg):
    cfg = replace(tmp_cfg, jobs_smartrecruiters=("acme",))
    assert JobsConnector().is_enabled(cfg) is True


def test_jobs_connector_enabled_for_recruitee(tmp_cfg):
    cfg = replace(tmp_cfg, jobs_recruitee=("someco",))
    assert JobsConnector().is_enabled(cfg) is True


# ---------------------- registry has new connectors -------------------

def test_registry_includes_rss_and_imap():
    from secondbrain.connectors import all_connectors
    names = {cls().name for cls in all_connectors()}
    assert "rss" in names
    assert "imap" in names
