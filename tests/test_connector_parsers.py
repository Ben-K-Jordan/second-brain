"""Pure-parser tests for connectors that ship hand-rolled feed/JSON walkers.

We don't want to spin up real HTTP. These tests pin the *parsing* behaviour
so future audits can detect when an upstream payload shape change breaks
extraction.
"""

from __future__ import annotations

from secondbrain.connectors.hacker_news import _strip_html as hn_strip
from secondbrain.connectors.obsidian import (
    _parse_frontmatter,
    _resolve_wikilinks,
)
from secondbrain.connectors.substack import _strip_html as substack_strip
from secondbrain.connectors.x_archive import _PREFIX_RE, _twitter_ts_to_epoch

# ----------------------------- Obsidian -----------------------------

def test_obsidian_frontmatter_basic():
    fm, body = _parse_frontmatter("---\ntitle: Hello\ntags: [foo, bar]\n---\n\nbody")
    assert fm == {"title": "Hello", "tags": ["foo", "bar"]}
    assert body.strip() == "body"


def test_obsidian_frontmatter_bulleted_list():
    fm, _ = _parse_frontmatter(
        "---\naliases:\n  - HelloNote\n  - greeting\n---\n\nbody"
    )
    assert fm == {"aliases": ["HelloNote", "greeting"]}


def test_obsidian_frontmatter_quoted_string():
    fm, _ = _parse_frontmatter('---\ntitle: "Quoted Title"\n---\nbody')
    assert fm["title"] == "Quoted Title"


def test_obsidian_no_frontmatter_returns_full_body():
    fm, body = _parse_frontmatter("just body, no frontmatter")
    assert fm == {}
    assert body == "just body, no frontmatter"


def test_obsidian_wikilink_display_text_wins():
    out = _resolve_wikilinks("see [[Note|click me]] for more", set())
    assert out == "see click me for more"


def test_obsidian_wikilink_heading_anchor():
    out = _resolve_wikilinks("see [[Note#Heading]]", set())
    assert out == "see Note > Heading"


def test_obsidian_wikilink_bare():
    out = _resolve_wikilinks("see [[Note]]", set())
    assert out == "see Note"


# ------------------------------ Substack ----------------------------

def test_substack_strip_html_preserves_paragraphs():
    out = substack_strip("<p>First.</p><p>Second.</p>")
    assert "First." in out and "Second." in out
    assert "<p>" not in out


def test_substack_strip_html_handles_br():
    out = substack_strip("line1<br>line2<br/>line3")
    assert out.count("\n") >= 2


# ------------------------------ HN ---------------------------------

def test_hn_strip_html_decodes_entities():
    out = hn_strip("AT&amp;T &lt;3&gt; &quot;hi&quot;")
    assert "AT&T" in out
    assert "<3>" in out
    assert '"hi"' in out


# ------------------------------ X archive ---------------------------

def test_x_prefix_strips_window_assignment():
    sample = "window.YTD.tweets.part0 = [{}]"
    assert _PREFIX_RE.sub("", sample, count=1) == "[{}]"


def test_x_prefix_strips_leading_semicolon():
    sample = ";\nwindow.YTD.like.part0 = [{}]"
    assert _PREFIX_RE.sub("", sample, count=1) == "[{}]"


def test_x_twitter_ts_parses_known_format():
    ts = _twitter_ts_to_epoch("Wed Oct 10 20:19:24 +0000 2018")
    assert ts > 0


def test_x_twitter_ts_returns_zero_on_garbage():
    assert _twitter_ts_to_epoch("not a date") == 0.0
    assert _twitter_ts_to_epoch(None) == 0.0
