"""Named domain presets for watchlists.

A preset is just a curated list of hostnames to pass through to Anthropic's
web_search tool's ``allowed_domains``. Picking the right preset matters a
lot — without it the model wanders across SEO-junk pages, with it you get
clean focused results.

The user can extend or override any preset by passing additional ``--domain``
flags to ``secondbrain watch add``. The CLI merges the preset with the
extras before persisting.

Maintenance: Anthropic's allowed_domains matches subdomains, so listing
``linkedin.com`` covers ``www.linkedin.com``, ``recruiter.linkedin.com``, etc.
"""

from __future__ import annotations

# Hostnames the user might plausibly want; lower-case, no scheme, no path.
PRESETS: dict[str, list[str]] = {
    # Internship / job hunting. Mixes major job boards (LinkedIn / Indeed /
    # Handshake), ATS-hosted boards used by most tech companies (Greenhouse /
    # Lever / Ashby / Workday), startup-focused (Wellfound, Otta, YC), and
    # community-maintained internship lists (Built In, Levels.fyi, Simplify).
    "jobs": [
        "linkedin.com",
        "indeed.com",
        "glassdoor.com",
        "joinhandshake.com",
        "simplify.jobs",
        "lever.co",
        "jobs.lever.co",
        "greenhouse.io",
        "boards.greenhouse.io",
        "job-boards.greenhouse.io",
        "ashbyhq.com",
        "jobs.ashbyhq.com",
        "workdayjobs.com",
        "myworkdayjobs.com",
        "wellfound.com",
        "otta.com",
        "builtin.com",
        "builtinla.com",
        "builtinnyc.com",
        "ycombinator.com",
        "levels.fyi",
        "ripplematch.com",
        "github.com",  # GitHub-hosted internship lists like jobright-ai/...
    ],

    # General news. Skewed toward English-language outlets with paywalled
    # but indexable headlines.
    "news": [
        "nytimes.com",
        "wsj.com",
        "ft.com",
        "bbc.com",
        "reuters.com",
        "apnews.com",
        "bloomberg.com",
        "axios.com",
        "cnbc.com",
        "theverge.com",
        "techcrunch.com",
        "arstechnica.com",
        "wired.com",
        "theatlantic.com",
        "newyorker.com",
        "substack.com",
    ],

    # Markets / finance. Some overlap with `news` but tighter scope.
    "markets": [
        "bloomberg.com",
        "ft.com",
        "wsj.com",
        "marketwatch.com",
        "cnbc.com",
        "reuters.com",
        "finance.yahoo.com",
        "seekingalpha.com",
        "barrons.com",
        "sec.gov",
        "investing.com",
        "morningstar.com",
    ],

    # Academic / research. arXiv first, then peer-reviewed venues.
    "research": [
        "arxiv.org",
        "openreview.net",
        "biorxiv.org",
        "medrxiv.org",
        "nature.com",
        "science.org",
        "semanticscholar.org",
        "researchgate.net",
        "ssrn.com",
    ],

    # AI / ML specifically. Useful for "what AI papers / launches happened today".
    "ai": [
        "arxiv.org",
        "openreview.net",
        "huggingface.co",
        "anthropic.com",
        "openai.com",
        "deepmind.google",
        "google.com/research",
        "research.google",
        "mistral.ai",
        "venturebeat.com",
        "techcrunch.com",
        "theverge.com",
        "github.com",
        "twitter.com",
        "x.com",
        "news.ycombinator.com",
    ],

    # Developer-flavoured. Useful for "what's new in <library>".
    "dev": [
        "github.com",
        "news.ycombinator.com",
        "lobste.rs",
        "reddit.com",
        "stackoverflow.com",
        "dev.to",
        "medium.com",
        "substack.com",
    ],
}


def names() -> list[str]:
    """Sorted list of available preset names for help text and dropdowns."""
    return sorted(PRESETS.keys())


def resolve(preset: str | None, extras: list[str] | None = None) -> list[str] | None:
    """Combine a preset name + extra domains into a single list.

    - ``resolve(None, None)`` -> None  (no domain restriction)
    - ``resolve("jobs", None)`` -> the jobs preset
    - ``resolve(None, ["x.com"])`` -> ["x.com"]
    - ``resolve("jobs", ["mycompany.com"])`` -> jobs preset + ["mycompany.com"]
    """
    out: list[str] = []
    if preset:
        if preset not in PRESETS:
            raise ValueError(
                f"Unknown preset {preset!r}. Available: {', '.join(names())}"
            )
        out.extend(PRESETS[preset])
    for d in extras or []:
        d = d.strip().lower()
        # Defensive: users sometimes paste full URLs. We want the bare host.
        for prefix in ("https://", "http://"):
            if d.startswith(prefix):
                d = d[len(prefix):]
                break
        d = d.rstrip("/").split("/", 1)[0]  # drop any path component
        if d and d not in out:
            out.append(d)
    return out or None
