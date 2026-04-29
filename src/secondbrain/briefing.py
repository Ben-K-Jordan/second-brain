"""Daily briefing — Claude reads what's new in your brain and writes a summary.

Calls Claude Opus 4.7 with adaptive thinking. The system prompt is cached so
repeated briefings only pay the ~10% cache-read rate on the framing.

Output: a markdown briefing covering:
  - What entered your brain (file counts by kind, notable additions)
  - Recurring entities (people, orgs, projects that show up)
  - Connected discovery (recent files that relate to older indexed content)
  - Anything worth your attention (anomalies, gaps, unfinished threads)
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass

from .budget import BudgetExceededError, check_budget, record_usage
from .config import Config

log = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
You are the user's second-brain briefing writer. The user runs a personal local-first \
knowledge base that auto-ingests files, transcripts, screenshots, and URLs. They've \
asked you to scan what entered the brain recently and produce a short, useful briefing.

Your output is markdown. Be concise and specific - the user will skim, not read. \
Quote file paths verbatim when referencing them. Don't summarize what you don't have \
evidence for. Don't pad with preamble or sign-offs.

Structure your briefing with these sections, in order, omitting any that have nothing useful:

## What's new
2-5 bullets covering the most notable additions. Group when sensible - "5 PDFs from \
your finance class" is better than listing all 5. Highlight anything that looks \
high-signal (a long-form document, a meeting transcript, a one-off URL).

## Recurring threads
Entities (people, orgs, projects) that appear across multiple recent files. Skip \
single-mention noise. If the user has been working on the same theme, name it.

## Worth your attention
Anomalies or gaps - e.g. "5 PDFs about a topic but no notes on it", "this transcript \
references a doc you haven't indexed", "you saved this article 3 days ago and haven't \
opened it." If nothing fits, omit this section.

If the input contains no meaningful new content, say so in one sentence. Don't fabricate.
"""


@dataclass
class BriefingDigest:
    """Structured view of what's new, fed to the LLM."""

    hours: int
    file_count: int
    chunk_count: int
    files_by_kind: dict[str, int]
    recent_files: list[tuple[str, str, float]]  # (path, kind, mtime)
    top_recent_entities: list[tuple[str, str, int]]  # (text, label, count)
    sample_chunks: list[tuple[str, int, str]]  # (path, chunk_index, text)


def collect_digest(conn: sqlite3.Connection, hours: int = 24) -> BriefingDigest:
    """Pull a structured digest of what's been indexed in the last N hours."""
    cutoff = time.time() - hours * 3600

    files = conn.execute(
        "SELECT id, path, kind, mtime FROM files "
        "WHERE indexed_at >= ? ORDER BY indexed_at DESC",
        (cutoff,),
    ).fetchall()
    file_ids = [r["id"] for r in files]

    chunk_count = 0
    sample_chunks: list[tuple[str, int, str]] = []
    top_entities: list[tuple[str, str, int]] = []
    if file_ids:
        placeholders = ",".join("?" * len(file_ids))
        chunk_count = conn.execute(
            f"SELECT COUNT(*) AS c FROM chunks WHERE file_id IN ({placeholders})",
            file_ids,
        ).fetchone()["c"]

        # Two sample chunks per file (capped at 24 total) — enough flavor for the model
        # without blowing the prompt budget on a big ingest day.
        rows = conn.execute(
            f"SELECT c.text, c.chunk_index, f.path "
            f"FROM chunks c JOIN files f ON f.id = c.file_id "
            f"WHERE f.id IN ({placeholders}) "
            f"ORDER BY f.id, c.chunk_index LIMIT 24",
            file_ids,
        ).fetchall()
        sample_chunks = [(r["path"], r["chunk_index"], r["text"]) for r in rows]

        ent_rows = conn.execute(
            f"SELECT e.text, e.label, COUNT(DISTINCT e.chunk_id) AS n "
            f"FROM entities e JOIN chunks c ON c.id = e.chunk_id "
            f"WHERE c.file_id IN ({placeholders}) "
            f"  AND e.label IN ('PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'WORK_OF_ART') "
            f"GROUP BY e.text_lower, e.label "
            f"ORDER BY n DESC LIMIT 20",
            file_ids,
        ).fetchall()
        top_entities = [(r["text"], r["label"], r["n"]) for r in ent_rows]

    return BriefingDigest(
        hours=hours,
        file_count=len(files),
        chunk_count=chunk_count,
        files_by_kind=dict(Counter(r["kind"] for r in files)),
        recent_files=[(r["path"], r["kind"], r["mtime"]) for r in files[:30]],
        top_recent_entities=top_entities,
        sample_chunks=sample_chunks,
    )


def _format_digest_for_llm(d: BriefingDigest) -> str:
    """Render the digest as a single user-message string."""
    lines = [f"# Brain digest — last {d.hours} hours", ""]
    if d.file_count == 0:
        lines.append("No new files were indexed in this window.")
        return "\n".join(lines)

    lines.append(f"**Counts**: {d.file_count} files, {d.chunk_count} chunks")
    if d.files_by_kind:
        kinds = ", ".join(f"{k}={v}" for k, v in sorted(d.files_by_kind.items()))
        lines.append(f"**By kind**: {kinds}")
    lines.append("")

    if d.recent_files:
        lines.append("## Files added (newest first)")
        for path, kind, mtime in d.recent_files:
            age_h = (time.time() - mtime) / 3600
            lines.append(f"- [{kind}] {path}  ({age_h:.1f}h ago)")
        lines.append("")

    if d.top_recent_entities:
        lines.append("## Top entities in these files")
        for text, label, n in d.top_recent_entities:
            lines.append(f"- {n:3d}× [{label}] {text}")
        lines.append("")

    if d.sample_chunks:
        lines.append("## Sample chunks (for flavor — not exhaustive)")
        for path, idx, text in d.sample_chunks:
            snippet = text if len(text) <= 600 else text[:600] + "..."
            lines.append(f"### {path} (chunk {idx})")
            lines.append(snippet)
            lines.append("")

    return "\n".join(lines)


def generate_briefing(
    conn: sqlite3.Connection,
    cfg: Config,
    hours: int = 24,
) -> str:
    """Generate a markdown briefing of what's entered the brain in the last N hours.

    Requires ANTHROPIC_API_KEY in the environment. Uses Claude Opus 4.7 with
    adaptive thinking. Prompt cache lives on the system message so repeat
    briefings amortise the framing tokens.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return (
            "(daily briefing requires ANTHROPIC_API_KEY — set it via "
            "`[Environment]::SetEnvironmentVariable(\"ANTHROPIC_API_KEY\", ..., \"User\")` "
            "and restart the dashboard / MCP server.)"
        )

    try:
        import anthropic
    except ImportError:
        return "(install with `pip install anthropic` to enable briefings)"

    digest = collect_digest(conn, hours=hours)
    if digest.file_count == 0:
        return f"# Daily briefing\n\nNothing new in the last {hours} hours."

    user_content = _format_digest_for_llm(digest)

    try:
        check_budget(cfg, "anthropic")
    except BudgetExceededError as e:
        return f"(briefing skipped: {e})"

    client = anthropic.Anthropic()
    try:
        response = client.messages.create(
            model=cfg.briefing_model,
            max_tokens=cfg.briefing_max_tokens,
            thinking={"type": "adaptive"},
            system=[{
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_content}],
        )
    except anthropic.APIError as e:
        log.warning("briefing API call failed: %s", e)
        return f"(briefing failed: {e})"

    record_usage(
        cfg, "anthropic", cfg.briefing_model,
        input_tokens=response.usage.input_tokens + response.usage.cache_read_input_tokens,
        output_tokens=response.usage.output_tokens,
        note=f"briefing/{digest.hours}h",
    )

    text_parts: list[str] = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
    body = "\n".join(text_parts).strip() or "(empty response)"

    usage = response.usage
    footer = (
        f"\n\n---\n"
        f"_{cfg.briefing_model} · "
        f"in: {usage.input_tokens} (cached: {usage.cache_read_input_tokens}) · "
        f"out: {usage.output_tokens} · "
        f"sources: {digest.file_count} files, {digest.chunk_count} chunks_"
    )
    return body + footer
