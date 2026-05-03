"""Auto-tagging — Claude reads each chunk and emits 1-3 short topic tags.

Tags become a parallel facet on top of full-text + vector search: you can
filter ``search_brain`` to chunks that match a tag, or browse by tag in the
dashboard. Useful when you remember the *topic* of something but not the
exact wording.

Cost: ~200 input + 30 output tokens per chunk on Haiku 4.5 → roughly
$0.0003 per chunk. A 6,000-chunk index tags for ~$2.

Opt-in via the CLI: ``secondbrain tag`` (with optional --since-days /
--limit). The indexer doesn't run tagging at ingest — too expensive on
bulk runs and most users will only want tags on a curated subset.
"""

from __future__ import annotations

import json
import logging
import os
import re

from .config import Config

log = logging.getLogger(__name__)


_TAG_SYSTEM_PROMPT = """\
You are tagging a chunk of text from a personal knowledge base. Read the \
chunk and reply with 1-3 short, lowercase, single-or-two-word topic tags \
that describe what the chunk is *about*.

Output format: a JSON array of strings, e.g. ["capital budgeting", "npv"]. \
No prose, no explanation, no markdown — just the JSON array.

Good tags are specific concepts, named entities, or activities — not \
generic words like "document" or "text". Avoid duplicate tags. Stop at 3 \
tags even if you can think of more."""


def _parse_tags(text: str) -> list[str]:
    """Best-effort: try JSON first, fall back to regex extraction."""
    text = text.strip()
    if not text:
        return []
    try:
        # Strip optional ```json fences
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
        data = json.loads(text)
        if isinstance(data, list):
            return [str(t).strip().lower() for t in data if str(t).strip()][:3]
    except json.JSONDecodeError:
        pass
    # Fallback: pull anything in quotes
    matches = re.findall(r'"([^"]+)"', text)
    return [m.strip().lower() for m in matches if m.strip()][:3]


def generate_tags(
    chunk_text: str,
    cfg: Config,
    model: str | None = None,
    *,
    conn=None,
    file_id: int | None = None,
) -> list[str]:
    """Ask Claude for 1-3 tags. Returns [] if SDK / key missing or call fails.

    Tagging is best-effort — we never raise from here, since tagging is
    nice-to-have and shouldn't break a tag pass for a single bad chunk.

    Round 11 (audit-found gap) — when ``conn`` is given, every call
    writes one ``ai_actions`` row.
    """
    def _audit(status: str, response_chars: int = 0,
               cents: float = 0.0, error: str = ""):
        if conn is None:
            return
        try:
            from . import ai_audit
            ai_audit.record_action(
                conn, kind="tag", feature="tag",
                model=model or cfg.tag_model,
                status=status,
                prompt_chars=len(chunk_text),
                response_chars=response_chars,
                cents=cents,
                summary=f"tagged chunk (file_id={file_id})",
                error=error, file_id=file_id,
            )
        except Exception:  # noqa: BLE001
            pass

    if not chunk_text.strip():
        return []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        _audit("no_provider", error="ANTHROPIC_API_KEY not set")
        return []
    try:
        import anthropic
    except ImportError:
        _audit("no_provider", error="anthropic SDK missing")
        return []
    # Round 15 (audit-found gap A1) — gate on the per-feature
    # 'tag' bucket. Tagging runs in a tight loop during bulk index
    # (one call per chunk), so without this an unbounded auto-tag
    # pass on a fresh import could blow past the daily Anthropic
    # cap before the user notices.
    try:
        from .budget import BudgetExceededError, check_budget
        check_budget(cfg, "anthropic", feature="tag")
    except BudgetExceededError as e:
        log.warning("tag: budget exceeded, skipping: %s", e)
        _audit("budget_exceeded", error=str(e)[:200])
        return []
    except ImportError:
        # budget module unavailable — fail open (best-effort tagging).
        pass

    # Round 10 (#4) — redact secret-shaped tokens before send. Tags
    # are about topic / theme, not literal contents — masking doesn't
    # hurt tag quality and prevents API keys from leaking via the
    # auto-tagger pipeline.
    try:
        from .safety import redact_text
        chunk_text = redact_text(chunk_text)
    except ImportError:
        pass
    snippet = chunk_text if len(chunk_text) <= 1500 else chunk_text[:1500] + "..."
    model = model or cfg.tag_model
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=128,
            system=_TAG_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": snippet}],
        )
        text = "\n".join(b.text for b in response.content if b.type == "text")
        tags = _parse_tags(text)
        # Record cost (best-effort)
        cents = 0.0
        try:
            from .budget import estimate_cost, record_usage
            record_usage(
                cfg, "anthropic", model,
                input_tokens=response.usage.input_tokens
                    + response.usage.cache_read_input_tokens,
                output_tokens=response.usage.output_tokens,
                note="tag",
            )
            cents = estimate_cost(
                model,
                input_tokens=response.usage.input_tokens
                    + response.usage.cache_read_input_tokens,
                output_tokens=response.usage.output_tokens,
            ).cents
        except Exception:
            pass
        _audit("success", response_chars=len(text), cents=cents)
        return tags
    except Exception as e:
        log.warning("tag generation failed: %s", e)
        _audit("api_error", error=str(e)[:200])
        return []
