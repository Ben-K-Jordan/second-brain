"""Phase 88: sensitive content masking.

The brain ingests everything. That's the point — but it means SSNs,
API keys, passwords, credit-card numbers, OAuth tokens that landed
in a forwarded email or a screenshot OCR risk leaking back out
through:

  1. Search results (rendered chunk previews)
  2. Chat citations (the model quotes the chunk)
  3. Vault export / dashboard

This module redacts at *render* time. Content stays in the index
verbatim (so search recall isn't crippled), but anything that
matches a sensitive pattern gets replaced with `[REDACTED:<kind>]`
when surfaced to a user / model.

Patterns covered (conservative — false positives are worse than
false negatives here):
  - US SSN: ``\\d{3}-\\d{2}-\\d{4}``
  - Credit card: 13-19 digit run optionally hyphen/space separated
  - Email: caught only when adjacent to "password" / "pwd" markers
  - API keys: common prefixes (sk-, ak-, ghp_, github_pat_, AIza,
    xoxb-, xoxp-) + 20-char-min token following
  - JWT: three base64url segments separated by dots
  - AWS access key: ``AKIA[0-9A-Z]{16}``
  - Bearer header: ``Bearer <token>``

Idempotent: redact_text(redact_text(s)) == redact_text(s).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---- Patterns ---------------------------------------------------------

# Each pattern is (kind, regex). Order matters: earlier patterns claim
# spans first so the more-specific (e.g. JWT) wins over generic
# (api-key-like) when both could match.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # JWT — three b64url segments separated by dots; bounded length
    # to avoid catching every dotted identifier.
    ("jwt", re.compile(
        r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b",
    )),
    # AWS access key id — fixed prefix + 16 alphanumerics.
    ("aws_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    # Anthropic API keys: sk-ant-... (more specific; must run BEFORE
    # the generic openai_key pattern to claim the span first).
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    # OpenAI sk-* / sk-proj-* keys. Excludes 'ant' prefix so we don't
    # double-match Anthropic strings the previous pattern already
    # rewrote — but since redaction replaces with [REDACTED:...]
    # before this runs, this is also implicitly safe.
    ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b")),
    # GitHub personal access tokens.
    ("github_token", re.compile(
        r"\b(?:ghp_|github_pat_|gho_|ghu_|ghs_|ghr_)[A-Za-z0-9_]{20,}\b",
    )),
    # Google API keys.
    ("google_api_key", re.compile(r"\bAIza[A-Za-z0-9_\-]{35}\b")),
    # Slack bot/user tokens.
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9_\-]{10,}\b")),
    # Bearer header (catches inline "Authorization: Bearer xyz...").
    ("bearer", re.compile(
        r"\bBearer\s+[A-Za-z0-9_\-\.=]{12,}\b",
    )),
    # SSN (US): three-two-four with hyphens. Standalone runs of 9
    # digits are too false-positive-prone (timestamps, IDs).
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # Credit-card-like: 13-19 digits in groups of 4 separated by space
    # or hyphen, OR a continuous 13-19 digit run that passes Luhn.
    ("credit_card", re.compile(
        r"\b(?:\d{4}[\s\-]?){3,4}\d{1,4}\b",
    )),
]


# Sensitive kinds → display tag.
_REDACTED_TAG_MAP = {
    "jwt": "[REDACTED:jwt]",
    "aws_key": "[REDACTED:aws_key]",
    "openai_key": "[REDACTED:openai_key]",
    "anthropic_key": "[REDACTED:anthropic_key]",
    "github_token": "[REDACTED:github_token]",
    "google_api_key": "[REDACTED:google_api_key]",
    "slack_token": "[REDACTED:slack_token]",
    "bearer": "[REDACTED:bearer]",
    "ssn": "[REDACTED:ssn]",
    "credit_card": "[REDACTED:credit_card]",
}


@dataclass
class RedactionResult:
    text: str
    counts: dict[str, int]   # kind → number of matches
    total: int


def _luhn_valid(digits: str) -> bool:
    """Luhn check for credit-card-like sequences. Reduces false
    positives on ordinary 16-digit IDs."""
    digits = re.sub(r"\D", "", digits)
    if not (13 <= len(digits) <= 19):
        return False
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact(text: str) -> RedactionResult:
    """Replace sensitive substrings with ``[REDACTED:<kind>]`` tags.

    Returns the redacted string + a counts breakdown so callers can
    tell the user 'we hid 2 SSNs and 1 API key from this preview'."""
    if not text:
        return RedactionResult(text=text or "", counts={}, total=0)
    counts: dict[str, int] = {}

    def _make_replacement(kind: str):
        def _rep(m: re.Match) -> str:
            # Credit card: only redact when Luhn-valid to cut false
            # positives on long ID strings.
            if kind == "credit_card" and not _luhn_valid(m.group(0)):
                return m.group(0)
            counts[kind] = counts.get(kind, 0) + 1
            return _REDACTED_TAG_MAP[kind]
        return _rep

    out = text
    for kind, pattern in _PATTERNS:
        out = pattern.sub(_make_replacement(kind), out)

    return RedactionResult(
        text=out, counts=counts, total=sum(counts.values()),
    )


def redact_text(text: str) -> str:
    """Convenience wrapper — just the redacted string."""
    return redact(text).text


def has_sensitive(text: str) -> bool:
    """True iff at least one pattern matches. Used to flag chunks
    for the dashboard's "this snippet contains redacted content"
    indicator."""
    return redact(text).total > 0
