"""Resume embedding + posting-fit scoring.

The user drops a resume (or many — different roles you target) at a
configured path; we embed it via the same Voyage embedder used for the
index, and at search/watchlist time we compute cosine similarity between
each posting and the resume to score "fit".

Setup:

    [resume]
    paths = ["~/Documents/resume-pm.md", "~/Documents/resume-eng.md"]
    # Roles can have a name so the dashboard shows which resume matched.

Or set ``RESUME_PATH`` env var to a single file path.

Outputs:

- ``score_postings(resume_doc, citations)`` returns a list of (citation,
  score) sorted by score desc. Score is in [0, 1] (cosine in our
  Voyage embedding space, which is L2-normalised).
- The watchlist runner attaches ``fit_score`` to each citation it
  produced via the jobs / linkedin / etc. paths, so the dashboard can
  surface them as "great fit" / "decent" / "stretch".

The fit score is intentionally cheap — it's one cosine, no LLM call.
For deeper "why this fits" reasoning the chat agent can take over.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .embedder import Embedder

log = logging.getLogger(__name__)


# Bands tuned on Voyage-3 cosine: anything > .55 is genuinely related,
# > .65 is a tight fit, < .40 is mostly noise.
FIT_BANDS = (
    (0.65, "great fit"),
    (0.55, "decent fit"),
    (0.45, "stretch"),
    (0.0, "weak"),
)


@dataclass
class ResumeProfile:
    """One resume's worth of embedded text."""
    name: str
    path: str
    text: str
    embedding: list[float]
    indexed_at: float


def _resume_paths(cfg: Config) -> list[Path]:
    paths: list[Path] = []
    for p in (getattr(cfg, "resume_paths", ()) or ()):
        try:
            rp = Path(p).expanduser().resolve()
        except OSError:
            continue
        if rp.is_file() and rp not in paths:
            paths.append(rp)
    env = os.environ.get("RESUME_PATH", "").strip()
    if env:
        try:
            ep = Path(env).expanduser().resolve()
            if ep.is_file() and ep not in paths:
                paths.append(ep)
        except OSError:
            pass
    return paths


def _read_resume(path: Path) -> str:
    """Read a resume file as text. Markitdown handles PDF/DOCX/etc.;
    plain text is read directly."""
    suffix = path.suffix.lower()
    if suffix in (".md", ".markdown", ".txt", ".rst"):
        try:
            return path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError as e:
            log.warning("resume %s read failed: %s", path, e)
            return ""
    # Anything else: defer to markitdown.
    try:
        from markitdown import MarkItDown
        md = MarkItDown()
        result = md.convert(str(path))
        return (result.text_content or "").strip()
    except Exception as e:  # noqa: BLE001
        log.warning("resume %s convert failed: %s", path, e)
        return ""


def load_resumes(cfg: Config, embedder: Embedder) -> list[ResumeProfile]:
    """Read every configured resume + embed it. Returns one profile per
    file. Errors are logged and skipped; an empty list means no resumes
    are configured (or none could be read), and scoring is skipped.

    Embedded once per call; the watchlist runner caches by passing the
    profile list around explicitly.
    """
    paths = _resume_paths(cfg)
    if not paths:
        return []
    out: list[ResumeProfile] = []
    for p in paths:
        text = _read_resume(p)
        if not text:
            continue
        # Embed as a "query" (single document); Voyage's voyage-3 model
        # treats query + doc symmetrically enough that this is fine.
        try:
            emb = embedder.embed_query(text)
        except Exception as e:  # noqa: BLE001
            log.warning("resume %s embed failed: %s", p, e)
            continue
        out.append(ResumeProfile(
            name=p.stem, path=str(p), text=text,
            embedding=emb, indexed_at=time.time(),
        ))
    return out


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Voyage embeddings are L2-normalised, so this
    reduces to a dot product, but we compute the general form for safety."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def fit_label(score: float) -> str:
    """Translate a cosine score into a human-readable band."""
    for threshold, label in FIT_BANDS:
        if score >= threshold:
            return label
    return "weak"


def score_against_text(
    resumes: list[ResumeProfile], embedder: Embedder, text: str,
) -> tuple[float, str, str] | None:
    """Embed ``text`` once and return (best_score, best_resume_name, label).

    Returns None when no resumes are configured (caller should treat as
    "scoring disabled"). Picks the highest-scoring resume across the user's
    profiles — different resumes for different role types.
    """
    if not resumes or not text:
        return None
    try:
        text_emb = embedder.embed_query(text)
    except Exception as e:  # noqa: BLE001
        log.warning("resume scoring: embedding failed: %s", e)
        return None
    best = (-1.0, "", "weak")
    for r in resumes:
        s = cosine(r.embedding, text_emb)
        if s > best[0]:
            best = (s, r.name, fit_label(s))
    return best
