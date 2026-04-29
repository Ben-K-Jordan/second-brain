"""Named-entity extraction. Default: spaCy's small English model.

Entities (people, organizations, places, dates, money, etc.) are pulled per
chunk and stored alongside in the DB so MCP tools can answer:

- "what files mention <entity>?"
- "who/what comes up most in my brain?"
- "timeline of mentions of <entity>"

This is the foundation of the knowledge-graph layer; future phases add
entity-merging (canonicalisation), relation extraction, and graph queries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .config import Config

log = logging.getLogger(__name__)


# spaCy NER labels worth keeping for a personal brain. We exclude noisy ones
# (CARDINAL, ORDINAL, PERCENT, QUANTITY, TIME) by default - they bloat the
# table without adding searchable value. Users can tweak via config.
DEFAULT_KEEP_LABELS: frozenset[str] = frozenset({
    "PERSON",       # Sarah, J.K. Rowling
    "ORG",          # Anthropic, Cornell
    "GPE",          # New York, Japan (geo-political entities)
    "LOC",          # Mt. Everest, Pacific Ocean
    "FAC",          # buildings, airports, highways
    "PRODUCT",      # Tesla Model 3, iPhone
    "EVENT",        # WWII, the Olympics
    "WORK_OF_ART",  # Mona Lisa, Hamlet
    "LAW",          # GDPR, the Patriot Act
    "DATE",         # 2024, last Tuesday
    "MONEY",        # $5, €100
    "LANGUAGE",     # Python, English
    "NORP",         # nationalities/religions/political groups
})


@dataclass(frozen=True)
class Entity:
    text: str
    label: str

    def normalized(self) -> str:
        """Canonical form for de-duping near-matches within a single chunk."""
        return " ".join(self.text.split()).strip()


@runtime_checkable
class EntityExtractor(Protocol):
    name: str

    def extract(self, text: str) -> list[Entity]: ...


class SpacyEntityExtractor:
    """spaCy NER. Loads on construction; one model serves many chunks."""

    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        keep_labels: frozenset[str] = DEFAULT_KEEP_LABELS,
    ):
        try:
            import spacy
        except ImportError as e:
            raise ImportError(
                "Entity extraction requires the [ner] extra. "
                "Install with: pip install -e .[ner]"
            ) from e

        try:
            # Disable components we don't need - parser/tagger are slow and
            # NER doesn't depend on them.
            self._nlp = spacy.load(
                model_name,
                disable=["parser", "tagger", "lemmatizer", "attribute_ruler"],
            )
        except OSError as e:
            raise RuntimeError(
                f"spaCy model '{model_name}' is not installed. Download it once with:\n"
                f"  python -m spacy download {model_name}"
            ) from e

        self.name = f"spacy-{model_name}"
        self._keep = keep_labels

    def extract(self, text: str) -> list[Entity]:
        if not text or not text.strip():
            return []
        # spaCy has a default 1MB cap on docs; chunk text fits comfortably.
        doc = self._nlp(text)
        seen: set[tuple[str, str]] = set()
        out: list[Entity] = []
        for ent in doc.ents:
            if ent.label_ not in self._keep:
                continue
            txt = " ".join(ent.text.split()).strip()
            if not txt or len(txt) > 200:
                continue
            key = (txt.casefold(), ent.label_)
            if key in seen:
                continue
            seen.add(key)
            out.append(Entity(text=txt, label=ent.label_))
        return out


def make_entity_extractor(cfg: Config) -> EntityExtractor | None:
    if not cfg.entities_enabled:
        return None
    return SpacyEntityExtractor(model_name=cfg.spacy_model)
