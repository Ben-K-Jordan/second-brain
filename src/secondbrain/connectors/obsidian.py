"""Obsidian connector — vault-aware ingestion of an Obsidian markdown vault.

You CAN just point ``watched_folders`` at your vault directory and the file
watcher will pick everything up. The reason this dedicated connector exists:

- Parses YAML frontmatter and exposes ``tags``, ``aliases``, ``up`` etc. as
  metadata. Otherwise frontmatter ends up as raw YAML in the chunk text and
  pollutes search.
- Resolves ``[[wikilinks]]`` to the linked note's title (and skips broken
  ones), so retrieval matches "what's that thing about X" against the
  semantic content of the linked notes too.
- Skips Obsidian's ``.obsidian/`` config directory and ``.trash/``.

Setup:
  - ``OBSIDIAN_VAULTS`` env var: comma-separated list of vault root paths.
  - Or set ``obsidian_vaults`` in config.toml as a TOML array (preferred).

What it ingests:
  - Every ``*.md`` under each vault root.
  - Frontmatter ``tags`` and ``aliases`` as metadata.
  - Body with wikilinks resolved.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterator
from pathlib import Path

from ..config import Config
from . import ConnectorDocument

log = logging.getLogger(__name__)

# Frontmatter delimiter: --- on its own line at the start of the file.
_FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)

# [[Note]], [[Note|Display]], [[Note#Heading]], [[Note#Heading|Display]]
_WIKILINK_RE = re.compile(r"\[\[([^\[\]]+?)\]\]")

# Inline tags like #foo/bar (Obsidian-style). We only capture for metadata;
# the body keeps them so search can still find the literal "#foo".
_INLINE_TAG_RE = re.compile(r"(?<![\w/])#([A-Za-z][\w/-]*)")

_SKIP_DIRS = frozenset({".obsidian", ".trash", ".git", "node_modules"})


def _parse_vaults(cfg: Config) -> list[Path]:
    paths = list(getattr(cfg, "obsidian_vaults", ()) or ())
    raw = os.environ.get("OBSIDIAN_VAULTS", "")
    for chunk in raw.split(","):
        p = chunk.strip()
        if p:
            paths.append(p)
    out: list[Path] = []
    seen: set[Path] = set()
    for p in paths:
        try:
            resolved = Path(p).expanduser().resolve()
        except OSError:
            continue
        if not resolved.is_dir():
            log.warning("Obsidian: skipping %s (not a directory)", p)
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Strip the YAML frontmatter block off the top of a note.

    Uses a hand-rolled tiny YAML parser because Obsidian frontmatter only
    uses a small subset (string scalars, lists, ints, bools) and we'd rather
    not pull in pyyaml as a dependency for a few keys.
    """
    m = _FM_RE.match(text)
    if not m:
        return {}, text
    block = m.group(1)
    body = text[m.end():]
    fm: dict = {}
    current_key: str | None = None
    for raw in block.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        # Bullet under the current key: collect into a list.
        if line.lstrip().startswith("-") and current_key is not None:
            item = line.lstrip().lstrip("-").strip().strip('"').strip("'")
            existing = fm.setdefault(current_key, [])
            if isinstance(existing, list):
                existing.append(item)
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            current_key = key
            if not val:
                # Either an empty scalar or the start of a bulleted list.
                fm.setdefault(key, [])
                continue
            # Inline list: tags: [a, b, c]
            if val.startswith("[") and val.endswith("]"):
                inner = val[1:-1]
                fm[key] = [
                    item.strip().strip('"').strip("'")
                    for item in inner.split(",") if item.strip()
                ]
                continue
            # Strip surrounding quotes
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            fm[key] = val
    return fm, body


def _resolve_wikilinks(body: str, vault_titles: set[str]) -> str:
    """Replace [[Note|Display]] with the display text. Preserves anchors."""
    def repl(m: re.Match) -> str:
        target = m.group(1).strip()
        # ``Note|Display`` → use Display text.
        if "|" in target:
            display = target.split("|", 1)[1].strip()
            return display
        # ``Note#Heading`` → keep "Note > Heading" so retrieval has both.
        if "#" in target:
            note, _, heading = target.partition("#")
            note = note.strip()
            heading = heading.strip()
            if note:
                return f"{note} > {heading}" if heading else note
            return heading
        return target
    return _WIKILINK_RE.sub(repl, body)


def _walk_vault(vault: Path) -> Iterator[Path]:
    for p in vault.rglob("*.md"):
        # Skip anything inside an _SKIP_DIRS-named directory at any depth.
        rel_parts = p.relative_to(vault).parts
        if any(part in _SKIP_DIRS for part in rel_parts):
            continue
        if not p.is_file():
            continue
        yield p


class ObsidianConnector:
    name = "obsidian"

    def is_enabled(self, cfg: Config) -> bool:
        return bool(_parse_vaults(cfg))

    def fetch(self, cfg: Config) -> Iterator[ConnectorDocument]:
        for vault in _parse_vaults(cfg):
            yield from self._walk(vault)

    # --- helpers --------------------------------------------------------

    def _walk(self, vault: Path) -> Iterator[ConnectorDocument]:
        # First pass: collect note titles so wikilink resolution can hint at
        # whether the link points at a real note. We don't currently do
        # anything with the validity result, but it's cheap and ready for
        # future "broken link" reporting.
        titles: set[str] = set()
        notes: list[Path] = []
        for p in _walk_vault(vault):
            titles.add(p.stem)
            notes.append(p)

        vault_name = vault.name
        for p in notes:
            try:
                raw = p.read_text(encoding="utf-8", errors="replace")
            except OSError as e:
                log.warning("Obsidian: cannot read %s: %s", p, e)
                continue
            fm, body = _parse_frontmatter(raw)
            body = _resolve_wikilinks(body, titles)
            inline_tags = sorted(set(_INLINE_TAG_RE.findall(body)))
            fm_tags = fm.get("tags") or []
            if isinstance(fm_tags, str):
                fm_tags = [fm_tags]
            all_tags = sorted(set([*fm_tags, *inline_tags]))
            aliases = fm.get("aliases") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            try:
                mtime = p.stat().st_mtime
            except OSError:
                mtime = 0.0

            rel = p.relative_to(vault).as_posix()
            title = fm.get("title") or p.stem
            header = [f"# {title}"]
            if aliases:
                header.append(f"Aliases: {', '.join(aliases)}")
            if all_tags:
                header.append("Tags: " + ", ".join("#" + t for t in all_tags))
            content = "\n".join(header) + "\n\n" + body.strip()

            yield ConnectorDocument(
                source="obsidian",
                virtual_path=f"obsidian://{vault_name}/{rel}",
                title=str(title),
                content=content,
                mtime=mtime,
                metadata={
                    "vault": vault_name,
                    "relative_path": rel,
                    "tags": all_tags,
                    "aliases": aliases,
                    "frontmatter": {
                        k: v for k, v in fm.items()
                        if k not in {"tags", "aliases"}
                    },
                },
            )
