"""Configuration, paths, and ignore rules for second-brain."""

from __future__ import annotations

import fnmatch
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_data_path

APP_NAME = "secondbrain"

DEFAULT_IGNORE_GLOBS: tuple[str, ...] = (
    ".git/*",
    "*/.git/*",
    "node_modules/*",
    "*/node_modules/*",
    ".venv/*",
    "*/.venv/*",
    "venv/*",
    "*/venv/*",
    "__pycache__/*",
    "*/__pycache__/*",
    ".pytest_cache/*",
    "*/.pytest_cache/*",
    ".mypy_cache/*",
    "*/.mypy_cache/*",
    ".ruff_cache/*",
    "*/.ruff_cache/*",
    "AppData/*",
    "*/AppData/*",
    ".cache/*",
    "*/.cache/*",
    ".secondbrain/*",
    "*/.secondbrain/*",
    "*.env",
    "*.env.*",
    ".env",
    ".env.*",
    "*.key",
    "*.pem",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "id_ed25519*",
    "*secret*",
    "*credentials*",
    "*password*",
    "*.exe",
    "*.dll",
    "*.so",
    "*.dylib",
    "*.msi",
    "*.iso",
    "*.dmg",
    "*.zip",
    "*.7z",
    "*.tar",
    "*.tar.gz",
    "*.tgz",
    "*.rar",
    "*.bin",
    "*.dat",
    "*.db",
    "*.sqlite",
    "*.sqlite3",
)

DOCUMENT_EXTENSIONS: frozenset[str] = frozenset({
    ".md", ".markdown", ".txt", ".rst", ".org",
    ".pdf", ".docx", ".doc", ".odt", ".rtf",
    ".pptx", ".ppt", ".odp",
    ".xlsx", ".xls", ".csv", ".ods", ".tsv",
    ".html", ".htm", ".xml", ".epub",
    ".json", ".yaml", ".yml", ".toml", ".ini",
})

CODE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go", ".java",
    ".kt", ".swift", ".c", ".cpp", ".h", ".hpp", ".cs", ".rb",
    ".php", ".lua", ".sh", ".bash", ".zsh", ".fish", ".ps1",
    ".sql", ".r", ".scala", ".clj", ".ex", ".exs", ".elm",
})

AUDIO_VIDEO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus",
    ".mp4", ".mov", ".avi", ".mkv", ".webm",
})

IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".heic",
})

# Backwards-compatibility alias; remove after one release if nothing depends on it.
MEDIA_EXTENSIONS: frozenset[str] = AUDIO_VIDEO_EXTENSIONS | IMAGE_EXTENSIONS


def app_data_dir() -> Path:
    """Return the cross-platform app data directory."""
    p = user_data_path(APP_NAME, appauthor=False, ensure_exists=True)
    return Path(p)


@dataclass
class Config:
    """Runtime configuration."""

    # Paths
    data_dir: Path = field(default_factory=app_data_dir)
    watched_folders: list[Path] = field(default_factory=list)

    # Embedder
    embedder_provider: str = "auto"  # "auto" | "voyage" | "local"
    voyage_model: str = "voyage-3"
    voyage_api_key: str | None = None
    local_model: str = "all-MiniLM-L6-v2"

    # Indexing
    chunk_size: int = 800
    chunk_overlap: int = 150
    max_file_bytes: int = 200 * 1024 * 1024  # 200 MB; media bypasses this when transcribed
    extra_ignore_globs: tuple[str, ...] = ()

    # Search
    hybrid_alpha: float = 0.5  # weight: 0=keyword only, 1=vector only
    adaptive_alpha: bool = True  # auto-tune alpha per query (proper nouns -> BM25, prose -> vector)
    rerank_enabled: bool = True
    rerank_model: str = "rerank-2-lite"
    rerank_overfetch: int = 50  # how many candidates to fetch before reranking down to k

    # Time-decay scoring: gently boost recently-modified files. Half-life is in days.
    time_decay_enabled: bool = True
    time_decay_weight: float = 0.1  # 0=ignore time, 1=time only
    time_decay_half_life_days: float = 365.0

    # HyDE query rewriting: for vague conceptual queries, draft a hypothetical
    # answer and embed *that*. Big retrieval bump on "what was that thing about X"
    # style searches. Requires ANTHROPIC_API_KEY; falls back silently to raw
    # query when unavailable. ~1ยข per query at default Haiku 4.5 model.
    hyde_enabled: bool = False
    hyde_model: str = "claude-haiku-4-5"

    # Source-aware ranking: lift personal-content paths over passive downloads.
    # Match is case-insensitive substring, so "/Documents/" hits any path
    # containing that segment. Multiplier of 1.0 = no change.
    personal_path_prefixes: tuple[str, ...] = ("/Documents/", "/notes/", "/OneDrive/")
    personal_path_boost: float = 1.3
    download_path_prefixes: tuple[str, ...] = ("/Downloads/",)
    download_path_demote: float = 0.85

    # Media transcription (audio/video -> text via faster-whisper)
    transcribe_enabled: bool = True
    whisper_model_size: str = "small"  # tiny/base/small/medium/large-v3

    # Image OCR (text-in-image -> text via Tesseract)
    ocr_enabled: bool = True
    ocr_lang: str = "eng"

    # Named-entity recognition (people, orgs, places, dates, money, ...) via spaCy
    entities_enabled: bool = True
    spacy_model: str = "en_core_web_sm"

    # Multimodal image embeddings: semantic image search via voyage-multimodal-3.
    # Images are also OCR'd in parallel; the two paths complement each other.
    image_embed_enabled: bool = True
    multimodal_model: str = "voyage-multimodal-3"

    # Daily briefing — Claude reads what's new and writes a summary.
    # Requires ANTHROPIC_API_KEY. Default model is Opus 4.7 (most capable);
    # swap to claude-sonnet-4-6 for lower cost or claude-haiku-4-5 for fastest.
    briefing_model: str = "claude-opus-4-7"
    briefing_max_tokens: int = 4096

    # Auto-tagging — Claude assigns 1-3 topic tags per chunk.
    # Run on demand via `secondbrain tag`; defaults to Haiku 4.5 (~$0.0003/chunk).
    tag_model: str = "claude-haiku-4-5"

    # Daily spend caps in cents (so $5 = 500). Refuses paid calls once today's
    # cumulative spend hits the cap. Set to 0 to disable. Defense in depth -
    # catches runaway loops fast; not a substitute for provider-side billing limits.
    daily_budget_cents_voyage: int = 500
    daily_budget_cents_anthropic: int = 500

    @property
    def db_path(self) -> Path:
        return self.data_dir / "index.db"

    @property
    def config_path(self) -> Path:
        return self.data_dir / "config.toml"

    @property
    def ignore_globs(self) -> tuple[str, ...]:
        return DEFAULT_IGNORE_GLOBS + tuple(self.extra_ignore_globs)


def default_config_toml() -> str:
    return """\
# second-brain config
# Edit this file to customize behavior. Restart any running daemon to apply.

# Folders to watch and index. Paths can be absolute or use ~ for home.
# Used by `secondbrain daemon` and `secondbrain tray`.
# Example:
# watched_folders = ["C:/Users/me/Downloads", "C:/Users/me/Documents/notes"]
watched_folders = []

# Embedder: "auto" picks Voyage if VOYAGE_API_KEY is set, else local.
# Override with "voyage" or "local".
embedder_provider = "auto"
voyage_model = "voyage-3"
local_model = "all-MiniLM-L6-v2"

# Chunking
chunk_size = 800
chunk_overlap = 150

# Hard cap; files this large will be skipped unless they're media (which get transcribed).
max_file_bytes = 209715200  # 200 MB

# Extra glob patterns to skip (added to built-in defaults).
extra_ignore_globs = []

# Hybrid search: 0.0 = keyword only, 1.0 = vector only.
hybrid_alpha = 0.5

# Cross-encoder reranking. When enabled, hybrid search over-fetches candidates
# and reranks them with a cross-encoder for ~30% precision lift on top results.
# Requires Voyage API access. Adds ~50-100ms latency per query and a small
# extra API cost (rerank-2-lite is ~$0.05/1M tokens).
rerank_enabled = true
rerank_model = "rerank-2-lite"
rerank_overfetch = 50

# Adaptive alpha: per-query tuning of vector vs keyword weight. Proper-noun-
# heavy queries get more BM25; conceptual prose gets more vector.
adaptive_alpha = true

# Time-decay scoring: gently boost recently-modified files in ranking. With a
# half-life of 365 days, a year-old file gets ~50% of the recency bonus a
# brand-new file gets. Set time_decay_weight to 0 to disable entirely.
time_decay_enabled = true
time_decay_weight = 0.1
time_decay_half_life_days = 365.0

# HyDE query rewriting: for vague conceptual queries, ask Claude Haiku to
# draft a hypothetical answer and embed that for vector search. Big quality
# bump on "what was that thing about X" queries. Requires ANTHROPIC_API_KEY;
# silently falls back to raw query when unavailable. Costs ~1c per qualifying
# query (most short queries skip HyDE entirely - see should_use_hyde).
hyde_enabled = false
hyde_model = "claude-haiku-4-5"

# Source-aware ranking: lift personal-content paths over passive downloads.
# Match is case-insensitive substring. Multiplier of 1.0 means no change.
# Tweak the path lists to fit how YOU organize your stuff.
personal_path_prefixes = ["/Documents/", "/notes/", "/OneDrive/"]
personal_path_boost = 1.3
download_path_prefixes = ["/Downloads/"]
download_path_demote = 0.85

# Media transcription. When enabled, audio and video files are transcribed
# locally via faster-whisper and the transcript flows into the regular index.
# Requires the [whisper] extra: pip install -e .[whisper]
transcribe_enabled = true
whisper_model_size = "small"  # tiny/base/small/medium/large-v3

# Image OCR. When enabled, image files are OCR'd via Tesseract and the text
# flows into the regular index. Requires the [ocr] extra AND a Tesseract
# binary on PATH (see README for install).
ocr_enabled = true
ocr_lang = "eng"

# Named-entity recognition. When enabled, spaCy extracts people, orgs,
# places, dates, money, etc. per chunk and stores them for graph queries.
# Requires the [ner] extra AND a one-time model download:
#   python -m spacy download en_core_web_sm
# For higher-quality NER on prose, swap to en_core_web_lg (~750MB):
#   python -m spacy download en_core_web_lg
#   then set spacy_model = "en_core_web_lg" below
entities_enabled = true
spacy_model = "en_core_web_sm"

# Multimodal image embeddings. When enabled, images are embedded via
# voyage-multimodal-3 alongside being OCR'd, so semantic image search
# ("the diagram of X") works in addition to text-in-image search.
# Requires VOYAGE_API_KEY (no extra needed - voyage SDK already pulled in).
image_embed_enabled = true
multimodal_model = "voyage-multimodal-3"

# Daily briefing - Claude summarises what's new in your brain.
# Requires ANTHROPIC_API_KEY. Switch to claude-sonnet-4-6 for cheaper runs,
# or claude-haiku-4-5 for fastest.
briefing_model = "claude-opus-4-7"
briefing_max_tokens = 4096

# Auto-tagging - Claude assigns 1-3 topic tags per chunk on demand.
# Run via `secondbrain tag`. Haiku 4.5 is cheap enough to tag a 6K-chunk
# index for ~$2; bump to claude-sonnet-4-6 for nicer tags at higher cost.
tag_model = "claude-haiku-4-5"

# Daily spend caps in cents ($5 = 500). Refuses paid API calls once today's
# spend hits the cap. Set to 0 to disable. Defense in depth - catches runaway
# loops fast. Not a substitute for provider-side billing limits, but a useful
# floor on damage when something misbehaves.
daily_budget_cents_voyage = 500
daily_budget_cents_anthropic = 500
"""


def load_config(path: Path | None = None) -> Config:
    """Load config from disk, falling back to defaults."""
    cfg = Config()
    config_path = path or cfg.config_path
    if config_path.exists():
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        if "watched_folders" in data:
            cfg.watched_folders = [Path(p).expanduser() for p in data["watched_folders"]]
        if "embedder_provider" in data:
            cfg.embedder_provider = data["embedder_provider"]
        if "voyage_model" in data:
            cfg.voyage_model = data["voyage_model"]
        if "local_model" in data:
            cfg.local_model = data["local_model"]
        if "chunk_size" in data:
            cfg.chunk_size = int(data["chunk_size"])
        if "chunk_overlap" in data:
            cfg.chunk_overlap = int(data["chunk_overlap"])
        if "max_file_bytes" in data:
            cfg.max_file_bytes = int(data["max_file_bytes"])
        if "extra_ignore_globs" in data:
            cfg.extra_ignore_globs = tuple(data["extra_ignore_globs"])
        if "hybrid_alpha" in data:
            cfg.hybrid_alpha = float(data["hybrid_alpha"])
        if "rerank_enabled" in data:
            cfg.rerank_enabled = bool(data["rerank_enabled"])
        if "rerank_model" in data:
            cfg.rerank_model = data["rerank_model"]
        if "rerank_overfetch" in data:
            cfg.rerank_overfetch = int(data["rerank_overfetch"])
        if "adaptive_alpha" in data:
            cfg.adaptive_alpha = bool(data["adaptive_alpha"])
        if "time_decay_enabled" in data:
            cfg.time_decay_enabled = bool(data["time_decay_enabled"])
        if "time_decay_weight" in data:
            cfg.time_decay_weight = float(data["time_decay_weight"])
        if "time_decay_half_life_days" in data:
            cfg.time_decay_half_life_days = float(data["time_decay_half_life_days"])
        if "hyde_enabled" in data:
            cfg.hyde_enabled = bool(data["hyde_enabled"])
        if "hyde_model" in data:
            cfg.hyde_model = data["hyde_model"]
        if "personal_path_prefixes" in data:
            cfg.personal_path_prefixes = tuple(data["personal_path_prefixes"])
        if "personal_path_boost" in data:
            cfg.personal_path_boost = float(data["personal_path_boost"])
        if "download_path_prefixes" in data:
            cfg.download_path_prefixes = tuple(data["download_path_prefixes"])
        if "download_path_demote" in data:
            cfg.download_path_demote = float(data["download_path_demote"])
        if "transcribe_enabled" in data:
            cfg.transcribe_enabled = bool(data["transcribe_enabled"])
        if "whisper_model_size" in data:
            cfg.whisper_model_size = data["whisper_model_size"]
        if "ocr_enabled" in data:
            cfg.ocr_enabled = bool(data["ocr_enabled"])
        if "ocr_lang" in data:
            cfg.ocr_lang = data["ocr_lang"]
        if "entities_enabled" in data:
            cfg.entities_enabled = bool(data["entities_enabled"])
        if "spacy_model" in data:
            cfg.spacy_model = data["spacy_model"]
        if "image_embed_enabled" in data:
            cfg.image_embed_enabled = bool(data["image_embed_enabled"])
        if "multimodal_model" in data:
            cfg.multimodal_model = data["multimodal_model"]
        if "briefing_model" in data:
            cfg.briefing_model = data["briefing_model"]
        if "briefing_max_tokens" in data:
            cfg.briefing_max_tokens = int(data["briefing_max_tokens"])
        if "tag_model" in data:
            cfg.tag_model = data["tag_model"]
        if "daily_budget_cents_voyage" in data:
            cfg.daily_budget_cents_voyage = int(data["daily_budget_cents_voyage"])
        if "daily_budget_cents_anthropic" in data:
            cfg.daily_budget_cents_anthropic = int(data["daily_budget_cents_anthropic"])

    cfg.voyage_api_key = os.environ.get("VOYAGE_API_KEY")
    return cfg


def write_default_config(cfg: Config) -> None:
    """Write a default config.toml if none exists."""
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.config_path.exists():
        cfg.config_path.write_text(default_config_toml(), encoding="utf-8")


def is_ignored(path: Path, ignore_globs: tuple[str, ...]) -> bool:
    """Test whether a path matches any ignore glob."""
    s = path.as_posix()
    name = path.name
    for pattern in ignore_globs:
        if fnmatch.fnmatch(s, pattern) or fnmatch.fnmatch(name, pattern):
            return True
    return False


def classify_file(path: Path) -> str:
    """Return one of: 'document', 'code', 'audio_video', 'image', 'other'."""
    ext = path.suffix.lower()
    if ext in DOCUMENT_EXTENSIONS:
        return "document"
    if ext in CODE_EXTENSIONS:
        return "code"
    if ext in AUDIO_VIDEO_EXTENSIONS:
        return "audio_video"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    return "other"
