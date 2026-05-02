"""Typer CLI entry point."""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import typer

# Force UTF-8 for stdout/stderr so non-ASCII content (smart quotes, em-dashes,
# Unicode punctuation in extracted text) renders correctly on Windows consoles
# that default to cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from . import __version__
from .config import Config, load_config, write_default_config
from .db import connect, connect_readonly, init_schema, stats
from .embedder import make_embedder
from .entities import make_entity_extractor
from .image_embedder import make_image_embedder
from .imager import make_ocr_engine
from .indexer import (
    IndexResult,
    dedupe_existing,
    index_folder,
    index_text,
    index_url,
    walk_folder,
)
from .reranker import make_reranker
from .search import hybrid_search
from .transcriber import make_transcriber

app = typer.Typer(
    name="secondbrain",
    help="A personal knowledge base that auto-ingests files and exposes them to AI assistants.",
    no_args_is_help=True,
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _open_state(cfg: Config):
    embedder = make_embedder(cfg)
    conn = connect(cfg.db_path)
    init_schema(conn, embedder.dim, embedder.name)
    return conn, embedder


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"second-brain {__version__}")


@app.command()
def init() -> None:
    """Create the data directory and a default config file.

    The index database is created lazily when you first run `index` or `search`.
    """
    cfg = load_config()
    write_default_config(cfg)
    console.print(f"[green]OK[/] Initialised at [cyan]{cfg.data_dir}[/]")
    console.print(f"  config: [cyan]{cfg.config_path}[/]")
    console.print(f"  db (will be created on first index): [cyan]{cfg.db_path}[/]")
    if cfg.voyage_api_key:
        console.print(f"  embedder: [cyan]Voyage ({cfg.voyage_model})[/] - VOYAGE_API_KEY detected")
    else:
        console.print(
            "  embedder: [yellow]not configured[/]\n"
            "    - Set VOYAGE_API_KEY for the API embedder, or\n"
            "    - Run `pip install -e .[local]` for the local fallback."
        )


@app.command()
def status() -> None:
    """Show what's currently in the index. Read-only — safe to run while the
    daemon is bulk-indexing."""
    cfg = load_config()
    try:
        conn = connect_readonly(cfg.db_path)
    except FileNotFoundError as e:
        console.print(f"[yellow]{e}[/]")
        raise typer.Exit(code=1) from None
    s = stats(conn)
    table = Table(show_header=False, box=None)
    table.add_row("Files", str(s["files"]))
    table.add_row("Aliases", f"{s.get('aliases', 0)} (duplicate paths)")
    table.add_row("Chunks", str(s["chunks"]))
    table.add_row("Entities", str(s.get("entities", 0)))
    table.add_row("Embedder", f"{s['embedder']} (dim {s['embedding_dim']})")
    table.add_row("DB path", str(cfg.db_path))
    table.add_row("Watched folders", ", ".join(str(p) for p in cfg.watched_folders) or "(none)")
    console.print(table)
    conn.close()


@app.command()
def dedupe(
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n",
        help="Report what would be aliased without changing anything.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="List every (canonical, duplicate) pair that gets converted.",
    ),
) -> None:
    """Find files in the index sharing a content hash and convert duplicates
    to aliases. Frees their chunks/entities/vectors, keeps the path.

    Idempotent. Safe to run repeatedly. Worth running once after bulk-indexing
    folders with overlap (e.g. Downloads + OneDrive). Acquires a write lock,
    so it's safest to run with the daemon stopped."""
    cfg = load_config()
    conn, _ = _open_state(cfg)
    if dry_run:
        console.print("[yellow]DRY RUN[/] - nothing will be modified")
    console.print("Scanning for content-hash duplicates...")
    result = dedupe_existing(conn, dry_run=dry_run)
    aliased = result.get("aliased") or []
    if verbose and aliased:
        for canonical, dup in aliased:
            console.print(f"  [dim]alias[/] {dup}\n         [dim]→[/] {canonical}")
    suffix = " (dry-run)" if dry_run else ""
    console.print(
        f"[green]Done.[/]{suffix} groups={result['groups_with_duplicates']} "
        f"converted={result['duplicate_files_converted']} "
        f"chunks_freed={result['chunks_freed']}"
    )
    conn.close()


@app.command()
def index(
    folder: Path = typer.Argument(..., help="Folder to index (recursively)."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    no_transcribe: bool = typer.Option(
        False, "--no-transcribe", help="Skip Whisper transcription for audio/video."
    ),
    no_ocr: bool = typer.Option(False, "--no-ocr", help="Skip OCR for images."),
    no_entities: bool = typer.Option(
        False, "--no-entities", help="Skip spaCy entity extraction."
    ),
) -> None:
    """One-shot index a folder. Skips files whose content is unchanged."""
    _setup_logging(verbose)
    folder = folder.expanduser().resolve()
    if not folder.exists():
        console.print(f"[red]Folder does not exist:[/] {folder}")
        raise typer.Exit(code=1)

    cfg = load_config()
    if no_transcribe:
        cfg.transcribe_enabled = False
    if no_ocr:
        cfg.ocr_enabled = False
    if no_entities:
        cfg.entities_enabled = False
    conn, embedder = _open_state(cfg)

    transcriber = None
    if cfg.transcribe_enabled:
        try:
            transcriber = make_transcriber(cfg)
            if transcriber:
                console.print(
                    f"[dim]Transcriber:[/] {transcriber.name} "
                    f"(loads on first audio/video file)"
                )
        except ImportError as e:
            console.print(f"[yellow]Transcription disabled:[/] {e}")
            transcriber = None

    ocr_engine = None
    if cfg.ocr_enabled:
        try:
            ocr_engine = make_ocr_engine(cfg)
            if ocr_engine:
                console.print(f"[dim]OCR:[/] {ocr_engine.name}")
        except ImportError as e:
            console.print(f"[yellow]OCR disabled:[/] {e}")
            ocr_engine = None

    entity_extractor = None
    if cfg.entities_enabled:
        try:
            entity_extractor = make_entity_extractor(cfg)
            if entity_extractor:
                console.print(f"[dim]Entities:[/] {entity_extractor.name}")
        except (ImportError, RuntimeError) as e:
            console.print(f"[yellow]Entity extraction disabled:[/] {e}")
            entity_extractor = None

    image_embedder = None
    if cfg.image_embed_enabled:
        try:
            image_embedder = make_image_embedder(cfg)
            if image_embedder:
                console.print(f"[dim]Image embedder:[/] {image_embedder.name}")
        except Exception as e:
            console.print(f"[yellow]Multimodal image embedder disabled:[/] {e}")
            image_embedder = None

    candidates = list(walk_folder(folder, cfg))
    console.print(
        f"Scanning [cyan]{folder}[/]: {len(candidates)} candidate file(s) "
        f"using [cyan]{embedder.name}[/]"
    )

    counts: dict[str, int] = {"indexed": 0, "skipped": 0, "unchanged": 0, "error": 0}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Indexing", total=len(candidates))

        def on_result(r: IndexResult) -> None:
            counts[r.status] = counts.get(r.status, 0) + 1
            progress.update(task, advance=1, description=f"Indexing {r.path.name}")
            if verbose and r.status in {"skipped", "error"}:
                console.print(f"  [yellow]{r.status}[/] {r.path}: {r.reason}")

        index_folder(
            conn, embedder, cfg, folder, progress=on_result,
            transcriber=transcriber, ocr_engine=ocr_engine,
            entity_extractor=entity_extractor,
            image_embedder=image_embedder,
        )

    console.print(
        f"[green]Done.[/] indexed={counts.get('indexed', 0)} "
        f"unchanged={counts.get('unchanged', 0)} "
        f"skipped={counts.get('skipped', 0)} "
        f"errors={counts.get('error', 0)}"
    )
    conn.close()


@app.command()
def search(
    query: str = typer.Argument(..., help="Query string."),
    k: int = typer.Option(10, "--k", "-k", help="Number of results."),
    alpha: float = typer.Option(
        None, "--alpha", help="0=keyword only, 1=vector only. Default uses config."
    ),
    no_rerank: bool = typer.Option(
        False, "--no-rerank", help="Skip cross-encoder reranking."
    ),
    folder: str = typer.Option(None, "--folder", help="Restrict to files under a path prefix."),
    kind: str = typer.Option(None, "--kind", help="Restrict to a kind: document/code/audio_video/image/url."),
    since_days: int = typer.Option(None, "--since-days", help="Restrict to files modified in the last N days."),
) -> None:
    """Search the index from the command line."""
    cfg = load_config()
    conn, embedder = _open_state(cfg)
    reranker = None if no_rerank else make_reranker(cfg)
    a = alpha if alpha is not None else cfg.hybrid_alpha
    # If the user pinned alpha explicitly via --alpha, skip adaptive override.
    use_adaptive = cfg.adaptive_alpha and alpha is None
    results = hybrid_search(
        conn, embedder, query, k=k, alpha=a,
        reranker=reranker, rerank_overfetch=cfg.rerank_overfetch,
        use_adaptive_alpha=use_adaptive,
        time_decay_weight=cfg.time_decay_weight if cfg.time_decay_enabled else 0.0,
        time_decay_half_life_days=cfg.time_decay_half_life_days,
        path_prefix=folder,
        kind=kind,
        since_days=since_days,
        use_hyde=cfg.hyde_enabled,
        hyde_model=cfg.hyde_model,
        personal_prefixes=cfg.personal_path_prefixes,
        personal_boost=cfg.personal_path_boost,
        download_prefixes=cfg.download_path_prefixes,
        download_demote=cfg.download_path_demote,
        click_boost_max=cfg.click_boost_max if cfg.click_boost_enabled else 1.0,
        click_boost_half_life_days=cfg.click_boost_half_life_days,
        cfg=cfg,
    )
    if not results:
        console.print("[yellow]No matches.[/]")
        return
    for i, r in enumerate(results, 1):
        tag = "reranked" if r.reranked else "+".join(r.sources)
        console.rule(f"[bold]{i}.[/] {r.file_path}  [dim](chunk {r.chunk_index} | {tag} | {r.score:.4f})")
        console.print(r.text if len(r.text) <= 1200 else r.text[:1200] + "...")
    conn.close()


@app.command()
def watch(
    folder: Path = typer.Argument(..., help="Folder to watch (recursively)."),
    bootstrap: bool = typer.Option(
        True, "--bootstrap/--no-bootstrap", help="Index the folder once before watching."
    ),
) -> None:
    """Bootstrap-index a folder and then watch for changes until interrupted."""
    from .watcher import Watcher  # heavy import; lazy

    folder = folder.expanduser().resolve()
    if not folder.exists():
        console.print(f"[red]Folder does not exist:[/] {folder}")
        raise typer.Exit(code=1)

    cfg = load_config()
    conn, embedder = _open_state(cfg)

    transcriber = None
    if cfg.transcribe_enabled:
        try:
            transcriber = make_transcriber(cfg)
        except ImportError as e:
            console.print(f"[yellow]Transcription disabled:[/] {e}")

    ocr_engine = None
    if cfg.ocr_enabled:
        try:
            ocr_engine = make_ocr_engine(cfg)
        except ImportError as e:
            console.print(f"[yellow]OCR disabled:[/] {e}")

    entity_extractor = None
    if cfg.entities_enabled:
        try:
            entity_extractor = make_entity_extractor(cfg)
        except (ImportError, RuntimeError) as e:
            console.print(f"[yellow]Entity extraction disabled:[/] {e}")

    image_embedder = None
    if cfg.image_embed_enabled:
        try:
            image_embedder = make_image_embedder(cfg)
        except Exception as e:
            console.print(f"[yellow]Multimodal image embedder disabled:[/] {e}")

    if bootstrap:
        console.print(f"Bootstrapping index for [cyan]{folder}[/]...")
        index_folder(
            conn, embedder, cfg, folder,
            transcriber=transcriber, ocr_engine=ocr_engine,
            entity_extractor=entity_extractor,
            image_embedder=image_embedder,
        )
        console.print("[green]Bootstrap complete.[/]")

    def on_event(r: IndexResult) -> None:
        if r.status in {"indexed", "deleted"}:
            console.print(f"[dim]{r.status}[/] {r.path}")

    watcher = Watcher(
        cfg, conn, embedder, on_event=on_event,
        transcriber=transcriber, ocr_engine=ocr_engine,
        entity_extractor=entity_extractor,
        image_embedder=image_embedder,
    )
    watcher.start([folder])
    console.print(f"[green]Watching[/] {folder}. Press Ctrl-C to stop.")
    try:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping...[/]")
        watcher.stop()
        conn.close()


@app.command()
def auth(
    provider: str = typer.Argument(
        "google",
        help="Auth provider to set up: 'google' (Gmail / Calendar / Drive) "
             "or 'extension' (browser-extension bearer token).",
    ),
) -> None:
    """One-time OAuth flow for a cloud provider, or print the extension token.

    For 'google': make sure you've created an OAuth Desktop client in
    https://console.cloud.google.com and saved the JSON as
    ~/.secondbrain/google_client_secret.json. Then this command opens a
    browser, captures the redirect, and stores credentials. Subsequent
    Gmail / Google Calendar syncs auto-refresh.

    For 'extension': prints (and creates if missing) the per-install bearer
    token for the browser extension. Paste it into the extension popup; the
    token is required for /api/extension/* calls.
    """
    cfg = load_config()

    if provider == "extension":
        from .dashboard import get_or_create_extension_token

        token = get_or_create_extension_token(cfg)
        console.print(
            "[green]Browser-extension token[/] "
            "(paste this into the extension popup):"
        )
        console.print(f"  [cyan]{token}[/]")
        console.print()
        console.print(
            "Anyone with this token + access to 127.0.0.1:8765 can read your "
            "index. Don't share it; rotate by deleting "
            f"{cfg.data_dir / 'extension_token.txt'}."
        )
        return

    if provider != "google":
        console.print(f"[red]Unknown auth provider:[/] {provider}")
        console.print("Try: 'google' or 'extension'.")
        raise typer.Exit(code=1)

    from .connectors._google_oauth import GoogleAuthError, run_oauth_flow
    from .connectors.gmail import GMAIL_SCOPES
    from .connectors.google_calendar import GOOGLE_CALENDAR_SCOPES
    from .connectors.google_drive import GOOGLE_DRIVE_SCOPES

    scopes = list({*GMAIL_SCOPES, *GOOGLE_CALENDAR_SCOPES, *GOOGLE_DRIVE_SCOPES})
    try:
        creds = run_oauth_flow(cfg, scopes, open_browser=True)
    except GoogleAuthError as e:
        console.print(f"[red]Auth failed:[/] {e}")
        raise typer.Exit(code=1) from e
    console.print(
        f"[green]Authorized.[/] Stored credentials at "
        f"{cfg.data_dir / 'google_credentials.json'}"
    )
    console.print(f"  scopes: {', '.join(creds.scopes)}")
    console.print()
    console.print("Now run:")
    console.print('  [cyan]secondbrain sync gmail[/]')
    console.print('  [cyan]secondbrain sync google_calendar[/]')


@app.command()
def sync(
    source: str = typer.Argument(
        "all",
        help="Connector name (github / notion / browser / calendar) or 'all'.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Pull documents from cloud connectors into the index.

    Each connector reads its credentials from environment variables:
        GITHUB_TOKEN, NOTION_TOKEN, CALENDAR_ICS_URL.
    The browser connector needs no auth; it reads Chrome/Edge SQLite history.

    Run periodically to keep the index fresh — connectors are idempotent
    via hash-based dedup, so re-runs are cheap.
    """
    _setup_logging(verbose)
    from .connectors import all_connectors, get_connector

    cfg = load_config()
    conn, embedder = _open_state(cfg)
    entity_extractor = None
    if cfg.entities_enabled:
        try:
            entity_extractor = make_entity_extractor(cfg)
        except (ImportError, RuntimeError) as e:
            console.print(f"[yellow]Entity extraction disabled:[/] {e}")

    if source == "all":
        connector_classes = all_connectors()
    else:
        cls = get_connector(source)
        if cls is None:
            console.print(f"[red]Unknown connector:[/] {source}")
            raise typer.Exit(code=1)
        connector_classes = [cls]

    grand_totals: dict[str, int] = {}
    for cls in connector_classes:
        c = cls()
        if not c.is_enabled(cfg):
            console.print(f"[dim]skip[/]  {c.name:10s}  (not configured — see `secondbrain --help`)")
            continue
        console.print(f"[cyan]sync[/]  {c.name:10s}  fetching...")
        counts = {"indexed": 0, "skipped": 0, "unchanged": 0, "alias": 0, "error": 0}
        try:
            for doc in c.fetch(cfg):
                result = index_text(
                    conn, embedder, cfg,
                    virtual_path=doc.virtual_path,
                    title=doc.title,
                    content=doc.content,
                    mtime=doc.mtime,
                    kind=doc.kind,
                    source=doc.source,
                    entity_extractor=entity_extractor,
                )
                counts[result.status] = counts.get(result.status, 0) + 1
                total = sum(counts.values())
                if verbose or total % 25 == 0:
                    console.print(
                        f"  [{c.name}] {total:4d}: "
                        f"indexed={counts.get('indexed',0)} "
                        f"unchanged={counts.get('unchanged',0)} "
                        f"alias={counts.get('alias',0)} "
                        f"err={counts.get('error',0)}"
                    )
        except Exception as e:
            console.print(f"[red]error in {c.name}:[/] {e}")
        for k, v in counts.items():
            grand_totals[k] = grand_totals.get(k, 0) + v
        console.print(
            f"[green]done[/]  {c.name:10s}  "
            f"indexed={counts.get('indexed',0)} "
            f"unchanged={counts.get('unchanged',0)} "
            f"alias={counts.get('alias',0)} "
            f"errors={counts.get('error',0)}"
        )

    console.print()
    console.print(
        f"[bold]Total:[/] indexed={grand_totals.get('indexed', 0)} "
        f"unchanged={grand_totals.get('unchanged', 0)} "
        f"alias={grand_totals.get('alias', 0)} "
        f"errors={grand_totals.get('error', 0)}"
    )
    conn.close()


@app.command()
def ingest(
    url: str = typer.Argument(..., help="URL to fetch and index (article, PDF, YouTube, ...)."),
) -> None:
    """Fetch a URL and index its content into the brain.

    Markitdown handles HTML article extraction, PDF download+parse, and
    YouTube transcript fetch. The URL is stored as a virtual file with
    kind='url' so it shows up in `list_folders` / `search_brain` / etc.
    """
    cfg = load_config()
    conn, embedder = _open_state(cfg)

    entity_extractor = None
    if cfg.entities_enabled:
        try:
            entity_extractor = make_entity_extractor(cfg)
        except (ImportError, RuntimeError) as e:
            console.print(f"[yellow]Entity extraction disabled:[/] {e}")

    console.print(f"Fetching [cyan]{url}[/] ...")
    result = index_url(conn, embedder, cfg, url, entity_extractor=entity_extractor)
    if result.status == "indexed":
        console.print(f"[green]Indexed:[/] {url} ({result.chunks} chunks)")
    elif result.status == "unchanged":
        console.print(f"[dim]Unchanged:[/] {url}")
    else:
        console.print(f"[yellow]{result.status}:[/] {url} ({result.reason})")
    conn.close()


@app.command()
def spend() -> None:
    """Show today's API spend per provider with current daily caps."""
    from .budget import spend_summary

    cfg = load_config()
    summary = spend_summary(cfg)
    table = Table(show_header=True, box=None)
    table.add_column("Provider")
    table.add_column("Calls", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Spent ($)", justify="right")
    table.add_column("Cap ($)", justify="right")
    for provider, bucket in summary.items():
        cap = (
            cfg.daily_budget_cents_voyage if provider == "voyage"
            else cfg.daily_budget_cents_anthropic if provider == "anthropic"
            else 0
        )
        cap_str = "disabled" if cap == 0 else f"{cap / 100:.2f}"
        table.add_row(
            provider,
            str(bucket["calls"]),
            f"{bucket['tokens']:,}",
            f"{bucket['cents'] / 100:.4f}",
            cap_str,
        )
    console.print(table)
    console.print(f"\n[dim]Ledger:[/] {cfg.data_dir / 'spend.jsonl'}")


@app.command()
def tag(
    since_days: int = typer.Option(
        None, "--since-days",
        help="Only tag chunks from files indexed in the last N days. Default: tag all untagged chunks.",
    ),
    limit: int = typer.Option(
        None, "--limit",
        help="Hard cap on chunks to tag this run. Useful for testing cost.",
    ),
    model: str = typer.Option(
        None, "--model",
        help="Override the model. Default: cfg.tag_model (claude-haiku-4-5).",
    ),
) -> None:
    """Use Claude to assign 1-3 topic tags per chunk. Idempotent — chunks
    that already have tags are skipped. Run periodically after big ingests.

    Cost: ~$0.0003 per chunk on Haiku 4.5. A 6,000-chunk index ≈ $2."""
    from .tagger import generate_tags

    cfg = load_config()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print("[red]ANTHROPIC_API_KEY not set.[/]")
        raise typer.Exit(code=1)

    conn, _ = _open_state(cfg)

    # Find untagged chunks. Optionally scope by file.indexed_at.
    query = (
        "SELECT c.id, c.text FROM chunks c "
        "JOIN files f ON f.id = c.file_id "
        "LEFT JOIN chunk_tags t ON t.chunk_id = c.id "
        "WHERE t.id IS NULL "
    )
    params: list = []
    if since_days is not None:
        query += "AND f.indexed_at >= ? "
        params.append(time.time() - since_days * 86400)
    query += "GROUP BY c.id ORDER BY c.id "
    if limit is not None:
        query += "LIMIT ? "
        params.append(limit)

    rows = conn.execute(query, params).fetchall()
    total = len(rows)
    if total == 0:
        console.print("[yellow]Nothing to tag.[/] All matching chunks already have tags.")
        return

    console.print(f"Tagging [cyan]{total}[/] chunks via [cyan]{model or cfg.tag_model}[/]...")
    tagged = 0
    failed = 0
    for i, r in enumerate(rows, 1):
        tags = generate_tags(r["text"], cfg, model=model)
        if not tags:
            failed += 1
        else:
            for t in tags:
                conn.execute(
                    "INSERT OR IGNORE INTO chunk_tags(chunk_id, tag) VALUES (?, ?)",
                    (r["id"], t),
                )
            tagged += 1
        if i % 25 == 0:
            conn.commit()
            console.print(f"  [{i}/{total}] tagged={tagged} failed={failed}")
    conn.commit()
    conn.close()
    console.print(f"[green]Done.[/] tagged={tagged} failed={failed}")


@app.command()
def briefing(
    hours: int = typer.Option(24, "--hours", "-h", help="Look-back window in hours."),
) -> None:
    """Generate a Claude-written summary of what's entered your brain recently.

    Requires ANTHROPIC_API_KEY. Uses Claude Opus 4.7 by default; configurable
    via `briefing_model` in config.toml.
    """
    from .briefing import generate_briefing

    cfg = load_config()
    conn, _ = _open_state(cfg)
    text = generate_briefing(conn, cfg, hours=hours)
    console.print(text)
    conn.close()


@app.command()
def serve() -> None:
    """Start the MCP server over stdio (for Claude Desktop / Claude Code / Cursor)."""
    from .mcp_server import run

    run()


@app.command()
def chat(
    question: str = typer.Argument(
        None,
        help="One-shot question. Omit for an interactive REPL.",
    ),
    no_rerank: bool = typer.Option(False, "--no-rerank"),
) -> None:
    """Ask your brain a question. Claude with `search_brain` as a tool;
    answers cite their sources.

    With no argument, drops into an interactive REPL. Conversation history
    is kept in memory for the session; type `/reset` to start over.
    Requires ANTHROPIC_API_KEY.
    """
    from .chat import stream_chat

    cfg = load_config()
    conn, embedder = _open_state(cfg)
    reranker = None if no_rerank else make_reranker(cfg)
    history: list[dict] = []

    def run_turn(q: str) -> None:
        # Stream tokens to stdout; print citations after.
        last_citations: list[dict] = []
        nonlocal_history: list[dict] | None = None
        # We can't reassign `history` inside this closure easily without
        # nonlocal — Typer's argument quirks make it cleaner to mutate.
        printed_anything = False
        for ev in stream_chat(cfg, conn, embedder, reranker, q, history):
            if ev.kind == "text":
                console.print(ev.data, end="", markup=False, highlight=False)
                printed_anything = True
            elif ev.kind == "search":
                console.print(
                    f"\n[dim]⌕ searching: {ev.data['query']} (k={ev.data['k']})[/]",
                    highlight=False,
                )
            elif ev.kind == "results":
                console.print(
                    f"[dim]→ {len(ev.data)} chunk(s)[/]",
                    highlight=False,
                )
            elif ev.kind == "done":
                last_citations = ev.data.get("citations") or []
                # Replace text if model emitted nothing during streaming
                # (force-answer iteration after tool use).
                if not printed_anything and ev.data.get("text"):
                    console.print(ev.data["text"], markup=False, highlight=False)
                nonlocal_history = ev.data.get("history")
            elif ev.kind == "error":
                console.print(f"\n[red]error:[/] {ev.data}")
        console.print()  # newline after the answer
        if last_citations:
            console.print("[dim]Sources:[/]")
            for i, c in enumerate(last_citations, 1):
                console.print(
                    f"  [dim]({i})[/] {c['file_path']} "
                    f"[dim]· chunk {c['chunk_index']} · {c['score']:.3f}[/]"
                )
        if nonlocal_history is not None:
            history.clear()
            history.extend(nonlocal_history)

    if question:
        run_turn(question)
        conn.close()
        return

    # REPL
    console.print(
        "[green]chat with your brain[/] · "
        f"model={cfg.chat_model} · type [cyan]/reset[/] to clear history, "
        "[cyan]/quit[/] or Ctrl-D to exit"
    )
    try:
        while True:
            try:
                line = console.input("[bold]you ›[/] ")
            except (EOFError, KeyboardInterrupt):
                console.print()
                break
            line = line.strip()
            if not line:
                continue
            if line in ("/quit", "/exit"):
                break
            if line == "/reset":
                history.clear()
                console.print("[dim]history cleared[/]")
                continue
            console.print("[bold]brain ›[/] ", end="")
            run_turn(line)
    finally:
        conn.close()


@app.command()
def dashboard(
    port: int = typer.Option(8765, "--port", "-p", help="HTTP port."),
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address (keep on localhost)."),
    no_open: bool = typer.Option(False, "--no-open", help="Don't open the browser."),
) -> None:
    """Launch the local web dashboard at http://localhost:8765 by default.

    Browse stats, recent files, top entities, search with filters, ingest
    URLs, and drill into any entity's mentions and co-occurring entities.
    Requires the [dashboard] extra: pip install -e .[dashboard]
    """
    from .dashboard import run_dashboard

    run_dashboard(host=host, port=port, open_browser=not no_open)


@app.command()
def daemon() -> None:
    """Headless watcher. Bootstrap-indexes all watched_folders, then watches forever.

    Reads `watched_folders` from config (~/.secondbrain/config.toml). Logs to
    `~/.secondbrain/daemon.log`. Stop with Ctrl-C. For autostart, schedule via
    Windows Task Scheduler / launchd / systemd (see README).
    """
    from .daemon import run_daemon

    cfg = load_config()
    run_daemon(cfg)


@app.command()
def tray() -> None:
    """Run as a system-tray app: bootstrap-indexes, then watches with a tray icon.

    Right-click the icon for status / open data dir / quit. Requires the [tray]
    extra: pip install -e .[tray]
    """
    from .daemon import run_tray

    cfg = load_config()
    run_tray(cfg)


@app.command()
def reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Delete the index database. Config is preserved."""
    cfg = load_config()
    if not cfg.db_path.exists():
        console.print(f"[yellow]No index at[/] {cfg.db_path}")
        return
    if not yes:
        confirm = typer.confirm(f"Delete index at {cfg.db_path}?")
        if not confirm:
            console.print("[yellow]Aborted.[/]")
            return
    cfg.db_path.unlink()
    for sidecar in (".db-wal", ".db-shm", ".db-journal"):
        p = cfg.db_path.with_suffix(sidecar)
        if p.exists():
            p.unlink()
    console.print(f"[green]Deleted[/] {cfg.db_path}")


if __name__ == "__main__":
    app()
