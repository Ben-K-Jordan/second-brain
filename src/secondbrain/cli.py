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


def _auth_canvas(cfg: Config) -> None:
    """Walk the user through generating + saving a Canvas personal access
    token. Designed for institutions with SSO + Duo (Cornell, etc.) — the
    one-time SSO+Duo happens in their browser when they visit the token
    page; the token itself is plain Bearer auth from then on.
    """
    import webbrowser

    from .connectors.canvas import (
        _credentials_path,
        save_canvas_credentials,
        verify_canvas_token,
    )

    console.print("[bold green]Canvas LMS setup[/]")
    console.print(
        "We'll generate a personal access token. The Canvas web UI requires "
        "your school's SSO + Duo, but the token itself doesn't — once "
        "generated, the connector authenticates with just the token. No "
        "SSO checks per sync.\n"
    )

    # Smart default: if the user already has CANVAS_BASE_URL set, use it.
    default_base = (os.environ.get("CANVAS_BASE_URL") or "").strip()
    prompt = (
        f"Canvas root URL [{default_base}]: " if default_base
        else "Canvas root URL (e.g. https://canvas.cornell.edu): "
    )
    base = console.input(prompt).strip() or default_base
    if not base:
        console.print("[red]Need a Canvas URL.[/]")
        raise typer.Exit(code=1)
    if not base.startswith(("http://", "https://")):
        base = "https://" + base
    base = base.rstrip("/")

    settings_url = f"{base}/profile/settings"
    console.print()
    console.print("[bold]1.[/] Sign into Canvas (this is when SSO + Duo happen).")
    console.print(f"[bold]2.[/] Visit [cyan]{settings_url}[/]")
    console.print("[bold]3.[/] Scroll to [bold]Approved Integrations[/] → "
                  "click [bold]+ New Access Token[/].")
    console.print("[bold]4.[/] Set Purpose = 'second-brain'. "
                  "Leave [bold]Expires[/] blank for a long-lived token.")
    console.print("[bold]5.[/] Click [bold]Generate Token[/], then copy the "
                  "long string Canvas shows you (it's only shown once).")
    console.print()

    if console.input("Open the settings page now? [Y/n]: ").strip().lower() not in ("n", "no"):
        try:
            webbrowser.open(settings_url)
        except Exception:  # noqa: BLE001
            pass

    token = console.input("\nPaste the access token here: ").strip()
    if not token:
        console.print("[red]No token provided.[/]")
        raise typer.Exit(code=1)

    console.print("\n[dim]Verifying...[/]")
    me = verify_canvas_token(base, token)
    if me is None:
        console.print(
            "[red]Verification failed.[/]  Check:\n"
            "  - The base URL is your Canvas root (e.g. https://canvas.cornell.edu)\n"
            "  - You pasted the full token (it's a long string, no spaces)\n"
            "  - Your school hasn't disabled personal access tokens "
            "(rare, but the iCal fallback is documented in the connector)"
        )
        raise typer.Exit(code=1)

    save_canvas_credentials(cfg, base, token)
    name = me.get("name") or me.get("short_name") or "?"
    login = me.get("login_id") or me.get("primary_email") or ""
    console.print(f"[green]✓[/] Authorized as [bold]{name}[/]"
                  + (f" ({login})" if login else ""))
    console.print(f"  saved to [dim]{_credentials_path(cfg)}[/]")
    console.print()
    console.print("Now try:")
    console.print("  [cyan]secondbrain sync canvas[/]")


@app.command()
def auth(
    provider: str = typer.Argument(
        "google",
        help="Auth provider to set up: 'google' (Gmail / Calendar / Drive), "
             "'canvas' (LMS personal access token), or 'extension' "
             "(browser-extension bearer token).",
    ),
) -> None:
    """Guided setup for a cloud provider.

    For 'google': make sure you've created an OAuth Desktop client in
    https://console.cloud.google.com and saved the JSON as
    ~/.secondbrain/google_client_secret.json. Then this command opens a
    browser, captures the redirect, and stores credentials. Subsequent
    Gmail / Google Calendar syncs auto-refresh.

    For 'canvas': opens the Canvas token-generation page in your browser
    (you'll do SSO + Duo *once*), prompts for the token, verifies it, and
    saves it. From then on, syncs use the API directly — no SSO each time.

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

    if provider == "canvas":
        _auth_canvas(cfg)
        return

    if provider != "google":
        console.print(f"[red]Unknown auth provider:[/] {provider}")
        console.print("Try: 'google', 'canvas', or 'extension'.")
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
def capture(
    silence_seconds: float = typer.Option(
        None, "--silence", "-s",
        help="Stop after this many seconds of silence (default from config).",
    ),
    save_audio: bool = typer.Option(
        None, "--save-audio/--no-save-audio",
        help="Override cfg.voice_save_audio.",
    ),
    label: str = typer.Option(
        "voice note", "--label", "-l",
        help="Document title prefix.",
    ),
) -> None:
    """Voice-capture a note into the brain. Speak after the prompt, stop
    when you're done — VAD detects silence and ends recording.

    Requires the [voice] extra: ``pip install -e .[voice]``.

    The transcript is ingested as a regular brain document (kind=document,
    source=voice), so it's searchable via the same hybrid retrieval as
    everything else. The chat agent can find your voice notes via
    search_brain.
    """
    from .voice import VoiceCaptureUnavailable
    from .voice import capture as voice_capture

    cfg = load_config()
    if save_audio is not None:
        cfg.voice_save_audio = save_audio
    conn, embedder = _open_state(cfg)

    # Live status renderer — show a simple progress bar of recent volume
    # so the user knows the mic is hearing them.
    live_state = {"max_rms": 0.0, "started": False}

    def status(kind: str, payload):
        if kind == "start":
            live_state["started"] = True
            console.print("[bold green]●[/] recording — speak now…  "
                          "[dim](stops after silence)[/]")
        elif kind == "volume":
            live_state["max_rms"] = max(live_state["max_rms"], payload)
        elif kind == "stop":
            reason = "silence detected" if payload == "silence" else (
                "max-duration reached" if payload == "max_duration"
                else str(payload)
            )
            console.print(f"[bold yellow]■[/] stopped ({reason})")

    try:
        result = voice_capture(
            cfg, conn, embedder,
            silence_seconds=silence_seconds, on_status=status,
            note_label=label,
        )
    except VoiceCaptureUnavailable as e:
        console.print(f"[red]capture failed:[/] {e}")
        conn.close()
        raise typer.Exit(code=1) from None
    except KeyboardInterrupt:
        console.print("\n[yellow]capture cancelled[/]")
        conn.close()
        raise typer.Exit(code=130) from None

    if not result.transcript:
        console.print(
            f"[yellow]no transcript[/] (recorded {result.duration_seconds:.1f}s; "
            "either silence or the mic captured nothing)"
        )
        conn.close()
        return

    console.print()
    console.print(f"[dim]{result.duration_seconds:.1f}s · "
                  f"{len(result.transcript)} chars · "
                  f"{result.chunks_indexed} chunk(s) indexed[/]")
    console.print(f"[dim]vp:[/] [cyan]{result.virtual_path}[/]")
    if result.audio_path:
        console.print(f"[dim]audio:[/] [cyan]{result.audio_path}[/]")
    console.print()
    console.print(result.transcript)
    conn.close()


@app.command("transcripts")
def transcripts_list(
    course: str | None = typer.Option(
        None, "--course", "-c",
        help="Filter to a specific Canvas course code (e.g. 'BME 410').",
    ),
    limit: int = typer.Option(50, "--limit", "-n"),
) -> None:
    """List ingested lecture / meeting transcripts.

    Plaud, Otter, and generic transcript-shaped emails ingested via the
    IMAP connector show up here with their inferred course code and
    recording timestamp.
    """
    from .db import connect_readonly

    cfg = load_config()
    try:
        conn = connect_readonly(cfg.db_path)
    except FileNotFoundError as e:
        console.print(f"[yellow]{e}[/]")
        raise typer.Exit(code=1) from None
    rows = conn.execute(
        "SELECT path, mtime, indexed_at FROM files "
        "WHERE path LIKE 'transcript://%' "
        "ORDER BY mtime DESC LIMIT ?",
        (limit * 3,),
    ).fetchall()
    if not rows:
        console.print(
            "[yellow]No transcripts ingested yet.[/]\n"
            "  - Set up Plaud's auto-export to email\n"
            "  - Configure a Gmail filter that labels them (e.g. 'Plaud')\n"
            "  - Add that label to imap_folders in config.toml\n"
            "  - Run [cyan]secondbrain sync imap[/]"
        )
        conn.close()
        return
    table = Table(show_header=True, box=None, title="Transcripts")
    table.add_column("when", style="dim", width=18)
    table.add_column("course", style="cyan", width=10)
    table.add_column("provider", style="dim", width=10)
    table.add_column("title")
    matched = 0
    for r in rows:
        path = r["path"]
        # Path shape: transcript://<provider>/<...>
        try:
            provider = path.split("//", 1)[1].split("/", 1)[0]
        except IndexError:
            provider = "?"
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["mtime"]))
        # Pull title + course out of the rendered first chunk.
        chunk = conn.execute(
            "SELECT text FROM chunks WHERE file_id = ("
            "  SELECT id FROM files WHERE path = ?"
            ") ORDER BY chunk_index LIMIT 1",
            (path,),
        ).fetchone()
        title = "(unknown)"
        course_code = ""
        if chunk:
            head = (chunk["text"].split("\n", 1)[0] or "").lstrip("# ").strip()
            if head.startswith("["):
                end = head.find("]")
                if end > 0:
                    course_code = head[1:end]
                    title = head[end + 1:].strip()
                else:
                    title = head
            else:
                title = head
        if course and course_code != course:
            continue
        matched += 1
        if matched > limit:
            break
        table.add_row(when, course_code, provider, title[:60])
    if matched == 0:
        console.print("[yellow]No transcripts match that filter.[/]")
    else:
        console.print(table)
    conn.close()


read_app = typer.Typer(
    no_args_is_help=True,
    help="Reading queue. Watchlist runs auto-enqueue high-fit jobs and "
         "every news/research item; the daemon writes a 60-second pre-read "
         "summary so you can scan instead of opening every link.",
)
app.add_typer(read_app, name="read")


@read_app.command("list")
def read_list(
    history: bool = typer.Option(
        False, "--history", "-H",
        help="Show read + skipped items instead of unread.",
    ),
    limit: int = typer.Option(20, "--limit", "-n"),
) -> None:
    """List items in your reading queue (or history)."""
    from .db import reading_queue_history, reading_queue_unread

    cfg = load_config()
    conn, _ = _open_state(cfg)
    rows = (
        reading_queue_history(conn, limit=limit) if history
        else reading_queue_unread(conn, limit=limit)
    )
    if not rows:
        msg = "No history yet." if history else "Nothing in your queue."
        console.print(f"[yellow]{msg}[/]")
        conn.close()
        return
    label = "History" if history else "Unread reading queue"
    table = Table(show_header=True, box=None, title=label)
    table.add_column("id", style="dim", width=4)
    table.add_column("title")
    table.add_column("source", style="dim", width=14)
    table.add_column("fit", justify="center", width=10)
    table.add_column("status" if history else "summary", justify="center")
    for r in rows:
        title = (r["title"] or r["url"])[:60]
        fit = r["fit_label"] or ""
        if history:
            if r["read_at"]:
                status = "[green]read[/]"
            elif r["skipped_at"]:
                status = "[dim]skipped[/]"
            else:
                status = "[dim]?[/]"
        elif r["summary_error"]:
            status = "[red]err[/]"
        elif r["summary"]:
            status = "[green]✓[/]"
        else:
            status = "[dim]…[/]"
        table.add_row(str(r["id"]), title, r["source"], fit, status)
    console.print(table)
    if not history:
        console.print(
            "\n[dim]use[/] [cyan]secondbrain read show <id>[/] [dim]to read a summary[/]"
        )
    conn.close()


@read_app.command("show")
def read_show(
    queue_id: int = typer.Argument(..., help="Queue id from `read list`."),
) -> None:
    """Print the summary for a queued item."""
    from .db import reading_queue_get

    cfg = load_config()
    conn, _ = _open_state(cfg)
    row = reading_queue_get(conn, queue_id)
    if row is None:
        console.print(f"[red]No queue item #{queue_id}.[/]")
        conn.close()
        raise typer.Exit(code=1)
    console.print(f"[bold]{row['title'] or row['url']}[/]")
    console.print(f"[dim]{row['url']}[/]")
    console.print(f"[dim]source: {row['source']} · fit: {row['fit_label'] or 'n/a'}[/]")
    console.print()
    if row["summary_error"]:
        console.print(f"[red]summary errored:[/] {row['summary_error']}")
    elif row["summary"]:
        console.print(row["summary"])
    else:
        console.print("[dim](summary still pending — daemon will pick it up)[/]")
    conn.close()


@read_app.command("add")
def read_add(
    url: str = typer.Argument(..., help="URL to summarise."),
    title: str = typer.Option("", "--title", "-t",
                              help="Optional human title."),
) -> None:
    """Manually enqueue a URL to read later (with auto-summary)."""
    from .db import reading_queue_enqueue

    cfg = load_config()
    conn, _ = _open_state(cfg)
    rid = reading_queue_enqueue(
        conn, url=url, title=title, source="manual",
    )
    if rid is None:
        console.print(f"[yellow]Already in queue:[/] {url}")
    else:
        console.print(f"[green]Queued #{rid}:[/] {url}")
    conn.close()


@read_app.command("mark")
def read_mark(
    queue_id: int = typer.Argument(...),
    state: str = typer.Argument(
        "read",
        help="One of 'read' or 'skipped'.",
    ),
) -> None:
    """Mark a queue item as read or skipped (removes from unread view)."""
    from .db import (
        reading_queue_get,
        reading_queue_mark_read,
        reading_queue_mark_skipped,
    )

    cfg = load_config()
    conn, _ = _open_state(cfg)
    if reading_queue_get(conn, queue_id) is None:
        console.print(f"[red]No queue item #{queue_id}.[/]")
        conn.close()
        raise typer.Exit(code=1)
    if state == "read":
        reading_queue_mark_read(conn, queue_id)
    elif state == "skipped":
        reading_queue_mark_skipped(conn, queue_id)
    else:
        console.print(f"[red]Unknown state[/] {state!r}; use 'read' or 'skipped'.")
        conn.close()
        raise typer.Exit(code=1)
    console.print(f"[green]✓[/] #{queue_id} marked {state}")
    conn.close()


@read_app.command("summarise")
def read_summarise(
    limit: int = typer.Option(5, "--limit", "-n"),
) -> None:
    """Run the summariser now for up to N pending items."""
    from .reading_queue import summarise_pending

    cfg = load_config()
    conn, embedder = _open_state(cfg)
    reranker = make_reranker(cfg)
    n = summarise_pending(cfg, conn, embedder, reranker)
    console.print(f"[green]Generated {n} summar{'y' if n == 1 else 'ies'}.[/]")
    conn.close()


brief_app = typer.Typer(
    no_args_is_help=True,
    help="Pre-event briefings. The daemon auto-generates these for events "
         "starting soon; these subcommands let you inspect / regenerate / "
         "trigger ad-hoc briefings.",
)
app.add_typer(brief_app, name="brief")


@brief_app.command("upcoming")
def brief_upcoming(
    minutes: int = typer.Option(
        120, "--minutes", "-m",
        help="Look-ahead window in minutes.",
    ),
) -> None:
    """List upcoming events from your calendars and whether they have
    a briefing yet."""
    from .db import event_briefing_get
    from .event_briefing import iter_upcoming_events

    cfg = load_config()
    conn, _ = _open_state(cfg)
    events = sorted(
        iter_upcoming_events(cfg, minutes * 60),
        key=lambda e: e.starts_at,
    )
    if not events:
        console.print(f"[yellow]No events in the next {minutes} minutes.[/]")
        conn.close()
        return
    table = Table(show_header=True, box=None,
                  title=f"Upcoming events (next {minutes}m)")
    table.add_column("when", style="dim")
    table.add_column("title")
    table.add_column("source", style="dim", width=10)
    table.add_column("brief", justify="center")
    for ev in events:
        when = time.strftime("%a %H:%M", time.localtime(ev.starts_at))
        existing = event_briefing_get(conn, ev.event_id, ev.source)
        if existing is None:
            mark = "[dim]—[/]"
        elif existing["error"]:
            mark = "[red]✗[/]"
        else:
            mark = "[green]✓[/]"
        table.add_row(when, ev.title[:60], ev.source, mark)
    console.print(table)
    conn.close()


@brief_app.command("next")
def brief_next() -> None:
    """Show the briefing for the next upcoming event (generating one
    if needed)."""
    from .db import event_briefing_get
    from .event_briefing import generate_for_event, iter_upcoming_events

    cfg = load_config()
    conn, embedder = _open_state(cfg)
    reranker = make_reranker(cfg)
    # Look 6 hours out so "next" works even on a quiet day.
    events = sorted(
        iter_upcoming_events(cfg, 6 * 3600),
        key=lambda e: e.starts_at,
    )
    if not events:
        console.print("[yellow]No upcoming events in the next 6 hours.[/]")
        conn.close()
        return
    ev = events[0]
    existing = event_briefing_get(conn, ev.event_id, ev.source)
    if existing is None or existing["error"]:
        console.print(
            f"[cyan]Generating briefing for[/] [bold]{ev.title}[/]..."
        )
        result = generate_for_event(cfg, conn, embedder, reranker, ev)
        if not result.get("ok"):
            console.print(f"[red]failed:[/] {result.get('error', '?')}")
            conn.close()
            raise typer.Exit(code=1)
        text = result["text"]
    else:
        text = existing["briefing_text"] or "(no text)"
    console.print()
    console.print(f"[bold]{ev.title}[/]  [dim]({time.strftime('%a %H:%M', time.localtime(ev.starts_at))})[/]")
    console.print()
    console.print(text)
    conn.close()


@brief_app.command("now")
def brief_now(
    title: str = typer.Argument(..., help="Title of the ad-hoc event."),
    starts_at: str = typer.Option(
        ..., "--at",
        help="ISO-8601 start time, e.g. '2026-04-15T14:00:00' or '2026-04-15T14:00:00-05:00'.",
    ),
    attendees: list[str] = typer.Option(
        None, "--attendee", "-a",
        help="Repeatable: emails or names of attendees.",
    ),
    description: str = typer.Option("", "--description", "-d"),
    location: str = typer.Option("", "--location", "-l"),
) -> None:
    """Generate an ad-hoc briefing for an event that isn't on a calendar
    yet (e.g. a phone screen scheduled by email)."""
    from .event_briefing import generate_for_event, manual_event

    cfg = load_config()
    conn, embedder = _open_state(cfg)
    reranker = make_reranker(cfg)
    try:
        ev = manual_event(
            title=title, starts_at_iso=starts_at,
            description=description,
            attendees=list(attendees or []),
            location=location,
        )
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        conn.close()
        raise typer.Exit(code=1) from None
    console.print(f"[cyan]Generating ad-hoc briefing for[/] [bold]{ev.title}[/]...")
    result = generate_for_event(cfg, conn, embedder, reranker, ev)
    if not result.get("ok"):
        console.print(f"[red]failed:[/] {result.get('error', '?')}")
        conn.close()
        raise typer.Exit(code=1)
    console.print()
    console.print(result["text"])
    conn.close()


@brief_app.command("regenerate")
def brief_regenerate(
    event_id: str = typer.Argument(..., help="event_id from `brief upcoming`."),
    source: str = typer.Option(
        "google_calendar", "--source",
        help="Event source. One of: google_calendar / ics / manual.",
    ),
) -> None:
    """Force-regenerate a briefing (e.g. after the meeting agenda changed)."""
    from .event_briefing import generate_for_event, iter_upcoming_events

    cfg = load_config()
    conn, embedder = _open_state(cfg)
    reranker = make_reranker(cfg)
    # Find the matching event in the next 24h.
    matched = None
    for ev in iter_upcoming_events(cfg, 24 * 3600):
        if ev.event_id == event_id and ev.source == source:
            matched = ev
            break
    if matched is None:
        console.print(f"[red]Event not found in upcoming calendar.[/] "
                      f"id={event_id} source={source}")
        conn.close()
        raise typer.Exit(code=1)
    result = generate_for_event(cfg, conn, embedder, reranker, matched)
    if not result.get("ok"):
        console.print(f"[red]failed:[/] {result.get('error', '?')}")
        conn.close()
        raise typer.Exit(code=1)
    console.print(f"[green]✓[/] Regenerated briefing for {matched.title!r}")
    conn.close()


@brief_app.command("today")
def brief_today(
    markdown: bool = typer.Option(
        False, "--markdown/--no-markdown",
        help="Print raw Markdown (e.g. for piping into a file or email).",
    ),
) -> None:
    """Phase 44: morning brief that aggregates everything else.

    Pulls together today's calendar, Canvas assignments due in the next
    72h, open action items from recent meeting transcripts, the top of
    your reading queue, and watchlist hits from the last day. Pure
    aggregation — no LLM call, no network beyond the calendar fetch.
    """
    from .daily_brief import generate_brief_markdown

    cfg = load_config()
    conn, _ = _open_state(cfg)
    md = generate_brief_markdown(cfg, conn)
    if markdown:
        # Bypass Rich's markdown renderer so the output stays exactly
        # what we'll send by email / save to a file.
        print(md)
    else:
        from rich.markdown import Markdown
        console.print(Markdown(md))
    conn.close()


links_app = typer.Typer(
    no_args_is_help=True,
    help="Phase 52 auto-backlinks. After a doc lands in the index, "
         "we compute its top-K most-similar siblings and store the "
         "pairs both ways. Use these subcommands to inspect or "
         "rebuild the graph.",
)
app.add_typer(links_app, name="links")


@links_app.command("show")
def links_show(
    path: str = typer.Argument(
        ...,
        help="Doc path (or substring) — supports virtual paths like "
             "'transcript://granola/abc' or filesystem paths.",
    ),
    limit: int = typer.Option(10, "--limit", "-n"),
) -> None:
    """Show the top related docs for a path. Lower distance = more similar."""
    from .backlinks import get_backlinks_for_path

    cfg = load_config()
    conn, _ = _open_state(cfg)
    rows = get_backlinks_for_path(conn, path, limit=limit)
    if not rows:
        # Try a substring match before giving up — paths are long.
        candidate = conn.execute(
            "SELECT path FROM files WHERE path LIKE ? "
            "ORDER BY indexed_at DESC LIMIT 1",
            (f"%{path}%",),
        ).fetchone()
        if candidate is None:
            console.print(f"[yellow]No doc matches[/] {path!r}")
            conn.close()
            return
        rows = get_backlinks_for_path(conn, candidate["path"], limit=limit)
        console.print(f"[dim]Matched:[/] {candidate['path']}")
    if not rows:
        console.print("[yellow]No backlinks recorded for this doc.[/] "
                      "Run `secondbrain links rebuild` to compute them.")
        conn.close()
        return
    table = Table(show_header=True, box=None, title="Related docs")
    table.add_column("score", style="dim", width=6)
    table.add_column("title")
    table.add_column("path", style="dim")
    for r in rows:
        table.add_row(
            f"{r.percent}%",
            r.title[:60],
            r.path[-50:] if len(r.path) > 50 else r.path,
        )
    console.print(table)
    conn.close()


@links_app.command("rebuild")
def links_rebuild(
    k: int = typer.Option(5, "--k", "-k", help="Neighbours per file."),
    max_distance: float = typer.Option(
        1.0, "--max-distance",
        help="Distance threshold; pairs above this aren't recorded.",
    ),
) -> None:
    """Drop the existing graph and recompute from scratch. Useful
    after switching embedders or for a one-time backfill."""
    from .backlinks import rebuild_all

    cfg = load_config()
    conn, _ = _open_state(cfg)
    n_files = conn.execute(
        "SELECT COUNT(*) AS n FROM files",
    ).fetchone()["n"]
    if n_files == 0:
        console.print("[yellow]Index is empty.[/]")
        conn.close()
        return
    console.print(f"[cyan]Rebuilding backlinks across {n_files} files...[/]")

    last_pct = -1
    def on_progress(done: int, total: int) -> None:
        nonlocal last_pct
        pct = (done * 100) // max(1, total)
        if pct != last_pct and pct % 5 == 0:
            console.print(f"[dim]  {done}/{total} ({pct}%)[/]")
            last_pct = pct

    written = rebuild_all(
        conn, k=k, max_distance=max_distance, on_progress=on_progress,
    )
    console.print(f"[green]✓[/] Recorded {written} pair-rows.")
    conn.close()


tasks_app = typer.Typer(
    no_args_is_help=True,
    help="Phase 47 tasks. First-class action items extracted from "
         "meeting transcripts (Granola, Plaud, generic) and ad-hoc "
         "ones you add manually. Use `tasks list` for what's open and "
         "`tasks done <id>` to close one.",
)
app.add_typer(tasks_app, name="tasks")


@tasks_app.command("list")
def tasks_list(
    show_done: bool = typer.Option(
        False, "--done/--no-done",
        help="Also show recently-completed tasks.",
    ),
    extract: bool = typer.Option(
        True, "--extract/--no-extract",
        help="Materialise tasks from recent transcripts before listing.",
    ),
    limit: int = typer.Option(50, "--limit", "-n"),
) -> None:
    """List open tasks (and recently-done with ``--done``)."""
    from . import tasks as tasks_mod

    cfg = load_config()
    conn, _ = _open_state(cfg)
    if extract:
        n = tasks_mod.materialize_from_transcripts(conn)
        if n:
            console.print(f"[dim]extracted {n} new task(s) from transcripts[/]")
    open_rows = tasks_mod.list_open(conn, limit=limit)
    if not open_rows and not show_done:
        console.print("[green]Inbox zero.[/] No open tasks.")
        conn.close()
        return
    if open_rows:
        table = Table(show_header=True, box=None, title="Open tasks")
        table.add_column("id", style="dim", width=4)
        table.add_column("text")
        table.add_column("source", style="dim")
        for t in open_rows:
            src = "" if t.source_path == "manual" else t.source_title
            table.add_row(str(t.id), t.text, src[:40])
        console.print(table)
        console.print(f"[dim]{len(open_rows)} open task(s)[/]")
    if show_done:
        done = tasks_mod.list_recent_done(conn, limit=limit)
        if done:
            table = Table(
                show_header=True, box=None, title="Recently done",
            )
            table.add_column("id", style="dim", width=4)
            table.add_column("text")
            table.add_column("done", style="dim")
            for t in done:
                when = (
                    time.strftime("%Y-%m-%d %H:%M",
                                  time.localtime(t.completed_at))
                    if t.completed_at else "—"
                )
                table.add_row(str(t.id), t.text, when)
            console.print(table)
    conn.close()


@tasks_app.command("add")
def tasks_add(
    text: str = typer.Argument(..., help="Task text. Quote it if it has spaces."),
) -> None:
    """Add a manual task. Useful for things that didn't come out of a
    meeting (or that you want to track outside the brain)."""
    from . import tasks as tasks_mod

    cfg = load_config()
    conn, _ = _open_state(cfg)
    tid = tasks_mod.add_manual(conn, text)
    if tid is None:
        console.print("[red]Empty task text — nothing to add.[/]")
        conn.close()
        raise typer.Exit(code=1)
    console.print(f"[green]Added task #{tid}:[/] {text}")
    conn.close()


@tasks_app.command("done")
def tasks_done(
    task_id: int = typer.Argument(..., help="Task id (from `tasks list`)."),
) -> None:
    """Mark a task complete. It stops showing up in the daily brief."""
    from . import tasks as tasks_mod

    cfg = load_config()
    conn, _ = _open_state(cfg)
    t = tasks_mod.get(conn, task_id)
    if t is None:
        console.print(f"[red]Task #{task_id} not found.[/]")
        conn.close()
        raise typer.Exit(code=1)
    if not tasks_mod.mark_done(conn, task_id):
        console.print(f"[yellow]Task #{task_id} was already done.[/]")
    else:
        console.print(f"[green]✓[/] #{task_id}: {t.text}")
    conn.close()


@tasks_app.command("cancel")
def tasks_cancel(
    task_id: int = typer.Argument(..., help="Task id."),
) -> None:
    """Mark a task cancelled (didn't do it; not going to)."""
    from . import tasks as tasks_mod

    cfg = load_config()
    conn, _ = _open_state(cfg)
    t = tasks_mod.get(conn, task_id)
    if t is None:
        console.print(f"[red]Task #{task_id} not found.[/]")
        conn.close()
        raise typer.Exit(code=1)
    tasks_mod.mark_cancelled(conn, task_id)
    console.print(f"[dim]✗[/] #{task_id}: {t.text}")
    conn.close()


@tasks_app.command("rm")
def tasks_rm(
    task_id: int = typer.Argument(..., help="Task id."),
) -> None:
    """Hard-delete a task. Useful for fixing typos in manual adds."""
    from . import tasks as tasks_mod

    cfg = load_config()
    conn, _ = _open_state(cfg)
    if not tasks_mod.delete(conn, task_id):
        console.print(f"[red]Task #{task_id} not found.[/]")
        conn.close()
        raise typer.Exit(code=1)
    console.print(f"[dim]Deleted #{task_id}[/]")
    conn.close()


@tasks_app.command("extract")
def tasks_extract(
    days: int = typer.Option(
        14, "--days", "-d",
        help="How many days of transcripts to scan.",
    ),
) -> None:
    """Force-extract tasks from recent transcripts. Idempotent —
    won't duplicate tasks you've already extracted or completed."""
    from . import tasks as tasks_mod

    cfg = load_config()
    conn, _ = _open_state(cfg)
    n = tasks_mod.materialize_from_transcripts(conn, lookback_days=days)
    if n == 0:
        console.print("[dim]No new tasks found.[/]")
    else:
        console.print(f"[green]Extracted {n} new task(s).[/]")
    conn.close()


apply_app = typer.Typer(
    no_args_is_help=True,
    help="Track jobs you've applied to. The watchlist agent skips "
         "already-applied roles when surfacing 'new' items, and the chat "
         "agent can answer 'have I applied to X?' against this list.",
)
app.add_typer(apply_app, name="apply")


@apply_app.command("add")
def apply_add(
    company: str = typer.Argument(..., help="Company name (e.g. 'Anthropic')."),
    role: str = typer.Argument(..., help="Role title (e.g. 'PM Intern, Summer 2026')."),
    url: str | None = typer.Option(None, "--url", "-u", help="Canonical posting URL."),
    source: str | None = typer.Option(
        None, "--source", "-s",
        help="Where you found it: 'linkedin', 'greenhouse:anthropic', 'referral', etc.",
    ),
    notes: str | None = typer.Option(None, "--notes", "-n"),
) -> None:
    """Record a new application."""
    from .db import application_create

    cfg = load_config()
    conn, _ = _open_state(cfg)
    aid = application_create(
        conn, company=company, role_title=role,
        role_url=url, source=source, notes=notes,
    )
    console.print(
        f"[green]Recorded[/] application #{aid}: [bold]{company}[/] · {role}"
    )
    if url:
        console.print(f"  url: [dim]{url}[/]")
    conn.close()


@apply_app.command("list")
def apply_list(
    status: str | None = typer.Option(None, "--status"),
    company: str | None = typer.Option(None, "--company"),
) -> None:
    """List applications, optionally filtered by status / company."""
    from .db import application_list

    cfg = load_config()
    conn, _ = _open_state(cfg)
    rows = application_list(conn, status=status, company=company)
    if not rows:
        console.print("[yellow]No applications match.[/]")
        return
    table = Table(show_header=True, box=None, title="Applications")
    table.add_column("id", style="dim", width=4)
    table.add_column("company")
    table.add_column("role")
    table.add_column("status", justify="center")
    table.add_column("applied", style="dim")
    table.add_column("source", style="dim")
    for r in rows:
        when = time.strftime(
            "%Y-%m-%d", time.localtime(r["applied_at"]),
        )
        status_color = {
            "applied": "cyan", "screen": "yellow", "interview": "yellow",
            "offer": "green", "rejected": "red", "withdrawn": "dim",
            "ghosted": "dim",
        }.get(r["status"], "white")
        table.add_row(
            str(r["id"]),
            r["company"],
            r["role_title"][:50],
            f"[{status_color}]{r['status']}[/]",
            when,
            r["source"] or "",
        )
    console.print(table)
    console.print(f"[dim]{len(rows)} application(s)[/]")
    conn.close()


@apply_app.command("status")
def apply_status(
    application_id: int = typer.Argument(..., help="Application id."),
    new_status: str = typer.Argument(
        ..., help="New status: applied / screen / interview / offer / rejected / withdrawn / ghosted",
    ),
    notes: str | None = typer.Option(None, "--notes", "-n"),
) -> None:
    """Update an application's status (e.g. moved from 'applied' to 'interview')."""
    from .db import APPLICATION_STATUSES, application_get, application_set_status

    cfg = load_config()
    conn, _ = _open_state(cfg)
    if new_status not in APPLICATION_STATUSES:
        console.print(
            f"[red]Unknown status[/] {new_status!r}. "
            f"Valid: {', '.join(APPLICATION_STATUSES)}"
        )
        conn.close()
        raise typer.Exit(code=1)
    if application_get(conn, application_id) is None:
        console.print(f"[red]Application #{application_id} not found.[/]")
        conn.close()
        raise typer.Exit(code=1)
    application_set_status(conn, application_id, new_status, notes=notes)
    console.print(f"[green]Updated[/] #{application_id} → {new_status}")
    conn.close()


@apply_app.command("remove")
def apply_remove(
    application_id: int = typer.Argument(..., help="Application id."),
) -> None:
    """Delete an application record."""
    from .db import application_delete

    cfg = load_config()
    conn, _ = _open_state(cfg)
    application_delete(conn, application_id)
    console.print(f"[green]Deleted[/] application #{application_id}.")
    conn.close()


digest_app = typer.Typer(
    no_args_is_help=True,
    help="Send / inspect the daily watchlist email digest. Configure SMTP "
         "in config.toml (digest_*) and put your password in the "
         "SECONDBRAIN_SMTP_PASSWORD env var.",
)
app.add_typer(digest_app, name="digest")


@digest_app.command("send")
def digest_send_cmd(
    force: bool = typer.Option(
        False, "--force",
        help="Send even if digest_enabled=false in config.",
    ),
) -> None:
    """Send the email digest now (regardless of the daily schedule)."""
    from .digest import send_digest

    cfg = load_config()
    if force:
        cfg.digest_enabled = True
    conn, _ = _open_state(cfg)
    ok, msg = send_digest(cfg, conn)
    if ok:
        console.print(f"[green]✓[/] {msg}")
    else:
        console.print(f"[red]digest failed:[/] {msg}")
        conn.close()
        raise typer.Exit(code=1)
    conn.close()


@digest_app.command("preview")
def digest_preview_cmd() -> None:
    """Render the digest body to stdout without sending. Useful to verify
    config + see what tonight's email will look like."""
    from .digest import _gather, _render_text, last_digest_sent_at

    cfg = load_config()
    conn, _ = _open_state(cfg)
    since = last_digest_sent_at(conn)
    rows = _gather(conn, since)
    console.print(_render_text(rows, since))
    conn.close()


@digest_app.command("history")
def digest_history_cmd(
    limit: int = typer.Option(20, "--limit", "-n"),
) -> None:
    """Show recent digest send attempts."""
    cfg = load_config()
    conn, _ = _open_state(cfg)
    # ensure table exists for fresh installs
    import time as _time

    from .digest import _ensure_digest_runs_table

    _ensure_digest_runs_table(conn)
    rows = conn.execute(
        "SELECT * FROM digest_runs ORDER BY sent_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    if not rows:
        console.print("[dim]No digest sends yet.[/]")
        conn.close()
        return
    table = Table(show_header=True, box=None, title="Digest history")
    table.add_column("when")
    table.add_column("ok?", justify="center", width=4)
    table.add_column("watchlists", justify="right")
    table.add_column("new", justify="right")
    table.add_column("recipients/error", overflow="fold")
    for r in rows:
        when = _time.strftime("%Y-%m-%d %H:%M", _time.localtime(r["sent_at"]))
        ok = "[green]✓[/]" if r["success"] else "[red]✗[/]"
        info = r["recipients"] if r["success"] else (r["error"] or "?")
        table.add_row(
            when, ok,
            str(r["watchlists_summarized"]),
            str(r["new_items_total"]),
            info[:80],
        )
    console.print(table)
    conn.close()


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


watch_app = typer.Typer(
    no_args_is_help=True,
    help="Manage saved recurring queries (watchlists). The daemon runs each "
         "watchlist on its schedule and saves a synthesized 'what's new' "
         "summary you can read on the /watch dashboard page.",
)
app.add_typer(watch_app, name="watch")


@watch_app.command("list")
def watch_list() -> None:
    """List all watchlists."""
    from .db import watchlist_list

    cfg = load_config()
    conn, _ = _open_state(cfg)
    rows = watchlist_list(conn)
    if not rows:
        console.print("[yellow]No watchlists yet.[/]  Add one with: [cyan]secondbrain watch add ...[/]")
        return
    table = Table(show_header=True, box=None, title="Watchlists")
    table.add_column("id", style="dim", width=4)
    table.add_column("name", style="bold")
    table.add_column("query")
    table.add_column("every", style="dim")
    table.add_column("last run", style="dim")
    table.add_column("on?", justify="center", width=3)
    import time as _time

    for r in rows:
        sched = r["schedule_minutes"]
        if sched >= 1440 and sched % 1440 == 0:
            every = f"{sched // 1440}d"
        elif sched >= 60 and sched % 60 == 0:
            every = f"{sched // 60}h"
        else:
            every = f"{sched}m"
        last = "(never)" if not r["last_run_at"] else _time.strftime(
            "%Y-%m-%d %H:%M", _time.localtime(r["last_run_at"])
        )
        on = "✓" if r["enabled"] else "·"
        q = r["query"] if len(r["query"]) <= 60 else r["query"][:60] + "…"
        table.add_row(str(r["id"]), r["name"], q, every, last, on)
    console.print(table)
    conn.close()


@watch_app.command("add")
def watch_add(
    name: str = typer.Argument(..., help="Short name for this watchlist."),
    query: str = typer.Argument(..., help="The query to run on each schedule."),
    every: str = typer.Option(
        "1d", "--every", "-e",
        help="How often to run (e.g. '15m', '2h', '1d'). Min 5 minutes.",
    ),
    preset: str | None = typer.Option(
        None, "--preset", "-p",
        help="Named domain preset to scope web search. "
             "One of: jobs, news, markets, research, ai, dev. "
             "See `secondbrain watch presets`.",
    ),
    domains: list[str] = typer.Option(
        None, "--domain", "-d",
        help="Extra hostname to allow in web search (repeatable). "
             "Combines with --preset.",
    ),
) -> None:
    """Save a new recurring query.

    Examples:
      # Generic watchlist (uses cfg.web_search_allowed_domains).
      secondbrain watch add pm-internships \\
        "PM internships posted today at top US tech" --every 1d

      # Scoped to job sites only.
      secondbrain watch add pm-internships \\
        "PM internships posted today" --preset jobs --every 1d

      # News on a specific topic, plus a custom domain.
      secondbrain watch add ai-news \\
        "What new AI launches happened today?" --preset news \\
        --domain anthropic.com --domain openai.com --every 6h
    """
    from .db import watchlist_create
    from .presets import resolve as resolve_preset

    minutes = _parse_every(every)
    try:
        allowed = resolve_preset(preset, list(domains or []))
    except ValueError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(code=1) from None
    cfg = load_config()
    conn, _ = _open_state(cfg)
    wid = watchlist_create(
        conn, name, query, schedule_minutes=minutes,
        allowed_domains=allowed,
    )
    console.print(
        f"[green]Created[/] watchlist #{wid} '[bold]{name}[/]' running every "
        f"{minutes} minute(s)."
    )
    if allowed:
        scope = (
            f"preset={preset} ({len(allowed)} hosts)" if preset
            else f"{len(allowed)} hosts"
        )
        console.print(f"  web search scoped to: [dim]{scope}[/]")
    console.print("It will run automatically once the daemon picks it up.")
    console.print("Run it now: [cyan]secondbrain watch run " + str(wid) + "[/]")
    conn.close()


@watch_app.command("presets")
def watch_presets() -> None:
    """Print the available domain presets."""
    from .presets import PRESETS

    for name in sorted(PRESETS):
        domains = PRESETS[name]
        console.print(f"[bold]{name}[/] [dim]({len(domains)} hosts)[/]")
        for d in domains:
            console.print(f"  · {d}")
        console.print()


def _parse_every(s: str) -> int:
    """Parse '15m' / '2h' / '1d' into minutes. Bare integer = minutes."""
    s = (s or "").strip().lower()
    if not s:
        return 1440
    unit_map = {"m": 1, "h": 60, "d": 24 * 60}
    if s[-1] in unit_map:
        try:
            n = int(s[:-1])
        except ValueError as exc:
            raise typer.BadParameter(f"can't parse --every {s!r}") from exc
        return max(5, n * unit_map[s[-1]])
    try:
        return max(5, int(s))
    except ValueError as exc:
        raise typer.BadParameter(f"can't parse --every {s!r}") from exc


@watch_app.command("remove")
def watch_remove(
    watchlist_id: int = typer.Argument(..., help="Watchlist id from `watch list`."),
) -> None:
    """Delete a watchlist (cascades to all its run history)."""
    from .db import watchlist_delete

    cfg = load_config()
    conn, _ = _open_state(cfg)
    watchlist_delete(conn, watchlist_id)
    console.print(f"[green]Deleted[/] watchlist #{watchlist_id}.")
    conn.close()


@watch_app.command("run")
def watch_run(
    watchlist_id: int | None = typer.Argument(
        None, help="Specific watchlist id. Omit to run all due watchlists.",
    ),
) -> None:
    """Run a watchlist now (don't wait for the schedule)."""
    from .db import watchlist_get
    from .watchlist import run_due_watchlists, run_watchlist

    cfg = load_config()
    conn, embedder = _open_state(cfg)
    reranker = make_reranker(cfg)
    if watchlist_id is None:
        n = run_due_watchlists(cfg, conn, embedder, reranker)
        console.print(f"[green]Ran[/] {n} due watchlist(s).")
        conn.close()
        return
    row = watchlist_get(conn, watchlist_id)
    if row is None:
        console.print(f"[red]Watchlist #{watchlist_id} not found.[/]")
        raise typer.Exit(code=1)
    console.print(f"[cyan]Running[/] '{row['name']}'...")
    response = run_watchlist(
        cfg, conn, embedder, reranker,
        row["id"], row["query"], row["last_run_at"],
    )
    if response is None:
        console.print("[red]Run failed; see daemon log for details.[/]")
        conn.close()
        raise typer.Exit(code=1)
    console.print()
    console.print(response.text)
    console.print()
    console.print(f"[dim]{len(response.citations)} source(s)[/]")
    conn.close()


@watch_app.command("show")
def watch_show(
    watchlist_id: int = typer.Argument(..., help="Watchlist id."),
) -> None:
    """Show the most recent run output for a watchlist."""
    from .db import watchlist_get
    from .watchlist import latest_summary

    cfg = load_config()
    conn, _ = _open_state(cfg)
    row = watchlist_get(conn, watchlist_id)
    if row is None:
        console.print(f"[red]Watchlist #{watchlist_id} not found.[/]")
        raise typer.Exit(code=1)
    s = latest_summary(conn, watchlist_id)
    console.print(f"[bold]{row['name']}[/]  [dim]({row['query']})[/]")
    if s is None:
        console.print("[dim]Never run.[/]")
        conn.close()
        return
    if s["error"]:
        console.print(f"[red]Last run errored:[/] {s['error']}")
        conn.close()
        return
    console.print(s["answer"] or "(no answer)")
    console.print()
    console.print(f"[dim]{len(s['citations'])} source(s)[/]")
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
