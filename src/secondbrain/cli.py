"""Typer CLI entry point."""

from __future__ import annotations

import logging
import sys
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
from .db import connect, init_schema, stats
from .embedder import make_embedder
from .entities import make_entity_extractor
from .image_embedder import make_image_embedder
from .imager import make_ocr_engine
from .indexer import IndexResult, index_folder, index_url, walk_folder
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
    """Show what's currently in the index."""
    cfg = load_config()
    conn, _ = _open_state(cfg)
    s = stats(conn)
    table = Table(show_header=False, box=None)
    table.add_row("Files", str(s["files"]))
    table.add_row("Chunks", str(s["chunks"]))
    table.add_row("Embedder", f"{s['embedder']} (dim {s['embedding_dim']})")
    table.add_row("DB path", str(cfg.db_path))
    table.add_row("Watched folders", ", ".join(str(p) for p in cfg.watched_folders) or "(none)")
    console.print(table)
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
