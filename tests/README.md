# Test suite

Hand-rolled pytest. ~60 fast tests covering the parts the audit found
fragile: budget guardrails, connector protocol contracts, indexer dedup,
search-ranker math, and chat persistence.

## Run it

```bash
pip install -e ".[dev]"
pytest                  # all fast tests
pytest -m "not slow"    # explicit (same as above today)
pytest -k click         # just the click-feedback tests
pytest -x --tb=short    # stop on first failure with short traceback
```

## What's gated

- `slow` marker — tests that touch the real index, the network, or paid
  APIs. None defined yet; reserved for the future.
- `_no_network_env` autouse fixture — strips every credential env var
  (`VOYAGE_API_KEY`, `ANTHROPIC_API_KEY`, all the connector tokens)
  before every test, so a misbehaving fast test can't accidentally hit a
  paid endpoint.

## How tests use a fake DB

`conftest.py` ships:

- `tmp_cfg` — a `Config` rooted at an isolated `tmp_path`.
- `fresh_db` — a freshly-migrated SQLite connection on a temp file.
- `fake_embedder` — deterministic 16-dim hash-based vectors so search /
  indexer tests don't need Voyage. Same input always produces the same
  vector, so retrieval is repeatable across runs.

If your test needs a different embedding dim (e.g. 8 to match a fixture),
build the connection inline rather than reusing `fresh_db`. See
`test_indexer.py::test_index_text_dedup_by_hash` for the pattern.

## What we deliberately don't test (yet)

- Whisper / OCR / spaCy / Voyage / Anthropic round-trips. These are heavy
  and would slow CI to a crawl. The `slow` marker is reserved for them
  if we ever want them.
- The dashboard FastAPI surface. Lots of HTML scaffolding; smoke value is
  low. We instead test the underlying library functions and trust the
  thin endpoint wrappers.
- The MCP stdio protocol. FastMCP handles the wire format; we test the
  tool implementations directly.

## CI

`.github/workflows/test.yml` runs `ruff check` + `pytest -m "not slow"`
on Python 3.11 and 3.12 against every push and PR to `main`.
