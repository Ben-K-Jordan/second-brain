# Second-brain audit — Phase 14

Scope: a pre-feature-add audit covering secrets hygiene, connector error handling, virtual_path collisions, daemon/WAL/lock integrity, budget guardrails, search-quality regressions, ingest fragility, browser-extension surface, and type/contract consistency.

## Critical (data loss, secret leak, broken sync)

### C1. OAuth refresh leaks `refresh_token` in error path — `src/secondbrain/connectors/_google_oauth.py:218,247`
**Issue:** `GoogleAuthError(f"Token exchange failed: {resp.status_code} {resp.text}")` and `f"Token refresh failed: {resp.status_code} {resp.text}. ..."` interpolate the full Google response body. On a token-exchange race or refresh-rotation case, Google's body contains the new `access_token`/`refresh_token` as JSON. If that exception bubbles up to a CLI command, it ends up on stdout/stderr and (for `daemon`) in `daemon.log`.
**Fix:** Truncate and strip token-shaped fields:
```python
def _scrub(body: str) -> str:
    return re.sub(r'("(?:access_token|refresh_token|id_token)"\s*:\s*")[^"]+"', r'\1<redacted>"', body)[:500]
raise GoogleAuthError(f"Token exchange failed: {resp.status_code} {_scrub(resp.text)}")
```
Or just log the status code and `resp.headers.get("WWW-Authenticate", "")`; never the body.

### C2. Reddit password-grant logs full response body (may contain secrets) — `src/secondbrain/connectors/reddit.py:84`
**Issue:** `log.warning("Reddit token fetch failed: %s %s", token_resp.status_code, token_resp.text[:200])`. Reddit's password-grant endpoint takes `username`, `password`, basic-auth `(client_id, client_secret)` and returns `{"access_token": "...", "refresh_token": "...", ...}` on success. On *misuse* (e.g., incorrect content-type, the wrong app type), Reddit echoes back form fields including the password into JSON error messages. With `text[:200]` we'd write that to the daemon log and tray notifications.
**Fix:** Log only `status_code` and a fixed reason string. Never include `token_resp.text` from a credentials endpoint:
```python
log.warning("Reddit token fetch failed: HTTP %s (check REDDIT_CLIENT_ID/SECRET/USERNAME/PASSWORD)", token_resp.status_code)
```

### C3. Pocket access_token + consumer_key leak in error path — `src/secondbrain/connectors/pocket.py:77`
**Issue:** `log.warning("Pocket /v3/get failed: %s %s", r.status_code, r.text[:200])`. The Pocket API echoes the full request payload (which contains `consumer_key` and `access_token` as JSON fields) in some error responses, and may also expose the consumer_key in the `X-Error-Code` flow. The first 200 chars of an error body easily fits both.
**Fix:** Same as C2 — log status + a static reason; never `r.text` from a credentialed endpoint:
```python
log.warning("Pocket /v3/get failed: HTTP %s (check POCKET_CONSUMER_KEY/POCKET_ACCESS_TOKEN)", r.status_code)
```

### C4. SQLite connection used cross-thread without `check_same_thread=False` — `src/secondbrain/db.py:23`, `daemon.py:141,212`, `watcher.py:106`
**Issue:** `connect()` opens the connection on the main thread. The daemon hands the same `conn` to a `Watcher` whose `_run` worker thread (line 106) calls `index_file(self._conn, ...)`. Python's `sqlite3` defaults to `check_same_thread=True`, which raises `ProgrammingError: SQLite objects created in a thread can only be used in that same thread.` This is silent today only because the watcher worker swallows everything in `log.exception` (`watcher.py:114`) — every file event is logged as "watcher failed" while the bootstrap thread succeeds. Re-indexing on file change does not actually work. Same bug for `run_tray` where `bootstrap_async` runs on a background thread and the watcher runs on another.
**Fix:** Pass `check_same_thread=False` and serialize via the existing `busy_timeout`:
```python
conn = sqlite3.connect(str(db_path), check_same_thread=False)
```
Add a thread lock around mutating calls if there's any chance of concurrent writers within a process.

### C5. WAL file is never checkpointed — `src/secondbrain/db.py:29-34`, `daemon.py:166`
**Issue:** `PRAGMA journal_mode = WAL` is set, but no code ever runs `PRAGMA wal_checkpoint(TRUNCATE)`. On a long-running daemon doing heavy ingest, the `-wal` file can grow until disk is full (the user previously hit 893 MB). SQLite auto-checkpoints on `commit` *only* when no readers hold a snapshot; the dashboard's read-only connection plus the daemon's writer keeps the WAL pinned indefinitely.
**Fix:** Add a periodic checkpoint in the watcher loop or after `_bootstrap`:
```python
# In Watcher._run, every N events or every 10 minutes
self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
```
And explicitly `conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")` before `conn.close()` in `run_daemon`/`run_tray`.

### C6. Budget cap silently disabled if config field is `None` — `src/secondbrain/budget.py:157-162`
**Issue:** `if cap <= 0: return` will raise `TypeError: '<=' not supported between instances of 'NoneType' and 'int'` if `cfg.daily_budget_cents_voyage` is somehow `None` — but `record_usage` is called in `try/except` paths around `check_budget`. Several call sites use a generic `except Exception` that swallows this; the call proceeds and there's no cap. More damaging: there is no fail-closed default. If the dataclass field is ever optional in the future, the cap silently disappears.
**Fix:** Coerce and fail closed:
```python
def check_budget(cfg: Config, provider: str) -> None:
    cap = (cfg.daily_budget_cents_voyage if provider == "voyage"
           else cfg.daily_budget_cents_anthropic)
    if cap is None:
        raise BudgetExceededError(provider, 0, 0)  # fail closed
    if cap <= 0:
        return  # explicit "disabled" sentinel
    ...
```
And keep `int` types non-Optional in the dataclass so the load_config path can never produce None.

### C7. HyDE Anthropic call bypasses budget tracking — `src/secondbrain/search.py:60-67`
**Issue:** `hyde_rewrite` calls `client.messages.create(...)` directly with no `check_budget(cfg, "anthropic")` and no `record_usage(...)`. Every search with `hyde_enabled = true` (and a 4+ word query) makes an uncounted, uncapped Anthropic call. If a UI integration loops on a query refresh, this will burn through real money with no entry in the spend ledger.
**Fix:** Plumb `cfg` into `hyde_rewrite` (and through `hybrid_search`) and gate it:
```python
def hyde_rewrite(cfg: Config, query: str, model: str = "claude-haiku-4-5", ...):
    check_budget(cfg, "anthropic")
    ...
    record_usage(cfg, "anthropic", model,
                 input_tokens=response.usage.input_tokens,
                 output_tokens=response.usage.output_tokens, note="hyde")
```

### C8. Spend ledger writes have no fsync and no cross-process lock — `src/secondbrain/budget.py:111-114`
**Issue:** `_LEDGER_LOCK` is a `threading.Lock` — *in-process only*. The daemon, the dashboard (`uvicorn` is its own process via `run_dashboard`), and the MCP server can each be running and writing `spend.jsonl` concurrently. Linux's `O_APPEND` saves us for atomic-line semantics on writes < PIPE_BUF, but Windows offers no such guarantee. There's also no `f.flush(); os.fsync(f.fileno())` — a power loss loses the most-recent spend, so the cap under-reports and re-runs the loop. Combined with C6 this breaks the only safety net.
**Fix:** Use a file-based lock and fsync on write:
```python
import portalocker  # or msvcrt/fcntl by platform
with open(path, "a", encoding="utf-8") as f:
    portalocker.lock(f, portalocker.LOCK_EX)
    try:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())
    finally:
        portalocker.unlock(f)
```

### C9. Browser-extension API is unauthenticated; any allowed origin can exfiltrate the brain — `src/secondbrain/dashboard.py:1467-1535`, `browser-extension/manifest.json`
**Issue:** `/api/extension/search` returns up to 10 chunks (1500 chars each) of the user's indexed content, gated only by an Origin allow-list (`chatgpt.com`, `gemini.google.com`, `x.com`, etc.). Any compromised script on those origins — a Chrome extension on chat.openai.com, an XSS in any `*.openai.com` subdomain (cookies aside, `Origin` is just a header), a malicious user-script — can call `http://127.0.0.1:8765/api/extension/search?q=*&k=10` and read arbitrary content from the index. `q=password` or `q=ssn` is a fast exfil. Origin checks are not authentication.
**Fix:** Issue a per-install token: write a random secret to `cfg.data_dir/extension_token.txt` on first run, surface it in `secondbrain extension token` for the user to paste into the extension popup, then require it as `Authorization: Bearer ...` on every `/api/extension/*` call. Also reject if `request.client.host != "127.0.0.1"`. Bonus: rate-limit per-token.

### C10. Browser-extension manifest grants content-script access to chat origins (foundation for C9) — `browser-extension/manifest.json:18-49`
**Issue:** Content scripts are injected into `chat.openai.com`, `chatgpt.com`, `gemini.google.com`, `perplexity.ai`, `x.com`, `grok.com`, `chat.deepseek.com` — plus `host_permissions` for `127.0.0.1:8765`. Once C9 is fixed (auth token), the extension itself becomes the trust boundary. Right now the *extension* and *any other JS on those origins* are equivalently powerful. The token issued in C9 must live in `chrome.storage.local` and be added by the extension's `background.js`, never readable from a content script.
**Fix:** Move token storage to background service worker (`chrome.storage.local`), have content scripts message-pass requests through the background, and never expose the token to page DOM. Document this in the extension README.

### C11. `dedupe_existing` picks canonical by lowest id, not by oldest indexed_at — `src/secondbrain/indexer.py:399-411`
**Issue:** Comment says "pick the oldest indexed_at as the canonical primary - lowest id is a good proxy". But `AUTOINCREMENT` rowid only correlates with insertion *order*, not with `indexed_at`. After a `reset --reembed` or a partial re-index, files re-inserted later get higher ids; the truly oldest record can be a higher id. More importantly, this throws away the older file's chunks (cascade) and replaces with a possibly-stale alias. If the canonical file later gets deleted from disk and the watcher fires `delete_file`, the alias rows are now orphaned (no canonical, paths still in `file_aliases` pointing at a deleted file_id). Cascade delete drops the aliases via `ON DELETE CASCADE`, but the user has lost the actual content.
**Fix:** Pick canonical by `MIN(indexed_at)` and keep that file's row:
```sql
SELECT id FROM files WHERE content_hash = ? ORDER BY indexed_at ASC LIMIT 1
```
And when the canonical file is deleted from disk, promote the oldest alias to canonical instead of cascading.

## High (degraded UX, partial failure, hard to debug)

### H1. Google access tokens never refresh mid-sync — `src/secondbrain/connectors/_google_oauth.py:274-285`
**Issue:** `authorized_session` snapshots `creds.access_token` into a static header at session-creation time. Google access tokens expire after 1 hour. On a first-run Drive or Gmail sync of a few thousand items, fetches partway through start returning 401 once the token expires; the connectors `log.warning("Gmail fetch %s failed: %s", msg_id, r.status_code)` and skip — the rest of the sync is silently lost.
**Fix:** Wrap the Session in an auth class that re-checks expiry before each request:
```python
class _AutoRefreshSession(requests.Session):
    def __init__(self, cfg, scopes): ...
    def request(self, method, url, **kw):
        creds = get_credentials(self._cfg, self._scopes)  # refreshes if expired
        self.headers["Authorization"] = f"Bearer {creds.access_token}"
        return super().request(method, url, **kw)
```
Or, on every 401, refresh once and retry.

### H2. Notion `/blocks/{id}/children` failures swallowed silently — `src/secondbrain/connectors/notion.py:127-128`
**Issue:** `if r.status_code != 200: break` exits the pagination loop with whatever blocks were collected, no logging. A page that hits Notion's 429 partway through ends up indexed as a half-page with no error. The user sees stale Notion content and thinks indexing is fine.
**Fix:** `log.warning("Notion blocks fetch %s failed: HTTP %s", block_id, r.status_code); break`.

### H3. Token refresh not retried on 401 in any Google connector — `gmail.py:155-158`, `google_calendar.py:81-84`, `google_drive.py:108-111`
**Issue:** Compounds H1. When a 401 hits mid-sync, all three connectors `return` and abandon the rest of the sync. Re-running `sync gmail` won't recover the missed window because mtime-based incremental skips the already-skipped messages.
**Fix:** Distinguish 401 from 5xx. On 401, force `_refresh()` once and retry; only bail if refresh itself fails. (Handled jointly with H1.)

### H4. 429 rate-limits treated identically to 500s — every connector
**Issue:** Slack's `_slack_get` (line 79-81) just records `error`. Notion / Linear / GitHub return on any non-200. None implement `Retry-After`. A first-run Gmail or Drive sync that trips Google's 100 RPM cap aborts at the cap and silently truncates.
**Fix:** Add a small util in `connectors/__init__.py`:
```python
def respect_retry_after(r, attempts=3):
    if r.status_code == 429:
        wait = float(r.headers.get("Retry-After", "5"))
        time.sleep(min(60.0, wait))
        return True
    return False
```
Wrap each connector's request loop with it.

### H5. Daemon log file has no rotation and is always appended — `src/secondbrain/daemon.py:122-129`, `205-209`
**Issue:** `logging.FileHandler(log_path, encoding="utf-8")` writes append-only with no rotation. Combined with `INFO` level + `info("indexing %s", path)` per file, a daemon running for weeks can produce hundreds of MB of `daemon.log`. The user previously hit a similar symptom with `index.db-wal`.
**Fix:** Use `RotatingFileHandler`:
```python
from logging.handlers import RotatingFileHandler
h = RotatingFileHandler(log_path, maxBytes=10*1024*1024, backupCount=3, encoding="utf-8")
```

### H6. `get_credentials` returns `None` for missing scope without distinguishing from "not authed" — `src/secondbrain/connectors/_google_oauth.py:264-271`
**Issue:** Both "no creds file" and "creds exist but lack a scope" return `None`. Callers print `"no Google credentials. Run `secondbrain auth google`"` (gmail.py:130). After running `auth google`, if the user only granted Gmail and the Drive connector then asks for `drive.readonly`, they get the same misleading message. They re-run auth, get redirected, re-grant Gmail, and are baffled.
**Fix:** Return a small enum or raise a typed error:
```python
class ScopeMissing(GoogleAuthError): pass
...
if any(s not in creds.scopes for s in required_scopes):
    raise ScopeMissing(f"missing scopes: {set(required_scopes) - set(creds.scopes)}")
```
And surface the missing scopes in the log message.

### H7. Reranker can return out-of-range index if Voyage truncates input — `src/secondbrain/search.py:393-397`
**Issue:** `rerank_pairs = reranker.rerank(query, docs, top_k=k)` and then `cid = cids[orig_idx]`. If Voyage's API returns a stale `index` (e.g., bug in their library, or an oversize `documents` list that gets server-truncated), an `IndexError` aborts the search. There's also no defensive `if cid not in hydrated` here even though the same code defends elsewhere.
**Fix:**
```python
for orig_idx, score in rerank_pairs:
    if orig_idx < 0 or orig_idx >= len(cids):
        continue
    cid = cids[orig_idx]
    if cid not in hydrated:
        continue
    ...
```

### H8. `index_url` doesn't dedup against existing content hashes — `src/secondbrain/indexer.py:604-607`
**Issue:** `index_text` and `index_file` both call `find_file_by_hash` and convert to alias on hash match. `index_url` only checks `get_file_by_path(conn, url)`. So ingesting the same Wikipedia page twice via two slightly different URLs (canonical vs. mobile, or with/without `?utm_source`) makes duplicate primary rows with duplicate chunks and duplicate Voyage spend.
**Fix:** Add the same `find_file_by_hash` / `add_alias` block right after computing `chash`:
```python
if existing is None:
    twin = find_file_by_hash(conn, chash)
    if twin is not None:
        add_alias(conn, twin["id"], url); conn.commit()
        return IndexResult(label_path, "alias", reason=f"duplicate of {twin['path']}")
```

### H9. Google Calendar virtual_path collides across calendars — `src/secondbrain/connectors/google_calendar.py:170`
**Issue:** `virtual_path=f"google_calendar://{ev_id}"`. Event IDs are unique *within* a calendar, not across calendars. A user with a personal calendar + a shared "team" calendar where someone forwarded an invite will have the same `ev_id` show up in both. The second run of the iterator overwrites the first via the upsert in `index_text`. The user loses the team-calendar copy of their meeting and now searches return only the calendar that was iterated last.
**Fix:** Include calendar id:
```python
virtual_path=f"google_calendar://{cal['id']}/{ev_id}",
```

### H10. Browser-history virtual_path embeds full URL including query secrets — `src/secondbrain/connectors/browser.py:108`
**Issue:** `virtual_path=f"browser://{label}/{url}"`. The URL goes verbatim into the `files.path` column and the connector content. If the user visited `https://example.com/reset?token=abc123def`, that exact token is now in the index and surfaces in search results / the dashboard / MCP responses. Also, search palette UI displays paths.
**Fix:** Strip query string before using as virtual_path; keep full URL only in metadata:
```python
from urllib.parse import urlsplit, urlunsplit
clean = urlunsplit(urlsplit(url)._replace(query="", fragment=""))
virtual_path=f"browser://{label}/{clean}",
```

### H11. HN scraping: same item flips between `favorite` and `own` on re-runs — `src/secondbrain/connectors/hacker_news.py:69-83`
**Issue:** The `seen` set is per-fetch only. Run #1: item 12345 is in your favorites, gets indexed with `metadata.label = "favorite"`. Run #2: same id is now also in your submissions; `_iter_favorites` is iterated first so it stays "favorite". But if you delete a favorite, run #3 sees it only in `own`, the upsert path overwrites with `label="own"`, and now the stored `kind` is wrong. More general: HN-API `kind` (story/comment/poll) is captured per-item too — those don't change, but the *metadata* nondeterminism makes search filters (`kind=favorite`) lie.
**Fix:** Either merge labels on conflict, or pick a single canonical label. Easiest: emit both labels into a list:
```python
metadata={"labels": list({source_label, *prev_labels})}
```
or stop using "favorite" vs "own" as a `kind` and store both flags as metadata booleans.

### H12. Pocket `data["list"]` can be a *list* not just dict-or-empty-array — `src/secondbrain/connectors/pocket.py:82-86`
**Issue:** Comment claims `list` is dict-or-empty-array, but Pocket's API has occasionally returned `list: []` for empty pages. Today's code handles that. What it doesn't handle: `data` itself being absent (auth failure response), or `list` being `None` due to a backend hiccup. With `(r.json() or {}).get("list")`, a `None` value is treated as "empty" and the loop ends — but there's no `log.warning`, so a transient auth issue silently looks like "you have no Pocket items".
**Fix:** Distinguish:
```python
if "list" not in data:
    log.warning("Pocket unexpected payload: %s", list(data)[:5])
    return
items_obj = data.get("list") or {}
```

### H13. X archive: silent on non-Twitter-archive directories — `src/secondbrain/connectors/x_archive.py:91-118`
**Issue:** If `X_ARCHIVE_PATH` points at a directory with no `tweets.js` / `tweet.js` / `bookmark.js` / `like.js`, `is_enabled` returns True (it's a dir), and `fetch` yields zero documents with no log. The user thinks the connector synced.
**Fix:** Log once when no files found, suggest the expected layout:
```python
if not (tweets_path or bookmarks_path or likes_path):
    log.warning("X archive at %s has no tweets.js/bookmark.js/like.js — wrong directory?", root)
```

### H14. CSRF on dashboard `/ingest` POST — `src/secondbrain/dashboard.py:1423`
**Issue:** No CSRF token, no Origin/Referer check on `POST /ingest`. A malicious page that the user visits in the same browser (any origin, not just the CORS-allowed list — form POSTs aren't gated by CORS) can submit `<form action="http://127.0.0.1:8765/ingest" method="post">` with `name="url"` set to e.g. an internal SSRF target. The server fetches that URL with a real-browser User-Agent (`indexer.py:435`) and indexes whatever comes back. SSRF + content disclosure to anyone who can read `/file?path=` afterward.
**Fix:** Require a same-origin Origin header, or a CSRF cookie+token pair, on `POST /ingest`:
```python
@app.post("/ingest")
def ingest_action(request: Request, url: str = Form(...)):
    origin = request.headers.get("origin", "")
    if origin and not origin.startswith(("http://127.0.0.1", "http://localhost")):
        return Response(status_code=403)
    ...
```

## Medium (latent bugs, edge cases)

### M1. `ConnectorDocument.kind` field is dead — `src/secondbrain/connectors/__init__.py:38`
**Issue:** Default `"url"`. Every connector sets `metadata["kind"]` (e.g., `"own_tweet"`, `"favorite"`, `"bookmark"`, `"comment"`) and never sets the dataclass `kind` field. Downstream `index_text` accepts `kind: str = "url"` as its own arg, which is then stored on `files.kind`. The dataclass field is never read.
**Fix:** Either remove `kind` from `ConnectorDocument`, or actually use it: in the sync CLI, pass `doc.kind` into `index_text(..., kind=doc.kind)`. Pick one.

### M2. `_fetch_user_directory` Slack pagination can spin forever on a malformed cursor — `src/secondbrain/connectors/slack.py:87-112`
**Issue:** If `users.list` returns `{ok: true, response_metadata: {next_cursor: "x"}}` and then on the next call returns the same `next_cursor` (rare Slack bug), the loop runs indefinitely. There's no max-iteration cap.
**Fix:** Bound to ~50 pages (covers ten thousand users):
```python
for _ in range(50):
    ...
```

### M3. `index_text` `find_file_by_hash` can mark a connector doc as alias of a *file* — `src/secondbrain/indexer.py:511-515`
**Issue:** If the user's local file `/notes/reddit-saved.md` has the same SHA-1 as a Reddit `selftext`, the Reddit connector doc becomes an alias of the local file. It vanishes from search even though it's a different "thing" the user might query for via "what did I save on Reddit". Practically rare for SHA-1, but content-only hash is too coarse — they share text but mean different things.
**Fix:** Salt the hash by source tier:
```python
chash = hashlib.sha1(("connector:" + virtual_path[:virtual_path.find("://")] + ":\n" + text).encode()).hexdigest()
```
Or skip cross-source dedup (only dedup connector-to-connector, file-to-file).

### M4. `chunk_text` paragraph-overlap drift — `src/secondbrain/indexer.py:166-168`
**Issue:** When a single paragraph exceeds `target_size`, we split with `step = target_size - overlap` but emit `p_stripped[i : i + target_size]`, so the overlap region is consistent. However, `p_offset + i` is the offset *within the paragraph*, not within the original document — `p_offset = cursor` is set to where the paragraph began, but `cursor += len(p) + len(paragraph_pattern)` was already applied, so `p_offset` is correct. Verified — but `flush()` resets `current_start` to whatever the next paragraph offset is, even if the current accumulator started earlier. After a long-paragraph split, the next chunk's `current_start` is off by the long-paragraph length. Citation offsets will point into the wrong region.
**Fix:** After a long-paragraph split, set `current_start = cursor` so the next regular paragraph's offset is the post-split position.

### M5. `_path_score_multiplier` matches case-insensitively but config says "Documents" — `src/secondbrain/search.py:158-163`, `config.py:148`
**Issue:** Default `personal_path_prefixes = ("/Documents/", ...)`. The implementation lowercases both sides — fine on Windows, where path case varies. But on macOS/Linux, `~/Documents` is the user's notes dir and matches. Then a user-overridable `extra_ignore_globs`-style mishap: if a user adds `personal_path_prefixes = ["/"]` to favor everything, it silently boosts every result. No bound check.
**Fix:** Document the surprising substring-match semantics, and add a sanity floor:
```python
if any(len(p) <= 2 for p in personal_prefixes):
    log.warning("personal_path_prefixes contains very short prefix; will match too broadly")
```

### M6. `replace_chunks` deletes vec rows then chunks but holds neither in a savepoint — `src/secondbrain/db.py:294-331`
**Issue:** No `SAVEPOINT` around the delete-then-insert. If embedding takes a long time and the daemon crashes between `DELETE FROM chunks WHERE file_id` and the `INSERT INTO chunks`, the file row exists with `content_hash = chash` but no chunks. On restart, the file looks "indexed" (hash matches) but is invisible to FTS/vector search until the user manually `secondbrain reset` or the file is touched.
**Fix:** Wrap in a savepoint, and roll back content_hash on failure:
```python
conn.execute("SAVEPOINT replace_chunks")
try:
    ...
    conn.execute("RELEASE SAVEPOINT replace_chunks")
except Exception:
    conn.execute("ROLLBACK TO SAVEPOINT replace_chunks"); raise
```
Or set the `files.content_hash` only AFTER the new chunks/vectors are committed.

### M7. ICS calendar `e` may include URL — `src/secondbrain/connectors/calendar.py:107`
**Issue:** `log.warning("calendar fetch failed: %s", e)` — `requests.RequestException.__str__` may include the URL with query string. The "secret address" iCal feed has the secret in the URL. On DNS failure or 5xx, the secret leaks to log.
**Fix:**
```python
log.warning("calendar fetch failed: %s", type(e).__name__)
```
Or scrub the URL before logging.

### M8. `dedupe_existing` can race with active writers — `src/secondbrain/indexer.py:387-430`
**Issue:** Iterates rows, then does multi-statement updates per row, with one big commit at the end. If the daemon is also indexing during a manual `secondbrain dedupe`, both write to `files`/`chunks`/`vec_chunks` and a busy_timeout retry storm can leave inconsistent state (one alias added, but the canonical's chunks half-deleted by another writer).
**Fix:** Either acquire `BEGIN IMMEDIATE` at the start, or document that `dedupe` should only run with the daemon stopped.

### M9. `chunk_text` doesn't guard against `target_size < overlap` — `src/secondbrain/indexer.py:166`
**Issue:** `step = max(1, target_size - overlap)`. If a user sets `chunk_size = 100, chunk_overlap = 200` (silly but possible), step = 1 — every chunk is a 100-char window stepped by 1 char. A 50KB doc produces 50,000 chunks, all embedded.
**Fix:** Validate at config load time:
```python
if cfg.chunk_overlap >= cfg.chunk_size:
    raise ValueError("chunk_overlap must be < chunk_size")
```

### M10. X archive JS prefix regex strips only one prefix — `src/secondbrain/connectors/x_archive.py:43,63`
**Issue:** `_PREFIX_RE.sub("", raw, count=1)`. The archive is supposed to start with one assignment, but if Twitter ever adds a comment or a leading semicolon (the old `; window.YTD...` pattern), the JSON parse fails silently (`return []`). Also, extremely large `tweets.js` files (>1 GB on long-running accounts) are read into memory in one shot — `path.read_text(encoding="utf-8")` blows up RAM.
**Fix:** Make the regex tolerant of `;` and whitespace:
```python
_PREFIX_RE = re.compile(r"^\s*;?\s*window\.YTD\.[A-Za-z0-9_]+\.part\d+\s*=\s*", re.DOTALL)
```
And for huge archives, fall back to streaming via `ijson` when filesize > 100 MB.

### M11. `_render_event` GCal attendees list keeps `None` entries — `src/secondbrain/connectors/google_calendar.py:181-183`
**Issue:** `[a.get("email") for a in attendees_list if a.get("email")]` — fine. But the *displayed* attendees on line 142-144 use `a.get("email") or a.get("displayName") or ""`, which can produce empty strings. Joined with `, ` you get `"alice@x.com, , bob@y.com"`.
**Fix:** Filter empties before joining:
```python
attendees = ", ".join(filter(None, (a.get("email") or a.get("displayName") for a in attendees_list)))
```

### M12. Briefing calls `check_budget` but doesn't catch `BudgetExceededError` for the user — `src/secondbrain/briefing.py:191`
**Issue:** Confirmed via grep — briefing uses `check_budget` but if the cap fires from CLI/MCP, the user sees an unhandled exception. (Sub-issue of C6/C7 ergonomics.)
**Fix:** Catch `BudgetExceededError` at the CLI/MCP boundaries and render a friendly "skip this run" message.

### M13. Gmail HTML fallback strips `<style>` but not preformatted `<pre>` — `src/secondbrain/connectors/gmail.py:46-50`
**Issue:** A code-heavy email gets its `<pre>` blocks tag-stripped, joining lines with no newline preservation. Search recall on code-in-email drops.
**Fix:** Convert `<pre>` and `<br>` and `</p>` to `\n` before the catch-all `<[^>]+>` strip:
```python
text = re.sub(r"<pre[^>]*>", "\n```\n", text, flags=re.IGNORECASE)
text = re.sub(r"</pre>", "\n```\n", text, flags=re.IGNORECASE)
```

### M14. `x_archive._iter_dms` ignores `is_enabled` opt-in for the connector — `src/secondbrain/connectors/x_archive.py:113`
**Issue:** DMs are gated by `SB_X_INCLUDE_DMS == "1"` *inside* `fetch`, but `is_enabled` doesn't mention it. A user who enables the connector and gets DMs they didn't expect will be confused by the env-var default ("off"). And conversely, opting into DMs just by env var with no UI affordance is easy to forget about during a `sync`.
**Fix:** Either log a one-line "DMs included" notice at start of fetch, or surface DM opt-in in dashboard config.

## Low (cleanup, consistency)

### L1. Inconsistent timeout values — calendar 60s, all others 30s — `connectors/*.py`
**Issue:** No reason for the asymmetry. Pick one default; document.
**Fix:** Constant `_DEFAULT_TIMEOUT = 30` in `connectors/__init__.py`; bump only for known-large endpoints (Drive export, Linear graphql).

### L2. `add_alias` imports `time` locally with alias — `src/secondbrain/db.py:247`
**Issue:** `import time as _time` inside a function. The module already imports `time` at the top.
**Fix:** Drop the local import and use the module-level one.

### L3. `RedditConnector._iter_listing` uses 500-default but env var doesn't override — `src/secondbrain/connectors/reddit.py:107`
**Issue:** Reddit hardcodes `limit=500`; HN exposes `SB_HN_MAX`, Slack exposes `SB_SLACK_DAYS`, etc. Inconsistent.
**Fix:** Add `SB_REDDIT_MAX` and read at the top of `fetch`.

### L4. `dedupe_existing` returns counts but no list of paths converted — `src/secondbrain/indexer.py:426-430`
**Issue:** A dry-run mode and a verbose log of which paths were aliased would be useful for debugging. Currently the user has to query `file_aliases` post-hoc.
**Fix:** Add `dry_run: bool = False` and either return the path list or log them at INFO.

### L5. Voyage rerank-2 (non-lite) is in the price table but not in the default config — `budget.py:42`
**Issue:** `rerank-2` priced at $0.10/1M, `rerank-2-lite` at $0.05/1M; default uses lite, which is fine. But `rerank-2-lite` is faster and `rerank-2` is more accurate — users have no signal to switch.
**Fix:** Comment in `default_config_toml` mentioning the trade-off.

### L6. `dashboard.py` imports `_log_query` from `mcp_server` lazily inside endpoint — `dashboard.py:1523`
**Issue:** Lazy import inside a hot path; works but adds a ~ms per request and obscures dependencies.
**Fix:** Import at module top.

### L7. `index_url` doesn't pass `entity_extractor` to its retry/error paths — `src/secondbrain/indexer.py:638-648`
**Issue:** Same try/except boilerplate copy/pasted three times across `index_text`/`index_url`/`index_file`. Drift is inevitable.
**Fix:** Extract `_run_entity_extraction(conn, chunk_ids, chunk_texts, entity_extractor)` helper.

### L8. `_extract_title` Notion uses `_, prop` from items() — `src/secondbrain/connectors/notion.py:138`
**Issue:** Idiomatic but the loop variable `_` is conventionally for "unused", here we never use the key — fine. Just noisy.
**Fix:** `for prop in (page.get("properties") or {}).values(): ...`.

### L9. `_eligible_chunk_ids` materializes potentially large set — `src/secondbrain/search.py:243-270`
**Issue:** For an index with millions of chunks, `path_prefix` filter materializes the whole chunk-id set into a Python set, then membership-tests each candidate. Fine at current scale; would matter at 10M+ chunks.
**Fix:** Move the filter into the FTS/vec SQL via a `WHERE c.id IN (SELECT ...)` subquery instead of post-filtering Python-side.

### L10. `_fetch_url_to_tempfile` doesn't honor `Content-Length` cap — `src/secondbrain/indexer.py:450-455`
**Issue:** No `stream=True`, no max-size guard. A malicious URL serving a 10 GB response gets fully buffered into memory.
**Fix:**
```python
resp = requests.get(url, stream=True, timeout=60, ...)
size = int(resp.headers.get("content-length", "0"))
if size > 200 * 1024 * 1024:
    raise RuntimeError(f"refusing to download {size} bytes")
```

### L11. Calendar `seen_uids` dedup discards potentially-different RECURRENCE-ID overrides — `src/secondbrain/connectors/calendar.py:111-117`
**Issue:** ICS uses UID + RECURRENCE-ID for occurrence overrides; deduping on UID alone drops legitimate recurrence exceptions.
**Fix:** Key on `(uid, recurrence_id)` if RECURRENCE-ID is present.

## Nit (style, naming, dead code)

### N1. `MEDIA_EXTENSIONS` "backwards-compat alias" — `src/secondbrain/config.py:96-97`
**Issue:** Comment says "remove after one release if nothing depends on it." Search the tree once and remove.
**Fix:** `grep -r MEDIA_EXTENSIONS src/`; if zero hits, delete.

### N2. Dataclass `ConnectorDocument` field order — `connectors/__init__.py:24-39`
**Issue:** Required fields and one defaulted field (`kind`) interleaved with `metadata` having a `field(default_factory=dict)`. Works but a future addition of a required field will trip the dataclass ordering rule.
**Fix:** Move `kind` after `metadata`, or remove (M1).

### N3. `print()` in `run_oauth_flow` instead of `log.info` — `src/secondbrain/connectors/_google_oauth.py:192-193`
**Issue:** Stdout printing during a CLI command that may be invoked from `secondbrain auth google` is fine, but inconsistent with the rest of the module's logging.
**Fix:** Use a Rich console or `log.info` for consistency.

### N4. `CalendarConnector` shadowed by Google Calendar — `connectors/__init__.py:60-86`
**Issue:** Two classes both register: ICS-based `CalendarConnector` (`name="calendar"`) and Google's `GoogleCalendarConnector` (`name="google_calendar"`). Distinct names, but ergonomics: a user who set `CALENDAR_ICS_URL` *and* ran `auth google` ends up double-indexing the same events.
**Fix:** Document the overlap; or have the Google connector emit a metadata flag the ICS one looks for.

### N5. `_render_thing` walrus opportunity — `src/secondbrain/connectors/reddit.py:134-141`
**Issue:** Pure style.
**Fix:** Skip.

### N6. `BrowserHistoryConnector._read_profile` cleanup uses bare `OSError` — `src/secondbrain/connectors/browser.py:115-118`
**Issue:** `tmp_path.unlink()` swallowing any `OSError` is fine for cleanup, but on Windows the file may still be locked by sqlite if the close above failed.
**Fix:** Wrap `conn.close()` in try/finally too.

### N7. Default `SB_GMAIL_QUERY` excludes Promotions but not `category:forums` — `gmail.py:39`
**Issue:** Forums emails (mailing lists) are arguably the highest-value email content to index but are excluded by `_DEFAULT_QUERY`'s `-category:updates`.
**Fix:** Just a doc comment noting the user can override.

### N8. Several connectors set `User-Agent: second-brain/0.0.1` hardcoded — multiple files
**Issue:** Should track `__version__`. Today we'll never bump.
**Fix:** Single constant in `secondbrain/__init__.py`; everyone imports.
