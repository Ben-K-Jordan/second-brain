"""Google OAuth2 scaffold — reusable across Gmail / Calendar / Drive / Photos.

Handles the redirect-server dance once and stores access + refresh tokens
locally. Every Google connector calls ``get_credentials(scopes)`` and gets
back an authenticated requests.Session that auto-refreshes when the access
token expires.

Setup is one-time, manual:
  1. Go to https://console.cloud.google.com → New Project (or pick one).
  2. APIs & Services → Library → enable Gmail API, Calendar API.
  3. APIs & Services → OAuth consent screen → External, Testing mode.
     Add your Gmail address as a test user.
  4. APIs & Services → Credentials → Create Credentials → OAuth client ID,
     application type = Desktop app.
  5. Download the JSON, save as ``~/.secondbrain/google_client_secret.json``.

Then run ``secondbrain auth google`` once. Credentials persist at
``~/.secondbrain/google_credentials.json``; refresh happens automatically.
"""

from __future__ import annotations

import http.server
import json
import logging
import re
import socket
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

import requests

from ..config import Config

log = logging.getLogger(__name__)

_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_REDIRECT_HOST = "127.0.0.1"


# Strip token-shaped fields from any response body before it ends up in a
# log line or exception message. Google's token endpoint returns access_token
# / refresh_token / id_token as JSON, and on some error paths echoes them
# back. Truncate to 500 chars regardless.
_TOKEN_FIELD_RE = re.compile(
    r'("(?:access_token|refresh_token|id_token)"\s*:\s*")[^"]+"', re.IGNORECASE
)


def _scrub(body: str) -> str:
    return _TOKEN_FIELD_RE.sub(r'\1<redacted>"', body or "")[:500]


@dataclass
class GoogleCredentials:
    """Persisted Google OAuth state. We don't include client_secret because
    the user's client_secret.json holds it; we only persist tokens."""

    access_token: str
    refresh_token: str
    expires_at: float
    scopes: list[str] = field(default_factory=list)
    token_type: str = "Bearer"

    @property
    def expired(self) -> bool:
        # Refresh 60s early so we don't race with API calls.
        return time.time() >= self.expires_at - 60


class GoogleAuthError(RuntimeError):
    pass


class ScopeMissing(GoogleAuthError):
    """Raised by ``get_credentials`` when stored creds don't cover required scopes.

    Distinguished from ``GoogleAuthError`` so callers can show a useful message
    ("you authorized for X but this connector needs Y") instead of a generic
    "no Google credentials" prompt.
    """

    def __init__(self, missing: set[str]) -> None:
        self.missing = missing
        super().__init__(f"Missing Google scopes: {sorted(missing)}")


def _client_secret_path(cfg: Config) -> Path:
    return cfg.data_dir / "google_client_secret.json"


def _credentials_path(cfg: Config) -> Path:
    return cfg.data_dir / "google_credentials.json"


def _load_client_secret(cfg: Config) -> dict:
    path = _client_secret_path(cfg)
    if not path.exists():
        raise GoogleAuthError(
            f"Google client_secret.json not found at {path}. "
            "Download it from https://console.cloud.google.com → OAuth credentials "
            "and save it there."
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Google wraps the credentials under either "installed" or "web"
    if "installed" in data:
        return data["installed"]
    if "web" in data:
        return data["web"]
    raise GoogleAuthError(f"Unexpected client_secret.json shape at {path}")


def _save_credentials(cfg: Config, creds: GoogleCredentials) -> None:
    path = _credentials_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "access_token": creds.access_token,
            "refresh_token": creds.refresh_token,
            "expires_at": creds.expires_at,
            "scopes": creds.scopes,
            "token_type": creds.token_type,
        }, f, indent=2)


def _load_credentials(cfg: Config) -> GoogleCredentials | None:
    path = _credentials_path(cfg)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return GoogleCredentials(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=float(data["expires_at"]),
            scopes=data.get("scopes", []),
            token_type=data.get("token_type", "Bearer"),
        )
    except (OSError, KeyError, ValueError) as e:
        log.warning("could not read google_credentials.json: %s", e)
        return None


def _free_port() -> int:
    """Find a free port for the redirect callback. Google requires the port
    to be in the OAuth consent screen, but for Desktop app type any port works."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((_REDIRECT_HOST, 0))
        return s.getsockname()[1]


class _OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    """One-shot HTTP handler for the OAuth redirect."""
    received_code: str | None = None
    received_error: str | None = None

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        if "code" in params:
            type(self).received_code = params["code"][0]
            body = (
                "<html><body style='font-family: sans-serif; padding: 40px;'>"
                "<h2>Authorization complete</h2>"
                "<p>You can close this tab and return to the terminal.</p>"
                "</body></html>"
            )
        elif "error" in params:
            type(self).received_error = params["error"][0]
            body = f"<html><body><h2>Authorization failed: {params['error'][0]}</h2></body></html>"
        else:
            body = "<html><body>Waiting...</body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, format: str, *args) -> None:  # noqa: A002, ARG002
        # Suppress default access logging - we have our own.
        pass


def run_oauth_flow(cfg: Config, scopes: list[str], open_browser: bool = True) -> GoogleCredentials:
    """Drive the full OAuth2 redirect flow. Returns fresh credentials."""
    client = _load_client_secret(cfg)
    client_id = client["client_id"]
    client_secret = client["client_secret"]

    port = _free_port()
    redirect_uri = f"http://{_REDIRECT_HOST}:{port}/oauth/callback"

    auth_params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(scopes),
        "access_type": "offline",       # we want a refresh_token
        "prompt": "consent",            # force re-issue refresh_token
        "include_granted_scopes": "true",
    }
    auth_url = f"{_AUTH_URL}?{urllib.parse.urlencode(auth_params)}"

    # Reset class-level state on the handler in case run_oauth_flow is called twice.
    _OAuthCallbackHandler.received_code = None
    _OAuthCallbackHandler.received_error = None

    server = http.server.HTTPServer((_REDIRECT_HOST, port), _OAuthCallbackHandler)
    thread = threading.Thread(target=server.handle_request, daemon=True)
    thread.start()

    log.info("Opening browser for Google authorization...")
    log.info("If it doesn't open, visit: %s", auth_url)
    if open_browser:
        try:
            webbrowser.open(auth_url)
        except Exception:
            pass

    # Wait up to 5 minutes for the user to complete the flow.
    thread.join(timeout=300)
    server.server_close()

    if _OAuthCallbackHandler.received_error:
        raise GoogleAuthError(f"Authorization failed: {_OAuthCallbackHandler.received_error}")
    if not _OAuthCallbackHandler.received_code:
        raise GoogleAuthError("Authorization timed out (no callback received)")

    # Exchange code for tokens.
    resp = requests.post(_TOKEN_URL, data={
        "code": _OAuthCallbackHandler.received_code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }, timeout=30)
    if resp.status_code != 200:
        raise GoogleAuthError(
            f"Token exchange failed: {resp.status_code} {_scrub(resp.text)}"
        )
    body = resp.json()
    if "refresh_token" not in body:
        raise GoogleAuthError(
            "No refresh_token in response. Revoke the app at "
            "https://myaccount.google.com/permissions and try again."
        )

    creds = GoogleCredentials(
        access_token=body["access_token"],
        refresh_token=body["refresh_token"],
        expires_at=time.time() + int(body.get("expires_in", 3600)),
        scopes=body.get("scope", " ".join(scopes)).split(),
        token_type=body.get("token_type", "Bearer"),
    )
    _save_credentials(cfg, creds)
    return creds


def _refresh(cfg: Config, creds: GoogleCredentials) -> GoogleCredentials:
    client = _load_client_secret(cfg)
    resp = requests.post(_TOKEN_URL, data={
        "client_id": client["client_id"],
        "client_secret": client["client_secret"],
        "refresh_token": creds.refresh_token,
        "grant_type": "refresh_token",
    }, timeout=30)
    if resp.status_code != 200:
        raise GoogleAuthError(
            f"Token refresh failed: {resp.status_code} {_scrub(resp.text)}. "
            "Re-run `secondbrain auth google` to re-authorize."
        )
    body = resp.json()
    creds.access_token = body["access_token"]
    creds.expires_at = time.time() + int(body.get("expires_in", 3600))
    # Google sometimes rotates refresh tokens; respect any new value.
    if "refresh_token" in body:
        creds.refresh_token = body["refresh_token"]
    _save_credentials(cfg, creds)
    return creds


def get_credentials(cfg: Config, required_scopes: list[str]) -> GoogleCredentials | None:
    """Return valid credentials covering ``required_scopes``, refreshing if
    expired. Returns None if no credentials are stored yet — caller should
    prompt the user to run ``secondbrain auth google``.

    Raises ``ScopeMissing`` (a ``GoogleAuthError``) if creds exist but lack
    one or more required scopes — distinct from "not authed at all" so the
    UX can tell the user *which* scope is missing instead of asking them to
    re-auth blind.
    """
    creds = _load_credentials(cfg)
    if creds is None:
        return None
    missing = {s for s in required_scopes if s not in creds.scopes}
    if missing:
        raise ScopeMissing(missing)
    if creds.expired:
        creds = _refresh(cfg, creds)
    return creds


class _AutoRefreshSession(requests.Session):
    """A requests.Session that re-checks token expiry before every request.

    The previous ``authorized_session`` snapshotted the access token into a
    static header at session-creation time. Google access tokens expire after
    1h, so a long Drive or Gmail first-sync would 401 partway through and
    silently truncate. This subclass calls ``get_credentials`` (which refreshes
    on expiry) before each request, and on a 401 forces one refresh + retry.
    """

    def __init__(self, cfg: Config, scopes: list[str]) -> None:
        super().__init__()
        self._cfg = cfg
        self._scopes = list(scopes)
        self.headers.update({"User-Agent": "second-brain/0.0.1"})

    def _apply_token(self) -> None:
        creds = get_credentials(self._cfg, self._scopes)
        if creds is None:
            raise GoogleAuthError(
                "No Google credentials. Run `secondbrain auth google` first."
            )
        self.headers["Authorization"] = f"{creds.token_type} {creds.access_token}"

    def request(self, method, url, **kwargs):  # type: ignore[override]
        self._apply_token()
        resp = super().request(method, url, **kwargs)
        # On 401, force one refresh + retry. Covers the case where the token
        # was valid at request-prep time but Google rejected it (clock skew,
        # rotation, etc.).
        if resp.status_code == 401:
            creds = _load_credentials(self._cfg)
            if creds is not None:
                try:
                    _refresh(self._cfg, creds)
                except GoogleAuthError as e:
                    log.warning("token refresh after 401 failed: %s", e)
                    return resp
                self._apply_token()
                resp = super().request(method, url, **kwargs)
        return resp


def authorized_session(cfg: Config, scopes: list[str]) -> requests.Session | None:
    """Return a requests.Session with auto-token-refresh on every request.

    Returns None if no creds stored. Raises ``ScopeMissing`` if creds exist
    but don't cover ``scopes``.
    """
    creds = get_credentials(cfg, scopes)
    if creds is None:
        return None
    return _AutoRefreshSession(cfg, scopes)


def is_authorized(cfg: Config, scopes: list[str]) -> bool:
    """Check whether stored credentials cover the requested scopes."""
    creds = _load_credentials(cfg)
    if creds is None:
        return False
    return all(s in creds.scopes for s in scopes)
