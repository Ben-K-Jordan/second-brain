"""Cross-platform desktop notifications.

Stdlib-only, best-effort. Each platform shells out to its native
notification command:

- Windows  : PowerShell + Windows.UI.Notifications XML toast (Windows 10+).
- macOS    : ``osascript -e 'display notification ...'``
- Linux    : ``notify-send`` if installed; falls back to a log line.

Failure modes never raise — a notification that didn't appear must not
take down the daemon. We swallow exceptions and log a single warning.

Used by:
- Watchlist runs: when a run finds new items vs. the previous run.
- Daily email digest: when the digest sends successfully (optional).

The ``url`` arg is a best-effort "click this to go straight to the
relevant page" target. Windows toasts honor it; macOS / Linux ignore it.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys

log = logging.getLogger(__name__)


def notify(title: str, message: str, url: str | None = None) -> bool:
    """Show a desktop notification. Returns True if it likely went through.

    The Windows path uses PowerShell with a XAML toast; the message is
    embedded via base64 so we don't have to worry about quote escaping.
    """
    title = (title or "").strip() or "second-brain"
    message = (message or "").strip()
    if not message:
        return False

    try:
        if sys.platform == "win32":
            return _notify_windows(title, message, url)
        if sys.platform == "darwin":
            return _notify_macos(title, message)
        return _notify_linux(title, message)
    except Exception as e:  # noqa: BLE001
        log.warning("notify failed: %s", e)
        return False


# --------------------------- platform impls ---------------------------

def _notify_windows(title: str, message: str, url: str | None) -> bool:
    """Show a Windows 10+ toast via PowerShell + Windows.UI.Notifications.

    Wraps the PowerShell script in a base64 -EncodedCommand to avoid all
    quote-escaping headaches. The toast is "transient" (no Action Center
    persistence) since we treat watchlist diffs as ephemeral.
    """
    import base64

    # PowerShell: build a XAML toast and dispatch via the ToastNotificationManager.
    ps_script = (
        "[Windows.UI.Notifications.ToastNotificationManager, "
        "Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null;"
        "$xml = New-Object Windows.Data.Xml.Dom.XmlDocument;"
        f"$xml.LoadXml('<toast><visual><binding template=\"ToastGeneric\">"
        f"<text>{_xml_escape(title)}</text>"
        f"<text>{_xml_escape(message)}</text>"
        f"</binding></visual></toast>');"
        "$toast = New-Object Windows.UI.Notifications.ToastNotification $xml;"
        "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("
        "'second-brain').Show($toast);"
    )
    encoded = base64.b64encode(ps_script.encode("utf-16-le")).decode("ascii")
    try:
        subprocess.run(
            [
                "powershell.exe", "-NoProfile", "-NonInteractive",
                "-WindowStyle", "Hidden", "-EncodedCommand", encoded,
            ],
            check=False, timeout=8,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("Windows toast failed (%s); falling back to msg.exe", e)
        # Last-ditch fallback: msg.exe pops a message box. Ugly but
        # always works on Windows. Truncate to keep the box manageable.
        try:
            subprocess.run(
                ["msg", "*", f"[{title}] {message[:200]}"],
                check=False, timeout=4,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.SubprocessError):
            return False
    return True


def _notify_macos(title: str, message: str) -> bool:
    # display notification escapes via "" -> double-quote in AppleScript.
    safe_title = title.replace('"', "''")
    safe_msg = message.replace('"', "''")
    script = (
        f'display notification "{safe_msg}" with title "{safe_title}" '
        f'sound name "Glass"'
    )
    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=False, timeout=5,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _notify_linux(title: str, message: str) -> bool:
    # Prefer notify-send when installed (gnome / KDE / most desktops).
    if shutil.which("notify-send"):
        try:
            subprocess.run(
                ["notify-send", "-a", "second-brain", title, message],
                check=False, timeout=5,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return True
        except (OSError, subprocess.SubprocessError):
            pass
    log.info("notify (no notify-send): [%s] %s", title, message)
    return False


def _xml_escape(s: str) -> str:
    """Minimal escape for the XAML toast template."""
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("'", "&apos;")
    )
