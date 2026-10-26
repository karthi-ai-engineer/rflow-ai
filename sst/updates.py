"""In-app updates from GitHub Releases: find a newer Rflow, download its installer, check it, install it.

A release carries Rflow-Setup.exe and Rflow-Setup.exe.sha256 (the Release workflow publishes both). The installer is
only run when its SHA-256 matches, and only files served by GitHub are accepted. It then installs with a small
progress window (/SILENT) and starts the new Rflow again (/UPDATE=1, see packaging/installer.iss).
"""
import hashlib
import json
import logging
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from sst import __version__

REPO = "karthi-ai-engineer/rflow-ai"
LATEST_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
INSTALLER = "Rflow-Setup.exe"
CHECKSUM = INSTALLER + ".sha256"
# Where GitHub serves release files from (downloads redirect from github.com to one of its file hosts).
ALLOWED_HOSTS = {"api.github.com", "github.com", "objects.githubusercontent.com", "release-assets.githubusercontent.com"}
CHECK_EVERY_HOURS = 6
# On some laptops (security software, networks) a program's first connection stalls; the next attempt gets
# through at once. So: a short limit per attempt, and a few attempts (each read of a download also gets this limit).
ATTEMPTS = 3
ATTEMPT_TIMEOUT = 8.0

log = logging.getLogger(__name__)


class UpdateError(Exception):
    pass


@dataclass
class Update:
    version: str  # e.g. "1.1.0"
    notes: str  # the release notes (markdown)
    page: str  # the release page on GitHub
    installer_url: str
    checksum_url: str
    size: int  # bytes


def parse_version(text: str) -> tuple[int, ...]:
    """'v1.2.0' or '1.2' -> (1, 2, 0); anything after the numbers (e.g. '-beta') is ignored."""
    numbers = re.match(r"v?(\d+(?:\.\d+)*)", text.strip())
    parts = [int(n) for n in numbers.group(1).split(".")] if numbers else [0]
    return tuple((parts + [0, 0, 0])[:3])


def check(current: str = __version__, url: str = LATEST_URL) -> Update | None:
    """The newest published release if it is newer than `current`, else None. Raises UpdateError when GitHub can't
    be asked (offline, rate limit...)."""
    try:
        release = json.loads(_get(url))
    except ValueError:
        raise UpdateError("GitHub sent an unreadable answer") from None
    if not isinstance(release, dict):
        raise UpdateError("GitHub sent an unexpected answer")
    tag = str(release.get("tag_name", ""))
    if release.get("draft") or release.get("prerelease") or parse_version(tag) <= parse_version(current):
        return None
    assets = {a.get("name"): a for a in release.get("assets", []) if isinstance(a, dict)}
    if INSTALLER not in assets or CHECKSUM not in assets:
        log.warning("Release %s has no %s and %s; not offering it", tag, INSTALLER, CHECKSUM)
        return None
    return Update(version=".".join(map(str, parse_version(tag))), notes=str(release.get("body") or ""),
                  page=str(release.get("html_url") or ""), installer_url=assets[INSTALLER]["browser_download_url"],
                  checksum_url=assets[CHECKSUM]["browser_download_url"], size=int(assets[INSTALLER].get("size") or 0))


def download(update: Update, folder: Path, progress=lambda done, total: None) -> Path:
    """Download the installer into `folder` and check it against the published SHA-256. Returns its path."""
    expected = _get(update.checksum_url).decode("ascii", "replace").split()
    if not expected or not re.fullmatch(r"[0-9a-fA-F]{64}", expected[0]):
        raise UpdateError("the release has no valid checksum")
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"Rflow-Setup-{update.version}.exe"
    partial = target.with_suffix(".part")
    digest, done = hashlib.sha256(), 0
    try:
        with _open(update.installer_url) as response, partial.open("wb") as out:
            total = int(response.headers.get("Content-Length") or update.size or 0)
            while chunk := response.read(1 << 20):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                progress(done, total)
        if digest.hexdigest().lower() != expected[0].lower():
            raise UpdateError("the download is damaged (checksum mismatch); try again")
        partial.replace(target)
        return target
    finally:
        partial.unlink(missing_ok=True)


def install(installer: Path) -> None:
    """Start the installer (small progress window, no questions); it restarts Rflow when done. The caller must quit
    right after this, and release its "running" mutex first so the installer doesn't stop to ask. It writes its log
    next to itself, for the next start to tell whether it finished (`unfinished`)."""
    log_file = installer.parent / INSTALL_LOG
    log_file.unlink(missing_ok=True)
    subprocess.Popen([str(installer), "/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/UPDATE=1", f"/LOG={log_file}"],
                     creationflags=subprocess.DETACHED_PROCESS)


INSTALL_LOG = "install.log"
_HOLDER = re.compile(r"RestartManager found an application using one of our files: (.+)")


def unfinished(folder: Path, current: str = __version__) -> str:
    """Why the last update didn't install ("" if none was tried or it installed), from the installer's log: in silent
    mode it rolls back when a program it can't close holds Rflow's files (an editor Rflow once opened, say)."""
    tried = [parse_version(p.stem.removeprefix("Rflow-Setup-")) for p in folder.glob("Rflow-Setup-*.exe")]
    try:
        text = (folder / INSTALL_LOG).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    newer = [v for v in tried if v > parse_version(current)]
    if not newer or "Installation process succeeded" in text:
        return ""
    version = ".".join(map(str, max(newer)))
    # a terminal's console host goes with the terminal: not worth naming
    holders = list(dict.fromkeys(m.strip() for m in _HOLDER.findall(text) if not m.startswith("Console Window")))
    if not holders:
        return f"Rflow {version} didn't install. Click Update now to try again."
    names = holders[0] if len(holders) == 1 else ", ".join(holders[:-1]) + " and " + holders[-1]
    return (f"Rflow {version} didn't install: {names} {'was' if len(holders) == 1 else 'were'} using Rflow's files. "
            f"Close {'it' if len(holders) == 1 else 'them'}, then click Update now.")


def _open(url: str, timeout: float = ATTEMPT_TIMEOUT):
    request = urllib.request.Request(url, headers={"User-Agent": f"Rflow/{__version__}",
                                                   "Accept": "application/vnd.github+json, application/octet-stream"})
    error: OSError | None = None
    for _ in range(ATTEMPTS):
        try:
            response = urllib.request.urlopen(request, timeout=timeout)  # uses the Windows proxy settings, if any
            break
        except urllib.error.HTTPError as e:  # GitHub answered (e.g. a rate limit): trying again won't help
            raise UpdateError(f"GitHub answered {e.code} {e.reason}") from None
        except OSError as e:  # no connection: on some networks the first attempt stalls and the next one works
            error = e
    else:
        raise UpdateError(f"could not reach GitHub: {getattr(error, 'reason', error)}")
    host = urlsplit(response.geturl()).hostname or ""
    if host not in ALLOWED_HOSTS:  # after redirects: only GitHub's own servers
        response.close()
        raise UpdateError(f"unexpected download location: {host}")
    return response


def _get(url: str) -> bytes:
    with _open(url) as response:
        return response.read()
