"""Speech models downloaded when the user chooses them, into DOWNLOADS_DIR.

Each model is a fixed list of files at a pinned revision, with each file's size and SHA-256. A file is only kept when
its checksum matches, and the model only counts as installed when every file is there (a marker file is written
last). A broken connection resumes where it stopped; the first connection, which stalls on some laptops, is retried;
and the user can cancel at any time (the part already downloaded stays for next time). Only Hugging Face's own
servers are accepted after redirects, and k2-fsa's GitHub releases (sherpa-onnx's own models, which they publish only
as archives): such a model's archive is pinned and checked like a file, then only the model's files are read out of it,
by their own names (no path in the archive is used), each checked again, and the archive is deleted.
"""
import hashlib
import http.client
import json
import logging
import shutil
import tarfile
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from sst import DOWNLOADS_DIR, __version__

ATTEMPTS = 4  # per request: the first connection sometimes stalls, the next gets through
ATTEMPT_TIMEOUT = 15.0  # seconds to connect, and for each read
CHUNK = 1 << 20
MARKER = "complete.json"
K2_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download/"  # sherpa-onnx's makers' own releases
GITHUB_ASSETS = {"release-assets.githubusercontent.com", "objects.githubusercontent.com"}  # where those redirect

log = logging.getLogger(__name__)


class DownloadError(Exception):
    pass


class Cancelled(DownloadError):
    pass


@dataclass(frozen=True)
class ModelFile:
    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class Download:
    folder: str  # its folder in DOWNLOADS_DIR
    base_url: str  # each file is base_url + its name
    files: tuple[ModelFile, ...]
    # Or the files are in this archive (base_url + its name, a .tar.bz2 whose top folder is named like it)
    archive: ModelFile | None = None

    @property
    def size(self) -> int:
        """The bytes to download."""
        return self.archive.size if self.archive else sum(f.size for f in self.files)

    def path(self, root: Path | None = None) -> Path:
        return (root or DOWNLOADS_DIR) / self.folder

    def installed(self, root: Path | None = None) -> bool:
        """Everything downloaded and checked (no hashing here: the marker is only written after the checks)."""
        folder = self.path(root)
        return (folder / MARKER).exists() and all(
            (folder / f.name).exists() and (folder / f.name).stat().st_size == f.size for f in self.files)


def allowed(url: str) -> bool:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    return (host == "huggingface.co" or host.endswith(".huggingface.co") or host.endswith(".hf.co")
            or url.startswith(K2_RELEASES) or (parts.scheme == "https" and host in GITHUB_ASSETS))


def download(model: Download, progress: Callable[[int, int], None] = lambda done, total: None,
             cancelled: Callable[[], bool] = lambda: False, root: Path | None = None) -> Path:
    """Fetch every missing file of `model`, check it, and return the model's folder. Raises Cancelled or DownloadError
    (with a readable reason)."""
    folder = model.path(root)
    if model.installed(root):
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    if model.archive is not None:
        _from_archive(model, folder, progress, cancelled)
    else:
        done = sum(f.size for f in model.files if _verified(folder / f.name, f))
        progress(done, model.size)
        for f in model.files:
            target = folder / f.name
            if _verified(target, f):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)  # a file in a subfolder (a voice's "lang/gmw/en-US")
            done = _fetch(model.base_url + f.name, target, f, done, model.size, progress, cancelled)
    (folder / MARKER).write_text(json.dumps({"files": [f.name for f in model.files], "version": __version__}),
                                 encoding="utf-8")
    log.info("Downloaded %s (%d MB)", model.folder, model.size // 2**20)
    return folder


def _from_archive(model: Download, folder: Path, progress, cancelled) -> None:
    """The model's files out of its archive (see the module's notes). The archive stays until every file is out, so a
    failure halfway needs no second download."""
    packed, top = folder / model.archive.name, model.archive.name.removesuffix(".tar.bz2")
    if not all(_verified(folder / f.name, f) for f in model.files):
        if not _verified(packed, model.archive):
            progress(0, model.size)
            _fetch(model.base_url + model.archive.name, packed, model.archive, 0, model.size, progress, cancelled)
        try:
            with tarfile.open(packed, "r:bz2") as tar:
                for f in model.files:
                    target = folder / f.name
                    if not _verified(target, f):
                        _unpack(tar, f"{top}/{f.name}", target, f)
        except (tarfile.TarError, EOFError, OSError) as e:  # checked, yet unreadable: a disk problem
            raise DownloadError(f"couldn't unpack {model.archive.name}: {e}") from None
    progress(model.size, model.size)
    packed.unlink(missing_ok=True)


def _unpack(tar: tarfile.TarFile, name: str, target: Path, f: ModelFile) -> None:
    """One file read out of the archive by its own name, into target (never a path from the archive), checked."""
    try:
        member = tar.getmember(name)
    except KeyError:
        member = None
    if member is None or not member.isfile() or member.size != f.size:
        raise DownloadError(f"{f.name} isn't in the archive as expected")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    with tar.extractfile(member) as source, partial.open("wb") as out:
        shutil.copyfileobj(source, out, CHUNK)
    if _sha256(partial) != f.sha256:
        partial.unlink()
        raise DownloadError(f"{f.name} arrived damaged (checksum mismatch); try again")
    partial.replace(target)


def remove(model: Download, root: Path | None = None) -> None:
    shutil.rmtree(model.path(root), ignore_errors=True)


_checked: set[tuple[str, int]] = set()  # files already hashed in this run: a 1.6 GB file takes seconds to hash


def _verified(path: Path, f: ModelFile) -> bool:
    if not path.exists() or path.stat().st_size != f.size:
        return False
    key = (str(path), path.stat().st_mtime_ns)
    if key in _checked:
        return True
    if _sha256(path) != f.sha256:
        path.unlink()
        return False
    _checked.add(key)
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch(url: str, target: Path, f: ModelFile, done: int, total: int, progress, cancelled) -> int:
    """Download one file into target.part, resuming it, then check and rename it. Returns the bytes done so far."""
    partial = target.with_name(target.name + ".part")
    if partial.exists() and partial.stat().st_size > f.size:
        partial.unlink()  # not this file: start again
    have = partial.stat().st_size if partial.exists() else 0
    error: Exception | None = None
    for _ in range(ATTEMPTS):
        if have == f.size:
            break
        try:
            with _open(url, have) as response, partial.open("ab" if have else "wb") as out:
                if have and response.status != 206:  # the server sent the whole file again: start over
                    out.truncate(0)
                    have = 0
                while chunk := response.read(CHUNK):
                    if cancelled():
                        raise Cancelled("cancelled")
                    out.write(chunk)
                    have += len(chunk)
                    progress(done + have, total)
        except (OSError, http.client.HTTPException) as e:  # a broken or stalled connection: go on from where it stopped
            error = e
            have = partial.stat().st_size if partial.exists() else 0
            continue
    if have != f.size:
        raise DownloadError(f"could not download {f.name}: {getattr(error, 'reason', error) or 'incomplete'}")
    if _sha256(partial) != f.sha256:
        partial.unlink()
        raise DownloadError(f"{f.name} arrived damaged (checksum mismatch); try again")
    partial.replace(target)
    return done + f.size


def _open(url: str, start: int):
    headers = {"User-Agent": f"Rflow/{__version__}"}
    if start:
        headers["Range"] = f"bytes={start}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=ATTEMPT_TIMEOUT)  # uses the Windows proxy settings, if any
    except urllib.error.HTTPError as e:  # the server answered: trying again won't help
        raise DownloadError(f"the model server answered {e.code} {e.reason}") from None
    if not allowed(response.geturl()):  # after redirects: only Hugging Face's own servers
        response.close()
        raise DownloadError(f"unexpected download location: {urlsplit(response.geturl()).hostname}")
    return response
