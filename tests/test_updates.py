"""In-app updates against a fake GitHub on this machine: which release is offered, and that an installer is only
accepted with a matching checksum from an allowed server."""
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from sst import updates
from sst.updates import UpdateError, parse_version

INSTALLER_BYTES = b"MZ fake installer " * 5000


class FakeGitHub:
    def __init__(self):
        self.release = {}
        self.checksum = hashlib.sha256(INSTALLER_BYTES).hexdigest() + "  Rflow-Setup.exe\n"
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/releases/latest":
                    body, kind = json.dumps(fake.release).encode(), "application/json"
                elif self.path.endswith("/Rflow-Setup.exe"):
                    body, kind = INSTALLER_BYTES, "application/octet-stream"
                elif self.path.endswith("/Rflow-Setup.exe.sha256"):
                    body, kind = fake.checksum.encode(), "text/plain"
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def publish(self, tag, assets=("Rflow-Setup.exe", "Rflow-Setup.exe.sha256"), **extra):
        self.release = {"tag_name": tag, "body": "What changed", "html_url": f"{self.base}/release/{tag}",
                        "assets": [{"name": a, "size": len(INSTALLER_BYTES), "browser_download_url": f"{self.base}/dl/{tag}/{a}"}
                                   for a in assets], **extra}


@pytest.fixture
def github(monkeypatch):
    server = FakeGitHub()
    monkeypatch.setattr(updates, "LATEST_URL", server.base + "/releases/latest")
    monkeypatch.setattr(updates, "ALLOWED_HOSTS", {"127.0.0.1"})
    yield server
    server.server.shutdown()


@pytest.mark.parametrize("text, expected", [("v1.2.3", (1, 2, 3)), ("1.2", (1, 2, 0)), ("v2", (2, 0, 0)),
                                            ("v1.10.0-beta", (1, 10, 0)), ("nonsense", (0, 0, 0))])
def test_parse_version(text, expected):
    assert parse_version(text) == expected


def test_a_newer_release_is_offered(github):
    github.publish("v1.1.0")
    update = updates.check(current="1.0.0", url=updates.LATEST_URL)
    assert update.version == "1.1.0" and update.notes == "What changed"
    assert update.installer_url.endswith("/v1.1.0/Rflow-Setup.exe") and update.size == len(INSTALLER_BYTES)


@pytest.mark.parametrize("tag", ["v1.0.0", "v0.9.0"])
def test_the_same_or_an_older_release_is_not_offered(github, tag):
    github.publish(tag)
    assert updates.check(current="1.0.0", url=updates.LATEST_URL) is None


def test_10_is_newer_than_9(github):
    github.publish("v1.10.0")
    assert updates.check(current="1.9.0", url=updates.LATEST_URL).version == "1.10.0"


def test_drafts_and_releases_without_the_installer_are_not_offered(github):
    github.publish("v2.0.0", draft=True)
    assert updates.check(current="1.0.0", url=updates.LATEST_URL) is None
    github.publish("v2.0.0", assets=("something-else.zip",))
    assert updates.check(current="1.0.0", url=updates.LATEST_URL) is None


def test_download_checks_the_checksum(github, tmp_path):
    github.publish("v1.1.0")
    update = updates.check(current="1.0.0", url=updates.LATEST_URL)
    seen = []
    path = updates.download(update, tmp_path, progress=lambda done, total: seen.append((done, total)))
    assert path.read_bytes() == INSTALLER_BYTES and path.name == "Rflow-Setup-1.1.0.exe"
    assert seen[-1] == (len(INSTALLER_BYTES), len(INSTALLER_BYTES))


def test_a_damaged_download_is_refused_and_removed(github, tmp_path):
    github.publish("v1.1.0")
    github.checksum = "0" * 64 + "  Rflow-Setup.exe\n"
    update = updates.check(current="1.0.0", url=updates.LATEST_URL)
    with pytest.raises(UpdateError, match="checksum"):
        updates.download(update, tmp_path)
    assert list(tmp_path.iterdir()) == []  # nothing left behind to be run by mistake


def test_downloads_from_other_servers_are_refused(github, tmp_path, monkeypatch):
    github.publish("v1.1.0")
    update = updates.check(current="1.0.0", url=updates.LATEST_URL)
    monkeypatch.setattr(updates, "ALLOWED_HOSTS", {"github.com"})
    with pytest.raises(UpdateError, match="unexpected download location"):
        updates.download(update, tmp_path)


def test_a_stalled_first_connection_is_retried(github, monkeypatch):
    # On some laptops a program's first connection hangs until the timeout; the next attempt gets through at once.
    github.publish("v1.1.0")
    real_urlopen, calls = updates.urllib.request.urlopen, []

    def stalls_once(request, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise updates.urllib.error.URLError(TimeoutError("timed out"))
        return real_urlopen(request, timeout=timeout)

    monkeypatch.setattr(updates.urllib.request, "urlopen", stalls_once)
    assert updates.check(current="1.0.0", url=updates.LATEST_URL).version == "1.1.0"
    assert calls == [updates.ATTEMPT_TIMEOUT] * 2


def test_an_answer_from_github_is_not_retried(github, monkeypatch):
    calls = []

    def rate_limited(request, timeout):
        calls.append(1)
        raise updates.urllib.error.HTTPError(request.full_url, 403, "rate limit exceeded", {}, None)

    monkeypatch.setattr(updates.urllib.request, "urlopen", rate_limited)
    with pytest.raises(UpdateError, match="GitHub answered 403"):
        updates.check(current="1.0.0", url=updates.LATEST_URL)
    assert calls == [1]


def test_github_unreachable_is_a_readable_error(monkeypatch):
    with pytest.raises(UpdateError, match="could not reach GitHub"):
        updates.check(current="1.0.0", url="http://127.0.0.1:9/releases/latest")


def test_install_starts_the_installer_silently_and_asks_it_to_restart_rflow(monkeypatch, tmp_path):
    started = []
    monkeypatch.setattr(updates.subprocess, "Popen", lambda args, **kwargs: started.append(args))
    (tmp_path / "install.log").write_text("an earlier try", encoding="utf-8")
    updates.install(tmp_path / "Rflow-Setup-1.1.0.exe")
    assert started == [[str(tmp_path / "Rflow-Setup-1.1.0.exe"), "/SILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/UPDATE=1",
                        f"/LOG={tmp_path / 'install.log'}"]]
    assert not (tmp_path / "install.log").exists()  # this try's log only


ROLLED_BACK = """2026-10-26 12:08:07.068   RestartManager found an application using one of our files: Visual Studio Code
2026-10-26 12:08:07.068   RestartManager found an application using one of our files: Visual Studio Code
2026-10-26 12:08:07.068   RestartManager found an application using one of our files: Console Window and PTY Host (Open Source)
2026-10-26 12:08:07.068   RestartManager found an application using one of our files: Windows PowerShell
2026-10-26 12:08:07.290   Some applications could not be shut down.
2026-10-26 12:08:07.290   User canceled the installation process.
2026-10-26 12:08:07.290   Rolling back changes.
"""


def test_an_update_that_rolled_back_says_which_programs_held_rflows_files(tmp_path):
    assert updates.unfinished(tmp_path, current="2.6.2") == ""  # no update tried
    (tmp_path / "Rflow-Setup-2.6.3.exe").write_bytes(b"")
    assert updates.unfinished(tmp_path, current="2.6.2") == ""  # no log: an installer before 2.6.3 started it
    (tmp_path / "install.log").write_text(ROLLED_BACK, encoding="utf-8")
    assert updates.unfinished(tmp_path, current="2.6.2") == (
        "Rflow 2.6.3 didn't install: Visual Studio Code and Windows PowerShell were using Rflow's files. "
        "Close them, then click Update now.")
    assert updates.unfinished(tmp_path, current="2.6.3") == ""  # it did install: this is the new version
    (tmp_path / "install.log").write_text(ROLLED_BACK.replace("Windows PowerShell", "Visual Studio Code"), encoding="utf-8")
    assert updates.unfinished(tmp_path, current="2.6.2").endswith(
        "Visual Studio Code was using Rflow's files. Close it, then click Update now.")
    (tmp_path / "install.log").write_text("Setup failed for another reason.\n", encoding="utf-8")
    assert updates.unfinished(tmp_path, current="2.6.2") == "Rflow 2.6.3 didn't install. Click Update now to try again."
    (tmp_path / "install.log").write_text("Installation process succeeded.\n", encoding="utf-8")
    assert updates.unfinished(tmp_path, current="2.6.2") == ""
