"""Downloading a speech model, against a fake model server on this machine: files checked, resumed, retried."""
import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from sst import downloads
from sst.downloads import Download, ModelFile

FILES = {"config.json": b'{"model": "tiny"}', "model.bin": bytes(range(256)) * 4000}  # ~1 MB


class FakeModelServer:
    """Serves FILES with Range support. `break_after[name]` makes that file's next answers stop early (a broken
    connection)."""

    def __init__(self):
        self.requests, self.break_after, self.status = [], {}, {}
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                name = self.path.rsplit("/", 1)[-1]
                fake.requests.append((name, self.headers.get("Range")))
                if name in fake.status:
                    self.send_error(fake.status[name])
                    return
                data = FILES[name]
                start = int(self.headers["Range"].split("=")[1].split("-")[0]) if self.headers.get("Range") else 0
                self.send_response(206 if start else 200)
                self.send_header("Content-Length", str(len(data) - start))
                self.end_headers()
                body = data[start:]
                if fake.break_after.get(name):  # send part of it, then drop the connection
                    body = body[:fake.break_after[name].pop(0)]
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/repo/resolve/abc/"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def model(self, damaged: str = "") -> Download:
        files = tuple(ModelFile(name, len(data), hashlib.sha256(data + (b"x" if name == damaged else b"")).hexdigest())
                      for name, data in FILES.items())
        return Download("tiny-model", self.url, files)


@pytest.fixture
def server(monkeypatch):
    monkeypatch.setattr(downloads, "allowed", lambda url: url.startswith("http://127.0.0.1"))
    monkeypatch.setattr(downloads, "CHUNK", 64 * 1024)
    monkeypatch.setattr(downloads, "ATTEMPT_TIMEOUT", 2.0)
    fake = FakeModelServer()
    yield fake
    fake.server.shutdown()


def test_a_model_is_downloaded_checked_and_installed(server, tmp_path):
    model = server.model()
    seen = []
    assert not model.installed(tmp_path)
    folder = downloads.download(model, lambda done, total: seen.append((done, total)), root=tmp_path)
    assert model.installed(tmp_path) and (folder / "model.bin").read_bytes() == FILES["model.bin"]
    assert seen[-1] == (model.size, model.size)
    assert not list(folder.glob("*.part"))
    downloads.download(model, root=tmp_path)  # already there: nothing fetched again
    assert len(server.requests) == 2


def test_a_broken_connection_resumes_where_it_stopped(server, tmp_path):
    server.break_after["model.bin"] = [300_000]  # its first answer stops after 300 kB
    model = server.model()
    downloads.download(model, root=tmp_path)
    assert model.installed(tmp_path)
    ranges = [r for name, r in server.requests if name == "model.bin"]
    assert ranges == [None, "bytes=300000-"]


def test_a_damaged_file_is_not_kept(server, tmp_path):
    model = server.model(damaged="model.bin")
    with pytest.raises(downloads.DownloadError, match="checksum"):
        downloads.download(model, root=tmp_path)
    assert not model.installed(tmp_path) and not (model.path(tmp_path) / "model.bin").exists()
    assert not (model.path(tmp_path) / "model.bin.part").exists()


def test_cancelling_keeps_what_was_downloaded_for_next_time(server, tmp_path):
    model = server.model()
    calls = []
    with pytest.raises(downloads.Cancelled):
        downloads.download(model, cancelled=lambda: calls.append(1) or len(calls) > 3, root=tmp_path)
    part = model.path(tmp_path) / "model.bin.part"
    kept = part.stat().st_size
    assert 0 < kept < len(FILES["model.bin"])
    downloads.download(model, root=tmp_path)  # goes on from there
    assert model.installed(tmp_path)
    assert [r for name, r in server.requests if name == "model.bin"][-1] == f"bytes={kept}-"


def test_an_answer_from_the_server_is_not_retried(server, tmp_path):
    server.status["config.json"] = 404
    with pytest.raises(downloads.DownloadError, match="404"):
        downloads.download(server.model(), root=tmp_path)
    assert len(server.requests) == 1


def test_only_hugging_face_servers_and_k2_fsas_releases_are_accepted():
    assert downloads.allowed("https://huggingface.co/dropbox-dash/x/resolve/abc/model.bin")
    assert downloads.allowed("https://us.aws.cdn.hf.co/xet-bridge/abc")
    assert downloads.allowed("https://cdn-lfs.huggingface.co/repos/abc")
    assert not downloads.allowed("https://example.com/model.bin")
    assert not downloads.allowed("https://huggingface.co.evil.example/model.bin")
    assert downloads.allowed("https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/x.tar.bz2")
    assert downloads.allowed("https://release-assets.githubusercontent.com/github-production-release-asset/1/2")
    assert not downloads.allowed("https://github.com/someone/sherpa-onnx/releases/download/kws-models/x.tar.bz2")
    assert not downloads.allowed("https://github.com/k2-fsa/sherpa-onnx/archive/main.zip")
    assert not downloads.allowed("http://release-assets.githubusercontent.com/x")


def _archive(top: str, members: dict[str, bytes]) -> bytes:
    import io
    import tarfile
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:bz2") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name if name.startswith(("..", "/")) else f"{top}/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return out.getvalue()


def test_a_model_published_as_an_archive_keeps_only_its_own_files_checked(server, tmp_path):
    # Only the pinned files are read out, by their own names; nothing else in the archive (a path climbing out of the
    # folder, a test recording) is written anywhere, and the archive is deleted once they're out.
    packed = _archive("kws", {"a.onnx": b"model" * 100, "tokens.txt": b"x 1", "test.wav": b"RIFF", "../evil.txt": b"!"})
    FILES["kws.tar.bz2"] = packed
    try:
        files = (ModelFile("a.onnx", 500, hashlib.sha256(b"model" * 100).hexdigest()),
                 ModelFile("tokens.txt", 3, hashlib.sha256(b"x 1").hexdigest()))
        model = Download("kws", server.url, files, ModelFile("kws.tar.bz2", len(packed), hashlib.sha256(packed).hexdigest()))
        seen = []
        folder = downloads.download(model, lambda done, total: seen.append((done, total)), root=tmp_path)
        assert model.installed(tmp_path) and (folder / "a.onnx").read_bytes() == b"model" * 100
        assert sorted(p.name for p in folder.iterdir()) == ["a.onnx", "complete.json", "tokens.txt"]
        assert not (tmp_path / "evil.txt").exists() and seen[-1] == (len(packed), len(packed))
        bad = Download("bad", server.url, (ModelFile("a.onnx", 500, "0" * 64),), model.archive)
        with pytest.raises(downloads.DownloadError, match="damaged"):
            downloads.download(bad, root=tmp_path)
        assert not bad.installed(tmp_path) and (bad.path(tmp_path) / "kws.tar.bz2").exists()  # kept for the next try
    finally:
        del FILES["kws.tar.bz2"]


def test_removing_a_model_frees_its_folder(server, tmp_path):
    model = server.model()
    downloads.download(model, root=tmp_path)
    downloads.remove(model, root=tmp_path)
    assert not model.path(tmp_path).exists() and not model.installed(tmp_path)
