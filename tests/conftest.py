"""Shared test setup: no test sees the speech models on the machine running it."""
import pytest

from sst import downloads
from sst.engines import parakeet


@pytest.fixture(autouse=True)
def no_downloaded_models(monkeypatch, tmp_path_factory):
    """An empty download folder for every test, so results don't depend on what this laptop has downloaded."""
    monkeypatch.setattr(downloads, "DOWNLOADS_DIR", tmp_path_factory.mktemp("downloads"))


@pytest.fixture(autouse=True)
def parakeet_on_disk(monkeypatch, tmp_path_factory):
    """Parakeet counts as being next to the program (as after an update from Rflow 1.4), whether or not this machine
    has the real model: no test loads it."""
    folder = tmp_path_factory.mktemp("parakeet")
    (folder / "encoder.int8.onnx").write_bytes(b"")
    monkeypatch.setattr(parakeet, "MODEL_DIR", folder)


@pytest.fixture
def no_parakeet(monkeypatch, tmp_path_factory):
    """A new install: Parakeet isn't downloaded."""
    monkeypatch.setattr(parakeet, "MODEL_DIR", tmp_path_factory.mktemp("no-parakeet"))


@pytest.fixture
def whisper_downloaded(monkeypatch):
    """Whisper turbo counts as downloaded (its real files are 1.6 GB)."""
    monkeypatch.setattr(downloads.Download, "installed", lambda self, root=None: True)


@pytest.fixture
def wake(monkeypatch):
    """sst.wake as it is, its models faked: a spotter that hears any sound louder than 0.05, and a voice print made of
    numbers. The listeners the app makes are kept (WakeListener.made), with the end phrases they were given and how
    often the app ended a turn; the tests call their callbacks (on_wake, on_pause, on_end) themselves."""
    import threading

    import numpy as np

    from sst import wake as api

    class Spotter:
        def __init__(self, phrases):
            self.name = next(iter(phrases))

        def reset(self):
            pass

        def feed(self, frame):
            return self.name if len(frame) and float(np.abs(frame).max()) > 0.05 else None

    class WakeListener(api.WakeListener):
        made: list = []

        def __init__(self, setup, end_phrases=api.END_PHRASES, *, voice_check=True):
            super().__init__(setup, end_phrases, voice_check=voice_check, spotter_factory=Spotter)
            self.end_phrases, self.ended, self.thread = list(end_phrases), 0, threading.current_thread().name
            WakeListener.made.append(self)

        def end(self):
            self.ended += 1
            super().end()

    class Enrollment(api.Enrollment):
        def __init__(self, keyword, transcribe=None):  # the app's transcriber is a real model: not here
            super().__init__(keyword, spotter_factory=Spotter, voice_factory=lambda takes: api.VoicePrint([0.1, 0.2, 0.3]))

    monkeypatch.setattr(api, "WakeListener", WakeListener)
    monkeypatch.setattr(api, "Enrollment", Enrollment)
    return api
