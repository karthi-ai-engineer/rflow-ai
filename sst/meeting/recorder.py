"""The meeting's sound, saved on this laptop: others.wav (what the laptop plays: the other people) and you.wav (the
microphone), 16 kHz mono 16-bit, as live translation captures it (sst.live.wasapi.Capture).

Each frame is written as it comes and the WAV header patched and flushed with it (wave's writeframes patches the sizes
when they change), so a crash or a power cut of the app (not of the disk) still leaves files any player opens, with
everything heard until then. Two tracks rather than one mix: the minutes tell "You" from the others by the track.
"""
import logging
import threading
import wave
from collections.abc import Callable
from pathlib import Path

from sst.live.contracts import RATE
from sst.live.wasapi import Capture

log = logging.getLogger(__name__)

OTHERS = "others.wav"  # Capture.speakers(): everyone else in the meeting
YOU = "you.wav"  # Capture.microphone(): the user
ROOM = "room.wav"  # Capture.microphone() in a meeting in person: everyone in the room, the user among them
SAMPLE_BYTES = 2  # PCM16


class _Track:
    """One WAV file, written from its capture's thread."""

    def __init__(self, path: Path):
        self.path = path
        self._file = open(path, "wb")  # noqa: SIM115 - kept open for the whole meeting; wave doesn't close a file it was given
        self._wav: wave.Wave_write | None = wave.open(self._file, "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(SAMPLE_BYTES)
        self._wav.setframerate(RATE)
        self.samples = 0
        self._lock = threading.Lock()

    def write(self, frame: bytes) -> None:
        with self._lock:
            if self._wav is None:
                return
            try:
                self._wav.writeframes(frame)  # the header's sizes patched each time: playable whenever it stops
                self._file.flush()
                self.samples += len(frame) // SAMPLE_BYTES
            except OSError as e:  # a full disk: keep what's written, say so once
                log.warning("Meeting recording: couldn't write %s: %s", self.path.name, e)
                self._close()

    def close(self) -> None:
        with self._lock:
            self._close()

    def _close(self) -> None:
        if self._wav is not None:
            try:
                self._wav.close()
            finally:
                self._wav = None
                self._file.close()


class Recorder:
    """start() opens both captures and writes their frames into `folder`; stop() closes them. A capture that can't
    open is left out (logged, in `problems`) while the other records; when neither opens, start() raises the first
    error. `seconds`: how long the longer track is; `files`: the tracks being recorded. In person (`room`), only the
    microphone, into room.wav: nobody is in the laptop's sound."""

    def __init__(self, folder: Path, *, speakers: Callable[[], Capture] = Capture.speakers,
                 microphone: Callable[[], Capture] = Capture.microphone, room: bool = False):
        self.folder = Path(folder)
        self._factories = ((ROOM, microphone),) if room else ((OTHERS, speakers), (YOU, microphone))
        self._running: list[tuple[_Track, Capture]] = []
        self.problems: list[str] = []

    @property
    def files(self) -> list[Path]:
        return [track.path for track, _ in self._running]

    @property
    def seconds(self) -> float:
        return max((track.samples for track, _ in self._running), default=0) / RATE

    def start(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        errors: list[Exception] = []
        for name, factory in self._factories:
            track = _Track(self.folder / name)
            try:
                capture = factory()
                capture.start(track.write)
            except Exception as e:
                log.warning("Meeting recording: %s left out: %s", name, e)
                self.problems.append(f"{name}: {e}")
                errors.append(e)
                track.close()
                track.path.unlink(missing_ok=True)  # an empty file would look like a silent track
                continue
            self._running.append((track, capture))
        if not self._running:
            raise errors[0]

    def stop(self) -> None:
        for track, capture in self._running:
            try:
                capture.stop()
            except Exception as e:  # the file is closed whatever the capture does
                log.warning("Meeting recording: stopping %s: %s", track.path.name, e)
            track.close()
