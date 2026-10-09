"""A stand-in for sst.wake (built in its own part, #164) with the names of the contract between the two parts, so the
app's tests (#165) run without its models: no spotter, no voice print, no audio. The tests put it in sys.modules as
"sst.wake" (install()) and drive the listener's callbacks by hand."""
import re
import sys
import types
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from sst.downloads import Download, ModelFile

THRESHOLD, TRIES, PASS, VOICE_THRESHOLD = 0.25, 10, 8, 0.2
END_PHRASES = ("stop it",)
SAFETY_SILENCE = 20.0
KWS = Download("wake-kws-test", "https://huggingface.co/test/kws/", (ModelFile("kws.onnx", 5_500_000, "0" * 64),))
VOICE = Download("wake-voice-test", "https://huggingface.co/test/voice/", (ModelFile("voice.onnx", 40_300_000, "1" * 64),))
COMMON = {"hey", "okay", "yes", "no", "done", "hello", "stop"}


def prompts(keyword: str) -> list[str]:
    ways = ["normally", "quietly", "loudly", "fast", "slowly", "close to the microphone", "leaning back",
            "in a sentence", "normally", "normally"]
    return [f"Say “{keyword}” {way}" for way in ways]


def check_keyword(text: str) -> str | None:
    words = text.split()
    if not words:
        return "Type a keyword."
    if not re.fullmatch(r"[A-Za-z' ]+", text):
        return "Use English letters."
    if len(words) == 1 and words[0].casefold() in COMMON:
        return f"“{text}” is said too often: it would start dictations by itself."
    if len(words) == 1 and len(words[0]) < 5:
        return "One short word is easily missed: use two or three syllables."
    return None


def spellings(text: str) -> list[str]:
    return [text.casefold()]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.casefold())


def strip_keyword(text: str, keyword: str, spelling: str) -> str:
    for said in (keyword, spelling):
        n = len(_words(said))
        if n and _words(text)[:n] == _words(said):
            matches = list(re.finditer(r"[A-Za-z']+", text))
            return text[matches[n - 1].end():].lstrip(" ,.!?").strip()
    return text


def strip_end(text: str, phrases: Sequence[str]) -> str:
    for phrase in phrases:
        n = len(_words(phrase))
        if n and _words(text)[-n:] == _words(phrase):
            matches = list(re.finditer(r"[A-Za-z']+", text))
            return text[:matches[-n].start()].rstrip(" ,").strip()
    return text


class VoicePrint:
    def __init__(self, vector):
        self.vector = list(vector)

    @classmethod
    def from_takes(cls, takes):
        return cls([0.1, 0.2, 0.3])

    def score(self, sound) -> float:
        return 0.9

    def to_json(self) -> dict:
        return {"vector": self.vector}

    @classmethod
    def from_json(cls, data):
        return cls(data["vector"])


@dataclass
class WakeSetup:
    keyword: str
    spelling: str
    voice: VoicePrint | None
    heard: int

    def to_json(self) -> dict:
        return {"keyword": self.keyword, "spelling": self.spelling, "heard": self.heard,
                "voice": self.voice.to_json() if self.voice else None}

    @classmethod
    def from_json(cls, data):
        return cls(str(data["keyword"]), str(data["spelling"]),
                   VoicePrint.from_json(data["voice"]) if data.get("voice") else None, int(data["heard"]))


class Enrollment:
    """Heard when the try is louder than 0.05 (a test says it by the audio it gives)."""

    def __init__(self, keyword, spotter_factory=None, voice_factory=None):
        self.keyword, self.prompts = keyword, prompts(keyword)
        self.heard = self.tries = 0
        self.takes: list[np.ndarray] = []

    @property
    def done(self) -> bool:
        return self.tries >= TRIES

    def add(self, sound) -> bool:
        self.tries += 1
        self.takes.append(sound)
        heard = float(np.abs(sound).max()) > 0.05
        self.heard += heard
        return heard

    def result(self):
        if self.heard < PASS:
            return None
        return WakeSetup(self.keyword, spellings(self.keyword)[0], VoicePrint.from_takes(self.takes), self.heard)


class WakeListener:
    """Keeps what it's fed and told; a test calls on_wake / on_pause / on_end itself."""
    made: list = []  # every listener made, newest last

    def __init__(self, setup, end_phrases=END_PHRASES, *, voice_check=True, spotter_factory=None):
        self.setup, self.end_phrases, self.voice_check = setup, list(end_phrases), voice_check
        self.on_wake = lambda: None
        self.on_pause = lambda: None
        self.on_end = lambda keep: None
        self.frames: list[tuple[int, float]] = []
        self.ended = 0
        self.active = False
        WakeListener.made.append(self)

    def feed(self, frame, now: float) -> None:
        self.frames.append((len(frame), now))

    def end(self) -> None:
        self.ended += 1
        self.active = False


def install(monkeypatch) -> types.ModuleType:
    """Make this module sst.wake for one test."""
    module = sys.modules[__name__]
    WakeListener.made = []
    monkeypatch.setitem(sys.modules, "sst.wake", module)
    return module
