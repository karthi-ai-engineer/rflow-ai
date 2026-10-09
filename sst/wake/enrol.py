"""Setting the wake word up: the user types a keyword and says it 10 times (prompts()); the setup keeps the spelling the
spotter heard most and refuses a keyword heard fewer than 8 times, then makes the voice print from the 10 recordings.

The spike: a threshold fitted to the 10 tries over-tightened (92.5% -> 71.7%) or fell to the floor, so the threshold
stays 0.25 and the tries choose the spelling and pass or fail the keyword. Each spelling has its own spotter: spellings
listened for together compete (all six of "Rflow"'s at once: 66-70%, "our flow" alone 76%).
"""
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from sst.wake.spotter import RATE, Spotter
from sst.wake.voiceprint import VoicePrint
from sst.wake.words import check_keyword, prompts, spellings

TRIES = 10
PASS = 8
LEAD, TAIL = 0.5, 1.0  # seconds of silence around a try: the spotter needs the end of the keyword followed by quiet

log = logging.getLogger(__name__)


@dataclass
class WakeSetup:
    """What the setup found, saved in the profile: numbers only, never audio."""
    keyword: str  # as the user typed it
    spelling: str  # what the spotter listens for
    voice: VoicePrint | None  # None: no voice check
    heard: int  # of the 10 tries, how many the spotter heard with this spelling

    def to_json(self) -> dict:
        return {"keyword": self.keyword, "spelling": self.spelling, "heard": self.heard,
                "voice": self.voice.to_json() if self.voice else None}

    @classmethod
    def from_json(cls, data: dict) -> "WakeSetup":
        voice = data.get("voice")
        return cls(str(data["keyword"]), str(data["spelling"]), VoicePrint.from_json(voice) if voice else None,
                   int(data.get("heard", 0)))


class Enrollment:
    """The 10 tries' logic; the window records each try and add()s it."""

    def __init__(self, keyword: str, spotter_factory: Callable[[dict[str, str]], Spotter] = Spotter,
                 voice_factory: Callable[[Sequence[np.ndarray]], VoicePrint] | None = VoicePrint.from_takes):
        problem = check_keyword(keyword)
        if problem:
            raise ValueError(problem)
        self.keyword = keyword.strip()
        self.spellings = spellings(self.keyword)
        self.prompts: list[str] = prompts(self.keyword)
        self._spotters = [spotter_factory({spelling: spelling}) for spelling in self.spellings]
        self._voice_factory = voice_factory
        self._counts = [0] * len(self.spellings)
        self._takes: list[np.ndarray] = []

    @property
    def tries(self) -> int:
        return len(self._takes)

    @property
    def heard(self) -> int:
        """The most tries one spelling caught: what has to reach PASS."""
        return max(self._counts)

    @property
    def done(self) -> bool:
        return self.tries >= TRIES

    def add(self, audio: np.ndarray) -> bool:
        """One try's recording (16 kHz float32): was the keyword heard, by any of its spellings?"""
        if self.done:
            raise RuntimeError("all the tries are done")
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        padded = np.concatenate([np.zeros(int(LEAD * RATE), np.float32), audio, np.zeros(int(TAIL * RATE), np.float32)])
        heard = False
        for i, spotter in enumerate(self._spotters):
            spotter.reset()
            if spotter.feed(padded) is not None:
                self._counts[i] += 1
                heard = True
        self._takes.append(audio)
        log.info("Wake word setup: try %d of %d %s (spellings: %s)", self.tries, TRIES, "heard" if heard else "missed",
                 self._counts)
        return heard

    def result(self) -> WakeSetup | None:
        """The setup: the spelling heard most (the likelier one when two tie); None when the tries aren't done or the
        keyword was heard fewer than PASS times (it would be missed too often: choose another)."""
        if not self.done or self.heard < PASS:
            return None
        best = self._counts.index(self.heard)
        voice = self._voice_factory(self._takes) if self._voice_factory else None
        return WakeSetup(self.keyword, self.spellings[best], voice, self.heard)
