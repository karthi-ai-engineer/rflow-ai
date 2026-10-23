"""Setting the wake word up: the user types a keyword and says it until it has been heard 10 times (prompts()). A try
not heard isn't used up: the same prompt is said again (the owner's wish: "if it fails why does it go to the next? it
can stay on the same"), and after MISSES in a row the window offers advice and another keyword. The setup keeps the
spelling the spotter heard most, refuses it when it heard fewer than PASS of the takes, and makes the voice print from
the tries heard.

The spike: a threshold fitted to the tries over-tightened (92.5% -> 71.7%) or fell to the floor, so the threshold stays
0.25 and the tries choose the spelling and pass or fail the keyword. Each spelling has its own spotter: spellings
listened for together compete (all six of "Rflow"'s at once: 66-70%, "our flow" alone 76%).

The spellings tried: the keyword's own (spellings()), those that sound alike (sounds_like()), and what the spotter's
model heard in a try it missed, when that could stand for the keyword (same_sound(): "CARTHY" for "karthi"). A name
is heard as the English spelling the voice sounds like, which differs from voice to voice (sst.wake.words), so the
user's own voice decides. A spelling learned from a take is scored on every take but that one.

check_hearing(): before any try, whether the spotter can hear the keyword at all, from a synthetic voice saying it in
a few sentences (the voice garbles a name said alone: Parakeet heard its "Kathy" as "Hi"); and, when it can't, which
short phrase with the keyword it hears instead.
"""
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np

from sst.wake.spotter import RATE, Spotter
from sst.wake.voiceprint import VoicePrint
from sst.wake.words import MAX_WORDS, check_keyword, prompts, same_sound, sounds_like, spellings

TRIES = 10
PASS = 8
MISSES = 3  # misses in a row on one try before the window offers advice and another keyword
MAX_SPELLINGS = 8  # the keyword's own and those that sound alike
MAX_LEARNED = 3  # spellings learned from what the model heard in the tries missed
LEAD, TAIL = 0.5, 1.0  # seconds of silence around a try: the spotter needs the end of the keyword followed by quiet
# The loudest 100 ms of a take, dBFS. Measured (Zira, a -70 dBFS noise floor): "okay type" and "computer" are heard down
# to -55, "hey karthi" only to -45; speech near a microphone is about -35 to -15.
QUIET_DB = -50.0
# check_hearing: the sentences the synthetic voice says the keyword in, at these speeds; heard in half of them = heard
CARRIERS = ("I asked {} about it.", "So {}, what now?", "Well, {} said yes.")
SPEEDS = (1.0, 0.9, 1.1)
PHRASES = ("hey {}", "okay {}", "hello {}")  # with the name last: a word after it was heard as one ("carth is start")
GREETINGS = {"hey", "okay", "ok", "hello", "hi"}
FALLBACK = "okay type"

log = logging.getLogger(__name__)


def level_db(audio: np.ndarray) -> float:
    """How loud a take is: its loudest 100 ms (the speech, not the silence around it), in dBFS."""
    audio = np.asarray(audio, dtype=np.float64).reshape(-1)
    step = RATE // 10
    frames = [audio[i:i + step] for i in range(0, len(audio), step) if len(audio[i:i + step])] or [np.zeros(1)]
    return max(10 * float(np.log10(np.mean(f ** 2) + 1e-12)) for f in frames)


@dataclass
class WakeSetup:
    """What the setup found, saved in the profile: numbers only, never audio."""
    keyword: str  # as the user typed it
    spelling: str  # what the spotter listens for
    voice: VoicePrint | None  # None: no voice check
    heard: int  # how many of the takes the spotter heard with this spelling
    said: int = 0  # how many takes were said (0: saved before misses stopped counting as tries, when it was 10)

    def to_json(self) -> dict:
        return {"keyword": self.keyword, "spelling": self.spelling, "heard": self.heard, "said": self.said,
                "voice": self.voice.to_json() if self.voice else None}

    @classmethod
    def from_json(cls, data: dict) -> "WakeSetup":
        voice = data.get("voice")
        return cls(str(data["keyword"]), str(data["spelling"]), VoicePrint.from_json(voice) if voice else None,
                   int(data.get("heard", 0)), int(data.get("said", 0)))


class Enrollment:
    """The tries' logic; the window records each take and add()s it. transcribe(audio) -> the words the spotter's model
    hears (spotter.Transcriber), to learn spellings from; None: only the keyword's own and those that sound alike."""

    def __init__(self, keyword: str, spotter_factory: Callable[[dict[str, str]], Spotter] = Spotter,
                 voice_factory: Callable[[Sequence[np.ndarray]], VoicePrint] | None = VoicePrint.from_takes,
                 transcribe: Callable[[np.ndarray], str] | None = None):
        problem = check_keyword(keyword)
        if problem:
            raise ValueError(problem)
        self.keyword = keyword.strip()
        self.prompts: list[str] = prompts(self.keyword)
        self._factory, self._voice_factory, self._transcribe = spotter_factory, voice_factory, transcribe
        self.spellings: list[str] = []
        self._spotters: list = []
        self._hits: list[list[bool]] = []  # for each spelling, each take: heard?
        self._learned: dict[int, int] = {}  # a learned spelling -> the take it was learned from
        self._takes: list[np.ndarray] = []  # every take, padded (memory only: a learned spelling is scored on them)
        self._counted: list[int] = []  # the takes heard: the tries done
        self.missed = 0  # takes missed in a row on this try
        own = spellings(self.keyword)
        for spelling in dict.fromkeys([*own, *sounds_like(self.keyword)]):
            if len(self.spellings) >= MAX_SPELLINGS:
                break
            try:
                self._add_spelling(spelling)
            except ValueError:  # pieces the spotter doesn't know: only a problem for the keyword's own spelling
                if spelling == own[0]:
                    raise

    @property
    def tries(self) -> int:
        """Tries done: the takes heard."""
        return len(self._counted)

    @property
    def said(self) -> int:
        return len(self._takes)

    @property
    def counts(self) -> list[int]:
        return [sum(row) for row in self._hits]

    @property
    def heard(self) -> int:
        """The most takes one spelling caught: what has to reach PASS."""
        return max(self.counts, default=0)

    @property
    def best(self) -> str:
        """The spelling heard most (the likelier one, listed first, when two tie)."""
        counts = self.counts
        return self.spellings[counts.index(max(counts))]

    @property
    def done(self) -> bool:
        return self.tries >= TRIES

    def _add_spelling(self, spelling: str) -> None:
        spotter = self._factory({spelling: spelling})  # ValueError: the spotter can't listen for it
        self.spellings.append(spelling)
        self._spotters.append(spotter)
        self._hits.append([self._hear(spotter, take) for take in self._takes])

    @staticmethod
    def _hear(spotter, padded: np.ndarray) -> bool:
        spotter.reset()
        return spotter.feed(padded) is not None

    def add(self, audio: np.ndarray) -> bool:
        """One take (16 kHz float32): was the keyword heard, by any of its spellings? A take heard is a try done; one
        missed is said again (missed counts them)."""
        if self.done:
            raise RuntimeError("all the tries are done")
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        padded = np.concatenate([np.zeros(int(LEAD * RATE), np.float32), audio, np.zeros(int(TAIL * RATE), np.float32)])
        index = len(self._takes)
        self._takes.append(padded)
        for spotter, row in zip(self._spotters, self._hits, strict=True):
            row.append(self._hear(spotter, padded))
        heard = any(row[index] for row in self._hits)
        learned = "" if heard else self._learn(padded, index)
        if heard:
            self._counted.append(index)
            self.missed = 0
        else:
            self.missed += 1
        log.info("Wake word setup: take %d %s, try %d of %d (level %.0f dBFS; spellings heard: %s%s)", index + 1,
                 "heard" if heard else "missed", self.tries if heard else self.tries + 1, TRIES, level_db(audio),
                 self.counts, "; learned one" if learned else "")
        return heard

    def _learn(self, padded: np.ndarray, index: int) -> str:
        """What the model heard in a take missed, as a new spelling when it could stand for the keyword ("" if not)."""
        if self._transcribe is None or len(self._learned) >= MAX_LEARNED:
            return ""
        try:
            words = " ".join(self._transcribe(padded).lower().split())
        except Exception as e:  # learning is a bonus: the tries go on without it
            log.warning("Wake word setup: the take couldn't be transcribed (%s)", type(e).__name__)
            return ""
        if not words or words in self.spellings or len(words.split()) > MAX_WORDS or not same_sound(words, self.keyword):
            return ""
        try:
            self._add_spelling(words)
        except ValueError:
            return ""
        self._learned[len(self.spellings) - 1] = index
        self._hits[-1][index] = False  # heard in the take it came from by construction: that one doesn't count
        return words

    def result(self) -> WakeSetup | None:
        """The setup: the spelling heard most; None when the tries aren't done or no spelling was heard in PASS takes
        (they were heard as different words: it would be missed too often)."""
        if not self.done or self.heard < PASS:
            return None
        takes = [self._takes[i][int(LEAD * RATE):len(self._takes[i]) - int(TAIL * RATE)] for i in self._counted]
        voice = self._voice_factory(takes) if self._voice_factory else None
        return WakeSetup(self.keyword, self.best, voice, self.heard, self.said)


# ---------------------------------------------------------------- can it be heard at all?

@dataclass
class Hearing:
    """How well the spotter heard a phrase said by the synthetic voice."""
    phrase: str
    spelling: str  # the spelling heard most
    heard: int
    said: int

    @property
    def ok(self) -> bool:
        return self.said > 0 and 2 * self.heard >= self.said


def hear(phrase: str, say: Callable[[str, float], np.ndarray], spotter_factory: Callable = Spotter,
         transcribe: Callable[[np.ndarray], str] | None = None) -> Hearing:
    """`phrase` said by say(text, speed) (16 kHz audio) in each of CARRIERS at SPEEDS, listened for as the setup does;
    it stops once the answer can't change (heard in half of all the takes, or can't be any more)."""
    enrollment = Enrollment(phrase, spotter_factory, None, transcribe)
    said, total = spellings(phrase)[0], len(CARRIERS) * len(SPEEDS)  # "Rflow" said as "our flow", not letter by letter
    for speed, carrier in product(SPEEDS, CARRIERS):
        audio = np.asarray(say(carrier.format(said), speed), dtype=np.float32).reshape(-1)
        enrollment.add(audio / (float(np.abs(audio).max()) + 1e-9) * 0.3)
        if 2 * enrollment.heard >= total or 2 * (enrollment.heard + total - enrollment.said) < total:
            break
    return Hearing(phrase, enrollment.best, enrollment.heard, enrollment.said)


def check_hearing(keyword: str, say: Callable[[str, float], np.ndarray], spotter_factory: Callable = Spotter,
                  transcribe: Callable[[np.ndarray], str] | None = None) -> tuple[Hearing, str | None]:
    """Is `keyword` heard (Hearing.ok), and if not, a phrase to suggest: the best heard of PHRASES with the keyword in
    it, else FALLBACK. Takes a few seconds (the voice and the spotter run ~20 times): call it on a thread."""
    hearing = hear(keyword, say, spotter_factory, transcribe)
    log.info("Wake word: the keyword was heard in %d of %d synthetic takes", hearing.heard, hearing.said)
    if hearing.ok:
        return hearing, None
    best, words = None, keyword.split()
    name = " ".join(words[1:] if len(words) > 1 and words[0].lower() in GREETINGS else words)  # not "hey okay karthi"
    for pattern in PHRASES:
        phrase = pattern.format(name)
        if phrase.lower() == " ".join(words).lower() or check_keyword(phrase) is not None:
            continue
        other = hear(phrase, say, spotter_factory, transcribe)
        if other.ok and (best is None or other.heard * best.said > best.heard * other.said):
            best = other
    return hearing, best.phrase if best is not None else FALLBACK
