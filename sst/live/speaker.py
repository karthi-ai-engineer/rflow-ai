"""Speaking live translations aloud: the engine's own voice as it streams, else the laptop's, sentence by sentence.

    the engine's voice (Gemini's, 24 kHz PCM16) of the ways spoken (`own`, any language) -> hear_audio() -> a queue
    TRANSLATION / LINE events of the ways the laptop's voice speaks (`lanes`, its language) -> whole sentences, in
        order -> said by the engine's voice already? dropped; else the laptop's voice (~0.2 s a sentence) -> the queue
    the queue -> one thread -> the player (Windows' default output): never two voices at once

Gemini Live Translate speaks its translation itself (its transcript is what the bar shows), so that's what's played,
piece by piece as it comes; the laptop's voice (sst.live.voice) only stands in for a sentence the engine's voice didn't
say: one with none of the engine's audio from OWN_RECENT seconds before it was whole to AUDIO_WAIT after (the audio
comes before its transcript). The window leans towards "said": a sentence said twice is worse than one not said when
Gemini's audio stops in the middle of a turn. Gemini's audio is played at its own pace (the speed setting is the
laptop's voice's); when it falls behind, its long pauses are shortened, and audio older than STALE is skipped.

A sentence is spoken once the translation goes on past it, when its line ends, or when it ends with a full stop and
nothing has come for SETTLE seconds: so "It costs 3." isn't said before ".5 million" arrives, and the last sentence
before a pause isn't held until the line ends (1.5 s later). An interpreter mustn't fall behind for good: with
sentences waiting it speaks faster (up to MAX_SPEED), and one older than STALE seconds is skipped while newer ones wait
(it stays on screen and in the transcript). Only translations are spoken: speech already in the voice's language is
heard as it is. Like an interpreter, it keeps the other apps lower while it runs, if given a Ducker (sst.live.ducking).

Gemini revises what it has translated: it rewrites earlier words, turns a comma into a full stop, re-sends the whole
line, and starts a new line with the end of the last. So what was said is remembered by what it says (words(): case,
spaces and punctuation aside), in its line and for the session's last RECENT sentences, and a sentence that matches one
of them (alike(), or part of one its line said) is a repeat: never said again. Only new sentences are said, in order.
"""
import logging
import os
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np

from sst.live.contracts import Kind, LiveEvent

SETTLE = 0.6  # seconds without a new piece after a full stop: that sentence is complete
STALE = 10.0  # seconds a sentence may wait while newer ones queue behind it
CATCH_UP = 0.15  # faster for each sentence waiting
MAX_SPEED = 1.6
TAIL = 0.4  # seconds after the voice ends that a microphone (the room) may still hear it
RECENT = 20  # the session's last sentences remembered: one coming again (a new line re-sends it) isn't said again
SIMILAR = 0.85  # this alike (difflib's ratio of their words()): the same sentence, a word revised
SHORT = 4  # words: a shorter sentence is the same only word for word ("I think so" isn't "I think not")
SHORT_RECENT = 10.0  # seconds a short sentence of another line counts: "Yes." may well be said again later
OWN_RECENT = 5.0  # seconds before a sentence is whole that the engine's voice counts as saying it (it comes first)
AUDIO_WAIT = 1.2  # seconds a sentence waits for the engine's voice before the laptop's says it
MERGE_S = 0.5  # seconds of the engine's voice played at once (its pieces are small)
BEHIND_S = 2.0  # seconds of the engine's voice waiting: its long pauses are shortened to catch up
PAUSE_S, PAUSE_LEVEL = 0.25, 10 ** (-45 / 20)  # a pause: below -45 dBFS; cut to PAUSE_S when behind
_END = re.compile(r"[.!?…。！？]+[\"'”’)\]]*")
_ENDS = re.compile(r"[.!?…。！？]+[\"'”’)\]]*\s*$")
_SHORT = {"mr", "mrs", "ms", "dr", "st", "vs", "etc", "e.g", "i.e", "jr", "sr", "no", "prof", "inc", "ltd", "co",
          "approx", "u.s", "u.k", "a.m", "p.m"}  # a full stop after these doesn't end the sentence
_WORDS = re.compile(r"\w+")
_NUMBERS = re.compile(r"\d+")

log = logging.getLogger(__name__)


def whole_sentences(text: str) -> tuple[list[str], int]:
    """The sentences at the start of `text` that the text goes on past, and where the last of them ends."""
    spans = _spans(text)
    return [sentence for sentence, _ in spans], spans[-1][1] if spans else 0


def _spans(text: str) -> list[tuple[str, int]]:
    """whole_sentences(), each with where it ends."""
    found, start = [], 0
    for m in _END.finditer(text):
        end = m.end()
        if end >= len(text) or not text[end].isspace():
            continue  # the text doesn't go on past it yet ("3." of "3.5"), or a dot inside a word ("e.g")
        words = text[start:m.start()].split()
        if m.group().startswith(".") and words and words[-1].lower().rstrip(".") in _SHORT:
            continue
        if sentence := text[start:end].strip():
            found.append((sentence, end))
        start = end
    return found


def _all(text: str) -> list[str]:
    """Every sentence of `text`, the unfinished last one too."""
    sentences, end = whole_sentences(text)
    return sentences + ([text[end:].strip()] if text[end:].strip() else [])


def words(sentence: str) -> str:
    """What a sentence says: its words, lower case, without punctuation."""
    return " ".join(_WORDS.findall(sentence.lower()))


def alike(a: str, b: str) -> bool:
    """Two sentences (as words()) that say the same: word for word when short, else nearly (a word revised), and never
    with other numbers ("it costs 3 million" isn't "it costs 5 million")."""
    if a == b:
        return True
    if min(a.count(" "), b.count(" ")) + 1 < SHORT or _NUMBERS.findall(a) != _NUMBERS.findall(b):
        return False
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    return matcher.real_quick_ratio() >= SIMILAR and matcher.quick_ratio() >= SIMILAR and matcher.ratio() >= SIMILAR


def shorten_pauses(samples: np.ndarray, rate: int, keep: float = PAUSE_S) -> np.ndarray:
    """`samples` with every pause (20 ms blocks below PAUSE_LEVEL) longer than `keep` cut to `keep`: the words stay
    whole, only the silence between them goes."""
    block = max(1, rate // 50)
    n = len(samples) // block
    if n < 2:
        return samples
    quiet = np.sqrt(np.mean(samples[:n * block].reshape(n, block) ** 2, axis=1)) < PAUSE_LEVEL
    most, run, kept = round(keep * 50), 0, []
    for i in range(n):
        run = run + 1 if quiet[i] else 0
        if run <= most:
            kept.append(samples[i * block:(i + 1) * block])
    kept.append(samples[n * block:])
    return np.concatenate(kept)


@dataclass
class _Sound:
    """A piece of the engine's own voice: float samples at `rate`."""
    samples: np.ndarray
    rate: int

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.rate


@dataclass
class _Line:
    text: str = ""
    spoken: int = 0  # how much of `text` is said or queued
    at: float = 0.0  # when its last piece came
    said: list[str] = field(default_factory=list)  # its sentences said, queued or let go while muted, as words()


class Speaker:
    """hear(event) and hear_audio(lane, pcm, rate) from any thread; it speaks on its own thread. `speaking`: a voice can
    be heard now, or was a moment ago (the microphone's way is muted then, where it could hear it). `voice_loader()`
    gives the laptop's voice, an object with `rate` and synthesize(text, speed), or None while it isn't downloaded;
    `player` has play(samples, rate), interrupt(), close() and `private`. `lanes`: the ways the laptop's voice speaks
    (translated into its language); `own`: the ways whose engine's own voice is played (any language)."""

    def __init__(self, voice_loader: Callable, player, lanes: Iterable[str], language: str = "en", speed: float = 1.0,
                 on_problem: Callable[[str], None] = lambda message: None, clock: Callable[[], float] = time.monotonic,
                 ducker=None, own: Iterable[str] = ()):
        self._load, self.player, self.language, self.speed = voice_loader, player, language, speed
        self.lanes, self.own = set(lanes), set(own)
        self._problem, self._clock = on_problem, clock
        self.ducker = ducker  # keeps the other apps lower while it runs (sst.live.ducking), if given
        self._lines: dict[str, _Line] = {}
        self._queue: deque[tuple[float, str, str | _Sound]] = deque()  # (when it was whole or came, way, what)
        self._waiting: deque[tuple[float, str, str]] = deque()  # sentences waiting for the engine's voice, as _queue
        self._sound_at: dict[str, float] = {}  # way -> when the engine's voice last came
        self._recent: deque[tuple[str, float]] = deque(maxlen=RECENT)  # the session's last sentences: (words, when)
        self._lock = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._voice = None  # the laptop's voice, once loaded
        self._voice_lock = threading.Lock()
        self._voice_broken = False
        self._audible_until = 0.0
        self._last_problem = ""
        self.muted = False
        self.said = self.skipped = self.repeats = 0  # repeats: sentences that came again and weren't said again
        self.voiced = 0  # sentences the engine's own voice said (the laptop's didn't)
        self.sound_s = self.sound_skipped_s = 0.0  # seconds of the engine's voice played, and left out to keep up
        self.unvoiced = 0  # sentences no voice said: the engine's didn't, the laptop's isn't downloaded

    @property
    def speaking(self) -> bool:
        return self._clock() < self._audible_until + TAIL

    @property
    def private(self) -> bool:
        """The voice plays through headphones or a headset: no microphone hears it."""
        return bool(self.player.private)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="live-voice", daemon=True)
        self._thread.start()
        if self.lanes:  # loaded meanwhile (~1 s): the engine's voice isn't held up, and the first stand-in is quick
            threading.Thread(target=self._local, name="live-voice-load", daemon=True).start()
        if self.ducker is not None:
            self.ducker.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            self._queue.clear()
            self._waiting.clear()
            self._lock.notify()
        self.player.interrupt()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(3)
        if self.ducker is not None:
            self.ducker.stop()  # the other apps back to their own volume

    def set_lanes(self, lanes: Iterable[str], own: Iterable[str] | None = None) -> None:
        """The ways the laptop's voice speaks, and (if given) those whose engine's own voice is played."""
        with self._lock:
            self.lanes = set(lanes)
            if own is not None:
                self.own = set(own)
            self._queue = deque(item for item in self._queue
                                if item[1] in (self.own if isinstance(item[2], _Sound) else self.lanes))
            self._waiting = deque(item for item in self._waiting if item[1] in self.lanes)
            self._lines = {lane: line for lane, line in self._lines.items() if lane in self.lanes}

    def set_speed(self, speed: float) -> None:
        self.speed = speed

    def set_muted(self, muted: bool) -> None:
        """Muted: nothing more is said (what's queued is dropped, what's playing stops) until unmuted."""
        with self._lock:
            self.muted = muted
            if muted:
                self._queue.clear()
                self._waiting.clear()
                for lane, line in self._lines.items():
                    self._say(lane, line, _all(line.text[line.spoken:]))  # let go: not said after unmuting either
                    line.spoken = len(line.text)
        if muted:
            self.player.interrupt()

    def hear_audio(self, lane: str, pcm: bytes, rate: int) -> None:
        """A piece of the engine's own voice (PCM16 mono at `rate`) for a way: played in turn; the sentences of that way
        waiting for it are its to say."""
        with self._lock:
            if self.muted or self._stop.is_set() or lane not in self.own or len(pcm) < 2:
                return
            now = self._clock()
            self._sound_at[lane] = now
            kept = deque(item for item in self._waiting if item[1] != lane)
            self.voiced += len(self._waiting) - len(kept)
            self._waiting = kept
            samples = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768
            self._queue.append((now, lane, _Sound(samples, rate)))
            self._lock.notify()

    def hear(self, event: LiveEvent) -> None:
        if event.kind not in (Kind.TRANSLATION, Kind.LINE):
            return
        with self._lock:
            if event.lane not in self.lanes:
                return
            line = self._lines.setdefault(event.lane, _Line())
            text = event.text.lstrip()  # a line's first piece may start with a space; its LINE never does
            start = line.spoken
            if not text.startswith(line.text[:start]):  # the model revised the line: from the first sentence it changed
                same = len(os.path.commonprefix([text, line.text[:start]]))
                start = max((end for _, end in _spans(text) if end <= same), default=0)
            line.text, line.at = text, self._clock()
            if event.kind is Kind.LINE:
                self._say(event.lane, line, _all(text[start:]), together=True)
                del self._lines[event.lane]
            else:
                spans = _spans(text[start:])
                self._say(event.lane, line, [sentence for sentence, _ in spans])
                line.spoken = start + (spans[-1][1] if spans else 0)
            self._lock.notify()

    def _say(self, lane: str, line: _Line, sentences: list[str], together: bool = False) -> None:
        """Queue the sentences not said before (all of them as one, `together`) and remember them; in a way the
        engine's voice speaks, only those it hasn't said, once it's had AUDIO_WAIT to. Under the lock."""
        new = []
        for sentence in sentences:
            said = words(sentence)
            if not said:
                continue  # punctuation alone
            if self._repeats(said, line):
                self.repeats += not self.muted
                continue
            line.said.append(said)
            self._recent.append((said, self._clock()))
            if not self.muted:
                new.append(sentence)
        now = self._clock()
        for sentence in [" ".join(new)] if together and new else new:
            if lane not in self.own:
                self._queue.append((now, lane, sentence))
            elif self._sound_at.get(lane, -np.inf) >= now - OWN_RECENT:
                self.voiced += 1  # the engine's voice has said it, or is saying it
            else:
                self._waiting.append((now, lane, sentence))  # the laptop's voice says it if the engine's doesn't come

    def _repeats(self, said: str, line: _Line) -> bool:
        """Whether a sentence (as words()) was said already: in its line (alike one, two or three of its sentences in a
        row, or a part of one: a comma made a full stop), or of late in the session."""
        if any(f" {said} " in f" {other} " for other in line.said):
            return True
        for i in range(len(line.said)):
            if any(alike(said, " ".join(line.said[i:i + n])) for n in (1, 2, 3) if i + n <= len(line.said)):
                return True
        now, short = self._clock(), said.count(" ") + 1 < SHORT
        return any(alike(said, other) for other, at in self._recent if not short or now - at <= SHORT_RECENT)

    def _settle(self) -> None:
        """A line whose text ends with a full stop and has had nothing new for SETTLE seconds: that sentence is said.
        A sentence the engine's voice hasn't said in AUDIO_WAIT seconds: the laptop's says it. Under the lock."""
        now = self._clock()
        for lane, line in self._lines.items():
            rest = line.text[line.spoken:]
            if rest.strip() and _ENDS.search(rest) and now - line.at >= SETTLE:
                self._say(lane, line, _all(rest), together=True)
                line.spoken = len(line.text)
        while self._waiting and now - self._waiting[0][0] >= AUDIO_WAIT:
            self._queue.append(self._waiting.popleft())

    def _next(self) -> tuple[str | _Sound, int] | None:
        """What to say next (a sentence, or a stretch of the engine's voice) and how many wait behind it, or None
        (nothing for now, or stopping)."""
        with self._lock:
            while not self._stop.is_set():
                self._settle()
                while self._queue:
                    at, lane, what = self._queue.popleft()
                    if self._queue and self._clock() - at > STALE:
                        if isinstance(what, _Sound):
                            self.sound_skipped_s += what.seconds
                        else:
                            self.skipped += 1  # newer ones wait: this one stays on screen and in the transcript
                        continue
                    if isinstance(what, _Sound):
                        what = self._stretch(lane, what)
                    return what, len(self._queue)
                self._lock.wait(0.1)
        return None

    def _stretch(self, lane: str, first: _Sound) -> _Sound:
        """The engine's voice from `first` on, up to MERGE_S of it in one piece; its long pauses shortened while more
        than BEHIND_S of it waits. Under the lock."""
        parts, seconds = [first.samples], first.seconds
        while self._queue and seconds < MERGE_S:
            _, other, what = self._queue[0]
            if other != lane or not isinstance(what, _Sound) or what.rate != first.rate:
                break
            self._queue.popleft()
            parts.append(what.samples)
            seconds += what.seconds
        samples = np.concatenate(parts) if len(parts) > 1 else first.samples
        behind = sum(what.seconds for _, other, what in self._queue if other == lane and isinstance(what, _Sound))
        if behind > BEHIND_S:
            shorter = shorten_pauses(samples, first.rate)
            self.sound_skipped_s += (len(samples) - len(shorter)) / first.rate
            samples = shorter
        return _Sound(samples, first.rate)

    def _local(self):
        """The laptop's voice, loaded the first time it's wanted (None while it isn't downloaded: asked again later)."""
        with self._voice_lock:
            if self._voice is None and not self._voice_broken:
                try:
                    self._voice = self._load()
                except Exception as e:
                    self._voice_broken = True
                    log.exception("The voice couldn't be loaded")
                    self._report(f"The voice couldn't start: {e}")
                else:
                    if self._voice is not None:
                        log.info("The voice is ready (%s, %d Hz): it speaks the ways %s", self.language, self._voice.rate,
                                 sorted(self.lanes))
            return self._voice

    def _run(self) -> None:
        try:
            while (item := self._next()) is not None:
                what, waiting = item
                try:
                    if isinstance(what, _Sound):
                        samples, rate = what.samples, what.rate
                    else:
                        voice = self._local()
                        if voice is None:
                            self.unvoiced += 1  # not downloaded (yet), or it couldn't load: said once already
                            continue
                        speed = min(self.speed * (1 + CATCH_UP * waiting), max(MAX_SPEED, self.speed))
                        samples, rate = voice.synthesize(what, speed), voice.rate
                    if self.muted or self._stop.is_set():
                        continue
                    self._audible_until = self._clock() + len(samples) / rate + 0.25  # + Windows' buffer
                    left = self.player.play(samples, rate)
                    self._audible_until = self._clock() + left
                    if isinstance(what, _Sound):
                        self.sound_s += len(samples) / rate
                    else:
                        self.said += 1
                    self._last_problem = ""
                except Exception as e:  # one sentence (or piece) lost; the next is tried
                    log.warning("The voice couldn't say a sentence: %s", e)
                    self._report(f"The voice couldn't play: {e}")
        finally:
            self.player.close()

    def _report(self, message: str) -> None:
        if message != self._last_problem:  # once, not for every sentence
            self._last_problem = message
            self._problem(message)
