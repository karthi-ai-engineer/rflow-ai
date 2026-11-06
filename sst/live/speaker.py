"""Speaking live translations aloud: Gemini's own voice, played as it streams.

    the engine's voice (Gemini Live Translate: 24 kHz PCM16) of the ways spoken -> hear_audio() -> a queue
    the queue -> one thread -> the player (Windows' default output)

Gemini Live Translate speaks its translation itself (its transcript is what the bar shows), and that voice is the only
one (the owner's choice of 2026-11-06, #264). Until 2.8.1 a voice on the laptop (Piper) stood in for sentences Gemini's
voice didn't say: in every session measured Gemini's came for every sentence and the laptop's said nothing, at the
price of a download, a speed setting and the matching of sentences to sound.

How the voice comes (measured that day): in pieces of 250 ms, one every 250 ms, as long as what Gemini was sent,
silence for silence, a piece now and then late: up to 125 ms from the connection alone, up to 320 ms with the whole
app running (captures, the other apps lowered). Played the moment it came, a late piece left a hole inside a word, and
every piece after it came as the one before ended: a click at each. So the voice is played with a cushion: each time
it starts (the first piece after a pause) that piece is held `cushion` seconds, and what follows plays as it comes,
that far ahead of its arrival; the pauses between sentences last as long as Gemini made them. A hole that happens
anyway (the output ran dry inside the voice) is followed by CUSHION again, so the rest of the sentence is whole, and
the cushion doubles for the next time the voice starts, up to MAX_CUSHION.

A stretch that is all pause (below PAUSE_LEVEL) isn't played once the voice has ended, and never makes it `speaking`.
Else a way that gets silence while the voice speaks (the whole mix, a microphone near speakers) sends silence, Gemini
answers it with silence, played as voice, and that way stays silenced for good: nothing after the first sentence is
translated.

An interpreter mustn't fall behind for good (the network stalls, then sends it all at once). A piece played more than
LATE after it should have been is behind: from then on a pause is kept to PAUSE_S, so the voice is back with the
speaker within a few sentences and no word is lost. While more than BEHIND_S waits, the pauses inside its pieces are
shortened too, and a piece older than STALE is skipped while newer ones wait (its words stay on screen and in the
transcript). Like an interpreter, it keeps the other apps lower while it runs, if
given a Ducker (sst.live.ducking).
"""
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cached_property

import numpy as np

CUSHION = 0.5  # seconds the voice's first piece waits, so the pieces after it may come that late
MAX_CUSHION = 1.5  # as far as holes inside the voice make it grow
STALE = 10.0  # seconds a piece may wait while newer ones queue behind it
TAIL = 0.4  # seconds after the voice ends that a microphone (the room) may still hear it; its pauses this long are played
MERGE_S = 0.5  # seconds of the voice handed to the player at once (its pieces are small)
LATE = 0.5  # seconds past its cushion a piece is played: behind, so pauses are kept short until it's caught up
BEHIND_S = 2.0  # seconds of the voice waiting: the pauses inside its pieces are shortened too
PAUSE_S, PAUSE_LEVEL = 0.25, 10 ** (-45 / 20)  # a pause: below -45 dBFS; cut to PAUSE_S when behind
EDGE = 0.02  # seconds at a piece's start or end that say whether the voice goes on across it

log = logging.getLogger(__name__)


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


def heard_for(samples: np.ndarray, rate: int) -> float:
    """Seconds from the start of `samples` to the end of their last sound (a 20 ms block at PAUSE_LEVEL or louder); 0
    when they're a pause all through."""
    block = max(1, rate // 50)
    n = -(-len(samples) // block)
    if not n:
        return 0.0
    padded = np.zeros(n * block, np.float32)
    padded[:len(samples)] = samples
    loud = np.flatnonzero(np.sqrt(np.mean(padded.reshape(n, block) ** 2, axis=1)) >= PAUSE_LEVEL)
    return min(len(samples), (int(loud[-1]) + 1) * block) / rate if len(loud) else 0.0


def _loud(samples: np.ndarray) -> bool:
    return bool(len(samples)) and float(np.sqrt(np.mean(samples * samples))) >= PAUSE_LEVEL


@dataclass
class _Sound:
    """A piece of the voice: float samples at `rate`."""
    samples: np.ndarray
    rate: int

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.rate

    @cached_property
    def pause(self) -> bool:
        """A pause all through: nothing of the voice in it."""
        return not heard_for(self.samples, self.rate)

    def edge(self, end: bool) -> bool:
        """Whether it starts (or ends) inside the voice: sound in its first (last) EDGE seconds."""
        n = max(1, round(EDGE * self.rate))
        return _loud(self.samples[-n:] if end else self.samples[:n])


class Speaker:
    """hear_audio(lane, pcm, rate) from any thread; it plays on its own thread. `speaking`: the voice can be heard now,
    or was a moment ago (a way that could hear it gets silence then). `player` has play(samples, rate), which returns
    the seconds still to be heard, interrupt(), close() and `private`. `lanes`: the ways whose voice is played."""

    def __init__(self, player, lanes: Iterable[str], on_problem: Callable[[str], None] = lambda message: None,
                 clock: Callable[[], float] = time.monotonic, ducker=None):
        self.player, self.lanes = player, set(lanes)
        self._problem, self._clock = on_problem, clock
        self.ducker = ducker  # keeps the other apps lower while it runs (sst.live.ducking), if given
        self._queue: deque[tuple[float, str, _Sound]] = deque()  # (when it came, way, the piece)
        self._lock = threading.Condition()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._audible_until = 0.0
        self._end = 0.0  # when what the player was given will have played: after it, nothing plays
        self._voiced = False  # what the player was given last ended inside the voice
        self._paused = 0.0  # seconds of pause at the end of what the player was given
        self._torn = False  # the output ran dry inside the voice: it starts again after CUSHION, not the grown cushion
        self._last_problem = ""
        self.cushion = CUSHION
        self.muted = False
        self.sound_s = self.sound_skipped_s = 0.0  # seconds of the voice played, and left out to keep up
        self.quiet_s = 0.0  # seconds of it not played: its silence once the voice had ended
        self.holes = 0  # times the output ran dry inside the voice (a piece came later than the cushion)

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
        if self.ducker is not None:
            self.ducker.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            self._queue.clear()
            self._lock.notify()
        self.player.interrupt()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(3)
        if self.ducker is not None:
            self.ducker.stop()  # the other apps back to their own volume

    def set_lanes(self, lanes: Iterable[str]) -> None:
        """The ways whose voice is played."""
        with self._lock:
            self.lanes = set(lanes)
            self._queue = deque(item for item in self._queue if item[1] in self.lanes)

    def set_muted(self, muted: bool) -> None:
        """Muted: nothing more is played (what's queued is dropped, what's playing stops) until unmuted."""
        with self._lock:
            self.muted = muted
            if muted:
                self._queue.clear()
        if muted:
            self.player.interrupt()

    def hear_audio(self, lane: str, pcm: bytes, rate: int) -> None:
        """A piece of the engine's own voice (PCM16 mono at `rate`) for a way: played in turn."""
        with self._lock:
            if self.muted or self._stop.is_set() or lane not in self.lanes or len(pcm) < 2:
                return
            samples = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768
            self._queue.append((self._clock(), lane, _Sound(samples, rate)))
            self._lock.notify()

    def _next(self) -> _Sound | None:
        """The stretch of the voice to play next, or None (stopping). Waits while there's none, or its cushion fills."""
        with self._lock:
            while not self._stop.is_set():
                sound, wait = self._take(self._clock())
                if sound is not None:
                    return sound
                self._lock.wait(wait)
        return None

    def _take(self, now: float) -> tuple[_Sound | None, float]:
        """The stretch of the voice to play now, or None and how long to wait before looking again (a piece that comes
        wakes it sooner). Under the lock."""
        while True:
            while len(self._queue) > 1 and now - self._queue[0][0] > STALE:
                self.sound_skipped_s += self._queue.popleft()[2].seconds  # newer ones wait: this one is skipped
            if not self._queue:
                return None, 0.1
            at, lane, first = self._queue[0]
            if first.pause:
                if now >= self._end or now >= self._audible_until + TAIL:
                    self._queue.popleft()  # Gemini's answer to silence, the voice over: never played
                    self.quiet_s += first.seconds
                    self._voiced = False
                    continue
                if now - at > self.cushion + LATE and self._paused >= PAUSE_S:
                    self._queue.popleft()  # behind: this pause has lasted long enough
                    self.sound_skipped_s += first.seconds
                    continue
            elif not self._voiced:  # the voice starts (again, after a pause or a hole): its cushion first
                hold = min(self.cushion, CUSHION) if self._torn else self.cushion
                if (wait := at + hold - now) > 0 and self._waiting() - first.seconds < hold:
                    return None, min(wait, 0.05)  # (with that much waiting behind it, the cushion is there already)
                self._torn = False
            elif now >= self._end and first.edge(end=False):  # the output ran dry inside the voice
                self.holes += 1
                self.cushion = min(MAX_CUSHION, self.cushion * 2)  # from the next time the voice starts
                self._voiced, self._torn = False, True  # and now it starts again, a cushion ahead: no click a piece
                continue
            self._queue.popleft()
            return self._stretch(lane, first), 0.0

    def _waiting(self) -> float:
        """Seconds of the voice in the queue. Under the lock."""
        return sum(what.seconds for _, _, what in self._queue)

    def _stretch(self, lane: str, first: _Sound) -> _Sound:
        """The voice from `first` on, up to MERGE_S of it in one piece (a pause stays a piece of its own: _take decides
        whether it's played); the pauses inside it shortened while more than BEHIND_S of it waits. Under the lock."""
        parts, seconds = [first.samples], first.seconds
        while self._queue and seconds < MERGE_S and not first.pause:
            _, other, what = self._queue[0]
            if other != lane or what.rate != first.rate or what.pause:
                break
            self._queue.popleft()
            parts.append(what.samples)
            seconds += what.seconds
        samples = np.concatenate(parts) if len(parts) > 1 else first.samples
        if sum(what.seconds for _, other, what in self._queue if other == lane) > BEHIND_S:
            shorter = shorten_pauses(samples, first.rate)
            self.sound_skipped_s += (len(samples) - len(shorter)) / first.rate
            samples = shorter
        return _Sound(samples, first.rate)

    def _run(self) -> None:
        try:
            while (sound := self._next()) is not None:
                if self.muted or self._stop.is_set():
                    continue
                try:
                    self._play(sound)
                    self._last_problem = ""
                except Exception as e:  # one piece lost; the next is tried
                    log.warning("The voice couldn't play a piece: %s", e)
                    self._report(f"The voice couldn't play: {e}")
        finally:
            self.player.close()

    def _play(self, sound: _Sound) -> None:
        """Hand a stretch to the player (which returns once it's all in Windows' buffer) and note until when it's heard."""
        samples, rate = sound.samples, sound.rate
        heard = heard_for(samples, rate)
        after = len(samples) / rate - heard  # the pause at its end: nobody hears it
        if heard:  # a pause between words is played but makes nothing `speaking` (TAIL covers it)
            self._audible_until = self._clock() + heard + 0.25  # + Windows' buffer
        left = self.player.play(samples, rate)
        now = self._clock()
        if heard:
            self._audible_until = now + max(0.0, left - after)
        self._end, self._voiced = now + left, sound.edge(end=True)
        self._paused = after if heard else self._paused + after
        self.sound_s += len(samples) / rate

    def _report(self, message: str) -> None:
        if message != self._last_problem:  # once, not for every piece
            self._last_problem = message
            self._problem(message)
