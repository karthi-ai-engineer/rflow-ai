"""The wake word's turns, fed every microphone frame: idle until the keyword is heard (on_wake: the app starts a
dictation), then a turn until the end phrase is heard and followed by silence (on_end). No threads of its own: the
callbacks run in feed(), on the feeder's thread, which also calls end() (from a callback too). feed() takes up to ~0.3 s
when it rechecks or checks the voice: feed it from a worker thread, not the microphone's callback.

    idle --keyword--> turn --end phrase + END_SILENCE of quiet--> on_end(True)
                           --SAFETY_SILENCE of quiet-----------> on_end(True)
                           --a voice not the user's------------> on_end(False): the app discards the dictation
                           --end() (a command, Ctrl+Win, Esc)--> back to idle, no callback

An end phrase counts only with silence after it (the spike: "stop it" also fires on "stop it from sending"): speech
within END_SILENCE takes it back, and the dictation goes on with those words. A short first sentence then a pause
(on_pause) lets the app check for a Text Transform command ("<keyword>, make it a Teams message").

The end phrase is listened for twice. A stream running since the keyword hears most; whether it hears a phrase depends
on where in the stream it falls (the same "stop it" heard or missed as the stream started 0.3 s earlier or later). So
when speech is followed by END_SILENCE of quiet and no phrase was heard, fresh streams go over the last RECHECK_SECONDS
at RECHECK_OFFSETS, and a phrase heard there ends the turn if it was the last thing said (its hit no earlier than the
last speech: not "stop it from sending"). Measured on the spike's clips (a sentence, then "..., stop it."): the running
stream 42-44 of 48, with the recheck 47 of 48, and still no end on "stop it from sending" or "stop it before it spreads".

The voice check scores the keyword (the KEYWORD_AUDIO seconds before the hit) plus the turn's first VOICE_SECONDS of
speech, as soon as there is that much, or earlier when the turn ends or pauses first (the app may act on a command).

Silence and speech: an energy gate on each frame (Gate). The room is the quietest frame of the last ROOM_SECONDS (even
fluent speech pauses for breath in that time; a fan switched on becomes the room within it); a frame GATE_DB above it,
and above SPEECH_FLOOR_DB, is speech. Simple, and enough for "is anyone talking": the spotter, not the gate, decides
what was said.
"""
import logging
from collections import deque
from collections.abc import Callable, Sequence

import numpy as np

from sst.wake.enrol import WakeSetup
from sst.wake.spotter import RATE, Spotter
from sst.wake.voiceprint import VOICE_THRESHOLD
from sst.wake.words import END_PHRASES, spellings

SAFETY_SILENCE = 20.0  # seconds of silence that end a turn anyway (the end phrase was missed)
END_SILENCE = 0.7  # quiet needed after the end phrase
END_GRACE = 0.15  # the phrase's own last sound, just after the hit, isn't "speaking on"
PAUSE_SILENCE = 1.2  # a pause after a short first sentence: on_pause
COMMAND_SECONDS = 4.0  # "short": the speech ended within this long of the keyword
RECHECK_SECONDS = 2.5  # the phrase (~0.6 s), what came before it, and END_SILENCE
RECHECK_OFFSETS = (0.0, 0.1, 0.2)  # seconds into that audio each fresh stream starts
KEYWORD_AUDIO = 1.5  # seconds before the hit kept for the voice check (the keyword, and the hit's ~0.4 s delay)
VOICE_SECONDS = 3.0  # of the turn's speech, with the keyword, for the voice check
GATE_DB = 10.0
SPEECH_FLOOR_DB = -65.0  # dBFS: quieter is never speech (a muted or disconnected microphone)
ROOM_SECONDS = 10.0  # long enough that fluent speech has a gap in it, short enough for a fan switched on

log = logging.getLogger(__name__)


class Gate:
    """Is this frame speech? See the module's notes."""

    def __init__(self):
        self._levels: deque[tuple[float, int]] = deque()  # the last ROOM_SECONDS: (dBFS, samples)
        self._samples = 0
        self.room = SPEECH_FLOOR_DB

    def __call__(self, frame: np.ndarray) -> bool:
        if not len(frame):
            return False
        level = 10 * float(np.log10(np.mean(frame.astype(np.float64) ** 2) + 1e-12))
        self._levels.append((level, len(frame)))
        self._samples += len(frame)
        while self._samples - self._levels[0][1] >= ROOM_SECONDS * RATE:
            self._samples -= self._levels.popleft()[1]
        self.room = min(db for db, _ in self._levels)
        return level > SPEECH_FLOOR_DB and level > self.room + GATE_DB


def _ignore(*_args) -> None:
    pass


class WakeListener:
    def __init__(self, setup: WakeSetup, end_phrases: Sequence[str] = END_PHRASES, *, voice_check: bool = True,
                 spotter_factory: Callable[[dict[str, str]], Spotter] = Spotter):
        if not end_phrases:
            raise ValueError("a turn needs an end phrase")
        self.setup = setup
        self.voice_check = voice_check and setup.voice is not None
        self.on_wake: Callable[[], None] = _ignore
        self.on_pause: Callable[[], None] = _ignore
        self.on_end: Callable[[bool], None] = _ignore
        self.active = False
        self.turns = 0
        self.voice_score: float | None = None  # the last turn's voice check
        self._keyword = spotter_factory({"keyword": setup.spelling})
        ends = {phrase: spellings(phrase)[0] for phrase in dict.fromkeys(end_phrases)}
        self._end = spotter_factory(ends)
        self._rechecks = [spotter_factory(ends) for _ in RECHECK_OFFSETS]
        self._gate = Gate()
        self._recent: deque[np.ndarray] = deque()  # the last seconds: the keyword's for the voice check, the recheck's
        self._recent_samples = 0
        self._idle()

    def _idle(self) -> None:
        self.active = False
        self._keyword.reset()  # nothing half-heard carries over (the end of the last turn's audio)
        self._woke_at = self._last_speech = 0.0
        self._ended_at: float | None = None  # the end phrase's hit, waiting for silence after it
        self._spoke = self._paused = self._rechecked = False
        self._voice: list[np.ndarray] | None = None  # audio for the voice check; None once checked (or not wanted)
        self._voice_speech = 0

    def feed(self, frame: np.ndarray, now: float) -> None:
        """One microphone frame (16 kHz mono float32, ~100 ms) at `now` (seconds, any steady clock)."""
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        speech = self._gate(frame)
        self._remember(frame)
        if not self.active:
            if self._keyword.feed(frame) is not None:
                self._wake(now)
            return
        self._turn(frame, speech, now)

    def end(self) -> None:
        """The app ended the turn itself (a command, Ctrl+Win, Esc): back to listening for the keyword."""
        if self.active:
            log.info("Wake word: turn %d ended by the app", self.turns)
            self._idle()

    def _remember(self, frame: np.ndarray) -> None:
        self._recent.append(frame)
        self._recent_samples += len(frame)
        while self._recent and self._recent_samples - len(self._recent[0]) >= max(KEYWORD_AUDIO, RECHECK_SECONDS) * RATE:
            self._recent_samples -= len(self._recent.popleft())

    def _wake(self, now: float) -> None:
        self._idle()
        self.active, self.turns, self.voice_score = True, self.turns + 1, None
        self._woke_at = self._last_speech = now
        self._end.reset()
        if self.voice_check:
            self._voice = [np.concatenate(self._recent)[-int(KEYWORD_AUDIO * RATE):]]
        log.info("Wake word: heard (turn %d)", self.turns)
        self.on_wake()

    def _turn(self, frame: np.ndarray, speech: bool, now: float) -> None:
        if speech:
            if self._ended_at is not None and now - self._ended_at > END_GRACE:
                log.info("Wake word: the end phrase was followed by speech; dictation goes on")
                self._ended_at = None
            self._last_speech, self._spoke, self._rechecked = now, True, False
            if self._voice is not None:
                self._voice.append(frame)
                self._voice_speech += len(frame)
                if self._voice_speech >= VOICE_SECONDS * RATE and not self._voice_ok():
                    return self._finish(False)
        if self._end.feed(frame) is not None:
            self._ended_at = now
        if self._ended_at is not None and now - self._ended_at >= END_SILENCE:
            return self._finish(True)
        quiet = now - self._last_speech
        if self._spoke and not self._rechecked and self._ended_at is None and quiet >= END_SILENCE:
            self._rechecked = True
            if self._heard_at_the_end(now):
                return self._finish(True)
        if quiet >= SAFETY_SILENCE:
            log.info("Wake word: %.0f s of silence ended turn %d", quiet, self.turns)
            return self._finish(True)
        if (not self._paused and self._spoke and quiet >= PAUSE_SILENCE
                and self._last_speech - self._woke_at <= COMMAND_SECONDS):
            self._paused = True
            if not self._voice_ok():
                return self._finish(False)
            self.on_pause()

    def _heard_at_the_end(self, now: float) -> bool:
        """The recheck (see the module's notes): was an end phrase the last thing said before this silence?"""
        audio = np.concatenate(self._recent)[-int(RECHECK_SECONDS * RATE):]
        start, step = now - len(audio) / RATE, int(0.1 * RATE)
        for spotter, offset in zip(self._rechecks, RECHECK_OFFSETS, strict=True):
            spotter.reset()
            first = int(offset * RATE)
            for i in range(first, len(audio), step):
                at = start + min(i + step, len(audio)) / RATE  # when the frame that brought the hit ended
                if spotter.feed(audio[i:i + step]) is not None and at >= self._last_speech - 0.1 and at > self._woke_at:
                    log.info("Wake word: the end phrase was heard on a second listen")
                    return True
        return False

    def _voice_ok(self) -> bool:
        """The voice check, once a turn (True when it's off or already passed). Logs the score, never the words."""
        if self._voice is None:
            return True
        audio, self._voice = np.concatenate(self._voice) if self._voice else np.zeros(0, np.float32), None
        try:
            self.voice_score = self.setup.voice.score(audio)
        except Exception:  # the voice model went missing: closed, as when it's someone else (the app checks it's there)
            log.exception("Wake word: the voice check failed to run")
            return False
        passed = self.voice_score >= VOICE_THRESHOLD
        log.info("Wake word: voice check %.2f (%s) on %.1f s", self.voice_score, "passed" if passed else "not the user",
                 len(audio) / RATE)
        return passed

    def _finish(self, keep: bool) -> None:
        keep = keep and self._voice_ok()
        log.info("Wake word: turn %d %s after %.1f s", self.turns, "ends" if keep else "discarded",
                 self._last_speech - self._woke_at)
        self._idle()
        self.on_end(keep)
