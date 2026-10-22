"""One live captions session: one or two ways (lanes), each a capture feeding its own engine, and their events going to
one transcript, the log and whoever shows them (the caption bar). Qt-free: the app moves the events to its own thread.

    SYSTEM  what the laptop plays (a meeting, a video)       -> target
    MIC     what the microphone hears (the room, or the user)  -> mic_target (Japanese)

The microphone's lines that come back untranslated (speech already in mic_target) depend on the source. With Both they
are dropped, never shown or saved: they are the meeting's own words from the speakers when there are no headphones
(echo, already shown and translated by the SYSTEM way), or the user speaking the language their words are translated
into, which the bar never shows for the user anyway (it shows the user's lines by their translation only). With the
microphone alone they are the room's words: shown and saved as heard (the bar marks them as already in the target
language), never spoken, since the voice says translations only.

While a way has shown no words, the bar's status line says why (NOTE events), from the levels of the frames it hears:

    the engine listens                                   -> "Listening…"
    the last LiveConfig.note_after_s s: sound, no words  -> "Hearing sound, but no speech yet"
                                        near silence     -> "Nothing heard yet: is the sound playing on this laptop?"
                                                            (the microphone: "Nothing heard from the microphone")
    words come                                           -> cleared, for good
    microphone alone, lines already in its language      -> "You're speaking English, …" until a line is translated

With Both the microphone says nothing of its own silence: the user listening is silent. note(lane, message) shows
something a capture found out instead (a device with no sound on it), until words come.

With speaking on, the translation also goes to a Speaker (sst.live.speaker), and a way that could hear the voice gets
silence while it speaks: the microphone near speakers (not headphones), and what the laptop plays where Windows can't
leave Rflow's own sound out (before Windows 11). So the spoken translation is never heard and translated again.
"""
import logging
import statistics
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from sst.live.contracts import MIC, RATE, SYSTEM, Kind, LiveConfig, LiveEvent, already_in, language_name
from sst.live.transcript import Transcript

LISTENING = "Listening…"
NO_SPEECH = "Hearing sound, but no speech yet"
NOTHING = {SYSTEM: "Nothing heard yet: is the sound playing on this laptop?", MIC: "Nothing heard from the microphone"}

log = logging.getLogger(__name__)


def speaking_target(language: str) -> str:
    return f"You're speaking {language_name(language)}, the language it translates into"


def dbfs(frame: bytes) -> float:
    """A PCM16 frame's loudness (RMS) in dB below full scale; -inf for silence."""
    samples = np.frombuffer(frame[: len(frame) // 2 * 2], dtype="<i2").astype(np.float32)
    rms = float(np.sqrt(np.mean(samples * samples))) / 32768 if len(samples) else 0.0
    return 20 * np.log10(rms) if rms > 0 else -np.inf


@dataclass
class _Hearing:
    """What a way has heard so far, for its note."""
    listening: bool = False  # the engine said it listens: from then on, silence or sound without words is worth a note
    heard: bool = False  # words came: no note about silence any more
    told: bool = False  # the note is note()'s: kept until words come or it's taken back
    note: str = ""
    recent: deque = field(default_factory=deque)  # (ms, loud) of the last frames, note_after_s of them
    ms: int = 0


class LiveSession:
    """add(lane, capture, engine_factory) starts a way: `capture` has start(on_frame) and stop(); `engine_factory(on_event)`
    makes an engine with start(), feed(frame) and stop() (GeminiLiveTranslate, or a fake in the tests). A capture with
    an `on_problem` attribute gets a callable that shows an error, one with `on_note` one that shows a note."""

    def __init__(self, config: LiveConfig, transcript: Transcript | None = None,
                 on_event: Callable[[LiveEvent], None] = lambda event: None):
        self.config, self.transcript, self.on_event = config, transcript, on_event
        self.lanes: dict[str, tuple] = {}  # lane -> (capture, engine)
        self.lags: dict[str, list[float]] = {}  # per way, per line that ended in a pause: complete this long after it
        self.lines: dict[str, int] = {}
        self.speaker = None  # speaks the translation aloud (sst.live.speaker), while the user wants it
        self._hearing: dict[str, _Hearing] = {}
        self._lock = threading.Lock()  # the notes: frames come on the capture's thread, events on the engine's

    @property
    def running(self) -> bool:
        return bool(self.lanes)

    def add(self, lane: str, capture, engine_factory: Callable) -> None:
        """Start a way; if it can't start, nothing of it keeps running and the error is raised."""
        if lane in self.lanes:
            return
        with self._lock:
            self._hearing[lane] = _Hearing()
        engine = engine_factory(self._event)
        engine.start()
        if hasattr(capture, "on_problem"):  # why it hears nothing (Windows' sound muted), shown like the engine's
            capture.on_problem = lambda message: self._event(LiveEvent(Kind.ERROR, message, lane=lane))
        if hasattr(capture, "on_note"):  # what it found out (the sound is on another device), on the status line
            capture.on_note = lambda message: self.note(lane, message)
        try:
            capture.start(self._feed(lane, capture, engine))
        except Exception:
            engine.stop()
            with self._lock:
                self._hearing.pop(lane, None)
            raise
        self.lanes[lane] = (capture, engine)
        config = self.config.for_lane(lane)
        log.info("Live captions started (%s): into %s, %s", lane, config.target, config.model)

    def remove(self, lane: str) -> None:
        capture, engine = self.lanes.pop(lane, (None, None))
        if capture is None:
            return
        capture.stop()
        engine.stop()
        with self._lock:
            self._hearing.pop(lane, None)
        lags, lines = self.lags.get(lane, []), self.lines.get(lane, 0)
        if lags:
            log.info("Live captions stopped (%s): %d lines; translation complete %.1f s after the voice paused "
                     "(median of %d), %.1f s at most", lane, lines, statistics.median(lags), len(lags), max(lags))
        else:
            log.info("Live captions stopped (%s): %d lines", lane, lines)

    def stop(self) -> None:
        for lane in list(self.lanes):
            self.remove(lane)
        self.set_speaker(None)

    def set_speaker(self, speaker) -> None:
        """Speak the translation with `speaker` (started here), or stop speaking (None)."""
        old, self.speaker = self.speaker, speaker
        if old is not None:
            old.stop()
            log.info("The voice said %d sentences (%d skipped to keep up); the voice skipped a repeat %d times", old.said,
                     old.skipped, old.repeats)
        if speaker is not None:
            ducker = getattr(speaker, "ducker", None)
            if ducker is not None:  # what the computer's way translates is never lowered (sst.live.ducking)
                ducker.captured = lambda: SYSTEM in self.lanes
            speaker.start()

    def note(self, lane: str, message: str) -> None:
        """Show `message` on the bar's status line for this way (what a capture found out, in plain words), in place of
        what the session would say, until words come; "" takes it back. Any thread."""
        with self._lock:
            hearing = self._hearing.get(lane)
            if hearing is None:
                return
            hearing.told = bool(message)
            event = self._set(lane, hearing, message)
        self._emit(event)

    def _feed(self, lane: str, capture, engine) -> Callable[[bytes], None]:
        """What a way's engine is fed: its frames, or silence while the voice plays where this way hears it."""
        def feed(frame: bytes) -> None:
            self._listen(lane, frame)
            speaker = self.speaker
            if speaker is not None and speaker.speaking and (
                    not speaker.private if lane == MIC else getattr(capture, "hears_self", False)):
                frame = bytes(len(frame))
            engine.feed(frame)
        return feed

    # -- the status line

    def _says_silence(self, lane: str) -> bool:
        """Whether this way's silence is news: with Both the microphone is the user, silent while they listen."""
        return lane == SYSTEM or (lane == MIC and self.config.source == "microphone")

    def _listen(self, lane: str, frame: bytes) -> None:
        """A frame heard: once note_after_s seconds came without words, the note says whether there was sound."""
        c = self.config
        with self._lock:
            hearing = self._hearing.get(lane)
            if hearing is None or hearing.heard or not hearing.listening or not self._says_silence(lane):
                return
            ms = len(frame) * 1000 // (2 * RATE)
            hearing.recent.append((ms, dbfs(frame) >= c.quiet_dbfs))
            hearing.ms += ms
            window = round(c.note_after_s * 1000)
            while hearing.recent and hearing.ms - hearing.recent[0][0] >= window:
                hearing.ms -= hearing.recent.popleft()[0]
            if hearing.ms < window or hearing.told:  # a capture's own finding wins over a guess from the levels
                return
            loud = sum(ms for ms, sound in hearing.recent if sound)
            event = self._set(lane, hearing, NO_SPEECH if loud >= c.sound_share * hearing.ms else NOTHING[lane])
        self._emit(event)

    def _hear(self, event: LiveEvent) -> None:
        """What the engine says moves the note: listening starts it, words end it."""
        with self._lock:
            hearing = self._hearing.get(event.lane)
            if hearing is None:
                return
            note = hearing.note
            if event.kind is Kind.STATUS and event.text == "Listening":
                hearing.listening = True
                if not (hearing.heard or note) and self._says_silence(event.lane):
                    note = LISTENING
            elif event.kind in (Kind.SOURCE, Kind.TRANSLATION, Kind.LINE):
                hearing.heard, hearing.told = True, False
                target = self.config.for_lane(event.lane).target
                if event.kind is Kind.LINE and event.lane == MIC and self.config.source == "microphone" \
                        and already_in(event, target):
                    note = speaking_target(target)
                elif event.kind is Kind.LINE or note != speaking_target(target):
                    note = ""  # a line came; words in progress keep "you're speaking …" until their line is translated
            else:
                return
            update = self._set(event.lane, hearing, note)
        self._emit(update)

    @staticmethod
    def _set(lane: str, hearing: _Hearing, note: str) -> LiveEvent | None:
        """Under the lock: the event that shows the way's new note, or None if it's the same."""
        if note == hearing.note:
            return None
        hearing.note = note
        return LiveEvent(Kind.NOTE, note, lane=lane)

    def _emit(self, event: LiveEvent | None) -> None:
        if event is not None:
            log.debug("Live captions note (%s): %s", event.lane, event.text or "(cleared)")
            self.on_event(event)

    # -- the engines' events

    def _event(self, event: LiveEvent) -> None:
        if event.kind is Kind.LINE:
            if event.lane == MIC and not event.text and self.config.source == "both":
                log.debug("Live captions: an untranslated line from the microphone dropped (echo): %s", event.source)
                return
            self.lines[event.lane] = self.lines.get(event.lane, 0) + 1
            if event.seconds:
                self.lags.setdefault(event.lane, []).append(event.seconds)
            log.debug("Live line (%s, complete %.1f s after the pause): %s -> %s", event.lane, event.seconds,
                      event.source, event.text)
            if self.transcript is not None:
                self.transcript.add(event)
        if self.speaker is not None:
            self.speaker.hear(event)
        self.on_event(event)
        self._hear(event)  # after the line: the bar shows it before its note goes
