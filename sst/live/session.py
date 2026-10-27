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

With speaking on, the translation also goes to a Speaker (sst.live.speaker): the engine's own voice (an engine with an
`on_audio` attribute: Gemini's), else the words to say with the laptop's voice. A way that could hear the voice gets
silence while it speaks: the microphone near speakers (not headphones), and what the laptop plays where Rflow's own sound
can't be left out (before Windows 11, or the output's whole mix when process loopback misses a call, while the voice
plays on that output). So the spoken translation is never heard and translated again; the meeting's words under the
voice are lost meanwhile (the mix can't be told apart from the voice in it).

With Both on speakers (the output isn't headphones or a headset), the microphone hears the others' words from the
speakers, which would be translated as the user's. So the microphone gets silence while the computer plays sound one
would hear and ECHO_HANGOVER after (the room's echo; the sound reaches the capture before the speakers play it). Gated,
not echo-cancelled: Windows' echo cancellation is the device's (many microphones have none), comes with communications
mode, which lowers the other apps by Windows' own setting and treats the room's voices as noise. The user's words while
the others speak are lost; theirs are translated by the computer's way anyway.

What the session costs (spend()) is the engines' own count (an engine with `spend()`: Gemini's) at the Live API's price,
each second sent billed in and Gemini's voice billed out; with the cost saver on (set_saving), what wasn't sent is what
was saved, priced both ways since Gemini's voice comes back as long as what's sent. quiet_for() says how long no words
came, for the saver's auto-stop.
"""
import logging
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from sst import costs
from sst.live.contracts import MIC, RATE, SYSTEM, Kind, LiveConfig, LiveEvent, already_in, language_name
from sst.live.transcript import Transcript

ECHO_HANGOVER = 0.5  # seconds the microphone stays silenced after the speakers played (Both on speakers)
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


def _path(capture) -> str:
    """How a way's sound was captured, for the log: the capture's own word for it (`path`) if it has one, else what its
    attributes say (sst.live.wasapi.Capture: the device, and whether Rflow's own sound was in it)."""
    path = getattr(capture, "path", None)
    if isinstance(path, str) and path:
        return path
    what = getattr(capture, "what", "") or type(capture).__name__
    if what == "output device":
        what += ", the whole mix" if getattr(capture, "hears_self", False) else ", process loopback"
    rate = getattr(capture, "device_rate", 0)
    return f"{what} at {rate} Hz" if rate else what


@dataclass(frozen=True)
class Spend:
    """What a session cost so far, all its ways together."""
    cost: float = 0.0  # USD: the seconds sent and the seconds of Gemini's voice received
    saved: float = 0.0  # USD: the seconds not sent, priced in and out
    seconds: float = 0.0  # heard (sent or not), by the way that heard the longest: the session's length
    unsent_s: float = 0.0  # not sent, all ways


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
        self._gated = False  # the microphone was silenced for the speakers' sound (said once in the log)
        self._spent: dict[str, tuple[float, float, float]] = {}  # per way: what its stopped engines streamed (spend())
        self._words_at: float | None = None  # when words last came, or the session started (quiet_for)

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
        if hasattr(engine, "saving"):  # the cost saver, as the session has it now
            engine.saving = self.config.save
        if hasattr(engine, "on_audio"):  # the translation spoken by the engine itself: to the speaker, if one speaks
            engine.on_audio = lambda pcm, rate: self._audio(lane, pcm, rate)
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
        if not self.lanes:
            self._words_at = time.monotonic()
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
        if callable(getattr(engine, "spend", None)):  # kept for spend(): the bar and the page show the last session's
            done = self._spent.get(lane, (0.0, 0.0, 0.0))
            self._spent[lane] = tuple(a + b for a, b in zip(done, engine.spend(), strict=True))
        lags, lines = self.lags.get(lane, []), self.lines.get(lane, 0)
        report = getattr(engine, "report", None)  # the engine's numbers (GeminiLiveTranslate.report): never any text
        engine_says = f"; {report()}" if callable(report) else ""
        if lags:
            log.info("Live captions stopped (%s, %s): %d lines; translation complete %.1f s after the voice paused "
                     "(median of %d), %.1f s at most%s", lane, _path(capture), lines, statistics.median(lags), len(lags),
                     max(lags), engine_says)
        else:
            log.info("Live captions stopped (%s, %s): %d lines%s", lane, _path(capture), lines, engine_says)

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
            if getattr(old, "sound_s", 0) or getattr(old, "voiced", 0):
                log.info("Gemini's own voice: %.0f s played (%.0f s left out to keep up), %d sentences said by it, %d by "
                         "the laptop's voice in its place, %d by neither", old.sound_s, old.sound_skipped_s, old.voiced,
                         old.said, old.unvoiced)
        if speaker is not None:
            speaker.start()

    def set_saving(self, on: bool) -> None:
        """Turn the cost saver on or off: for the engines running now, and the ways added later."""
        self.config.save = on
        for _, engine in list(self.lanes.values()):
            if hasattr(engine, "saving"):
                engine.saving = on
        log.info("Live captions: the cost saver %s", "on" if on else "off")

    def spend(self) -> Spend:
        """What the session cost so far, and what the saver saved (its ways now, and the ones that stopped)."""
        lanes = {lane: list(done) for lane, done in self._spent.items()}
        for lane, (_, engine) in list(self.lanes.items()):
            if callable(getattr(engine, "spend", None)):
                lanes[lane] = [a + b for a, b in zip(lanes.get(lane, (0.0, 0.0, 0.0)), engine.spend(), strict=True)]
        rate_in, rate_out = costs.live_rates(self.config.model)
        sent = sum(s for s, _, _ in lanes.values())
        received = sum(r for _, r, _ in lanes.values())
        unsent = sum(u for _, _, u in lanes.values())
        heard = max((s + u for s, _, u in lanes.values()), default=0.0)
        return Spend(cost=sent * rate_in + received * rate_out, saved=unsent * (rate_in + rate_out), seconds=heard,
                     unsent_s=unsent)

    def quiet_for(self, now: float | None = None) -> float:
        """Seconds since words last came (heard or translated), or since the session started; 0 if it hasn't."""
        at = self._words_at
        return 0.0 if at is None else max(0.0, (time.monotonic() if now is None else now) - at)

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
            elif lane == MIC and self._speakers_heard():
                frame = bytes(len(frame))
            engine.feed(frame)
        return feed

    def _speakers_heard(self) -> bool:
        """With Both, the computer plays through speakers (not headphones) and had sound just now: the microphone may
        be hearing it."""
        system = self.lanes.get(SYSTEM)
        if self.config.source != "both" or system is None or getattr(system[0], "private", None) is not False:
            return False
        if not system[0].sounding(ECHO_HANGOVER):
            return False
        if not self._gated:
            self._gated = True
            log.info("Live captions: Both on speakers: the microphone is silenced while the computer plays sound")
        return True

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

    def _audio(self, lane: str, pcm: bytes, rate: int) -> None:
        """A piece of the translation spoken by the engine (its thread): to the speaker, which knows the ways it plays."""
        speaker = self.speaker
        if speaker is not None and hasattr(speaker, "hear_audio"):
            speaker.hear_audio(lane, pcm, rate)

    def _event(self, event: LiveEvent) -> None:
        if event.kind in (Kind.SOURCE, Kind.TRANSLATION) and event.text:
            self._words_at = max(self._words_at or 0.0, event.at)
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
