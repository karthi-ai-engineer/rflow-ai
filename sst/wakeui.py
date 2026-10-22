"""Start with your voice: the wake word in the app. Spotting the keyword, the voice check and a turn's logic are sst.wake;
this is where its microphone, its thread, its turns and its 10-try setup meet the rest of Rflow.

While it's on and set up, every block the always-on microphone delivers goes to a WakeListener: Recorder.tap hands it
on, and a thread of its own (WakeFeeder, never the UI thread) brings it to 16 kHz, cuts 100 ms frames and feeds them.
A Bluetooth headset's microphone is never kept open (it would drop the headset to call quality): with one chosen,
another microphone that isn't one listens (the built-in one), and the turn records there too, so the moment before the
keyword is heard is kept; with none, the wake word doesn't listen and the Settings card says why. Listening pauses
while live translation uses the microphone, while a dictation started with the keys or Text Transform runs, and while
the setup's tries are recorded.

A turn: the keyword starts a hands-free dictation, exactly like Ctrl+Win+Space (the beep, the pill). The microphone
keeps PREROLL seconds before the keyword was heard, so the first words said after it are kept even though the spotter
hears it about half a second late; the keyword itself is trimmed from the words heard (sst.wake.strip_keyword), and
so is the end phrase (strip_end), before the voice commands and the cleanup. The end phrase types the text; a failed
voice check drops it silently. Ctrl+Win and Esc still end a turn. A short pause after a few words: the take so far is
transcribed, and if it's a voice command ("make it concise") the turn ends and the command runs on the text box, as
with Ctrl+Win. The log says when turns start and end, and why; never the words.
"""
import importlib
import logging
import math
import queue
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QLineEdit, QStackedWidget, QVBoxLayout, QWidget

from sst import audio, theme
from sst.audio import TARGET_RATE, Recorder, StreamResampler, resample
from sst.dictate import trimmed
from sst.live.contracts import MIC
from sst.pipeline.audio_stream import trim_preroll
from sst.pipeline.contracts import VoiceConfig
from sst.settings import read_wake
from sst.ui import Lamp, Meter, Orb, button, card, label, set_tone

FRAME = TARGET_RATE // 10  # the spotter's 100 ms frames
# Seconds kept from before the keyword was heard: the keyword (up to ~1 s) and the spotter's delay (0.65 s p90 in the
# spike), so the words said right after it are in the recording. The voice pipeline trims the rest back to the
# speech running into the start (sst.pipeline.audio_stream.trim_preroll).
PREROLL = 2.0
QUEUE_BLOCKS = 1000  # ~10 s of the microphone's blocks; more means the listener is stuck, and blocks are dropped
CHECK_EVERY = 3.0  # seconds between looks at whether the microphone listened to is still open and still the right one
SUGGESTIONS = ("okay type", "computer")  # heard best in the spike (94.5% and 90.9%); "Rflow" alone 74.5%
BLUETOOTH = ("Your Bluetooth headset's microphone is chosen, and keeping it on would drop the headset to call quality. "
             "Rflow listens on another microphone when there is one: connect one, or choose another in AI & models.")

log = logging.getLogger(__name__)


def wake_api():
    """sst.wake, imported when first needed: its models load only when the wake word is used."""
    return importlib.import_module("sst.wake")


def _unload() -> None:
    """The wake word was switched off: its models' memory goes back (they load again when it's switched on)."""
    api = sys.modules.get("sst.wake")
    for part in ("spotter", "voiceprint"):
        unload = getattr(getattr(api, part, None), "unload", None)
        if unload is not None:
            unload()


def keyword_trim(setup, end_phrases) -> Callable[[str], str]:
    """A turn's trim (Dictation.trim): the words heard without the keyword in front and the end phrase at the end."""
    api, phrases = wake_api(), tuple(end_phrases)

    def trim(text: str) -> str:
        return api.strip_end(api.strip_keyword(text, setup.keyword, setup.spelling), phrases)
    return trim


@dataclass
class WakeState:
    """What the Settings card shows (TrayApp.wake_state; PreviewApp makes its own)."""
    keyword: str = ""  # the keyword set up ("" before the setup)
    heard: int = 0  # how many of the setup's tries were heard
    tries: int = 10
    models: str = "missing"  # ready, missing or downloading
    percent: int = 0  # downloaded so far, while downloading
    size: int = 0  # bytes still to download
    problem: str = ""  # why the last download failed
    listening: str = ""  # the microphone listened to now ("" when it isn't listening)
    note: str = ""  # why it isn't listening or is paused ("" when it listens, or is off)


# ---------------------------------------------------------------- the listener's thread

class WakeFeeder:
    """Feeds a WakeListener on a thread of its own. The microphone's audio callback only queues its block (put); here it
    is brought to 16 kHz and cut into 100 ms frames, each fed with its time on the audio's own clock (frames that wait
    in the queue keep their spacing). Paused, frames are dropped, and the listener starts afresh after: what was said
    meanwhile wasn't heard. end() asks for the listener's end() between two frames, so it's never called during one."""

    def __init__(self, listener, clock: Callable[[], float] = time.monotonic):
        self.listener, self._clock = listener, clock
        self.paused = False
        self.dropped = 0  # blocks dropped because the queue was full
        self._blocks: queue.Queue = queue.Queue(maxsize=QUEUE_BLOCKS)
        self._end = threading.Event()
        self._stopping = threading.Event()
        self._rate, self._resample = 0, None
        self._pending = np.zeros(0, dtype=np.float32)
        self._since, self._fed = 0.0, 0  # the clock when this stretch of frames began, and samples fed since
        self._was_paused = False
        self._errors = 0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="wake-word", daemon=True)
        self._thread.start()

    def put(self, block: np.ndarray, rate: int) -> None:
        """Recorder.tap: on the audio thread, so only a queue put."""
        try:
            self._blocks.put_nowait((block, rate))
        except queue.Full:
            self.dropped += 1

    def end(self) -> None:
        self._end.set()

    def stop(self) -> None:
        self._stopping.set()
        if self._thread is not None:
            self._thread.join(1.0)

    def _run(self) -> None:
        while not self._stopping.is_set():
            try:
                block, rate = self._blocks.get(timeout=0.2)
            except queue.Empty:
                if self._end.is_set():
                    self.handle(np.zeros(0, dtype=np.float32), self._rate or TARGET_RATE)
                continue
            self.handle(block, rate)

    def handle(self, block: np.ndarray, rate: int) -> None:
        """One block from the microphone (the thread calls it; the tests too)."""
        if self._end.is_set():
            self._end.clear()
            self._call(self.listener.end)
        if self.paused:
            self._was_paused = True
            return
        if self._was_paused:
            self._was_paused = False
            self._restart()
            self._call(self.listener.end)
        if rate != self._rate:
            self._rate, self._resample = rate, StreamResampler(rate)
            self._restart()
        if not len(block):
            return
        self._pending = np.concatenate([self._pending, self._resample(block)])
        while len(self._pending) >= FRAME:
            frame, self._pending = self._pending[:FRAME], self._pending[FRAME:]
            if not self._fed:
                self._since = self._clock()
            self._fed += FRAME
            self._call(self.listener.feed, frame, self._since + self._fed / TARGET_RATE)

    def _restart(self) -> None:
        self._pending, self._fed = np.zeros(0, dtype=np.float32), 0

    def _call(self, method, *args) -> None:
        try:
            method(*args)
        except Exception:  # keep listening; say it in the log, but not for every frame
            self._errors += 1
            if self._errors <= 3 or self._errors % 1000 == 0:
                log.exception("The wake word's listener failed (%d times)", self._errors)


# ---------------------------------------------------------------- the turns

class WakeController(QObject):
    """Listens while the wake word is on and set up, and runs the turns it starts (on the UI thread; the listener's
    callbacks arrive from the feeder's thread through signals). `app` is the TrayApp: its settings, profile, dictation,
    recorder, live translation and Text Transform. Nothing here polls: tick() comes with the keys' events and the app's
    once-a-second housekeeping, which runs while a microphone is open (the wake word keeps one open: wants_ticks).

    The listener is made on a thread (`background`): its spotter loads a model the first time (0.8-2.2 s in the spike),
    which mustn't hold up the tray or the window. on_change() is called when what the Settings card shows changes."""

    _woke = Signal()
    _paused = Signal()
    _ended = Signal(bool)
    _checked = Signal(int, object)  # (turn, the voice command it was, or None)

    def __init__(self, app, *, microphones: Callable[[], list[str]] | None = None,
                 default_microphone: Callable[[], str] | None = None,
                 call_quality: Callable[[str | None], bool] | None = None,
                 new_recorder: Callable[[str | None], Recorder] | None = None,
                 clock: Callable[[], float] = time.monotonic, background=None):
        super().__init__()
        self.app = app
        self.background = background or (lambda work, done: _in_background(self, work, done))
        self.on_change: Callable[[], None] | None = None
        # Looked up when used (sst.audio's by default), so a test can replace them after the app is built
        self._microphones = microphones or (lambda: audio.input_device_names())
        self._default = default_microphone or (lambda: audio.default_microphone())
        self._call_quality = call_quality or (lambda device: audio.call_quality(device))
        self._new_recorder = new_recorder or (lambda device: Recorder(device))
        self._clock = clock
        self.setup = None  # the profile's sst.wake.WakeSetup, None before the setup
        self.listener = None  # sst.wake.WakeListener while listening
        self.feeder: WakeFeeder | None = None
        self.recorder: Recorder | None = None  # the microphone listened to: dictation's own, or one of its own
        self.shared = True  # listening on dictation's microphone (else a wired one, a Bluetooth headset being chosen)
        self.microphone = ""  # the name of the microphone listened to, for the card
        self.problem = ""  # why it doesn't listen
        self.note = ""  # why it's paused now
        self.active = False  # a turn is being recorded
        self.turn = 0
        self.setting_up = False  # the setup's tries are being recorded: they mustn't start a dictation
        self._trim: Callable[[str], str] | None = None  # this turn's
        self._phrases: list[str] = []  # the end phrases listened for
        self._key: tuple | None = None  # what the listener was made for
        self._making: tuple | None = None  # the key of a listener being made on a thread
        self._made: tuple | None = None  # (key, listener) made and not yet listening
        self._shown: tuple | None = None  # what on_change last told
        self._began = 0.0
        self._looked = 0.0
        self._woke.connect(self._on_wake)
        self._paused.connect(self._on_pause)
        self._ended.connect(self._on_end)
        self._checked.connect(self._on_checked)

    # -- the setup and whether to listen

    def load(self) -> None:
        """The profile's setup (at start-up and when the profile changes); a damaged one counts as none."""
        data = read_wake(self.app.profile.wake_file)
        self.setup = None
        if data is not None:
            try:
                self.setup = wake_api().WakeSetup.from_json(data)
            except Exception as e:
                log.warning("The wake word's setup couldn't be read (%s): set it up again", type(e).__name__)

    @property
    def listening(self) -> bool:
        return self.listener is not None

    def models_ready(self) -> bool:
        try:
            api = wake_api()
            return api.KWS.installed() and api.VOICE.installed()
        except Exception:
            return False

    def choose_microphone(self) -> tuple[bool, str | None, str, str]:
        """Where the wake word listens: (on dictation's own microphone, the device to open (None: Windows' default),
        its name, why it can't listen). A Bluetooth headset's microphone is never kept open: another one that isn't
        one listens instead (Windows' default first), and with none, nothing does."""
        chosen = self.app.settings.microphone
        default = self._default()
        if not self._call_quality(chosen or None):
            return True, chosen or None, chosen or default, ""
        others = [name for name in self._microphones() if name != chosen and not self._call_quality(name)]
        pick = default if default in others else others[0] if others else ""
        if not pick:
            return False, None, "", BLUETOOTH
        return False, pick, pick, ""

    @property
    def wants_ticks(self) -> bool:
        """tick() is needed once a second: listening (on an open microphone), or waiting for a microphone to listen on."""
        return self.listening or self.active or self.problem == BLUETOOTH or self.problem.startswith("Couldn't open")

    def apply(self) -> None:
        """Listen, or stop, as the settings, the setup, the models and the microphones say now. Called whenever one of
        them may have changed (settings saved, profile switched, model loaded or downloaded, setup saved). Dictation's
        microphone gets its own settings back from TrayApp._configure_dictation_mic, which calls this after."""
        try:
            self._apply()
        finally:
            self._changed()

    def _apply(self) -> None:
        s, app = self.app.settings, self.app
        self.problem = ""
        if not s.wake_enabled:
            was = self.listener is not None or self._making is not None or self._made is not None
            self._stop_listening()
            self._made = None
            if was:
                _unload()
            return
        if not self.models_ready():
            self.problem = "Download the voice models to start."
        elif self.setup is None:
            self.problem = "Set up your keyword to start."
        elif app.dictation is None:
            self.problem = "Waiting for the speech model."
        if self.problem:
            self._stop_listening()
            return
        shared, device, name, problem = self.choose_microphone()
        if problem:
            self._stop_listening()
            self.problem = problem
            return
        phrases = [p for p in s.wake_end_phrases if p.strip()] or ["stop it"]
        key = (repr(self.setup.to_json()), tuple(phrases), s.wake_voice_check, shared, device, s.raw_audio)
        if self.listener is None or key != self._key:
            if self.listener is None and self._making == key:
                return  # being made: apply() comes again when it's ready
            self._stop_listening()
            made, self._made = self._made, None
            if made is None or made[0] != key:
                self._make(key, phrases, s.wake_voice_check)  # apply() again once it's made
                return
            listener = made[1]
            listener.on_wake, listener.on_pause, listener.on_end = self._woke.emit, self._paused.emit, self._ended.emit
            recorder = app.recorder if shared else self._new_recorder(device)
            if not shared:
                recorder.raw, recorder.tail = s.raw_audio, app.recorder.tail  # the same short tail as dictation's
            self.listener, self.feeder, self.recorder = listener, WakeFeeder(listener, self._clock), recorder
            self.shared, self.microphone, self._key, self._phrases = shared, name, key, phrases
            self.feeder.start()
            log.info("Wake word: listening on %s (%s; voice check %s)", "dictation's microphone" if shared else
                     "another microphone, not the Bluetooth headset's", name or "Windows' default",
                     "on" if s.wake_voice_check else "off")
        try:
            self._hold_open()  # also after dictation's microphone was set up again: it stays open for listening
        except Exception as e:  # no microphone: said on the card, tried again at the next look
            log.warning("The wake word couldn't open the microphone: %s", e)
            self._stop_listening()
            self.problem = f"Couldn't open the microphone: {e}"

    def _make(self, key: tuple, phrases: list[str], voice_check: bool) -> None:
        """The listener for `key`, made on a thread; then apply() again, unless something else is wanted by then."""
        self._making, setup = key, self.setup

        def done(listener, error) -> None:
            if self._making != key:
                return  # switched off, or changed meanwhile: another one is being made, or none is wanted
            self._making = None
            if error is not None:
                log.error("The wake word couldn't start: %s", error)
                self.problem = f"The wake word couldn't start: {error}"
                self._changed()
                return
            self._made = (key, listener)
            self.apply()
        self.background(lambda: wake_api().WakeListener(setup, phrases, voice_check=voice_check), done)

    def _changed(self) -> None:
        shown = (self.listening, self.microphone, self.problem, self.note)
        if shown != self._shown:
            self._shown = shown
            if self.on_change is not None:
                self.on_change()

    def _hold_open(self) -> None:
        """The microphone listened to stays open, with PREROLL seconds kept, and taps into the feeder."""
        r = self.recorder
        r.warm_seconds = math.inf
        r.preroll_seconds = max(r.preroll_seconds, PREROLL)
        r.tap = self.feeder.put
        r.keep_open()

    def _stop_listening(self) -> None:
        if self.active:  # a turn going on: what was said is typed (its end phrase can't be heard any more)
            dictation = self.app.dictation
            if dictation is not None and dictation.recording:
                dictation.end(keep=True)
            self._finish("listening stopped")
        if self.recorder is not None:
            self.recorder.tap = None
            if not self.shared:
                self.recorder.close()
        if self.feeder is not None:
            self.feeder.stop()
        if self.listener is not None:
            log.info("Wake word: stopped listening")
        self.listener, self.feeder, self.recorder, self._key, self.microphone = None, None, None, None, ""
        self._making = None  # a listener still being made is dropped when it's ready (apply() makes the one wanted)

    def stop(self) -> None:
        """Rflow quits or starts over."""
        self._stop_listening()

    # -- with the keys' events and the app's once-a-second housekeeping (on the UI thread)

    def pause_reason(self) -> str:
        app = self.app
        if self.setting_up:
            return "Paused while you set up the keyword."
        live = getattr(app, "live", None)
        if live is not None and live.running and MIC in live.config.lanes:
            return "Paused while live translation uses the microphone."
        dictation = app.dictation
        if dictation is not None and dictation.recording and not self.active:
            return "Paused while you dictate with the keys."
        transforms = getattr(app, "transforms", None)
        if transforms is not None and (transforms.busy or transforms.menu.isVisible()):
            return "Paused while Text Transform works."
        return ""

    def tick(self, now: float) -> None:
        dictation = self.app.dictation
        if self.active and (dictation is None or not dictation.recording):
            # Ctrl+Win, Esc, the time limit: the dictation ended by itself, and the listener goes back to waiting
            self._finish("ended with the keys or the time limit")
            if self.feeder is not None:
                self.feeder.end()
        if self.feeder is not None:
            self.note = self.pause_reason()
            self.feeder.paused = bool(self.note)
        else:
            self.note = ""
        if self.recorder is not None and not self.shared and not self.active:
            self.recorder.tick(now)  # its own microphone follows Windows' devices, as dictation's does
        if now - self._looked >= CHECK_EVERY:
            self._looked = now
            self._look()
        self._changed()

    def _look(self) -> None:
        """A microphone that closed (a Bluetooth headset became Windows' default), or a Bluetooth headset that now has
        a wired microphone to stand in for it: listen where it should be."""
        if not self.app.settings.wake_enabled or self.active:
            return
        if self.listening and self.recorder is not None and not self.recorder.warm and not self.recorder.busy:
            self.apply()
        elif not self.listening and (self.problem == BLUETOOTH or self.problem.startswith("Couldn't open")):
            self.apply()

    # -- a turn

    def _on_wake(self) -> None:
        dictation = self.app.dictation
        if self.listener is None or dictation is None:
            return
        if dictation.recording or self.pause_reason():
            log.info("Wake word heard while busy: ignored")
            self.feeder.end()
            return
        self.turn += 1
        self._trim = keyword_trim(self.setup, self._phrases)
        if not self.shared:
            dictation.recorder = self.recorder  # the turn records where the keyword was heard, with its moment before
        dictation.trim = self._trim
        dictation.handle("handsfree", self._clock())  # exactly Ctrl+Win+Space: the beep, the pill, Esc cancels
        dictation.trim = None  # it went with this recording
        if not dictation.recording:
            log.info("Wake word: turn %d couldn't start", self.turn)
            self._restore()
            self.feeder.end()
            return
        self.active, self._began = True, self._clock()
        log.info("Wake word: turn %d started", self.turn)

    def _on_end(self, keep: bool) -> None:
        if not self.active:
            return  # ended already (Ctrl+Win, Esc, a command)
        dictation = self.app.dictation
        if dictation is not None and dictation.recording:
            dictation.end(keep=keep, quiet=not keep)  # a voice that isn't the user's: dropped without a sound
        self._finish("the end phrase or a long silence: typed" if keep else "not the voice set up: dropped")

    def _on_pause(self) -> None:
        """A few words, then a pause: if they were a voice command, it runs now, without waiting for the end phrase."""
        dictation = self.app.dictation
        if not self.active or dictation is None or not dictation.recording or dictation.command is None:
            return
        take = getattr(dictation.recorder, "current_take", None)
        if take is None:
            return
        turn, chunks, preroll, rate = self.turn, list(take.chunks), take.preroll, take.rate
        engine, command, trim = dictation.engine, dictation.command, self._trim

        def work() -> None:
            found = None
            try:
                sound = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
                sound = sound[trim_preroll(sound, rate, preroll, VoiceConfig().vad):]  # what the dictation would hear
                heard = engine.transcribe(sound, rate) if len(sound) else ""
                found = command(trimmed(trim, heard)) if heard else None
            except Exception as e:
                log.warning("Wake word: checking for a voice command failed (%s)", type(e).__name__)
            self._checked.emit(turn, found)
        threading.Thread(target=work, name="wake-command", daemon=True).start()

    def _on_checked(self, turn: int, command) -> None:
        dictation = self.app.dictation
        if not command or turn != self.turn or not self.active or dictation is None or not dictation.recording:
            return  # dictation goes on
        dictation.end(keep=False, quiet=True)  # not typed, not saved: a command, not a dictation
        self._finish(f"a voice command ({command})")
        if self.feeder is not None:
            self.feeder.end()
        self.app.transforms.run_command(command)  # on the text box, exactly as Ctrl+Win + the command

    def _finish(self, why: str) -> None:
        self.active = False
        self._restore()
        log.info("Wake word: turn %d ended after %.1f s (%s)", self.turn, self._clock() - self._began, why)

    def _restore(self) -> None:
        dictation = self.app.dictation
        if dictation is not None and dictation.recorder is not self.app.recorder and not dictation.recording:
            dictation.recorder = self.app.recorder


# ---------------------------------------------------------------- the 10-try setup

class _Relay(QObject):
    done = Signal(object, object)  # (result, error)


def _in_background(owner: QObject, work, done) -> None:
    """work() on a thread (models load, the spotter and the voice print run), then done(result, error) on the UI's."""
    relay = _Relay(owner)
    relay.done.connect(lambda result, error: (done(result, error), relay.deleteLater()),
                       Qt.ConnectionType.QueuedConnection)

    def run() -> None:
        try:
            relay.done.emit(work(), None)
        except Exception as e:
            relay.done.emit(None, e)
    threading.Thread(target=run, name="wake-setup", daemon=True).start()


class _RecordOrb(Orb):
    """The voice orb as the big record button: mint while waiting, coral with the voice's level while recording."""
    clicked = Signal()

    def __init__(self, level):
        super().__init__(112, level)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Record (Space)")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()


def _level(rms: float) -> float:
    """A recorder's loudness as 0..1 (speech is roughly -45..-15 dBFS)."""
    return min(1.0, max(0.0, (20 * math.log10(rms + 1e-6) + 55) / 40))


class WakeSetupDialog(QDialog):
    """The keyword typed, then said ten ways, one at a time (sst.wake.Enrollment's prompts), each followed by heard or
    not heard: a try not heard counts as one of the ten, and that prompt may be tried once more. Saved when heard at
    least `passes` times; else refused, with advice. The tries are used in memory only: nothing is recorded to disk.

    check(text): why the keyword can't be used, or None. enroll(keyword): an Enrollment (built on a thread: the spotter
    loads its model). new_recorder(): the microphone (the Bluetooth rule applied by the app). save(setup): keep it."""

    MAX_SECONDS = 6.0  # a try stops by itself

    def __init__(self, parent=None, *, microphone: str, problem: str, check, enroll, new_recorder, save,
                 tries: int = 10, passes: int = 8):
        super().__init__(parent)
        self._check, self._enroll, self._new_recorder, self._save = check, enroll, new_recorder, save
        self.tries, self.passes, self.problem = tries, passes, problem
        self.enrollment = None
        self.recorder = None
        self.recording = False
        self.busy = False
        self.results: list[bool] = []  # each try: heard or not
        self.prompt = 0  # which of the prompts is said now
        self.retried = False  # this prompt was tried again already
        self.saved = None  # the WakeSetup once saved
        self._closed = False
        theme.load_fonts()
        theme.make_host(self)
        self.setWindowTitle("Rflow: set up your keyword")
        self.setStyleSheet(theme.stylesheet())
        self.setFont(theme.font(14))
        self.setFixedSize(560, 540)
        self.pages = QStackedWidget()
        self.pages.addWidget(self._keyword_page(microphone))
        self.pages.addWidget(self._tries_page())
        self.pages.addWidget(self._result_page())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.addWidget(self.pages)
        self._meter = QTimer(self, interval=50, timeout=self._show_level)
        self._limit = QTimer(self, singleShot=True, timeout=self._stop_try)
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self._space)
        self._keyword_changed()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), theme.tok("base"))
        theme.paint_hosted(self, p)  # the cards' soft depth
        p.end()

    # -- 1: the keyword

    def _keyword_page(self, microphone: str) -> QWidget:
        page, layout = card(12, (24, 24, 24, 24))
        layout.addWidget(label("Your keyword", "title"))
        layout.addWidget(label("Say it in any app to start dictating, then say your end phrase to type the text. Choose "
                               "words you don't say while dictating: two or three syllables are heard best.", "caption",
                               "2"))
        self.keyword = QLineEdit()
        self.keyword.setPlaceholderText("okay type")
        theme.surface(self.keyword, "field")
        self.keyword.textChanged.connect(self._keyword_changed)
        self.keyword.returnPressed.connect(self._start)
        layout.addWidget(self.keyword)
        self.warning = label("", "caption", "warn")
        layout.addWidget(self.warning)
        chips = QHBoxLayout()
        chips.setSpacing(8)
        chips.addWidget(label("Heard well:", "caption", "3", wrap=False))
        for suggestion in SUGGESTIONS:
            chips.addWidget(button(suggestion, lambda _=False, s=suggestion: self.keyword.setText(s), kind="chip",
                                   size="sm"))
        chips.addStretch()
        layout.addLayout(chips)
        layout.addWidget(label("A short name alone, like “Rflow”, is often missed: in a phrase it's heard better.",
                               "caption", "3"))
        layout.addStretch()
        mic = QHBoxLayout()
        mic.setSpacing(8)
        self.mic_lamp = Lamp("warn" if self.problem else "ok")
        self.mic_words = label(self.problem or f"Microphone: {microphone or 'Windows default'}", "caption", "2")
        mic.addWidget(self.mic_lamp, 0, Qt.AlignmentFlag.AlignTop)
        mic.addWidget(self.mic_words, 1)
        layout.addLayout(mic)
        layout.addWidget(label("Your tries are used in memory to learn the keyword and your voice; no recording is kept.",
                               "caption", "3"))
        self.start_button = button("Start the 10 tries", self._start, primary=True)
        footer = QHBoxLayout()
        footer.addWidget(button("Cancel", self.reject, kind="quiet"))
        footer.addStretch()
        footer.addWidget(self.start_button)
        layout.addLayout(footer)
        return page

    def _keyword_changed(self, *_) -> None:
        text = self.keyword.text().strip()
        why = self._check(text) if text else None
        self.warning.setText(why or "")
        self.warning.setVisible(bool(why))
        self.start_button.setEnabled(bool(text) and not why and not self.problem and not self.busy)

    def _start(self) -> None:
        if not self.start_button.isEnabled():
            return
        keyword = self.keyword.text().strip()
        self.busy = True
        self.start_button.setEnabled(False)
        self.results, self.prompt, self.retried = [], 0, False
        self.pages.setCurrentIndex(1)
        self._show_try("Getting ready…", "")
        _in_background(self, lambda: self._enroll(keyword), self._ready)

    # -- 2: the ten tries

    def _tries_page(self) -> QWidget:
        page, layout = card(14, (24, 24, 24, 24))
        top = QHBoxLayout()
        self.counter = label("", "caps", "3", wrap=False)
        top.addWidget(self.counter)
        top.addStretch()
        self.dots = [Lamp("off") for _ in range(self.tries)]
        for dot in self.dots:
            top.addWidget(dot)
        layout.addLayout(top)
        self.prompt_label = label("", "heading")
        self.prompt_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prompt_label.setMinimumHeight(56)
        layout.addWidget(self.prompt_label)
        self.orb = _RecordOrb(lambda: _level(self.recorder.level) if self.recording and self.recorder else 0.0)
        self.orb.clicked.connect(self._toggle)
        layout.addWidget(self.orb, 0, Qt.AlignmentFlag.AlignHCenter)
        controls = QHBoxLayout()
        controls.addStretch()
        self.record_button = button("Record", self._toggle, primary=True, size="lg", icon="mic")
        controls.addWidget(self.record_button)
        self.level = Meter()
        self.level.setToolTip("Say something: the bars light up.")
        controls.addWidget(self.level, 0, Qt.AlignmentFlag.AlignVCenter)
        controls.addStretch()
        layout.addLayout(controls)
        verdict = QHBoxLayout()
        verdict.addStretch()
        self.verdict_icon = QLabel()
        self.verdict = label("", tone="2", wrap=False)
        verdict.addWidget(self.verdict_icon)
        verdict.addWidget(self.verdict)
        verdict.addStretch()
        layout.addLayout(verdict)
        layout.addStretch()
        footer = QHBoxLayout()
        footer.addWidget(button("Cancel", self.reject, kind="quiet"))
        footer.addStretch()
        layout.addLayout(footer)
        return page

    def _ready(self, enrollment, error) -> None:
        if self._closed:
            return
        self.busy = False
        if error is not None:
            log.warning("The wake word's setup couldn't start: %s", error)
            self.pages.setCurrentIndex(0)
            self._keyword_changed()
            self.warning.setText(f"The setup couldn't start: {error}")
            self.warning.show()
            return
        self.enrollment = enrollment
        try:
            self.recorder = self._new_recorder()
            self.recorder.warm_seconds = math.inf  # open from now on: a try starts at once, with the moment before it
            self.recorder.keep_open()
        except Exception as e:
            self._show_try(f"Couldn't open the microphone: {e}", "err")
            self.record_button.setEnabled(False)
            return
        self._show_try("", "")

    def _show_try(self, message: str, tone: str) -> None:
        """The try's page as it is now; `message`: a word under the button (tone ok, err, or "" for a plain one)."""
        n = len(self.results)
        self.counter.setText(f"Try {min(n + 1, self.tries)} of {self.tries}")
        for dot, heard in zip(self.dots, self.results + [None] * self.tries, strict=False):
            dot.set_state("off" if heard is None else "ok" if heard else "err")
        prompts = self.enrollment.prompts if self.enrollment is not None else []
        self.prompt_label.setText(prompts[self.prompt] if self.prompt < len(prompts) else "")
        self.record_button.setText("Stop" if self.recording else "Try again" if tone == "err" and self.retried
                                   else "Record")
        self.record_button.setEnabled(self.enrollment is not None and not self.busy)
        self.orb.set_state("live" if self.recording else "ai" if self.busy else "ready")
        if not message and not self.recording and self.enrollment is not None:
            message = "Press Record (or Space), say it, then press Stop."
        icon = {"ok": "check", "err": "close"}.get(tone)
        if icon:
            self.verdict_icon.setPixmap(theme.icon_pixmap(icon, theme.tok(tone).name(), 18, self.devicePixelRatioF()))
        self.verdict_icon.setVisible(bool(icon))
        self.verdict.setText(message)
        set_tone(self.verdict, tone or "2")

    def _space(self) -> None:
        if self.pages.currentIndex() == 1 and self.record_button.isEnabled():
            self._toggle()

    def _toggle(self) -> None:
        if self.recording:
            self._stop_try()
            return
        if self.enrollment is None or self.busy or self.recorder is None:
            return
        try:
            self.recorder.start()
        except Exception as e:
            self._show_try(f"Couldn't open the microphone: {e}", "err")
            return
        self.recording = True
        self._meter.start()
        self._limit.start(int(self.MAX_SECONDS * 1000))
        self._show_try("Listening… press Stop when you've said it.", "")

    def _show_level(self) -> None:
        if self.recorder is not None:
            self.level.set_value(_level(self.recorder.level))

    def _stop_try(self) -> None:
        if not self.recording:
            return
        self.recording = False
        self._meter.stop()
        self._limit.stop()
        self.level.set_value(0)
        sound, rate = self.recorder.stop(), self.recorder.rate
        if len(sound) < 0.3 * rate:
            self._show_try("Too short: press Record, then say it.", "")
            return
        self.busy = True
        self._show_try("Checking…", "")
        sound = resample(sound, rate)  # 16 kHz, as the spotter and the voice print want it
        _in_background(self, lambda: self.enrollment.add(sound), self._added)

    def _added(self, heard, error) -> None:
        if self._closed:
            return
        self.busy = False
        if error is not None:
            log.warning("A try couldn't be checked: %s", error)
            self._show_try(f"That try couldn't be checked ({error}). Press Record to say it again.", "err")
            return
        heard = bool(heard)
        self.results.append(heard)
        if heard:
            self.prompt, self.retried, message, tone = self.prompt + 1, False, "Heard", "ok"
        elif not self.retried:
            self.retried, message, tone = True, "Not heard: try again", "err"  # the same prompt, once more
        else:
            self.prompt, self.retried, message, tone = self.prompt + 1, False, "Not heard again: on to the next", "err"
        log.info("Wake word setup: try %d %s", len(self.results), "heard" if heard else "not heard")
        if getattr(self.enrollment, "done", False) or len(self.results) >= self.tries:
            self._finish()
            return
        self._show_try(message, tone)

    # -- 3: the result

    def _result_page(self) -> QWidget:
        page, layout = card(12, (24, 24, 24, 24))
        layout.addStretch()
        self.result_orb = Orb(96)
        self.result_orb.set_state("ai")
        layout.addWidget(self.result_orb, 0, Qt.AlignmentFlag.AlignHCenter)
        self.result_title = label("Learning your voice…", "title")
        self.result_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result_title)
        self.result_words = label("", tone="2")
        self.result_words.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result_words)
        self.result_advice = label("", "caption", "2")
        self.result_advice.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result_advice)
        layout.addStretch()
        footer = QHBoxLayout()
        self.again_button = button("Start again", self._again, kind="quiet")
        self.done_button = button("Done", self.accept, primary=True)
        footer.addWidget(self.again_button)
        footer.addStretch()
        footer.addWidget(self.done_button)
        layout.addLayout(footer)
        return page

    def _finish(self) -> None:
        self.busy = True
        self._close_recorder()
        self.pages.setCurrentIndex(2)
        self.result_orb.set_state("ai")
        self.result_title.setText("Learning your voice…")
        self.result_words.setText("")
        self.result_advice.setText("")
        self.again_button.hide()
        self.done_button.setEnabled(False)
        _in_background(self, self.enrollment.result, self._finished)

    def _finished(self, setup, error) -> None:
        if self._closed:
            return
        self.busy = False
        keyword = self.keyword.text().strip()
        heard = getattr(self.enrollment, "heard", sum(self.results))
        if error is None and setup is not None:
            try:
                self._save(setup)
                self.saved = setup
            except Exception as e:
                error = e
        self.done_button.setEnabled(True)
        self.done_button.setFocus()
        if self.saved is not None:
            log.info("Wake word set up: heard %d of %d", heard, self.tries)
            self.result_orb.set_state("done")
            self.result_title.setText("Saved")
            self.result_words.setText(f"Rflow heard “{keyword}” {heard} of {self.tries} times.")
            self.result_advice.setText("Say it in any app to start dictating, and your end phrase to type the text.")
            self.done_button.setText("Done")
            return
        self.result_orb.set_state("off")
        self.again_button.show()
        self.done_button.setText("Close")
        if error is not None:
            log.warning("The wake word's setup couldn't be finished: %s", error)
            self.result_title.setText("Not saved")
            self.result_words.setText(f"The setup couldn't be finished: {error}")
            self.result_advice.setText("Try again in a moment.")
            return
        log.info("Wake word setup refused: heard %d of %d", heard, self.tries)
        self.result_title.setText("Not saved")
        self.result_words.setText(f"Rflow heard “{keyword}” only {heard} of {self.tries} times.")
        self.result_advice.setText(f"It needs to be heard at least {self.passes} times. Try a longer keyword (two or three "
                                   "syllables, like “okay type”), or a quieter room.")

    def _again(self) -> None:
        self.enrollment = None
        self.pages.setCurrentIndex(0)
        self.keyword.selectAll()
        self.keyword.setFocus()
        self._keyword_changed()

    # -- closing

    def _close_recorder(self) -> None:
        self._meter.stop()
        self._limit.stop()
        if self.recorder is not None:
            if self.recording:
                self.recording = False
                self.recorder.stop()  # dropped
            self.recorder.close()
        self.recorder = None

    def done(self, result: int) -> None:
        self._closed = True
        self._close_recorder()
        super().done(result)
