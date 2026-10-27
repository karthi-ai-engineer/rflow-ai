"""Rflow, the app: the window (sst/window.py), a tray icon, and dictation without a console window.

Opening Rflow shows its window; closing the window keeps Rflow in the tray, where dictation goes on working. While
you dictate, a small "pill" near the bottom of the screen shows the recording level, then the transcribing, then
"Typed". The dictation itself is sst.dictate.Dictation, the same as the console command.

  uv run sst app                        from the source checkout
  Rflow.exe                             the installed app (the Start menu shortcut); again: shows the running window
  Rflow.exe --startup                   at sign-in: only the tray icon
  Rflow.exe --self-test                 build check: builds every window off-screen and transcribes once
"""
import contextlib
import ctypes
import dataclasses
import faulthandler
import gc
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import date, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QCursor, QFont, QFontMetrics, QGuiApplication, QIcon, QPainter
from PySide6.QtNetwork import QLocalServer
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon, QWidget

from sst import DOWNLOADS_DIR, __version__, bench, downloads, instance, scan, shell, textaccess, theme, updates, usage
from sst.audio import PREROLL_SECONDS, TAIL_SECONDS, Recorder, default_microphone, input_device_names
from sst.commands import UNDO, match_command, phrases_for
from sst.dictate import DEFAULT_HOTKEY, Dictation, wispr_flow_running
from sst.engines import DEFAULT_MODEL, SPEECH_MODELS, load_engine, usable
from sst.engines.cloud import CLOUD, REMOTE, SERVER, CloudEngine
from sst.gateway import PROVIDERS, SPEECH_SERVER, GatewayConfig, Polisher
from sst.hotkey import HotkeyListener, parse_hotkey
from sst.live.captions import LiveCaptions
from sst.live.contracts import MIC, Kind, LiveConfig, LiveEvent
from sst.live.ducking import Ducker, put_back_left
from sst.live.gemini import GeminiLiveTranslate
from sst.live.session import LiveSession
from sst.live.speaker import Speaker
from sst.live.transcript import Transcript, for_copy
from sst.live.voice import LOCAL as LOCAL_VOICE
from sst.live.voice import PiperVoice, Voice
from sst.live.voice import install as install_voice
from sst.live.voice import local as local_voice
from sst.live.wasapi import Capture, Player
from sst.pipeline.asr import ASRScheduler, EngineBackend
from sst.pipeline.contracts import VoiceConfig
from sst.pipeline.dictionary import DictionaryEngine, DictionaryStore, TermMode, speech_hints
from sst.pipeline.formatting import Formatter
from sst.pipeline.guard import Guard
from sst.pipeline.learning import CorrectionEvent, Learner
from sst.pipeline.polish import GatewayLLM
from sst.pipeline.session import LazyBackend, Stages, VoicePipeline
from sst.settings import (
    CONFIG_DIR,
    LOAD_PROBLEMS,
    Profiles,
    Settings,
    Stats,
    add_to_history,
    read_history,
    write_wake,
)
from sst.snippets import Snippet
from sst.snippets import load as load_snippets
from sst.transform import Context as TransformContext
from sst.transform import Transformer
from sst.transformui import TransformController
from sst.translate import Translator, system_language
from sst.translateui import TranslateController
from sst.wakeui import WakeController, WakeSetupDialog, WakeState, WakeTools, wake_api
from sst.window import APP_NAME, ICON_FILE, LOG_DIR, MainWindow, PreviewApp

if TYPE_CHECKING:
    from sst import evaluate  # imported when a reading test is scored (score_reading), not at every start

UPDATE_DIR = Path(os.environ.get("TEMP", Path.home())) / "Rflow-update"  # downloaded installers
# The sentence models are tried on (the scan, a cloud or server card's Test): 7.4 s of a LibriVox reading, as in
# sherpa-onnx's model packages. It comes with Rflow, since Parakeet may not be downloaded.
SAMPLE_FILE = Path(__file__).resolve().parent / "static" / "sample.wav"
SERVER_NAME = instance.SERVER_NAME  # the local pipe a second Rflow asks to show the window (sst/instance.py)
WARM_SECONDS = 300  # the microphone stays open this long after a dictation (the owner chose 5 minutes)
TICK_MS = 1000  # dictation's housekeeping: none of it needs to be sooner (Dictation.tick, Recorder.tick)
DEBUG_DIR = LOG_DIR.parent / "debug"  # each dictation's stages and chunk audio, when Settings asks for them

log = logging.getLogger("sst.app")


def setup_logging() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(LOG_DIR / "sst.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[handler], force=True)
    return LOG_DIR


CRASH_FILE, RUNNING_FILE = "crash.log", "running"
_crash_file = None  # kept open for the whole run: faulthandler writes a native crash's traceback into it


def install_crash_handlers(log_dir: Path) -> str:
    """Errors nobody catches (in Qt slots, in threads) go to the log instead of nowhere: Rflow.exe has no console. A
    native crash (in onnxruntime, Qt...) leaves its traceback in crash.log, and a marker says Rflow is running until it
    quits. Returns the name the last run's crash report was kept under, if that run ended in a crash, else ""."""
    global _crash_file
    sys.excepthook = lambda kind, value, tb: log.critical("Unexpected error", exc_info=(kind, value, tb))
    threading.excepthook = lambda args: log.critical(
        "Unexpected error in %s", getattr(args.thread, "name", "a thread"),
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
    crash, running, left = log_dir / CRASH_FILE, log_dir / RUNNING_FILE, ""
    try:
        if running.exists() and crash.exists() and crashed(crash.read_text(encoding="utf-8", errors="replace")):
            kept = log_dir / f"crash-{time.strftime('%Y%m%d-%H%M%S')}.log"
            crash.replace(kept)
            left = kept.name
        running.write_text(str(os.getpid()), encoding="utf-8")
        _crash_file = crash.open("w", encoding="utf-8")
        faulthandler.enable(_crash_file)
    except OSError as e:
        log.warning("Couldn't set up the crash report: %s", e)
    return left


def crashed(report: str) -> bool:
    """What faulthandler wrote shows a real crash. On Windows it also writes exceptions that are raised and handled
    inside Windows (COM's 0x8001xxxx, e.g. 0x8001010d seen on the owner's laptop while the hook thread waits): those
    are no crash, and alone they don't make one."""
    if "Fatal Python error" in report:
        return True
    return any(not re.fullmatch(r"code 0x8001[0-9a-f]{4}", line.split(": ", 1)[1].strip(), re.IGNORECASE)
               for line in report.splitlines() if line.startswith("Windows fatal exception: "))


def mark_clean_exit(log_dir: Path) -> None:
    """Rflow quit on purpose, or Windows ended the session: there's no crash to report at the next start."""
    with contextlib.suppress(OSError):
        (log_dir / RUNNING_FILE).unlink()


def close_log_files() -> None:
    """Let go of the log and of the crash report, so their folder can be deleted (Start over). Nothing is logged after."""
    global _crash_file
    root = logging.getLogger()
    for handler in [h for h in root.handlers if isinstance(h, logging.FileHandler)]:
        root.removeHandler(handler)
        handler.close()
    if _crash_file is not None:
        faulthandler.disable()
        _crash_file.close()
        _crash_file = None


# ---------------------------------------------------------------- Start over: Rflow as if just installed

# What Rflow keeps: %APPDATA%\sst (every profile's settings, words, snippets, history, stats, keys, dictionary, live
# transcripts; the last scan) and %LOCALAPPDATA%\sst (logs, reading tests, the installed app's recordings, the debug
# steps, and the downloaded speech models and voice in "models"). The source code's recordings/ is never touched.
DATA_DIRS = (CONFIG_DIR, LOG_DIR.parent)
AFTER = instance.AFTER  # Rflow started again by Start over waits for the one that started it (its process id) to end
wait_for_exit = instance.wait_for_exit


def erase_data(folders=DATA_DIRS, keep: tuple[str, ...] = ()) -> list[Path]:
    """Delete everything in Rflow's data folders but the names in `keep` (e.g. "models"), and the folders themselves
    when nothing is kept. Only folders named "sst" are touched, whatever the environment says. Returns what couldn't be
    deleted (a file another program holds open)."""
    left = []
    for folder in map(Path, folders):
        if folder.name != "sst" or not folder.is_dir():
            continue
        for item in folder.iterdir():
            if item.name in keep:
                continue
            if item.is_dir() and not item.is_symlink():
                shutil.rmtree(item, ignore_errors=True)
            else:
                with contextlib.suppress(OSError):
                    item.unlink()
            if item.exists() or item.is_symlink():
                left.append(item)
        with contextlib.suppress(OSError):
            folder.rmdir()  # empty only: a kept folder keeps it
    return left


def restart_command(pid: int) -> list[str]:
    """How to start this Rflow again once process `pid` (this one) has ended: Rflow.exe, or the app from the source."""
    if getattr(sys, "frozen", False):
        return [sys.executable, f"{AFTER}{pid}"]
    return [sys.executable, "-c", "import sys; from sst.instance import launch; sys.exit(launch())", f"{AFTER}{pid}"]


# ---------------------------------------------------------------- the recording pill

class Pill(QWidget):
    """A small capsule near the bottom of the screen, in the design's popup look (always Obsidian): a coral waveform and
    the time while listening, violet dots while the AI cleans up, a mint check when typed. It never takes keyboard focus
    and ignores the mouse, so the dictated text still goes to the app you were typing in. Kept small, about two thirds of
    its first 40 x 176 px (the owner's wish of 2026-10-27: it hid what was under it); Qt scales it with the DPI."""

    HEIGHT = 28
    MIN_WIDTH = 104
    PAD = 12  # inside the capsule, on each side of its content
    GAP = 7  # between the waveform, dots or icon and the words
    MARGIN = 24  # transparent room around the capsule for its soft shadow (the "float" shadow reaches about 20 px)
    BARS, BAR_STEP, BAR_WIDTH = 16, 3.5, 2.0
    DOTS, DOT_STEP = 3, 8.0
    ICON = 12

    def __init__(self, level=lambda: 0.0):
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.WindowDoesNotAcceptFocus | Qt.WindowType.WindowTransparentForInput)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self._level = level
        self.state, self.message = "hidden", ""
        self.ai = False  # the AI cleanup runs after the speech model: "Cleaning up" in violet, else "Typing"
        self._levels: deque[float] = deque([0.0] * self.BARS, maxlen=self.BARS)
        self._phase, self._started = 0.0, 0.0
        self._animation = QTimer(self, interval=33, timeout=self._animate)
        self._hide_timer = QTimer(self, singleShot=True, timeout=self.hide)
        theme.load_fonts()
        self._font = theme.font(11, 500)
        self._mono = theme.font(11, 500, mono=True)

    def show_state(self, state: str, message: str = "") -> None:
        """recording, transcribing, typed, typed_raw, typed_local, cancelled, ignored, warning, error, or idle/hidden to
        hide."""
        self._hide_timer.stop()
        if state in ("idle", "hidden"):
            self.state = "hidden"
            self.hide()
            return
        if state == "recording" and self.state != "recording":
            self._started = time.monotonic()
        self.state, self.message = state, message
        if state == "recording":
            self._levels.extend([0.0] * self.BARS)
        self._fit()
        if not self.isVisible():
            self.show()
            _no_activate(int(self.winId()))
        self._animation.start()
        if state in ("typed", "cancelled", "ignored"):
            self._hide_timer.start(900)
        elif state == "transformed":
            self._hide_timer.start(1500)
        elif state in ("typed_raw", "typed_local"):
            self._hide_timer.start(1800)
        elif state in ("warning", "error"):
            self._hide_timer.start(4000)
        self.update()

    def hideEvent(self, event):
        self._animation.stop()
        super().hideEvent(event)

    def _fit(self) -> None:
        self.resize(self._width() + 2 * self.MARGIN, self.HEIGHT + 2 * self.MARGIN)
        self._place()

    def _timer(self) -> str:
        """The time since the recording started: 0:07, 27:41, 1:02:05 (a dictation has no length limit to speak of)."""
        seconds = int(time.monotonic() - self._started)
        hours, minutes = divmod(seconds // 60, 60)
        return f"{hours}:{minutes:02}:{seconds % 60:02}" if hours else f"{minutes}:{seconds % 60:02}"

    def _lead(self) -> float:
        """The width of what comes before the words: the waveform, the dots or the icon."""
        if self.state == "recording":
            return self.BARS * self.BAR_STEP - (self.BAR_STEP - self.BAR_WIDTH)
        if self.state in ("transcribing", "transforming"):
            return (self.DOTS - 1) * self.DOT_STEP + 5
        return self.ICON

    def _words(self) -> tuple[str, QFont]:
        return (self._timer(), self._mono) if self.state == "recording" else (self._text(), self._font)

    def _content(self) -> float:
        words, font = self._words()
        return self._lead() + self.GAP + QFontMetrics(font).horizontalAdvance(words)

    def _width(self) -> int:
        """The capsule's width (the window has MARGIN more on each side)."""
        return max(self.MIN_WIDTH, math.ceil(self._content()) + 2 * self.PAD)

    def _text(self) -> str:
        return {"typed": "Typed", "typed_raw": "Typed as heard · AI offline",
                "typed_local": "Typed with Parakeet · cloud offline", "cancelled": "Cancelled",
                "ignored": "Too short", "transcribing": "Cleaning up" if self.ai else "Typing",
                "transforming": "Transforming"}.get(self.state, self.message)

    def _place(self) -> None:
        screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
        area = screen.availableGeometry()
        self.move(area.center().x() - self.width() // 2, area.bottom() - self.HEIGHT - 20 - self.MARGIN)

    def _animate(self) -> None:
        self._phase += 0.033
        if self.state == "recording":
            rms = self._level()
            db = 20 * math.log10(rms + 1e-6)  # speech is roughly -45..-15 dBFS
            self._levels.append(min(1.0, max(0.0, (db + 55) / 40)))
            if self._width() != self.width() - 2 * self.MARGIN:  # the time gained a digit (10:00, 1:00:00)
                self._fit()
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        body = QRectF(self.MARGIN, self.MARGIN, self.width() - 2 * self.MARGIN, self.HEIGHT)
        theme.paint_surface(p, "float", body, self.HEIGHT / 2, popup=True, dpr=self.devicePixelRatioF())
        cy = body.center().y()
        dpr = self.devicePixelRatioF()
        x = body.center().x() - self._content() / 2
        text = QRectF(x + self._lead() + self.GAP, body.top(), body.width(), body.height())
        words, font = self._words()
        p.setPen(Qt.PenStyle.NoPen)
        if self.state == "recording":
            p.setBrush(theme.tok("live", popup=True))
            for i, level in enumerate(self._levels):
                h = 3 + level * 13
                p.drawRoundedRect(QRectF(x + i * self.BAR_STEP, cy - h / 2, self.BAR_WIDTH, h), 1, 1)
            p.setPen(theme.tok("text", popup=True))
        elif self.state in ("transcribing", "transforming"):
            colour = theme.tok("violet" if self.ai or self.state == "transforming" else "text2", popup=True)
            for i in range(self.DOTS):
                dot = QColor(colour)
                dot.setAlphaF(0.35 + 0.65 * max(0.0, math.sin(self._phase * 7 - i * 0.9)))
                p.setBrush(dot)
                p.drawEllipse(QPointF(x + 2.5 + i * self.DOT_STEP, cy), 2, 2)
            p.setPen(colour)
        else:
            half = self.ICON / 2
            if self.state in ("typed", "transformed", "cancelled", "ignored"):
                icon, colour = ("check", "ok") if self.state in ("typed", "transformed") else ("minus", "text2")
                p.drawPixmap(QPointF(x, cy - half), theme.icon_pixmap(icon, theme.tok(colour, popup=True).name(), self.ICON,
                                                                       dpr))
            else:  # a lamp: amber for a warning or a fallback, rose for an error
                p.setBrush(theme.tok("well", popup=True))
                p.drawEllipse(QPointF(x + half, cy), 4.5, 4.5)
                p.setBrush(theme.tok("err" if self.state == "error" else "warn", popup=True))
                p.drawEllipse(QPointF(x + half, cy), 2.3, 2.3)
            p.setPen(theme.tok("text2" if self.state in ("cancelled", "ignored") else "text", popup=True))
        p.setFont(font)
        p.drawText(text, Qt.AlignmentFlag.AlignVCenter, words)
        p.end()


def _no_activate(hwnd: int, click_through: bool = True) -> None:
    """Belt and braces on top of Qt's flags: a window that got focus would receive the paste instead of the user's app.
    `click_through` (the pill) also lets clicks pass to the window below; a popup with buttons must not: with it, every
    click on Translate's popup and the Text Transform menu went to the app behind them (until 1.10.1)."""
    user32 = ctypes.windll.user32
    user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
    user32.GetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int]
    user32.SetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
    GWL_EXSTYLE, WS_EX_NOACTIVATE, WS_EX_TRANSPARENT, WS_EX_TOOLWINDOW = -20, 0x08000000, 0x20, 0x80
    style = user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE) | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
    style = style | WS_EX_TRANSPARENT if click_through else style & ~WS_EX_TRANSPARENT
    user32.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, style)


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.CreateMutexW.restype = ctypes.c_void_p
_kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
_kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
_running_mutex = None


def hold_running_mutex() -> None:
    # While it exists, the installer asks to close Rflow before updating or uninstalling it. The name is from the
    # first versions; kept so that new installers still recognise a running old version.
    global _running_mutex
    _running_mutex = _kernel32.CreateMutexW(None, False, "SST-Dictation-running")


def release_running_mutex() -> None:
    global _running_mutex
    if _running_mutex:
        _kernel32.CloseHandle(_running_mutex)
        _running_mutex = None


def show_running_window() -> bool:
    """Ask a running Rflow to show its window; False if none answers (e.g. the console `sst dictate` or 1.1 and older)."""
    return instance.show_running_window(SERVER_NAME)


# ---------------------------------------------------------------- the app

MEETING_PARTS = ("translate", "record", "minutes", "screen")  # Settings.meeting_<part>, MeetingOptions' switches
ROOM_PARTS = ("translate", "record", "minutes")  # Settings.room_<part>: a meeting in person has no window to look at
MEETING_NOTICE = (
    "Meeting mode records the meeting's sound on this laptop: online, the others (what your laptop plays) and you "
    "(your microphone); in person, the room through your microphone. Tell the people in the meeting that it is "
    "recorded.\n\n"
    "For the minutes, the recording and (online) the meeting window's changed slides go to Google, which writes the transcript, "
    "the decisions and the action items with Gemini: about $0.25 an hour. Live translation, when on, costs as on its "
    "page. With a free Gemini key Google may use them to improve its products. Everything stays on this laptop "
    "afterwards; turn parts off on the Meeting mode page.\n\nStart?")


class _MeetingLive:
    """Live translation as a meeting's part (MeetingSession's `live`): its bar started with the meeting and stopped with
    it, unless it was already running (then it's left as it was). stop() may come on the meeting's watcher thread (the
    meeting window closed): the bar is stopped on the app's thread."""

    def __init__(self, app: "TrayApp", in_person: bool = False):
        self.app, self.started, self.in_person = app, False, in_person

    def start(self) -> None:
        if self.app.live.running:
            return
        problem = self.app.start_live(self.app.room_live_config() if self.in_person else None)
        if problem:
            raise RuntimeError(problem)
        self.started = True

    def stop(self) -> None:
        if self.started:
            self.started = False
            self.app.signals.meeting_live_stop.emit()

    def spend(self):
        return self.app.live.spend()


class _Signals(QObject):
    state = Signal(str, str)  # from Dictation, possibly on its worker thread
    result = Signal(str, str, float)  # (heard, typed, seconds of audio), from Dictation's worker thread
    command = Signal(str)  # a voice command for Text Transform ("make it concise"), from Dictation's worker thread
    loaded = Signal(object)
    failed = Signal(str, str)  # (speech model, why it couldn't load)
    update_found = Signal(object)  # the rest come from the update threads
    update_none = Signal()
    update_progress = Signal(str)
    update_ready = Signal(str)
    update_failed = Signal(str, bool)  # (message, tell the user)
    download_progress = Signal(str, int, int)  # (speech model, bytes done, bytes in all)
    download_done = Signal(str, str)  # (speech model, "" or "cancelled" or why it failed)
    voice_progress = Signal(int)  # percent of live translation's voice downloaded
    voice_done = Signal(str)  # "" or why the voice couldn't be downloaded
    wake_progress = Signal(int)  # percent of the wake word's models downloaded
    wake_done = Signal(str)  # "" or "cancelled" or why they couldn't be downloaded
    scan_progress = Signal(str)
    scan_done = Signal(object)  # the scan's result (sst.scan.save), or {"error": why}
    key = Signal()  # the dictation shortcut's listener queued an event, from the keyboard hook's thread
    live_key = Signal()  # the same for live translation's shortcut
    meeting_key = Signal()  # the same for Meeting mode's shortcut
    meeting = Signal()  # the meeting's state or result changed (MeetingSession.on_change, from any thread)
    minutes_written = Signal(object, object, str)  # (folder, MinutesResult or None, why not), a past meeting's minutes
    meeting_live_stop = Signal()  # the meeting stopped (perhaps on its watcher thread): its live bar stops on this one
    usage = Signal()  # a Gemini call was counted (sst.usage), from the thread that made it


class TrayApp:
    """Owns the settings and the dictation; the window (sst/window.py) shows them and asks this class for changes."""

    def __init__(self, quiet_start: bool = False):
        self._told_later: list[str] = []  # warnings from before the tray icon is there (sst.app._tell)
        self._settings_unsaved = False  # told once that the settings couldn't be saved
        self._history_unsaved = False  # told once that the history couldn't be saved
        self.profiles = Profiles.load()
        self._load_profile()
        self.recorder = self.new_recorder()
        self._configure_dictation_mic()
        self.dictation: Dictation | None = None
        # Start with your voice (sst.wakeui): listens once the speech model is loaded, if it's on and set up
        self.wake = WakeController(self)
        self.wake.load()
        self.wake_downloading: int | None = None  # percent, while its models download
        self.wake_problem = ""  # why they couldn't be downloaded, until tried again
        self._cancel_wake = threading.Event()
        self.loading_speech = ""  # the speech model being loaded in the background, if any
        self.downloading: tuple[str, int, int] | None = None  # (speech model, bytes done, bytes in all)
        self.scanning = ""  # what the scan is doing now, while it runs
        self.last_scan = scan.load()
        self._cancel_download = threading.Event()
        self.voice_downloading: int | None = None  # percent, while live translation's voice downloads
        self.voice_problem = ""  # why the voice couldn't be downloaded, until it's tried again
        self.listener: HotkeyListener | None = None
        self.quiet_start = quiet_start
        self._cleanup_notice = -1e9  # when the user was last told that the cleanup couldn't help
        self._speech_notice = -1e9  # when the user was last told that a cloud speech model couldn't help
        self._local = None  # Parakeet, once loaded: a cloud speech model falls back on it
        self._told_no_model = False  # "Rflow needs a speech model", said once
        self._local_lock = threading.Lock()
        self.update: updates.Update | None = None
        self._update_told = ""  # the version the user was last notified about
        self.install_problem = ""  # why the last update didn't install (updates.unfinished), said with the next offer

        self.signals = _Signals()
        self.signals.state.connect(self._on_state)
        self.signals.result.connect(self._on_result)
        self.signals.command.connect(lambda command: self.transforms.run_command(command))
        self.signals.loaded.connect(self._on_loaded)
        self.signals.failed.connect(self._on_failed)
        self.signals.update_found.connect(self._on_update_found)
        self.signals.update_none.connect(self._on_update_none)
        self.signals.update_progress.connect(lambda message: self.window.show_update(message, busy=True))
        self.signals.update_ready.connect(self._on_update_ready)
        self.signals.update_failed.connect(self._on_update_failed)
        self.signals.download_progress.connect(self._on_download_progress)
        self.signals.download_done.connect(self._on_download_done)
        self.signals.voice_progress.connect(self._on_voice_progress)
        self.signals.voice_done.connect(self._on_voice_done)
        self.signals.wake_progress.connect(self._on_wake_progress)
        self.signals.wake_done.connect(self._on_wake_done)
        self.signals.scan_progress.connect(self._on_scan_progress)
        self.signals.scan_done.connect(self._on_scan_done)
        self.signals.key.connect(self._pump)  # the keys wake the app up: nothing polls between them
        self.signals.live_key.connect(self._live_key_pump)
        self.signals.meeting_key.connect(self._meeting_key_pump)
        self.signals.meeting.connect(self._meeting_changed)
        self.signals.minutes_written.connect(self._minutes_written)
        self.signals.meeting_live_stop.connect(lambda: self.stop_live())
        self.signals.usage.connect(self._on_usage)

        self.icon = QIcon(str(ICON_FILE))
        self.recording_icon = _with_red_dot(self.icon)
        # the microphone recording now: a wake word turn may record on another one (a Bluetooth headset's stand-in)
        self.pill = Pill(level=lambda: (self.dictation.recorder if self.dictation else self.recorder).level)
        # Text Transform: its own shortcut and keyboard hook (HotkeyListener looked up when used: the tests replace it)
        self.transforms = TransformController(self, listener_factory=lambda key: HotkeyListener(key))
        self.transforms.start(self.settings.transform_shortcut)
        self.translator = TranslateController(self, listener_factory=lambda key: HotkeyListener(key))
        self.translator.start(self.settings.translate_shortcut)
        # Live captions: a pipeline of its own (sst.live), started and stopped by the user, never by dictation
        self.live = LiveCaptions(lambda config, on_event: self._live_session(config, on_event),
                                 lambda lane, config: self._live_lane(lane, config),
                                 lambda config: self._live_speaker(config))
        self.live.changed.connect(self._live_changed)
        self.live.moved.connect(self._live_moved)
        self.live.speak_toggled.connect(lambda on: self.set_live_speak(on))
        self.live.copy_requested.connect(self._copy_live_from_bar)
        self.live.save_toggled.connect(lambda on: self.set_live_save(on))
        self.live.quiet_stop.connect(lambda why: self._notify(APP_NAME, f"Live translation stopped: {why}."))
        self.live.spent.connect(lambda: self.window.isVisible() and self.window.refresh())  # the Live page shows the cost
        self.live.noted.connect(lambda _: self.window.isVisible() and self.window.refresh())  # the Live page shows it
        self.live_listener = None  # live translation's shortcut (Ctrl+Alt+L)
        # Meeting mode (sst.meeting, imported when a meeting is first looked for: start-up stays fast)
        self.last_meeting = None  # this run's latest MeetingSession: recording, writing its minutes, done or failed
        self._meeting_state = ""  # its state as last told (a notification once per change)
        self.writing_minutes: set[Path] = set()  # past meetings whose minutes are being written again (write_minutes)
        self.meeting_listener = None  # Meeting mode's shortcut (Ctrl+Alt+M)
        self._on_message = None  # what clicking the last notification does, if not opening the window
        self.window = MainWindow(self)
        self.wake.on_change = self._on_wake_changed

        self.tray = QSystemTrayIcon(self.icon)
        menu = QMenu()
        self.status_action = QAction("Loading the speech model...", menu)
        self.status_action.setEnabled(False)
        menu.addAction(self.status_action)
        self.update_action = QAction("", menu)
        self.update_action.triggered.connect(lambda: self.window.open("home"))
        self.update_action.setVisible(False)
        menu.addAction(self.update_action)
        menu.addSeparator()
        menu.addAction(f"Open {APP_NAME}", lambda: self.window.open("home"))
        menu.addAction("Words", lambda: self.window.open("dictionary"))
        menu.addAction("Text Transform", lambda: self.window.open("transform"))
        menu.addAction("Translate", lambda: self.window.open("translate"))
        self.live_action = QAction("Live translation", menu, checkable=True)
        self.live_action.triggered.connect(lambda: self.toggle_live())
        menu.addAction(self.live_action)
        self._start_live_shortcut()
        self.meeting_action = QAction("Meeting mode", menu, checkable=True)
        self.meeting_action.triggered.connect(lambda: self.toggle_meeting())
        menu.addAction(self.meeting_action)
        self._start_meeting_shortcut()
        menu.addAction("AI && models", lambda: self.window.open("models"))
        menu.addAction("Reading test", lambda: self.window.open("reading"))
        menu.addAction("Settings", lambda: self.window.open("settings"))
        menu.addAction("Check for updates", lambda: self.check_for_updates(manual=True))
        menu.addAction("Retry the last dictation", self.retry_last_dictation)
        menu.addSeparator()
        menu.addAction(f"Quit {APP_NAME}", self.quit)
        self.menu = menu  # keep a reference; the tray only borrows it
        self.tray.setContextMenu(menu)
        self.tray.activated.connect(lambda reason: self.window.open("home")
                                    if reason == QSystemTrayIcon.ActivationReason.Trigger else None)
        self.tray.messageClicked.connect(self._message_clicked)  # e.g. "Rflow 1.2.0 is available"
        self.tray.setToolTip(f"{APP_NAME}: loading...")
        self.tray.show()
        if self._told_later:  # what went wrong while starting: a damaged settings file or dictionary
            self._notify(APP_NAME, "\n".join(self._told_later), QSystemTrayIcon.MessageIcon.Warning)
            self._told_later.clear()
        self._update_waiting = ""  # a downloaded update, waiting for Rflow to be idle
        self._update_timer = QTimer(interval=2000, timeout=self._install_when_idle)
        self._set_status("Loading the speech model...")

        self.server = QLocalServer()
        self.server.newConnection.connect(self._on_connection)
        if not self.server.listen(SERVER_NAME):
            log.warning("No window server (%s); opening Rflow again won't show this window", self.server.errorString())

        # Dictation's housekeeping (the 30-minute limit, closing or watching the open microphone), once a second and only
        # while recording or while the microphone is open; the keys themselves arrive through signals.key.
        self.ticker = QTimer()
        self.ticker.setInterval(TICK_MS)
        self.ticker.timeout.connect(self._tick)
        self._load_speech()
        if getattr(sys, "frozen", False):  # the source checkout is updated with git, not by the app
            self.install_problem = updates.unfinished(UPDATE_DIR)
            if self.install_problem:  # the installer rolled back: this is still the old version
                log.warning("The update didn't install: %s", self.install_problem)
            # The installer of an update that has finished (it started this version) isn't needed any more.
            QTimer.singleShot(60_000, lambda: shutil.rmtree(UPDATE_DIR, ignore_errors=True))
            QTimer.singleShot(20_000, self.check_for_updates)
            self.update_timer = QTimer()
            self.update_timer.setInterval(updates.CHECK_EVERY_HOURS * 3600 * 1000)
            self.update_timer.timeout.connect(self.check_for_updates)
            self.update_timer.start()
        if not quiet_start or not self.settings.welcomed:
            self.window.open()

    # -- start-up

    def _load_speech(self) -> None:
        """Load the profile's speech model on a thread. A dictation already running keeps its model until the new one
        is ready, so switching never leaves the user without dictation."""
        key = usable(self.settings.speech_model, self.gateway)
        if not key:  # no speech model yet (Parakeet not downloaded, nothing else set up): the window asks for one
            self.loading_speech = ""
            if not self.dictation:
                self._set_status("Choose a speech model to start dictating")
                if self.settings.welcomed and not self._told_no_model:
                    self._told_no_model = True
                    self._notify(APP_NAME, "Rflow needs a speech model: choose one on AI & models.",
                                 QSystemTrayIcon.MessageIcon.Warning)
            self.window.refresh()
            return
        if key == self.loading_speech or (self.dictation and self.dictation.engine.name == key and not self.loading_speech):
            return
        self.loading_speech = key
        if self.dictation:
            self._set_status(f"Loading {SPEECH_MODELS[key].name}...", True)
        self.window.refresh()

        def work() -> None:
            try:
                t0 = time.perf_counter()
                local = self._local
                engine = local if key == DEFAULT_MODEL and local else self._new_engine(key)
                log.info("Speech model %s loaded in %.1fs", key, time.perf_counter() - t0)
                self.signals.loaded.emit(engine)
            except Exception as e:
                log.exception("Could not load the speech model %s", key)
                self.signals.failed.emit(key, str(e))
        threading.Thread(target=work, name="model-loader", daemon=True).start()

    def _new_engine(self, key: str, fallback: bool = True):
        """A speech model as the settings say. A cloud model gets its key and model, an own server its address too, and
        both get Parakeet to fall back on."""
        s = self.settings
        if key not in REMOTE:
            return load_engine(key, s.speech_language)
        on_failure = self._fallback_engine if fallback else None
        if key in CLOUD:
            return load_engine(key, s.speech_language, self.gateway.key_for(key), s.speech_cloud_models.get(key, ""),
                               on_failure)
        address, api_key = self.gateway.speech_server()
        return load_engine(key, s.speech_language, api_key, s.speech_server_model, on_failure, address)

    def _fallback_engine(self):
        """Parakeet, for a cloud model that couldn't help: the one loaded already, or loaded now (once). Using only
        a cloud model keeps Rflow light until then."""
        with self._local_lock:
            if self._local is None:
                if not SPEECH_MODELS[DEFAULT_MODEL].installed():
                    raise RuntimeError("Parakeet isn't downloaded to take over")
                self._local = load_engine(DEFAULT_MODEL)
            return self._local

    def _on_loaded(self, engine) -> None:
        if self.loading_speech == engine.name:
            self.loading_speech = ""
        first = self.dictation is None
        if not first and engine.name != usable(self.settings.speech_model, self.gateway):
            self._load_speech()  # another model was chosen while this one loaded: drop it, load (or await) that one
            return
        if engine.name == DEFAULT_MODEL:
            self._local = engine  # a cloud model chosen later falls back on it without loading it again
        elif engine.name not in REMOTE:
            self._local = None  # another model runs on this computer: Parakeet's memory can go
        if first:
            self.dictation = Dictation(engine, self.recorder, sounds=self.settings.sounds,
                                       save=self.settings.save_recordings)
            self.dictation.on_state = self.signals.state.emit
            self.dictation.on_result = self.signals.result.emit
            self.dictation.command = self.voice_command
            self.dictation.snippets = self.snippets
            self.dictation.on_command = self.signals.command.emit
            self.dictation.on_notice = lambda message: self._notify(APP_NAME, message)  # e.g. the chosen mic is missing
        else:
            self.dictation.engine = engine  # the next dictation uses it; one being transcribed finishes with the old
        self._apply_cleanup()  # also gives the new model Your words, and builds its voice pipeline
        if first:
            self._start_listener()
            self.wake.apply()  # the wake word needs a dictation to start (and keeps the microphone open: ticks)
            self._wake_ticker()
        self.window.refresh()
        if not first:
            return
        if engine.name != usable(self.settings.speech_model, self.gateway):
            self._load_speech()  # dictation works now; the chosen model follows
            return
        if self.listener and wispr_flow_running() and self.listener.hotkey.modifiers == {"ctrl", "win"}:
            self._notify("Wispr Flow is running", "It also listens to Ctrl+Win, so both would type. Quit Wispr Flow.",
                         QSystemTrayIcon.MessageIcon.Warning)
        elif self.listener and not self.quiet_start and not self.window.isVisible():
            self._notify(f"{APP_NAME} is ready", f"Hold {self.listener.hotkey.label} in any app and speak.")

    def _on_failed(self, key: str, message: str) -> None:
        if self.loading_speech == key:
            self.loading_speech = ""
        if self.dictation:  # a switch failed: dictation goes on with the model it had
            self._update_status()
            self.window.refresh()
            self._notify(APP_NAME, f"Could not load {SPEECH_MODELS[key].name}; still using "
                         f"{SPEECH_MODELS[self.dictation.engine.name].name}. {message}",
                         QSystemTrayIcon.MessageIcon.Warning)
            return
        self._set_status("Could not load the speech model")
        QMessageBox.critical(None, APP_NAME, f"Could not load the speech model:\n\n{message}\n\n"
                             f"Details are in the log: {LOG_DIR}")

    def _start_listener(self) -> None:
        if self.listener:
            self.listener.stop()
        try:
            key = parse_hotkey(self.settings.hotkey)
        except ValueError as e:
            log.warning("Bad hotkey %r in the settings (%s); using %s", self.settings.hotkey, e, DEFAULT_HOTKEY)
            key = parse_hotkey(DEFAULT_HOTKEY)
        self.listener = HotkeyListener(key)
        self.listener.on_event = self.signals.key.emit
        self.listener.start()
        self.dictation.listener = self.listener
        self._update_status()
        log.info("Listening for %s", key.text)

    def _apply_speech(self) -> None:
        """The speech model's options that need no reload, from the next dictation on: the language, Your words, and a
        cloud model's model and key (an own server's address too)."""
        s, engine = self.settings, self.dictation.engine
        if hasattr(engine, "language"):
            engine.language = s.speech_language
        if hasattr(engine, "words"):  # the recogniser listens for them (hotwords, a provider's prompt)
            # Names and terms only: everyday words as hints get heard where they weren't said (speech_hints)
            engine.words = speech_hints(self.dictionary.hint_terms() if s.voice_pipeline else s.vocabulary)
        if hasattr(engine, "timestamp_mode"):
            engine.timestamp_mode = VoiceConfig().asr.timestamp_mode  # Gemini Transcribe: word times for the merge
        if engine.name in CLOUD:
            engine.model = s.speech_cloud_models.get(engine.name) or CLOUD[engine.name].models[0]
            engine.api_key = self.gateway.key_for(engine.name) or engine.api_key  # no key: Parakeet is on its way
        elif engine.name in REMOTE:
            address, engine.api_key = self.gateway.speech_server()
            engine.model = s.speech_server_model
            if address:  # none: Parakeet is on its way
                engine.set_url(address)

    def _apply_cleanup(self) -> None:
        """Use Your words and the model chosen in AI cleanup from the next dictation on."""
        s = self.settings
        self._apply_speech()
        model = s.cleanup_model if s.cleanup else ""
        polisher = None
        if model and self.gateway.address:
            polisher = Polisher(self.gateway, model, speech_hints(s.vocabulary), fallback=s.cleanup_fallback or None)
            polisher.prepare()  # connect now, so the first dictation doesn't wait for it
        elif s.cleanup:
            self._notify(APP_NAME, "AI cleanup needs a provider and a model: connect one on AI & models.",
                         QSystemTrayIcon.MessageIcon.Warning)
        self.dictation.cleanup = polisher
        self._build_pipeline()
        self._update_status()
        log.info("Text cleanup: %s, backup %s (%s, %d words)", model or "off", s.cleanup_fallback or "none",
                 self.gateway, len(s.vocabulary))

    def _build_pipeline(self) -> None:
        """The staged voice pipeline (sst.pipeline) for the speech model in use: a new one when the model changes, new
        text stages when the cleanup, the dictionary or the formatting change."""
        s, dictation = self.settings, self.dictation
        if not s.voice_pipeline:
            dictation.pipeline = None  # the classic way: the whole recording at once
            return
        engine, pipeline = dictation.engine, dictation.pipeline
        if pipeline is None or getattr(pipeline, "engine", None) is not engine:
            config = VoiceConfig()
            remote = engine.name in REMOTE
            config.asr.max_in_flight = 2 if remote else 1  # a local model already uses every core
            # A cloud model or own server falls back on Parakeet chunk by chunk; it is loaded only if needed.
            fallback = LazyBackend("parakeet", lambda: EngineBackend(self._fallback_engine())) if remote else None
            scheduler = ASRScheduler(EngineBackend(engine), config.asr, fallback=fallback, on_event=self._on_pipeline_event)
            pipeline = VoicePipeline(scheduler, config)
            pipeline.engine = engine
        pipeline.debug_dir = DEBUG_DIR if s.debug_pipeline else None
        pipeline.keep_audio = s.debug_pipeline
        model = s.cleanup_model if s.cleanup else ""
        llm = GatewayLLM(self.gateway, model, s.cleanup_fallback or None) if model and self.gateway.address else None
        config = pipeline.config
        config.formatting.enabled = s.format_text
        pipeline.stages = Stages(dictionary=DictionaryEngine(self.dictionary, config.dictionary),
                                 formatter=Formatter(config.formatting) if s.format_text else None,
                                 llm=llm, guard=Guard(config.guard),
                                 terms=lambda: speech_hints(self.dictionary.hint_terms()),  # protected: not "move"
                                 command=self.voice_command, snippets=self.snippets)
        dictation.pipeline = pipeline

    def _on_pipeline_event(self, name: str, data: dict) -> None:
        # The ASR workers' events: the log tells where a slow or failed dictation spent its time.
        (log.warning if name in ("asr_failed", "asr_fallback") else log.debug)("%s %s", name, data)

    def retry_last_dictation(self) -> None:
        if not (self.dictation and self.dictation.retry_last()):
            self._notify(APP_NAME, "There is no failed dictation to retry.")

    def _update_status(self) -> None:
        if not self.listener or self.loading_speech:
            return
        model = self.dictation.cleanup.model.rsplit("/", 1)[-1] if self.dictation and self.dictation.cleanup else None
        engine = self.dictation.engine if self.dictation else None
        speech = f" · speech: {engine.title}" if engine and engine.name != DEFAULT_MODEL else ""
        self._set_status(f"Ready: hold {self.listener.hotkey.label}" + speech + (f" · cleanup: {model}" if model else ""),
                         True)

    # -- running

    def _pump(self) -> None:
        """The dictation shortcut's events, as they come (signals.key)."""
        if self.listener is None or self.dictation is None:
            return
        events = self.listener.events
        while not events.empty():
            event, at = events.get_nowait()
            self.dictation.handle(event, at)
        self.wake.tick(time.monotonic())  # a turn ended with the keys, or a dictation with the keys pauses it
        self._wake_ticker()  # a recording may have started, and the microphone opened

    def _tick(self) -> None:
        now = time.monotonic()
        if self.dictation is not None:
            self.dictation.tick(now)
        self.wake.tick(now)  # a turn ended by the time limit, a pause for live translation or Text Transform
        if not self._needs_ticks():
            self.ticker.stop()

    def _needs_ticks(self) -> bool:
        r = self.recorder
        # The wake word listens on an open microphone, so its ticks are those of an open microphone: once a second
        return self.dictation is not None and (self.dictation.recording or r.warm or r.busy or self.wake.wants_ticks)

    def _wake_ticker(self) -> None:
        if self._needs_ticks() and not self.ticker.isActive():
            self.ticker.start()

    def _on_wake_changed(self) -> None:
        """The wake word started or stopped listening, or paused: the ticks it needs, and the Settings card's status."""
        if hasattr(self, "ticker"):
            self._wake_ticker()
        if self.window.isVisible():
            self.window.show_wake()

    def _on_result(self, heard: str, typed: str, seconds: float) -> None:
        self.transforms.note_typed(typed + " ")  # as Dictation pasted it: Text Transform's text where Ctrl+A copies nothing
        self._add_history(typed, heard)
        self.stats.add(typed, seconds, date.today())
        try:
            self.stats.save(self.profile.stats_file)
        except OSError as e:
            log.warning("Could not save the stats: %s", e)
        if self.window.isVisible():
            self.window.refresh()

    def _on_state(self, state: str, message: str) -> None:
        if state in ("warning", "error"):
            log.warning(message)
            self._notify(APP_NAME, message, QSystemTrayIcon.MessageIcon.Warning)
        if state == "typed_local":
            log.warning("Typed with Parakeet: %s", message)
            if time.monotonic() - self._speech_notice > 600:  # at most one notification per 10 minutes
                self._speech_notice = time.monotonic()
                self._notify("Cloud speech unavailable", f"Parakeet typed it on this computer ({message}).",
                             QSystemTrayIcon.MessageIcon.Warning)
        if state == "typed_raw":
            log.warning("Typed as heard: %s", message)
            if time.monotonic() - self._cleanup_notice > 600:  # at most one notification per 10 minutes
                self._cleanup_notice = time.monotonic()
                self._notify("Text cleanup unavailable", f"Typed what was heard ({message}).",
                             QSystemTrayIcon.MessageIcon.Warning)
        recording = bool(self.dictation and self.dictation.recording) or self.meeting is not None
        self.tray.setIcon(self.recording_icon if recording else self.icon)
        if recording and state not in ("recording",):
            return  # a previous dictation finished while a new one is being recorded: keep showing the recording
        self.pill.ai = bool(self.dictation and self.dictation.cleanup)  # "Cleaning up" while the AI works on it
        self.pill.show_state(state, message)

    def _on_connection(self) -> None:
        while (socket := self.server.nextPendingConnection()) is not None:
            socket.disconnected.connect(socket.deleteLater)
            self.window.open()  # someone opened Rflow again (Start menu, desktop): show this one

    # -- what the window asks for

    def hotkey_label(self) -> str:
        if self.listener:
            return self.listener.hotkey.label
        try:
            return parse_hotkey(self.settings.hotkey).label
        except ValueError:
            return parse_hotkey(DEFAULT_HOTKEY).label

    def history_entries(self) -> list[dict]:
        return read_history(self.profile.history_file)

    def microphones(self) -> list[str]:
        return input_device_names()

    def default_microphone(self) -> str:
        """Windows' default microphone now: what "Windows default" records."""
        return default_microphone()

    def new_recorder(self) -> Recorder:
        """A recorder set up as the settings say (dictation's, and the reading test's, so a test hears what dictation
        hears)."""
        recorder = Recorder()
        self._configure(recorder)
        return recorder

    def _configure(self, recorder: Recorder) -> None:
        s = self.settings
        recorder.device, recorder.raw, recorder.tail = s.microphone or None, s.raw_audio, TAIL_SECONDS
        recorder.warm_seconds = WARM_SECONDS if s.warm_mic else 0.0

    def _configure_dictation_mic(self) -> None:
        """Dictation's microphone (not the reading test's): with the voice pipeline it keeps 2 s of pre-roll, and the
        always-on mode keeps it open while Rflow runs (the audio stays in RAM; nothing is kept or sent before a key
        press)."""
        s, recorder = self.settings, self.recorder
        self._configure(recorder)
        # The pipeline trims its 2 s pre-roll to the speech running into the key press; the classic path doesn't, so it
        # keeps the short pre-roll it always had.
        recorder.preroll_seconds = VoiceConfig().audio.pre_roll_ms / 1000 if s.voice_pipeline else PREROLL_SECONDS
        if s.always_on_mic:
            recorder.warm_seconds = math.inf
            try:
                recorder.keep_open()
            except Exception as e:  # no microphone yet: it opens at the first dictation
                log.warning("Could not keep the microphone open: %s", e)
        wake = getattr(self, "wake", None)
        if wake is not None:
            wake.apply()  # listening keeps a microphone open, with the moment before the keyword
        if hasattr(self, "ticker"):  # not while starting: the ticks begin with dictation (_on_loaded)
            self._wake_ticker()

    def apply_settings(self, new: Settings) -> None:
        """Save the settings and use them at once (the Settings page and the welcome change them one by one)."""
        old, self.settings = self.settings, new
        try:
            new.save(self.profile.settings_file)
        except OSError as e:  # a full disk, a folder that can't be written: the change works until Rflow closes
            log.warning("Couldn't save the settings: %s", e)
            if not self._settings_unsaved:
                self._settings_unsaved = True
                self._tell(f"Rflow couldn't save your settings ({e}): your changes work until Rflow closes.")
        self._configure_dictation_mic()
        if new.vocabulary != old.vocabulary:
            self.dictionary.sync_vocabulary(new.vocabulary)
        if self.dictation:
            self.dictation.sounds, self.dictation.save = new.sounds, new.save_recordings
            if (new.cleanup, new.cleanup_model, new.cleanup_fallback, new.vocabulary) != (
                    old.cleanup, old.cleanup_model, old.cleanup_fallback, old.vocabulary):
                self._apply_cleanup()  # the chosen model is active from the next dictation
            elif (new.speech_language, new.speech_cloud_models, new.speech_server_model) != (
                    old.speech_language, old.speech_cloud_models, old.speech_server_model):
                self._apply_speech()  # from the next dictation; no reload needed
            if (new.voice_pipeline, new.format_text, new.debug_pipeline) != (
                    old.voice_pipeline, old.format_text, old.debug_pipeline):
                self._build_pipeline()
            if new.hotkey != old.hotkey:
                self._start_listener()
        if new.transform_shortcut != old.transform_shortcut and hasattr(self, "transforms"):
            self.transforms.start(new.transform_shortcut)
        if new.translate_shortcut != old.translate_shortcut and hasattr(self, "translator"):
            self.translator.start(new.translate_shortcut)
        if (new.live_shortcut, new.translate_shortcut, new.transform_shortcut) != (
                old.live_shortcut, old.translate_shortcut, old.transform_shortcut) and hasattr(self, "live_action"):
            self._start_live_shortcut()
        shortcuts = ("meeting_shortcut", "live_shortcut", "translate_shortcut", "transform_shortcut")
        if any(getattr(new, k) != getattr(old, k) for k in shortcuts) and hasattr(self, "meeting_action"):
            self._start_meeting_shortcut()
        if new.speech_model != old.speech_model:
            self._load_speech()
        self.window.refresh()
        log.info("Settings saved: %s", new)

    def save_cleanup(self, on: bool, model: str, fallback: str, gateway: GatewayConfig) -> None:
        if gateway != self.gateway:
            gateway.save(self.profile.gateway_file)
            self.gateway = gateway
        self.apply_settings(dataclasses.replace(self.settings, cleanup=on, cleanup_model=model, cleanup_fallback=fallback))
        if self.dictation:
            self._apply_cleanup()  # also when only the endpoint or key changed (a cloud speech model's key too)
        self._load_speech()  # a key added or removed can change which speech model is usable

    # -- Your API keys (AI & models): one key per provider, which speech, AI cleanup and live translation read

    def save_key(self, provider: str, key: str) -> None:
        """A provider's key added, changed or removed (""). Which provider and model AI cleanup uses never changes here
        (the user testing's M-04: a Gemini key added for live translation switched the cleanup to Gemini, no model)."""
        if provider in PROVIDERS and key != self.gateway.key_for(provider):
            self._keys_changed(self.gateway.with_key(provider, key))

    def save_server(self, address: str, key: str, names: tuple[str, ...] = ("vllm",)) -> None:
        """Your own server's address and key ("" removes it), for AI cleanup's server ("vllm"), the speech model's
        (SPEECH_SERVER), or both when they are the same server. Neither becomes the one in use here."""
        gateway = self.gateway
        for name in names:
            gateway = gateway.with_entry(name, address, key)
        if gateway != self.gateway:
            self._keys_changed(gateway)

    def _keys_changed(self, gateway: GatewayConfig) -> None:
        self.gateway = gateway
        gateway.save(self.profile.gateway_file)
        if self.dictation:
            self._apply_cleanup()  # the cleanup's provider or the speech model in use may use the key: no reload
        self._load_speech()  # a key added or removed can change which speech model is usable
        self.window.refresh()

    def add_words(self, new_words: list[str]) -> int:
        known = {w.lower() for w in self.settings.vocabulary}
        added = [w for w in dict.fromkeys(new_words) if w.lower() not in known]
        if added:
            # The cleanup uses the new words from the next dictation (and the next scoring).
            self.apply_settings(dataclasses.replace(self.settings, vocabulary=self.settings.vocabulary + added))
        return len(added)

    def remove_word(self, word: str) -> None:
        term = self.dictionary.find(word)
        if term is not None and term.source != "vocabulary":
            self.dictionary.remove_term(term.id)  # a term that only had sound-alikes
        if word in self.settings.vocabulary:
            self.apply_settings(dataclasses.replace(self.settings, vocabulary=[w for w in self.settings.vocabulary
                                                                               if w != word]))
        elif self.dictation:
            self._apply_speech()

    # -- Text Transform (sst.transformui): the AI cleanup's model, under Text Transform's own prompt and checks

    def transform_model(self) -> str:
        return self.settings.cleanup_model if self.gateway.address else ""

    def transform_ready(self) -> bool:
        return bool(self.transform_model())

    def translate_ready(self) -> bool:
        return self.transform_ready()  # the same model as the AI cleanup

    # -- live translation (sst.live)

    def live_problem(self) -> str:
        """Why live translation can't start, or "" when it can."""
        if not self.gateway.key_for("gemini"):
            return "Live translation uses Google Gemini 3.5 Live Translate: add a Gemini key in AI & models."
        return ""

    def live_running(self) -> bool:
        return self.live.running

    def live_note(self) -> str:
        """The bar's status line while live translation runs: why nothing shows yet, or ""."""
        return self.live.note

    def live_languages(self) -> tuple[str, str]:
        """What live translation translates into now (the computer's sound, the microphone's): while it runs, the ones
        it started with, since a language changed meanwhile waits for the next start."""
        c = self.live.config if self.live.running else self.live_config()
        return c.target, c.mic_target

    def live_folder(self) -> Path:
        return self.profile.folder() / "live captions"  # its name since 2.1.0, so earlier sessions stay listed

    def live_sessions(self, limit: int = 8) -> list[tuple[datetime, int, Path]]:
        """The latest transcripts, newest first: when each began, how many lines it has, its file."""
        found = []
        for path in sorted(self.live_folder().glob("* live captions.txt"), reverse=True)[:limit]:
            try:
                began = datetime.strptime(path.name[:19], "%Y-%m-%d %H-%M-%S")
                lines = path.read_text(encoding="utf-8").count("\n[")
            except (ValueError, OSError):
                continue
            found.append((began, lines, path))
        return found

    def open_live_session(self, path: Path) -> None:
        shell.open_url(QUrl.fromLocalFile(str(path)))

    def copy_live_session(self, path: Path, translation_only: bool = False) -> str:
        """A session's transcript onto the clipboard, to paste anywhere: "" or why it couldn't be copied."""
        return self._copy_live(path, translation_only)[1]

    def _copy_live(self, path: Path, translation_only: bool = False) -> tuple[int, str]:
        """(how many lines were copied, "" or why nothing was)."""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            log.warning("Couldn't read the live translation transcript: %s", e)
            return 0, "The transcript couldn't be read"
        try:
            textaccess.set_clipboard(for_copy(text, translation_only))
        except OSError as e:  # another app holds the clipboard
            log.warning("Couldn't copy the live translation transcript: %s", e)
            return 0, "Another app is using the clipboard: try again"
        return text.count("\n["), ""

    def _copy_live_from_bar(self) -> None:
        """The bar's copy button: the session so far, both languages; the bar says what happened."""
        path = self.live.transcript_path
        if path is None:
            self.live.flash("Nothing to copy yet")
            return
        lines, problem = self._copy_live(path)
        self.live.flash(problem or f"Copied {lines} line{'' if lines == 1 else 's'}")

    def live_config(self) -> LiveConfig:
        s = self.settings
        return LiveConfig(target=s.live_target, mic_target=s.live_mic_target, source=s.live_source,
                          hide_from_capture=s.live_hide_from_share, speak=s.live_speak, speak_speed=s.live_speak_speed,
                          duck=s.live_duck, save=s.live_save)

    def room_live_config(self) -> LiveConfig:
        """Live translation in a meeting in person: the microphone hears the room, whose words go into the user's own
        language (live_target, not the microphone's usual mic_target, which is for the user speaking); words already in
        it show as heard, so the bar is the meeting's live transcript. Never spoken aloud in the room."""
        return dataclasses.replace(self.live_config(), source="microphone", mic_target=self.settings.live_target,
                                   speak=False, duck=False)

    # -- the spoken translation (sst.live.voice, sst.live.speaker)

    def live_voice(self) -> Voice:
        """The voice on this laptop (Ryan; an older Rflow's Danny until Ryan is downloaded): Gemini speaks the
        translation in its own voice, this one stands in when Gemini's audio doesn't come."""
        return local_voice()

    def live_voice_state(self) -> tuple[str, int, str]:
        """("ready" / "downloading" / "missing", the percent downloaded, why the last download failed)."""
        if self.voice_downloading is not None:
            return "downloading", self.voice_downloading, ""
        return ("ready" if self.live_voice().ready() else "missing"), 0, self.voice_problem

    def _live_speaker(self, config: LiveConfig) -> Speaker | None:
        """What says the translation aloud: Gemini's own voice at once, the laptop's in its place once downloaded (the
        voice is looked for at each load: a download finished meanwhile is used)."""
        language = self.live_voice().language

        def load() -> PiperVoice | None:
            voice = local_voice()
            return PiperVoice(voice) if voice.ready() else None
        return Speaker(load, Player(), config.spoken_lanes(language), language, config.speak_speed,
                       on_problem=lambda message: self.live.event.emit(LiveEvent(Kind.ERROR, message, lane="voice")),
                       ducker=Ducker(config.duck), own=config.spoken_lanes())

    def set_live_speak(self, on: bool) -> None:
        """The translation spoken aloud, or not: saved, at once while it runs; the laptop's voice is downloaded the
        first time (Ryan also where an older Rflow has Danny), to stand in when Gemini's audio doesn't come."""
        self.apply_settings(dataclasses.replace(self.settings, live_speak=on))
        if on and not LOCAL_VOICE.ready():
            self.download_live_voice()
        self.live.set_speak(on)
        if self.window.isVisible():
            self.window.refresh()

    def set_live_save(self, on: bool) -> None:
        """The cost saver (audio sent only while someone talks): saved, and at once while it runs; the bar's button and
        the Live page's toggle stay in step."""
        self.apply_settings(dataclasses.replace(self.settings, live_save=on))
        self.live.set_save(on)
        if self.window.isVisible():
            self.window.refresh()

    def live_spend(self):
        """The running session's Spend while the cost saver is on, else None."""
        return self.live.spend() if self.live.running and self.live.config.save else None

    def live_last_spend(self):
        """The last saving session's Spend, until the next start, or None."""
        return self.live.last_spend

    def live_stopped_why(self) -> str:
        """Why the cost saver stopped the last session ("no speech for 15 min"), or ""."""
        return self.live.stopped_why

    def set_live_speak_speed(self, speed: float) -> None:
        self.apply_settings(dataclasses.replace(self.settings, live_speak_speed=speed))
        self.live.set_speak_speed(speed)

    def set_live_duck(self, depth: float) -> None:
        """How loud the other apps stay while the voice speaks (1.0: as they are): saved, and at once."""
        self.apply_settings(dataclasses.replace(self.settings, live_duck=depth))
        self.live.set_duck(depth)

    def download_live_voice(self) -> None:
        """Download and prepare the voice in the background (once; the progress shows on the bar and the page)."""
        if self.voice_downloading is not None:
            return
        voice, last = LOCAL_VOICE, [-1]
        self.voice_downloading, self.voice_problem = 0, ""

        def progress(done: int, total: int) -> None:
            percent = done * 100 // total if total else 0
            if percent != last[0]:  # the window needn't redraw for every megabyte
                last[0] = percent
                self.signals.voice_progress.emit(percent)

        def work() -> None:
            try:
                install_voice(voice, progress)
                self.signals.voice_done.emit("")
            except Exception as e:
                log.exception("Downloading the voice %s failed", voice.key)
                self.signals.voice_done.emit(str(e) or type(e).__name__)
        threading.Thread(target=work, name="voice-download", daemon=True).start()
        self.live.set_voice_note("Downloading the voice…")

    def _on_voice_progress(self, percent: int) -> None:
        self.voice_downloading = percent
        self.live.set_voice_note(f"Downloading the voice: {percent}%")
        if self.window.isVisible():
            self.window.refresh()

    def _on_voice_done(self, error: str) -> None:
        self.voice_downloading, self.voice_problem = None, error
        self.live.set_voice_note("")
        if error:
            self._notify(APP_NAME, f"The voice couldn't be downloaded: {error}", QSystemTrayIcon.MessageIcon.Warning)
        else:
            log.info("The voice %s is ready", LOCAL_VOICE.name)
            self.live.set_speak(self.settings.live_speak)  # here now: it speaks if speaking is still wanted
        if self.window.isVisible():
            self.window.refresh()

    # -- Start with your voice (sst.wakeui; the keyword spotting, voice check and turns are sst.wake)

    def wake_state(self) -> WakeState:
        """What Settings' "Start with your voice" card shows."""
        wake = self.wake
        state = WakeState(listening=wake.microphone if wake.listening else "", note=wake.problem or wake.note,
                          problem=self.wake_problem, instead=wake.instead if wake.listening else "")
        if wake.setup is not None:
            state.keyword, state.heard, state.said = wake.setup.keyword, wake.setup.heard, wake.setup.said
        try:
            api = wake_api()
            missing = [m for m in (api.KWS, api.VOICE) if not m.installed()]
            state.tries, state.size = api.TRIES, sum(m.size for m in missing)
            state.models = "downloading" if self.wake_downloading is not None else "missing" if missing else "ready"
            state.percent = self.wake_downloading or 0
        except Exception as e:  # not in this build
            state.models, state.problem = "missing", state.problem or f"The wake word isn't available ({e})."
        return state

    def set_wake_enabled(self, on: bool) -> None:
        """Switched on or off: saved, and at once; switched on the first time, its models are downloaded."""
        self.apply_settings(dataclasses.replace(self.settings, wake_enabled=on))
        if on and self.wake_state().models == "missing":
            self.download_wake_models()

    def download_wake_models(self) -> None:
        """The keyword spotter and the voice check (sst.wake.KWS, VOICE), in the background, one after the other."""
        if self.wake_downloading is not None:
            return
        try:
            api = wake_api()
        except Exception as e:
            self.wake_problem = str(e) or type(e).__name__
            self.window.refresh()
            return
        models = [m for m in (api.KWS, api.VOICE) if not m.installed()]
        if not models:
            self.wake.apply()
            return
        total = sum(m.size for m in models) or 1
        self.wake_downloading, self.wake_problem = 0, ""
        self._cancel_wake.clear()
        done_before, last = [0], [-1]

        def progress(done: int, _total: int) -> None:
            percent = (done_before[0] + done) * 100 // total
            if percent != last[0]:  # the window needn't redraw for every megabyte
                last[0] = percent
                self.signals.wake_progress.emit(percent)

        def work() -> None:
            try:
                for model in models:
                    downloads.download(model, progress, self._cancel_wake.is_set)
                    done_before[0] += model.size
                self.signals.wake_done.emit("")
            except downloads.Cancelled:
                self.signals.wake_done.emit("cancelled")
            except Exception as e:
                log.exception("Downloading the wake word's models failed")
                self.signals.wake_done.emit(str(e) or type(e).__name__)
        threading.Thread(target=work, name="wake-download", daemon=True).start()
        self.window.refresh()

    def cancel_wake_download(self) -> None:
        self._cancel_wake.set()  # what is downloaded so far stays for the next try

    def _on_wake_progress(self, percent: int) -> None:
        self.wake_downloading = percent
        if self.window.isVisible():
            self.window.refresh()

    def _on_wake_done(self, error: str) -> None:
        self.wake_downloading = None
        if error and error != "cancelled":
            self.wake_problem = error
            self._notify(APP_NAME, f"The wake word's models couldn't be downloaded: {error}",
                         QSystemTrayIcon.MessageIcon.Warning)
        elif not error:
            log.info("The wake word's models are downloaded")
            self.wake.apply()
        if self.window.isVisible():
            self.window.refresh()

    def open_wake_setup(self, parent=None) -> None:
        """The 10-try setup (a dialog): the keyword typed (checked with live translation's voice, when it's downloaded),
        then said ten ways; kept when heard often enough. It records on the microphone the wake word listens on (a
        Bluetooth headset's only when the user lets it), in memory only."""
        try:
            api = wake_api()
        except Exception as e:
            self._notify(APP_NAME, f"The wake word isn't available: {e}", QSystemTrayIcon.MessageIcon.Warning)
            return
        _, device, name, problem = self.wake.choose_microphone()

        def new_recorder() -> Recorder:
            recorder = Recorder(device)
            recorder.raw = self.settings.raw_audio
            recorder.keep_call_quality = True  # a headset's only when the wake word listens there: open for the tries
            return recorder
        def english_voice() -> PiperVoice | None:  # the laptop's voice (Ryan, or Danny until it's here): English, as the spotter
            voice = local_voice()
            return PiperVoice(voice) if voice.ready() else None
        tools = WakeTools(api, english_voice)
        dialog = WakeSetupDialog(parent, microphone=name, problem=problem, check=api.check_keyword,
                                 enroll=lambda keyword: api.Enrollment(keyword, transcribe=tools.transcribe),
                                 hearing=tools.hearing, new_recorder=new_recorder, save=self.save_wake_setup,
                                 tries=api.TRIES, passes=api.PASS, misses=api.MISSES, quiet_db=api.QUIET_DB)
        self.wake.setting_up = True  # the tries mustn't start a dictation
        try:
            dialog.exec()
        finally:
            self.wake.setting_up = False
        if self.window.isVisible():
            self.window.refresh()

    def save_wake_setup(self, setup) -> None:
        """A setup that passed: kept in the profile (words and numbers, no audio), and listened for at once."""
        write_wake(self.profile.wake_file, setup.to_json())
        self.wake.setup = setup
        self.wake.apply()
        log.info("Wake word set up: heard %d of the tries", setup.heard)

    def _live_session(self, config: LiveConfig, on_event) -> LiveSession:
        return LiveSession(config, Transcript(self.live_folder(), config), on_event)

    def _live_lane(self, lane: str, config: LiveConfig) -> tuple:
        """One way of live translation: what the laptop plays, or the microphone; each its own Gemini session."""
        key, lane_config = self.gateway.key_for("gemini"), config.for_lane(lane)
        # The microphone chosen for dictation (read at each look: a new choice is followed), else Windows' default
        capture = Capture.microphone(lambda: self.settings.microphone) if lane == MIC else Capture.speakers()
        return capture, lambda emit: GeminiLiveTranslate(key, lane_config, emit, lane=lane)

    def start_live(self, config: LiveConfig | None = None) -> str:
        """Start live translation of what the chosen source hears (or as `config` says); "" or why it didn't start."""
        if self.live.running:
            return ""
        problem = self.live_problem()
        if problem:
            return problem
        s = self.settings
        if not s.live_told:
            answer = QMessageBox.question(
                None, f"{APP_NAME}: live translation",
                "Live translation sends what it listens to (what your laptop plays, your microphone, or both, as you "
                "choose) to Google, which translates it with Gemini 3.5 Live Translate, a preview model, and shows "
                "the translation in a bar you can move anywhere.\n\n"
                "With a paid Gemini key this costs about $2.20 an hour for each source (both: twice); with a free key "
                "Google may use the audio to improve its products. Each session's transcript is saved on this "
                "laptop.\n\nStart?")
            if answer != QMessageBox.StandardButton.Yes:
                return "Not started."
            self.apply_settings(dataclasses.replace(s, live_told=True))
        if s.live_speak and not self.voice_problem and not LOCAL_VOICE.ready():
            self.download_live_voice()  # e.g. an older Rflow's Danny here: Ryan comes in the background, once a run
        try:
            self.live.start(config or self.live_config(), geometry=list(self.settings.live_bar) or None)
        except Exception as e:  # e.g. no output device or microphone to listen to
            log.warning("Live translation didn't start: %s", e)
            return f"Live translation couldn't start: {e}"
        return ""

    def stop_live(self) -> None:
        self.live.stop()

    def set_live_source(self, source: str) -> str:
        """Listen to the computer, the microphone or both: saved, and at once while it runs; "" or why not."""
        self.apply_settings(dataclasses.replace(self.settings, live_source=source))
        return self.live.set_source(source) if self.live.running else ""

    def set_live_hidden(self, hidden: bool) -> None:
        """The bar left out of screen shares, or shown in them: saved, and at once while it runs."""
        self.apply_settings(dataclasses.replace(self.settings, live_hide_from_share=hidden))
        self.live.set_hidden(hidden)

    def toggle_live(self) -> None:
        if self.live.running:
            self.stop_live()
            return
        problem = self.start_live()
        if problem and problem != "Not started.":
            self._notify(APP_NAME, problem, QSystemTrayIcon.MessageIcon.Warning)
        self._live_changed(self.live.running)

    def _live_changed(self, running: bool) -> None:
        self.live_action.setChecked(running)
        if not running and self.live.last_problem:
            self._notify(APP_NAME, f"Live translation stopped: {self.live.last_problem}",
                         QSystemTrayIcon.MessageIcon.Warning)
            self.live.last_problem = ""
        if self.window.isVisible():
            self.window.refresh()

    def _live_moved(self, geometry: list) -> None:
        """Where the user put the bar, for the next start (saved quietly: nothing else depends on it)."""
        self.settings = dataclasses.replace(self.settings, live_bar=list(geometry))
        self.settings.save(self.profile.settings_file)

    def live_shortcut_clash(self) -> str:
        """The tool already using live translation's shortcut, or "": the shortcut then stays with that tool."""
        s = self.settings
        for name, shortcut in (("Translate", s.translate_shortcut), ("Text Transform", s.transform_shortcut)):
            if s.live_shortcut and shortcut == s.live_shortcut:
                return name
        return ""

    def _start_live_shortcut(self) -> None:
        """Ctrl+Alt+L (or the user's choice) starts and stops live translation from any app."""
        if self.live_listener is not None:
            self.live_listener.stop()
            self.live_listener = None
        shortcut, label = self.settings.live_shortcut, ""
        if shortcut and not self.live_shortcut_clash():
            try:
                self.live_listener = HotkeyListener(parse_hotkey(shortcut))
                self.live_listener.on_event = self.signals.live_key.emit
                self.live_listener.start()
                label = parse_hotkey(shortcut).label
            except (ValueError, OSError) as e:
                log.warning("Live translation's shortcut %r doesn't work: %s", shortcut, e)
                self.live_listener = None
        self.live_action.setText(f"Live translation\t{label.title()}" if label else "Live translation")

    def _live_key_pump(self) -> None:
        listener = self.live_listener
        while listener is not None and not listener.events.empty():
            event, _ = listener.events.get_nowait()
            if event == "press":
                self.toggle_live()

    def open_live_folder(self) -> None:
        folder = self.live_folder()
        folder.mkdir(parents=True, exist_ok=True)
        shell.open_url(QUrl.fromLocalFile(str(folder)))

    # -- Meeting mode (sst.meeting): live translation, the recording, the minutes and the slides, with one Start

    def meetings_folder(self) -> Path:
        return self.profile.meetings_folder

    def meeting_windows(self) -> list:
        """The meetings open now (Teams, Zoom, Google Meet, Webex), the likeliest first; [] when none."""
        from sst.meeting import apps
        try:
            return apps.find()
        except OSError as e:  # a window list that can't be read is no meeting, not a crash
            log.warning("Couldn't look for a meeting window: %s", e)
            return []

    @property
    def meeting(self):
        """The running MeetingSession, or None (its minutes being written: no longer running)."""
        m = self.last_meeting
        return m if m is not None and m.state == "recording" else None

    def meeting_options(self, in_person: bool = False):
        from sst.meeting.session import MeetingOptions
        s = self.settings
        if in_person:
            return MeetingOptions(**{part: getattr(s, f"room_{part}") for part in ROOM_PARTS}, screen=False)
        return MeetingOptions(**{part: getattr(s, f"meeting_{part}") for part in MEETING_PARTS})

    def set_meeting_option(self, name: str, on: bool, in_person: bool = False) -> None:
        """A part switched on or off: for the next meeting (a running one keeps what it started with)."""
        if name not in (ROOM_PARTS if in_person else MEETING_PARTS):
            raise ValueError(f"No meeting part {name!r}")
        self.apply_settings(dataclasses.replace(self.settings, **{f"{'room' if in_person else 'meeting'}_{name}": on}))

    def meeting_problem(self, in_person: bool = False) -> str:
        """Why a meeting can't start with the parts switched on, or ""."""
        on = self.meeting_options(in_person).effective()
        if not on.anything():
            return "Switch on live translation or Record first."
        if (on.translate or on.minutes) and not self.gateway.key_for("gemini"):
            return "Live translation and the minutes use Google Gemini: add a Gemini key in AI & models."
        return ""

    def meeting_status(self) -> tuple:
        """(state, window, folder, result, error) of this run's latest meeting; state "" before the first."""
        m = self.last_meeting
        return (m.state, m.window, m.folder, m.result, m.error) if m is not None else ("", None, None, None, "")

    def meeting_cost(self):
        """The latest meeting's MeetingCost (so far, while it runs), or None."""
        m = self.last_meeting
        if m is None or not m.state:
            return None
        try:
            return m.cost()
        except Exception as e:  # a cost figure never stops the meeting
            log.warning("The meeting's cost: %s", e)
            return None

    def start_meeting(self, window) -> str:
        """Start a meeting in `window` (a MeetingWindow) with the parts switched on; "" or why it didn't start."""
        if self.meeting is not None:
            return ""
        if window is None:
            return "Start a meeting first: Teams, Zoom, Google Meet or Webex."
        from sst.meeting.apps import in_person
        room = in_person(window)
        problem = self.meeting_problem(room)
        if problem:
            return problem
        options, s = self.meeting_options(room), self.settings
        if not s.meeting_told:
            answer = QMessageBox.question(None, f"{APP_NAME}: meeting mode", MEETING_NOTICE)
            if answer != QMessageBox.StandardButton.Yes:
                return "Not started."
            # the notice said what live translation sends too: its own isn't asked again in the meeting
            self.apply_settings(dataclasses.replace(s, meeting_told=True, live_told=s.live_told or options.translate))
        from sst.meeting.session import MeetingSession
        session = MeetingSession(window, options, self._new_meeting_folder(window),
                                 recorder_factory=lambda folder: self._meeting_recorder(folder, room),
                                 frames_factory=self._meeting_frames, live=_MeetingLive(self, room),
                                 summarise=self._summarise_meeting)
        before, self.last_meeting = self.last_meeting, session
        session.on_change = self.signals.meeting.emit  # queued onto this thread when it comes from another
        try:
            session.start()
        except Exception as e:  # e.g. no output device or microphone; the parts started are stopped again
            log.warning("The meeting didn't start: %s", e)
            self.last_meeting = before
            self._meeting_changed()
            return f"The meeting couldn't start: {e}"
        log.info("Meeting started in %s: %s", window.app, session.options)
        return ""

    def stop_meeting(self) -> None:
        """Stop the running meeting; its minutes are written in the background (a notification says when)."""
        m = self.meeting
        if m is not None:
            m.stop()

    def toggle_meeting(self) -> None:
        """The shortcut and the tray: stop the meeting, or start one in the meeting found (several: the page asks); in
        person when Meeting mode's In person tab was the one last open."""
        if self.meeting is not None:
            self.stop_meeting()
            return
        from sst.meeting import apps
        if self.settings.meeting_place == "room":  # the In person tab was the one last open: a meeting in the room
            problem = self.start_meeting(apps.IN_PERSON)
            if problem and problem != "Not started.":
                self._notify(APP_NAME, problem, QSystemTrayIcon.MessageIcon.Warning)
            self.meeting_action.setChecked(self.meeting is not None)
            return
        found = self.meeting_windows()
        window = apps.choose(found)
        if window is None:
            self.window.open("meeting")  # none, or several to pick from: the page shows them
            if not found:
                self._notify(APP_NAME, "Start a meeting first: Teams, Zoom, Google Meet or Webex.")
        else:
            problem = self.start_meeting(window)
            if problem and problem != "Not started.":
                self._notify(APP_NAME, problem, QSystemTrayIcon.MessageIcon.Warning)
        self.meeting_action.setChecked(self.meeting is not None)

    def _new_meeting_folder(self, window) -> Path:
        """<profile>/meetings/2026-10-27 1003 Microsoft Teams, made unique when two start in the same minute."""
        root, name = self.meetings_folder(), f"{datetime.now():%Y-%m-%d %H%M} {window.app}"
        folder, n = root / name, 2
        while folder.exists():
            folder, n = root / f"{name} ({n})", n + 1
        return folder

    def _meeting_recorder(self, folder: Path, room: bool = False):
        from sst.meeting.recorder import Recorder
        # the microphone chosen for dictation (read when it opens), else Windows' default, as live translation does
        return Recorder(folder, speakers=Capture.speakers,
                        microphone=lambda: Capture.microphone(lambda: self.settings.microphone), room=room)

    def _meeting_frames(self, window, folder: Path):
        from sst.meeting.screen import WindowFrames
        return WindowFrames(window, folder)

    def _summarise_meeting(self, folder: Path):
        from sst.meeting.minutes import summarise  # it counts its Gemini calls as the cost tracker's "meeting"
        return summarise(folder, self.gateway.key_for("gemini"))

    def _meeting_changed(self) -> None:
        """The meeting's state changed (on this thread): the tray, a notification once per change, the page."""
        state, window, folder, result, error = self.meeting_status()
        recording = state == "recording"
        self.meeting_action.setChecked(recording)
        self.tray.setIcon(self.recording_icon if recording or (self.dictation and self.dictation.recording) else self.icon)
        self.tray.setToolTip(f"{APP_NAME}: recording the meeting" if recording else f"{APP_NAME}: {self._status[0]}")
        if state != self._meeting_state:
            self._meeting_state = state
            if recording:
                where = "in person" if window.app == "In person" else f"in {window.app}"
                self._notify(APP_NAME, f"Recording the meeting {where}." if self.last_meeting.options.record
                             else f"Meeting mode is on {where}.")
            elif state == "done" and result is not None:
                self._notify("Minutes ready", f"Minutes ready: {result.title}", on_click=lambda f=folder: self.open_meeting(f))
            elif state == "done":
                self._notify(APP_NAME, "The meeting ended." + (" Its recording is saved." if self.last_meeting.options.record
                                                               else ""))
            elif state == "failed":
                self._notify(APP_NAME, f"The minutes couldn't be written: {error}. The recording is kept.",
                             QSystemTrayIcon.MessageIcon.Warning)
        if self.window.isVisible():
            self.window.refresh()

    def past_meetings(self, limit: int = 8) -> list[tuple[Path, datetime, str, float, bool]]:
        """The latest meetings, newest first (not the one recording): (folder, started, app, seconds, has minutes)."""
        try:
            folders = sorted((p for p in self.meetings_folder().iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)
        except OSError:  # no meeting yet
            return []
        running = self.meeting.folder if self.meeting is not None else None
        found = []
        for folder in folders:
            try:
                started = datetime.strptime(folder.name[:15], "%Y-%m-%d %H%M")
            except ValueError:  # not a meeting's folder
                continue
            if folder == running:
                continue
            try:
                meta = json.loads((folder / "meeting.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = {}
            meta = meta if isinstance(meta, dict) else {}
            app = str(meta.get("app") or folder.name[16:].split(" (")[0])
            seconds = meta.get("seconds")
            found.append((folder, started, app, float(seconds) if isinstance(seconds, (int, float)) else 0.0,
                          (folder / "minutes.md").is_file()))
            if len(found) >= limit:
                break
        return found

    def write_minutes(self, folder: Path) -> None:
        """A past meeting's minutes from its recording, on a worker thread: Rflow quit while they were written, or Gemini
        failed then. The page's "Write minutes"; the answer comes back through signals.minutes_written."""
        folder = Path(folder)
        if folder in self.writing_minutes:
            return
        self.writing_minutes.add(folder)

        def work():
            try:
                self.signals.minutes_written.emit(folder, self._summarise_meeting(folder), "")
            except Exception as e:  # noqa: BLE001 - MinutesError in plain words; anything else said as it is
                log.warning("The minutes of %s couldn't be written: %s", folder.name, e)
                self.signals.minutes_written.emit(folder, None, str(e).rstrip("."))

        threading.Thread(target=work, name="meeting-minutes-again", daemon=True).start()
        if self.window.isVisible():
            self.window.refresh()

    def _minutes_written(self, folder: Path, result, error: str) -> None:
        self.writing_minutes.discard(folder)
        if result is not None:
            self._notify("Minutes ready", f"Minutes ready: {result.title}", on_click=lambda f=folder: self.open_meeting(f))
        else:
            self._notify(APP_NAME, f"The minutes couldn't be written: {error}. The recording is kept.",
                         QSystemTrayIcon.MessageIcon.Warning)
        if self.window.isVisible():
            self.window.refresh()

    def open_meeting(self, folder: Path) -> None:
        """A meeting's minutes, or its folder before they're written."""
        minutes = Path(folder) / "minutes.md"
        shell.open_url(QUrl.fromLocalFile(str(minutes if minutes.is_file() else folder)))

    def open_meeting_folder(self, folder: Path | None = None) -> None:
        """A meeting's folder (the recording, the slides, the minutes), or the one holding them all."""
        folder = Path(folder) if folder is not None else self.meetings_folder()
        folder.mkdir(parents=True, exist_ok=True)
        shell.open_url(QUrl.fromLocalFile(str(folder)))

    def copy_minutes(self, folder: Path) -> str:
        """A meeting's minutes onto the clipboard, to paste anywhere: "" or why they couldn't be copied."""
        try:
            text = (Path(folder) / "minutes.md").read_text(encoding="utf-8")
        except OSError as e:
            log.warning("Couldn't read the minutes: %s", e)
            return "The minutes couldn't be read"
        try:
            textaccess.set_clipboard(text)
        except OSError as e:  # another app holds the clipboard
            log.warning("Couldn't copy the minutes: %s", e)
            return "Another app is using the clipboard: try again"
        return ""

    def meeting_shortcut_clash(self) -> str:
        """The tool already using Meeting mode's shortcut, or "": the shortcut then stays with that tool."""
        s = self.settings
        for name, shortcut in (("Live translation", s.live_shortcut), ("Translate", s.translate_shortcut),
                               ("Text Transform", s.transform_shortcut)):
            if s.meeting_shortcut and shortcut == s.meeting_shortcut:
                return name
        return ""

    def _start_meeting_shortcut(self) -> None:
        """Ctrl+Alt+M (or the user's choice) starts and stops a meeting from any app."""
        if self.meeting_listener is not None:
            self.meeting_listener.stop()
            self.meeting_listener = None
        shortcut, label = self.settings.meeting_shortcut, ""
        if shortcut and not self.meeting_shortcut_clash():
            try:
                self.meeting_listener = HotkeyListener(parse_hotkey(shortcut))
                self.meeting_listener.on_event = self.signals.meeting_key.emit
                self.meeting_listener.start()
                label = parse_hotkey(shortcut).label
            except (ValueError, OSError) as e:
                log.warning("Meeting mode's shortcut %r doesn't work: %s", shortcut, e)
                self.meeting_listener = None
        self.meeting_action.setText(f"Meeting mode\t{label.title()}" if label else "Meeting mode")

    def _meeting_key_pump(self) -> None:
        listener = self.meeting_listener
        while listener is not None and not listener.events.empty():
            event, _ = listener.events.get_nowait()
            if event == "press":
                self.toggle_meeting()

    def run_translation(self, text: str, target: str, second: str = ""):
        """`text` translated (sst.translate.Translation); raises when the provider can't be asked."""
        s = self.settings

        def complete(prompt: str, message: str) -> str:
            return Polisher(self.gateway, s.cleanup_model, [], fallback=s.cleanup_fallback or None,
                            system_prompt=prompt).complete(message)
        with usage.feature("translate"):  # Translate's "Try it" has said "tests" already
            result = Translator(complete).translate(text, target, second)
        log.info("Translated %d characters into %s in %.1f s%s", len(text), result.target, result.seconds,
                 f" ({'; '.join(result.warnings)})" if result.warnings else "")
        return result

    def snippets(self) -> list[Snippet]:
        """The profile's snippets, as set now (sst.snippets): read at each dictation, so a change counts at once."""
        return load_snippets(self.settings.snippets)

    def voice_command(self, text: str) -> str | None:
        """The Text Transform command a whole dictation is ("make it concise", "undo that"), or None: type it. On the
        dictation's thread. Without an AI model a transform's phrase is typed as said (nothing is lost); "undo" is
        always a command, so the pill can say there is nothing to undo."""
        s = self.settings
        if not s.voice_commands:
            return None
        command = match_command(text, phrases_for(s.command_phrases))
        return command if command == UNDO or (command and self.transform_ready()) else None

    def run_transform(self, text: str, key: str):
        """The transform of `text` (sst.transform.TransformResult); raises when the provider can't be asked."""
        s = self.settings
        terms = [t for t in speech_hints(self.dictionary.hint_terms()) if t.casefold() in text.casefold()]

        def complete(prompt: str, message: str) -> str:
            return Polisher(self.gateway, s.cleanup_model, [], fallback=s.cleanup_fallback or None,
                            system_prompt=prompt).complete(message)
        # The user's name for Japanese emails, Translate's languages (as the Translate popup has them)
        context = TransformContext.of(s, system_language())
        with usage.feature("transform"):  # Text Transform's "Try it" has said "tests" already
            result = Transformer(complete, context=context).transform(text, key, terms)
        log.info("Text Transform %s: %s in %.1f s after %d attempt(s)%s", key, "done" if result.accepted else "kept the text",
                 result.seconds, result.attempts, f" ({'; '.join(result.reasons)})" if result.reasons else "")
        return result

    def ai_models(self, gateway: GatewayConfig) -> list[str]:
        """The models a provider lists for this key (the welcome picks its fast one)."""
        return Polisher(gateway, "").models()

    def check_ai(self, gateway: GatewayConfig, model: str) -> str:
        """A short answer from the model, or an error (AI & models' Test, the welcome's Connect)."""
        return Polisher(gateway, model, self.settings.vocabulary).check()

    def open_window(self, page: str) -> None:
        """The window at a page (Translate's "Set up AI cleanup")."""
        self.window.open(page)

    def say(self, state: str, message: str = "") -> None:
        """A short word in the pill (and the log, for warnings)."""
        if state == "warning":
            log.info("Text Transform: %s", message)
        self.pill.show_state(state, message)

    def remember(self, text: str, original: str) -> None:
        """A transform in Home's history, with the text it came from (its tooltip)."""
        self._add_history(text, original)
        if self.window.isVisible():
            self.window.refresh()

    def _add_history(self, text: str, heard: str | None) -> None:
        """Into Home's history. A folder that can't be written is said once; it never fails the dictation (typed)."""
        try:
            add_to_history(text, heard, path=self.profile.history_file)
        except OSError as e:
            log.warning("Couldn't save to the history: %s", e)
            if not self._history_unsaved:
                self._history_unsaved = True
                self._tell(f"Rflow couldn't save your dictations to the history ({e}). They're still typed.")

    def dictionary_terms(self) -> list:
        return self.dictionary.terms(enabled_only=False)

    def correct_dictation(self, typed: str, corrected: str) -> None:
        """Home: the user corrected a dictation. Learned as a suggestion once the same correction comes twice."""
        new = self.learner.observe(CorrectionEvent(typed, corrected, app="history"))
        if new:
            s = new[0]
            self._notify(APP_NAME, f"You corrected \u201c{s.original_phrase}\u201d to {s.corrected_phrase} "
                                   f"{s.seen_count} times. Learn it on the Words page?")
        self.window.refresh()

    def correction_suggestions(self) -> list:
        return self.learner.suggestions()

    def accept_suggestion(self, suggestion) -> None:
        self.learner.accept(suggestion)
        if self.dictation:
            self._apply_speech()
        self.window.refresh()

    def reject_suggestion(self, suggestion) -> None:
        self.learner.reject(suggestion)
        self.window.refresh()

    def add_sound_alike(self, heard: str, meant: str) -> None:
        """The Dictionary page: what the speech model writes for a term, written as the term from now on (an exact
        match: the dictionary fixes it before the formatting and the AI cleanup). Raises ValueError when the sound-alike
        already stands for another term."""
        term = self.dictionary.find(meant)
        if term is None:
            self.dictionary.add_term(meant, aliases=[heard], mode=TermMode.AUTOMATIC, source="user")  # asked for: always
        else:
            self.dictionary.add_alias(term.id, heard)
        log.info("Sound-alike added for a dictionary term (%d terms)", len(self.dictionary.terms()))
        if self.dictation:
            self._apply_speech()  # the speech model listens for the term too

    def score_reading(self, folders: list[Path], progress) -> "evaluate.Results":
        """The speech model alone, then with the cleanup model and the backup model, one at a time (the endpoint may be
        small).
        The report goes into the test's folder, or into 'summary' when several tests are scored together."""
        from sst import evaluate
        if not self.dictation:
            raise RuntimeError("The speech model is still loading; try again in a moment." if self.loading_speech
                               else "Choose a speech model first, on the Speech recognition page.")
        s = self.settings
        models = [m for m in dict.fromkeys((s.cleanup_model, s.cleanup_fallback)) if m] if self.gateway.address else []
        polishers = {m.rsplit("/", 1)[-1]: Polisher(self.gateway, m, speech_hints(s.vocabulary)) for m in models}
        engine = self.dictation.engine
        if engine.name in REMOTE:  # scored without Parakeet to fall back on, so that a provider's failure shows
            engine = self._new_engine(engine.name, fallback=False)
            engine.words = speech_hints(s.vocabulary)
        with usage.feature("tests"):
            results = evaluate.run(folders, engine, evaluate.pipelines_for(polishers, title=engine.title), progress)
        results.save(evaluate.output_folder(folders))
        return results

    def choose_speech_model(self, key: str) -> None:
        """The Speech recognition page: use this model from now on (it loads in the background)."""
        if key in SPEECH_MODELS and usable(key, self.gateway) == key and key != self.settings.speech_model:
            self.apply_settings(dataclasses.replace(self.settings, speech_model=key))

    def use_cloud_speech(self, provider: str, api_key: str, model: str) -> None:
        """A cloud card's "Use this model", or "Save" while it is in use: keep its key (shared with AI cleanup) and its
        model, and use it from the next dictation. The window has asked first: the voice goes to the provider."""
        if provider not in CLOUD or not api_key:
            return
        if api_key != self.gateway.key_for(provider):
            self.gateway = self.gateway.with_key(provider, api_key)
            self.gateway.save(self.profile.gateway_file)
            if self.dictation:
                self._apply_cleanup()  # AI cleanup may use the same provider, and so the same key
        models = {**self.settings.speech_cloud_models, provider: model or CLOUD[provider].models[0]}
        self.apply_settings(dataclasses.replace(self.settings, speech_model=provider, speech_cloud_models=models))

    def test_cloud_speech(self, provider: str, api_key: str, model: str, address: str | None = None) -> str:
        """A cloud or server card's Test button (the window runs it on a thread): the sample sentence through the
        provider. Raises with a readable reason."""
        sample = _sample_sentence()
        if sample is None:
            raise RuntimeError("the sample sentence is missing (it comes with Parakeet)")
        engine = load_engine(provider, "", api_key, model, url=address)  # the sample is English, whatever the language
        engine.words = speech_hints(self.settings.vocabulary)
        return engine.check(*sample)

    def use_server_speech(self, address: str, api_key: str, model: str) -> None:
        """The server card's "Use this model", or "Save" while it is in use: keep the address and key (apart from AI
        cleanup's, encrypted) and the model, and use it from the next dictation."""
        if not address or not model:
            return
        if (address, api_key) != self.gateway.speech_server():
            self.gateway = self.gateway.with_entry(SPEECH_SERVER, address, api_key)
            self.gateway.save(self.profile.gateway_file)
            if self.dictation:
                self._apply_speech()  # a server in use goes to the new address from the next dictation
        self.apply_settings(dataclasses.replace(self.settings, speech_model=SERVER.key, speech_server_model=model))

    def test_server_speech(self, address: str, api_key: str, model: str) -> str:
        return self.test_cloud_speech(SERVER.key, api_key, model, address)

    def server_models(self, address: str, api_key: str) -> list[str]:
        """The server card's "Load models" (on a thread): the server's models, its speech models first."""
        return CloudEngine(SERVER.key, api_key, url=address).models()

    def set_speech_language(self, code: str) -> None:
        if code != self.settings.speech_language:
            self.apply_settings(dataclasses.replace(self.settings, speech_language=code))

    def download_speech_model(self, key: str) -> None:
        """Download a speech model in the background (one at a time), then use it."""
        model = SPEECH_MODELS.get(key)
        if not model or not model.download or self.downloading:
            return
        self.downloading = (key, 0, model.download.size)
        self._cancel_download.clear()
        self.window.refresh()
        last = [-1]

        def progress(done: int, total: int) -> None:
            percent = done * 100 // total if total else 0
            if percent != last[0]:  # the window needn't redraw for every megabyte
                last[0] = percent
                self.signals.download_progress.emit(key, done, total)

        def work() -> None:
            try:
                downloads.download(model.download, progress, self._cancel_download.is_set)
                self.signals.download_done.emit(key, "")
            except downloads.Cancelled:
                self.signals.download_done.emit(key, "cancelled")
            except Exception as e:
                log.exception("Downloading %s failed", key)
                self.signals.download_done.emit(key, str(e) or type(e).__name__)
        threading.Thread(target=work, name="model-download", daemon=True).start()

    def cancel_download(self) -> None:
        self._cancel_download.set()  # what is downloaded so far stays, and the next try goes on from there

    def remove_speech_model(self, key: str) -> None:
        """Free the disk space of a downloaded model that isn't in use."""
        model = SPEECH_MODELS.get(key)
        if model and model.download and model.download.installed() and \
                key not in (self.settings.speech_model, self.speech_in_use()) and \
                not (self.downloading and self.downloading[0] == key):
            downloads.remove(model.download)
            log.info("Removed the download of %s", key)
            self.window.refresh()

    def scan_computer(self) -> None:
        """Scan my computer, in the background: the hardware, a benchmark, and each downloaded model timed on the
        speech model's sample sentence (the model in use is timed as it is; others are loaded for it if memory allows)."""
        if self.scanning:
            return
        self.scanning = "Reading this computer..."
        self.window.refresh()
        in_use = self.dictation.engine if self.dictation else None
        local = [m for m in SPEECH_MODELS.values() if m.where == "local" and m.ready]

        def work() -> None:
            try:
                pc = scan.computer()
                measured = {}
                sample = _sample_sentence()
                for model in local if sample is not None else []:
                    if not model.installed():
                        continue
                    if in_use is not None and in_use.name == model.key:
                        engine = in_use
                    elif pc.free_memory_gb >= scan.MEMORY_GB.get(model.key, 1.0) + 1:
                        self.signals.scan_progress.emit(f"Loading {model.name} to try it...")
                        engine = load_engine(model.key)
                    else:
                        continue  # not enough free memory to load it next to the one in use: estimated instead
                    self.signals.scan_progress.emit(f"Trying {model.name} on a short sentence...")
                    t0 = time.perf_counter()
                    engine.transcribe(*sample)
                    measured[model.key] = time.perf_counter() - t0
                    del engine
                self.signals.scan_done.emit(scan.save(pc, scan.judge(pc, local, measured)))
            except Exception as e:
                log.exception("Scanning the computer failed")
                self.signals.scan_done.emit({"error": str(e) or type(e).__name__})
        threading.Thread(target=work, name="scan", daemon=True).start()

    def _on_scan_progress(self, message: str) -> None:
        self.scanning = message
        self.window.refresh()

    def _on_scan_done(self, data: dict) -> None:
        self.scanning = ""
        if "error" in data:
            self._notify(APP_NAME, f"The scan didn't finish: {data['error']}", QSystemTrayIcon.MessageIcon.Warning)
        else:
            self.last_scan = data
            log.info("Scan: %s", data["computer"])
        self.window.refresh()

    def _on_download_progress(self, key: str, done: int, total: int) -> None:
        self.downloading = (key, done, total)
        self.window.refresh()

    def _on_download_done(self, key: str, error: str) -> None:
        self.downloading = None
        name = SPEECH_MODELS[key].name
        if not error:
            log.info("%s downloaded", name)
            self.choose_speech_model(key)  # what the user asked for: download it and use it
            self._load_speech()  # also when it was chosen already (Parakeet, the first time)
        elif error != "cancelled":
            self._notify(APP_NAME, f"Could not download {name}: {error}", QSystemTrayIcon.MessageIcon.Warning)
        self.window.refresh()

    def speech_in_use(self) -> str:
        """The speech model dictation uses right now ("" while the first one loads)."""
        return self.dictation.engine.name if self.dictation else ""

    def finish_welcome(self, name: str = "") -> None:
        if name.strip():
            self.rename_profile(self.profile.id, name)
        self.apply_settings(dataclasses.replace(self.settings, welcomed=True))

    def window_closed(self) -> None:
        if not self.settings.told_about_tray:
            self._notify(f"{APP_NAME} is still running", f"Dictate with {self.hotkey_label()} as usual. Click the tray "
                         f"icon to open {APP_NAME}; right-click it to quit.")
            self.apply_settings(dataclasses.replace(self.settings, told_about_tray=True))

    # -- profiles: each person's own settings, words, AI provider and keys, history, stats and reading tests

    def _load_profile(self) -> None:
        self.profile = self.profiles.current
        self.settings = Settings.load(self.profile.settings_file)
        while LOAD_PROBLEMS:
            self._tell(LOAD_PROBLEMS.pop(0))
        self.gateway = GatewayConfig.load(self.profile.gateway_file)
        self.stats = Stats.load(self.profile.stats_file, history=self.profile.history_file)
        # What Gemini costs (AI & models' card): the profile's own count; the one before is saved as it goes
        self.usage = usage.Ledger(self.profile.usage_file, on_change=self._usage_counted)
        usage.use(self.usage)
        old = getattr(self, "dictionary", None)
        # Each person's dictionary (terms, sound-alikes, learned corrections), with "Your words" mirrored into it.
        self.dictionary, problem = DictionaryStore.open(self.profile.folder() / "dictionary.db")
        if problem:
            self._tell(problem)
        self.dictionary.sync_vocabulary(self.settings.vocabulary)
        self.learner = Learner(self.dictionary)  # corrections made twice become suggestions, never rules by themselves
        if old is not None:
            old.close()

    def _usage_counted(self) -> None:
        """A Gemini call was counted, on the thread that made it: the cost card updates on the UI thread."""
        signals = getattr(self, "signals", None)
        if signals is not None:
            signals.usage.emit()

    def _on_usage(self) -> None:
        window = getattr(self, "window", None)
        if window is not None and window.isVisible():
            window.usage_changed()

    def bench_dir(self) -> Path:
        return self.profile.folder(bench.BENCH_DIR)

    def switch_profile(self, profile_id: str) -> None:
        if profile_id == self.profile.id or not self.profiles.get(profile_id):
            return
        self.profiles.active = profile_id
        self.profiles.save()
        self._activate_profile()

    def create_profile(self, name: str) -> None:
        profile = self.profiles.add(name)
        self.profiles.save()
        self.switch_profile(profile.id)  # a new profile starts with the welcome

    def rename_profile(self, profile_id: str, name: str) -> None:
        profile = self.profiles.get(profile_id)
        if profile and name.strip() and name.strip() != profile.name:
            profile.name = name.strip()
            self.profiles.save()
            self.window.refresh()

    def delete_profile(self, profile_id: str) -> None:
        """Delete another profile and its files (settings, keys, history, stats, reading tests)."""
        profile = self.profiles.get(profile_id)
        if not profile or profile.id == self.profile.id:
            return  # the one in use can't go; switch first (and the first profile can't go at all)
        self.profiles.remove(profile_id)
        self.profiles.save()
        shutil.rmtree(profile.folder(), ignore_errors=True)
        shutil.rmtree(profile.folder(bench.BENCH_DIR), ignore_errors=True)
        self.window.refresh()
        log.info("Deleted profile %s", profile.id)

    def _activate_profile(self) -> None:
        if self.dictation and self.dictation.recording:
            self.dictation.close()  # a recording started for the other profile is dropped
        old_hotkey = self.settings.hotkey
        self._load_profile()
        self.wake.load()  # the other profile's keyword and voice (applied with the microphone, next)
        self._configure_dictation_mic()
        if self.dictation:
            self.dictation.sounds, self.dictation.save = self.settings.sounds, self.settings.save_recordings
            self._apply_cleanup()
            if self.settings.hotkey != old_hotkey:
                self._start_listener()
            self._load_speech()  # the other profile may use another speech model
        self.transforms.start(self.settings.transform_shortcut)
        self.translator.start(self.settings.translate_shortcut)
        self.live.stop()  # the other profile may use another key, and keeps its own transcripts
        self._start_live_shortcut()
        # Every page shows the profile's own data: build the window again rather than update each field.
        old, self.window = self.window, MainWindow(self)
        self.window.set_status(*self._status)
        if self.update:
            self.window.show_update(f"Rflow {self.update.version} is available (you have {__version__}).",
                                    version=self.update.version)
        if old.isVisible():
            self.window.setGeometry(old.geometry())
            self.window.open()
        old.hide()
        old.deleteLater()
        log.info("Profile: %s (%s)", self.profile.label, self.profile.id)

    # -- updates

    def check_for_updates(self, manual: bool = False) -> None:
        if manual:
            self.window.set_update_status("Checking for updates...")

        def work() -> None:
            try:
                update = updates.check()
            except Exception as e:  # offline, rate-limited...: try again at the next check
                self.signals.update_failed.emit(f"Could not check for updates: {e}", manual)
                return
            if update:
                self.signals.update_found.emit(update)
            elif manual:
                self.signals.update_none.emit()
        threading.Thread(target=work, name="update-check", daemon=True).start()

    def _on_update_none(self) -> None:
        self.window.set_update_status(f"You have the latest version ({__version__}).")
        if not self.window.isVisible():
            self._notify(APP_NAME, f"You have the latest version ({__version__}).")

    def _on_update_found(self, update) -> None:
        self.update = update
        message = self.install_problem or f"Rflow {update.version} is available (you have {__version__})."
        self.window.show_update(message, version=update.version)
        self.window.set_update_status(message)
        self.update_action.setText(f"Update to Rflow {update.version}...")
        self.update_action.setVisible(True)
        log.info("Update available: %s", update.version)
        if self.install_problem:
            self._notify(f"Rflow {update.version} didn't install", self.install_problem.partition(": ")[2] or
                         "Click here, then Update now.", QSystemTrayIcon.MessageIcon.Warning)
            self._update_told = update.version
        elif self._update_told != update.version:
            self._update_told = update.version
            self._notify(f"Rflow {update.version} is available", "Click here, then Update now.")

    def start_update(self) -> None:
        update = self.update
        if not update:
            return
        self.install_problem = ""  # tried again: the next start says it again if it fails again
        if not getattr(sys, "frozen", False):
            shell.open_url(QUrl(update.page))  # a source checkout: just show the release
            return
        self.window.show_update(f"Downloading Rflow {update.version}...", busy=True)
        last = [-1]

        def progress(done: int, total: int) -> None:
            percent = int(done * 100 / total) if total else 0
            if percent != last[0]:
                last[0] = percent
                self.signals.update_progress.emit(f"Downloading Rflow {update.version}... {percent}%")

        def work() -> None:
            try:
                path = updates.download(update, UPDATE_DIR, progress)
            except Exception as e:
                log.exception("Update download failed")
                self.signals.update_failed.emit(f"Update failed: {e}", True)
                return
            self.signals.update_ready.emit(str(path))
        threading.Thread(target=work, name="update-download", daemon=True).start()

    def busy(self) -> str:
        """What Rflow is in the middle of that an update mustn't cut ("" when nothing)."""
        if self.dictation is not None and self.dictation.busy:
            return "this dictation"
        if self.live.running:
            return "live translation stops"
        return ""

    def _install_when_idle(self) -> None:
        if self._update_waiting and not self.busy():
            self._on_update_ready(self._update_waiting)

    def _on_update_ready(self, path: str) -> None:
        if doing := self.busy():  # never cut a dictation or a live translation: install once it's over
            self._update_waiting = path
            version = self.update.version if self.update else "The update"
            self.window.show_update(f"Rflow {version} is ready: it installs when {doing} is done.", busy=True)
            self._update_timer.start()
            log.info("The update waits: %s", doing)
            return
        self._update_timer.stop()
        self._update_waiting = ""
        log.info("Installing %s; Rflow restarts when it's done", path)
        self.window.show_update("Installing... Rflow restarts by itself when it's done.", busy=True)
        release_running_mutex()  # otherwise the installer stops to ask for Rflow to be closed
        self.server.close()  # the new version opens its own
        try:
            updates.install(Path(path))
        except OSError as e:
            hold_running_mutex()
            self.server.listen(SERVER_NAME)
            self._on_update_failed(f"Could not start the installer: {e}", True)
            return
        self.quit()

    def _on_update_failed(self, message: str, tell: bool) -> None:
        log.warning(message)
        if tell:
            self._notify(APP_NAME, message, QSystemTrayIcon.MessageIcon.Warning)
        self.window.set_update_status(message)
        if self.update:
            self.window.show_update(f"{message} You can try again.")

    def open_release_notes(self) -> None:
        if self.update and self.update.page:
            shell.open_url(QUrl(self.update.page))

    def _stop_all(self) -> None:
        """Stop the shortcuts' keyboard hooks, live translation and dictation (a recording is dropped), before quitting."""
        self.wake.stop()
        self.transforms.stop()
        self.translator.stop()
        self.stop_meeting()  # the recording is kept; minutes still being written are lost with Rflow
        self.live.stop()
        for listener in (self.live_listener, self.meeting_listener):
            if listener is not None:
                listener.stop()
        if self.listener:
            self.listener.stop()
        if self.dictation:
            self.dictation.close()
        self.server.close()
        self.usage.flush()  # after live translation stopped: its last minutes are counted

    def quit(self) -> None:
        self._stop_all()
        self.tray.hide()
        mark_clean_exit(LOG_DIR)
        QApplication.quit()

    def start_over(self, keep_models: bool = True) -> None:
        """Settings' Start over: stop dictation and live translation, let go of every open file (the dictionary, the
        log, the microphone), delete what Rflow keeps (every profile's settings, words, snippets, history, stats, keys,
        recordings, reading tests, live transcripts, logs; the downloaded speech models too unless kept), and start
        Rflow again: it opens on the welcome, like a new install."""
        log.info("Starting over: the downloaded speech models are %s", "kept" if keep_models else "deleted too")
        self._cancel_download.set()  # a download in progress stops writing into the models folder
        self.ticker.stop()
        self._stop_all()
        self.dictation, self._local = None, None  # the speech models let go of, so their files can be deleted too
        gc.collect()
        self.recorder.close()
        self.dictionary.close()
        close_log_files()
        erase_data(DATA_DIRS, keep=(DOWNLOADS_DIR.name,) if keep_models else ())
        release_running_mutex()
        try:  # the new Rflow waits for this one to end, then starts on the welcome
            subprocess.Popen(restart_command(os.getpid()), close_fds=True,
                             creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
        except OSError as e:
            QMessageBox.information(None, APP_NAME, f"Everything is deleted, but {APP_NAME} couldn't start again ({e}). "
                                                    "Start it from the Start menu.")
        self.tray.hide()
        self.window.hide()
        QApplication.exit(0)  # not quit(): that asks the windows first, and the main window only hides when closed

    def _set_status(self, message: str, ready: bool = False) -> None:
        self._status = (message, ready)  # a window built later (another profile) shows it too
        self.status_action.setText(message)
        self.tray.setToolTip(f"{APP_NAME}: recording the meeting" if self.meeting is not None else f"{APP_NAME}: {message}")
        self.window.set_status(message, ready)

    def _notify(self, title: str, message: str, icon=QSystemTrayIcon.MessageIcon.Information,
                on_click: Callable[[], None] | None = None) -> None:
        self._on_message = on_click  # a notification replaces the one before, and what clicking it did
        self.tray.showMessage(title, message, icon, 5000)

    def _message_clicked(self) -> None:
        action, self._on_message = self._on_message, None
        action() if action is not None else self.window.open()

    def _tell(self, message: str) -> None:
        """A warning the user must see: a notification now, or once the tray icon is there (while starting)."""
        log.warning(message)
        if getattr(self, "tray", None) is not None:
            self._notify(APP_NAME, message, QSystemTrayIcon.MessageIcon.Warning)
        else:
            self._told_later.append(message)


def _sample_sentence():
    """The sentence models are tried on (SAMPLE_FILE): (audio, rate), or None if it is missing."""
    from sst.audio import load_wav
    try:
        return load_wav(SAMPLE_FILE)
    except OSError:
        return None


def _with_red_dot(icon: QIcon) -> QIcon:
    pixmap = icon.pixmap(64, 64)
    p = QPainter(pixmap)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(QColor(255, 255, 255))
    p.setBrush(QColor(239, 68, 68))
    p.drawEllipse(QPointF(48, 48), 13, 13)
    p.end()
    return QIcon(pixmap)


# ---------------------------------------------------------------- entry points

def self_test() -> int:
    """Build every window off-screen and transcribe once; exit code 0 if it all works (used by the installer build)."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    app = QApplication(sys.argv)
    pill = Pill(level=lambda: 0.05)
    for state in ("recording", "transcribing", "typed", "error"):
        pill.show_state(state, "test")
        pill.grab()
    preview = PreviewApp(history=[{"time": "2026-09-30 10:15:00", "text": "Self-test."}])
    window = MainWindow(preview)
    window.show_update("Rflow 9.9.9 is available (you have 1.0.0).", version="9.9.9")
    if not theme.load_fonts():  # the shipped Geist fonts (sst/static/fonts) made it into the build
        return 1
    from sst.ui import brand_mark
    if brand_mark(64).isNull() or QIcon(str(ICON_FILE)).isNull():  # the logo files (sst/static/brand, sst.ico)
        return 1
    for name in ("dark", "light"):
        window.apply_theme(name)
        for page in window.pages:  # every page, so the icons (Qt SVG) and the soft shadows are drawn in the build too
            window.show_page(page)
            window.grab()
    window.pages["reading"].ensure_test().grab()
    from sst.transformui import TransformMenu
    from sst.translate import Translation
    from sst.translateui import TranslatePopup
    popup = TranslatePopup()
    popup.show_result(Translation("Self-test.", "English", "Self-test.", 0.1))
    popup.grab()
    menu = TransformMenu()
    menu.items, menu.source = [("concise", "1", "Concise"), ("undo", "U", "Undo")], "Self-test"
    menu.grab()
    WakeSetupDialog(microphone="Self-test", problem="", check=lambda text: None, enroll=lambda keyword: None,
                    new_recorder=lambda: None, save=lambda setup: None).grab()
    _with_red_dot(QIcon(str(ICON_FILE)))
    import ctranslate2  # noqa: F401 (Whisper's runtime: bundled and its DLLs load, without needing the downloaded model)
    import faster_whisper  # noqa: F401

    from sst.engines.parakeet import find_model
    sample = _sample_sentence()
    if sample is None:
        return 1
    if find_model():  # the build's smoke test puts Parakeet next to the program; the installer comes without it
        if not load_engine("parakeet").transcribe(*sample):
            return 1
    app.quit()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--self-test" in argv:
        return self_test()
    if instance.hand_over(argv):  # asked already by Rflow.exe's entry (instance.launch), before these imports
        return 0
    log_dir = setup_logging()
    log.info("%s %s starting (%s)", APP_NAME, __version__, scan.machine())  # which kind of computer the log is from
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setQuitOnLastWindowClosed(False)  # closing the window keeps Rflow in the tray
    app.setWindowIcon(QIcon(str(ICON_FILE)))
    hold_running_mutex()
    if not QSystemTrayIcon.isSystemTrayAvailable():
        QMessageBox.critical(None, APP_NAME, "Windows has no notification area (system tray) available.")
        return 1
    left = install_crash_handlers(log_dir)  # only now: a second copy that just hands over mustn't touch them
    put_back_left()  # apps a crash left lowered while live translation spoke: back to their own volume
    app.aboutToQuit.connect(lambda: mark_clean_exit(log_dir))
    app.commitDataRequest.connect(lambda manager: mark_clean_exit(log_dir))  # Windows signs out or shuts down
    tray_app = start(lambda: TrayApp(quiet_start="--startup" in argv), log_dir, left)
    if tray_app is None:
        return 1
    try:
        return app.exec()
    except Exception:
        log.exception("Crashed; logs in %s", log_dir)
        raise


def start(make: Callable[[], "TrayApp"], log_dir: Path, left: str = "") -> "TrayApp | None":
    """The app, or None after saying why it couldn't start (instead of vanishing). `left`: the crash report the last
    run left behind, mentioned once the app is up."""
    try:
        tray_app = make()
    except Exception as e:
        log.exception("Rflow couldn't start")
        mark_clean_exit(log_dir)  # it didn't crash: it said why
        QMessageBox.critical(None, APP_NAME, f"{APP_NAME} couldn't start: {e}\n\nThe details are in the log, in "
                                             f"{log_dir}. Your settings and words are kept.")
        return None
    if left:
        tray_app._notify(APP_NAME, f"{APP_NAME} closed unexpectedly last time. A crash report is in its logs folder "
                                   f"({left}).", QSystemTrayIcon.MessageIcon.Warning)
    return tray_app
