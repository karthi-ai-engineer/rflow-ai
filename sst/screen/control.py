"""Screen translator for Rflow's app (sst.app.TrayApp): its shortcut (Ctrl+Alt+S by default), start and stop, the
one-time notice, and where the frame was left (Settings.screen_lens). The frame and its threads are made only when it
starts, so Rflow's start-up stays as fast as before.

Started from the shortcut, the frame covers the window the user was in (its page area for a browser); from Rflow's
window or the tray, it opens where it was left. Needs the profile's Gemini key and Windows' Japanese OCR.
"""
import dataclasses
import logging
import os
import threading

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMessageBox

from sst.hotkey import HotkeyListener, parse_hotkey
from sst.screen import translate

log = logging.getLogger(__name__)
TITLE = "Screen translator"
OCR_LANGUAGE = "ja"


def ocr_problem(language: str = OCR_LANGUAGE) -> str:
    """"" when Windows can read this language, else why not, in plain words (tried on a thread of its own: WinRT
    objects belong to the thread that made them)."""
    found: list[str] = []

    def check():
        try:
            from sst.screen.winocr import WinOcr
            WinOcr(language).close()
        except Exception as e:  # OcrMissing, or no WinRT at all
            found.append(str(e) or type(e).__name__)

    t = threading.Thread(target=check, name="screen-ocr-check", daemon=True)
    t.start()
    t.join(10.0)
    return found[0] if found else ""


class _Signals(QObject):
    key = Signal()


class ScreenControl:
    """The app's side: app is the TrayApp (settings, apply_settings, gateway, _notify), action the tray's checkable
    "Screen translator"."""

    def __init__(self, app, action, *, make_lens=None, check_ocr=ocr_problem, ask=None):
        self.app, self.action = app, action
        self.lens = None
        self.listener = None
        self._make_lens, self._check_ocr = make_lens or self._new_lens, check_ocr
        self._ask = ask or _ask
        self.signals = _Signals()
        self.signals.key.connect(self._key_pump)
        action.setCheckable(True)
        action.triggered.connect(lambda: self.toggle())

    @property
    def running(self) -> bool:
        return self.lens is not None and self.lens.running

    # ------------------------------------------------------------ the shortcut

    def clash(self) -> str:
        """The tool already using the shortcut, or "": the shortcut then stays with that tool."""
        s = self.app.settings
        others = (("Live translation", s.live_shortcut), ("Meeting mode", s.meeting_shortcut),
                  ("Translate", s.translate_shortcut), ("Text Transform", s.transform_shortcut), ("Dictation", s.hotkey))
        return next((name for name, shortcut in others if s.screen_shortcut and shortcut == s.screen_shortcut), "")

    def start_shortcut(self) -> None:
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
        shortcut, label = self.app.settings.screen_shortcut, ""
        if shortcut and not self.clash():
            try:
                self.listener = HotkeyListener(parse_hotkey(shortcut))
                self.listener.on_event = self.signals.key.emit
                self.listener.start()
                label = parse_hotkey(shortcut).label
            except (ValueError, OSError) as e:
                log.warning("Screen translator's shortcut %r doesn't work: %s", shortcut, e)
                self.listener = None
        self.action.setText(f"{TITLE}\t{label.title()}" if label else TITLE)

    def _key_pump(self) -> None:
        listener = self.listener
        while listener is not None and not listener.events.empty():
            event, _ = listener.events.get_nowait()
            if event == "press":
                self.toggle(fit_foreground=True)

    # ------------------------------------------------------------ start and stop

    def problem(self) -> str:
        if not self.app.gateway.key_for("gemini"):
            return "Screen translator needs a Gemini API key: add one in AI & models."
        return ""

    def start(self, fit_foreground: bool = False) -> str:
        """Show the frame; "" or why it didn't start."""
        if self.running:
            return ""
        if problem := self.problem() or self._check_ocr():
            return problem
        s = self.app.settings
        if not s.screen_told:
            if not self._ask(
                    "Screen translator reads the Japanese in a frame you put over any app, with Windows' own OCR on "
                    "this laptop, and covers it with English. The text it reads (never a picture of the screen) goes to "
                    f"Google's Gemini to be translated: about ${translate.estimate():.4f} for each new screen, nothing for "
                    "a screen read again. With a free Gemini key Google may use that text to improve its products.\n\n"
                    "Start?"):
                return "Not started."
            self.app.apply_settings(dataclasses.replace(s, screen_told=True))
        from sst.screen import win32
        fit = win32.foreground_other(os.getpid()) if fit_foreground else None
        self.lens = self._make_lens()
        self.lens.start(list(self.app.settings.screen_lens) or None, fit=fit)
        self.action.setChecked(True)
        log.info("Screen translator started")
        return ""

    def _new_lens(self):
        from sst.screen.lens import ScreenLens
        from sst.screen.winocr import WinOcr
        return ScreenLens(translate.Gemini(self.app.gateway.key_for("gemini")), make_ocr=lambda: WinOcr(OCR_LANGUAGE),
                          on_close=self._closed)

    def stop(self) -> None:
        if self.lens is not None:
            self.lens.close()  # _closed follows

    def toggle(self, fit_foreground: bool = False) -> None:
        if self.running:
            self.stop()
            return
        problem = self.start(fit_foreground)
        if problem and problem != "Not started.":
            from PySide6.QtWidgets import QSystemTrayIcon
            self.app._notify("Rflow", problem, QSystemTrayIcon.MessageIcon.Warning)
        self.action.setChecked(self.running)
        self._refresh()

    def _closed(self, geometry: list[int]) -> None:
        """The frame closed (its ✕, the shortcut, the tray): where it was is kept for the next start."""
        self.lens = None
        self.action.setChecked(False)
        log.info("Screen translator stopped")
        if list(self.app.settings.screen_lens) != geometry:
            self.app.settings = dataclasses.replace(self.app.settings, screen_lens=list(geometry))
            self.app.settings.save(self.app.profile.settings_file)  # quietly: nothing else depends on it
        self._refresh()

    def _refresh(self) -> None:
        window = getattr(self.app, "window", None)
        if window is not None and window.isVisible():
            window.refresh()

    def quit(self) -> None:
        self.stop()
        if self.listener is not None:
            self.listener.stop()
            self.listener = None


def _ask(text: str) -> bool:
    return QMessageBox.question(None, f"Rflow: {TITLE.lower()}", text) == QMessageBox.StandardButton.Yes
