"""Translate (the owner's idea of 2026-10-02, like DeepL): select text in any app, press Ctrl+C twice, read it translated.

  Ctrl+C+C (the shortcut)   the app copies as usual; Rflow reads that copy, presses nothing, and opens a small popup
                            at the pointer: "Japanese → English" at the top, the translation below (sst.translate)
  a language at the top     translates again into it, and remembers it; More shows every language
  Copy (C)                  the translation on the clipboard ("Copied"; the popup stays)
  Replace (Enter)           the translation in place of the selected text (its window brought back first, like Text
                            Transform; if the text isn't there any more, the translation goes on the clipboard)
  Try again (Enter)         after an error, which is said in plain words (the provider's own below, small)
  Esc, ✕, a click outside   closes the popup (Esc in the language list goes back), as does going to another app

The popup never takes the keyboard focus, so the app keeps its selection; Esc is taken from the keyboard hook while the
popup is open, and so are C and Enter while their buttons are there. Text already in the chosen language goes to the
second language, or else to Windows' own language (or English), never into the same language. Another shortcut (not a
double copy) copies the selection itself, with Ctrl+Insert, never in a terminal. The popup is always in the design's
dark popup look (sst.theme.POPUP), like the pill: it reads on any app.
"""
import dataclasses
import logging
import os
import threading
import time

from PySide6.QtCore import QObject, QPoint, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QCursor, QFontMetrics, QGuiApplication, QPainter
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from sst import theme, translate
from sst.hotkey import VK_ESCAPE, HotkeyListener, parse_hotkey
from sst.theme import font, paint_surface
from sst.transformui import TERMINALS, place
from sst.translate import LANGUAGES, MAX_CHARS, Translation, choose_target, detect, fallback_second
from sst.ui import Button, Host, IconButton, label, set_tone

COPY_WAIT = 0.4  # seconds the app gets to copy after the second Ctrl+C, before the clipboard is read as it is

log = logging.getLogger(__name__)


def _pointer() -> QPoint:
    return QCursor.pos()  # apart, so the tests can say where the pointer is

KEY_REFUSED = "The provider refused the API key. Check it on AI & models (AI connection)."
UNREACHABLE = "Rflow can't reach the provider. Check the internet connection and the AI connection, then try again."
MARGIN = 24  # transparent room around the panel for its soft shadow
VK_C, VK_RETURN = 0x43, 0x0D  # Copy and Replace from the keyboard, as their keycaps say


def explain(error: str) -> str:
    """What a failed translation means for the user, from the provider's error (shown below it, smaller)."""
    e = error.casefold()
    if any(s in e for s in ("401", "403", "unauthorized", "forbidden", "api key", "api_key", "permission")):
        return KEY_REFUSED
    if any(s in e for s in ("429", "quota", "rate limit", "exhausted", "too many requests")):
        return "The provider's limit is reached (quota, or too many requests). Try again in a minute."
    if any(s in e for s in ("timed out", "timeout")):
        return "The provider took too long to answer. Try again."
    if any(s in e for s in ("getaddrinfo", "connection", "unreachable", "network", "11001", "10061", "no address")):
        return UNREACHABLE
    if "no translation" in e:
        return "The model gave no translation. Try again, or choose another model on AI & models."
    return "Couldn't translate this. Try again."


class TranslatePopup(QWidget):
    """The panel at the pointer, in the design's popup look (always Obsidian, over any app). The header says from what
    language ("Japanese →"), with the languages used most as one-click chips and More for all of them; then the
    translation, as tall as it needs (it scrolls past a limit); then Copy [C] and Replace [↵], or Try again [↵] after
    an error. It never takes the keyboard focus (the app keeps its selection for Replace), stays on the screen as it
    grows, and closes on Esc, ✕ or a click outside it."""

    language = Signal(str)  # a language picked in the header or the full list
    copy = Signal()
    replace = Signal()
    retry = Signal()
    setup = Signal()  # open the AI connection page: no model yet, or its key or connection failed
    closed = Signal()
    keys_changed = Signal()  # the keys it takes from the keyboard changed (Copy and Replace come with a translation)

    WIDTH = 460  # the panel; the window has MARGIN more on each side, for the shadow
    INNER = WIDTH - 40  # inside the panel's padding
    RESULT_MAX = 360  # at most, and at most 40% of the screen; then the translation scrolls
    COLUMNS = 3  # of the full language list

    def __init__(self):
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setProperty("neuPopup", True)
        theme.load_fonts()
        self.setStyleSheet(theme.stylesheet("dark"))
        self.setFont(font(14))
        self.setFixedWidth(self.WIDTH + 2 * MARGIN)
        self.anchor, self._above, self._view, self._back, self.target = QPoint(), False, "busy", "busy", ""
        self._choices: list[str] = []
        self._dot = 0
        outer = QVBoxLayout(self)
        outer.setContentsMargins(MARGIN, MARGIN, MARGIN, MARGIN)
        self.frame = Host()
        self.frame.setObjectName("popup")
        outer.addWidget(self.frame)
        layout = QVBoxLayout(self.frame)
        layout.setContentsMargins(20, 14, 14, 16)
        layout.setSpacing(10)

        self.head = QHBoxLayout()
        self.head.setSpacing(8)
        self.route = label("Into", "caption", "2", wrap=False)
        self.head.addWidget(self.route)
        self.chips: list[QPushButton] = []
        self.chip_row = QHBoxLayout()
        self.chip_row.setSpacing(6)
        self.head.addLayout(self.chip_row)
        self.more = self._button("More", "chip", self.toggle_languages, "All languages")
        self.more.icon_name, self.more.icon_after = "chevron-down", True
        self.more.setCheckable(True)
        self.head.addWidget(self.more)
        self.head.addStretch()
        self.close_button = IconButton("close", "Close (Esc)", 28, flat=True)
        self.close_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.close_button.clicked.connect(self.close_popup)
        self.head.addWidget(self.close_button)
        layout.addLayout(self.head)

        self.original = label("", "caption", "3", wrap=False)  # one line: the text is still selected in the app
        self.original.setMinimumHeight(20)  # Japanese falls back to a taller font than Geist: room for it
        layout.addWidget(self.original)
        self.waiting = label("", tone="2", wrap=False)
        layout.addWidget(self.waiting)
        self.dots = QTimer(self)
        self.dots.setInterval(350)
        self.dots.timeout.connect(self._tick)
        self.result = QTextBrowser()
        self.result.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.result.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.result.setFont(font(15))
        self.result.setStyleSheet("QTextBrowser { padding: 0; }")
        self.result.document().setDocumentMargin(0)
        layout.addWidget(self.result)
        self.grid_box = QWidget()
        self.grid = QGridLayout(self.grid_box)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(2)
        self.all_languages: dict[str, QPushButton] = {}
        for i, name in enumerate(LANGUAGES):
            b = self._button(name, "nav", lambda _=False, n=name: self._pick(n))
            b.setCheckable(True)
            self.all_languages[name] = b
            self.grid.addWidget(b, i // self.COLUMNS, i % self.COLUMNS)
        layout.addWidget(self.grid_box)
        self.status = label("", tone="warn")  # a warning about the translation, or what went wrong
        layout.addWidget(self.status)
        self.detail = label("", "caption", "3", wrap=False)  # the provider's own words for an error, small
        layout.addWidget(self.detail)

        self.buttons = QWidget()
        buttons = QHBoxLayout(self.buttons)
        buttons.setContentsMargins(0, 4, 6, 0)
        buttons.setSpacing(10)
        buttons.addStretch()
        self.setup_button = self._button("AI settings", "key", self.setup.emit, "Choose the AI model Translate uses")
        self.copy_button = self._button("Copy", "key", self.copy.emit, "Copy the translation (C)", hint="C")
        self.replace_button = self._button("Replace", "primary", self.replace.emit,
                                           "Put the translation in place of the selected text (Enter)", hint="↵")
        self.retry_button = self._button("Try again", "primary", self.retry.emit, "Try again (Enter)", hint="↵")
        self._offer_setup = False
        for b in (self.setup_button, self.copy_button, self.replace_button, self.retry_button):
            buttons.addWidget(b)
        layout.addWidget(self.buttons)
        self.copied_timer = QTimer(self)
        self.copied_timer.setSingleShot(True)
        self.copied_timer.setInterval(1500)
        self.copied_timer.timeout.connect(self._copy_done)

    def _button(self, text: str, kind: str, on_click, tip: str = "", hint: str | None = None) -> Button:
        b = Button(text, kind, "sm", hint=hint)
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        if tip:
            b.setToolTip(tip)
        b.clicked.connect(on_click)
        return b

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        paint_surface(p, "float", QRectF(self.frame.geometry()), 20, popup=True, dpr=self.devicePixelRatioF())
        p.end()

    def contains_global(self, point: QPoint) -> bool:
        """Is the point on the panel (not its shadow)? A click outside the panel closes it."""
        return self.frame.geometry().translated(self.pos()).contains(point)

    # -- what it shows

    def open_at(self, pos: QPoint, source: str, target: str, choices: list[str] | None = None,
                source_language: str = "") -> None:
        self.anchor = pos
        area = self._area()
        room_below = area.bottom() - pos.y() - 18
        self._above = room_below < 280 and pos.y() - area.top() > room_below  # little room below: open upwards
        shown = " ".join(source.split())
        self.original.setText(self.original.fontMetrics().elidedText(shown, Qt.TextElideMode.ElideRight, self.INNER))
        self.original.setToolTip(shown[:600])
        self.route.setText(f"{source_language} →" if source_language else "Into")
        self._choices = [c for c in (choices or [target]) if c in LANGUAGES]
        self.busy(target)
        self.show()
        try:
            from sst.app import _no_activate  # clicks never take the app's focus: its selection stays for Replace
            _no_activate(int(self.winId()), click_through=False)  # but they do reach the buttons
        except Exception:  # the off-screen test platform has no real window
            pass

    def busy(self, target: str) -> None:
        self.target = target
        self._set_chips(target)
        self._dot = 0
        self.waiting.setText(f"Translating into {target}")
        self.dots.start()
        self.result.setPlainText("")
        self._show("busy")

    def show_result(self, translation: Translation, model: str = "") -> None:
        self.target = translation.target
        self._set_chips(translation.target)
        self.result.setPlainText(translation.text)
        self.result.setToolTip(f"{translation.target}, by {model} in {translation.seconds:.1f} s" if model else "")
        self._say("\n".join(translation.warnings), "warn")
        self.detail.setText("")
        self._show("result")

    def show_error(self, message: str, detail: str = "", setup: bool = False) -> None:
        """What went wrong, Try again, and with `setup` the way to the AI connection (a key or connection problem)."""
        self._say(message, "warn")
        detail = " ".join(detail.split())
        self.detail.setText(self.detail.fontMetrics().elidedText(detail, Qt.TextElideMode.ElideRight, self.INNER))
        self.detail.setToolTip(detail)
        self._offer_setup = setup
        self._show("error")

    def show_setup(self) -> None:
        """No AI model is connected yet: say so here, at the text, with the way to set one up (not a passing notice)."""
        self.dots.stop()
        self._say("Translate needs an AI model. Connect one on AI & models; Translate uses the same.", "warn")
        self.detail.setText("")
        self._show("setup")

    def keys(self) -> set[int]:
        """The keys the popup takes from the keyboard now: Esc always, C and Enter when their buttons are there."""
        if self._view == "result":
            return {VK_ESCAPE, VK_C, VK_RETURN}
        return {VK_ESCAPE, VK_RETURN} if self._view in ("error", "setup") else {VK_ESCAPE}

    def key(self, vk: int) -> None:
        """A key taken from the keyboard while the popup is open."""
        if vk == VK_ESCAPE:
            self.escape()
        elif vk == VK_C and self._view == "result":
            self.copy.emit()
        elif vk == VK_RETURN:
            signal = {"result": self.replace, "error": self.retry, "setup": self.setup}.get(self._view)
            if signal is not None:
                signal.emit()

    def toggle_languages(self) -> None:
        self._show("languages" if self._view != "languages" else self._back)

    def escape(self) -> None:
        """Esc: from the full language list back to the translation; else the popup closes."""
        if self._view == "languages":
            self._show(self._back)
        else:
            self.close_popup()

    def copied(self) -> None:
        self.copy_button.setText("Copied")
        self.copy_button.icon_name, self.copy_button.hint = "check", None
        self.copy_button.updateGeometry()
        self.copied_timer.start()

    def close_popup(self) -> None:
        self.dots.stop()
        if self.isVisible():
            self.hide()
            self.closed.emit()

    # -- inside

    def _pick(self, language: str) -> None:
        self.language.emit(language)

    def _set_chips(self, current: str) -> None:
        """The languages used most, as many as fit beside "More", the current one always among them (and lit)."""
        while self.chip_row.count():
            chip = self.chip_row.takeAt(0).widget()
            chip.hide()
            chip.deleteLater()
        metrics = QFontMetrics(font(13, 600))
        room = self.INNER - metrics.horizontalAdvance(self.route.text()) - metrics.horizontalAdvance("More") - 120
        names = list(dict.fromkeys(self._choices))
        shown: list[str] = []
        for name in names:
            width = metrics.horizontalAdvance(name) + 30
            if len(shown) == 3 or width > room:
                break
            shown.append(name)
            room -= width
        if current not in shown:
            shown = (shown[:-1] if len(shown) == 3 or not shown else shown) + [current]
        self.chips = []
        for name in shown:
            chip = self._button(name, "chip", lambda _=False, n=name: self._pick(n), f"Translate into {name}")
            chip.setCheckable(True)
            chip.setChecked(name == current)
            self.chip_row.addWidget(chip)
            self.chips.append(chip)
        for name, b in self.all_languages.items():
            b.setChecked(name == current)

    def _say(self, message: str, tone: str) -> None:
        set_tone(self.status, tone)
        self.status.setText(message)

    def _show(self, view: str) -> None:
        if view != "languages":
            self._back = view
        self._view = view
        if view != "busy":
            self.dots.stop()
        self.more.setChecked(view == "languages")
        self.waiting.setVisible(view == "busy")
        self.result.setVisible(view == "result")
        self.grid_box.setVisible(view == "languages")
        self.status.setVisible(view in ("result", "error", "setup") and bool(self.status.text()))
        self.detail.setVisible(view == "error" and bool(self.detail.text()))
        self.buttons.setVisible(view in ("result", "error", "setup"))
        self.copy_button.setVisible(view == "result")
        self.replace_button.setVisible(view == "result")
        self.retry_button.setVisible(view == "error")
        self.setup_button.setVisible(view == "setup" or (view == "error" and self._offer_setup))
        self.setup_button.setText("Connect an AI" if view == "setup" else "AI settings")
        self.setup_button.set_kind("primary" if view == "setup" else "key")
        self.setup_button.set_hint("↵" if view == "setup" else None)
        self._fit()
        self.keys_changed.emit()

    def _area(self):
        screen = QGuiApplication.screenAt(self.anchor) or QGuiApplication.primaryScreen()
        return screen.availableGeometry()

    def _fit(self) -> None:
        """Every part as tall as its text (so nothing overlaps), the translation at most RESULT_MAX or 40% of the
        screen; then the popup is placed again, so a grown popup stays on the screen."""
        area = self._area()
        if not self.result.isHidden():
            self.result.ensurePolished()  # the style sheet's font, which the height depends on
            document = self.result.document().clone()  # measured apart: the box lays out its own at its current width,
            document.setDefaultFont(self.result.font())  # which isn't the final one yet (it measured too few lines)
            document.setTextWidth(self.INNER - 2 * self.result.frameWidth())
            height = document.size().height() + 2 * self.result.frameWidth() + 6
            self.result.setFixedHeight(int(min(max(height, 30), self.RESULT_MAX, 0.4 * area.height())))
        if not self.status.isHidden():
            self.status.setFixedHeight(self.status.heightForWidth(self.INNER))  # wrapped lines, measured, not guessed
        for _ in range(2):  # twice: the panel's own layout settles after the window's first pass (it was too tall)
            self.frame.layout().activate()
            self.layout().activate()
            self.adjustSize()
        width, height = self.width() - 2 * MARGIN, self.height() - 2 * MARGIN  # the panel, without its shadow
        x = min(max(area.left(), self.anchor.x() + 12), area.right() - width)
        y = self.anchor.y() - 10 - height if self._above else self.anchor.y() + 18
        self.move(x - MARGIN, min(max(area.top(), y), area.bottom() - height) - MARGIN)

    def _tick(self) -> None:
        self._dot = self._dot % 3 + 1
        self.waiting.setText(f"Translating into {self.target}" + "." * self._dot)

    def _copy_done(self) -> None:
        self.copy_button.setText("Copy")
        self.copy_button.icon_name, self.copy_button.hint = None, "C"
        self.copy_button.updateGeometry()


class TranslateController(QObject):
    """The shortcut, the copied text, the popup and the translation. `app` is the TrayApp (or a stand-in in the tests):
    settings, `translate_ready()`, `run_translation(text, target, second) -> Translation`, `transform_model()`,
    `say(state, message)` and `apply_settings(settings)`. `access` is sst.textaccess (fakes in the tests)."""

    _got = Signal(object)  # (text, window) or why there is none (str), from the reading thread
    _done = Signal(object)  # (request number, Translation or error str), from the translating thread
    _placed = Signal(str)  # what happened to Replace, from the pasting thread
    _key = Signal()  # the shortcut's listener queued an event, from the keyboard hook's thread

    def __init__(self, app, access=None, listener_factory=HotkeyListener):
        super().__init__()
        if access is None:
            from sst import textaccess as access
        self.app, self.access, self._listener_factory = app, access, listener_factory
        self.listener: HotkeyListener | None = None
        self.double_copy = False  # the shortcut is a double copy: the app copies, Rflow reads
        self.popup = TranslatePopup()
        self.popup.language.connect(self._pick)
        self.popup.copy.connect(self._copy)
        self.popup.replace.connect(self._replace)
        self.popup.retry.connect(self._retry)
        self.popup.setup.connect(self._setup)
        self.popup.closed.connect(self._closed)
        self.popup.keys_changed.connect(self._take_keys)
        self._held = True  # a mouse button was down at the last look: only a new click outside closes the popup
        self._asked = ("", "")  # (target, second) of the last translation, for Retry
        self._got.connect(self._show)
        self._done.connect(self._finish)
        self._placed.connect(self._after_replace)
        self._key.connect(self._pump)  # queued from the hook's thread: nothing runs here between keys
        # Only while the popup is open: it closes when the user goes to another app or clicks outside it.
        self.watch = QTimer(self)
        self.watch.setInterval(15)
        self.watch.timeout.connect(self._watch_popup)
        self.reading = False  # the copied text is being read
        self.request = 0  # each translation's number: an answer to an older one is dropped
        self.source: tuple[str, int] | None = None  # (the text, its window) the popup is for
        self.result: Translation | None = None

    def start(self, shortcut: str) -> None:
        """Listen for the shortcut ("" = Translate off)."""
        self.stop()
        if not shortcut:
            return
        try:
            hotkey = parse_hotkey(shortcut)
            self.listener = self._listener_factory(hotkey)
            self.listener.on_event = self._key.emit
            self.listener.start()
        except (ValueError, OSError) as e:
            log.warning("Translate's shortcut %r doesn't work: %s", shortcut, e)
            self.listener = None
            return
        self.double_copy = hotkey.double and hotkey.key is not None
        log.info("Translate on %s", shortcut)

    def stop(self) -> None:
        self.popup.close_popup()
        self.watch.stop()
        if self.listener is not None:
            self.listener.stop()
            self.listener = None

    @property
    def busy(self) -> bool:
        return self.reading

    def _pump(self) -> None:
        listener = self.listener
        if listener is not None:
            while not listener.events.empty():
                event, _ = listener.events.get_nowait()
                if event == "release":
                    self.trigger()
                elif event.startswith("key:"):
                    self.popup.key(int(event[4:]))  # Esc, and C (Copy) and Enter (Replace, Try again) when they're there

    def _watch_popup(self) -> None:
        if not self.popup.isVisible():
            self.watch.stop()
            return
        if self.source:
            front = self.access.foreground_window()
            if front not in (self.source[1], 0) and self.access.window_process(front) != os.getpid():
                self.popup.close_popup()  # the user went to another app: the popup was for the text left behind
                # (Rflow's own windows don't count)
            held = bool(getattr(self.access, "mouse_down", lambda: False)())
            if held and not self._held and not self.popup.contains_global(_pointer()):
                self.popup.close_popup()  # a click outside it, e.g. back in the text: like any popup, it goes
            self._held = held

    def trigger(self) -> None:
        if self.reading:
            return
        self.reading = True  # without an AI model the popup still opens, at the text, and says how to set one up
        hwnd, before = self.access.foreground_window(), self.access.clipboard_sequence()
        threading.Thread(target=self._read, args=(hwnd, before), name="translate-read", daemon=True).start()

    def _read(self, hwnd: int, before: int) -> None:
        """The text: what the app copied for the double Ctrl+C (read as soon as it lands, before any clipboard tool
        rewrites it), or the selection copied by Rflow for another shortcut."""
        try:
            if self.double_copy:
                end = time.monotonic() + COPY_WAIT
                while self.access.clipboard_sequence() == before and time.monotonic() < end:
                    time.sleep(0.005)
                text = self.access.clipboard_text()
            elif self.access.window_class(hwnd) in TERMINALS:
                text = None
                self._got.emit("In a terminal, select the text and press Ctrl+C twice.")
                return
            else:
                text = self.access.copy_selection()
        except Exception as e:  # a closed window, a busy clipboard: say so, never leave Translate stuck
            log.exception("Reading the text to translate failed")
            self._got.emit(f"Couldn't read the text ({e})")
            return
        self._got.emit((text, hwnd) if text and text.strip() else "Select some text first, then try again.")

    def _show(self, payload) -> None:
        self.reading = False
        if isinstance(payload, str):
            self.app.say("warning", payload)
            return
        text, hwnd = payload
        if len(text) > MAX_CHARS:
            self.app.say("warning", f"That's {len(text):,} characters: Translate takes up to {MAX_CHARS:,} at a time.")
            return
        self.source, self.result = (text, hwnd), None
        s, second = self.app.settings, self._second()
        language = detect(text)
        self._held = True
        self.popup.open_at(_pointer(), text, choose_target(text, s.translate_to, second),
                           self._choices(language, second), language)
        self.watch.start()
        self._take_keys()
        if not self.app.translate_ready():
            self.popup.show_setup()
            return
        self._translate(s.translate_to, second)

    def _second(self) -> str:
        """Where text already in the chosen language goes: the user's second language, else Windows' own language
        (or English), so English text is never "translated" into English."""
        s = self.app.settings
        return s.translate_second or fallback_second(s.translate_to, translate.system_language())

    def _choices(self, source_language: str, second: str) -> list[str]:
        """The popup's one-click languages: the chosen ones, Windows' language, English, then the rest; never the
        language the text is already in."""
        s = self.app.settings
        names = [s.translate_to, second, translate.system_language(), "English", *LANGUAGES]
        return [n for n in dict.fromkeys(names) if n in LANGUAGES and not (source_language
                                                                            and n.startswith(source_language))]

    def _translate(self, target: str, second: str) -> None:
        self.request += 1
        request, (text, _) = self.request, self.source
        self._asked = (target, second)
        self.result = None
        self.popup.busy(choose_target(text, target, second))

        def work() -> None:
            try:
                self._done.emit((request, self.app.run_translation(text, target, second)))
            except Exception as e:  # no answer, no key, a refusal: shown in the popup
                log.warning("Translate failed: %s", e)
                self._done.emit((request, str(e) or type(e).__name__))
        threading.Thread(target=work, name="translate", daemon=True).start()

    def _finish(self, payload) -> None:
        request, result = payload
        if request != self.request or not self.popup.isVisible():
            return  # a newer translation was asked for, or the popup was closed
        if isinstance(result, str):
            message = explain(result)
            self.popup.show_error(message, result, setup=message in (KEY_REFUSED, UNREACHABLE))
            return
        self.result = result
        self.popup.show_result(result, self.app.transform_model())

    def _retry(self) -> None:
        if self.source is not None:
            self._translate(*self._asked)

    def _setup(self) -> None:
        self.popup.close_popup()
        open_window = getattr(self.app, "open_window", None)
        if open_window is not None:
            open_window("cleanup")

    def _pick(self, language: str) -> None:
        """A language picked in the popup: translate into it (exactly that one), and remember it."""
        if self.source is None:
            return
        if language != self.app.settings.translate_to:
            self.app.apply_settings(dataclasses.replace(self.app.settings, translate_to=language))
        self._translate(language, "")

    def _copy(self) -> None:
        if self.result is None:
            return
        try:
            self.access.set_clipboard(self.result.text)
            self.popup.copied()  # it stays open: the button says so, and a click outside closes it
        except OSError as e:
            self.app.say("warning", f"Couldn't copy the translation ({e})")

    def _replace(self) -> None:
        if self.result is None or self.source is None:
            return
        text, hwnd = self.source
        translation = self.result.text
        self.popup.close_popup()
        self.reading = True  # the shortcut waits until the paste is done

        def work() -> None:
            try:
                if place(self.access, text, hwnd):
                    self.access.paste_rich(translation)
                    self._placed.emit("")
                else:  # never paste over something else
                    self.access.set_clipboard(translation)
                    self._placed.emit("Your text wasn't where it was any more, so the translation is on the clipboard: "
                                      "press Ctrl+V.")
            except Exception as e:
                log.exception("Pasting the translation failed")
                self._placed.emit(f"Couldn't type the translation ({e})")
        threading.Thread(target=work, name="translate-paste", daemon=True).start()

    def _after_replace(self, message: str) -> None:
        self.reading = False
        if message:
            self.app.say("warning", message)
        else:
            self.app.say("transformed", "Translated")

    def _take_keys(self) -> None:
        """While the popup is open, its keys are taken from the keyboard (the app keeps the focus and its selection)."""
        if self.listener is not None and self.popup.isVisible():
            self.listener.capture(frozenset(self.popup.keys()))

    def _closed(self) -> None:
        if self.listener is not None:
            self.listener.capture(None)
