"""The translation bar, and the Qt side of a live translation session.

The bar floats over every app, where the user puts it: drag it to move it, drag an edge to resize it (both are
remembered). Each line shows the words heard, small, above their translation; the whole session stays in it to scroll
back through, and it follows new lines until the user scrolls up. The ✕ stops live translation. It never takes the
keyboard from the app in front (WS_EX_NOACTIVATE: clicks move, resize and scroll it, typing stays where it was), and by
default it isn't in screen shares or recordings (SetWindowDisplayAffinity); a switch shows it, for colleagues to read.

With Both (an online meeting) the microphone's lines are the user's own: marked "You" in the accent colour, without the
words heard (the user knows what they said). With the microphone alone, words already in the language they'd be
translated into are shown as heard, marked "already English" in the small line. A quiet status line at the bottom says
why nothing shows yet (the session's notes: listening, nothing heard, sound but no speech). The speaker button turns the
spoken translation on and off, and the copy button puts the session so far on the clipboard (the app does it, from the
transcript: the bar never takes the keyboard, so its text can't be selected). The cost saver button sends audio to
Google only while someone talks (LiveConfig.save); while it's on, the title says what the session cost and saved, and a
session with no words for 15 minutes stops by itself, after a warning in the status line.
"""
import ctypes
import dataclasses
import html
import logging
import math
from collections.abc import Callable

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QPainter, QTextBlockFormat, QTextCursor
from PySide6.QtWidgets import QHBoxLayout, QLabel, QTextBrowser, QToolButton, QVBoxLayout, QWidget

from sst import theme
from sst.costs import money
from sst.live.contracts import AUTO_STOP_S, AUTO_STOP_WARN_S, MIC, SYSTEM, Kind, LiveConfig, LiveEvent, already_in, language_name

EDGE = 8  # px along the border where a drag resizes instead of moving
TICK_S = 5.0  # how often the cost is refreshed and a quiet saving session is checked
SAVE_TIP = "Cost saver: send audio only while someone talks"
MIN_SIZE, DEFAULT_SIZE = QSize(360, 140), QSize(760, 240)
KEEP_LINES = 2000  # finished lines kept to scroll back to (a long meeting); the transcript keeps them all
WDA_NONE, WDA_EXCLUDEFROMCAPTURE = 0x0, 0x11  # Windows 10 2004+: drawn on the screen, left out of screen capture

log = logging.getLogger(__name__)


def _no_focus(hwnd: int) -> None:
    """Never activated: the user's app keeps the keyboard, while clicks move, resize and scroll the bar. Live
    translation's own copy of what Rflow's popups do: the two pipelines share no code."""
    user32 = ctypes.windll.user32
    user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
    user32.GetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int]
    user32.SetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
    GWL_EXSTYLE, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW = -20, 0x08000000, 0x80
    user32.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE) | WS_EX_NOACTIVATE
                             | WS_EX_TOOLWINDOW)


def dragged(start: QRect, edges: tuple[bool, bool, bool, bool], by: QPoint, minimum: QSize) -> QRect:
    """`start` moved by `by` (no edges), or resized by its (left, top, right, bottom) edges, never below `minimum`."""
    left, top, right, bottom = edges
    if not any(edges):
        return start.translated(by)
    r = QRect(start)
    if left:
        r.setLeft(min(start.left() + by.x(), start.right() - minimum.width() + 1))
    if right:
        r.setRight(max(start.right() + by.x(), start.left() + minimum.width() - 1))
    if top:
        r.setTop(min(start.top() + by.y(), start.bottom() - minimum.height() + 1))
    if bottom:
        r.setBottom(max(start.bottom() + by.y(), start.top() + minimum.height() - 1))
    return r


_CURSORS = {(True, False): Qt.CursorShape.SizeHorCursor, (False, True): Qt.CursorShape.SizeVerCursor}


class CaptionBar(QWidget):
    closed = Signal()  # the ✕: stop live translation
    moved = Signal(object)  # [x, y, width, height] after a move or a resize, to remember
    speak_toggled = Signal(bool)  # the speaker button: the translation spoken aloud, or not
    copy_requested = Signal()  # the copy button: the session so far onto the clipboard
    save_toggled = Signal(bool)  # the cost saver button: audio sent only while someone talks, or all of it

    def __init__(self, config: LiveConfig | None = None):
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setMouseTracking(True)
        self.setMinimumSize(MIN_SIZE)
        self.resize(DEFAULT_SIZE)
        self.config = config or LiveConfig()
        self.hide_from_capture = self.config.hide_from_capture
        self.finished: list[tuple[str, str, str]] = []  # (way, words heard, translation) of every finished line
        self.current: dict[str, tuple[str, str]] = {}  # way -> (words heard, translation) of the line being spoken
        self.problems: dict[str, str] = {}  # way -> what went wrong, until words come again
        self.notes: dict[str, str] = {}  # way -> why it shows nothing yet (the status line), until the session clears it
        self.status = "Starting…"
        self.voice_note = ""  # the voice's state in the title while it isn't simply on or off (downloading, loading)
        self.flash_note = ""  # what a click just did ("Copied 42 lines"), in the title for a moment
        self.spend_note = ""  # what the session cost and saved ("$0.42 · saved $0.26"), while the cost saver is on
        self.warning = ""  # the cost saver's auto-stop warning, first in the status line
        self._flash_timer = QTimer(self, singleShot=True, interval=2500)
        self._flash_timer.timeout.connect(lambda: self.flash(""))
        self._drag: tuple[QPoint, QRect, tuple] | None = None  # (where the press was, the geometry then, the edges)
        self._tail_at = 0  # where the finished lines end in the document: what follows is redrawn as words come

        outer = QVBoxLayout(self)
        outer.setContentsMargins(EDGE + 12, EDGE + 4, EDGE + 4, EDGE + 6)
        outer.setSpacing(2)
        head = QHBoxLayout()
        self.title = QLabel()
        self.title.setFont(theme.font(12, 600))
        self.title.setStyleSheet(f"color: {theme.tok('text3', popup=True).name()}; background: transparent;")
        self.speak_button = self._head_button("Speak the translation")
        self.speak_button.setCheckable(True)
        self.speak_button.clicked.connect(lambda on: self.speak_toggled.emit(on))
        self.copy_button = self._head_button("Copy the translation so far (both languages)")
        self.copy_button.setIcon(QIcon(theme.icon_pixmap("copy", theme.tok("text2", popup=True).name(), 15, 2.0)))
        self.copy_button.clicked.connect(self.copy_requested.emit)
        self.save_button = self._head_button(SAVE_TIP)
        self.save_button.setCheckable(True)
        self.save_button.clicked.connect(lambda on: self.save_toggled.emit(on))
        self.close_button = self._head_button("Stop live translation")
        self.close_button.setIcon(QIcon(theme.icon_pixmap("close", theme.tok("text2", popup=True).name(), 14, 2.0)))
        self.close_button.clicked.connect(self.closed.emit)
        head.addWidget(self.title, 1)
        head.addWidget(self.save_button)
        head.addWidget(self.copy_button)
        head.addWidget(self.speak_button)
        head.addWidget(self.close_button)
        outer.addLayout(head)
        self.view = QTextBrowser()
        self.view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.view.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setStyleSheet(
            "QTextBrowser { background: transparent; border: none; }"
            "QScrollBar:vertical { width: 8px; background: transparent; margin: 2px 0; }"
            "QScrollBar::handle:vertical { background: rgba(255, 255, 255, 46); border-radius: 4px; min-height: 28px; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }"
            "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }")
        self.view.viewport().setAutoFillBackground(False)
        self.view.document().setDefaultFont(theme.font(17, 500))
        self.view.document().setDocumentMargin(2)
        self.view.viewport().installEventFilter(self)  # its presses move the bar too; the wheel still scrolls it
        outer.addWidget(self.view, 1)
        self.footer = QLabel()  # the status line: why nothing shows yet (the session's notes), hidden when all is well
        self.footer.setFont(theme.font(12, 400))
        self.footer.setStyleSheet(f"color: {theme.tok('text3', popup=True).name()}; background: transparent;")
        self.footer.setWordWrap(True)
        self.footer.setContentsMargins(2, 0, 0, 0)  # in line with the text above (the document's margin)
        self.footer.hide()
        outer.addWidget(self.footer)
        self.set_speaking(False)
        self.set_saving(self.config.save)
        self._update_title()
        self._redraw_tail()

    @staticmethod
    def _head_button(tip: str) -> QToolButton:
        b = QToolButton()
        b.setToolTip(tip)
        b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setStyleSheet("QToolButton { border: none; border-radius: 6px; padding: 4px; background: transparent; }"
                        "QToolButton:hover { background: rgba(255, 255, 255, 28); }")
        return b

    def set_speaking(self, on: bool, hint: str = "") -> None:
        """The speaker button shows whether the translation is spoken; `hint` says why nothing is (its tooltip)."""
        self.speak_button.setChecked(on)
        colour = theme.tok("primary" if on else "text2", popup=True).name()
        self.speak_button.setIcon(QIcon(theme.icon_pixmap("speaker" if on else "speaker-off", colour, 15, 2.0)))
        self.speak_button.setToolTip(hint or ("Stop speaking the translation" if on else "Speak the translation"))
        self.problems.pop("voice", None)

    def set_saving(self, on: bool) -> None:
        """The cost saver button shows whether it's on (the accent, like the speaker); off, the cost and the
        warning go."""
        self.save_button.setChecked(on)
        colour = theme.tok("primary" if on else "text2", popup=True).name()
        self.save_button.setIcon(QIcon(theme.icon_pixmap("cloud", colour, 15, 2.0)))
        if not on:
            self.spend_note, self.warning = "", ""
            self._update_footer()
        self._update_title()

    def set_spend(self, cost: float, saved: float) -> None:
        """What the session cost so far, and what the cost saver kept from being sent, in the title."""
        self.spend_note = f"{_dollars(cost)} · saved {_dollars(saved)}"
        self._update_title()

    def set_warning(self, text: str) -> None:
        if text != self.warning:
            self.warning = text
            self._update_footer()

    def set_voice_note(self, note: str) -> None:
        self.voice_note = note
        self._update_title()

    def flash(self, note: str) -> None:
        """Say in the title what a click just did, for a moment."""
        self.flash_note = note
        if note:
            self._flash_timer.start()
        self._update_title()
    # -- what's shown

    def show_event(self, event: LiveEvent) -> None:
        lane, (heard, said) = event.lane, self.current.get(event.lane, ("", ""))
        if event.kind is Kind.SOURCE:
            self.current[lane] = (event.text, said)
            self.problems.pop(lane, None)
        elif event.kind is Kind.TRANSLATION:
            self.current[lane] = (heard, event.text)
            self.problems.pop(lane, None)
        elif event.kind is Kind.LINE:
            self.current.pop(lane, None)
            target = self.config.for_lane(lane).target
            note = f"already {language_name(target)}" if lane == MIC and already_in(event, target) else ""
            self._add_finished(lane, event.source, event.text or event.source, note)  # in the target language: as heard
            return
        elif event.kind is Kind.STATUS:
            if lane == self.config.lanes[0]:  # the main way's: connecting, listening, reconnecting
                self.status = event.text
                self._update_title()
                if not self.finished and not self.current:
                    self._redraw_tail()
            return
        elif event.kind is Kind.ERROR:
            self.problems[lane] = event.text
        elif event.kind is Kind.NOTE:
            if event.text:
                self.notes[lane] = event.text
            else:
                self.notes.pop(lane, None)
            self._update_footer()
            return
        self._redraw_tail()

    def note(self) -> str:
        """The status line: the cost saver's warning, then each way's note once, the computer's first."""
        order = [SYSTEM, MIC] + [lane for lane in self.notes if lane not in (SYSTEM, MIC)]
        notes = [self.warning] + [self.notes[lane] for lane in order if self.notes.get(lane)]
        return "  ·  ".join(dict.fromkeys(n for n in notes if n))

    def _update_footer(self) -> None:
        text = self.note()
        self.footer.setText(text)
        self.footer.setVisible(bool(text))

    def set_config(self, config: LiveConfig) -> None:
        """The source changed while live translation runs: the title and the marks follow."""
        self.config = config
        self._update_title()
        self._redraw_tail()

    def drop_lane(self, lane: str) -> None:
        """A way stopped: its line in progress, its problem and its note go (its finished lines stay)."""
        self.current.pop(lane, None)
        self.problems.pop(lane, None)
        self.notes.pop(lane, None)
        self._update_footer()
        self._redraw_tail()

    def text(self) -> str:
        """Everything the bar shows, as plain text (for tests and checks)."""
        return self.view.toPlainText()

    def set_hidden_from_capture(self, hidden: bool) -> None:
        """Left out of screen shares, or shown in them: at once, while live translation runs."""
        self.hide_from_capture = hidden
        if self.isVisible() and QGuiApplication.platformName() != "offscreen":
            ctypes.windll.user32.SetWindowDisplayAffinity(ctypes.c_void_p(int(self.winId())),
                                                          WDA_EXCLUDEFROMCAPTURE if hidden else WDA_NONE)

    def _update_title(self) -> None:
        c = self.config
        parts = ["Live translation"]
        if c.source == "microphone":
            parts.append(f"microphone into {language_name(c.mic_target)}")
        else:
            parts.append(f"into {language_name(c.target)}")
            if c.source == "both":
                parts.append(f"you into {language_name(c.mic_target)}")
        if self.status not in ("Listening", ""):
            parts.append(self.status)
        if self.voice_note:
            parts.append(self.voice_note)
        if self.spend_note:
            parts.append(self.spend_note)
        if self.flash_note:
            parts.append(self.flash_note)
        self.title.setText("  ·  ".join(parts))

    def _add_finished(self, lane: str, heard: str, said: str, note: str = "") -> None:
        follow = self._at_bottom()
        self.finished.append((lane, heard, said))
        cursor = self._clear_tail()
        entry = self._entry(lane, heard, said, live=False, note=note)
        if entry:
            if self._tail_at:
                cursor.insertBlock()
            cursor.insertHtml(entry)
            cursor.setBlockFormat(self._spacing())  # after: inserting into the first block resets its format
            self._tail_at = cursor.position()
        if len(self.finished) > KEEP_LINES:  # the oldest line leaves the bar (it's in the transcript)
            self.finished.pop(0)
            first = QTextCursor(self.view.document())
            first.movePosition(QTextCursor.MoveOperation.NextBlock, QTextCursor.MoveMode.KeepAnchor)
            removed = first.selectionEnd() - first.selectionStart()
            first.removeSelectedText()
            self._tail_at = max(0, self._tail_at - removed)
        self._redraw_tail(follow)

    def _clear_tail(self) -> QTextCursor:
        cursor = QTextCursor(self.view.document())
        cursor.setPosition(self._tail_at)
        cursor.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        return cursor

    def _redraw_tail(self, follow: bool | None = None) -> None:
        """The lines being spoken, problems, or what the bar waits for: after the finished lines, redrawn as words
        come. Only this part changes, so a user reading back up the bar isn't moved."""
        follow = self._at_bottom() if follow is None else follow
        cursor = self._clear_tail()
        blocks = [self._entry(lane, *self.current[lane], live=True) for lane in (SYSTEM, MIC) if lane in self.current]
        warn = theme.tok("warn", popup=True).name()
        for lane, message in self.problems.items():
            who = "Your speech: " if lane == MIC and self.config.marks_mine else ""
            blocks.append(f'<span style="color:{warn}">{html.escape(who + message)}</span>')
        blocks = [b for b in blocks if b]
        if not blocks and not self.finished:
            muted = theme.tok("text3", popup=True).name()
            blocks = [f'<span style="color:{muted}">{html.escape(self._waiting())}</span>']
        for i, block in enumerate(blocks):
            if self._tail_at or i:
                cursor.insertBlock()
            cursor.insertHtml(block)
            cursor.setBlockFormat(self._spacing())
        if follow:
            self._to_bottom()
            QTimer.singleShot(0, self._to_bottom)  # again once the document's new size is laid out

    def _entry(self, lane: str, heard: str, said: str, live: bool, note: str = "") -> str:
        """A line: the words heard (small) above their translation; `note` instead of the words heard says why there's
        no translation ("already English": the words heard are shown as they are)."""
        you = lane == MIC and self.config.marks_mine
        parts = []
        small = note or (heard if heard and not you and heard != said else "")
        if small:
            muted = theme.tok("text3", popup=True).name()
            parts.append(f'<span style="color:{muted}; font-size:12px; font-weight:400">{html.escape(small)}</span>')
        if said and you:
            accent = theme.tok("primary", popup=True)
            colour = accent.name() if live else _mix(accent, theme.tok("base", popup=True), 0.75)
            parts.append(f'<span style="color:{accent.name()}; font-weight:600">You&nbsp;·&nbsp;</span>'
                         f'<span style="color:{colour}">{html.escape(said)}</span>')
        elif said:
            colour = theme.tok("text" if live else "text2", popup=True).name()
            parts.append(f'<span style="color:{colour}">{html.escape(said)}</span>')
        return "<br>".join(parts)

    @staticmethod
    def _spacing() -> QTextBlockFormat:
        block = QTextBlockFormat()
        block.setBottomMargin(10)
        return block

    def _waiting(self) -> str:
        if self.status == "Listening":
            into = language_name(self.config.mic_target if self.config.source == "microphone" else self.config.target)
            return f"Translations into {into} appear here when someone speaks."  # "Listening…": the status line
        return self.status

    def _at_bottom(self) -> bool:
        bar = self.view.verticalScrollBar()
        return bar.value() >= bar.maximum() - 4

    def _to_bottom(self) -> None:
        bar = self.view.verticalScrollBar()
        bar.setValue(bar.maximum())

    # -- where it is

    def place(self, geometry: list | None = None) -> None:
        """Where the user left it, if that's still on a screen; else at the bottom of the screen the pointer is on."""
        if geometry and len(geometry) == 4:
            rect = QRect(*geometry)
            rect.setSize(rect.size().expandedTo(MIN_SIZE))
            for screen in QGuiApplication.screens():
                seen = screen.availableGeometry().intersected(rect)
                if seen.width() >= 120 and seen.height() >= 60:  # enough of it to grab and drag back
                    self.setGeometry(rect)
                    return
        screen = QGuiApplication.screenAt(self.cursor().pos()) or QGuiApplication.primaryScreen()
        area = screen.availableGeometry()
        width = min(DEFAULT_SIZE.width(), int(area.width() * 0.8))
        self.setGeometry(area.left() + (area.width() - width) // 2, area.bottom() - DEFAULT_SIZE.height() - 24, width,
                         DEFAULT_SIZE.height())

    def _edges(self, pos: QPoint) -> tuple[bool, bool, bool, bool]:
        return (pos.x() < EDGE, pos.y() < EDGE, pos.x() >= self.width() - EDGE, pos.y() >= self.height() - EDGE)

    def _press(self, global_pos: QPoint, pos: QPoint) -> None:
        self._drag = (global_pos, QRect(self.geometry()), self._edges(pos))

    def _move_to(self, global_pos: QPoint, pos: QPoint) -> None:
        if self._drag is None:
            left, top, right, bottom = self._edges(pos)
            if (left and top) or (right and bottom):
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            elif (right and top) or (left and bottom):
                self.setCursor(Qt.CursorShape.SizeBDiagCursor)
            else:
                self.setCursor(_CURSORS.get((left or right, top or bottom), Qt.CursorShape.ArrowCursor))
            return
        start, geometry, edges = self._drag
        self.setGeometry(dragged(geometry, edges, global_pos - start, self.minimumSize()))

    def _release(self) -> None:
        if self._drag is not None:
            self._drag = None
            g = self.geometry()
            self.moved.emit([g.x(), g.y(), g.width(), g.height()])

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press(event.globalPosition().toPoint(), event.position().toPoint())

    def mouseMoveEvent(self, event):
        self._move_to(event.globalPosition().toPoint(), event.position().toPoint())

    def mouseReleaseEvent(self, event):
        self._release()

    def eventFilter(self, watched, event):
        if watched is self.view.viewport():
            kind = event.type()
            if kind == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                at = event.globalPosition().toPoint()
                self._press(at, self.mapFromGlobal(at))
                return True
            if kind == QEvent.Type.MouseMove and self._drag is not None:
                at = event.globalPosition().toPoint()
                self._move_to(at, self.mapFromGlobal(at))
                return True
            if kind == QEvent.Type.MouseButtonRelease:
                self._release()
                return True
        return super().eventFilter(watched, event)

    def showEvent(self, event):
        super().showEvent(event)
        if QGuiApplication.platformName() == "offscreen":  # tests: no real window
            return
        hwnd = int(self.winId())
        try:
            _no_focus(hwnd)
            if self.hide_from_capture:
                ctypes.windll.user32.SetWindowDisplayAffinity(ctypes.c_void_p(hwnd), WDA_EXCLUDEFROMCAPTURE)
        except Exception as e:  # never lose the translation over a window style
            log.warning("Translation bar window style: %s", e)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        base = QColor(theme.tok("base", popup=True))
        base.setAlpha(236)
        p.setPen(QColor(255, 255, 255, 34))
        p.setBrush(base)
        p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 14, 14)
        p.end()


class LiveCaptions(QObject):
    """Starts and stops a live translation session and shows it in the translation bar; thread-safe towards the
    engines (their events arrive on their own threads and are moved to Qt's). The app wires the parts:
    `make_session(config, on_event)` builds the LiveSession (with its transcript), `make_lane(lane, config)` gives a
    way's (capture, engine_factory), `make_speaker(config)` the Speaker that says the translation aloud (None if nothing
    can speak it)."""

    event = Signal(object)
    changed = Signal(bool)  # running or not
    moved = Signal(object)  # the bar's new [x, y, width, height], to remember
    speak_toggled = Signal(bool)  # the bar's speaker button
    copy_requested = Signal()  # the bar's copy button
    save_toggled = Signal(bool)  # the bar's cost saver button
    noted = Signal(str)  # the bar's status line changed (the Live page shows it too)
    quiet_stop = Signal(str)  # the cost saver stopped a session with no words (why: "no speech for 15 min")
    spent = Signal()  # the cost of a saving session was refreshed (the Live page shows it too)

    def __init__(self, make_session: Callable, make_lane: Callable, make_speaker: Callable | None = None):
        super().__init__()
        self._make_session, self._make_lane, self._make_speaker = make_session, make_lane, make_speaker
        self.event.connect(self._on_event)
        self.session = None
        self.config = LiveConfig()
        self.bar: CaptionBar | None = None
        self.last_problem = ""
        self.stopped_why = ""  # "no speech for 15 min": why the cost saver stopped the last session, until the next start
        self.last_spend = None  # the last saving session's Spend (cost, saved, seconds heard), until the next start
        self._ticks = QTimer(self, interval=int(TICK_S * 1000))
        self._ticks.timeout.connect(self.tick)

    @property
    def running(self) -> bool:
        return self.session is not None

    @property
    def note(self) -> str:
        """The bar's status line: why nothing shows yet ("" when all is well, or not running)."""
        return self.bar.note() if self.bar is not None else ""

    def start(self, config: LiveConfig, geometry: list | None = None) -> None:
        """Every way of config.source. Raises if none can start; a way that can't is said on the bar, and the others
        go on."""
        if self.session is not None:
            return
        self.config, self.last_problem, self.stopped_why, self.last_spend = config, "", "", None
        bar = CaptionBar(config)
        session = self._make_session(config, self.event.emit)
        failed = []
        for lane in config.lanes:
            try:
                session.add(lane, *self._make_lane(lane, config))
            except Exception as e:  # e.g. no microphone, or it's blocked in Windows' privacy settings
                failed.append((lane, e))
        if not session.lanes:
            bar.deleteLater()
            raise failed[0][1]
        self.session, self.bar = session, bar
        bar.closed.connect(self.stop)
        bar.moved.connect(self.moved.emit)
        bar.speak_toggled.connect(self.speak_toggled.emit)
        bar.copy_requested.connect(self.copy_requested.emit)
        bar.save_toggled.connect(self.save_toggled.emit)
        for lane, error in failed:
            self._couldnt_start(lane, error)
        self._update_speaker()
        bar.place(geometry)
        bar.show()
        self._ticks.start()
        self.tick()
        self.changed.emit(True)

    def set_source(self, source: str) -> str:
        """Listen to something else while live translation runs; "" or why a way couldn't start."""
        self.config = dataclasses.replace(self.config, source=source)
        if self.session is None:
            return ""
        problem = ""
        for lane in list(self.session.lanes):
            if lane not in self.config.lanes:
                self.session.remove(lane)
                self.bar.drop_lane(lane)
        self.session.config = self.config
        for lane in self.config.lanes:
            if lane not in self.session.lanes:
                try:
                    self.session.add(lane, *self._make_lane(lane, self.config))
                except Exception as e:
                    problem = self._couldnt_start(lane, e)
        if self.session.transcript is not None:
            self.session.transcript.config = self.config
        self.bar.set_config(self.config)
        if not self.session.lanes:
            self.stop()
            return problem
        self._update_speaker()
        return problem

    def set_speak(self, on: bool) -> None:
        """The translation spoken aloud, or not: at once if live translation runs, and for the next start."""
        self.config = dataclasses.replace(self.config, speak=on)
        self._update_speaker()

    def set_save(self, on: bool) -> None:
        """The cost saver on or off: at once if live translation runs (every way's engine), and for the next start."""
        self.config = dataclasses.replace(self.config, save=on)
        if self.session is None:
            return
        self.session.config = self.config
        self.session.set_saving(on)
        self.bar.set_saving(on)
        self.tick()
        if not on:
            self.noted.emit(self.note)  # a warning went with it

    def spend(self):
        """The running session's Spend (cost, saved, seconds heard, seconds not sent), or None; never raises."""
        if self.session is None:
            return None
        try:
            return self.session.spend()
        except Exception as e:  # a cost figure never stops the translation
            log.warning("Live translation's cost: %s", e)
            return None

    def tick(self) -> None:
        """Every TICK_S while it runs: with the cost saver on, what the session cost and saved on the bar, and a session
        with no words warned at AUTO_STOP_S - AUTO_STOP_WARN_S, then stopped at AUTO_STOP_S."""
        if self.session is None or self.bar is None or not self.config.save:
            return
        spend = self.spend()
        if spend is not None:
            self.bar.set_spend(spend.cost, spend.saved)
            self.spent.emit()
        quiet = self.session.quiet_for()
        if quiet >= AUTO_STOP_S:
            log.info("Live translation: no speech for %.0f min, the cost saver stops it", quiet / 60)
            self.stopped_why = f"no speech for {round(AUTO_STOP_S / 60)} min"
            self.stop()
            self.quiet_stop.emit(self.stopped_why)
            return
        warning = ""
        if quiet >= AUTO_STOP_S - AUTO_STOP_WARN_S:
            warning = (f"No speech for {int(quiet // 60)} min: stops in "
                       f"{max(1, math.ceil((AUTO_STOP_S - quiet) / 60))} min")
        if warning != self.bar.warning:
            self.bar.set_warning(warning)
            self.noted.emit(self.note)

    def set_speak_speed(self, speed: float) -> None:
        self.config = dataclasses.replace(self.config, speak_speed=speed)
        if self.session is not None and self.session.speaker is not None:
            self.session.speaker.set_speed(speed)

    def set_duck(self, depth: float) -> None:
        """How loud the other apps stay while live translation speaks: at once if it runs, and for the next start."""
        self.config = dataclasses.replace(self.config, duck=depth)
        ducker = getattr(self.session.speaker, "ducker", None) if self.session is not None else None
        if ducker is not None:
            ducker.set_depth(depth)

    @property
    def transcript_path(self):
        """The running session's transcript file, or None (not running, or nobody has spoken yet)."""
        transcript = getattr(self.session, "transcript", None)
        return transcript.path if transcript is not None else None

    def flash(self, note: str) -> None:
        if self.bar is not None:
            self.bar.flash(note)

    def set_voice_note(self, note: str) -> None:
        """The voice's state in the bar's title (downloading, loading), or "" when it's simply on or off."""
        if self.bar is not None:
            self.bar.set_voice_note(note)

    def _update_speaker(self) -> None:
        """The voice while speaking is on: started, told which ways it speaks now; stopped when speaking is off."""
        session = self.session
        if session is None:
            return
        speaker = session.speaker
        if self.config.speak and speaker is None and self._make_speaker is not None:
            speaker = self._make_speaker(self.config)  # None if nothing can speak it
            if speaker is not None:
                session.set_speaker(speaker)
        elif not self.config.speak and speaker is not None:
            session.set_speaker(None)
            speaker = None
        hint = ""
        if speaker is not None:
            # Gemini speaks every way spoken itself, in any language; the laptop's voice stands in only in its own
            lanes, own = self.config.spoken_lanes(speaker.language), self.config.spoken_lanes()
            speaker.set_lanes(lanes, own)
            speaker.set_speed(self.config.speak_speed)
            if not lanes:
                hint = (f"Stop speaking the translation (Gemini's voice; the voice on this laptop speaks only "
                        f"{language_name(speaker.language)}, so it can't stand in for it here)")
        if self.bar is not None:
            self.bar.set_speaking(speaker is not None, hint)

    def _couldnt_start(self, lane: str, error: Exception) -> str:
        log.warning("Live translation: the %s didn't start: %s", "microphone" if lane == MIC else "computer's sound",
                    error)
        message = (f"The microphone couldn't be opened ({error})." if lane == MIC
                   else f"The computer's sound couldn't be captured ({error}).")
        self.event.emit(LiveEvent(Kind.ERROR, message, lane=lane))
        return message

    def set_hidden(self, hidden: bool) -> None:
        """Left out of screen shares or shown in them: at once if live translation runs, and for the next start."""
        self.config = dataclasses.replace(self.config, hide_from_capture=hidden)
        if self.bar is not None:
            self.bar.set_hidden_from_capture(hidden)

    def stop(self) -> None:
        self._ticks.stop()
        if self.session is not None and self.config.save:
            self.last_spend = self.spend()  # before the engines go: what this session cost and saved, for the Live page
        session, self.session = self.session, None
        if session is not None:
            session.stop()
        if self.bar is not None:
            self.bar.close()
            self.bar.deleteLater()
            self.bar = None
        if session is not None:
            self.changed.emit(False)

    def _on_event(self, event: LiveEvent) -> None:
        if event.kind is Kind.ERROR:
            mine = event.lane == MIC and self.config.marks_mine
            self.last_problem = ("Your speech: " if mine else "") + event.text
        if self.bar is not None:
            self.bar.show_event(event)
            if event.kind is Kind.NOTE:
                self.noted.emit(self.note)
        if event.kind is Kind.STATUS and event.text == "Stopped" and self.session is not None \
                and event.lane in self.session.lanes:  # an engine gave up (a refused key, ...), not one we stopped
            if len(self.session.lanes) > 1:
                self.session.remove(event.lane)  # the other way goes on; this one's reason stays on the bar
            else:
                self.stop()  # nothing left: the bar goes; the reason stays in last_problem


def _dollars(amount: float) -> str:
    """$0.42; a little that isn't nothing: "<$0.01" (as the cost card says it)."""
    return "$0.00" if amount <= 0 else "<$0.01" if amount < 0.005 else money(amount)


def _mix(colour: QColor, other: QColor, amount: float) -> str:
    """`amount` of `colour` over `other`, as #rrggbb: a dimmer accent for the user's finished lines."""
    return QColor(*(round(a * amount + b * (1 - amount)) for a, b in (
        (colour.red(), other.red()), (colour.green(), other.green()), (colour.blue(), other.blue())))).name()
