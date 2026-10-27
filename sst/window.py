"""The Rflow window: a small app like Wispr Flow's, next to the tray icon, in the "Obsidian Signal" design (Rflow UI 2.0;
sst.theme draws it, sst.ui has its widgets, docs/design/rflow-ui.html is the design).

Ten sections in the sidebar (phase 31: each explains itself at a glance; in the narrow rail, short names under icons):
  Home              the voice orb and how to dictate, a stats strip, the dictations (searchable, copy and correct)
  Live translation  speech translated while people speak, in a bar of its own
  Words             Your words: the dictionary, sound-alikes, suggestions
  Snippets          a short phrase said, your own text typed
  Text Transform    its switch, how to use it, examples of each transform, a box to try them, the phrases and the menu
  Translate         its switch, how to use it, examples, the shortcut and the languages, a box to try it
  Formatting        "Write numbers as numbers", with what the real formatting stage types, and a box to try it
  AI & models       how Rflow hears you and the AI connection; Speech models and AI connection one level down
  Settings          the dictation key, the everyday switches and the Reading test; Advanced has the voice pipeline and
                    Profiles; Start over at the end
  Report a problem  a GitHub issue, the version information to paste into it, the logs
A first-run welcome in three steps sets up how Rflow hears you, a first dictation and an AI connection. It is native
Qt, following Windows' light or dark mode (an embedded browser would add ~150 MB for the same look).

The window keeps no state of its own. It reads and changes everything through `app`: the TrayApp (sst/app.py), which
applies a change at once (hotkey, microphone, cleanup model...), or PreviewApp below for the self-test, the tests and
the website's screenshots.
"""
import dataclasses
import html
import logging
import math
import os
import platform
import re
import sys
import threading
import time
from collections.abc import Callable, Mapping
from datetime import date, datetime, timedelta
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRect, QRectF, QSize, QSortFilterProxyModel, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QFontMetrics, QGuiApplication, QIcon, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QCompleter,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLayout,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionComboBox,
    QStyleOptionViewItem,
    QStylePainter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
    QWidgetItem,
)

from sst import RECORDINGS_DIR, __version__, bench, costs, setups, shell, theme, usage
from sst.audio import LevelMeter, Take, call_quality, save_wav
from sst.commands import DEFAULT_PHRASES, UNDO, normalize, parse_phrases, phrases_for
from sst.engines import DEFAULT_MODEL, SPEECH_MODELS, usable
from sst.engines.cloud import CLOUD, SPEECH
from sst.engines.whisper import LANGUAGES
from sst.gateway import PROVIDERS, SPEECH_SERVER, GatewayConfig, Polisher, offered
from sst.hotkey import parse_hotkey
from sst.live.contracts import LANGUAGES as LIVE_LANGUAGES
from sst.live.contracts import LiveConfig, language_name
from sst.live.voice import LOCAL as LOCAL_VOICE
from sst.meeting.apps import IN_PERSON
from sst.pipeline.contracts import VoiceConfig
from sst.pipeline.dictionary import speech_hints
from sst.pipeline.formatting import Formatter
from sst.scan import MEMORY_GB, SYSTEM_GB, Computer, machine
from sst.settings import (
    Profiles,
    Settings,
    Stats,
    can_start_with_windows,
    set_start_with_windows,
    starts_with_windows,
)
from sst.snippets import MAX_CUE_WORDS as SNIPPET_WORDS
from sst.snippets import Snippet
from sst.snippets import alone as snippet_alone
from sst.snippets import compact as compact_cue
from sst.snippets import expand as expand_snippets
from sst.snippets import load as load_snippets
from sst.snippets import protect as protect_snippets
from sst.theme import font, surface, tok
from sst.transform import GROUPS, TRANSFORMS, menu_items
from sst.translate import LANGUAGES as TRANSLATE_LANGUAGES
from sst.translate import fallback_second, system_language
from sst.ui import (
    BlinkLamp,
    Button,
    Card,
    GlowCard,
    Host,
    IconButton,
    Lamp,
    Mark,
    Meter,
    Orb,
    PowerButton,
    Segmented,
    Slider,
    Toast,
    Toggle,
    WaveProgress,
    button,
    card,
    divider,
    icon_button,
    keycap,
    keys,
    label,
    lamp_row,
    set_tone,
    setting_row,
)
from sst.wakeui import WakeState

APP_NAME = "Rflow"
ICON_FILE = Path(__file__).parent / "static" / "sst.ico"
UI_IMAGES = Path(__file__).parent / "static" / "ui"  # drawn by scripts/make_ui_images.py
LOG_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "sst" / "logs"
WEBSITE = "https://rflow-ai.vercel.app"
REPO = "https://github.com/karthi-ai-engineer/rflow-ai"
HOTKEY_CHOICES = [("Ctrl+Win", "ctrl+win"), ("Menu key", "menu"), ("Ctrl+Alt+D", "ctrl+alt+d")]
# Text Transform's menu shortcut: not Ctrl+Win+... (that starts a dictation) and not a plain Ctrl+letter (apps use those).
# A double tap of Ctrl is the easiest; PowerToys' "Find My Mouse" uses a double Ctrl too (Rflow's still works with it).
# Translate's shortcut: a double copy (the app copies; Rflow reads it), or a shortcut after which Rflow copies.
LIVE_SOURCES = [("computer", "Computer"), ("microphone", "Microphone"), ("both", "Both")]
LIVE_SOURCE_WORDS = {
    "computer": "What your laptop plays: a Teams or Zoom meeting, a video.",
    "microphone": "What your microphone hears: people talking in the room. Speech already in the language you choose "
                  "isn't shown.",
    "both": "An online meeting: the others from your laptop, and your own words from the microphone, marked “You”. "
            "Use headphones, or the microphone hears the meeting too.",
}
LIVE_SPEEDS = [("Normal", 1.0), ("A little faster", 1.15), ("Faster", 1.3)]
LIVE_SHORTCUTS = [("Ctrl+Alt+L", "ctrl+alt+l"), ("Ctrl+Shift+L", "ctrl+shift+l"), ("Win+Alt+L", "win+alt+l"), ("Off", "")]
MEETING_SHORTCUTS = [("Ctrl+Alt+M", "ctrl+alt+m"), ("Ctrl+Shift+M", "ctrl+shift+m"), ("Win+Alt+M", "win+alt+m"),
                     ("Off", "")]
# Meeting mode's parts (Settings.meeting_<part>): its title and what it does, one line each
MEETING_PARTS = [
    ("translate", "Live translation", "The translation bar runs during the meeting, as set on Live translation."),
    ("record", "Record", "The others (what the laptop plays) and you (the microphone), saved as sound files on this laptop."),
    ("minutes", "Minutes", "After the meeting: who said what, a summary, the decisions, and the action items with owners "
                           "and dates. Needs Record."),
    ("screen", "Screen", "The meeting window seen every 5 seconds; a slide is kept when the screen changes, and given to "
                         "the minutes."),
]
# The In person tab's parts (Settings.room_<part>): one microphone hears the room, so no screen and no "you" track
ROOM_PARTS = [
    ("translate", "Live transcript", "What the room says, live on the bar: translated into your language, or as said "
                                     "when it already is."),
    ("record", "Record", "The room as your microphone hears it, saved as a sound file on this laptop."),
    ("minutes", "Minutes", "After the meeting: who said what, a summary, the decisions, and the action items with owners "
                           "and dates. Needs Record."),
]
MEETING_TABS = [("online", "Online"), ("room", "In person")]
MEETING_SUBTITLES = {
    "online": "Your meeting translated, recorded and written up: the minutes, decisions and action items, from Teams, "
              "Zoom, Google Meet or Webex.",
    "room": "A meeting in a room, around a table: your microphone hears everyone. A live transcript, the recording and "
            "the minutes, with each voice told apart.",
}
MEETING_REFRESH_MS = 5000  # the meeting's line (time, cost, slides) and the meetings found, while the page is shown
TRANSLATE_SHORTCUTS = [("Ctrl+C+C (press Ctrl+C twice)", "ctrl+c+c"), ("Ctrl+Alt+L", "ctrl+alt+l"), ("Off", "")]
TRANSFORM_HOTKEYS = [("Double-tap Ctrl", "double ctrl"), ("Ctrl+Alt+T", "ctrl+alt+t"), ("F8", "f8"), ("Off", "")]
WHERE_LABELS = {"local": "On this PC", "cloud": "In the cloud", "server": "Your own server"}
# The top bar's look: as Windows is set, or always light or dark (Settings.appearance)
APPEARANCE_CHOICES = [("system", "System"), ("light", "Light"), ("dark", "Dark")]
APPEARANCE_ICONS = {"system": "display", "light": "sun", "dark": "moon"}
APPEARANCE_TIPS = {"system": "Light or dark, as Windows is set", "light": "Always light (Porcelain)",
                   "dark": "Always dark (Obsidian)"}
COMPACT_WIDTH = 900  # below it, the sidebar becomes a rail of icons with their names under them

log = logging.getLogger("sst.window")


def dark_mode() -> bool:
    return theme.dark_mode()


def stylesheet(name: str) -> str:
    """The style sheet of a theme ("dark" Obsidian, "light" Porcelain)."""
    theme.load_fonts()
    return theme.stylesheet(name)


def text(value: str = "", name: str | None = None, muted: bool = False, wrap: bool = True) -> QLabel:
    """A label by its old names (h1, h2, section, warning, ok, stat): kept so the pages read as before."""
    roles = {"h1": ("title", None), "h2": ("heading", None), "section": ("caps", None), "warning": (None, "warn"),
             "ok": (None, "ok"), "stat": ("number", None), "sentence": ("hero", None), "brand": ("wordmark", None)}
    role, tone = roles.get(name or "", (None, None))
    widget = label(value, role, tone or ("2" if muted else None), wrap)
    if name and name not in roles:
        widget.setObjectName(name)
    return widget


def caption(value: str = "", tone: str | None = "2", wrap: bool = True) -> QLabel:
    return label(value, "caption", tone, wrap)


def fixed_height(widget: QWidget, height: int) -> None:
    """One height, and no asking the card for more: a text box's size policy grows, so a fixed height alone still made
    its card taller, and the spare height went to the card's heading (a large gap above "Try it" and "Add a snippet")."""
    widget.setFixedHeight(height)
    widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)


def row(*items, stretch_at: int | None = None, spacing: int = 8) -> QHBoxLayout:
    layout = QHBoxLayout()
    layout.setSpacing(spacing)
    for i, item in enumerate(items):
        if i == stretch_at:
            layout.addStretch()
        if isinstance(item, QHBoxLayout | QVBoxLayout):
            layout.addLayout(item)
        else:
            layout.addWidget(item)
    if stretch_at is not None and stretch_at >= len(items):
        layout.addStretch()
    return layout


def clear(layout: QLayout) -> None:
    """Empty a layout that is filled again (history, words)."""
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget:
            widget.hide()  # at once: deleteLater waits for the event loop, and until then it would still be drawn
            widget.deleteLater()
        elif item.layout():
            clear(item.layout())


def field(widget: QWidget, kind: str = "field") -> QWidget:
    """A text box or list on a well with an edge (focus: a 2 px Iris edge), painted by its card."""
    return surface(widget, kind)


def bare(edit: QLineEdit) -> QLineEdit:
    """A line edit inside a field of its own (a search box with an icon): no padding of its own."""
    edit.setProperty("bare", True)
    return edit


def key_names(label_text: str) -> str:
    """A hotkey's label as keys: "Ctrl+Win" stays, "Menu key" is the Menu key."""
    return "+".join(part.strip().removesuffix(" key") for part in label_text.split("+") if part.strip())


def back_button(title: str, go_to, page: str) -> Button:
    return button(title, lambda: go_to(page), link=True, size="sm", icon="chevron-left")


# ---------------------------------------------------------------- form controls
# A settings page must not change by accident, long lists must be searchable, and what is saved must be visible (the
# owner, 2026-10-04: scrolling over a dropdown changed the model and the language, and nothing said what was saved).

class Choice(QComboBox):
    """A dropdown the mouse wheel never changes: over a closed list the wheel scrolls the page. `search` gives a long
    list a search box at its top; an `editable` one filters its list by what is typed (any other name still goes)."""

    def __init__(self, search: bool = False, editable: bool = False, small: bool = False):
        super().__init__()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)  # the wheel doesn't take the focus either
        self._search = search
        self._popup: _SearchPopup | None = None
        if small:
            self.setProperty("size", "sm")
        if editable:
            self.setEditable(True)
            self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
            completer = QCompleter(self.model(), self)
            completer.setFilterMode(Qt.MatchFlag.MatchContains)
            completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
            self.setCompleter(completer)
            bare(self.lineEdit())
        surface(self, "field", 10 if small else 14)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def wheelEvent(self, event) -> None:
        event.ignore()  # to the page, which scrolls

    def showPopup(self) -> None:
        if not self._search:
            super().showPopup()
            return
        if self._popup is None:
            self._popup = _SearchPopup(self)
        self._popup.open()


class _SearchPopup(QFrame):
    """A long dropdown's list with a search box on top: type a few letters, then Enter (or a click) picks."""

    def __init__(self, combo: QComboBox):
        super().__init__(combo, Qt.WindowType.Popup)
        self.setObjectName("searchPopup")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)  # else a window of its own isn't painted
        theme.make_host(self)
        self.combo = combo
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search...")
        self.search.setClearButtonEnabled(True)
        field(self.search)
        self.proxy = QSortFilterProxyModel(self)
        self.proxy.setSourceModel(combo.model())
        self.proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.view = QListView()
        self.view.setModel(self.proxy)
        self.view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.empty = caption("Nothing matches", "3")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        for widget in (self.search, self.view, self.empty):
            layout.addWidget(widget)
        self.search.textChanged.connect(self._filter)
        self.search.installEventFilter(self)  # arrows and Enter work while typing
        self.view.clicked.connect(self.pick)
        self.view.activated.connect(self.pick)

    def paintEvent(self, event):
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        theme.paint_hosted(self, p)
        p.end()

    def open(self) -> None:
        self.search.clear()
        self._filter("")
        width, height = max(self.combo.width(), 260), 320
        below = self.combo.mapToGlobal(QPoint(0, self.combo.height() + 4))
        screen = self.combo.screen().availableGeometry()
        y = below.y() if below.y() + height <= screen.bottom() else self.combo.mapToGlobal(QPoint(0, 0)).y() - height - 4
        self.setGeometry(below.x(), y, width, height)
        self.show()
        self.search.setFocus()

    def _filter(self, value: str) -> None:
        self.proxy.setFilterFixedString(value.strip())
        if value.strip():
            current = self.proxy.index(0, 0)  # Enter picks the best match
        else:
            current = self.proxy.mapFromSource(self.combo.model().index(self.combo.currentIndex(), 0))
        self.view.setCurrentIndex(current)
        self.view.scrollTo(current, QAbstractItemView.ScrollHint.PositionAtCenter)
        self.empty.setVisible(not self.proxy.rowCount())

    def pick(self, index=None) -> None:
        index = index if index is not None and index.isValid() else self.view.currentIndex()
        if index.isValid():
            self.combo.setCurrentIndex(self.proxy.mapToSource(index).row())
        self.hide()

    def eventFilter(self, obj, event) -> bool:
        if obj is self.search and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            if key in (Qt.Key.Key_Down, Qt.Key.Key_Up, Qt.Key.Key_PageDown, Qt.Key.Key_PageUp):
                QGuiApplication.sendEvent(self.view, event)
                return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.pick()
                return True
            if key == Qt.Key.Key_Escape:
                self.hide()
                return True
        return super().eventFilter(obj, event)


def _clipboard_text() -> str:
    return QGuiApplication.clipboard().text()  # apart, so the tests never touch the real clipboard


def _masked(key: str) -> str:
    return "•" * 8 + (f"  {key[-4:]}" if len(key) >= 12 else "") if key else ""  # a short key shows nothing of it


class KeyField(QWidget):
    """An API key. Saved, it shows as dots and its last four characters, with a pen to change it; being changed, it's
    a hidden field with Show, Paste and a cross back to the saved key. It's saved with the rest of its section, by
    that section's Save. text() is the key as it would be saved; setText() types one."""

    textChanged = Signal(str)

    def __init__(self, saved: str = "", placeholder: str = ""):
        super().__init__()
        self.saved = saved
        surface(self, "field")
        self.view = bare(QLineEdit())
        self.view.setReadOnly(True)
        self.view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.view.setToolTip("Saved, encrypted on this computer")
        self.view.setFont(font(13, 500, mono=True, spacing=0.06))
        self.field = bare(QLineEdit())
        self.field.setEchoMode(QLineEdit.EchoMode.Password)
        self.field.setPlaceholderText(placeholder)
        self.edit = icon_button("edit", "Change the key", lambda _=False: self.start_editing(), 28, flat=True)
        self.reveal = icon_button("eye", "Show the key", lambda _=False: self._toggle_shown(), 28, flat=True)
        self.paste = icon_button("paste", "Paste a key", lambda _=False: self.setText(_clipboard_text().strip()), 28,
                                 flat=True)
        self.undo = icon_button("close", "Keep the saved key", lambda _=False: self.show_saved(), 28, flat=True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 6, 0)
        layout.setSpacing(2)
        for widget in (self.view, self.edit, self.field, self.reveal, self.paste, self.undo):
            layout.addWidget(widget)
        layout.setStretch(0, 1)
        layout.setStretch(2, 1)
        self.setFixedHeight(40)
        self.field.textChanged.connect(lambda _="": self.textChanged.emit(self.text()))
        self.show_saved(saved)

    @property
    def editing(self) -> bool:
        return not self.field.isHidden()

    def text(self) -> str:
        return self.field.text() if self.editing else self.saved

    def setText(self, value: str) -> None:
        self.start_editing()
        self.field.setText(value)

    def setPlaceholderText(self, value: str) -> None:
        self.field.setPlaceholderText(value)

    def edited(self) -> bool:
        return self.text().strip() != self.saved

    def show_saved(self, saved: str | None = None) -> None:
        """Back to the saved key, or a new one saved meanwhile (e.g. on the other page); with none, the box to type it."""
        if saved is not None:
            self.saved = saved
        self.view.setText(_masked(self.saved))
        self._set_editing(not self.saved)
        self.field.blockSignals(True)
        self.field.setText(self.saved)
        self.field.blockSignals(False)
        self.textChanged.emit(self.text())

    def start_editing(self) -> None:
        if self.editing:
            return
        self._set_editing(True)
        self.field.setFocus()
        self.field.selectAll()  # typing or pasting replaces the saved key

    def _set_editing(self, editing: bool) -> None:
        for widget in (self.view, self.edit):
            widget.setVisible(not editing)
        for widget in (self.field, self.reveal, self.paste):
            widget.setVisible(editing)
        self.undo.setVisible(editing and bool(self.saved))
        self.field.setEchoMode(QLineEdit.EchoMode.Password)
        self.reveal.set_icon("eye")
        self.reveal.setToolTip("Show the key")

    def _toggle_shown(self) -> None:
        shown = self.field.echoMode() == QLineEdit.EchoMode.Password
        self.field.setEchoMode(QLineEdit.EchoMode.Normal if shown else QLineEdit.EchoMode.Password)
        self.reveal.set_icon("eye-off" if shown else "eye")
        self.reveal.setToolTip("Hide the key" if shown else "Show the key")


class SaveBar(QWidget):
    """A section's Save, Cancel while there are changes, and its state in words: unsaved changes, or saved."""

    def __init__(self, on_save, on_cancel, label_text: str = "Save"):
        super().__init__()
        self.save = button(label_text, lambda _=False: on_save(), primary=True, size="sm")
        self.cancel = button("Cancel", lambda _=False: on_cancel(), kind="quiet", size="sm")
        self.cancel.hide()
        self.mark = QLabel()
        self.mark.hide()
        self.state = caption("", None, wrap=True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self.save)
        layout.addWidget(self.cancel)
        layout.addSpacing(4)
        layout.addWidget(self.mark)
        layout.addWidget(self.state, 1)

    def show_state(self, edited: bool, enabled: bool | None = None) -> None:
        """After every change. Save is on when there is something to save (or `enabled` says otherwise)."""
        self.save.setEnabled(edited if enabled is None else enabled)
        self.cancel.setVisible(edited)
        if edited:
            self._say("Unsaved changes", "warn")
        elif self.state.property("tone") == "warn":
            self._say("", None)

    def saved(self, message: str) -> None:
        self.cancel.hide()
        self._say(message, "ok")

    def _say(self, message: str, tone: str | None) -> None:
        set_tone(self.state, tone)
        self.state.setText(message)
        self.mark.setVisible(tone == "ok" and bool(message))
        if tone == "ok":
            self.mark.setPixmap(theme.icon_pixmap("check", tok("ok").name(), 14, self.devicePixelRatioF()))


# ---------------------------------------------------------------- what a model costs (sst.costs), next to it

COST_TONES = {costs.AMBER: "warn", costs.RED: "err"}  # the colour of a mid-priced and of an expensive model


class _CostDelegate(QStyledItemDelegate):
    """A model list's lines with each model's estimate on the right ("≈ $0.42/mo"), amber or red for the dear ones."""

    def __init__(self, estimate, parent=None):
        super().__init__(parent)
        self.estimate = estimate  # model id -> costs.Estimate

    def _words(self, index) -> tuple[str, str]:
        model = str(index.data() or "").strip()
        if not model:
            return "", ""
        estimate = self.estimate(model)
        return estimate.short, COST_TONES.get(estimate.tier, "text3")

    def paint(self, painter, option, index):
        words, colour = self._words(index)
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        room = opt.fontMetrics.horizontalAdvance(words) + 20 if words else 0
        opt.text = opt.fontMetrics.elidedText(opt.text, Qt.TextElideMode.ElideRight, max(40, opt.rect.width() - room - 16))
        widget = opt.widget
        (widget.style() if widget else QApplication.style()).drawControl(
            QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget)
        if words:
            painter.save()
            painter.setPen(tok(colour))
            painter.setFont(font(12, 500))
            painter.drawText(opt.rect.adjusted(0, 0, -12, 0), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                             words)
            painter.restore()

    def sizeHint(self, option, index):
        hint = super().sizeHint(option, index)
        words, _ = self._words(index)
        return QSize(hint.width() + (QFontMetrics(font(12, 500)).horizontalAdvance(words) + 28 if words else 0),
                     hint.height())


def cost_list(box: QComboBox, estimate) -> QComboBox:
    """Show each model's estimate in a model list (and in what an editable one finds while typing)."""
    box.setItemDelegate(_CostDelegate(estimate, box))
    box.view().setMinimumWidth(340)  # the model's name and its estimate side by side
    if box.completer() is not None:
        box.completer().popup().setItemDelegate(_CostDelegate(estimate, box))
    return box


class CostLine(QWidget):
    """Under a model: what it costs a month for typical use; amber or red with a lamp and why, for the dear ones."""

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.lamp = Lamp("warn")
        self.words = caption("", "3")
        lamp = QVBoxLayout()  # the lamp on the first line of the words
        lamp.setContentsMargins(0, 3, 0, 0)
        lamp.addWidget(self.lamp)
        lamp.addStretch()
        layout.addLayout(lamp)
        layout.addWidget(self.words, 1, Qt.AlignmentFlag.AlignTop)
        self.estimate: costs.Estimate | None = None

    def show_estimate(self, estimate: costs.Estimate | None) -> None:
        """None: no model chosen (the line goes)."""
        self.estimate = estimate
        self.setVisible(estimate is not None)
        if estimate is None:
            return
        tone = COST_TONES.get(estimate.tier)
        if estimate.warning:
            words = estimate.warning
        elif estimate.tier == costs.FREE:
            words = "Free: it runs on this PC."
        elif estimate.monthly is None:
            words = "Price unknown: Rflow has no price for this model."
        else:
            words = f"{estimate.words} for {costs.TYPICAL_WORDS}."
        self.words.setText(words)
        set_tone(self.words, tone or "3")
        self.lamp.setVisible(bool(tone))
        if tone:
            self.lamp.set_state(tone)
        self.lamp.setToolTip(f"Prices seen on the providers' pages on {costs.SEEN}.")


def expensive(model: str, estimate: costs.Estimate) -> str:
    """What a question says before an expensive (red) model is saved: "" for the others."""
    return f"{model} is an expensive model. {estimate.warning}" if estimate.tier == costs.RED else ""


def open_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    shell.open_url(QUrl.fromLocalFile(str(path)))


class _Relay(QObject):
    done = Signal(object, object)  # (result, error)


def run_in_background(owner: QObject, work, done) -> None:
    """work() on a thread (network calls), then done(result, error) back on the UI thread."""
    relay = _Relay(owner)  # parented, so it lives until it has delivered
    relay.done.connect(lambda result, error: (done(result, error), relay.deleteLater()),
                       Qt.ConnectionType.QueuedConnection)

    def run() -> None:
        try:
            relay.done.emit(work(), None)
        except Exception as e:
            relay.done.emit(None, e)
    threading.Thread(target=run, name="window-task", daemon=True).start()


class FlowLayout(QLayout):
    """Items left to right, wrapping to the next line (Your words as chips)."""

    def __init__(self, spacing: int = 8):
        super().__init__()
        self._items: list = []
        self._spacing = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:
        self._items.append(item)

    def addWidget(self, widget: QWidget) -> None:
        self.addChildWidget(widget)
        self.addItem(QWidgetItem(widget))

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._place(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        self._place(rect, apply=True)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        return size

    def _place(self, rect: QRect, apply: bool) -> int:
        x, y, line = rect.x(), rect.y(), 0
        for item in self._items:
            if item.widget() is not None and item.widget().isHidden():
                continue
            hint = item.sizeHint()
            if x + hint.width() > rect.right() + 1 and line > 0:
                x, y, line = rect.x(), y + line + self._spacing, 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + self._spacing
            line = max(line, hint.height())
        return y + line - rect.y()


class Page(QScrollArea):
    """A page: its header (a title, a subtitle, a slot on the right, and a way back for a page one level down), then its
    cards; it scrolls when the window is small. The page itself paints the soft depth of the cards on it."""

    def __init__(self, title: str = "", subtitle: str = "", right: QWidget | None = None, back: Button | None = None):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = Host("base")
        body.setObjectName("page")
        self.body = QVBoxLayout(body)
        # 32 around: a card's soft light shadow fades out before the page's edge (at 24 it showed as a line in Porcelain)
        self.body.setContentsMargins(32, 32, 32, 32)
        self.body.setSpacing(24)
        self.title = text(title, "h1")
        self.subtitle = label(subtitle, "subtitle")
        self.header = QWidget()
        head = QHBoxLayout(self.header)
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(16)
        words = QVBoxLayout()
        words.setSpacing(4)
        if back is not None:
            words.addWidget(back, 0, Qt.AlignmentFlag.AlignLeft)
            words.addSpacing(2)
        words.addWidget(self.title)
        if subtitle:
            words.addWidget(self.subtitle)
        head.addLayout(words, 1)
        if right is not None:
            head.addWidget(right, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        if title:
            self.body.addWidget(self.header)
        self.setWidget(body)

    def add(self, item) -> None:
        if isinstance(item, QLayout):
            self.body.addLayout(item)
        else:
            self.body.addWidget(item)

    def set_compact(self, compact: bool) -> None:
        margin = 24 if compact else 32
        self.body.setContentsMargins(margin, margin, margin, margin)
        self.body.setSpacing(16 if compact else 24)


def advanced_row(title: str, caption_text: str, on_click) -> QPushButton:
    """The flat line that opens a page's Advanced part: a chevron, its name, what's in it."""
    b = Button(title, "quiet", icon="chevron-right")
    b.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    b.setToolTip(caption_text)
    holder = QWidget()
    layout = QVBoxLayout(holder)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(divider())
    inner = QHBoxLayout()
    inner.setContentsMargins(0, 6, 0, 6)
    inner.setSpacing(12)
    b.clicked.connect(on_click)
    inner.addWidget(b, 0)
    hint = caption(caption_text, "3", wrap=False)
    inner.addWidget(hint, 1)
    layout.addLayout(inner)
    holder.button = b
    return holder


def link_row(title: str, caption_text: str, on_click, icon: str = "chevron-right") -> QWidget:
    """A line that leads somewhere: its name and what's there, and a chevron."""
    w = QPushButton()
    w.setCursor(Qt.CursorShape.PointingHandCursor)
    w.setFlat(True)
    w.setMinimumHeight(56)
    layout = QHBoxLayout(w)
    layout.setContentsMargins(0, 8, 0, 8)
    layout.setSpacing(12)
    words = QVBoxLayout()
    words.setSpacing(2)
    words.addWidget(label(title, "rowtitle"))
    if caption_text:
        words.addWidget(caption(caption_text))
    layout.addLayout(words, 1)
    chevron = QLabel()
    chevron.setPixmap(theme.icon_pixmap(icon, tok("text3").name(), 16, 1.0))
    layout.addWidget(chevron)
    w.clicked.connect(on_click)
    w.setAccessibleName(title)
    return w


# ---------------------------------------------------------------- the microphone box (AI & models and the welcome)

def fit_to_width(box: QComboBox, letters: int = 12) -> QComboBox:
    """A dropdown that asks for the room of a few letters, not for its longest item's: a long name never makes its page
    wider than the window (the user testing's M-03 and N-04: a microphone's or a model's name)."""
    box.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    box.setMinimumContentsLength(letters)
    box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return box


class ElidedChoice(Choice):
    """A dropdown whose closed box cuts a long name with … ("Microphone Array (Intel® Smart Sound…"), the whole name in
    its tooltip; the list shows every name whole."""

    def __init__(self):
        super().__init__()
        fit_to_width(self)
        self.currentIndexChanged.connect(lambda _=0: self.setToolTip(self.currentText()))

    def addItem(self, text: str, data=None) -> None:
        super().addItem(text, data)
        self.setItemData(self.count() - 1, text, Qt.ItemDataRole.ToolTipRole)
        self.setToolTip(self.currentText())  # also while its signals are blocked (the list filled again)

    def paintEvent(self, event):
        p = QStylePainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        p.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)
        room = self.style().subControlRect(QStyle.ComplexControl.CC_ComboBox, option,
                                           QStyle.SubControl.SC_ComboBoxEditField, self).width()
        option.currentText = self.fontMetrics().elidedText(option.currentText, Qt.TextElideMode.ElideRight, room - 4)
        p.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, option)
        p.end()


class ElidedText(QLabel):
    """A label on one line, cut with … when the space is short (the whole text is its tooltip), so a long name never
    sets the width of its page. `role` and `tone` as label()'s; no role for a font set by hand (a key, an address)."""

    def __init__(self, value: str = "", role: str | None = "caption", tone: str | None = "3"):
        super().__init__()
        if role:
            self.setProperty("role", role)
        if tone:
            self.setProperty("tone", tone)
        self.setMinimumWidth(1)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._full = ""
        self.setText(value)

    def setText(self, value: str) -> None:
        self._full = value
        self._elide()

    def text(self) -> str:
        return self._full

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        width = self.width() if self.width() > 1 else 10_000  # before the first layout: whole
        shown = self.fontMetrics().elidedText(self._full, Qt.TextElideMode.ElideRight, width)
        super().setText(shown)
        self.setToolTip(self._full if shown != self._full else "")


class MicrophoneBox(QWidget):
    """A microphone choice with a live level meter, so the user sees at once that the microphone hears them. The list
    follows Windows while it's shown (a headset plugged in or out shows up within FOLLOW_MS), "Windows default" says
    which microphone that is now, and a chosen one that isn't connected says what Rflow uses meanwhile. Long names are
    cut with … (whole in the tooltip), so they never widen the page."""

    changed = Signal(str)  # the chosen device name ("" = the Windows default)
    followed = Signal()  # the list changed (a microphone plugged in or out, a new default)

    FOLLOW_MS = 2000

    def __init__(self, current: str, microphones: list[str], default: str = "", source=None):
        """`source` () -> (microphones, default name) is asked again every FOLLOW_MS while the box is shown."""
        super().__init__()
        self._source, self._listed = source, None
        self.combo = ElidedChoice()
        self.combo.currentIndexChanged.connect(self._chosen)
        self.level = Meter()
        self.level.setToolTip("Say something: the bars light up.")
        self.note = caption("", "3")
        self.note.hide()
        self.hearing = ElidedText()  # the microphone the meter (and so dictation) actually opened
        self.hearing.hide()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        line = QHBoxLayout()
        line.setSpacing(12)
        line.addWidget(self.combo, 1)
        line.addWidget(self.level, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(line)
        layout.addWidget(self.note)
        layout.addWidget(self.hearing)
        self.meter = LevelMeter(current or None)
        self._timer = QTimer(self, interval=50, timeout=self._show_level)
        self._follow_timer = QTimer(self, interval=self.FOLLOW_MS, timeout=self._follow)
        self._error = ""
        self.set_microphones(microphones, default, current)

    def device(self) -> str:
        return self.combo.currentData() or ""

    def set_microphones(self, microphones: list[str], default: str = "", current: str | None = None) -> None:
        """The list as Windows has it now; the choice is kept (one not connected stays, marked so)."""
        current = self.device() if current is None else current
        self._listed = (list(microphones), default)
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem(f"Windows default (now: {default})" if default else "Windows default", "")
        for name in microphones:
            self.combo.addItem(name, name)
        if current and self.combo.findData(current) < 0:
            self.combo.addItem(f"{current} (not connected)", current)
        self.combo.setCurrentIndex(max(0, self.combo.findData(current)))
        self.combo.blockSignals(False)
        self.combo.setToolTip(self.combo.currentText())
        self._show_note()

    def _show_note(self) -> None:
        microphones, default = self._listed or ([], "")
        current = self.device()
        if self._error:
            text = self._error
        elif current and current not in microphones:
            text = f"Not connected now: Rflow uses {default or 'the default microphone'} until it is."
        else:
            text = ""
        self.note.setText(text)
        self.note.setVisible(bool(text))

    def _follow(self) -> None:
        if self._source is None:
            return
        try:
            microphones, default = self._source()
        except Exception as e:  # never break the page over a list
            log.warning("Couldn't list the microphones: %s", e)
            return
        if (list(microphones), default) != self._listed:
            self.set_microphones(microphones, default)
            if self._timer.isActive():
                self._restart()  # the meter listens to the microphone Rflow would use now
            self.followed.emit()

    def _chosen(self) -> None:
        self.meter.device = self.device() or None
        if self._timer.isActive():
            self._restart()
        self._show_note()
        self.changed.emit(self.device())

    def _restart(self) -> None:
        self.meter.stop()
        try:
            self.meter.start()
            self._error = ""
            hearing = self.meter.describe().get("device", "")
            self.hearing.setText(f"Hearing: {hearing}" if hearing else "")
            self.hearing.setVisible(bool(hearing))
        except Exception as e:
            self._error = f"Could not open this microphone: {e}"
            self.hearing.hide()
        self._show_note()

    def _show_level(self) -> None:
        db = 20 * math.log10(self.meter.level + 1e-6)  # speech is roughly -45..-15 dBFS
        self.level.set_value(min(1.0, max(0.0, (db + 55) / 40)))

    def showEvent(self, event):
        super().showEvent(event)
        if QGuiApplication.platformName() != "offscreen":  # tests and the self-test don't open a microphone
            self._follow()
            self._restart()
            self._timer.start()
            self._follow_timer.start()

    def hideEvent(self, event):
        self._timer.stop()
        self._follow_timer.stop()
        self.meter.stop()
        self.level.set_value(0)
        super().hideEvent(event)


# ---------------------------------------------------------------- Home

def _day_title(day: date, today: date) -> str:
    if day == today:
        return "Today"
    if day == today - timedelta(days=1):
        return "Yesterday"
    return day.strftime("%A %d %B") if day.year == today.year else day.strftime("%d %B %Y")


def how_to_dictate(label_text: str) -> str:
    extra = f" {label_text}+Space starts hands-free as well." if label_text == "Ctrl+Win" else ""
    return f"Tap once for hands-free, tap again to finish.{extra} Esc cancels."


class ElidedLabel(QLabel):
    """One line of text, cut with … when the space is short (the full text is its tooltip)."""

    def __init__(self, value: str = ""):
        super().__init__(value)
        self.setMinimumWidth(1)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setFont(self.font())
        p.setPen(self.palette().color(self.foregroundRole()))
        shown = self.fontMetrics().elidedText(" ".join(self.text().split()), Qt.TextElideMode.ElideRight, self.width())
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, shown)
        p.end()


class HistoryRow(QWidget):
    """One dictation: its time, its text on one line, and Correct and Copy when the pointer is over it."""

    def __init__(self, when: datetime, entry: dict, on_copy, on_correct):
        super().__init__()
        surface(self, "row", 14)
        self.setFixedHeight(42)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 7, 0)
        layout.setSpacing(16)
        time_label = QLabel(when.strftime("%H:%M"))
        time_label.setFont(font(12, 500, mono=True))
        time_label.setProperty("tone", "3")
        time_label.setFixedWidth(40)
        self.body = ElidedLabel(entry["text"])
        self.body.setToolTip(entry["text"] + (f"\n\nHeard: {entry['heard']}" if entry.get("heard") else ""))
        self.copy = icon_button("copy", "Copy", size=28)
        self.copy.clicked.connect(lambda: on_copy(self.copy, entry["text"]))
        self.correct = icon_button("edit", "Correct it: Rflow learns from corrections you make twice", size=28)
        self.correct.clicked.connect(lambda: on_correct(entry["text"]))
        self.actions = QWidget()
        actions = QHBoxLayout(self.actions)
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(8)
        actions.addWidget(self.correct)
        actions.addWidget(self.copy)
        self.actions.setVisible(False)
        layout.addWidget(time_label)
        layout.addWidget(self.body, 1)
        layout.addWidget(self.actions)

    def enterEvent(self, event):
        self.actions.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.actions.setVisible(False)
        super().leaveEvent(event)


class HomePage(Page):
    """The voice orb and how to dictate (or what Rflow still needs), the stats strip, and the dictations."""

    SHOWN = 30  # dictations listed before "Show all"

    def __init__(self, app, go_to=None):
        super().__init__()
        self.app, self.go_to = app, go_to or (lambda page: None)
        self.ready, self.show_all, self.compact = False, False, False

        # the hero: the orb, the state, how to dictate (or what's missing), and what Rflow uses
        self.hero, hero = card(24, (24, 24, 24, 24), horizontal=True)
        self.orb = Orb(128, level=lambda: 0.0)
        hero.addWidget(self.orb, 0, Qt.AlignmentFlag.AlignVCenter)
        column = QVBoxLayout()
        column.setSpacing(8)
        self.status_row, self.status_lamp, self.status_text = lamp_row("ok", "Ready in any app")
        column.addWidget(self.status_row)
        self.hero_keys = QWidget()  # "Hold [Ctrl] + [Win] and talk. Let go, it's typed."
        keys_line = QHBoxLayout(self.hero_keys)
        keys_line.setContentsMargins(0, 0, 0, 0)
        keys_line.setSpacing(8)
        self.hold = label("Hold", "hero", wrap=False)
        self.hero_caps = QWidget()
        self.hero_caps_layout = QHBoxLayout(self.hero_caps)
        self.hero_caps_layout.setContentsMargins(0, 0, 0, 0)
        self.and_talk = label("and talk.", "hero", wrap=False)
        self.let_go = label("Let go, it's typed.", "hero", wrap=False)
        for widget in (self.hold, self.hero_caps, self.and_talk, self.let_go):
            keys_line.addWidget(widget, 0, Qt.AlignmentFlag.AlignVCenter)
        keys_line.addStretch()
        column.addWidget(self.hero_keys)
        self.hero_title = label("", "hero")  # the line when Rflow isn't ready
        column.addWidget(self.hero_title)
        self.hero_text = label("", tone="2")
        self.hero_text.setMaximumWidth(520)
        column.addWidget(self.hero_text)
        self.meta = QWidget()
        meta = QHBoxLayout(self.meta)
        meta.setContentsMargins(0, 4, 0, 0)
        meta.setSpacing(16)
        self.mic_meta = QLabel()
        self.mic_meta_text = caption("", "3", wrap=False)
        self.ai_meta = QLabel()
        self.ai_meta_text = caption("", "violet", wrap=False)
        for widget in (self.mic_meta, self.mic_meta_text):
            meta.addWidget(widget)
        meta.addSpacing(8)
        for widget in (self.ai_meta, self.ai_meta_text):
            meta.addWidget(widget)
        meta.addStretch()
        column.addWidget(self.meta)
        self.progress_row = QWidget()
        progress = QHBoxLayout(self.progress_row)
        progress.setContentsMargins(0, 8, 0, 0)
        progress.setSpacing(16)
        self.wave = WaveProgress()
        self.progress_text = QLabel()
        self.progress_text.setFont(font(12, 500, mono=True))
        self.progress_text.setProperty("tone", "3")
        progress.addWidget(self.wave)
        progress.addWidget(self.progress_text)
        progress.addStretch()
        column.addWidget(self.progress_row)
        self.actions = QWidget()
        actions = QHBoxLayout(self.actions)
        actions.setContentsMargins(0, 8, 0, 0)
        actions.setSpacing(12)
        self.get_parakeet = button("Download Parakeet", lambda _=False: app.download_speech_model(DEFAULT_MODEL),
                                   primary=True, icon="download")
        self.use_cloud = button("Use a cloud model instead", lambda _=False: self.go_to("speech"), icon="cloud")
        self.pause = button("Pause", lambda _=False: app.cancel_download(), kind="quiet")
        for widget in (self.get_parakeet, self.use_cloud, self.pause):
            actions.addWidget(widget)
        actions.addStretch()
        column.addWidget(self.actions)
        hero.addLayout(column, 1)
        self.add(self.hero)

        # the stats strip: one inset readout
        self.readout, strip = card(0, (20, 0, 20, 0), kind="well", horizontal=True)
        self.readout.setProperty("neuRadius", 14)
        self.readout.setFixedHeight(48)
        self.stat_values: dict[str, QLabel] = {}
        self.stat_captions: dict[str, QLabel] = {}
        for i, (key, words) in enumerate([("week", "this week"), ("speed", "words a minute"), ("streak", "days in a row"),
                                          ("total", "all time")]):
            if i:
                line = divider(vertical=True)
                line.setFixedHeight(24)
                strip.addSpacing(20)
                strip.addWidget(line, 0, Qt.AlignmentFlag.AlignVCenter)
                strip.addSpacing(20)
            value = QLabel("0")
            value.setFont(font(16, 600, tabular=True))
            words_label = caption(words, "3", wrap=False)
            pair = QHBoxLayout()
            pair.setSpacing(6)
            pair.addWidget(value, 0, Qt.AlignmentFlag.AlignVCenter)
            pair.addWidget(words_label, 0, Qt.AlignmentFlag.AlignVCenter)
            pair.addStretch()
            strip.addLayout(pair, 1)
            self.stat_values[key], self.stat_captions[key] = value, words_label
        self.add(self.readout)

        # the dictations
        self.list_card, column = card(2, (16, 16, 16, 8))
        head = QHBoxLayout()
        head.setContentsMargins(8, 0, 0, 8)
        head.setSpacing(12)
        self.list_title = label("Today", "heading", wrap=False)
        head.addWidget(self.list_title)
        head.addStretch()
        self.search_box = Card("field", 10)
        search = QHBoxLayout(self.search_box)
        search.setContentsMargins(10, 0, 6, 0)
        search.setSpacing(8)
        icon = QLabel()
        icon.setPixmap(theme.icon_pixmap("search", tok("text3").name(), 16, 1.0))
        self.search = bare(QLineEdit())
        self.search.setPlaceholderText("Search")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(lambda _="": self.refresh())
        search.addWidget(icon)
        search.addWidget(self.search, 1)
        self.search_keys = keys("Ctrl+K", "sm")
        search.addWidget(self.search_keys)
        self.search_box.setFixedSize(236, 32)
        head.addWidget(self.search_box)
        self.all_button = button("Show all", self._toggle_all, link=True, size="sm")
        head.addWidget(self.all_button)
        column.addLayout(head)
        self.history = QVBoxLayout()
        self.history.setSpacing(2)
        column.addLayout(self.history)
        column.addStretch()
        self.add(self.list_card)
        self.body.addStretch()
        QShortcut(QKeySequence("Ctrl+K"), self, activated=self._find,
                  context=Qt.ShortcutContext.WidgetWithChildrenShortcut)

    def _find(self) -> None:
        self.search.setFocus()
        self.search.selectAll()

    def _toggle_all(self) -> None:
        self.show_all = not self.show_all
        self.refresh()

    def set_ready(self, ready: bool) -> None:
        self.ready = ready

    def set_compact(self, compact: bool) -> None:
        super().set_compact(compact)
        self.compact = compact
        self.let_go.setVisible(not compact)
        self.search_box.setFixedWidth(180 if compact else 236)
        self.search_keys.setVisible(not compact)
        self.hero.layout().setContentsMargins(*(4 * [20 if compact else 24]))

    def _hero_keys(self, label_text: str) -> None:
        clear(self.hero_caps_layout)
        caps = keys(key_names(label_text))
        self.hero_caps_layout.addWidget(caps)
        self.hero_keys.setAccessibleName(f"Hold {label_text} and talk. Let go, it's typed.")
        self.hero_keys.setToolTip(self.hero_keys.accessibleName())

    def refresh(self) -> None:
        now = datetime.now()
        today = now.date()
        app = self.app
        stats: Stats = app.stats
        label_text = app.hotkey_label()
        self._show_state(label_text)
        wpm = stats.words_per_minute
        self.stat_values["week"].setText(f"{stats.words_this_week(today):,}")
        self.stat_values["total"].setText(f"{stats.words:,}")
        self.stat_values["speed"].setText(str(wpm) if wpm else "–")
        self.stat_values["streak"].setText(str(stats.streak(today)))
        self.stat_values["speed"].setToolTip("" if wpm else "Shown after half a minute of dictation.")
        self._fill_history(app.history_entries(), today)

    def _show_state(self, label_text: str) -> None:
        """Ready, or what Rflow is waiting for: Parakeet downloading, a model loading, or no speech model yet."""
        app = self.app
        in_use, loading, downloading = app.speech_in_use(), app.loading_speech, app.downloading
        fetching = bool(downloading) and downloading[0] == DEFAULT_MODEL
        state = "ready" if self.ready and in_use and not fetching else ("fetching" if fetching else
                                                                          "loading" if (loading or in_use) else "none")
        ready = state == "ready"
        self.hero_keys.setVisible(ready)
        self.hero_title.setVisible(not ready)
        self.meta.setVisible(ready)
        self.progress_row.setVisible(state == "fetching")
        self.actions.setVisible(state in ("fetching", "none"))
        self.get_parakeet.setVisible(state == "none" and not downloading)
        self.pause.setVisible(state == "fetching")
        self.use_cloud.setVisible(state in ("fetching", "none"))
        self.use_cloud.setText("Use a cloud model instead" if state == "fetching" else "Use a cloud model")
        parakeet = SPEECH_MODELS[DEFAULT_MODEL]
        if ready:
            self.orb.set_state("ready")
            self.status_lamp.set_state("ok")
            self.status_text.setText("Ready in any app")
            self._hero_keys(label_text)
            self.hero_text.setText(how_to_dictate(label_text))
            microphone = app.settings.microphone or "Windows default microphone"
            self.mic_meta.setPixmap(theme.icon_pixmap("mic", tok("text3").name(), 16, self.devicePixelRatioF()))
            self.mic_meta_text.setText(microphone)
            model = app.transform_model() if app.settings.cleanup else ""
            provider = PROVIDERS[app.gateway.service.key].name if model else ""
            self.ai_meta.setVisible(bool(model))
            self.ai_meta_text.setVisible(bool(model))
            if model:
                self.ai_meta.setPixmap(theme.icon_pixmap("tools", tok("violet").name(), 16, self.devicePixelRatioF()))
                self.ai_meta_text.setText(f"Cleanup by {provider.split(' (')[0]}")
        elif state == "fetching":
            done, total = downloading[1], downloading[2] or 1
            self.orb.set_state("loading", done / total)
            self.status_lamp.set_state("warn")
            self.status_text.setText("Not ready yet · downloading")
            self.hero_title.setText("Getting Parakeet, your private speech model")
            self.hero_text.setText("Rflow types as soon as it's here. Until then, the key says “Not ready” "
                                   "instead of doing nothing.")
            self.wave.set_value(done / total)
            self.progress_text.setText(f"{_size(done)} / {_size(total)}")
        elif state == "loading":
            name = SPEECH_MODELS.get(loading or in_use)
            self.orb.set_state("off")
            self.status_lamp.set_state("warn")
            self.status_text.setText("Getting ready")
            self.hero_title.setText(f"Loading {name.name if name else 'the speech model'}…")
            self.hero_text.setText("A few seconds. Dictation starts as soon as it's loaded.")
        else:
            self.orb.set_state("off")
            self.status_lamp.set_state("warn")
            self.status_text.setText("Not ready yet")
            self.hero_title.setText("Choose how Rflow hears you")
            self.hero_text.setText(f"Download NVIDIA Parakeet ({_size(parakeet.download.size)}) to dictate privately on "
                                   "this PC, or use a cloud model with your own key.")

    def _fill_history(self, entries: list[dict], today: date) -> None:
        clear(self.history)
        query = self.search.text().strip().casefold()
        found = [e for e in entries if not query or query in e.get("text", "").casefold()]
        self.all_button.setVisible(len(found) > self.SHOWN and not query)
        self.all_button.setText("Show fewer" if self.show_all else "Show all")
        self.search_box.setVisible(bool(entries))
        if not entries:
            self.list_title.setText("Your dictations")
            self.history.addWidget(self._empty())
            return
        if not found:
            self.list_title.setText("Search")
            self.history.addWidget(caption(f"No dictation has “{self.search.text().strip()}”.", "3"))
            return
        shown = found if self.show_all or query else found[:self.SHOWN]
        day = None
        for entry in shown:
            try:
                when = datetime.strptime(entry.get("time", ""), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            if when.date() != day:
                day = when.date()
                if self.history.count() == 0:
                    self.list_title.setText(_day_title(day, today) if not query else "Found")
                else:
                    heading = caption(_day_title(day, today), "3", wrap=False)
                    heading.setContentsMargins(8, 12, 0, 6)
                    self.history.addWidget(heading)
            self.history.addWidget(HistoryRow(when, entry, self._copy, self._correct))

    def _empty(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 16, 0, 16)
        layout.setSpacing(12)
        layout.addWidget(keys(key_names(self.app.hotkey_label()), "lg"), 0, Qt.AlignmentFlag.AlignHCenter)
        title = label("Nothing dictated yet", "heading")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(title)
        words = caption(f"Click in any text box, hold {self.app.hotkey_label()} and speak. Your dictations appear "
                        "here, kept on this PC, ready to copy again.")
        words.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(words)
        return w

    def ask_correction(self, typed: str) -> str | None:
        """The corrected text, or None if cancelled (a dialog; the tests replace it)."""
        value, ok = QInputDialog.getText(self, APP_NAME, "What should it have been?", text=typed)
        return value if ok else None

    def _correct(self, typed: str) -> None:
        corrected = self.ask_correction(typed)
        if corrected is not None and corrected.strip() and corrected.strip() != typed:
            self.app.correct_dictation(typed, corrected.strip())

    def _copy(self, source: IconButton, value: str) -> None:
        QGuiApplication.clipboard().setText(value)
        source.set_icon("check")
        source.setToolTip("Copied")

        def back() -> None:
            try:
                source.set_icon("copy")
                source.setToolTip("Copy")
            except RuntimeError:  # the list was filled again meanwhile (a new dictation): that button is gone
                pass
        QTimer.singleShot(1200, back)


# ---------------------------------------------------------------- Words (Your words) and Snippets: a section each

class WordChip(QWidget):
    """A word as a chip: flat at rest, raised with ✕ under the pointer."""

    def __init__(self, word: str, note: str = "", tip: str = "", on_remove=None):
        super().__init__()
        surface(self, "chip", 999)
        self.word = word
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 6, 0)
        layout.setSpacing(6)
        name = QLabel(word)
        name.setFont(font(13, 500))
        layout.addWidget(name)
        if note:
            extra = QLabel(note)
            extra.setFont(font(12, 400))
            extra.setProperty("tone", "3")
            layout.addWidget(extra)
        self.remove = icon_button("close", f"Remove {word}", lambda _=False: on_remove(word) if on_remove else None, 20,
                                  flat=True)
        self.remove.setVisible(False)
        layout.addWidget(self.remove)
        self.setFixedHeight(32)
        if tip:
            self.setToolTip(tip)
        layout.addSpacing(0)

    def sizeHint(self) -> QSize:
        hint = super().sizeHint()
        return QSize(hint.width() + (0 if self.remove.isVisible() else 20), 32)

    def enterEvent(self, event):
        self.remove.setVisible(True)
        self.setProperty("neuDown", True)
        theme.refresh(self)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.remove.setVisible(False)
        self.setProperty("neuDown", False)
        theme.refresh(self)
        super().leaveEvent(event)


class DictionaryPage(Page):
    """Your words: names and terms Rflow should always spell right, what it hears them as, and corrections to learn."""

    def __init__(self, app, go_to):
        super().__init__("Words", "Names and terms Rflow should always get right: your colleagues, products, places. Add "
                                  "them once; Rflow listens for them and spells them your way.")
        self.app, self.go_to = app, go_to
        # one box: type to find a word among yours; Enter (or Add) adds it (several: separate them with commas)
        self.entry_box = Card("field", 14)
        entry = QHBoxLayout(self.entry_box)
        entry.setContentsMargins(12, 0, 10, 0)
        entry.setSpacing(10)
        icon = QLabel()
        icon.setPixmap(theme.icon_pixmap("search", tok("text3").name(), 16, 1.0))
        self.entry = bare(QLineEdit())
        self.entry.setPlaceholderText("Add or find a word (several: separate them with commas)")
        self.entry.returnPressed.connect(self._add)
        self.entry.textChanged.connect(lambda _="": self._filter())
        entry.addWidget(icon)
        entry.addWidget(self.entry, 1)
        entry.addWidget(keycap("↵", "sm"))
        entry.addWidget(caption("adds it", "3", wrap=False))
        self.entry_box.setFixedHeight(44)
        self.add_button = button("Add", self._add)
        self.add_button.setFixedHeight(44)
        self.add(row(self.entry_box, self.add_button, spacing=12))

        self.cleanup_off, off = card(12, (16, 12, 12, 12), kind="well", horizontal=True)
        off_icon = QLabel()
        off_icon.setPixmap(theme.icon_pixmap("info", tok("text3").name(), 16, 1.0))
        off.addWidget(off_icon)
        off.addWidget(caption("Speech recognition uses your words. AI cleanup is off, so it doesn't.", "2"), 1)
        off.addWidget(button("Turn on AI cleanup", lambda: go_to("models"), link=True, size="sm"))
        self.add(self.cleanup_off)

        self.suggestions_card = QWidget()
        self.suggestions = QVBoxLayout(self.suggestions_card)
        self.suggestions.setContentsMargins(0, 0, 0, 0)
        self.suggestions.setSpacing(12)
        self.add(self.suggestions_card)

        self.list_card, column = card(16, (20, 20, 20, 20))
        head = QHBoxLayout()
        head.addWidget(label("Your words", "heading"))
        head.addStretch()
        self.count = caption("", "3", wrap=False)
        head.addWidget(self.count)
        column.addLayout(head)
        self.chips_box = QWidget()
        self.list = FlowLayout(8)
        self.chips_box.setLayout(self.list)
        column.addWidget(self.chips_box)
        self.none_found = caption("", "3")
        self.none_found.hide()
        column.addWidget(self.none_found)
        self.add(self.list_card)

        # sound-alikes: what the speech model writes for a term ("post grass" for PostgreSQL), fixed before anything else
        self.alikes_card, column = card(12, (20, 16, 20, 20))
        column.addWidget(label("Heard as", "heading"))
        column.addWidget(caption("When Rflow writes a word wrong the same way, tell it what you meant: it's fixed "
                                 "before anything else. Corrections you make twice are learned here too."))
        self.alikes = QVBoxLayout()
        self.alikes.setSpacing(0)
        column.addLayout(self.alikes)
        self.heard = QLineEdit()
        self.heard.setPlaceholderText("When Rflow writes… (e.g. post grass)")
        field(self.heard)
        self.meant = QLineEdit()
        self.meant.setPlaceholderText("…write instead (e.g. PostgreSQL)")
        self.meant.returnPressed.connect(self._add_sound_alike)
        field(self.meant)
        arrow = QLabel()
        arrow.setPixmap(theme.icon_pixmap("arrow-right", tok("text3").name(), 16, 1.0))
        column.addLayout(row(self.heard, arrow, self.meant, button("Add sound-alike", self._add_sound_alike), spacing=10))
        self.alike_note = caption("", "2")
        self.alike_note.hide()  # until there is something to say: an empty line would leave a gap
        column.addWidget(self.alike_note)
        self.add(self.alikes_card)
        self.body.addStretch()
        self._entries: list[tuple[str, list[str]]] = []

    def refresh(self) -> None:
        settings: Settings = self.app.settings
        self.cleanup_off.setVisible(not settings.cleanup)
        clear(self.suggestions)
        suggestions = self.app.correction_suggestions()
        self.suggestions_card.setVisible(bool(suggestions))
        for s in suggestions[:5]:
            banner, line = card(12, (16, 10, 10, 10), kind="well", horizontal=True)
            sparkle = QLabel()
            sparkle.setPixmap(theme.icon_pixmap("tools", tok("violet").name(), 18, 1.0))
            line.addWidget(sparkle)
            times = "twice" if s.seen_count == 2 else f"{s.seen_count} times"
            said = label(f"Heard “{html.escape(s.original_phrase)}” {times}. You meant "
                         f"<b>{html.escape(s.corrected_phrase)}</b>.")
            said.setTextFormat(Qt.TextFormat.RichText)
            line.addWidget(said, 1)
            line.addWidget(button("Not now", lambda _=False, s=s: self._suggestion(s, False), kind="quiet", size="sm"))
            line.addWidget(button("Learn it", lambda _=False, s=s: self._suggestion(s, True), primary=True, size="sm"))
            self.suggestions.addWidget(banner)
        # Your words, plus terms that only have sound-alikes; each with what it is heard as.
        entries = {w.lower(): (w, []) for w in settings.vocabulary}
        sources: dict[str, str] = {}
        for term in self.app.dictionary_terms():
            word, aliases = entries.get(term.preferred.lower(), (term.preferred, []))
            entries[term.preferred.lower()] = (word, aliases + [a for a in term.aliases if a not in aliases])
            sources[term.preferred.lower()] = term.source
        self._entries = sorted(entries.values(), key=lambda e: e[0].lower())
        self._sources = sources
        self.count.setText(f"{len(entries)} word{'s' if len(entries) != 1 else ''}")
        self.list_card.setVisible(bool(entries))
        self._filter()
        clear(self.alikes)
        for word, aliases in self._entries:
            for alias in aliases:
                line = QWidget()
                layout = QHBoxLayout(line)
                layout.setContentsMargins(0, 10, 0, 10)
                layout.setSpacing(12)
                heard = label(f"<s>{html.escape(alias)}</s>", tone="3", wrap=False)
                heard.setTextFormat(Qt.TextFormat.RichText)
                heard.setMinimumWidth(140)
                arrow = QLabel()
                arrow.setPixmap(theme.icon_pixmap("arrow-right", tok("text3").name(), 14, 1.0))
                layout.addWidget(heard)
                layout.addWidget(arrow)
                layout.addWidget(label(word, "rowtitle", wrap=False), 1)
                learned = sources.get(word.lower()) == "learned"
                layout.addWidget(caption("Learned from your fixes" if learned else "You added it", "3", wrap=False))
                if self.alikes.count():
                    self.alikes.addWidget(divider())
                self.alikes.addWidget(line)

    def _filter(self) -> None:
        """The chips, those matching what's typed in the box first (typing finds a word among yours)."""
        clear(self.list)
        typed = self.entry.text().strip().casefold()
        self.add_button.setEnabled(bool(typed))
        shown = [(w, a) for w, a in self._entries if not typed or typed.split(",")[-1].strip() in w.casefold()
                 or "," in typed]
        for word, aliases in shown:
            note, tip = "", ""
            if not speech_hints([word]):
                note = "everyday word"
                tip = ("Speech models spell everyday words right by themselves; listed as hints, they get heard where "
                       "you didn't say them. Keep names and terms here.")
            elif aliases:
                note = "heard as " + ", ".join(f"“{a}”" for a in aliases[:2])
            self.list.addWidget(WordChip(word, note, tip, self._remove))
        self.none_found.setVisible(bool(typed) and not shown and "," not in typed)
        self.none_found.setText(f"“{self.entry.text().strip()}” isn't in your words yet: press Enter to "
                                "add it.")
        self.chips_box.updateGeometry()

    def _suggestion(self, suggestion, accept: bool) -> None:
        if accept:
            self.app.accept_suggestion(suggestion)
        else:
            self.app.reject_suggestion(suggestion)
        self.refresh()

    def _say(self, message: str) -> None:
        self.alike_note.setText(message)
        self.alike_note.setVisible(bool(message))

    def _add_sound_alike(self) -> None:
        heard, meant = self.heard.text().strip(), self.meant.text().strip()
        if not heard or not meant:
            self._say("Fill in both: what Rflow writes, and what it should write.")
            return
        try:
            self.app.add_sound_alike(heard, meant)
        except ValueError as e:  # e.g. the same sound-alike already stands for another word
            self._say(str(e))
            return
        self._say(f"From now on “{heard}” is written as {meant}.")
        self.heard.clear()
        self.meant.clear()
        self.refresh()

    def _add(self) -> None:
        new = [w.strip() for w in re.split(r"[,\n]", self.entry.text()) if w.strip()]
        if new:
            self.app.add_words(new)
            self.entry.clear()
            self.refresh()

    def _remove(self, word: str) -> None:
        aliases = next((a for w, a in self._entries if w == word), [])
        self.app.remove_word(word)
        self.refresh()
        toast = getattr(self.window(), "toast", None)
        if toast is not None:
            toast.show_message(f"Removed “{word}”", "close", "err", "Undo", lambda: self._undo_remove(word, aliases))

    def _undo_remove(self, word: str, aliases: list[str]) -> None:
        self.app.add_words([word])
        for alias in aliases:
            try:
                self.app.add_sound_alike(alias, word)
            except ValueError:
                pass
        self.refresh()


class SnippetsPage(Page):
    """Snippets (sst.snippets): say a short phrase, get your own text typed, exactly as written here."""

    def __init__(self, app, go_to=None):
        super().__init__("Snippets", "Say a short phrase, get your own text: “my email” types your email address, "
                                     "“my signature” your signature, line breaks and all. Typed exactly as "
                                     "written here, and never sent to the AI.")
        self.app = app
        self.editing: str | None = None  # the cue of the snippet being edited
        form, layout = card(12, (20, 20, 20, 20))
        self.form_title = label("Add a snippet", "heading")
        layout.addWidget(self.form_title)
        self.cue = QLineEdit()
        self.cue.setPlaceholderText("When I say… (e.g. my email)")
        field(self.cue)
        layout.addWidget(self.cue)
        self.snippet_text = QPlainTextEdit()
        self.snippet_text.setPlaceholderText("…type this (e.g. xyz@gmail.com). Several lines are fine.")
        field(self.snippet_text)
        fixed_height(self.snippet_text, 88)
        layout.addWidget(self.snippet_text)
        self.anywhere = Toggle("Also inside a sentence")
        anywhere, _, _ = setting_row("Also inside a sentence", "“send it to my email”. Off: only when you say "
                                     "the phrase on its own, so “I checked my email” stays as you said it.",
                                     self.anywhere)
        anywhere.layout().setContentsMargins(0, 4, 0, 4)
        layout.addWidget(anywhere)
        self.note = caption("", "2")
        self.note.hide()
        layout.addWidget(self.note)
        self.save_button = button("Add", self._save, primary=True, size="sm")
        self.cancel_button = button("Cancel", self._cancel, kind="quiet", size="sm")
        self.cancel_button.hide()
        layout.addLayout(row(self.save_button, self.cancel_button, stretch_at=2))
        self.add(form)

        trial, layout = card(10, (20, 18, 20, 20))
        layout.addWidget(label("Try it", "heading"))
        self.trial = QLineEdit()
        self.trial.setPlaceholderText("Type what you would say, e.g. send it to my email")
        field(self.trial)
        self.trial.textChanged.connect(self._try)
        layout.addWidget(self.trial)
        self.trial_result = caption("", "2")
        layout.addWidget(self.trial_result)
        self.add(trial)

        self.list_card, self.list = card(0, (20, 12, 12, 12))
        head = QHBoxLayout()
        head.setContentsMargins(0, 4, 8, 8)
        head.addWidget(label("Your snippets", "heading"))
        head.addStretch()
        self.count = caption("", "3", wrap=False)
        head.addWidget(self.count)
        self.list.addLayout(head)
        self.rows = QVBoxLayout()
        self.rows.setSpacing(0)
        self.list.addLayout(self.rows)
        self.add(self.list_card)
        self.body.addStretch()

    def refresh(self) -> None:
        mine = load_snippets(self.app.settings.snippets)
        clear(self.rows)
        self.count.setText(f"{len(mine)} snippet{'s' if len(mine) != 1 else ''}")
        self.list_card.setVisible(bool(mine))
        for snippet in mine:
            line = QWidget()
            layout = QHBoxLayout(line)
            layout.setContentsMargins(0, 8, 0, 8)
            layout.setSpacing(12)
            layout.addWidget(label(f"“{snippet.cue}”", "rowtitle", wrap=False))
            arrow = QLabel()
            arrow.setPixmap(theme.icon_pixmap("arrow-right", tok("text3").name(), 14, 1.0))
            layout.addWidget(arrow)
            lines = snippet.text.strip().splitlines()
            shown = ElidedLabel(lines[0] + (" …" if len(lines) > 1 else ""))
            shown.setProperty("tone", "2")
            shown.setToolTip(snippet.text)
            layout.addWidget(shown, 1)
            if snippet.anywhere:
                layout.addWidget(caption("also inside sentences", "3", wrap=False))
            layout.addWidget(icon_button("edit", f"Edit “{snippet.cue}”", lambda _=False, s=snippet: self._edit(s),
                                         28))
            layout.addWidget(icon_button("trash", f"Remove “{snippet.cue}”",
                                         lambda _=False, s=snippet: self._remove(s), 28))
            if self.rows.count():
                self.rows.addWidget(divider())
            self.rows.addWidget(line)
        self._try()

    def _say(self, message: str) -> None:
        self.note.setText(message)
        self.note.setVisible(bool(message))

    def _save(self) -> None:
        cue, body = " ".join(self.cue.text().split()), self.snippet_text.toPlainText().strip()
        if not cue or not body:
            self._say("Fill in both: what you say, and the text to type.")
            return
        new = Snippet.from_dict({"cue": cue, "text": body, "anywhere": self.anywhere.isChecked()})
        if new is None:
            self._say(f"A phrase of at most {SNIPPET_WORDS} words, please.")
            return
        mine = [s for s in load_snippets(self.app.settings.snippets) if s.cue != self.editing]
        if any(compact_cue(s.cue) == compact_cue(cue) for s in mine):
            self._say(f"There is already a snippet for “{cue}”.")
            return
        self.app.apply_settings(dataclasses.replace(self.app.settings, snippets=[s.to_dict() for s in [*mine, new]]))
        warning = ""
        if new.anywhere and len(cue.split()) == 1:
            warning = f"Saved. Note: “{cue}” will be replaced every time you say it in a sentence."
        self._cancel()
        self._say(warning or f"Saved: say “{cue}” to type it.")

    def _edit(self, snippet: Snippet) -> None:
        self.editing = snippet.cue
        self.cue.setText(snippet.cue)
        self.snippet_text.setPlainText(snippet.text)
        self.anywhere.setChecked(snippet.anywhere)
        self.form_title.setText("Edit the snippet")
        self.save_button.setText("Save")
        self.cancel_button.show()
        self._say("")
        self.cue.setFocus()

    def _cancel(self) -> None:
        self.editing = None
        self.cue.clear()
        self.snippet_text.clear()
        self.anywhere.setChecked(False)
        self.form_title.setText("Add a snippet")
        self.save_button.setText("Add")
        self.cancel_button.hide()
        self._say("")
        self.refresh()

    def _remove(self, snippet: Snippet) -> None:
        mine = [s.to_dict() for s in load_snippets(self.app.settings.snippets) if s.cue != snippet.cue]
        self.app.apply_settings(dataclasses.replace(self.app.settings, snippets=mine))
        if self.editing == snippet.cue:
            self._cancel()
        self.refresh()

    def _try(self, *_) -> None:
        said, mine = self.trial.text().strip(), load_snippets(self.app.settings.snippets)
        if not said:
            self.trial_result.setText("What Rflow would type appears here.")
            return
        if (snippet := snippet_alone(said, mine)) is not None:
            self.trial_result.setText(f"Types: {snippet.text}")
            return
        protected, slots = protect_snippets(said, mine)
        typed = expand_snippets(protected, slots) or said
        self.trial_result.setText(f"Types: {typed}" if slots else "No snippet here: typed as you said it.")


# ---------------------------------------------------------------- Reading test

class ReadingTest(QWidget):
    """The reading test: read a set of sentences aloud, then see how many words each setup gets wrong (sst/evaluate.py)."""

    scored = Signal(object)  # from the scoring thread
    progressed = Signal(str)
    failed = Signal(str)
    restart = Signal()  # "New test"

    def __init__(self, recorder, score, add_words, microphone: str, folder: Path | None = None, block: str = "A"):
        super().__init__()
        self.recorder, self._score, self._add_words, self.microphone = recorder, score, add_words, microphone
        self.folder = folder or bench.BENCH_DIR / time.strftime("%Y-%m-%d_%H%M%S")
        self.report_folder = self.folder  # where the shown results were saved: this test, or 'summary' for all
        # A test already started keeps its set; a new one reads the set it was given.
        started = (self.folder / bench.SESSION_FILE).exists() or bool(self.recorded())
        self.block = bench.read_session(self.folder)["block"] if started else block
        self.sentences = bench.BLOCKS[self.block]
        self.index, self.recording = 0, False
        self.scored.connect(self._show_results)
        self.progressed.connect(lambda message: self.status.setText(message))
        self.failed.connect(self._score_failed)

        # page 1: reading
        purpose = "a practice set" if self.block in bench.TUNING else "a test set"
        intro = caption(f"Read each sentence aloud the way you normally dictate. Set {self.block} of {len(bench.BLOCKS)} "
                        f"({purpose}), microphone: {microphone}. Rflow then counts the words it gets wrong, with and "
                        "without AI cleanup. About 6 minutes; you can stop and continue later.", "2")
        self.counter = caption("", "3", wrap=False)
        self.sentence = text("", "sentence")
        self.sentence.setMinimumHeight(110)
        self.level = QProgressBar()
        self.level.setRange(0, 100)
        self.level.setTextVisible(False)
        self.level.setFixedHeight(6)
        self.status = label("Press Record (or Space), read the sentence, then Stop.", tone="2")
        self.record_button = button("Record", self.toggle_recording, primary=True)
        self.redo_button = button("Redo", self.toggle_recording)
        self.back_button = button("Back", lambda: self.go(self.index - 1), kind="quiet")
        self.next_button = button("Next", lambda: self.go(self.index + 1), kind="quiet")
        self.score_button = button("Score", self.start_scoring)
        reading, reading_layout = card(14, (24, 24, 24, 24))
        for widget in (intro, self.counter, self.sentence, self.level, self.status):
            reading_layout.addWidget(widget)
        reading_layout.addStretch()
        reading_layout.addLayout(row(self.record_button, self.redo_button, self.back_button, self.next_button,
                                     self.score_button, stretch_at=4))

        # page 2: results
        self.report = QTextBrowser()
        self.report.setOpenExternalLinks(False)
        self.report.setMinimumHeight(200)
        field(self.report, "well")
        self.suggestions = QListWidget()
        self.suggestions.setMaximumHeight(130)
        field(self.suggestions, "well")
        self.results_status = caption("", "2")
        results, results_layout = card(12, (24, 24, 24, 24))
        results_layout.addWidget(self.report, 1)
        results_layout.addWidget(label("Worth adding to Your words (untick any you don't want):"))
        results_layout.addWidget(self.suggestions)
        add_row = row(button("Add to Your words", self._add_selected, primary=True, size="sm"), self.results_status)
        add_row.setStretch(1, 1)  # the note takes the rest of the line
        results_layout.addLayout(add_row)
        # Two rows: one would make the page wider than the window's smallest size.
        results_layout.addLayout(row(button("Score again", self.start_scoring, size="sm"),
                                     button("Score all tests", lambda: self.start_scoring(every=True), size="sm"),
                                     button("Open folder", lambda: open_folder(self.report_folder), size="sm"),
                                     button("New test", self.restart.emit, size="sm"), stretch_at=4))

        self.pages = QStackedWidget()
        self.pages.addWidget(reading)
        self.pages.addWidget(results)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.pages)
        # Only while the test has the focus: Space must still type spaces on the other pages.
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self._space,
                  context=Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.meter = QTimer(self)
        self.meter.setInterval(50)
        self.meter.timeout.connect(lambda: self.level.setValue(min(100, int(self.recorder.level * 400))))
        self.go(next(iter(self._to_read()), 0))  # a resumed test opens at the first sentence not read yet
        if self.recorded():
            self.status.setText(f"Continuing your unfinished test ({self.recorded()} of {len(self.sentences)} read).")

    def recorded(self) -> int:
        return len(bench.recordings(self.folder)) if self.folder.exists() else 0

    def _to_read(self) -> list[int]:
        return [i for i in range(len(self.sentences)) if not (self.folder / f"{i + 1:02}.wav").exists()]

    def go(self, index: int) -> None:
        if self.recording:
            return
        self.index = max(0, min(index, len(self.sentences) - 1))
        done = (self.folder / f"{self.index + 1:02}.wav").exists()
        self.counter.setText(f"Set {self.block}  ·  sentence {self.index + 1} of {len(self.sentences)}"
                             + ("  ·  recorded" if done else ""))
        self.sentence.setText(self.sentences[self.index])
        self.record_button.setEnabled(not done)
        self.redo_button.setEnabled(done)
        self.back_button.setEnabled(self.index > 0)
        self.next_button.setEnabled(self.index < len(self.sentences) - 1)
        count = self.recorded()
        self.score_button.setEnabled(count > 0)
        self.score_button.setText(f"Score ({count} recorded)" if count else "Score")

    def _space(self) -> None:
        if self.pages.currentIndex() == 0:
            self.toggle_recording()

    def toggle_recording(self) -> None:
        if not self.recording:
            try:
                self.recorder.start()
            except Exception as e:
                self.status.setText(f"Could not open the microphone: {e}")
                return
            self.recording = True
            self.meter.start()
            for b in (self.redo_button, self.back_button, self.next_button, self.score_button):
                b.setEnabled(False)
            self.record_button.setEnabled(True)
            self.record_button.setText("Stop")
            self.status.setText("Recording... read the sentence, then press Stop (or Space).")
            return
        self.recording = False
        self.meter.stop()
        self.level.setValue(0)
        self.record_button.setText("Record")
        stop_later = getattr(self.recorder, "stop_later", None)
        take = stop_later() if stop_later else Take.ready(self.recorder.stop(), self.recorder.rate)
        if take.seconds < 0.5:
            self.status.setText("That was too short; press Record and read the sentence again.")
            self.go(self.index)
            return
        if take.done.is_set():
            self._save(take)
        else:  # the moment after Stop is still being recorded, as in dictation; don't hold up the window for it
            self.record_button.setEnabled(False)
            QTimer.singleShot(int(self.recorder.tail * 1000) + 50, lambda: self._save(take))

    def _save(self, take: Take) -> None:
        audio = take.audio()
        if not (self.folder / bench.SESSION_FILE).exists():  # which set, microphone and rate: for the report
            describe = getattr(self.recorder, "describe", None)
            bench.write_session(self.folder, self.block, self.microphone,
                                describe() if describe else {"rate": self.recorder.rate})
        stem = self.folder / f"{self.index + 1:02}"
        save_wav(stem.with_suffix(".wav"), audio, take.rate)
        stem.with_suffix(".txt").write_text(self.sentences[self.index], encoding="utf-8")
        remaining = self._to_read()
        if remaining:
            self.status.setText("Saved. Next sentence:")
            later = [i for i in remaining if i > self.index]
            self.go(later[0] if later else remaining[0])
        else:
            self.status.setText("All sentences recorded. Press Score.")
            self.go(self.index)

    def stop(self) -> None:
        """Leaving the page: don't leave the microphone open (not even warm)."""
        if self.recording:
            self.recording = False
            self.meter.stop()
            self.recorder.stop()
            self.record_button.setText("Record")
            self.status.setText("Recording stopped. Press Record to read the sentence again.")
            self.go(self.index)
        close = getattr(self.recorder, "close", None)
        if close:
            close()

    def start_scoring(self, every: bool = False) -> None:
        """This test, or every test of the profile together (their folders sit next to this one)."""
        if self.recording or not self.recorded():
            return
        folders = bench.sessions(self.folder.parent) if every else [self.folder]
        self.score_button.setEnabled(False)
        self.results_status.setText("Scoring...")
        self.status.setText("Scoring...")

        def work() -> None:
            try:
                self.scored.emit(self._score(folders, self.progressed.emit))
            except Exception as e:
                log.exception("Scoring the reading test failed")
                self.failed.emit(str(e))
        threading.Thread(target=work, name="reading-test-score", daemon=True).start()

    def _show_results(self, results) -> None:
        best = min(results.scores, key=lambda s: s.error_rate)
        rows = "".join(
            f"<tr><td>{'<b>' if s is best else ''}{s.name}{'</b>' if s is best else ''}</td>"
            f"<td align=right>{s.error_rate:.1%}</td><td align=right>{_interval(s.interval)}</td>"
            f"<td>{_verdict(s.difference)}</td><td align=right>{s.term_error_rate:.1%}</td>"
            f"<td align=right>{s.percentile(50):.2f} s</td></tr>" for s in results.scores)
        misheard = "".join(f"<li>{said or '<i>(extra word)</i>'} → {heard or '<i>(missed)</i>'} ({times}×)</li>"
                           for said, heard, times in results.misheard[:10])
        sessions = len({r.session for r in results.recordings})
        self.report.setHtml(
            f"<h3>{len(results.recordings)} sentences scored" + (f" from {sessions} tests" if sessions > 1 else "") + "</h3>"
            "<table cellpadding=5><tr><th align=left>Setup</th><th>Word errors</th><th>95% range</th>"
            f"<th align=left>Against the first</th><th>Names and terms</th><th>Time</th></tr>{rows}</table>"
            f"<p>Lowest error rate: <b>{best.name}</b>."
            + (" One set gives a wide range: read more sets and score all tests for a surer answer." if sessions == 1
               else "") + "</p>"
            + "".join(f"<p>⚠ {warning}</p>" for warning in microphone_warnings(results))
            + (f"<h4>Most misheard (by speech recognition)</h4><ul>{misheard}</ul>" if misheard else ""))
        self.suggestions.clear()
        for word in results.suggestions:
            item = QListWidgetItem(word)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.suggestions.addItem(item)
        # The folder name only: a full path would make the window wider than the screen allows for it.
        self.report_folder = self.folder.parent / "summary" if sessions > 1 else self.folder
        self.results_status.setText(f"Saved as report.md in the {self.report_folder.name} folder.")
        self.results_status.setToolTip(str(self.report_folder))
        self.pages.setCurrentIndex(1)
        self.go(self.index)

    def _score_failed(self, message: str) -> None:
        self.status.setText(f"Scoring failed: {message}")
        self.results_status.setText(f"Scoring failed: {message}")
        self.go(self.index)

    def _add_selected(self) -> None:
        chosen = [self.suggestions.item(i).text() for i in range(self.suggestions.count())
                  if self.suggestions.item(i).checkState() == Qt.CheckState.Checked]
        added = self._add_words(chosen)
        self.results_status.setText(f"Added {added} word(s) to Your words. Press Score again to see the difference."
                                    if added else "Those words are already in Your words.")


def _interval(interval) -> str:
    return f"{interval[0]:.0%}–{interval[1]:.0%}" if interval else ""


def _verdict(difference) -> str:
    """A setup against the first one: better or worse only when the whole 95% range says so."""
    if difference is None:
        return "the baseline"
    mean, low, high = (100 * x for x in difference)
    return f"{mean:+.1f} points, " + ("better" if high < 0 else "worse" if low > 0 else "no clear difference")


def microphone_warnings(results) -> list[str]:
    """Plain advice from the measured audio: phone-quality Bluetooth, clipping, a very quiet microphone."""
    out = []
    for microphone, indexes in results.microphones().items():
        flags = [flag for k in indexes for flag in results.recordings[k].stats.flags]
        if flags.count("narrowband") * 2 > len(indexes):
            out.append(f"<b>{microphone}</b> sounds like a phone call (nothing above 4 kHz), as a Bluetooth headset "
                       "does while its microphone is on. Speech recognition loses consonants there; the laptop's own "
                       "microphone is usually better.")
        if "clipped" in flags:
            out.append(f"<b>{microphone}</b> was too loud in {flags.count('clipped')} recording(s): lower its level in "
                       "Windows' sound settings.")
        if flags.count("quiet") * 2 > len(indexes):
            out.append(f"<b>{microphone}</b> is very quiet: speak closer to it or raise its level.")
    return out


class ReadingTestPage(Page):
    def __init__(self, app, go_to=None):
        go_to = go_to or (lambda page: None)
        super().__init__("Reading test", "How well does Rflow understand your voice, microphone and words? Read a set of "
                                         "30 short sentences, then compare speech recognition alone and with AI "
                                         "cleanup. There are 5 sets; the more you read, the surer the numbers.",
                         back=back_button("Settings", go_to, "settings"))
        self.app = app
        self.test: ReadingTest | None = None
        self.holder = QVBoxLayout()
        self.add(self.holder)
        self.body.addStretch()

    def ensure_test(self, folder: Path | None = None, new: bool = False) -> ReadingTest:
        if self.test is None or new:
            if self.test is not None:
                self.test.stop()
                self.test.deleteLater()
            microphone, root = self.app.settings.microphone, self.app.bench_dir()  # each profile has its own tests
            folder = None if new else folder or bench.unfinished(root)
            self.test = ReadingTest(self.app.new_recorder(), self.app.score_reading, self.app.add_words,
                                    microphone or "Windows default", folder=folder or root / time.strftime("%Y-%m-%d_%H%M%S"),
                                    block=bench.next_block(root))
            self.test.restart.connect(lambda: self.ensure_test(new=True))
            self.holder.addWidget(self.test)
        return self.test

    def showEvent(self, event):
        super().showEvent(event)
        self.ensure_test()

    def hideEvent(self, event):
        if self.test:
            self.test.stop()
        super().hideEvent(event)


# ---------------------------------------------------------------- How Rflow hears you (speech models)

def _size(n: int) -> str:
    return f"{n / 1e9:.1f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def _parakeet_takes_over(provider: str) -> str:
    """What happens when a cloud model or own server can't be reached: Parakeet types it, if it is downloaded."""
    if SPEECH_MODELS[DEFAULT_MODEL].installed():
        return f"If {provider} can't be reached, Parakeet types it on this computer."
    return f"Download Parakeet too (On this PC) to have it type when {provider} can't be reached."


def _status_line() -> tuple[QWidget, Lamp, QLabel]:
    widget, lamp, words = lamp_row("ok", "", stretch=False)
    lamp.hide()
    return widget, lamp, words


def _set_status(lamp: Lamp, words: QLabel, message: str, state: str = "") -> None:
    words.setText(message)
    lamp.setVisible(bool(state and message))
    if state:
        lamp.set_state(state)


def _heading(title: QWidget, status: QWidget) -> QHBoxLayout:
    """A card's heading on the left (cut with … if it must be), its state on the right."""
    line = QHBoxLayout()
    line.setSpacing(12)
    line.addWidget(title, 1)
    line.addWidget(status, 0, Qt.AlignmentFlag.AlignVCenter)
    return line


class _ModelCard:
    """One speech model on this computer: what it is, its state, and the buttons that change it."""

    def __init__(self, app, model):
        self.app, self.model = app, model
        self.frame, layout = card(8, (20, 18, 20, 20))
        self.status_row, self.lamp, self.status = _status_line()
        layout.addLayout(_heading(ElidedText(model.name, "heading", None), self.status_row))
        layout.addWidget(label(model.summary, tone="2"))
        layout.addWidget(caption(f"Languages: {model.languages}  ·  Size: {model.size}", "3"))
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        layout.addWidget(self.progress)
        size = _size(model.download.size) if model.download else ""
        self.download = button(f"Download and use ({size})", lambda _=False: app.download_speech_model(model.key),
                               primary=True, size="sm", icon="download")
        self.cancel = button("Cancel", lambda _=False: app.cancel_download(), kind="quiet", size="sm")
        self.choose = button("Use this model", lambda _=False: app.choose_speech_model(model.key), primary=True, size="sm")
        self.remove = button("Remove download", lambda _=False: app.remove_speech_model(model.key), kind="danger",
                             size="sm")
        layout.addSpacing(4)
        layout.addLayout(row(self.download, self.cancel, self.choose, self.remove, stretch_at=3))

    def refresh(self) -> None:
        app, model, key = self.app, self.model, self.model.key
        chosen, in_use, loading, downloading = (app.settings.speech_model, app.speech_in_use(), app.loading_speech,
                                                app.downloading)
        installed = model.installed()
        here = bool(downloading) and downloading[0] == key
        state = ""
        if not model.ready:
            status = "Coming soon"
        elif here:
            done, total = downloading[1], downloading[2] or 1
            status, state = f"Downloading {done * 100 // total}%  ({_size(done)} of {_size(total)})", "warn"
            self.progress.setValue(done * 1000 // total)
        elif key == loading:
            status, state = "Loading...", "warn"
        elif key == in_use:
            status, state = "In use", "ok"
        else:
            status = "Downloaded" if installed and model.download else ""
        _set_status(self.lamp, self.status, status, state)
        self.progress.setVisible(here)
        self.download.setVisible(model.ready and not installed and not here)
        self.download.setEnabled(not downloading)  # one download at a time
        self.cancel.setVisible(here)
        self.choose.setVisible(model.ready and installed and key != chosen)
        self.choose.setEnabled(not loading)
        # Only a download can be removed (not Parakeet next to an older Rflow's program).
        self.remove.setVisible(bool(model.download) and model.download.installed() and key not in (chosen, in_use))


def _form() -> QFormLayout:
    form = QFormLayout()
    form.setHorizontalSpacing(16)
    form.setVerticalSpacing(10)
    form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    return form


class _CloudCard:
    """A cloud speech model, opened by its tile: the provider's key (the one in Your API keys, or one pasted here, which
    goes there too), its model, a Test, and "Use this model" after a question, since the voice goes to the provider."""

    def __init__(self, page, app, model):
        self.page, self.app, self.model = page, app, model
        provider = CLOUD[model.key]
        self.frame, layout = card(10, (20, 18, 20, 20))
        self.status_row, self.lamp, self.status = _status_line()
        layout.addLayout(_heading(ElidedText(model.name, "heading", None), self.status_row))
        layout.addWidget(label(model.summary, tone="2"))
        layout.addWidget(caption(f"Languages: {model.languages}  ·  {model.size}", "3"))
        self.privacy = caption("", "warn")
        layout.addWidget(self.privacy)
        form = _form()
        self.key = KeyField(app.gateway.key_for(model.key), "Paste your key")
        key_link = button("Get a free key" if model.key == "gemini" else "Get a key",
                          lambda _=False: shell.open_url(QUrl(provider.key_page)), link=True, size="sm",
                          icon="external", icon_after=True)
        key_row = row(self.key, key_link)
        key_row.setStretch(0, 1)
        form.addRow(caption("API key", "2", wrap=False), key_row)
        self.key_note = caption("", "3")
        form.addRow("", self.key_note)
        self.model_box = fit_to_width(Choice(editable=True))  # one of the usual models, or any other name it knows
        self.model_box.addItems(provider.models)
        self.model_box.setCurrentText(self._saved_model())
        self.model_box.lineEdit().setPlaceholderText("Type to search, or any model name the provider knows")
        cost_list(self.model_box, lambda m: costs.speech_cost(model.key, m))  # each model's cost a month, in the list
        self.test = button("Test", self._test, size="sm")
        model_row = row(self.model_box, self.test)
        model_row.setStretch(0, 1)
        form.addRow(caption("Model", "2", wrap=False), model_row)
        self.cost = CostLine()
        form.addRow("", self.cost)
        layout.addLayout(form)
        self.result = caption("", "2")
        self.result.hide()  # until there is something to say: an empty line would leave a gap
        layout.addWidget(self.result)
        self.bar = SaveBar(self._use, self.discard)
        self.choose = self.bar.save  # "Use this model", or "Save" once it is the one in use
        layout.addWidget(self.bar)
        self.key.textChanged.connect(lambda _="": self._show_buttons())
        self.model_box.currentTextChanged.connect(lambda _="": self._show_buttons())

    def _saved_model(self) -> str:
        return self.app.settings.speech_cloud_models.get(self.model.key) or CLOUD[self.model.key].models[0]

    def edited(self) -> bool:
        return self.key.edited() or self.model_box.currentText().strip() != self._saved_model()

    def discard(self) -> None:
        self.key.show_saved()
        self.model_box.setCurrentText(self._saved_model())
        self._say("")
        self._show_buttons()

    def refresh(self) -> None:
        app, key = self.app, self.model.key
        name = CLOUD[key].name
        self.privacy.setText(f"Your voice is sent to {name} each time you dictate. " + _parakeet_takes_over(name))
        saved = app.gateway.key_for(key)
        if not self.key.edited() and saved != self.key.saved:
            self.key.show_saved(saved)  # changed in Your API keys or the AI connection; a key being typed stays
        self.key_note.setText("From Your API keys on AI & models." if saved else
                              "A key pasted here is kept in Your API keys, encrypted on this PC.")
        if key == app.loading_speech:
            _set_status(self.lamp, self.status, "Loading...", "warn")
        elif key == app.speech_in_use():
            _set_status(self.lamp, self.status, "In use", "ok")
        else:
            _set_status(self.lamp, self.status, "")
        self._show_buttons()

    def _show_buttons(self) -> None:
        chosen, edited = self.model.key == self.app.settings.speech_model, self.edited()
        self.choose.setText("Save" if chosen else "Use this model")
        self.bar.show_state(edited, (edited or not chosen) and not self.app.loading_speech)
        model = self.model_box.currentText().strip()
        self.cost.show_estimate(costs.speech_cost(self.model.key, model) if model else None)

    def _say(self, message: str) -> None:
        self.result.setText(message)
        self.result.setVisible(bool(message))

    def _use(self) -> None:
        name, api_key, model = CLOUD[self.model.key].name, self.key.text().strip(), self.model_box.currentText().strip()
        if not api_key:
            self._say(f"Enter your {name} API key first (Get a key).")
            return
        new = self.model.key != self.app.settings.speech_model
        # Asked once: before the voice first goes to the provider, and before an expensive model is first saved
        dear = expensive(model, costs.speech_cost(self.model.key, model)) if new or model != self._saved_model() else ""
        question = (f"Use {name} for speech recognition?\n\nEach time you dictate, the recording of your voice is sent "
                    f"to {name}, which turns it into text. " + _parakeet_takes_over(name)) if new else ""
        if dear:
            question = f"{question}\n\n{dear}" if question else f"{dear}\n\nUse it anyway?"
        if question and not self.page.confirm(question):
            return
        self.app.use_cloud_speech(self.model.key, api_key, model)
        self.key.show_saved(api_key)
        self._say("")
        self.page.refresh()  # the model in use, at the top of the page
        self.bar.saved("Saved. Active from the next dictation.")

    def _test(self) -> None:
        name, api_key, model = CLOUD[self.model.key].name, self.key.text().strip(), self.model_box.currentText().strip()
        if not api_key:
            self._say(f"Enter your {name} API key first (Get a key).")
            return
        self.test.setEnabled(False)
        self._say(f"Sending a sample sentence to {name}...")

        def done(answer, error) -> None:
            self.test.setEnabled(True)
            self._say(f"Failed: {error}" if error else f"OK: {answer}")
        run_in_background(self.frame, lambda: self.app.test_cloud_speech(self.model.key, api_key, model), done)


class _ServerCard:
    """The speech model on the user's own server, opened by its tile: its address, a key if it needs one, the model
    (Load models lists the server's speech models first), a Test, and "Use this model". The address and key are kept
    apart from the AI connection's; a new card starts from your own server in Your API keys (e.g. a company gateway)."""

    def __init__(self, page, app, model):
        self.page, self.app, self.model = page, app, model
        self.frame, layout = card(10, (20, 18, 20, 20))
        self.status_row, self.lamp, self.status = _status_line()
        layout.addLayout(_heading(ElidedText(model.name, "heading", None), self.status_row))
        layout.addWidget(label(model.summary, tone="2"))
        layout.addWidget(caption(f"Languages: {model.languages}  ·  {model.size}", "3"))
        self.note = caption("", "warn")
        layout.addWidget(self.note)
        self._saved = app.gateway.speech_server()  # (address, key) as last saved
        address, key = self._saved if self._saved[0] else app.gateway.entries().get("vllm", ("", ""))
        form = _form()
        self.address = QLineEdit(address)
        self.address.setPlaceholderText("e.g. http://localhost:8000/v1, or your company's AI gateway")
        field(self.address)
        form.addRow(caption("Address", "2", wrap=False), self.address)
        self.key = KeyField(key, "Only if your server needs one")
        self.load = button("Load models", self._load_models, size="sm")
        key_row = row(self.key, self.load)
        key_row.setStretch(0, 1)
        form.addRow(caption("API key", "2", wrap=False), key_row)
        self.model_box = fit_to_width(Choice(editable=True))  # one of the loaded models, or any name the server knows
        self.model_box.lineEdit().setPlaceholderText("e.g. whisper-1: Load models, then type to search")
        if app.settings.speech_server_model:
            self.model_box.addItem(app.settings.speech_server_model)
        self.model_box.setCurrentText(app.settings.speech_server_model)
        self.test = button("Test", self._test, size="sm")
        model_row = row(self.model_box, self.test)
        model_row.setStretch(0, 1)
        form.addRow(caption("Model", "2", wrap=False), model_row)
        layout.addLayout(form)
        self.result = caption("", "2")
        self.result.hide()  # until there is something to say: an empty line would leave a gap
        layout.addWidget(self.result)
        if not self._saved[0] and address:
            self._say("Filled in from your own server in Your API keys. Load models to see what it offers.")
        self._shown = self._fields()  # what the card showed when it was last in step: changes are measured from it
        self.bar = SaveBar(self._use, self.discard)
        self.choose = self.bar.save  # "Use this model", or "Save" once it is the one in use
        layout.addWidget(self.bar)
        for box in (self.address, self.key):
            box.textChanged.connect(lambda _="": self._show_buttons())
        self.model_box.currentTextChanged.connect(lambda _="": self._show_buttons())

    def _fields(self) -> tuple[str, str, str]:
        return self.address.text().strip(), self.key.text().strip(), self.model_box.currentText().strip()

    def edited(self) -> bool:
        return self._fields() != self._shown

    def discard(self) -> None:
        address, key, model = self._shown
        self.address.setText(address)
        self.key.show_saved(key)
        self.model_box.setCurrentText(model)
        self._show_buttons()

    def refresh(self) -> None:
        app, key = self.app, self.model.key
        self.note.setText("Your voice goes to this server each time you dictate. " + _parakeet_takes_over("the server"))
        saved = app.gateway.speech_server()
        if saved != self._saved and saved[0] and not self.edited():
            self.address.setText(saved[0])  # changed elsewhere; what is being typed here is left alone
            self.key.show_saved(saved[1])
            self._shown = self._fields()
        self._saved = saved
        if key == app.loading_speech:
            _set_status(self.lamp, self.status, "Loading...", "warn")
        elif key == app.speech_in_use():
            _set_status(self.lamp, self.status, "In use", "ok")
        else:
            _set_status(self.lamp, self.status, "")
        self._show_buttons()

    def _show_buttons(self) -> None:
        chosen, edited = self.model.key == self.app.settings.speech_model, self.edited()
        self.choose.setText("Save" if chosen else "Use this model")
        self.bar.show_state(edited, (edited or not chosen) and not self.app.loading_speech)

    def _say(self, message: str) -> None:
        self.result.setText(message)
        self.result.setVisible(bool(message))

    def _ready(self, need_model: bool = True) -> bool:
        address, _, model = self._fields()
        if not address:
            self._say("Enter your server's address first.")
        elif need_model and not model:
            self._say("Choose a model first (Load models).")
        return bool(address and (model or not need_model))

    def _use(self) -> None:
        if self._ready():
            address, api_key, model = self._fields()
            self.app.use_server_speech(address, api_key, model)
            self._saved = self.app.gateway.speech_server()
            self.key.show_saved(api_key)
            self._shown = self._fields()
            self._say("")
            self.page.refresh()  # the model in use, at the top of the page
            self.bar.saved("Saved. Active from the next dictation.")

    def _busy(self, busy: bool, message: str = "") -> None:
        self.load.setEnabled(not busy)
        self.test.setEnabled(not busy)
        if message:
            self._say(message)

    def _load_models(self) -> None:
        if not self._ready(need_model=False):
            return
        address, api_key, _ = self._fields()
        self._busy(True, "Loading models...")

        def done(models, error) -> None:
            self._busy(False)
            if error:
                self._say(f"Failed: {error}")
                return
            current = self.model_box.currentText()
            self.model_box.clear()
            self.model_box.addItems(models)
            speech = [m for m in models if SPEECH.search(m)]
            self.model_box.setCurrentText(current or (speech[0] if speech else ""))
            self._say(f"Loaded {len(models)} models, {len(speech)} of them for speech (listed first): choose one, then "
                      "Test." if models else "The server lists no models; type the model's name.")
        run_in_background(self.frame, lambda: self.app.server_models(address, api_key), done)

    def _test(self) -> None:
        if not self._ready():
            return
        address, api_key, model = self._fields()
        self._busy(True, "Sending a sample sentence to the server...")

        def done(answer, error) -> None:
            self._busy(False)
            self._say(f"Failed: {error}" if error else f"OK: {answer}")
        run_in_background(self.frame, lambda: self.app.test_server_speech(address, api_key, model), done)


# What a scan verdict looks like: (lamp, words).
SCAN_LEVELS = {"recommended": ("ok", "Recommended"), "fast": ("ok", "Fast here"),
               "usable": ("off", "Works, with a short wait"), "slow": ("warn", "Slow on this computer"),
               "no": ("err", "Won't run well here")}
SCAN_HINTS = {"whisper-turbo": "Choose it for other languages, or on a computer with an NVIDIA card."}


class _ScanCard:
    """Scan my computer: the button, its progress, then this computer and a verdict for each model."""

    def __init__(self, app):
        self.app = app
        self.frame, layout = card(8, (20, 18, 20, 20))
        layout.addWidget(label("Not sure which model suits this PC?", "heading"))
        layout.addWidget(label("Scan this PC checks the memory, disk space, processor and graphics card, and tries each "
                               "downloaded model on a short sentence (about half a minute with Whisper). Models not "
                               "downloaded yet are estimated.", tone="2"))
        self.button = button("Scan this PC", lambda _=False: app.scan_computer(), size="sm")
        self.status = ElidedText("", "caption", "3")
        line = QHBoxLayout()
        line.setSpacing(12)
        line.addWidget(self.button)
        line.addWidget(self.status, 1)
        layout.addLayout(line)
        self.result = QWidget()
        result = QVBoxLayout(self.result)
        result.setContentsMargins(0, 8, 0, 0)
        result.setSpacing(8)
        self.computer = caption("", "2")
        result.addWidget(self.computer)
        self.lines = QVBoxLayout()
        self.lines.setSpacing(6)
        result.addLayout(self.lines)
        layout.addWidget(self.result)

    def refresh(self) -> None:
        scanning, data = self.app.scanning, self.app.last_scan
        self.button.setEnabled(not scanning)
        self.button.setText("Scan again" if data else "Scan this PC")
        self.status.setText(scanning or (f"Last scan: {data['time']}" if data else ""))
        self.result.setVisible(bool(data))
        if not data:
            return
        self.computer.setText("This PC: " + Computer(**data["computer"]).summary())
        clear(self.lines)
        for verdict in data["verdicts"]:
            model = SPEECH_MODELS.get(verdict["key"])
            if not model:
                continue
            state, words = SCAN_LEVELS.get(verdict["level"], ("off", verdict["level"]))
            reason = verdict["reason"][:1].upper() + verdict["reason"][1:]
            hint = SCAN_HINTS.get(model.key, "") if verdict["level"] in ("usable", "slow") else ""
            line = QWidget()
            layout = QHBoxLayout(line)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(10)
            layout.addWidget(Lamp(state), 0, Qt.AlignmentFlag.AlignTop)
            layout.addWidget(label(f"{model.name}: {words}. {reason}." + (f" {hint}" if hint else ""), tone="2"), 1)
            self.lines.addWidget(line)


class _InUseCard:
    """The top of How Rflow hears you: the model in use, and the language you speak. It's one setting for every model
    that can choose (Whisper, the cloud, your own server), so it's set here, once, not on each model's card."""

    def __init__(self, page, app):
        self.page, self.app = page, app
        self.frame, layout = card(10, (20, 18, 20, 20))
        self.lamp = Lamp("ok")
        self.title = ElidedText("", "heading", None)
        heading = QHBoxLayout()
        heading.setSpacing(10)
        heading.addWidget(self.lamp, 0, Qt.AlignmentFlag.AlignVCenter)
        heading.addWidget(self.title, 1)
        layout.addLayout(heading)
        self.detail = label("", tone="2")
        layout.addWidget(self.detail)
        self.language = Choice(search=True)
        self.language.setMinimumWidth(260)
        for code, name in LANGUAGES.items():
            self.language.addItem(name, code)
        self.language.currentIndexChanged.connect(lambda _=0: self._show_state())
        layout.addLayout(row(caption("Language you speak", "2", wrap=False), self.language, stretch_at=2, spacing=16))
        self.note = caption("", "3")
        layout.addWidget(self.note)
        self.bar = SaveBar(self._save, self.discard)
        layout.addWidget(self.bar)
        self._select(app.settings.speech_language)

    def edited(self) -> bool:
        return self.language.currentData() != self.app.settings.speech_language

    def discard(self) -> None:
        self._select(self.app.settings.speech_language)
        self._show_state()

    def _select(self, code: str) -> None:
        self.language.blockSignals(True)  # showing the setting isn't changing it
        self.language.setCurrentIndex(max(0, self.language.findData(code)))
        self.language.blockSignals(False)
        self._shown = self.language.currentData()  # what it showed when last in step with the setting

    def refresh(self) -> None:
        app = self.app
        model = SPEECH_MODELS.get(app.speech_in_use())
        if app.loading_speech in SPEECH_MODELS:
            title, detail, state = (f"Switching to {SPEECH_MODELS[app.loading_speech].name}...",
                                    "Dictation goes on meanwhile.", "warn")
        elif model is None:
            title, detail, state = ("No speech model yet", "Download NVIDIA Parakeet (On this PC), or choose a cloud "
                                    "model.", "warn")
        else:
            title, state = f"In use: {model.name}", "ok"
            if model.where == "cloud":
                detail = f"In the cloud · {app.settings.speech_cloud_models.get(model.key) or CLOUD[model.key].models[0]}"
            elif model.where == "server":
                detail = f"Your own server · {app.settings.speech_server_model}"
            else:
                detail = "On this PC: your voice stays here"
        self.title.setText(title)
        self.detail.setText(detail)
        self.lamp.set_state(state)
        self.note.setText(f"{model.name} hears English; the language is for Whisper, the cloud models and your own "
                          "server." if model is not None and not model.language_choice else "")
        self.note.setVisible(bool(self.note.text()))
        if self.language.currentData() == self._shown:
            self._select(app.settings.speech_language)  # a language saved meanwhile shows; one being chosen stays
        self._show_state()

    def _show_state(self) -> None:
        self.bar.show_state(self.edited())

    def _save(self) -> None:
        self.app.set_speech_language(self.language.currentData())
        self._shown = self.language.currentData()
        self._show_state()
        self.bar.saved("Saved. Active from the next dictation.")


# How Rflow hears you, in two groups; the cloud group's tiles, in this order: a cloud speech model (its SPEECH_MODELS
# key), its name on the tile, and the icon that stands in for its logo until there is one. Only the providers offered
# (sst.gateway.OFFERED): Gemini and your own server for now.
SPEECH_GROUPS = [("local", "On this PC"), ("cloud", "Cloud")]
CLOUD_TILES = [tile for tile in [("openai", "OpenAI", "cloud"), ("groq", "Groq", "cloud"), ("gemini", "Gemini", "cloud"),
                                 ("server", "Your own server", "models")] if offered(tile[0])]


class _CloudTile(Card):
    """A cloud speech model as a tile: room for the provider's logo, its name, and its state (in use, its key saved,
    what it needs). A click, Space or Enter chooses it: raised, or pressed in with an Iris edge once chosen."""

    clicked = Signal()

    def __init__(self, key: str, name: str, icon: str):
        super().__init__("tile")
        self.key, self.icon, self.chosen = key, icon, False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName(name)
        self.setMinimumWidth(100)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)
        self.logo = QLabel()  # the provider's logo goes here, 28 px square; an icon stands in until then
        self.logo.setFixedSize(28, 28)
        self.logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        surface(self.logo, "well", 8)
        layout.addWidget(self.logo)
        layout.addSpacing(2)
        self.title = ElidedText(name, "rowtitle", None)
        layout.addWidget(self.title)
        self.state_row, self.lamp, self.state = lamp_row("ok", "", stretch=True)
        layout.addWidget(self.state_row)

    def set_chosen(self, chosen: bool) -> None:
        self.chosen = chosen
        self.set_kind("chosen" if chosen else "tile")
        self._show_logo()

    def set_state(self, words: str, lamp: str = "") -> None:
        self.state.setText(words)
        self.lamp.setVisible(bool(lamp))
        if lamp:
            self.lamp.set_state(lamp)
        self._show_logo()

    def _show_logo(self) -> None:
        colour = tok("iris" if self.chosen else "text2").name()
        self.logo.setPixmap(theme.icon_pixmap(self.icon, colour, 16, self.devicePixelRatioF()))

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.clicked.emit()
            return
        super().keyPressEvent(event)


def _show_only(stack: QStackedWidget, index: int) -> None:
    """Show one page of a stack, and let the stack be as tall as that page (not as its tallest)."""
    for i in range(stack.count()):
        stack.widget(i).setSizePolicy(QSizePolicy.Policy.Preferred,
                                      QSizePolicy.Policy.Preferred if i == index else QSizePolicy.Policy.Ignored)
    stack.setCurrentIndex(index)
    stack.updateGeometry()


class SpeechPage(Page):
    """Which model turns the voice into text, a building block of its own, chosen apart from the AI connection. Two
    groups: On this PC (Parakeet and Whisper, with their download, size and Scan this PC), and Cloud: a tile for each
    provider offered and your own server (CLOUD_TILES: Gemini and your own server for now), each opening its key, its
    model, a Test and Use."""

    def __init__(self, app, go_to=None):
        go_to = go_to or (lambda page: None)
        super().__init__("How Rflow hears you", "The model that turns your voice into text. Any speech model works with "
                                                "any AI connection, and your words help every model.",
                         back=back_button("AI & models", go_to, "models"))
        self.app = app
        self.in_use = _InUseCard(self, app)
        self.add(self.in_use.frame)
        self.tabs = Segmented(SPEECH_GROUPS)
        self.where: dict[str, QPushButton] = self.tabs.buttons
        self.tabs.changed.connect(self.show_where)
        self.add(self.tabs)
        self.models: dict[str, _ModelCard | _CloudCard | _ServerCard] = {}
        self.tiles: dict[str, _CloudTile] = {}
        self.tile = ""  # the cloud tile chosen ("" = none yet)
        self.groups = QStackedWidget()
        self.groups.addWidget(self._local_group())
        self.groups.addWidget(self._cloud_group())
        self.add(self.groups)
        self.body.addStretch()
        chosen = SPEECH_MODELS.get(setups.chosen_speech(app.settings, app.gateway), SPEECH_MODELS[DEFAULT_MODEL])
        self.show_where("local" if chosen.where == "local" else chosen.key)

    def _group(self, words: str) -> tuple[QWidget, QVBoxLayout]:
        group = Host()
        layout = QVBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        layout.addWidget(label(words, tone="2"))
        return group, layout

    def _local_group(self) -> QWidget:
        group, layout = self._group("Your voice stays on this PC: private, free, and it works offline. Each model is "
                                    "downloaded once.")
        for model in (m for m in SPEECH_MODELS.values() if m.where == "local"):
            self.models[model.key] = _ModelCard(self.app, model)
            layout.addWidget(self.models[model.key].frame)
        self.scan = _ScanCard(self.app)
        layout.addWidget(self.scan.frame)
        layout.addStretch()
        return group

    def _cloud_group(self) -> QWidget:
        group, layout = self._group("The provider turns your voice into text on its servers, with your API key: nothing "
                                    "to download, quick on any PC. Your voice goes to the provider each time you dictate.")
        tiles = QHBoxLayout()
        tiles.setSpacing(12)
        for key, name, icon in CLOUD_TILES:
            self.tiles[key] = _CloudTile(key, name, icon)
            self.tiles[key].clicked.connect(lambda k=key: self.show_where(k))
            tiles.addWidget(self.tiles[key], 1)
        layout.addLayout(tiles)
        self.pick = caption("Choose one: then its key, the model, and Use.", "3")
        layout.addWidget(self.pick)
        for key, _, _ in CLOUD_TILES:  # the chosen tile's card, under the tiles (a card for each, kept while hidden)
            model = SPEECH_MODELS[key]
            panel = _ServerCard(self, self.app, model) if model.where == "server" else _CloudCard(self, self.app, model)
            self.models[key] = panel
            panel.frame.hide()
            layout.addWidget(panel.frame)
        layout.addStretch()
        return group

    def show_where(self, where: str) -> None:
        """A group ("local", "cloud"), or a cloud tile by its model ("gemini", "server")."""
        group = "local" if where == "local" else "cloud"
        self.where[group].setChecked(True)
        _show_only(self.groups, [key for key, _ in SPEECH_GROUPS].index(group))
        if where in self.tiles:
            self.tile = where
        for key, tile in self.tiles.items():
            tile.set_chosen(key == self.tile)
            self.models[key].frame.setVisible(key == self.tile)
        self.pick.setVisible(not self.tile)
        self.refresh()

    def refresh(self) -> None:
        self.in_use.refresh()
        for model_card in self.models.values():
            model_card.refresh()
        self.scan.refresh()
        app = self.app
        for key, tile in self.tiles.items():
            if key == app.loading_speech:
                tile.set_state("Loading", "warn")
            elif key == app.speech_in_use():
                tile.set_state("In use", "ok")
            elif key == "server":  # its own address, or your own server's in Your API keys to start from
                server = app.gateway.speech_server()[0] or app.gateway.entries().get("vllm", ("", ""))[0]
                tile.set_state("Address saved" if server else "Needs an address")
            else:
                tile.set_state("Key saved" if app.gateway.key_for(key) else "Needs a key")

    def _editable(self) -> list:
        return [self.in_use, *(c for c in self.models.values() if hasattr(c, "edited"))]

    def unsaved(self) -> bool:
        return any(part.edited() for part in self._editable())

    def discard_changes(self) -> None:
        for part in self._editable():
            if part.edited():
                part.discard()

    def confirm(self, question: str) -> bool:
        return QMessageBox.question(self, APP_NAME, question) == QMessageBox.StandardButton.Yes


# ---------------------------------------------------------------- AI connection (the provider, key and models)

def _model_box(hint: str) -> Choice:
    box = fit_to_width(Choice(editable=True))  # pick from the loaded list (typing filters it), or type any model name
    box.lineEdit().setPlaceholderText(hint)
    return box


class CleanupPage(Page):
    """The AI connection: the provider, its key (the one in Your API keys), the model and a backup model. The AI cleanup
    uses it, and so do Text Transform and Translate."""

    def __init__(self, app, go_to=None):
        go_to = go_to or (lambda page: None)
        super().__init__("AI connection", "An AI model adds punctuation, removes filler words and spells your words "
                                          "right. Only the finished text goes to the provider.",
                         back=back_button("AI & models", go_to, "models"))
        self.app = app
        frame, layout = card(14, (20, 18, 20, 20))
        self.cleanup_on = Toggle("Clean up dictation")
        self.cleanup_on.setToolTip("If the model fails, the backup model is used; if the provider can't help in time, "
                                   "the text is typed as heard.")
        switch, _, _ = setting_row("Clean up dictation", "Punctuation, no filler words, your names spelled right.",
                                   self.cleanup_on)
        switch.layout().setContentsMargins(0, 0, 0, 4)
        layout.addWidget(switch)
        layout.addWidget(divider())
        self.form = _form()
        self.form.setVerticalSpacing(12)
        self.provider = fit_to_width(Choice(), 16)
        for provider in PROVIDERS.values():
            if offered(provider.key):  # Gemini, Ollama and your own server for now (sst.gateway.OFFERED)
                self.provider.addItem(provider.name, provider.key)
        self.form.addRow(caption("Provider", "2", wrap=False), self.provider)
        self.gateway_url = QLineEdit()
        field(self.gateway_url)
        self.form.addRow(caption("Address", "2", wrap=False), self.gateway_url)
        self.api_key = KeyField()
        self.load_button = button("Load models", self._load_models, size="sm")
        key_row = row(self.api_key, self.load_button)
        key_row.setStretch(0, 1)
        self.form.addRow(caption("API key", "2", wrap=False), key_row)
        self.key_link = button("Get a key", self._open_key_page, link=True, size="sm", icon="external", icon_after=True)
        self.key_note = ElidedText("", "caption", "3")
        self.key_row = QHBoxLayout()
        self.key_row.setSpacing(8)
        self.key_row.addWidget(self.key_link)
        self.key_row.addWidget(self.key_note, 1)
        self.form.addRow("", self.key_row)
        self.model = _model_box("")
        self.test_button = button("Test", self._test, size="sm")
        model_row = row(self.model, self.test_button)
        model_row.setStretch(0, 1)
        self.form.addRow(caption("Model", "2", wrap=False), model_row)
        self.cost = CostLine()  # what the model costs a month: amber or red, and why, for the dear ones
        self.form.addRow("", self.cost)
        self.fallback = _model_box("optional: used if the model fails")
        self.form.addRow(caption("Backup model", "2", wrap=False), self.fallback)
        for box in (self.model, self.fallback):  # each model's cost a month, in the lists
            cost_list(box, lambda m: costs.cleanup_cost(self._provider, m))
        layout.addLayout(self.form)
        self.test_result = caption("", "2")
        layout.addWidget(self.test_result)
        self.bar = SaveBar(self._save, self.discard_changes)
        layout.addWidget(self.bar)
        self.add(frame)
        tools = QHBoxLayout()
        tools.setSpacing(8)
        tools.addWidget(ElidedText("This connection also powers Text Transform and Translate.", "caption", "2"), 1)
        for title, key in (("Text Transform", "transform"), ("Translate", "translate")):
            tools.addWidget(button(title, lambda _=False, k=key: go_to(k), link=True, size="sm", icon="chevron-right",
                                   icon_after=True))
        self.add(tools)
        self.body.addStretch()
        self.discard_changes()  # the fields as saved
        self.provider.currentIndexChanged.connect(self._provider_changed)
        for changed in (self.cleanup_on.toggled, self.gateway_url.textChanged, self.api_key.textChanged,
                        self.model.currentTextChanged, self.fallback.currentTextChanged):
            changed.connect(lambda *_: self._show_state())

    def discard_changes(self) -> None:
        """The fields as saved: when the page is built, and Cancel."""
        settings, gateway = self.app.settings, self.app.gateway
        # The (address, key, model, backup model) of each provider left on this page, so switching back loses nothing.
        # The others come from app.gateway, where Your API keys and the speech page may also have saved a key.
        self._memory: dict[str, tuple[str, str, str, str]] = {}
        # Nothing chosen yet: start with the first provider in the list rather than an empty custom server. A provider
        # no longer offered shows as that one too, with no model: its settings stay as saved until Save.
        if self._moved():
            log.info("AI cleanup's provider %s is no longer offered: AI connection shows %s", gateway.chosen,
                     self.provider.itemData(0))
        self._provider = gateway.chosen if gateway.chosen and not self._moved() else self.provider.itemData(0)
        self.provider.blockSignals(True)
        self.provider.setCurrentIndex(self.provider.findData(self._provider))
        self.provider.blockSignals(False)
        self.cleanup_on.blockSignals(True)
        self.cleanup_on.setChecked(settings.cleanup)
        self.cleanup_on.blockSignals(False)
        self.gateway_url.setText(gateway.base_url if gateway.chosen == self._provider else self._fresh(self._provider)[0])
        self.api_key.show_saved(gateway.key_for(self._provider))
        for box, value in zip((self.model, self.fallback), self._saved_models(), strict=True):
            box.clear()
            if value:
                box.addItem(value)
            box.setCurrentText(value)
        self.test_result.setText("")
        self._show_provider()
        self._saved = self._fields()
        self._show_state()

    def _moved(self) -> bool:
        """AI cleanup's provider is one Rflow no longer offers (sst.gateway.OFFERED): this page shows the first offered."""
        chosen = self.app.gateway.chosen
        return bool(chosen) and not offered(chosen)

    def _saved_models(self) -> tuple[str, str]:
        """The model and the backup model as saved, as this page shows them (none for a provider no longer offered)."""
        s = self.app.settings
        return ("", "") if self._moved() else (s.cleanup_model, s.cleanup_fallback)

    def _fresh(self, key: str) -> tuple[str, str, str, str]:
        """A provider's fields before any change here: its saved address and key."""
        url, secret = self.app.gateway.entries().get(key, ("", ""))
        return url or (PROVIDERS[key].url if PROVIDERS[key].own_server else ""), secret, "", ""

    def _fields(self) -> tuple:
        """What Save would change: the switch, the provider and its fields, and the address and key typed for another
        provider (their models aren't kept)."""
        others = tuple(sorted((key, value[:2]) for key, value in self._memory.items()
                              if key != self._provider and value[:2] != self._fresh(key)[:2]))
        return (self.cleanup_on.isChecked(), self._provider, self.gateway_url.text().strip(), self.api_key.text().strip(),
                self.model.currentText().strip(), self.fallback.currentText().strip(), others)

    def unsaved(self) -> bool:
        return self._fields() != self._saved

    def _show_state(self) -> None:
        self.bar.show_state(self.unsaved())
        model = self.model.currentText().strip()
        self.cost.show_estimate(costs.cleanup_cost(self._provider, model) if model else None)

    def _show_provider(self) -> None:
        """What the chosen provider needs: a key (cloud), an address (own server), or both."""
        p = PROVIDERS[self._provider]
        self.form.setRowVisible(self.gateway_url, p.own_server)
        self.gateway_url.setPlaceholderText(p.url or "e.g. http://localhost:8000/v1, or your company's AI gateway")
        self.api_key.setPlaceholderText("Paste your key" if p.needs_key else "Only if your server needs one")
        self.form.setRowVisible(self.key_row, bool(p.key_page))
        self.key_link.setVisible(bool(p.key_page))
        self.key_note.setText(f"from {p.name}, kept in Your API keys" if p.key_page else "")
        self.model.lineEdit().setPlaceholderText(p.hint or "choose after Load models, or type a model name")

    def _provider_changed(self) -> None:
        new = self.provider.currentData()
        self._memory[self._provider] = (self.gateway_url.text().strip(), self.api_key.text().strip(),
                                        self.model.currentText().strip(), self.fallback.currentText().strip())
        url, key, model, fallback = self._memory.get(new, self._fresh(new))
        self._provider = new
        self.gateway_url.setText(url)
        saved_key = self._fresh(new)[1]
        self.api_key.show_saved(saved_key)
        if key != saved_key:
            self.api_key.setText(key)  # typed here before switching away: still a change, not the saved key
        for box, value in ((self.model, model), (self.fallback, fallback)):
            box.clear()
            box.setCurrentText(value)
        self.test_result.setText("")
        self._show_provider()
        self._show_state()

    def _open_key_page(self) -> None:
        shell.open_url(QUrl(PROVIDERS[self._provider].key_page))

    def confirm(self, question: str) -> bool:
        return QMessageBox.question(self, APP_NAME, question) == QMessageBox.StandardButton.Yes

    def refresh(self) -> None:
        clean = not self.unsaved()
        if clean and (self.app.settings.cleanup != self.cleanup_on.isChecked()
                      or self._saved_models()[0] != self.model.currentText().strip()):
            self.discard_changes()  # changed elsewhere (the switch on AI & models, the welcome): show what's saved
            return
        saved = self.app.gateway.key_for(self._provider)
        if not self.api_key.edited() and saved != self.api_key.saved:
            self.api_key.show_saved(saved)  # changed in Your API keys or on the speech page; a key being typed stays
        if clean:
            self._saved = self._fields()  # a key saved elsewhere isn't a change made here
        self._show_state()

    def result(self) -> tuple[bool, str, str, GatewayConfig]:
        p = PROVIDERS[self._provider]
        entries = self.app.gateway.entries()
        entries.update({key: (url, secret) for key, (url, secret, _, _) in self._memory.items()})
        others = {key: (url, secret) for key, (url, secret) in entries.items() if key != p.key and (url or secret)}
        gateway = GatewayConfig(self.gateway_url.text().strip() if p.own_server else "", self.api_key.text().strip(),
                                p.key, others)
        return self.cleanup_on.isChecked(), self.model.currentText().strip(), self.fallback.currentText().strip(), gateway

    def _save(self) -> None:
        on, model, fallback, gateway = self.result()
        chosen = self.app.gateway.chosen
        if not model and chosen and self._provider != chosen and self.app.settings.cleanup_model:
            # The user testing's M-04: switching the provider only to save its key left the cleanup with no model.
            self.test_result.setText(f"Choose a {provider_name(self._provider)} model first (Load models), or Cancel "
                                     f"to keep {provider_name(chosen)}. To only add a key, use Your API keys on AI & "
                                     "models: the cleanup stays as it is.")
            return
        # An expensive model asks once: when it's first saved, not again with every change after
        dear = expensive(model, costs.cleanup_cost(self._provider, model)) if model and (
            model, self._provider) != (self.app.settings.cleanup_model, chosen) else ""
        if dear and not self.confirm(f"{dear}\n\nUse it anyway?"):
            return
        self.app.save_cleanup(on, model, fallback, gateway)
        self.api_key.show_saved(gateway.api_key)
        self._saved = self._fields()
        self._show_state()
        self.bar.saved("Saved. Active from the next dictation." if on and model else "Saved. AI cleanup is off.")

    def _busy(self, busy: bool, message: str = "") -> None:
        self.load_button.setEnabled(not busy)
        self.test_button.setEnabled(not busy)
        if message:
            self.test_result.setText(message)

    def _load_models(self) -> None:
        gateway = self.result()[3]
        if not gateway.address:
            self.test_result.setText("Enter your server's address first.")
            return
        self._busy(True, "Loading models...")

        def done(models, error) -> None:
            self._busy(False)
            if error:
                self.test_result.setText(f"Failed: {error}")
                return
            for box in (self.model, self.fallback):
                current = box.currentText()
                box.clear()
                if box is self.fallback:
                    box.addItem("")  # no backup model
                box.addItems(models)
                box.setCurrentText(current)
            self.test_result.setText(f"Loaded {len(models)} models: choose one, then Test." if models
                                     else "The provider lists no models; type a model name.")
        run_in_background(self, lambda: Polisher(gateway, "").models(), done)

    def _test(self) -> None:
        on, model, _, gateway = self.result()
        if not model:
            self.test_result.setText("Choose a model first (Load models).")
            return
        self._busy(True, "Testing...")
        words = self.app.settings.vocabulary

        def done(answer, error) -> None:
            self._busy(False)
            self.test_result.setText(f"Failed: {error}" if error else f"OK: {answer}")
        run_in_background(self, lambda: Polisher(gateway, model, words).check(), done)


# ---------------------------------------------------------------- Your API keys (the top of AI & models)

def provider_name(key: str) -> str:
    return PROVIDERS[key].name.split(" (")[0] if key in PROVIDERS else key


# A key each, for the providers offered (sst.gateway.OFFERED): Gemini for now. Keys saved for the others stay in the file.
KEY_PROVIDERS = [key for key, p in PROVIDERS.items() if p.needs_key and offered(key)]
SERVERS = ("vllm", SPEECH_SERVER)  # your own server: AI cleanup's and the speech model's (mostly the same one)


def _uses(app, name: str) -> list[str]:
    """What uses a provider's key (or a server) now: AI cleanup, speech, live translation."""
    s = app.settings
    uses = []
    if name == app.gateway.chosen and s.cleanup_model:  # "vllm": AI cleanup's own server
        uses.append("AI cleanup")
    if setups.chosen_speech(s, app.gateway) == ("server" if name == SPEECH_SERVER else name):
        uses.append("speech")
    if name == "gemini":
        uses.append("live translation")  # Gemini Live Translate, whenever it's started
    return uses


def _and(words: list[str]) -> str:
    return ", ".join(words[:-1]) + " and " + words[-1] if len(words) > 1 else "".join(words)


class _KeyRow(QWidget):
    """One line of Your API keys: the provider, what uses its key, the key masked (your own server: its address), and
    Edit. Edit opens the key (Show, Paste), Get a key, Save, Cancel and Remove beneath it."""

    def __init__(self, keys, name: str):
        super().__init__()
        self.keys, self.app, self.name = keys, keys.app, name
        self.server = name in SERVERS
        self.names: tuple[str, ...] = (name,)  # what Save changes: your own server may be AI cleanup's and speech's
        self.saved = ("", "")  # (address, key) as saved
        self.used: list[str] = []
        p = PROVIDERS.get(name)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 10, 0, 10)
        layout.setSpacing(10)
        line = QHBoxLayout()
        line.setSpacing(16)
        words = QVBoxLayout()
        words.setSpacing(2)
        self.title = ElidedText("Your own server" if self.server else provider_name(name), "rowtitle", None)
        self.uses = ElidedText("", "caption", "3")
        words.addWidget(self.title)
        words.addWidget(self.uses)
        line.addLayout(words, 3)
        self.value = ElidedText("", None, "2")
        self.value.setFont(font(13, 500, mono=True))
        line.addWidget(self.value, 2)
        self.edit = button("Edit", lambda _=False: self.open(), size="sm", icon="edit")
        self.edit.setAccessibleName(f"Edit {self.title.text()}")
        line.addWidget(self.edit, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(line)

        self.editor = QWidget()
        form = QVBoxLayout(self.editor)
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        self.address = QLineEdit()
        self.address.setPlaceholderText("Your server's address, e.g. http://localhost:8000/v1")
        field(self.address)
        if not self.server:  # hide only: shown before it has a parent, it would flash up as its own window
            self.address.hide()
        form.addWidget(self.address)
        self.key = KeyField("", "Your server's key, only if it needs one" if self.server else "Paste your key")
        self.get_key = button("Get a free key" if name == "gemini" else "Get a key",
                              lambda _=False: shell.open_url(QUrl(p.key_page)), link=True, size="sm",
                              icon="external", icon_after=True)
        if not (p and p.key_page) or self.server:
            self.get_key.hide()
        key_line = row(self.key, self.get_key)
        key_line.setStretch(0, 1)
        form.addLayout(key_line)
        self.note = caption("", "3")
        form.addWidget(self.note)
        self.bar = SaveBar(self.save, self.close_editor)
        self.remove = button("Remove", lambda _=False: self.remove_key(), kind="danger", size="sm", icon="trash")
        self.bar.layout().addWidget(self.remove)
        form.addWidget(self.bar)
        layout.addWidget(self.editor)
        self.editor.hide()
        for changed in (self.address.textChanged, self.key.textChanged):
            changed.connect(lambda *_: self._show_state())

    @property
    def editing(self) -> bool:
        return not self.editor.isHidden()

    def label(self) -> str:
        return "your own server" if self.server else f"your {provider_name(self.name)} key"

    def fields(self) -> tuple[str, str]:
        return (self.address.text().strip() if self.server else ""), self.key.text().strip()

    def edited(self) -> bool:
        return self.editing and self.fields() != self.saved

    def show_saved(self, saved: tuple[str, str], names: tuple[str, ...], used: list[str], title: str) -> None:
        """What is saved now (it may have changed on another page); a line being edited keeps what is typed."""
        self.names, self.used = names, used
        self.title.setText(title)
        self.uses.setText(f"Used for {_and(used)}" if used else "Not used yet")
        address, key = saved
        self.value.setText(" · ".join(x for x in (address, _masked(key)) if x))
        if not self.editing:
            self.saved = saved
        self.setVisible(any(saved) or self.editing)

    def open(self) -> None:
        """Edit (or Add a key): the key field, Get a key, Save, Cancel, and Remove when there is one to remove."""
        if not self.editing:
            self.address.setText(self.saved[0])
            self.key.show_saved(self.saved[1])
        self.editor.show()
        self.edit.hide()
        self.remove.setVisible(any(self.saved))
        self.note.setText(self.keys.stays(self))
        self.show()
        if self.server and not self.saved[0]:
            self.address.setFocus()
        else:
            self.key.start_editing()
            self.key.field.setFocus()
        self._show_state()
        self.keys.changed()

    def close_editor(self) -> None:
        """Cancel, or after Save: back to the line as saved."""
        self.editor.hide()
        self.edit.show()
        self.key.show_saved(self.saved[1])
        self.address.setText(self.saved[0])
        self.bar.show_state(False)
        self.setVisible(any(self.saved))
        self.keys.changed()

    def _show_state(self) -> None:
        edited = self.edited()
        self.bar.show_state(edited, edited and bool(self.fields()[0] if self.server else self.fields()[1]))
        self.bar.cancel.setVisible(self.editing)  # also with nothing typed: a line opened by mistake closes again

    def save(self) -> None:
        address, key = self.fields()
        if self.server:
            self.app.save_server(address, key, self.names)
        else:
            self.app.save_key(self.name, key)
        added = not any(self.saved)
        self.saved = (address, key)
        self.close_editor()
        what = self.label()
        self.keys.saved(f"{what[0].upper()}{what[1:]}: {'added' if added else 'saved'}. {self.keys.stays(self)}")

    def remove_key(self) -> None:
        what = self.label()
        question = f"Remove {what}?"
        if self.used:
            after = {"AI cleanup": "AI cleanup stops until it has one again",
                     "speech": "speech goes back to Parakeet on this PC",
                     "live translation": "live translation can't start without it"}
            question += f"\n\nIt's used for {_and(self.used)}: " + _and([after[use] for use in self.used]) + "."
        if not self.keys.page.confirm(question):
            return
        if self.server:
            self.app.save_server("", "", self.names)
        else:
            self.app.save_key(self.name, "")
        self.saved = ("", "")
        self.close_editor()
        self.keys.saved(f"{what[0].upper()}{what[1:]}: removed.")


class _KeysCard:
    """Your API keys, at the top of AI & models: a line for each provider offered that has a key (KEY_PROVIDERS: Google
    Gemini for now) and for your own server, masked, each with Edit; "Add a key" for the others. Speech, AI cleanup and
    live translation all read their keys from here, and saving one never changes which provider or model they use (the
    user testing's M-04: a Gemini key added for live translation switched the cleanup to Gemini, with no model)."""

    def __init__(self, page, app):
        self.page, self.app = page, app
        self.frame, layout = card(4, (20, 18, 20, 16))
        layout.addWidget(label("Your API keys", "heading", wrap=False))
        layout.addWidget(caption("Encrypted on this PC. Speech, AI cleanup and live translation use them; adding or "
                                 "changing a key changes nothing else.", "2"))
        layout.addSpacing(4)
        self.rows: dict[str, _KeyRow] = {}
        for name in (*KEY_PROVIDERS, *SERVERS):
            self.rows[name] = _KeyRow(self, name)
            layout.addWidget(self.rows[name])
        self.empty = caption("No keys yet. A provider's key lets Rflow use its models: for AI cleanup, speech in the "
                             "cloud, or live translation.", "3")
        layout.addWidget(self.empty)
        self.adding = QWidget()
        flow = FlowLayout(8)
        self.adding.setLayout(flow)
        add_label = caption("Add a key", "2", wrap=False)
        add_label.setFixedHeight(28)
        flow.addWidget(add_label)
        self.add_buttons: dict[str, Button] = {}
        for name in (*KEY_PROVIDERS, SERVERS[0]):
            title = "Your own server" if name in SERVERS else provider_name(name)
            self.add_buttons[name] = button(title, lambda _=False, n=name: self.rows[n].open(), kind="chip", size="sm",
                                            icon="plus")
            self.add_buttons[name].setAccessibleName(f"Add {title}")
            flow.addWidget(self.add_buttons[name])
        layout.addSpacing(6)
        layout.addWidget(self.adding)
        self.state = caption("", "ok")
        self.state.hide()  # after a save: what was saved, and what stayed as it was
        layout.addWidget(self.state)
        self.refresh()

    def refresh(self) -> None:
        app = self.app
        gateway = app.gateway
        for name in KEY_PROVIDERS:
            self.rows[name].show_saved(("", gateway.key_for(name)), (name,), _uses(app, name), provider_name(name))
        cleanup, speech = gateway.entries().get(SERVERS[0], ("", "")), gateway.speech_server()
        both = any(cleanup) and any(speech) and cleanup != speech  # two servers: a line each, named by what it's for
        if any(cleanup) and cleanup == speech:  # one server for both: one line, and Save changes both
            names, used = SERVERS, _uses(app, SERVERS[0]) + _uses(app, SERVERS[1])
            self.rows[SERVERS[0]].show_saved(cleanup, names, used, "Your own server")
            self.rows[SERVERS[1]].show_saved(("", ""), (SERVERS[1],), [], "Your own server")
        else:
            self.rows[SERVERS[0]].show_saved(cleanup, (SERVERS[0],), _uses(app, SERVERS[0]),
                                             "Your server for AI cleanup" if both else "Your own server")
            self.rows[SERVERS[1]].show_saved(speech, (SERVERS[1],), _uses(app, SERVERS[1]),
                                             "Your server for speech" if both else "Your own server")
        self.changed()

    def changed(self) -> None:
        """Which lines and "Add a key" buttons show."""
        for name, add in self.add_buttons.items():
            shown = self.rows[name].isVisibleTo(self.frame) or (name in SERVERS and any(
                self.rows[server].isVisibleTo(self.frame) for server in SERVERS))
            add.setVisible(not shown)
        self.empty.setVisible(not any(row_.isVisibleTo(self.frame) for row_ in self.rows.values()))
        self.adding.setVisible(any(not add.isHidden() for add in self.add_buttons.values()))
        self.adding.updateGeometry()

    def stays(self, row_: _KeyRow) -> str:
        """What a key saved on this line leaves as it is: the cleanup's provider, and the speech model."""
        s, gateway = self.app.settings, self.app.gateway
        kept = []
        if gateway.chosen and s.cleanup_model and gateway.chosen not in row_.names:
            kept.append(f"AI cleanup keeps {provider_name(gateway.chosen)}")
        speech = SPEECH_MODELS.get(self.app.speech_in_use())
        if speech is not None and speech.key not in (*row_.names, "server" if SPEECH_SERVER in row_.names else ""):
            kept.append(f"speech keeps {speech.name}")
        if row_.used:
            return f"Used for {_and(row_.used)} from now on" + (f"; {_and(kept)}." if kept else ".")
        return f"Nothing else changes: {_and(kept)}." if kept else "Nothing else changes."

    def saved(self, message: str) -> None:
        self.page.refresh()
        self.state.setText(message)
        self.state.show()

    def unsaved(self) -> bool:
        return any(row_.edited() for row_ in self.rows.values())

    def discard(self) -> None:
        for row_ in self.rows.values():
            if row_.editing:
                row_.close_editor()


# ---------------------------------------------------------------- Setups (the top of AI & models, and the first run)

SETUP_ICONS = {"recommended": "check", "fastest": "arrow-right", "multilingual": "words", "local": "laptop",
               "custom": "settings"}
SETUP_COLUMNS_WIDTH = 740  # a page narrower than this shows the setups two by two
KEY_NAMES = {"gemini": "Gemini", "groq": "Groq"}
LOCAL_CHOICES = [("parakeet", "Parakeet · English"), ("whisper-turbo", "Whisper · 99 languages")]


def _key_name(provider: str) -> str:
    return KEY_NAMES.get(provider, provider_name(provider))


def _speech_words(key: str, model: str = "") -> str:
    """A speech model in a few words: "Parakeet, on this PC", "Whisper large-v3 turbo, on Groq"."""
    speech = SPEECH_MODELS.get(key)
    if speech is None:
        return "No speech model yet"
    if speech.where == "local":
        return f"{speech.name.removeprefix('NVIDIA ').removeprefix('OpenAI ')}, on this PC"
    if speech.where == "server":
        return f"{model or 'A model'}, on your own server"
    return f"{setups.model_name(model or CLOUD[key].models[0])}, on {CLOUD[key].name.removeprefix('Google ')}"


def setup_uses(app, setup: setups.Setup, local: str = "parakeet") -> tuple[str, str]:
    """What a setup uses, in two short lines: (speech, AI). Custom: what is chosen now."""
    s = app.settings
    if setup.key == "custom":
        speech = setups.chosen_speech(s, app.gateway)
        model = s.speech_cloud_models.get(speech, "") if speech in CLOUD else s.speech_server_model
        chosen = app.gateway.chosen
        ai = (f"{setups.model_name(s.cleanup_model)}, on {provider_name(chosen).removeprefix('Google ')}"
              if s.cleanup and s.cleanup_model and chosen else "No AI cleanup")
        return _speech_words(speech, model), ai
    if setup.key == "local":
        return "Parakeet or Whisper, on this PC", "AI on this PC: coming soon"
    return (_speech_words(setup.speech, setups.speech_model_for(setup, s.speech_language)),
            f"{setups.model_name(setup.model)}, on {provider_name(setup.provider).removeprefix('Google ')}")


def setup_cost(app, setup: setups.Setup, local: str = "parakeet") -> costs.Estimate:
    if setup.key == "custom":
        return costs.total(*setups.in_use_cost(app.settings, app.gateway))
    return setups.cost(setup, app.settings.speech_language, local)


def _icon(name: str, tone: str = "text3", size: int = 14) -> QLabel:
    picture = QLabel()
    picture.setFixedSize(size, size)
    picture.setPixmap(theme.icon_pixmap(name, tok(tone).name(), size, 1.0))
    return picture


class _SetupTile(Card):
    """A setup as a tile: its name and whether it's in use, what it uses, the key it needs, and what it costs a month.
    A click, Space or Enter opens it under the tiles: pressed in with an Iris edge while open."""

    clicked = Signal()

    def __init__(self, setup: setups.Setup):
        super().__init__("tile")
        self.key, self.chosen = setup.key, False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName(setup.name)
        self.setMinimumWidth(150)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(5)
        top = QHBoxLayout()
        top.setSpacing(8)
        self.icon = QLabel()
        self.icon.setFixedSize(16, 16)
        self.icon_name = SETUP_ICONS[setup.key]
        self.title = ElidedText(setup.name, "rowtitle", None)
        self.state_row, self.lamp, self.state = lamp_row("ok", "", stretch=False)
        top.addWidget(self.icon, 0, Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(self.title, 1)
        top.addWidget(self.state_row, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(top)
        layout.addSpacing(2)
        self.speech, self.ai, self.needs = (ElidedText("", "caption", "2"), ElidedText("", "caption", "2"),
                                            ElidedText("", "caption", "3"))
        for icon, words in (("mic", self.speech), ("tools", self.ai), ("key", self.needs)):
            line = QHBoxLayout()
            line.setSpacing(8)
            line.addWidget(_icon(icon), 0, Qt.AlignmentFlag.AlignVCenter)
            line.addWidget(words, 1)
            layout.addLayout(line)
        cost = QHBoxLayout()
        cost.setSpacing(8)
        self.cost_lamp = Lamp("warn")
        self.cost = ElidedText("", "caption", "2")
        cost.addWidget(self.cost_lamp, 0, Qt.AlignmentFlag.AlignVCenter)
        cost.addWidget(self.cost, 1)
        layout.addSpacing(2)
        layout.addLayout(cost)
        layout.addStretch()
        self._show_icon()

    def set_chosen(self, chosen: bool) -> None:
        self.chosen = chosen
        self.set_kind("chosen" if chosen else "tile")
        self._show_icon()

    def show_setup(self, uses: tuple[str, str], needs: str, estimate: costs.Estimate | None, state: str = "",
                   lamp: str = "ok") -> None:
        """`estimate` None: Custom while another setup is in use (its cost shows once its parts are chosen)."""
        self.speech.setText(uses[0])
        self.ai.setText(uses[1])
        self.needs.setText(needs)
        self.cost.setText("Its cost shows as you choose" if estimate is None else estimate.words)
        tone = COST_TONES.get(estimate.tier) if estimate is not None else None
        set_tone(self.cost, tone or "2")
        self.cost_lamp.setVisible(bool(tone))
        if tone:
            self.cost_lamp.set_state(tone)
        self.state.setText(state)
        self.state_row.setVisible(bool(state))
        self.lamp.set_state(lamp)
        self._show_icon()

    def _show_icon(self) -> None:
        self.icon.setPixmap(theme.icon_pixmap(self.icon_name, tok("iris" if self.chosen else "text2").name(), 16,
                                              self.devicePixelRatioF()))

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Space, Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.clicked.emit()
            return
        super().keyPressEvent(event)


def _fact(icon: str) -> tuple[QWidget, QLabel, QLabel]:
    """A line of a setup's details: an icon and words."""
    widget = QWidget()
    line = QHBoxLayout(widget)
    line.setContentsMargins(0, 0, 0, 0)
    line.setSpacing(10)
    picture = _icon(icon, "text2", 16)
    words = label("", tone="2")
    line.addWidget(picture, 0, Qt.AlignmentFlag.AlignTop)
    line.addWidget(words, 1)
    return widget, picture, words


class _SetupsCard:
    """Setups, first on AI & models: ready-made pairs of a speech model and an AI, each the cheapest that does the job
    well (sst.setups), with what they use, the key they need and what they cost a month; the one in use is marked
    (Custom when the settings match none). A tile opens its details under the tiles, with Use: a missing key is asked
    for there and saved in Your API keys (nothing else changes until Use); a model not on this PC is downloaded first."""

    def __init__(self, page, app):
        self.page, self.app = page, app
        self.frame = QWidget()  # not a host: the page paints the tiles' and the card's soft depth, past this edge
        layout = QVBoxLayout(self.frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(label("Setups", "heading", wrap=False))
        layout.addWidget(caption("Not sure which models to use? Each setup pairs a speech model with an AI, the cheapest "
                                 f"that does the job well. Costs are for {costs.TYPICAL_WORDS}.", "2"))
        layout.addSpacing(2)
        self.grid = QGridLayout()
        self.grid.setSpacing(12)
        self.tiles: dict[str, _SetupTile] = {}
        for key in setups.ORDER:
            self.tiles[key] = _SetupTile(setups.SETUPS[key])
            self.tiles[key].clicked.connect(lambda k=key: self.select(k))
        layout.addLayout(self.grid)
        self.columns = 0
        self.set_columns(3)
        self.hint = caption("Choose one to see what it uses and to use it. Any part can be changed later, below.", "3")
        layout.addWidget(self.hint)

        self.panel, column = card(12, (20, 18, 20, 20))
        self.panel_title = ElidedText("", "heading", None)
        self.panel_status, self.panel_lamp, self.panel_state = lamp_row("ok", "", stretch=False)
        column.addLayout(_heading(self.panel_title, self.panel_status))
        self.blurb = label("", tone="2")
        column.addWidget(self.blurb)
        self.speech_fact, _, self.speech_words = _fact("mic")
        self.ai_fact, _, self.ai_words = _fact("tools")
        column.addWidget(self.speech_fact)
        self.local = Segmented(LOCAL_CHOICES)  # Local: Parakeet or Whisper
        self.local.changed.connect(self._local_chosen)
        self.local_choice = "parakeet"
        column.addWidget(self.local, 0, Qt.AlignmentFlag.AlignLeft)
        self.check_row = QWidget()
        check = QHBoxLayout(self.check_row)
        check.setContentsMargins(0, 0, 0, 0)
        check.setSpacing(10)
        self.check_lamp = Lamp("ok")
        self.check_words = caption("", "2")
        self.scan = button("Check this PC", lambda _=False: app.scan_computer(), size="sm")
        check.addWidget(self.check_lamp, 0, Qt.AlignmentFlag.AlignVCenter)
        check.addWidget(self.check_words, 1)
        check.addWidget(self.scan, 0, Qt.AlignmentFlag.AlignVCenter)
        column.addWidget(self.check_row)
        column.addWidget(self.ai_fact)
        self.links = QWidget()  # Custom: each part, one level down
        links = QHBoxLayout(self.links)
        links.setContentsMargins(0, 0, 0, 0)
        links.setSpacing(16)
        for title, page_key in (("Choose the speech model", "speech"), ("Choose the AI", "cleanup")):
            links.addWidget(button(title, lambda _=False, p=page_key: page.go_to(p), link=True, size="sm",
                                   icon="chevron-right", icon_after=True))
        links.addStretch()
        column.addWidget(self.links)
        self.key_area = QWidget()  # a missing key, asked for here and kept in Your API keys
        keys = QVBoxLayout(self.key_area)
        keys.setContentsMargins(0, 4, 0, 0)
        keys.setSpacing(8)
        self.key_title = caption("", "2")
        keys.addWidget(self.key_title)
        self.key = KeyField("", "Paste your key")
        self.get_key = button("Get a free key", self._open_key_page, link=True, size="sm", icon="external",
                              icon_after=True)
        key_line = row(self.key, self.get_key)
        key_line.setStretch(0, 1)
        keys.addLayout(key_line)
        column.addWidget(self.key_area)
        column.addWidget(divider())
        self.cost_row = QWidget()
        cost = QHBoxLayout(self.cost_row)
        cost.setContentsMargins(0, 0, 0, 0)
        cost.setSpacing(10)
        self.cost_lamp = Lamp("warn")
        self.cost_words = label("", tone="2")
        cost.addWidget(self.cost_lamp, 0, Qt.AlignmentFlag.AlignTop)
        cost.addWidget(self.cost_words, 1)
        column.addWidget(self.cost_row)
        self.warnings = caption("", "warn")
        column.addWidget(self.warnings)
        self.free_note = caption("", "3")
        column.addWidget(self.free_note)
        self.actions = QWidget()  # Use, and what it did
        actions = QHBoxLayout(self.actions)
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(12)
        self.use = button("Use", lambda _=False: self._use(), primary=True, size="sm")
        self.result = caption("", "2")
        actions.addWidget(self.use, 0, Qt.AlignmentFlag.AlignVCenter)
        actions.addWidget(self.result, 1)
        column.addWidget(self.actions)
        layout.addWidget(self.panel)
        self.panel.hide()
        self.selected = ""  # the tile whose details are open ("" = none)
        self.key.textChanged.connect(lambda _="": self._show_use())

    # -- the tiles

    def set_columns(self, columns: int) -> None:
        """Three tiles a row (Custom across under them), or two on a narrow page."""
        if columns == self.columns:
            return
        self.columns = columns
        for tile in self.tiles.values():
            self.grid.removeWidget(tile)
        # The setups offered in order (setups.ORDER: no Fastest while Groq isn't offered), and Custom after them: the
        # rest of the last line, or a line of its own
        keys = [key for key in setups.ORDER if key != "custom"]
        places = {key: (i // columns, i % columns, 1) for i, key in enumerate(keys)}
        line, column = divmod(len(keys), columns)
        places["custom"] = (line, column, columns - column)
        for key, (line, column, span) in places.items():
            self.grid.addWidget(self.tiles[key], line, column, 1, span)
        for column in range(3):
            self.grid.setColumnStretch(column, 1 if column < columns else 0)

    def in_use(self) -> list[str]:
        app = self.app
        pending = app.downloading[0] if app.downloading else ""
        return setups.in_use(app.settings, app.gateway, pending)

    def refresh(self) -> None:
        app, in_use = self.app, self.in_use()
        if self.selected != "local" and app.settings.speech_model in setups.LOCAL_MODELS:
            self.local_choice = app.settings.speech_model  # Local opens on the model on this PC in use
        for key, tile in self.tiles.items():
            setup = setups.for_app(app, key)  # what it would use now, for the language spoken
            local = self.local_choice if key == "local" else "parakeet"
            if setup.needs:
                needs = f"{_key_name(setup.needs)} key" + (": saved" if app.gateway.key_for(setup.needs) else
                                                           ", free to start")
            else:
                needs = "No key" if key == "local" else "Keys from Your API keys"
            state, lamp = "", "ok"
            if key in in_use:
                downloading = app.downloading
                if downloading and downloading[0] in SPEECH_MODELS:
                    done, total = downloading[1], downloading[2] or 1
                    state, lamp = f"Downloading {done * 100 // total}%", "warn"
                else:
                    state = "In use"
            if key == "custom" and "custom" not in in_use:  # what another setup uses isn't Custom's
                uses, estimate = ("Any speech model you choose", "Any AI you choose"), None
            else:
                uses, estimate = setup_uses(app, setup, local), setup_cost(app, setup, local)
            tile.show_setup(uses, needs, estimate, state, lamp)
        if self.selected:
            self._show_panel()

    def select(self, key: str) -> None:
        if key != self.selected:
            self.key.show_saved("")
            self.result.setText("")
        self.selected = key
        for name, tile in self.tiles.items():
            tile.set_chosen(name == key)
        self.hint.hide()
        self._show_panel()
        self.panel.show()
        QTimer.singleShot(0, lambda: self.page.ensureWidgetVisible(self.panel, 0, 24))

    def _local_chosen(self, key: str) -> None:
        self.local_choice = key
        self.refresh()

    # -- the details of the open one

    def _show_panel(self) -> None:
        app, setup, key = self.app, setups.for_app(self.app, self.selected), self.selected
        current = key in self.in_use()
        self.panel_title.setText(setup.name)
        self.panel_state.setText("In use" if current else "")
        self.panel_status.setVisible(current)
        self.blurb.setText(setup.blurb)
        local = key == "local"
        self.local.setVisible(local)
        self.local.set_current(self.local_choice)
        self.check_row.setVisible(local)
        self.links.setVisible(key == "custom")
        self.speech_fact.setVisible(not local)
        if local:
            self._show_check()
            self.ai_words.setText("AI cleanup on this PC is coming soon. Until then it's off: Rflow types what it hears, "
                                  "and Text Transform and Translate wait for an AI.")
        else:
            self.speech_words.setText(self._speech_fact(setup))
            self.ai_words.setText(self._ai_fact(setup))
        missing = setups.key_missing(setup, app.gateway)
        self.key_area.setVisible(bool(missing))
        if missing:
            self.key_title.setText(f"{setup.name} needs your {provider_name(missing)} key. It's kept in Your API "
                                   "keys, encrypted on this PC; nothing else changes until you press Use.")
            self.get_key.setText("Get a free key")
        self._show_cost(setup)
        self.free_note.setText(setup.free_key)
        self.free_note.setVisible(bool(setup.free_key))
        self.use.setText(f"Use {setup.name}")
        self.use.setVisible(key != "custom" and not current)
        self._show_use()
        self.actions.setVisible(not self.use.isHidden() or bool(self.result.text()))

    def _speech_fact(self, setup: setups.Setup) -> str:
        s = self.app.settings
        if setup.key == "custom":
            speech = SPEECH_MODELS.get(setups.chosen_speech(s, self.app.gateway))
            if speech is None:
                return "Speech: no model yet."
            if speech.where == "cloud":
                model = s.speech_cloud_models.get(speech.key) or CLOUD[speech.key].models[0]
                return f"Speech: {model}, at {speech.name}. Your voice goes there each time you dictate."
            if speech.where == "server":
                return f"Speech: {s.speech_server_model or 'a model'}, on your own server."
            return f"Speech: {speech.name}, on this PC. Your voice stays here."
        speech = SPEECH_MODELS[setup.speech]
        if speech.where == "local":
            return f"Speech: {speech.name}, on this PC. Your voice stays here. {speech.languages}."
        name, model = CLOUD[setup.speech].name, setups.speech_model_for(setup, s.speech_language)
        words = f"Speech: {setups.model_name(model)}, at {name}. Your voice goes to {name} each time you dictate."
        if setup.speech_model == "gemini-3.5-transcribe":
            words += (" Tamil goes to Gemini 3.5 Flash-Lite: Transcribe has no Tamil." if model != setup.speech_model
                      else " For Tamil, Rflow uses Gemini 3.5 Flash-Lite (Transcribe has no Tamil).")
        return words

    def _ai_fact(self, setup: setups.Setup) -> str:
        s, gateway = self.app.settings, self.app.gateway
        if setup.key == "custom":
            if not (s.cleanup and s.cleanup_model and gateway.chosen):
                return "AI cleanup: off."
            return f"AI cleanup: {s.cleanup_model}, at {provider_name(gateway.chosen)}. Only the text is sent."
        return (f"AI cleanup, Text Transform and Translate: {setups.model_name(setup.model)}, at "
                f"{provider_name(setup.provider)}. Only the text is sent, never your voice.")

    def _show_check(self) -> None:
        """Local: whether this PC can run the model chosen, from Scan this PC (sst.scan), in plain words."""
        app, key = self.app, self.local_choice
        level, words = setups.pc_check(app.last_scan, key)
        speech = SPEECH_MODELS[key]
        if app.scanning:
            state, words = "warn", f"Checking this PC: {app.scanning}"
        elif not level:
            state, words = "off", (f"Check this PC first: {speech.name} needs about {scan_memory(key)} of memory and "
                                   f"{speech.size.split(',')[0]} of disk.")
        else:
            state = {"recommended": "ok", "fast": "ok", "usable": "off", "slow": "warn", "no": "err"}[level]
        self.check_lamp.set_state(state)
        self.check_words.setText(words)
        self.scan.setText("Check again" if app.last_scan else "Check this PC")
        self.scan.setEnabled(not app.scanning)

    def _show_cost(self, setup: setups.Setup) -> None:
        app = self.app
        if setup.key == "custom":
            speech, ai = setups.in_use_cost(app.settings, app.gateway)
            total = costs.total(speech, ai)
            parts = f"speech {_part(speech)}, AI {_part(ai)}"
            warnings = [f"{what}: {part.warning}" for what, part in (("Speech", speech), ("AI", ai)) if part.warning]
        else:
            local = self.local_choice
            speech, ai = setups.speech_cost(setup, app.settings.speech_language, local), setups.cleanup_cost(setup)
            total = setups.cost(setup, app.settings.speech_language, local)
            parts = f"speech {_part(speech)}, AI {_part(ai)}"
            warnings = []
        if total.tier == costs.FREE:
            words = "Free: nothing is sent anywhere." if setup.key == "local" else "Free: everything runs on this PC."
        elif total.monthly is None:
            words = f"Price unknown: {parts}."
        else:
            words = f"{total.words} for {costs.TYPICAL_WORDS} ({parts})."
        tone = COST_TONES.get(total.tier)
        self.cost_words.setText(words)
        set_tone(self.cost_words, tone or "2")
        self.cost_lamp.setVisible(bool(tone))
        if tone:
            self.cost_lamp.set_state(tone)
        self.warnings.setText("\n".join(warnings))
        self.warnings.setVisible(bool(warnings))
        set_tone(self.warnings, "err" if any(p.tier == costs.RED for p in (speech, ai)) else "warn")

    def _show_use(self) -> None:
        if not self.selected:
            return
        setup = setups.SETUPS[self.selected]
        if setup.key == "local":  # after the check, and only if this PC can run the model
            level = setups.pc_check(self.app.last_scan, self.local_choice)[0]
            self.use.setEnabled(level not in ("", "no") and not self.app.loading_speech)
        else:
            self.use.setEnabled(not self.app.loading_speech)

    def _open_key_page(self) -> None:
        missing = setups.for_app(self.app, self.selected).needs if self.selected else ""
        if missing:
            shell.open_url(QUrl(PROVIDERS[missing].key_page))

    def _say(self, message: str, tone: str | None = "2") -> None:
        self.result.setText(message)
        set_tone(self.result, tone)
        self.actions.setVisible(not self.use.isHidden() or bool(message))

    def _use(self) -> None:
        """Use the open setup: its key first if it's missing (saved, nothing else changed), then its two parts."""
        app, setup = self.app, setups.for_app(self.app, self.selected)
        missing = setups.key_missing(setup, app.gateway)
        typed = self.key.text().strip()
        if missing and not typed:
            self._say(f"Paste your {provider_name(missing)} key first (Get a free key).", "warn")
            return
        speech = self.local_choice if setup.key == "local" else setup.speech
        model = SPEECH_MODELS[speech]
        if model.where == "cloud" and app.settings.speech_model != speech:
            name = CLOUD[speech].name
            if not self.page.confirm(f"Use {setup.name}?\n\nEach time you dictate, the recording of your voice is sent "
                                     f"to {name}, which turns it into text. " + _parakeet_takes_over(name)):
                return
        if missing:
            app.save_key(missing, typed)  # Your API keys: changes nothing else
        downloading = setups.apply(app, setup.key, self.local_choice)
        self.key.show_saved("")
        self.page.refresh()
        if downloading and app.downloading and app.downloading[0] != downloading:  # one download at a time
            self._say(f"{SPEECH_MODELS[downloading].name} can download once the other download has stopped: press Use "
                      "again in a moment.", "warn")
        elif downloading:
            fetched = SPEECH_MODELS[downloading]
            self._say(f"{setup.name} is set. {fetched.name} is downloading ({_size(fetched.download.size)}); it takes "
                      "over when it's here.", "ok")
        else:
            self._say(f"{setup.name} is in use from the next dictation.", "ok")

    def unsaved(self) -> bool:
        return bool(self.selected) and not self.key_area.isHidden() and bool(self.key.text().strip())

    def discard(self) -> None:
        self.key.show_saved("")


def _part(estimate: costs.Estimate) -> str:
    """A part of a setup's cost: "free", "$1.05", "price unknown"."""
    if estimate.tier == costs.FREE:
        return "free"
    return costs.money(estimate.monthly) if estimate.monthly is not None else "price unknown"


def scan_memory(key: str) -> str:
    """The memory a model on this PC needs in all (sst.scan: its own and Windows' with the other programs)."""
    return f"{MEMORY_GB.get(key, 1.0) + SYSTEM_GB:.1f} GB"


# ---------------------------------------------------------------- What Gemini costs (under Your API keys)

def dollars(amount: float) -> str:
    """$1.24, $13; a little that isn't nothing: "<$0.01"."""
    return "$0.00" if amount <= 0 else "<$0.01" if amount < 0.005 else costs.money(amount)


def nice_day(day: str) -> str:
    """"2026-10-06" as "6 Oct 2026"."""
    try:
        when = date.fromisoformat(day)
    except ValueError:
        return day
    return f"{when.day} {when:%b %Y}"


class _CostsCard:
    """What Gemini costs, an estimate (sst.usage): today and this month in dollars, the month by feature, a link to
    AI Studio's Billing page (the real figure) and the day the prices were read. Shown once there is a Gemini key or
    anything counted; refreshed when the page is shown and when a call is counted (MainWindow.usage_changed)."""

    def __init__(self, app):
        self.app = app
        self.frame, layout = card(10, (20, 18, 20, 18))
        self.billing = button("Billing in AI Studio", lambda _=False: open_link(usage.BILLING), link=True, size="sm",
                              icon="external", icon_after=True)
        layout.addLayout(row(label("What Gemini costs (estimate)", "heading", wrap=False), self.billing, stretch_at=1))
        totals = QHBoxLayout()
        totals.setSpacing(32)
        self.today, self.month = label("", "hero", wrap=False), label("", "hero", wrap=False)
        for title, value in (("Today", self.today), ("This month", self.month)):
            column = QVBoxLayout()
            column.setSpacing(2)
            column.addWidget(label(title.upper(), "caps", wrap=False))
            column.addWidget(value)
            totals.addLayout(column)
        totals.addStretch()
        layout.addLayout(totals)
        self.free = caption("", "ok")  # a free key: nothing is billed
        layout.addWidget(self.free)
        self.table, self.lines = card(0, (14, 6, 14, 6), kind="well")
        layout.addWidget(self.table)
        self.empty = caption("Nothing counted this month yet. Each call to Gemini is: speech, AI cleanup, Text Transform, "
                             "Translate and live translation.", "3")
        layout.addWidget(self.empty)
        self.saved = caption("", "ok")  # what the cost saver kept from being sent this month
        layout.addWidget(self.saved)
        self.note = caption("", "3")
        layout.addWidget(self.note)

    def refresh(self) -> None:
        ledger, key = getattr(self.app, "usage", None), self.app.gateway.key_for("gemini")
        summary = ledger.summary() if ledger is not None else usage.Summary()
        shown = bool(key) or summary.used
        self.frame.setVisible(shown)
        if not shown:
            return
        # TODO(owner): the key's tier can't be read from the key; Settings.gemini_free_tier waits for a way to set it.
        free = bool(getattr(self.app.settings, "gemini_free_tier", False))
        self.today.setText("$0.00" if free else dollars(summary.today))
        self.month.setText("$0.00" if free else dollars(summary.month))
        self.free.setVisible(free and summary.used)
        self.free.setText(f"$0 billed: a free key (would cost {dollars(summary.month)} this month).")
        clear(self.lines)
        for i, line in enumerate(summary.lines):
            if i:
                self.lines.addWidget(divider())
            self.lines.addWidget(self._line(line))
        self.table.setVisible(bool(summary.lines))
        self.empty.setVisible(not summary.lines)
        saved = getattr(summary, "saved", 0.0)
        self.saved.setText(f"Saved by the cost saver this month: {dollars(saved)}")
        self.saved.setVisible(saved > 0)
        note = (f"Estimated from what Rflow sent and got back, at Google's list prices as of {nice_day(costs.SEEN)}. "
                "Google's Billing page has the real figure.")
        if summary.unpriced:
            note += f" {summary.unpriced} call{'s' if summary.unpriced > 1 else ''} to a model Rflow has no price for " \
                    "aren't in it."
        self.note.setText(note)

    @staticmethod
    def _line(line: "usage.Line") -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(12)
        layout.addWidget(label(line.title, wrap=False), 1)
        amount = caption(line.amount, "2", wrap=False)
        layout.addWidget(amount)
        cost = label(dollars(line.cost), wrap=False)
        cost.setFont(font(13, 500, mono=True))
        cost.setMinimumWidth(72)
        cost.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(cost)
        return widget


# ---------------------------------------------------------------- AI & models (the overview)

PAIR_WIDTH = 760  # a page narrower than this shows How Rflow hears you above the AI connection, not beside it


class ModelsPage(Page):
    """AI & models: Setups, Your API keys, How Rflow hears you and the AI connection side by side, the microphone, and
    the cleanup switch. Change leads one level down (SpeechPage, CleanupPage)."""

    def __init__(self, app, go_to):
        super().__init__("AI & models", "How Rflow hears you, the AI that polishes what you say, and the keys they use.")
        self.app, self.go_to = app, go_to
        # Setups, first on the page: ready-made pairs of a speech model and an AI (Recommended, Multilingual, Local,
        # Custom), with what they cost
        self.setups = QWidget()
        self.setups_layout = QVBoxLayout(self.setups)
        self.setups_layout.setContentsMargins(0, 0, 0, 0)
        self.setups_layout.setSpacing(16)
        self.setup_card = _SetupsCard(self, app)
        self.setups_layout.addWidget(self.setup_card.frame)
        self.add(self.setups)

        self.keys = _KeysCard(self, app)
        self.add(self.keys.frame)
        self.costs = _CostsCard(app)
        self.add(self.costs.frame)

        self.pair = QBoxLayout(QBoxLayout.Direction.LeftToRight)
        self.pair.setSpacing(24)
        self.speech_card, column = card(14, (20, 20, 20, 20))
        self.speech_status, self.speech_lamp, self.speech_state = lamp_row("ok", "In use", stretch=False)
        column.addLayout(_heading(ElidedText("How Rflow hears you", "heading", None), self.speech_status))
        self.speech_name = ElidedText("", "rowtitle", None)
        self.speech_detail = ElidedText("", "caption", "2")
        self.speech_change = button("Change", lambda: go_to("speech"), size="sm")
        column.addLayout(self._named(self.speech_name, self.speech_detail, self.speech_change))
        column.addStretch()
        self.pair.addWidget(self.speech_card, 1)

        self.ai_card, column = card(14, (20, 20, 20, 20))
        self.ai_status, self.ai_lamp, self.ai_state = lamp_row("ok", "Connected", stretch=False)
        column.addLayout(_heading(ElidedText("AI connection", "heading", None), self.ai_status))
        self.ai_name = ElidedText("", "rowtitle", None)
        self.ai_model = ElidedText("", None, "2")
        self.ai_change = button("Change", lambda: go_to("cleanup"), size="sm")
        column.addLayout(self._named(self.ai_name, self.ai_model, self.ai_change))
        self.ai_divider = divider()
        column.addWidget(self.ai_divider)
        self.key_line = ElidedText("", "caption", "2")
        self.test_button = button("Test", self._test, size="sm")
        self.test_row = QWidget()
        test_line = QHBoxLayout(self.test_row)
        test_line.setContentsMargins(0, 0, 0, 0)
        test_line.setSpacing(12)
        test_line.addWidget(self.key_line, 1)
        test_line.addWidget(self.test_button)
        column.addWidget(self.test_row)
        self.test_note = caption("", "3")
        self.test_note.hide()
        column.addWidget(self.test_note)
        column.addStretch()
        self.pair.addWidget(self.ai_card, 1)
        self.add(self.pair)

        self.mic_card, column = card(10, (20, 18, 20, 20))
        column.addWidget(label("Microphone", "heading", wrap=False))
        column.addWidget(caption("The microphone Rflow listens to. Say something: the bars beside it light up.", "2"))
        column.addSpacing(2)
        self.microphone = MicrophoneBox(app.settings.microphone, app.microphones(), app.default_microphone(),
                                        source=lambda: (app.microphones(), app.default_microphone()))
        self.microphone.changed.connect(self._microphone_chosen)
        self.microphone.followed.connect(self._show_call_warning)
        column.addWidget(self.microphone)
        self.call_warning = caption("This is a Bluetooth headset's microphone. It records in call quality (like a "
                                    "phone), so Rflow gets more words wrong, and the headset plays sound in call "
                                    "quality while it's open. The laptop's own microphone is usually clearer.", "warn")
        column.addWidget(self.call_warning)
        self.add(self.mic_card)

        self.cleanup_card, column = card(0, (20, 4, 20, 4))
        self.cleanup = Toggle("Clean up dictation")
        self.cleanup.toggled.connect(self._cleanup_toggled)
        switch, _, self.cleanup_caption = setting_row(
            "Clean up dictation", "Punctuation, no filler words, your names spelled right. Only text is sent, never your "
            "voice.", self.cleanup)
        column.addWidget(switch)
        column.addWidget(divider())
        powers = QHBoxLayout()
        powers.setContentsMargins(0, 12, 0, 12)
        powers.setSpacing(12)
        self.sparkle = QLabel()
        powers.addWidget(self.sparkle)
        powers.addWidget(caption("This connection also powers Text Transform and Translate.", "2"), 1)
        for title, key in (("Text Transform", "transform"), ("Translate", "translate")):
            powers.addWidget(button(title, lambda _=False, k=key: go_to(k), link=True, size="sm", icon="chevron-right",
                                    icon_after=True))
        column.addLayout(powers)
        self.add(self.cleanup_card)

        self.advanced = advanced_row("Advanced", "Every speech model, the backup AI model, the Reading test",
                                     self._toggle_advanced)
        self.add(self.advanced)
        self.advanced_card, column = card(0, (20, 4, 20, 4))
        for i, (title, words, page) in enumerate([
                ("Every speech model", "Parakeet and Whisper on this PC, the cloud models, your own server, Scan this PC",
                 "speech"),
                ("AI provider and backup model", "Gemini, Ollama on this PC or your own server", "cleanup"),
                ("Reading test", "Read 30 sentences: how many words does Rflow get wrong?", "reading")]):
            if i:
                column.addWidget(divider())
            column.addWidget(link_row(title, words, lambda _=False, p=page: go_to(p)))
        self.advanced_card.hide()
        self.add(self.advanced_card)
        self.body.addStretch()
        self.call_warning.hide()  # until shown (refresh): asking means restarting PortAudio (refresh_devices)

    @staticmethod
    def _named(name: QLabel, detail: QLabel, change: QPushButton) -> QHBoxLayout:
        """A card's name and what it is, cut with … when narrow, and its Change button."""
        words = QVBoxLayout()
        words.setSpacing(2)
        words.addWidget(name)
        words.addWidget(detail)
        line = QHBoxLayout()
        line.setSpacing(12)
        line.addLayout(words, 1)
        line.addWidget(change, 0, Qt.AlignmentFlag.AlignVCenter)
        return line

    def resizeEvent(self, event):
        super().resizeEvent(event)
        side_by_side = self.viewport().width() >= PAIR_WIDTH
        direction = QBoxLayout.Direction.LeftToRight if side_by_side else QBoxLayout.Direction.TopToBottom
        if self.pair.direction() != direction:
            self.pair.setDirection(direction)
            self.pair.setSpacing(24 if side_by_side else self.body.spacing())
        self.setup_card.set_columns(3 if self.viewport().width() >= SETUP_COLUMNS_WIDTH else 2)

    def _toggle_advanced(self) -> None:
        open_ = self.advanced_card.isHidden()
        self.advanced_card.setVisible(open_)
        self.advanced.button.icon_name = "chevron-down" if open_ else "chevron-right"
        self.advanced.button.update()

    def _microphone_chosen(self, device: str) -> None:
        self.app.apply_settings(dataclasses.replace(self.app.settings, microphone=device))
        self._show_call_warning()

    def _show_call_warning(self) -> None:
        self.call_warning.setVisible(call_quality(self.microphone.device() or None))

    def _cleanup_toggled(self, on: bool) -> None:
        s = self.app.settings
        if on != s.cleanup:
            self.app.save_cleanup(on, s.cleanup_model, s.cleanup_fallback, self.app.gateway)

    def add_key(self, provider: str) -> None:
        """Open Your API keys at a provider's key, e.g. for live translation's "Add a Gemini key"."""
        line = self.keys.rows[provider]
        line.open()
        self.ensureWidgetVisible(line)
        QTimer.singleShot(0, lambda: self.ensureWidgetVisible(line, 0, 48))  # again once the page is laid out

    def unsaved(self) -> bool:
        return self.keys.unsaved() or self.setup_card.unsaved()

    def discard_changes(self) -> None:
        self.keys.discard()
        self.setup_card.discard()

    def confirm(self, question: str) -> bool:
        return QMessageBox.question(self, APP_NAME, question) == QMessageBox.StandardButton.Yes

    def refresh(self) -> None:
        app = self.app
        s = app.settings
        self.setup_card.refresh()
        self.keys.refresh()
        self.costs.refresh()
        model = SPEECH_MODELS.get(app.speech_in_use())
        loading, downloading = app.loading_speech, app.downloading
        fetching = bool(downloading) and downloading[0] == DEFAULT_MODEL and model is None
        if fetching:
            done, total = downloading[1], downloading[2] or 1
            self.speech_lamp.set_state("warn")
            self.speech_state.setText(f"Downloading {done * 100 // total}%")
            self.speech_name.setText("NVIDIA Parakeet")
            self.speech_detail.setText("On this PC · English · arriving")
        elif loading in SPEECH_MODELS:
            self.speech_lamp.set_state("warn")
            self.speech_state.setText("Loading")
            self.speech_name.setText(SPEECH_MODELS[loading].name)
            self.speech_detail.setText("Dictation goes on meanwhile")
        elif model is None:
            self.speech_lamp.set_state("warn")
            self.speech_state.setText("Not ready")
            self.speech_name.setText("No speech model yet")
            self.speech_detail.setText("Choose one: Parakeet on this PC, or a cloud model")
        else:
            self.speech_lamp.set_state("ok")
            self.speech_state.setText("In use")
            self.speech_name.setText(model.name)
            if model.where == "cloud":
                self.speech_detail.setText(f"In the cloud · {s.speech_cloud_models.get(model.key) or CLOUD[model.key].models[0]}")
            elif model.where == "server":
                self.speech_detail.setText(f"Your own server · {s.speech_server_model or 'no model chosen'}")
            else:
                verdict = self._verdict(model.key)
                self.speech_detail.setText(" · ".join(x for x in ("On this PC", model.languages, verdict) if x))
        self._show_call_warning()

        gateway = app.gateway
        connected = bool(gateway.address and s.cleanup_model)
        key = gateway.service.key
        p = PROVIDERS[key]
        self.ai_name.setText(provider_name(key) if connected or gateway.provider else "Not connected")
        self.ai_model.setFont(font(12, 500, mono=connected))
        self.ai_model.setText(s.cleanup_model if connected else "Add punctuation and drop filler words")
        self.ai_change.setText("Change" if connected else "Connect")
        self.ai_change.set_kind("key" if connected else "primary")
        if connected:
            self.ai_lamp.set_state("ok" if s.cleanup else "off")
            self.ai_state.setText("Connected" if s.cleanup else "Connected · cleanup off")
        else:
            self.ai_lamp.set_state("off")
            self.ai_state.setText("Not connected")
        if p.own_server:
            self.key_line.setText(f"At {gateway.address}")
        elif gateway.key_for(key):
            self.key_line.setText(f"With your {provider_name(key)} key, from Your API keys")
        else:
            self.key_line.setText(f"No {provider_name(key)} key yet: add one in Your API keys")
        for widget in (self.ai_divider, self.test_row):
            widget.setVisible(connected)
        self.test_button.setEnabled(connected)
        self.cleanup.blockSignals(True)
        self.cleanup.setChecked(s.cleanup and connected)
        self.cleanup.blockSignals(False)
        self.cleanup.setEnabled(connected)
        self.cleanup_caption.setText("Punctuation, no filler words, your names spelled right. Only text is sent, never "
                                     "your voice." if connected else "Connect an AI first (AI connection).")
        self.sparkle.setPixmap(theme.icon_pixmap("tools", tok("violet").name(), 16, self.devicePixelRatioF()))

    def _verdict(self, key: str) -> str:
        scan = self.app.last_scan
        for verdict in (scan or {}).get("verdicts", []):
            if verdict["key"] == key:
                return {"recommended": "fast here", "fast": "fast here", "usable": "a short wait",
                        "slow": "slow here"}.get(verdict["level"], "")
        return ""

    def _test(self) -> None:
        s, gateway = self.app.settings, self.app.gateway
        self.test_button.setEnabled(False)
        self.ai_state.setText("Testing…")
        self.test_note.hide()
        started = time.monotonic()

        def done(answer, error) -> None:
            self.test_button.setEnabled(True)
            if error:
                self.ai_lamp.set_state("err")
                self.ai_state.setText("Failed")
                self.test_note.setText(str(error))
                self.test_note.show()
            else:
                self.ai_lamp.set_state("ok")
                self.ai_state.setText(f"Connected · {time.monotonic() - started:.1f} s")
        run_in_background(self, lambda: self.app.check_ai(gateway, s.cleanup_model), done)


# ---------------------------------------------------------------- how a feature is used, at a glance
# Text Transform, Translate and Formatting each say how they're used in numbered steps, with the keys drawn as keys, and
# show examples of what goes in and what comes out (the owner, phase 31: every section explains itself to anyone).

def _shortcut_caps(shortcut: str, size: str = "md") -> QWidget:
    """A shortcut as keys: "double ctrl" → [Ctrl] (the words around it say "double-tap"), "ctrl+c+c" → [Ctrl] [C] [C]."""
    if shortcut == "double ctrl":
        return keys("Ctrl", size)
    if shortcut == "ctrl+c+c":
        return keys("Ctrl C C", size)
    try:
        return keys(key_names(parse_hotkey(shortcut).label), size)
    except ValueError:
        return keys(shortcut, size)


def _caps_label(value: str, tone: str = "3") -> QLabel:
    """A small label in capitals over a box ("YOU SAY"), as the design's caps."""
    w = label(value.upper(), None, tone, wrap=False)
    w.setFont(font(11, 600, spacing=0.06))
    return w


def _light_markdown(value: str) -> str:
    """A transform's light markdown as a label's HTML: a **Heading** line in bold, "- " bullets as • lines."""
    lines = []
    for line in value.splitlines():
        s = line.strip()
        if len(s) > 4 and s.startswith("**") and s.endswith("**"):
            lines.append(f"<b>{html.escape(s[2:-2])}</b>")
        elif s.startswith("- "):
            lines.append("•&nbsp;&nbsp;" + html.escape(s[2:]))
        elif s:
            lines.append(html.escape(s))
    return "<br>".join(lines)


class Example(QWidget):
    """One example: what goes in, an arrow, what comes out, as two wells side by side ("You say" → "Rflow types")."""

    def __init__(self, before: str, after_html: str, before_label: str, after_label: str):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.before = label(before, tone="2")
        self.after = label(after_html)
        self.after.setTextFormat(Qt.TextFormat.RichText)
        for words, text_label in ((before_label, self.before), (after_label, self.after)):
            box = Card("well", 14)
            inner = QVBoxLayout(box)
            inner.setContentsMargins(14, 10, 14, 12)
            inner.setSpacing(4)
            inner.addWidget(_caps_label(words))
            inner.addWidget(text_label)
            inner.addStretch()
            layout.addWidget(box, 1)
            if text_label is self.before:
                arrow = QLabel()
                arrow.setPixmap(theme.icon_pixmap("arrow-right", tok("iris").name(), 16, 1.0))
                layout.addWidget(arrow, 0, Qt.AlignmentFlag.AlignVCenter)


class Steps(QWidget):
    """How to use a feature, as numbered steps under a line (in the card of the feature's switch). A step is a list of
    words and keys on one line ("Press", [Ctrl C C], "twice."; its last words wrap), or (that list, a widget under it)."""

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 16)
        layout.setSpacing(16)
        layout.addWidget(divider())
        self.rows = QVBoxLayout()
        self.rows.setSpacing(12)
        layout.addLayout(self.rows)

    def set_steps(self, steps: list) -> None:
        clear(self.rows)
        for i, step in enumerate(steps, 1):
            parts, below = step if isinstance(step, tuple) else (step, None)
            number = QLabel(str(i))
            number.setFixedSize(24, 24)
            number.setAlignment(Qt.AlignmentFlag.AlignCenter)
            number.setFont(font(12, 600, mono=True))
            surface(number, "well", 12)
            line = QHBoxLayout()
            line.setSpacing(6)
            for j, part in enumerate(parts):
                last = j == len(parts) - 1
                if isinstance(part, str):
                    line.addWidget(label(part, wrap=last), 1 if last else 0, Qt.AlignmentFlag.AlignVCenter)
                else:
                    line.addWidget(part, 0, Qt.AlignmentFlag.AlignVCenter)
            if not isinstance(parts[-1], str):
                line.addStretch()
            column = QVBoxLayout()
            column.setSpacing(10)
            column.addLayout(line)
            if below is not None:
                column.addWidget(below)
            holder = QWidget()
            row_layout = QHBoxLayout(holder)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(12)
            row_layout.addWidget(number, 0, Qt.AlignmentFlag.AlignTop)
            row_layout.addLayout(column, 1)
            self.rows.addWidget(holder)


class AiNeeded(Card):
    """The AI connection a feature needs, at the top of its page: the model it uses, or a way to connect one."""

    def __init__(self, go_to, feature: str):
        super().__init__("well", 14)
        self.feature = feature
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 12, 12)
        layout.setSpacing(12)
        self.icon = QLabel()
        self.words = caption("", "2")
        self.setup = button("Connect an AI", lambda: go_to("cleanup"), primary=True, size="sm")
        layout.addWidget(self.icon)
        layout.addWidget(self.words, 1)  # the line takes the width: a model's name never breaks in the middle
        layout.addWidget(self.setup)

    def show_model(self, model: str) -> None:
        self.words.setText(f"Uses your AI connection: {model}" if model else
                           f"{self.feature} needs an AI connection. Connect one: it takes a minute.")
        self.setup.setVisible(not model)
        self.icon.setPixmap(theme.icon_pixmap("info", tok("text3" if model else "warn").name(), 16,
                                              self.devicePixelRatioF()))


def live_doing(source: str, target: str, mic_target: str) -> str:
    """What live translation does with a source, in words: "what the computer plays into English"."""
    computer = f"what the computer plays into {language_name(target)}"
    if source == "microphone":
        return f"what the microphone hears into {language_name(mic_target)}"
    if source == "both":
        return f"{computer}, and what you say into {language_name(mic_target)}"
    return computer


def _minutes(seconds: float) -> str:
    """11 min; under a minute: "<1 min"."""
    return "<1 min" if seconds < 30 else f"{round(seconds / 60)} min"


class LivePage(Page):
    """Live translation (sst.live), a main feature: a big Start (a Stop while it runs, with a blinking Live light and
    what it's doing), the translation spoken aloud, then what it listens to and into which languages, its shortcut, how
    the bar shows in screen shares, and the past sessions' transcripts. It shows the real state, however live
    translation was started or stopped: here, the shortcut, the tray or the bar's ✕."""

    def __init__(self, app, go_to):
        super().__init__("Live translation", "Speech translated while people speak, in a bar you can move anywhere: a "
                                             "meeting, a video, or the people in the room.")
        self.app = app
        self._running = False
        self.connect_card, connect = card(12, (16, 12, 12, 12), kind="well", horizontal=True)
        info = QLabel()
        info.setPixmap(theme.icon_pixmap("info", tok("warn").name(), 16, 1.0))
        connect.addWidget(info)
        connect.addWidget(caption("Live translation uses Google Gemini 3.5 Live Translate: add a Gemini key.", "2"), 1)
        connect.addWidget(button("Add a Gemini key", lambda: self._add_key(go_to), primary=True, size="sm"))
        self.add(self.connect_card)

        # Start / Stop first, with the light and what it's doing next to it
        self.start_card = GlowCard()
        column = QVBoxLayout(self.start_card)
        column.setContentsMargins(24, 24, 24, 20)
        column.setSpacing(16)
        self.start_button = PowerButton("Start")
        self.start_button.clicked.connect(self._start_clicked)
        self.light = BlinkLamp("off")
        self.state = label("Off", "heading", wrap=False)
        status = QHBoxLayout()
        status.setSpacing(8)
        status.addWidget(self.light, 0, Qt.AlignmentFlag.AlignVCenter)
        status.addWidget(self.state, 0, Qt.AlignmentFlag.AlignVCenter)
        status.addStretch()
        words = QVBoxLayout()
        words.setSpacing(4)
        words.addStretch()  # centred on the button
        words.addLayout(status)
        self.doing = label("", tone="2")
        words.addWidget(self.doing)
        self.hearing = caption("", "3")  # the bar's status line while it runs: why nothing shows yet
        self.hearing.hide()
        words.addWidget(self.hearing)
        self.spend = caption("", "2")  # the cost saver: what this session costs and saves, or what the last one did
        self.spend.hide()
        words.addWidget(self.spend)
        words.addStretch()
        top = QHBoxLayout()
        top.setSpacing(20)
        top.addWidget(self.start_button, 0, Qt.AlignmentFlag.AlignVCenter)
        top.addLayout(words, 1)
        column.addLayout(top)
        self.how = QHBoxLayout()
        self.how.setSpacing(6)
        column.addLayout(self.how)
        self.note = caption("", "3")
        self.note.hide()
        column.addWidget(self.note)
        self.add(self.start_card)

        # Hear it: the translation spoken aloud in Gemini's voice, or the laptop's (sst.live.voice), a row of its own
        self.voice_card, column = card(6, (20, 18, 20, 18))
        self.speak = Toggle("Speak the translation")
        self.speak.toggled.connect(self._speak_toggled)
        self.speak_icon = QLabel()
        head = row(self.speak_icon, label("Speak the translation", "heading", wrap=False), self.speak, stretch_at=2,
                   spacing=10)
        column.addLayout(head)
        self.speak_caption = caption("", "2")
        column.addWidget(self.speak_caption)
        self.speed = Choice(small=True)
        for label_text, value in LIVE_SPEEDS:
            self.speed.addItem(label_text, value)
        self.speed.setMinimumWidth(200)
        self.speed.currentIndexChanged.connect(self._speed_chosen)
        self.speed_row = QWidget()  # the speed, under a line: only while it speaks
        speed = QVBoxLayout(self.speed_row)
        speed.setContentsMargins(0, 10, 0, 0)
        speed.setSpacing(0)
        speed.addWidget(divider())
        line = setting_row("Speed", "Of the voice on this laptop; it speeds up by itself when it falls behind. Gemini's "
                                    "own voice keeps its pace.", self.speed)[0]
        line.layout().setContentsMargins(0, 14, 0, 0)
        speed.addWidget(line)
        # How loud the other apps stay while it runs: lowered like under an interpreter, the whole time, back when it stops
        self.duck = Slider(0, 100, 5)
        self.duck.setAccessibleName("Other sounds while it runs")
        self.duck.setFixedWidth(160)
        self.duck_value = label("", wrap=False)
        self.duck_value.setMinimumWidth(QFontMetrics(self.duck_value.font()).horizontalAdvance("Unchanged") + 4)
        self.duck.valueChanged.connect(self._duck_moved)
        self.duck.sliderReleased.connect(self._duck_chosen)
        line, _, self.duck_caption = setting_row("Other sounds while it runs", " ", self.duck, self.duck_value)
        line.layout().setContentsMargins(0, 14, 0, 0)
        speed.addWidget(line)
        column.addWidget(self.speed_row)
        self.add(self.voice_card)

        self.source_card, column = card(6, (20, 20, 20, 10))
        column.addWidget(label("Listen to", "heading"))
        self.sources = Segmented(LIVE_SOURCES)
        self.sources.changed.connect(self._source_changed)
        column.addWidget(self.sources, 0, Qt.AlignmentFlag.AlignLeft)
        self.source_words = caption("", "2")
        column.addWidget(self.source_words)
        self.target, self.mic_target = self._languages(), self._languages()
        self.target_row = setting_row("Computer sound into", "", self.target)[0]
        column.addWidget(self.target_row)
        self.mic_row, self.mic_title, _ = setting_row("Microphone into", "", self.mic_target)
        column.addWidget(self.mic_row)
        self.add(self.source_card)

        options, column = card(4, (20, 10, 20, 14))
        self.shortcut = Choice(small=True)
        for label_text, value in LIVE_SHORTCUTS:
            self.shortcut.addItem(label_text, value)
        self.shortcut.setMinimumWidth(200)
        self.shortcut.currentIndexChanged.connect(self._apply)
        self.shortcut_row, _, self.shortcut_caption = setting_row("Shortcut", "", self.shortcut)
        column.addWidget(self.shortcut_row)
        column.addWidget(divider())
        self.hide_share = Toggle("Hide the bar from screen sharing")
        self.hide_share.toggled.connect(self._apply)
        column.addWidget(setting_row("Hide the bar from screen sharing", "On: screen shares and recordings leave it "
                                     "out. Off: colleagues can read it in your share.", self.hide_share)[0])
        column.addWidget(divider())
        self.save = Toggle("Cost saver")
        self.save.toggled.connect(self._save_toggled)
        column.addWidget(setting_row("Cost saver", "Sends audio to Google only while someone talks, and stops a session "
                                     "after 15 minutes without speech. Google bills every second sent.", self.save)[0])
        self.add(options)

        sessions, column = card(4, (20, 18, 20, 16))
        column.addLayout(row(label("Past sessions", "heading"), button("Open the folder", lambda: app.open_live_folder(),
                                                                         link=True, size="sm", icon="folder"),
                             stretch_at=1))
        self.session_rows = QVBoxLayout()
        self.session_rows.setSpacing(0)
        column.addLayout(self.session_rows)
        self.add(sessions)
        self.add(caption("About $2.20 an hour for each source with a paid Gemini key (Both: twice). The audio goes to "
                         "Google, and with a free key Google may use it to improve its products; the transcripts stay "
                         "on this laptop.", "3"))
        self.body.addStretch()

    def _add_key(self, go_to) -> None:
        """AI & models, with the Gemini line of Your API keys open to paste the key."""
        go_to("models")
        models = getattr(self.window(), "pages", {}).get("models")
        if models is not None and self.window().current_page() == "models":
            models.add_key("gemini")

    def _languages(self) -> Choice:
        box = Choice(search=True, small=True)
        for name, code in LIVE_LANGUAGES.items():
            box.addItem(name, code)
        box.setMinimumWidth(220)
        box.currentIndexChanged.connect(self._apply)
        return box

    def refresh(self) -> None:
        s, app = self.app.settings, self.app
        problem, running = app.live_problem(), app.live_running()
        if running != self._running:  # started or stopped, here or elsewhere: a note about the last session goes
            self._running = running
            self._say("")
        source = s.live_source if s.live_source in LIVE_SOURCE_WORDS else "computer"
        doing = live_doing(source, *app.live_languages())  # while it runs, a new language waits for the next start
        self.connect_card.setVisible(bool(problem))
        self.start_button.set_running(running, "Stop" if running else "Start")
        self.start_button.setEnabled(running or not problem)
        self.light.set_state("live" if running else "off")
        self.light.set_blinking(running)  # its timer also stops while the page is hidden
        self.state.setText("Live" if running else "Off")
        set_tone(self.state, "live" if running else None)
        self.doing.setText(f"Translating {doing}" if running else "Add a Gemini key to start it" if problem
                           else f"Ready to translate {doing}")
        hearing = app.live_note() if running else ""
        self.hearing.setText(hearing)
        self.hearing.setVisible(bool(hearing))
        spend = self._spend_words(running)
        self.spend.setText(spend)
        self.spend.setVisible(bool(spend))
        clash = app.live_shortcut_clash()
        clear(self.how)
        if s.live_shortcut and not clash:
            self.how.addWidget(caption("Start and stop it in any app with", "2", wrap=False))
            self.how.addWidget(_shortcut_caps(s.live_shortcut))
        else:
            self.how.addWidget(caption("Start it here, or from the tray menu", "2", wrap=False))
        self.how.addStretch()
        self.shortcut_caption.setText(f"{key_names(parse_hotkey(s.live_shortcut).label)} is {clash}'s shortcut, so it "
                                      f"stays with {clash}: choose another." if clash
                                      else "Starts and stops live translation in any app.")
        self.shortcut_caption.show()
        self.sources.set_current(source)
        self.source_words.setText(LIVE_SOURCE_WORDS[source])
        for box, value in ((self.target, s.live_target), (self.mic_target, s.live_mic_target),
                           (self.shortcut, s.live_shortcut)):
            box.blockSignals(True)
            if box.findData(value) < 0:  # a shortcut set by hand
                box.insertItem(box.count() - 1, value, value)
            box.setCurrentIndex(box.findData(value))
            box.blockSignals(False)
        self.target_row.setVisible(source != "microphone")
        self.mic_row.setVisible(source != "computer")
        self.mic_title.setText("Your speech into" if source == "both" else "Microphone into")
        self.hide_share.blockSignals(True)
        self.hide_share.setChecked(s.live_hide_from_share)
        self.hide_share.blockSignals(False)
        self.save.blockSignals(True)
        self.save.setChecked(s.live_save)
        self.save.blockSignals(False)
        self._refresh_voice(s, source)
        clear(self.session_rows)
        found = app.live_sessions()
        if not found:
            self.session_rows.addWidget(caption("Each session's transcript is kept here once someone has spoken: both "
                                                "languages, line by line.", "3"))
        for began, lines, path in found:
            line = QWidget()
            layout = QHBoxLayout(line)
            layout.setContentsMargins(0, 6, 0, 6)
            layout.setSpacing(12)
            layout.addWidget(label(f"{began:%a} {began.day} {began:%b}, {began:%H:%M}", wrap=False))
            layout.addWidget(caption(f"{lines} line{'' if lines == 1 else 's'}", "3", wrap=False), 1)
            for words, only in (("Copy", False), ("Copy translation", True)):
                copy = button(words, link=True, size="sm", icon="copy")
                copy.setToolTip("The translation alone, line by line" if only else "Both languages, line by line")
                copy.clicked.connect(lambda _=False, b=copy, p=path, o=only, w=words: self._copy_session(b, p, o, w))
                layout.addWidget(copy)
            layout.addWidget(button("Open", lambda _=False, p=path: self.app.open_live_session(p), link=True, size="sm"))
            self.session_rows.addWidget(line)

    def _spend_words(self, running: bool) -> str:
        """The cost saver's figures: this session's while it runs and saves, else the last saving session's."""
        if running:
            spend = self.app.live_spend() if self.app.settings.live_save else None
            return (f"This session: {dollars(spend.cost)} · {_minutes(spend.seconds)} · saved {dollars(spend.saved)}"
                    if spend is not None else "")
        why, spend = self.app.live_stopped_why(), self.app.live_last_spend()
        words = [f"Stopped: {why}." if why else ""]
        if spend is not None:
            words.append(f"Last session: {dollars(spend.cost)} for {_minutes(spend.seconds)}, saved {dollars(spend.saved)}")
        return " ".join(w for w in words if w)

    def _save_toggled(self, on: bool) -> None:
        self.app.set_live_save(on)
        self.refresh()

    def _copy_session(self, copy: Button, path, translation_only: bool, words: str) -> None:
        """A session's transcript onto the clipboard; the button says "Copied", or why not, for a moment."""
        problem = self.app.copy_live_session(path, translation_only)
        if problem:
            copy.setToolTip(problem)
        self._show_copied(copy, "Couldn't copy" if problem else "Copied", "warning" if problem else "check")
        QTimer.singleShot(1500, lambda: self._show_copied(copy, words, "copy"))

    @staticmethod
    def _show_copied(copy: Button, words: str, icon: str) -> None:
        try:
            copy.icon_name = icon
            copy.setText(words)
            copy.update()
        except RuntimeError:  # the page was refreshed meanwhile: a new button stands there
            pass

    def _refresh_voice(self, s, source: str) -> None:
        voice, (state, percent, problem) = self.app.live_voice(), self.app.live_voice_state()
        speaks = language_name(voice.language)
        spoken = LiveConfig(target=s.live_target, mic_target=s.live_mic_target, source=source).spoken_lanes(voice.language)
        self.speak_icon.setPixmap(theme.icon_pixmap("speaker" if s.live_speak else "speaker-off", tok("text2").name(), 18,
                                                    self.devicePixelRatioF()))
        self.speak.blockSignals(True)
        self.speak.setChecked(s.live_speak)
        self.speak.blockSignals(False)
        self.speed.blockSignals(True)
        self.speed.setCurrentIndex(max(0, self.speed.findData(s.live_speak_speed)))
        self.speed.blockSignals(False)
        if not self.duck.isSliderDown():
            self.duck.blockSignals(True)
            self.duck.setValue(round(s.live_duck * 100))
            self.duck.blockSignals(False)
            self._show_duck(self.duck.value())
        # Gemini says the translation in its own voice, in any language; the voice on this laptop stands in for it
        gemini = "Gemini says the translation in its own voice as it's translated."
        if state == "downloading":
            words = (f"Downloading {LOCAL_VOICE.name}, the voice on this laptop: {percent}%. {gemini} {LOCAL_VOICE.name} "
                     f"stands in when Gemini's voice doesn't come.")
        elif problem:
            words = (f"The voice on this laptop couldn't be downloaded ({problem}). {gemini} Switch it on again to try "
                     f"once more.")
        elif not spoken:
            words = (f"{gemini} {voice.name}, the voice on this laptop, speaks only {speaks}: it stands in when Gemini's "
                     f"doesn't come only for what's translated into {speaks}.")
        elif state == "missing":
            words = (f"{gemini} {voice.name}, an {speaks} voice that runs on this laptop, stands in when Gemini's "
                     f"doesn't come. Downloaded the first time ({round(voice.size / 1e6)} MB).")
        else:
            words = f"{gemini} {voice.name}, an {speaks} voice on this laptop, stands in when Gemini's doesn't come."
            if source != "computer":
                words += " Through speakers the microphone pauses while it speaks; headphones keep it listening."
        self.speak_caption.setText(words)
        self.speed_row.setVisible(s.live_speak)

    def _speak_toggled(self, on: bool) -> None:
        self.app.set_live_speak(on)
        self.refresh()

    def _speed_chosen(self, *_) -> None:
        self.app.set_live_speak_speed(self.speed.currentData())
        self.refresh()

    def _duck_moved(self, value: int) -> None:
        self._show_duck(value)
        if not self.duck.isSliderDown():  # the arrow keys: chosen at once; a drag: once it's let go
            self._duck_chosen()

    def _duck_chosen(self) -> None:
        self.app.set_live_duck(self.duck.value() / 100)

    def _show_duck(self, value: int) -> None:
        self.duck_value.setText("Unchanged" if value >= 100 else f"{value}%")
        if value >= 100:
            words = "Left as they are: the voice speaks over them."
        else:
            words = ("Other apps go silent for you" if value <= 0 else "Other apps stay this quiet") + \
                " while live translation runs, and come back when it stops. Rflow still hears and translates them."
        self.duck_caption.setText(words)

    def _say(self, message: str) -> None:
        self.note.setText(message)
        self.note.setVisible(bool(message))

    def _start_clicked(self) -> None:
        if self.app.live_running():
            self.app.stop_live()
            self._say("")
        else:
            problem = self.app.start_live()
            self._say("" if problem == "Not started." else problem)
        self.refresh()

    def _source_changed(self, source: str) -> None:
        problem = self.app.set_live_source(source)
        self.refresh()
        if problem:
            self._say(problem)

    def _apply(self, *_) -> None:
        s = self.app.settings
        if self.hide_share.isChecked() != s.live_hide_from_share:
            self.app.set_live_hidden(self.hide_share.isChecked())  # at once, also while it runs
            s = self.app.settings
        chosen = (self.target.currentData(), self.mic_target.currentData(), self.shortcut.currentData())
        if chosen != (s.live_target, s.live_mic_target, s.live_shortcut):
            new_language = chosen[:2] != (s.live_target, s.live_mic_target)
            self.app.apply_settings(dataclasses.replace(s, live_target=chosen[0], live_mic_target=chosen[1],
                                                        live_shortcut=chosen[2]))
            if new_language and self.app.live_running():
                self._say("Saved: the next start uses the new language.")
        self.refresh()


def meeting_line(app_name: str, cost, screen: bool) -> str:
    """The running meeting in a line: "Microsoft Teams · 14 min · $0.12 so far · 6 slides"."""
    if cost is None:
        return app_name
    words = [app_name, _minutes(cost.seconds), f"{dollars(cost.total)} so far"]
    if screen:
        words.append(f"{cost.frames} slide{'' if cost.frames == 1 else 's'}")
    return " · ".join(words)


class MeetingPage(Page):
    """Meeting mode (sst.meeting): the meeting found by itself (several: a choice; none: "Start a meeting first"), one
    switch per part (live translation, the recording, the minutes, the slides), a big Start / Stop with a red Recording
    light and the meeting's time and cost, then "Writing the minutes…" and "Minutes ready" with Open and Copy, and the
    past meetings. Like the Live page it shows the real state, however the meeting was started: here, the shortcut, the
    tray, or the meeting window closing by itself. Two tabs at the top: Online (all that) and In person (a meeting in a
    room: the microphone alone, no meeting app, no screen; it runs until stopped)."""

    def __init__(self, app, go_to):
        super().__init__("Meeting mode", MEETING_SUBTITLES["online"])
        self.app = app
        self.found: list | None = None  # the meetings open now (MeetingWindow), from the last look
        self.chosen = None  # the one picked of them
        self.place = "room" if app.settings.meeting_place == "room" else "online"
        self._timer = QTimer(self, interval=MEETING_REFRESH_MS, timeout=self._tick)
        self.tabs = Segmented(MEETING_TABS)
        self.tabs.set_current(self.place)
        self.tabs.changed.connect(self._tab_chosen)
        self.add(self.tabs)
        self.connect_card, connect = card(12, (16, 12, 12, 12), kind="well", horizontal=True)
        info = QLabel()
        info.setPixmap(theme.icon_pixmap("info", tok("warn").name(), 16, 1.0))
        connect.addWidget(info)
        self.connect_words = caption("", "2")
        connect.addWidget(self.connect_words, 1)
        self.connect_button = button("Add a Gemini key", lambda: LivePage._add_key(self, go_to), primary=True, size="sm")
        connect.addWidget(self.connect_button)
        self.add(self.connect_card)

        # Start / Stop first, with the Recording light and the meeting's line next to it
        self.start_card = GlowCard()
        column = QVBoxLayout(self.start_card)
        column.setContentsMargins(24, 24, 24, 20)
        column.setSpacing(16)
        self.start_button = PowerButton("Start")
        self.start_button.clicked.connect(self._start_clicked)
        self.light = BlinkLamp("off")
        self.state = label("Off", "heading", wrap=False)
        status = QHBoxLayout()
        status.setSpacing(8)
        status.addWidget(self.light, 0, Qt.AlignmentFlag.AlignVCenter)
        status.addWidget(self.state, 0, Qt.AlignmentFlag.AlignVCenter)
        status.addStretch()
        words = QVBoxLayout()
        words.setSpacing(4)
        words.addStretch()  # centred on the button
        words.addLayout(status)
        self.doing = label("", tone="2")
        words.addWidget(self.doing)
        self.done_row = QWidget()  # the minutes written: Open and Copy them, or the folder
        done = QHBoxLayout(self.done_row)
        done.setContentsMargins(0, 4, 0, 0)
        done.setSpacing(4)
        self.open_minutes = button("Open the minutes", lambda: self._open_last(), link=True, size="sm", icon="external")
        self.copy_last = button("Copy", link=True, size="sm", icon="copy")
        self.copy_last.clicked.connect(lambda: self._copy(self.copy_last, self.app.meeting_status()[2], "Copy"))
        self.open_last_folder = button("Open the folder", lambda: self.app.open_meeting_folder(self.app.meeting_status()[2]),
                                       link=True, size="sm", icon="folder")
        for b in (self.open_minutes, self.copy_last, self.open_last_folder):
            done.addWidget(b)
        done.addStretch()
        self.done_row.hide()
        words.addWidget(self.done_row)
        words.addStretch()
        top = QHBoxLayout()
        top.setSpacing(20)
        top.addWidget(self.start_button, 0, Qt.AlignmentFlag.AlignVCenter)
        top.addLayout(words, 1)
        column.addLayout(top)
        self.how = QHBoxLayout()
        self.how.setSpacing(6)
        column.addLayout(self.how)
        self.note = caption("", "3")
        self.note.hide()
        column.addWidget(self.note)
        self.add(self.start_card)

        # The meeting: found by itself; several to pick from; none yet
        self.app_card, column = card(8, (20, 18, 20, 18))
        column.addLayout(row(label("Your meeting", "heading"), button("Look again", lambda: self._scan(), link=True,
                                                                     size="sm", icon="search"), stretch_at=1))
        self.app_line = QWidget()
        line = QHBoxLayout(self.app_line)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(10)
        self.app_lamp = Lamp("ok")
        self.app_name = label("", wrap=False)
        self.app_name.setFont(font(14, 600))
        self.app_title = ElidedLabel("")
        set_tone(self.app_title, "2")
        line.addWidget(self.app_lamp, 0, Qt.AlignmentFlag.AlignVCenter)
        line.addWidget(self.app_name, 0, Qt.AlignmentFlag.AlignVCenter)
        line.addWidget(self.app_title, 1, Qt.AlignmentFlag.AlignVCenter)
        column.addWidget(self.app_line)
        self.picker = Choice(small=True)
        self.picker.setMinimumWidth(320)
        self.picker.currentIndexChanged.connect(self._picked)
        column.addWidget(self.picker, 0, Qt.AlignmentFlag.AlignLeft)
        self.app_words = caption("", "2")
        column.addWidget(self.app_words)
        self.add(self.app_card)

        # One switch per part, each tab its own
        self.parts: dict[str, Toggle] = {}
        self.part_rows: dict[str, QWidget] = {}
        self.room_parts: dict[str, Toggle] = {}
        self.room_rows: dict[str, QWidget] = {}
        self.parts_card = self.room_card = None
        for place, items, toggles, rows in (("online", MEETING_PARTS, self.parts, self.part_rows),
                                            ("room", ROOM_PARTS, self.room_parts, self.room_rows)):
            parts, column = card(4, (20, 6, 20, 6))
            for i, (part, title, words_text) in enumerate(items):
                if i:
                    column.addWidget(divider())
                toggle = Toggle(title)
                toggle.toggled.connect(lambda on, p=part, r=place == "room": self._part_toggled(p, on, r))
                toggles[part] = toggle
                rows[part] = setting_row(title, words_text, toggle)[0]
                column.addWidget(rows[part])
            self.add(parts)
            if place == "online":
                self.parts_card = parts
            else:
                self.room_card = parts

        options, column = card(4, (20, 6, 20, 6))
        self.shortcut = Choice(small=True)
        for label_text, value in MEETING_SHORTCUTS:
            self.shortcut.addItem(label_text, value)
        self.shortcut.setMinimumWidth(200)
        self.shortcut.currentIndexChanged.connect(self._shortcut_chosen)
        self.shortcut_row, _, self.shortcut_caption = setting_row("Shortcut", "", self.shortcut)
        column.addWidget(self.shortcut_row)
        self.add(options)

        meetings, column = card(4, (20, 18, 20, 16))
        column.addLayout(row(label("Past meetings", "heading"), button("Open the folder", lambda: app.open_meeting_folder(),
                                                                         link=True, size="sm", icon="folder"),
                             stretch_at=1))
        self.meeting_rows = QVBoxLayout()
        self.meeting_rows.setSpacing(0)
        column.addLayout(self.meeting_rows)
        self.add(meetings)
        self.add(caption("The minutes cost about $0.25 an hour of meeting with a paid Gemini key; live translation costs "
                         "as on its page. The recording and the slides go to Google for the minutes, and with a free key "
                         "Google may use them to improve its products. Everything is kept on this laptop.", "3"))
        self.body.addStretch()

    # ---- the meetings found

    def _scan(self) -> None:
        """Look for the meeting windows again; the one picked stays picked while it's open."""
        found = list(self.app.meeting_windows())
        keep = self.chosen.hwnd if self.chosen is not None else None
        self.found = found
        self.chosen = next((w for w in found if w.hwnd == keep), found[0] if len(found) == 1 else None)
        self.refresh()

    def _picked(self, index: int) -> None:
        if self.found and 0 < index <= len(self.found):  # the first choice is "Pick your meeting"
            self.chosen = self.found[index - 1]
        else:
            self.chosen = None
        self.refresh()

    def _fill_picker(self) -> None:
        self.picker.blockSignals(True)
        self.picker.clear()
        self.picker.addItem("Pick your meeting", None)
        for w in self.found or []:
            self.picker.addItem(f"{w.app}: {w.title}", w.hwnd)
        at = self.picker.findData(self.chosen.hwnd) if self.chosen is not None else 0
        self.picker.setCurrentIndex(max(0, at))
        self.picker.blockSignals(False)

    # ---- on screen

    def showEvent(self, event):
        super().showEvent(event)
        self._timer.start()

    def hideEvent(self, event):
        self._timer.stop()  # a hidden page looks for nothing
        super().hideEvent(event)

    def _tick(self) -> None:
        if self.app.meeting_status()[0] == "recording" or self.place == "room":
            self.refresh()  # the time, the cost and the slides go on (in person, there are no windows to look for)
        else:
            self._scan()  # a meeting started meanwhile shows by itself

    def _tab_chosen(self, place: str) -> None:
        """Online or In person; remembered, so the shortcut and the tray start that kind next time."""
        self.place = place
        if self.app.settings.meeting_place != place:
            self.app.apply_settings(dataclasses.replace(self.app.settings, meeting_place=place))
        self._say("")
        self.refresh()

    @property
    def target(self):
        """What Start starts: the room in person, else the meeting window picked (None: none yet)."""
        return IN_PERSON if self.place == "room" else self.chosen

    def refresh(self) -> None:
        if self.found is None and self.place == "online":
            self._scan()  # the first time it's shown (it refreshes itself)
            return
        s, app = self.app.settings, self.app
        state, window, folder, result, error = app.meeting_status()
        running, writing = state == "recording", state == "writing minutes"
        if (running or writing) and window is not None:
            self.place = "room" if window == IN_PERSON else "online"  # the tab of the meeting going on
        room = self.place == "room"
        self.tabs.set_current(self.place)
        for key, b in self.tabs.buttons.items():
            b.setEnabled(not running or key == self.place)  # one meeting at a time
        self.subtitle.setText(MEETING_SUBTITLES[self.place])
        problem = app.meeting_problem(room)
        self.connect_card.setVisible(bool(problem) and not running)
        self.connect_words.setText(problem)
        self.connect_button.setVisible("Gemini key" in problem)
        # a finished meeting of the other kind isn't shown on this tab
        mine = window is not None and (window == IN_PERSON) == room
        if not mine and not running:
            state, folder, result, error = "", None, None, ""
        ready = self.target is not None and not problem and not writing
        self.start_button.set_running(running, "Stop" if running else "Start")
        self.start_button.setEnabled(running or ready)
        self.light.set_state({"recording": "live", "writing minutes": "ai", "done": "ok", "failed": "err"}.get(state, "off"))
        self.light.set_blinking(running or writing)  # its timer also stops while the page is hidden
        if running:
            heading, tone = "Recording" if self._options_of_last("record") else "On", "live"
            doing = meeting_line(window.app, app.meeting_cost(), self._options_of_last("screen"))
        elif writing:
            heading, tone = "Writing the minutes…", None
            doing = f"{meeting_line(window.app, app.meeting_cost(), False)}: they're ready in a few minutes."
        elif state == "done" and result is not None:
            heading, tone, doing = "Minutes ready", None, result.title
        elif state == "done":
            where = "in person" if window == IN_PERSON else f"in {window.app}"
            heading, tone, doing = "Saved", None, f"The meeting {where} is saved on this laptop."
        elif state == "failed":
            heading, tone, doing = "The minutes failed", None, f"{error}. The recording is kept: nothing is lost."
        else:
            heading, tone = "Off", None
            doing = ("Add a Gemini key to start it" if "Gemini key" in problem else "Switch on a part to start it" if problem
                     else "Ready: put the laptop where its microphone hears everyone" if room
                     else f"Ready for your meeting in {self.chosen.app}" if self.chosen is not None
                     else "Pick your meeting below" if self.found else "Start a meeting first")
        self.state.setText(heading)
        set_tone(self.state, tone)
        self.doing.setText(doing)
        finished = state in ("done", "failed") and folder is not None
        self.done_row.setVisible(finished)
        self.open_minutes.setVisible(result is not None)
        self.copy_last.setVisible(result is not None)
        self.open_last_folder.setVisible(result is None)
        clash = app.meeting_shortcut_clash()
        clear(self.how)
        if s.meeting_shortcut and not clash:
            self.how.addWidget(caption("Start and stop it in any app with", "2", wrap=False))
            self.how.addWidget(_shortcut_caps(s.meeting_shortcut))
        else:
            self.how.addWidget(caption("Start it here, or from the tray menu", "2", wrap=False))
        self.how.addStretch()
        self.app_card.setVisible(not room)
        self.parts_card.setVisible(not room)
        self.room_card.setVisible(room)
        if not room:
            self._refresh_app(running or writing, window)
        self._refresh_parts(s, running)
        self.shortcut.blockSignals(True)
        if self.shortcut.findData(s.meeting_shortcut) < 0:  # a shortcut set by hand
            self.shortcut.insertItem(self.shortcut.count() - 1, s.meeting_shortcut, s.meeting_shortcut)
        self.shortcut.setCurrentIndex(self.shortcut.findData(s.meeting_shortcut))
        self.shortcut.blockSignals(False)
        self.shortcut_caption.setText(f"{key_names(parse_hotkey(s.meeting_shortcut).label)} is {clash}'s shortcut, so "
                                      f"it stays with {clash}: choose another." if clash
                                      else "Starts and stops the meeting in any app: in the room when In person "
                                           "was the tab last open, else the meeting found.")
        self.shortcut_caption.show()
        self._refresh_past()

    def _options_of_last(self, part: str) -> bool:
        """A part as the meeting running started with it (a switch changed meanwhile waits for the next one)."""
        last = getattr(self.app, "last_meeting", None)
        options = getattr(last, "options", None)
        if options is not None:
            return getattr(options, part)
        if self.place == "room":
            return part != "screen" and getattr(self.app.settings, f"room_{part}")
        return getattr(self.app.settings, f"meeting_{part}")

    def _refresh_app(self, busy: bool, window) -> None:
        shown = window if busy else self.chosen
        several = not busy and len(self.found or []) > 1
        self.app_line.setVisible(shown is not None and not several)
        if shown is not None:
            self.app_name.setText(shown.app)
            self.app_title.setText(shown.title)
            self.app_lamp.set_state("live" if busy else "ok")
        self.picker.setVisible(several)
        if several:
            self._fill_picker()
        self.picker.setEnabled(not busy)
        if busy:
            words = ""
        elif several:
            words = f"{len(self.found)} meetings are open: pick the one to record."
        elif shown is not None:
            words = "Found by itself. Start when the meeting begins; it stops by itself when the meeting window closes."
        else:
            words = ("Start a meeting first: Microsoft Teams, Zoom, Google Meet in a browser, or Webex. Rflow finds it "
                     "by itself.")
        self.app_words.setText(words)
        self.app_words.setVisible(bool(words))

    def _refresh_parts(self, s, running: bool) -> None:
        for kind, toggles, rows in (("meeting", self.parts, self.part_rows), ("room", self.room_parts, self.room_rows)):
            on = {part: getattr(s, f"{kind}_{part}", False) for part, *_ in MEETING_PARTS}
            works = {"translate": on["translate"], "record": on["record"], "minutes": on["minutes"] and on["record"]}
            works["screen"] = on["screen"] and works["minutes"]
            for part, toggle in toggles.items():
                toggle.blockSignals(True)
                toggle.setChecked(works[part])
                toggle.blockSignals(False)
                # a running meeting keeps the parts it started with; the minutes need the sound, the slides the minutes
                needs = {"minutes": on["record"], "screen": works["minutes"]}.get(part, True)
                toggle.setEnabled(not running and needs)
                rows[part].setToolTip("" if needs else "Needs Record" if part == "minutes" else "Needs the minutes")

    def _refresh_past(self) -> None:
        clear(self.meeting_rows)
        found = self.app.past_meetings()
        if not found:
            self.meeting_rows.addWidget(caption("Each meeting is kept here: its recording, its slides and its minutes.",
                                                "3"))
        for folder, began, app_name, seconds, has_minutes in found:
            line = QWidget()
            layout = QHBoxLayout(line)
            layout.setContentsMargins(0, 6, 0, 6)
            layout.setSpacing(12)
            layout.addWidget(label(f"{began:%a} {began.day} {began:%b}, {began:%H:%M}", wrap=False))
            layout.addWidget(caption(f"{app_name} · {_minutes(seconds)}" + ("" if has_minutes else " · no minutes"), "3",
                                     wrap=False), 1)
            if has_minutes:
                copy = button("Copy minutes", link=True, size="sm", icon="copy")
                copy.clicked.connect(lambda _=False, b=copy, f=folder: self._copy(b, f, "Copy minutes"))
                layout.addWidget(copy)
                layout.addWidget(button("Open minutes", lambda _=False, f=folder: self.app.open_meeting(f), link=True,
                                        size="sm"))
            elif (folder / "others.wav").is_file() or (folder / "you.wav").is_file():  # quit or failed while written
                writing = folder in self.app.writing_minutes
                write = button("Writing the minutes…" if writing else "Write minutes",
                               lambda _=False, f=folder: self.app.write_minutes(f), link=True, size="sm")
                write.setEnabled(not writing)
                write.setToolTip("The minutes from the recording, by Gemini (about $0.25 an hour)")
                layout.addWidget(write)
            layout.addWidget(button("Open folder", lambda _=False, f=folder: self.app.open_meeting_folder(f), link=True,
                                    size="sm"))
            self.meeting_rows.addWidget(line)

    # ---- what the user does

    def _say(self, message: str) -> None:
        self.note.setText(message)
        self.note.setVisible(bool(message))

    def _start_clicked(self) -> None:
        if self.app.meeting_status()[0] == "recording":
            self.app.stop_meeting()
            self._say("")
        else:
            problem = self.app.start_meeting(self.target)
            self._say("" if problem == "Not started." else problem)
        self.refresh()

    def _open_last(self) -> None:
        folder = self.app.meeting_status()[2]
        if folder is not None:
            self.app.open_meeting(folder)

    def _copy(self, copy: Button, folder, words: str) -> None:
        """A meeting's minutes onto the clipboard; the button says "Copied", or why not, for a moment."""
        problem = self.app.copy_minutes(folder) if folder is not None else "Nothing to copy"
        if problem:
            copy.setToolTip(problem)
        LivePage._show_copied(copy, "Couldn't copy" if problem else "Copied", "warning" if problem else "check")
        QTimer.singleShot(1500, lambda: LivePage._show_copied(copy, words, "copy"))

    def _part_toggled(self, part: str, on: bool, in_person: bool = False) -> None:
        if in_person:
            self.app.set_meeting_option(part, on, True)
        else:
            self.app.set_meeting_option(part, on)
        self.refresh()

    def _shortcut_chosen(self, *_) -> None:
        chosen = self.shortcut.currentData()
        if chosen != self.app.settings.meeting_shortcut:
            self.app.apply_settings(dataclasses.replace(self.app.settings, meeting_shortcut=chosen))
        self.refresh()


# What each transform does, shown on its page: realistic dictations and what Rflow makes of them. The tests check every
# one with the real TransformGuard, so each result is one Rflow would accept (nothing invented, nothing lost).
TRANSFORM_EXAMPLES = {
    "concise": [
        ("I checked the deployment and everything looks good, but we still have one issue with the database migration, "
         "and I think we should fix that before production.",
         "Deployment looks good, but the database migration issue needs to be fixed before production."),
        ("So basically what I'm saying is that we could deploy on Friday, or maybe Monday, depending on testing.",
         "We could deploy Friday or maybe Monday, depending on testing."),
    ],
    "professional": [
        ("hey can you send me the report by friday, the client keeps asking about it",
         "Could you please send me the report by Friday? The client keeps asking about it."),
        ("the build is broken again, I think it's the new login code, I'll look at it after lunch",
         "The build is failing again. I think the new login code is the cause; I will look into it after lunch."),
    ],
    "bullets": [
        ("The frontend is basically done. The backend API is also mostly done, but we still need to finish "
         "authentication and then test everything together.",
         "**Status**\n- Frontend: Complete\n- Backend API: Mostly complete\n- Authentication: Pending\n- Testing: Pending"),
        ("For the trip we need to book the flights, find a hotel near the office and maybe rent a car.",
         "- Book the flights\n- Find a hotel near the office\n- Maybe rent a car"),
    ],
    "actions": [
        ("During the meeting we agreed that John is going to handle the API documentation, Sarah is going to take care "
         "of the database migration, and I'm going to prepare the deployment checklist.",
         "**Action items**\n- John — API documentation\n- Sarah — Database migration\n- Me — Deployment checklist"),
        ("Before we deploy, I want to check the database migration, verify the environment variables, and make sure "
         "the rollback procedure works.",
         "**Action items**\n- Check the database migration\n- Verify the environment variables\n"
         "- Test the rollback procedure"),
    ],
    "rewrite": [
        ("the update it didn't install because of the disk it was full",
         "The update didn't install because the disk was full."),
        ("The meeting is probably going to be next week.", "The meeting will probably be next week."),
    ],
    "grammar": [
        ("i finish the test for login page, two bug found, one is critical",
         "I finished the test for the login page; two bugs were found, and one is critical."),
        ("明日の会議わ10時からです。資料を確認しといてください。", "明日の会議は10時からです。資料を確認しておいてください。"),
    ],
    # The research's examples (research_notes "Rflow Text Transform expansion", 1.6): names stay as written, so "suzuki
    # san" is Suzukiさん, never a guessed 鈴木さん.
    "teams": [
        ("田中さん、明日の会議なんだけど、資料まだできてなくて、たぶん15時ぐらいになると思う、ごめん",
         "田中さん\nお疲れ様です。\n明日の会議資料ですが、まだ完成しておらず、15時頃になる見込みです。\n遅くなり申し訳ありません。\n"
         "よろしくお願いいたします。"),
        ("hi suzuki san, i finish the test for login page, two bug found, one is critical, i will fix by tomorrow morning, "
         "can you review PR after that",
         "Suzukiさん\nお疲れ様です。\nログイン画面のテストが完了しました。\nバグが2件見つかり、うち1件はクリティカルです。\n"
         "明日の午前中までに修正しますので、その後PRのレビューをお願いできますか。\nよろしくお願いいたします。"),
    ],
    "email_internal": [
        ("佐藤さん、来週の定例の資料、金曜までに共有します、たぶん10ページぐらい",
         "佐藤さん\n\nお疲れ様です。\n\n来週の定例会議の資料は、金曜日までに共有いたします。\n分量は10ページ程度になる見込みです。\n\n"
         "よろしくお願いいたします。"),
        ("tanaka san, the server maintenance moved to 10/20 from 9 PM, please tell your team",
         "Tanakaさん\n\nお疲れ様です。\n\nサーバーメンテナンスが10/20の21時からに変更になりました。\n"
         "お手数ですが、チームの皆さんにもお伝えください。\n\nよろしくお願いいたします。"),
    ],
    "email_external": [
        ("山本さん、見積もりの件ありがとうございます、ちょっと社内で確認するんで、来週の水曜までに返事します",
         "山本様\n\nいつもお世話になっております。\n\nお見積もりの件、ありがとうございます。\n社内で確認のうえ、来週水曜日までに"
         "ご連絡いたします。\n\n何卒よろしくお願いいたします。"),
        ("Mr. Kato from ABC, thanks for the meeting yesterday, I'll send the revised quote by Friday",
         "ABC\nKato様\n\nいつもお世話になっております。\n\n昨日はお打ち合わせのお時間をいただき、ありがとうございました。\n"
         "修正したお見積もりを金曜日までにお送りいたします。\n\n何卒よろしくお願いいたします。"),
    ],
    "translate": [
        ("Can you send me the invoice by Friday?", "金曜日までに請求書を送っていただけますか？"),
        ("来週の定例会議は木曜日の午後3時からに変更になりました。", "Next week's regular meeting has moved to Thursday at 3 PM."),
    ],
}


class _GroupedTabs(QWidget):
    """The transforms as tabs, one row per group (Tone, Format, Language): one chosen across all the rows."""

    changed = Signal(str)

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        self.rows: list[Segmented] = []
        self.buttons: dict[str, QPushButton] = {}
        for group in GROUPS:
            options = [(key, t.name) for key, t in TRANSFORMS.items() if t.group == group]
            tabs = Segmented(options)
            tabs.changed.connect(self.set_current)
            tabs.changed.connect(self.changed)
            self.rows.append(tabs)
            self.buttons |= tabs.buttons
            name = caption(group, "3", wrap=False)
            name.setFixedWidth(68)
            layout.addLayout(row(name, tabs, stretch_at=2, spacing=10))

    def set_current(self, key: str) -> None:
        for tabs in self.rows:
            if key in tabs.buttons:
                tabs.set_current(key)
                continue
            for b in tabs.buttons.values():  # another row's choice: none chosen here
                b.setAutoExclusive(False)
                b.setChecked(False)
                b.setAutoExclusive(True)


class TransformPage(Page):
    """Text Transform (sst.transformui): on or off, how it's used (the user's own key, phrase and menu), what each
    transform does with examples, a box to try them, then the voice commands' phrases and the menu (sst.commands)."""

    def __init__(self, app, go_to):
        super().__init__("Text Transform", "Rewrite text you already have: shorter, more professional, grammar fixed, as "
                                           "a Japanese Teams message or email, or translated. It works in any app, on the "
                                           "text you select, or on all the text in the box you're typing in. Numbers, "
                                           "names, dates and your “maybe” stay.")
        self.app = app
        s = app.settings
        self.ai = AiNeeded(go_to, "Text Transform")
        self.model, self.setup = self.ai.words, self.ai.setup
        self.add(self.ai)

        how, layout = card(0, (20, 4, 20, 4))
        self.transform_on = Toggle("Text Transform")
        self.transform_on.toggled.connect(self._switched)
        switch, _, self.switch_words = setting_row("Text Transform", " ", self.transform_on)  # its words: refresh()
        layout.addWidget(switch)
        self.steps = Steps()
        layout.addWidget(self.steps)
        self.add(how)

        examples, layout = card(12, (20, 18, 20, 20))
        layout.addWidget(label("What each one does", "heading"))
        self.example_key = "concise"
        self.example_tabs = _GroupedTabs()
        self.example_tabs.set_current("concise")
        self.example_tabs.changed.connect(self._show_examples)
        layout.addWidget(self.example_tabs)
        self.example_words = caption("", "2")
        layout.addWidget(self.example_words)
        self.examples = QVBoxLayout()
        self.examples.setSpacing(10)
        layout.addLayout(self.examples)
        self.add(examples)

        trial, layout = card(10, (20, 18, 20, 20))
        layout.addWidget(label("Try it", "heading"))
        self.sample = QPlainTextEdit()
        self.sample.setPlainText("I checked the deployment and everything looks good, but we still have one issue with the "
                                 "database migration, and I think we should fix that before production.")
        field(self.sample, "well")
        fixed_height(self.sample, 88)
        layout.addWidget(self.sample)
        self.try_buttons: dict[str, QPushButton] = {}
        holder = QWidget()
        buttons = FlowLayout(8)  # ten of them: they wrap
        holder.setLayout(buttons)
        for group in GROUPS:
            for key, transform in TRANSFORMS.items():
                if transform.group == group:
                    b = button(transform.name, lambda _=False, k=key: self._try(k), size="sm")
                    self.try_buttons[key] = b
                    buttons.addWidget(b)
        layout.addWidget(holder)
        self.result = QTextBrowser()
        field(self.result, "well")
        self.result.setFixedHeight(120)
        self.result.hide()
        layout.addWidget(self.result)
        self.note = caption("", "3")
        self.note.hide()
        layout.addWidget(self.note)
        self.add(trial)

        voice, layout = card(10, (20, 18, 20, 20))
        self.voice = Toggle("Voice commands")
        self.voice.setChecked(s.voice_commands)
        self.voice.toggled.connect(self._apply)
        switch, _, _ = setting_row("Voice commands", "Hold the dictation key and say one. Say only the command; anything "
                                                     "longer is typed as dictated.", self.voice)
        switch.layout().setContentsMargins(0, 0, 0, 4)
        layout.addWidget(switch)
        layout.addWidget(caption("Your own phrases: separate them with commas.", "3"))
        names = {key: transform.name for key, transform in TRANSFORMS.items()} | {UNDO: "Undo"}
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)
        self.phrases: dict[str, QLineEdit] = {}
        for i, key in enumerate(DEFAULT_PHRASES):
            edit = QLineEdit()
            edit.setPlaceholderText("no phrase: this command is off")
            edit.setToolTip(f"What you say for {names[key]}. The defaults: {', '.join(DEFAULT_PHRASES[key])}." + (
                " A language works too, while Translate has a phrase: “translate it into Japanese”, “in English, formally”, "
                "“英語にして”." if key == "translate" else ""))
            edit.editingFinished.connect(self._save_phrases)
            field(edit)
            self.phrases[key] = edit
            grid.addWidget(label(names[key], "rowtitle", wrap=False), i, 0)
            grid.addWidget(edit, i, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        self.phrase_note = text("", "warning")
        layout.addWidget(self.phrase_note)
        self.reset = button("Use the default phrases", self._reset_phrases, link=True, size="sm")
        layout.addLayout(row(self.reset, stretch_at=1))
        self.add(voice)

        frame, layout = card(4, (20, 18, 20, 16))
        layout.addWidget(label("The menu", "heading"))
        self.hotkey = Choice(small=True)
        for label_text, value in TRANSFORM_HOTKEYS:
            self.hotkey.addItem(label_text, value)
        if self.hotkey.findData(s.transform_shortcut) < 0:
            self.hotkey.insertItem(self.hotkey.count() - 1, s.transform_shortcut, s.transform_shortcut)  # set by hand
        self.hotkey.setCurrentIndex(self.hotkey.findData(s.transform_shortcut))
        self.hotkey.currentIndexChanged.connect(self._apply)
        self.hotkey.setMinimumWidth(200)
        shortcut, _, self.how = setting_row("Shortcut", "", self.hotkey)
        layout.addWidget(shortcut)
        layout.addWidget(divider())
        layout.addWidget(caption("In the menu: numbered 1-9 in this order; Translate is T", "2"))
        self.choices: dict[str, Toggle] = {}
        for group in GROUPS:
            heading = _caps_label(group)
            heading.setContentsMargins(0, 10, 0, 0)
            layout.addWidget(heading)
            for key, transform in TRANSFORMS.items():
                if transform.group != group:
                    continue
                box = Toggle(transform.name)
                box.setChecked(key in app.settings.transforms)
                box.toggled.connect(self._apply)
                self.choices[key] = box
                choice, _, _ = setting_row(transform.name, transform.description, box)
                choice.layout().setContentsMargins(0, 8, 0, 8)
                layout.addWidget(choice)
        self.add(frame)

        sender, layout = card(10, (20, 18, 20, 20))
        layout.addWidget(label("Your name for emails", "heading"))
        layout.addWidget(caption("Internal and External email write the line that introduces you (株式会社△△の□□です。) "
                                 "from these. Left empty, that line is left out: Rflow never makes a name up.", "2"))
        self.signature_name, self.signature_company = QLineEdit(s.signature_name), QLineEdit(s.signature_company)
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(8)
        for i, (title, edit, hint) in enumerate((("Name", self.signature_name, "山田 / Yamada"),
                                                 ("Company", self.signature_company, "株式会社〇〇 (for external emails)"))):
            edit.setPlaceholderText(hint)
            edit.editingFinished.connect(self._save_sender)
            field(edit)
            grid.addWidget(label(title, "rowtitle", wrap=False), i, 0)
            grid.addWidget(edit, i, 1)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)
        self.add(sender)
        self.body.addStretch()
        self._show_examples("concise")

    def refresh(self) -> None:
        s, model = self.app.settings, self.app.transform_model()
        shortcut = s.transform_shortcut
        press = "double-tap Ctrl" if shortcut == "double ctrl" else f"press {self.hotkey.currentText()}"
        items = menu_items(s.transforms)
        numbers = sum(hint.isdigit() for _, hint, _ in items)
        letters = "".join(f" or {hint}" for _, hint, _ in items if not hint.isdigit())
        self.how.setText("The menu is off: voice commands still work." if not shortcut else
                         f"Select text, or don't (then all the text in the box is used), {press}, then 1-"
                         f"{max(1, numbers)}{letters} or a click. U undoes the last transform; Esc closes the menu.")
        self.how.show()
        phrases = phrases_for(s.command_phrases)
        for key, edit in self.phrases.items():
            if not edit.hasFocus():  # never rewrite what the user is typing
                edit.setText(", ".join(phrases[key]))
                edit.setCursorPosition(0)  # a long list shows its first phrases, not its last
            edit.setEnabled(s.voice_commands)
        on = bool(shortcut or s.voice_commands)
        for box, value in ((self.voice, s.voice_commands), (self.transform_on, on)):
            box.blockSignals(True)
            box.setChecked(value)
            box.blockSignals(False)
        self.hotkey.blockSignals(True)
        self.hotkey.setCurrentIndex(max(0, self.hotkey.findData(shortcut)))
        self.hotkey.blockSignals(False)
        self.reset.setVisible(bool(s.command_phrases))
        self._check_phrases(phrases)
        self.ai.show_model(model)
        for b in self.try_buttons.values():
            b.setEnabled(bool(model))
        self.switch_words.setText("On: rewrite text by voice or from a menu, in any app. Here's how:" if on else
                                  "Off. Turn it on to rewrite text by voice or from a menu, in any app.")
        self.steps.setVisible(on)
        if on:
            self.steps.set_steps(self._steps(phrases))
        self._show_examples(self.example_key)  # again: its arrows take the theme's colour

    def _steps(self, phrases: dict[str, list[str]]) -> list:
        """How to use it, with the user's own dictation key, first phrase, menu shortcut and menu."""
        s = self.app.settings
        steps: list = [["Select text in any app. Or select nothing: then all the text in the box you're typing in is used "
                        "(if it's not a long document)."]]
        said = phrases.get("concise", [])
        if s.voice_commands and said:
            steps.append(["Hold", keys(key_names(self.app.hotkey_label())), f"and say “{said[0]}”, or another "
                                                                            "command below."])
        if s.transform_shortcut:
            first = "Or" if len(steps) > 1 else "Then"
            press = ([f"{first} double-tap", keys("Ctrl")] if s.transform_shortcut == "double ctrl" else
                     [f"{first} press", _shortcut_caps(s.transform_shortcut)])
            letters = [hint for _, hint, _ in menu_items(s.transforms) if not hint.isdigit()]
            steps.append(([*press, f"for the menu, and press a number{' or ' + letters[0] if letters else ''}:"],
                          self._menu_preview()))
        undo = phrases.get(UNDO, [])
        undo = next((p for p in undo if " " in p), undo[0]) if undo else ""  # "undo that" reads better than "undo"
        steps.append(["Rflow writes the result in place of the text." + (f" Not what you wanted? Say “{undo}”."
                                                                         if s.voice_commands and undo else "")])
        return steps

    def _menu_preview(self) -> QWidget:
        """The menu's transforms as they're numbered in it: [1] Concise  [2] Professional..."""
        holder = QWidget()
        flow = FlowLayout(16)
        holder.setLayout(flow)
        for _, hint, name in menu_items(self.app.settings.transforms):
            pair = QWidget()
            layout = QHBoxLayout(pair)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(8)
            layout.addWidget(keycap(hint, "sm"), 0, Qt.AlignmentFlag.AlignVCenter)
            layout.addWidget(label(name, tone="2", wrap=False), 0, Qt.AlignmentFlag.AlignVCenter)
            flow.addWidget(pair)
        return holder

    def _show_examples(self, key: str) -> None:
        self.example_key = key
        self.example_tabs.set_current(key)
        self.example_words.setText(TRANSFORMS[key].description)
        clear(self.examples)
        for before, after in TRANSFORM_EXAMPLES.get(key, []):
            self.examples.addWidget(Example(before, _light_markdown(after), "Your text", TRANSFORMS[key].name))

    def _switched(self, on: bool) -> None:
        """The whole of Text Transform on or off: the voice commands and the menu (back to double-tap Ctrl)."""
        s = self.app.settings
        shortcut = (s.transform_shortcut or "double ctrl") if on else ""
        self.app.apply_settings(dataclasses.replace(s, transform_shortcut=shortcut, voice_commands=on))
        self.refresh()

    def _apply(self, *_) -> None:
        chosen = [key for key, box in self.choices.items() if box.isChecked()]
        self.app.apply_settings(dataclasses.replace(self.app.settings, transform_shortcut=self.hotkey.currentData(),
                                                    transforms=chosen, voice_commands=self.voice.isChecked()))
        self.refresh()

    def _save_phrases(self) -> None:
        """A phrase box was left: keep the user's phrases where they differ from the defaults (the defaults can then
        improve in an update)."""
        custom = {}
        for key, edit in self.phrases.items():
            said = parse_phrases(edit.text())
            if said != list(DEFAULT_PHRASES[key]):
                custom[key] = ", ".join(said)
        if custom != self.app.settings.command_phrases:
            self.app.apply_settings(dataclasses.replace(self.app.settings, command_phrases=custom))
        self.refresh()

    def _save_sender(self) -> None:
        """The name and company for Japanese emails, as typed (spaces tidied)."""
        name, company = (" ".join(e.text().split()) for e in (self.signature_name, self.signature_company))
        s = self.app.settings
        if (name, company) != (s.signature_name, s.signature_company):
            self.app.apply_settings(dataclasses.replace(s, signature_name=name, signature_company=company))

    def _reset_phrases(self) -> None:
        self.app.apply_settings(dataclasses.replace(self.app.settings, command_phrases={}))
        for edit in self.phrases.values():
            edit.clearFocus()
        self.refresh()

    def _check_phrases(self, phrases: dict[str, list[str]]) -> None:
        """A phrase given to two commands: the first one gets it; say so, rather than surprise the user."""
        seen: dict[str, str] = {}
        twice = []
        names = {key: transform.name for key, transform in TRANSFORMS.items()} | {UNDO: "Undo"}
        for key, options in phrases.items():
            for phrase in options:
                said = normalize(phrase)
                if said in seen and seen[said] != key:
                    twice.append(f"\"{phrase}\" is in {names[seen[said]]} and {names[key]}: {names[seen[said]]} is used.")
                seen.setdefault(said, key)
        self.phrase_note.setText(" ".join(twice))
        self.phrase_note.setVisible(bool(twice))

    def _say(self, message: str) -> None:
        self.note.setText(message)
        self.note.setVisible(bool(message))

    def _try(self, key: str) -> None:
        sample = self.sample.toPlainText().strip()
        if not sample:
            self._say("Type or paste some text first.")
            return
        for b in self.try_buttons.values():
            b.setEnabled(False)
        self._say(f"{TRANSFORMS[key].name}...")

        def done(result, error) -> None:
            self.refresh()
            if error:
                self._say(f"Didn't work: {error}")
            elif not result.accepted:
                self.result.hide()
                self._say("Kept the text: " + "; ".join(result.reasons))
            else:
                self.result.setHtml(result.html or html.escape(result.plain))
                self.result.show()
                self._say(f"{TRANSFORMS[key].name} in {result.seconds:.1f} s" +
                          (f" (checked and fixed once: {result.attempts} answers)" if result.attempts > 1 else ""))
        run_in_background(self, lambda: usage.within("tests", self.app.run_transform, sample, key), done)


# What Translate shows for some selected text (into English, the default): names, numbers and times stay as written.
TRANSLATE_EXAMPLES = [
    ("¿Podrías enviarme el informe actualizado antes del viernes?", "Could you send me the updated report by Friday?"),
    ("来週の定例会議は木曜日の午後3時からに変更になりました。", "Next week's regular meeting has moved to Thursday at 3 PM."),
    ("Danke für die schnelle Antwort! Ich melde mich morgen bei Priya.",
     "Thanks for the quick reply! I'll get in touch with Priya tomorrow."),
]


class TranslatePage(Page):
    """Translate (sst.translateui): on or off, how it's used (select, the shortcut, the popup's keys), examples, the
    shortcut and the languages, and a box to try it in."""

    def __init__(self, app, go_to):
        super().__init__("Translate", "Read or answer text in another language without leaving the app you're in. "
                                      "Names, numbers, dates and links stay as written.")
        self.app = app
        s = app.settings
        self.ai = AiNeeded(go_to, "Translate")
        self.model, self.setup = self.ai.words, self.ai.setup
        self.add(self.ai)

        how, layout = card(0, (20, 4, 20, 4))
        self.translate_on = Toggle("Translate")
        self.translate_on.toggled.connect(self._switched)
        switch, _, self.switch_words = setting_row("Translate", " ", self.translate_on)  # its words: refresh()
        layout.addWidget(switch)
        self.steps = Steps()
        layout.addWidget(self.steps)
        self.add(how)

        examples, layout = card(10, (20, 18, 20, 20))
        layout.addWidget(label("For example, into English", "heading"))
        self.examples = QVBoxLayout()
        self.examples.setSpacing(10)
        layout.addLayout(self.examples)
        self.add(examples)

        frame, layout = card(4, (20, 10, 20, 14))
        self.shortcut = Choice(small=True)
        for label_text, value in TRANSLATE_SHORTCUTS:
            self.shortcut.addItem(label_text, value)
        if self.shortcut.findData(s.translate_shortcut) < 0:
            self.shortcut.insertItem(self.shortcut.count() - 1, s.translate_shortcut, s.translate_shortcut)  # by hand
        self.shortcut.setCurrentIndex(self.shortcut.findData(s.translate_shortcut))
        self.shortcut.currentIndexChanged.connect(self._apply)
        self.shortcut.setMinimumWidth(240)
        layout.addWidget(setting_row("Shortcut", "", self.shortcut)[0])
        layout.addWidget(divider())
        self.target = Choice(search=True, small=True)
        self.target.addItems(list(TRANSLATE_LANGUAGES))
        self.target.setCurrentText(s.translate_to)
        self.target.currentIndexChanged.connect(self._apply)
        self.target.setMinimumWidth(240)
        layout.addWidget(setting_row("Translate into", "", self.target)[0])
        layout.addWidget(divider())
        self.second = Choice(search=True, small=True)
        self.second.addItem("Automatic: Windows' language, or English", "")
        for name in TRANSLATE_LANGUAGES:
            self.second.addItem(name, name)
        self.second.setCurrentIndex(max(0, self.second.findData(s.translate_second)))
        self.second.currentIndexChanged.connect(self._apply)
        self.second.setMinimumWidth(240)
        layout.addWidget(setting_row("Text already in that language goes into", "", self.second)[0])
        self.how = caption("", "2")
        layout.addWidget(self.how)
        self.add(frame)

        trial, layout = card(10, (20, 18, 20, 20))
        layout.addWidget(label("Try it", "heading"))
        self.sample = QPlainTextEdit()
        self.sample.setPlainText("Could you send me the updated report by Friday? The budget is $25,000.")
        field(self.sample, "well")
        fixed_height(self.sample, 72)
        layout.addWidget(self.sample)
        self.try_button = button("Translate", self._try, primary=True, size="sm")
        layout.addLayout(row(self.try_button, stretch_at=1))
        self.result = QTextBrowser()
        field(self.result, "well")
        self.result.setFixedHeight(90)
        self.result.hide()
        layout.addWidget(self.result)
        self.note = caption("", "3")
        self.note.hide()
        layout.addWidget(self.note)
        self.add(trial)
        self.body.addStretch()

    def _second(self) -> str:
        s = self.app.settings
        return s.translate_second or fallback_second(s.translate_to, system_language())

    def refresh(self) -> None:
        s, model, second = self.app.settings, self.app.transform_model(), self._second()
        for box, value in ((self.shortcut, s.translate_shortcut), (self.second, s.translate_second)):
            box.blockSignals(True)
            box.setCurrentIndex(max(0, box.findData(value)))
            box.blockSignals(False)
        self.target.blockSignals(True)
        self.target.setCurrentText(s.translate_to)
        self.target.blockSignals(False)
        on = bool(s.translate_shortcut)
        self.translate_on.blockSignals(True)
        self.translate_on.setChecked(on)
        self.translate_on.blockSignals(False)
        already = f" (text already in {s.translate_to}: in {second})" if second else ""
        self.how.setText("Translate is off." if not on else
                         f"Select text, press {self.shortcut.currentText().split(' (')[0]}: the window shows it in "
                         f"{s.translate_to}{already}. A language at its top translates again; Esc or a click outside "
                         "closes it.")
        self.switch_words.setText("On: translate the text you select, in any app. Here's how:" if on else
                                  "Off. Turn it on to translate the text you select, in any app.")
        self.steps.setVisible(on)
        if on:
            self.steps.set_steps(self._steps(second))
        clear(self.examples)  # again: the arrows take the theme's colour
        for before, after in TRANSLATE_EXAMPLES:
            self.examples.addWidget(Example(before, html.escape(after), "You select", "The window shows"))
        self.ai.show_model(model)
        self.try_button.setEnabled(bool(model))

    def _steps(self, second: str) -> list:
        """How to use it, with the user's own shortcut and languages."""
        s = self.app.settings
        press = (["Press", _shortcut_caps(s.translate_shortcut), "(Ctrl+C twice, quickly)."]
                 if s.translate_shortcut == "ctrl+c+c" else ["Press", _shortcut_caps(s.translate_shortcut),
                                                             "to translate it."])
        already = f" Text already in {s.translate_to} goes into {second}." if second else ""
        return [["Select text in any app: a message, an email, a web page."],
                press,
                [f"A small window at the pointer shows it in {s.translate_to}.{already}"],
                ["Press", keycap("C"), "to copy the translation, or", keycap("↵"), "to put it in place of your text."]]

    def _switched(self, on: bool) -> None:
        s = self.app.settings
        self.app.apply_settings(dataclasses.replace(s, translate_shortcut=(s.translate_shortcut or "ctrl+c+c") if on
                                                    else ""))
        self.refresh()

    def _apply(self, *_) -> None:
        self.app.apply_settings(dataclasses.replace(self.app.settings, translate_shortcut=self.shortcut.currentData(),
                                                    translate_to=self.target.currentText(),
                                                    translate_second=self.second.currentData()))
        self.refresh()

    def _say(self, message: str) -> None:
        self.note.setText(message)
        self.note.setVisible(bool(message))

    def _try(self) -> None:
        sample = self.sample.toPlainText().strip()
        if not sample:
            self._say("Type or paste some text first.")
            return
        s = self.app.settings
        self.try_button.setEnabled(False)
        self._say(f"Translating into {s.translate_to}…")

        def done(result, error) -> None:
            self.refresh()
            if error:
                self.result.hide()
                self._say(f"Didn't work: {error}")
                return
            self.result.setPlainText(result.text)
            self.result.show()
            self._say(f"{result.target} in {result.seconds:.1f} s" + (": " + "; ".join(result.warnings)
                                                                        if result.warnings else ""))
        second = self._second()
        run_in_background(self, lambda: usage.within("tests", self.app.run_translation, sample, s.translate_to, second),
                          done)


# ---------------------------------------------------------------- Formatting

# What the formatting stage writes, shown on its page. The page runs each phrase through the real stage, with the app's
# own policy, so what it shows is what Rflow types; the tests check the outputs too.
FORMAT_EXAMPLES = ["sales went up twenty five percent this quarter", "the budget is twenty five thousand dollars",
                   "it costs nine dollars ninety nine", "let's meet at three thirty pm",
                   "the launch is on october first twenty twenty six", "we need twenty five hundred copies",
                   "send it to john dot smith at gmail dot com"]
FORMAT_KEPT = ["we have two options", "meet me at five", "the first time"]  # prose, or unclear: stays as said
# For each: what a formatter that turned every number word into digits would type, and why Rflow doesn't. The page shows
# it struck out, so a phrase typed exactly as said reads as a choice, not as nothing happening (the owner's question).
FORMAT_KEPT_WHY = {"we have two options": ("we have 2 options", "A small number in a sentence reads better as a word."),
                   "meet me at five": ("meet me at 5:00", "Five what? 5 AM, 5 PM or five minutes: Rflow won't guess."),
                   "the first time": ("the 1st time", "An everyday word, not a date or a list.")}


def formatted(said: str) -> tuple[str, str]:
    """What the formatting stage types for `said`: (the text, the text as HTML with each change in Iris)."""
    result = Formatter(VoiceConfig().formatting).format(said)
    parts, pos = [], 0
    for change in result.changes:
        at = result.text.find(change.replacement, pos)
        if at < 0:
            continue
        parts += [html.escape(result.text[pos:at]),
                  f"<span style='color:{tok('iris').name()}; font-weight:600'>{html.escape(change.replacement)}</span>"]
        pos = at + len(change.replacement)
    parts.append(html.escape(result.text[pos:]))
    return result.text, "".join(parts)


class FormattingPage(Page):
    """Formatting (sst.pipeline.formatting): the switch for writing spoken numbers, money, times and dates the usual
    way, what it changes and what it leaves (run through the real stage), and a box to try it."""

    def __init__(self, app, go_to=None):
        super().__init__("Formatting", "Numbers, amounts of money, times and dates you say are typed the way people "
                                       "write them. Ordinary words stay as you said them.")
        self.app = app
        self.pipeline_off, off = card(12, (16, 12, 12, 12), kind="well", horizontal=True)
        self.off_icon = QLabel()
        off.addWidget(self.off_icon)
        off.addWidget(caption("Formatting is part of the voice pipeline, which is off (Settings, Advanced): your "
                              "dictations are typed as heard.", "2"), 1)
        off.addWidget(button("Turn it on", self._pipeline_on, link=True, size="sm"))
        self.add(self.pipeline_off)

        switch_card, layout = card(0, (20, 4, 20, 4))
        self.format_text = Toggle("Write numbers as numbers")
        self.format_text.toggled.connect(self._apply)
        switch, _, self.switch_words = setting_row("Write numbers as numbers", " ", self.format_text)  # words: refresh()
        layout.addWidget(switch)
        self.add(switch_card)

        changes, layout = card(12, (20, 18, 20, 20))
        layout.addWidget(label("What changes", "heading"))
        self.changed_box = self._table(FORMAT_EXAMPLES)
        layout.addWidget(self.changed_box)
        layout.addSpacing(4)
        layout.addWidget(label("What stays as you said it", "heading"))
        layout.addWidget(caption("Rflow leaves these alone on purpose. Turning every number word into digits would type "
                                 "what's struck out, and change how it reads or what you meant.", "2"))
        self.kept_box = self._kept(FORMAT_KEPT)
        layout.addWidget(self.kept_box)
        self.add(changes)

        trial, layout = card(10, (20, 18, 20, 20))
        layout.addWidget(label("Try it", "heading"))
        self.trial = QLineEdit()
        self.trial.setPlaceholderText("Type what you would say, e.g. the call is at nine am tomorrow")
        field(self.trial)
        self.trial.textChanged.connect(self._try)
        layout.addWidget(self.trial)
        self.trial_result = label("", tone="2")
        self.trial_result.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.trial_result)
        self.add(trial)
        self.body.addStretch()
        self._try()

    def _table(self, phrases: list[str]) -> Card:
        """Phrases as said, and as Rflow types them (its changes in Iris), one per line in a well."""
        box = Card("well", 14)
        grid = QGridLayout(box)
        grid.setContentsMargins(16, 12, 16, 12)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)
        grid.addWidget(_caps_label("You say"), 0, 0)
        grid.addWidget(_caps_label("Rflow types"), 0, 2)
        box.arrows = []
        for i, said in enumerate(phrases, 1):
            grid.addWidget(label(said, tone="2"), i, 0)
            arrow = QLabel()
            box.arrows.append(arrow)
            grid.addWidget(arrow, i, 1)
            typed = label(formatted(said)[1])
            typed.setTextFormat(Qt.TextFormat.RichText)
            typed.setObjectName("typed")
            typed.said = said
            grid.addWidget(typed, i, 2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(2, 1)
        return box

    def _kept(self, phrases: list[str]) -> Card:
        """Phrases Rflow types as said, one well: each with a check, why it's kept, and the wrong version struck out."""
        box = Card("well", 14)
        grid = QGridLayout(box)
        grid.setContentsMargins(16, 12, 16, 14)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(0)
        grid.addWidget(_caps_label("Typed as you said it"), 0, 1)
        grid.addWidget(_caps_label("Not this"), 0, 2)
        grid.setRowMinimumHeight(1, 8)
        box.arrows, box.checks = [], []  # arrows: none here; refresh() colours both kinds of box alike
        for i, said in enumerate(phrases):
            wrong, why = FORMAT_KEPT_WHY[said]
            at = 2 + 3 * i
            if i:
                grid.setRowMinimumHeight(at - 1, 12)
            check = QLabel()
            box.checks.append(check)
            grid.addWidget(check, at, 0, Qt.AlignmentFlag.AlignVCenter)
            typed = label(formatted(said)[1])
            typed.setTextFormat(Qt.TextFormat.RichText)
            typed.setObjectName("typed")
            typed.said = said
            grid.addWidget(typed, at, 1)
            naive = label(f"<s>{html.escape(wrong)}</s>", tone="3")
            naive.setTextFormat(Qt.TextFormat.RichText)
            naive.setObjectName("naive")
            grid.addWidget(naive, at, 2)
            grid.addWidget(caption(why, "3"), at + 1, 1, 1, 2)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        return box

    def refresh(self) -> None:
        s = self.app.settings
        self.format_text.blockSignals(True)
        self.format_text.setChecked(s.format_text)
        self.format_text.blockSignals(False)
        self.switch_words.setText("On: “twenty five percent” is typed as 25%." if s.format_text else
                                  "Off: numbers are typed in words, as you said them.")
        self.pipeline_off.setVisible(not s.voice_pipeline)
        self.off_icon.setPixmap(theme.icon_pixmap("info", tok("warn").name(), 16, self.devicePixelRatioF()))
        for box in (self.changed_box, self.kept_box):  # the theme's colours: the arrows, the changes in Iris
            for arrow in box.arrows:
                arrow.setPixmap(theme.icon_pixmap("arrow-right", tok("text3").name(), 14, self.devicePixelRatioF()))
            for check in getattr(box, "checks", []):
                check.setPixmap(theme.icon_pixmap("check", tok("ok").name(), 16, self.devicePixelRatioF()))
            for typed in box.findChildren(QLabel, "typed"):
                typed.setText(formatted(typed.said)[1])
        self._try()

    def _apply(self, *_) -> None:
        self.app.apply_settings(dataclasses.replace(self.app.settings, format_text=self.format_text.isChecked()))
        self.refresh()

    def _pipeline_on(self) -> None:
        self.app.apply_settings(dataclasses.replace(self.app.settings, voice_pipeline=True))
        self.refresh()

    def _try(self, *_) -> None:
        said = self.trial.text().strip()
        if not said:
            self.trial_result.setText("What Rflow would type appears here.")
        elif not self.app.settings.format_text:
            self.trial_result.setText("Formatting is off: typed as you said it.")
        else:
            typed, shown = formatted(said)
            self.trial_result.setText(f"Rflow types: {shown}" if typed != said else "Nothing to change: typed as you "
                                                                                    "said it.")


# ---------------------------------------------------------------- Report a problem

def version_info() -> str:
    """What a bug report needs to know about this Rflow: its version, Windows' and the kind of computer (x64, or x64
    emulated on ARM64). Nothing the user wrote, said or set."""
    source = "" if getattr(sys, "frozen", False) else " (from the source code)"
    try:
        build = sys.getwindowsversion().build
        windows = f"Windows {'11' if build >= 22000 else platform.release()} (build {build})"
    except AttributeError:  # not Windows: the tests on another system
        windows = platform.platform()
    return f"{APP_NAME} {__version__}{source}\n{windows}\n{machine()}"


def open_link(url: str) -> None:
    shell.open_url(QUrl(url))


def copy_text(value: str) -> None:
    QGuiApplication.clipboard().setText(value)  # apart, so the tests never touch the clipboard


class ReportPage(Page):
    """Report a problem: a new issue on GitHub, the version information to paste into it, and the logs, with a plain
    word on what they can hold."""

    def __init__(self, app, go_to=None):
        super().__init__("Report a problem", "Something doesn't work as it should? Tell us on GitHub: what you did, "
                                             "what you expected, and what happened instead.")
        self.app = app
        issue, layout = card(8, (20, 18, 20, 20))
        layout.addWidget(label("Open an issue on GitHub", "heading"))
        layout.addWidget(caption("It needs a free GitHub account. Issues are public: anyone can read what you write, "
                                 "so leave out anything private."))
        layout.addSpacing(4)
        self.issue_button = button("Open a GitHub issue", lambda: open_link(REPO + "/issues/new/choose"), primary=True,
                                   icon="external", icon_after=True)
        layout.addLayout(row(self.issue_button, stretch_at=1))
        self.add(issue)

        version, layout = card(10, (20, 18, 20, 20))
        layout.addWidget(label("Version information", "heading"))
        layout.addWidget(caption("Paste it into your report: it says which Rflow, which Windows and which kind of "
                                 "computer. Nothing you dictated or set is in it."))
        well, inner = card(0, (16, 12, 16, 12), kind="well")
        self.version = label(version_info(), selectable=True)
        self.version.setFont(font(13, 500, mono=True))
        inner.addWidget(self.version)
        layout.addWidget(well)
        self.copy_button = button("Copy version info", self._copy, icon="copy")
        layout.addLayout(row(self.copy_button, stretch_at=1))
        self.add(version)

        logs, layout = card(10, (20, 18, 20, 20))
        layout.addWidget(label("Logs", "heading"))
        layout.addWidget(caption("What Rflow did, and the errors it met, kept on this PC. They help find a problem, "
                                 "and a crash leaves its report there too."))
        warning, line = card(12, (16, 12, 16, 12), kind="well", horizontal=True)
        self.warn_icon = QLabel()
        line.addWidget(self.warn_icon, 0, Qt.AlignmentFlag.AlignTop)
        self.privacy = label("The logs can contain text you dictated, your words and snippets, and the names of your "
                             "microphones. Read a log before you share it, remove anything private, and never attach "
                             "it to a public issue as it is.", tone="warn")
        line.addWidget(self.privacy, 1)
        layout.addWidget(warning)
        self.logs_button = button("Open logs folder", lambda: open_folder(LOG_DIR), icon="folder")
        layout.addLayout(row(self.logs_button, stretch_at=1))
        self.add(logs)
        self.body.addStretch()

    def refresh(self) -> None:
        self.version.setText(version_info())
        self.warn_icon.setPixmap(theme.icon_pixmap("warning", tok("warn").name(), 16, self.devicePixelRatioF()))

    def _copy(self) -> None:
        copy_text(version_info())
        self._show_copied(True)
        QTimer.singleShot(1500, lambda: self._show_copied(False))

    def _show_copied(self, copied: bool) -> None:
        try:
            self.copy_button.icon_name = "check" if copied else "copy"
            self.copy_button.setText("Copied" if copied else "Copy version info")
            self.copy_button.update()
        except RuntimeError:  # the window was built again meanwhile (another profile)
            pass


# ---------------------------------------------------------------- Settings

MIC_READY = [("While Rflow runs", "always"), ("5 minutes after dictating", "warm"), ("Only while dictating", "off")]


def parse_end_phrases(value: str) -> list[str]:
    """The end phrases as typed, separated by commas: trimmed, each once; none typed is the default, "stop it"."""
    phrases = []
    for part in value.split(","):
        phrase = " ".join(part.split())
        if phrase and phrase.casefold() not in (p.casefold() for p in phrases):
            phrases.append(phrase)
    return phrases or ["stop it"]


def risky_end_phrases(phrases: list[str]) -> list[str]:
    """End phrases said in ordinary dictation, which would end it early: "done" fired on 23 of 24 real ones (spike)."""
    common = {"done", "im done", "finished", "end", "stop", "thats it", "over", "ok", "okay", "thanks", "thank you"}
    return [p for p in phrases if " ".join(re.sub(r"[^\w\s]", "", p.casefold()).split()) in common]


class WakeCard:
    """Settings' "Start with your voice" (sst.wakeui): the switch; when on, where it listens, the models it needs, the
    keyword and its 10-try setup, the end phrases, the voice check, and a word on privacy and battery. Its status line
    is told when the wake word starts, stops or pauses (MainWindow.show_wake): nothing polls."""

    def __init__(self, app):
        self.app = app
        s = app.settings
        self.frame, layout = card(0, (20, 4, 20, 4))
        self.toggle = Toggle("Start with your voice")
        self.toggle.setChecked(s.wake_enabled)
        self.toggle.toggled.connect(self._switched)
        layout.addWidget(setting_row("Start with your voice", "Say your keyword in any app, speak, then say your end "
                                     "phrase: the text is typed where the cursor is. Ctrl+Win keeps working.",
                                     self.toggle)[0])
        self.details = QWidget()
        details = QVBoxLayout(self.details)
        details.setContentsMargins(0, 0, 0, 0)
        details.setSpacing(0)
        status = QWidget()
        status_layout = QHBoxLayout(status)
        status_layout.setContentsMargins(0, 0, 0, 14)
        status_layout.setSpacing(8)
        self.lamp = Lamp("off")
        self.status = caption("", "2")
        status_layout.addWidget(self.lamp, 0, Qt.AlignmentFlag.AlignTop)
        status_layout.addWidget(self.status, 1)
        details.addWidget(status)

        self.models = QWidget()
        models = QVBoxLayout(self.models)
        models.setContentsMargins(0, 0, 0, 0)
        models.setSpacing(0)
        models.addWidget(divider())
        self.progress = WaveProgress(24)
        self.download = button("Download", lambda _=False: app.download_wake_models(), primary=True, size="sm",
                               icon="download")
        self.cancel = button("Cancel", lambda _=False: app.cancel_wake_download(), kind="quiet", size="sm")
        models_row, _, self.models_caption = setting_row("Voice models", "", self.progress, self.cancel, self.download)
        models.addWidget(models_row)
        details.addWidget(self.models)

        details.addWidget(divider())
        self.setup = button("Set up", lambda _=False: self._set_up(), size="sm")
        keyword_row, _, self.keyword = setting_row("Keyword", "", self.setup)
        details.addWidget(keyword_row)
        details.addWidget(divider())
        self.phrases = QLineEdit(", ".join(s.wake_end_phrases))
        self.phrases.setPlaceholderText("stop it")
        self.phrases.setMinimumWidth(200)
        self.phrases.editingFinished.connect(self._save_phrases)
        field(self.phrases)
        phrases_row, _, self.phrases_caption = setting_row("End phrases", "", self.phrases)
        details.addWidget(phrases_row)
        details.addWidget(divider())
        self.voice = Toggle("Only my voice")
        self.voice.setChecked(s.wake_voice_check)
        self.voice.toggled.connect(self._voice_toggled)
        for words in (self.models_caption, self.keyword, self.phrases_caption):
            words.show()  # made empty, so hidden: refresh() fills them
        details.addWidget(setting_row("Only my voice", "It starts only for the voice it learned in the setup, not for a "
                                      "colleague or a video.", self.voice)[0])
        details.addWidget(divider())
        self.bluetooth = Toggle("Listen on my Bluetooth headset")
        self.bluetooth.setChecked(s.wake_bluetooth)
        self.bluetooth.toggled.connect(self._bluetooth_toggled)
        details.addWidget(setting_row("Listen on my Bluetooth headset", "When your headset is the microphone, listen on it "
                                      "even when you aren't dictating. The headset plays in call quality meanwhile.",
                                      self.bluetooth)[0])
        details.addWidget(divider())
        privacy = caption("Private: Rflow listens for the keyword on this laptop, and nothing is kept or sent before it "
                          "hears it. The microphone stays on meanwhile (Windows shows its icon), which costs a little "
                          "battery: about 2% of one core.", "3")
        privacy.setContentsMargins(0, 12, 0, 14)
        details.addWidget(privacy)
        layout.addWidget(self.details)
        self.refresh()

    def refresh(self) -> None:
        s, state = self.app.settings, self.app.wake_state()
        self.toggle.blockSignals(True)
        self.toggle.setChecked(s.wake_enabled)
        self.toggle.blockSignals(False)
        for toggle, on in ((self.voice, s.wake_voice_check), (self.bluetooth, s.wake_bluetooth)):
            toggle.blockSignals(True)
            toggle.setChecked(on)
            toggle.blockSignals(False)
        self.details.setVisible(s.wake_enabled)
        if not self.phrases.hasFocus():
            self.phrases.setText(", ".join(s.wake_end_phrases))
        risky = risky_end_phrases(s.wake_end_phrases)
        self.phrases_caption.setText(
            f"“{risky[0]}” ends a dictation whenever you say the word: choose words you never dictate, like “stop it”."
            if risky else "Say one to type the text; Rflow waits for a short pause after it. Separate several with "
                          "commas. Not “done”: it's said too often.")
        set_tone(self.phrases_caption, "warn" if risky else "2")
        size = f" ({_size(state.size)})" if state.size else ""
        self.models.setVisible(state.models != "ready")
        if state.models == "downloading":
            self.models_caption.setText(f"Downloading: {state.percent}%")
            self.progress.set_value(state.percent / 100)
        elif state.problem:
            self.models_caption.setText(f"The download didn't finish ({state.problem}). Try again: it goes on from where "
                                        "it stopped.")
        else:
            self.models_caption.setText("The keyword spotter and the voice check run on this laptop: downloaded once"
                                        f"{size}.")
        downloading = state.models == "downloading"
        self.progress.setVisible(downloading)
        self.cancel.setVisible(downloading)
        self.download.setVisible(not downloading)
        self.download.setText(f"Download{size}")
        if state.keyword:
            self.keyword.setText(f"“{state.keyword}”  ·  heard {state.heard} of {state.said or state.tries} in the setup")
        else:
            self.keyword.setText(f"Not set up yet: say it {state.tries} times, and Rflow learns it and your voice.")
        self.setup.setText("Set up again" if state.keyword else "Set up")
        self.setup.setEnabled(state.models == "ready")
        self.setup.setToolTip("" if state.models == "ready" else "Download the voice models first.")
        self._show_status(state)

    def show_status(self) -> None:
        if self.frame.isVisible():
            self._show_status(self.app.wake_state())

    def _show_status(self, state) -> None:
        if state.models == "downloading":
            words, lamp = f"Downloading the voice models: {state.percent}%.", "warn"
        elif state.note:
            words, lamp = state.note, "warn"
        elif state.listening:
            words, lamp = (f"Listening for “{state.keyword}” on {state.listening}. Say it, speak, then say "
                           f"“{self.app.settings.wake_end_phrases[0]}”."), "ok"
            if state.instead:  # dictation's headset isn't kept open: say so, and how to listen there
                words += (f" Not on {state.instead}: turn on “Listen on my Bluetooth headset” to use it.")
        else:
            words, lamp = "Not listening yet.", "off"
        self.status.setText(words)
        self.lamp.set_state(lamp)

    def _switched(self, on: bool) -> None:
        self.app.set_wake_enabled(on)
        self.refresh()

    def _voice_toggled(self, on: bool) -> None:
        self.app.apply_settings(dataclasses.replace(self.app.settings, wake_voice_check=on))

    def _bluetooth_toggled(self, on: bool) -> None:
        self.app.apply_settings(dataclasses.replace(self.app.settings, wake_bluetooth=on))

    def _save_phrases(self) -> None:
        phrases = parse_end_phrases(self.phrases.text())
        self.phrases.setText(", ".join(phrases))
        if phrases != self.app.settings.wake_end_phrases:
            self.app.apply_settings(dataclasses.replace(self.app.settings, wake_end_phrases=phrases))
        self.refresh()

    def _set_up(self) -> None:
        self.app.open_wake_setup(self.frame.window())
        self.refresh()


class SettingsPage(Page):
    def __init__(self, app, go_to=None):
        go_to = go_to or (lambda page: None)
        super().__init__("Settings", "Changes save as you make them.")
        self.app = app
        s: Settings = app.settings

        everyday, layout = card(0, (20, 4, 20, 4))
        self.hotkey = Choice(small=True)
        for label_text, value in HOTKEY_CHOICES:
            self.hotkey.addItem(label_text, value)
        if self.hotkey.findData(s.hotkey) < 0:
            self.hotkey.addItem(s.hotkey, s.hotkey)  # a custom one set with --hotkey or by hand
        self.hotkey.setCurrentIndex(self.hotkey.findData(s.hotkey))
        self.hotkey.setMinimumWidth(220)
        self.hotkey_caps = QWidget()
        self.hotkey_caps_layout = QHBoxLayout(self.hotkey_caps)
        self.hotkey_caps_layout.setContentsMargins(0, 0, 0, 0)
        key_row, _, _ = setting_row("Dictation key", "Hold to talk, tap for hands-free. With Ctrl+Win, Ctrl+Win+Space is "
                                                     "hands-free as well.", self.hotkey_caps, self.hotkey)
        layout.addWidget(key_row)
        layout.addWidget(divider())
        self.sounds = Toggle("Sounds")
        self.sounds.setChecked(s.sounds)
        layout.addWidget(setting_row("Sounds", "A soft click when recording starts and stops", self.sounds)[0])
        layout.addWidget(divider())
        self.start_with_windows = Toggle("Start when I sign in")
        self.start_with_windows.setChecked(starts_with_windows())
        self.start_with_windows.setEnabled(can_start_with_windows())
        layout.addWidget(setting_row("Start when I sign in", "Rflow waits quietly in the tray" if can_start_with_windows()
                                     else "Available in the installed app", self.start_with_windows)[0])
        layout.addWidget(divider())
        self.appearance = Segmented(APPEARANCE_CHOICES, icons=APPEARANCE_ICONS)
        self.appearance.set_current(s.appearance)
        self.appearance.changed.connect(lambda name: getattr(self.window(), "set_appearance", lambda _: None)(name))
        layout.addWidget(setting_row("Appearance", "Light or dark, or as Windows is set. Also in the bar at the top.",
                                     self.appearance)[0])
        layout.addWidget(divider())
        self.reading = link_row("Reading test", "Read 30 sentences aloud: how many words does Rflow get wrong?",
                                lambda _=False: go_to("reading"))
        layout.addWidget(self.reading)
        self.add(everyday)
        self.wake = WakeCard(app)  # starting with the voice, next to starting with the keys
        self.add(self.wake.frame)

        privacy, layout = card(0, (20, 4, 20, 4))
        self.save_recordings = Toggle("Keep recordings on this PC")
        self.save_recordings.setChecked(s.save_recordings)
        layout.addWidget(setting_row("Keep recordings on this PC", "Audio and text stay in your folder, never uploaded",
                                     button("Open folder", lambda: open_folder(RECORDINGS_DIR), link=True, size="sm"),
                                     self.save_recordings)[0])
        layout.addWidget(divider())
        self.mic_ready = Choice(small=True)
        for label_text, value in MIC_READY:
            self.mic_ready.addItem(label_text, value)
        self.mic_ready.setCurrentIndex(self.mic_ready.findData("always" if s.always_on_mic else "warm" if s.warm_mic
                                                               else "off"))
        self.mic_ready.setMinimumWidth(220)
        self.mic_ready.setToolTip("Dictation starts at once and keeps the moment before you pressed the key, so a word "
                                  "you began early isn't cut off. Those seconds stay in memory and are replaced all the "
                                  "time: nothing is kept or sent until you press the key. Never done for Bluetooth "
                                  "headsets.")
        layout.addWidget(setting_row("Microphone stays ready", "Your first word is never lost. Windows shows its "
                                     "microphone icon meanwhile.", self.mic_ready)[0])
        layout.addWidget(divider())
        self.update_lamp = Lamp("ok")
        self.update_status = caption("Rflow checks for updates by itself and tells you when one is ready.", "2")
        version = QVBoxLayout()
        version.setSpacing(2)
        version.addWidget(label(f"{APP_NAME} {__version__}", "rowtitle"))
        version.addLayout(row(self.update_lamp, self.update_status, spacing=8))
        about = QWidget()
        about_layout = QHBoxLayout(about)
        about_layout.setContentsMargins(0, 14, 0, 14)
        about_layout.setSpacing(16)
        about_layout.addLayout(version, 1)
        about_layout.addWidget(button("Check for updates", lambda: app.check_for_updates(manual=True), size="sm"))
        layout.addWidget(about)
        self.add(privacy)

        self.advanced = advanced_row("Advanced", "Voice pipeline, troubleshooting, profiles, logs, source code",
                                     self._toggle_advanced)
        self.add(self.advanced)
        self.advanced_card, layout = card(0, (20, 4, 20, 4))
        self.voice_pipeline = Toggle("Voice pipeline")
        self.voice_pipeline.setChecked(s.voice_pipeline)
        self.voice_pipeline.setToolTip("Long dictations are cut at your pauses and transcribed while you are still "
                                       "speaking, so the text is ready sooner. Then your dictionary, the formatting "
                                       "and the AI cleanup run, and a check keeps the AI from changing numbers, names "
                                       "or meaning. Off: the whole recording is transcribed at once, as before.")
        layout.addWidget(setting_row("Voice pipeline", "Transcribe in parts while you speak, then fix, format and check "
                                     "the text", self.voice_pipeline)[0])
        layout.addWidget(divider())
        self.debug_pipeline = Toggle("Keep each dictation's steps")
        self.debug_pipeline.setChecked(s.debug_pipeline)
        self.debug_pipeline.setToolTip("Saves the text after each step, and the audio parts, in a folder on this "
                                       "laptop. Turn it off when done: it keeps your voice.")
        layout.addWidget(setting_row("Keep each dictation's steps", "For troubleshooting. It keeps your voice: turn it "
                                     "off when done.", button("Open folder", lambda: open_folder(LOG_DIR.parent / "debug"),
                                                              link=True, size="sm"), self.debug_pipeline)[0])
        layout.addWidget(divider())
        self.raw_audio = Toggle("Turn off Windows' voice effects")
        self.raw_audio.setChecked(s.raw_audio)
        self.raw_audio.setToolTip("Records the microphone as it is, without Windows' or the driver's noise suppression "
                                  "and gain. Try it with the Reading test: it may help or hurt, depending on the "
                                  "microphone and the room.")
        layout.addWidget(setting_row("Turn off Windows' voice effects", "The microphone as it is: no noise suppression "
                                     "or gain. Check it with the Reading test.", self.raw_audio)[0])
        for title, words, on_click in [
                ("Profiles", "One setup per person sharing this PC", lambda: go_to("profiles")),
                ("Logs", "Open the folder with Rflow's logs", lambda: open_folder(LOG_DIR)),
                ("Website", WEBSITE.removeprefix("https://"), lambda: shell.open_url(QUrl(WEBSITE))),
                ("Source code", "Free and open source, MIT License", lambda: shell.open_url(QUrl(REPO)))]:
            layout.addWidget(divider())
            layout.addWidget(link_row(title, words, lambda _=False, f=on_click: f()))
        self.advanced_card.hide()
        self.add(self.advanced_card)

        # the very end: Rflow as if just installed (sst.app.TrayApp.start_over)
        start_over, layout = card(0, (20, 4, 20, 4))
        self.start_over = button("Start over…", self._start_over, kind="danger", size="sm")
        layout.addWidget(setting_row("Start over", "Delete everything Rflow keeps on this PC, for every profile, and begin "
                                     "again with the welcome, like a new install.", self.start_over)[0])
        self.add(start_over)
        self.body.addStretch()

        self.hotkey.currentIndexChanged.connect(self._apply)
        self.mic_ready.currentIndexChanged.connect(self._apply)
        for box in (self.voice_pipeline, self.debug_pipeline, self.raw_audio, self.sounds, self.save_recordings):
            box.toggled.connect(self._apply)
        self.start_with_windows.toggled.connect(lambda on: set_start_with_windows(on) if can_start_with_windows() else None)
        self._show_keys()

    def confirm_start_over(self) -> bool | None:
        """Asks before Start over, saying plainly what goes: None to cancel, else whether to keep the downloaded speech
        models (a dialog; the tests replace it)."""
        box, keep, delete = self.start_over_box()
        box.exec()
        return keep.isChecked() if box.clickedButton() is delete else None

    def start_over_box(self) -> tuple[QMessageBox, QCheckBox, QPushButton]:
        box = QMessageBox(self)
        box.setWindowTitle(APP_NAME)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setText("Start over and delete everything Rflow keeps on this PC?")
        box.setInformativeText("For every profile: the settings, your words and snippets, the dictation history and "
                               "stats, the API keys, the recordings, the reading tests and the live translation "
                               "transcripts. Rflow's logs too. This can't be undone.\n\nRflow then restarts with the "
                               "welcome, like a new install.")
        keep = QCheckBox("Keep the downloaded speech models")
        c = {name: theme.css(tok(name)) for name in ("rim", "well", "primary")}
        keep.setStyleSheet(  # a box that shows unticked too (Windows' own is barely visible on these colours)
            f"QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {c['rim']}; border-radius: 5px; "
            f"background: {c['well']}; }} QCheckBox::indicator:checked {{ background: {c['primary']}; "
            f"border-color: {c['primary']}; image: url({(UI_IMAGES / 'check.png').as_posix()}); }}")
        keep.setChecked(True)
        keep.setToolTip("Parakeet, Whisper and the live translation voice stay: no need to download them again.")
        box.setCheckBox(keep)
        delete = box.addButton("Delete everything and restart", QMessageBox.ButtonRole.DestructiveRole)
        delete.setStyleSheet(f"QPushButton {{ color: {theme.css(tok('err'))}; }}")  # in red, like the button that asked
        cancel = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel)
        return box, keep, delete

    def _start_over(self) -> None:
        keep_models = self.confirm_start_over()
        if keep_models is not None:
            self.app.start_over(keep_models)

    def _toggle_advanced(self) -> None:
        open_ = self.advanced_card.isHidden()
        self.advanced_card.setVisible(open_)
        self.advanced.button.icon_name = "chevron-down" if open_ else "chevron-right"
        self.advanced.button.update()

    def _show_keys(self) -> None:
        clear(self.hotkey_caps_layout)
        try:
            label_text = parse_hotkey(self.hotkey.currentData()).label
        except ValueError:
            label_text = self.hotkey.currentText()
        self.hotkey_caps_layout.addWidget(keys(key_names(label_text)))

    def result(self, current: Settings) -> Settings:
        """The current settings with this page's choices (the other pages own the rest)."""
        ready = self.mic_ready.currentData()
        warm = ready == "warm" or (ready == "always" and current.warm_mic)  # always on keeps the warm choice
        return dataclasses.replace(current, hotkey=self.hotkey.currentData(), sounds=self.sounds.isChecked(),
                                   save_recordings=self.save_recordings.isChecked(),
                                   always_on_mic=ready == "always", warm_mic=warm,
                                   raw_audio=self.raw_audio.isChecked(), voice_pipeline=self.voice_pipeline.isChecked(),
                                   debug_pipeline=self.debug_pipeline.isChecked())

    def refresh(self) -> None:
        s = self.app.settings
        for box, on in ((self.sounds, s.sounds), (self.voice_pipeline, s.voice_pipeline),
                        (self.debug_pipeline, s.debug_pipeline), (self.raw_audio, s.raw_audio),
                        (self.save_recordings, s.save_recordings)):
            box.blockSignals(True)
            box.setChecked(on)
            box.blockSignals(False)
        self.wake.refresh()

    def _apply(self, *_) -> None:
        self._show_keys()
        self.app.apply_settings(self.result(self.app.settings))  # changes apply at once, like a phone's settings


# ---------------------------------------------------------------- the first-run welcome

PROVIDER_TILES = [tile for tile in [("gemini", "Gemini", "Free key"), ("groq", "Groq", "Free key, fastest"),
                                    ("openai", "OpenAI", "Pay as you go"), ("anthropic", "Anthropic", "Pay as you go"),
                                    ("ollama", "Ollama", "Runs on this PC")] if offered(tile[0])]  # sst.gateway.OFFERED
# The model the welcome picks for each provider from its own list, as the research notes of 2026-10-06 say (the newest
# match of the first pattern that finds one): the pinned id first. Groq's llama-3.1-8b-instant was shut down on
# 2026-08-16; gpt-oss-20b replaces it. OpenAI's "any mini" could pick a pricier or retiring one (o4-mini).
FAST_MODELS = {"gemini": [r"^gemini-3\.5-flash-lite$", r"flash-lite"], "openai": [r"^gpt-4\.1-mini$", r"^gpt-4o-mini$"],
               "anthropic": [r"^claude-haiku-4-5", r"haiku"], "groq": [r"^openai/gpt-oss-20b$", r"gpt-oss-20b"],
               "ollama": [r"."]}
# Not chat models: speech, text-to-speech (Groq's Orpheus), images, classifiers (prompt guards, safeguards), previews
_NOT_CHAT = re.compile(r"embed|tts|orpheus|audio|speech|image|vision|live|transcri|whisper|guard|moderation|search|"
                       r"realtime|preview|exp", re.IGNORECASE)


def fast_model(provider: str, models: list[str]) -> str:
    """The provider's fast chat model for cleanup (a Flash-Lite, a mini, a Haiku, gpt-oss...), the newest version first;
    else its cheapest chat model with a known price; else, only on a server of the user's own, its first chat model.
    Never a speech, text-to-speech or classifier model: "" when there is no chat model to pick."""
    chat = [m for m in models if not _NOT_CHAT.search(m)]

    def version(name: str) -> tuple:
        return tuple(int(n) for n in re.findall(r"\d+", name)[:3])
    for pattern in FAST_MODELS.get(provider, [r"."]):
        found = sorted((m for m in chat if re.search(pattern, m, re.IGNORECASE)), key=version, reverse=True)
        if found:
            return found[0]
    cheapest = costs.cheapest_known(provider, chat)
    if cheapest:
        return cheapest
    own = provider in PROVIDERS and PROVIDERS[provider].own_server
    return chat[0] if chat and own else ""


class OptionCard(Card):
    """A choice as a card: raised, or pressed in with an Iris edge when chosen. clicked()."""

    clicked = Signal()

    def __init__(self, kind: str = "card"):
        super().__init__(kind)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chosen = False

    def set_chosen(self, chosen: bool) -> None:
        self.chosen = chosen
        self.set_kind("chosen" if chosen else "card")

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


class Stepper(QWidget):
    """The steps of the welcome: done ones with a check, the current one lit. Their names follow the setup chosen."""

    def __init__(self, steps: list[str]):
        super().__init__()
        self.row = QHBoxLayout(self)
        self.row.setContentsMargins(0, 0, 0, 0)
        self.row.setSpacing(12)
        self.dots: list[QLabel] = []
        self.names: list[QLabel] = []
        self.set_names(steps)

    def set_names(self, steps: list[str]) -> None:
        if [words.text() for words in self.names] == list(steps):
            return
        clear(self.row)
        self.dots, self.names = [], []
        for i, name in enumerate(steps):
            if i:
                line = divider()
                line.setFixedWidth(32)
                self.row.addWidget(line, 0, Qt.AlignmentFlag.AlignVCenter)
            dot = QLabel(str(i + 1))
            dot.setAlignment(Qt.AlignmentFlag.AlignCenter)
            dot.setFixedSize(24, 24)
            dot.setFont(font(12, 600, mono=True))
            words = caption(name, "3", wrap=False)
            self.row.addWidget(dot)
            self.row.addWidget(words)
            self.dots.append(dot)
            self.names.append(words)

    def set_step(self, step: int) -> None:
        for i, (dot, words) in enumerate(zip(self.dots, self.names, strict=True)):
            if i < step:
                dot.setText("✓")
                dot.setStyleSheet(f"background: {theme.css(tok('well'))}; color: {theme.css(tok('ok'))}; "
                                  "border-radius: 12px;")
                set_tone(words, "2")
                words.setFont(font(12, 500))
            elif i == step:
                dot.setText(str(i + 1))
                dot.setStyleSheet(f"background: {theme.css(tok('primary'))}; color: {theme.css(tok('on_primary'))}; "
                                  "border-radius: 12px;")
                set_tone(words, None)
                words.setFont(font(13, 600))
            else:
                dot.setText(str(i + 1))
                dot.setStyleSheet(f"background: {theme.css(tok('well'))}; color: {theme.css(tok('text3'))}; "
                                  "border-radius: 12px;")
                set_tone(words, "3")
                words.setFont(font(12, 500))


# The welcome's steps for each setup, and their names on the stepper
WELCOME_FLOWS = {"recommended": ["setup", "key", "try"], "fastest": ["setup", "key", "try"],
                 "multilingual": ["setup", "key", "try"], "local": ["setup", "check", "try"],
                 "custom": ["setup", "hear", "try", "ai"]}
WELCOME_STEPS = {"setup": "Setup", "key": "Key", "check": "This PC", "hear": "Hear you", "try": "Try it", "ai": "AI"}


class WelcomePage(QWidget):
    """The first run. First a setup (sst.setups: Recommended first, Multilingual, Local or Custom), then what it
    needs: its key (checked with the provider, then saved), or for Local a check that this PC can run the model; then a
    first dictation. Custom keeps the steps of before: how Rflow hears you (Parakeet on this PC, or a cloud model), a
    first dictation, and an optional AI connection (the model is chosen for the user)."""

    def __init__(self, app, go_to):
        super().__init__()
        self.app, self.go_to = app, go_to
        self.step, self.setup, self.choice, self.provider = 0, "recommended", "local", "gemini"
        self.local_choice = "parakeet"
        root = Host("base")
        root.setObjectName("page")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        head = QGridLayout()
        head.setContentsMargins(40, 24, 40, 16)
        brand = QHBoxLayout()
        brand.setSpacing(12)
        self.mark = Mark()
        brand.addWidget(self.mark)
        brand.addWidget(label(APP_NAME, "wordmark", wrap=False))
        brand.addStretch()
        head.addLayout(brand, 0, 0)
        self.stepper = Stepper([WELCOME_STEPS[name] for name in self.flow])
        head.addWidget(self.stepper, 0, 1, Qt.AlignmentFlag.AlignCenter)
        self.skip = button("Skip setup", self.finish, kind="quiet", size="sm")
        head.addWidget(self.skip, 0, 2, Qt.AlignmentFlag.AlignRight)
        for column in (0, 2):
            head.setColumnStretch(column, 1)
        layout.addLayout(head)

        self.steps = QStackedWidget()
        self.step_pages: dict[str, QWidget] = {}
        for name, build in (("setup", self._setups), ("key", self._key), ("check", self._check), ("hear", self._hear),
                            ("try", self._try), ("ai", self._connect)):
            self.step_pages[name] = build()
            self.steps.addWidget(self.step_pages[name])
        layout.addWidget(self.steps, 1)

        layout.addWidget(divider())
        foot = QHBoxLayout()
        foot.setContentsMargins(40, 14, 40, 14)
        foot.setSpacing(12)
        self.back = button("Back", lambda: self.show_step(self.step - 1), kind="quiet")
        self.foot_note = caption("", "3", wrap=False)
        self.secondary = button("", self._secondary)
        self.primary = button("", self._primary, primary=True, size="lg")
        for widget in (self.back, self.foot_note):
            foot.addWidget(widget)
        foot.addStretch()
        foot.addWidget(self.secondary)
        foot.addWidget(self.primary)
        footer = QWidget()
        footer.setLayout(foot)
        footer.setFixedHeight(72)
        layout.addWidget(footer)
        self.title = label("Welcome to Rflow")  # for the window's "unsaved changes" question (never asked here)
        self.ready = False
        self.show_step(0)

    @property
    def flow(self) -> list[str]:
        return WELCOME_FLOWS[self.setup]

    def current(self) -> str:
        """The step shown: setup, key, check, hear, try or ai."""
        return self.flow[self.step]

    # -- step 1: a setup

    def _setups(self) -> QWidget:
        scroll = QScrollArea()  # a small window scrolls the setups rather than squeezing them
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        page = Host()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 0, 40, 16)
        layout.setSpacing(8)
        layout.addStretch()
        title = label("How should Rflow work?", "display")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(title)
        sub = label("Pick a setup: each is the cheapest that does the job well. You can change any part later on AI & "
                    "models.", tone="2")
        sub.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(sub)
        layout.addSpacing(14)
        self.options: dict[str, OptionCard] = {}
        self.option_costs: dict[str, QLabel] = {}
        self.option_keys: dict[str, QLabel] = {}
        self.option_uses: dict[str, QLabel] = {}
        for key in setups.ORDER:
            setup = setups.SETUPS[key]
            option = OptionCard()
            option.setFixedWidth(640)
            option.setAccessibleName(setup.name)
            line = QHBoxLayout(option)
            line.setContentsMargins(20, 10, 20, 10)
            line.setSpacing(14)
            line.addWidget(_icon(SETUP_ICONS[key], "text2", 20), 0, Qt.AlignmentFlag.AlignVCenter)
            words = QVBoxLayout()
            words.setSpacing(2)
            words.addWidget(label(setup.name, "heading", wrap=False))
            self.option_uses[key] = caption(setups.uses_line(setup), "2", wrap=False)  # for the language: _show_setups
            words.addWidget(self.option_uses[key])
            line.addLayout(words, 1)
            right = QVBoxLayout()
            right.setSpacing(2)
            self.option_costs[key] = caption("", "2", wrap=False)
            self.option_keys[key] = caption("", "3", wrap=False)
            for widget in (self.option_costs[key], self.option_keys[key]):
                widget.setAlignment(Qt.AlignmentFlag.AlignRight)
                right.addWidget(widget)
            line.addLayout(right)
            option.clicked.connect(lambda k=key: self.choose_setup(k))
            self.options[key] = option
            layout.addWidget(option, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch()
        scroll.setWidget(page)
        return scroll

    def choose_setup(self, key: str) -> None:
        self.setup = key
        for name, option in self.options.items():
            option.set_chosen(name == key)
        self.stepper.set_names([WELCOME_STEPS[name] for name in self.flow])
        self.stepper.set_step(self.step)
        self._show_footer()

    def _show_setups(self) -> None:
        app = self.app
        for key in setups.ORDER:
            setup = setups.for_app(app, key)  # what it would use, for the language spoken
            self.option_uses[key].setText(setups.uses_line(setup, app.settings.speech_language))
            if key == "custom":
                cost, needs = "Its cost shows as you choose", "The keys you choose"
                tone = None
            else:
                estimate = setups.cost(setup, app.settings.speech_language)
                cost, tone = estimate.words, COST_TONES.get(estimate.tier)
                needs = f"{_key_name(setup.needs)} key, free to start" if setup.needs else "No key, nothing leaves this PC"
            self.option_costs[key].setText(cost)
            set_tone(self.option_costs[key], tone or "2")
            self.option_keys[key].setText(needs)

    # -- step 2: the setup's key

    def _key(self) -> QWidget:
        page = Host()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 0, 40, 0)
        layout.setSpacing(8)
        layout.addStretch()
        self.key_heading = label("", "display")
        self.key_about = label("", tone="2")
        for widget in (self.key_heading, self.key_about):
            widget.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            layout.addWidget(widget)
        layout.addSpacing(24)
        self.setup_card, column = card(12, (24, 20, 24, 20))
        self.setup_card.setFixedWidth(620)
        self.key_title = caption("", "2", wrap=False)
        self.setup_key_link = button("Get a free key", self._open_setup_key_page, link=True, size="sm", icon="external",
                                     icon_after=True)
        column.addLayout(row(self.key_title, self.setup_key_link, stretch_at=1))
        self.setup_key = QLineEdit()
        self.setup_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.setup_key.setPlaceholderText("Paste your key")
        field(self.setup_key)
        self.setup_key.textChanged.connect(lambda _="": self._show_footer())
        paste = button("Paste", lambda: self.setup_key.setText(_clipboard_text().strip()))
        column.addLayout(row(self.setup_key, paste, spacing=12))
        self.key_voice = caption("", "2")
        column.addWidget(self.key_voice)
        self.key_note = caption("", "3")
        column.addWidget(self.key_note)
        self.key_status = caption("", "2")
        self.key_status.hide()
        column.addWidget(self.key_status)
        layout.addWidget(self.setup_card, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch()
        return page

    def _show_key(self) -> None:
        setup = setups.for_app(self.app, self.setup)
        if not setup.needs:
            return
        name = provider_name(setup.needs)
        self.key_heading.setText(f"Add your {_key_name(setup.needs)} key")
        speech_model = SPEECH_MODELS[setup.speech]
        if speech_model.where == "local":
            hears = f"{speech_model.name.removeprefix('NVIDIA ')} on this PC"
            polish = f"{setups.model_name(setup.model)} at {name}"
        else:
            hears = setups.model_name(setups.speech_model_for(setup, self.app.settings.speech_language))
            polish = f"{setups.model_name(setup.model)}, both at {name}"
        self.key_about.setText(f"{setup.name} hears you with {hears} and polishes the text with {polish}.")
        self.key_title.setText(f"{name} API key")
        if not self.setup_key.text():
            self.setup_key.setText(self.app.gateway.key_for(setup.needs))
        self.key_voice.setText(f"Your voice is sent to {CLOUD[setup.speech].name} each time you dictate, and the text "
                               "to polish it." if speech_model.where == "cloud" else
                               f"Your voice stays on this PC; only the text goes to {name}, to polish it.")
        set_tone(self.key_voice, "warn" if speech_model.where == "cloud" else "2")
        estimate = setups.cost(setup, self.app.settings.speech_language)
        self.key_note.setText(f"With a paid key it's {estimate.words.lower()} for 20 minutes of dictation a day. "
                              f"{setup.free_key}")

    def _open_setup_key_page(self) -> None:
        needs = setups.for_app(self.app, self.setup).needs
        if needs:
            shell.open_url(QUrl(PROVIDERS[needs].key_page))

    def _say_key(self, message: str, tone: str | None = "2") -> None:
        self.key_status.setText(message)
        set_tone(self.key_status, tone)
        self.key_status.setVisible(bool(message))

    def _connect_setup(self) -> None:
        """Check the key with the setup's AI model, then save the key (alone) and use the setup."""
        app, setup = self.app, setups.for_app(self.app, self.setup)
        p, api_key = PROVIDERS[setup.provider], self.setup_key.text().strip()
        gateway = GatewayConfig("", api_key, p.key, {k: v for k, v in app.gateway.entries().items() if k != p.key})
        self.primary.setEnabled(False)
        self._say_key(f"Connecting to {provider_name(p.key)}…")

        def done(_, error) -> None:
            self.primary.setEnabled(True)
            if error:
                self._say_key(f"Couldn't connect: {error}", "err")
                return
            if api_key != app.gateway.key_for(setup.needs):
                app.save_key(setup.needs, api_key)  # Your API keys: changes nothing else
            setups.apply(app, setup.key)
            self._say_key("")
            self.show_step(self.step + 1)
        run_in_background(self, lambda: app.check_ai(gateway, setup.model), done)

    # -- step 2 for Local: can this PC do it?

    def _check(self) -> QWidget:
        page = Host()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 0, 40, 0)
        layout.setSpacing(8)
        layout.addStretch()
        for widget in (label("Can this PC do it?", "display"),
                       label("Local keeps everything on this PC. Choose how it hears you: Rflow checks this PC can run "
                             "it.", tone="2")):
            widget.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            layout.addWidget(widget)
        layout.addSpacing(24)
        check_card, column = card(14, (24, 20, 24, 20))
        check_card.setFixedWidth(620)
        self.local = Segmented(LOCAL_CHOICES)
        self.local.changed.connect(self._local_chosen)
        column.addWidget(self.local, 0, Qt.AlignmentFlag.AlignLeft)
        check = QHBoxLayout()
        check.setSpacing(10)
        self.check_lamp = Lamp("off")
        self.check_words = label("", tone="2")
        lamp = QVBoxLayout()
        lamp.setContentsMargins(0, 4, 0, 0)
        lamp.addWidget(self.check_lamp)
        lamp.addStretch()
        check.addLayout(lamp)
        check.addWidget(self.check_words, 1)
        column.addLayout(check)
        column.addWidget(divider())
        soon = QHBoxLayout()
        soon.setSpacing(10)
        soon.addWidget(_icon("tools", "text3", 16), 0, Qt.AlignmentFlag.AlignTop)
        soon.addWidget(caption("AI cleanup on this PC is coming soon; until then Rflow types what it hears.", "3"), 1)
        column.addLayout(soon)
        layout.addWidget(check_card, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch()
        return page

    def _local_chosen(self, key: str) -> None:
        self.local_choice = key
        self.refresh(self.ready)

    def _local_level(self) -> str:
        return setups.pc_check(self.app.last_scan, self.local_choice)[0]

    def _show_check(self) -> None:
        app, key = self.app, self.local_choice
        self.local.set_current(key)
        level, words = setups.pc_check(app.last_scan, key)
        if app.scanning:
            state, words = "warn", f"Checking this PC: {app.scanning}"
        elif not level:
            state, words = "off", ("Rflow checks the memory, disk space and processor of this PC (Check this PC): "
                                   f"{SPEECH_MODELS[key].name} needs about {scan_memory(key)} of memory.")
        else:
            state = {"recommended": "ok", "fast": "ok", "usable": "off", "slow": "warn", "no": "err"}[level]
        self.check_lamp.set_state(state)
        self.check_words.setText(words)

    # -- for Custom: how Rflow hears you

    def _hear(self) -> QWidget:
        page = Host()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 0, 40, 0)
        layout.addStretch()
        title = label("How should Rflow hear you?", "display")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(title)
        sub = label("Pick one to start. You can switch any time on AI & models.", tone="2")
        sub.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(sub)
        layout.addSpacing(32)
        options = QHBoxLayout()
        options.setSpacing(24)
        options.addStretch()
        parakeet = SPEECH_MODELS[DEFAULT_MODEL]
        self.local_option = self._option("laptop", "On this PC", "NVIDIA Parakeet · English", "Recommended", [
            ("check", "ok", "Private: your voice stays on this PC"),
            ("check", "ok", "Works offline, free, no account"),
            ("check", "ok", "Fast here: about 0.4 s a sentence")],
            ("download", f"One download of {_size(parakeet.download.size)}"))
        self.cloud_option = self._option("cloud", "In the cloud", "Google Gemini", "", [
            ("check", "ok", "Many languages, Tamil and Japanese too"),
            ("check", "ok", "Nothing to download"),
            ("info", "warn", "Your voice goes to Google")],
            ("key", "Needs a Gemini API key"))
        for option, key in ((self.local_option, "local"), (self.cloud_option, "cloud")):
            option.clicked.connect(lambda k=key: self._choose(k))
            option.setFixedWidth(340)
            options.addWidget(option)
        options.addStretch()
        layout.addLayout(options)
        self.speech_status = caption("", "2")
        self.speech_status.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addSpacing(16)
        layout.addWidget(self.speech_status)
        layout.addStretch()
        return page

    def _option(self, icon: str, title: str, subtitle: str, badge: str, ticks: list, foot: tuple) -> OptionCard:
        option = OptionCard()
        layout = QVBoxLayout(option)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)
        top = QHBoxLayout()
        picture = QLabel()
        picture.setPixmap(theme.icon_pixmap(icon, tok("iris" if icon == "laptop" else "text2").name(), 24, 1.0))
        top.addWidget(picture)
        top.addStretch()
        if badge:
            top.addWidget(caption(badge, "2", wrap=False))
        layout.addLayout(top)
        words = QVBoxLayout()
        words.setSpacing(2)
        words.addWidget(label(title, "hero"))
        words.addWidget(caption(subtitle, "3", wrap=False))
        layout.addLayout(words)
        for tick, tone, line in ticks:
            mark = QLabel()
            mark.setPixmap(theme.icon_pixmap(tick, tok(tone).name(), 16, 1.0))
            ticked = row(mark, label(line, tone="2", wrap=False), spacing=10)
            ticked.setStretch(1, 1)
            layout.addLayout(ticked)
        layout.addWidget(divider())
        mark = QLabel()
        mark.setPixmap(theme.icon_pixmap(foot[0], tok("text3").name(), 16, 1.0))
        layout.addLayout(row(mark, caption(foot[1], "3", wrap=False), stretch_at=2, spacing=8))
        return option

    def _choose(self, key: str) -> None:
        self.choice = key
        self.local_option.set_chosen(key == "local")
        self.cloud_option.set_chosen(key == "cloud")
        self._show_footer()

    # -- try it

    def _try(self) -> QWidget:
        page = Host()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(64, 0, 64, 0)
        layout.setSpacing(56)
        left = QVBoxLayout()
        left.addStretch()
        self.orb = Orb(200)
        left.addWidget(self.orb, 0, Qt.AlignmentFlag.AlignHCenter)
        left.addSpacing(20)
        self.orb_row, self.orb_lamp, self.orb_text = lamp_row("ok", "", stretch=False)
        left.addWidget(self.orb_row, 0, Qt.AlignmentFlag.AlignHCenter)
        left.addStretch()
        holder = QWidget()
        holder.setLayout(left)
        holder.setFixedWidth(260)
        layout.addWidget(holder)
        right = QVBoxLayout()
        right.setSpacing(20)
        right.addStretch()
        right.addWidget(label("Say something", "display"))
        self.try_line = QHBoxLayout()
        self.try_line.setSpacing(6)
        right.addLayout(self.try_line)
        self.try_text = label("", tone="2")
        right.addWidget(self.try_text)
        right.addWidget(caption("Microphone", "2", wrap=False))
        self.microphone = MicrophoneBox(self.app.settings.microphone, self.app.microphones(), self.app.default_microphone(),
                                        source=lambda: (self.app.microphones(), self.app.default_microphone()))
        self.microphone.changed.connect(self._microphone_chosen)
        right.addWidget(self.microphone)
        self.typed_well, typed = card(6, (20, 16, 20, 16), kind="well")
        typed.addWidget(caption("Typed here", "3", wrap=False))
        self.try_box = QPlainTextEdit()
        self.try_box.setPlaceholderText("Click here first. Your words will appear here.")
        self.try_box.setFont(font(16, 400))
        self.try_box.setStyleSheet("QPlainTextEdit { padding: 0; }")
        fixed_height(self.try_box, 64)
        self.try_box.textChanged.connect(self._typed)
        typed.addWidget(self.try_box)
        right.addWidget(self.typed_well)
        self.status = label("", tone="2")
        right.addWidget(self.status)
        right.addStretch()
        layout.addLayout(right, 1)
        return page

    def _microphone_chosen(self, device: str) -> None:
        self.app.apply_settings(dataclasses.replace(self.app.settings, microphone=device))

    def _typed(self) -> None:
        if self.current() == "try":
            self.refresh(self.ready)

    # -- for Custom: connect an AI

    def _connect(self) -> QWidget:
        page = Host()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 0, 40, 0)
        layout.setSpacing(8)
        layout.addStretch()
        for widget in (caption("Optional", "2", wrap=False), label("Make it write better", "display"),
                       label("Connect an AI to add punctuation, drop filler words and unlock Text Transform and "
                             "Translate. Only text is sent, never your voice.", tone="2")):
            widget.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            layout.addWidget(widget)
        layout.addSpacing(24)
        tiles = QHBoxLayout()
        tiles.setSpacing(10)
        tiles.addStretch()
        self.tiles: dict[str, OptionCard] = {}
        for key, name, note in PROVIDER_TILES:
            tile = OptionCard("tile")
            tile.setFixedSize(128, 64)  # five side by side in the smallest window
            box = QVBoxLayout(tile)
            box.setContentsMargins(12, 10, 12, 10)
            box.setSpacing(2)
            box.addWidget(label(name, "heading", wrap=False))
            box.addWidget(caption(note, "3", wrap=False))
            tile.clicked.connect(lambda k=key: self._pick_provider(k))
            self.tiles[key] = tile
            tiles.addWidget(tile)
        tiles.addStretch()
        layout.addLayout(tiles)
        layout.addSpacing(16)
        self.key_card, column = card(14, (24, 20, 24, 20))
        self.key_card.setFixedWidth(620)
        self.ai_key_title = caption("", "2", wrap=False)
        self.key_link = button("Get a free key", self._open_key_page, link=True, size="sm", icon="external",
                               icon_after=True)
        column.addLayout(row(self.ai_key_title, self.key_link, stretch_at=1))
        self.ai_key = QLineEdit()
        self.ai_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.ai_key.setPlaceholderText("Paste your key")
        field(self.ai_key)
        self.ai_key.textChanged.connect(lambda _="": self._show_footer())
        self.paste_key = button("Paste", lambda: self.ai_key.setText(_clipboard_text().strip()))
        column.addLayout(row(self.ai_key, self.paste_key, spacing=12))
        self.model_line = caption("", "3")
        column.addWidget(self.model_line)
        layout.addWidget(self.key_card, 0, Qt.AlignmentFlag.AlignHCenter)
        layout.addStretch()
        return page

    def _pick_provider(self, key: str) -> None:
        self.provider = key
        for name, tile in self.tiles.items():
            tile.set_chosen(name == key)
        p = PROVIDERS[key]
        self.ai_key_title.setText(f"{provider_name(key)} API key" if p.needs_key else "Ollama runs on this PC: no key "
                                                                                      "needed")
        self.key_link.setText("Get a free key" if key in ("gemini", "groq") else "Get a key")
        self.key_link.setVisible(bool(p.key_page))
        self.ai_key.setVisible(p.needs_key)
        self.paste_key.setVisible(p.needs_key)
        self.ai_key.setText(self.app.gateway.key_for(key))
        self.model_line.setText("Rflow picks the provider's fast model for you (you can change it later).")
        self._show_footer()

    def _open_key_page(self) -> None:
        shell.open_url(QUrl(PROVIDERS[self.provider].key_page))

    # -- the steps and the footer

    def show_step(self, step: int) -> None:
        self.step = max(0, min(len(self.flow) - 1, step))
        name = self.current()
        self.steps.setCurrentWidget(self.step_pages[name])
        self.stepper.set_names([WELCOME_STEPS[key] for key in self.flow])
        self.stepper.set_step(self.step)
        if name == "setup":
            self.choose_setup(self.setup)
        elif name == "key":
            self._say_key("")
            self._show_key()
        elif name == "check" and not self.app.last_scan and not self.app.scanning:
            self.app.scan_computer()  # Local: whether this PC can run the model, before it's used
        elif name == "hear":
            self._choose(self.choice)
        elif name == "ai" and not any(tile.chosen for tile in self.tiles.values()):
            self._pick_provider(self.provider)
        self.refresh(self.ready)
        if name == "try":
            self.try_box.setFocus()

    def go_to_step(self, name: str) -> None:
        """A step of the setup's flow by its name ("try")."""
        if name in self.flow:
            self.show_step(self.flow.index(name))

    def _show_footer(self) -> None:
        name, app = self.current(), self.app
        installed = SPEECH_MODELS[DEFAULT_MODEL].installed()
        fetching = bool(app.downloading) and app.downloading[0] == DEFAULT_MODEL
        last = self.step == len(self.flow) - 1
        self.back.setVisible(self.step > 0)
        self.skip.setVisible(name != "ai")
        self.foot_note.setVisible(name in ("setup", "hear", "check"))
        self.secondary.setVisible(name in ("try", "ai", "check"))
        self.primary.setEnabled(True)
        self.primary.icon_name, self.primary.icon_after = "arrow-right", True
        self.primary.setText("Continue")
        if name == "setup":
            setup = setups.for_app(app, self.setup)  # Recommended and Fastest choose by the language spoken
            parakeet = setup.speech == DEFAULT_MODEL and not installed and not fetching and not app.speech_in_use()
            self.foot_note.setText(f"Parakeet ({_size(SPEECH_MODELS[DEFAULT_MODEL].download.size)}) downloads once "
                                   "you've added the key." if parakeet else
                                   {"local": "Next: a check of this PC.", "custom": "Next: each part, one by one."}
                                   .get(self.setup, f"Next: your {_key_name(setup.needs)} key."))
        elif name == "key":
            self.primary.setText("Connect and continue")
            self.primary.setEnabled(bool(self.setup_key.text().strip()))
        elif name == "check":
            level, model = self._local_level(), SPEECH_MODELS[self.local_choice]
            if not model.installed() and not (app.downloading and app.downloading[0] == model.key):
                self.primary.setText(f"Download {'Parakeet' if model.key == DEFAULT_MODEL else 'Whisper'} and continue")
                self.primary.icon_name, self.primary.icon_after = "download", False
            self.primary.setEnabled(level not in ("", "no") and not app.scanning)
            self.secondary.setText("Check again" if app.last_scan else "Check this PC")
            self.secondary.setEnabled(not app.scanning)
            self.foot_note.setText("Nothing leaves this PC.")
        elif name == "hear":
            chosen = app.speech_in_use() or app.loading_speech
            if self.choice == "cloud":
                self.primary.setText("Set up a cloud model")
                self.foot_note.setText("Choose a provider and paste its key on the next page.")
            elif installed or chosen or fetching:
                self.foot_note.setText("Parakeet is here." if installed else "The download keeps going while you try the "
                                                                             "next step.")
            else:
                self.primary.setText("Download Parakeet and continue")
                self.primary.icon_name, self.primary.icon_after = "download", False
                self.foot_note.setText("The download keeps going while you try the next step.")
        elif name == "try":
            self.secondary.setText("Try again")
            if last:
                self.primary.setText("Finish")
                self.primary.icon_name = None
        else:
            self.secondary.setText("Skip for now")
            self.primary.setText("Connect and finish")
            self.primary.icon_name = None
            p = PROVIDERS[self.provider]
            self.primary.setEnabled(bool(self.ai_key.text().strip()) or not p.needs_key)
        self.primary.updateGeometry()
        self.primary.update()

    @property
    def get_parakeet(self) -> QPushButton:
        """The footer's main button (on the first steps it downloads Parakeet when Parakeet isn't here)."""
        return self.primary

    def _primary(self) -> None:
        name, app = self.current(), self.app
        if name == "key":
            self._connect_setup()
        elif name == "check":
            setups.apply(app, "local", self.local_choice)  # downloads the model first when it isn't here
            self.show_step(self.step + 1)
        elif name == "hear":
            if self.choice == "cloud":
                self._to_speech()
                return
            fetching = bool(app.downloading) and app.downloading[0] == DEFAULT_MODEL
            if not SPEECH_MODELS[DEFAULT_MODEL].installed() and not fetching and not app.speech_in_use():
                app.download_speech_model(DEFAULT_MODEL)
            self.show_step(self.step + 1)
        elif name == "ai":
            self._connect_ai()
        elif self.step == len(self.flow) - 1:
            self.finish()
        else:
            self.show_step(self.step + 1)

    def _secondary(self) -> None:
        name = self.current()
        if name == "try":
            self.try_box.clear()
            self.try_box.setFocus()
        elif name == "check":
            self.app.scan_computer()
            self.refresh(self.ready)
        elif name == "ai":
            self.finish()

    def _connect_ai(self) -> None:
        """Ask the provider for its models, pick its fast one, check it answers, then save and finish."""
        p = PROVIDERS[self.provider]
        api_key = self.ai_key.text().strip()
        gateway = GatewayConfig(p.url if p.own_server else "", api_key, p.key,
                                {k: v for k, v in self.app.gateway.entries().items() if k != p.key})
        self.primary.setEnabled(False)
        self.model_line.setText(f"Connecting to {provider_name(p.key)}…")
        set_tone(self.model_line, "2")

        def work():
            models = self.app.ai_models(gateway)
            model = fast_model(p.key, models)
            if not model:
                raise RuntimeError(f"{provider_name(p.key)} lists none of its usual fast models: choose one on AI "
                                   "connection later" if models else "the provider lists no models")
            self.app.check_ai(gateway, model)
            return model

        def done(model, error) -> None:
            self.primary.setEnabled(True)
            if error:
                set_tone(self.model_line, "err")
                self.model_line.setText(f"Couldn't connect: {error}")
                return
            set_tone(self.model_line, "ok")
            self.model_line.setText(f"Connected. Model chosen for you: {model}")
            self.app.save_cleanup(True, model, "", gateway)
            self.finish()
        run_in_background(self, work, done)

    def refresh(self, ready: bool) -> None:
        self.ready = ready
        app = self.app
        label_text = app.hotkey_label()
        in_use, loading, downloading = app.speech_in_use(), app.loading_speech, app.downloading
        fetching = bool(downloading) and downloading[0] in SPEECH_MODELS
        chosen = in_use or loading
        self.mark.set_state("ok" if ready else "warn")
        self._show_setups()
        if self.current() == "check":
            self._show_check()
        if fetching and downloading[0] == DEFAULT_MODEL:
            done, total = downloading[1], downloading[2] or 1
            self.speech_status.setText(f"Downloading Parakeet: {done * 100 // total}%  ({_size(done)} of {_size(total)}).")
        else:
            self.speech_status.setText(f"In use: {SPEECH_MODELS[chosen].name}" if chosen else "")
        self.speech_status.setVisible(bool(self.speech_status.text()))
        clear(self.try_line)
        self.try_line.addWidget(label("Hold", tone="2", wrap=False))
        self.try_line.addWidget(keys(key_names(label_text), "sm"))
        self.try_line.addWidget(label("and say “Hello Rflow, this is my first dictation”, then let go.",
                                      tone="2"), 1)
        self.try_text.setText(f"Click in the box below first. {how_to_dictate(label_text)}")
        typed = bool(self.try_box.toPlainText().strip())
        if typed:
            self.orb.set_state("done")
            self.orb_lamp.set_state("ok")
            self.orb_text.setText("Typed, on this PC" if (in_use and SPEECH_MODELS[in_use].where == "local") else "Typed")
            self.status.setText("It works. Now try it in any app.")
        elif ready:
            self.orb.set_state("ready")
            self.orb_lamp.set_state("ok")
            self.orb_text.setText("Ready: go ahead")
            self.status.setText("Ready: go ahead.")
        elif fetching:
            name = SPEECH_MODELS[downloading[0]].name.removeprefix("NVIDIA ").removeprefix("OpenAI ")
            done, total = downloading[1], downloading[2] or 1
            self.orb.set_state("loading", done / total)
            self.orb_lamp.set_state("warn")
            self.orb_text.setText(f"Downloading {name}, {done * 100 // total}%")
            self.status.setText(f"Waiting for {name}'s download... Meanwhile, choose your microphone.")
        elif chosen:
            self.orb.set_state("off")
            self.orb_lamp.set_state("warn")
            self.orb_text.setText("Loading the speech model")
            self.status.setText("Loading the speech model (a few seconds)...")
        else:
            self.orb.set_state("off")
            self.orb_lamp.set_state("warn")
            self.orb_text.setText("No speech model yet")
            self.status.setText("Choose how Rflow recognises your speech first (step 1).")
        self._show_footer()

    def _to_speech(self) -> None:
        self.finish()
        self.go_to("speech")
        page = getattr(self.window(), "pages", {}).get("speech")
        if page is not None:
            page.show_where("cloud")

    def finish(self) -> None:
        self.app.finish_welcome("")
        self.go_to("home")


# ---------------------------------------------------------------- profiles

class ProfilesPage(Page):
    """People sharing this computer: each profile has its own setup."""

    def __init__(self, app, go_to=None):
        go_to = go_to or (lambda page: None)
        super().__init__("Profiles", "Each profile has its own dictation key and microphone, words, AI provider and keys, "
                                     "dictations, stats and reading tests. Useful when several people share this PC, "
                                     "or to keep a work and a private setup apart. Click a name to change it.",
                         back=back_button("Settings", go_to, "settings"))
        self.app = app
        self.list = QVBoxLayout()
        self.list.setSpacing(16)
        self.add(self.list)
        new, layout = card(12, (20, 18, 20, 20))
        layout.addWidget(label("New profile", "heading"))
        self.new_name = QLineEdit()
        self.new_name.setPlaceholderText("Name, e.g. Rahul")
        field(self.new_name)
        self.new_name.returnPressed.connect(self._create)
        layout.addLayout(row(self.new_name, button("Create and switch to it", self._create, primary=True), spacing=12))
        layout.addWidget(caption("A new profile starts with the welcome: how Rflow hears you, a first dictation, an AI "
                                 "connection.", "3"))
        self.add(new)
        self.body.addStretch()

    def refresh(self) -> None:
        clear(self.list)
        current = self.app.profiles.current
        for profile in self.app.profiles.items:
            frame, layout = card(12, (20, 14, 20, 14), horizontal=True)
            name = QLineEdit(profile.name)
            name.setPlaceholderText(profile.label)
            name.setToolTip("Type to rename, then press Enter")
            field(name)
            name.editingFinished.connect(lambda p=profile, box=name: self.app.rename_profile(p.id, box.text()))
            layout.addWidget(name, 1)
            if profile.id == current.id:
                in_use, _, _ = lamp_row("ok", "In use", stretch=False)
                layout.addWidget(in_use)
            else:
                layout.addWidget(button("Switch to this profile", lambda _=False, p=profile: self.app.switch_profile(p.id),
                                        size="sm"))
                if profile.id != "default":  # the first profile's files are the settings folder itself
                    layout.addWidget(button("Delete", lambda _=False, p=profile: self._delete(p), kind="danger",
                                            size="sm"))
            self.list.addWidget(frame)

    def _create(self) -> None:
        name = self.new_name.text().strip()
        if name:
            self.new_name.clear()
            self.app.create_profile(name)

    def _delete(self, profile) -> None:
        if self.confirm(f"Delete the profile {profile.label}, with its words, keys, dictations, stats and reading tests?"):
            self.app.delete_profile(profile.id)
            self.refresh()

    def confirm(self, question: str) -> bool:
        return QMessageBox.question(self, APP_NAME, question) == QMessageBox.StandardButton.Yes


# ---------------------------------------------------------------- the window

NAV = [("home", "Home"), ("live", "Live translation"), ("meeting", "Meeting mode"), ("words", "Words"),
       ("snippets", "Snippets"), ("transform", "Text Transform"), ("translate", "Translate"), ("formatting", "Formatting"),
       ("models", "AI & models"), ("settings", "Settings"), ("report", "Report a problem")]
NAV_ICONS = {"home": "home", "live": "live", "meeting": "meeting", "words": "words", "snippets": "snippets",
             "transform": "transform", "translate": "translate", "formatting": "formatting", "models": "models",
             "settings": "settings", "report": "report"}
RAIL_NAMES = {"live": "Live", "meeting": "Meeting", "transform": "Transform", "formatting": "Format", "models": "AI",
              "report": "Report"}
# The section each page belongs to (its sidebar button), and the page a section opens on. "tools" was the section of
# Text Transform and Translate until phase 31: a link to it (from another page, an older build) opens Text Transform.
SECTION = {"home": "home", "live": "live", "meeting": "meeting", "dictionary": "words", "snippets": "snippets",
           "transform": "transform", "translate": "translate", "formatting": "formatting", "models": "models",
           "speech": "models", "cleanup": "models", "settings": "settings", "reading": "settings", "profiles": "settings",
           "report": "report"}
OPENS = {"words": "dictionary", "tools": "transform"}
REFRESHED = ("home", "dictionary", "snippets", "profiles", "speech", "cleanup", "transform", "translate", "formatting",
             "live", "meeting", "models", "settings", "report")
# A sidebar button's height (roomy, tight), in the sidebar and in the narrow rail: eleven sections fit the smallest window.
NAV_HEIGHTS = {False: (40, 26), True: (52, 33)}


class NavButton(Button):
    """A sidebar button whose height the window sets (MainWindow._fit_sidebar), so every section fits a short window.
    In the narrow rail its icon sits over its short name, centred in that height."""

    def __init__(self, text: str, icon: str):
        super().__init__(text, "nav", icon=icon)
        self.height_now = NAV_HEIGHTS[False][0]

    def _metrics(self):
        _, padding, px, weight = super()._metrics()
        return self.height_now, padding, px, weight

    def paintEvent(self, event):
        if not self.compact:
            super().paintEvent(event)
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r, checked, colour = QRectF(self.rect()), self.isChecked(), self._colour()
        icon = 18 if r.height() >= 44 else 16
        top = (r.height() - icon - 16) / 2  # the icon, then its name in a line of 16
        if self.icon_name:
            p.drawPixmap(QPointF(r.center().x() - icon / 2, top),
                         theme.icon_pixmap(self.icon_name, (tok("iris") if checked else colour).name(), icon,
                                           self.devicePixelRatioF()))
        p.setFont(font(11, 600 if checked else 500))
        p.setPen(colour)
        p.drawText(QRectF(0, top + icon + 1, r.width(), 16), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                   self.text())
        self._focus(p, r)
        p.end()


class Pages(Mapping):
    """The window's pages by key, each built the first time it is wanted (shown, or asked for as `pages[key]`). All
    fifteen built at once took 3-5 s, 1,170 widgets and 107 MB, mostly for pages never opened. Iterating gives every key;
    values() and items() build what they reach, like a dict would hold them. `built` has only those built so far."""

    def __init__(self, makers: dict[str, Callable[[], QWidget]], on_built: Callable[[str, QWidget], None]):
        self._makers, self._on_built = makers, on_built
        self.built: dict[str, QWidget] = {}

    def __getitem__(self, key: str) -> QWidget:
        page = self.built.get(key)
        if page is None:
            page = self.built[key] = self._makers[key]()
            self._on_built(key, page)
        return page

    def __contains__(self, key) -> bool:
        return key in self._makers

    def __iter__(self):
        return iter(self._makers)

    def __len__(self) -> int:
        return len(self._makers)


class MainWindow(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        theme.load_fonts()
        theme.set_theme("dark" if dark_mode() else "light")
        theme.make_host(self)
        self.setObjectName("root")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(QIcon(str(ICON_FILE)))
        self.setFont(font(14))
        self.resize(1000, 700)
        self.setMinimumSize(780, 540)
        self.ready = False
        self.status_message = ""
        self.compact = None

        self.sidebar = Host("base")
        self.sidebar.setObjectName("sidebar")
        side = QVBoxLayout(self.sidebar)
        side.setContentsMargins(20, 24, 0, 20)
        side.setSpacing(4)
        brand = QHBoxLayout()
        brand.setContentsMargins(4, 0, 0, 0)
        brand.setSpacing(12)
        self.mark = Mark()
        self.wordmark = label(APP_NAME, "wordmark", wrap=False)
        brand.addWidget(self.mark)
        brand.addWidget(self.wordmark)
        brand.addStretch()
        side.addLayout(brand)
        self.brand_gap = QWidget()  # 24 px, less in a short window (_fit_sidebar)
        side.addWidget(self.brand_gap)
        self.nav_box = QVBoxLayout()
        self.nav: dict[str, NavButton] = {}
        for key, label_text in NAV:
            b = NavButton(label_text, NAV_ICONS[key])
            b.setCheckable(True)
            b.setAutoExclusive(True)
            b.clicked.connect(lambda _=False, k=key: self.show_page(k))
            self.nav[key] = b
            self.nav_box.addWidget(b)
        side.addLayout(self.nav_box)
        side.addStretch()
        self.update_link = button("", lambda: app.start_update(), link=True, size="sm", icon="update")
        self.update_link.hide()
        side.addWidget(self.update_link)
        self.status_card = Card("well", 14)
        status = QHBoxLayout(self.status_card)
        status.setContentsMargins(12, 12, 12, 12)
        status.setSpacing(10)
        self.status_lamp = Lamp("warn")
        words = QVBoxLayout()
        words.setSpacing(0)
        self.status_title = label("Getting ready", wrap=False)
        self.status_title.setFont(font(12, 600))
        self.status_label = caption("", "2", wrap=False)
        words.addWidget(self.status_title)
        words.addWidget(self.status_label)
        status.addWidget(self.status_lamp, 0, Qt.AlignmentFlag.AlignVCenter)
        status.addLayout(words, 1)
        side.addSpacing(8)
        side.addWidget(self.status_card)
        self.profile_button = button("", self._profile_menu, kind="quiet", size="sm", icon="profile")
        self.profile_button.setToolTip("Switch profile")
        side.addWidget(self.profile_button)

        self.banner = Card("tile", 14)
        banner = QHBoxLayout(self.banner)
        banner.setContentsMargins(16, 10, 10, 10)
        banner.setSpacing(12)
        mark = QLabel()
        mark.setPixmap(theme.icon_pixmap("update", tok("iris").name(), 18, 1.0))
        self.banner_text = label("")
        self.notes_button = button("What's new", lambda: app.open_release_notes(), link=True, size="sm")
        self.update_button = button("Update now", lambda: app.start_update(), primary=True, size="sm")
        banner.addWidget(mark)
        banner.addWidget(self.banner_text, 1)
        banner.addWidget(self.notes_button)
        banner.addWidget(self.update_button)
        self.banner.hide()

        self._update_status = ""  # Settings' update line, kept for when that page is built
        kinds = {"home": HomePage, "dictionary": DictionaryPage, "snippets": SnippetsPage, "live": LivePage,
                 "meeting": MeetingPage, "transform": TransformPage, "translate": TranslatePage, "formatting": FormattingPage,
                 "report": ReportPage, "models": ModelsPage, "speech": SpeechPage, "cleanup": CleanupPage,
                 "settings": SettingsPage, "reading": ReadingTestPage, "profiles": ProfilesPage, "welcome": WelcomePage}
        self.stack = QStackedWidget()
        self.pages = Pages({key: lambda kind=kind: kind(app, self.show_page) for key, kind in kinds.items()},
                           self._page_built)
        self.pages["home"]  # noqa: B018 (built now: the page the window opens on, and the one set_status keeps current)
        self.content = Host("base")
        content = QVBoxLayout(self.content)
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)
        # The bar over every page: the look (as Windows is set, light or dark) and Check for updates, one click each
        self.topbar = QWidget()
        top = QHBoxLayout(self.topbar)
        top.setContentsMargins(32, 16, 32, 0)
        top.setSpacing(12)
        top.addStretch()
        self.appearance = Segmented(APPEARANCE_CHOICES, icons=APPEARANCE_ICONS)
        for key, b in self.appearance.buttons.items():
            b.setToolTip(APPEARANCE_TIPS[key])
        self.appearance.set_current(theme.APPEARANCES[0])
        self.appearance.changed.connect(self.set_appearance)
        top.addWidget(self.appearance, 0, Qt.AlignmentFlag.AlignVCenter)
        self.check_button = button("Check for updates", self._check_for_updates, primary=True, size="sm", icon="update")
        top.addWidget(self.check_button, 0, Qt.AlignmentFlag.AlignVCenter)
        self._checking = False  # the top bar's check is waiting for its answer (a toast says it)
        content.addWidget(self.topbar)
        self.banner_holder = QWidget()
        banner_holder = QVBoxLayout(self.banner_holder)
        banner_holder.setContentsMargins(24, 20, 24, 0)
        banner_holder.addWidget(self.banner)
        self.banner_holder.hide()
        content.addWidget(self.banner_holder)
        content.addWidget(self.stack, 1)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.sidebar)
        layout.addWidget(self.content, 1)
        self.toast = Toast(self)

        theme.set_appearance(app.settings.appearance)
        self.appearance.set_current(app.settings.appearance)
        self.apply_theme()
        QGuiApplication.styleHints().colorSchemeChanged.connect(self.apply_theme)
        self._set_compact(False)
        self.show_page("home" if app.settings.welcomed else "welcome")

    @property
    def toast_offset(self) -> int:
        return 0 if self.sidebar.isHidden() else self.sidebar.width() // 2  # the toast centres on the page, not the window

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), tok("base"))
        p.end()

    def apply_theme(self, name: str | None = None) -> None:
        """Obsidian or Porcelain, as Windows is set (or `name`, e.g. "light" for the website's screenshots). It restyles
        the pages built so far; one built later takes the theme in use from the window's style sheet as it joins."""
        theme.set_theme(name if isinstance(name, str) else "dark" if dark_mode() else "light")
        self.setStyleSheet(theme.stylesheet())
        for widget in self.findChildren(QWidget):
            widget.update()
        self.update()
        self.refresh()

    def set_appearance(self, name: str) -> None:
        """The look chosen in the top bar or in Settings: saved in the profile, used at once."""
        name = name if name in theme.APPEARANCES else "system"
        if self.app.settings.appearance != name:
            self.app.apply_settings(dataclasses.replace(self.app.settings, appearance=name))
        theme.set_appearance(name)
        self.appearance.set_current(name)
        if "settings" in self.pages.built:
            self.pages["settings"].appearance.set_current(name)
        self.apply_theme()

    def _check_for_updates(self) -> None:
        """The top bar's button: it says Checking… until the answer, which a toast then gives (or the update banner)."""
        self._checking = True
        self.check_button.setText("Checking…")
        self.check_button.setEnabled(False)
        self.app.check_for_updates(manual=True)

    def _checked(self, message: str = "") -> None:
        if not self._checking:
            return
        self._checking = False
        self.check_button.setText("Check for updates")
        self.check_button.setEnabled(True)
        if message:
            latest = message.startswith("You have the latest")
            self.toast.show_message(message, "check" if latest else "warning", "ok" if latest else "warn")

    def _page_built(self, key: str, page: QWidget) -> None:
        """A page built on first use: the window's present state, as it would have had from the start."""
        self.stack.addWidget(page)
        if isinstance(page, Page) and self.compact is not None:
            page.set_compact(self.compact)
        if key == "home":
            page.set_ready(self.ready)
        elif key == "settings" and self._update_status:
            page.update_status.setText(self._update_status)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._set_compact(self.width() < COMPACT_WIDTH)
        self._fit_sidebar()
        if self.toast.isVisible():
            self.toast.move((self.width() - self.toast.width()) // 2 + self.toast_offset, self.height() - self.toast.height())

    def _fit_sidebar(self) -> None:
        """Every section's button in the window's height. In a short window (the smallest is 540 px) the gaps, the
        margins and then the buttons get smaller, down to NAV_HEIGHTS' tight height, before anything is cut off."""
        if self.compact is None:
            return
        roomy, tight = NAV_HEIGHTS[self.compact]
        side, count = self.sidebar.layout(), len(self.nav)
        margins = (10, 20, 10, 16) if self.compact else (20, 24, 0, 20)
        side.setContentsMargins(*margins)
        self.brand_gap.setFixedHeight(24)
        self.nav_box.setSpacing(4)
        self._set_nav_height(roomy)
        if side.sizeHint().height() > self.height():  # short: smaller gaps and margins, then shorter buttons
            margins = (10, 12, 10, 10) if self.compact else (20, 16, 0, 14)
            side.setContentsMargins(*margins)
            self.brand_gap.setFixedHeight(8 if self.compact else 12)
            self.nav_box.setSpacing(2)
            others = side.sizeHint().height() - count * roomy  # all but the buttons themselves
            self._set_nav_height(max(tight, min(roomy, (self.height() - others) // count)))

    def _set_nav_height(self, height: int) -> None:
        for b in self.nav.values():
            b.height_now = height
            b.updateGeometry()
        self.sidebar.layout().invalidate()

    def _set_compact(self, compact: bool) -> None:
        if compact == self.compact:
            return
        self.compact = compact
        self.sidebar.setFixedWidth(84 if compact else 216)
        self.wordmark.setVisible(not compact)
        for key, b in self.nav.items():
            b.compact = compact
            b.setText(RAIL_NAMES.get(key, dict(NAV)[key]) if compact else dict(NAV)[key])
            b.setSizePolicy(QSizePolicy.Policy.Fixed if compact else QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            b.setFixedWidth(64) if compact else b.setMinimumWidth(0)
            if not compact:
                b.setMaximumWidth(16777215)
            b.updateGeometry()
        self.status_label.setVisible(not compact)
        status = self.status_card.layout()  # the rail: the lamp over its word
        status.setDirection(QBoxLayout.Direction.TopToBottom if compact else QBoxLayout.Direction.LeftToRight)
        status.setContentsMargins(*((6, 10, 6, 10) if compact else (12, 12, 12, 12)))
        status.setAlignment(self.status_lamp, Qt.AlignmentFlag.AlignHCenter if compact else Qt.AlignmentFlag.AlignVCenter)
        self.status_title.setFont(font(11 if compact else 12, 600))
        self.status_title.setAlignment(Qt.AlignmentFlag.AlignHCenter if compact else Qt.AlignmentFlag.AlignLeft)
        self.profile_button.setVisible(not compact and len(self.app.profiles.items) > 1)
        self.update_link.setVisible(not compact and bool(self.update_link.text()))
        for page in self.pages.built.values():
            if isinstance(page, Page):
                page.set_compact(compact)
        self._fit_sidebar()
        self.layout().invalidate()

    def show_page(self, key: str) -> None:
        key = OPENS.get(key, key)
        page, current = self.pages[key], self.stack.currentWidget()
        if current is not page and getattr(current, "unsaved", lambda: False)():
            if not self.leave_unsaved(current.title.text()):
                section = SECTION.get(self.current_page())
                if section in self.nav:
                    self.nav[section].setChecked(True)  # the sidebar button just clicked lets go again
                return
            current.discard_changes()
        self._show_profile()
        if key in REFRESHED:
            page.refresh()
        elif key == "welcome":
            page.refresh(self.ready)
        self.stack.setCurrentWidget(page)
        self.sidebar.setVisible(key != "welcome")  # the welcome has the whole window
        section = SECTION.get(key)
        if section in self.nav:
            self.nav[section].setChecked(True)
        else:
            for b in self.nav.values():
                b.setAutoExclusive(False)
                b.setChecked(False)
                b.setAutoExclusive(True)

    def current_section(self) -> str:
        return SECTION.get(self.current_page(), "")

    def leave_unsaved(self, page: str) -> bool:
        """Leaving a page whose changes weren't saved: True to discard them, False to stay on it."""
        box = QMessageBox(self)
        box.setWindowTitle(APP_NAME)
        box.setIcon(QMessageBox.Icon.Question)
        box.setText(f"{page} has changes that aren't saved.")
        box.setInformativeText("Stay to save them with Save, or discard them.")
        stay = box.addButton("Stay", QMessageBox.ButtonRole.RejectRole)
        discard = box.addButton("Discard changes", QMessageBox.ButtonRole.DestructiveRole)
        box.setDefaultButton(stay)
        box.exec()
        return box.clickedButton() is discard

    def _show_profile(self) -> None:
        self.profile_button.setText(self.app.profiles.current.label)
        shown = not self.compact and len(self.app.profiles.items) > 1
        if shown != self.profile_button.isVisibleTo(self):
            self.profile_button.setVisible(shown)
            self._fit_sidebar()
        counts = len(self.app.correction_suggestions()) if hasattr(self.app, "correction_suggestions") else 0
        self.nav["words"].badge = str(counts) if counts and not self.compact else ""
        self.nav["words"].update()

    def _profile_menu(self) -> None:
        menu = QMenu(self)
        current = self.app.profiles.current
        for profile in self.app.profiles.items:
            action = menu.addAction(profile.label, lambda p=profile: self.app.switch_profile(p.id))
            action.setCheckable(True)
            action.setChecked(profile.id == current.id)
        menu.addSeparator()
        menu.addAction("New profile...", self._new_profile)
        menu.addAction("Manage profiles", lambda: self.show_page("profiles"))
        menu.exec(self.profile_button.mapToGlobal(self.profile_button.rect().bottomLeft()))

    def _new_profile(self) -> None:
        self.show_page("profiles")
        self.pages["profiles"].new_name.setFocus()

    def current_page(self) -> str:
        return next(key for key, page in self.pages.built.items() if page is self.stack.currentWidget())

    def open(self, page: str | None = None) -> None:
        if not self.app.settings.welcomed:
            page = "welcome"
        self.show_page(page or self.current_page())
        if self.isMinimized():
            self.showNormal()
        self.show()
        self.raise_()
        self.activateWindow()

    def refresh(self) -> None:
        """New dictation, words, settings or profile name: update what is on screen."""
        wanted = self.app.settings.appearance
        if wanted in theme.APPEARANCES and wanted != theme.appearance():  # e.g. another profile's look
            self.set_appearance(wanted)
            return  # apply_theme refreshes
        self._show_profile()
        self._show_status()
        current = self.current_page()
        if current in REFRESHED:
            self.pages[current].refresh()
        elif current == "welcome":
            self.pages[current].refresh(self.ready)

    def usage_changed(self) -> None:
        """A Gemini call was counted: AI & models' cost card, if it is on screen (it refreshes itself when shown)."""
        if "models" in self.pages.built and self.current_page() == "models":
            self.pages["models"].costs.refresh()

    def set_status(self, message: str, ready: bool) -> None:
        self.ready, self.status_message = ready, message
        if "home" in self.pages.built:
            self.pages["home"].set_ready(ready)
        self._show_status()
        current = self.current_page()
        if current == "welcome":
            self.pages["welcome"].refresh(ready)
        elif current == "home":
            self.pages["home"].refresh()

    def _show_status(self) -> None:
        """The sidebar's lamp: Ready and the speech model, or what Rflow is waiting for."""
        app = self.app
        in_use, loading, downloading = app.speech_in_use(), app.loading_speech, app.downloading
        fetching = bool(downloading) and downloading[0] == DEFAULT_MODEL and not self.ready
        model = SPEECH_MODELS.get(in_use)
        if self.ready and model is not None:
            where = {"local": "on this PC", "cloud": "in the cloud", "server": "your server"}.get(model.where, "")
            title, detail, state = "Ready", f"{model.name.removeprefix('NVIDIA ').split(' large')[0]}, {where}", "ok"
        elif fetching:
            done, total = downloading[1], downloading[2] or 1
            title, detail, state = "Not ready yet", f"Downloading, {done * 100 // total}%", "warn"
        elif loading or in_use:
            title, detail, state = "Getting ready", "Loading the speech model", "warn"
        elif self.status_message:
            title, detail, state = "Not ready yet", "Choose a speech model", "warn"
        else:
            title, detail, state = "Getting ready", "", "warn"
        self.status_title.setText(title)
        self.status_label.setText(detail)
        self.status_card.setToolTip(self.status_message)
        self.status_lamp.set_state(state)
        self.mark.set_state(state)

    def show_update(self, message: str, busy: bool = False, version: str = "") -> None:
        self._checked()
        self.banner_text.setText(message)
        self.update_button.setEnabled(not busy)
        self.banner.show()
        self.banner_holder.show()
        if version:
            self.update_link.setText(f"Update to {version}")
            self.update_link.setVisible(not self.compact)  # the rail has no room for it: the banner says it
            self._fit_sidebar()

    def show_wake(self) -> None:
        """The wake word's status changed (TrayApp._on_wake_changed): its card says so, if it's built."""
        if "settings" in self.pages.built:
            self.pages["settings"].wake.show_status()

    def set_update_status(self, message: str) -> None:
        self._update_status = message
        if not message.startswith("Checking"):
            self._checked("" if self.banner.isVisible() else message)  # an update found: the banner says it
        if "settings" in self.pages.built:
            self.pages["settings"].update_status.setText(message)

    def closeEvent(self, event):
        # Closing the window doesn't quit: dictation keeps working from the tray, like Wispr Flow.
        event.ignore()
        self.hide()  # hiding also stops a reading-test recording and the level meters (their hideEvent)
        self.app.window_closed()


# ---------------------------------------------------------------- a stand-in for the TrayApp

class PreviewApp:
    """Stands in for the TrayApp: the self-test, the tests and the website's screenshots use it (no model, no hook)."""

    def __init__(self, settings: Settings | None = None, history: list[dict] | None = None, stats: Stats | None = None,
                 microphones: list[str] | None = None, gateway: GatewayConfig | None = None,
                 profiles: Profiles | None = None, bench: Path | None = None):
        self.settings = settings or Settings(welcomed=True)
        self.profiles = profiles or Profiles()
        self._bench = bench or Path(os.environ.get("TEMP", ".")) / "rflow-preview-bench"
        self.gateway = gateway or GatewayConfig()
        self.history = history or []
        self.stats = stats or Stats()
        self._microphones = microphones if microphones is not None else ["Microphone (Realtek(R) Audio)"]
        self.loading_speech = ""
        self.downloading: tuple[str, int, int] | None = None
        self.scanning = ""
        self.last_scan: dict | None = None
        self.calls: list[tuple] = []  # what the window asked for
        self._dictionary = None
        self.usage = usage.Ledger(None)  # in memory: what Gemini cost, for screenshots and tests

    def hotkey_label(self) -> str:
        return parse_hotkey(self.settings.hotkey).label

    def history_entries(self) -> list[dict]:
        return self.history

    def microphones(self) -> list[str]:
        return self._microphones

    def default_microphone(self) -> str:
        return self._microphones[0] if self._microphones else ""

    def new_recorder(self):
        from sst.audio import Recorder
        return Recorder(self.settings.microphone or None, raw=self.settings.raw_audio)

    def bench_dir(self) -> Path:
        return self.profiles.current.folder(self._bench)

    def speech_in_use(self) -> str:
        return usable(self.settings.speech_model, self.gateway)

    def choose_speech_model(self, key: str) -> None:
        self.settings.speech_model = key
        self.calls.append(("choose_speech_model", key))

    def set_speech_language(self, code: str) -> None:
        self.settings.speech_language = code
        self.calls.append(("set_speech_language", code))

    def download_speech_model(self, key: str) -> None:
        self.calls.append(("download_speech_model", key))

    def use_cloud_speech(self, provider: str, api_key: str, model: str) -> None:
        self.gateway = self.gateway.with_key(provider, api_key)
        self.settings.speech_cloud_models = {**self.settings.speech_cloud_models, provider: model}
        self.settings.speech_model = provider
        self.calls.append(("use_cloud_speech", provider, model))

    def test_cloud_speech(self, provider: str, api_key: str, model: str) -> str:
        self.calls.append(("test_cloud_speech", provider, model))
        return f"{model} answered in 0.6 s: After early nightfall the yellow lamps would light up."

    def use_server_speech(self, address: str, api_key: str, model: str) -> None:
        from sst.gateway import SPEECH_SERVER
        self.gateway = self.gateway.with_entry(SPEECH_SERVER, address, api_key)
        self.settings.speech_server_model, self.settings.speech_model = model, "server"
        self.calls.append(("use_server_speech", address, model))

    def test_server_speech(self, address: str, api_key: str, model: str) -> str:
        self.calls.append(("test_server_speech", address, model))
        return f"{model} answered in 0.3 s: After early nightfall the yellow lamps would light up."

    def server_models(self, address: str, api_key: str) -> list[str]:
        self.calls.append(("server_models", address))
        return ["whisper-1", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"]

    def save_key(self, provider: str, key: str) -> None:
        self.gateway = self.gateway.with_key(provider, key)
        self.calls.append(("save_key", provider))  # never the key itself

    def save_server(self, address: str, key: str, names: tuple[str, ...] = ("vllm",)) -> None:
        for name in names:
            self.gateway = self.gateway.with_entry(name, address, key)
        self.calls.append(("save_server", address, names))

    def ai_models(self, gateway: GatewayConfig) -> list[str]:
        self.calls.append(("ai_models", gateway.service.key))
        return {"gemini": ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash", "text-embedding-004"],
                "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4.1-mini"]}.get(gateway.service.key, ["model-a"])

    def check_ai(self, gateway: GatewayConfig, model: str) -> str:
        self.calls.append(("check_ai", gateway.service.key, model))
        return f"{model} answered in 0.6 s"

    def cancel_download(self) -> None:
        self.calls.append(("cancel_download",))

    def scan_computer(self) -> None:
        self.calls.append(("scan_computer",))

    def remove_speech_model(self, key: str) -> None:
        self.calls.append(("remove_speech_model", key))

    def apply_settings(self, new: Settings) -> None:
        self.settings = new
        self.calls.append(("apply_settings", new))

    def save_cleanup(self, on: bool, model: str, fallback: str, gateway: GatewayConfig) -> None:
        self.settings.cleanup, self.settings.cleanup_model, self.settings.cleanup_fallback = on, model, fallback
        self.gateway = gateway
        self.calls.append(("save_cleanup", on, model, fallback, gateway))

    def add_words(self, words: list[str]) -> int:
        known = {w.lower() for w in self.settings.vocabulary}
        added = [w for w in dict.fromkeys(words) if w.lower() not in known]
        self.settings.vocabulary = self.settings.vocabulary + added
        return len(added)

    def remove_word(self, word: str) -> None:
        self.settings.vocabulary = [w for w in self.settings.vocabulary if w != word]
        term = self.dictionary.find(word)
        if term is not None and term.source != "vocabulary":
            self.dictionary.remove_term(term.id)

    @property
    def dictionary(self):
        if self._dictionary is None:
            from sst.pipeline.dictionary import DictionaryStore
            self._dictionary = DictionaryStore()  # in memory: the preview keeps nothing
            self._dictionary.sync_vocabulary(self.settings.vocabulary)
        return self._dictionary

    def dictionary_terms(self) -> list:
        self.dictionary.sync_vocabulary(self.settings.vocabulary)
        return self.dictionary.terms(enabled_only=False)

    def correct_dictation(self, typed: str, corrected: str) -> None:
        from sst.pipeline.learning import CorrectionEvent, Learner
        Learner(self.dictionary).observe(CorrectionEvent(typed, corrected, app="history"))
        self.calls.append(("correct_dictation", typed, corrected))

    def correction_suggestions(self) -> list:
        from sst.pipeline.learning import Learner
        return Learner(self.dictionary).suggestions()

    def accept_suggestion(self, suggestion) -> None:
        from sst.pipeline.learning import Learner
        Learner(self.dictionary).accept(suggestion)

    def reject_suggestion(self, suggestion) -> None:
        from sst.pipeline.learning import Learner
        Learner(self.dictionary).reject(suggestion)

    def transform_model(self) -> str:
        return self.settings.cleanup_model if self.gateway.address else ""

    def translate_ready(self) -> bool:
        return bool(self.transform_model())

    _live = False  # live translation never really starts in the preview
    _live_sessions: tuple = ()  # (began, lines, path) of past sessions, for screenshots and tests

    def live_problem(self) -> str:
        return "" if self.gateway.key_for("gemini") else \
            "Live translation uses Google Gemini 3.5 Live Translate: add a Gemini key in AI & models."

    def live_running(self) -> bool:
        return self._live

    _live_note = ""  # the bar's status line while it runs, for screenshots and tests

    def live_note(self) -> str:
        return self._live_note if self._live else ""

    _live_languages =("", "")  # what the session running translates into (a language changed meanwhile waits)

    def live_languages(self) -> tuple[str, str]:
        s = self.settings
        return self._live_languages if self._live and all(self._live_languages) else (s.live_target, s.live_mic_target)

    def live_sessions(self, limit: int = 8) -> list:
        return list(self._live_sessions)[:limit]

    def live_shortcut_clash(self) -> str:
        s = self.settings
        for name, shortcut in (("Translate", s.translate_shortcut), ("Text Transform", s.transform_shortcut)):
            if s.live_shortcut and shortcut == s.live_shortcut:
                return name
        return ""

    def start_live(self) -> str:
        self.calls.append(("start_live", self.settings.live_source))
        self._live = not self.live_problem()
        self._live_languages = (self.settings.live_target, self.settings.live_mic_target)
        return self.live_problem()

    def stop_live(self) -> None:
        self.calls.append(("stop_live",))
        self._live = False

    def set_live_source(self, source: str) -> str:
        self.calls.append(("set_live_source", source))
        self.apply_settings(dataclasses.replace(self.settings, live_source=source))
        return ""

    def set_live_hidden(self, hidden: bool) -> None:
        self.calls.append(("set_live_hidden", hidden))
        self.apply_settings(dataclasses.replace(self.settings, live_hide_from_share=hidden))

    def open_live_folder(self) -> None:
        self.calls.append(("open_live_folder",))

    def open_live_session(self, path) -> None:
        self.calls.append(("open_live_session", path))

    def copy_live_session(self, path, translation_only: bool = False) -> str:
        self.calls.append(("copy_live_session", path, translation_only))
        return ""

    _live_spend = None  # the running session's Spend while it saves, for screenshots and tests
    _live_last = None  # the last saving session's Spend
    _live_why = ""  # why the cost saver stopped the last session

    def set_live_save(self, on: bool) -> None:
        self.calls.append(("set_live_save", on))
        self.apply_settings(dataclasses.replace(self.settings, live_save=on))

    def live_spend(self):
        return self._live_spend if self._live and self.settings.live_save else None

    def live_last_spend(self):
        return self._live_last

    def live_stopped_why(self) -> str:
        return self._live_why

    _voice_state: tuple = ("missing", 0, "")  # the voice isn't downloaded in the preview

    def live_voice(self):
        return LOCAL_VOICE

    def live_voice_state(self) -> tuple:
        return self._voice_state

    def set_live_speak(self, on: bool) -> None:
        self.calls.append(("set_live_speak", on))
        self.apply_settings(dataclasses.replace(self.settings, live_speak=on))

    def set_live_speak_speed(self, speed: float) -> None:
        self.calls.append(("set_live_speak_speed", speed))
        self.apply_settings(dataclasses.replace(self.settings, live_speak_speed=speed))

    def set_live_duck(self, depth: float) -> None:
        self.calls.append(("set_live_duck", depth))
        self.apply_settings(dataclasses.replace(self.settings, live_duck=depth))

    # Meeting mode, for screenshots and tests: nothing is recorded, looked at or sent
    _meeting_windows: tuple = ()  # the meetings "open now" (MeetingWindow or alike: app, title, hwnd, pid)
    _meeting: tuple = ("", None, None, None, "")  # (state, window, folder, result, error), as TrayApp.meeting_status
    _meeting_cost = None  # the meeting's MeetingCost (or alike: total, seconds, frames)
    _past_meetings: tuple = ()  # (folder, started, app, seconds, has minutes)
    last_meeting = None

    def meeting_windows(self) -> list:
        return list(self._meeting_windows)

    def meeting_status(self) -> tuple:
        return self._meeting

    def meeting_cost(self):
        return self._meeting_cost if self._meeting[0] else None

    def meeting_problem(self, in_person: bool = False) -> str:
        s, kind = self.settings, "room" if in_person else "meeting"
        on = {part: getattr(s, f"{kind}_{part}") for part in ("translate", "record", "minutes")}
        if not (on["translate"] or on["record"]):
            return "Switch on live translation or Record first."
        if (on["translate"] or on["minutes"] and on["record"]) and not self.gateway.key_for("gemini"):
            return "Live translation and the minutes use Google Gemini: add a Gemini key in AI & models."
        return ""

    def start_meeting(self, window) -> str:
        self.calls.append(("start_meeting", getattr(window, "app", None)))
        problem = ("Start a meeting first: Teams, Zoom, Google Meet or Webex." if window is None
                   else self.meeting_problem(window.app == "In person"))
        if not problem:
            self._meeting = ("recording", window, Path("meetings") / f"2026-10-27 1003 {window.app}", None, "")
        return problem

    def stop_meeting(self) -> None:
        self.calls.append(("stop_meeting",))
        state, window, folder, _, _ = self._meeting
        if state == "recording":
            minutes = self.settings.room_minutes if window.app == "In person" else self.settings.meeting_minutes
            self._meeting = ("writing minutes" if minutes else "done", window, folder, None, "")

    def set_meeting_option(self, name: str, on: bool, in_person: bool = False) -> None:
        self.calls.append(("set_meeting_option", name, on, *(("in person",) if in_person else ())))
        self.apply_settings(dataclasses.replace(self.settings, **{f"{'room' if in_person else 'meeting'}_{name}": on}))

    def past_meetings(self, limit: int = 8) -> list:
        return list(self._past_meetings)[:limit]

    def open_meeting(self, folder) -> None:
        self.calls.append(("open_meeting", folder))

    writing_minutes: set = set()

    def write_minutes(self, folder) -> None:
        self.calls.append(("write_minutes", folder))

    def open_meeting_folder(self, folder=None) -> None:
        self.calls.append(("open_meeting_folder", folder))

    def copy_minutes(self, folder) -> str:
        self.calls.append(("copy_minutes", folder))
        return ""

    def meeting_shortcut_clash(self) -> str:
        s = self.settings
        for name, shortcut in (("Live translation", s.live_shortcut), ("Translate", s.translate_shortcut),
                               ("Text Transform", s.transform_shortcut)):
            if s.meeting_shortcut and shortcut == s.meeting_shortcut:
                return name
        return ""

    _wake: WakeState | None = None  # the wake word as the card shows it, for screenshots and tests

    def wake_state(self) -> WakeState:
        return self._wake or WakeState(size=57_900_000)

    def set_wake_enabled(self, on: bool) -> None:
        self.calls.append(("set_wake_enabled", on))
        self.apply_settings(dataclasses.replace(self.settings, wake_enabled=on))

    def download_wake_models(self) -> None:
        self.calls.append(("download_wake_models",))

    def cancel_wake_download(self) -> None:
        self.calls.append(("cancel_wake_download",))

    def open_wake_setup(self, parent=None) -> None:
        self.calls.append(("open_wake_setup",))  # the dialog itself is tested on its own (tests/test_wakeui.py)

    def run_translation(self, text: str, target: str, second: str = ""):
        from sst.translate import Translation
        self.calls.append(("run_translation", target))
        return Translation("¿Podrías enviarme el informe actualizado antes del viernes? El presupuesto es de $25,000.",
                           target, text, 0.9)

    def run_transform(self, text: str, key: str):
        from sst.transform import TransformResult
        self.calls.append(("run_transform", key))
        plain = "Deployment looks good, but the database migration issue needs to be fixed before production."
        return TransformResult(key, text, plain, plain, f"<p>{plain}</p>", True, [], 1, 0.8)

    def add_sound_alike(self, heard: str, meant: str) -> None:
        from sst.pipeline.dictionary import TermMode
        term = self.dictionary.find(meant)
        if term is None:
            self.dictionary.add_term(meant, aliases=[heard], mode=TermMode.AUTOMATIC, source="user")  # asked for: always
        else:
            self.dictionary.add_alias(term.id, heard)
        self.calls.append(("add_sound_alike", heard, meant))

    def score_reading(self, folders, progress):
        raise RuntimeError("No speech model in the preview.")

    def finish_welcome(self, name: str = "") -> None:
        if name.strip():
            self.profiles.current.name = name.strip()
        self.settings.welcomed = True
        self.calls.append(("finish_welcome", name))

    def switch_profile(self, profile_id: str) -> None:
        self.profiles.active = profile_id
        self.calls.append(("switch_profile", profile_id))

    def create_profile(self, name: str) -> None:
        self.switch_profile(self.profiles.add(name).id)

    def rename_profile(self, profile_id: str, name: str) -> None:
        if name.strip():
            self.profiles.get(profile_id).name = name.strip()

    def delete_profile(self, profile_id: str) -> None:
        self.profiles.remove(profile_id)
        self.calls.append(("delete_profile", profile_id))

    def window_closed(self) -> None:
        self.calls.append(("window_closed",))

    def start_over(self, keep_models: bool = True) -> None:
        self.calls.append(("start_over", keep_models))  # the preview deletes nothing

    def check_for_updates(self, manual: bool = False) -> None:
        self.calls.append(("check_for_updates", manual))

    def start_update(self) -> None:
        self.calls.append(("start_update",))

    def open_release_notes(self) -> None:
        self.calls.append(("open_release_notes",))
