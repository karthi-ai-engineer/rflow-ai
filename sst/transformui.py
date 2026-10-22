"""Text Transform (the owner's idea, 2026-10-02): speak normally first, transform the text afterwards, only on request.

  say "make it concise"         (holding the dictation key) the transform, right away: sst.commands, run_command
  double-tap Ctrl (the shortcut) the menu for the text selected in the focused app, or else all the text in its box
  1-9, T or a click             the transform (Concise, Professional, Bullet points, Action items..., T: Translate)
  U, or say "undo that"         undo: the last transform's original comes back (Ctrl+Z in the app works too)
  Esc                           close the menu; nothing changes

The text is the selection; with nothing selected, all the text in the box being typed in (the owner's rule of 2026-10-08:
Ctrl+A there), unless it is longer than BOX_LIMIT, a whole document: then the caret goes back where it was and the user
is asked to select a part. Rflow's own record of the last dictation (or transform) typed in that window, selected again
by counting caret steps, is only the fallback where Ctrl+A copies nothing, and how undo finds a transform of part of a
box. The menu takes no keyboard focus, so the app keeps its selection; the keys the menu needs are taken from the
keyboard hook while it is open. The text is transformed by the AI cleanup's model under a strict prompt and checked by
sst.transform.TransformGuard: a result that loses or invents anything isn't typed, and the user is told why. Before the
result replaces the text, its window is brought back and the text checked to still be selected there; if that can't be
made sure, the result goes on the clipboard instead ("press Ctrl+V"). Otherwise the user's clipboard is left as it was
(sst.textaccess).
"""
import logging
import threading
import time
from dataclasses import dataclass

from PySide6.QtCore import QObject, QPoint, QRectF, Qt, Signal
from PySide6.QtGui import QCursor, QFontMetrics, QGuiApplication, QPainter
from PySide6.QtWidgets import QWidget

from sst import theme
from sst.hotkey import HotkeyListener, parse_hotkey
from sst.transform import TRANSFORMS, menu_items

LAST_TEXT_SECONDS = 15 * 60  # the last dictation counts as "the text" this long, in the window it was typed into
BOX_LIMIT = 2000  # characters: with nothing selected, a box holding more is likely a whole document, not a message
TOO_LONG = "Select the part to rewrite: there is too much text here to rewrite all of it."
# Keys the open menu takes from the keyboard: 1-9, numpad 1-9, Up, Down, Enter, Esc, U (undo), T (Translate).
MENU_KEYS = frozenset([*range(0x31, 0x3A), *range(0x61, 0x6A), 0x26, 0x28, 0x0D, 0x1B, 0x55, 0x54])
UNDO = "undo"
# Console windows: Ctrl+C there stops the running program instead of copying, so Text Transform never presses it there.
TERMINALS = frozenset({"ConsoleWindowClass", "CASCADIA_HOSTING_WINDOW_CLASS", "mintty", "VirtualConsoleClass",
                       "PuTTY", "KiTTY"})

log = logging.getLogger(__name__)


@dataclass
class Target:
    text: str  # exactly as selected (or as typed), with its surrounding whitespace
    source: str  # "selection", "box" (all the text in the box, nothing being selected) or "last" (the last dictation or
    # transform, selected again by Rflow)
    hwnd: int  # the window it is in: the result is only pasted there
    before: str | None = None  # "box": the text that was before the caret, so the caret can go back (textaccess.unselect_all)


@dataclass
class Done:
    original: str
    pasted: str
    hwnd: int
    transform: str


def edges(text: str) -> tuple[str, str]:
    """The whitespace around a text: kept around its replacement, so a transform doesn't join it to its neighbours."""
    core = text.strip()
    if not core:
        return text, ""
    start = text.index(core[0])
    return text[:start], text[start + len(core):]


def same_text(a: str, b: str) -> bool:
    return " ".join(a.split()) == " ".join(b.split())


def place(access, text: str, hwnd: int, whole: bool = False) -> bool:
    """Make sure `text` is selected again in window `hwnd`, the moment before it is replaced: the user may have clicked
    elsewhere or switched windows while a menu or popup was open or the model answered. The window comes back to the
    front, and the text must be the selection, or be found right before the caret (as typed), or, when it was `whole`
    box, be all the text in the box again; else False, with nothing left selected."""
    if access.foreground_window() != hwnd:
        if not access.activate(hwnd):
            log.info("The text's window couldn't be brought back")
            return False
        log.info("The text's window brought back")
    copied = access.copy_selection(fallback=True)
    if copied is not None:  # something is selected: replace it only if it is still that text
        if not same_text(copied, text):
            log.info("Other text is selected now (%d characters)", len(copied))
        return same_text(copied, text)
    if whole:
        log.info("The text isn't selected any more; selecting all the text in the box again")
        before = access.text_before_caret()
        again = access.select_all()
        if again is not None and same_text(again, text):
            return True
        log.info("The box holds other text now (%d characters)", len(again or ""))
        access.unselect_all(before, again)
        return False
    log.info("The text isn't selected any more; looking for it before the caret")
    return access.select_last(text) is not None


class TransformMenu(QWidget):
    """A small panel at the mouse pointer, in the design's popup look: "Text Transform" and the source of the text, then
    one row per transform with its key drawn as a key (Undo apart, below a line). It never takes focus."""

    chosen = Signal(str)  # a transform key, or UNDO
    closed = Signal()

    ROW = 36
    WIDTH = 296  # the panel; the window has MARGIN more on each side, for its soft shadow
    MARGIN = 24
    TOP = 66  # from the panel's top to its first row: the title and the source above

    def __init__(self):
        super().__init__(None, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                         | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setMouseTracking(True)
        self.items: list[tuple[str, str, str]] = []  # (key, hint, label)
        self.source = ""
        self.current = 0
        theme.load_fonts()

    def _row_rect(self, i: int) -> QRectF:
        """Where row i is: Undo sits below a line, apart from the transforms."""
        apart = 9 if self.items and self.items[i][0] == UNDO and i > 0 else 0
        return QRectF(self.MARGIN + 12, self.MARGIN + self.TOP + i * self.ROW + apart, self.WIDTH - 24, self.ROW - 2)

    def _height(self) -> int:
        undo = any(key == UNDO for key, _, _ in self.items) and len(self.items) > 1
        return self.TOP + self.ROW * len(self.items) + (9 if undo else 0) + 12

    def open_at(self, pos: QPoint, items: list[tuple[str, str, str]], source: str) -> None:
        self.items, self.source, self.current = items, source, 0
        self.resize(self.WIDTH + 2 * self.MARGIN, self._height() + 2 * self.MARGIN)
        screen = QGuiApplication.screenAt(pos) or QGuiApplication.primaryScreen()
        area = screen.availableGeometry()
        height = self._height()
        x = min(max(area.left(), pos.x() + 12), area.right() - self.WIDTH)
        y = pos.y() + 16 if pos.y() + 16 + height <= area.bottom() else pos.y() - height - 8
        self.move(x - self.MARGIN, max(area.top(), y) - self.MARGIN)
        self.show()
        try:
            from sst.app import _no_activate  # like the pill, clicks never take the app's focus; unlike it, the
            _no_activate(int(self.winId()), click_through=False)  # menu's rows can be clicked
        except Exception:  # the off-screen test platform has no real window
            pass
        self.update()

    def key(self, vk: int) -> None:
        """A key taken from the keyboard while the menu is open."""
        if not self.isVisible() or not self.items:
            return
        if vk == 0x1B:
            self.close_menu()
        elif vk in (0x26, 0x28):
            self.current = (self.current + (1 if vk == 0x28 else -1)) % len(self.items)
            self.update()
        elif vk == 0x0D:
            self._choose(self.current)
        else:
            digit = chr(vk) if 0x31 <= vk <= 0x39 else chr(vk - 0x30) if 0x61 <= vk <= 0x69 else ""  # top row, numpad
            hint = chr(vk) if 0x41 <= vk <= 0x5A else digit  # a row's letter: U, T
            for i, (_, item_hint, _) in enumerate(self.items):
                if item_hint == hint:
                    self._choose(i)
                    return

    def close_menu(self) -> None:
        if self.isVisible():
            self.hide()
            self.closed.emit()

    def _choose(self, i: int) -> None:
        self.hide()
        self.chosen.emit(self.items[i][0])

    def _row_at(self, y: float) -> int | None:
        for i in range(len(self.items)):
            rect = self._row_rect(i)
            if rect.top() <= y < rect.top() + self.ROW:
                return i
        return None

    def mouseMoveEvent(self, event):
        i = self._row_at(event.position().y())
        if i is not None and i != self.current:
            self.current = i
            self.update()

    def mousePressEvent(self, event):
        i = self._row_at(event.position().y())
        if i is not None:
            self._choose(i)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        dpr = self.devicePixelRatioF()
        m, panel = self.MARGIN, QRectF(self.MARGIN, self.MARGIN, self.WIDTH, self._height())
        theme.paint_surface(p, "float", panel, 20, popup=True, dpr=dpr)
        text, quiet = theme.tok("text", popup=True), theme.tok("text3", popup=True)
        p.setPen(text)
        p.setFont(theme.font(14, 600))
        p.drawText(QRectF(m + 20, m + 14, self.WIDTH - 40, 20), Qt.AlignmentFlag.AlignVCenter, "Text Transform")
        small = theme.font(12, 500)
        p.setFont(small)
        p.setPen(quiet)
        p.drawText(QRectF(m + 20, m + 34, self.WIDTH - 40, 18), Qt.AlignmentFlag.AlignVCenter,
                   QFontMetrics(small).elidedText(self.source, Qt.TextElideMode.ElideRight, self.WIDTH - 40))
        for i, (key, hint, label) in enumerate(self.items):
            row = self._row_rect(i)
            if key == UNDO and i > 0:
                p.fillRect(QRectF(row.left() + 8, row.top() - 6, row.width() - 16, 1), theme.tok("line", popup=True))
            if i == self.current:
                theme.paint_surface(p, "pressed", row, 10, popup=True, dpr=dpr)
            cap = QRectF(row.left() + 10, row.center().y() - 10, 22, 20)
            theme.paint_surface(p, "keycap", cap, 6, popup=True, dpr=dpr)
            p.setPen(text)
            p.setFont(theme.font(11, 500, mono=True))
            p.drawText(cap.adjusted(0, -1, 0, -1), Qt.AlignmentFlag.AlignCenter, hint)
            p.setPen(text if key != UNDO else theme.tok("text2", popup=True))
            p.setFont(theme.font(14, 600 if i == self.current else 400))
            p.drawText(row.adjusted(44, 0, -8, 0), Qt.AlignmentFlag.AlignVCenter, label)
        p.end()


class TransformController(QObject):
    """The shortcut, the text, the menu, the transform and the replacement. `app` is the TrayApp (or a stand-in in the
    tests): it gives settings, `run_transform(text, key)`, `transform_ready()`, `say(state, message)` (the pill) and
    `remember(text, original)` (Home's history). `access` is sst.textaccess (fakes in the tests: nothing here presses a
    key or touches the clipboard by itself)."""

    _ready = Signal(object)  # a Target, or why there is no text (str), from the capture thread
    _done = Signal(object)  # (Target, TransformResult or error str), from the transform thread
    _restored = Signal(object)  # (Done, "") after an undo, or (None, "transformed:<name>" or what went wrong): the end
    _key = Signal()  # the shortcut's listener queued an event, from the keyboard hook's thread

    def __init__(self, app, access=None, listener_factory=HotkeyListener):
        super().__init__()
        if access is None:
            from sst import textaccess as access
        self.app, self.access, self._listener_factory = app, access, listener_factory
        self.listener: HotkeyListener | None = None
        self.menu = TransformMenu()
        self.menu.chosen.connect(self._chosen)
        self.menu.closed.connect(self._menu_closed)
        self._ready.connect(self._show_menu)
        self._done.connect(self._finish)
        self._restored.connect(self._after)
        self._key.connect(self._pump)  # queued from the hook's thread: nothing runs here between keys
        self.busy = False  # reading the text, transforming or pasting: the shortcut waits
        self.pending: Target | None = None  # the text the open menu is for
        self.last: Done | None = None  # the last transform, for undo
        self.last_typed: tuple[str, float, int] | None = None  # (text as typed, when, window): the "no selection" text

    # -- the shortcut

    def start(self, hotkey: str) -> None:
        """Listen for the shortcut ("" = Text Transform off)."""
        self.stop()
        if not hotkey:
            return
        try:
            self.listener = self._listener_factory(parse_hotkey(hotkey))
            self.listener.on_event = self._key.emit
            self.listener.start()
        except (ValueError, OSError) as e:
            log.warning("Text Transform's shortcut %r doesn't work: %s", hotkey, e)
            self.listener = None
            return
        log.info("Text Transform on %s", hotkey)

    def stop(self) -> None:
        self.menu.close_menu()
        if self.listener is not None:
            self.listener.stop()
            self.listener = None

    def note_typed(self, text: str, hwnd: int | None = None) -> None:
        """A dictation (or a transform) was just typed: the text where Ctrl+A copies nothing (_find)."""
        self.last_typed = (text, time.monotonic(), self.access.foreground_window() if hwnd is None else hwnd)

    def _pump(self) -> None:
        listener = self.listener
        if listener is None:
            return
        while not listener.events.empty():
            event, _ = listener.events.get_nowait()
            if event == "release":
                self.trigger()
            elif event.startswith("key:"):
                self.menu.key(int(event[4:]))

    def trigger(self) -> None:
        """The shortcut: the menu for the selected text (or all the text in the box)."""
        if self.menu.isVisible():  # the shortcut again closes the menu
            self.menu.close_menu()
            return
        if self.busy:
            return
        if not self.app.transform_ready():
            self.app.say("warning", "Text Transform needs an AI model: choose one in AI cleanup.")
            return
        self.busy = True
        threading.Thread(target=self._capture, args=(None,), name="transform-capture", daemon=True).start()

    def run_command(self, command: str) -> None:
        """A voice command (sst.commands): a transform, or UNDO, on the selected text or all the text in the box; no menu."""
        if self.busy or self.menu.isVisible():
            self.app.say("warning", "Text Transform is still busy with the last one.")
            return
        if command == UNDO and self.last is None:
            self.app.say("warning", "Nothing to undo: there was no transform yet.")
            return
        if command != UNDO and not self.app.transform_ready():
            self.app.say("warning", "Text Transform needs an AI model: choose one in AI cleanup.")
            return
        self.busy = True
        if command != UNDO:
            self.app.say("transforming", TRANSFORMS[command].name)
        threading.Thread(target=self._capture, args=(command,), name="transform-command", daemon=True).start()

    def _capture(self, command: str | None) -> None:
        """Find the text (on a thread: copying waits for the app), then the menu, or the command right away."""
        try:
            target = self._find(self.access.foreground_window(), undo=command == UNDO)
        except Exception as e:  # the clipboard was busy, a window closed...: say so, never leave Text Transform stuck
            log.exception("Reading the text to transform failed")
            target = f"Couldn't read the text ({e})"
        if command is None:
            self._ready.emit(target)
        elif isinstance(target, str):
            self._restored.emit((None, target))
        elif command == UNDO:
            self._undo(target)
        else:
            self._run(target, command)

    def _find(self, hwnd: int, undo: bool = False) -> Target | str:
        """The selected text, else all the text in the box (selected with Ctrl+A), else the last text typed in this window
        (selected again); or why there is none. For `undo`, the box only when it holds just the last transform."""
        cls = self.access.window_class(hwnd)
        if cls in TERMINALS:
            return "Text Transform doesn't work in a terminal: Ctrl+C there would stop the running program."
        text = self.access.copy_selection()
        if text and text.strip():
            log.info("Text Transform: %d characters selected in %s", len(text), cls)
            return Target(text, "selection", hwnd)
        # Nothing selected: all the text in the box (the owner's rule of 2026-10-08; counting caret steps back over the last
        # dictation missed by 3 in Teams' editor). The text before the caret is read first, so the caret can go back.
        before = self.access.text_before_caret()
        if before is not None and len(before) > BOX_LIMIT:
            log.info("Text Transform: nothing selected, and %d characters before the caret in %s", len(before), cls)
            return TOO_LONG
        whole = self.access.select_all()
        if whole is not None and len(whole) > BOX_LIMIT:
            log.info("Text Transform: nothing selected, and %d characters in the box in %s", len(whole), cls)
            self.access.unselect_all(before, whole)
            return TOO_LONG
        done = self.last if self.last and self.last.hwnd == hwnd else None
        if whole is not None and not (undo and done and not same_text(whole, done.pasted)):
            log.info("Text Transform: nothing selected; all the text in the box (%d characters) in %s", len(whole), cls)
            return Target(whole, "box", hwnd, before)
        self.access.unselect_all(before, whole)  # never leave a selection Rflow made behind
        # Ctrl+A copied nothing (an empty box, or an app without Ctrl+A or Ctrl+Insert), or undo of a transform of part of
        # the box: the text Rflow typed there last, selected again before the caret.
        last = (done.pasted, time.monotonic(), hwnd) if undo and done else self.last_typed
        if not last or last[2] != hwnd or time.monotonic() - last[1] >= LAST_TEXT_SECONDS:
            log.info("Text Transform: nothing selected in %s, and no recent dictation there", cls)
            return "Select some text first, then try again."
        if found := self.access.select_last(last[0]):  # as it is there now: the app may have changed it a little
            log.info("Text Transform: the last dictation (%d characters) selected again in %s", len(found), cls)
            return Target(found, "last", hwnd)
        log.info("Text Transform: the last dictation couldn't be found again in %s", cls)
        return "Rflow couldn't find your last dictation here to rewrite it. Select the text, then say it again."

    # -- the menu

    def _show_menu(self, target) -> None:
        self.busy = False
        if isinstance(target, str):
            self.app.say("warning", target)
            return
        self.pending = target
        items = menu_items(self.app.settings.transforms)  # Tone, Format, Language; 1-9, and T for Translate
        if self.last and self.last.hwnd == target.hwnd and same_text(target.text, self.last.pasted):
            items.append((UNDO, "U", "Undo: restore the original"))
        if not items:
            self.app.say("warning", "No transforms are chosen: pick some on the Text Transform page.")
            return
        words = len(target.text.split())
        kind = ("Selected text" if target.source == "selection" else
                "Your last transform" if self.last and same_text(target.text, self.last.pasted) else
                "All the text here" if target.source == "box" else "Your last dictation")
        self.menu.open_at(QCursor.pos(), items, f"{kind} · {words} word{'s' if words != 1 else ''}")
        if self.listener is not None:
            self.listener.capture(MENU_KEYS)

    def _menu_closed(self) -> None:
        if self.listener is not None:
            self.listener.capture(None)
        target, self.pending = self.pending, None  # the text stays as it was (and selected, unless Rflow selected it all)
        self._release(target)

    def _release(self, target: Target | None) -> None:
        """Nothing replaces the text: when Rflow selected all of the box for it, that selection goes (on a thread)."""
        if target is not None and target.source == "box":
            self.busy = True
            threading.Thread(target=self._unselect, args=(target, True), name="transform-release", daemon=True).start()

    def _unselect(self, target: Target, release: bool = False) -> None:
        """Unselect the box's text Rflow selected, the caret back where it was, if it is still selected there: the user may
        have clicked or typed meanwhile, and then their caret stays where they put it."""
        try:
            if target.source == "box" and self.access.foreground_window() == target.hwnd \
                    and same_text(self.access.copy_selection() or "", target.text):
                self.access.unselect_all(target.before, target.text)
        except Exception:
            log.exception("Unselecting the box's text failed")
        finally:
            if release:
                self.busy = False

    def _chosen(self, key: str) -> None:
        if self.listener is not None:
            self.listener.capture(None)
        target, self.pending = self.pending, None
        if target is None:
            return
        self.busy = True
        if key == UNDO:
            threading.Thread(target=self._undo, args=(target,), name="transform-undo", daemon=True).start()
            return
        self.app.say("transforming", TRANSFORMS[key].name)
        threading.Thread(target=self._run, args=(target, key), name="transform", daemon=True).start()

    # -- the transform and the replacement

    def _run(self, target: Target, key: str) -> None:
        try:
            self._done.emit((target, self.app.run_transform(target.text.strip(), key)))
        except Exception as e:  # no answer, no model, an error from the provider: the text stays as it was
            log.warning("Text Transform failed: %s", e)
            self._done.emit((target, str(e) or type(e).__name__))

    def _finish(self, payload) -> None:
        target, result = payload
        if isinstance(result, str):
            self.busy = False
            self.app.say("warning", f"Text Transform didn't work: {result}")
            self._release(target)
            return
        if not result.accepted:
            self.busy = False
            reason = result.reasons[0] if result.reasons else "the result changed the meaning"
            log.info("Text Transform %s kept the text: %s", result.transform, "; ".join(result.reasons))
            self.app.say("warning", f"Kept your text: {reason}")
            self._release(target)
            return
        lead, trail = edges(target.text)
        pasted = lead + result.plain + trail
        self.app.remember(pasted.strip(), target.text.strip())
        threading.Thread(target=self._paste, args=(target, result, pasted), name="transform-paste", daemon=True).start()

    def _place(self, target: Target) -> bool:
        return place(self.access, target.text, target.hwnd, whole=target.source == "box")

    def _paste(self, target: Target, result, pasted: str) -> None:
        try:
            if self._place(target):
                self.access.paste_rich(pasted, result.html or None)
                self.last = Done(target.text, pasted, target.hwnd, result.transform)
                self.last_typed = (pasted, time.monotonic(), target.hwnd)  # again: another transform, or undo
                self._restored.emit((None, f"transformed:{TRANSFORMS[result.transform].name}"))
            else:  # never paste over something else: the result waits on the clipboard instead
                self.access.set_clipboard(pasted.strip(), result.html or None)
                self._restored.emit((None, "Your text wasn't where it was any more, so the result is on the clipboard: "
                                           "press Ctrl+V."))
        except Exception as e:
            log.exception("Pasting the transform failed")
            self._restored.emit((None, f"Couldn't type the result ({e}); it is on Rflow's Home page."))

    def _undo(self, target: Target) -> None:
        last = self.last
        try:
            if last is None or not same_text(target.text, last.pasted):
                self._unselect(target)
                raise RuntimeError("undo works on the text a transform just typed; Ctrl+Z in the app works too")
            if not self._place(target):
                self.access.set_clipboard(last.original.strip())
                raise RuntimeError("Your text wasn't where it was any more, so the original is on the clipboard: press "
                                   "Ctrl+V")
            self.access.paste_rich(last.original)
            self.last = None
            self.last_typed = (last.original, time.monotonic(), target.hwnd)
            self._restored.emit((last, ""))
        except Exception as e:
            self._restored.emit((None, str(e)))

    def _after(self, payload) -> None:
        self.busy = False
        last, message = payload
        if message.startswith("transformed:"):
            self.app.say("transformed", f"Transformed: {message.split(':', 1)[1]}")
        elif last is not None:
            self.app.say("transformed", "Original restored")
        else:
            self.app.say("warning", message)
