"""Read and replace the text selected in whichever app has keyboard focus (Text Transform).

Apps don't share their text with other programs in one common way, so this does what the user would do: Ctrl+C copies
the selection, Ctrl+A selects all the text in the field, Shift+Left selects what Rflow has just typed, and Ctrl+V pastes
the result over the selection, as formatted HTML too for the editors that take it. The user's clipboard is put back
every time, except by set_clipboard: when the result can't go back where the text was, it is left on the clipboard for
the user to paste.

Every Windows call sits behind a small module-level function, so the tests replace them with fakes.
"""
import ctypes
import difflib
import logging
import time
import unicodedata
from ctypes import wintypes

from sst.hotkey import send_keys
from sst.paste import (
    CF_EXCLUDE_FROM_HISTORY,
    CF_UNICODETEXT,
    RESTORE_DELAY,
    VK_CONTROL,
    VK_SHIFT,
    VK_V,
    _clipboard,
    _put,
    _snapshot,
    _wait_for_modifiers_released,
    settle,
)

log = logging.getLogger(__name__)
# The text typed last, changed by the app as it was pasted: how many more caret steps are looked at, and how close the
# text there must be to what was typed (difflib's ratio; "Thank you." typed, "Thank you" there: 0.95).
LOOK_FURTHER, CLOSE_ENOUGH = 12, 0.85
user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

user32.GetForegroundWindow.restype = wintypes.HWND
user32.IsWindow.argtypes = [wintypes.HWND]
user32.IsIconic.argtypes = [wintypes.HWND]
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.BringWindowToTop.argtypes = [wintypes.HWND]
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, wintypes.BOOL]
kernel32.GetCurrentThreadId.restype = wintypes.DWORD
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetClipboardSequenceNumber.restype = wintypes.DWORD
user32.GetClipboardData.argtypes = [wintypes.UINT]
user32.GetClipboardData.restype = wintypes.HANDLE
user32.RegisterClipboardFormatW.argtypes = [wintypes.LPCWSTR]
user32.RegisterClipboardFormatW.restype = wintypes.UINT
kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
kernel32.GlobalLock.restype = wintypes.LPVOID
kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
kernel32.GlobalSize.argtypes = [wintypes.HGLOBAL]
kernel32.GlobalSize.restype = ctypes.c_size_t

VK_A, VK_C, VK_HOME, VK_LEFT, VK_RIGHT, VK_INSERT = 0x41, 0x43, 0x24, 0x25, 0x27, 0x2D
SW_RESTORE = 9
CF_HTML = user32.RegisterClipboardFormatW("HTML Format")  # where browsers, Office, Teams and Slack look for formatted text
_NO_HISTORY = (CF_EXCLUDE_FROM_HISTORY, b"\0" * 4)
_CTRL_C = [(VK_CONTROL, False), (VK_C, False), (VK_C, True), (VK_CONTROL, True)]
# Rflow copies with Ctrl+Insert, Windows' other copy key (browsers, Office, Qt, Electron, edit boxes): translators and
# clipboard tools act on Ctrl+C, and two of them in a second ("Ctrl+C+C") make one rewrite the clipboard or open a window.
_CTRL_INSERT = [(VK_CONTROL, False), (VK_INSERT, False), (VK_INSERT, True), (VK_CONTROL, True)]
SETTLE_DELAY = 0.15  # seconds between putting the text on the clipboard and pressing Ctrl+V (paste_rich)
_CTRL_V = [(VK_CONTROL, False), (VK_V, False), (VK_V, True), (VK_CONTROL, True)]
_BATCH = 50  # Shift+Left presses per SendInput call: a long dictation would otherwise be one array of thousands of events
_QUOTES = str.maketrans("\u2018\u2019\u201a\u201b\u201c\u201d\u201e\u201f", "''''\"\"\"\"")
_HTML_HEADER = "Version:0.9\r\nStartHTML:{:010d}\r\nEndHTML:{:010d}\r\nStartFragment:{:010d}\r\nEndFragment:{:010d}\r\n"


# ---------------------------------------------------------------- the focused window

def foreground_window() -> int:
    """The top-level window the user is working in (0 while none is, e.g. as the desktop switches)."""
    return user32.GetForegroundWindow() or 0


def window_title(hwnd: int) -> str:
    """The window's caption, for logs. Windows reads another process's caption without asking it, so a hung app can't
    block this."""
    buffer = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, buffer, len(buffer))
    return buffer.value


def window_process(hwnd: int) -> int:
    """The id of the process a window belongs to (0 for none): Rflow's own windows are told from the user's apps."""
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def window_class(hwnd: int) -> str:
    """The window's class name. Terminals (ConsoleWindowClass, CASCADIA_HOSTING_WINDOW_CLASS) are worth recognising: there
    Ctrl+C with nothing selected stops the running program instead of copying."""
    buffer = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buffer, len(buffer))
    return buffer.value


user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short


def mouse_down() -> bool:
    """A mouse button is held now (left, right or middle): Rflow's popups close on a click outside them."""
    return any(user32.GetAsyncKeyState(vk) & 0x8000 for vk in (0x01, 0x02, 0x04))


def activate(hwnd: int, timeout: float = 0.3) -> bool:
    """Bring the window `hwnd` back to the front (restored if minimised), so a paste goes there. True once it is the
    foreground window; False when it no longer exists or Windows kept another window in front."""
    if not hwnd or not _is_window(hwnd):
        return False
    if _is_iconic(hwnd):
        _show_window(hwnd, SW_RESTORE)
    elif foreground_window() == hwnd:
        return True
    # Windows lets a background app like Rflow take the foreground only with the input of the window in front: joined
    # to its thread (and the target's) for the call, SetForegroundWindow works instead of just flashing the taskbar.
    me = _current_thread()
    threads = dict.fromkeys(_window_thread(h) for h in (foreground_window(), hwnd) if h)
    attached = [thread for thread in threads if thread and thread != me and _attach_input(me, thread, True)]
    try:
        _bring_to_top(hwnd)
        _set_foreground(hwnd)
    finally:
        for thread in attached:  # never leave our input joined to another app's: its keyboard state would be shared
            _attach_input(me, thread, False)
    deadline = time.monotonic() + timeout
    while foreground_window() != hwnd:  # the switch can land a moment later
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


def _is_window(hwnd: int) -> bool:
    return bool(user32.IsWindow(hwnd))


def _is_iconic(hwnd: int) -> bool:
    return bool(user32.IsIconic(hwnd))


def _show_window(hwnd: int, command: int) -> None:
    user32.ShowWindow(hwnd, command)


def _bring_to_top(hwnd: int) -> None:
    user32.BringWindowToTop(hwnd)


def _set_foreground(hwnd: int) -> None:
    user32.SetForegroundWindow(hwnd)


def _window_thread(hwnd: int) -> int:
    return user32.GetWindowThreadProcessId(hwnd, None)


def _current_thread() -> int:
    return kernel32.GetCurrentThreadId()


def _attach_input(thread: int, to: int, attach: bool) -> bool:
    return bool(user32.AttachThreadInput(thread, to, attach))


# ---------------------------------------------------------------- reading the selection

def copy_selection(timeout: float = 0.6, fallback: bool = False) -> str | None:
    """The text selected in the focused app, copied with Ctrl+Insert ("\\r\\n" becomes "\\n"); with `fallback`, with
    Ctrl+C too when that copied nothing (an app without Ctrl+Insert, where text is known to be selected). None when
    nothing was copied: with no selection most apps leave the clipboard alone. Only whitespace counts as nothing too.
    Never raises for clipboard trouble, and the user's clipboard is put back unless they copied something new meanwhile."""
    _wait_for_modifiers_released()  # Ctrl+C pressed while the user still holds Alt from the shortcut would be Ctrl+Alt+C
    settle()  # a dictation's paste puts the user's clipboard back in the background: save theirs, not the dictation
    try:
        with _clipboard():
            saved, before = _snapshot(), _sequence()
    except OSError as error:  # without a copy of the user's clipboard, Ctrl+C would lose it
        log.warning("Could not read the selection: %s", error)
        return None
    for keys in (_CTRL_INSERT, _CTRL_C) if fallback else (_CTRL_INSERT,):
        send_keys(keys)
        deadline = time.monotonic() + timeout
        while _sequence() == before and time.monotonic() < deadline:
            time.sleep(0.005)
        if _sequence() != before:
            break
    else:
        return None  # nothing copied, so the user's clipboard was never touched: no need to rewrite it
    text = ours = None
    try:
        for attempt in range(3):  # an app writing a long selection (Word: text, HTML, RTF) may hold it open over a second
            try:
                with _clipboard():  # waits a second while the app still holds the clipboard open to write it
                    ours = _sequence()
                    text = _read_text()
                break
            except OSError:
                if attempt == 2:
                    raise
    except Exception:  # whatever went wrong, the user's clipboard must come back
        log.exception("Could not read the copied selection")
    finally:
        _restore(saved, ours)
    text = text.replace("\r\n", "\n") if text else ""
    return text if text.strip() else None


# ---------------------------------------------------------------- selecting what was just typed

def select_back(chars: int) -> None:
    """Select the `chars` caret steps before the caret (Shift+Left that many times)."""
    _wait_for_modifiers_released()  # Ctrl still held from the shortcut would make it Ctrl+Shift+Left: whole words
    if chars <= 0:
        return
    send_keys([(VK_SHIFT, False)])
    try:
        _steps(VK_LEFT, chars)
    finally:
        send_keys([(VK_SHIFT, True)])  # never leave Shift held down


def _steps(vk: int, count: int) -> None:
    for done in range(0, count, _BATCH):
        send_keys([(vk, False), (vk, True)] * min(_BATCH, count - done))


def collapse_selection(to_start: bool = False) -> None:
    """Press Right once: standard text boxes, browsers and Word collapse a selection to its end, where the caret was
    (Left, with `to_start`: to its start)."""
    _wait_for_modifiers_released()
    vk = VK_LEFT if to_start else VK_RIGHT
    send_keys([(vk, False), (vk, True)])


# ---------------------------------------------------------------- all the text in the field (nothing selected)

def text_before_caret() -> str | None:
    """The text from the start of the focused field to the caret: Ctrl+Shift+Home selects it, a copy reads it, and Right
    puts the caret back where it was (a selection collapses to its end). Read before select_all, so its selection can be
    undone with the caret where it was. None when nothing was copied (the caret at the start, or an app that doesn't
    select this way): then nothing more is pressed."""
    _wait_for_modifiers_released()
    send_keys([(VK_CONTROL, False), (VK_SHIFT, False), (VK_HOME, False), (VK_HOME, True), (VK_SHIFT, True),
               (VK_CONTROL, True)])
    before = copy_selection()
    if before is not None:
        collapse_selection()
    return before


def select_all() -> str | None:
    """Select all the text in the focused field (Ctrl+A) and return it, read with a copy: Text Transform's text when
    nothing is selected. The copy is Ctrl+Insert alone: with no text known to be selected, Ctrl+C could stop a program in
    a console Rflow doesn't recognise. None when nothing was copied (an empty field, an app without Ctrl+A or Ctrl+Insert);
    whatever Ctrl+A selected is still selected then (unselect_all)."""
    _wait_for_modifiers_released()
    send_keys([(VK_CONTROL, False), (VK_A, False), (VK_A, True), (VK_CONTROL, True)])
    return copy_selection()


def unselect_all(before: str | None, whole: str | None) -> None:
    """Undo select_all's selection, with the caret back where it was: `before` is the text that was before it
    (text_before_caret), `whole` what select_all copied. The caret goes to the end when it was there, or when that isn't
    known (where a dictation leaves it); else it takes the fewest caret steps from the start or from the end."""
    after = whole[len(before):] if whole and before and whole.startswith(before) else ""
    if not after:
        collapse_selection()
    elif _clusters(before) <= _clusters(after):
        collapse_selection(to_start=True)
        _steps(VK_RIGHT, _clusters(before))
    else:
        collapse_selection()
        _steps(VK_LEFT, _clusters(after))


def select_last(text: str) -> str | None:
    """Select `text`, which Rflow has just typed before the caret, check with a copy what is selected, and return it;
    None when it can't be found. Apps often change pasted text a little (the trailing space dropped, a comma or a
    capital added, quotes curled): then a few more caret steps are looked at, the closest text ending at the caret is
    found among them, and exactly that is selected. Never leaves a wrong selection behind."""
    plain = text.replace("\r\n", "\n")  # a line break is one caret step
    if not plain.strip():
        return None
    want = _loose(plain)
    # Most editors step over a whole cluster (a Tamil letter with its vowel sign, an emoji with its skin tone), a few over
    # each code point; the second count is tried only when they differ and the first selected the wrong text.
    for steps in dict.fromkeys((_clusters(plain), len(plain))):
        select_back(steps)
        copied = copy_selection(timeout=0.6 + steps / 500, fallback=True)  # the app works through every Shift+Left
        if copied is not None and _loose(copied) == want:
            return copied
        log.info("The text typed last wasn't found before the caret (%d steps): %s", steps,
                 "nothing copied" if copied is None else f"{len(copied)} characters copied, {len(plain)} expected")
        collapse_selection()
        if copied is None:
            return None  # nothing copied: this app doesn't select or copy this way, and another count won't change that
    return _select_close(plain, want)


def _select_close(plain: str, want: str) -> str | None:
    """The app changed the text as it was pasted: look LOOK_FURTHER more steps back, find where the typed text starts
    there (closely, not exactly), and select from there to the caret."""
    look = len(plain) + LOOK_FURTHER
    select_back(look)
    around = copy_selection(timeout=0.6 + look / 500, fallback=True)
    collapse_selection()
    start = _closest_start(around, want) if around else None
    if start is None:
        log.info("Nothing before the caret is close to the text typed last (%d characters looked at)",
                 len(around or ""))
        return None
    found = around[start:].replace("\r\n", "\n")
    for steps in dict.fromkeys((_clusters(found), len(found))):
        select_back(steps)
        copied = copy_selection(timeout=0.6 + steps / 500, fallback=True)
        if copied is not None and _loose(copied) == _loose(found):
            log.info("The text typed last was found as the app changed it (%d characters typed, %d there)",
                     len(plain), len(found))
            return copied
        collapse_selection()
        if copied is None:
            break
    return None


def _closest_start(around: str, want: str) -> int | None:
    """Where in `around` (the text before the caret) the text typed last starts, allowing the small changes apps make
    (difflib's ratio of at least CLOSE_ENOUGH, compared loosely); None when nothing there is that close."""
    # The app changes a few characters, not a fifth of a long dictation: the starts looked at are bounded (a 3,300-character
    # dictation took 50 s here, with Text Transform showing nothing meanwhile), and `want` is indexed once for all of them.
    best, best_score, slack = None, CLOSE_ENOUGH, max(3, min(len(want) // 5, 2 * LOOK_FURTHER + 8))
    matcher = difflib.SequenceMatcher(None, autojunk=False)
    matcher.set_seq2(want)
    for start in range(len(around)):
        tail = _loose(around[start:])
        if not tail:
            break
        if around[start].isspace() or abs(len(tail) - len(want)) > slack:
            continue  # a selection never starts on a space
        matcher.set_seq1(tail)
        if matcher.real_quick_ratio() < best_score or matcher.quick_ratio() < best_score:
            continue  # can't beat the best one: the full comparison isn't needed
        score = matcher.ratio()
        if score > best_score or (score == best_score and best is None):
            best, best_score = start, score
    return best


def _loose(text: str) -> str:
    # Word's AutoCorrect changes quotes and capitals as Rflow types, without changing how many characters there are.
    return " ".join(text.translate(_QUOTES).casefold().split())


def _clusters(text: str) -> int:
    """How many user-perceived characters `text` has, close to Unicode's grapheme clusters: combining marks (accents,
    vowel signs, viramas, variation selectors), skin tones and zero-width-joined emoji join the character before them,
    and two regional indicators make one flag."""
    count, joining, flag = 0, False, False
    for char in text:
        code = ord(char)
        regional = 0x1F1E6 <= code <= 0x1F1FF
        extends = count and (joining or code == 0x200D or (regional and flag) or 0x1F3FB <= code <= 0x1F3FF
                             or unicodedata.category(char) in ("Mn", "Mc", "Me"))
        count += not extends
        flag = regional and not (flag and extends)
        joining = code == 0x200D
    return count


# ---------------------------------------------------------------- pasting the result

def paste_rich(text: str, html: str | None = None) -> None:
    """Paste over the selection in the focused app: `text` for plain editors and, when given, the HTML fragment `html`
    for rich ones (Word, Outlook, Teams, Slack, browsers), so headings and lists arrive formatted. The user's clipboard
    comes back after RESTORE_DELAY, unless they copied something new meanwhile."""
    items = _items(text, html)
    _wait_for_modifiers_released()  # Ctrl+V pressed while the user still holds Alt would arrive as Ctrl+Alt+V
    settle()
    with _clipboard():
        saved = _snapshot()
        _put(items + [_NO_HISTORY])
    ours = _sequence()
    try:
        # An app that has just copied (Text Transform's check) still owns the clipboard in its own eyes until it hears of
        # the change: Ctrl+V at once would paste what it copied, over itself. A moment lets the news arrive.
        time.sleep(SETTLE_DELAY)
        send_keys(_CTRL_V)
        # The app reads the clipboard when it handles Ctrl+V, a moment later; longer for a long text with its HTML (a rich
        # editor busy laying it out could otherwise read the user's clipboard back instead).
        time.sleep(RESTORE_DELAY + min(1.0, len(text) / 10000))
    finally:
        _restore(saved, ours)


def clipboard_sequence() -> int:
    """Windows' count of clipboard changes: it moves when the user's app copies (Ctrl+C+C, the translator)."""
    return _sequence()


def clipboard_text() -> str | None:
    """The text the user copied ("\\r\\n" as "\\n"), read without pressing anything; None when there is none, or the
    clipboard stays busy."""
    try:
        with _clipboard():
            text = _read_text()
    except OSError as error:
        log.warning("Could not read the clipboard: %s", error)
        return None
    text = text.replace("\r\n", "\n") if text else ""
    return text if text.strip() else None


def set_clipboard(text: str, html: str | None = None) -> None:
    """Put `text` (and, when given, the HTML fragment `html`) on the clipboard for the user to paste themselves: the
    fallback when the result can't go back where the text was. Unlike paste_rich this is a copy the user wants, so it
    replaces theirs for good and is kept in Win+V history. Raises OSError while another app holds the clipboard."""
    settle()  # or a dictation's restore, still to come, would replace it
    with _clipboard():
        _put(_items(text, html))


def _items(text: str, html: str | None) -> list[tuple[int, bytes]]:
    # Windows text on the clipboard has "\r\n" line breaks; the classic edit box shows a bare "\n" as nothing.
    items = [(CF_UNICODETEXT, (text.replace("\r\n", "\n").replace("\n", "\r\n") + "\0").encode("utf-16-le"))]
    if html is not None:
        items.append((CF_HTML, cf_html(html) + b"\0"))
    return items


def cf_html(fragment: str) -> bytes:
    """The clipboard's "HTML Format" for an HTML fragment: a header with the byte offsets (in the UTF-8 encoding) of the
    whole document and of the fragment, which is what the app pastes."""
    head, body, tail = b"<html><body>\r\n<!--StartFragment-->", fragment.encode("utf-8"), b"<!--EndFragment-->\r\n</body></html>"
    start = len(_HTML_HEADER.format(0, 0, 0, 0))  # the numbers have a fixed width, so the header's length is known
    fragment_start = start + len(head)
    fragment_end = fragment_start + len(body)
    header = _HTML_HEADER.format(start, fragment_end + len(tail), fragment_start, fragment_end)
    return header.encode("ascii") + head + body + tail


# ---------------------------------------------------------------- the clipboard

def _restore(saved: list[tuple[int, bytes]], ours: int | None) -> None:
    """Put the user's clipboard back, unless it changed since `ours` (the user copied something new)."""
    try:
        with _clipboard():
            if ours is None or _sequence() == ours:
                _put([item for item in saved if item[0] != CF_EXCLUDE_FROM_HISTORY] + [_NO_HISTORY] if saved else [])
    except OSError as error:
        log.warning("Could not put the clipboard back: %s", error)


def _sequence() -> int:
    """Windows counts every change to the clipboard; a new number means someone copied."""
    return user32.GetClipboardSequenceNumber()


def _read_text() -> str | None:
    """The clipboard's text (the clipboard must be open)."""
    handle = user32.GetClipboardData(CF_UNICODETEXT)
    size = kernel32.GlobalSize(handle) if handle else 0
    pointer = kernel32.GlobalLock(handle) if size else None
    if not pointer:
        return None
    try:
        return ctypes.wstring_at(pointer, size // 2).split("\0", 1)[0]  # the block can be longer than the text
    finally:
        kernel32.GlobalUnlock(handle)
