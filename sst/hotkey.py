"""Global hotkeys on Windows through a low-level keyboard hook (the way Wispr Flow does it).

A hotkey is modifiers plus an optional key, or a double tap of Ctrl or Shift:
  ctrl+win     modifiers held together on their own    (default, like Wispr Flow)
  menu         the Menu key, next to right Alt         its right-click menu is blocked
  ctrl+alt+d   modifiers + a key                       the key is blocked, the modifiers pass through
  double ctrl  Ctrl tapped twice quickly, on its own   Ctrl passes through: alone it does nothing in apps
  ctrl+c+c     Ctrl+C pressed twice quickly            nothing is blocked: the app copies as usual (the translator)

One hook (HOOK), on its own thread, serves every listener: Windows calls a low-level hook for every key typed in any
app, so four hooks cost four trips through Python per key. Each listener reports "press", "release", "cancel" (Esc
while recording), "handsfree" (Space added to a modifier-only hotkey: Ctrl+Win+Space, as in Wispr Flow) and "interrupt"
(any other key added, e.g. Ctrl+Win+D: a Windows shortcut, not dictation) on its queue, and calls its `on_event` then, so
the app wakes up only when there is something to do (no timer polling the queue). A double tap has no "press": it
reports "release" as the second tap is let go.

While a menu that must not take the keyboard focus is open (Text Transform's, so the app keeps its selection), the
listener can capture keys: their presses are hidden from every app and reported as "key:<virtual-key code>".
"""
import ctypes
import logging
import queue
import threading
import time
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass

log = logging.getLogger(__name__)
user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

LRESULT = ctypes.c_ssize_t
HOOKPROC = ctypes.WINFUNCTYPE(LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
user32.SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD]
user32.SetWindowsHookExW.restype = wintypes.HHOOK
user32.CallNextHookEx.argtypes = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
user32.CallNextHookEx.restype = LRESULT
user32.UnhookWindowsHookEx.argtypes = [wintypes.HHOOK]
user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT]
user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short
user32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
kernel32.GetModuleHandleW.restype = wintypes.HMODULE

WH_KEYBOARD_LL, HC_ACTION, WM_QUIT, WM_TIMER = 13, 0, 0x0012, 0x0113
# Windows silently drops a keyboard hook that answers too slowly (Python busy loading a model), and nothing says so:
# a fresh one is put in this often while no key is held, so the hotkey can't stay dead until Rflow restarts.
REHOOK_SECONDS = 30
user32.SetTimer.argtypes = [wintypes.HWND, ctypes.c_size_t, wintypes.UINT, ctypes.c_void_p]
user32.SetTimer.restype = ctypes.c_size_t
user32.KillTimer.argtypes = [wintypes.HWND, ctypes.c_size_t]
WM_KEYDOWN, WM_SYSKEYDOWN = 0x0100, 0x0104
VK_ESCAPE, VK_SPACE, VK_MASK = 0x1B, 0x20, 0xE8  # 0xE8 is unassigned: pressing it tells Windows "Win was used with another key"
OUR_INPUT = 0x53535431  # dwExtraInfo on keys we send ourselves, so the hook lets them through untouched


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD), ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class MOUSEINPUT(ctypes.Structure):  # only here so INPUT has the size SendInput expects
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class INPUT(ctypes.Structure):
    class _Union(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]

    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _Union)]


user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.MapVirtualKeyW.argtypes = [wintypes.UINT, wintypes.UINT]
INPUT_KEYBOARD, KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP = 1, 0x1, 0x2
# The keys of the navigation block (arrows, Home, End, Page Up/Down, Insert, Delete), and the right-hand Ctrl and Alt, the
# Windows and Menu keys. Sent without the extended flag, an arrow is the number pad's: with NumLock on, Windows then
# lifts Shift around it, and Shift+Left moves the caret instead of selecting (Text Transform found nothing to transform).
EXTENDED_KEYS = frozenset({0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E, 0x5B, 0x5C, 0x5D, 0xA3, 0xA5})


def key_input(vk: int, up: bool) -> INPUT:
    """One key event as SendInput takes it, with its scan code, like a real keyboard's."""
    flags = (KEYEVENTF_KEYUP if up else 0) | (KEYEVENTF_EXTENDEDKEY if vk in EXTENDED_KEYS else 0)
    return INPUT(type=INPUT_KEYBOARD, ki=KEYBDINPUT(wVk=vk, wScan=user32.MapVirtualKeyW(vk, 0) & 0xFF, dwFlags=flags,
                                                    dwExtraInfo=OUR_INPUT))


def send_keys(events: list[tuple[int, bool]]) -> None:
    """Press/release keys as (virtual-key code, is_release), marked so our own hook ignores them."""
    inputs = (INPUT * len(events))(*(key_input(vk, up) for vk, up in events))
    user32.SendInput(len(inputs), inputs, ctypes.sizeof(INPUT))


# The hook reports left/right variants (0xA0-0xA5, 0x5B/0x5C); the generic codes appear in injected input.
MODIFIER_OF = {0x10: "shift", 0xA0: "shift", 0xA1: "shift", 0x11: "ctrl", 0xA2: "ctrl", 0xA3: "ctrl",
               0x12: "alt", 0xA4: "alt", 0xA5: "alt", 0x5B: "win", 0x5C: "win"}
_MODIFIER_NAMES = {"ctrl": "ctrl", "control": "ctrl", "alt": "alt", "shift": "shift", "win": "win"}
_KEYS = {"menu": 0x5D, "apps": 0x5D, "space": 0x20, "enter": 0x0D, "tab": 0x09, "insert": 0x2D, "home": 0x24,
         "end": 0x23, "pageup": 0x21, "pagedown": 0x22, "pause": 0x13, "scrolllock": 0x91,
         "muhenkan": 0x1D, "henkan": 0x1C}  # 無変換 / 変換 on Japanese keyboards
_KEYS.update({f"f{n}": 0x6F + n for n in range(1, 25)})
_KEYS.update({c: ord(c.upper()) for c in "abcdefghijklmnopqrstuvwxyz0123456789"})
_DOUBLE_TAPS = ("ctrl", "shift")  # Alt or Win tapped alone opens the menu bar or the Start menu, so they can't be tapped
# A double tap: each tap is held shorter than TAP_HOLD, and the second starts within TAP_GAP of the first one's release.
TAP_HOLD, TAP_GAP = 0.35, 0.4
# ...and the mouse pointer stays put: two quick Ctrl+clicks (selecting files, opening links) move it between the clicks.
# The keyboard hook can't see clicks, and a mouse hook would slow every mouse move down, so the pointer tells instead.
TAP_POINTER_PX = 12
PRESS_GAP = 0.5  # ctrl+c+c: the second press of the key within this many seconds of the first


@dataclass(frozen=True)
class Hotkey:
    text: str
    modifiers: frozenset[str]
    key: int | None  # None: the modifiers alone are the hotkey
    double: bool = False  # the one modifier tapped twice (key None), or the key pressed twice (ctrl+c+c)

    @property
    def label(self) -> str:
        if self.double and self.key is None:
            return f"Double-tap {min(self.modifiers).capitalize()}"
        return "+".join("Menu key" if part == "menu" else part.capitalize() for part in self.text.split("+"))


def parse_hotkey(text: str) -> Hotkey:
    """'ctrl+win', 'menu', 'ctrl+alt+d', 'double ctrl' (also 'double-shift') or 'ctrl+c+c' -> Hotkey."""
    words = text.lower().replace("-", " ").split()
    if len(words) == 2 and words[0] == "double":
        modifier = _MODIFIER_NAMES.get(words[1])
        if modifier not in _DOUBLE_TAPS:
            raise ValueError(f"'{text}' can't be double-tapped; use double ctrl or double shift")
        return Hotkey(f"double {modifier}", frozenset({modifier}), None, double=True)
    parts = [part.strip().lower() for part in text.split("+")]
    if len(parts) >= 3 and parts[-1] == parts[-2] and parts[-1] in _KEYS:  # "ctrl+c+c": the key pressed twice
        single = parse_hotkey("+".join(parts[:-1]))
        if not single.modifiers:
            raise ValueError(f"'{text}' needs a modifier, like ctrl+c+c")
        return Hotkey("+".join(parts), single.modifiers, single.key, double=True)
    modifiers, key = set(), None
    for i, part in enumerate(parts):
        if part in _MODIFIER_NAMES:
            modifiers.add(_MODIFIER_NAMES[part])
        elif part in _KEYS and i == len(parts) - 1:
            key = _KEYS[part]
        else:
            raise ValueError(f"unknown key '{part}' in hotkey '{text}'. Use e.g. ctrl+win, menu or ctrl+alt+d")
    if key is None and len(modifiers) < 2:
        raise ValueError(f"'{text}' alone would get in the way of normal typing; combine two modifiers (e.g. ctrl+win)")
    return Hotkey("+".join(parts), frozenset(modifiers), key)


def _pointer() -> tuple[int, int]:
    point = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(point))
    return point.x, point.y


class Matcher:
    """Decides, for each physical key event, whether the hotkey went down or up and whether other apps
    should see the event. No Windows calls, so every case can be tested without a keyboard."""

    def __init__(self, hotkey: Hotkey, is_held: Callable[[int], bool] | None = None,
                 clock: Callable[[], float] = time.monotonic, pointer: Callable[[], tuple[int, int]] | None = None):
        self.hotkey = hotkey
        self.recording = False  # set by the dictation loop: Esc only cancels while recording
        self._is_held = is_held  # asks Windows whether a key is really down
        self._clock = clock  # times a double tap's taps
        self._pointer = pointer  # where the mouse pointer is (None: not checked)
        self._tap_at: tuple[int, int] | None = None  # double tap: the pointer as the first tap of a pair went down
        self._down: set[int] = set()  # keys physically held now
        self._blocked: set[int] = set()  # keys whose repeats and release must stay hidden too
        self._active = False  # the hotkey is held
        self.capture: frozenset[int] = frozenset()  # keys a menu takes while it is open (HotkeyListener.capture)
        self._captured: set[int] = set()  # keys whose press went to the menu: their repeats and release are hidden too
        self._tap: tuple[int, float] | None = None  # double tap: the modifier key down as a tap, and since when
        self._first_tap: float | None = None  # double tap: when the first tap of a pair was let go
        self._first_press: float | None = None  # ctrl+c+c: when the key was first pressed with the modifiers

    def feed(self, vk: int, is_down: bool) -> tuple[bool, str | None]:
        """Returns (hide the event from other apps, event or None)."""
        if is_down and self._is_held:
            # The hook misses releases on the lock screen (Win+L) and in admin windows; drop keys that were let go.
            self._down = {k for k in self._down if k == vk or self._is_held(k)}
            self._blocked &= self._down
            self._captured &= self._down
            if self._active and not (self.hotkey.key in self._down if self.hotkey.key else self._hotkey_held()):
                self._active = False
        repeat = is_down and vk in self._down
        (self._down.add if is_down else self._down.discard)(vk)
        # Every key counts, even one a menu captures or a repeat: any key but the tapped modifier breaks a double tap.
        tapped = self.hotkey.double and (self._double_tap(vk, is_down, repeat) if self.hotkey.key is None
                                         else self._double_press(vk, is_down, repeat))

        # The app never saw a captured key go down, so its repeats and release stay hidden even after the capture ends;
        # a key already held when the capture began is left to the app, so it can't get stuck there.
        if vk in self._captured or (is_down and not repeat and self._captures(vk)):
            if not is_down:
                self._captured.discard(vk)
                return True, None
            self._captured.add(vk)
            return True, f"key:{vk}" if vk in self.capture else None  # repeats too: holding Down walks the menu
        if vk in self._blocked:
            if is_down:
                return True, None
            self._blocked.discard(vk)
            if self._active and vk == self.hotkey.key:
                self._active = False
                return True, "release"
            return True, None
        if repeat:
            return False, None
        if is_down and vk == VK_ESCAPE and self.recording:
            self._blocked.add(vk)
            return True, "cancel"
        if self.hotkey.double:
            return False, "release" if tapped else None  # the apps see every tap: Ctrl or Shift alone does nothing there
        if self.hotkey.key is not None:
            if is_down and vk == self.hotkey.key and self._held_modifiers() == self.hotkey.modifiers:
                self._blocked.add(vk)
                self._active = True
                return True, "press"
            return False, None
        return self._modifier_only(vk, is_down)

    def _modifier_only(self, vk: int, is_down: bool) -> tuple[bool, str | None]:
        wanted = self.hotkey.modifiers
        if is_down:
            if self._active and vk == VK_SPACE:
                self._active = False  # Ctrl+Win+Space: hands-free, as in Wispr Flow; Windows doesn't get the Space
                self._blocked.add(vk)
                return True, "handsfree"
            if self._active and MODIFIER_OF.get(vk) not in wanted:
                # Another key joined: a Windows shortcut (Ctrl+Win+D) when it comes at once, else a key brushed while
                # speaking. The dictation tells them apart by the time; the hotkey stays held, so its release comes.
                return False, "interrupt"
            if not self._active and self._held_modifiers() == wanted and all(k in MODIFIER_OF for k in self._down):
                self._active = True
                return False, "press"
        elif self._active and MODIFIER_OF.get(vk) in wanted and MODIFIER_OF[vk] not in self._held_modifiers():
            self._active = False
            return False, "release"
        return False, None

    def _double_tap(self, vk: int, is_down: bool, repeat: bool) -> bool:
        """Follows the taps of a double-tap hotkey's modifier (left and right count alike); True as the second tap of a
        pair is let go."""
        now = self._clock()
        if is_down and not repeat and MODIFIER_OF.get(vk) in self.hotkey.modifiers and self._down == {vk}:
            if self._first_tap is not None and now - self._first_tap >= TAP_GAP:
                self._first_tap = None  # too late to be the second tap, but it can be the first of a new pair
            if self._first_tap is None and self._pointer is not None:
                self._tap_at = self._pointer()
            self._tap = (vk, now)
            return False
        if not is_down and self._tap and self._tap[0] == vk:
            quick, first = now - self._tap[1] < TAP_HOLD, self._first_tap
            self._tap = self._first_tap = None
            if quick and first is not None:
                if not self._pointer_moved():
                    return True  # the pair is used up: a third tap starts a new one
                if self._pointer is not None:  # Ctrl+clicks, not a double tap; this tap may be the first of a real one
                    self._tap_at = self._pointer()
            self._first_tap = now if quick else None
            return False
        # Another key (a shortcut like Ctrl+C), the modifier pressed with a key held, or a repeat: it's held too long.
        self._tap = self._first_tap = None
        return False

    def _double_press(self, vk: int, is_down: bool, repeat: bool) -> bool:
        """ctrl+c+c: True as the key goes down the second time with exactly the modifiers held, soon after the first.
        Ctrl may stay held or be pressed again in between; any other key, or a key held down (repeats), breaks it."""
        if not is_down or MODIFIER_OF.get(vk) in self.hotkey.modifiers:
            return False
        now = self._clock()
        if vk != self.hotkey.key or repeat or self._held_modifiers() != self.hotkey.modifiers:
            self._first_press = None
            return False
        if self._first_press is not None and now - self._first_press < PRESS_GAP:
            self._first_press = None  # a third press starts a new pair
            return True
        self._first_press = now
        return False

    def _pointer_moved(self) -> bool:
        if self._pointer is None or self._tap_at is None:
            return False
        (x0, y0), (x1, y1) = self._tap_at, self._pointer()
        return max(abs(x1 - x0), abs(y1 - y0)) > TAP_POINTER_PX

    def _captures(self, vk: int) -> bool:
        if vk not in self.capture or (vk == VK_ESCAPE and self.recording):
            return False  # Esc cancels a recording first, the innermost thing going on; the next Esc reaches the menu
        # The hotkey keeps working while a menu is open, even when its key is one the menu takes.
        return not (vk == self.hotkey.key and self._held_modifiers() == self.hotkey.modifiers)

    def _held_modifiers(self) -> frozenset[str]:
        return frozenset(MODIFIER_OF[vk] for vk in self._down if vk in MODIFIER_OF)

    @property
    def idle(self) -> bool:
        """No key held and no hotkey under way: a moment to put in a fresh hook."""
        return not self._down and not self._active

    def _hotkey_held(self) -> bool:
        return self.hotkey.modifiers <= self._held_modifiers()


class HotkeyListener:
    """One shortcut, watched through the shared keyboard hook (HOOK); events arrive on `events` as
    (event, time.monotonic()), and `on_event()` (if set) is called on the hook's thread after each one is queued: the
    app connects it to a Qt signal, so its thread wakes up only for an event."""

    def __init__(self, hotkey: Hotkey):
        self.hotkey = hotkey
        self.events: queue.Queue[tuple[str, float]] = queue.Queue()
        self.on_event: Callable[[], None] | None = None
        self._matcher = Matcher(hotkey, is_held=lambda vk: bool(user32.GetAsyncKeyState(vk) & 0x8000),
                                pointer=_pointer if hotkey.double else None)
        # Pressing Win or Alt with no other key opens the Start menu / an app's menu bar on release.
        self._mask = bool(hotkey.modifiers & {"win", "alt"})

    @property
    def recording(self) -> bool:
        return self._matcher.recording

    @recording.setter
    def recording(self, value: bool) -> None:
        self._matcher.recording = value

    def capture(self, keys: set[int] | None) -> None:
        """Take these keys (virtual-key codes) from every app and report their presses as "key:<code>" events, for a
        menu that leaves the keyboard focus with the app; None gives them back."""
        self._matcher.capture = frozenset(keys or ())

    def start(self) -> None:
        """Start listening, through the shared hook (installed with the first listener)."""
        HOOK.add(self)

    def stop(self) -> None:
        HOOK.remove(self)

    def feed(self, vk: int, is_down: bool) -> bool:
        """One physical key event, on the hook's thread: True hides it from the other apps. Kept short (no waiting):
        Windows holds every app's keystroke until the hook answers."""
        hide, event = self._matcher.feed(vk, is_down)
        if event:
            # A captured key is hidden, so a Win or Alt held with it would look pressed alone on release.
            if (event == "press" and self._mask) or (
                    event.startswith("key:") and self._matcher._held_modifiers() & {"win", "alt"}):
                send_keys([(VK_MASK, False), (VK_MASK, True)])
            self.events.put((event, time.monotonic()))
            if self.on_event is not None:
                self.on_event()
        return hide


class KeyboardHook:
    """The one low-level keyboard hook, on its own thread, that feeds every started HotkeyListener (dictation, Text
    Transform, Translate, live translation). It is installed with the first listener and removed with the last."""

    def __init__(self):
        self.listeners: tuple[HotkeyListener, ...] = ()  # replaced, never changed in place: the hook reads it unlocked
        self._lock = threading.Lock()
        self._thread_id = 0  # the hook's thread, while it runs
        self._ident = 0  # the same thread, as Python knows it: a hook still ending after a restart feeds nobody
        self._ready = threading.Event()
        self._error = ""

    def add(self, listener: HotkeyListener) -> None:
        with self._lock:
            if listener in self.listeners:
                return
            self.listeners = (*self.listeners, listener)
            if self._thread_id:
                return
            self._ready.clear()
            self._error = ""
            threading.Thread(target=self._run, name="keyboard-hook", daemon=True).start()
            self._ready.wait(5)
            if self._error:
                self.listeners = tuple(item for item in self.listeners if item is not listener)
                raise OSError(self._error)

    def remove(self, listener: HotkeyListener) -> None:
        with self._lock:
            self.listeners = tuple(item for item in self.listeners if item is not listener)
            if not self.listeners and self._thread_id:
                user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
                self._thread_id = self._ident = 0

    @property
    def idle(self) -> bool:
        return all(listener._matcher.idle for listener in self.listeners)

    def _run(self) -> None:
        proc = HOOKPROC(self._on_key)  # kept while the hook lives: Windows calls it
        self._thread_id, self._ident = kernel32.GetCurrentThreadId(), threading.get_ident()
        hook = self._hook(proc)
        if not hook:
            self._error = f"could not install the keyboard hook (Windows error {ctypes.get_last_error()})"
            self._thread_id = self._ident = 0
        self._ready.set()
        if not hook:
            return
        timer = user32.SetTimer(None, 0, REHOOK_SECONDS * 1000, None)
        msg = wintypes.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:  # Windows calls the hook from in here
                if msg.message == WM_TIMER and self.idle and (fresh := self._hook(proc)):
                    user32.UnhookWindowsHookEx(hook)  # the new one is in first: there's never a moment with none
                    hook = fresh
        finally:
            if timer:
                user32.KillTimer(None, timer)
            user32.UnhookWindowsHookEx(hook)

    @staticmethod
    def _hook(proc):
        return user32.SetWindowsHookExW(WH_KEYBOARD_LL, proc, kernel32.GetModuleHandleW(None), 0)

    def _on_key(self, code: int, wparam: int, lparam: int) -> int:
        try:
            if code == HC_ACTION and threading.get_ident() == self._ident:
                info = ctypes.cast(lparam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
                if info.dwExtraInfo != OUR_INPUT:
                    vk, is_down, hide = info.vkCode, wparam in (WM_KEYDOWN, WM_SYSKEYDOWN), False
                    for listener in self.listeners:  # every one sees every key, so each knows which keys are held
                        try:
                            hide = listener.feed(vk, is_down) or hide
                        except Exception:  # one listener's error never costs the others their key
                            log.exception("Keyboard hook error (%s)", listener.hotkey.text)
                    if hide:
                        return 1
        except Exception:  # never let an error here swallow or delay the user's typing
            log.exception("Keyboard hook error")
        return user32.CallNextHookEx(None, code, wparam, lparam)


HOOK = KeyboardHook()
