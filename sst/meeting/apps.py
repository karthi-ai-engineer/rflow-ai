"""Find the meeting app's window: Microsoft Teams, Zoom, Google Meet in a browser, Webex.

The visible top-level windows are listed (EnumWindows, cloaked ones left out: Windows keeps hidden app windows "visible"
but cloaked) with their process's image name and title, and each app's rules (APPS) say which window is its meeting:

    Microsoft Teams  ms-teams.exe (new) or Teams.exe (classic). A meeting or call window, not the main window: its title's
                     first part names one of Teams' own views ("Chat | ... | Microsoft Teams", "Calendar | ..."), which
                     a meeting's never does. "Meeting with ..."/"Call with ..." titles rank first.
    Zoom             Zoom.exe, "Zoom Meeting ..."/"Zoom Webinar" first. Zoom renames the meeting window "Zoom" when it
                     loses focus, and Zoom Workplace's own window is "Zoom Workplace": both are kept, ranked below.
    Google Meet      a browser (chrome, msedge, firefox, brave, opera, vivaldi) whose active tab is "Meet - abc-defg-hij"
                     (the title starts with "Meet - "), else one whose title has "Google Meet" (its home page, ranked
                     below). A browser shows only the active tab's title: a Meet tab behind another isn't seen.
    Webex            webex.exe, CiscoCollabHost.exe or atmgr.exe (Webex Meetings): a meeting window first.

Teams, Zoom and Webex hold one meeting at a time but open several windows for it (the call, a compact view, a shared
screen), so each gives one MeetingWindow, its best; each Meet window is its own meeting. The most likely come first,
then the one used last (EnumWindows lists the top of the z-order first).
"""
import ctypes
import re
import sys
from collections.abc import Callable
from ctypes import wintypes
from dataclasses import dataclass

TEAMS, ZOOM, MEET, WEBEX = "Microsoft Teams", "Zoom", "Google Meet", "Webex"
# The words of Teams' own views, which a meeting window's title never starts with (new and classic Teams).
TEAMS_VIEWS = frozenset({"activity", "chat", "teams", "calendar", "calls", "onedrive", "files", "apps", "meet", "copilot",
                         "communities", "help", "settings", "search", "assignments", "approvals", "shifts", "planner",
                         "people", "viva", "notifications", "microsoft teams"})
STRONG, WEAK = 2, 1  # how sure a title is the meeting's


@dataclass(frozen=True)
class MeetingWindow:
    app: str    # one of APPS' names
    title: str
    hwnd: int
    pid: int


IN_PERSON = MeetingWindow("In person", "", 0, 0)  # a meeting in a room: no app, no window; the microphone hears everyone


def in_person(window) -> bool:
    return window == IN_PERSON


def _teams(title: str) -> int:
    first = title.split("|", 1)[0].strip().lower()
    if not first or first in TEAMS_VIEWS or "notification" in first:
        return 0
    return STRONG if re.search(r"\b(meeting|call|meet now|huddle)\b", first) else WEAK


def _zoom(title: str) -> int:
    low = title.strip().lower()
    if low.startswith(("zoom meeting", "zoom webinar")) or "meeting id" in low:
        return STRONG
    return WEAK if low in ("zoom", "zoom workplace") else 0


def _meet(title: str) -> int:
    if title.startswith("Meet - "):
        return STRONG
    return WEAK if "Google Meet" in title else 0


def _webex(title: str) -> int:
    low = title.strip().lower()
    if not low:
        return 0
    return STRONG if "meeting" in low or low not in ("webex", "cisco webex meetings", "webex meetings") else WEAK


# name, process image names (lower-case), how sure a title is its meeting, one meeting at a time (one window given)
APPS: tuple[tuple[str, frozenset[str], Callable[[str], int], bool], ...] = (
    (TEAMS, frozenset({"ms-teams.exe", "teams.exe"}), _teams, True),
    (ZOOM, frozenset({"zoom.exe"}), _zoom, True),
    (MEET, frozenset({"chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe", "vivaldi.exe"}), _meet, False),
    (WEBEX, frozenset({"webex.exe", "ciscocollabhost.exe", "atmgr.exe"}), _webex, True),
)

if sys.platform == "win32":
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    dwmapi = ctypes.WinDLL("dwmapi")
    _ENUM = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [_ENUM, wintypes.LPARAM]
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                                    ctypes.POINTER(wintypes.DWORD)]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    dwmapi.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD]
PROCESS_QUERY_LIMITED_INFORMATION, DWMWA_CLOAKED = 0x1000, 14


def _pid(hwnd: int) -> int:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def _exe(pid: int) -> str:
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ""  # another user's or a protected process: not a meeting app the user runs
    try:
        buffer, size = ctypes.create_unicode_buffer(1024), wintypes.DWORD(1024)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return ""
        return buffer.value.rsplit("\\", 1)[-1].lower()
    finally:
        kernel32.CloseHandle(handle)


def _cloaked(hwnd: int) -> bool:
    value = wintypes.DWORD()
    return dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, ctypes.byref(value), ctypes.sizeof(value)) == 0 and bool(value.value)


def _top_windows() -> list[tuple[int, int, str, str]]:
    """(hwnd, pid, exe name lower-case, title) of the visible, titled top-level windows, top of the z-order first."""
    found, exes = [], {}

    def each(hwnd, _):
        length = user32.GetWindowTextLengthW(hwnd)
        if length and user32.IsWindowVisible(hwnd) and not _cloaked(hwnd):
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            pid = _pid(hwnd)
            if pid not in exes:
                exes[pid] = _exe(pid)
            found.append((int(hwnd), pid, exes[pid], buffer.value))
        return True

    user32.EnumWindows(_ENUM(each), 0)
    return found


def find(windows: Callable[[], list[tuple[int, int, str, str]]] = _top_windows) -> list[MeetingWindow]:
    """The meetings running now, the most likely first: one per meeting."""
    scored = []  # (sureness, order, app index, window)
    for order, (hwnd, pid, exe, title) in enumerate(windows()):
        for index, (name, exes, sure, _) in enumerate(APPS):
            if exe in exes and (score := sure(title)):
                scored.append((score, order, index, MeetingWindow(name, title, hwnd, pid)))
    scored.sort(key=lambda s: (-s[0], s[1]))
    result, taken = [], set()
    for _, _, index, window in scored:
        if APPS[index][3]:
            if index in taken:
                continue  # another window of the same meeting
            taken.add(index)
        result.append(window)
    return result


def choose(found: list[MeetingWindow]) -> MeetingWindow | None:
    """One meeting found: that one; none or several: None (the user picks)."""
    return found[0] if len(found) == 1 else None


def alive(window: MeetingWindow) -> bool:
    """The meeting window is still there (a closed window's handle can be reused by another process's). A meeting in
    person has none: it lasts until the user stops it."""
    if in_person(window):
        return True
    return bool(user32.IsWindow(window.hwnd)) and _pid(window.hwnd) == window.pid
