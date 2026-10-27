"""The few Win32 calls the frame needs, through ctypes. Every coordinate is in physical pixels: Rflow's Qt process is
per-monitor DPI aware, so GetWindowRect and GetCursorPos aren't scaled.

Raw input (wheel, keys) is how stickers are hidden before the text under them scrolls: unlike a keyboard or mouse hook
it never sits in the input's path, so it can't add lag even when Rflow's UI thread is busy.
"""
import ctypes
import sys
from ctypes import wintypes

WDA_NONE, WDA_EXCLUDEFROMCAPTURE = 0x00, 0x11  # Windows 10 2004+: on the screen, left out of every capture
GWL_EXSTYLE = -20
WS_EX_TRANSPARENT, WS_EX_TOOLWINDOW, WS_EX_LAYERED, WS_EX_NOACTIVATE = 0x20, 0x80, 0x80000, 0x08000000
SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 76, 77, 78, 79
SRCCOPY, CAPTUREBLT = 0x00CC0020, 0x40000000
DWMWA_EXTENDED_FRAME_BOUNDS = 9
PAGE_CLASSES = {"Chrome_RenderWidgetHostHWND"}  # where Chromium draws the page: Chrome, Edge, Teams and other Electron apps

WM_INPUT = 0x00FF
RIDEV_REMOVE, RIDEV_INPUTSINK, RID_INPUT = 0x1, 0x100, 0x10000003
RIM_TYPEMOUSE, RIM_TYPEKEYBOARD = 0, 1
RI_MOUSE_LEFT_BUTTON_DOWN, RI_MOUSE_WHEEL, RI_MOUSE_HWHEEL = 0x0001, 0x0400, 0x0800
WM_KEYDOWN, WM_SYSKEYDOWN = 0x0100, 0x0104
NAV_KEYS = {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28}  # Page Up/Down, End, Home, the arrows

if sys.platform == "win32":
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    dwmapi = ctypes.WinDLL("dwmapi")
    user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
    user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
    user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
    user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    user32.GetForegroundWindow.restype = wintypes.HWND
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    _ENUMCHILD = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumChildWindows.argtypes = [wintypes.HWND, _ENUMCHILD, wintypes.LPARAM]
    user32.GetDC.argtypes = [wintypes.HWND]
    user32.GetDC.restype = wintypes.HDC
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user32.GetRawInputData.argtypes = [wintypes.HANDLE, wintypes.UINT, ctypes.c_void_p, ctypes.POINTER(wintypes.UINT),
                                       wintypes.UINT]
    user32.GetRawInputData.restype = wintypes.UINT
    gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    gdi32.CreateCompatibleDC.restype = wintypes.HDC
    gdi32.CreateDIBSection.argtypes = [wintypes.HDC, ctypes.c_void_p, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p),
                                       wintypes.HANDLE, wintypes.DWORD]
    gdi32.CreateDIBSection.restype = wintypes.HBITMAP
    gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    gdi32.SelectObject.restype = wintypes.HGDIOBJ
    gdi32.BitBlt.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HDC,
                             ctypes.c_int, ctypes.c_int, wintypes.DWORD]
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.DeleteDC.argtypes = [wintypes.HDC]
    dwmapi.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD]


class _PowerStatus(ctypes.Structure):
    _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte), ("BatteryLifePercent", ctypes.c_ubyte),
                ("SystemStatusFlag", ctypes.c_ubyte), ("BatteryLifeTime", wintypes.DWORD),
                ("BatteryFullLifeTime", wintypes.DWORD)]


class _RawInputDevice(ctypes.Structure):
    _fields_ = [("usUsagePage", ctypes.c_ushort), ("usUsage", ctypes.c_ushort), ("dwFlags", wintypes.DWORD),
                ("hwndTarget", wintypes.HWND)]


class _RawInputHeader(ctypes.Structure):
    _fields_ = [("dwType", wintypes.DWORD), ("dwSize", wintypes.DWORD), ("hDevice", wintypes.HANDLE),
                ("wParam", wintypes.WPARAM)]


class BitmapInfoHeader(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG), ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]


def exclude_from_capture(hwnd: int) -> bool:
    """The window left out of screen capture (BitBlt, DXGI, Graphics Capture): Rflow never reads its own stickers."""
    return bool(user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE))


def add_ex_style(hwnd: int, bits: int) -> None:
    user32.SetWindowLongPtrW(hwnd, GWL_EXSTYLE, user32.GetWindowLongPtrW(hwnd, GWL_EXSTYLE) | bits)


def window_rect(hwnd: int) -> tuple[int, int, int, int]:
    r = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    return r.left, r.top, r.right - r.left, r.bottom - r.top


def foreground_rect() -> tuple[int, int, int, int] | None:
    hwnd = user32.GetForegroundWindow()
    return window_rect(hwnd) if hwnd else None


def foreground_other(own_pid: int) -> int | None:
    """The active window, or None when it is Rflow's own or there is none."""
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return None if pid.value == own_pid else int(hwnd)


def _class_name(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(128)
    user32.GetClassNameW(hwnd, buf, 128)
    return buf.value


def _visible_bounds(hwnd: int) -> tuple[int, int, int, int]:
    """The window without the invisible resize border Windows 10/11 put around it."""
    r = wintypes.RECT()
    if dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(r), ctypes.sizeof(r)) == 0:
        return r.left, r.top, r.right - r.left, r.bottom - r.top
    return window_rect(hwnd)


def _client_bounds(hwnd: int) -> tuple[int, int, int, int]:
    r, p = wintypes.RECT(), wintypes.POINT(0, 0)
    user32.GetClientRect(hwnd, ctypes.byref(r))
    user32.ClientToScreen(hwnd, ctypes.byref(p))
    return p.x, p.y, r.right - r.left, r.bottom - r.top


def _page_child(hwnd: int) -> tuple[int, int, int, int] | None:
    found: list[tuple[int, int, int, int]] = []

    def visit(child, _):
        if user32.IsWindowVisible(child) and _class_name(child) in PAGE_CLASSES:
            r = window_rect(child)
            if r[2] > 50 and r[3] > 50:
                found.append(r)
        return True

    user32.EnumChildWindows(hwnd, _ENUMCHILD(visit), 0)
    return max(found, key=lambda r: r[2] * r[3]) if found else None


def fit_rect(hwnd: int | None) -> tuple[int, int, int, int] | None:
    """What the frame should cover of a window: a browser's page, else its client area; None if minimised or hidden."""
    if not hwnd or user32.IsIconic(hwnd) or not user32.IsWindowVisible(hwnd):
        return None
    rect = _page_child(hwnd) or _client_bounds(hwnd)
    return _visible_bounds(hwnd) if rect[2] < 60 or rect[3] < 40 else rect


def on_battery() -> bool:
    """On battery (AC line status 0); unknown counts as plugged in."""
    st = _PowerStatus()
    return bool(kernel32.GetSystemPowerStatus(ctypes.byref(st))) and st.ACLineStatus == 0


def cursor_pos() -> tuple[int, int]:
    p = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(p))
    return p.x, p.y


def virtual_screen() -> tuple[int, int, int, int]:
    gm = user32.GetSystemMetrics
    return gm(SM_XVIRTUALSCREEN), gm(SM_YVIRTUALSCREEN), gm(SM_CXVIRTUALSCREEN), gm(SM_CYVIRTUALSCREEN)


def register_raw_input(hwnd: int) -> bool:
    devices = (_RawInputDevice * 2)(_RawInputDevice(0x01, 0x02, RIDEV_INPUTSINK, hwnd),  # the mouse
                                    _RawInputDevice(0x01, 0x06, RIDEV_INPUTSINK, hwnd))  # the keyboard
    return bool(user32.RegisterRawInputDevices(devices, 2, ctypes.sizeof(_RawInputDevice)))


def unregister_raw_input() -> None:
    """Rflow's process no longer told of the mouse and keyboard (RIDEV_REMOVE), when the frame closes."""
    devices = (_RawInputDevice * 2)(_RawInputDevice(0x01, 0x02, RIDEV_REMOVE, None),
                                    _RawInputDevice(0x01, 0x06, RIDEV_REMOVE, None))
    user32.RegisterRawInputDevices(devices, 2, ctypes.sizeof(_RawInputDevice))


_MSG_ID_OFFSET = ctypes.sizeof(ctypes.c_void_p)  # MSG: HWND, then the UINT message
_HEADER = ctypes.sizeof(_RawInputHeader)
_raw = ctypes.create_string_buffer(256)


def message_id(message_ptr: int) -> int:
    """A native MSG's number, read without building the struct: Qt hands every message to nativeEvent."""
    return ctypes.c_uint.from_address(message_ptr + _MSG_ID_OFFSET).value


def raw_input_kind(message_ptr: int) -> str | None:
    """A WM_INPUT message as "wheel", "click", "navkey", "key", or None."""
    msg = wintypes.MSG.from_address(message_ptr)
    size = wintypes.UINT(len(_raw))
    if user32.GetRawInputData(msg.lParam, RID_INPUT, _raw, ctypes.byref(size), _HEADER) in (0, 0xFFFFFFFF):
        return None
    kind = _RawInputHeader.from_buffer(_raw).dwType
    if kind == RIM_TYPEMOUSE:
        flags = int.from_bytes(_raw.raw[_HEADER + 4:_HEADER + 6], "little")
        return "wheel" if flags & (RI_MOUSE_WHEEL | RI_MOUSE_HWHEEL) else "click" if flags & RI_MOUSE_LEFT_BUTTON_DOWN \
            else None
    if kind == RIM_TYPEKEYBOARD:
        vkey = int.from_bytes(_raw.raw[_HEADER + 6:_HEADER + 8], "little")
        message = int.from_bytes(_raw.raw[_HEADER + 8:_HEADER + 12], "little")
        if message in (WM_KEYDOWN, WM_SYSKEYDOWN):
            return "navkey" if vkey in NAV_KEYS else "key"
    return None
