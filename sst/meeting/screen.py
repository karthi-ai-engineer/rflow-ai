"""See the meeting window every few seconds and keep a frame when what it shows changed (a new slide, page or screen).

grab_window asks the window itself to draw into a bitmap (PrintWindow with PW_RENDERFULLCONTENT, Windows 8.1+, which is
in winuser.h though not on PrintWindow's page): unlike copying the screen, the apps and Rflow bars over it never appear,
and a window behind others is still seen. Without that flag, windows drawn by the GPU through DirectComposition
(Chromium, so Teams, Meet in a browser and Zoom's web parts) come back black. A minimised window has nothing to draw:
it is skipped, as is a grab that came back all black. The grab runs per-monitor DPI aware
(SetThreadDpiAwarenessContext) so the window's rectangle is in real pixels on a scaled display, not a blurred, cut-off
virtualised one; the invisible resize borders around Windows 10/11 windows are cut off (DWMWA_EXTENDED_FRAME_BOUNDS).
https://learn.microsoft.com/windows/win32/api/winuser/nf-winuser-printwindow
https://learn.microsoft.com/windows/win32/api/winuser/nf-winuser-setthreaddpiawarenesscontext

changed() compares two frames on a coarse grid of the grey picture's cell averages: a slide change moves many cells,
while a webcam tile (a few cells at most) or a speaker's moving head inside one doesn't reach CHANGED_SHARE.
"""
import ctypes
import logging
import sys
import threading
import time
from collections.abc import Callable
from ctypes import wintypes
from pathlib import Path

import numpy as np

from sst.meeting.apps import MeetingWindow

log = logging.getLogger(__name__)
INTERVAL_S = 5.0      # how often the window is seen
MAX_WIDTH = 1600      # frames are scaled down to this width: slides stay readable, files stay small
JPEG_QUALITY = 80
GRID = (27, 48)       # rows, columns of the comparison grid (16:9 cells on a 16:9 window)
CELL_DIFF = 6.0       # grey levels (of 255) a cell's average must move to count as changed: above JPEG/scaling noise
CHANGED_SHARE = 0.10  # share of the cells that must change for a new frame: more than a webcam tile covers
BLACK = 2.0           # a grab whose brightest grey is below this drew nothing (a GPU window that wouldn't print)

if sys.platform == "win32":
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)
    dwmapi = ctypes.WinDLL("dwmapi")
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
    user32.GetDC.argtypes = [wintypes.HWND]
    user32.GetDC.restype = wintypes.HDC
    user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user32.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
    user32.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
    gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
    gdi32.CreateCompatibleDC.restype = wintypes.HDC
    gdi32.CreateDIBSection.argtypes = [wintypes.HDC, ctypes.c_void_p, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p),
                                       wintypes.HANDLE, wintypes.DWORD]
    gdi32.CreateDIBSection.restype = wintypes.HBITMAP
    gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
    gdi32.SelectObject.restype = wintypes.HGDIOBJ
    gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
    gdi32.DeleteDC.argtypes = [wintypes.HDC]
    gdi32.GdiFlush.restype = wintypes.BOOL
    dwmapi.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD]
PW_RENDERFULLCONTENT, DIB_RGB_COLORS, DWMWA_EXTENDED_FRAME_BOUNDS = 0x2, 0, 9
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4


class _BitmapInfoHeader(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG), ("biYPelsPerMeter", wintypes.LONG),
                ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]


def _visible(hwnd: int, rect: wintypes.RECT) -> tuple[int, int, int, int]:
    """The window's drawn part inside its rectangle (left, top, right, bottom in the bitmap), without the invisible borders."""
    width, height = rect.right - rect.left, rect.bottom - rect.top
    frame = wintypes.RECT()
    if dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(frame), ctypes.sizeof(frame)) != 0:
        return 0, 0, width, height
    left, top = max(0, frame.left - rect.left), max(0, frame.top - rect.top)
    right, bottom = min(width, frame.right - rect.left), min(height, frame.bottom - rect.top)
    return (left, top, right, bottom) if right > left and bottom > top else (0, 0, width, height)


def grab_window(hwnd: int) -> "np.ndarray | None":
    """The window as it draws itself, HxWx3 uint8 RGB; None when it is minimised, gone or drew nothing."""
    if not user32.IsWindow(hwnd) or user32.IsIconic(hwnd):
        return None
    old = user32.SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
    screen = memory = bitmap = None
    try:
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        width, height = rect.right - rect.left, rect.bottom - rect.top
        if width <= 0 or height <= 0:
            return None
        crop = _visible(hwnd, rect)
        screen = user32.GetDC(None)
        memory = gdi32.CreateCompatibleDC(screen)
        header = _BitmapInfoHeader(biSize=ctypes.sizeof(_BitmapInfoHeader), biWidth=width, biHeight=-height,  # top-down
                                   biPlanes=1, biBitCount=32, biCompression=0)
        bits = ctypes.c_void_p()
        bitmap = gdi32.CreateDIBSection(memory, ctypes.byref(header), DIB_RGB_COLORS, ctypes.byref(bits), None, 0)
        if not bitmap or not bits.value:
            return None
        previous = gdi32.SelectObject(memory, bitmap)
        drawn = user32.PrintWindow(hwnd, memory, PW_RENDERFULLCONTENT)
        gdi32.GdiFlush()
        gdi32.SelectObject(memory, previous)
        if not drawn:
            return None
        buffer = (ctypes.c_ubyte * (width * height * 4)).from_address(bits.value)
        bgra = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4)
        left, top, right, bottom = crop
        image = bgra[top:bottom, left:right, 2::-1].copy()  # BGRA -> RGB, copied before the bitmap goes
        return None if image.max() < BLACK else image
    except OSError:
        log.exception("Couldn't see the meeting window")
        return None
    finally:
        if bitmap:
            gdi32.DeleteObject(bitmap)
        if memory:
            gdi32.DeleteDC(memory)
        if screen:
            user32.ReleaseDC(None, screen)
        if old:
            user32.SetThreadDpiAwarenessContext(old)


def _cells(image: np.ndarray) -> np.ndarray:
    """The grey picture's average over each cell of GRID (fewer cells for a tiny image)."""
    grey = image[..., 0] * 0.299 + image[..., 1] * 0.587 + image[..., 2] * 0.114 if image.ndim == 3 else image.astype(float)
    rows, cols = min(GRID[0], grey.shape[0]), min(GRID[1], grey.shape[1])
    height, width = grey.shape[0] // rows * rows, grey.shape[1] // cols * cols
    return grey[:height, :width].reshape(rows, height // rows, cols, width // cols).mean(axis=(1, 3))


def changed(a: np.ndarray, b: np.ndarray) -> bool:
    """b shows something new compared with a: a share of the grid's cells, not one tile, moved."""
    if a.shape != b.shape:
        return True  # the window was resized, or the share started or stopped
    return float((np.abs(_cells(a) - _cells(b)) > CELL_DIFF).mean()) >= CHANGED_SHARE


def frame_name(number: int, seconds: float) -> str:
    """'0001 00m35s.jpg': in order, with the time into the meeting it was seen."""
    minutes, rest = divmod(int(seconds), 60)
    return f"{number:04d} {minutes:02d}m{rest:02d}s.jpg"


def save_jpeg(image: np.ndarray, path: Path) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage

    height, width = image.shape[:2]
    data = np.ascontiguousarray(image)
    picture = QImage(data.data, width, height, 3 * width, QImage.Format.Format_RGB888)
    if width > MAX_WIDTH:
        picture = picture.scaledToWidth(MAX_WIDTH, Qt.TransformationMode.SmoothTransformation)
    if not picture.save(str(path), "JPG", JPEG_QUALITY):
        raise OSError(f"Couldn't save {path.name}")


class WindowFrames:
    """Sees one window every `interval` seconds on its own thread and keeps the frames that changed, as JPEG files in
    folder/frames. `frames`: (seconds since start, file); `on_frame(file, seconds)` for each one kept."""

    def __init__(self, window: MeetingWindow, folder: Path, on_frame: Callable[[Path, float], None] | None = None, *,
                 grab: Callable[[int], "np.ndarray | None"] = grab_window, clock=time.monotonic, interval: float = INTERVAL_S):
        self.window, self.folder, self.on_frame = window, Path(folder) / "frames", on_frame
        self.grab, self.clock, self.interval = grab, clock, interval
        self.frames: list[tuple[float, Path]] = []
        self.seen = self.kept = self.skipped = 0
        self._last: np.ndarray | None = None
        self._started: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread:
            return
        self._started = self.clock()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="meeting-frames", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=self.interval + 5)
        self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.look()
            except Exception:  # one bad grab or a full disk mustn't end the meeting's frames
                log.exception("Couldn't keep a meeting frame")
            self._stop.wait(self.interval)

    def look(self) -> Path | None:
        """One look at the window: the file kept, or None (unchanged, minimised, covered or failed)."""
        if self._started is None:
            self._started = self.clock()
        self.seen += 1
        image = self.grab(self.window.hwnd)
        if image is None:
            self.skipped += 1
            return None
        if self._last is not None and not changed(self._last, image):
            return None
        seconds = self.clock() - self._started
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / frame_name(self.kept + 1, seconds)
        save_jpeg(image, path)
        self._last = image
        self.kept += 1
        self.frames.append((seconds, path))
        if self.on_frame:
            self.on_frame(path, seconds)
        return path
