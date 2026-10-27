"""The frame's inside, copied from the screen: GDI's BitBlt from the screen, which honours WDA_EXCLUDEFROMCAPTURE, so
the stickers' window is never in it. A Frame is BGRA in absolute physical screen pixels, with its grey version (the
tracker's and OCR's) made once, on first use; numpy only."""
import ctypes
import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)
MIN_SIDE = 8  # px: anything narrower isn't worth a grab


def to_gray(bgra: np.ndarray) -> np.ndarray:
    """Rec. 601 luma, as OpenCV's BGRA2GRAY, in integers (the frame is small and this runs every tick)."""
    b, g, r = (bgra[..., i].astype(np.uint32) for i in range(3))
    return ((r * 299 + g * 587 + b * 114 + 500) // 1000).astype(np.uint8)


@dataclass
class Frame:
    """A captured image and where it sits on the virtual screen."""

    img: np.ndarray  # (h, w, 4) BGRA uint8
    x: int
    y: int
    t: float = 0.0
    _gray: np.ndarray | None = field(default=None, repr=False)

    @classmethod
    def of_gray(cls, gray: np.ndarray, x: int = 0, y: int = 0, t: float = 0.0) -> "Frame":
        """A frame from a grey picture (tests, and anything already grey)."""
        img = np.repeat(gray[..., None], 4, axis=2)
        img[..., 3] = 255
        return cls(img, x, y, t, np.ascontiguousarray(gray))

    @property
    def w(self) -> int:
        return self.img.shape[1]

    @property
    def h(self) -> int:
        return self.img.shape[0]

    @property
    def rect(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    @property
    def gray(self) -> np.ndarray:
        if self._gray is None:
            self._gray = to_gray(self.img)
        return self._gray

    def crop_gray(self, x: int, y: int, w: int, h: int) -> np.ndarray | None:
        """A grey crop by absolute coordinates; None unless it lies wholly inside the frame."""
        lx, ly = x - self.x, y - self.y
        if lx < 0 or ly < 0 or w <= 0 or h <= 0 or lx + w > self.w or ly + h > self.h:
            return None
        return self.gray[ly:ly + h, lx:lx + w]


class Capturer:
    """Grabs screen rectangles. One per thread: its device contexts belong to the thread that made them."""

    def __init__(self):
        from sst.screen import win32
        self._w = win32
        self._screen = win32.user32.GetDC(None)
        self._memory = win32.gdi32.CreateCompatibleDC(self._screen)
        self._bitmap = self._bits = None
        self._size = (0, 0)

    def _ensure(self, width: int, height: int) -> None:
        if self._size == (width, height):
            return
        if self._bitmap:
            self._w.gdi32.DeleteObject(self._bitmap)
        header = self._w.BitmapInfoHeader(biSize=ctypes.sizeof(self._w.BitmapInfoHeader), biWidth=width,
                                          biHeight=-height, biPlanes=1, biBitCount=32, biCompression=0)  # top-down
        bits = ctypes.c_void_p()
        self._bitmap = self._w.gdi32.CreateDIBSection(self._memory, ctypes.byref(header), 0, ctypes.byref(bits), None, 0)
        if not self._bitmap or not bits.value:
            self._size, self._bitmap = (0, 0), None
            raise OSError("Couldn't make a bitmap to copy the screen into")
        self._w.gdi32.SelectObject(self._memory, self._bitmap)
        self._bits, self._size = bits.value, (width, height)

    def grab(self, rect: tuple[int, int, int, int], t: float = 0.0) -> Frame | None:
        """The screen inside rect (x, y, w, h), clipped to the virtual screen; None when too little is left."""
        x, y, w, h = rect
        vx, vy, vw, vh = self._w.virtual_screen()
        x0, y0, x1, y1 = max(x, vx), max(y, vy), min(x + w, vx + vw), min(y + h, vy + vh)
        width, height = x1 - x0, y1 - y0
        if width < MIN_SIDE or height < MIN_SIDE:
            return None
        self._ensure(width, height)
        if not self._w.gdi32.BitBlt(self._memory, 0, 0, width, height, self._screen, x0, y0,
                                    self._w.SRCCOPY | self._w.CAPTUREBLT):
            raise OSError("The screen couldn't be copied (a secure desktop, or the screen locked)")
        buffer = (ctypes.c_ubyte * (width * height * 4)).from_address(self._bits)
        img = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 4).copy()  # before the next grab reuses it
        return Frame(img, x0, y0, t)

    def close(self) -> None:
        if self._bitmap:
            self._w.gdi32.DeleteObject(self._bitmap)
            self._bitmap = None
        if self._memory:
            self._w.gdi32.DeleteDC(self._memory)
            self._memory = None
        if self._screen:
            self._w.user32.ReleaseDC(None, self._screen)
            self._screen = None
