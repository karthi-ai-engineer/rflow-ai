"""What the OCR read, as lines in absolute screen pixels, with Windows OCR's quirks for Japanese dealt with:
  * a Japanese line comes back with a space between every character: the text is rebuilt from the words, with a space
    only between two Latin words or numbers, then NFKC
  * small UI fonts misread (ー as -, a dropped ャ): the frame is read upscaled, 2x (1.5x when that would be too big)
  * light text on a dark background reads better inverted
  * a table column of short cells can come back as one vertical line (担/佐/鈴/高): rebuilt as rows
  * table cells and a form's label and value come glued in one line: split at wide gaps
  * a short label cut by the frame's edge misreads (件名 -> 名 -> "Given name"): dropped
  * the thousands comma read as a period (128,000円 -> 128.000円), which changes the amount: put back before 円 and such
The engine (sst.screen.winocr.WinOcr, a fake in tests) reads an 8-bit grey picture into lines of words.
"""
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from sst.screen.capture import Frame

MAX_PIXELS = 16_000_000  # an upscaled frame larger than this is read at a smaller scale
DARK = 110  # a median grey below this is a dark theme: inverted before reading


class Engine(Protocol):
    max_dim: int

    def recognize(self, gray: bytes, width: int, height: int) -> list: ...  # lines of words (.text .x .y .width .height)


@dataclass
class OcrLine:
    text: str
    x: float  # absolute physical pixels
    y: float
    w: float
    h: float
    char_h: float  # the median glyph height

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h


def _is_latin(ch: str) -> bool:
    return ch.isascii() and ch.isalnum()


_KEEP_LEADING = set("「『（(【[<〈《\"'#$¥￥+-")
_GROUPED = re.compile(r"(?<![\d.])\d{1,3}(?:\.\d{3})+(?![\d.])")
_UNIT_AFTER = re.compile(r"\s*[円件名人個台回枚点]")


def _strip_leading_junk(text: str) -> str:
    i = 0
    while i < len(text) and unicodedata.category(text[i])[0] in "PSZ" and text[i] not in _KEEP_LEADING:
        i += 1
    return text[i:]


def _fix_thousands(text: str) -> str:
    """Only groups of exactly three digits that are clearly money or counts (before 円, 件..., or after ¥)."""
    def repl(m: re.Match) -> str:
        if _UNIT_AFTER.match(text, m.end()) or text[:m.start()].rstrip().endswith(("¥", "￥")):
            return m.group(0).replace(".", ",")
        return m.group(0)
    return _GROUPED.sub(repl, text)


def join_words(words: list[str]) -> str:
    out = ""
    for w in words:
        if not w:
            continue
        if out and _is_latin(out[-1]) and _is_latin(w[0]):
            out += " "
        out += w
    return _strip_leading_junk(_fix_thousands(unicodedata.normalize("NFKC", out)))


def upscale(gray: np.ndarray, scale: float) -> np.ndarray:
    """The picture `scale` times larger, smoothly (Qt's bilinear filter): small kana read right at 2x."""
    if scale == 1.0:
        return np.ascontiguousarray(gray)
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QImage
    h, w = gray.shape
    image = QImage(w, h, QImage.Format.Format_Grayscale8)  # Qt's own memory: an image over a Python buffer can outlive it
    np.frombuffer(image.bits(), np.uint8).reshape(h, image.bytesPerLine())[:, :w] = gray
    big = image.scaled(round(w * scale), round(h * scale), Qt.AspectRatioMode.IgnoreAspectRatio,
                       Qt.TransformationMode.SmoothTransformation)
    rows = np.frombuffer(big.constBits(), np.uint8).reshape(big.height(), big.bytesPerLine())
    return rows[:, :big.width()].copy()


class ScreenOcr:
    """A frame's lines. Use from one thread (the engine's)."""

    def __init__(self, engine: Engine, max_scale: float = 2.0):
        self.engine, self.max_scale = engine, max_scale
        self.last_ms = 0.0

    def scale_for(self, w: int, h: int) -> float:
        for s in (2.0, 1.5, 1.0):
            if s <= self.max_scale and max(w, h) * s <= self.engine.max_dim and w * h * s * s <= MAX_PIXELS:
                return s
        return min(1.0, self.engine.max_dim / max(w, h))

    def recognize(self, frame: Frame) -> list[OcrLine]:
        t0 = time.perf_counter()
        gray = frame.gray
        if float(np.median(gray)) < DARK:
            gray = 255 - gray
        s = self.scale_for(frame.w, frame.h)
        big = upscale(gray, s)
        raw = self.engine.recognize(big.tobytes(), big.shape[1], big.shape[0])
        lines = [_make_line(seg, s, frame.x, frame.y) for words in _regroup_vertical(raw) for seg in _split_on_gaps(words)]
        self.last_ms = (time.perf_counter() - t0) * 1000
        return [ln for ln in lines if ln.text.strip() and not _cut_by_edge(ln, frame)]


def _cut_by_edge(ln: OcrLine, frame: Frame) -> bool:
    """Ink between the frame's edge and a short line: a glyph the edge sliced, so the line is likely misread."""
    if len(ln.text) > 6:
        return False
    y0, y1 = max(0, int(ln.y - frame.y)), min(frame.h, int(ln.bottom - frame.y) + 1)
    if y1 - y0 < 2:
        return False
    lx, rx = int(ln.x - frame.x), int(ln.right - frame.x)
    margin = 1.6 * ln.char_h
    strips = []
    if lx < margin:
        strips.append(frame.gray[y0:y1, 0:max(0, lx - 1)])
    if frame.w - rx < margin:
        strips.append(frame.gray[y0:y1, min(frame.w, rx + 2):frame.w])
    return any(st.size and int(st.max()) - int(st.min()) > 90 for st in strips)


def _bbox(words) -> tuple[float, float, float, float]:
    return (min(w.x for w in words), min(w.y for w in words), max(w.x + w.width for w in words),
            max(w.y + w.height for w in words))


def _median_height(words) -> float:
    hs = sorted(w.height for w in words)
    return hs[len(hs) // 2]


def _regroup_vertical(raw_lines: list) -> list:
    """A table column of short cells read as one vertical line: its words, and the short lines beside it, pooled and
    rebuilt as horizontal rows."""
    vertical, keep = [], []
    for words in raw_lines:
        if not words:
            continue
        x0, y0, x1, y1 = _bbox(words)
        ch = _median_height(words)
        (vertical if len(words) >= 2 and (y1 - y0) > 2.2 * ch and (x1 - x0) < 2.0 * ch else keep).append(words)
    if not vertical:
        return [w for w in raw_lines if w]
    boxes = [_bbox(w) for w in vertical]
    pool = [w for words in vertical for w in words]
    rest = []
    for words in keep:
        x0, y0, x1, y1 = _bbox(words)
        ch = max(1.0, y1 - y0)
        near = any(x0 < vx1 + 1.5 * ch and x1 > vx0 - 1.5 * ch and y0 < vy1 and y1 > vy0 for vx0, vy0, vx1, vy1 in boxes)
        if near and len(words) <= 4:
            pool.extend(words)
        else:
            rest.append(words)
    pool.sort(key=lambda w: w.y + w.height / 2)
    rows: list[list] = []
    for w in pool:
        cy = w.y + w.height / 2
        for row in rows:
            r0 = row[0]
            if abs((r0.y + r0.height / 2) - cy) < 0.5 * max(w.height, r0.height):
                row.append(w)
                break
        else:
            rows.append([w])
    for row in rows:
        row.sort(key=lambda w: w.x)
    return rest + rows


def _split_on_gaps(words: list) -> list[list]:
    """Split a line where the gap between two words is wider than ~1.2 glyph heights (table cells, label and value)."""
    if not words:
        return []
    ch = _median_height(words)
    segs, cur = [], [words[0]]
    for prev, nxt in zip(words, words[1:], strict=False):
        if nxt.x - (prev.x + prev.width) > max(1.2 * ch, 10):
            segs.append(cur)
            cur = []
        cur.append(nxt)
    segs.append(cur)
    return segs


def _make_line(seg: list, s: float, ox: int, oy: int) -> OcrLine:
    x0, y0, x1, y1 = _bbox(seg)
    return OcrLine(text=join_words([w.text for w in seg]), x=ox + x0 / s, y=oy + y0 / s, w=(x1 - x0) / s,
                   h=(y1 - y0) / s, char_h=_median_height(seg) / s)
