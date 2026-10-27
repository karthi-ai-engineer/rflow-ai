"""Stickers as tracked objects, and the change detector that says when to read the screen again.

Each sticker remembers the pixels it was read from (with a little around them). Every tick it must prove it is still
true:
  1. the same place, the same pixels   -> kept
  2. the pixels moved (scrolling)      -> followed: the shift another sticker just found is tried first, then a search
  3. the pixels gone (a popup, a page) -> hidden, until they come back or the next reading replaces it
Boxes are in absolute physical screen pixels, so moving the frame itself needs nothing. A sticker is drawn only while
its text stands still, so it never trails behind text that is scrolling. Numpy only: the search is TM_SQDIFF_NORMED
template matching, its cross term by FFT.
"""
import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from sst.screen.capture import Frame
from sst.screen.layout import Block

_ids = itertools.count(1)

DIFF_LEVEL = 40  # grey-level change that counts as a changed pixel
DIFF_FRACTION = 0.015  # changed pixels tolerated in a patch: a caret's blink, anti-aliasing
SEARCH_X = 48  # px of horizontal slack for the search
MATCH_MAX = 0.04  # TM_SQDIFF_NORMED below this is a match
SEARCH_BUDGET_S = 0.012  # per tick: one search usually finds the shift for every sticker
RETRY_BASE_S, RETRY_MAX_S = 0.15, 2.0  # a sticker whose search failed waits this long, doubling each miss
HEADING_RATIO = 1.3  # text this much taller than the screen's median is a heading (bold)


@dataclass
class Sticker:
    id: int
    src: str
    x: int
    y: int
    w: int
    h: int
    char_h: float
    n_lines: int
    patch: np.ndarray  # grey, the padded region it was read from
    px: int  # the patch's absolute position
    py: int
    bg: tuple[int, int, int]  # RGB
    fg: tuple[int, int, int]
    visible: bool = True
    stable: int = 0  # ticks in a row matched in the same place
    bold: bool = False
    centered: bool = False
    room_r: int | None = None  # px from the text to its container's right edge
    fails: int = 0  # searches failed in a row (the retry's backoff)
    retry_at: float = 0.0

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h

    def shift(self, dx: int, dy: int) -> None:
        self.x, self.y, self.px, self.py = self.x + dx, self.y + dy, self.px + dx, self.py + dy

    def contains(self, x: int, y: int) -> bool:
        return self.x <= x < self.x + self.w and self.y <= y < self.y + self.h


def absdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, b) - np.minimum(a, b)


def match_template(image: np.ndarray, templ: np.ndarray) -> tuple[float, tuple[int, int]]:
    """OpenCV's TM_SQDIFF_NORMED minimum over every place templ fits in image: (its value, (x, y)). The cross term by
    FFT, the windows' energy by an integral image."""
    img, t = image.astype(np.float64), templ.astype(np.float64)
    H, W = img.shape
    h, w = t.shape
    shape = (H, W)
    cross = np.fft.irfft2(np.fft.rfft2(img, shape) * np.conj(np.fft.rfft2(t, shape)), shape)[:H - h + 1, :W - w + 1]
    sq = np.zeros((H + 1, W + 1))
    sq[1:, 1:] = (img * img).cumsum(0).cumsum(1)
    energy = sq[h:, w:] - sq[:-h, w:] - sq[h:, :-w] + sq[:-h, :-w]
    t2 = float((t * t).sum())
    diff = np.maximum(energy - 2 * cross + t2, 0.0)
    norm = np.sqrt(np.maximum(energy * t2, 1e-9))
    result = diff / norm
    y, x = np.unravel_index(int(np.argmin(result)), result.shape)
    return float(result[y, x]), (int(x), int(y))


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter or 1)


def _luma(c) -> float:
    r, g, b = c
    return 0.299 * r + 0.587 * g + 0.114 * b


def _rel_lum(c) -> float:
    def ch(v):
        v /= 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = c
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast(a, b) -> float:
    """WCAG's contrast ratio: anti-aliased thin glyphs sample as a washed-out grey."""
    la, lb = _rel_lum(a), _rel_lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def sample_colors(frame: Frame, box) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """(background, text) as RGB: the median of a ring just outside the box, and of the box's pixels that clearly differ
    from it; black or white text when that wouldn't read."""
    x, y, w, h = box
    pad = 3
    lx0, ly0 = max(0, x - frame.x - pad), max(0, y - frame.y - pad)
    lx1, ly1 = min(frame.w, x - frame.x + w + pad), min(frame.h, y - frame.y + h + pad)
    region = frame.img[ly0:ly1, lx0:lx1, :3].reshape(-1, 3).astype(np.int16)
    if region.size == 0:
        return (255, 255, 255), (0, 0, 0)
    mask = np.ones((ly1 - ly0, lx1 - lx0), bool)
    ix0, iy0 = x - frame.x - lx0, y - frame.y - ly0
    mask[max(0, iy0):max(0, iy0 + h), max(0, ix0):max(0, ix0 + w)] = False
    ring = region[mask.reshape(-1)]
    if len(ring) < 8:
        ring = region
    bg_bgr = np.median(ring, axis=0)
    inner = region[~mask.reshape(-1)]
    far = inner[np.abs(inner - bg_bgr).sum(axis=1) > 120] if len(inner) else inner
    if len(far) >= 12:  # the strongest glyph pixels, not the anti-aliased fringe
        dist = np.abs(far - bg_bgr).sum(axis=1)
        far = far[dist >= np.percentile(dist, 60)]
    bg = tuple(int(v) for v in bg_bgr[::-1])
    if len(far) >= max(6, 0.01 * len(inner)):
        fg = tuple(int(v) for v in np.median(far, axis=0)[::-1])
    else:
        fg = (0, 0, 0) if _luma(bg) > 128 else (255, 255, 255)
    if abs(_luma(fg) - _luma(bg)) < 90 or contrast(fg, bg) < 4.5:
        fg = (0, 0, 0) if _luma(bg) > 128 else (255, 255, 255)
    return bg, fg


def detect_layout(frame: Frame, box, bg) -> tuple[bool, int | None]:
    """(centered, room_right) for the text at box: centred in a container (a button, a header cell) whose two edges are
    found; px from the text to the container's right edge, None without one (page text). An edge is a run of background
    ending in a line as tall as the text; ending at other text doesn't count."""
    x, y, w, h = box
    lx, ly = x - frame.x, y - frame.y
    gray = frame.gray
    y0, y1 = max(0, ly), min(frame.h, ly + h)
    if min(frame.w, lx + w) - max(0, lx) < 4 or y1 - y0 < 4:
        return False, None
    bg_l = int(_luma(bg))
    row = gray[min(frame.h - 1, max(0, ly + h // 2)), :].astype(np.int16)
    reach = int(6 * h)

    def run(start: int, step: int) -> int | None:
        n, i = 0, start
        while 0 <= i < frame.w and n <= reach:
            if abs(int(row[i]) - bg_l) > 14:
                tall = float(np.mean(np.abs(gray[y0:y1, i].astype(np.int16) - bg_l) > 14))
                return n if tall >= 0.85 else None
            i += step
            n += 1
        return None

    left, right = run(lx - 1, -1), run(lx + w, 1)
    centered = left is not None and right is not None and left + right > 8 and abs(left - right) <= 0.2 * (left + right) + 2
    return centered, right


def make_sticker(block: Block, frame: Frame) -> Sticker | None:
    x, y, w, h = block.box
    pad_y, pad_x = int(0.35 * block.char_h) + 2, int(1.0 * block.char_h) + 2
    px0, py0 = max(frame.x, x - pad_x), max(frame.y, y - pad_y)
    px1, py1 = min(frame.x + frame.w, x + w + pad_x), min(frame.y + frame.h, y + h + pad_y)
    patch = frame.crop_gray(px0, py0, px1 - px0, py1 - py0)
    if patch is None or patch.size == 0:
        return None
    bg, fg = sample_colors(frame, (x, y, w, h))
    centered, room_r = detect_layout(frame, (x, y, w, h), bg) if len(block.lines) == 1 else (False, None)
    return Sticker(next(_ids), block.text, x, y, w, h, block.char_h, len(block.lines), patch.copy(), px0, py0, bg, fg,
                   centered=centered, room_r=room_r)


class Tracker:
    def __init__(self):
        self.stickers: list[Sticker] = []
        self.hint = (0, 0)  # the last shift that worked

    def replace_from_ocr(self, blocks: list[Block], frame: Frame) -> None:
        """A fresh reading, keeping the objects of unchanged text so nothing flickers."""
        old, new = self.stickers, []
        for b in blocks:
            s = make_sticker(b, frame)
            if s is None:
                continue
            for o in old:
                if o.src == s.src and _iou(o.box, s.box) > 0.5:
                    s.id, s.stable = o.id, o.stable
                    break
            new.append(s)
        if len(new) >= 2:  # small bold text can't be told from regular, but headings are larger than the rest
            hs = sorted(s.char_h for s in new)
            typical = hs[(len(hs) - 1) // 2]
            for s in new:
                s.bold = s.n_lines == 1 and s.char_h >= HEADING_RATIO * typical
        self.stickers = new

    def clear(self) -> None:
        self.stickers = []

    def hide_all(self) -> bool:
        """A scroll is coming: the stickers go before the text moves, and come back once their pixels stand still."""
        changed = any(s.visible for s in self.stickers)
        for s in self.stickers:
            s.visible, s.stable = False, 0
        return changed

    def _matches(self, s: Sticker, frame: Frame, dx: int, dy: int) -> bool | None:
        """Whether the patch is at this offset; None when it is mostly off the frame."""
        ph, pw = s.patch.shape
        ax, ay = s.px + dx, s.py + dy
        ix0, iy0 = max(ax, frame.x), max(ay, frame.y)
        ix1, iy1 = min(ax + pw, frame.x + frame.w), min(ay + ph, frame.y + frame.h)
        if ix1 <= ix0 or iy1 <= iy0 or (ix1 - ix0) * (iy1 - iy0) < 0.6 * pw * ph:
            return None
        cur = frame.gray[iy0 - frame.y:iy1 - frame.y, ix0 - frame.x:ix1 - frame.x]
        ref = s.patch[iy0 - ay:iy1 - ay, ix0 - ax:ix1 - ax]
        return np.count_nonzero(absdiff(cur, ref) > DIFF_LEVEL) <= max(3, DIFF_FRACTION * ref.size)

    def _search(self, s: Sticker, frame: Frame) -> tuple[int, int] | None:
        ph, pw = s.patch.shape
        lx0 = max(0, s.px - frame.x - SEARCH_X)
        lx1 = min(frame.w, s.px - frame.x + pw + SEARCH_X)
        band = frame.gray[:, lx0:lx1]
        if band.shape[0] < ph or band.shape[1] < pw or float(s.patch.std()) < 6:  # featureless: can't be tracked
            return None
        best, (bx, by) = match_template(band, s.patch)
        if best > MATCH_MAX:
            return None
        dx, dy = frame.x + lx0 + bx - s.px, frame.y + by - s.py
        return (dx, dy) if self._matches(s, frame, dx, dy) else None

    def track(self, frame: Frame, search: bool = True) -> bool:
        """Check or move every sticker; True if anything changed. search=False skips the search (a busy screen).

        A scroll moves every sticker by the same offset, so one successful search moves them all; the budget goes first
        to the stickers likeliest to be found: far from the frame's edges (text near one may have scrolled out) and not
        already failing."""
        now = time.perf_counter()
        before = {id(s): (s.visible, s.x, s.y) for s in self.stickers}
        lost: list[Sticker] = []
        found = None
        for s in self.stickers:
            ok = self._matches(s, frame, 0, 0)
            if ok:
                s.stable += 1
                s.fails = 0
                continue
            s.stable = 0
            if ok is None:
                continue
            shift = next((c for c in (found, self.hint) if c and c != (0, 0) and self._matches(s, frame, *c)), None)
            if shift:
                self._move(s, shift)
                found = shift
            else:
                lost.append(s)
        if lost and search:
            deadline = now + SEARCH_BUDGET_S

            def edge_gap(st: Sticker) -> int:
                return min(st.y - frame.y, frame.y + frame.h - (st.y + st.h))

            for s in sorted(lost, key=lambda st: (st.fails, -edge_gap(st))):
                if time.perf_counter() >= deadline:
                    break
                if now < s.retry_at:
                    continue
                shift = self._search(s, frame)
                if shift:
                    self._move(s, shift)
                    found = shift
                    break
                s.fails += 1  # text that is gone isn't searched for every tick
                s.retry_at = now + min(RETRY_MAX_S, RETRY_BASE_S * 2 ** s.fails)
            if found:  # the rest of the page moved the same way
                for s in lost:
                    if (s.x, s.y) == before[id(s)][1:] and self._matches(s, frame, *found):
                        self._move(s, found)
        if found:
            self.hint = found
        changed = False
        for s in self.stickers:
            s.visible = s.stable >= 1  # a sticker that just moved waits a tick: it never trails scrolling text
            changed |= before[id(s)] != (s.visible, s.x, s.y)
        return changed

    @staticmethod
    def _move(s: Sticker, shift: tuple[int, int]) -> None:
        s.shift(*shift)
        s.fails = s.stable = 0


def _small(gray: np.ndarray, f: int = 4) -> np.ndarray:
    """The picture f times smaller, by block averages (OpenCV's INTER_AREA at 1/f)."""
    h, w = (gray.shape[0] // f) * f, (gray.shape[1] // f) * f
    if h == 0 or w == 0:
        return gray.astype(np.uint8)
    return gray[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3)).astype(np.uint8)


class ChangeDetector:
    """Says when the screen has settled and differs from what was last read."""

    LEVEL = 18

    def __init__(self, settle_s: float = 0.30, min_gap_s: float = 0.6, force_s: float = 2.0,
                 clock: Callable[[], float] = time.monotonic):
        self.settle_s, self.min_gap_s, self.force_s, self.clock = settle_s, min_gap_s, force_s, clock
        self.prev = self.ref = None  # (rect, small picture): the last tick's, and the last reading's
        self.stable_since = clock()
        self.last_ocr = 0.0
        self.dirty = True  # differs from the last reading
        self.moving = False  # changed since the last tick

    def _differs(self, a, b) -> tuple[bool, float]:
        (ra, sa), (rb, sb) = a, b
        if ra != rb or sa.shape != sb.shape:
            return True, 1.0
        n = np.count_nonzero(absdiff(sa, sb) > self.LEVEL)
        return n >= 12, n / max(1, sa.size)

    def update(self, frame: Frame) -> None:
        now = self.clock()
        cur = (frame.rect, _small(frame.gray))
        if self.prev is not None:
            _, frac = self._differs(self.prev, cur)
            self.moving = frac > 0.0015
            if frac > 0.004:
                self.stable_since = now
        self.prev = cur
        self.dirty = self.ref is None or self._differs(self.ref, cur)[0]

    def should_ocr(self) -> bool:
        if not self.dirty:
            return False
        now = self.clock()
        since = now - self.last_ocr
        return (now - self.stable_since >= self.settle_s and since >= self.min_gap_s) or since >= self.force_s

    def mark_ocr_started(self, frame: Frame) -> None:
        self.last_ocr = self.clock()
        self.ref = (frame.rect, _small(frame.gray))
        self.dirty = False

    def force(self) -> None:
        self.ref, self.dirty, self.last_ocr = None, True, 0.0
