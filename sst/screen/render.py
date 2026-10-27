"""The stickers, painted: each on its text's own background colour, in its text colour, with the largest font that
fits its box (a one-line sticker grows to the right into free space first: English is usually wider). A sticker still
waiting for its translation is a dashed outline in Rflow's Iris; the one under the pointer fades, to see the original.
Segoe UI, Windows' own interface font, so the English looks like the app it covers."""
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPen

from sst import theme
from sst.screen.tracker import Sticker

FONT_FAMILY = "Segoe UI"
WRAP = Qt.TextFlag.TextWordWrap
_fit_cache: dict[tuple, tuple[float, float]] = {}


def fit_font(text: str, w: float, h: float, start_px: float, bold: bool = False) -> tuple[QFont, float]:
    """The largest pixel size (at most start_px, at least 70% of it) whose wrapped text fits w x h, and the height that
    text needs (more than h at the smallest size)."""
    key = (text, round(w), round(h), round(start_px, 1), bold)
    font = QFont(FONT_FAMILY)
    font.setBold(bold)
    if hit := _fit_cache.get(key):
        font.setPixelSize(max(1, round(hit[0])))
        return font, hit[1]
    min_px = max(10.0, start_px * 0.7)
    px = max(min_px, start_px)
    flags = WRAP | Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
    while True:
        font.setPixelSize(max(1, round(px)))
        br = QFontMetricsF(font).boundingRect(QRectF(0, 0, w, 100000), int(flags), text)
        need = br.height()
        if (need <= h * 1.08 and br.width() <= w + 1) or px <= min_px:
            break
        px = max(min_px, px - max(0.5, px * 0.06))
    if len(_fit_cache) > 4000:
        _fit_cache.clear()
    _fit_cache[key] = (px, max(h, need))
    return font, max(h, need)


def _row_limit(s: Sticker, others: list[Sticker], right_edge: float) -> float:
    """The absolute x where the next sticker on the same row begins."""
    limit = right_edge
    for o in others:
        if o is not s and o.x > s.x and min(o.y + o.h, s.y + s.h) - max(o.y, s.y) > 0.4 * min(o.h, s.h):
            limit = min(limit, o.x)
    return limit


def paint_stickers(p: QPainter, stickers: list[Sticker], lookup, origin: tuple[int, int], scale: float,
                   hover_id: int | None = None, clip_w: float | None = None) -> int:
    """Paint the visible stickers; how many were drawn. origin: the absolute physical position of the painter's (0, 0);
    scale: physical px per logical px; clip_w: the logical width there is; lookup(src): the translation, "" for none,
    None while it's on its way."""
    ox, oy = origin
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    p.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    shown = sorted((s for s in stickers if s.visible), key=lambda s: (s.y, s.x))
    right_edge = ox + (clip_w if clip_w is not None else 1e9) * scale
    drawn, pad = 0, 2.0
    for s in shown:
        x, y = (s.x - ox) / scale - pad, (s.y - oy) / scale - pad
        w, h = s.w / scale + 2 * pad, s.h / scale + 2 * pad
        text = lookup(s.src)
        if text is None:
            pending = QColor(theme.tok("iris"))
            pending.setAlpha(200)
            p.setPen(QPen(pending, 1.2, Qt.PenStyle.DashLine))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(QRectF(x, y, w, h), 3, 3)
            continue
        if not text:
            continue
        # A CJK glyph fills its em box, Latin letters about half of it: English needs a larger size to look as big.
        start_px = min(32.0, max(10.0, s.char_h / scale * 1.08))
        align = Qt.AlignmentFlag.AlignHCenter if s.centered else Qt.AlignmentFlag.AlignLeft
        flags = WRAP | align | Qt.AlignmentFlag.AlignVCenter
        need_h = h
        font = QFont(FONT_FAMILY)
        font.setBold(s.bold)
        if s.n_lines == 1:
            room = (_row_limit(s, shown, right_edge) - ox) / scale - 3 - x
            font.setPixelSize(round(start_px))
            natural = QFontMetricsF(font).horizontalAdvance(text) + 8
            if s.centered:
                room = w  # inside its own button or cell
            elif s.room_r is not None:  # inside the bubble or cell it is in
                room = min(room, (s.x + s.w + s.room_r - ox) / scale - 2 - x)
            if natural <= max(w, room):
                w = max(w, natural)
                flags = align | Qt.AlignmentFlag.AlignVCenter
            else:
                w = max(w, room)
                font, need_h = fit_font(text, w - 6, h - 2, start_px, s.bold)
        else:
            font, need_h = fit_font(text, w - 6, h - 2, start_px, s.bold)
        hover = s.id == hover_id
        if clip_w is not None:  # never out of the frame
            x = min(max(0.0, x), max(0.0, clip_w - w))
        box = QRectF(x, y, w, max(h, need_h + 2))
        p.setPen(QPen(QColor(0, 0, 0, 60), 1) if need_h > h else Qt.PenStyle.NoPen)
        p.setBrush(QColor(*s.bg, 40 if hover else 255))
        p.drawRoundedRect(box, 3, 3)
        p.setFont(font)
        p.setPen(QColor(*s.fg, 60 if hover else 255))
        p.drawText(box.adjusted(3, 0, -3, 0), int(flags), text)
        drawn += 1
    return drawn
