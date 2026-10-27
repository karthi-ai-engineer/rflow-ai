"""Rflow's look, "Obsidian Signal" (Rflow UI 2.0; the design is docs/design/rflow-ui.html, in Figma as "Rflow UI 2.0").

One surface colour whose depth comes from light and shadow, a few instrument lamps, and one job per colour: Iris is
what can be pressed, Violet is the AI, Coral is listening, Mint is ready, Amber wants attention, Rose went wrong. Dark is
"Obsidian", light is "Porcelain"; every text is 4.5:1 and every control edge 3:1 on any surface of its theme.

Qt's style sheets have no shadows, so the soft depth is painted here. A widget asks for a surface with
`surface(widget, "key")`, and its nearest *host* among its parents (a card, a page, the sidebar) paints that surface
beneath it: raised keys and cards with a light and a dark soft shadow, pressed wells and fields with inner ones. The
widgets stay ordinary Qt widgets with transparent backgrounds, so text, focus and keyboard work as always. Shadows are
blurred once per size and colour (numpy) and drawn nine-sliced, so a page repaints quickly.
"""
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PySide6.QtCore import QByteArray, QEvent, QObject, QPointF, QRect, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QGuiApplication,
    QIcon,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QAbstractButton, QApplication, QComboBox, QWidget

FONT_DIR = Path(__file__).parent / "static" / "fonts"  # Geist and Geist Mono (SIL Open Font License, OFL.txt)
SANS = "Geist"
MONO = "Geist Mono"
FALLBACK = ["Segoe UI Variable Text", "Segoe UI", "Yu Gothic UI"]  # if the fonts couldn't be loaded; Japanese falls back too


def rgba(r: int, g: int, b: int, a: float) -> QColor:
    return QColor(r, g, b, round(a * 255))


# ---------------------------------------------------------------- tokens

TOKENS = {
    "dark": {  # Obsidian
        "base": QColor("#191C22"), "well": QColor("#101216"), "cx_a": QColor("#1E2128"), "cx_b": QColor("#15171C"),
        "line": rgba(255, 255, 255, .06), "chip_bg": rgba(255, 255, 255, .04), "chip_line": rgba(255, 255, 255, .07),
        "rim": QColor("#6B7386"), "text": QColor("#ECEEF3"), "text2": QColor("#A6ACBA"), "text3": QColor("#8A91A1"),
        "iris": QColor("#8C9DFF"), "violet": QColor("#B48CFF"), "live": QColor("#FF6B6B"), "ok": QColor("#5BE3A6"),
        "warn": QColor("#FFC56B"), "err": QColor("#FF7D93"), "primary": QColor("#8C9DFF"),
        "on_primary": QColor("#0B0D13"), "knob_on": QColor("#0B0D13"), "knob_off": QColor("#8A91A1"),
        "key_face": QColor("#22252C"), "key_lip": rgba(0, 0, 0, .45), "edge": rgba(255, 255, 255, .06),
        "glow": rgba(140, 157, 255, .28), "selection": QColor("#3A4699"),
    },
    "light": {  # Porcelain
        "base": QColor("#E8EBF1"), "well": QColor("#DFE3EA"), "cx_a": QColor("#F3F5F9"), "cx_b": QColor("#E0E4EB"),
        "line": rgba(22, 25, 32, .08), "chip_bg": rgba(22, 25, 32, .035), "chip_line": rgba(22, 25, 32, .08),
        "rim": QColor("#787F91"), "text": QColor("#161920"), "text2": QColor("#3A4050"), "text3": QColor("#5E6575"),
        "iris": QColor("#3F4ACB"), "violet": QColor("#6A3ED4"), "live": QColor("#B42F2F"), "ok": QColor("#0B6B42"),
        "warn": QColor("#8A4E00"), "err": QColor("#B02640"), "primary": QColor("#4651D2"),
        "on_primary": QColor("#FFFFFF"), "knob_on": QColor("#FFFFFF"), "knob_off": QColor("#787F91"),
        "key_face": QColor("#F6F7FA"), "key_lip": rgba(136, 146, 170, .45), "edge": rgba(255, 255, 255, .70),
        "glow": rgba(70, 81, 210, .26), "selection": QColor("#C5CAF4"),
    },
}
# Popups over other apps (the pill, Translate, the Text Transform menu) are always Obsidian: dark reads on any app.
POPUP = dict(TOKENS["dark"], base=QColor("#16181E"), cx_a=QColor("#1C1F26"), cx_b=QColor("#14161B"))

# (x, y, blur, colour) per depth, as in the design's CSS box-shadows (blur = 2 sigma)
SHADOWS = {
    "dark": {
        "card": [(-6, -6, 16, rgba(255, 255, 255, .05)), (8, 8, 20, rgba(0, 0, 0, .60))],
        "tile": [(-4, -4, 10, rgba(255, 255, 255, .045)), (5, 5, 12, rgba(0, 0, 0, .55))],
        "key": [(-2, -2, 6, rgba(255, 255, 255, .04)), (3, 3, 7, rgba(0, 0, 0, .50))],
        "in1": [(5, 5, 12, rgba(0, 0, 0, .60)), (-4, -4, 10, rgba(255, 255, 255, .04))],
        "in2": [(2, 2, 6, rgba(0, 0, 0, .58)), (-2, -2, 5, rgba(255, 255, 255, .045))],
        "float": [(0, 6, 14, rgba(0, 0, 0, .45)), (0, 1, 3, rgba(0, 0, 0, .35))],  # fades out inside a popup's 24 px
    },
    "light": {
        "card": [(-8, -8, 20, rgba(255, 255, 255, .95)), (10, 10, 24, rgba(136, 146, 170, .42))],
        "tile": [(-5, -5, 12, rgba(255, 255, 255, .90)), (6, 6, 14, rgba(136, 146, 170, .40))],
        "key": [(-3, -3, 7, rgba(255, 255, 255, .85)), (3, 3, 8, rgba(136, 146, 170, .38))],
        "in1": [(5, 5, 12, rgba(136, 146, 170, .45)), (-4, -4, 10, rgba(255, 255, 255, .90))],
        "in2": [(2, 2, 6, rgba(136, 146, 170, .45)), (-2, -2, 5, rgba(255, 255, 255, .90))],
        "float": [(0, 6, 14, rgba(30, 36, 52, .20)), (0, 1, 3, rgba(30, 36, 52, .16))],
    },
}

_theme = "dark"
APPEARANCES = ("system", "light", "dark")  # Settings.appearance
_appearance = "system"


def set_appearance(name: str) -> None:
    """The user's choice: as Windows is set, or always light or dark (dark_mode() follows it)."""
    global _appearance
    _appearance = name if name in APPEARANCES else "system"


def appearance() -> str:
    return _appearance


def set_theme(name: str) -> None:
    global _theme
    _theme = name if name in TOKENS else "dark"


def theme() -> str:
    return _theme


def dark_mode() -> bool:
    if _appearance != "system":
        return _appearance == "dark"
    return QGuiApplication.styleHints().colorScheme() != Qt.ColorScheme.Light  # Obsidian unless Windows says light


def tok(name: str, popup: bool = False) -> QColor:
    return QColor((POPUP if popup else TOKENS[_theme])[name])


def css(colour: QColor) -> str:
    """A colour for a style sheet."""
    return f"rgba({colour.red()}, {colour.green()}, {colour.blue()}, {colour.alphaF():.3f})"


# ---------------------------------------------------------------- fonts

_families: dict[str, str] = {}


def load_fonts() -> bool:
    """Register the shipped Geist fonts with Qt (once). False if they couldn't be loaded: Windows' fonts are used."""
    if _families:
        return bool(_families.get(SANS))
    for path in sorted(FONT_DIR.glob("*.ttf")):
        font_id = QFontDatabase.addApplicationFont(str(path))
        for family in QFontDatabase.applicationFontFamilies(font_id) if font_id >= 0 else []:
            _families.setdefault(MONO if "Mono" in family else SANS, family)
    _families.setdefault("loaded", "1")
    return bool(_families.get(SANS))


def font(px: float, weight: int = 400, mono: bool = False, spacing: float = 0.0, tabular: bool = False) -> QFont:
    """A font in pixels (the design's sizes), e.g. font(26, 600) for a page title. `spacing` in em, as in CSS."""
    load_fonts()
    f = QFont()
    f.setFamilies([_families.get(MONO if mono else SANS, MONO if mono else SANS), *FALLBACK])
    f.setPixelSize(round(px))
    f.setWeight(QFont.Weight(weight))
    f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    if spacing:
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing * px)
    if tabular:
        try:
            f.setFeature(QFont.Tag("tnum"), 1)  # numbers line up (Qt 6.7+)
        except (AttributeError, TypeError):
            pass
    return f


# ---------------------------------------------------------------- icons (the design's line icons, 24-unit grid)

ICONS = {
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2.5V5M12 19v2.5M2.5 12H5M19 12h2.5M5.3 5.3l1.8 1.8M16.9 16.9l1.8 '
           '1.8M5.3 18.7l1.8-1.8M16.9 7.1l1.8-1.8"/>',
    "moon": '<path d="M19.5 14.5A7.5 7.5 0 0 1 9.5 4.5a7.5 7.5 0 1 0 10 10z"/>',
    "display": '<rect x="3.5" y="4.5" width="17" height="11.5" rx="2"/><path d="M9 20h6M12 16v4"/>',  # as Windows is set
    "meeting": '<circle cx="9" cy="8.5" r="3"/><path d="M3.5 19.5a5.5 5.5 0 0 1 11 0"/>'
               '<circle cx="16.5" cy="9.5" r="2.5"/><path d="M16 14.1a4.5 4.5 0 0 1 4.5 5.4"/>',  # two people at a meeting
    "home":'<path d="M3.5 10.2 12 3.5l8.5 6.7V19a1.5 1.5 0 0 1-1.5 1.5h-4.5v-6h-5v6H5A1.5 1.5 0 0 1 3.5 19z"/>',
    "words": '<path d="M12 6.5C10 5 7.5 4.5 4 4.5v14c3.5 0 6 .5 8 2 2-1.5 4.5-2 8-2v-14c-3.5 0-6 .5-8 2zM12 6.5v14"/>',
    "tools": '<path d="M11 3.5l1.6 4.4 4.4 1.6-4.4 1.6L11 15.5l-1.6-4.4L5 9.5l4.4-1.6zM18 14.5l.8 2.2 2.2.8-2.2.8-.8 2.2'
             '-.8-2.2-2.2-.8 2.2-.8z"/>',
    "models": '<rect x="6" y="6" width="12" height="12" rx="3"/><path d="M10 10h4v4h-4zM9 3v3M15 3v3M9 18v3M15 18v3M3 '
              '9h3M3 15h3M18 9h3M18 15h3"/>',
    "settings": '<path d="M4 7h9M17 7h3M4 17h3M11 17h9"/><circle cx="15" cy="7" r="2"/><circle cx="9" cy="17" r="2"/>',
    "mic": '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21"/>',
    "search": '<circle cx="11" cy="11" r="6.5"/><path d="M20 20l-4.2-4.2"/>',
    "edit": '<path d="M4 20l1-4L16 5l3 3L8 19zM14 7l3 3"/>',
    "copy": '<rect x="9" y="9" width="11" height="11" rx="2.5"/><path d="M15 9V6.5A2.5 2.5 0 0 0 12.5 4h-6A2.5 2.5 0 0 0 '
            '4 6.5v6A2.5 2.5 0 0 0 6.5 15H9"/>',
    "check": '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    "close": '<path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/>',
    "speaker": '<path d="M4 9.5h3l4.5-4v13L7 14.5H4z"/><path d="M15 9.2a4 4 0 0 1 0 5.6M17.6 6.6a7.6 7.6 0 0 1 0 10.8"/>',
    "speaker-off": '<path d="M4 9.5h3l4.5-4v13L7 14.5H4z"/><path d="M15.5 9.5l5 5M20.5 9.5l-5 5"/>',
    "chevron-down": '<path d="M6 9.5l6 6 6-6"/>',
    "chevron-right": '<path d="M9.5 6l6 6-6 6"/>',
    "chevron-left": '<path d="M14.5 6l-6 6 6 6"/>',
    "arrow-right": '<path d="M5 12h14M13 6l6 6-6 6"/>',
    "download": '<path d="M12 4v11M7.5 10.5L12 15l4.5-4.5M5 19.5h14"/>',
    "cloud": '<path d="M7.5 18.5a4 4 0 0 1-.7-7.94A5.5 5.5 0 0 1 17.4 9.1a4.7 4.7 0 0 1 .1 9.4z"/>',
    "laptop": '<rect x="4" y="5" width="16" height="11" rx="2"/><path d="M2.5 19h19"/>',
    "key": '<circle cx="8" cy="15" r="3.5"/><path d="M10.5 12.5l8-8M15.5 7.5l2.5 2.5"/>',
    "external": '<path d="M14 4.5h5.5V10M19.5 4.5L11 13M17 14v4.5a1.5 1.5 0 0 1-1.5 1.5h-10A1.5 1.5 0 0 1 4 18.5v-10A1.5 '
                '1.5 0 0 1 5.5 7H10"/>',
    "info": '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5v5M12 16v.5"/>',
    "minus": '<circle cx="12" cy="12" r="8.5"/><path d="M8 12h8"/>',
    "eye": '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="3"/>',
    "eye-off": '<path d="M4 4l16 16M9.9 5.8A9.6 9.6 0 0 1 12 5.5c6 0 9.5 6.5 9.5 6.5a16 16 0 0 1-3 3.8M6.4 6.9C3.9 8.6 '
               '2.5 12 2.5 12S6 18.5 12 18.5c1.6 0 3-.4 4.2-1M9.9 9.9a3 3 0 0 0 4.2 4.2"/>',
    "paste": '<rect x="6" y="4.5" width="12" height="16" rx="2.5"/><path d="M9.5 4.5V3.5h5v1M9.5 10h5M9.5 14h5"/>',
    "trash": '<path d="M4.5 7h15M10 11v6M14 11v6M6 7l1 12.5A1.5 1.5 0 0 0 8.5 21h7a1.5 1.5 0 0 0 1.5-1.5L18 7M9 7V4.5h6V7"/>',
    "profile": '<circle cx="12" cy="8.5" r="3.5"/><path d="M5 20a7 7 0 0 1 14 0"/>',
    "update": '<path d="M12 4v11M7.5 8.5L12 4l4.5 4.5M5 19.5h14"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "folder": '<path d="M3.5 7.5A1.5 1.5 0 0 1 5 6h4.5l2 2.5H19a1.5 1.5 0 0 1 1.5 1.5v8A1.5 1.5 0 0 1 19 19.5H5A1.5 1.5 0 '
              '0 1 3.5 18z"/>',
    "warning": '<path d="M12 4 21 19.5H3zM12 10v4M12 17v.3"/>',
    "reading": '<path d="M5 5h14M5 9.5h14M5 14h9M5 18.5h6"/>',
    "live": '<rect x="3" y="5" width="18" height="14" rx="3"/><path d="M7 10.5h5M14.5 10.5h2.5M7 14.5h2.5M12 14.5h5"/>',
    # the sections of phase 31: a page with a bolt (a phrase that types your text), a wand, A to 文, #, a flag
    "snippets": '<path d="M13.5 3.5H7A1.5 1.5 0 0 0 5.5 5v14A1.5 1.5 0 0 0 7 20.5h10a1.5 1.5 0 0 0 1.5-1.5V8.5zM13.5 3.5v5h5'
                'M12.5 11 10 14.5h4L11.5 18"/>',
    "transform": '<path d="M4.5 19.5l10-10M13 8l3 3M18 3.5v3.5M16.25 5.25h3.5M19.5 12.5v2.5M18.25 13.75h2.5M9.5 3.5v2.5'
                 'M8.25 4.75h2.5"/>',
    "translate": '<path d="M3.5 6h9M8 3.5V6M10.5 6c-.6 3.6-2.9 6.6-6 8M6 9.5c1 1.9 2.6 3.4 4.5 4.3M12.5 20.5l3.75-9'
                 ' 3.75 9M13.9 17.3h4.7"/>',
    "formatting": '<path d="M10 4 8 20M16.5 4l-2 16M5 9h15M4 15h15"/>',
    "report": '<path d="M5.5 21V4M5.5 4.5h12.5l-2.5 4.25L18 13H5.5"/>',
}


@lru_cache(maxsize=512)
def icon_pixmap(name: str, colour: str, size: int = 18, dpr: float = 1.0, stroke: float = 1.75) -> QPixmap:
    """A design icon in one colour (a CSS colour string), at `size` logical pixels."""
    from PySide6.QtSvg import QSvgRenderer
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{colour}" '
           f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')
    px = max(1, round(size * dpr))
    pixmap = QPixmap(px, px)
    pixmap.fill(Qt.GlobalColor.transparent)
    p = QPainter(pixmap)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    QSvgRenderer(QByteArray(svg.encode())).render(p, QRectF(0, 0, px, px))
    p.end()
    pixmap.setDevicePixelRatio(dpr)
    return pixmap


def _hex(colour: QColor) -> str:
    return colour.name(QColor.NameFormat.HexArgb) if colour.alpha() < 255 else colour.name()


def make_icon(name: str, colour: QColor, active: QColor | None = None, size: int = 18) -> QIcon:
    """An icon with its normal colour, a colour for checked or hovered, and a quieter one when disabled."""
    dpr = max(1.0, QGuiApplication.primaryScreen().devicePixelRatio() if QGuiApplication.primaryScreen() else 1.0)
    icon = QIcon()
    for mode, c in ((QIcon.Mode.Normal, colour), (QIcon.Mode.Active, active or colour),
                    (QIcon.Mode.Selected, active or colour), (QIcon.Mode.Disabled, tok("text3"))):
        icon.addPixmap(icon_pixmap(name, _hex(c), size, dpr), mode, QIcon.State.Off)
        icon.addPixmap(icon_pixmap(name, _hex(active or c) if mode != QIcon.Mode.Disabled else _hex(c), size, dpr),
                       mode, QIcon.State.On)
    return icon


# ---------------------------------------------------------------- soft shadows

def _blur(alpha: np.ndarray, sigma: float) -> np.ndarray:
    """A Gaussian blur, rows then columns (separable)."""
    if sigma <= 0:
        return alpha
    r = max(1, math.ceil(sigma * 3))
    x = np.arange(-r, r + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    for axis in (1, 0):
        padded = np.pad(alpha, [(0, 0), (r, r)] if axis == 1 else [(r, r), (0, 0)], mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * r + 1, axis=axis)
        alpha = windows @ kernel
    return alpha


def _mask(size: int, pad: int, core: int, radius: float, inner: bool) -> np.ndarray:
    """A rounded square (or, `inner`, everything around it), antialiased, as floats 0..1."""
    image = QImage(size, size, QImage.Format.Format_Grayscale8)
    image.fill(255 if inner else 0)
    p = QPainter(image)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(QColor(0, 0, 0) if inner else QColor(255, 255, 255))
    p.drawRoundedRect(QRectF(pad, pad, core, core), radius, radius)
    p.end()
    raw = np.frombuffer(image.constBits(), dtype=np.uint8, count=image.sizeInBytes())
    return raw.reshape(size, image.bytesPerLine())[:, :size].astype(np.float32) / 255.0


@lru_cache(maxsize=256)
def _tile(radius: int, blur: int, rgba_: tuple[int, int, int, int], inner: bool, reach: int, dpr: float) -> tuple:
    """A blurred shadow tile to draw nine-sliced: (pixmap, pad, cut) in logical pixels."""
    sigma = blur / 2
    pad = math.ceil(sigma * 3) + 1 + (reach if inner else 0)
    core = 2 * radius + 4
    size = core + 2 * pad
    scale = dpr
    big = round(size * scale)
    alpha = _blur(_mask(big, round(pad * scale), round(core * scale), radius * scale, inner), sigma * scale)
    r, g, b, a = rgba_
    alpha = np.clip(alpha * (a / 255.0), 0.0, 1.0)
    argb = np.empty((big, big, 4), dtype=np.uint8)  # premultiplied BGRA in memory
    argb[..., 0] = np.round(alpha * b)
    argb[..., 1] = np.round(alpha * g)
    argb[..., 2] = np.round(alpha * r)
    argb[..., 3] = np.round(alpha * 255)
    image = QImage(argb.tobytes(), big, big, big * 4, QImage.Format.Format_ARGB32_Premultiplied).copy()
    pixmap = QPixmap.fromImage(image)
    pixmap.setDevicePixelRatio(scale)
    return pixmap, pad, pad + radius + 1  # the 2 pixels in the middle are stretched


def _nine(p: QPainter, pixmap: QPixmap, cut: int, target: QRectF) -> None:
    """Draw a square tile into `target`, its corners `cut` wide kept, its edges and middle stretched. The slices meet
    on whole device pixels, so no hairline shows between them at 125% or 150%."""
    s = pixmap.width() / pixmap.devicePixelRatio()
    k = pixmap.devicePixelRatio()
    device = p.device()
    dpr = device.devicePixelRatioF() if device is not None else 1.0

    def axis(start: float, size: float) -> tuple[list[float], list[float]]:
        """Where the slices go along one side, and where they come from: three slices, or one (scaled) when the side
        is no longer than its two corners (a circle, the round ends of a pill)."""
        def snap(v: float) -> float:
            return round(v * dpr) / dpr
        if size <= 2 * cut:
            return [snap(start), snap(start + size)], [0, s]
        return [snap(start), snap(start + cut), snap(start + size - cut), snap(start + size)], [0, cut, s - cut, s]
    xs, src_x = axis(target.x(), target.width())
    ys, src_y = axis(target.y(), target.height())
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            tw, th = xs[i + 1] - xs[i], ys[j + 1] - ys[j]
            sw, sh = src_x[i + 1] - src_x[i], src_y[j + 1] - src_y[j]
            if tw > 0 and th > 0 and sw > 0 and sh > 0:
                p.drawPixmap(QRectF(xs[i], ys[j], tw, th), pixmap, QRectF(src_x[i] * k, src_y[j] * k, sw * k, sh * k))


def _key(colour: QColor) -> tuple[int, int, int, int]:
    return colour.red(), colour.green(), colour.blue(), colour.alpha()


def outer_shadow(p: QPainter, rect: QRectF, radius: float, shadows: list, dpr: float = 1.0) -> None:
    radius = int(min(radius, rect.width() / 2, rect.height() / 2))
    for dx, dy, blur, colour in shadows:
        pixmap, pad, cut = _tile(radius, blur, _key(colour), False, 0, round(dpr * 2) / 2)
        _nine(p, pixmap, cut, rect.translated(dx, dy).adjusted(-pad, -pad, pad, pad))


def inner_shadow(p: QPainter, rect: QRectF, radius: float, shadows: list, dpr: float = 1.0) -> None:
    radius = int(min(radius, rect.width() / 2, rect.height() / 2))
    path = QPainterPath()
    path.addRoundedRect(rect, radius, radius)
    p.save()
    p.setClipPath(path, Qt.ClipOperation.IntersectClip)
    for dx, dy, blur, colour in shadows:
        reach = max(abs(dx), abs(dy))
        pixmap, pad, cut = _tile(radius, blur, _key(colour), True, reach, round(dpr * 2) / 2)
        _nine(p, pixmap, cut, rect.translated(dx, dy).adjusted(-pad, -pad, pad, pad))
    p.restore()


def css_gradient(rect: QRectF, angle: float, start: QColor, end: QColor) -> QLinearGradient:
    """CSS's linear-gradient(<angle>deg, start, end) over `rect`."""
    a = math.radians(angle)
    dx, dy = math.sin(a), -math.cos(a)
    length = abs(rect.width() * dx) + abs(rect.height() * dy)
    c = rect.center()
    g = QLinearGradient(QPointF(c.x() - dx * length / 2, c.y() - dy * length / 2),
                        QPointF(c.x() + dx * length / 2, c.y() + dy * length / 2))
    g.setColorAt(0, start)
    g.setColorAt(1, end)
    return g


def rounded(rect: QRectF, radius: float) -> QPainterPath:
    path = QPainterPath()
    radius = min(radius, rect.width() / 2, rect.height() / 2)
    path.addRoundedRect(rect, radius, radius)
    return path


def lip(p: QPainter, rect: QRectF, radius: float, dy: float, colour: QColor) -> None:
    """CSS's inset 0 <dy>px 0 <colour>: a thin crescent along the top (dy > 0) or the bottom (dy < 0) edge."""
    shape = rounded(rect, radius)
    p.fillPath(shape.subtracted(rounded(rect.translated(0, dy), radius)), colour)


# ---------------------------------------------------------------- painting a surface

RADIUS = {"card": 20, "tile": 14, "key": 14, "primary": 14, "well": 14, "well1": 20, "field": 14, "pressed": 14,
          "chip": 999, "seg": 14, "segment": 10, "nav": 14, "row": 14, "keycap": 6, "orb": 999, "mark": 10,
          "toast": 999, "danger": 14, "flat": 20}


def paint_surface(p: QPainter, kind: str, rect: QRectF, radius: float | None = None, *, popup: bool = False,
                  hover: bool = False, down: bool = False, checked: bool = False, focus: bool = False,
                  enabled: bool = True, dpr: float = 1.0) -> None:
    """One surface of the design, painted at `rect` (shadows reach outside it)."""
    t = POPUP if popup else TOKENS[_theme]
    sh = SHADOWS["dark" if popup else _theme]
    r = RADIUS.get(kind, 14) if radius is None else radius
    r = min(r, rect.width() / 2, rect.height() / 2)

    def raised(level: str, face=None) -> None:
        outer_shadow(p, rect, r, sh[level], dpr)
        p.fillPath(rounded(rect, r), face or css_gradient(rect, 145, t["cx_a"], t["cx_b"]))
        lip(p, rect, r, 1, t["edge"])

    def pressed(level: str = "in2", fill: str = "base") -> None:
        p.fillPath(rounded(rect, r), t[fill])
        inner_shadow(p, rect, r, sh[level], dpr)

    def border(colour: QColor, width: float) -> None:
        p.save()
        p.setPen(QPen(colour, width))
        p.setBrush(Qt.BrushStyle.NoBrush)
        inset = width / 2
        p.drawRoundedRect(rect.adjusted(inset, inset, -inset, -inset), max(0, r - inset), max(0, r - inset))
        p.restore()

    if kind in ("card", "tile", "toast"):
        raised("card" if kind == "card" else "tile")
    elif kind == "flat":
        border(t["line"], 1)
    elif kind in ("key", "danger"):
        if not enabled:
            pressed()
        elif down or checked:
            pressed()
        else:
            raised("key")
    elif kind == "primary":
        if not enabled:
            pressed()
            return
        glow = [(0, 4, 14, t["glow"])] if not down else []
        outer_shadow(p, rect, r, glow, dpr)
        face = t["primary"].darker(108) if down else t["primary"].lighter(104) if hover else t["primary"]
        p.fillPath(rounded(rect, r), face)
        lip(p, rect, r, 1, rgba(255, 255, 255, .30))
    elif kind == "well":
        pressed("in2", "well")
    elif kind == "well1":
        pressed("in1", "well")
    elif kind == "pressed":
        pressed()
    elif kind == "chosen":  # a chosen option: pressed in, with an Iris edge
        pressed("in1")
        border(t["iris"], 2)
    elif kind == "field":
        pressed("in2", "well")
        if focus:
            border(t["iris"], 2)
        else:
            border(t["rim"], 1)
    elif kind == "chip":
        if checked:
            pressed()
        elif hover or down:
            raised("key")
        else:
            p.fillPath(rounded(rect, r), t["chip_bg"])
            border(t["chip_line"], 1)
    elif kind == "seg":
        pressed("in2", "well")
    elif kind == "segment":
        if checked:
            raised("key")
    elif kind in ("nav", "row"):
        if checked or down:
            pressed()
        elif hover and kind == "row":
            pressed()
    elif kind == "keycap":
        outer_shadow(p, rect, r, sh["key"], dpr)
        p.fillPath(rounded(rect, r), t["key_face"])
        lip(p, rect, r, -2, t["key_lip"])
        lip(p, rect, r, 1, t["edge"])
    elif kind == "mark":
        outer_shadow(p, rect, r, sh["key"], dpr)
        p.fillPath(rounded(rect, r), t["key_face"])
        lip(p, rect, r, -2, t["key_lip"])
        lip(p, rect, r, 1, t["edge"])
    elif kind == "orb":
        outer_shadow(p, rect, r, sh["card"], dpr)
        p.fillPath(rounded(rect, r), css_gradient(rect, 145, t["cx_a"], t["cx_b"]))
        lip(p, rect, r, 1, t["edge"])
    elif kind == "float":
        outer_shadow(p, rect, r, sh["float"], dpr)
        p.fillPath(rounded(rect, r), t["base"])
        border(rgba(255, 255, 255, .07), 1)


# ---------------------------------------------------------------- surfaces painted by their host

SURFACE = "neuSurface"  # the dynamic property naming a widget's surface
_HOST = "_neu_host"


def surface(widget: QWidget, kind: str, radius: float | None = None) -> QWidget:
    """Give `widget` a surface of the design, painted beneath it by its nearest host. Returns the widget."""
    widget.setProperty(SURFACE, kind)
    if radius is not None:
        widget.setProperty("neuRadius", radius)
    widget.installEventFilter(_watcher())
    if isinstance(widget, QComboBox) and widget.lineEdit() is not None:
        widget.lineEdit().installEventFilter(_watcher())  # its focus is the box's focus
    if isinstance(widget, QAbstractButton):
        widget.toggled.connect(lambda *_: refresh(widget))
    refresh(widget)
    return widget


def make_host(widget: QWidget) -> None:
    """Mark `widget` as a surface host: its paintEvent must call paint_hosted(widget, painter)."""
    setattr(widget, _HOST, True)


def host_of(widget: QWidget) -> QWidget | None:
    parent = widget.parentWidget()
    while parent is not None:
        if getattr(parent, _HOST, False):
            return parent
        if parent.isWindow():
            return None
        parent = parent.parentWidget()
    return None


def refresh(widget: QWidget, old: QRect | None = None) -> None:
    """Repaint the part of the host where this widget's surface (and its shadows) are."""
    host = host_of(widget)
    if host is None:
        return
    margin = 40
    top_left = widget.mapTo(host, widget.rect().topLeft())
    area = QRect(top_left, widget.size()).adjusted(-margin, -margin, margin, margin)
    if old is not None:
        area = area.united(old.adjusted(-margin, -margin, margin, margin))
    host.update(area)


def _hovered(widget: QWidget) -> bool:
    return widget.underMouse() and widget.isEnabled()


def _focused(widget: QWidget) -> bool:
    focus = QApplication.focusWidget()
    return focus is not None and (focus is widget or widget.isAncestorOf(focus))


def paint_hosted(host: QWidget, p: QPainter, popup: bool | None = None) -> None:
    """Paint the surfaces of the host's widgets (those whose nearest host is this one), in tree order. In a popup over
    other apps (its window has the neuPopup property) they are always Obsidian."""
    if popup is None:
        window = host.window()
        popup = bool(window is not None and window.property("neuPopup"))
    dpr = host.devicePixelRatioF()
    clip = p.clipBoundingRect() if p.hasClipping() else QRectF(host.rect())
    for widget in host.findChildren(QWidget):
        kind = widget.property(SURFACE)
        if not kind or host_of(widget) is not host or not widget.isVisibleTo(host):
            continue
        top_left = widget.mapTo(host, widget.rect().topLeft())
        rect = QRectF(top_left.x(), top_left.y(), widget.width(), widget.height())
        if not clip.intersects(rect.adjusted(-40, -40, 40, 40)):
            continue
        radius = widget.property("neuRadius")
        button = widget if isinstance(widget, QAbstractButton) else None
        paint_surface(p, kind, rect, radius, popup=popup, hover=_hovered(widget),
                      down=bool(button and button.isDown()) or bool(widget.property("neuDown")),
                      checked=bool(button and button.isCheckable() and button.isChecked()) or bool(widget.property("neuChecked")),
                      focus=_focused(widget), enabled=widget.isEnabled(), dpr=dpr)


class _Watcher(QObject):
    """Repaints a surface when what it shows changes: hover, press, focus, a move, showing or hiding."""

    EVENTS = {QEvent.Type.Enter, QEvent.Type.Leave, QEvent.Type.FocusIn, QEvent.Type.FocusOut, QEvent.Type.Show,
              QEvent.Type.Hide, QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease, QEvent.Type.EnabledChange,
              QEvent.Type.ParentChange, QEvent.Type.KeyPress, QEvent.Type.KeyRelease}

    def eventFilter(self, obj, event) -> bool:
        kind = event.type()
        if kind in self.EVENTS or kind in (QEvent.Type.Move, QEvent.Type.Resize):
            widget = obj
            if widget.property(SURFACE) is None and isinstance(widget.parentWidget(), QComboBox):
                widget = widget.parentWidget()  # an editable box's line edit: the box shows its focus
            host, old = host_of(widget), None
            if host is not None and kind in (QEvent.Type.Move, QEvent.Type.Resize):
                now = widget.mapTo(host, widget.rect().topLeft())  # where it is in the host, to clear where it was
                if kind == QEvent.Type.Move:
                    old = QRect(now - (event.pos() - event.oldPos()), widget.size())
                else:
                    old = QRect(now, event.oldSize())
            refresh(widget, old)
        return False


@lru_cache(maxsize=1)
def _watcher() -> _Watcher:
    return _Watcher(QApplication.instance())


# ---------------------------------------------------------------- the style sheet (text, sizes, transparent controls)

def stylesheet(name: str | None = None) -> str:
    """The window's style sheet for the theme: text colours and sizes, and controls with no background of their own
    (their surfaces are painted by their hosts)."""
    t = TOKENS[name or _theme]
    c = {k: css(v) for k, v in t.items()}
    image = 'arrow-dark.png' if (name or _theme) == 'dark' else 'arrow-light.png'
    arrow = f"url({(Path(__file__).parent / 'static' / 'ui' / image).as_posix()})"
    return f"""
    #root, #page, #sidebar, #rail {{ background: {c['base']}; }}
    QWidget {{ color: {c['text']}; }}
    QLabel {{ background: transparent; }}
    QLabel[tone="2"] {{ color: {c['text2']}; }}
    QLabel[tone="3"] {{ color: {c['text3']}; }}
    QLabel[tone="iris"] {{ color: {c['iris']}; }}
    QLabel[tone="violet"] {{ color: {c['violet']}; }}
    QLabel[tone="ok"] {{ color: {c['ok']}; }}
    QLabel[tone="warn"] {{ color: {c['warn']}; }}
    QLabel[tone="err"] {{ color: {c['err']}; }}
    QLabel[tone="live"] {{ color: {c['live']}; }}
    QLabel[role="display"] {{ font-size: 30px; font-weight: 600; }}
    QLabel[role="title"] {{ font-size: 26px; font-weight: 600; }}
    QLabel[role="hero"] {{ font-size: 20px; font-weight: 600; }}
    QLabel[role="heading"] {{ font-size: 15px; font-weight: 600; }}
    QLabel[role="subtitle"] {{ font-size: 13px; color: {c['text2']}; }}
    QLabel[role="rowtitle"] {{ font-size: 14px; font-weight: 500; }}
    QLabel[role="caption"] {{ font-size: 12px; font-weight: 500; }}
    QLabel[role="caps"] {{ font-size: 11px; font-weight: 600; color: {c['text3']}; }}
    QLabel[role="wordmark"] {{ font-size: 18px; font-weight: 600; }}
    QLabel[role="number"] {{ font-size: 16px; font-weight: 600; }}
    QToolTip {{ color: {c['text']}; background: {c['well']}; border: 1px solid {c['line']}; padding: 6px 8px;
                border-radius: 8px; font-size: 12px; }}

    QPushButton, QToolButton {{ background: transparent; border: none; color: {c['text']}; outline: none; }}
    QPushButton:disabled, QToolButton:disabled {{ color: {c['text3']}; }}

    QLineEdit, QPlainTextEdit, QTextBrowser, QComboBox {{ background: transparent; border: none; color: {c['text']};
        selection-background-color: {c['selection']}; selection-color: {c['text']}; font-size: 14px; }}
    QLineEdit {{ padding: 0 12px; min-height: 40px; }}
    QLineEdit[bare="true"], QComboBox QLineEdit {{ padding: 0; min-height: 0; }}
    QLineEdit:read-only {{ color: {c['text2']}; }}
    QPlainTextEdit, QTextBrowser {{ padding: 10px 12px; }}
    QPlainTextEdit > QWidget, QTextBrowser > QWidget {{ background: transparent; }}
    QComboBox {{ padding: 0 36px 0 12px; min-height: 40px; }}
    QComboBox[size="sm"] {{ min-height: 32px; font-size: 13px; }}
    QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right; width: 34px; border: none;
                            background: transparent; }}
    QComboBox::down-arrow {{ image: {arrow}; width: 12px; height: 12px; }}
    QComboBox QAbstractItemView, QListView {{ background: {c['well']}; color: {c['text']}; border: 1px solid {c['line']};
        border-radius: 10px; padding: 4px; outline: none; selection-background-color: {c['cx_a']};
        selection-color: {c['text']}; }}
    QListView::item, QComboBox QAbstractItemView::item {{ min-height: 32px; padding: 0 8px; border-radius: 8px; }}
    QListView::item:hover, QComboBox QAbstractItemView::item:hover {{ background: {c['cx_a']}; }}
    QListView::item:selected {{ background: {c['cx_a']}; color: {c['text']}; }}
    QFrame#searchPopup {{ background: {c['well']}; border: 1px solid {c['line']}; border-radius: 12px; }}
    QFrame#searchPopup QListView {{ border: none; background: transparent; }}
    QMenu {{ background: {c['well']}; color: {c['text']}; border: 1px solid {c['line']}; border-radius: 10px; padding: 6px; }}
    QMenu::item {{ padding: 8px 28px 8px 12px; border-radius: 8px; }}
    QMenu::item:selected {{ background: {c['cx_a']}; }}
    QMenu::separator {{ height: 1px; background: {c['line']}; margin: 4px 8px; }}
    QCheckBox {{ background: transparent; spacing: 10px; }}
    QListWidget {{ background: transparent; border: none; }}
    QListWidget::item {{ padding: 6px 4px; }}
    QProgressBar {{ background: {c['well']}; border: none; border-radius: 3px; max-height: 6px; min-height: 6px; }}
    QProgressBar::chunk {{ background: {c['warn']}; border-radius: 3px; }}
    QScrollArea {{ background: transparent; border: none; }}
    QScrollArea > QWidget > QWidget {{ background: transparent; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 4px 2px; }}
    QScrollBar::handle:vertical {{ background: {c['line']}; border-radius: 3px; min-height: 32px; }}
    QScrollBar::handle:vertical:hover {{ background: {c['rim']}; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical, QScrollBar::add-page:vertical,
    QScrollBar::sub-page:vertical {{ height: 0; background: transparent; }}
    QScrollBar:horizontal {{ height: 0; }}
    QMessageBox, QInputDialog {{ background: {c['base']}; }}
    QMessageBox QLabel, QInputDialog QLabel {{ font-size: 14px; }}
    QMessageBox QPushButton, QInputDialog QPushButton {{ background: {c['cx_a']}; border: 1px solid {c['line']};
        border-radius: 10px; min-width: 88px; min-height: 32px; padding: 0 14px; font-size: 13px; }}
    QMessageBox QPushButton:hover, QInputDialog QPushButton:hover {{ border-color: {c['rim']}; }}
    QMessageBox QPushButton:default, QInputDialog QPushButton:default {{ background: {c['primary']};
        color: {c['on_primary']}; border: none; font-weight: 600; }}
    QInputDialog QLineEdit {{ background: {c['well']}; border: 1px solid {c['rim']}; border-radius: 10px; }}
    """
