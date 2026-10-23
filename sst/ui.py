"""The widgets of Rflow UI 2.0 (sst.theme has the colours, fonts, icons and the soft depth they are drawn with).

Cards, wells and pages are surface hosts: they paint the surfaces of the widgets inside them (sst.theme.paint_hosted).
Buttons, toggles, lamps, keycaps and the voice orb draw themselves, so they look the same on any page and in tests.
"""
import math
import random
from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QEvent, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFontMetrics, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QAbstractButton,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from sst import theme
from sst.theme import font, icon_pixmap, paint_hosted, paint_surface, surface, tok


def _t(widget: QWidget, name: str) -> QColor:
    """A colour of the theme, or of the popups (always Obsidian) for a widget inside one."""
    return tok(name, popup=in_popup(widget))


def _shade(widget: QWidget) -> str:
    return "dark" if in_popup(widget) else theme.theme()


def in_popup(widget: QWidget) -> bool:
    window = widget.window()
    return bool(window is not None and window.property("neuPopup"))


# ---------------------------------------------------------------- hosts


class Host(QWidget):
    """A plain widget that paints its children's surfaces (and its background, when `background` is set)."""

    def __init__(self, background: str = "", parent=None):
        super().__init__(parent)
        theme.make_host(self)
        self.background = background

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.background:
            p.fillRect(self.rect(), _t(self, self.background))
        paint_hosted(self, p)
        p.end()


class Card(QFrame):
    """A raised card (or a "well", a "tile", a "flat" outline): painted by its own host, and a host for what's inside."""

    def __init__(self, kind: str = "card", radius: float | None = None, parent=None):
        super().__init__(parent)
        theme.make_host(self)
        surface(self, kind, radius)
        self.setFrameShape(QFrame.Shape.NoFrame)

    def set_kind(self, kind: str) -> None:
        self.setProperty(theme.SURFACE, kind)
        theme.refresh(self)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        paint_hosted(self, p)
        p.end()


def card(spacing: int = 12, padding: tuple[int, int, int, int] = (20, 20, 20, 20), kind: str = "card",
         horizontal: bool = False) -> tuple[Card, QVBoxLayout | QHBoxLayout]:
    frame = Card(kind)
    layout = QHBoxLayout(frame) if horizontal else QVBoxLayout(frame)
    layout.setContentsMargins(*padding)
    layout.setSpacing(spacing)
    return frame, layout


# ---------------------------------------------------------------- text

def label(value: str = "", role: str | None = None, tone: str | None = None, wrap: bool = True,
          selectable: bool = False) -> QLabel:
    """A label in the design's type scale: role display, title, hero, heading, subtitle, rowtitle, caption, caps;
    tone 2 or 3 (quieter text), or a colour's job (iris, violet, ok, warn, err, live)."""
    w = QLabel(value)
    w.setWordWrap(wrap)
    if role:
        w.setProperty("role", role)
    if tone:
        w.setProperty("tone", tone)
    if selectable:
        w.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return w


def set_tone(widget: QLabel, tone: str | None) -> None:
    """Change a label's colour job (its style is applied again)."""
    if widget.property("tone") != tone:
        widget.setProperty("tone", tone)
        widget.style().unpolish(widget)
        widget.style().polish(widget)


class Divider(QWidget):
    """A 1 px line in the theme's line colour."""

    def __init__(self, vertical: bool = False, parent=None):
        super().__init__(parent)
        if vertical:
            self.setFixedWidth(1)
        else:
            self.setFixedHeight(1)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), _t(self, "line"))
        p.end()


def divider(vertical: bool = False) -> Divider:
    return Divider(vertical)


# ---------------------------------------------------------------- buttons

SIZES = {"sm": (32, 12, 13, 10), "md": (36, 16, 14, 14), "lg": (44, 20, 14, 14)}  # height, padding, font px, radius


class Button(QPushButton):
    """A button of the design. kind: key (raised), primary (the one lit button of a view), ghost (Iris text), quiet
    (secondary text), danger (Rose text), nav (the sidebar), chip, segment. A `hint` is drawn as a keycap inside it
    (Copy [C]); an `icon` is one of sst.theme.ICONS. text() stays the plain label."""

    def __init__(self, text: str = "", kind: str = "key", size: str = "md", icon: str | None = None,
                 hint: str | None = None, icon_after: bool = False, parent=None):
        super().__init__(text, parent)
        self.kind, self.size_name, self.icon_name, self.hint, self.icon_after = kind, size, icon, hint, icon_after
        self.badge = ""  # a count on a nav button
        self.suffix = ""  # quieter words after the label (a tab's count: "Your words 17")
        self.compact = False  # a nav button in the narrow rail: icon over its label
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Fixed if kind not in ("nav",) else QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)
        kinds = {"key": "key", "primary": "primary", "danger": "danger", "nav": "nav", "chip": "chip",
                 "segment": "segment"}
        if kind in kinds:
            surface(self, kinds[kind], self._radius())
        else:
            self.setProperty("variant", kind)

    def _radius(self) -> float:
        if self.kind in ("chip",):
            return 999
        if self.kind == "segment":
            return 10
        return SIZES[self.size_name][3] if self.size_name in SIZES else 14

    def set_kind(self, kind: str) -> None:
        """key <-> primary, e.g. the button that becomes the one to press."""
        self.kind = kind
        self.setProperty(theme.SURFACE, kind if kind in ("key", "primary", "danger") else None)
        theme.refresh(self)
        self.update()

    def set_hint(self, hint: str | None) -> None:
        self.hint = hint
        self.updateGeometry()
        self.update()

    def setText(self, text: str) -> None:
        super().setText(text)
        self.updateGeometry()

    def _metrics(self):
        height, padding, px, _ = SIZES.get(self.size_name, SIZES["md"])
        if self.kind == "nav":
            height, padding, px = (52, 0, 11) if self.compact else (40, 12, 14)
        elif self.kind == "chip":
            height, padding, px = (28 if self.size_name == "sm" else 32), 12, 13
        elif self.kind == "segment":
            height, padding, px = 28, 12, 13
        elif self.kind in ("ghost", "quiet"):
            padding = 8 if self.size_name != "flush" else 0
        weight = 600 if (self.kind == "primary" or (self.isCheckable() and self.isChecked()
                                                    and self.kind in ("nav", "chip", "segment"))) else 500
        return height, padding, px, weight

    def sizeHint(self) -> QSize:
        height, padding, px, weight = self._metrics()
        fm = QFontMetrics(font(px, 600))  # measured bold: the width doesn't jump when the button is checked
        width = fm.horizontalAdvance(self.text()) + 2 * padding
        if self.suffix:
            width += 6 + QFontMetrics(font(px, 500)).horizontalAdvance(self.suffix)
        if self.icon_name:
            width += (18 if self.kind == "nav" else 16) + (12 if self.kind == "nav" else 8) * bool(self.text())
        if self.hint:
            width += 8 + max(22, QFontMetrics(font(11, 500, mono=True)).horizontalAdvance(self.hint) + 12)
        if self.badge:
            width += 26
        if self.kind == "nav" and self.compact:
            width = 64
        return QSize(width, height)

    def minimumSizeHint(self) -> QSize:
        hint = self.sizeHint()
        return QSize(hint.width() if self.kind != "nav" else 40, hint.height())

    def hitButton(self, pos) -> bool:
        return self.rect().contains(pos)

    def _colour(self) -> QColor:
        enabled, hover = self.isEnabled(), self.underMouse()
        if not enabled:
            return _t(self, "text3")
        checked = self.isCheckable() and self.isChecked()
        return {
            "primary": _t(self, "on_primary"),
            "ghost": _t(self, "text") if hover else _t(self, "iris"),
            "quiet": _t(self, "text") if hover else _t(self, "text2"),
            "danger": _t(self, "err"),
            "nav": _t(self, "text") if checked or hover else _t(self, "text2"),
            "segment": _t(self, "text") if checked else _t(self, "text2"),
        }.get(self.kind, _t(self, "text"))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        height, padding, px, weight = self._metrics()
        colour = self._colour()
        r = QRectF(self.rect())
        dpr = self.devicePixelRatioF()
        if self.kind == "nav" and self.compact:  # the rail: the icon over its label
            checked = self.isChecked()
            if self.icon_name:
                c = _t(self, "iris") if checked else colour
                p.drawPixmap(QPointF(r.center().x() - 9, 8), icon_pixmap(self.icon_name, c.name(), 18, dpr))
            p.setFont(font(11, 600 if checked else 500))
            p.setPen(colour)
            p.drawText(QRectF(0, 28, r.width(), 16), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                       self.text())
            self._focus(p, r)
            p.end()
            return
        f = font(px, weight)
        fm = QFontMetrics(f)
        icon_w = (18 if self.kind == "nav" else 16) if self.icon_name else 0
        gap = (12 if self.kind == "nav" else 8) if self.icon_name and self.text() else 0
        hint_w = max(22, QFontMetrics(font(11, 500, mono=True)).horizontalAdvance(self.hint) + 12) if self.hint else 0
        suffix_w = 6 + QFontMetrics(font(px, 500)).horizontalAdvance(self.suffix) if self.suffix else 0
        content = icon_w + gap + fm.horizontalAdvance(self.text()) + suffix_w + (8 + hint_w if self.hint else 0)
        x = padding if self.kind == "nav" else (r.width() - content) / 2
        if self.kind == "nav" and self.isChecked():
            colour_icon = _t(self, "iris")
        else:
            colour_icon = colour
        if self.icon_name and not self.icon_after:
            size = icon_w
            p.drawPixmap(QPointF(x, (r.height() - size) / 2), icon_pixmap(self.icon_name, colour_icon.name(), size, dpr))
            x += icon_w + gap
        p.setFont(f)
        p.setPen(colour)
        text_w = fm.horizontalAdvance(self.text())
        p.drawText(QRectF(x, 0, text_w + 2, r.height()), Qt.AlignmentFlag.AlignVCenter, self.text())
        x += text_w
        if self.suffix:
            p.setFont(font(px, 500))
            p.setPen(_t(self, "text3"))
            p.drawText(QRectF(x + 6, 0, suffix_w, r.height()), Qt.AlignmentFlag.AlignVCenter, self.suffix)
            x += suffix_w
            p.setFont(f)
            p.setPen(colour)
        if self.icon_name and self.icon_after:
            x += gap
            p.drawPixmap(QPointF(x, (r.height() - icon_w) / 2), icon_pixmap(self.icon_name, colour_icon.name(), icon_w, dpr))
            x += icon_w
        if self.hint:
            x += 8
            cap = QRectF(x, (r.height() - 20) / 2, hint_w, 20)
            if self.kind == "primary":
                p.fillPath(theme.rounded(cap, 6), QColor(0, 0, 0, 36) if _shade(self) == "dark" else QColor(255, 255, 255, 46))
                p.setPen(colour)
            else:
                paint_surface(p, "keycap", cap, 6, popup=in_popup(self), dpr=dpr)
                p.setPen(_t(self, "text"))
            p.setFont(font(11, 500, mono=True))
            p.drawText(cap, Qt.AlignmentFlag.AlignCenter, self.hint)
        if self.badge:
            b = QRectF(r.width() - 12 - 18, (r.height() - 18) / 2, max(18, fm.horizontalAdvance(self.badge) + 10), 18)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(_t(self, "violet"))
            p.drawRoundedRect(b, 9, 9)
            p.setPen(_t(self, "on_primary") if _shade(self) == "dark" else QColor("#FFFFFF"))
            p.setFont(font(11, 600))
            p.drawText(b, Qt.AlignmentFlag.AlignCenter, self.badge)
        self._focus(p, r)
        p.end()

    def _focus(self, p: QPainter, r: QRectF) -> None:
        if self.hasFocus() and self.focusPolicy() != Qt.FocusPolicy.NoFocus and getattr(self, "_keyboard_focus", False):
            p.setPen(QPen(_t(self, "iris"), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            radius = self._radius()
            p.drawRoundedRect(r.adjusted(1, 1, -1, -1), min(radius, r.height() / 2) - 1, min(radius, r.height() / 2) - 1)

    def focusInEvent(self, event):
        self._keyboard_focus = event.reason() in (Qt.FocusReason.TabFocusReason, Qt.FocusReason.BacktabFocusReason)
        super().focusInEvent(event)

    def enterEvent(self, event):
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self.update()


def button(text: str, on_click=None, primary: bool = False, link: bool = False, kind: str | None = None,
           size: str = "md", icon: str | None = None, hint: str | None = None, icon_after: bool = False) -> Button:
    """A button: raised by default, `primary` for the one lit button of a view, `link` for an Iris text button."""
    b = Button(text, kind or ("primary" if primary else "ghost" if link else "key"), size, icon, hint, icon_after)
    if on_click:
        b.clicked.connect(on_click)
    return b


class IconButton(QToolButton):
    """A small raised square with one of the design's icons (Copy, Correct, Remove...). Its tooltip names it."""

    def __init__(self, name: str, tip: str, size: int = 32, flat: bool = False, parent=None):
        super().__init__(parent)
        self.icon_name, self.flat = name, flat
        self.setToolTip(tip)
        self.setAccessibleName(tip)
        self.setFixedSize(size, size)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)
        if not flat:
            surface(self, "key", 10)

    def set_icon(self, name: str) -> None:
        self.icon_name = name
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        colour = _t(self, "text3") if not self.isEnabled() else _t(self, "text") if self.underMouse() else _t(self, "text2")
        if self.icon_name == "check":
            colour = _t(self, "ok")
        size = 16 if self.width() <= 32 else 18
        p.drawPixmap(QPointF((self.width() - size) / 2, (self.height() - size) / 2),
                     icon_pixmap(self.icon_name, colour.name(), size, self.devicePixelRatioF()))
        p.end()

    def enterEvent(self, event):
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self.update()


def icon_button(name: str, tip: str, on_click=None, size: int = 32, flat: bool = False) -> IconButton:
    b = IconButton(name, tip, size, flat)
    if on_click:
        b.clicked.connect(on_click)
    return b


# ---------------------------------------------------------------- toggles, lamps, keycaps


class Toggle(QCheckBox):
    """The design's switch: an inset track with an edge; on, a solid Iris track with a dark knob. Its text is only its
    accessible name: the row around it says what it does."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self.setAccessibleName(text)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.toggled.connect(lambda *_: self.update())

    def sizeHint(self) -> QSize:
        return QSize(44, 24)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def hitButton(self, pos) -> bool:
        return self.rect().contains(pos)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = QRectF(0, (self.height() - 24) / 2, 44, 24)
        on, enabled = self.isChecked(), self.isEnabled()
        if on and enabled:
            p.fillPath(theme.rounded(track, 12), _t(self, "primary"))
            theme.lip(p, track, 12, 1, theme.rgba(255, 255, 255, .25))
        else:
            p.fillPath(theme.rounded(track, 12), _t(self, "well"))
            theme.inner_shadow(p, track, 12, theme.SHADOWS[_shade(self)]["in2"], self.devicePixelRatioF())
            p.setPen(QPen(_t(self, "rim") if enabled else _t(self, "line"), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(track.adjusted(.5, .5, -.5, -.5), 11.5, 11.5)
        knob = QRectF(track.x() + (23 if on else 3), track.y() + 3, 18, 18).adjusted(1, 1, -1, -1)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_t(self, "knob_on") if on and enabled else _t(self, "knob_off") if enabled else _t(self, "line"))
        p.drawEllipse(knob)
        if self.hasFocus() and getattr(self, "_keyboard_focus", False):
            p.setPen(QPen(_t(self, "iris"), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(track.adjusted(-2, -2, 2, 2), 14, 14)
        p.end()

    def focusInEvent(self, event):
        self._keyboard_focus = event.reason() in (Qt.FocusReason.TabFocusReason, Qt.FocusReason.BacktabFocusReason)
        super().focusInEvent(event)


class Slider(QSlider):
    """The design's slider, kin to the switch: an inset track filled with Iris up to a round knob. A click or a drag puts
    the knob where the pointer is, in whole steps; the arrow keys move it a step."""

    KNOB = 18

    def __init__(self, minimum: int = 0, maximum: int = 100, step: int = 5, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setRange(minimum, maximum)
        self.setSingleStep(step)
        self.setPageStep(step * 2)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(24)
        self.setMinimumWidth(140)

    def sizeHint(self) -> QSize:
        return QSize(200, 24)

    def value_at(self, x: float) -> int:
        """The value a pointer at `x` chooses: the nearest whole step."""
        span, step = self.maximum() - self.minimum(), self.singleStep() or 1
        part = min(max((x - self.KNOB / 2) / max(self.width() - self.KNOB, 1), 0.0), 1.0)
        return self.minimum() + round(part * span / step) * step

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setSliderDown(True)
            self.setValue(self.value_at(event.position().x()))
            event.accept()

    def mouseMoveEvent(self, event):
        if self.isSliderDown():
            self.setValue(self.value_at(event.position().x()))
            event.accept()

    def mouseReleaseEvent(self, event):
        if self.isSliderDown():
            self.setSliderDown(False)  # sliderReleased: the value is chosen
            event.accept()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        enabled, k = self.isEnabled(), self.KNOB
        track = QRectF(k / 2, (self.height() - 6) / 2, self.width() - k, 6)
        span = (self.maximum() - self.minimum()) or 1
        x = track.x() + track.width() * (self.value() - self.minimum()) / span
        p.fillPath(theme.rounded(track, 3), _t(self, "well"))
        p.setPen(QPen(_t(self, "rim") if enabled else _t(self, "line"), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(track.adjusted(.5, .5, -.5, -.5), 2.5, 2.5)
        if x > track.x():
            p.fillPath(theme.rounded(QRectF(track.x(), track.y(), x - track.x(), track.height()), 3),
                       _t(self, "primary") if enabled else _t(self, "line"))
        knob = QRectF(x - k / 2, (self.height() - k) / 2, k, k)
        p.setPen(QPen(_t(self, "primary") if enabled else _t(self, "line"), 2))
        p.setBrush(_t(self, "knob_on") if enabled else _t(self, "well"))
        p.drawEllipse(knob.adjusted(1, 1, -1, -1))
        if self.hasFocus() and getattr(self, "_keyboard_focus", False):
            p.setPen(QPen(_t(self, "iris"), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(knob.adjusted(-2, -2, 2, 2))
        p.end()

    def focusInEvent(self, event):
        self._keyboard_focus = event.reason() in (Qt.FocusReason.TabFocusReason, Qt.FocusReason.BacktabFocusReason)
        super().focusInEvent(event)


class Lamp(QWidget):
    """An instrument lamp: an 8 px light in a 12 px inset ring. Never alone: a word always goes with it."""

    COLOURS = {"ok": "ok", "warn": "warn", "live": "live", "err": "err", "ai": "violet", "off": "text3"}

    def __init__(self, state: str = "ok", parent=None):
        super().__init__(parent)
        self.state = state
        self.setFixedSize(12, 12)

    def set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        ring = QRectF(0, 0, 12, 12)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_t(self, "well"))
        p.drawEllipse(ring)
        theme.inner_shadow(p, ring, 6, theme.SHADOWS[_shade(self)]["in2"], self.devicePixelRatioF())
        colour = _t(self, self.COLOURS.get(self.state, "text3"))
        if _shade(self) == "dark" and self.state in ("ok", "warn", "live"):
            halo = QColor(colour)
            halo.setAlpha(70)
            p.setBrush(halo)
            p.drawEllipse(QPointF(6, 6), 4.5, 4.5)
        p.setBrush(colour)
        p.drawEllipse(QPointF(6, 6), 3, 3)
        p.end()


def lamp_row(state: str, words: str, role: str = "caption", tone: str | None = "2",
             stretch: bool = True) -> tuple[QWidget, Lamp, QLabel]:
    """A lamp and its word ("● Ready"), as a row."""
    w = QWidget()
    layout = QHBoxLayout(w)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    lamp = Lamp(state)
    words_label = label(words, role, tone, wrap=False)
    layout.addWidget(lamp, 0, Qt.AlignmentFlag.AlignVCenter)
    layout.addWidget(words_label, 0, Qt.AlignmentFlag.AlignVCenter)
    if stretch:
        layout.addStretch()
    return w, lamp, words_label


KEY_SIZES = {"sm": (20, 22, 11, 6), "md": (24, 26, 12, 6), "lg": (40, 56, 14, 10)}  # height, min width, font px, radius


class KeyCap(QWidget):
    """A key drawn as a key (Ctrl, Win, Esc, 1...), so a shortcut is recognised before it is read."""

    def __init__(self, text: str, size: str = "md", parent=None):
        super().__init__(parent)
        self.key_text, self.size_name = text, size
        height, min_width, px, _ = KEY_SIZES[size]
        width = max(min_width, QFontMetrics(font(px, 500, mono=True)).horizontalAdvance(text) + (24 if size == "lg" else 14))
        self.setFixedSize(width, height)
        self.setAccessibleName(text)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        height, _, px, radius = KEY_SIZES[self.size_name]
        r = QRectF(self.rect())
        p.fillPath(theme.rounded(r, radius), _t(self, "key_face"))
        theme.lip(p, r, radius, -2, _t(self, "key_lip"))
        theme.lip(p, r, radius, 1, _t(self, "edge"))
        p.setPen(_t(self, "text"))
        p.setFont(font(px, 500, mono=True))
        p.drawText(r.adjusted(0, -1, 0, -1), Qt.AlignmentFlag.AlignCenter, self.key_text)
        p.end()


def keycap(text: str, size: str = "md") -> KeyCap:
    cap = KeyCap(text, size)
    surface(cap, "keycap", KEY_SIZES[size][3])  # its soft shadow, painted by the card or page it sits on
    cap.paint_face = True
    return cap


def keys(label_text: str, size: str = "md") -> QWidget:
    """A shortcut as keycaps: "Ctrl+Win" gives [Ctrl] + [Win]; "Ctrl C C" gives three keys side by side."""
    w = QWidget()
    layout = QHBoxLayout(w)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    parts = [k for k in label_text.replace("+", " + ").split() if k]
    for part in parts:
        if part == "+":
            plus = QLabel("+")
            plus.setProperty("tone", "3")
            plus.setFont(font(12, 500, mono=True))
            layout.addWidget(plus, 0, Qt.AlignmentFlag.AlignVCenter)
        else:
            layout.addWidget(keycap(part, size), 0, Qt.AlignmentFlag.AlignVCenter)
    return w


def inline(*parts, spacing: int = 6) -> QWidget:
    """Words and keycaps on one line: inline("Hold", keys("Ctrl+Win"), "and talk.")."""
    w = QWidget()
    layout = QHBoxLayout(w)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    for part in parts:
        if isinstance(part, str):
            layout.addWidget(label(part, wrap=False), 0, Qt.AlignmentFlag.AlignVCenter)
        else:
            layout.addWidget(part, 0, Qt.AlignmentFlag.AlignVCenter)
    layout.addStretch()
    return w


# ---------------------------------------------------------------- segmented control


class Segmented(Card):
    """A row of choices in a well; the chosen one is raised. changed(key)."""

    changed = Signal(str)

    def __init__(self, options: list[tuple[str, str]], parent=None):
        super().__init__("seg", 14, parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        self.buttons: dict[str, Button] = {}
        for key, text in options:
            b = Button(text, "segment")
            b.setCheckable(True)
            b.setAutoExclusive(True)
            b.clicked.connect(lambda _=False, k=key: self.changed.emit(k))
            self.buttons[key] = b
            layout.addWidget(b)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def set_current(self, key: str) -> None:
        if key in self.buttons:
            self.buttons[key].setChecked(True)

    def set_label(self, key: str, text: str) -> None:
        self.buttons[key].setText(text)
        self.updateGeometry()


# ---------------------------------------------------------------- the voice orb


class Orb(QWidget):
    """What Rflow is doing, at a glance: ready (a mint ring, bars at rest), live (coral, the voice's level), ai
    (violet, cleaning up), loading (an amber arc with the percentage), done (a check), off (paused).
    The raised disc is its host's surface ("orb"); the ring, the core and what's in it are drawn here."""

    # Only these move. Each frame repaints the orb and the soft surfaces under it (18-40 ms): a breathing "ready" kept
    # half a core busy for as long as Home was open, so the waiting states are drawn once, still.
    ANIMATED = ("live", "ai")

    def __init__(self, size: int = 128, level=lambda: 0.0, parent=None):
        super().__init__(parent)
        self.state, self.progress, self._phase = "ready", 0.0, 0.0
        self._level, self._levels = level, [0.0] * 5
        self.setFixedSize(size, size)
        surface(self, "orb", size / 2)
        self._timer = QTimer(self, interval=40, timeout=self._tick)

    @property
    def animating(self) -> bool:
        return self._timer.isActive()

    def set_state(self, state: str, progress: float = 0.0) -> None:
        if (state, progress) == (self.state, self.progress):
            return  # the window refreshes often: an unchanged orb isn't painted again
        self.state, self.progress = state, progress
        animated = state in self.ANIMATED
        if animated and not self._timer.isActive() and self.isVisible():
            self._timer.start()
        elif not animated:
            self._timer.stop()
        self.update()

    def showEvent(self, event):
        super().showEvent(event)
        if self.state in self.ANIMATED:
            self._timer.start()

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def _tick(self) -> None:
        self._phase += 0.04
        if self.state == "live":
            level = max(0.0, min(1.0, self._level()))
            self._levels = self._levels[1:] + [level]
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        size = self.width()
        ring_d, core_d = size * 0.8125, size * 0.656
        c = QPointF(size / 2, size / 2)
        ring = QRectF(c.x() - ring_d / 2, c.y() - ring_d / 2, ring_d, ring_d)
        core = QRectF(c.x() - core_d / 2, c.y() - core_d / 2, core_d, core_d)
        colours = {"ready": "ok", "done": "ok", "live": "live", "ai": "violet", "loading": "warn", "off": "line"}
        colour = _t(self, colours.get(self.state, "line"))
        dpr = self.devicePixelRatioF()
        if self.state == "loading":
            p.setPen(QPen(_t(self, "line"), 3))
            p.drawEllipse(ring.adjusted(1.5, 1.5, -1.5, -1.5))
            p.setPen(QPen(colour, 3, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawArc(ring.adjusted(1.5, 1.5, -1.5, -1.5), 90 * 16, -int(360 * 16 * max(0.0, min(1.0, self.progress))))
        elif self.state != "off":
            glow = QColor(colour)
            glow.setAlpha(80 if _shade(self) == "dark" else 40)
            theme.outer_shadow(p, ring, ring_d / 2, [(0, 0, 16, glow)], dpr)
            p.setPen(QPen(colour, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(ring.adjusted(1, 1, -1, -1))
        else:
            p.setPen(QPen(_t(self, "line"), 2))
            p.drawEllipse(ring.adjusted(1, 1, -1, -1))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_t(self, "well"))
        p.drawEllipse(core)
        theme.inner_shadow(p, core, core_d / 2, theme.SHADOWS[_shade(self)]["in1"], dpr)
        unit = size / 128
        if self.state == "loading":
            p.setPen(colour)
            p.setFont(font(max(12, 18 * unit), 600, tabular=True))
            p.drawText(core, Qt.AlignmentFlag.AlignCenter, f"{int(self.progress * 100)}%")
        elif self.state == "done":
            p.drawPixmap(QPointF(c.x() - 26 * unit, c.y() - 26 * unit),
                         icon_pixmap("check", colour.name(), round(52 * unit), dpr, 2.0))
        elif self.state == "ai":
            for i in range(3):
                dot = QColor(colour)
                lift = max(0.0, math.sin(self._phase * 6 - i * 0.9))
                dot.setAlphaF(0.35 + 0.65 * lift)
                p.setBrush(dot)
                p.drawEllipse(QPointF(c.x() + (i - 1) * 11 * unit, c.y()), 2.6 * unit, 2.6 * unit)
        else:
            if self.state == "live":
                heights = [4 + 30 * v for v in self._levels]
            elif self.state == "ready":
                heights = [6, 10, 14, 10, 6]
            else:
                heights = [4] * 5
            bar = QColor(colour)
            if self.state == "ready":
                bar.setAlphaF(0.6)
            p.setBrush(bar)
            for i, h in enumerate(heights):
                h *= unit
                x = c.x() + (i - 2) * 8 * unit - 2 * unit
                p.drawRoundedRect(QRectF(x, c.y() - h / 2, 4 * unit, h), 2 * unit, 2 * unit)
        p.end()


# ---------------------------------------------------------------- a feature's big Start / Stop and its light


class PowerButton(QPushButton):
    """The big Start / Stop of something that runs until it's stopped (live translation): Mint with a play mark to
    start, Coral with a stop mark while it runs. It's its page's one main button, so it glows in its colour; the glow
    reaches outside the button, so its card (a GlowCard) paints it beneath, as hosts paint the other soft depth."""

    HEIGHT, PADDING, MIN_WIDTH, RADIUS = 48, 28, 148, 14

    def __init__(self, text: str = "Start", parent=None):
        super().__init__(text, parent)
        self.running = False
        self.setAccessibleName(text)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def set_running(self, running: bool, text: str) -> None:
        """Start (Mint) or Stop (Coral)."""
        self.running = running
        self.setText(text)
        self.setAccessibleName(text)
        self._glow_changed()

    def setText(self, text: str) -> None:
        super().setText(text)
        self.updateGeometry()

    def colour(self) -> QColor:
        return _t(self, "live" if self.running else "ok")

    def sizeHint(self) -> QSize:
        width = 12 + 10 + QFontMetrics(font(16, 600)).horizontalAdvance(self.text()) + 2 * self.PADDING
        return QSize(max(self.MIN_WIDTH, width), self.HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def hitButton(self, pos) -> bool:
        return self.rect().contains(pos)

    def paint_glow(self, p: QPainter, host: QWidget) -> None:
        """The glow beneath it, painted by its card (none while it's pressed in or off)."""
        if not self.isVisibleTo(host) or not self.isEnabled() or self.isDown():
            return
        top_left = self.mapTo(host, self.rect().topLeft())
        glow = self.colour()
        glow.setAlphaF(.34 if _shade(self) == "dark" else .30)
        theme.outer_shadow(p, QRectF(top_left.x(), top_left.y(), self.width(), self.height()), self.RADIUS,
                           [(0, 4, 16, glow)], host.devicePixelRatioF())

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect())
        if self.isEnabled():
            colour = self.colour()
            face = colour.darker(110) if self.isDown() else colour.lighter(106) if self.underMouse() else colour
            p.fillPath(theme.rounded(r, self.RADIUS), face)
            theme.lip(p, r, self.RADIUS, 1, theme.rgba(255, 255, 255, .30))
            ink = _t(self, "on_primary")  # dark on Obsidian's light Mint and Coral, white on Porcelain's deep ones
        else:  # off (live translation without its key): pressed in, quiet
            p.fillPath(theme.rounded(r, self.RADIUS), _t(self, "base"))
            theme.inner_shadow(p, r, self.RADIUS, theme.SHADOWS[_shade(self)]["in2"], self.devicePixelRatioF())
            ink = _t(self, "text3")
        f = font(16, 600)
        text_w = QFontMetrics(f).horizontalAdvance(self.text())
        x = (r.width() - (12 + 10 + text_w)) / 2
        mid = r.height() / 2
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(ink)
        if self.running:  # a stop mark: a small rounded square
            p.drawRoundedRect(QRectF(x + 1, mid - 5, 10, 10), 2, 2)
        else:  # a play mark: a triangle with soft corners
            p.setPen(QPen(ink, 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            p.drawPolygon(QPolygonF([QPointF(x + 2, mid - 5.5), QPointF(x + 11, mid), QPointF(x + 2, mid + 5.5)]))
        p.setFont(f)
        p.setPen(ink)
        p.drawText(QRectF(x + 22, 0, text_w + 2, r.height()), Qt.AlignmentFlag.AlignVCenter, self.text())
        if self.hasFocus() and getattr(self, "_keyboard_focus", False):
            p.setPen(QPen(_t(self, "iris"), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(r.adjusted(1, 1, -1, -1), self.RADIUS - 1, self.RADIUS - 1)
        p.end()

    def _glow_changed(self) -> None:
        self.update()
        theme.refresh(self)  # the card repaints the glow around it

    def focusInEvent(self, event):
        self._keyboard_focus = event.reason() in (Qt.FocusReason.TabFocusReason, Qt.FocusReason.BacktabFocusReason)
        super().focusInEvent(event)

    def enterEvent(self, event):
        super().enterEvent(event)
        self.update()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self.update()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self._glow_changed()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self._glow_changed()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.EnabledChange:
            self._glow_changed()

    def moveEvent(self, event):
        super().moveEvent(event)
        host = theme.host_of(self)
        if host is not None:
            host.update()  # the glow moves with it: its card (small) is painted again whole


class GlowCard(Card):
    """A card that paints the glow of the PowerButton on it, beneath the button, along with its other surfaces."""

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        paint_hosted(self, p)
        for b in self.findChildren(PowerButton):
            if theme.host_of(b) is self:
                b.paint_glow(p, self)
        p.end()


class BlinkLamp(Lamp):
    """A lamp that blinks while something runs (live translation's red light): lit half a second, dimmed the other
    half. Its timer runs only while it is blinking and shown, so a page left open, hidden or minimised costs nothing."""

    def __init__(self, state: str = "off", size: int = 14, parent=None):
        super().__init__(state, parent)
        self.setFixedSize(size, size)
        self.blinking, self.lit = False, True
        self._timer = QTimer(self, interval=500, timeout=self._tick)

    def set_blinking(self, on: bool) -> None:
        if on == self.blinking:
            return
        self.blinking, self.lit = on, True
        if on and self.isVisible():
            self._timer.start()
        elif not on:
            self._timer.stop()
        self.update()

    def showEvent(self, event):
        super().showEvent(event)
        if self.blinking:
            self.lit = True
            self._timer.start()

    def hideEvent(self, event):
        self._timer.stop()
        super().hideEvent(event)

    def _tick(self) -> None:
        self.lit = not self.lit
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        d = self.width()
        c = QPointF(d / 2, d / 2)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(_t(self, "well"))
        p.drawEllipse(QRectF(0, 0, d, d))
        theme.inner_shadow(p, QRectF(0, 0, d, d), d / 2, theme.SHADOWS[_shade(self)]["in2"], self.devicePixelRatioF())
        colour = _t(self, self.COLOURS.get(self.state, "text3"))
        if not self.lit:
            colour.setAlphaF(.22)  # dimmed, not gone: the light is still there between blinks
        elif _shade(self) == "dark" and self.state in ("ok", "warn", "live"):
            halo = QColor(colour)
            halo.setAlpha(80)
            p.setBrush(halo)
            p.drawEllipse(c, d * .375, d * .375)
        p.setBrush(colour)
        p.drawEllipse(c, d * .25, d * .25)
        p.end()


# ---------------------------------------------------------------- meters


class Meter(QWidget):
    """The microphone's level as a few mint bars: say something, they light up."""

    HEIGHTS = (6, 9, 13, 16, 12, 8, 5)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0.0
        self.setFixedSize(len(self.HEIGHTS) * 7, 18)

    def set_value(self, value: float) -> None:
        value = max(0.0, min(1.0, value))
        if abs(value - self.value) > 0.01:
            self.value = value
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        lit = round(self.value * len(self.HEIGHTS))
        for i, h in enumerate(self.HEIGHTS):
            p.setBrush(_t(self, "ok") if i < lit else _t(self, "line"))
            p.drawRoundedRect(QRectF(i * 7, 18 - h, 4, h), 2, 2)
        p.end()


class WaveProgress(QWidget):
    """Progress as a waveform of 40 bars, lit in Amber as far as it has come (a download)."""

    def __init__(self, bars: int = 40, colour: str = "warn", parent=None):
        super().__init__(parent)
        rng = random.Random(7)
        self.heights = [rng.choice((10, 12, 14, 16, 18, 20, 22, 24, 26, 28)) for _ in range(bars)]
        self.value, self.colour = 0.0, colour
        self.setFixedSize(bars * 5 - 2, 28)

    def set_value(self, value: float) -> None:
        self.value = max(0.0, min(1.0, value))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        lit = round(self.value * len(self.heights))
        for i, h in enumerate(self.heights):
            p.setBrush(_t(self, self.colour) if i < lit else _t(self, "line"))
            p.drawRoundedRect(QRectF(i * 5, (28 - h) / 2, 3, h), 1.5, 1.5)
        p.end()


# ---------------------------------------------------------------- the brand mark


class Mark(QWidget):
    """Rflow's logo: the blue-to-violet ribbon "R" (sst/static/brand, made by scripts/make_brand.py from the owner's
    artwork). It reads on Obsidian and on Porcelain alike, so it is drawn as it is; whether Rflow is ready is said by
    the lamp beside it, not by the logo."""

    def __init__(self, size: int = 32, parent=None):
        super().__init__(parent)
        self.state = "ok"
        self.setFixedSize(size, size)
        self.setAccessibleName("Rflow")

    def set_state(self, state: str) -> None:
        self.state = state  # kept for the callers; the sidebar's lamp shows the state

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawPixmap(self.rect(), brand_mark(round(self.width() * self.devicePixelRatioF())))
        p.end()


def brand_mark(pixels: int) -> QPixmap:
    """The logo file closest above `pixels` (64, 128, 256 or 512 px), so it's scaled down a little, never up."""
    for size in BRAND_SIZES:
        if size >= pixels or size == BRAND_SIZES[-1]:
            return _brand_pixmap(size)
    return _brand_pixmap(BRAND_SIZES[-1])


BRAND_DIR = Path(__file__).parent / "static" / "brand"
BRAND_SIZES = (64, 128, 256, 512)


@lru_cache(maxsize=8)
def _brand_pixmap(size: int) -> QPixmap:
    return QPixmap(str(BRAND_DIR / f"rflow-mark-{size}.png"))


# ---------------------------------------------------------------- a toast at the bottom of the window


class Toast(QWidget):
    """A short message with an optional action (Undo), at the bottom centre of its parent; it goes by itself."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._action = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(36, 22, 28, 30)  # room for its own shadow
        layout.setSpacing(12)
        self.icon = QLabel()
        self.text = label("", wrap=False)
        self.action = button("Undo", self._act, link=True, size="sm")
        layout.addWidget(self.icon)
        layout.addWidget(self.text, 1)
        layout.addWidget(self.action)
        self._timer = QTimer(self, singleShot=True, timeout=self.hide)
        self.hide()

    def show_message(self, message: str, icon: str = "check", tone: str = "ok", action: str = "", on_action=None,
                     seconds: float = 5.0) -> None:
        self.text.setText(message)
        self.icon.setPixmap(icon_pixmap(icon, _t(self, tone).name(), 16, self.devicePixelRatioF()))
        self._action = on_action
        self.action.setText(action or "")
        self.action.setVisible(bool(action and on_action))
        self.adjustSize()
        parent = self.parentWidget()
        self.move((parent.width() - self.width()) // 2 + getattr(parent, "toast_offset", 0), parent.height() - self.height())
        self.raise_()
        self.show()
        self._timer.start(int(seconds * 1000))

    def _act(self) -> None:
        self.hide()
        if self._action:
            self._action()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        body = QRectF(self.rect()).adjusted(20, 14, -20, -22)
        paint_surface(p, "toast", body, body.height() / 2, dpr=self.devicePixelRatioF())
        p.end()


def stretch_row(*items, spacing: int = 12) -> QHBoxLayout:
    """Items on a line; None is a stretch."""
    layout = QHBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    for item in items:
        if item is None:
            layout.addStretch()
        elif isinstance(item, QHBoxLayout | QVBoxLayout):
            layout.addLayout(item)
        else:
            layout.addWidget(item)
    return layout


def setting_row(title: str, caption: str = "", *controls, tooltip: str = "") -> tuple[QWidget, QLabel, QLabel]:
    """A settings line: a title and its caption on the left, the controls on the right (sentence case, no icon tile)."""
    w = QWidget()
    layout = QHBoxLayout(w)
    layout.setContentsMargins(0, 14, 0, 14)
    layout.setSpacing(16)
    words = QVBoxLayout()
    words.setSpacing(2)
    title_label = label(title, "rowtitle")
    caption_label = label(caption, "caption", "2")
    if not caption:  # never setVisible(True) here: a widget without a parent yet would flash up as its own window
        caption_label.hide()
    words.addWidget(title_label)
    words.addWidget(caption_label)
    layout.addLayout(words, 1)
    for control in controls:
        if isinstance(control, QHBoxLayout | QVBoxLayout):
            layout.addLayout(control)
        elif control is not None:
            layout.addWidget(control, 0, Qt.AlignmentFlag.AlignVCenter)
    if tooltip:
        w.setToolTip(tooltip)
    return w, title_label, caption_label


def is_button(widget) -> bool:
    return isinstance(widget, QAbstractButton)
