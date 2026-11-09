"""Render Rflow's demo videos from its parts:

- docs/media/demo.mp4, the tour (1920x1080, about a minute: every feature), and its dictation scene as the README's GIF
  (docs/media/demo.gif);
- docs/media/demo-linkedin.mp4, the cut for LinkedIn (1080x1350, portrait). LinkedIn plays videos muted and people
  decide in two seconds, so it starts with the text being dictated, big captions say what happens, each feature gets a
  short clip, and the end card comes last.

The pill, the Translate popup, the translation bar and the window are the app's own widgets with example data, drawn off
the screen like scripts/make_site_screenshots.py: nothing appears, takes focus, presses keys or touches the clipboard.
The editor, the meeting and the captions around them are painted here, in the app's colours and fonts. The newer
features (the wake word, the Text tab, the Screen translator, Meeting mode) are stretches of their website clips
(scripts/make_feature_clips_c.py and _d.py), imported and shown as a screen. The frames are
composed with QPainter (30 fps) and encoded with PyAV (H.264, no sound). Each video is made in a process of its own (the
portrait cut draws Rflow's parts larger, so Qt starts at another scale); writing the GIF needs Pillow, which Rflow
itself doesn't use, so it's added only for this run:

  uv run --no-sync --with pillow python scripts/make_demo_video.py [demo | linkedin]
"""
import importlib
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]  # this checkout's Rflow, whichever one is installed
JOBS = ("demo", "linkedin")
JOB = sys.argv[1] if len(sys.argv) > 1 else ""
os.environ["QT_QPA_PLATFORM"] = "windows"  # the real fonts
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"  # every part drawn at 2x (3x for the portrait cut), whatever the screen's
os.environ["QT_SCALE_FACTOR"] = "3" if JOB == "linkedin" else "2"  # scaling: sharp at the size the video shows it
DATA = tempfile.mkdtemp(prefix="rflow-demo-")
os.environ["APPDATA"] = os.environ["LOCALAPPDATA"] = DATA  # the example data only: the user's own Rflow stays unread

import av  # noqa: E402
import make_site_screenshots as site  # noqa: E402
import numpy as np  # noqa: E402
from av.video.reformatter import Colorspace  # noqa: E402
from PIL import Image  # noqa: E402
from PySide6.QtCore import QPointF, QRectF, Qt  # noqa: E402
from PySide6.QtGui import (  # noqa: E402
    QColor,
    QFontMetricsF,
    QImage,
    QLinearGradient,
    QPainter,
    QPen,
    QRadialGradient,
    QTextLayout,
    QTextOption,
)
from PySide6.QtWidgets import QApplication  # noqa: E402

from sst import theme  # noqa: E402

OUT = ROOT / "docs" / "media"
FPS = 30
FADE = 0.6  # seconds between scenes: the one fades out, then the next fades in
GIF_WIDTH, GIF_STEP = 800, 0.07  # the README's GIF: 800 px wide, about 14 frames a second
T, POP = theme.TOKENS["dark"], theme.POPUP  # Obsidian; the popups are always Obsidian
BAR = 40  # the editor's title bar


@dataclass(frozen=True)
class Stage:
    """A video's frame: w x h pixels, drawn on a stage s times smaller, and where the parts go on that stage."""
    w: int
    h: int
    s: float
    editor: QRectF  # the plain "Notes" window the text goes into
    pad: float  # its text's left margin
    text_px: float
    line: float
    pill_y: float  # centres
    pill_scale: float  # the pill a little larger than life: it's what the video is about
    keys_y: float
    keys_scale: float
    caption_y: float
    caption_px: float
    caption_weight: int
    brand: bool = False  # the logo and the name at the top (the portrait cut: it starts without a title)

    @property
    def sw(self) -> float:
        return self.w / self.s

    @property
    def sh(self) -> float:
        return self.h / self.s


# The tour: a 1280x720 stage, Rflow's parts at 150% as on a laptop screen at 150% scaling. The portrait cut: a 540x675
# stage at 200%, the captions big enough for a phone, at the top where LinkedIn's own buttons don't cover them.
WIDE = Stage(1920, 1080, 1.5, QRectF(170, 36, 940, 400), 44, 22, 36, 500, 1.3, 590, 1.0, 664, 22, 500)
TALL = Stage(1080, 1350, 2.0, QRectF(20, 168, 500, 272), 24, 22, 34, 508, 1.5, 594, 1.2, 108, 36, 700, brand=True)
STAGE = WIDE

DICTATED = site.EXAMPLES[0][1]  # "Can you review the pull request before lunch ...", the website's example
JAPANESE = "来週の定例会議は木曜日の午後3時からに変更になりました。資料は水曜日までに共有フォルダにアップロードしてください。"
ENGLISH = ("Next week's regular meeting has moved to Thursday at 3 PM. Please upload the materials to the shared folder by "
           "Wednesday.")  # the website's Translate example
LIVE_LINES = [  # (start, end, words heard, translation): a meeting in Japanese, in English as people speak
    (0.45, 1.85, "では、第3四半期のロードマップから始めましょう。", "Let's start with the Q3 roadmap."),
    (2.15, 3.75, "新しい画面のデザインは、来週の月曜日までに仕上がる予定です。",
     "The designs for the new screens should be finished by next Monday."),
    (4.05, 5.35, "テストが終わったら、金曜日にリリースしましょう。", "Once testing is done, let's release on Friday."),
]
MEETING_LINES = [  # the portrait cut's meeting: a schedule change, a budget question
    (0.35, 2.3, "来週の定例会議は、木曜日の午後3時に変更します。", "We're moving next week's meeting to Thursday at 3 PM."),
    (2.6, 4.45, "第3四半期の予算について、質問はありますか？", "Any questions about the Q3 budget?"),
    (5.0, 7.2, "見積もりは金曜日までに更新します。", "I'll update the estimate by Friday."),
    (7.75, 9.75, "承認は来週の月曜日になる予定です。", "Approval should come next Monday."),
]


def use(stage: Stage) -> None:
    global STAGE
    STAGE = stage
    backdrop.cache_clear()
    scrim.cache_clear()


# ---------------------------------------------------------------- motion and drawing helpers

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def ease(x: float) -> float:
    """Ease-in-out (cubic), 0..1."""
    x = clamp(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ramp(t: float, start: float, length: float = 0.4) -> float:
    return ease((t - start) / length)


def mix(a: float, b: float, k: float) -> float:
    return a + (b - a) * k


def blend(a: QColor, b: QColor, k: float) -> QColor:
    return QColor.fromRgbF(*(mix(x, y, k) for x, y in zip(a.getRgbF(), b.getRgbF(), strict=True)))


def alpha(colour: QColor, a: float) -> QColor:
    c = QColor(colour)
    c.setAlphaF(clamp(c.alphaF() * a))
    return c


@contextmanager
def faded(p: QPainter, amount: float):
    """Paint at `amount` of the current opacity."""
    p.save()
    p.setOpacity(p.opacity() * clamp(amount))
    try:
        yield
    finally:
        p.restore()


def write(p: QPainter, rect: QRectF, text: str, px: float, weight: int = 400, colour: QColor | None = None,
          align=Qt.AlignmentFlag.AlignCenter, mono: bool = False) -> None:
    p.setFont(theme.font(px, weight, mono=mono))
    p.setPen(T["text"] if colour is None else colour)
    p.drawText(rect, align, text)


def draw_image(p: QPainter, image: QImage, x: float, y: float) -> None:
    """A grabbed widget (drawn at 2x or 3x) at x, y on the stage, at its own size."""
    dpr = image.devicePixelRatio()
    p.drawImage(QRectF(x, y, image.width() / dpr, image.height() / dpr), image)


def icon(p: QPainter, name: str, colour: QColor, x: float, y: float, size: int = 16) -> None:
    p.drawPixmap(QPointF(x, y), theme.icon_pixmap(name, colour.name(), size, 2 * STAGE.s))


def settle() -> None:
    for _ in range(3):
        QApplication.processEvents()


def voice(t: float, spans: list[tuple[float, float]]) -> float:
    """How loud someone speaking is at t, 0..1: syllables, words, short breaths; quiet outside the spans."""
    for start, end in spans:
        if start <= t <= end:
            edge = min(1.0, (t - start) / 0.12, (end - t) / 0.12)
            syllables = abs(math.sin(t * math.tau * 2.4)) ** 0.7
            words = 0.6 + 0.4 * math.sin(t * math.tau * 0.45 + 1.3)
            grain = 0.75 + 0.25 * math.sin(t * 41.0) * math.sin(t * 13.7 + 0.4)
            return clamp(0.15 + 0.85 * syllables * words * grain * edge)
    return 0.04 + 0.03 * abs(math.sin(t * 23.0))


# ---------------------------------------------------------------- the stage: backdrop, captions, keys

@lru_cache(maxsize=1)
def backdrop() -> QImage:
    """Obsidian with two soft glows, Iris and Violet: the desktop everything sits on."""
    w, h = STAGE.w, STAGE.h
    image = QImage(w, h, QImage.Format.Format_RGB32)
    p = QPainter(image)
    g = QLinearGradient(0, 0, 0, h)
    g.setColorAt(0, QColor("#171920"))
    g.setColorAt(1, QColor("#0F1115"))
    p.fillRect(image.rect(), g)
    for x, y, radius, colour, a in [(0.16, 0.10, 0.62, T["iris"], 0.10), (0.88, 0.92, 0.55, T["violet"], 0.07)]:
        glow = QRadialGradient(QPointF(x * w, y * h), radius * max(w, h))
        glow.setColorAt(0, alpha(colour, a))
        glow.setColorAt(1, alpha(colour, 0))
        p.fillRect(image.rect(), glow)
    p.end()
    return image


@lru_cache(maxsize=1)
def scrim() -> QImage:
    """The backdrop's bottom band, fading in from transparent: the caption stays clear of what's above it."""
    top = round(596 * STAGE.s)
    band = backdrop().copy(0, top, STAGE.w, STAGE.h - top).convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    p = QPainter(band)
    p.setCompositionMode(QPainter.CompositionMode.CompositionMode_DestinationIn)
    g = QLinearGradient(0, 0, 0, 44 * STAGE.s)
    g.setColorAt(0, QColor(0, 0, 0, 0))
    g.setColorAt(1, QColor(0, 0, 0, 255))
    p.fillRect(band.rect(), g)
    p.end()
    return band


@lru_cache(maxsize=1)
def logo() -> QImage:
    return QImage(str(ROOT / "sst" / "static" / "brand" / "rflow-mark-512.png"))


def draw_logo(p: QPainter, cx: float, cy: float, size: float) -> None:
    """The mark, in a soft Iris halo."""
    halo = QRadialGradient(QPointF(cx, cy), size * 1.6)
    halo.setColorAt(0, alpha(T["iris"], 0.22))
    halo.setColorAt(1, alpha(T["iris"], 0))
    p.fillRect(QRectF(cx - size * 1.6, cy - size * 1.6, size * 3.2, size * 3.2), halo)
    p.drawImage(QRectF(cx - size / 2, cy - size / 2, size, size), logo())


def draw_brand(p: QPainter) -> None:
    """The mark and the name, small, at the top of the portrait cut."""
    width = 26 + 9 + QFontMetricsF(theme.font(19, 600)).horizontalAdvance("Rflow")
    x = STAGE.sw / 2 - width / 2
    p.drawImage(QRectF(x, 23, 26, 26), logo())
    write(p, QRectF(x + 35, 20, 120, 32), "Rflow", 19, 600, T["text2"], Qt.AlignmentFlag.AlignVCenter)


def draw_caption(p: QPainter, captions: list[tuple[float, str]], t: float) -> None:
    """The scene's captions, (from, text) each: the last one begun, fading in; the one before fading out. The portrait
    cut's are big and bold, wrapped on two lines at most."""
    shown = [i for i, (start, _) in enumerate(captions) if start <= t]
    if not shown:
        return
    i, tall = shown[-1], STAGE is TALL
    rect = QRectF(28, STAGE.caption_y - 60, STAGE.sw - 56, 120) if tall else QRectF(0, STAGE.caption_y - 24, 1280, 48)
    flags = Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap
    start, text = captions[i]
    if i > 0 and t < start + 0.15:  # the one before goes first
        with faded(p, 1 - ramp(t, start, 0.15)):
            write(p, rect, captions[i - 1][1], STAGE.caption_px, STAGE.caption_weight, T["text"], flags)
    k = ramp(t, start + (0.12 if i > 0 else 0), 0.3 if tall else 0.45)
    with faded(p, k):
        write(p, rect.translated(0, 6 * (1 - k)), text, STAGE.caption_px, STAGE.caption_weight, T["text"], flags)


def draw_keys(p: QPainter, keys: list, amount: float) -> None:
    """Keycaps in a row, shown `amount` (0..1): (label, how far it's pressed 0..1), or a separator such as "+"."""
    if amount <= 0:
        return
    caps, seps = QFontMetricsF(theme.font(17, 600)), QFontMetricsF(theme.font(19, 500))
    items = [(k, None, seps.horizontalAdvance(k) + 2) if isinstance(k, str)
             else (k[0], k[1], max(58.0, caps.horizontalAdvance(k[0]) + 36)) for k in keys]
    gap = 10
    x = -(sum(w for *_, w in items) + gap * (len(items) - 1)) / 2
    with faded(p, amount):
        p.translate(STAGE.sw / 2, STAGE.keys_y + 8 * (1 - amount))
        p.scale(STAGE.keys_scale, STAGE.keys_scale)
        for label, pressed, w in items:
            if pressed is None:
                write(p, QRectF(x, -22, w, 40), label, 19, 500, T["text3"])
            else:
                keycap(p, QRectF(x, -22, w, 42), label, pressed)
            x += w + gap


def keycap(p: QPainter, rect: QRectF, label: str, pressed: float) -> None:
    """A key in the design's keycap look; pressed, it sinks a little and lights up in Iris."""
    rect = rect.translated(0, 2.5 * pressed)
    dpr = STAGE.s * STAGE.keys_scale
    theme.paint_surface(p, "keycap", rect, 9, popup=True, dpr=dpr)
    if pressed > 0:
        with faded(p, pressed):
            theme.outer_shadow(p, rect, 9, [(0, 0, 18, alpha(T["iris"], 0.45))], dpr)
            p.fillPath(theme.rounded(rect, 9), QColor("#262B3A"))
            p.setPen(QPen(alpha(T["iris"], 0.9), 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(rect.adjusted(0.75, 0.75, -0.75, -0.75), 8.25, 8.25)
    write(p, rect.adjusted(0, -1, 0, -1), label, 17, 600, blend(T["text"], T["iris"], pressed))


def chip(p: QPainter, rect: QRectF, name: str, colour: QColor, text: str) -> None:
    """A small floating capsule in the popups' look, an icon and a few words: what Rflow does meanwhile."""
    theme.paint_surface(p, "float", rect, rect.height() / 2, popup=True, dpr=STAGE.s)
    icon(p, name, colour, rect.left() + 14, rect.center().y() - 8)
    write(p, QRectF(rect.left() + 38, rect.top(), rect.width() - 40, rect.height()), text, 13, 500, POP["text"],
          Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)


# ---------------------------------------------------------------- the editor ("Notes", a plain text window)

@lru_cache(maxsize=256)
def layout(text: str, px: float, width: float, line_height: float) -> QTextLayout:
    lay = QTextLayout(text, theme.font(px))
    option = QTextOption()
    option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
    lay.setTextOption(option)
    lay.beginLayout()
    y = 0.0
    while True:
        line = lay.createLine()
        if not line.isValid():
            break
        line.setLineWidth(width)
        line.setPosition(QPointF(0, y))
        y += line_height
    lay.endLayout()
    return lay


def text_layout(text: str) -> QTextLayout:
    return layout(text, STAGE.text_px, STAGE.editor.width() - 2 * STAGE.pad, STAGE.line)


def text_at() -> QPointF:
    return QPointF(STAGE.editor.left() + STAGE.pad, STAGE.editor.top() + BAR + 32)


def _x(line, position: int) -> float:
    x = line.cursorToX(position)
    return x[0] if isinstance(x, tuple) else x


def caret_at(text: str, position: int) -> QPointF:
    """The top of the caret before `position`, on the stage."""
    lay, at = text_layout(text), text_at()
    if not text or not lay.lineCount():
        return at
    line = lay.lineForTextPosition(position)
    if not line.isValid():
        line = lay.lineAt(lay.lineCount() - 1)
    return QPointF(at.x() + _x(line, position), at.y() + line.y())


def draw_editor(p: QPainter) -> None:
    r = STAGE.editor
    theme.outer_shadow(p, r, 12, [(0, 18, 44, theme.rgba(0, 0, 0, .55)), (0, 2, 6, theme.rgba(0, 0, 0, .30))], STAGE.s)
    p.fillPath(theme.rounded(r, 12), QColor("#1C1F25"))
    p.save()
    p.setClipPath(theme.rounded(r, 12))
    p.fillRect(QRectF(r.left(), r.top(), r.width(), BAR), QColor("#16181D"))
    p.fillRect(QRectF(r.left(), r.top() + BAR - 1, r.width(), 1), theme.rgba(255, 255, 255, .06))
    p.restore()
    p.setPen(QPen(theme.rgba(255, 255, 255, .08), 1))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), 12, 12)
    icon(p, "reading", T["text3"], r.left() + 16, r.top() + 12)
    write(p, QRectF(r.left() + 42, r.top(), 200, BAR), "Notes", 13, 500, T["text2"], Qt.AlignmentFlag.AlignVCenter)
    cy = r.top() + BAR / 2  # minimise, maximise, close: plain lines
    p.setPen(QPen(T["text3"], 1.3))
    x = r.right() - 104
    p.drawLine(QPointF(x - 6, cy), QPointF(x + 6, cy))
    p.drawRect(QRectF(x + 40 - 5.5, cy - 5.5, 11, 11))
    p.drawLine(QPointF(x + 80 - 5.5, cy - 5.5), QPointF(x + 80 + 5.5, cy + 5.5))
    p.drawLine(QPointF(x + 80 + 5.5, cy - 5.5), QPointF(x + 80 - 5.5, cy + 5.5))


def draw_text(p: QPainter, text: str, selected: int = 0, tint: float = 0.0) -> None:
    """The document's text, its first `selected` characters highlighted as a selection; `tint` (0..1) colours it Violet,
    the AI's colour, as it settles after a transform."""
    lay, at = text_layout(text), text_at()
    for i in range(lay.lineCount()):
        line = lay.lineAt(i)
        start, end = line.textStart(), min(selected, line.textStart() + line.textLength())
        if end > start:
            x0, x1 = _x(line, start), _x(line, end)
            p.fillRect(QRectF(at.x() + x0 - 1, at.y() + line.y() - 3, x1 - x0 + 2, STAGE.line - 2), T["selection"])
    p.setPen(blend(T["text"], T["violet"], tint))
    lay.draw(p, at)


def draw_caret(p: QPainter, text: str, position: int, t: float, steady: bool = False) -> None:
    """A blinking Iris caret (steady while text is being typed)."""
    on = 1.0 if steady else clamp(abs(((t % 1.06) / 1.06) * 2 - 1) * 6 - 2.5)  # about 0.5 s on, 0.5 s off, soft edges
    if on > 0:
        at = caret_at(text, position)
        p.fillRect(QRectF(at.x(), at.y() - 1, 2, STAGE.text_px + 8), alpha(T["iris"], on))


def words_typed(text: str, k: float) -> str:
    """The first k (0..1) of the text, whole words at a time."""
    ends = [i for i, ch in enumerate(text) if ch == " "] + [len(text)]
    return text[:ends[round(k * (len(ends) - 1))]] if k > 0 else ""


# ---------------------------------------------------------------- Rflow's own parts, grabbed off the screen

class PillTrack:
    """The real pill (sst.app.Pill) through a scene: shown, its states one after another, hidden."""

    def __init__(self, show: float, hide: float, states: list[tuple[float, str, bool, str]],
                 speech: list[tuple[float, float]]):
        from sst.app import Pill
        self.pill = Pill(level=lambda: 0.0)
        self.show, self.hide, self.states, self.speech = show, hide, states, speech

    def image(self, i: int, t: float) -> QImage:
        start, state, ai, message = self.states[i]
        pill = self.pill
        pill.state, pill.message, pill.ai, pill._phase = state, message, ai, t
        if state == "recording":  # the pill paints its timer from the clock, and the last BARS levels, one a frame
            pill._started = time.monotonic() - max(0.0, t - start)
            pill._levels.clear()
            pill._levels.extend(voice(t - (pill.BARS - 1 - k) / 30, self.speech) for k in range(pill.BARS))
        pill.resize(pill._width() + 2 * pill.MARGIN, pill.HEIGHT + 2 * pill.MARGIN)
        return pill.grab().toImage()

    def draw(self, p: QPainter, t: float) -> None:
        shown = ramp(t, self.show, 0.25) * (1 - ramp(t, self.hide, 0.35))
        if shown <= 0:
            return
        i = max(k for k, s in enumerate(self.states) if s[0] <= t) if t >= self.states[0][0] else 0
        p.save()
        p.translate(STAGE.sw / 2, STAGE.pill_y)
        scale = STAGE.pill_scale * mix(0.9, 1.0, ramp(t, self.show, 0.3))
        p.scale(scale, scale)
        with faded(p, shown):
            change = ramp(t, self.states[i][0], 0.2) if i > 0 else 1.0
            for k, amount in ((i - 1, 1 - change), (i, change)):
                if k >= 0 and amount > 0:
                    image = self.image(k, t)
                    dpr = image.devicePixelRatio()
                    with faded(p, amount):
                        draw_image(p, image, -image.width() / dpr / 2, -image.height() / dpr / 2)
        p.restore()


@lru_cache(maxsize=1)
def translate_popups() -> tuple[list[QImage], QImage]:
    """The real Translate popup: translating (three steps of its dots), then the translation."""
    from sst.translate import Translation
    from sst.translateui import TranslatePopup
    popup = TranslatePopup()
    popup._choices = ["English", "Spanish", "Japanese"]
    popup.route.setText("Japanese →")
    popup.original.setText(popup.original.fontMetrics().elidedText(JAPANESE, Qt.TextElideMode.ElideRight, popup.INNER))
    popup.busy("English")
    popup.dots.stop()  # its dots are set here, frame by frame
    busy = []
    for dots in (1, 2, 3):
        popup.waiting.setText("Translating into English" + "." * dots)
        settle()
        busy.append(popup.grab().toImage())
    popup.show_result(Translation(ENGLISH, "English", JAPANESE, 0.9))
    settle()
    return busy, popup.grab().toImage()


def line_events(start: float, end: float, heard: str, said: str) -> list:
    """A line of a meeting as the translation bar gets it: the words heard as they come, the translation close behind,
    then the finished line."""
    from sst.live.contracts import Kind, LiveEvent
    events = []
    for k in range(1, 9):
        events.append((start + (end - start) * 0.7 * k / 8, LiveEvent(Kind.SOURCE, heard[:math.ceil(len(heard) * k / 8)])))
    words = said.split()
    for k in range(1, len(words) + 1):
        events.append((start + 0.35 + (end - start - 0.4) * k / len(words), LiveEvent(Kind.TRANSLATION, " ".join(words[:k]))))
    events.append((end, LiveEvent(Kind.LINE, said, source=heard)))
    return events


class Feed:
    """The real translation bar (sst.live.captions.CaptionBar), fed a meeting as it goes: (time, event), the event a
    LiveEvent or a change to the bar (its speaker button)."""

    def __init__(self, events: list, size: tuple[int, int]):
        self.events, self.size = sorted(events, key=lambda e: e[0]), size
        self.bar, self.last, self.applied, self.image = None, 0.0, 0, None

    def at(self, t: float) -> QImage:
        from sst.live.captions import CaptionBar
        from sst.live.contracts import Kind, LiveConfig, LiveEvent
        if self.bar is None or t < self.last:
            self.bar = CaptionBar(LiveConfig(target="en", source="computer"))
            self.bar.resize(*self.size)
            self.bar.show_event(LiveEvent(Kind.STATUS, "Listening"))
            self.applied, self.image = 0, None
        self.last = t
        while self.applied < len(self.events) and self.events[self.applied][0] <= t:
            event = self.events[self.applied][1]
            if callable(event):
                event(self.bar)
            else:
                self.bar.show_event(event)
            self.applied, self.image = self.applied + 1, None
        if self.image is None:
            settle()
            self.image = self.bar.grab().toImage()
        return self.image


def draw_call(p: QPainter, r: QRectF, title: str, tints: tuple, rows: int, who: int, level: float,
              you: int = -1) -> None:
    """An online meeting, abstract: people as soft shapes in tiles, the one speaking (`who`) ringed in Mint."""
    theme.outer_shadow(p, r, 14, [(0, 18, 44, theme.rgba(0, 0, 0, .5))], STAGE.s)
    p.fillPath(theme.rounded(r, 14), QColor("#14161B"))
    p.setPen(QPen(theme.rgba(255, 255, 255, .07), 1))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), 14, 14)
    head = 40 if rows > 1 else 34
    write(p, QRectF(r.left() + 18, r.top(), 300, head), title, 13, 500, T["text2"], Qt.AlignmentFlag.AlignVCenter)
    write(p, QRectF(r.right() - 220, r.top(), 202, head), f"{len(tints)} people", 13, 400, T["text3"],
          Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
    cols, gap, pad = math.ceil(len(tints) / rows), 12 if rows > 1 else 9, 16 if rows > 1 else 12
    tw = (r.width() - 2 * pad - gap * (cols - 1)) / cols
    th = (r.height() - head - pad - gap * (rows - 1)) / rows
    for i, tint in enumerate(tints):
        tile = QRectF(r.left() + pad + (i % cols) * (tw + gap), r.top() + head + (i // cols) * (th + gap), tw, th)
        colour = T[tint]
        g = QLinearGradient(tile.topLeft(), tile.bottomRight())
        g.setColorAt(0, blend(QColor("#1D2029"), colour, 0.10))
        g.setColorAt(1, QColor("#171920"))
        p.fillPath(theme.rounded(tile, 12), g)
        p.save()
        p.setClipPath(theme.rounded(tile, 12))
        c, u = tile.center(), min(th, tw * 1.2)
        shape = QRadialGradient(QPointF(c.x(), c.y() - 0.04 * u), 0.58 * u)
        shape.setColorAt(0, alpha(blend(QColor("#2A2E3A"), colour, 0.32), 0.85))
        shape.setColorAt(1, alpha(blend(QColor("#2A2E3A"), colour, 0.18), 0.55))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(shape)
        p.drawEllipse(QPointF(c.x(), c.y() - 0.083 * u), 0.116 * u, 0.116 * u)
        p.drawEllipse(QRectF(c.x() - 0.256 * u, c.y() + 0.083 * u, 0.512 * u, 0.455 * u))
        p.restore()
        if i == you:
            write(p, QRectF(tile.left() + 10, tile.top() + 8, 60, 22), "You", 12, 600, T["text2"],
                  Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        if i == who:
            p.setPen(QPen(alpha(T["ok"], 0.55 + 0.45 * level), 2.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(tile.adjusted(1.25, 1.25, -1.25, -1.25), 11, 11)


# ---------------------------------------------------------------- the scenes

class Scene:
    duration = 0.0
    captions: list[tuple[float, str]] = []  # (from, text), in the scene's seconds
    chrome = None  # what stays from one scene to the next (the editor), if anything: drawn beneath draw()
    brand = True  # the portrait cut's logo at the top (not on the end card)

    def __init__(self, **changes):
        """The scene's timing as in the tour, or with `changes` (its attributes) for the portrait cut."""
        for name, value in changes.items():
            setattr(self, name, value)

    def draw(self, p: QPainter, t: float) -> None:
        raise NotImplementedError


class Title(Scene):
    duration = 2.5
    brand = False

    def draw(self, p, t):
        k = ramp(t, 0.0, 0.7)
        with faded(p, k):
            draw_logo(p, 640, 228, mix(132, 144, k))
        for start, y, text, px, weight, colour in [(0.25, 372, "Rflow", 64, 600, T["text"]),
                                                   (0.5, 440, "Speak anywhere. It types.", 30, 500, T["text"]),
                                                   (0.75, 492, "Dictation, translation and meeting notes for Windows", 19, 500,
                                                    T["text2"])]:
            k = ramp(t, start, 0.55)
            with faded(p, k):
                write(p, QRectF(0, y - 40 + 10 * (1 - k), 1280, 80), text, px, weight, colour)


class Dictation(Scene):
    duration = 9.0
    captions = [(0.15, "Hold Ctrl+Win and speak")]
    text = DICTATED
    KEYS, DOWN, UP, TYPED, HIDE = 0.85, 1.2, 5.0, 6.4, 7.5  # the keys shown, pressed, let go; the text typed; pill gone
    SPEECH = (1.45, 4.75)
    chrome = staticmethod(draw_editor)

    def __init__(self, **changes):
        super().__init__(**changes)
        self.pill = PillTrack(self.DOWN + 0.05, self.HIDE, [(self.DOWN + 0.05, "recording", False, ""),
                                                            (self.UP, "transcribing", True, ""),
                                                            (self.TYPED, "typed", False, "")], [self.SPEECH])

    def draw(self, p, t):
        typing = ramp(t, self.TYPED - 0.05, 0.7) if t >= self.TYPED - 0.05 else 0.0
        text = words_typed(self.text, typing)
        draw_text(p, text)
        draw_caret(p, text, len(text), t, steady=0 < typing < 1)
        pressed = ramp(t, self.DOWN, 0.12) * (1 - ramp(t, self.UP, 0.12))
        draw_keys(p, [("Ctrl", pressed), "+", ("Win", pressed)], ramp(t, self.KEYS, 0.3) * (1 - ramp(t, self.UP + 0.3, 0.35)))
        self.pill.draw(p, t)


class Transform(Scene):
    duration = 7.0
    captions = [(0.15, "Say “make it concise”")]
    SELECT, KEYS, DOWN, UP, DONE = 0.3, 1.05, 1.35, 3.1, 4.35  # selected; the keys pressed, let go; the text replaced
    SPEECH = (1.6, 2.75)
    chrome = staticmethod(draw_editor)

    def __init__(self, **changes):
        super().__init__(**changes)
        from sst.window import TRANSFORM_EXAMPLES
        self.before, self.after = TRANSFORM_EXAMPLES["concise"][0]  # a pair the app's own page shows, checked by its tests
        self.pill = PillTrack(self.DOWN + 0.05, self.DONE + 1.55, [
            (self.DOWN + 0.05, "recording", False, ""), (self.UP, "transforming", False, ""),
            (self.DONE, "transformed", False, "Transformed: Concise")], [self.SPEECH])

    def draw(self, p, t):
        gone, came = ramp(t, self.DONE - 0.15, 0.2), ramp(t, self.DONE + 0.05, 0.3)  # one text out, then the other in
        if gone < 1:
            with faded(p, 1 - gone):
                draw_text(p, self.before, round(len(self.before) * ramp(t, self.SELECT, 0.8)))
                if t < self.SELECT:
                    draw_caret(p, self.before, len(self.before), t)
        if came > 0:
            with faded(p, came):
                draw_text(p, self.after, tint=1 - ramp(t, self.DONE + 0.5, 1.0))
                if t > self.DONE + 0.5:
                    draw_caret(p, self.after, len(self.after), t - self.DONE)
        pressed = ramp(t, self.DOWN, 0.12) * (1 - ramp(t, self.UP, 0.12))
        draw_keys(p, [("Ctrl", pressed), "+", ("Win", pressed)], ramp(t, self.KEYS, 0.3) * (1 - ramp(t, self.UP + 0.3, 0.35)))
        self.pill.draw(p, t)


class Translate(Scene):
    duration = 6.0
    captions = [(0.15, "Select text and press Ctrl+C twice to translate")]
    SELECT, KEYS, CTRL, FIRST, SECOND, KEYS_OFF = 0.3, 0.95, 1.15, 1.38, 1.7, 2.55  # selected; the keys
    OPEN, DONE = 1.95, 2.75  # the popup opens, translating; the translation is there
    chrome = staticmethod(draw_editor)

    def draw(self, p, t):
        draw_text(p, JAPANESE, round(len(JAPANESE) * ramp(t, self.SELECT, 0.7)))
        if t < self.SELECT:
            draw_caret(p, JAPANESE, len(JAPANESE), t)
        ctrl = ramp(t, self.CTRL, 0.1) * (1 - ramp(t, self.SECOND + 0.55, 0.12))
        first = ramp(t, self.FIRST, 0.08) * (1 - ramp(t, self.FIRST + 0.14, 0.08))
        second = ramp(t, self.SECOND, 0.08) * (1 - ramp(t, self.SECOND + 0.14, 0.08))
        draw_keys(p, [("Ctrl", ctrl), "+", ("C", first), ",", ("C", second)],
                  ramp(t, self.KEYS, 0.3) * (1 - ramp(t, self.KEYS_OFF, 0.35)))
        shown = ramp(t, self.OPEN, 0.3)
        if shown > 0:
            busy, result = translate_popups()
            end = caret_at(JAPANESE, len(JAPANESE))
            right = STAGE.sw - 460 - 24 if STAGE is WIDE else (STAGE.sw - 460) / 2  # where the app puts it: by the
            x = min(end.x() + 12, right) - 24  # selection, on the screen (centred in the portrait cut)
            y = end.y() + STAGE.line + 10 - 24 + 8 * (1 - shown)
            done = ramp(t, self.DONE, 0.18)
            with faded(p, shown):
                if done < 1:
                    with faded(p, 1 - done):
                        draw_image(p, busy[int(max(0.0, t - self.OPEN) / 0.35) % 3], x, y)
                if done > 0:
                    with faded(p, done):
                        draw_image(p, result, x, y)


class Live(Scene):
    duration = 6.0
    captions = [(0.15, "Live translation: meetings translated as people speak")]
    CALL = QRectF(150, 24, 980, 552)

    def __init__(self, **changes):
        super().__init__(**changes)
        self.feed = Feed([e for line in LIVE_LINES for e in line_events(*line)], (760, 240))

    def draw(self, p, t):
        speaking = next((i for i, (start, end, *_) in enumerate(LIVE_LINES) if start - 0.1 <= t <= end), None)
        who = {0: 0, 1: 1, 2: 0}.get(speaking, -1)  # tiles the translation bar doesn't cover
        draw_call(p, self.CALL, "Weekly sync", ("iris", "violet", "ok", "warn"), 2, who,
                  voice(t, [(s, e) for s, e, *_ in LIVE_LINES]))
        image = self.feed.at(t)
        dpr = image.devicePixelRatio()
        draw_image(p, image, 640 - image.width() / dpr / 2, 596 - image.height() / dpr)


class Meeting(Scene):
    """The portrait cut's live translation: a meeting in Japanese, the bar translating it as people speak; then the
    translation read aloud while the meeting's sound is lowered, like an interpreter's; the meeting turned right down,
    still translated; and the bar left out of the screen share."""
    duration = 12.2
    captions = [(-1, "And in meetings…"), (1.5, "Live translation, while people speak"),
                (4.5, "Read aloud, like an interpreter"), (7.5, "Even at low volume"), (9.9, "Hidden from screen sharing")]
    CALL, BAR_AT, BAR_SIZE = QRectF(20, 168, 500, 214), QPointF(20, 440), (500, 210)
    TINTS, YOU = ("iris", "violet", "ok"), 2
    SPEAK_ON, SPEAK_OFF, READ = 4.5, 7.4, (4.75, 6.65)  # the bar's speaker button; the voice reading the second line
    LOWER = 7.6  # the meeting turned down to 5%
    SPLIT = 9.95  # the screen share beside the screen
    FRAME = QRectF(20, 168, 500, 482)  # the meeting and the bar: what the screen shows

    def __init__(self, **changes):
        super().__init__(**changes)

        def speaking(on: bool):
            return lambda bar: bar.set_speaking(on)
        events = [e for line in MEETING_LINES for e in line_events(*line)]
        self.feed = Feed(events + [(self.SPEAK_ON, speaking(True)), (self.SPEAK_OFF, speaking(False))], self.BAR_SIZE)

    def volume(self, t: float) -> float:
        """The meeting app's volume: lowered to 30% while the voice reads, back after; then turned down to 5%."""
        start, end = self.READ
        ducked = mix(1.0, 0.3, ramp(t, start, 0.15) * (1 - ramp(t, end + 0.2, 0.6)))
        return mix(ducked, 0.05, ramp(t, self.LOWER, 0.45))

    def screen(self, p: QPainter, t: float, bar: bool = True) -> None:
        """The meeting, the bar over it, and what Rflow does meanwhile (chips between them)."""
        spans = [(s, e) for s, e, *_ in MEETING_LINES]
        speaking = next((i for i, (s, e) in enumerate(spans) if s - 0.1 <= t <= e), None)
        draw_call(p, self.CALL, "Weekly sync", self.TINTS, 1, -1 if speaking is None else speaking % 2,
                  voice(t, spans), you=self.YOU)
        reading = ramp(t, self.READ[0] - 0.1, 0.25) * (1 - ramp(t, self.READ[1], 0.25))
        shown = ramp(t, self.SPEAK_ON, 0.3) * (1 - ramp(t, self.SPLIT - 0.3, 0.3))
        y = self.CALL.bottom() + 12
        if reading > 0:
            with faded(p, reading):
                r = QRectF(20, y, 214, 36)
                chip(p, r, "speaker", POP["iris"], "Reading it aloud")
                for k in range(4):  # the voice, as it speaks
                    h = 4 + 12 * abs(math.sin(t * 7.3 + k * 1.1)) * (0.6 + 0.4 * math.sin(t * 2.1 + k))
                    p.fillRect(QRectF(r.right() - 46 + k * 7, r.center().y() - h / 2, 3.5, h), POP["iris"])
        if shown > 0:
            with faded(p, shown):
                level = self.volume(t)
                r = QRectF(244, y, 276, 36)
                chip(p, r, "speaker" if level > 0.2 else "speaker-off", POP["text2"], "Meeting")
                track = QRectF(r.left() + 108, r.center().y() - 3, 100, 6)
                p.fillPath(theme.rounded(track, 3), POP["well"])
                p.fillPath(theme.rounded(QRectF(track.left(), track.top(), max(6.0, track.width() * level), 6), 3),
                           blend(POP["iris"], POP["warn"], ramp(t, self.LOWER, 0.45)))
                write(p, QRectF(track.right() + 8, r.top(), 52, r.height()), f"{round(level * 100)}%", 13, 500,
                      POP["text"], Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, mono=True)
        if bar:
            draw_image(p, self.feed.at(t), self.BAR_AT.x(), self.BAR_AT.y())

    def draw(self, p, t):
        k = ramp(t, self.SPLIT, 0.7)
        if k <= 0:
            self.screen(p, t)
            return
        f, scale = self.FRAME, mix(1.0, 0.47, k)
        w, h = f.width() * 0.47, f.height() * 0.47
        top = (f.top() + f.bottom()) / 2 - h / 2 + 20
        for x, bar, amount, words, name in ((20, True, 1.0, "On your screen", "eye"),
                                            (f.right() - w, False, ramp(t, self.SPLIT + 0.45, 0.5), "In the screen share",
                                             "eye-off")):
            if amount <= 0:
                continue
            with faded(p, amount):
                p.save()
                p.translate(mix(f.left(), x, k), mix(f.top(), top, k))
                p.scale(scale, scale)
                p.translate(-f.left(), -f.top())
                panel = f.adjusted(-16, -16, 16, 16)
                with faded(p, k):
                    p.fillPath(theme.rounded(panel, 22), QColor("#101216"))
                    p.setPen(QPen(theme.rgba(255, 255, 255, .10), 2))
                    p.setBrush(Qt.BrushStyle.NoBrush)
                    p.drawRoundedRect(panel, 22, 22)
                self.screen(p, t, bar)
                if not bar:  # where the bar is on the screen: nothing in the share
                    hole = QRectF(self.BAR_AT.x(), self.BAR_AT.y(), *self.BAR_SIZE).adjusted(24, 24, -24, -24)
                    p.setPen(QPen(alpha(T["text3"], 0.7), 3, Qt.PenStyle.DashLine))
                    p.drawRoundedRect(hole, 14, 14)
                    p.drawPixmap(QPointF(hole.center().x() - 24, hole.center().y() - 24),
                                 theme.icon_pixmap("eye-off", T["text3"].name(), 48, 2 * STAGE.s))
                p.restore()
                with faded(p, k):  # what the panel is, over it
                    left = x + w / 2 - (24 + QFontMetricsF(theme.font(14, 500)).horizontalAdvance(words)) / 2
                    icon(p, name, T["text2"], left, top - 38)
                    write(p, QRectF(left + 24, top - 44, w, 28), words, 14, 500, T["text2"],
                          Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)


class WindowShot(Scene):
    """A real window, grabbed, with a slow zoom from `start` to `end`: each (x, y, scale, at_x, at_y), the window's point
    x, y shown at the stage's at_x, at_y. Clipped to `view` if given, else whole with its shadow."""
    start = end = (0, 0, 1, 0, 0)
    view: QRectF | None = None
    MOVE = (0.4, 4.2)

    def image(self, t: float) -> QImage:
        raise NotImplementedError

    def draw(self, p, t):
        image = self.image(t)
        dpr = image.devicePixelRatio()
        rect = QRectF(0, 0, image.width() / dpr, image.height() / dpr)
        k = ramp(t, *self.MOVE)
        fx, fy, scale, cx, cy = (mix(a, b, k) for a, b in zip(self.start, self.end, strict=True))
        p.save()
        if self.view is not None:
            theme.outer_shadow(p, self.view, 16, [(0, 18, 44, theme.rgba(0, 0, 0, .55))], STAGE.s)
            p.setClipPath(theme.rounded(self.view, 16))
        p.translate(cx, cy)
        p.scale(scale, scale)
        p.translate(-fx, -fy)
        if self.view is None:
            theme.outer_shadow(p, rect, 14, [(0, 18, 44, theme.rgba(0, 0, 0, .55)), (0, 2, 6, theme.rgba(0, 0, 0, .3))],
                               STAGE.s)
            p.setClipPath(theme.rounded(rect, 14))
        p.drawImage(rect, image)
        p.restore()
        p.setPen(QPen(theme.rgba(255, 255, 255, .08), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        if self.view is not None:
            p.drawRoundedRect(self.view.adjusted(0.5, 0.5, -0.5, -0.5), 16, 16)
            return
        p.save()
        p.translate(cx, cy)
        p.scale(scale, scale)
        p.translate(-fx, -fy)
        p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 14, 14)
        p.restore()
        band = scrim()
        p.drawImage(QRectF(0, STAGE.sh - band.height() / STAGE.s, STAGE.sw, band.height() / STAGE.s), band)


class Clip(Scene):
    """A stretch of a feature clip (scripts/make_feature_clips_c.py and _d.py: their CLIPS, name -> (seconds, frame)),
    from `start`, shown as a screen: in the tour above the caption, in the portrait cut under it. The clip draws itself
    (its own stage, its own widgets): this only places its frames. `crop`, in fractions of the clip's frame, shows a part
    of it: the portrait cut's narrow screen would make the whole 16:9 frame too small to read on a phone."""
    start = 0.0
    crop = QRectF(0, 0, 1, 1)

    def __init__(self, script: str, name: str, **changes):
        super().__init__(**changes)
        self.script, self.name = script, name

    def draw(self, p, t):
        _, frame = clips(self.script)[self.name]
        image = frame(self.start + t)
        c = self.crop
        source = QRectF(c.x() * image.width(), c.y() * image.height(), c.width() * image.width(), c.height() * image.height())
        width = 1050 if STAGE is WIDE else STAGE.sw - 40  # in the tour, 82% of the stage: the caption stays clear
        rect = QRectF(0, 0, width, width * source.height() / source.width())
        rect.moveCenter(QPointF(STAGE.sw / 2, 311 if STAGE is WIDE else STAGE.sh / 2 + 40))
        theme.outer_shadow(p, rect, 16, [(0, 18, 44, theme.rgba(0, 0, 0, .55))], STAGE.s)
        p.save()
        p.setClipPath(theme.rounded(rect, 16))
        p.drawImage(rect, image, source)
        p.restore()
        p.setPen(QPen(theme.rgba(255, 255, 255, .10), 1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 16, 16)


@lru_cache(maxsize=4)
def clips(script: str) -> dict:
    """A clip script's CLIPS, imported when first shown (after Qt has started: the script uses Rflow's widgets)."""
    return importlib.import_module(script).CLIPS


class Home(WindowShot):
    """The real window's Home: the month's usage card in view, zoomed in, then the theme switched to light."""
    duration = 5.5
    captions = [(0.15, "See what you spend, in light or dark")]
    start, end = (500, 330, 0.85, 640, 318), (600, 360, 1.3, 640, 300)  # from the whole window to the usage card
    MOVE, LIGHT = (0.3, 2.6), (3.0, 0.8)

    def image(self, t):
        dark, light = home_window("dark"), home_window("light")
        k = ramp(t, *self.LIGHT)
        if k <= 0 or k >= 1:
            return light if k >= 1 else dark
        image = dark.copy()
        p = QPainter(image)
        p.setOpacity(k)
        p.drawImage(0, 0, light)
        p.end()
        return image


@lru_cache(maxsize=2)
def home_window(theme_name: str) -> QImage:
    """The real window on Home, with a month of example use on its usage card (make_site_screenshots.spend)."""
    from sst import window as w
    window = w.MainWindow(site.preview())
    window.apply_theme(theme_name)
    window.resize(1000, 680)
    window.set_status("Ready: hold Ctrl+Win · cleanup: gemini-3.5-flash-lite", True)
    window.show_page("home")
    settle()
    return window.grab().toImage()


class End(Scene):
    duration = 3.5
    brand = False

    def draw(self, p, t):
        with faded(p, ramp(t, 0.0, 0.6)):
            draw_logo(p, 640, 200, 128)
        k = ramp(t, 0.25, 0.55)
        with faded(p, k):
            button(p, "Download free: rflow-ai.vercel.app", 640, 360 + 8 * (1 - k), 24, 60)
        for start, y, text, px, colour, mono in [(0.45, 436, "github.com/karthi-ai-engineer/rflow-ai", 20, T["text2"], True),
                                                 (0.6, 478, "MIT License", 16, T["text3"], False)]:
            k = ramp(t, start, 0.55)
            with faded(p, k):
                write(p, QRectF(0, y - 20 + 8 * (1 - k), 1280, 40), text, px, 500, colour, mono=mono)


class EndCard(Scene):
    """The portrait cut's end card: what Rflow is, in two lines, and where to get it."""
    duration = 3.0
    brand = False

    def draw(self, p, t):
        with faded(p, ramp(t, 0.0, 0.6)):
            draw_logo(p, 270, 196, 132)
        for start, y, text, px, weight, colour in [(0.15, 318, "Rflow", 54, 600, T["text"]),
                                                   (0.3, 384, "Voice typing + live translation", 27, 600, T["text"]),
                                                   (0.45, 428, "Free · Open source · Windows", 22, 500, T["text2"])]:
            k = ramp(t, start, 0.5)
            with faded(p, k):
                write(p, QRectF(0, y - 40 + 8 * (1 - k), 540, 80), text, px, weight, colour)
        k = ramp(t, 0.6, 0.5)
        with faded(p, k):
            button(p, "rflow-ai.vercel.app", 270, 516 + 8 * (1 - k), 26, 64)


def button(p: QPainter, label: str, cx: float, cy: float, px: float, height: float) -> None:
    """The design's primary button, as a call to action."""
    width = QFontMetricsF(theme.font(px, 600)).horizontalAdvance(label) + px * 8 / 3
    rect = QRectF(cx - width / 2, cy - height / 2, width, height)
    theme.paint_surface(p, "primary", rect, 16, popup=True, dpr=STAGE.s)
    write(p, rect, label, px, 600, T["on_primary"])


# ---------------------------------------------------------------- the videos

def tour() -> list[Scene]:
    """About a minute, every feature: voice typing (with the keys, then the wake word), Text Transform by one word,
    translating selected text, typed text and the screen, live translation, Meeting mode and its minutes, the costs."""
    c, d = "make_feature_clips_c", "make_feature_clips_d"
    return [
        Title(),
        Dictation(duration=8.0, HIDE=7.3),
        Clip(d, "wake", duration=6.8, start=1.0, captions=[(0.15, "Or say your wake word: no keys at all")]),
        Transform(captions=[(0.15, "Select text and say “concise”")]),
        Translate(duration=5.0),
        Clip(c, "text", duration=5.5, start=1.5, captions=[(0.15, "Or type it on Translate's Text tab")]),
        Clip(c, "screen", duration=6.8, start=0.6, captions=[(0.15, "Ctrl+Alt+S translates anything on the screen")]),
        Live(captions=[(0.15, "Live translation: meetings and videos, as people speak")]),
        Clip(d, "meeting", duration=5.5, start=3.0, captions=[(0.15, "Meeting mode translates and records the meeting")]),
        Clip(d, "meeting", duration=5.4, start=10.8, captions=[(0.15, "Then it writes the minutes for you")]),
        Home(),
        End(),
    ]


def linkedin() -> list[Scene]:
    """About 12 s of voice typing from the first frame, the screen translated, 10 s of live translation, a meeting's
    minutes, the end card."""
    return [
        Dictation(duration=4.3, text=site.EXAMPLES[1][1], KEYS=-1, DOWN=-1.6, UP=1.25, TYPED=1.55, HIDE=3.0,
                  SPEECH=(-1.4, 1.1), captions=[(-1, "Hold Ctrl + Win"), (0.7, "Speak"), (1.5, "It types. Anywhere.")]),
        Transform(duration=4.2, SELECT=0.05, KEYS=0.35, DOWN=0.55, UP=1.85, DONE=2.6, SPEECH=(0.75, 1.65),
                  captions=[(-1, "Say “concise”")]),
        Translate(duration=3.9, SELECT=0.05, KEYS=0.5, CTRL=0.6, FIRST=0.8, SECOND=1.08, KEYS_OFF=1.3, OPEN=1.45, DONE=2.1,
                  captions=[(-1, "Translate any text")]),
        Clip("make_feature_clips_c", "screen", duration=4.2, start=2.8, crop=QRectF(0.12, 0.05, 0.5, 0.7),
             captions=[(-1, "Translate the screen")]),
        Meeting(duration=9.6),
        Clip("make_feature_clips_d", "meeting", duration=5.0, start=11.6, crop=QRectF(0.18, 0.03, 0.64, 0.75),
             captions=[(-1, "Then the minutes, written for you")]),
        EndCard(),
    ]


# ---------------------------------------------------------------- frames and files

def painter(image: QImage) -> QPainter:
    """A painter on a frame, in stage coordinates."""
    p = QPainter(image)
    for hint in (QPainter.RenderHint.Antialiasing, QPainter.RenderHint.TextAntialiasing,
                 QPainter.RenderHint.SmoothPixmapTransform):
        p.setRenderHint(hint)
    p.scale(STAGE.s, STAGE.s)
    return p


def paint(p: QPainter, scene: Scene, t: float, chrome: bool = True) -> None:
    if chrome:
        if STAGE.brand and scene.brand:
            draw_brand(p)
        if scene.chrome is not None:
            scene.chrome(p)
    scene.draw(p, t)
    draw_caption(p, scene.captions, t)


def render(scene: Scene, t: float) -> QImage:
    image = backdrop().copy()
    p = painter(image)
    paint(p, scene, t)
    p.end()
    return image


def frame(scenes: list[Scene], starts: list[float], g: float) -> QImage:
    """The video at g seconds: one scene; around the start of the next, the first fades out, then the next fades in
    (no text over text). What both show, the editor and the logo at the top, stays."""
    for i in range(1, len(scenes)):
        a, b, at = scenes[i - 1], scenes[i], starts[i]
        if at - FADE / 2 <= g < at + FADE / 2:
            k = (g - at + FADE / 2) / FADE
            shared = a.chrome is not None and a.chrome is b.chrome
            branded = STAGE.brand and a.brand and b.brand
            image = backdrop().copy()
            p = painter(image)
            if branded:
                draw_brand(p)
            if shared:
                a.chrome(p)
            for scene, t, amount in ((a, g - starts[i - 1], 1 - ease(2 * k)), (b, max(0.0, g - at), ease(2 * k - 1))):
                if amount > 0:  # each scene faded as a whole, so its own layers don't show through each other
                    layer = QImage(STAGE.w, STAGE.h, QImage.Format.Format_ARGB32_Premultiplied)
                    layer.fill(Qt.GlobalColor.transparent)
                    q = painter(layer)
                    if not branded and STAGE.brand and scene.brand:
                        draw_brand(q)
                    if not shared and scene.chrome is not None:
                        scene.chrome(q)
                    scene.draw(q, t)
                    draw_caption(q, scene.captions, t)
                    q.end()
                    p.save()
                    p.resetTransform()
                    p.setOpacity(amount)
                    p.drawImage(0, 0, layer)
                    p.restore()
            p.end()
            return image
    i = max(k for k, start in enumerate(starts) if start <= g)
    return render(scenes[i], g - starts[i])


def pixels(image: QImage) -> np.ndarray:
    """The frame's pixels as an array (BGRA), a copy: the image may be gone once this returns."""
    rows = np.frombuffer(image.constBits(), np.uint8, count=image.sizeInBytes()).reshape(image.height(), -1)
    return rows[:, :image.width() * 4].reshape(image.height(), image.width(), 4).copy()


def write_video(name: str, scenes: list[Scene]) -> float:
    starts = [sum(s.duration for s in scenes[:i]) for i in range(len(scenes))]
    total = starts[-1] + scenes[-1].duration
    path = OUT / name
    with av.open(str(path), "w", container_options={"movflags": "+faststart"}) as container:  # plays while it loads
        stream = container.add_stream("libx264", rate=FPS, options={
            "crf": "20", "preset": "slow", "tune": "animation",
            "x264-params": "colorprim=bt709:transfer=bt709:colormatrix=bt709"})  # HD's colours, said in the file
        stream.width, stream.height, stream.pix_fmt = STAGE.w, STAGE.h, "yuv420p"
        for n in range(round(total * FPS)):
            video = av.VideoFrame.from_ndarray(pixels(frame(scenes, starts, n / FPS)), format="bgra")
            container.mux(stream.encode(video.reformat(format="yuv420p", dst_colorspace=Colorspace.ITU709)))
        container.mux(stream.encode())
    print(f"Wrote {name}: {total:.1f} s, {STAGE.w}x{STAGE.h}, {FPS} fps, {path.stat().st_size / 1e6:.2f} MB")
    return total


def write_gif(name: str, scene: Scene) -> None:
    """One scene as a looping GIF, GIF_WIDTH wide."""
    size = (GIF_WIDTH, round(STAGE.h * GIF_WIDTH / STAGE.w))
    frames = []
    for n in range(round(scene.duration / GIF_STEP)):
        bgra = pixels(render(scene, n * GIF_STEP))
        frames.append(Image.fromarray(bgra[..., 2::-1].copy()).resize(size, Image.Resampling.LANCZOS))
    # One palette for every frame keeps the colours steady: the frames' own (from frames across the scene), plus the
    # lamps' colours and their edges, which are too few pixels to win a place on their own. Index 255 means "unchanged".
    sample = Image.new("RGB", (size[0], size[1] * 6))
    for i, f in enumerate(frames[:: max(1, len(frames) // 6)][:6]):
        sample.paste(f, (0, i * size[1]))
    lamps = [blend(POP["base"], T[name], k) for name in ("live", "violet", "ok", "iris", "selection") for k in (1, .6, .3)]
    colours = sample.quantize(colors=255 - len(lamps), method=Image.Quantize.MEDIANCUT).getpalette()[:3 * (255 - len(lamps))]
    colours += [v for c in lamps for v in (c.red(), c.green(), c.blue())]
    colours += [255, 0, 255] * (256 - len(colours) // 3)  # the rest, far from every colour in the frames
    palette = Image.new("P", (1, 1))
    palette.putpalette(colours)
    indexed = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    # Pixels that didn't change are left transparent: each frame stores only what moved.
    still = np.asarray(indexed[0]).copy()
    out = [indexed[0]]
    for f in indexed[1:]:
        now = np.asarray(f)
        image = Image.fromarray(np.where(now == still, 255, now).astype(np.uint8), "P")
        image.putpalette(colours)
        still = now
        out.append(image)
    out[0].save(OUT / name, save_all=True, append_images=out[1:], duration=round(GIF_STEP * 1000), loop=0,
                transparency=255, disposal=1, optimize=False)
    print(f"Wrote {name}: {len(out)} frames, {size[0]}x{size[1]}, {(OUT / name).stat().st_size / 1e6:.2f} MB")


def main() -> None:
    try:
        if not JOB:  # each video in a process of its own, side by side
            jobs = [subprocess.Popen([sys.executable, __file__, job]) for job in JOBS]
            sys.exit(max(job.wait() for job in jobs))
        if JOB not in JOBS:
            sys.exit(f"usage: make_demo_video.py [{' | '.join(JOBS)}]")
        QApplication([])
        theme.load_fonts()
        site.parakeet.find_model = lambda: Path(".")  # as if Parakeet were downloaded: Rflow shows as ready
        OUT.mkdir(parents=True, exist_ok=True)
        if JOB == "demo":
            scenes = tour()
            write_video("demo.mp4", scenes)
            write_gif("demo.gif", scenes[1])
        else:
            use(TALL)
            write_video("demo-linkedin.mp4", linkedin())
    finally:
        shutil.rmtree(DATA, ignore_errors=True)


if __name__ == "__main__":
    main()
